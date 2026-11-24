"""
CONCEPTUAL DESIGN - entry point.

    .venv/bin/python conceptual-design/conceptual.py          (from design/)

Runs the sizing loop (sizing.py) and writes
    outputs/design_point.json      -> hand-off to preliminary design (AeroSandbox)
    outputs/constraint_diagram.png
and prints the design point, constraint margins, BMF breakdown, mass
breakdown, the inputs the team still has to measure, and a sensitivity table.

The battery-mass-fraction equations from "Electric Propulsion Fundamentals
(Metric)" live in propulsion.py; mission.py applies them per segment with
CALCULATED L/D, speeds, efficiencies and battery specific energy.
"""
import json
import math
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

import requirements  # noqa: F401
import constraints
import sizing
from atmosphere import G0
from catalog import airfoil, real_batteries
from parameters import P

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")

# values the first version of this file ASSUMED - shown next to what is now calculated
OLD_ASSUMED = dict(LD=10.0, V=15.0, eta_prop=0.65, eta_b2s=0.80, U=150.0, roc=2.5)

SENSITIVITY = [
    ("mu_grass_ref", 0.07, 0.15), ("bekker_n", 0.5, 1.1), ("mu_brake_grass", 0.10, 0.30),
    ("cg_balance_tol_m", 0.001, 0.004), ("mass_sigma_structure", 0.08, 0.25),
    ("eta_esc", 0.90, 0.97), ("avionics_power_factor", 0.7, 1.5),
    ("weight_growth_allowance", 0.0, 0.20), ("wing_internal_ratio", 0.2, 0.6),
    ("fus_frames_ratio", 0.10, 0.40), ("roughness_k_m", 1e-5, 1e-4),
    ("leak_protub_frac", 0.02, 0.10), ("q_gear", 1.0, 1.5),
    ("lipo_sag_at_rated_v", 0.25, 0.6), ("prop_mass_10in_kg", 0.018, 0.030),
    ("isa_temp_offset_K", 0.0, 25.0), ("field_elevation_m", 0.0, 1500.0),
]


def _f(x, nd=3):
    return float(round(float(x), nd))


def flight_thrust_table(best):
    """Thrust AVAILABLE (at the throttle cap) vs. drag at the speeds that size the
    aircraft. Thrust comes from the APC CT(J, RPM) data at each airspeed with the
    motor/battery operating point solved there - not from static thrust."""
    from propulsion import V_GRID
    d = best.detail
    perf, pt = d["perf"], d["pt"]
    ft = pt.full_throttle()
    W = best.m * G0
    pts = [("static (V=0)", 0.0), ("liftoff 1.1 Vs", perf["takeoff"]["V_lo"]),
           ("V_mc engine-out 1.2 Vs", d["engine_out"].get("V_mc", P.vmc_ratio * perf["V_s"])),
           ("loiter", perf["loiter"]["V"]), ("best climb V_y", perf["climb"]["V"]),
           ("max level speed", perf["V_max"])]
    rows = []
    for name, V in pts:
        it = lambda arr: float(np.interp(V, V_GRID, arr))
        T, Ps = it(ft["T"]), it(ft["P_shaft"])
        Dl = it(perf["D_curve"]) if V >= perf["V_s"] else float("nan")
        rows.append(dict(point=name, V_ms=_f(V, 2), thrust_avail_N=_f(T, 3),
                         thrust_per_weight=_f(T / W, 3),
                         drag_level_N=_f(Dl, 3) if Dl == Dl else None,
                         excess_N=_f(T - Dl, 3) if Dl == Dl else None,
                         eta_prop=_f(T * V / Ps, 3) if Ps > 0 else None,
                         rpm=_f(it(ft["rpm"]), 0), I_motor_A=_f(it(ft["I_motor"]), 2),
                         I_batt_A=_f(it(ft["I_batt"]), 2)))
    lo = perf["loiter"]
    rows.append(dict(point="loiter (thrust REQUIRED, part throttle)", V_ms=_f(lo["V"], 2),
                     thrust_avail_N=_f(lo["D"], 3), thrust_per_weight=_f(lo["D"] / W, 3),
                     drag_level_N=_f(lo["D"], 3), excess_N=0.0, eta_prop=_f(lo["eta_prop"], 3),
                     rpm=_f(lo["rpm"], 0), I_motor_A=None, I_batt_A=_f(lo["I_batt"], 2),
                     duty=_f(lo["duty"], 3)))
    return rows


def print_twist(rows, best):
    root = best.combo.airfoil
    tip_now = best.combo.tip_airfoil or root
    print(f"\n TWIST TRADE STUDY  (root {root}; each row re-optimized S, AR, tail arm with the same parts)")
    print("   aerodynamic twist = different tip airfoil (lofted root->tip); geometric twist = washout")
    print(f"  {'tip airfoil':11s} {'washout':>7s} {'mass kg':>8s} {'span m':>7s} {'loiter':>7s} {'landing':>8s} "
          f"{'CLmax':>6s} {'CLmax_L':>7s} {'stall@':>7s} {'e':>6s} {'L/D':>6s}  status")
    for r in sorted(rows, key=lambda r: (r["tip"] != root, r["tip"], r["washout"])):
        if not r.get("ok"):
            print(f"  {r['tip']:11s} {r['washout']:7.1f}   (did not converge)")
            continue
        mark = " <- chosen" if (r["tip"] == tip_now and abs(r["washout"] - best.detail["washout"]) < 0.05) else ""
        st = "ok" if r["feasible"] else "misses " + "+".join(r["misses"])
        print(f"  {r['tip']:11s} {r['washout']:7.1f} {r['m']:8.4f} {r['span']:7.3f} {r['loiter_min']:6.1f}m "
              f"{r['landing_m']:7.1f}m {r['CLmax']:6.3f} {r['CLmax_landing']:7.3f} {r['stall_eta']:6.0%} "
              f"{r['e']:6.3f} {r['LD_loiter']:6.2f}  {st}{mark}")
    print(f"  (optimizer's free choice: tip {tip_now}, washout {best.detail['washout']:.2f} deg)")


def sensitivity(best):
    base = best.detail["bmf"]
    rows = []
    for name, lo, hi in SENSITIVITY:
        row = dict(param=name, base=getattr(P, name), cases=[])
        for v in (lo, hi):
            with P.override(**{name: v}):
                r = sizing.evaluate(best.x, best.combo)
            if not r.ok:
                row["cases"].append(dict(value=v, ok=False))
                continue
            p, b = r.detail["perf"], r.detail["bmf"]
            row["cases"].append(dict(value=v, ok=True, feasible=r.feasible,
                                     misses=[k for k, c in r.cons.items() if c < -1e-4],
                                     d_loiter_min=b["loiter_max_min"] - base["loiter_max_min"],
                                     takeoff_m=p["takeoff"]["s"], landing_m=p["s_landing"], m=r.m))
        rows.append(row)
    def impact(c):          # worst fractional change against any requirement
        if not c.get("ok"):
            return 99.0
        base_to, base_ld = best.detail["perf"]["takeoff"]["s"], best.detail["perf"]["s_landing"]
        return max(abs(c["d_loiter_min"]) / P.endurance_min,
                   abs(c["takeoff_m"] - base_to) / P.takeoff_roll_max_grass_m,
                   abs(c["landing_m"] - base_ld) / P.landing_roll_max_grass_m)
    rows.sort(key=lambda r: -max(impact(c) for c in r["cases"]))
    return rows


def print_balance(best):
    b, tl, g = best.detail["balance"], best.detail["tail"], best.detail["geom"]
    if not b:
        return
    c = g.wing.mac
    print("\n BALANCE  (CG band from the as-built scatter; scissor plot hangs it from the aft limit)")
    print(f"  as-built CG scatter 1-sigma {b['sigma_x'] * 1000:.1f} mm; x{P.cg_coverage_sigma:.0f} = "
          f"{P.cg_coverage_sigma * b['sigma_x'] * 1000:.1f} mm, battery slide corrects "
          f"{b['slide_authority'] * 1000:.1f} mm, balancing accuracy +/-{P.cg_balance_tol_m * 1000:.1f} mm")
    print(f"  -> CG band {b['band_m'] * 1000:.1f} mm = {b['band_frac_mac']:.1%} MAC "
          f"(was 15% MAC from the fixed 5-20% static-margin range)")
    print(f"  stability line: aft CG {tl['h_cg_aft']:.1%} MAC = NP {tl['h_n']:.1%} - {P.static_margin_min:.0%};  "
          f"control line: forward CG {tl['h_cg_fwd']:.1%} MAC (elevator {tl['elevator']['ratio']:.2f}x required)")
    print(f"  wing placed at x_qc {g.x_qc * 1000:.0f} mm so the nominal CG ({b['x_cg'] * 1000:.0f} mm, "
          f"{b['h_cg']:.1%} MAC) sits mid-band" + ("  [CLAMPED: wing at the nose]" if g.x_qc_clamped else ""))
    top = sorted(b["items"], key=lambda i: -abs(i["sigma"] * i["m"] * (i["x"] - b["x_cg"])))[:5]
    print("  biggest scatter contributors: " + ", ".join(
        f"{i['name']} ({i['sigma'] * i['m'] * abs(i['x'] - b['x_cg']) / b['m'] * 1000:.2f} mm)" for i in top))


def print_lateral(best):
    lat = best.detail.get("lateral")
    if not lat:
        return
    eo, xw, dv = lat["engine_out"], lat["crosswind"], lat["derivatives"]
    print("\n COUPLED LATERAL-DIRECTIONAL TRIM  (side force + roll + yaw solved together)")
    print(f"  derivatives /rad: Cn_beta {dv['cn_beta']:+.3f} (fuselage {dv['cn_beta_fus']:+.3f}), "
          f"Cl_beta {dv['cl_beta']:+.3f} (wing height {dv['cl_beta_wing']:+.3f}, no dihedral), "
          f"CY_beta {dv['cy_beta']:+.3f}, full aileron Cl {dv['cl_aileron_full']:.4f} / adverse Cn {dv['cn_aileron_full']:+.4f}")
    print(f"  {'case':28s} {'V':>5s} {'bank':>6s} {'slip':>6s} {'aileron':>8s} {'rudder':>7s} {'usable':>7s} {'fin a':>6s}")
    for name, c in (("engine-out (right motor dead)", eo), ("crosswind 0.2 V_SO from right", xw)):
        if c:
            print(f"  {name:28s} {c['V']:5.1f} {c['phi_deg']:+6.1f} {c['beta_deg']:+6.1f} {c['aileron_frac']:+8.0%} "
                  f"{c['rudder_deg']:+7.1f} {c['rudder_usable_deg']:7.1f} {c['fin_alpha_deg']:+6.1f}")
    if eo:
        print(f"  engine-out: {eo['T_one']:.2f} N live motor + {eo['D_dead_prop']:.2f} N stopped-prop drag at "
              f"{eo['V']:.1f} m/s -> yaw {eo['yaw_moment']:.3f} N.m; slipstream roll {eo['roll_moment']:.3f} N.m "
              f"(+{eo['slipstream_dV']:.1f} m/s over the wing). Fin needed {lat['sv_needed']:.4f} m^2 "
              f"({lat['rudder_cf']:.0%} rudder)")


def print_landing(best):
    p = best.detail["perf"]
    ap = p["approach"]
    print("\n LANDING / APPROACH  (stall speeds trimmed at the forward CG; deflected-surface drag included)")
    print(f"  V_stall landing {p['V_s_landing']:.2f} m/s (wing CLmax {p['CLmax_landing']:.3f}, trimmed "
          f"{p['CLmax_landing_trimmed']:.3f}); touchdown {p['V_td']:.1f} m/s; approach {ap['V']:.1f} m/s")
    if ap.get("trim"):
        t = ap["trim"]
        print(f"  approach trim: tail CL {t['CL_tail']:+.3f}, elevator {t['elevator_deg']:+.1f} deg, trim drag "
              f"{t['dCD'] * 1e4:.0f} counts; glide (motors idle) {ap['glide_deg']:.1f} deg; balked-landing "
              f"gradient {ap['balked_gradient']:.3f}")
    mg, mp = p["mu_grass"], p["mu_pavement"]
    print(f"  wheels: grass rolling mu {mg['roll']:.3f} (main {mg['main']:.3f}, nose {mg['nose']:.3f}); braked "
          f"{mg['brake']:.3f} grass / {mp['brake']:.3f} pavement" + ("" if mg["braked"] else "  (NO BRAKES)"))
    crow = p.get("crow")
    print(f"  braked roll: grass {p['s_landing']:.1f} m, pavement {p['s_landing_pavement']:.1f} m "
          f"(after {P.brake_delay_s:.0f} s free roll)" + (
              f"; crow ailerons up {crow['deflection_up_deg']:.0f} deg (without crow {p['s_landing_nocrow']:.1f} m)"
              if crow else ""))


def print_wheels(rows):
    print("\n WHEEL TRADE STUDY  (design frozen; same wheel on all three legs)")
    print(f"  {'wheel':14s} {'d mm':>5s} {'mass g':>7s} {'gear CD':>8s} {'grass mu':>8s} {'TO grass':>9s} "
          f"{'LDG grass':>9s} {'brake':>6s}")
    for r in rows:
        if not r.get("ok"):
            print(f"  {r['wheel']:14s} (failed)")
            continue
        mark = "  <- chosen" if r["chosen"] else ""
        print(f"  {r['wheel']:14s} {r['d_mm']:5.0f} {r['m'] * 1000:7.1f} {r['CD_gear'] * 1e4:7.0f}c {r['mu_grass']:8.3f} "
              f"{r['takeoff_grass_m']:8.1f}m {r['landing_grass_m']:8.1f}m {'fits' if r['brake_fits'] else 'NO':>6s}{mark}")


def print_stall(rows, best):
    print("\n STALL-SPEED TRADE STUDY  (lightest design at each stall-speed cap, same parts, re-optimized)")
    print(f"  {'cap':>5s} {'V_s':>6s} {'mass kg':>8s} {'score':>7s} {'S m2':>7s} {'AR':>5s} {'span':>6s} "
          f"{'W/S g/dm2':>9s} {'V_td':>5s} {'TO grs':>7s} {'LDG grs':>7s}  status")
    for r in rows:
        if not r.get("ok"):
            print(f"  {r['cap']:5.1f}  (did not converge)")
            continue
        st = "ok" if r["feasible"] else "misses " + "+".join(r["misses"])
        print(f"  {r['cap']:5.1f} {r['V_s_landing']:6.2f} {r['m']:8.4f} {r['score']:7.4f} {r['S']:7.4f} {r['AR']:5.2f} "
              f"{r['span']:6.3f} {r['W_S'] / G0 * 10:9.1f} {r['V_td']:5.1f} {r['takeoff_m']:6.1f}m "
              f"{r['landing_m']:6.1f}m  {st}")
    print(f"  chosen: V_s {best.vs:.2f} m/s, {best.m:.4f} kg (objective weight "
          f"{P.stall_speed_weight_g_per_ms:.0f} g per m/s)")


def wheel_study(best):
    """Every catalog wheel on all three legs, design frozen (same x, parts)."""
    from catalog import wheels, wheel_brake
    rows = []
    for name, wh in wheels().items():
        with P.override(main_wheel=name, nose_wheel=name):
            r = sizing.evaluate(best.x, best.combo)
        if not r.ok:
            rows.append(dict(wheel=name, ok=False))
            continue
        p = r.detail["perf"]
        rows.append(dict(wheel=name, ok=True, d_mm=wh.d * 1000, m=r.m, CD_gear=r.detail["aero"].CD_gear,
                         mu_grass=p["mu_grass"]["roll"], takeoff_grass_m=p["takeoff"]["s"],
                         landing_grass_m=p["s_landing"], brake_fits=wheel_brake().fits(wh),
                         feasible=r.feasible, chosen=(name == P.main_wheel)))
    return rows


def report(best, alts, cdiag, sens, runtime, twist=None, wheels_=None, stall=None):
    d = best.detail
    g, perf, bmf, ms, batt = d["geom"], d["perf"], d["bmf"], d["masses"], d["battery"]
    lo, cl, to = perf["loiter"], perf["climb"], perf["takeoff"]
    W = best.m * G0
    line = "=" * 78
    print("\n" + line)
    print(" CONCEPTUAL DESIGN POINT  (min  mass + %.0f g per m/s of stall speed, meeting every requirement)"
          % P.stall_speed_weight_g_per_ms)
    print(line)
    print(f" Objective {best.score:.4f} kg = mass {best.m:.4f} kg + {P.stall_speed_weight_g_per_ms:.0f} g/(m/s) x "
          f"V_stall {best.vs:.2f} m/s (landing config, trimmed at forward CG)")
    print(f" Takeoff mass {best.m:.3f} kg   W/S {W / g.S:.1f} N/m^2   "
          f"P/W {cdiag['design'][1]:.1f} W/N (installed shaft)   search {runtime:.0f} s")
    print(f" Atmosphere: {P.field_elevation_m:.0f} m, ISA{P.isa_temp_offset_K:+.0f} K, "
          f"rho = {d['atm'].rho:.4f} kg/m^3")

    print("\n REQUIREMENTS / CONSTRAINTS                         value        limit    margin")
    rows = [
        ("Endurance (loiter only, not counting climb)", f"{bmf['loiter_max_min']:.1f} min", f">= {P.endurance_min:.0f}", best.cons.get("endurance", 0)),
        ("  total airborne time", f"{bmf['airborne_max_min']:.1f} min", "", None),
        ("Takeoff ground roll (soft grass)", f"{to['s']:.1f} m", f"<= {P.takeoff_roll_max_grass_m:.0f}", best.cons["takeoff_roll_grass"]),
        ("Takeoff ground roll (pavement)", f"{perf['takeoff_pavement']['s']:.1f} m", f"<= {P.takeoff_roll_max_pavement_m:.0f}", best.cons["takeoff_roll_pavement"]),
        ("Landing ground roll (soft grass, braked)", f"{perf['s_landing']:.1f} m", f"<= {P.landing_roll_max_grass_m:.0f}", best.cons["landing_roll_grass"]),
        ("Landing ground roll (pavement, braked)", f"{perf['s_landing_pavement']:.1f} m", f"<= {P.landing_roll_max_pavement_m:.0f}", best.cons["landing_roll_pavement"]),
        ("Balked-landing climb gradient", f"{perf['approach']['balked_gradient']:.3f}", f">= {P.balked_landing_gradient_min:.3f}", best.cons["balked_landing"]),
        ("Wingspan", f"{g.b:.3f} m", f"<= {P.span_max_m}", best.cons["span"]),
        ("Climb rate (at throttle cap)", f"{cl['roc']:.2f} m/s", f">= {max(P.min_roc_ms, P.climb_rate_target_ms):.2f}", best.cons["climb_rate"]),
        ("Climb gradient", f"{cl['grad_max']:.3f}", f">= {P.min_climb_gradient:.3f}", best.cons["climb_gradient"]),
        ("Motor current (max)", f"{perf['I_motor_max']:.1f} A", f"<= {sizing.motors()[best.combo.motor].imax:.0f}", best.cons["motor_current"]),
        ("Battery current (max)", f"{perf['I_batt_max']:.1f} A", f"<= {batt.i_max:.0f}", best.cons["battery_current"]),
        ("Spar fits in root airfoil", f"{d['winfo']['spar']['util']:.2f}", "< 1", best.cons["spar_fits"]),
        ("Tip Reynolds number at loiter", f"{(best.cons['tip_reynolds'] + 1) * P.re_tip_min / 1e3:.0f}k", f">= {P.re_tip_min / 1e3:.0f}k", best.cons["tip_reynolds"]),
        ("Roll helix pb/2V (approach)", f"{d['controls']['aileron']['pb2v']:.3f}", f">= {P.roll_helix_min}", best.cons["roll_helix"]),
        ("Elevator authority (forward CG)", f"{d['tail']['elevator']['ratio']:.2f}x", ">= 1", best.cons["elevator_authority"]),
        ("Crosswind rudder (0.2 V_SO)", f"{d['lateral']['crosswind']['ratio']:.2f}", "<= 1 of usable", best.cons["crosswind_rudder"]),
        ("Crosswind aileron", f"{abs(d['lateral']['crosswind']['aileron_frac']):.0%}", f"<= {P.trim_aileron_max_frac:.0%}", best.cons["crosswind_aileron"]),
        ("Engine-out aileron", f"{abs(d['lateral']['engine_out']['aileron_frac']):.0%}", f"<= {P.trim_aileron_max_frac:.0%}", best.cons["engine_out_aileron"]),
        ("Directional stability Cn_beta", f"{d['lateral']['derivatives']['cn_beta']:.3f}/rad", "> 0", best.cons["cn_beta"]),
    ]
    for name, val, lim, mar in rows:
        ms_ = "" if mar is None else (f"{mar * 100:+6.1f}%" + ("  ACTIVE" if abs(mar) < 0.02 else ""))
        print(f"  {name:46s} {val:>11s} {lim:>10s}  {ms_}")

    print("\n CONFIGURATION")
    print(f"  Wing   S={g.S:.4f} m^2  b={g.b:.3f} m  AR={g.AR:.2f}  taper={g.wing.taper}  "
          f"c_root={g.wing.c_root * 1000:.0f} mm  c_tip={g.wing.c_tip * 1000:.0f} mm  MAC={g.wing.mac * 1000:.0f} mm")
    tip = best.combo.tip_airfoil or best.combo.airfoil
    print(f"         airfoil {best.combo.airfoil} (root t/c {g.wing.t_c:.3f}) -> {tip} at the tip, washout "
          f"{d['washout']:.2f} deg   construction: {best.combo.construction}   span efficiency e {d['aero'].e:.3f}")
    print(f"         spar: carbon tube OD {d['winfo']['spar']['D'] * 1000:.1f} mm x {d['winfo']['spar']['t'] * 1000:.2f} mm wall"
          f"  (n_limit {d['winfo']['n_limit']:.2f}, ultimate {d['winfo']['n_ult']:.2f}, V_C {d['V_C']:.1f} m/s)")
    sh_drv = ("scissor plot (elevator at forward CG)" if P.htail_sizing == "scissor" else "tail volume / elevator")
    print(f"  H-tail S={g.htail.S:.4f} m^2  span={g.htail.b * 1000:.0f} mm  tail arm={g.l_t * 1000:.0f} mm  "
          f"(sized by {sh_drv}; Raymer tail-volume value {g.sh_volume:.4f})")
    sv_drv = "coupled engine-out trim" if g.sv_engine_out > g.sv_volume else "Raymer tail volume (floor)"
    print(f"  Fin    S={g.vtail.S:.4f} m^2  height={g.vtail.b * 1000:.0f} mm  (sized by {sv_drv}; "
          f"volume-coeff value {g.sv_volume:.4f}, engine-out needs {g.sv_engine_out:.4f})")
    print(f"  Fuselage L={g.fus_L * 1000:.0f} mm  {g.fus_w * 1000:.0f} x {g.fus_h * 1000:.0f} mm section  bay {g.bay_L * 1000:.0f} mm")
    print(f"  Motors at y=+/-{g.y_eng * 1000:.0f} mm   gear height {g.gear_h * 1000:.0f} mm  track {g.track * 1000:.0f} mm")

    cj = controls_json(best)
    semi = g.b / 2
    print("\n CONTROL SURFACES  (sized from requirements; NeuralFoil on the deflected sections)")
    ai = cj["ailerons"]
    print(f"  Configuration: {cj['configuration']}")
    print(f"  {'Flaperons' if ai['is_flaperon'] else 'Ailerons'}: {ai['y_inboard_m'] * 1000:.0f}-{ai['y_outboard_m'] * 1000:.0f} mm "
          f"from centerline ({ai['eta_inboard']:.0%}-{ai['eta_outboard']:.0%} semispan), {ai['chord_fraction']:.0%} chord, "
          f"+/-{ai['throw_deg']:.0f} deg -> pb/2V {ai['roll_helix_pb_2V']:.3f} (req >= {P.roll_helix_min})")
    if cj["flaps"]:
        fl = cj["flaps"]
        label = "Flaperon droop" if ai["is_flaperon"] else "Flaps"
        print(f"  {label}: {fl['y_inboard_m'] * 1000:.0f}-{fl['y_outboard_m'] * 1000:.0f} mm ({fl['eta_inboard']:.0%}-"
              f"{fl['eta_outboard']:.0%}), {fl['chord_fraction']:.0%} chord, {fl['landing_deflection_deg']:.0f} deg for landing "
              f"-> CLmax {perf['CLmax']:.3f} clean / {perf['CLmax_landing']:.3f} landing, "
              f"V_stall {perf['V_s']:.2f} -> {perf['V_s_landing']:.2f} m/s")
    el, ru, stb = cj["elevator"], cj["rudder"], cj["stability"]
    print(f"  Elevator: {el['chord_fraction']:.0%} chord, {el['span_fraction']:.0%} span, {el['throw_deg']:.0f} deg up; "
          f"authority {el['authority_ratio']:.2f}x required (landing trim {el['ratio_landing_trim']:.2f}, "
          f"takeoff {el['ratio_takeoff']:.2f})")
    print(f"  Rudder:   {ru['chord_fraction']:.0%} chord, {ru['span_fraction']:.0%} height; engine-out needs "
          f"{ru['engine_out_deg']:.1f} deg of {ru['throw_usable_deg']:.1f} trusted, crosswind {ru['crosswind_deg']:.1f} deg")
    print(f"  Stabilizer {stb['stabilizer_area_m2']:.4f} m^2 (tail-volume value {stb['stabilizer_area_tail_volume_m2']:.4f})")
    print(f"  Neutral point {stb['neutral_point_frac_mac']:.1%} MAC (fuselage moves it {stb['np_shift_fuselage']:+.1%}); "
          f"CG band {stb['cg_forward_frac_mac']:.1%}-{stb['cg_aft_frac_mac']:.1%} MAC (static margin "
          f"{stb['static_margin_range'][1]:.1%}-{stb['static_margin_range'][0]:.1%})")
    print(f"  Incidence: wing {stb['wing_incidence_deg']:+.2f} deg, stabilizer {stb['tail_incidence_deg']:+.2f} deg "
          f"(decalage {stb['decalage_deg']:+.2f}) relative to the fuselage axis")
    print(f"  Takeoff: {stb['takeoff_technique']}" + (f", nose-up ground stance {stb['ground_stance_deg']:.1f} deg"
                                                    if stb["takeoff_technique"] == "stance" else ""))
    print(f"  Stall starts at {perf['stall_eta']:.0%} semispan (clean), {perf['stall_eta_landing']:.0%} (landing)")
    print_balance(best)
    print_lateral(best)
    print_landing(best)

    print("\n PROPULSION (selected from catalogs)")
    st = perf["static"]
    print(f"  2 x {best.combo.motor}   2 x {best.combo.prop}   2 x {d['esc'].name}")
    print(f"  Battery: {batt.name}  {batt.energy_wh:.1f} Wh  {batt.mass_kg * 1000:.0f} g  "
          f"U={batt.u_wh_kg:.0f} Wh/kg  K={batt.k}")
    cap = d["throttle_cap"]
    print(f"  THROTTLE CAP {cap:.3f} (max ESC duty) - set as the max output on BOTH motor channels "
          f"in the flight controller so it applies in every mode, incl. manual")
    print(f"  Static (capped): thrust {st['T']:.1f} N (T/W {st['T'] / W:.2f})  {st['I_motor']:.1f} A/motor  {st['rpm']:.0f} rpm")
    print(f"  Avionics load {d['p_av']:.1f} W")
    print("\n IN-FLIGHT THRUST (at the throttle cap; APC data at each airspeed)")
    print(f"  {'point':38s} {'V m/s':>6s} {'T avail N':>9s} {'T/W':>6s} {'drag N':>7s} "
          f"{'excess':>7s} {'eta_p':>6s} {'rpm':>6s} {'I_mot':>6s} {'I_bat':>6s}")
    for r in flight_thrust_table(best):
        fmt = lambda v, f: (f % v) if v is not None else "-"
        print(f"  {r['point']:38s} {r['V_ms']:6.2f} {r['thrust_avail_N']:9.3f} {r['thrust_per_weight']:6.3f} "
              f"{fmt(r['drag_level_N'], '%7.3f'):>7s} {fmt(r['excess_N'], '%7.3f'):>7s} "
              f"{fmt(r['eta_prop'], '%6.3f'):>6s} {r['rpm']:6.0f} {fmt(r['I_motor_A'], '%6.2f'):>6s} "
              f"{fmt(r['I_batt_A'], '%6.2f'):>6s}")

    print("\n AERODYNAMICS")
    print(f"  V_stall {perf['V_s']:.2f} m/s (CLmax {perf['CLmax']:.3f})  V_loiter {lo['V']:.2f} m/s  "
          f"V_max {perf['V_max']:.1f} m/s  best glide L/D {perf['glide']['LD']:.2f}")
    print(f"  Polar fit CD0 {cdiag['CD0']:.4f}  K {cdiag['K']:.4f}   inviscid e {d['aero'].e:.3f}")
    parts = lo["drag_parts"]
    tot = sum(parts.values())
    print("  Drag at loiter (counts): " + ", ".join(f"{k} {v * 1e4:.0f}" for k, v in parts.items())
          + f"  = {tot * 1e4:.0f}")

    print("\n MISSION BMF  (image equations, calculated inputs)")
    print(f"  takeoff  {bmf['takeoff']:.4f}   t={to['t']:.1f} s, eta_b2s={bmf['eta_b2s_takeoff']:.3f}")
    print(f"  climb    {bmf['climb']:.4f}   {P.climb_altitude_m:.0f} m at {cl['V']:.1f} m/s, V_v={cl['roc']:.2f} m/s, "
          f"L/D={cl['LD']:.2f}, eta_total={cl['eta_total']:.3f}")
    print(f"  loiter   {bmf['loiter']:.4f}   {P.endurance_min:.0f} min at {lo['V']:.2f} m/s, L/D={lo['LD']:.2f}, "
          f"eta_prop={lo['eta_prop']:.3f} x eta_b2s={lo['eta_b2s']:.3f} = {lo['eta_total']:.3f}")
    print(f"  avionics {bmf['avionics']:.4f}   {d['p_av']:.1f} W for {bmf['t_total_s'] / 60:.1f} min")
    print(f"  TOTAL    {bmf['total']:.4f}   -> needs {bmf['battery_mass_required'] * 1000:.0f} g of this chemistry; "
          f"pack is {batt.mass_kg * 1000:.0f} g (margin {(bmf['endurance_margin'] - 1) * 100:+.1f}%)")

    print("\n PREVIOUSLY ASSUMED  ->  NOW CALCULATED")
    for k, new, unit in [("LD", lo["LD"], ""), ("V", lo["V"], "m/s"), ("eta_prop", lo["eta_prop"], ""),
                         ("eta_b2s", lo["eta_b2s"], ""), ("U", batt.u_wh_kg, "Wh/kg"), ("roc", cl["roc"], "m/s")]:
        print(f"  {k:9s} {OLD_ASSUMED[k]:8.3f}  ->  {new:8.3f} {unit}")

    print("\n MASS BREAKDOWN")
    for k, v in sorted(ms.items(), key=lambda kv: -kv[1]):
        if v > 0:
            print(f"  {k:18s} {v * 1000:7.1f} g  {v / best.m * 100:5.1f}%")

    print("\n ALTERNATIVES (next-lightest feasible designs)")
    for r in alts:
        print(f"  {r.m:.3f} kg  {r.combo.label()}")

    print("\n SENSITIVITY (design frozen at the point above; ranked by worst impact on any requirement)")
    print("   'MISSES' = the frozen design would miss a requirement -> margin needed / measure it")
    for row in sens[:12]:
        cs = "   ".join(
            f"{c['value']:g}: {c['d_loiter_min']:+5.1f} min, TO {c['takeoff_m']:.0f} m, LDG {c['landing_m']:.0f} m"
            + ("" if c["feasible"] else " MISSES " + "+".join(c["misses"])) if c.get("ok") else f"{c['value']:g}: failed"
            for c in row["cases"])
        print(f"  {row['param']:24s} (now {row['base']:g})  {cs}")

    if twist:
        print_twist(twist, best)
    if stall:
        print_stall(stall, best)
    if wheels_:
        print_wheels(wheels_)

    print("\n INPUTS TO MEASURE / VERIFY (they replace estimates in parameters.py and data/*.csv)")
    for p in P.all():
        if p.measure:
            print(f"  - {p.name} = {p.value} {p.units}: {p.source}")
    print(line)


def layout(best):
    """Part positions and dimensions for CAD / preliminary design (the single
    source for dimensions.txt and visualize.py). Metres and degrees.
    Frame: x aft from the nose tip, y right, z up, fuselage centreline at z = 0.
    The wing station comes from the balance (nominal CG mid-band); the CG band
    from the scissor plot; the main gear 15 deg behind the AFT CG (Raymer)."""
    from catalog import airfoil_shape, components
    from geometry import fuselage_profile
    from weights import nacelle_stations
    d = best.detail
    g, tl, ct, wi, bal = d["geom"], d["tail"], d["controls"], d["winfo"], d["balance"]
    r = lambda v: _f(v, 5)
    x_qc = g.x_qc
    z_wing = g.fus_h / 2
    semi = g.b / 2
    c_eng = float(g.chord_at(g.y_eng))
    nac = nacelle_stations(g)
    z_axis = z_wing + 0.5 * g.wing.t_c * c_eng
    x_tail = g.x_tail
    z_ground = -(g.fus_h / 2 + g.gear_h)
    x_main = tl["x_main_gear"]
    x_nose = 0.8 * g.L_nose
    stance = tl["stance_deg"]
    root_af = best.combo.airfoil
    tip_af = best.combo.tip_airfoil or best.combo.airfoil
    wsh = airfoil_shape(root_af)
    x_spar = x_qc + (wsh["x_tmax"] - 0.25) * g.wing.c_root
    c_tip = g.wing.c_tip
    lam = g.wing.taper
    y_mac = semi / 3 * (1 + 2 * lam) / (1 + lam)

    def le(c):
        return x_qc - 0.25 * c

    def hinge(x_qc_s, c, cf):
        return x_qc_s - 0.25 * c + (1 - cf) * c

    def surf_block(s_, x_q, extra):
        out = {"area": r(s_.S), "span": r(s_.b), "c_root": r(s_.c_root), "c_tip": r(s_.c_tip),
               "taper": r(s_.taper), "mac": r(s_.mac), "t_c": r(s_.t_c),
               "x_quarter_chord": r(x_q), "x_le_root": r(x_q - 0.25 * s_.c_root),
               "x_le_tip": r(x_q - 0.25 * s_.c_tip), "x_te_root": r(x_q + 0.75 * s_.c_root),
               "x_te_tip": r(x_q + 0.75 * s_.c_tip)}
        out.update(extra)
        return out

    a = ct["aileron"]
    c_a1, c_a2 = float(g.chord_at(a["y1"])), float(g.chord_at(a["y2"]))
    ailerons = {"y_inboard": r(a["y1"]), "y_outboard": r(a["y2"]), "chord_fraction": a["cf"],
                "chord_inboard": r(a["cf"] * c_a1), "chord_outboard": r(a["cf"] * c_a2),
                "hinge_x_inboard": r(hinge(x_qc, c_a1, a["cf"])), "hinge_x_outboard": r(hinge(x_qc, c_a2, a["cf"])),
                "throw_deg": a["throw"], "is_flaperon": ct["config"] == "flaperons",
                "crow_up_deg": (d["perf"].get("crow") or {}).get("deflection_up_deg", 0.0)}
    flaps = None
    if ct["flap"] and ct["config"] == "flaps":
        f = ct["flap"]
        c1, c2 = float(g.chord_at(f["y0"])), float(g.chord_at(f["y1"]))
        flaps = {"y_inboard": r(f["y0"]), "y_outboard": r(f["y1"]), "chord_fraction": f["cf"],
                 "hinge_x_inboard": r(hinge(x_qc, c1, f["cf"])), "hinge_x_outboard": r(hinge(x_qc, c2, f["cf"])),
                 "landing_deflection_deg": f["deflection"]}
    el, lat = tl["elevator"], d.get("lateral") or {}
    cf_r = lat.get("rudder_cf") or d["engine_out"].get("rudder_cf") or 0.4
    ht, vt = g.htail, g.vtail
    eo = lat.get("engine_out") or {}
    xw = lat.get("crosswind") or {}
    rudder_throw = max(abs(eo.get("rudder_deg", 0.0)), abs(xw.get("rudder_deg", 0.0)))
    bay_x0 = g.L_nose
    comps = [{"name": c.name, "qty": c.qty, "l": r(c.dims_m[0]), "w": r(c.dims_m[1]), "h": r(c.dims_m[2]),
              "mass_each": r(c.mass_kg), "in_fuselage": c.in_fuselage} for c in components()]
    batt = d["battery"]
    return {
        "frame": "x aft from nose tip, y right, z up, fuselage centerline z=0 [m]",
        "fuselage": {"nose_length": r(g.L_nose), "bay_length": r(g.bay_L), "length": r(g.fus_L),
                     "width": r(g.fus_w), "height": r(g.fus_h), "wall_t": P.fus_skin_t_m,
                     "bay_x_start": r(bay_x0), "bay_x_end": r(bay_x0 + g.bay_L),
                     "bay_inner_width": r(g.fus_w - 2 * P.fus_skin_t_m),
                     "bay_inner_height": r(g.fus_h - 2 * P.fus_skin_t_m),
                     "tail_end_width": r(0.3 * g.fus_w), "tail_end_height": r(0.4 * g.fus_h),
                     "stations": [[r(x), r(kw * g.fus_w), r(kh * g.fus_h)]
                                  for x, kw, kh in fuselage_profile(g.L_nose, g.bay_L, g.fus_L)],
                     "section_shape": "rounded box (superellipse exponent 2.6)"},
        "wing": surf_block(g.wing, x_qc, {
            "z_root_chord_line": r(z_wing), "incidence_deg": _f(tl["wing_incidence_deg"], 2),
            "washout_deg": _f(d["washout"], 2), "dihedral_deg": 0.0, "root_airfoil": root_af,
            "tip_airfoil": tip_af, "y_mac": r(y_mac), "x_le_mac": r(x_qc - 0.25 * g.wing.mac),
            "t_c_root": r(g.wing.t_c),
            "note": "unswept quarter-chord line; incidence puts the fuselage level at loiter; "
                    "washout linear root->tip, airfoil lofted linearly root->tip"}),
        "spar": {"od": r(wi["spar"]["D"]), "wall": r(wi["spar"]["t"]), "length": r(P.spar_span_frac * g.b),
                 "x": r(x_spar), "z": r(z_wing + wsh["camber_at_tmax"] * g.wing.c_root),
                 "xc_root": r(wsh["x_tmax"]), "xc_tip": r((x_spar - le(c_tip)) / c_tip),
                 "material": "carbon tube"},
        "ailerons": ailerons, "flaps": flaps,
        "htail": surf_block(ht, x_tail, {"z": 0.0, "incidence_deg": _f(tl["tail_incidence_deg"], 2),
                                         "airfoil": P.tail_airfoil}),
        "elevator": {"chord_fraction": el["cf"], "span_fraction": P.elevator_span_frac,
                     "hinge_x_root": r(hinge(x_tail, ht.c_root, el["cf"])),
                     "hinge_x_tip": r(hinge(x_tail, ht.c_tip, el["cf"])),
                     "throw_up_deg": _f(el["throw"], 2), "servo_throw_deg": P.elevator_max_deg},
        "vtail": surf_block(vt, x_tail, {"z_root": r(0.2 * g.fus_h), "height": r(vt.b),
                                         "airfoil": P.tail_airfoil}),
        "rudder": {"chord_fraction": cf_r, "span_fraction": P.rudder_span_frac,
                   "hinge_x_root": r(hinge(x_tail, vt.c_root, cf_r)),
                   "hinge_x_tip": r(hinge(x_tail, vt.c_tip, cf_r)),
                   "throw_needed_deg": _f(rudder_throw, 2), "servo_throw_deg": P.rudder_max_deg},
        "nacelles": {"y": r(g.y_eng), "z_axis": r(z_axis), "x_front": r(nac["x_front"]),
                     "x_rear": r(nac["x_rear"]), "diameter": r(g.nac_d), "length": r(g.nac_L)},
        "props": {"x": r(nac["x_prop"]), "y": r(g.y_eng), "z": r(z_axis), "diameter": r(g.prop_D),
                  "model": best.combo.prop,
                  "tip_to_fuselage": r(g.y_eng - g.prop_D / 2 - g.fus_w / 2),
                  "tip_to_ground": r(z_axis - g.prop_D / 2 - z_ground)},
        "cg": {"x_forward_limit": r(tl["x_cg_fwd"]), "x_aft_limit": r(tl["x_cg_aft"]),
               "x_neutral_point": r(tl["x_np"]), "x_nominal": r(bal["x_cg"]) if bal else None,
               "band": r(tl["x_cg_aft"] - tl["x_cg_fwd"]), "battery_slide": P.battery_slide_m,
               "note": "balance the finished aircraft so the CG falls in this range (slide the battery)"},
        "gear": {"z_ground": r(z_ground), "x_main": r(x_main), "y_main": r(g.track / 2),
                 "main_wheel_model": g.main_wheel.name, "main_wheel_d": r(g.main_wheel.d),
                 "main_wheel_w": r(g.main_wheel.w), "main_axle_mm": g.main_wheel.axle_mm,
                 "x_nose": r(x_nose), "nose_wheel_model": g.nose_wheel.name,
                 "nose_wheel_d": r(g.nose_wheel.d), "nose_wheel_w": r(g.nose_wheel.w),
                 "nose_axle_mm": g.nose_wheel.axle_mm,
                 "ground_stance_deg": _f(stance, 2),
                 "nose_gear_extra_length": r((x_main - x_nose) * math.tan(math.radians(stance))),
                 "gear_height": r(g.gear_h), "main_leg_length": r(g.main_leg_L),
                 "nose_strut_length": r(g.nose_strut_L), "strap_thickness": r(wi["gear_strap_t"]),
                 "strap_width": P.gear_plate_width_m, "nose_wire_d": r(wi["nose_wire_d"]),
                 "brake_model": g.brake.name if g.brake.active else None,
                 "brake_actuator": g.brake.actuator if g.brake.active else None,
                 "note": f"takeoff technique: {tl['takeoff_technique']}; main gear at "
                         f"{P.tipback_deg:.0f} deg tip-back behind the aft CG"},
        "bay_contents": {"battery": {"name": batt.name, "l": r(batt.dims_m[0]), "w": r(batt.dims_m[1]),
                                     "h": r(batt.dims_m[2]), "mass": r(batt.mass_kg)},
                         "components": comps,
                         "note": "CG budget assumes the bay contents' centroid at the bay centre; "
                                 "the battery slides +/- battery_slide to trim"},
        "servos": [{"surface": "ailerons", "count": 2, "location": "wing, at the aileron inboard end"},
                   {"surface": "elevator", "count": 1, "location": "fuselage bay"},
                   {"surface": "rudder", "count": 1, "location": "fuselage bay"},
                   {"surface": "nose steering", "count": 1, "location": "fuselage bay"}]
        + ([{"surface": "flaps", "count": 2, "location": "wing, flap inboard end"}] if flaps else [])
        + ([{"surface": "wheel brakes", "count": 1, "location": "fuselage bay (cables to both main wheels)"}]
           if g.brake.active else []),
    }


def controls_json(best):
    d = best.detail
    g, ct, tl, eo = d["geom"], d["controls"], d["tail"], d["engine_out"]
    lat = d.get("lateral") or {}
    semi = g.b / 2
    a = ct["aileron"]
    out = {"configuration": ct["config"],
           "ailerons": {"y_inboard_m": _f(a["y1"], 4), "y_outboard_m": _f(a["y2"], 4),
                        "eta_inboard": _f(a["y1"] / semi, 3), "eta_outboard": _f(a["y2"] / semi, 3),
                        "chord_fraction": a["cf"], "throw_deg": a["throw"],
                        "roll_helix_pb_2V": _f(a["pb2v"], 4), "requirement": P.roll_helix_min,
                        "is_flaperon": ct["config"] == "flaperons"},
           "flaps": None,
           "elevator": {"chord_fraction": tl["elevator"]["cf"], "throw_deg": tl["elevator"]["throw"],
                        "span_fraction": P.elevator_span_frac,
                        "authority_ratio": _f(tl["elevator"]["ratio"], 3),
                        "ratio_landing_trim": _f(tl["elevator"]["ratio_land"], 3),
                        "ratio_takeoff": _f(tl["elevator"]["ratio_takeoff"], 3)},
           "rudder": {"chord_fraction": lat.get("rudder_cf", eo.get("rudder_cf")),
                      "throw_usable_deg": _f((lat.get("engine_out") or {}).get("rudder_usable_deg",
                                                                              eo.get("rudder_defl_usable", 0)), 2),
                      "engine_out_deg": _f(abs((lat.get("engine_out") or {}).get("rudder_deg", 0.0)), 2),
                      "crosswind_deg": _f(abs((lat.get("crosswind") or {}).get("rudder_deg", 0.0)), 2),
                      "span_fraction": P.rudder_span_frac, "servo_throw_deg": P.rudder_max_deg},
           "stability": {"neutral_point_frac_mac": _f(tl["h_n"], 4),
                         "cg_forward_frac_mac": _f(tl["h_cg_fwd"], 4), "cg_aft_frac_mac": _f(tl["h_cg_aft"], 4),
                         "static_margin_range": [P.static_margin_min, _f(tl["h_n"] - tl["h_cg_fwd"], 4)],
                         "np_shift_fuselage": _f(tl["dh_n_fuselage"], 4),
                         "cm_alpha_fuselage_per_rad": _f(tl["cm_alpha_fuselage"], 4),
                         "cg_band_frac_mac": _f(tl["band_frac"], 4), "htail_sizing": P.htail_sizing,
                         "tail_volume_V_H": _f(tl["V_H"], 4), "downwash_gradient": _f(tl["deps_dalpha"], 4),
                         "wing_incidence_deg": _f(tl["wing_incidence_deg"], 2),
                         "tail_incidence_deg": _f(tl["tail_incidence_deg"], 2),
                         "decalage_deg": _f(tl["decalage_deg"], 2),
                         "takeoff_technique": tl["takeoff_technique"], "ground_stance_deg": _f(tl["stance_deg"], 2),
                         "stabilizer_area_m2": _f(g.htail.S, 5), "stabilizer_area_tail_volume_m2": _f(g.sh_volume, 5)}}
    if ct["flap"]:
        f = ct["flap"]
        out["flaps"] = {"y_inboard_m": _f(f["y0"], 4), "y_outboard_m": _f(f["y1"], 4),
                        "eta_inboard": _f(f["y0"] / semi, 3), "eta_outboard": _f(f["y1"] / semi, 3),
                        "chord_fraction": f["cf"], "landing_deflection_deg": f["deflection"],
                        "CLmax_landing": _f(f["CLmax_landing"], 4)}
    return out


def _parts(best):
    """Datasheet values of the selected parts (so the hand-off is self-contained)."""
    from catalog import motors as _motors, props as _props
    d = best.detail
    m, pr, b, e = _motors()[best.combo.motor], _props()[best.combo.prop], d["battery"], d["esc"]
    return {
        "motor": {"name": m.name, "kv_rpm_per_V": m.kv, "io_A": m.io, "rm_ohm": m.rm,
                  "imax_A": m.imax, "mass_kg": m.mass_kg, "od_m": m.od_m, "source": m.source},
        "prop": {"name": pr.name, "diameter_in": pr.d_in, "pitch_in": pr.pitch_in,
                 "mass_kg_estimate": _f(pr.mass_kg, 4), "source": "APC PER3 performance data"},
        "esc": {"name": e.name, "i_cont_A": e.i_cont, "mass_kg": e.mass_kg} if e else None,
        "battery": {"name": b.name, "chemistry": b.chemistry, "cells_series": b.s,
                    "parallel": b.p, "v_nom": b.v_nom, "energy_Wh": _f(b.energy_wh, 2),
                    "mass_kg": _f(b.mass_kg, 4), "specific_energy_Wh_kg": _f(b.u_wh_kg, 1),
                    "r_int_ohm": _f(b.r_int, 4), "i_max_A": _f(b.i_max, 1),
                    "dims_m": [_f(v, 4) for v in b.dims_m], "usable_fraction_K": b.k},
    }


def _clean(o):
    """JSON-safe copy (numpy -> float, drop heavy objects)."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if not callable(v)}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return _f(o, 6)
    if isinstance(o, float):
        return _f(o, 6) if math.isfinite(o) else None
    if isinstance(o, (str, int, bool)) or o is None:
        return o
    return str(o)


def to_json(best, alts, cdiag, sens, twist=None, wheels_=None, stall=None):
    d = best.detail
    g, perf, bmf = d["geom"], d["perf"], d["bmf"]
    conv = lambda v: (_f(v, 6) if isinstance(v, (float, np.floating, int)) and not isinstance(v, bool) else v)

    def surf(s):
        return {k: _f(v, 5) for k, v in vars(s).items()}

    return {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "takeoff_mass_kg": _f(best.m, 4),
        "wing_loading_N_m2": _f(best.m * G0 / g.S, 2),
        "power_loading_W_N": _f(cdiag["design"][1], 3),
        "selection": {"wing_airfoil": best.combo.airfoil,
                      "wing_tip_airfoil": best.combo.tip_airfoil or best.combo.airfoil,
                      "wing_washout_deg": _f(d["washout"], 3), "control_config": best.combo.controls,
                      "tail_airfoil": P.tail_airfoil,
                      "wing_construction": best.combo.construction, "motor": best.combo.motor,
                      "prop": best.combo.prop, "esc": d["esc"].name, "battery": d["battery"].name},
        "geometry": {"wing": surf(g.wing), "htail": surf(g.htail), "vtail": surf(g.vtail),
                     "tail_arm_m": _f(g.l_t, 4), "fuselage_length_m": _f(g.fus_L, 4),
                     "fuselage_width_m": _f(g.fus_w, 4), "fuselage_height_m": _f(g.fus_h, 4),
                     "motor_station_m": _f(g.y_eng, 4), "prop_diameter_m": _f(g.prop_D, 4),
                     "nacelle_diameter_m": _f(g.nac_d, 4), "nacelle_length_m": _f(g.nac_L, 4),
                     "gear_height_m": _f(g.gear_h, 4), "gear_track_m": _f(g.track, 4),
                     "spar_od_m": _f(d["winfo"]["spar"]["D"], 5), "spar_wall_m": _f(d["winfo"]["spar"]["t"], 5)},
        "aero": {"CD0": _f(cdiag["CD0"], 5), "K": _f(cdiag["K"], 5), "e_inviscid": _f(d["aero"].e, 4),
                 "CLmax": _f(perf["CLmax"], 4), "CL_alpha_per_rad": _f(d["aero"].CL_alpha, 4),
                 "V_stall_ms": _f(perf["V_s"]), "V_max_ms": _f(perf["V_max"]),
                 "loiter": {k: conv(v) for k, v in perf["loiter"].items() if k != "drag_parts"},
                 "loiter_drag_breakdown": {k: _f(v, 6) for k, v in perf["loiter"]["drag_parts"].items()},
                 "climb": {k: conv(v) for k, v in perf["climb"].items()},
                 "best_glide": {k: conv(v) for k, v in perf["glide"].items()}},
        "throttle_cap": _f(d["throttle_cap"], 4),
        "loads": {"n_limit": _f(d["winfo"]["n_limit"], 3), "n_ultimate": _f(d["winfo"]["n_ult"], 3),
                  "V_C_ms": _f(d["V_C"], 3)},
        "controls": controls_json(best),
        "twist_study": twist if twist else None,
        "thrust_in_flight": flight_thrust_table(best),
        "parts": _parts(best),
        "engine_out": {k: _f(v, 4) for k, v in d["engine_out"].items()},
        "objective": {"score_kg": _f(best.score, 5), "mass_kg": _f(best.m, 5),
                      "stall_speed_landing_ms": _f(best.vs, 4),
                      "stall_speed_weight_g_per_ms": P.stall_speed_weight_g_per_ms},
        "lateral": _clean({k: v for k, v in (d.get("lateral") or {}).items() if k != "cons"}),
        "balance": _clean({k: v for k, v in (d.get("balance") or {}).items()}),
        "landing": _clean({"approach": perf["approach"], "crow": perf.get("crow"),
                           "V_stall_landing_ms": perf["V_s_landing"], "V_touchdown_ms": perf["V_td"],
                           "CLmax_landing_wing": perf["CLmax_landing"],
                           "CLmax_landing_trimmed": perf["CLmax_landing_trimmed"],
                           "roll_grass_m": perf["s_landing"], "roll_pavement_m": perf["s_landing_pavement"],
                           "roll_grass_without_crow_m": perf.get("s_landing_nocrow"),
                           "mu_grass": perf["mu_grass"], "mu_pavement": perf["mu_pavement"]}),
        "wheel_study": _clean(wheels_) if wheels_ else None,
        "stall_study": _clean(stall) if stall else None,
        "performance": {"takeoff_roll_m": _f(perf["takeoff"]["s"], 2),
                        "takeoff_roll_pavement_m": _f(perf["takeoff_pavement"]["s"], 2),
                        "landing_roll_m": _f(perf["s_landing"], 2),
                        "landing_roll_pavement_m": _f(perf["s_landing_pavement"], 2),
                        "stall_speed_landing_ms": _f(perf["V_s_landing"], 3),
                        "stall_onset_semispan_clean": _f(perf["stall_eta"], 3),
                        "stall_onset_semispan_landing": _f(perf["stall_eta_landing"], 3),
                        "CLmax_trimmed_clean": _f(perf["CLmax_trimmed"], 4),
                        "static_thrust_N": _f(perf["static"]["T"]),
                        "max_motor_current_A": _f(perf["I_motor_max"]),
                        "max_battery_current_A": _f(perf["I_batt_max"])},
        "mission_bmf": {k: conv(v) for k, v in bmf.items()},
        "mass_breakdown_kg": {k: _f(v, 5) for k, v in d["masses"].items()},
        "constraint_margins": {k: _f(v, 4) for k, v in best.cons.items()},
        "layout": layout(best),
        "design_variables": dict(zip(sizing.X_NAMES, [_f(v, 5) for v in best.x])),
        "design_variable_bounds": dict(zip(sizing.X_NAMES, [list(b) for b in sizing.X_BOUNDS])),
        "alternatives": [{"mass_kg": _f(r.m, 4), "score_kg": _f(getattr(r, "score", r.m), 4),
                          "label": r.combo.label()} for r in alts],
        "sensitivity": [{"param": r["param"], "base": r["base"],
                         "cases": [{k: conv(v) for k, v in c.items()} for c in r["cases"]]} for r in sens],
        "inputs": [{"name": p.name, "value": p.value, "units": p.units, "tag": p.tag,
                    "source": p.source, "measure": p.measure} for p in P.all()],
    }


def _load_design_point():
    """Re-build the saved design (parts + design variables) for --evaluate."""
    with open(os.path.join(OUT, "design_point.json")) as f:
        j = json.load(f)
    s = j["selection"]
    batt = {b.name: b for b in real_batteries()}[s["battery"]]
    combo = sizing.Combo(s["wing_airfoil"], s["wing_construction"], s["motor"], s["prop"], battery=batt,
                         controls=s.get("control_config", "flaps"),
                         tip_airfoil=s.get("wing_tip_airfoil"), washout=s.get("wing_washout_deg", 0.0))
    x = [j["design_variables"][k] for k in sizing.X_NAMES if k in j["design_variables"]]
    alts = [SimpleNamespace(m=a["mass_kg"], combo=SimpleNamespace(label=(lambda l=a["label"]: l)))
            for a in j.get("alternatives", [])]
    return sizing.evaluate(x, combo), alts


def main(argv):
    """--evaluate : skip the search; re-evaluate the saved design point with the
    current parameters/data (use after replacing an estimate with a measurement)."""
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    if "--evaluate" in argv:
        best, alts = _load_design_point()
        print("[--evaluate] re-evaluated the saved design point (no search)")
    else:
        final = sizing.run(verbose=True)
        # safety net: re-evaluate from scratch and keep the first candidate that
        # still passes every constraint (guards against search/report mismatch)
        for i, cand in enumerate(final):
            best = sizing.evaluate(cand.x, cand.combo)
            if best.feasible:
                break
            print(f"[!] candidate {i} failed re-evaluation ({cand.combo.label()}), trying next")
        final = final[i:]
        seen, alts = {(best.combo.motor, best.combo.prop, best.combo.battery.name)}, []
        for r in final[1:]:
            k = (r.combo.motor, r.combo.prop, r.combo.battery.name)
            if k not in seen:
                seen.add(k)
                alts.append(r)
            if len(alts) == 5:
                break
    runtime = time.time() - t0
    twist = stall = None
    if "--evaluate" in argv:              # keep the previous run's trade studies unless recomputed
        try:
            with open(os.path.join(OUT, "design_point.json")) as f:
                prev = json.load(f)
            twist, stall = prev.get("twist_study"), prev.get("stall_study")
        except (OSError, ValueError):
            pass
    if "--evaluate" not in argv or "--twist" in argv:
        print("[F ] twist trade study (tip airfoil x washout, each re-optimized) ...")
        twist = sizing.twist_study(best)
    if "--evaluate" not in argv or "--stall" in argv:
        print("[G ] stall-speed trade study (lightest design at each stall-speed cap) ...")
        stall = sizing.stall_study(best)
    print("[H ] wheel trade study ...")
    wheels_ = wheel_study(best)
    print("[E ] sensitivity study ...")
    sens = sensitivity(best)
    cdiag = constraints.compute(best)
    constraints.plot(cdiag, os.path.join(OUT, "constraint_diagram.png"),
                     f"Constraint diagram - {best.m:.2f} kg, {best.combo.motor} + {best.combo.prop}")
    with open(os.path.join(OUT, "design_point.json"), "w") as f:
        json.dump(to_json(best, alts, cdiag, sens, twist, wheels_, stall), f, indent=2, default=str)
    report(best, alts, cdiag, sens, runtime, twist, wheels_, stall)
    import print_design
    import print_dimensions
    import print_parts
    import print_performance
    jp = os.path.join(OUT, "design_point.json")
    print_design.write(jp, os.path.join(OUT, "design_parameters.txt"))
    print_dimensions.write(jp, os.path.join(OUT, "dimensions.txt"))
    print_dimensions.write_airfoils(jp, os.path.join(OUT, "airfoils"))
    print_parts.write(jp, os.path.join(OUT, "parts_list.txt"), os.path.join(OUT, "parts_list.csv"))
    print_performance.write(jp, os.path.join(OUT, "performance.txt"))
    print(f" wrote {os.path.relpath(OUT)}/design_point.json, design_parameters.txt, dimensions.txt, "
          f"parts_list.txt/.csv, performance.txt, airfoils/*.dat, constraint_diagram.png")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
