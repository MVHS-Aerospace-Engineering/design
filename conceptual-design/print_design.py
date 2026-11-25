"""
Print EVERY final sizing variable of the conceptual design point, followed by
every input ratio / coefficient / value used to calculate them (with source).

    .venv/bin/python conceptual-design/print_design.py            (prints)
    .venv/bin/python conceptual-design/print_design.py -o out.txt  (also saves)

Reads outputs/design_point.json (written by conceptual.py). conceptual.py also
saves this printout automatically as outputs/design_parameters.txt.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, "outputs", "design_point.json")

SURFACE_UNITS = {"S": "m^2", "b": "m", "taper": "-", "c_root": "m", "c_tip": "m", "mac": "m",
                 "t_c": "-", "S_wet": "m^2"}
SURFACE_NAMES = {"S": "area", "b": "span (fin: height)", "taper": "taper ratio", "c_root": "root chord",
                 "c_tip": "tip chord", "mac": "mean aerodynamic chord", "t_c": "thickness/chord",
                 "S_wet": "wetted area"}


def _fmt(v):
    if isinstance(v, float):
        if v == 0:
            return "0"
        return f"{v:.6g}"
    if isinstance(v, list):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return str(v)


def render(j):
    L = []
    w = L.append

    def head(t):
        w("")
        w("=" * 86)
        w(f" {t}")
        w("=" * 86)

    def kv(k, v, unit="", note=""):
        w(f"  {k:44s} {_fmt(v):>16s} {unit:8s} {note}")

    def block(title, d, skip=()):
        w(f" -- {title}")
        for k, v in d.items():
            if k in skip or isinstance(v, dict):
                continue
            kv(k, v)

    w(f"CONCEPTUAL DESIGN PARAMETERS   (generated {j.get('generated', '?')})")
    w("Source: conceptual-design/outputs/design_point.json")

    # 1 ---------------------------------------------------------------- summary
    head("1. SUMMARY")
    kv("takeoff mass", j["takeoff_mass_kg"], "kg")
    if j.get("objective"):
        o = j["objective"]
        kv("objective = mass + w x V_stall", o["score_kg"], "kg",
           f"w = {o['stall_speed_weight_g_per_ms']} g per m/s")
        kv("stall speed (landing config, trimmed)", o["stall_speed_landing_ms"], "m/s")
    kv("wing loading W/S", j["wing_loading_N_m2"], "N/m^2")
    kv("installed shaft power / weight (capped)", j["power_loading_W_N"], "W/N")
    if "throttle_cap" in j:
        kv("throttle cap (max ESC duty)", j["throttle_cap"], "-", "set on both motor outputs in the FC")

    # 2 ---------------------------------------------------------------- variables
    head("2. DESIGN VARIABLES")
    w(" -- chosen by the optimizer (SLSQP)")
    bounds = j.get("design_variable_bounds", {})
    for k, v in j["design_variables"].items():
        b = bounds.get(k)
        kv(k, v, "", f"bounds {b}" if b else "")
    w(" -- sized inside the convergence loop")
    if "throttle_cap" in j:
        kv("throttle cap", j["throttle_cap"], "-", "smallest meeting takeoff + climb target x margin")
    g = j["geometry"]
    kv("fin area", g["vtail"]["S"], "m^2", "max(tail-volume, coupled engine-out trim)")
    kv("stabilizer area", g["htail"]["S"], "m^2", "scissor plot: stability at aft CG, elevator at forward CG")
    if j.get("balance"):
        kv("wing station x_qc (CG mid-band)", j["balance"].get("x_qc"), "m")
    kv("battery (snapped to a real pack)", j["selection"]["battery"])
    kv("spar OD / wall", f"{g['spar_od_m'] * 1000:.2f} / {g['spar_wall_m'] * 1000:.2f}", "mm")
    w(" -- discrete choices (searched over the catalogs)")
    for k, v in j["selection"].items():
        kv(k, v)

    # 3 ---------------------------------------------------------------- parts
    if "parts" in j:
        head("3. SELECTED PARTS (datasheet values)")
        for part, d in j["parts"].items():
            if d:
                block(part, d)

    # 4 ---------------------------------------------------------------- geometry
    head("4. GEOMETRY")
    for surf, title in (("wing", "main wing"), ("htail", "horizontal tail"), ("vtail", "vertical tail (fin)")):
        w(f" -- {title}")
        for k, v in g[surf].items():
            kv(f"{SURFACE_NAMES.get(k, k)} ({k})", v, SURFACE_UNITS.get(k, ""))
        if surf == "wing":
            kv("aspect ratio", g["wing"]["b"] ** 2 / g["wing"]["S"], "-")
    block("fuselage, nacelles, props, gear, spar",
          {k: v for k, v in g.items() if k not in ("wing", "htail", "vtail")})
    if "layout" in j:
        w(" -- layout (x aft from nose tip, y right, z up; metres)")
        for part, d in j["layout"].items():
            if isinstance(d, dict):
                for k, v in d.items():
                    if k != "note" and not isinstance(v, (dict, list)):
                        kv(f"{part}.{k}", v)
        w("  (full CAD dimensions in mm: outputs/dimensions.txt)")

    # 4b --------------------------------------------------------------- controls
    if j.get("controls"):
        head("4b. CONTROL SURFACES & STABILITY")
        ct = j["controls"]
        kv("configuration", ct["configuration"])
        for part in ("ailerons", "flaps", "elevator", "rudder", "stability"):
            if ct.get(part):
                block(part, ct[part])
    if j.get("balance"):
        head("4d. BALANCE (CG budget -> CG band -> scissor plot)")
        b = j["balance"]
        for k in ("m", "x_cg", "h_cg", "sigma_x", "slide_authority", "residual", "band_m", "band_frac_mac",
                  "x_qc", "x_qc_target", "h_mid", "clamped"):
            kv(k, b.get(k))
        w(f"  {'item':38s} {'mass kg':>9s} {'x m':>8s} {'sigma':>6s}  moves with wing")
        for it in b.get("items", []):
            w(f"  {it['name'][:38]:38s} {it['m']:9.4f} {it['x']:8.4f} {it['sigma']:6.2f}  {it['wing']}")
    if j.get("lateral"):
        head("4e. COUPLED LATERAL-DIRECTIONAL TRIM")
        la = j["lateral"]
        kv("rudder chord fraction", la.get("rudder_cf"))
        kv("fin area needed by engine-out", la.get("sv_needed"), "m^2")
        for part in ("engine_out", "crosswind", "derivatives"):
            if la.get(part):
                block(part, la[part])
    if j.get("dynamics"):
        head("4e2. LATERAL-DIRECTIONAL DYNAMICS (MIL-F-8785C Level 1, Class I)")
        dy = j["dynamics"]
        kv("dihedral (each panel)", dy["dihedral_deg"], "deg", "smallest meeting the spiral requirement")
        kv("Ixx", dy["Ixx"], "kg.m^2")
        kv("Izz", dy["Izz"], "kg.m^2")
        for name, c in dy["cases"].items():
            block(f"{name} modes", {k: v for k, v in c.items() if k not in ("eigenvalues", "derivatives")})
            block(f"{name} derivatives (per rad)", c["derivatives"])
    if j.get("landing"):
        head("4f. LANDING / APPROACH")
        ld = j["landing"]
        block("landing", ld)
        if ld.get("approach"):
            block("approach + balked landing", ld["approach"])
            if ld["approach"].get("trim"):
                block("approach trim (forward CG)", ld["approach"]["trim"])
        for k in ("crow", "mu_grass", "mu_pavement"):
            if ld.get(k):
                block(k, ld[k])
    if j.get("stall_study"):
        head("4g. STALL-SPEED TRADE STUDY (lightest design at each stall-speed cap)")
        w(f"  {'cap':>5s} {'V_s':>6s} {'mass':>7s} {'score':>7s} {'S':>7s} {'AR':>5s} {'span':>6s} status")
        for r in j["stall_study"]:
            if not r.get("ok"):
                w(f"  {r['cap']:5.1f}  (did not converge)")
                continue
            st = "ok" if r["feasible"] else "misses " + "+".join(r["misses"])
            w(f"  {r['cap']:5.1f} {r['V_s_landing']:6.2f} {r['m']:7.4f} {r['score']:7.4f} {r['S']:7.4f} "
              f"{r['AR']:5.2f} {r['span']:6.3f} {st}")
    if j.get("wheel_study"):
        head("4h. WHEEL TRADE STUDY (design frozen)")
        for r in j["wheel_study"]:
            if r.get("ok"):
                w(f"  {r['wheel']:14s} d {r['d_mm']:5.1f} mm  mass {r['m']:.4f} kg  grass mu {r['mu_grass']:.3f}  "
                  f"TO {r['takeoff_grass_m']:.1f} m  LDG {r['landing_grass_m']:.1f} m  brake fits {r['brake_fits']}"
                  + ("  <- chosen" if r["chosen"] else ""))
    if j.get("taper_study"):
        head("4i. TAPER TRADE STUDY (fixed taper, each re-optimized)")
        for r in j["taper_study"]:
            if r.get("ok"):
                st = "ok" if r["feasible"] else "misses " + "+".join(r["misses"])
                w(f"  taper {r['taper']:4.2f}  mass {r['m']:.4f}  score {r['score']:.4f}  span {r['span']:.3f}  "
                  f"root/tip {r['c_root'] * 1000:.0f}/{r['c_tip'] * 1000:.0f} mm  tip Re {r['tip_re'] / 1e3:.0f}k  "
                  f"V_s {r['V_s_landing']:.2f}  stall@ {r['stall_eta']:.0%}  {st}")
    if j.get("twist_study"):
        head("4c. TWIST TRADE STUDY (each row re-optimized)")
        w(f"  {'tip airfoil':11s} {'washout':>7s} {'mass':>7s} {'span':>6s} {'loiter':>7s} {'landing':>8s} "
          f"{'CLmax':>6s} {'stall@':>7s} {'e':>6s}  status")
        for r in j["twist_study"]:
            if not r.get("ok"):
                w(f"  {r['tip']:11s} {r['washout']:7.1f}  (did not converge)")
                continue
            st = "ok" if r["feasible"] else "misses " + "+".join(r["misses"])
            w(f"  {r['tip']:11s} {r['washout']:7.1f} {r['m']:7.4f} {r['span']:6.3f} {r['loiter_min']:6.1f}m "
              f"{r['landing_m']:7.1f}m {r['CLmax']:6.3f} {r['stall_eta']:6.0%} {r['e']:6.3f}  {st}")

    # 5 ---------------------------------------------------------------- aero
    head("5. AERODYNAMICS")
    a = j["aero"]
    block("coefficients & speeds", a)
    block("loiter point", a["loiter"])
    block("loiter drag breakdown (CD, referenced to wing area)", a["loiter_drag_breakdown"])
    block("best-rate climb point (at throttle cap)", a["climb"])
    block("best glide", a["best_glide"])

    # 6 ---------------------------------------------------------------- performance
    head("6. PERFORMANCE")
    block("field / currents", j["performance"])
    if "engine_out" in j:
        block("engine-out (one motor failed)", j["engine_out"])
    if "thrust_in_flight" in j:
        w(" -- in-flight thrust (thrust AVAILABLE at the throttle cap vs. level-flight drag)")
        w(f"  {'point':40s} {'V':>6s} {'T_avail':>8s} {'T/W':>6s} {'drag':>7s} {'excess':>7s} {'eta_p':>6s} {'rpm':>6s}")
        for r in j["thrust_in_flight"]:
            f = lambda x, s: (s % x) if x is not None else "-"
            w(f"  {r['point']:40s} {r['V_ms']:6.2f} {r['thrust_avail_N']:8.3f} {r['thrust_per_weight']:6.3f} "
              f"{f(r['drag_level_N'], '%7.3f'):>7s} {f(r['excess_N'], '%7.3f'):>7s} "
              f"{f(r['eta_prop'], '%6.3f'):>6s} {r['rpm']:6.0f}")

    # 7 ---------------------------------------------------------------- mission
    head("7. MISSION BATTERY-MASS-FRACTION BUILD-UP")
    block("segments", j["mission_bmf"])

    # 8 ---------------------------------------------------------------- mass
    head("8. MASS BREAKDOWN")
    tot = j["takeoff_mass_kg"]
    for k, v in sorted(j["mass_breakdown_kg"].items(), key=lambda kv_: -kv_[1]):
        kv(k, v, "kg", f"{v / tot * 100:5.1f}%")

    # 9 ---------------------------------------------------------------- margins
    head("9. CONSTRAINT MARGINS  (>= 0 satisfied; ~0 = ACTIVE, i.e. this requirement sizes the design)")
    for k, v in j["constraint_margins"].items():
        kv(k, v, "", "ACTIVE" if abs(v) < 0.02 else "")

    # 10/11 ------------------------------------------------------------ inputs
    inputs = j.get("inputs", [])
    ratios = [p for p in inputs if p["units"] == "-"]
    others = [p for p in inputs if p["units"] != "-"]
    head("10. INPUT RATIOS & COEFFICIENTS USED TO CALCULATE THE ABOVE (dimensionless)")
    for p in ratios:
        kv(p["name"], p["value"], "-", f"[{p['tag']}{', MEASURE' if p['measure'] else ''}] {p['source']}")
    head("11. OTHER INPUTS (dimensional)")
    for p in others:
        kv(p["name"], p["value"], p["units"], f"[{p['tag']}{', MEASURE' if p['measure'] else ''}] {p['source']}")
    w("")
    w("Parts catalogs: conceptual-design/data/*.csv (motors, lipo_packs, liion_cells, escs, components, materials, wheels, brakes)")
    return "\n".join(L)


def write(json_path=DEFAULT, out_path=None):
    with open(json_path) as f:
        text = render(json.load(f))
    if out_path:
        with open(out_path, "w") as f:
            f.write(text + "\n")
    return text


if __name__ == "__main__":
    args = sys.argv[1:]
    out = args[args.index("-o") + 1] if "-o" in args else None
    print(write(DEFAULT, out))
