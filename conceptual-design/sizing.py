"""
The sizing loop.

evaluate(x, combo)  - one design: converges takeoff mass with a fixed-point
                      loop   geometry -> aero -> powertrain -> performance ->
                      mission BMF -> weights -> new W_TO   (|dm| < 0.1 g),
                      together with the fin (coupled lateral trim), the
                      stabilizer (scissor plot), the CG band and the wing
                      station. Returns the objective + every constraint.

Objective (Result.score): takeoff mass + P.stall_speed_weight_g_per_ms x the
trimmed landing-configuration stall speed. Mass alone always shrinks the wing
to the fastest stall speed any constraint allows.

run()               - staged search:
   A  airframe technology (wing airfoil x construction in P.wing_constructions), screened with
      reference powertrains
   B  every motor x prop x battery-family combination, screened in parallel
      on a 2-D (wing area x aspect ratio) grid
   C  multi-start SLSQP over (S, AR, tail-arm) for the 24 best candidates,
      continuous battery
   D  snap to REAL catalog batteries and re-optimize with the pack fixed
"""
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize

import requirements  # noqa: F401  (registers requirement parameters)
from aerodynamics import Aero, WingSections, lift_slope_3d
import dynamics
import lateral
from controls import tail_and_elevator, wing_controls
from atmosphere import G0, isa
from catalog import (airfoil, battery_families, motors, pick_esc, props,
                     wing_airfoil_names)
from geometry import build
from mission import mission_bmf, performance
from parameters import P
from propulsion import V_GRID, Powertrain
from weights import (avionics_power_w, cg_budget, component_stations,
                     design_cruise_speed, gust_load_factor, mass_breakdown)

X_BOUNDS = [(0.04, 0.60), (4.0, 20.0), (1.8, 5.0), (0.0, 6.0), (0.35, 1.0)]   # S, AR, tail arm/MAC, washout deg, taper
X_NAMES = ("S_m2", "AR", "tail_arm_over_mac", "washout_deg", "taper")
CONS_KEYS = ("takeoff_roll_grass", "takeoff_roll_pavement", "landing_roll_grass", "landing_roll_pavement",
             "span", "climb_rate", "climb_gradient", "balked_landing", "motor_current", "battery_current",
             "esc_available", "motor_station", "spar_fits", "tip_reynolds", "roll_helix",
             "elevator_authority", "crosswind_rudder", "crosswind_aileron", "engine_out_aileron",
             "cn_beta", "dutch_roll", "spiral", "roll_mode", "short_period", "phugoid", "stall_speed", "brake_fits", "nose_gear_fits",
             "prop_data_range", "converged")
N_CONS = len(CONS_KEYS)   # constraints for a continuous battery (+1 endurance for a real pack)


@dataclass
class Combo:
    airfoil: str
    construction: str
    motor: str
    prop: str
    family: object = None          # BatteryFamily (continuous battery)
    battery: object = None         # real Battery (fixed)
    controls: str = "flaps"        # "none" (ailerons only) | "flaps" (+ailerons) | "flaperons"
    tip_airfoil: str = None        # None = same as root (no aerodynamic twist)
    washout: float = 0.0           # deg, used when x has only 3 variables
    taper: float = None            # used when x has fewer than 5 variables (None = P.wing_taper)

    def label(self):
        b = self.battery.name if self.battery is not None else self.family.name + " (continuous)"
        wing = self.airfoil if not self.tip_airfoil or self.tip_airfoil == self.airfoil \
            else f"{self.airfoil}->{self.tip_airfoil}"
        return f"{self.motor} + {self.prop} | {b} | {wing} {self.construction} | {self.controls}"


@dataclass
class Result:
    ok: bool
    m: float
    x: tuple
    combo: Combo
    cons: dict = field(default_factory=dict)
    detail: dict = field(default_factory=dict)
    vs: float = float("nan")        # trimmed landing-configuration stall speed [m/s]

    @property
    def score(self):
        """Objective [kg]: mass + (stall-speed worth) x V_stall."""
        if not math.isfinite(self.m):
            return math.inf
        w = P.stall_speed_weight_g_per_ms / 1000.0
        return self.m + (w * self.vs if math.isfinite(self.vs) else 0.0)

    @property
    def feasible(self):
        return self.ok and all(v >= -1e-4 for v in self.cons.values())

    @property
    def violation(self):
        return sum(max(-v, 0.0) for v in self.cons.values()) if self.cons else 1e3


def atmosphere():
    return isa(P.field_elevation_m, P.isa_temp_offset_K)


def throttle_cap_update(cap, perf, pt, W):
    """Smallest throttle cap meeting the takeoff roll, climb-rate target and
    climb gradient (x margin). Thrust ~ duty^2, so cap *= sqrt(k_required)."""
    ft = pt.full_throttle()
    V = V_GRID
    ok = (V >= perf["V_min"]) & (ft["T"] > 0.05 * ft["T"][0])
    if not ok.any():
        return cap
    T, D, Vo = ft["T"][ok], perf["D_curve"][ok], V[ok]
    roc_req = max(P.min_roc_ms, P.climb_rate_target_ms)
    k_roc = np.min((W * roc_req + D * Vo) / (T * Vo))
    k_grad = np.min((W * P.min_climb_gradient + D) / T)
    k_to = 0.0
    for to, s_max in ((perf["takeoff"], P.takeoff_roll_max_grass_m),
                      (perf["takeoff_pavement"], P.takeoff_roll_max_pavement_m)):
        V_lo = to["V_lo"]
        L_eff = s_max - V_lo * P.rotation_time_s
        if math.isfinite(to["s"]):
            s_eff = max(to["s"] - V_lo * P.rotation_time_s, 1e-3)
            resist = to["T07"] / W - V_lo ** 2 / (2 * G0 * s_eff)       # mean resistance / W
            k_to = max(k_to, (V_lo ** 2 / (2 * G0 * L_eff) + resist) / (to["T07"] / W))
        else:
            k_to = 2.0
    # balked landing (landing configuration, approach speed)
    ap = perf["approach"]
    k_bl = (W * P.balked_landing_gradient_min + ap["D"]) / max(ap["T"], 1e-6)
    k = max(k_roc, k_grad, k_to, k_bl) * P.throttle_cap_margin
    return float(np.clip(cap * math.sqrt(max(k, 1e-6)), 0.15, 1.0))


FREEZE_IT = 6     # iterations before discrete control-surface picks are frozen
CONTROL_CONFIGS = ("none", "flaps", "flaperons")


def evaluate(x, combo, max_iter=100, fast=False):
    """x = (S, AR, tail arm / MAC[, washout deg[, taper]]).
    fast=True: screening mode - no control-surface / elevator sizing (ailerons
    only, Raymer tails, rudder-only engine-out, wing over the bay centre, level
    ground stance). Used only to RANK powertrains."""
    S, AR, lt = (float(v) for v in x[:3])
    washout = float(x[3]) if len(x) > 3 else float(combo.washout)
    taper = float(x[4]) if len(x) > 4 else (combo.taper if combo.taper is not None else P.wing_taper)
    atm = atmosphere()
    secs = WingSections(airfoil(combo.airfoil), airfoil(combo.tip_airfoil or combo.airfoil))
    taf = airfoil(P.tail_airfoil)
    motor, prop = motors()[combo.motor], props()[combo.prop]
    p_av = avionics_power_w()
    rubber = combo.battery is None
    batt = combo.family.make(40.0) if rubber else combo.battery

    m, sv_eo, sh_trim, n_lim, V_C, cap = 1.5, 0.0, 0.0, P.n_limit_maneuver, 0.0, 1.0
    eo_info, lat, ctrl, tail, bal = {}, None, None, None, None
    ctrl_choice = tail_choice = rudder_cf = None
    flaps_prev, ground_cl = None, None
    x_qc, band, trim, dihedral, dyn = None, None, None, 0.0, None
    perf = bmf = None
    converged = False
    for it in range(max_iter):
        frozen = it >= FREEZE_IT
        geom = build(S, AR, lt, secs, motor, prop, batt, sv_eo, washout, sh_trim, x_qc, taper, dihedral)
        aero = Aero(geom, secs, taf, atm, flaps_prev, trim)
        if not fast:
            ctrl = wing_controls(geom, aero, m * G0, combo.controls, ctrl_choice if frozen else None)
            flaps_prev = aero.flaps
        pt = Powertrain(motor, prop, batt, atm.rho, p_av, cap)
        ft = pt.full_throttle()
        esc = pick_esc(float(np.max(ft["I_motor"])), batt.s)
        masses, winfo = mass_breakdown(geom, secs, combo.construction, motor, prop, esc, batt, m, n_lim,
                                       span_load=aero.span_load(),
                                       n_flap_servos=2 if (combo.controls == "flaps" and not fast) else 0)
        perf = performance(geom, aero, pt, m, batt, ground_cl, ctrl=ctrl, tail=tail)
        if perf is None:
            return Result(False, math.inf, tuple(x), combo, detail=dict(reason="no loiter speed"))
        bmf = mission_bmf(perf, batt, p_av)
        if bmf is None:
            return Result(False, math.inf, tuple(x), combo,
                          detail=dict(reason="cannot take off/climb"))
        # ---- couplings
        W = m * G0
        if fast:
            sv_new, eo_info = lateral.fin_engine_out_simple(geom, pt, perf["V_s"], atm)
            sh_new = 0.0
        else:
            # rudder chord: the whole grid is compared once (at the freeze iteration);
            # before that a mid-grid value, after it the frozen pick
            cf_use = rudder_cf if frozen else (None if it == FREEZE_IT - 1 else P.rudder_cf_grid[len(P.rudder_cf_grid) // 2])
            sv_new, lat = lateral.fin_requirements(geom, aero, pt, perf, ctrl, atm, W, cf_use)
            sv_new = float(np.clip(sv_new, 0.5 * geom.vtail.S, 2.0 * geom.vtail.S))
            eo = lat["engine_out"] or {}
            eo_info = dict(V_mc=eo.get("V", P.vmc_ratio * perf["V_s"]), T_one=eo.get("T_one", 0.0),
                           yaw_moment=eo.get("yaw_moment", 0.0), rudder_cf=lat["rudder_cf"],
                           rudder_defl_usable=eo.get("rudder_usable_deg", 0.0),
                           rudder_deg=eo.get("rudder_deg", 0.0), bank_deg=eo.get("phi_deg", 0.0),
                           sideslip_deg=eo.get("beta_deg", 0.0), aileron_frac=eo.get("aileron_frac", 0.0),
                           rudder_ratio=eo.get("ratio", 0.0))
            tail = tail_and_elevator(geom, aero, perf, pt, m, tail_choice if frozen else None, band)
            sh_new = float(np.clip(tail["sh_needed"] * 1.01, 0.5 * geom.htail.S, 2.0 * geom.htail.S))
            ground_cl = tail["CL_ground"]             # nose-up stance (or None: level)
            # balance: CG scatter -> CG band; wing station that puts the CG mid-band
            items = component_stations(geom, masses, winfo, batt, tail["x_main_gear"],
                                       0.8 * geom.L_nose, combo.airfoil, motor, esc, prop)
            bal = cg_budget(geom, items, tail["h_mid"])
            band_new = bal["band_frac_mac"]
            # lateral-directional dynamics: smallest dihedral meeting the spiral requirement
            dih_new, dyn = dynamics.size_dihedral(geom, aero, perf, atm, m, items, bal["x_cg"],
                                                  lat["rudder_cf"], ctrl["aileron"] if ctrl else None)
            dyn["longitudinal"] = dynamics.longitudinal_modes(geom, aero, perf, tail, items, m, atm)
            x_qc_new = bal["x_qc_target"]
            trim = dict(h=tail["h_cg_fwd"], c_lt=geom.wing.mac / geom.l_t)
        if it == FREEZE_IT - 1:
            ctrl_choice = ctrl["choice"] if ctrl else None
            tail_choice = tail["choice"] if tail else None
            rudder_cf = (lat or {}).get("rudder_cf") or eo_info.get("rudder_cf")
        cap_new = throttle_cap_update(cap, perf, pt, W)
        W_S = W / S
        V_C = design_cruise_speed(W_S, perf["V_max"])
        n_new = max(P.n_limit_maneuver,
                    gust_load_factor(W_S, V_C, perf["V_s"], atm.rho, geom.wing.mac, aero.CL_alpha))
        m_new = sum(masses.values())
        if rubber:
            e_req = bmf["energy_required_wh"]
            batt_new = combo.family.make(e_req)
            m_new = m_new - batt.mass_kg + batt_new.mass_kg
            de = abs(batt_new.energy_wh - batt.energy_wh)
            batt = batt_new
        else:
            de = 0.0
        dm = abs(m_new - m)
        m = 0.5 * m + 0.5 * m_new if dm > 0.05 else (m_new if it < 20 else (1 - w_r) * m + w_r * m_new)
        # under-relaxation (tightened late in the loop as a safety net against small
        # limit cycles from kinks such as the bank limit or trusted-throw ends)
        w_r = 0.5 if it < 60 else 0.25
        dsv, dsh = abs(sv_new - sv_eo), abs(sh_new - sh_trim)
        sv_eo = (1 - w_r) * sv_eo + w_r * sv_new
        # stabilizer: a smaller tail moves the neutral point and the CG band forward, so
        # the area update is steep (slope ~ -3) -> stronger under-relaxation
        w_h = 0.6 * w_r
        sh_trim = sh_new if sh_trim == 0.0 else (1 - w_h) * sh_trim + w_h * sh_new
        dx = db = ddih = 0.0
        if not fast:
            x_prev = geom.x_qc
            x_qc = (1 - w_r) * x_prev + w_r * x_qc_new
            dx = abs(x_qc_new - x_prev) if not geom.x_qc_clamped else 0.0
            db = abs(band_new - band) if band is not None else 1.0
            band = band_new if band is None else (1 - w_r) * band + w_r * band_new
            ddih = abs(dih_new - dihedral)
            dihedral = (1 - w_r) * dihedral + w_r * dih_new
        dcap = abs(cap_new - cap)
        cap = (1 - w_r) * cap + w_r * cap_new
        n_lim = n_new
        if m > 15.0:
            return Result(False, math.inf, tuple(x), combo, detail=dict(reason="mass diverged"))
        if os.environ.get("SIZING_DEBUG"):
            print(f"  it {it:2d} m {m:.5f} dm {dm:.1e} dsv {dsv:.1e} dsh {dsh:.1e} dcap {dcap:.1e} "
                  f"dx {dx:.1e} db {db:.1e} Sv {sv_eo:.5f} Sh {sh_trim:.5f}")
        if (frozen and dm < 1e-4 and de < 1e-3 and dsv < 1e-4 * max(sv_eo, 1e-3)
                and dsh < 1e-4 * max(sh_trim, 1e-3) and dcap < 1e-5 and dx < 2e-5 and db < 1e-4
                and ddih < 0.02):
            converged = True
            break

    st = perf["static"]
    lc = lat["cons"] if lat else {}
    ap = perf["approach"]
    cons = dict(
        takeoff_roll_grass=1 - perf["takeoff"]["s"] / P.takeoff_roll_max_grass_m,
        takeoff_roll_pavement=1 - perf["takeoff_pavement"]["s"] / P.takeoff_roll_max_pavement_m,
        landing_roll_grass=1 - perf["s_landing"] / P.landing_roll_max_grass_m,
        landing_roll_pavement=1 - perf["s_landing_pavement"] / P.landing_roll_max_pavement_m,
        span=1 - geom.b / P.span_max_m,
        climb_rate=perf["climb"]["roc"] / max(P.min_roc_ms, P.climb_rate_target_ms) - 1,
        climb_gradient=perf["climb"]["grad_max"] / P.min_climb_gradient - 1,
        balked_landing=ap["balked_gradient"] / P.balked_landing_gradient_min - 1,
        motor_current=1 - perf["I_motor_max"] / motor.imax,
        battery_current=1 - perf["I_batt_max"] / batt.i_max,
        esc_available=0.0 if esc is not None else -1.0,
        motor_station=1 - geom.y_eng / (P.max_motor_station_frac * geom.b),
        spar_fits=1 - winfo["spar"]["util"],
        tip_reynolds=(atm.rho * perf["loiter"]["V"] * geom.wing.c_tip / atm.mu) / P.re_tip_min - 1,
        roll_helix=(ctrl["aileron"]["pb2v"] / P.roll_helix_min - 1) if ctrl else 0.0,
        elevator_authority=(min(tail["elevator"]["ratio"], 2.0) - 1) if tail else 0.0,
        crosswind_rudder=min(lc.get("crosswind_rudder", 0.0), 1.0),
        crosswind_aileron=min(lc.get("crosswind_aileron", 0.0), 1.0),
        engine_out_aileron=min(lc.get("engine_out_aileron", 0.0), 1.0),
        cn_beta=min(lc.get("cn_beta", 0.0), 1.0),
        dutch_roll=min(min(c["margin_dutch"] for c in dyn["cases"].values()), 1.0) if dyn else 0.0,
        spiral=min(min(c["margin_spiral"] for c in dyn["cases"].values()), 1.0) if dyn else 0.0,
        roll_mode=min(min(c["margin_roll"] for c in dyn["cases"].values()), 1.0) if dyn else 0.0,
        short_period=min(min(c["margin"] for c in dyn["longitudinal"]["cases"].values()), 1.0) if dyn else 0.0,
        phugoid=min(dyn["longitudinal"]["margin_phugoid"], 1.0) if dyn else 0.0,
        stall_speed=(1 - perf["V_s_landing"] / P.stall_speed_max_ms) if P.stall_speed_max_ms else 1.0,
        brake_fits=0.0 if (not geom.brake.active or geom.brake.fits(geom.main_wheel)) else -1.0,
        nose_gear_fits=0.0 if winfo.get("nose_gear_fits", True) else -1.0,
        prop_data_range=0.0 if not st["rpm_limited"] else -1.0,
        converged=0.0 if converged else -1.0,
    )
    assert tuple(cons) == CONS_KEYS
    if not rubber:
        cons["endurance"] = bmf["endurance_margin"] - 1
    detail = dict(geom=geom, aero=aero, pt=pt, perf=perf, bmf=bmf, masses=masses,
                  winfo=winfo, esc=esc, battery=batt, atm=atm, p_av=p_av, V_C=V_C,
                  throttle_cap=cap, engine_out=eo_info, lateral=lat, controls=ctrl, tail=tail,
                  balance=bal, washout=washout, secs=secs, dynamics=dyn)
    return Result(True, m, tuple(x), combo, cons, detail, vs=float(perf["V_s_landing"]))


# ------------------------------------------------------------------ optimizer
SCREEN_S = np.geomspace(0.07, 0.45, 7)
SCREEN_AR = (5.0, 7.0, 9.5)


def screen(combo):
    """Cheap 2-D scan of wing area x aspect ratio (AR capped by the span limit).
    An earlier version scanned only along the span limit; that forced small
    wings to very high AR, which failed the tip-Reynolds constraint and hid
    good combinations from the optimizer. Returns the best Result (feasible
    preferred, else least violated)."""
    best = None
    pts = {(float(S), float(min(ar, P.span_max_m ** 2 / S)), 3.0) for S in SCREEN_S for ar in SCREEN_AR}
    for x in sorted(pts):
        r = evaluate(x, combo, fast=True)
        if not r.ok:
            continue
        key = (not r.feasible, r.violation if not r.feasible else r.score)
        if best is None or key < (not best.feasible, best.violation if not best.feasible else best.score):
            best = r
    return best


def optimize_multistart(combo, x0):
    """SLSQP from the given start and from a perturbed second start; the
    lightest feasible result wins (guards against local optima)."""
    S, AR, lt = x0[:3]
    alt = (S * 1.15, min(max(AR * 0.8, X_BOUNDS[1][0]), P.span_max_m ** 2 / (S * 1.15)), 2.6) \
        + tuple(x0[3:])
    res = [optimize(combo, x) for x in (x0, alt)]
    feas = [r for r in res if r.feasible]
    return min(feas, key=lambda r: r.score) if feas else min(res, key=lambda r: r.violation)


def optimize(combo, x0):
    nx = len(x0)
    bounds = list(X_BOUNDS[:nx])
    if nx > 3:
        bounds[3] = (0.0, P.washout_max_deg)
    if nx > 4:
        bounds[4] = tuple(P.wing_taper_bounds)
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    to_z = lambda x: (np.asarray(x) - lo) / (hi - lo)
    to_x = lambda z: lo + np.clip(z, 0, 1) * (hi - lo)
    cache = {}

    def ev(z):
        key = tuple(np.round(z, 10))
        if key not in cache:
            cache[key] = evaluate(to_x(z), combo)
        return cache[key]

    def f(z):
        r = ev(z)
        return r.score if r.ok else 50.0

    def cons_fun(z):
        r = ev(z)
        if not r.ok:
            return -np.ones(N_CONS + (combo.battery is not None))
        return np.array(list(r.cons.values()))

    res = minimize(f, to_z(x0), method="SLSQP", bounds=[(0, 1)] * nx,
                   constraints=[{"type": "ineq", "fun": cons_fun}],
                   options=dict(maxiter=60, ftol=1e-6, eps=2e-3))
    best = ev(res.x)
    # keep the best feasible point the optimizer visited
    feas = [r for r in cache.values() if r.feasible]
    if feas:
        cand = min(feas, key=lambda r: r.score)
        if not best.feasible or cand.score < best.score:
            best = cand
    return best


# ------------------------------------------------------------------ search
def propulsion_combos():
    """Motor x prop x battery family, pre-filtered on the static operating
    point (motor current limit, prop data range, ESC availability)."""
    atm = atmosphere()
    out = []
    for fam in battery_families():
        ref = fam.make(45.0)
        for m in motors().values():
            if fam.s > m.max_cells:
                continue
            for p in props().values():
                ft = Powertrain(m, p, ref, atm.rho, avionics_power_w()).full_throttle()
                I = float(ft["I_motor"][0])
                if ft["rpm_limited"][0] or I < 0.25 * m.imax:   # out of data / dead weight
                    continue
                # over-propped is OK if a throttle cap of ~0.6 brings it within limits
                I_cap = float(Powertrain(m, p, ref, atm.rho, avionics_power_w(), 0.6)
                              .full_throttle()["I_motor"][0])
                if I_cap > m.imax or pick_esc(min(I, m.imax), fam.s) is None:
                    continue
                out.append((m.name, p.name, fam, float(ft["T"][0])))
    return out


def _screen_job(args):
    af, cons_, mname, pname, fam = args
    r = screen(Combo(af, cons_, mname, pname, family=fam))
    if r is None:
        return None
    return (r.feasible, r.score, r.violation, r.x, af, cons_, mname, pname, fam.name)


def run(verbose=True, workers=None):
    import time
    t0 = time.time()
    _log = print if verbose else (lambda *a, **k: None)

    def log(msg):
        _log(f"{msg}   [{time.time() - t0:5.0f} s]")
    fams = {f.name: f for f in battery_families()}
    combos = propulsion_combos()
    log(f"[B0] {len(combos)} motor/prop/battery-family combinations pass the static check")

    # ---- A: airframe technology, screened with a stratified sample of powertrains
    order = sorted(range(len(combos)), key=lambda i: combos[i][3])
    refs = [combos[i] for i in order[::max(1, len(order) // 36)]]
    jobs = [(af, cn, r[0], r[1], r[2]) for af in wing_airfoil_names() for cn in P.wing_constructions for r in refs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        resA = [r for r in ex.map(_screen_job, jobs, chunksize=4) if r is not None]
    tech = {}
    for feas, m, viol, x, af, cn, *_ in resA:
        k = (af, cn)
        score = m if feas else 100 + viol
        tech[k] = min(tech.get(k, math.inf), score)
    ranked = sorted(tech.items(), key=lambda kv: kv[1])
    log("[A ] airframe technology ranking (best screened score = mass + stall-speed worth, kg):")
    for (af, cn), s in ranked:
        _log(f"       {af:9s} {cn:14s} {s:7.3f}" if s < 100 else f"       {af:9s} {cn:14s} infeasible")
    top_tech = [k for k, _ in ranked[:3]]

    # ---- B: every powertrain with the top airframe technologies
    jobs = [(af, cn, c[0], c[1], c[2]) for (af, cn) in top_tech for c in combos]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        resB = [r for r in ex.map(_screen_job, jobs, chunksize=4) if r is not None]
    feasB = sorted([r for r in resB if r[0]], key=lambda r: r[1])
    log(f"[B ] screened {len(jobs)} designs, {len(feasB)} feasible")
    if not feasB:
        raise RuntimeError("no feasible design found in screening - check requirements/inputs")

    # ---- C: full model (control surfaces, elevator, takeoff technique) for the best
    #         distinct candidates, in each control-surface configuration
    seen, cand = set(), []
    for r in feasB:
        key = (r[6], r[7], r[8])
        if key in seen:
            continue
        seen.add(key)
        cand.append(r)
        if len(cand) >= 12:
            break
    jobsC = [(Combo(r[4], r[5], r[6], r[7], family=fams[r[8]], controls=cfg), r[3], True)
             for r in cand for cfg in CONTROL_CONFIGS]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        resC = list(ex.map(_opt_job, jobsC))
    resC = sorted([r for r in resC if r is not None and r.feasible], key=lambda r: r.score)
    log(f"[C ] refined {len(jobsC)} candidate x control-configuration designs (multi-start SLSQP)")
    if not resC:
        raise RuntimeError("no feasible design with control surfaces sized")

    # ---- D: snap to real batteries - the best 12 overall PLUS the best 2 of every
    #         battery family, so no single chemistry can crowd out the others
    pick = list(resC[:12])
    for fam_name in {r.combo.family.name for r in resC}:
        pick += [r for r in resC if r.combo.family.name == fam_name][:2]
    pick = list({id(r): r for r in pick}.values())
    jobsD = []
    for r in pick:
        fam = r.combo.family
        e_req = r.detail["bmf"]["energy_required_wh"]
        opts = sorted([b for b in fam.options() if b.energy_wh >= 0.9 * e_req], key=lambda b: b.mass_kg)[:3]
        for b in opts:
            c = replace(r.combo, family=None, battery=b)
            jobsD.append((c, r.x, False))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        resD = list(ex.map(_opt_job, jobsD))
    resD = sorted([r for r in resD if r is not None and r.feasible], key=lambda r: r.score)
    log(f"[D ] {len(resD)} feasible designs with real catalog batteries")
    if not resD:
        raise RuntimeError("no feasible design with a real battery")

    # ---- T: wing twist - every tip airfoil (aerodynamic twist) with washout as a
    #         4th optimizer variable (geometric twist), for the best 3 designs
    jobsT = []
    for r in resD[:3]:
        for tip in wing_airfoil_names():
            c = replace(r.combo, tip_airfoil=tip)
            jobsT.append((c, tuple(r.x[:3]) + (1.0, P.wing_taper), False))   # washout AND taper free
    with ProcessPoolExecutor(max_workers=workers) as ex:
        resT = list(ex.map(_opt_job, jobsT))
    final = sorted([r for r in resD + resT if r is not None and r.feasible], key=lambda r: r.score)
    log(f"[T ] twist / tip-airfoil refinement: {len(jobsT)} designs")
    return final


def twist_study(best, workers=None):
    """Trade study for the winning design: every tip airfoil x fixed washout,
    each re-optimized (S, AR, tail arm) with the same parts. Returns rows."""
    washouts = (0.0, 2.0, 4.0, 6.0)
    jobs = []
    for tip in wing_airfoil_names():
        for w in washouts:
            c = replace(best.combo, tip_airfoil=tip, washout=w, taper=_taper_of(best))
            jobs.append((c, tuple(best.x[:3]), False))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_opt_job_full, jobs))
    rows = []
    for (c, _, _), r in zip(jobs, res):
        if r is None or not r.ok:
            rows.append(dict(tip=c.tip_airfoil, washout=c.washout, ok=False))
            continue
        rows.append(dict(tip=c.tip_airfoil, washout=c.washout, ok=True, feasible=r.feasible,
                         misses=[k for k, v in r.cons.items() if v < -1e-4], **r.detail["summary"]))
    return rows


def _taper_of(r):
    return float(r.x[4]) if len(r.x) > 4 else (r.combo.taper if r.combo.taper is not None else P.wing_taper)


TAPERS = (0.35, 0.45, 0.6, 0.8, 1.0)


def taper_study(best, workers=None):
    """Trade study for the winning design: fixed taper ratios, each re-optimized
    (S, AR, tail arm, washout) with the same parts. Shows why the optimizer's
    taper wins (tip Reynolds number, stall location, span efficiency)."""
    jobs = [(replace(best.combo, taper=t), tuple(best.x[:4]), False) for t in TAPERS]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_opt_job_full, jobs))
    rows = []
    for (c, _, _), r in zip(jobs, res):
        if r is None or not r.ok:
            rows.append(dict(taper=c.taper, ok=False))
            continue
        rows.append(dict(taper=c.taper, ok=True, feasible=r.feasible,
                         misses=[k for k, v in r.cons.items() if v < -1e-4], **r.detail["summary"]))
    return rows


STALL_CAPS = (8.5, 9.0, 9.5, 10.0, 11.0, 12.0, 13.0, 15.0)


def stall_study(best, workers=None):
    """Trade study for the winning design: the LIGHTEST design (mass objective,
    stall-speed weight 0) at each stall-speed cap, same parts, re-optimized
    (S, AR, tail arm, washout). Shows what each m/s of stall speed costs."""
    jobs = [(best.combo, tuple(best.x), cap) for cap in STALL_CAPS]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_opt_job_cap, jobs))
    rows = []
    for (_, _, cap), r in zip(jobs, res):
        if r is None or not r.ok:
            rows.append(dict(cap=cap, ok=False))
            continue
        rows.append(dict(cap=cap, ok=True, feasible=r.feasible,
                         misses=[k for k, v in r.cons.items() if v < -1e-4], **r.detail["summary"]))
    return rows


def _opt_job_cap(args):
    combo, x0, cap = args
    try:
        with P.override(stall_speed_max_ms=cap, stall_speed_weight_g_per_ms=0.0):
            r = optimize_multistart(combo, x0)
            if r.ok:
                r.detail["summary"] = _summary(r)
        return _strip(r, keep=("bmf", "summary"))
    except Exception as e:   # pragma: no cover
        print("optimize failed:", combo.label(), e)
        return None


def _opt_job(args):
    combo, x0, multistart = args
    try:
        return _strip(optimize_multistart(combo, x0) if multistart else optimize(combo, x0))
    except Exception as e:   # pragma: no cover - keep the search going
        print("optimize failed:", combo.label(), e)
        return None


def _summary(r):
    d = r.detail
    p, g, a = d["perf"], d["geom"], d["aero"]
    return dict(m=r.m, score=r.score, V_s_landing=r.vs, span=g.b, S=g.S, AR=g.AR, taper_ratio=g.wing.taper,
                c_root=g.wing.c_root, c_tip=g.wing.c_tip,
                tip_re=d["atm"].rho * p["loiter"]["V"] * g.wing.c_tip / d["atm"].mu,
                W_S=r.m * G0 / g.S, V_td=p["V_td"], takeoff_pavement_m=p["takeoff_pavement"]["s"],
                landing_pavement_m=p["s_landing_pavement"], endurance_margin=d["bmf"]["endurance_margin"],
                loiter_min=d["bmf"]["loiter_max_min"],
                landing_m=p["s_landing"], takeoff_m=p["takeoff"]["s"], CLmax=p["CLmax"],
                CLmax_landing=p["CLmax_landing"], stall_eta=p["stall_eta"],
                stall_eta_landing=p["stall_eta_landing"], e=a.e, LD_loiter=p["loiter"]["LD"],
                pb2v=d["controls"]["aileron"]["pb2v"] if d["controls"] else None,
                x=tuple(float(v) for v in r.x))


def _opt_job_full(args):
    combo, x0, multistart = args
    try:
        r = optimize_multistart(combo, x0) if multistart else optimize(combo, x0)
        if r.ok:
            r.detail["summary"] = _summary(r)
        return _strip(r, keep=("bmf", "summary"))
    except Exception as e:   # pragma: no cover
        print("optimize failed:", combo.label(), e)
        return None


def _strip(r, keep=("bmf",)):
    """Results cross process boundaries - drop heavy objects, re-evaluate later."""
    if r is None:
        return None
    keep = {k: r.detail[k] for k in keep if k in r.detail}
    return Result(r.ok, r.m, r.x, r.combo, r.cons, keep, r.vs)
