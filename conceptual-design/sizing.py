"""
The sizing loop.

evaluate(x, combo)  - one design: converges takeoff mass with a fixed-point
                      loop   geometry -> aero -> powertrain -> performance ->
                      mission BMF -> weights -> new W_TO   (|dm| < 0.1 g)
                      and returns the objective (W_TO) + every constraint.

run()               - staged search:
   A  airframe technology (wing airfoil x construction), screened with
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
from controls import tail_and_elevator, wing_controls
from atmosphere import G0, isa
from catalog import (airfoil, battery_families, motors, pick_esc, props,
                     wing_airfoil_names)
from geometry import build
from mission import mission_bmf, performance
from parameters import P
from propulsion import V_GRID, Powertrain
from weights import (CONSTRUCTIONS, avionics_power_w, design_cruise_speed, gust_load_factor,
                     mass_breakdown)

X_BOUNDS = [(0.04, 0.60), (4.0, 20.0), (1.8, 5.0), (0.0, 6.0)]   # S, AR, tail arm/MAC, washout deg
X_NAMES = ("S_m2", "AR", "tail_arm_over_mac", "washout_deg")
N_CONS = 15        # constraints for a continuous battery (+1 endurance for a real pack)


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

    @property
    def feasible(self):
        return self.ok and all(v >= -1e-4 for v in self.cons.values())

    @property
    def violation(self):
        return sum(max(-v, 0.0) for v in self.cons.values()) if self.cons else 1e3


def atmosphere():
    return isa(P.field_elevation_m, P.isa_temp_offset_K)


@lru_cache(maxsize=512)
def _rudder_at_re(re_key, cf, d_max, conf_min):
    """2-D side-force coefficient of the fin section with the rudder deflected,
    from NeuralFoil on the actual deflected geometry (alpha = 0: no sideslip),
    at one Reynolds number. The usable deflection ends where NeuralFoil's
    confidence crosses conf_min (interpolated, so results are continuous) -
    beyond it the flow separates over the rudder and XFOIL-type analysis
    over-predicts. Returns (cl, usable deflection deg)."""
    import aerosandbox as asb
    import neuralfoil as nf
    base = asb.Airfoil(P.tail_airfoil)
    ds = np.arange(0.0, d_max + 1e-9, 1.0)
    cl, conf = [], []
    for d in ds:
        coords = base.add_control_surface(deflection=float(d), hinge_point_x=1 - cf).coordinates
        r = nf.get_aero_from_coordinates(coords, alpha=0.0, Re=float(re_key), model_size="xlarge")
        cl.append(float(np.ravel(r["CL"])[0]))
        conf.append(float(np.ravel(r["analysis_confidence"])[0]))
    cl, conf = np.array(cl), np.array(conf)
    bad = np.flatnonzero(conf < conf_min)
    if not len(bad):
        return float(cl[-1]), float(ds[-1])
    i = int(bad[0])
    if i == 0:
        return 0.0, 0.0
    f = (conf[i - 1] - conf_min) / (conf[i - 1] - conf[i])
    return float(cl[i - 1] + f * (cl[i] - cl[i - 1])), float(ds[i - 1] + f * (ds[i] - ds[i - 1]))


def rudder_section_cl(re, cf, d_max, conf_min, step=10000):
    """Linear interpolation between cached Reynolds-number buckets (continuous in Re)."""
    lo = int(math.floor(re / step) * step)
    lo = max(lo, step)
    w = (re - lo) / step
    a, b = _rudder_at_re(lo, cf, d_max, conf_min), _rudder_at_re(lo + step, cf, d_max, conf_min)
    return (1 - w) * a[0] + w * b[0], (1 - w) * a[1] + w * b[1]


def fin_engine_out(geom, pt, V_s, atm, cf_fixed=None):
    """Fin area so the rudder holds one motor at full (capped) throttle at
    V_mc = 1.2 Vs (14 CFR 23.149), no sideslip.
      dCL_fin = cl_2D(rudder) * (a_3D / a_2D) * rudder span fraction"""
    if not P.engine_out_sizing:
        return 0.0, {}
    V_mc = P.vmc_ratio * V_s
    T_one = float(np.interp(V_mc, V_GRID, pt.full_throttle()["T"])) / P.n_motors
    N = T_one * geom.y_eng
    q = 0.5 * atm.rho * V_mc ** 2
    taf = airfoil(P.tail_airfoil)
    re_v = atm.rho * V_mc * max(geom.vtail.mac, 0.03) / atm.mu
    grid = (cf_fixed,) if cf_fixed else P.rudder_cf_grid
    cl2d, d_use, cf_r = max((rudder_section_cl(re_v, cf, P.rudder_max_deg, P.foil_conf_min) + (cf,)
                            for cf in grid), key=lambda t: t[0])
    a2 = float(taf.cl_alpha(re_v))
    a_v = lift_slope_3d(P.vtail_ar_eff_factor * P.ar_vtail, a2)
    dCL = cl2d * (a_v / a2) * P.rudder_span_frac
    info = dict(V_mc=V_mc, T_one=T_one, yaw_moment=N, rudder_cf=cf_r, rudder_cl2d=cl2d,
                rudder_defl_usable=d_use, dCL_fin=dCL)
    return N / (q * geom.l_t * max(dCL, 1e-3)), info


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
    to = perf["takeoff"]
    V_lo = to["V_lo"]
    L_eff = P.takeoff_roll_max_m - V_lo * P.rotation_time_s
    if math.isfinite(to["s"]):
        s_eff = max(to["s"] - V_lo * P.rotation_time_s, 1e-3)
        resist = to["T07"] / W - V_lo ** 2 / (2 * G0 * s_eff)       # mean resistance / W
        k_to = (V_lo ** 2 / (2 * G0 * L_eff) + resist) / (to["T07"] / W)
    else:
        k_to = 2.0
    k = max(k_roc, k_grad, k_to) * P.throttle_cap_margin
    return float(np.clip(cap * math.sqrt(max(k, 1e-6)), 0.15, 1.0))


FREEZE_IT = 6     # iterations before discrete control-surface picks are frozen
CONTROL_CONFIGS = ("none", "flaps", "flaperons")


def evaluate(x, combo, max_iter=80, fast=False):
    """x = (S, AR, tail arm / MAC[, washout deg]).
    fast=True: screening mode - no control-surface / elevator sizing (ailerons
    only, Raymer tails, level ground stance). Used only to RANK powertrains."""
    S, AR, lt = (float(v) for v in x[:3])
    washout = float(x[3]) if len(x) > 3 else float(combo.washout)
    atm = atmosphere()
    secs = WingSections(airfoil(combo.airfoil), airfoil(combo.tip_airfoil or combo.airfoil))
    taf = airfoil(P.tail_airfoil)
    motor, prop = motors()[combo.motor], props()[combo.prop]
    p_av = avionics_power_w()
    rubber = combo.battery is None
    batt = combo.family.make(40.0) if rubber else combo.battery

    m, sv_eo, sh_trim, n_lim, V_C, cap = 1.5, 0.0, 0.0, P.n_limit_maneuver, 0.0, 1.0
    eo_info, ctrl, tail = {}, None, None
    ctrl_choice = tail_choice = rudder_cf = None
    flaps_prev, ground_cl = None, None
    perf = bmf = None
    converged = False
    for it in range(max_iter):
        frozen = it >= FREEZE_IT
        geom = build(S, AR, lt, secs, motor, prop, batt, sv_eo, washout, sh_trim)
        aero = Aero(geom, secs, taf, atm, flaps_prev)
        if not fast:
            ctrl = wing_controls(geom, aero, m * G0, combo.controls, ctrl_choice if frozen else None)
            flaps_prev = aero.flaps
        pt = Powertrain(motor, prop, batt, atm.rho, p_av, cap)
        ft = pt.full_throttle()
        esc = pick_esc(float(np.max(ft["I_motor"])), batt.s)
        masses, winfo = mass_breakdown(geom, secs, combo.construction, motor, prop, esc, batt, m, n_lim,
                                       span_load=aero.span_load(),
                                       n_flap_servos=2 if (combo.controls == "flaps" and not fast) else 0)
        perf = performance(geom, aero, pt, m, batt, ground_cl)
        if perf is None:
            return Result(False, math.inf, tuple(x), combo, detail=dict(reason="no loiter speed"))
        bmf = mission_bmf(perf, batt, p_av)
        if bmf is None:
            return Result(False, math.inf, tuple(x), combo,
                          detail=dict(reason="cannot take off/climb"))
        # couplings
        sv_new, eo_info = fin_engine_out(geom, pt, perf["V_s"], atm, rudder_cf if frozen else None)
        if not fast:
            tail = tail_and_elevator(geom, aero, perf, pt, m, tail_choice if frozen else None)
            sh_new = tail["sh_needed"] * 1.01
            ground_cl = tail["CL_ground"]             # nose-up stance (or None: level)
        else:
            sh_new = 0.0
        if it == FREEZE_IT - 1:
            ctrl_choice = ctrl["choice"] if ctrl else None
            tail_choice = tail["choice"] if tail else None
            rudder_cf = eo_info.get("rudder_cf")
        cap_new = throttle_cap_update(cap, perf, pt, m * G0)
        W_S = m * G0 / S
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
        m = 0.5 * m + 0.5 * m_new if dm > 0.05 else m_new
        dsv, dsh = abs(sv_new - sv_eo), abs(sh_new - sh_trim)
        sv_eo = 0.5 * sv_eo + 0.5 * sv_new
        sh_trim = 0.5 * sh_trim + 0.5 * sh_new
        dcap = abs(cap_new - cap)
        cap = 0.5 * cap + 0.5 * cap_new
        n_lim = n_new
        if m > 15.0:
            return Result(False, math.inf, tuple(x), combo, detail=dict(reason="mass diverged"))
        if frozen and dm < 1e-4 and de < 1e-3 and dsv < 1e-6 and dsh < 1e-6 and dcap < 1e-5:
            converged = True
            break

    st = perf["static"]
    cons = dict(
        takeoff_roll=1 - perf["takeoff"]["s"] / P.takeoff_roll_max_m,
        landing_roll=1 - perf["s_landing"] / P.landing_roll_max_m,
        span=1 - geom.b / P.span_max_m,
        climb_rate=perf["climb"]["roc"] / max(P.min_roc_ms, P.climb_rate_target_ms) - 1,
        climb_gradient=perf["climb"]["grad_max"] / P.min_climb_gradient - 1,
        motor_current=1 - perf["I_motor_max"] / motor.imax,
        battery_current=1 - perf["I_batt_max"] / batt.i_max,
        esc_available=0.0 if esc is not None else -1.0,
        motor_station=1 - geom.y_eng / (P.max_motor_station_frac * geom.b),
        spar_fits=1 - winfo["spar"]["util"],
        tip_reynolds=(atm.rho * perf["loiter"]["V"] * geom.wing.c_tip / atm.mu) / P.re_tip_min - 1,
        roll_helix=(ctrl["aileron"]["pb2v"] / P.roll_helix_min - 1) if ctrl else 0.0,
        elevator_authority=(min(tail["elevator"]["ratio"], 2.0) - 1) if tail else 0.0,
        prop_data_range=0.0 if not st["rpm_limited"] else -1.0,
        converged=0.0 if converged else -1.0,
    )
    if not rubber:
        cons["endurance"] = bmf["endurance_margin"] - 1
    detail = dict(geom=geom, aero=aero, pt=pt, perf=perf, bmf=bmf, masses=masses,
                  winfo=winfo, esc=esc, battery=batt, atm=atm, p_av=p_av, V_C=V_C,
                  throttle_cap=cap, engine_out=eo_info, controls=ctrl, tail=tail,
                  washout=washout, secs=secs)
    return Result(True, m, tuple(x), combo, cons, detail)


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
        key = (not r.feasible, r.violation if not r.feasible else r.m)
        if best is None or key < (not best.feasible, best.violation if not best.feasible else best.m):
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
    return min(feas, key=lambda r: r.m) if feas else min(res, key=lambda r: r.violation)


def optimize(combo, x0):
    nx = len(x0)
    bounds = list(X_BOUNDS[:nx])
    if nx > 3:
        bounds[3] = (0.0, P.washout_max_deg)
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
        return r.m if r.ok else 50.0

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
        cand = min(feas, key=lambda r: r.m)
        if not best.feasible or cand.m < best.m:
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
    return (r.feasible, r.m, r.violation, r.x, af, cons_, mname, pname, fam.name)


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
    jobs = [(af, cn, r[0], r[1], r[2]) for af in wing_airfoil_names() for cn in CONSTRUCTIONS for r in refs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        resA = [r for r in ex.map(_screen_job, jobs, chunksize=4) if r is not None]
    tech = {}
    for feas, m, viol, x, af, cn, *_ in resA:
        k = (af, cn)
        score = m if feas else 100 + viol
        tech[k] = min(tech.get(k, math.inf), score)
    ranked = sorted(tech.items(), key=lambda kv: kv[1])
    log("[A ] airframe technology ranking (best screened W_TO, kg):")
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
    resC = sorted([r for r in resC if r is not None and r.feasible], key=lambda r: r.m)
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
    resD = sorted([r for r in resD if r is not None and r.feasible], key=lambda r: r.m)
    log(f"[D ] {len(resD)} feasible designs with real catalog batteries")
    if not resD:
        raise RuntimeError("no feasible design with a real battery")

    # ---- T: wing twist - every tip airfoil (aerodynamic twist) with washout as a
    #         4th optimizer variable (geometric twist), for the best 3 designs
    jobsT = []
    for r in resD[:3]:
        for tip in wing_airfoil_names():
            c = replace(r.combo, tip_airfoil=tip)
            jobsT.append((c, tuple(r.x[:3]) + (1.0,), False))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        resT = list(ex.map(_opt_job, jobsT))
    final = sorted([r for r in resD + resT if r is not None and r.feasible], key=lambda r: r.m)
    log(f"[T ] twist / tip-airfoil refinement: {len(jobsT)} designs")
    return final


def twist_study(best, workers=None):
    """Trade study for the winning design: every tip airfoil x fixed washout,
    each re-optimized (S, AR, tail arm) with the same parts. Returns rows."""
    washouts = (0.0, 2.0, 4.0, 6.0)
    jobs = []
    for tip in wing_airfoil_names():
        for w in washouts:
            c = replace(best.combo, tip_airfoil=tip, washout=w)
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
    return dict(m=r.m, span=g.b, S=g.S, AR=g.AR, loiter_min=d["bmf"]["loiter_max_min"],
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
    return Result(r.ok, r.m, r.x, r.combo, r.cons, keep)
