"""
Control surfaces: placement and size from requirements.

Every deflected-surface number comes from NeuralFoil run on the actual
deflected airfoil geometry (aerosandbox Airfoil.add_control_surface), trusted
only where NeuralFoil's analysis confidence >= P.foil_conf_min (beyond that the
flow separates over the surface and XFOIL-type methods over-predict).
Results are interpolated between Reynolds-number buckets so they vary smoothly.

  Ailerons  : outboard; smallest span meeting the roll-helix requirement
              pb/2V >= 0.07 at approach speed (strip theory: aileron rolling
              moment = roll damping).
  Flaps     : inboard of the ailerons, chord/deflection giving the highest
              landing CLmax (lifting line, critical section).
  Flaperons : alternative - full-span surfaces that droop as flaps and still
              meet the roll requirement about the drooped position.
  Elevator  : (a) trim at landing CLmax, flaps down, forward CG;
              (b) takeoff rotation about the main wheels at liftoff speed,
              forward CG. Neutral point from the tail-volume relation plus
              the fuselage (Raymer eq. 16.25, K_fus of Fig. 16.14).
  Stabilizer: SCISSOR PLOT (P.htail_sizing = 'scissor'): the CG band width
              comes from the as-built CG scatter (weights.cg_budget); its aft
              end sits on the stability line (static margin >= minimum), its
              forward end must be controllable (a)+(b). The stabilizer area is
              the smallest that does both - no historical tail-volume floor.
  Trim drag : elevator deflection + tail induced drag for a trimmed condition
              (elevator_trim_drag) - used for the approach / balked landing.
  (Rudder   : coupled lateral-directional trim, see lateral.py.)
"""
import math
from functools import lru_cache

import numpy as np

from aerodynamics import lift_slope_3d
from atmosphere import G0
from parameters import P
from propulsion import V_GRID

ALPHAS = np.arange(-8.0, 18.01, 0.5)
RE_STEP = 10000


@lru_cache(maxsize=4096)
def _sweep(af_name, cf, delta, re_key):
    """NeuralFoil alpha sweep of the deflected section (cached)."""
    import aerosandbox as asb
    import neuralfoil as nf
    af = asb.Airfoil(af_name)
    if abs(delta) > 1e-9:
        af = af.add_control_surface(deflection=float(delta), hinge_point_x=1 - cf)
    r = nf.get_aero_from_coordinates(af.coordinates, alpha=ALPHAS, Re=float(re_key), model_size="xlarge")
    return (np.ravel(r["CL"]), np.ravel(r["CD"]), np.ravel(r["CM"]),
            np.ravel(r["analysis_confidence"]))


@lru_cache(maxsize=4096)
def _derived(af_name, cf, delta, re_key, conf_min):
    """clmax, lift slope, zero-lift angle, cm0 and the pre-stall cd(cl) branch of
    the deflected section. A plain flap acts as added camber (thin-airfoil
    theory): it shifts the zero-lift angle and leaves the lift slope ~unchanged.
    The flap's lift increment is measured where BOTH the clean and the deflected
    section are confidently analysed in the linear range; deflections where the
    deflected section's own slope has collapsed (< 70% of clean: flow separating
    over the flap) are not trusted."""
    cl, cd, cm, conf = _sweep(af_name, cf, delta, re_key)
    good = conf >= conf_min
    if good.sum() < 6:
        return None
    pos = good & (ALPHAS > -4)
    if not pos.any():
        return None
    i_max = int(np.flatnonzero(pos)[np.argmax(cl[pos])])
    neg = good & (ALPHAS <= ALPHAS[i_max])
    i_min = int(np.flatnonzero(neg)[np.argmin(cl[neg])])
    bcl, bcd = [], []
    for i in range(i_min, i_max + 1):
        if good[i] and (not bcl or cl[i] > bcl[-1] + 1e-6):
            bcl.append(cl[i]); bcd.append(cd[i])
    lin = good & (ALPHAS >= ALPHAS[i_max] - 12) & (ALPHAS <= ALPHAS[i_max] - 4)
    if lin.sum() < 4:
        return None
    slope, icpt = np.polyfit(np.radians(ALPHAS[lin]), cl[lin], 1)
    out = dict(clmax=float(cl[i_max]), cm0=float(np.mean(cm[lin])), bcl=np.array(bcl), bcd=np.array(bcd))
    if abs(delta) < 1e-9:
        if not (0.5 * 2 * math.pi <= slope <= 1.3 * 2 * math.pi):
            return None
        out.update(cl_alpha=float(slope), alpha0=float(-icpt / slope))
        return out
    clean = _derived(af_name, cf, 0.0, re_key, conf_min)
    if clean is None or slope < 0.7 * clean["cl_alpha"]:
        return None
    ccl, _, ccm, cconf = _sweep(af_name, cf, 0.0, re_key)
    both = lin & (cconf >= conf_min)
    if both.sum() < 3:
        return None
    dcl = float(np.mean(cl[both] - ccl[both]))
    out.update(cl_alpha=clean["cl_alpha"], alpha0=clean["alpha0"] - dcl / clean["cl_alpha"],
               cm0=clean["cm0"] + float(np.mean(cm[both] - ccm[both])))
    return out


def _buckets(re):
    lo = max(int(math.floor(re / RE_STEP) * RE_STEP), RE_STEP)
    return lo, lo + RE_STEP, (re - lo) / RE_STEP


class FlapSection:
    """A section with a trailing-edge surface of chord fraction cf deflected by
    `delta` degrees (positive = trailing edge down)."""

    def __init__(self, af_name, cf, delta):
        self.af, self.cf, self.delta = af_name, float(cf), float(delta)

    def usable(self, re):
        lo, hi, _ = _buckets(re)
        return all(_derived(self.af, self.cf, self.delta, k, P.foil_conf_min) is not None for k in (lo, hi))

    def props(self, re):
        """Increments relative to the clean section + flapped lift slope."""
        lo, hi, w = _buckets(re)
        out = {}
        for k, wt in ((lo, 1 - w), (hi, w)):
            f = _derived(self.af, self.cf, self.delta, k, P.foil_conf_min)
            c = _derived(self.af, self.cf, 0.0, k, P.foil_conf_min)
            if f is None or c is None:            # no trusted analysis: credit no flap effect
                c = c or dict(clmax=1.0, cl_alpha=2 * math.pi, alpha0=0.0, cm0=0.0)
                f = c
            vals = dict(d_clmax=f["clmax"] - c["clmax"], d_alpha0=f["alpha0"] - c["alpha0"],
                        d_cm0=f["cm0"] - c["cm0"], cl_alpha=f["cl_alpha"], clmax=f["clmax"])
            for key, v in vals.items():
                out[key] = out.get(key, 0.0) + wt * v
        return out

    def cd(self, cl, re):
        """cd on the flapped pre-stall branch (smooth penalty past clmax)."""
        cl = np.asarray(cl, float)
        r = float(np.mean(re))
        lo, hi, w = _buckets(r)
        res = 0.0
        for k, wt in ((lo, 1 - w), (hi, w)):
            f = None
            for kk in range(k, k + 40 * RE_STEP, RE_STEP):     # nearest trusted Re (slow end of rollout)
                f = (_derived(self.af, self.cf, self.delta, kk, P.foil_conf_min)
                     or _derived(self.af, self.cf, 0.0, kk, P.foil_conf_min))
                if f is not None:
                    break
            b_cl, b_cd = f["bcl"], f["bcd"]
            v = np.interp(cl, b_cl, b_cd)
            over = cl > b_cl[-1]
            v = np.where(over, b_cd[-1] + 0.5 * (cl - b_cl[-1]) + 3 * (cl - b_cl[-1]) ** 2, v)
            res = res + wt * v
        return res

    def cd_at(self, alpha_deg, re):
        """section cd at an angle of attack, interpolated in alpha and Re."""
        lo, hi, w = _buckets(re)
        cd = 0.0
        for k, wt in ((lo, 1 - w), (hi, w)):
            _, c, _, _ = _sweep(self.af, self.cf, self.delta, k)
            cd += wt * float(np.interp(alpha_deg, ALPHAS, c))
        return cd

    def cl_at(self, alpha_deg, re):
        """(cl, confidence) at an angle of attack, interpolated in alpha and Re."""
        lo, hi, w = _buckets(re)
        cl = conf = 0.0
        for k, wt in ((lo, 1 - w), (hi, w)):
            c, _, _, cf = _sweep(self.af, self.cf, self.delta, k)
            cl += wt * float(np.interp(alpha_deg, ALPHAS, c))
            conf += wt * float(np.interp(alpha_deg, ALPHAS, cf))
        return cl, conf


def max_usable_deflection(af_name, cf, re, d_max, step=2.0):
    """Largest deflection (multiple of `step`) whose flapped section is still
    analysed with confidence (has a trusted clmax)."""
    best = 0.0
    for d in np.arange(step, d_max + 1e-9, step):
        if FlapSection(af_name, cf, d).usable(re):
            best = float(d)
        else:
            break
    return best


# =====================================================================
# Ailerons
# =====================================================================
def _int_cy(geom, y1, y2, power):
    """integral of c(y) y^power dy over [y1, y2]"""
    y = np.linspace(y1, y2, 60)
    return float(np.trapezoid(geom.chord_at(y) * y ** power, y))


def roll_helix(geom, aero, dcl_roll, y1, y2):
    """pb/2V from strip theory: rolling moment of the ailerons (section lift
    change dcl_roll per side) balanced by roll damping."""
    a0 = float(np.mean(aero.a0_c))
    damping = a0 * _int_cy(geom, 0.0, geom.b / 2, 2)
    return (geom.b / 2) * dcl_roll * _int_cy(geom, y1, y2, 1) / damping


def aileron_effectiveness(af_name, cf, alpha_deg, re, droop=0.0):
    """Per-side section lift change for antisymmetric deflection +/- da about
    `droop`, with the largest trusted throw <= aileron_max_deg."""
    base = FlapSection(af_name, cf, droop)
    for da in np.arange(P.aileron_max_deg, 4.9, -2.5):
        dn = FlapSection(af_name, cf, droop + da).cl_at(alpha_deg, re)
        up = FlapSection(af_name, cf, droop - da).cl_at(alpha_deg, re)
        if min(dn[1], up[1]) >= P.foil_conf_min:
            return 0.5 * (dn[0] - up[0]), float(da)
    del base
    return 0.0, 0.0


def size_ailerons(geom, aero, cf, af_name, alpha_deg, re, y_inner_min, droop=0.0, full_span=False):
    """Inboard edge of the smallest aileron meeting pb/2V >= requirement.
    Returns dict (ok False if even the full available span is not enough)."""
    y2 = P.aileron_outboard_eta * geom.b / 2
    dcl, da = aileron_effectiveness(af_name, cf, alpha_deg, re, droop)
    req = P.roll_helix_min
    full = roll_helix(geom, aero, dcl, y_inner_min, y2)
    if full_span:
        return dict(cf=cf, y1=y_inner_min, y2=y2, throw=da, dcl=dcl, pb2v=full, ok=full >= req)
    if full < req:
        return dict(cf=cf, y1=y_inner_min, y2=y2, throw=da, dcl=dcl, pb2v=full, ok=False)
    lo, hi = y_inner_min, y2
    for _ in range(40):                           # largest y1 still meeting the requirement
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if roll_helix(geom, aero, dcl, mid, y2) >= req else (lo, mid)
    return dict(cf=cf, y1=lo, y2=y2, throw=da, dcl=dcl,
                pb2v=roll_helix(geom, aero, dcl, lo, y2), ok=True)


# =====================================================================
# Wing control-surface layout (ailerons + flaps / flaperons)
# =====================================================================
def _local_alpha_deg(aero, CL, eta, config="clean"):
    w = aero.landing if config == "landing" else aero.clean
    i = int(np.argmin(np.abs(aero.eta_strip - eta)))
    nu = aero.atm.mu / aero.atm.rho
    re = 12.0 * aero.g.chord[i] / nu
    a0 = float(aero.secs.cl_alpha(re, aero.eta_strip[i]))
    al0 = float(aero.secs.alpha0(re, aero.eta_strip[i]))
    cl = CL * w.la[i] + w.lb[i]
    return math.degrees(cl / a0 + al0)


def wing_controls(geom, aero, W, config, choice=None):
    """Lay out ailerons and (config) flaps or flaperons on the current wing.
    `choice` freezes the discrete picks (chord fractions, deflections) once the
    sizing loop has settled, so the loop converges smoothly.
    Leaves `aero` in the chosen landing configuration and returns the layout."""
    rho, mu = aero.atm.rho, aero.atm.mu
    root_af, tip_af = aero.secs.root.name, aero.secs.tip.name
    y_in = geom.fus_w / 2                               # surfaces start at the fuselage side
    y_out = P.aileron_outboard_eta * geom.b / 2
    re_flap = rho * 10.0 * geom.wing.mac / mu
    eta_ail = 0.75                                      # representative aileron station
    re_ail = rho * 10.0 * float(geom.chord_at(eta_ail * geom.b / 2)) / mu
    gap = P.flap_aileron_gap_eta * geom.b / 2
    cfs = P.control_cf_grid

    def approach_alpha(cfg):
        V_app = P.approach_speed_ratio * aero.stall_speed(W, cfg)
        CL_app = W / (0.5 * rho * V_app ** 2 * geom.S)
        return _local_alpha_deg(aero, CL_app, eta_ail, cfg)

    def cfg_now():
        return "landing" if aero.flaps is not None else "clean"

    def best_ailerons(cf_list, alpha):
        cands = [size_ailerons(geom, aero, cf, tip_af, alpha, re_ail, y_in) for cf in cf_list]
        ok = [c for c in cands if c["ok"]]
        return max(ok, key=lambda c: c["y1"]) if ok else max(cands, key=lambda c: c["pb2v"])

    if config == "none":
        aero.clear_flaps()
        a = best_ailerons((choice["aileron_cf"],) if choice else cfs, approach_alpha("clean"))
        return dict(config="none", aileron=a, flap=None, choice=dict(aileron_cf=a["cf"]))

    if config == "flaps":
        # ailerons first (outboard, smallest span) -> flaps get everything inboard
        a = best_ailerons((choice["aileron_cf"],) if choice else cfs, approach_alpha(cfg_now()))
        y1f = max(a["y1"] - gap, y_in + 0.02)
        best = None
        for cf in ((choice["flap_cf"],) if choice else cfs):
            d = choice["flap_deg"] if choice else max_usable_deflection(root_af, cf, re_flap, P.flap_max_deg)
            if d <= 0:
                continue
            flaps = dict(y0=y_in, y1=y1f, section=FlapSection(root_af, cf, d))
            aero.set_flaps(flaps)
            clm = float(aero.CLmax(aero.stall_speed(W, "landing"), "landing"))
            if best is None or clm > best[0]:
                best = (clm, cf, d, flaps)
        if best is None:
            aero.clear_flaps()
            return dict(config="flaps", aileron=a, flap=None,
                        choice=dict(aileron_cf=a["cf"], flap_cf=cfs[0], flap_deg=0.0))
        clm, cf, d, flaps = best
        aero.set_flaps(flaps)
        return dict(config="flaps", aileron=a,
                    flap=dict(cf=cf, deflection=d, y0=y_in, y1=y1f, CLmax_landing=clm),
                    choice=dict(aileron_cf=a["cf"], flap_cf=cf, flap_deg=d))

    if config == "flaperons":
        # full-span surfaces: the largest droop that still leaves pb/2V >= requirement
        cands = []
        for cf in ((choice["aileron_cf"],) if choice else cfs):
            if choice:
                droops = (choice["flap_deg"],)
            else:
                dmax = max_usable_deflection(root_af, cf, re_flap, P.flaperon_max_droop_deg)
                droops = np.arange(dmax, -0.1, -2.0)
            for d in droops:
                d = float(d)
                if d > 0:
                    aero.set_flaps(dict(y0=y_in, y1=y_out, section=FlapSection(root_af, cf, d)))
                else:
                    aero.clear_flaps()
                a = size_ailerons(geom, aero, cf, tip_af, approach_alpha(cfg_now()), re_ail, y_in,
                                  droop=d, full_span=True)
                if a["ok"] or choice:
                    clm = float(aero.CLmax(aero.stall_speed(W, cfg_now()), cfg_now()))
                    cands.append((a["ok"], clm, cf, d, a))
                    break
        if not cands:                                 # nothing meets roll: best effort, no droop
            aero.clear_flaps()
            a = best_ailerons(cfs, approach_alpha("clean"))
            return dict(config="flaperons", aileron=a, flap=None,
                        choice=dict(aileron_cf=a["cf"], flap_deg=0.0))
        ok_, clm, cf, d, a = max(cands, key=lambda t: (t[0], t[1]))
        if d > 0:
            aero.set_flaps(dict(y0=y_in, y1=y_out, section=FlapSection(root_af, cf, d)))
        else:
            aero.clear_flaps()
        flap = dict(cf=cf, deflection=d, y0=y_in, y1=y_out, CLmax_landing=clm,
                    note="flaperon droop") if d > 0 else None
        return dict(config="flaperons", aileron=a, flap=flap, choice=dict(aileron_cf=cf, flap_deg=d))
    raise ValueError(config)


# =====================================================================
# Neutral point, CG range, tail incidence, elevator
# =====================================================================
def fuselage_k_fus(x_ratio):
    """Raymer Fig. 16.14 empirical pitching-moment factor K_fus [per deg] vs. the wing
    root quarter-chord position / fuselage length (quadratic fit of the figure used by
    FAST-GA, compute_k_fuselage.py; valid 0.1-0.62)."""
    x = min(max(x_ratio, 0.1), 0.62)
    return 0.01 - 0.063 * x + 0.211 * x ** 2


def fuselage_dh_n(geom, a_w):
    """Neutral-point shift from the fuselage (destabilizing), fraction of MAC:
    Cm_alpha_fus = K_fus W_f^2 L_f / (c S)  [per deg]  (Raymer eq. 16.25);  dh_n = -Cm_a / a_w."""
    k = fuselage_k_fus(geom.x_qc / geom.fus_L)
    cma = k * geom.fus_w ** 2 * geom.fus_L / (geom.wing.mac * geom.S) * 180 / math.pi
    return -cma / a_w, cma


def tail_and_elevator(geom, aero, perf, pt, m, choice=None, band_frac=None):
    """Returns dict with neutral point, CG range, tail incidence, elevator size
    and the stabilizer area the elevator checks need (scissor plot: the CG band
    of width band_frac [MAC] hangs from the aft limit h_n - static_margin_min)."""
    rho, mu = aero.atm.rho, aero.atm.mu
    W = m * G0
    S, c, AR = geom.S, geom.wing.mac, geom.AR
    S_h, l_t = geom.htail.S, geom.l_t
    tail_af = aero.tail_af
    eta_t = P.tail_efficiency
    a_w = aero.clean.CLa
    re_t = rho * perf["V_s"] * geom.htail.mac / mu
    a2t = float(tail_af.cl_alpha(re_t))
    a_t = lift_slope_3d(P.ar_htail, a2t)
    deda = 2 * a_w / (math.pi * AR)
    V_H = S_h * l_t / (S * c)
    h_ac = 0.25
    dh_fus, cma_fus = fuselage_dh_n(geom, a_w)
    h_n = h_ac + eta_t * V_H * (a_t / a_w) * (1 - deda) + dh_fus
    h_aft = h_n - P.static_margin_min
    if P.htail_sizing == "scissor" and band_frac is not None:
        h_fwd = h_aft - band_frac
    else:
        h_fwd = h_n - P.static_margin_max
    h_mid = 0.5 * (h_aft + h_fwd)

    # loiter trim at mid CG with the elevator neutral -> tail incidence
    CL_lo = perf["loiter"]["CL"]
    cm_clean = aero.cm_ac("clean")
    CLt_lo = (cm_clean + CL_lo * (h_mid - h_ac)) / (eta_t * V_H)
    alpha_t_lo = CLt_lo / a_t                                     # rad
    i_w = aero.clean.alpha_root(CL_lo)                            # fuselage level at loiter
    eps_lo = 2 * CL_lo / (math.pi * AR)
    i_t = alpha_t_lo + eps_lo                                     # relative to fuselage axis

    # (a) landing trim at CLmax, flaps down, forward CG
    cfg = "landing" if aero.flaps is not None else "clean"
    V_sl = aero.stall_speed(W, cfg)
    CL_land = float(aero.CLmax(V_sl, cfg))
    wcfg = aero.landing if cfg == "landing" else aero.clean
    cm_land = aero.cm_ac(cfg)
    CLt_req_land = (cm_land + CL_land * (h_fwd - h_ac)) / (eta_t * V_H)
    alpha_fus_land = wcfg.alpha_root(CL_land) - i_w
    alpha_t_land = alpha_fus_land - 2 * CL_land / (math.pi * AR) + i_t

    # takeoff, two techniques (forward CG, flaps up, full capped thrust at liftoff):
    #  "rotate": fuselage level on the ground, elevator rotates the nose up about the
    #            main wheels at liftoff speed (Raymer Ch.11/17)
    #  "stance": nose-up ground stance so the wing already makes liftoff CL; the
    #            aircraft flies off in that attitude and the elevator only has to
    #            trim it in free air (common on RC tricycle aircraft)
    x_qc = geom.x_qc
    x_lemac = x_qc - 0.25 * c
    x_cg_fwd, x_cg_aft = x_lemac + h_fwd * c, x_lemac + h_aft * c
    z_cg = geom.gear_h + geom.fus_h / 2
    x_mg = x_cg_aft + z_cg * math.tan(math.radians(P.tipback_deg))
    c_eng = float(geom.chord_at(geom.y_eng))
    z_T = geom.gear_h + geom.fus_h + 0.5 * geom.wing.t_c * c_eng
    V_R = perf["takeoff"]["V_lo"]
    q_R = 0.5 * rho * V_R ** 2
    T_R = float(np.interp(V_R, V_GRID, pt.full_throttle()["T"]))
    x_t = x_qc + l_t
    # rotate
    CL_g0 = min(CL_lo, 0.8 * perf["CLmax"])
    M_need = (W * (x_mg - x_cg_fwd) - q_R * S * CL_g0 * (x_mg - x_qc)
              - q_R * S * c * cm_clean + T_R * z_T)
    CLt_req_rot = -max(M_need, 0.0) / (x_t - x_mg) / (eta_t * q_R * S_h)
    alpha_t_rot = alpha_t_lo                                      # ground attitude = loiter attitude
    # stance: free-air trim at liftoff CL
    CL_lof = W / (q_R * S)
    CLt_req_st = (cm_clean + CL_lof * (h_fwd - h_ac) - T_R * (z_T - z_cg) / (q_R * S * c)) / (eta_t * V_H)
    alpha_t_st = (alpha_t_lo + (aero.clean.alpha_root(CL_lof) - i_w)
                  - 2 * (CL_lof - CL_lo) / (math.pi * AR))
    stance_deg = math.degrees(aero.clean.alpha_root(CL_lof) - i_w)

    def tail_cl(cf, alpha_rad):
        """3-D stabilizer CL with elevator full up: the largest trusted throw,
        interpolated where NeuralFoil's confidence crosses foil_conf_min (continuous,
        so the sizing loop does not jump between 2.5-deg steps)."""
        ad = math.degrees(alpha_rad)
        cl0, _ = FlapSection(tail_af.name, cf, 0.0).cl_at(ad, re_t)
        prev = None                                      # (deflection, cl, conf) of the last trusted step
        for de in np.arange(2.5, P.elevator_max_deg + 1e-9, 2.5):
            cle, conf = FlapSection(tail_af.name, cf, -float(de)).cl_at(ad, re_t)
            if conf >= P.foil_conf_min:
                prev = (float(de), cle, conf)
                continue
            if prev is not None:
                f = (prev[2] - P.foil_conf_min) / max(prev[2] - conf, 1e-9)
                d_use = prev[0] + f * (float(de) - prev[0])
                cl_use = prev[1] + f * (cle - prev[1])
                return (a_t / a2t) * (cl0 + P.elevator_span_frac * (cl_use - cl0)), d_use
            break
        if prev is None:
            return (a_t / a2t) * cl0, 0.0
        return (a_t / a2t) * (cl0 + P.elevator_span_frac * (prev[1] - cl0)), prev[0]

    def ratio(cl_avail, cl_req):
        """> 1: the stabilizer can produce the required (negative) lift"""
        return (cl_avail / cl_req) if cl_req < -1e-9 else 10.0

    best = None
    techs = (choice["technique"],) if choice else ("rotate", "stance")
    for tech in techs:
        req_to, a_to = (CLt_req_rot, alpha_t_rot) if tech == "rotate" else (CLt_req_st, alpha_t_st)
        for cf in ((choice["elevator_cf"],) if choice else P.elevator_cf_grid):
            cl_land, de_l = tail_cl(cf, alpha_t_land)
            cl_to, de_t = tail_cl(cf, a_to)
            r_land, r_to = ratio(cl_land, CLt_req_land), ratio(cl_to, req_to)
            cand = dict(cf=cf, throw=min(de_l, de_t), ratio=min(r_land, r_to), ratio_land=r_land,
                        ratio_takeoff=r_to, technique=tech)
            # smallest elevator that works; otherwise the most capable one
            if cand["ratio"] >= 1.0:
                if best is None or best["ratio"] < 1.0 or \
                        (cand["technique"] == best["technique"] and cf < best["cf"]) or \
                        (cand["technique"] != best["technique"] and cand["ratio"] > best["ratio"]):
                    best = cand
                break
            if best is None or (best["ratio"] < 1.0 and cand["ratio"] > best["ratio"]):
                best = cand
    # stabilizer area at which the authority is exactly sufficient (the sizing loop
    # iterates; the Raymer tail-volume value acts as a floor in geometry.build)
    sh_need = S_h / max(best["ratio"], 1e-3)
    tech = best["technique"]
    return dict(h_n=h_n, h_cg_fwd=h_fwd, h_cg_aft=h_aft, x_np=x_lemac + h_n * c,
                x_cg_fwd=x_cg_fwd, x_cg_aft=x_cg_aft, x_main_gear=x_mg, V_H=V_H,
                deps_dalpha=deda, a_tail=a_t, tail_incidence_deg=math.degrees(i_t),
                wing_incidence_deg=math.degrees(i_w), decalage_deg=math.degrees(i_w - i_t),
                CLt_loiter=CLt_lo, CLt_req_landing=CLt_req_land, CLt_req_rotation=CLt_req_rot,
                CLt_req_stance=CLt_req_st, CL_landing=CL_land, elevator=best, sh_needed=sh_need,
                dh_n_fuselage=dh_fus, cm_alpha_fuselage=cma_fus, h_mid=h_mid,
                band_frac=h_aft - h_fwd, i_w=i_w, i_t=i_t, CL_loiter=CL_lo, a_t2=a2t,
                takeoff_technique=tech, stance_deg=stance_deg if tech == "stance" else 0.0,
                CL_ground=CL_lof if tech == "stance" else None,
                choice=dict(elevator_cf=best["cf"], technique=tech))


def elevator_trim(tail_af_name, cf, CLt_req, alpha_t_rad, re_t, a_t, a2t):
    """Elevator deflection that makes the stabilizer produce CLt_req at tail angle
    alpha_t, and the section cd there (NeuralFoil on the deflected section).
    Returns (deflection deg, +down; section cd averaged over span; trusted)."""
    ad = math.degrees(alpha_t_rad)
    f = P.elevator_span_frac
    k3 = a_t / a2t
    base = FlapSection(tail_af_name, cf, 0.0)
    cl0 = base.cl_at(ad, re_t)[0]
    cd0 = base.cd_at(ad, re_t)
    if abs(k3 * cl0 - CLt_req) < 1e-6:
        return 0.0, cd0, True
    sgn = 1.0 if CLt_req > k3 * cl0 else -1.0
    prev = (0.0, k3 * cl0, cd0)
    for d in np.arange(2.5, P.elevator_max_deg + 1e-9, 2.5):
        fs = FlapSection(tail_af_name, cf, sgn * float(d))
        cle, conf = fs.cl_at(ad, re_t)
        CL = k3 * (cl0 + f * (cle - cl0))
        cd = (1 - f) * cd0 + f * fs.cd_at(ad, re_t)
        if (CL - CLt_req) * sgn >= 0:
            w = (CLt_req - prev[1]) / (CL - prev[1]) if abs(CL - prev[1]) > 1e-9 else 1.0
            return sgn * (prev[0] + w * (d - prev[0])), prev[2] + w * (cd - prev[2]), conf >= P.foil_conf_min
        prev = (float(d), CL, cd)
    return sgn * P.elevator_max_deg, prev[2], False


def elevator_trim_drag(geom, aero, tail, CL, V, config):
    """Extra drag (CD on the wing area) of flying trimmed at the FORWARD CG:
    stabilizer profile drag with the elevator deflected (minus the cl = 0 tail drag
    already in Aero.drag) + the stabilizer's induced drag."""
    from aerodynamics import oswald_inviscid
    rho, mu = aero.atm.rho, aero.atm.mu
    S, c, AR = geom.S, geom.wing.mac, geom.AR
    S_h, l_t = geom.htail.S, geom.l_t
    eta_t = P.tail_efficiency
    V_H = S_h * l_t / (S * c)
    wcfg = aero._cfg(config)
    CLt = (aero.cm_ac(config) + CL * (tail["h_cg_fwd"] - 0.25)) / (eta_t * V_H)
    alpha_t = (wcfg.alpha_root(CL) - tail["i_w"]) - 2 * CL / (math.pi * AR) + tail["i_t"]
    re_t = rho * V * geom.htail.mac / mu
    de, cd_e, ok = elevator_trim(aero.tail_af.name, tail["elevator"]["cf"], CLt, alpha_t, re_t,
                                 tail["a_tail"], tail["a_t2"])
    cd_ref = float(aero.tail_af.cd(0.0, re_t))
    e_t = oswald_inviscid(P.ar_htail, P.taper_htail, 0.0, geom.htail.b)
    dCD = (P.q_tail * eta_t * (cd_e - cd_ref) + eta_t * CLt ** 2 / (math.pi * e_t * P.ar_htail)) * S_h / S
    return dict(dCD=float(dCD), CL_tail=float(CLt), elevator_deg=float(de), trusted=bool(ok),
                alpha_tail_deg=math.degrees(alpha_t))


class CrowSection(FlapSection):
    """Aileron raised for crow on the landing rollout. Only the lift loss and the
    drag near the ground attitude matter (no stall point is needed), so the
    increments come straight from the NeuralFoil sweeps where both the clean and
    the raised section are trusted, over alpha in [-2, 6] deg."""
    A_LO, A_HI = -2.0, 6.0

    def _pair(self, re_key):
        cl, cd, cm, conf = _sweep(self.af, self.cf, self.delta, re_key)
        c0, _, m0, f0 = _sweep(self.af, self.cf, 0.0, re_key)
        k = (ALPHAS >= self.A_LO) & (ALPHAS <= self.A_HI) & (conf >= P.foil_conf_min) & (f0 >= P.foil_conf_min)
        return cl, cd, cm, c0, m0, k

    def usable(self, re):
        lo, hi, _ = _buckets(re)
        return all(self._pair(k)[5].sum() >= 4 for k in (lo, hi))

    def props(self, re):
        lo, hi, w = _buckets(re)
        out = dict(d_clmax=0.0, d_alpha0=0.0, d_cm0=0.0, cl_alpha=0.0, clmax=0.0)
        for kk, wt in ((lo, 1 - w), (hi, w)):
            cl, cd, cm, c0, m0, k = self._pair(kk)
            clean = _derived(self.af, self.cf, 0.0, kk, P.foil_conf_min)
            a0 = clean["cl_alpha"] if clean else 2 * math.pi
            if k.sum() < 2:
                dcl = dcm = 0.0
            else:
                dcl, dcm = float(np.mean(cl[k] - c0[k])), float(np.mean(cm[k] - m0[k]))
            vals = dict(d_clmax=dcl, d_alpha0=-dcl / a0, d_cm0=dcm, cl_alpha=a0,
                        clmax=(clean["clmax"] if clean else 1.0) + dcl)
            for key, v in vals.items():
                out[key] += wt * v
        return out

    def cd(self, cl, re):
        cl = np.asarray(cl, float)
        lo, hi, w = _buckets(float(np.mean(re)))
        res = 0.0
        for kk, wt in ((lo, 1 - w), (hi, w)):
            c, d, _, conf = _sweep(self.af, self.cf, self.delta, kk)
            k = (ALPHAS >= -6) & (ALPHAS <= 10) & (conf >= P.foil_conf_min)
            cc, dd = c[k], d[k]
            order = np.argsort(cc)
            res = res + wt * np.interp(cl, cc[order], dd[order])
        return res
