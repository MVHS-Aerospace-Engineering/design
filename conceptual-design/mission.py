"""
Performance + mission battery-mass-fraction build-up.

Segments (README mission):
  1. Takeoff  - ground roll integrated numerically (Raymer Ch.17) at full
                throttle with the real thrust-vs-speed curve, on soft grass
                (rolling friction from the wheel size, Bekker scaling) and on
                pavement; BMF (grass, the longer one) from the
                image's "specified endurance" equation with the computed time,
                shaft power and eta_b2s.
  2. Climb    - image "Climb Time" equation at the best-rate-of-climb speed,
                V_v and eta_total from the powertrain model.
  3. Loiter   - image "Level Flight Loiter Endurance" equation at the
                minimum-battery-power speed (>= 1.2 V_stall), with L/D,
                eta_prop and eta_b2s computed at that point.
  4. Descent  - power-off glide at best L/D (time only).
  +  Avionics - image "specified endurance" equation on the avionics power
                over the whole flight (Pi 5 + FPV are a real energy load).
Batteries do not get lighter, so segment BMFs ADD.

Landing: trimmed stall speed (forward CG), braked rollout on grass and on
pavement after a free-roll delay, with crow (ailerons up) if P.rollout_crow.
Balked landing: full-throttle climb gradient in landing configuration at the
approach speed, including the elevator trim drag at the forward CG.
"""
import math

import numpy as np

from atmosphere import G0
from parameters import P
from propulsion import (V_GRID, bmf_climb, bmf_loiter, bmf_specified_endurance)


def _refine_min(x, y, i):
    """Parabolic refinement of a grid minimum at index i (continuous result, so
    the optimizer sees smooth functions instead of 0.25 m/s grid jumps)."""
    if 0 < i < len(x) - 1 and np.isfinite(y[i - 1]) and np.isfinite(y[i + 1]):
        denom = y[i - 1] - 2 * y[i] + y[i + 1]
        if denom > 0:
            off = 0.5 * (y[i - 1] - y[i + 1]) / denom
            return x[i] + np.clip(off, -1, 1) * (x[1] - x[0])
    return x[i]


def _constrained_min(x, y, x_min):
    """Minimum of grid function y(x) subject to x >= x_min, continuous in x_min:
    refine the unconstrained minimum, then clamp to the bound."""
    i = int(np.argmin(y))
    return max(float(_refine_min(x, y, i)), x_min)


def _at(xq, x, y):
    return float(np.interp(xq, x, y))


def ground_roll_takeoff(aero, pt, W, CL_g, V_lo, mu):
    rho, S = aero.atm.rho, aero.S
    ft = pt.full_throttle()
    V = np.linspace(0.0, V_lo, 80)
    T = np.interp(V, V_GRID, ft["T"])
    q = 0.5 * rho * V ** 2
    CD, _ = aero.drag(np.full_like(V, CL_g), V, ground=True)
    L, D = q * S * CL_g, q * S * CD
    a = G0 / W * (T - D - mu * np.maximum(W - L, 0.0))
    if np.any(a <= 0.05):
        return dict(s=math.inf, t=math.inf, E_chem=math.inf, E_chem_prop=math.inf, E_shaft=math.inf,
                    V_lo=V_lo, T07=float(np.interp(0.7 * V_lo, V_GRID, ft["T"])))
    s = np.trapezoid(V / a, V) + V_lo * P.rotation_time_s
    t = np.trapezoid(1 / a, V) + P.rotation_time_s
    pc = np.interp(V, V_GRID, ft["P_chem"])
    ps = np.interp(V, V_GRID, ft["P_shaft"])
    pav = pt.p_av_term * pt.batt.v_nom / np.interp(V, V_GRID, ft["v_term"])
    E_chem = np.trapezoid(pc / a, V) + pc[-1] * P.rotation_time_s
    E_av = np.trapezoid(pav / a, V) + pav[-1] * P.rotation_time_s
    E_shaft = np.trapezoid(ps / a, V) + ps[-1] * P.rotation_time_s
    return dict(s=s, t=t, E_chem=E_chem, E_chem_prop=E_chem - E_av, E_shaft=E_shaft,
                V_lo=V_lo, T07=float(np.interp(0.7 * V_lo, V_GRID, ft["T"])))


def ground_roll_landing(aero, W, CL_g, V_td, mu, config="clean", t_free=0.0):
    """Rollout from touchdown: free roll for t_free seconds (counted at V_td, no
    deceleration - conservative), then D + mu (W - L) with mu the effective braking
    (or rolling) coefficient. Idle thrust ~ 0."""
    rho, S = aero.atm.rho, aero.S
    V = np.linspace(0.0, V_td, 60)
    q = 0.5 * rho * V ** 2
    CD, _ = aero.drag(np.full_like(V, CL_g), V, ground=True, config=config)
    L, D = q * S * CL_g, q * S * CD
    a = G0 / W * (D + mu * np.maximum(W - L, 0.0))
    return float(np.trapezoid(V / a, V)) + V_td * t_free


def grass_mu(wheel, load_n):
    """Soft-grass rolling resistance of one wheel by Bekker compaction scaling from
    the reference wheel:  mu ~ (W/W_ref)^(1/(2n+1)) (b_ref/b)^(1/(2n+1)) (D_ref/D)^((n+1)/(2n+1))."""
    n = P.bekker_n
    e1, e2 = 1 / (2 * n + 1), (n + 1) / (2 * n + 1)
    return (P.mu_grass_ref * (max(load_n, 1e-6) / P.grass_ref_wheel_load_n) ** e1
            * (P.grass_ref_wheel_w_m / wheel.w) ** e1 * (P.grass_ref_wheel_d_m / wheel.d) ** e2)


def surface_mu(geom, W, surface):
    """Effective rolling and braking coefficients (weight-share averaged over the
    main and nose wheels; brakes on the mains only)."""
    f = P.main_gear_load_frac
    if surface == "grass":
        m_main = grass_mu(geom.main_wheel, W * f / 2)
        m_nose = grass_mu(geom.nose_wheel, W * (1 - f))
        mb = P.mu_brake_grass
    else:
        m_main = m_nose = P.mu_roll_pavement
        mb = P.mu_brake_pavement
    roll = f * m_main + (1 - f) * m_nose
    brk = geom.brake.active and geom.brake.fits(geom.main_wheel)
    brake = f * max(mb, m_main) + (1 - f) * m_nose if brk else roll
    return dict(roll=roll, brake=brake, main=m_main, nose=m_nose, braked=brk)


def _crow(geom, aero, ctrl):
    """Crow surfaces for the rollout: landing flaps (if any) + ailerons up by
    crow_aileron_up_deg (limited to the largest trusted up-deflection)."""
    from controls import CrowSection
    a = ctrl["aileron"]
    nu = aero.atm.mu / aero.atm.rho
    re = 10.0 * float(geom.chord_at(0.5 * (a["y1"] + a["y2"]))) / nu
    tip_af = aero.secs.tip.name
    d = P.crow_aileron_up_deg
    while d > 0 and not CrowSection(tip_af, a["cf"], -d).usable(re):
        d -= 2.5
    if d <= 0:
        return None
    surfs = [aero.flaps] if aero.flaps is not None else []
    surfs.append(dict(y0=a["y1"], y1=a["y2"], section=CrowSection(tip_af, a["cf"], -d)))
    aero.set_rollout(surfs)
    return dict(deflection_up_deg=d, y0=a["y1"], y1=a["y2"], cf=a["cf"])


def performance(geom, aero, pt, m, battery, ground_cl=None, ctrl=None, tail=None):
    """ground_cl: wing CL in the ground stance (None = fuselage level = loiter CL).
    ctrl: wing control layout (crow on the rollout); tail: previous tail/elevator
    result (forward CG + elevator, for the trim drag on approach)."""
    atm, S = aero.atm, aero.S
    W = m * G0
    V_s = aero.stall_speed(W)
    V_min = P.v_min_ratio * V_s
    V = V_GRID
    q = 0.5 * atm.rho * np.maximum(V, 0.1) ** 2
    CL = W / (q * S)
    CD, parts = aero.drag(CL, V)
    D = q * S * CD
    valid = V >= V_min
    req = pt.required(D)
    ft = pt.full_throttle()

    # --- loiter: minimum battery power (best endurance), V >= 1.2 Vs
    flyable = (V > 1.02 * V_s) & req["ok"]
    pc = np.where(flyable, req["P_chem"], np.inf)
    if not np.isfinite(pc[V >= V_min]).any():
        return None
    V_lo = _constrained_min(V, pc, V_min)
    loiter = dict(V=V_lo, CL=_at(V_lo, V, CL), CD=_at(V_lo, V, CD), D=_at(V_lo, V, D),
                  P_chem=_at(V_lo, V, req["P_chem"]), P_chem_prop=_at(V_lo, V, req["P_chem_prop"]),
                  P_chem_av=_at(V_lo, V, req["P_chem_av"]), P_shaft=_at(V_lo, V, req["P_shaft"]),
                  eta_prop=_at(V_lo, V, req["eta_prop"]), eta_b2s=_at(V_lo, V, req["eta_b2s"]),
                  rpm=_at(V_lo, V, req["rpm"]), I_batt=_at(V_lo, V, req["I_batt"]),
                  duty=_at(V_lo, V, req["duty"]))
    loiter["LD"] = loiter["CL"] / loiter["CD"]
    loiter["eta_total"] = loiter["eta_prop"] * loiter["eta_b2s"]
    loiter["drag_parts"] = {k: _at(V_lo, V, np.broadcast_to(v, V.shape)) for k, v in parts.items()}

    # --- max level speed (full throttle): interpolated zero of thrust - drag
    excess = np.where(valid, ft["T"] - D, -np.inf)
    pos = np.flatnonzero(excess > 0)
    if not len(pos):
        V_max = float(V_min)
    elif pos[-1] == len(V) - 1:
        V_max = float(V[-1])
    else:
        k0 = pos[-1]
        V_max = float(V[k0] + (V[k0 + 1] - V[k0]) * excess[k0] / (excess[k0] - excess[k0 + 1]))

    # --- climb (full throttle): best rate and best gradient, V >= 1.2 Vs
    roc = np.where(valid, (ft["T"] - D) * V / W, -np.inf)
    grad = np.where(valid, (ft["T"] - D) / W, -np.inf)
    V_y = _constrained_min(V, -roc, V_min)
    p_prop = ft["P_chem"] - pt.p_av_term * battery.v_nom / ft["v_term"]
    eta_c = np.where(p_prop > 0, ft["T"] * V / np.maximum(p_prop, 1e-9), 0.0)
    roc_f = np.where(np.isfinite(roc), roc, (ft["T"] - D) * V / W)
    climb = dict(V=V_y, roc=_at(V_y, V, roc_f), grad_max=float(np.max(grad)),
                 LD=_at(V_y, V, CL / CD), eta_total=_at(V_y, V, eta_c),
                 P_chem=_at(V_y, V, ft["P_chem"]), I_batt=_at(V_y, V, ft["I_batt"]))

    # --- ground rolls (fuselage level on tricycle gear; wing incidence set for loiter
    #     -> ground-roll CL = loiter CL). Grass (wheel-size dependent) and pavement.
    CL_g = ground_cl if ground_cl is not None else min(loiter["CL"], 0.8 * float(aero.CLmax(V_s)))
    mu_g, mu_p = surface_mu(geom, W, "grass"), surface_mu(geom, W, "pavement")
    to = ground_roll_takeoff(aero, pt, W, CL_g, P.v_liftoff_ratio * V_s, mu_g["roll"])
    to_pav = ground_roll_takeoff(aero, pt, W, CL_g, P.v_liftoff_ratio * V_s, mu_p["roll"])
    # landing: flaps down (if any), stall speed trimmed at the forward CG. Same ground
    # attitude as takeoff, so the wing's CL on the ground comes from the lifting line at
    # the same root angle - in the ROLLOUT configuration (crow: ailerons up) if used.
    cfg_l = "landing" if aero.flaps is not None else "clean"
    V_sl = aero.stall_speed(W, cfg_l)
    CLmax_l = float(aero.CLmax(V_sl, cfg_l))
    CLmax_l_trim = float(aero.CLmax_trimmed(V_sl, cfg_l))
    crow = None
    cfg_r = cfg_l
    if P.rollout_crow and ctrl is not None and ctrl.get("config") != "flaperons":
        crow = _crow(geom, aero, ctrl)
        if crow is not None:
            cfg_r = "rollout"
    wr = aero._cfg(cfg_r)
    V_td = P.v_touchdown_ratio * V_sl
    CL_gl = wr.CL0 + wr.CLa * aero.clean.alpha_root(CL_g)
    CL_gl = min(CL_gl, 0.8 * CLmax_l, 0.98 * W / (0.5 * atm.rho * V_td ** 2 * S))   # L <= W on the ground
    s_ldg = ground_roll_landing(aero, W, CL_gl, V_td, mu_g["brake"], cfg_r, P.brake_delay_s)
    s_ldg_pav = ground_roll_landing(aero, W, CL_gl, V_td, mu_p["brake"], cfg_r, P.brake_delay_s)
    s_ldg_nocrow = None
    if crow is not None:
        CL_gl0 = aero._cfg(cfg_l).CL0 + aero._cfg(cfg_l).CLa * aero.clean.alpha_root(CL_g)
        CL_gl0 = min(CL_gl0, 0.8 * CLmax_l, 0.98 * W / (0.5 * atm.rho * V_td ** 2 * S))
        s_ldg_nocrow = ground_roll_landing(aero, W, CL_gl0, V_td, mu_g["brake"], cfg_l, P.brake_delay_s)

    # --- approach (motors idle) and balked landing (full throttle), landing config,
    #     trimmed at the forward CG (elevator deflection + tail induced drag)
    V_app = P.approach_speed_ratio * V_sl
    q_app = 0.5 * atm.rho * V_app ** 2
    CL_app = W / (q_app * S)
    CD_app = float(aero.drag(CL_app, V_app, config=cfg_l)[0])
    trim_app = None
    if tail is not None:
        from controls import elevator_trim_drag
        trim_app = elevator_trim_drag(geom, aero, tail, CL_app, V_app, cfg_l)
        CD_app += trim_app["dCD"]
    D_app = q_app * S * CD_app
    T_app = float(np.interp(V_app, V_GRID, ft["T"]))
    approach = dict(V=V_app, CL=CL_app, CD=CD_app, D=D_app, glide_deg=math.degrees(math.atan(CD_app / CL_app)),
                    balked_gradient=(T_app - D_app) / W, T=T_app, trim=trim_app)

    # --- glide (descent, motors off)
    ld = np.where(valid, CL / CD, 0.0)
    k = int(np.argmax(ld))
    glide = dict(V=float(V[k]), LD=float(ld[k]),
                 t=P.climb_altitude_m / (float(V[k]) / float(ld[k])))

    static = dict(T=float(ft["T"][0]), I_motor=float(ft["I_motor"][0]), I_batt=float(ft["I_batt"][0]),
                  rpm=float(ft["rpm"][0]), rpm_limited=bool(ft["rpm_limited"][0]))
    return dict(W=W, m=m, V_s=V_s, CLmax=float(aero.CLmax(V_s)), CLmax_trimmed=float(aero.CLmax_trimmed(V_s)),
                loiter=loiter, climb=climb,
                D_curve=D, V_min=V_min, V_s_landing=V_sl, CLmax_landing=CLmax_l,
                CLmax_landing_trimmed=CLmax_l_trim, V_td=V_td,
                CL_ground_landing=CL_gl, stall_eta=aero.stall_station(V_s),
                stall_eta_landing=aero.stall_station(V_sl, cfg_l),
                takeoff=to, takeoff_pavement=to_pav, s_landing=s_ldg, s_landing_pavement=s_ldg_pav,
                s_landing_nocrow=s_ldg_nocrow, crow=crow, mu_grass=mu_g, mu_pavement=mu_p,
                approach=approach,
                CL_ground=CL_g, V_max=V_max, glide=glide, static=static,
                I_motor_max=float(np.max(ft["I_motor"][valid | (V == 0)])),
                I_batt_max=float(np.max(ft["I_batt"][valid | (V == 0)])))


def mission_bmf(perf, battery, p_avionics_w):
    """Segment BMFs from the image equations with CALCULATED inputs."""
    m, U, K = perf["m"], battery.u_wh_kg, battery.k
    to, cl, lo = perf["takeoff"], perf["climb"], perf["loiter"]
    t_lo_hr = P.endurance_min / 60
    if not math.isfinite(to["s"]) or cl["roc"] <= 0:
        return None

    # 1. takeoff - specified endurance with computed t, mean shaft power, eta_b2s
    t_to_hr = to["t"] / 3600
    p_shaft_kw = to["E_shaft"] / to["t"] / 1000
    eta_b2s_to = to["E_shaft"] / to["E_chem_prop"]
    b_to = bmf_specified_endurance(t_to_hr, p_shaft_kw, m, U, eta_b2s_to, K)
    # 2. climb
    b_cl = bmf_climb(P.climb_altitude_m, cl["V"], cl["roc"], cl["LD"], U, cl["eta_total"], K)
    t_cl = P.climb_altitude_m / cl["roc"]
    # 3. loiter
    b_lo = bmf_loiter(t_lo_hr, lo["V"] * 3.6, lo["LD"], U, lo["eta_total"], K)
    # 4. avionics over the whole flight
    t_total_s = to["t"] + t_cl + P.endurance_min * 60 + perf["glide"]["t"]
    eta_av = p_avionics_w / lo["P_chem_av"]          # BEC + battery-IR losses
    b_av = bmf_specified_endurance(t_total_s / 3600, p_avionics_w / 1000, m, U, eta_av, K)

    total = b_to + b_cl + b_lo + b_av
    # what THIS battery actually delivers: max loiter time
    E_use = K * battery.energy_wh * 3600                          # J (usable)
    E_to = to["E_chem"]
    E_cl = (b_cl * U * K * m * 3600) + lo["P_chem_av"] * t_cl
    E_desc = lo["P_chem_av"] * perf["glide"]["t"]
    t_lo_max = (E_use - E_to - E_cl - E_desc) / lo["P_chem"]
    return dict(takeoff=b_to, climb=b_cl, loiter=b_lo, avionics=b_av, total=total,
                t_climb_s=t_cl, t_total_s=t_total_s, eta_b2s_takeoff=eta_b2s_to,
                battery_mass_required=total * m,
                energy_required_wh=total * m * U,
                endurance_margin=battery.mass_kg / (total * m),
                loiter_max_min=t_lo_max / 60,
                airborne_max_min=(to["t"] + t_cl + t_lo_max + perf["glide"]["t"]) / 60)
