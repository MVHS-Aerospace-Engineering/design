"""
Bottom-up weight build-up (replaces historical regressions, which are fit to
full-size aircraft and do not represent a 1.5 m printed RC twin).

  structure  = material density x wetted area (x internal-structure ratio)
  wing spar  = carbon tube sized from root bending moment at ultimate load
  gear       = aluminum strap / music wire sized from the landing load
  parts      = datasheet masses (data/*.csv)
"""
import math

import numpy as np

from atmosphere import G0
from catalog import components, materials
from parameters import P

CONSTRUCTIONS = ("printed_lwpla", "foam_glass")


def spar_sizing(geom, m_to, n_ult, span_load=None):
    """Carbon tube spar, root bending moment from the Schrenk distribution.
    Returns dict(D, t, mass, util) - util > 1 means it cannot fit in the wing."""
    mat = materials()["cfrp_tube"]
    sigma, rho = mat["allowable_pa"], mat["density_kg_m3"]
    load = span_load if span_load is not None else geom.schrenk * geom.chord
    y_bar = float(np.sum(load * geom.y) / np.sum(load))
    M = n_ult * (m_to * G0 / 2) * y_bar                      # no inertia relief (conservative)
    Z = M / sigma
    D_max = P.spar_depth_frac * geom.wing.t_c * geom.wing.c_root
    t_min, D_min = P.spar_min_wall_m, P.spar_min_od_m

    def z_tube(D, t):
        d = max(D - 2 * t, 0.0)
        return math.pi * (D ** 4 - d ** 4) / (32 * D)

    z_solid_max = z_tube(D_max, D_max / 2)
    util = Z / z_solid_max
    if util >= 1.0:
        D, t = D_max, D_max / 2
    else:
        d4 = D_max ** 4 - 32 * Z * D_max / math.pi
        t = (D_max - d4 ** 0.25) / 2
        D = D_max
        if t < t_min:                      # min gauge: shrink diameter at t_min
            t = t_min
            lo, hi = min(D_min, D_max), D_max
            if z_tube(lo, t) >= Z:
                D = lo
            else:
                for _ in range(50):
                    mid = 0.5 * (lo + hi)
                    lo, hi = (lo, mid) if z_tube(mid, t) >= Z else (mid, hi)
                D = hi
            t = min(t, D / 2)
    L = P.spar_span_frac * geom.b
    mass = rho * math.pi * (D ** 2 - max(D - 2 * t, 0) ** 2) / 4 * L
    return dict(D=D, t=t, mass=mass, util=util, M_root=M, y_bar=y_bar)


def design_cruise_speed(W_S, V_H):
    """14 CFR 23.335(a): V_C >= 33 sqrt(W/S) kt (W/S in psf), and V_C need not
    exceed 0.9 V_H (max level speed). Returns the lowest permissible V_C [m/s]."""
    vc_min = 33 * math.sqrt(W_S * 0.0208854) * 0.514444
    return min(vc_min, 0.9 * V_H)


def gust_load_factor(W_S, V_C, V_stall, rho, mac, cl_alpha):
    """14 CFR 23.341 (SI): n = 1 + Kg rho0 Ude V_eas a / (2 W/S), capped at the
    stall-limited load factor (V_C/V_s)^2 - the wing cannot exceed CLmax, and the
    linear Pratt formula overpredicts badly at gust/airspeed ratios ~1 (RC)."""
    rho0 = 1.225
    mu_g = 2 * W_S / (rho * mac * cl_alpha * G0)
    k_g = 0.88 * mu_g / (5.3 + mu_g)
    v_eas = V_C * math.sqrt(rho / rho0)
    n_pratt = 1 + k_g * rho0 * P.gust_ude_ms * v_eas * cl_alpha / (2 * W_S)
    return min(n_pratt, (V_C / V_stall) ** 2)


def mass_breakdown(geom, wing_af, construction, motor, prop, esc, battery, m_to, n_limit,
                   span_load=None, n_flap_servos=0):
    mat = materials()
    rho_lw = mat["lw_pla_foamed"]["density_kg_m3"]
    out = {}

    # ---- wing
    if construction == "printed_lwpla":
        out["wing_shell"] = rho_lw * P.wing_skin_t_m * geom.wing.S_wet * (1 + P.wing_internal_ratio)
    elif construction == "foam_glass":
        int_c2 = 2 * float(np.sum(geom.chord ** 2)) * geom.dy
        area_ratio = (0.5 * (wing_af.root.area_ratio + wing_af.tip.area_ratio)
                      if hasattr(wing_af, "root") else wing_af.area_ratio)
        core = mat["eps_foam_core"]["density_kg_m3"] * area_ratio * int_c2
        skin = mat["glass_laminate_25gsm"]["areal_kg_m2"] * geom.wing.S_wet
        out["wing_shell"] = core + skin
    else:
        raise ValueError(construction)
    n_ult = n_limit * P.factor_of_safety
    spar = spar_sizing(geom, m_to, n_ult, span_load)
    out["wing_spar"] = spar["mass"]

    # ---- tails, fuselage, nacelles (LW-PLA)
    tail_k = rho_lw * P.tail_skin_t_m * (1 + P.tail_internal_ratio)
    out["htail"] = tail_k * geom.htail.S_wet
    out["vtail"] = tail_k * geom.vtail.S_wet
    out["fuselage"] = rho_lw * P.fus_skin_t_m * geom.fus_S_wet * (1 + P.fus_frames_ratio)
    out["nacelles"] = P.n_motors * (rho_lw * P.nacelle_skin_t_m * geom.nac_S_wet
                                    + P.motor_mount_mass_kg)

    # ---- landing gear (sized to landing load)
    W = m_to * G0
    n_land_ult = P.n_landing_limit * P.factor_of_safety
    al = mat["aluminum_6061_t6"]
    F_leg = n_land_ult * W * P.main_gear_load_frac / 2
    M_leg = F_leg * geom.gear_h                       # 45 deg legs: lateral arm = height
    t_strap = math.sqrt(6 * M_leg / (al["allowable_pa"] * P.gear_plate_width_m))
    main = al["density_kg_m3"] * P.gear_plate_width_m * t_strap * (2 * geom.main_leg_L + geom.fus_w)
    mw = mat["music_wire"]
    F_nose = n_land_ult * W * (1 - P.main_gear_load_frac)
    M_nose = F_nose * 0.25 * geom.nose_strut_L        # trail/side-load arm (ESTIMATE)
    d_wire = max((32 * M_nose / (math.pi * mw["allowable_pa"])) ** (1 / 3), 0.0025)
    nose = mw["density_kg_m3"] * math.pi * d_wire ** 2 / 4 * (geom.nose_strut_L + 0.04)
    out["landing_gear"] = (main + nose + 2 * P.wheel_mass_main_kg + P.wheel_mass_nose_kg
                           + P.nose_steer_mass_kg)

    # ---- propulsion
    out["motors"] = P.n_motors * motor.mass_kg
    out["props_spinners"] = P.n_motors * (prop.mass_kg + P.spinner_mass_kg)
    out["escs"] = P.n_motors * esc.mass_kg if esc is not None else P.n_motors * 0.063
    out["power_wiring"] = 2 * P.n_motors * (geom.y_eng + 0.10) * P.power_wire_kg_m

    # ---- avionics + battery + payload
    out["avionics_servos"] = sum(c.qty * c.mass_kg for c in components())
    out["flap_servos"] = n_flap_servos * P.flap_servo_mass_kg
    out["growth_allowance"] = P.weight_growth_allowance * sum(out.values())
    out["battery"] = battery.mass_kg
    out["payload"] = P.payload_kg

    info = dict(spar=spar, n_limit=n_limit, n_ult=n_ult, gear_strap_t=t_strap, nose_wire_d=d_wire)
    return out, info


def avionics_power_w():
    return P.avionics_power_factor * sum(c.qty * c.power_w for c in components())
