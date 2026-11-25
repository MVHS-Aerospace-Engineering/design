"""
Lateral-directional dynamic stability (linear, small perturbations) and wing
dihedral sizing.

State [beta, p, r, phi] in stability axes, level flight (Etkin & Reid,
"Dynamics of Flight", ch. 4-6; Nelson, "Flight Stability and Automatic
Control", ch. 5):

    beta' = Yb/V beta + Yp/V p + (Yr/V - 1) r + g/V phi
    p'    = Lb beta + Lp p + Lr r          (product of inertia Ixz neglected)
    r'    = Nb beta + Np p + Nr r
    phi'  = p

Eigenvalues -> roll mode (fast real), spiral (slow real), Dutch roll (complex
pair). Requirements: MIL-F-8785C Level 1, Class I (small light aircraft):
    Dutch roll  zeta >= 0.08 and zeta*omega >= 0.15 rad/s   (Cat B and C)
    spiral      time to double amplitude >= 20 s (Cat B: loiter/cruise),
                >= 12 s (Cat C: approach) - a stable spiral passes
    roll mode   time constant <= 1.4 s (Cat B), <= 1.0 s (Cat C)
These are piloted full-size criteria; a small, autopilot-flown aircraft is
more forgiving on the spiral, so they are conservative here.

Derivatives (per rad, wing area / span reference):
  sideslip   CY_b, Cl_b, Cn_b from lateral.LateralModel (fin from NeuralFoil
             slope + sidewash, fuselage, wing height, dihedral - see
             lateral.body_derivatives)
  roll rate  Cl_p wing = -(4 / (S b^2)) int a c y^2 dy x (CL_a / a)  (strip
             theory with a lifting-line correction), fin 2 (z_v/b)^2 CY_b,v;
             Cn_p wing = -CL/8, fin -2 (l_v z_v / b^2) CY_b,v;
             CY_p fin = 2 (z_v/b) CY_b,v
  yaw rate   Cn_r fin = 2 (l_v/b)^2 CY_b,v, wing -CD/4;
             Cl_r wing = CL/4, fin -2 (l_v z_v / b^2) CY_b,v;
             CY_r fin = -2 (l_v/b) CY_b,v
  (CY_b,v = fin side-force slope WITHOUT sidewash for the rate terms)
Inertia: from the balance's component stations (x) and each item's spanwise
spread (wing shell chord-weighted over the span, spar as a bar, motors at the
motor station, main wheels at the track, ...); Ixz neglected.
"""
import math

import numpy as np

from atmosphere import G0
from parameters import P


# ------------------------------------------------------------------ inertia
def inertia(geom, items, x_cg):
    """Ixx, Izz [kg m^2] from the balance items (lists in weights.component_stations)."""
    semi = geom.b / 2
    y = np.linspace(0, semi, 60)
    c = geom.chord_at(y)
    r2_wing = float(np.trapezoid(c * y ** 2, y) / np.trapezoid(c, y))       # chord-weighted y^2
    L_sp = P.spar_span_frac * geom.b
    hb = geom.htail.b / 2
    z_v = 0.2 * geom.fus_h + geom.vtail.b * (1 + 2 * P.taper_vtail) / (3 * (1 + P.taper_vtail))
    r2 = {"wing shell": r2_wing, "wing spar": L_sp ** 2 / 12, "power wiring": geom.y_eng ** 2 / 3,
          "nacelles + mounts": geom.y_eng ** 2, "motors": geom.y_eng ** 2, "props + spinners": geom.y_eng ** 2,
          "ESCs": geom.y_eng ** 2, "main gear + wheels": (geom.track / 2) ** 2 * 0.8,
          "wheel brakes": (geom.track / 2) ** 2, "horizontal tail": hb ** 2 / 3, "fin": z_v ** 2,
          "aileron servos": (0.6 * semi) ** 2, "flap servos": (0.3 * semi) ** 2}
    r2_fus = (geom.fus_w ** 2 + geom.fus_h ** 2) / 12
    Ixx = Izz = 0.0
    for it in items:
        nm = it["name"]
        ry = r2_wing if "wing half" in nm else (hb ** 2 / 3 if "tail half" in nm else r2.get(nm, r2_fus))
        Ixx += it["m"] * ry
        Izz += it["m"] * (ry + (it["x"] - x_cg) ** 2)
    return Ixx, Izz


# ------------------------------------------------------------------ derivatives + modes
def lateral_derivatives(lm, geom, aero, CL, CD, dihedral_deg):
    """Nondimensional stability derivatives at one flight condition (lm: lateral.LateralModel)."""
    S, b = geom.S, geom.b
    a_w = aero.CL_alpha
    Fv = lm.Fb / lm.sig                                   # fin CY per rad of fin angle (no sidewash)
    lv, zv = lm.lv_b, lm.zv_b
    y = np.linspace(0, b / 2, 60)
    a0 = float(np.mean(aero.a0_c))
    clp_strip = -4 * a0 * float(np.trapezoid(geom.chord_at(y) * y ** 2, y)) / (S * b ** 2)
    return dict(
        CYb=lm.cy_beta, Clb=lm.cl_beta, Cnb=lm.cn_beta, Clb_dihedral=lm.bd["cl_beta_dihedral"],
        CYp=2 * zv * Fv, Clp=clp_strip * (a_w / a0) + 2 * zv ** 2 * Fv, Cnp=-CL / 8 - 2 * lv * zv * Fv,
        CYr=-2 * lv * Fv, Clr=CL / 4 - 2 * lv * zv * Fv, Cnr=2 * lv ** 2 * Fv - CD / 4)


def modes(der, m, Ixx, Izz, rho, V, S, b):
    q = 0.5 * rho * V ** 2
    Yb, Yp, Yr = q * S * der["CYb"] / m, q * S * b * der["CYp"] / (2 * m * V), q * S * b * der["CYr"] / (2 * m * V)
    Lb, Lp, Lr = q * S * b * der["Clb"] / Ixx, q * S * b ** 2 * der["Clp"] / (2 * V * Ixx), q * S * b ** 2 * der["Clr"] / (2 * V * Ixx)
    Nb, Np, Nr = q * S * b * der["Cnb"] / Izz, q * S * b ** 2 * der["Cnp"] / (2 * V * Izz), q * S * b ** 2 * der["Cnr"] / (2 * V * Izz)
    A = np.array([[Yb / V, Yp / V, Yr / V - 1, G0 / V],
                  [Lb, Lp, Lr, 0.0],
                  [Nb, Np, Nr, 0.0],
                  [0.0, 1.0, 0.0, 0.0]])
    ev = np.linalg.eigvals(A)
    cplx = [e for e in ev if abs(e.imag) > 1e-6]
    real = sorted([e.real for e in ev if abs(e.imag) <= 1e-6])
    out: dict = dict(eigenvalues=[complex(e) for e in ev])
    if cplx:
        e = max(cplx, key=lambda z: z.imag)
        wn = abs(e)
        out.update(dutch_roll_omega=wn, dutch_roll_zeta=-e.real / wn, dutch_roll_zeta_omega=-e.real,
                   dutch_roll_period=2 * math.pi / abs(e.imag))
    else:                                                     # overdamped / split Dutch roll
        out.update(dutch_roll_omega=0.0, dutch_roll_zeta=1.0, dutch_roll_zeta_omega=1.0, dutch_roll_period=math.inf)
        real = sorted(e.real for e in ev)
    lam_roll = min(real) if real else -10.0                  # most negative real root = roll mode
    lam_spiral = max(real) if real else 0.0                   # slowest real root = spiral
    out["roll_tau"] = -1 / lam_roll if lam_roll < 0 else math.inf
    out["spiral_lambda"] = lam_spiral
    out["spiral_t2"] = (math.log(2) / lam_spiral) if lam_spiral > 1e-9 else math.inf   # time to double (unstable)
    out["spiral_stable"] = lam_spiral <= 1e-9
    return out


def conditions(geom, aero, perf):
    """(name, V, CL, CD, category) for loiter (Cat B) and approach (Cat C)."""
    lo, ap = perf["loiter"], perf["approach"]
    return (("loiter", lo["V"], lo["CL"], lo["CD"], "B"), ("approach", ap["V"], ap["CL"], ap["CD"], "C"))


def _base(geom, aero, perf, atm, m, items, x_cg, rudder_cf, aileron):
    """Derivatives at zero dihedral for each condition + Cl_beta per degree of dihedral."""
    import lateral
    Ixx, Izz = inertia(geom, items, x_cg)
    out = []
    for name, V, CL, CD, cat in conditions(geom, aero, perf):
        lm = lateral.LateralModel(geom, atm, V, CL, rudder_cf, aileron, e=aero.e, dihedral_deg=0.0)
        der = lateral_derivatives(lm, geom, aero, CL, CD, 0.0)
        k = lateral.body_derivatives(geom, 1.0)["cl_beta_dihedral"]          # per degree (linear)
        out.append((name, V, cat, der, k))
    return Ixx, Izz, out


def _modes_at(base, m, atm, geom, dihedral_deg):
    Ixx, Izz, conds = base
    res = {}
    for name, V, cat, der0, k in conds:
        der = dict(der0, Clb=der0["Clb"] + k * dihedral_deg, Clb_dihedral=k * dihedral_deg)
        md: dict = modes(der, m, Ixx, Izz, atm.rho, V, geom.S, geom.b)
        t2_req = P.spiral_t2_min_B_s if cat == "B" else P.spiral_t2_min_C_s
        tau_max = P.roll_tau_max_B_s if cat == "B" else P.roll_tau_max_C_s
        md.update(V=V, category=cat, derivatives=der,
                  margin_dutch=min(md["dutch_roll_zeta"] / P.dutch_roll_zeta_min,
                                   md["dutch_roll_zeta_omega"] / P.dutch_roll_zeta_omega_min) - 1,
                  margin_spiral=(1.0 if md["spiral_stable"] else min(md["spiral_t2"] / t2_req - 1, 1.0)),
                  margin_roll=tau_max / max(md["roll_tau"], 1e-6) - 1)
        res[name] = md
    return dict(Ixx=Ixx, Izz=Izz, dihedral_deg=dihedral_deg, cases=res)


def evaluate_modes(geom, aero, perf, atm, m, items, x_cg, rudder_cf, aileron, dihedral_deg):
    return _modes_at(_base(geom, aero, perf, atm, m, items, x_cg, rudder_cf, aileron), m, atm, geom, dihedral_deg)


def size_dihedral(geom, aero, perf, atm, m, items, x_cg, rudder_cf, aileron):
    """Smallest dihedral (0..dihedral_max_deg) whose spiral meets its time-to-double
    requirement at every condition (dihedral costs a little lift and Dutch-roll
    damping, so no more than needed). Returns (dihedral_deg, modes result)."""
    base = _base(geom, aero, perf, atm, m, items, x_cg, rudder_cf, aileron)

    def worst(g):
        r = _modes_at(base, m, atm, geom, g)
        return min(c["margin_spiral"] for c in r["cases"].values()), r
    w0, r0 = worst(0.0)
    if w0 >= 0:
        return 0.0, r0
    gmax = P.dihedral_max_deg
    wm, rm = worst(gmax)
    if wm < 0:
        return gmax, rm
    lo, hi = 0.0, gmax
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if worst(mid)[0] < 0 else (lo, mid)
    return hi, worst(hi)[1]


# ------------------------------------------------------------------ longitudinal
def longitudinal_modes(geom, aero, perf, tail, items, m, atm):
    """Short period (Nelson ch. 4 short-period approximation) and phugoid
    (Lanchester approximation) at loiter, at the forward and aft CG limits.

    CL_a  = a_w + eta a_t (S_h/S)(1 - de/da);  Cm_a = -CL_a (h_n - h)
    Cm_q  = -2 eta a_t V_H (l_t/c);  Cm_adot = Cm_q de/da
    w_sp^2 = Z_a M_q / V - M_a;  2 zeta w_sp = -(M_q + M_adot + Z_a / V)
    phugoid: w_p = sqrt(2) g / V, zeta_p = (CD/CL) / sqrt(2)"""
    S, c = geom.S, geom.wing.mac
    lo = perf["loiter"]
    V, CL, CD = lo["V"], lo["CL"], lo["CD"]
    q = 0.5 * atm.rho * V ** 2
    eta, a_t, V_H, deda = P.tail_efficiency, tail["a_tail"], tail["V_H"], tail["deps_dalpha"]
    lt_c = geom.l_t / c
    CLa = aero.CL_alpha + eta * a_t * (geom.htail.S / S) * (1 - deda)
    Cmq = -2 * eta * a_t * V_H * lt_c
    Cmad = Cmq * deda
    out = dict(V=V, CL_alpha=CLa, Cm_q=Cmq, Cm_alphadot=Cmad, cases={})
    for name, h, x_cg in (("forward CG", tail["h_cg_fwd"], tail["x_cg_fwd"]), ("aft CG", tail["h_cg_aft"], tail["x_cg_aft"])):
        Iyy = sum(it["m"] * ((it["x"] - x_cg) ** 2 + (geom.fus_h / 2) ** 2 / 3) for it in items)
        Cma = -CLa * (tail["h_n"] - h)
        Za = -q * S * (CLa + CD) / m
        Ma = q * S * c * Cma / Iyy
        Mq = q * S * c ** 2 * Cmq / (2 * V * Iyy)
        Mad = q * S * c ** 2 * Cmad / (2 * V * Iyy)
        w2 = Za * Mq / V - Ma
        if w2 > 0:
            w = math.sqrt(w2)
            z = -(Mq + Mad + Za / V) / (2 * w)
        else:                                            # aperiodic / divergent
            w, z = 0.0, (10.0 if Ma < 0 else -1.0)
        out["cases"][name] = dict(static_margin=tail["h_n"] - h, Cm_alpha=Cma, Iyy=Iyy, short_period_omega=w,
                                  short_period_zeta=z,
                                  margin=min(z / P.short_period_zeta_min, P.short_period_zeta_max / max(z, 1e-6)) - 1)
    zp = (CD / CL) / math.sqrt(2)
    out.update(phugoid_omega=math.sqrt(2) * G0 / V, phugoid_period=2 * math.pi * V / (math.sqrt(2) * G0),
               phugoid_zeta=zp, margin_phugoid=zp / P.phugoid_zeta_min - 1)
    return out
