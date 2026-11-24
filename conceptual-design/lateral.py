"""
Coupled lateral-directional trim (Raymer Ch.16 "lateral-directional trim").

Side force, rolling moment and yawing moment are balanced TOGETHER, with
sideslip beta, bank phi, aileron and rudder as the unknowns - the way the
aircraft actually holds an engine failure or a crosswind - instead of sizing
each surface on its own.

  Engine-out  : V_mc = 1.2 V_s, live motor at full capped thrust, dead prop
                stopped (Raymer stopped-prop drag), live-prop slipstream adds
                lift on its wing (actuator disk). Zero-sideslip bank toward the
                live engine, at most 5 deg (14 CFR 23.149); beyond that 5 deg
                with sideslip. Sizes the fin (never below the Raymer
                tail-volume value, kept for the unmodelled Dutch-roll/spiral
                damping) and checks the aileron left over.
  Crosswind   : steady sideslip at the approach speed with a 90-deg crosswind
                of 0.2 V_SO (14 CFR 23.233). Checks rudder and aileron.

Derivatives (per radian, wing area / span reference):
  fin      : side force from NeuralFoil on the fin section at its actual
             angle (sidewash-corrected sideslip) and rudder deflection
             (symmetric section, table over alpha x deflection); 3-D by the
             fin's effective-aspect-ratio lift slope (Raymer Ch.16);
             sidewash (1 + dsigma/dbeta) eta_v = 0.724 + 3.06 (S_v/S)/(1+cos L)
             + 0.4 z_w/d + 0.009 AR (DATCOM / Raymer eq. 16.48)
  fuselage : Cn_beta = -1.3 Vol/(S b) (D/W) (Raymer eq. 16.47);
             CY_beta = -2 K_i S_0/S (DATCOM)
  wing     : Cl_beta of the wing height on the fuselage (DATCOM 5.2.2.1):
             1.2 sqrt(AR) (z_w/b)(2 D/b); no geometric dihedral
  ailerons : Cl from the strip theory used to size them (controls.py);
             adverse yaw from the induced drag, Cn = -(CL / (pi e AR)) Cl
Sign convention: body axes, beta > 0 = wind from the right, phi > 0 = right
wing down, rudder > 0 = trailing edge left (nose left), aileron u in
[-1, 1] of full throw, u > 0 rolls right. The dead engine is the RIGHT one.
"""
import math
from functools import lru_cache

import numpy as np

from aerodynamics import lift_slope_3d
from catalog import airfoil
from parameters import P
from propulsion import V_GRID

FIN_ALPHAS = np.arange(-16.0, 16.01, 1.0)
RE_STEP = 10000


# ------------------------------------------------------------------ fin section table
@lru_cache(maxsize=256)
def _fin_table(af_name, cf, re_key, d_max):
    """cl and NeuralFoil confidence of the fin section over (rudder deflection,
    alpha). Symmetric section: cl(a, -d) = -cl(-a, d), so only d >= 0 is run."""
    import aerosandbox as asb
    import neuralfoil as nf
    base = asb.Airfoil(af_name)
    ds = np.arange(0.0, d_max + 1e-9, 2.5)
    cl = np.zeros((len(ds), len(FIN_ALPHAS)))
    conf = np.zeros_like(cl)
    for i, d in enumerate(ds):
        af = base if d == 0 else base.add_control_surface(deflection=float(d), hinge_point_x=1 - cf)
        r = nf.get_aero_from_coordinates(af.coordinates, alpha=FIN_ALPHAS, Re=float(re_key), model_size="xlarge")
        cl[i] = np.ravel(r["CL"])
        conf[i] = np.ravel(r["analysis_confidence"])
    d_full = np.concatenate([-ds[:0:-1], ds])
    cl_full = np.vstack([-cl[:0:-1, ::-1], cl])
    conf_full = np.vstack([conf[:0:-1, ::-1], conf])
    return d_full, cl_full, conf_full


class FinSection:
    """Fin section cl(alpha, rudder) at a Reynolds number (interpolated between
    Re buckets). Deflection here is the SECTION flap angle (+ = toward the
    pressure side for positive alpha). Lookups go through the 1-D profile over
    deflection at the requested alpha (bilinear overall, linear extrapolation
    beyond the table in deflection)."""

    def __init__(self, af_name, cf, re, d_max=None):
        d_max = P.rudder_max_deg if d_max is None else d_max
        lo = max(int(math.floor(re / RE_STEP) * RE_STEP), RE_STEP)
        w = (re - lo) / RE_STEP
        a, b = _fin_table(af_name, cf, lo, d_max), _fin_table(af_name, cf, lo + RE_STEP, d_max)
        self.d = a[0]
        self._cl = (1 - w) * a[1] + w * b[1]
        self._cf = (1 - w) * a[2] + w * b[2]
        self.d_max = d_max
        self._prof = {}
        h = 2.5
        self.a_alpha = float(self.cl(1.0, 0) - self.cl(-1.0, 0)) / math.radians(2.0)
        self.a_delta = float(self.cl(0, h) - self.cl(0, -h)) / math.radians(2 * h)

    def _profile(self, alpha_deg):
        key = round(float(alpha_deg), 9)
        pr = self._prof.get(key)
        if pr is None:
            A = FIN_ALPHAS
            x = min(max(alpha_deg, A[0]), A[-1])
            i = min(int(np.searchsorted(A, x)) - 1, len(A) - 2)
            i = max(i, 0)
            t = (alpha_deg - A[i]) / (A[i + 1] - A[i])           # (extrapolates in alpha if outside)
            pr = ((1 - t) * self._cl[:, i] + t * self._cl[:, i + 1],
                  (1 - t) * self._cf[:, i] + t * self._cf[:, i + 1])
            if len(self._prof) > 4096:
                self._prof.clear()
            self._prof[key] = pr
        return pr

    @staticmethod
    def _lin(xq, x, y):
        xq = np.asarray(xq, float)
        v = np.interp(xq, x, y)
        lo, hi = xq < x[0], xq > x[-1]
        if np.any(lo):
            v = np.where(lo, y[0] + (xq - x[0]) * (y[1] - y[0]) / (x[1] - x[0]), v)
        if np.any(hi):
            v = np.where(hi, y[-1] + (xq - x[-1]) * (y[-1] - y[-2]) / (x[-1] - x[-2]), v)
        return v

    def cl(self, alpha_deg, d_deg):
        return float(self._lin(d_deg, self.d, self._profile(alpha_deg)[0]))

    def conf(self, alpha_deg, d_deg):
        return float(self._lin(d_deg, self.d, self._profile(alpha_deg)[1]))

    TRUST_BAND = 0.05      # confidence half-width of the soft trust threshold

    def usable(self, alpha_deg, sign):
        """Trusted |deflection| in direction `sign` at alpha. Trust per step ramps
        0 -> 1 as NeuralFoil's confidence goes foil_conf_min -/+ TRUST_BAND and is
        ACCUMULATED along the throw (product), so a dip below the threshold discounts
        everything beyond it. Continuous in alpha and Re - a hard threshold made the
        usable throw jump (15 <-> 22 deg) when a confidence dip crossed 0.7.
        Deflections that unload a fin at high angle can be trusted even where the
        undeflected fin is not (the product starts at the first trusted step)."""
        dd = np.arange(0.0, self.d_max + 1e-9, 0.5)
        c = self._lin(sign * dd, self.d, self._profile(alpha_deg)[1])
        t = np.clip((c - P.foil_conf_min + self.TRUST_BAND) / (2 * self.TRUST_BAND), 0.0, 1.0)
        if not (t > 0).any():
            return 0.0
        i0 = int(np.argmax(t > 0.5)) if (t > 0.5).any() else int(np.argmax(t))
        T = np.cumprod(t[i0:])
        return float(dd[i0] + np.sum(0.5 * (T[1:] + T[:-1])) * 0.5)

    def deflection_for(self, alpha_deg, cl_target, f_span, sign):
        """Section deflection (direction sign) giving the fin cl_target with
        rudder span fraction f_span (may exceed d_max by linear extrapolation -
        flagged by the caller)."""
        prof = self._profile(alpha_deg)[0]
        cl0 = float(self._lin(0.0, self.d, prof))
        dd = np.linspace(0.0, 2 * self.d_max, 401)
        g = (cl0 + f_span * (self._lin(sign * dd, self.d, prof) - cl0) - cl_target) * sign
        if g[0] >= 0:
            return 0.0
        k = np.flatnonzero(g >= 0)
        if not len(k):
            return float(dd[-1])
        j = int(k[0])
        return float(dd[j - 1] + (0 - g[j - 1]) / (g[j] - g[j - 1]) * (dd[j] - dd[j - 1]))


# ------------------------------------------------------------------ derivatives
def body_derivatives(geom):
    """Fuselage + wing-position derivatives (per rad)."""
    from geometry import fuselage_volume_centroid
    S, b = geom.S, geom.b
    vol, _ = fuselage_volume_centroid(geom)
    cn_b_fus = -1.3 * vol / (S * b) * (geom.fus_h / geom.fus_w)
    S0 = math.pi / 4 * geom.fus_w * geom.fus_h
    cy_b_fus = -2 * P.fus_ki_sideforce * S0 / S
    z_w = -geom.fus_h / 2                           # high wing: root quarter chord above the centreline
    cl_b_wing = 1.2 * math.sqrt(geom.AR) * (z_w / b) * (2 * geom.fus_d_eq / b)
    return dict(cn_beta_fus=cn_b_fus, cy_beta_fus=cy_b_fus, cl_beta_wing=cl_b_wing, fus_volume=vol)


def fin_geometry(geom, S_v=None):
    """Fin area, MAC, arm and height of its MAC above the CG (CG ~ on the
    fuselage centreline; fin root at 0.2 fuselage heights above it)."""
    S_v = geom.vtail.S if S_v is None else S_v
    h = math.sqrt(P.ar_vtail * S_v)
    lam = P.taper_vtail
    c_root = 2 * S_v / (h * (1 + lam))
    mac = (2 / 3) * c_root * (1 + lam + lam ** 2) / (1 + lam)
    z_mac = 0.2 * geom.fus_h + h * (1 + 2 * lam) / (3 * (1 + lam))
    return dict(S_v=S_v, h=h, mac=mac, l_v=geom.l_t, z_v=z_mac)


class LateralModel:
    """Linearized fin + body + aileron model at one flight condition, with the
    NeuralFoil fin table for the nonlinear rudder check."""

    def __init__(self, geom, atm, V, CL, rudder_cf, aileron=None, S_v=None, e=0.95):
        self.g, self.V, self.CL = geom, V, CL
        self.q = 0.5 * atm.rho * V ** 2
        fg = fin_geometry(geom, S_v)
        self.fg = fg
        re_v = atm.rho * V * max(fg["mac"], 0.02) / atm.mu
        taf = airfoil(P.tail_airfoil)
        a2 = float(taf.cl_alpha(re_v))
        self.k3d = lift_slope_3d(P.vtail_ar_eff_factor * P.ar_vtail, a2) / a2
        self.eta = P.tail_efficiency
        # (S_v/S clipped to the formula's range of application)
        self.ksw = (0.724 + 3.06 * min(fg["S_v"] / geom.S, 0.3) / 2 + 0.4 * (-geom.fus_h / 2) / geom.fus_h
                    + 0.009 * geom.AR)
        self.sig = self.ksw / self.eta                        # fin angle per unit sideslip
        self.fin = FinSection(P.tail_airfoil, rudder_cf, re_v)
        self.f = P.rudder_span_frac
        self.cf = rudder_cf
        self.bd = body_derivatives(geom)
        Sv_S = fg["S_v"] / geom.S
        # linear fin side force  CY_fin = Fb beta + Fd delta_r   (beta, delta_r in rad)
        self.Fb = -Sv_S * self.eta * self.k3d * self.fin.a_alpha * self.sig
        self.Fd = Sv_S * self.eta * self.k3d * self.f * self.fin.a_delta
        self.lv_b, self.zv_b = fg["l_v"] / geom.b, fg["z_v"] / geom.b
        if aileron is not None:
            y = np.linspace(aileron["y1"], aileron["y2"], 40)
            int_cy = float(np.trapezoid(geom.chord_at(y) * y, y))
            self.cl_a = 2 * aileron["dcl"] * int_cy / (geom.S * geom.b)     # full-throw rolling moment
        else:
            self.cl_a = 0.0
        self.cn_a = -(CL / (math.pi * e * geom.AR)) * self.cl_a
        self.cn_beta = self.bd["cn_beta_fus"] - self.lv_b * self.Fb
        self.cy_beta = self.bd["cy_beta_fus"] + self.Fb
        self.cl_beta = self.bd["cl_beta_wing"] + self.zv_b * self.Fb

    def matrix(self):
        """rows: Y, l, n ; columns: beta, delta_r, u"""
        bd = self.bd
        return np.array([[bd["cy_beta_fus"] + self.Fb, self.Fd, 0.0],
                         [bd["cl_beta_wing"] + self.zv_b * self.Fb, self.zv_b * self.Fd, self.cl_a],
                         [bd["cn_beta_fus"] - self.lv_b * self.Fb, -self.lv_b * self.Fd, self.cn_a]])

    def check(self, beta, CY_fin, full=True):
        """Nonlinear rudder check at the trimmed fin angle: required fin CL vs.
        what the trusted rudder throw gives. Returns ratio (<= 1 ok), the rudder
        deflection [deg, + = TE left] and the fin angle."""
        alpha_true = math.degrees(self.sig * beta)
        alpha = max(-FIN_ALPHAS[-1] + 1, min(FIN_ALPHAS[-1] - 1, alpha_true))   # table range
        CLv_req = -CY_fin * self.g.S / (self.fg["S_v"] * self.eta)       # fin lift, + toward -y
        cl_req = CLv_req / self.k3d
        cl0 = self.fin.cl(alpha, 0.0)
        sign = 1.0 if cl_req >= cl0 else -1.0                             # section flap direction
        d_lim = self.fin.usable(alpha, sign)
        cl_av = cl0 + self.f * (self.fin.cl(alpha, sign * d_lim) - cl0)
        need, have = cl_req - cl0, cl_av - cl0
        ratio = need / have if abs(have) > 1e-9 else (0.0 if abs(need) < 1e-9 else 10.0)
        if alpha != alpha_true:
            ratio = max(ratio, 10.0)            # fin far outside the analysed range: not trimmable
        d_sec = self.fin.deflection_for(alpha, cl_req, self.f, sign) if full else float("nan")
        return dict(ratio=float(min(ratio, 10.0)), rudder_deg=float(-sign * d_sec), rudder_usable_deg=float(d_lim),
                    fin_alpha_deg=alpha_true, CL_fin=CLv_req)


def _solve(M, rhs):
    try:
        return np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(M, rhs, rcond=None)[0]


# ------------------------------------------------------------------ cases
def engine_out_moments(geom, pt, atm, V, CL):
    """Yaw and roll from one failed motor (the right one) at airspeed V."""
    T_one = float(np.interp(V, V_GRID, pt.full_throttle()["T"])) / P.n_motors
    q = 0.5 * atm.rho * V ** 2
    A = math.pi * geom.prop_D ** 2 / 4
    D_dead = P.stopped_prop_dq_per_solidity * P.prop_solidity * A * q
    N = geom.y_eng * (T_one + D_dead)                                  # nose toward the dead engine
    # live-prop slipstream over the wing behind it (actuator disk; wing ~ prop dia/2 behind it)
    v_i = 0.5 * (-V + math.sqrt(V ** 2 + 2 * T_one / (atm.rho * A)))
    x_w = 0.55 * geom.nac_L + 0.01 + 0.25 * float(geom.chord_at(geom.y_eng))
    R = geom.prop_D / 2
    V_w = V + v_i * (1 + x_w / math.sqrt(x_w ** 2 + R ** 2))
    R_s = R * math.sqrt((V + v_i) / V_w)                               # contracted slipstream radius
    c_e = float(geom.chord_at(geom.y_eng))
    ar_s = 2 * R_s / c_e
    k_ll = ar_s / (ar_s + 2)                                           # lifting-line relief of the strip
    dL = ((V_w / V) ** 2 - 1) * q * c_e * 2 * R_s * CL * k_ll
    L_roll = geom.y_eng * dL                                           # rolls toward the dead engine
    Sb = q * geom.S * geom.b
    return dict(T_one=T_one, D_dead_prop=D_dead, yaw_moment=N, roll_moment=L_roll,
                Cn_T=N / Sb, Cl_T=L_roll / Sb, slipstream_dV=V_w - V, slipstream_k=k_ll)


def engine_out_trim(geom, aero, pt, atm, V_s, aileron, rudder_cf, W, S_v=None):
    """Engine-out trim: zero-sideslip bank toward the live (left) engine, <= 5 deg."""
    V = P.vmc_ratio * V_s
    q = 0.5 * atm.rho * V ** 2
    CL = W / (q * geom.S)
    lm = LateralModel(geom, atm, V, CL, rudder_cf, aileron, S_v, e=aero.e)
    mo = engine_out_moments(geom, pt, atm, V, CL)
    M = lm.matrix()
    lim = P.trim_aileron_max_frac

    def trim(phi_deg):
        s = math.sin(math.radians(phi_deg))
        rhs = np.array([-W * s / (q * geom.S), -mo["Cl_T"], -mo["Cn_T"]])
        beta, dr, u = _solve(M, rhs)
        ck = lm.check(beta, lm.Fb * beta + lm.Fd * dr, full=False)
        cand = dict(phi_deg=float(phi_deg), beta_deg=math.degrees(beta), aileron_frac=float(u),
                    _beta=beta, _cy=lm.Fb * beta + lm.Fd * dr, **ck)
        return ck["ratio"] + 10 * max(0.0, abs(u) - lim), cand

    # Bank: ZERO-SIDESLIP trim (bank just enough that gravity balances the fin side
    # force - the minimum-drag engine-out technique), unless that needs more than the
    # 14 CFR 23.149 limit of 5 deg; then 5 deg with the sideslip the equations give.
    # Unique and continuous in the design (choosing the min-rudder bank instead jumps
    # between two bank/sideslip basins).
    g_s = W / (q * geom.S)
    A0 = np.array([[g_s, M[0, 1], 0.0],
                   [0.0, M[1, 1], M[1, 2]],
                   [0.0, M[2, 1], M[2, 2]]])
    s0, _, _ = _solve(A0, np.array([0.0, -mo["Cl_T"], -mo["Cn_T"]]))
    s_lim = math.sin(math.radians(P.engine_out_bank_max_deg))
    phi0 = math.degrees(math.asin(max(-s_lim, min(s_lim, s0))))
    best = trim(phi0)
    out = best[1]
    out.update(lm.check(out.pop("_beta"), out.pop("_cy"), full=True))
    out.update(V=V, CL=CL, **mo, cn_beta=lm.cn_beta, cy_beta=lm.cy_beta, cl_beta=lm.cl_beta,
               S_v=lm.fg["S_v"], rudder_cf=rudder_cf)
    return out


def crosswind_trim(geom, aero, atm, V_s_landing, aileron, rudder_cf, W):
    """Steady sideslip on approach into a 0.2 V_SO crosswind from the right."""
    V = P.approach_speed_ratio * V_s_landing
    V_cw = P.crosswind_ratio * V_s_landing
    beta = math.asin(min(V_cw / V, 0.9))
    q = 0.5 * atm.rho * V ** 2
    CL = W / (q * geom.S)
    lm = LateralModel(geom, atm, V, CL, rudder_cf, aileron, e=aero.e)
    M = lm.matrix()
    # unknowns: sin(phi), delta_r, u   (beta given)
    A = np.array([[W / (q * geom.S), M[0, 1], 0.0],
                  [0.0, M[1, 1], M[1, 2]],
                  [0.0, M[2, 1], M[2, 2]]])
    rhs = -M[:, 0] * beta
    s, dr, u = _solve(A, rhs)
    CY_fin = lm.Fb * beta + lm.Fd * dr
    ck = lm.check(beta, CY_fin)
    return dict(V=V, V_crosswind=V_cw, beta_deg=math.degrees(beta),
                phi_deg=math.degrees(math.asin(max(-1.0, min(1.0, s)))), aileron_frac=float(u), **ck)


def _smallest_fin(geom, aero, pt, atm, V_s, aileron, cf, W, eo_now):
    """Smallest fin area whose best engine-out trim stays inside the trusted rudder
    throw: bisection on log S_v, started from a tight bracket around the secant
    estimate S_v x ratio (robust where the best trim jumps between bank/sideslip
    basins, which a plain 'S_v x ratio' update cannot follow)."""
    f = lambda Sv: engine_out_trim(geom, aero, pt, atm, V_s, aileron, cf, W, S_v=Sv)["ratio"] - 1.0
    S_est = geom.vtail.S * max(eo_now["ratio"], 0.05)
    lo, hi = S_est / 1.03, S_est * 1.03
    flo, fhi = f(lo), f(hi)
    for _ in range(10):                                   # widen until bracketed
        if flo > 0 and fhi <= 0:
            break
        if fhi > 0:
            lo, flo, hi = hi, fhi, hi * 1.15
            fhi = f(hi)
        else:
            hi, fhi, lo = lo, flo, lo / 1.15
            flo = f(lo)
    else:
        return hi
    for _ in range(6):                                    # 3 % -> ~0.1 %
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return hi


def fin_requirements(geom, aero, pt, perf, ctrl, atm, W, rudder_cf=None):
    """Fin area the coupled engine-out trim needs (S_v * rudder ratio: the fin
    side force needed is ~ fixed, the fin's capacity scales with its area),
    best rudder chord fraction, crosswind check and lateral constraints."""
    a = ctrl["aileron"] if ctrl else None
    grid = (rudder_cf,) if rudder_cf else P.rudder_cf_grid
    best = None
    for cf in grid:
        eo = engine_out_trim(geom, aero, pt, atm, perf["V_s"], a, cf, W) if P.engine_out_sizing else None
        need = geom.vtail.S * max(eo["ratio"], 0.05) if eo else 0.0      # (ranking estimate)
        xw = crosswind_trim(geom, aero, atm, perf["V_s_landing"], a, cf, W)
        key = (max(xw["ratio"], 1.0), need)        # crosswind-capable first, then the smallest fin
        if best is None or key < best[0]:
            best = (key, need, cf, eo, xw)
    _, sv_need, cf, eo, xw = best
    if eo:
        sv_need = _smallest_fin(geom, aero, pt, atm, perf["V_s"], a, cf, W, eo)
    lm = LateralModel(geom, atm, perf["loiter"]["V"], perf["loiter"]["CL"], cf, a, e=aero.e)
    lim = P.trim_aileron_max_frac
    cons = dict(
        crosswind_rudder=1 - xw["ratio"],
        crosswind_aileron=(1 - abs(xw["aileron_frac"]) / lim) if a else 0.0,
        engine_out_aileron=(1 - abs(eo["aileron_frac"]) / lim) if (a and eo) else 0.0,
        cn_beta=lm.cn_beta / max(abs(lm.bd["cn_beta_fus"]), 1e-6),
    )
    info = dict(rudder_cf=cf, sv_needed=sv_need, engine_out=eo, crosswind=xw,
                derivatives=dict(cn_beta=lm.cn_beta, cy_beta=lm.cy_beta, cl_beta=lm.cl_beta,
                                 cn_beta_fus=lm.bd["cn_beta_fus"], cl_beta_wing=lm.bd["cl_beta_wing"],
                                 cy_beta_fus=lm.bd["cy_beta_fus"], sidewash_factor=lm.ksw,
                                 cl_aileron_full=lm.cl_a, cn_aileron_full=lm.cn_a),
                cons=cons)
    return sv_need, info


# ------------------------------------------------------------------ screening (no ailerons yet)
@lru_cache(maxsize=512)
def _rudder_at_re(re_key, cf, d_max, conf_min):
    """2-D side-force coefficient of the fin section with the rudder deflected
    (alpha = 0), usable deflection where NeuralFoil's confidence crosses
    conf_min. Returns (cl, usable deflection deg)."""
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
    lo = max(int(math.floor(re / step) * step), step)
    w = (re - lo) / step
    a, b = _rudder_at_re(lo, cf, d_max, conf_min), _rudder_at_re(lo + step, cf, d_max, conf_min)
    return (1 - w) * a[0] + w * b[0], (1 - w) * a[1] + w * b[1]


def fin_engine_out_simple(geom, pt, V_s, atm, cf_fixed=None):
    """Screening only: rudder alone holds the engine-out yaw, no sideslip."""
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
