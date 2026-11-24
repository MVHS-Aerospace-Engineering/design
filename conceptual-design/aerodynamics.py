"""
Drag polar + lift curve (README critical input #2).

  Wing lift         : Prandtl lifting-line (Glauert Fourier solution) with the
                      real chord, washout (geometric twist), spanwise airfoil
                      blend root -> tip (aerodynamic twist) and deflected flaps.
                      Because the problem is linear in angle of attack, the
                      local lift is  cl(y) = CL * la(y) + lb(y)  ("additional"
                      + "basic" loading), solved once per configuration.
  Wing profile drag : strip integration of NeuralFoil section polars at the
                      local Reynolds number and local cl from the lifting line
                      (captures low-Re laminar-bubble drag and the lift-
                      dependent part of profile drag).
  Induced drag      : from the same lifting-line solution (exact for the twist
                      and loading), times the Nita & Scholz fuselage factor.
  CLmax             : critical-section method - the wing stalls when the first
                      strip reaches its own clmax at its own Reynolds number.
  Tail profile drag : section polar at cl = 0, tail Reynolds number.
  Fuselage/nacelles : Raymer Ch.12 component build-up  Cf * FF * Q * Swet/Sref
                      with Prandtl-Schlichting mixed laminar/turbulent Cf and
                      Raymer's surface-roughness cutoff Reynolds number.
  Landing gear      : Raymer Table 12.5 D/q per frontal area.
"""
import math

import numpy as np

from parameters import P

V_REF = 12.0      # m/s, reference speed for section lift slopes / zero-lift angles


def cf_flat_plate(re, length):
    """Raymer eq. 12.27 (M~0) with Prandtl-Schlichting transition correction
    and the roughness cutoff Reynolds number (Raymer eq. 12.28)."""
    re = np.asarray(re, float)
    re_cut = 38.21 * (length / P.roughness_k_m) ** 1.053
    re_e = np.clip(np.minimum(re, re_cut), 1e3, None)
    cf_turb = 0.455 / np.log10(re_e) ** 2.58 - P.re_transition_A / re_e
    cf_lam = 1.328 / np.sqrt(re_e)
    return np.maximum(cf_turb, cf_lam)


def oswald_inviscid(AR, taper, d_fus, b):
    """Nita & Scholz (2012): e = e_theo * k_eF, Hoerner taper function f(lambda).
    (Kept as an independent check on the lifting line - see tests.)"""
    dl = -0.357 + 0.45 * math.exp(0.0375 * 0.0)          # unswept quarter chord
    lam = taper - dl
    f = 0.0524 * lam ** 4 - 0.15 * lam ** 3 + 0.1659 * lam ** 2 - 0.0706 * lam + 0.0119
    e_theo = 1.0 / (1.0 + f * AR)
    return e_theo * fuselage_span_factor(d_fus, b)


def fuselage_span_factor(d_fus, b):
    """Nita & Scholz (2012) k_eF = 1 - 2 (d_F / b)^2"""
    return 1.0 - 2.0 * (d_fus / b) ** 2


def lift_slope_3d(AR, cla_2d, F=1.0, s_exp_ratio=1.0):
    """Raymer eq. 12.6 (DATCOM), unswept, M~0. Returns per radian.
    Raymer: if (S_exp/S_ref)*F > 1, use 0.98."""
    eta = cla_2d / (2 * math.pi)
    base = 2 * math.pi * AR / (2 + math.sqrt(4 + (AR / eta) ** 2))
    factor = s_exp_ratio * F
    return base * (0.98 if factor > 1.0 else factor)


# =====================================================================
# Lifting line
# =====================================================================
class LiftingLine:
    """Symmetric Glauert solution, odd Fourier terms, N collocation points on
    the half span (theta in (0, pi/2], root at pi/2)."""

    def __init__(self, b, S, chord_fn, N=24):
        th = (np.arange(N) + 1) * np.pi / (2 * N)
        self.N, self.b, self.S, self.AR = N, b, S, b * b / S
        self.y = b / 2 * np.cos(th)
        self.c = chord_fn(self.y)
        self.n = 2 * np.arange(N) + 1
        self.sin_nt = np.sin(np.outer(th, self.n))
        self.sin_t = np.sin(th)

    def solve(self, a0, alpha_eff):
        """a0: section lift slopes [1/rad]; alpha_eff = alpha_geo - alpha0 [rad]."""
        mu = a0 * self.c / (4 * self.b)
        M = self.sin_nt * (self.sin_t[:, None] + self.n[None, :] * mu[:, None])
        return np.linalg.solve(M, mu * self.sin_t * alpha_eff)

    def cl_local(self, A):
        return 4 * self.b * (self.sin_nt @ A) / self.c


class WingConfig:
    """Lifting-line result for one configuration (clean, flaps down, ...):
    cl(y) = CL*la(y) + lb(y) on the drag strips; CDi = k2 CL^2 + k1 CL + k0."""

    def __init__(self, ll, a0, alpha0, twist, y_strips):
        A1 = ll.solve(a0, np.ones_like(a0))                 # unit root angle
        A0 = ll.solve(a0, twist - alpha0)                   # twist + camber, root angle 0
        self.CLa = math.pi * ll.AR * A1[0]
        self.CL0 = math.pi * ll.AR * A0[0]
        u = A1 / self.CLa
        v = A0 - self.CL0 * u
        piA = math.pi * ll.AR
        self.k2 = piA * float(np.sum(ll.n * u * u))
        self.k1 = 2 * piA * float(np.sum(ll.n * u * v))
        self.k0 = piA * float(np.sum(ll.n * v * v))
        self.e = 1.0 / (piA * self.k2)                     # span efficiency (additional loading)
        order = np.argsort(ll.y)
        la_c = ll.cl_local(u)
        lb_c = ll.cl_local(v)
        self.la = np.interp(y_strips, ll.y[order], la_c[order])
        self.lb = np.interp(y_strips, ll.y[order], lb_c[order])

    def alpha_root(self, CL):
        """Root geometric angle of attack [rad] (relative to the zero-incidence
        reference) needed for CL."""
        return (CL - self.CL0) / self.CLa

    def cdi(self, CL):
        return self.k2 * CL ** 2 + self.k1 * CL + self.k0


class WingSections:
    """Spanwise airfoil: linear loft from root airfoil (eta = 0) to tip airfoil
    (eta = 1) - aerodynamic twist when they differ. Properties are blended
    linearly in span (standard approximation for a lofted wing)."""

    def __init__(self, root_af, tip_af):
        self.root, self.tip = root_af, tip_af
        self.same = root_af.name == tip_af.name

    def _blend(self, f, eta, *args):
        if self.same:
            return getattr(self.root, f)(*args)
        w = np.clip(eta, 0, 1)
        return (1 - w) * getattr(self.root, f)(*args) + w * getattr(self.tip, f)(*args)

    def cd(self, cl, re, eta):
        return self._blend("cd", eta, cl, re)

    def clmax(self, re, eta):
        return self._blend("clmax", eta, re)

    def cl_alpha(self, re, eta):
        return self._blend("cl_alpha", eta, re)

    def alpha0(self, re, eta):
        return self._blend("alpha0", eta, re)

    def cm0(self, re, eta):
        return self._blend("cm0", eta, re)

    @property
    def name(self):
        return self.root.name if self.same else f"{self.root.name}->{self.tip.name}"


class Aero:
    """Aircraft aerodynamics. `flaps` (optional) describes deflected trailing-
    edge surfaces for the landing configuration (see controls.py):
        dict(y0, y1, section=<FlapSection-like with props(re), cd(cl, re)>)"""

    def __init__(self, geom, wing_secs, tail_af, atm, flaps=None):
        self.g, self.secs, self.tail_af, self.atm = geom, wing_secs, tail_af, atm
        g = geom
        self.S = g.S
        nu = atm.mu / atm.rho
        self.k_eF = fuselage_span_factor(g.fus_w, g.b)
        # ground effect (Raymer eq. 12.61)
        hb = (g.wing_height / g.b) ** 1.5
        self.ground_factor = 33 * hb / (1 + 33 * hb)
        # form factors (Raymer 12.31, 12.32)
        f_f = g.fus_L / g.fus_d_eq
        self.ff_fus = 1 + 60 / f_f ** 3 + f_f / 400
        f_n = g.nac_L / g.nac_d
        self.ff_nac = 1 + 0.35 / f_n
        # landing gear, Raymer Table 12.5 (frontal areas)
        a_wheels = 2 * P.wheel_d_main_m * P.wheel_w_main_m
        a_nosewheel = P.wheel_d_nose_m * P.wheel_w_nose_m
        mat_t = 0.003    # nominal strap thickness for drag only (weights.py sizes it)
        a_legs = 2 * g.main_leg_L * mat_t
        a_nose_strut = g.nose_strut_L * 0.003
        self.CD_gear = P.q_gear * (P.dq_wheel * (a_wheels + a_nosewheel)
                                   + P.dq_flat_leg * a_legs
                                   + P.dq_round_strut * a_nose_strut) / self.S

        # ---- lifting line, clean configuration
        self.eta_strip = 2 * g.y / g.b
        self.ll = LiftingLine(g.b, g.S, g.chord_at)
        eta_c = 2 * self.ll.y / g.b
        re_c = V_REF * self.ll.c / nu
        self.a0_c = self.secs.cl_alpha(re_c, eta_c)
        self.al0_c = self.secs.alpha0(re_c, eta_c)
        self.twist_c = g.twist_at(self.ll.y)
        self.clean = WingConfig(self.ll, self.a0_c, self.al0_c, self.twist_c, g.y)
        self.CL_alpha = self.clean.CLa
        self.e = self.clean.e * self.k_eF
        self.flaps = None
        self.landing = self.clean
        if flaps is not None:
            self.set_flaps(flaps)

    # ------------------------------------------------------------------ flaps
    def clear_flaps(self):
        self.flaps, self.landing = None, self.clean

    def set_flaps(self, flaps):
        """Landing configuration: deflected surfaces shift the local zero-lift
        angle and raise the local clmax over their span (NeuralFoil on the
        deflected section, see controls.FlapSection)."""
        g = self.g
        nu = self.atm.mu / self.atm.rho
        self.flaps = flaps
        in_c = (self.ll.y >= flaps["y0"]) & (self.ll.y <= flaps["y1"])
        in_s = (g.y >= flaps["y0"]) & (g.y <= flaps["y1"])
        re_ref = V_REF * g.wing.mac / nu
        fp = flaps["section"].props(re_ref)
        a0 = np.where(in_c, fp["cl_alpha"], self.a0_c)
        al0 = np.where(in_c, self.al0_c + fp["d_alpha0"], self.al0_c)
        self.landing = WingConfig(self.ll, a0, al0, self.twist_c, g.y)
        self._flap_strips = in_s
        self._flap_dclmax = fp["d_clmax"]

    # ------------------------------------------------------------------ drag
    def _cfg(self, config):
        return self.landing if config == "landing" else self.clean

    def drag(self, CL, V, ground=False, config="clean"):
        """CD (and breakdown) for lift coefficient CL at true airspeed V.
        CL and V broadcast; returns (CD, parts dict)."""
        g, a = self.g, self.atm
        w = self._cfg(config)
        CL, V = np.broadcast_arrays(np.asarray(CL, float), np.asarray(V, float))
        V = np.maximum(V, 0.5)
        nu = a.mu / a.rho
        # wing profile (strips)
        cl_loc = CL[..., None] * w.la + w.lb
        re_loc = V[..., None] * g.chord / nu
        cd_loc = self.secs.cd(cl_loc, re_loc, self.eta_strip)
        if config == "landing" and self.flaps is not None:
            fs = self.flaps["section"]
            cd_f = fs.cd(cl_loc[..., self._flap_strips], re_loc[..., self._flap_strips])
            cd_loc = cd_loc.copy()
            cd_loc[..., self._flap_strips] = cd_f
        CD_wing = 2 * np.sum(cd_loc * g.chord, axis=-1) * g.dy / self.S
        # tails
        CD_ht = P.q_tail * self.tail_af.cd(0.0 * V, V * g.htail.mac / nu) * g.htail.S / self.S
        CD_vt = P.q_tail * self.tail_af.cd(0.0 * V, V * g.vtail.mac / nu) * g.vtail.S / self.S
        # fuselage & nacelles
        CD_fus = (cf_flat_plate(V * g.fus_L / nu, g.fus_L) * self.ff_fus * P.q_fuselage
                  * g.fus_S_wet / self.S)
        CD_nac = (P.n_motors * cf_flat_plate(V * g.nac_L / nu, g.nac_L) * self.ff_nac
                  * P.q_nacelle * g.nac_S_wet / self.S)
        CD_gear = self.CD_gear * np.ones_like(V)
        profile = CD_wing + CD_ht + CD_vt + CD_fus + CD_nac + CD_gear
        CD_leak = P.leak_protub_frac * profile
        CD_i = w.cdi(CL) / self.k_eF * (self.ground_factor if ground else 1.0)
        CD = profile + CD_leak + CD_i
        parts = dict(wing_profile=CD_wing, htail=CD_ht, vtail=CD_vt, fuselage=CD_fus,
                     nacelles=CD_nac, gear=CD_gear, leakage=CD_leak, induced=CD_i)
        return CD, parts

    # ------------------------------------------------------------------ lift
    def strip_clmax(self, V, config="clean"):
        g, a = self.g, self.atm
        re_loc = np.asarray(V, float)[..., None] * g.chord * a.rho / a.mu
        cm = self.secs.clmax(re_loc, self.eta_strip)
        if config == "landing" and self.flaps is not None:
            cm = cm + np.where(self._flap_strips, self._flap_dclmax, 0.0)
        return cm

    def CLmax(self, V, config="clean"):
        """Critical-section wing CLmax: first strip to reach its clmax."""
        w = self._cfg(config)
        cm = self.strip_clmax(V, config)
        return np.min((cm - w.lb) / np.maximum(w.la, 1e-6), axis=-1)

    def stall_station(self, V, config="clean"):
        """Spanwise position (fraction of semispan) where the stall starts."""
        w = self._cfg(config)
        cm = self.strip_clmax(V, config)
        i = int(np.argmin((cm - w.lb) / np.maximum(w.la, 1e-6)))
        return float(self.eta_strip[i])

    def stall_speed(self, W, config="clean"):
        V = 10.0
        for _ in range(8):
            V = math.sqrt(2 * W / (self.atm.rho * self.S * float(self.CLmax(V, config))))
        return V

    def cm_ac(self, config="clean"):
        """Wing pitching moment about its aerodynamic center (section cm0
        weighted by c^2 over the span; flapped strips use the flapped section)."""
        g = self.g
        nu = self.atm.mu / self.atm.rho
        re = V_REF * g.chord / nu
        cm = self.secs.cm0(re, self.eta_strip)
        if config == "landing" and self.flaps is not None:
            fp = self.flaps["section"].props(V_REF * g.wing.mac / nu)
            cm = np.where(self._flap_strips, cm + fp["d_cm0"], cm)
        return float(2 * np.sum(cm * g.chord ** 2) * g.dy / (g.S * g.wing.mac))

    def span_load(self):
        """Spanwise load shape (c * cl, additional loading) for structures."""
        return self.clean.la * self.g.chord

    def polar_fit(self, V):
        """Parabolic fit CD = CD0 + K CL^2 at airspeed V (for the constraint
        diagram and reporting only - the sizing uses the full polar)."""
        clmax = float(self.CLmax(V))
        cls = np.linspace(0.15, 0.85 * clmax, 25)
        cds, _ = self.drag(cls, V)
        K, CD0 = np.polyfit(cls ** 2, cds, 1)
        return CD0, K
