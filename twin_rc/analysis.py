"""Aerodynamic, stability, trim and performance analyses.

Most results come from AeroSandbox's AeroBuildup. It is a viscous buildup: the
airfoil polars come from NeuralFoil, and it adds fuselage, nacelle and
interference drag. AeroSandbox's vortex lattice method (VLM) then
cross-checks the lift slope and pitch stability. That is the same inviscid
method XFLR5 uses for its plane analyses.
"""

from __future__ import annotations

from dataclasses import dataclass

import aerosandbox as asb
import numpy as np
from scipy.optimize import brentq, fsolve

from .config import PlaneConfig
from .geometry import build_airplane
from .mass import mass_properties
from .propulsion import Propulsion

G = 9.81


def _f(x) -> float:
    return float(np.ravel(x)[0])


@dataclass
class TrimPoint:
    V: float
    alpha: float
    stab_deg: float
    CL: float
    CD: float
    L_over_D: float
    drag: float
    thrust_avail: float  # N, both motors at full throttle
    throttle_current: float | None  # A, total, at trim thrust
    electrical_power: float | None  # W, total


class TwinPlane:
    def __init__(self, cfg: PlaneConfig | None = None):
        self.cfg = cfg or PlaneConfig()
        self.mass = mass_properties(self.cfg)
        self.weight = self.mass.mass * G
        self.xyz_cg = [self.mass.x_cg, self.mass.y_cg, self.mass.z_cg]
        self.atmo = asb.Atmosphere(altitude=self.cfg.altitude)
        self.rho = _f(self.atmo.density())
        self.prop = Propulsion(self.cfg.propulsion)
        self.clean = self.airplane()

    # ------------------------------------------------------------------ geometry
    def airplane(self, stab=0.0, stab_diff=0.0, rudder=0.0, aileron=0.0) -> asb.Airplane:
        """`stab` = collective stabilator deflection; `stab_diff` is added to the
        left half and subtracted from the right half (roll)."""
        return build_airplane(
            self.cfg,
            stab_left_deg=stab + stab_diff,
            stab_right_deg=stab - stab_diff,
            rudder_deg=rudder,
            aileron_deg=aileron,
            xyz_ref=self.xyz_cg,
        )

    def op(self, V, alpha=0.0, beta=0.0) -> asb.OperatingPoint:
        return asb.OperatingPoint(atmosphere=self.atmo, velocity=V, alpha=alpha, beta=beta)

    def aero(self, V, alpha=0.0, beta=0.0, **controls) -> dict:
        r = asb.AeroBuildup(self.airplane(**controls), self.op(V, alpha, beta), xyz_ref=self.xyz_cg).run()
        return {k: _f(v) for k, v in r.items() if np.size(v) == 1 or k in ("CL", "CD", "Cm")}

    # ------------------------------------------------------------- polars/alpha
    def alpha_sweep(self, V, alphas, stab=0.0) -> dict:
        ap = self.airplane(stab=stab)
        r = asb.AeroBuildup(ap, self.op(V, np.array(alphas)), xyz_ref=self.xyz_cg).run()
        return {k: np.ravel(r[k]) for k in ("CL", "CD", "Cm")} | {"alpha": np.array(alphas)}

    def stability_derivatives(self, V, alpha) -> dict:
        r = asb.AeroBuildup(self.clean, self.op(V, alpha), xyz_ref=self.xyz_cg).run_with_stability_derivatives(
            alpha=True, beta=True, p=True, q=True, r=True
        )
        keys = [
            "CL", "CD", "Cm", "CLa", "Cma", "CYb", "Clb", "Cnb", "Clp", "Cnp", "Cmq", "Clr", "Cnr",
            "x_np",
        ]
        return {k: _f(r[k]) for k in keys if k in r}

    def neutral_point(self, V, alpha) -> float:
        d = self.stability_derivatives(V, alpha)
        if "x_np" in d:
            return d["x_np"]
        return self.xyz_cg[0] - d["Cma"] / d["CLa"] * self.clean.c_ref

    def static_margin(self, V, alpha) -> float:
        return (self.neutral_point(V, alpha) - self.xyz_cg[0]) / self.clean.c_ref

    def control_derivatives(self, V, alpha, h=2.0) -> dict:
        """Per-degree control effectiveness from central differences."""
        out = {}
        for name, kw, coefs in (
            ("stab", "stab", ("CL", "Cm")),
            ("stab_diff", "stab_diff", ("Cl", "Cn")),
            ("rudder", "rudder", ("CY", "Cn", "Cl")),
            ("aileron", "aileron", ("Cl", "Cn")),
        ):
            hi = self.aero(V, alpha, **{kw: h})
            lo = self.aero(V, alpha, **{kw: -h})
            for c in coefs:
                out[f"{c}_d{name}"] = (hi[c] - lo[c]) / (2 * h)
        return out

    # ---------------------------------------------------------------------- trim
    def trim(self, V, guess=(3.0, 0.0)) -> tuple[float, float, dict]:
        """Solve alpha and collective stabilator so that L = W and Cm = 0."""
        q = 0.5 * self.rho * V**2
        S = self.clean.s_ref
        CL_req = self.weight / (q * S)

        def resid(x):
            a, s = x
            r = self.aero(V, a, stab=s)
            return [r["CL"] - CL_req, r["Cm"] * 10]

        sol, info, ier, _ = fsolve(resid, guess, full_output=True, xtol=1e-6)
        a, s = sol
        return a, s, self.aero(V, a, stab=s) | {"converged": ier == 1}

    def trim_sweep(self, speeds) -> list[TrimPoint]:
        pts = []
        guess = (8.0, -2.0)
        for V in speeds:
            a, s, r = self.trim(V, guess)
            if not r["converged"] or a > 14:
                continue
            guess = (a, s)
            q = 0.5 * self.rho * V**2
            D = r["CD"] * q * self.clean.s_ref
            T_avail = 2 * self.prop.operating_point(V, self.rho, 1.0).thrust
            op = self.prop.throttle_for_thrust(D / 2, V, self.rho)
            pts.append(
                TrimPoint(
                    V, a, s, r["CL"], r["CD"], r["CL"] / r["CD"], D, T_avail,
                    2 * op.current if op else None,
                    2 * op.electrical_power if op else None,
                )
            )
        return pts

    def stall_speed(self, CL_max) -> float:
        return float(np.sqrt(2 * self.weight / (self.rho * self.clean.s_ref * CL_max)))

    # -------------------------------------------------------- engine-out (Vmc)
    def engine_out_rudder(self, V, alpha=None) -> dict:
        """Rudder needed to hold zero sideslip with the left motor dead and the
        right motor at full power. The asymmetric thrust yaws the nose left and
        the dead prop's drag adds to it. The rudder alone must cancel this
        (wings-level, beta = 0)."""
        p = self.cfg.propulsion
        if alpha is None:
            alpha, _, _ = self.trim(V)
        T = self.prop.operating_point(V, self.rho, 1.0).thrust
        # A stopped/windmilling 2-blade prop: drag ~ 0.1 * q * prop disk area (rough).
        q = 0.5 * self.rho * V**2
        D_dead = 0.1 * q * np.pi * (self.prop.D / 2) ** 2
        N_asym = -(T + D_dead) * p.motor_y  # yawing moment, nose-left negative (body z down)
        Cn_asym = N_asym / (q * self.clean.s_ref * self.clean.b_ref)
        h = 2.0
        Cn_dr = (self.aero(V, alpha, rudder=h)["Cn"] - self.aero(V, alpha, rudder=-h)["Cn"]) / (2 * h)
        rudder_req = -Cn_asym / Cn_dr
        return {"V": V, "thrust": T, "Cn_asym": Cn_asym, "Cn_dr": Cn_dr, "rudder_req": rudder_req}

    def vmc(self, v_lo, v_hi) -> float | None:
        """Minimum control speed: below it full rudder cannot hold the dead-engine yaw."""
        max_r = self.cfg.fin.max_rudder_deg

        def f(V):
            return abs(self.engine_out_rudder(V)["rudder_req"]) - max_r

        if f(v_hi) > 0:
            return None
        if f(v_lo) < 0:
            return v_lo
        return brentq(f, v_lo, v_hi, xtol=0.05)

    # ---------------------------------------------------------- VLM cross-check
    def vlm(self, V, alpha) -> dict:
        ap = self.clean
        ap_wings = asb.Airplane(
            name=ap.name, xyz_ref=self.xyz_cg, wings=ap.wings,
            s_ref=ap.s_ref, c_ref=ap.c_ref, b_ref=ap.b_ref,
        )

        def run(a):
            r = asb.VortexLatticeMethod(
                ap_wings, self.op(V, a), xyz_ref=self.xyz_cg,
                spanwise_resolution=8, chordwise_resolution=8,
            ).run()
            return _f(r["CL"]), _f(r["Cm"])

        cl0, cm0 = run(alpha - 1)
        cl1, cm1 = run(alpha + 1)
        CLa, Cma = (cl1 - cl0) / 2, (cm1 - cm0) / 2
        x_np = self.xyz_cg[0] - Cma / CLa * ap.c_ref
        return {"CL": (cl0 + cl1) / 2, "CLa_per_deg": CLa, "Cma_per_deg": Cma, "x_np": x_np}
