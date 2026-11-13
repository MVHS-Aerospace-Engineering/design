"""Aerodynamic, stability, trim and performance analyses.

Most results come from AeroSandbox's AeroBuildup. It is a viscous buildup: the
airfoil polars come from NeuralFoil, and it adds fuselage, nacelle and
interference drag. The design neutral point comes from AeroSandbox's vortex
lattice method (VLM), the same inviscid method XFLR5 uses for planes, with a
fuselage correction (see layout.neutral_point_offset).
"""

from __future__ import annotations

from dataclasses import dataclass

import aerosandbox as asb
import numpy as np
from scipy.optimize import brentq, fsolve

from .config import PlaneConfig
from .geometry import build_airplane
from .layout import Layout, rudder_power_classical, solve_layout
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
    motor_current: float | None  # A, both motors, at trim thrust
    total_power: float | None  # W from the battery: motors + avionics
    endurance_min: float | None


class TwinPlane:
    def __init__(self, cfg: PlaneConfig | None = None, layout: Layout | None = None):
        self.cfg = cfg or PlaneConfig()
        self.layout = layout or solve_layout(self.cfg)
        self.mass = mass_properties(self.layout)
        self.weight = self.mass.mass * G
        self.xyz_cg = [self.mass.x_cg, self.mass.y_cg, self.mass.z_cg]
        self.atmo = asb.Atmosphere(altitude=self.cfg.altitude)
        self.rho = _f(self.atmo.density())
        self.prop = Propulsion(self.cfg.propulsion)
        self.clean = self.airplane()

    # ------------------------------------------------------------------ power
    @property
    def avionics_power(self) -> float:
        a = self.cfg.avionics
        return (a.pi5_power + a.fc_power + a.servo_receiver_power) / a.bec_efficiency

    @property
    def usable_energy_wh(self) -> float:
        p = self.cfg.propulsion
        return (self.cfg.design.battery_mah / 1000) * p.battery_cells * p.cell_voltage * p.usable_battery_fraction

    # ------------------------------------------------------------------ geometry
    def airplane(self, stab=0.0, stab_diff=0.0, rudder=0.0, aileron=0.0) -> asb.Airplane:
        """`stab` = collective stabilator deflection; `stab_diff` is added to the
        left half and subtracted from the right half (roll)."""
        return build_airplane(self.cfg, self.layout, stab_left_deg=stab + stab_diff,
                              stab_right_deg=stab - stab_diff, rudder_deg=rudder, aileron_deg=aileron,
                              xyz_ref=self.xyz_cg)

    def op(self, V, alpha=0.0, beta=0.0) -> asb.OperatingPoint:
        return asb.OperatingPoint(atmosphere=self.atmo, velocity=V, alpha=alpha, beta=beta)

    def aero(self, V, alpha=0.0, beta=0.0, **controls) -> dict:
        r = asb.AeroBuildup(self.airplane(**controls), self.op(V, alpha, beta), xyz_ref=self.xyz_cg).run()
        return {k: _f(v) for k, v in r.items() if np.size(v) == 1}

    # ------------------------------------------------------------- polars/alpha
    def alpha_sweep(self, V, alphas, stab=0.0) -> dict:
        r = asb.AeroBuildup(self.airplane(stab=stab), self.op(V, np.array(alphas)), xyz_ref=self.xyz_cg).run()
        return {k: np.ravel(r[k]) for k in ("CL", "CD", "Cm")} | {"alpha": np.array(alphas)}

    def trimmed_clmax(self, V=10.0) -> float:
        """Highest CL that can be trimmed (Cm = 0) within stabilator travel."""
        alphas = np.linspace(0, 16, 33)
        lim = self.cfg.stabilator.max_deflection_deg
        best = 0.0
        sweeps = {s: self.alpha_sweep(V, alphas, stab=s) for s in (-lim, -lim / 2, 0.0)}
        for i in range(len(alphas)):
            cms = np.array([sweeps[s]["Cm"][i] for s in sweeps])
            cls = np.array([sweeps[s]["CL"][i] for s in sweeps])
            if cms.min() <= 0 <= cms.max():  # trimmable at this alpha
                order = np.argsort(cms)
                best = max(best, float(np.interp(0.0, cms[order], cls[order])))
        return best

    def stability_derivatives(self, V, alpha) -> dict:
        r = asb.AeroBuildup(self.clean, self.op(V, alpha), xyz_ref=self.xyz_cg).run_with_stability_derivatives(
            alpha=True, beta=True, p=True, q=True, r=True)
        keys = ["CL", "CD", "Cm", "CLa", "Cma", "CYb", "Clb", "Cnb", "Clp", "Cnp", "Cmq", "Clr", "Cnr", "x_np"]
        return {k: _f(r[k]) for k in keys if k in r}

    @property
    def static_margin(self) -> float:
        """Design static margin: VLM neutral point + fuselage correction."""
        return (self.layout.x_np - self.xyz_cg[0]) / self.layout.mac

    def control_derivatives(self, V, alpha, h=2.0) -> dict:
        """Per-degree control effectiveness from central differences."""
        out = {}
        for name, coefs in (("stab", ("CL", "Cm")), ("stab_diff", ("Cl", "Cn")),
                            ("rudder", ("CY", "Cn", "Cl")), ("aileron", ("Cl", "Cn"))):
            hi, lo = self.aero(V, alpha, **{name: h}), self.aero(V, alpha, **{name: -h})
            for c in coefs:
                out[f"{c}_d{name}"] = (hi[c] - lo[c]) / (2 * h)
        return out

    # ---------------------------------------------------------------------- trim
    def trim(self, V, guess=(3.0, 0.0)) -> tuple[float, float, dict]:
        """Solve alpha and collective stabilator so that L = W and Cm = 0."""
        CL_req = self.weight / (0.5 * self.rho * V**2 * self.clean.s_ref)

        def resid(x):
            r = self.aero(V, x[0], stab=x[1])
            return [r["CL"] - CL_req, r["Cm"] * 10]

        sol, _, ier, _ = fsolve(resid, guess, full_output=True, xtol=1e-6)
        return sol[0], sol[1], self.aero(V, sol[0], stab=sol[1]) | {"converged": ier == 1}

    def level_flight_power(self, drag: float, V: float):
        """Total battery power (W) and motor current (A, both) to hold `drag` at V."""
        op = self.prop.throttle_for_thrust(drag / 2, V, self.rho)
        if op is None:
            return None, None
        return 2 * op.electrical_power + self.avionics_power, 2 * op.current

    def trim_sweep(self, speeds) -> list[TrimPoint]:
        pts, guess = [], (8.0, -2.0)
        for V in speeds:
            a, s, r = self.trim(V, guess)
            if not r["converged"] or a > 14:
                continue
            guess = (a, s)
            D = r["CD"] * 0.5 * self.rho * V**2 * self.clean.s_ref
            P, I = self.level_flight_power(D, V)
            pts.append(TrimPoint(
                V, a, s, r["CL"], r["CD"], r["CL"] / r["CD"], D,
                2 * self.prop.operating_point(V, self.rho, 1.0).thrust, I, P,
                self.usable_energy_wh / P * 60 if P else None,
            ))
        return pts

    def stall_speed(self, CL_max) -> float:
        return float(np.sqrt(2 * self.weight / (self.rho * self.clean.s_ref * CL_max)))

    # -------------------------------------------------------- engine-out (Vmc)
    def rudder_power(self, method: str = "classical") -> float:
        """Cn per degree of rudder. "classical" (design value) is the handbook
        low-AR fin estimate; "buildup" is AeroBuildup, about 3x higher here and
        treated as optimistic."""
        if method == "classical":
            return rudder_power_classical(self.cfg, self.layout.Vv) / np.degrees(1)
        if not hasattr(self, "_cn_dr_buildup"):
            V, h = 12.0, 2.0
            a, _, _ = self.trim(V)
            self._cn_dr_buildup = (self.aero(V, a, rudder=h)["Cn"] - self.aero(V, a, rudder=-h)["Cn"]) / (2 * h)
        return self._cn_dr_buildup

    def engine_out_rudder(self, V, throttle=1.0, method: str = "classical") -> dict:
        """Rudder needed to hold zero sideslip, wings level, with the left motor
        dead and the right motor at `throttle`. The dead prop's drag adds to the yaw."""
        T = self.prop.operating_point(V, self.rho, throttle).thrust
        q = 0.5 * self.rho * V**2
        D_dead = 0.1 * q * np.pi * (self.prop.D / 2) ** 2  # stopped 2-blade prop, rough
        Cn_asym = -(T + D_dead) * self.layout.motor_y / (q * self.clean.s_ref * self.clean.b_ref)
        Cn_dr = self.rudder_power(method)
        return {"V": V, "thrust": T, "Cn_asym": Cn_asym, "Cn_dr": Cn_dr, "rudder_req": -Cn_asym / Cn_dr,
                "throttle": throttle}

    def vmc(self, v_lo, v_hi, method: str = "classical") -> float | None:
        """Full-power minimum control speed: below it, full rudder can't hold the dead-engine yaw."""
        max_r = self.cfg.fin.max_rudder_deg

        def f(V):
            return abs(self.engine_out_rudder(V, method=method)["rudder_req"]) - max_r

        if f(v_hi) > 0:
            return None
        if f(v_lo) < 0:
            return v_lo
        return brentq(f, v_lo, v_hi, xtol=0.05)

    def oei_throttle_limit(self, V) -> float:
        """Highest throttle on the live motor that full rudder can still hold at V.
        The flight controller enforces this automatically after a motor failure."""
        max_r = self.cfg.fin.max_rudder_deg
        if abs(self.engine_out_rudder(V, 1.0)["rudder_req"]) <= max_r:
            return 1.0
        return brentq(lambda t: abs(self.engine_out_rudder(V, t)["rudder_req"]) - max_r, 0.05, 1.0, xtol=0.01)
