"""Motor + 2-blade propeller model for each of the two nacelles.

The propeller uses quadratic fits of thrust and power coefficient against advance
ratio J = V / (n D). They come from published wind-tunnel data for 2-blade
10x6 electric props (UIUC propeller database, APC 10x6E). Other diameters and
pitches are scaled by the pitch/diameter ratio. That is good enough for sizing,
but swap in measured data for the prop you actually buy.

The motor is a standard first-order brushed/brushless DC model
(Kv, winding resistance, no-load current). Each operating point solves for the
RPM where motor torque matches propeller torque.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .config import PropulsionConfig

INCH = 0.0254

# Reference prop: 2-blade, pitch/diameter = 0.6
_CT = (0.105, -0.030, -0.147)  # Ct = a + b J + c J^2, zero thrust at J ~ 0.75
_CP = (0.045, 0.005, -0.045)
_PD_REF = 0.6


@dataclass
class PropPoint:
    rpm: float
    thrust: float  # N, per motor
    current: float  # A, per motor
    shaft_power: float  # W
    electrical_power: float  # W
    prop_efficiency: float
    J: float


class Propulsion:
    def __init__(self, cfg: PropulsionConfig):
        self.cfg = cfg
        self.D = cfg.prop_diameter_in * INCH
        pd = cfg.prop_pitch_in / cfg.prop_diameter_in
        blade_scale = cfg.n_blades / 2.0
        # Scale the zero-thrust advance ratio with pitch/diameter and the
        # coefficient magnitudes with blade count.
        self.j_scale = pd / _PD_REF
        self.ct_scale = blade_scale * (pd / _PD_REF) ** 0.5
        self.cp_scale = blade_scale * (pd / _PD_REF) ** 1.2
        self.v_batt = cfg.battery_cells * cfg.cell_voltage
        self.kt = 60.0 / (2 * np.pi * cfg.motor_kv)  # N*m / A

    def ct(self, J):
        j = J / self.j_scale
        return self.ct_scale * (_CT[0] + _CT[1] * j + _CT[2] * j**2)

    def cp(self, J):
        j = J / self.j_scale
        return max(self.cp_scale * (_CP[0] + _CP[1] * j + _CP[2] * j**2), 1e-4)

    def operating_point(self, V: float, rho: float, throttle: float = 1.0) -> PropPoint:
        c = self.cfg
        v_motor = throttle * self.v_batt
        r = c.motor_resistance + c.esc_resistance

        def torque_balance(rpm):
            n = rpm / 60.0
            J = V / (n * self.D)
            q_prop = self.cp(J) * rho * n**2 * self.D**5 / (2 * np.pi)
            current = (v_motor - rpm / c.motor_kv) / r
            q_motor = self.kt * (current - c.motor_no_load_current)
            return q_motor - q_prop

        rpm_max = v_motor * c.motor_kv * 0.999
        rpm = brentq(torque_balance, 200.0, rpm_max)
        n = rpm / 60.0
        J = V / (n * self.D)
        thrust = self.ct(J) * rho * n**2 * self.D**4
        current = (v_motor - rpm / c.motor_kv) / r
        p_shaft = self.cp(J) * rho * n**3 * self.D**5
        eta = thrust * V / p_shaft if p_shaft > 0 else 0.0
        return PropPoint(rpm, thrust, current, p_shaft, current * self.v_batt, eta, J)

    def throttle_for_thrust(self, T_req: float, V: float, rho: float) -> PropPoint | None:
        """Throttle that gives `T_req` newtons from ONE motor, or None if it can't."""
        if self.operating_point(V, rho, 1.0).thrust < T_req:
            return None
        th = brentq(lambda t: self.operating_point(V, rho, t).thrust - T_req, 0.05, 1.0)
        return self.operating_point(V, rho, th)
