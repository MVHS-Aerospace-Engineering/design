"""Motor + 2-blade propeller model for each of the two nacelles.

Propeller: thrust and power coefficients as quadratics in advance ratio
J = V / (n D), anchored to static values (`ct_static`, `cp_static`) and the J
at which thrust reaches zero. The defaults are for a 10x4.7 slow-fly prop on the
FT Radial 2218 and are calibrated against published 2218 bench data (see
config.PropulsionConfig). Swap in thrust-stand numbers when you have them.

Motor: first-order DC model (Kv, winding resistance, no-load current). Each
operating point solves for the RPM where motor torque equals prop torque.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .config import PropulsionConfig

INCH = 0.0254


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
        blade_scale = cfg.n_blades / 2.0  # coefficients are for 2 blades
        self.ct0 = cfg.ct_static * blade_scale
        self.cp0 = cfg.cp_static * blade_scale
        self.j0 = cfg.j_zero_thrust
        self.v_batt = cfg.battery_cells * cfg.cell_voltage
        self.kt = 60.0 / (2 * np.pi * cfg.motor_kv)  # N*m / A

    def ct(self, J):
        j = J / self.j0
        return self.ct0 * (1 - 0.3 * j - 0.7 * j**2)

    def cp(self, J):
        j = J / self.j0
        return max(self.cp0 * (1 + 0.1 * j - 0.75 * j**2), 1e-4)

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
