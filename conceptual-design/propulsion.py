"""
Propulsion (README critical input #1).

1. Battery-mass-fraction equations from "Electric Propulsion Fundamentals
   (Metric)" - unchanged from the first version of conceptual.py.
2. A physical powertrain model that REPLACES the old assumed efficiencies:
     battery (V_oc, R_int)  ->  ESC (eta_esc)  ->  motor (Kv, Rm, Io)
     ->  prop (APC CT/CP vs J and RPM)
   It returns, at every airspeed:
     - power available at full throttle        (README: available power vs airspeed)
     - battery power / current for a required thrust (README: energy draw vs time)
     - eta_b2s and eta_prop actually achieved at that operating point
"""
import math
from functools import lru_cache

import numpy as np

from atmosphere import G0
from parameters import P

G = G0


# =====================================================================
# BMF equations (image) - t in hours, V in km/h, power in kW, U in Wh/kg
# =====================================================================
def bmf_specified_endurance(t_hr, p_used_kw, mass_kg, u_batt, eta_b2s, k):
    """BMF = 1000 t / (U_batt eta_b2s K) * (P_used / m)"""
    return 1000.0 * t_hr / (u_batt * eta_b2s * k) * (p_used_kw / mass_kg)


def bmf_loiter(t_hr, v_kmh, ld, u_batt, eta_total, k):
    """Level Flight Loiter Endurance: BMF = t g V / (3.6 eta_total (L/D) U_batt K)"""
    return t_hr * G * v_kmh / (3.6 * eta_total * ld * u_batt * k)


def bmf_cruise_range(r_km, ld, u_batt, eta_total, k):
    """Level Flight Cruise Range: BMF = R g / (3.6 eta_total (L/D) U_batt K)"""
    return r_km * G / (3.6 * eta_total * ld * u_batt * k)


def bmf_climb(h_m, v_ms, v_v_ms, ld_climb, u_batt, eta_total, k):
    """Climb Time: BMF = h g / (3600 eta_total U_batt K) * (1 + V / (V_v L/D))"""
    return h_m * G / (3600.0 * eta_total * u_batt * k) * (1.0 + v_ms / (v_v_ms * ld_climb))


def climb_rate(p_avail_per_weight, v_ms, ld):
    """V_v = P_avail/W - V/(L/D)   [m/s], P_avail/W in W/N."""
    return p_avail_per_weight - v_ms / ld


def climb_power_required(v_v_ms, v_ms, ld):
    """P_avail/W [W/N] needed for a given V_v (inverse of climb_rate)."""
    return v_v_ms + v_ms / ld


def battery_mass(bmf, takeoff_mass_kg):
    """W_battery = BMF * W_TO"""
    return bmf * takeoff_mass_kg


# =====================================================================
# Prop tables on a fixed (airspeed x rev/s) grid - cached per prop and density
# =====================================================================
V_GRID = np.linspace(0.0, 40.0, 161)      # m/s, 0.25 m/s steps
N_PTS = 220


@lru_cache(maxsize=256)
def _prop_table(prop_name, rho_key):
    from catalog import props
    prop = props()[prop_name]
    rho = rho_key / 1e6
    n = np.linspace(2.0, prop.rpm_max / 60.0, N_PTS)          # rev/s
    Vg, ng = np.meshgrid(V_GRID, n, indexing="ij")
    ct, cp = prop.coeffs(60.0 * ng, Vg / (ng * prop.D))
    T = ct * rho * ng ** 2 * prop.D ** 4
    Ps = cp * rho * ng ** 3 * prop.D ** 5
    T = np.maximum.accumulate(T, axis=1)                      # enforce monotone in n
    return n, T, Ps


def _row_interp(x_tab, y_tab, x_req):
    """For each row r, interpolate y at x_req[r] on monotone x_tab[r, :].
    Returns (y, ok) where ok is False when x_req is outside the row range."""
    k = x_tab.shape[1]
    idx = np.clip((x_tab < x_req[:, None]).sum(axis=1), 1, k - 1)
    r = np.arange(x_tab.shape[0])
    x0, x1 = x_tab[r, idx - 1], x_tab[r, idx]
    y0, y1 = y_tab[r, idx - 1], y_tab[r, idx]
    w = np.clip((x_req - x0) / np.where(x1 > x0, x1 - x0, 1.0), 0.0, 1.0)
    ok = (x_req <= x_tab[:, -1]) & (x_req >= x_tab[:, 0])
    return y0 + w * (y1 - y0), ok


def _battery_current(v_oc, r, p_term):
    """Terminal power P = (V_oc - I R) I  ->  I. NaN if beyond max power."""
    disc = v_oc ** 2 - 4 * r * p_term
    return np.where(disc >= 0, (v_oc - np.sqrt(np.maximum(disc, 0))) / (2 * r), np.nan)


class Powertrain:
    """n_motors identical motor+prop units on one battery, plus avionics load."""

    def __init__(self, motor, prop, battery, rho, p_avionics_w, throttle_cap=1.0):
        self.motor, self.prop, self.batt = motor, prop, battery
        self.rho = rho
        self.cap = throttle_cap         # max ESC duty, set in the flight controller (all modes)
        self.n_m = P.n_motors
        self.p_av_term = p_avionics_w / P.eta_bec           # avionics draw at battery terminals
        self.n, self.T_tab, self.Ps_tab = _prop_table(prop.name, int(round(rho * 1e6)))
        self._ft = None

    # ------------------------------------------------------------------
    def _motor_state(self, n, Ps):
        m = self.motor
        omega = 2 * math.pi * n
        Q = Ps / np.maximum(omega, 1e-9)
        I = Q / m.kt + m.io
        Vm = I * m.rm + omega / m.kv_rad
        return I, Vm

    def required(self, T_total):
        """Operating point on V_GRID delivering total thrust T_total(V_GRID)."""
        T_m = np.asarray(T_total, float) / self.n_m
        n, ok = _row_interp(self.T_tab, np.broadcast_to(self.n, self.T_tab.shape), T_m)
        Ps, _ = _row_interp(self.T_tab, self.Ps_tab, T_m)
        I, Vm = self._motor_state(n, Ps)
        p_term = self.n_m * Vm * I / P.eta_esc + self.p_av_term
        Ib = _battery_current(self.batt.v_nom, self.batt.r_int, p_term)
        v_term = self.batt.v_nom - Ib * self.batt.r_int
        duty = Vm / (P.eta_esc * v_term)
        ok = ok & np.isfinite(Ib) & (duty <= self.cap) & (Ps > 0)
        p_chem = self.batt.v_nom * Ib
        p_chem_av = self.p_av_term * self.batt.v_nom / v_term
        p_chem_prop = p_chem - p_chem_av
        with np.errstate(divide="ignore", invalid="ignore"):
            eta_prop = np.where(Ps > 0, T_m * V_GRID / Ps, 0.0)
            eta_b2s = self.n_m * Ps / p_chem_prop
        return dict(ok=ok, rpm=60 * n, I_motor=I, V_motor=Vm, I_batt=Ib, P_shaft=self.n_m * Ps,
                    P_chem=p_chem, P_chem_prop=p_chem_prop, P_chem_av=p_chem_av,
                    eta_prop=eta_prop, eta_b2s=eta_b2s, duty=duty)

    def full_throttle(self):
        """Full-throttle (= throttle cap) thrust/current/power vs V_GRID, battery sag included."""
        if self._ft is not None:
            return self._ft
        m, b = self.motor, self.batt
        omega = 2 * math.pi * self.n
        q_prop = self.Ps_tab / omega
        v_term = np.full(V_GRID.shape, b.v_nom)
        for _ in range(6):
            vm = self.cap * P.eta_esc * v_term
            q_mot = m.kt * ((vm[:, None] - omega[None, :] / m.kv_rad) / m.rm - m.io)
            g = q_mot - q_prop                                  # decreasing in n
            # first sign change -> root
            neg = g < 0
            idx = np.clip(np.argmax(neg, axis=1), 1, len(self.n) - 1)
            r = np.arange(len(V_GRID))
            g0, g1 = g[r, idx - 1], g[r, idx]
            w = np.clip(g0 / np.where(g0 != g1, g0 - g1, 1.0), 0, 1)
            n_op = self.n[idx - 1] + w * (self.n[idx] - self.n[idx - 1])
            T = self.T_tab[r, idx - 1] + w * (self.T_tab[r, idx] - self.T_tab[r, idx - 1])
            Ps = self.Ps_tab[r, idx - 1] + w * (self.Ps_tab[r, idx] - self.Ps_tab[r, idx - 1])
            I = (vm - 2 * math.pi * n_op / m.kv_rad) / m.rm
            p_term = self.n_m * vm * I / P.eta_esc + self.p_av_term
            Ib = p_term / v_term
            v_new = b.v_nom - Ib * b.r_int
            v_term = 0.5 * v_term + 0.5 * np.maximum(v_new, 0.3 * b.v_nom)
        rpm_limited = ~neg.any(axis=1)          # prop data ran out before torque balance
        p_chem = b.v_nom * Ib
        self._ft = dict(T=self.n_m * T, P_shaft=self.n_m * Ps, I_motor=I, I_batt=Ib,
                        P_chem=p_chem, v_term=v_term, rpm=60 * n_op, rpm_limited=rpm_limited,
                        P_avail=self.n_m * T * V_GRID)
        return self._ft
