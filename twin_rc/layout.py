"""Derives the full geometry and mass layout from `DesignVars` + hardware.

What's derived:
* Fuselage cross-section: the smallest boxy (superellipse) section that holds
  the Pi 5, flight controller and battery with clearance and foam-board walls.
* Avionics bay: battery, flight controller and Pi 5 packed nose to tail, in the
  order that puts the flight controller (its IMU) nearest the CG; then the BEC
  and receiver.
* Wing position: chosen so that CG = neutral point - static_margin x MAC.
* Stabilator size: from the horizontal tail volume coefficient.
* Fin and rudder size: the larger of the vertical tail volume floor and what's
  needed to hold one engine out at `vmc_over_vs` x stall.
* Motor spacing: as close to the fuselage as prop tip clearance allows. That
  minimises the engine-out yawing moment.

`solve_layout(cfg)` computes the neutral point with AeroSandbox's VLM and
returns the converged layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from .config import BATTERIES, Box, PlaneConfig

INCH = 0.0254
G = 9.81


@dataclass
class Placed:
    name: str
    box: Box
    center: tuple  # x, y, z


@dataclass
class Layout:
    # wing
    span: float
    root_chord: float
    tip_chord: float
    y_break: float
    wing_x_le: float
    wing_z: float
    wing_area: float
    mac: float
    x_mac_le: float  # LE of the MAC
    # stabilators (each half)
    stab_semispan: float
    stab_root_chord: float
    stab_tip_chord: float
    stab_x_le: float
    stab_z: float
    stab_y_root: float
    stab_area: float  # both halves
    # fin
    fin_height: float
    fin_root_chord: float
    fin_tip_chord: float
    fin_x_le: float
    fin_z_root: float
    fin_area: float
    # fuselage
    fus_length: float
    fus_width: float
    fus_height: float
    fus_stations: list  # (x, width, height, z_centre)
    # propulsion
    motor_y: float
    prop_x: float
    motor_z: float
    nacelle_radius: float
    nacelle_length: float
    # payload + mass
    components: list = field(default_factory=list)  # Placed, inside the fuselage
    mass_items: dict = field(default_factory=dict)  # name -> (mass, x, y, z)
    tail_arm: float = 0.0
    Vh: float = 0.0
    Vv: float = 0.0
    np_offset_mac: float = 0.0  # (x_np - x_mac_le) / mac, used for placement

    @property
    def mass(self) -> float:
        return sum(m for m, *_ in self.mass_items.values())

    @property
    def cg(self) -> np.ndarray:
        m = np.array([v[0] for v in self.mass_items.values()])
        p = np.array([v[1:] for v in self.mass_items.values()])
        return (m[:, None] * p).sum(axis=0) / m.sum()

    @property
    def x_np(self) -> float:
        return self.x_mac_le + self.np_offset_mac * self.mac

    def fc_offset_from_cg(self) -> float:
        fc = next(c for c in self.components if c.name == "flight_controller")
        return fc.center[0] - self.cg[0]


def _wing_planform(span, cr, ct, taper_start):
    semi = span / 2
    yb = semi * taper_start
    area = 2 * (cr * yb + 0.5 * (cr + ct) * (semi - yb))
    # MAC of the two panels (area-weighted)
    a1, a2 = cr * yb, 0.5 * (cr + ct) * (semi - yb)
    lam = ct / cr
    mac2 = (2 / 3) * cr * (1 + lam + lam**2) / (1 + lam)
    y2 = yb + (semi - yb) * (1 + 2 * lam) / (3 * (1 + lam))
    mac = (a1 * cr + a2 * mac2) / (a1 + a2)
    # quarter-chord line is straight, so LE x at span station y is 0.25 (cr - c(y))
    c_at_y2 = cr + (ct - cr) * (y2 - yb) / (semi - yb)
    x_le_mac2 = 0.25 * (cr - c_at_y2)
    x_mac_le = (a2 * x_le_mac2) / (a1 + a2)  # panel 1 LE is at 0
    return area, mac, x_mac_le, yb


RUDDER_TAU = 0.62  # flap effectiveness of a 40 % chord rudder
FIN_EFFICIENCY = 0.9  # dynamic pressure ratio at the fin


def fin_lift_slope(cfg: PlaneConfig) -> float:
    """Helmbold lift slope (per rad) of the fin, end-plated by fuselage and stabilators."""
    A_eff = 1.6 * cfg.fin.aspect_ratio
    return 2 * np.pi * A_eff / (2 + np.sqrt(A_eff**2 + 4))


def rudder_power_classical(cfg: PlaneConfig, Vv: float) -> float:
    """Cn per radian of rudder (handbook estimate; more conservative than AeroBuildup for low-AR fins)."""
    return Vv * fin_lift_slope(cfg) * RUDDER_TAU * FIN_EFFICIENCY


def _superellipse_scale(n):
    """Factor by which a superellipse's half-axes must exceed a centred box's half-sizes to contain it."""
    return 2 ** (1 / n)


# Forward stacks, packed from the nose. "fc@cg" puts the flight controller on its
# own tray at the CG (or right behind the stack if the CG is further forward).
PAYLOAD_ORDERS = [
    ("battery", "pi5", "fc@cg"),
    ("pi5", "battery", "fc@cg"),
    ("battery", "flight_controller", "pi5"),
    ("battery", "pi5", "flight_controller"),
]


def build_layout(cfg: PlaneConfig, np_offset_mac: float, mass_guess: float | None = None,
                 clmax_est: float = 1.30) -> Layout:
    d, w, s, f, p, a, fz = cfg.design, cfg.wing, cfg.stabilator, cfg.fin, cfg.propulsion, cfg.avionics, cfg.fuselage
    batt = BATTERIES[d.battery_mah]

    # ---- fuselage cross-section from the payload
    k = _superellipse_scale(fz.section_exponent)
    payload = [batt, a.pi5, a.flight_controller]
    half_w = max(b.width for b in payload) / 2 + fz.clearance
    stack_h = max(b.height for b in payload) + 2 * fz.clearance + fz.floor_gap
    inner_w, inner_h = 2 * k * half_w, 2 * k * stack_h / 2
    W, H = inner_w + 2 * fz.wall, inner_h + 2 * fz.wall
    z_floor = -stack_h / 2 + fz.floor_gap + fz.clearance  # tray height; stack centred on z = 0

    # ---- wing planform
    cr, ct = d.root_chord, d.root_chord * d.taper
    area, mac, x_mac_le_rel, yb = _wing_planform(d.span, cr, ct, w.taper_start_frac)
    wing_z = H / 2 + 0.004  # high wing on the fuselage top (saddle)

    # ---- motors
    R = p.prop_diameter_in * INCH / 2
    motor_y = W / 2 + R + p.prop_tip_clearance
    motor_z = wing_z - 0.024  # thrust line under the wing, motor hung on a nacelle
    nacelle_r = p.motor_diameter / 2 + 0.006

    # ---- tail sizing
    l_t = d.tail_arm_over_mac * mac
    S_h = d.Vh * area * mac / l_t
    b_h = np.sqrt(s.aspect_ratio * S_h)
    cr_h = 2 * S_h / (b_h * (1 + s.taper))
    ct_h = cr_h * s.taper
    semi_h = b_h / 2
    mac_h = (2 / 3) * cr_h * (1 + s.taper + s.taper**2) / (1 + s.taper)
    y_mac_h = semi_h * (1 + 2 * s.taper) / (3 * (1 + s.taper))
    ac_h_from_le = np.tan(np.radians(s.sweep_le_deg)) * y_mac_h + 0.25 * mac_h

    # Fin: volume floor, or engine-out need at vmc_over_vs x estimated stall.
    m_guess = mass_guess if mass_guess is not None else 1.2 + batt.mass
    rho = 1.219
    v_s = np.sqrt(2 * m_guess * G / (rho * area * clmax_est))
    V = d.vmc_over_vs * v_s
    from .propulsion import Propulsion  # local import, avoids a cycle at module load

    T = Propulsion(p).operating_point(V, rho, 1.0).thrust
    q = 0.5 * rho * V**2
    D_dead = 0.1 * q * np.pi * R**2
    Cn_needed = (T + D_dead) * motor_y / (q * area * d.span)
    Vv_oei = Cn_needed / (rudder_power_classical(cfg, 1.0) * np.radians(f.max_rudder_deg))
    Vv = max(d.Vv_min, Vv_oei)
    S_v = Vv * area * d.span / l_t
    h_v = np.sqrt(f.aspect_ratio * S_v)
    cr_v = 2 * S_v / (h_v * (1 + f.taper))
    ct_v = cr_v * f.taper
    mac_v = (2 / 3) * cr_v * (1 + f.taper + f.taper**2) / (1 + f.taper)
    z_mac_v = h_v * (1 + 2 * f.taper) / (3 * (1 + f.taper))
    ac_v_from_le = np.tan(np.radians(f.sweep_le_deg)) * z_mac_v + 0.25 * mac_v

    def assemble(x_le: float, order: tuple, fc_target: float = 0.0) -> Layout:
        x_ac_w = x_le + x_mac_le_rel + 0.25 * mac
        stab_x_le = x_ac_w + l_t - ac_h_from_le
        fin_x_le = x_ac_w + l_t - ac_v_from_le
        tail_end = max(stab_x_le + cr_h, fin_x_le + cr_v)
        L = tail_end + 0.01

        # payload along x: contiguous, in the given order, starting behind the nose
        comps = []
        x = fz.nose_length
        boxes = {"battery": batt, "flight_controller": a.flight_controller, "pi5": a.pi5}
        for name in order:
            if name == "fc@cg":
                name = "flight_controller"
                x = max(x, fc_target - a.flight_controller.length / 2)
            b = boxes[name]
            comps.append(Placed(name, b, (x + b.length / 2, 0.0, z_floor + b.height / 2)))
            x += b.length + fz.gap
        comps.append(Placed("bec", a.bec, (x + a.bec.length / 2, -0.012, z_floor + a.bec.height / 2)))
        comps.append(Placed("receiver", a.receiver,
                            (x + a.receiver.length / 2, 0.014, z_floor + a.receiver.height / 2)))
        x += max(a.bec.length, a.receiver.length)
        bay_end = x + fz.clearance

        full_end = max(bay_end, x_le + cr + 0.02)
        boom_start = stab_x_le - 0.02
        stations = [
            (0.0, 0.40 * W, 0.45 * H, 0.0),
            (fz.nose_length * 0.5, 0.80 * W, 0.85 * H, 0.0),
            (fz.nose_length, W, H, 0.0),
            (full_end, W, H, 0.0),
            (boom_start, fz.tail_boom_width, fz.tail_boom_height, 0.0),
            (L, fz.tail_boom_width * 0.8, fz.tail_boom_height * 0.8, 0.0),
        ]
        # fuselage skin area (perimeter of a rounded box ~ 2(w+h) * 0.95)
        skin, moment = 0.0, 0.0
        for (x0, w0, h0, _), (x1, w1, h1, _) in zip(stations[:-1], stations[1:]):
            dA = 0.95 * (w0 + h0 + w1 + h1) * (x1 - x0)
            skin += dA
            moment += dA * (x0 + x1) / 2
        fus_mass = skin * fz.areal_mass + fz.structure_extra
        fus_cx = moment / skin

        prop_x = x_le - (p.motor_length + 0.035)
        nac_len = (x_le + 0.55 * cr) - (prop_x + 0.008)

        items = {
            "wing": (area * w.areal_mass, x_le + x_mac_le_rel + 0.42 * mac, 0.0, wing_z + 0.02),
            "stabilators": (S_h * s.areal_mass + 2 * s.pivot_hardware_mass,
                            stab_x_le + 0.35 * cr_h, 0.0, 0.0),
            "fin_rudder": (S_v * f.areal_mass, fin_x_le + 0.45 * cr_v, 0.0, fz.tail_boom_height / 2 + h_v * 0.4),
            "fuselage": (fus_mass, fus_cx, 0.0, 0.0),
            "landing_gear": (fz.landing_gear_mass, x_le + 0.05, 0.0, -H / 2 - 0.03),
            "wiring": (fz.wiring_mass, x_le, 0.0, 0.0),
            "servos_wing": (2 * a.servo_mass, x_le + 0.5 * cr, 0.0, wing_z),
            "servos_tail": (3 * a.servo_mass, stab_x_le - 0.10, 0.0, 0.0),
        }
        for side, sgn in (("L", -1), ("R", 1)):
            items[f"motor_{side}"] = (p.motor_mass, prop_x + 0.005 + p.motor_length / 2, sgn * motor_y, motor_z)
            items[f"prop_{side}"] = (p.prop_mass, prop_x, sgn * motor_y, motor_z)
            items[f"esc_{side}"] = (p.esc_mass, prop_x + p.motor_length + 0.05, sgn * motor_y, motor_z)
            items[f"nacelle_{side}"] = (0.015, prop_x + nac_len / 2, sgn * motor_y, motor_z)
        for c in comps:
            items[c.name] = (c.box.mass, *c.center)

        return Layout(
            span=d.span, root_chord=cr, tip_chord=ct, y_break=yb, wing_x_le=x_le, wing_z=wing_z,
            wing_area=area, mac=mac, x_mac_le=x_le + x_mac_le_rel,
            stab_semispan=semi_h - fz.tail_boom_width / 2, stab_root_chord=cr_h, stab_tip_chord=ct_h,
            stab_x_le=stab_x_le, stab_z=0.0, stab_y_root=fz.tail_boom_width / 2, stab_area=S_h,
            fin_height=h_v, fin_root_chord=cr_v, fin_tip_chord=ct_v, fin_x_le=fin_x_le,
            fin_z_root=fz.tail_boom_height / 2, fin_area=S_v,
            fus_length=L, fus_width=W, fus_height=H, fus_stations=stations,
            motor_y=motor_y, prop_x=prop_x, motor_z=motor_z, nacelle_radius=nacelle_r, nacelle_length=nac_len,
            components=comps, mass_items=items, tail_arm=l_t, Vh=d.Vh, Vv=Vv, np_offset_mac=np_offset_mac,
        )

    # Place the wing so the CG lands at the static-margin target, for each payload
    # order. Keep the order that puts the flight controller (IMU) closest to the CG.
    best = None
    for order in PAYLOAD_ORDERS:
        fc_target = 0.0
        for _ in range(4 if "fc@cg" in order else 1):  # the FC tray follows the CG
            x_le = brentq(lambda xl: (lambda L: L.cg[0] - (L.x_np - d.static_margin * mac))(
                assemble(xl, order, fc_target)), -0.5, 1.5)
            lay = assemble(x_le, order, fc_target)
            fc_target = lay.cg[0]
        if best is None or abs(lay.fc_offset_from_cg()) < abs(best.fc_offset_from_cg()):
            best = lay
    return best


def check_packaging(lay: Layout, cfg: PlaneConfig) -> list[dict]:
    """Check every payload box (plus clearance) fits inside the fuselage's inner skin."""
    fz = cfg.fuselage
    n = fz.section_exponent
    xs = np.array([s[0] for s in lay.fus_stations])
    ws = np.array([s[1] for s in lay.fus_stations])
    hs = np.array([s[2] for s in lay.fus_stations])
    out = []
    for c in lay.components:
        b = c.box
        worst = 0.0
        for dx in (-b.length / 2, 0, b.length / 2):
            x = c.center[0] + dx
            a_in = np.interp(x, xs, ws) / 2 - fz.wall
            h_in = np.interp(x, xs, hs) / 2 - fz.wall
            for dy in (-1, 1):
                for dz in (-1, 1):
                    y = abs(c.center[1] + dy * (b.width / 2 + fz.clearance))
                    z = abs(c.center[2] + dz * (b.height / 2 + fz.clearance))
                    if a_in <= 0 or h_in <= 0:
                        worst = np.inf
                        continue
                    worst = max(worst, (y / a_in) ** n + (z / h_in) ** n)
        out.append({"name": c.name, "fits": bool(worst <= 1.0 + 1e-6), "fill": float(worst),
                    "x_range": (c.center[0] - b.length / 2, c.center[0] + b.length / 2)})
    return out


def neutral_point_offset(cfg: PlaneConfig, lay: Layout, V: float = 12.0, alpha: float = 2.0) -> float:
    """Neutral point from AeroSandbox's VLM (lifting surfaces), plus a Gilruth
    fuselage/nacelle correction. Returned as (x_np - x_mac_le) / mac."""
    import aerosandbox as asb

    from .geometry import build_airplane

    ref = [lay.x_mac_le, 0.0, 0.0]
    ap = build_airplane(cfg, lay, xyz_ref=ref, lifting_only=True)
    atmo = asb.Atmosphere(altitude=cfg.altitude)

    def run(a):
        r = asb.VortexLatticeMethod(ap, asb.OperatingPoint(atmosphere=atmo, velocity=V, alpha=a), xyz_ref=ref,
                                    spanwise_resolution=8, chordwise_resolution=8).run()
        return float(np.ravel(r["CL"])[0]), float(np.ravel(r["Cm"])[0])

    cl0, cm0 = run(alpha - 1)
    cl1, cm1 = run(alpha + 1)
    CLa, Cma = (cl1 - cl0) / 2, (cm1 - cm0) / 2  # per degree
    x_np_rel = -Cma / CLa  # in MACs aft of the MAC LE
    # Fuselage + nacelles are destabilising: dCm/dalpha (per deg) = Kf w^2 L / (S c)
    S, c = lay.wing_area, lay.mac
    dCma = 0.012 * lay.fus_width**2 * lay.fus_length / (S * c)
    dCma += 2 * 0.006 * (2 * lay.nacelle_radius) ** 2 * lay.nacelle_length / (S * c)
    return x_np_rel - dCma / CLa


def solve_layout(cfg: PlaneConfig) -> Layout:
    """Fixed-point: neutral point and fin size (via the mass-dependent stall speed)."""
    lay = build_layout(cfg, np_offset_mac=0.45)
    for _ in range(3):
        lay = build_layout(cfg, np_offset_mac=neutral_point_offset(cfg, lay), mass_guess=lay.mass)
    return lay
