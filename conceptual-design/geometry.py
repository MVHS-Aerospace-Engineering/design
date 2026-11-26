"""
Builds the aircraft layout from the sizing variables.

Configuration (README notional concept): high wing on top of the fuselage,
twin wing-mounted tractor props, conventional tail, fixed tricycle gear.

The fuselage is sized by what it must carry (battery + avionics + battery trim
slide), the motor station by prop clearance, the gear height by prop ground
clearance / wheel size. The fin is the larger of the Raymer tail-volume value
and the coupled engine-out need (lateral.py); the stabilizer comes from the
scissor plot (controls.tail_and_elevator), never below the Raymer value while
P.htail_volume_floor is set, or, with P.htail_sizing = 'tail_volume', the Raymer value. The wing position along the fuselage (x_qc)
is solved in the sizing loop so the nominal CG sits mid-band (weights.cg_budget).
"""
import math
from dataclasses import dataclass

import numpy as np

from catalog import components, main_wheel, nose_wheel, wheel_brake
from parameters import P

N_STATIONS = 12   # spanwise strips per semispan


@dataclass
class Surface:
    S: float
    b: float
    taper: float
    c_root: float
    c_tip: float
    mac: float
    t_c: float
    S_wet: float


def _surface(S, AR, taper, t_c):
    """Trapezoidal surface. For a fin, b is its height and AR = h^2/S."""
    b = math.sqrt(AR * S)
    c_root = 2 * S / (b * (1 + taper))
    mac = (2 / 3) * c_root * (1 + taper + taper ** 2) / (1 + taper)
    s_wet = S * (1.977 + 0.52 * t_c)                  # Raymer eq. 7.11
    return Surface(S, b, taper, c_root, taper * c_root, mac, t_c, s_wet)


@dataclass
class Geometry:
    wing: Surface
    htail: Surface
    vtail: Surface
    l_t: float
    # spanwise strips (semispan)
    y: np.ndarray
    dy: float
    chord: np.ndarray
    schrenk: np.ndarray       # local cl / CL (Schrenk)
    # fuselage
    fus_w: float
    fus_h: float
    fus_d_eq: float
    fus_L: float
    fus_S_wet: float
    bay_L: float
    # nacelles, motors, props
    nac_d: float
    nac_L: float
    nac_S_wet: float
    y_eng: float
    prop_D: float
    # gear
    gear_h: float
    track: float
    main_leg_L: float
    nose_strut_L: float
    wing_height: float
    sv_volume: float
    sv_engine_out: float
    sh_volume: float = 0.0
    sh_trim: float = 0.0
    washout_deg: float = 0.0          # geometric twist: tip incidence lower by this
    t_c_tip: float = 0.0
    x_qc: float = 0.0                 # wing root quarter chord, aft of the nose tip
    L_nose: float = 0.0
    main_wheel: object = None
    nose_wheel: object = None
    brake: object = None
    x_qc_clamped: bool = False
    dihedral_deg: float = 0.0         # each wing panel, from the root (sized for spiral stability)

    @property
    def x_bay_center(self):
        return self.L_nose + self.bay_L / 2

    @property
    def x_tail(self):
        return self.x_qc + self.l_t

    def chord_at(self, y):
        y = np.abs(np.asarray(y, float))
        return self.wing.c_root * (1 - (1 - self.wing.taper) * 2 * y / self.wing.b)

    def twist_at(self, y):
        """local incidence relative to the root [rad] (linear washout)"""
        return -math.radians(self.washout_deg) * 2 * np.abs(np.asarray(y, float)) / self.wing.b

    @property
    def S(self):
        return self.wing.S

    @property
    def b(self):
        return self.wing.b

    @property
    def AR(self):
        return self.wing.b ** 2 / self.wing.S


def fuselage_profile(nose, bay, L_f):
    """Fuselage outer shape: (x, width factor, height factor) stations, rounded-box
    sections (the same shape preliminary-design/visualize.py draws)."""
    x_be = nose + bay
    return [(0.0, 0.20, 0.20), (0.3 * nose, 0.70, 0.72), (0.7 * nose, 0.95, 0.96), (nose, 1.0, 1.0),
            (x_be, 1.0, 1.0), (x_be + 0.35 * (L_f - x_be), 0.62, 0.70), (L_f, 0.30, 0.40)]


def fuselage_volume_centroid(geom):
    """Volume (elliptic sections) and shell-area centroid x of the fuselage."""
    st = fuselage_profile(geom.L_nose, geom.bay_L, geom.fus_L)
    x = np.array([s[0] for s in st])
    kw = np.array([s[1] for s in st])
    kh = np.array([s[2] for s in st])
    xs = np.linspace(0, geom.fus_L, 200)
    w = np.interp(xs, x, kw) * geom.fus_w
    h = np.interp(xs, x, kh) * geom.fus_h
    area = math.pi / 4 * w * h
    perim = math.pi * (w + h) / 2
    vol = float(np.trapezoid(area, xs))
    x_c = float(np.trapezoid(perim * xs, xs) / np.trapezoid(perim, xs))
    return vol, x_c


def fuselage_bay(battery):
    """Internal width/height/length needed for the battery + avionics (+ the
    battery trim slide, P.battery_slide_m each way)."""
    c = P.internal_clearance_m
    comps = [k for k in components() if k.in_fuselage and k.dims_m[0] > 0]
    w_i = max([battery.dims_m[1]] + [k.dims_m[1] for k in comps]) + 2 * c
    h_i = battery.dims_m[2] + P.pi_stack_height_m + 2 * c
    vol = sum(k.qty * np.prod(k.dims_m) for k in comps) + np.prod(battery.dims_m)
    L = max(P.fuselage_packing_factor * vol / (w_i * h_i),
            battery.dims_m[0] + 2 * c + 2 * P.battery_slide_m, max(k.dims_m[0] for k in comps) + 2 * c)
    return w_i, h_i, L


def build(S, AR, lt_ratio, wing_secs, motor, prop, battery, sv_engine_out=0.0,
          washout_deg=0.0, sh_trim=0.0, x_qc=None, taper=None, dihedral_deg=0.0):
    """wing_secs: aerodynamics.WingSections (root/tip airfoil). sv_engine_out: fin
    area the engine-out trim needs (the fin is never below the Raymer tail-volume
    value). sh_trim: stabilizer area from the scissor plot (0 = not sized yet ->
    Raymer value). x_qc: wing root quarter-chord station (None = bay centre)."""
    t_root, t_tip = wing_secs.root.t_c, wing_secs.tip.t_c
    wing = _surface(S, AR, P.wing_taper if taper is None else taper, 0.5 * (t_root + t_tip))
    wing.t_c = t_root                       # spar depth is set by the root section
    b = wing.b

    # spanwise strips + Schrenk lift distribution (average of planform and ellipse)
    y = (np.arange(N_STATIONS) + 0.5) * (b / 2) / N_STATIONS
    dy = (b / 2) / N_STATIONS
    chord = wing.c_root * (1 - (1 - wing.taper) * 2 * y / b)
    c_ell = 4 * S / (math.pi * b) * np.sqrt(1 - (2 * y / b) ** 2)
    schrenk = 0.5 * (chord + c_ell) / chord

    # tails (Raymer tail volume method; fin also engine-out)
    l_t = lt_ratio * wing.mac
    sh_vol = P.tail_cht * wing.mac * S / l_t
    if P.htail_sizing == "scissor" and sh_trim > 0:
        # scissor-plot area, floored at the Raymer tail volume (P.htail_volume_floor): without the
        # floor the scissor plot shrank the stabilizer until it sat at its stall angle on approach
        floor = sh_vol if P.htail_volume_floor else 0.05 * sh_vol   # (0.05: numerical guard only)
        S_h = max(sh_trim, floor)
    else:
        S_h = max(sh_vol, sh_trim)
    sv_vol = P.tail_cvt * b * S / l_t
    S_v = max(sv_vol, sv_engine_out)
    from catalog import airfoil
    t_c_tail = airfoil(P.tail_airfoil).t_c
    htail = _surface(S_h, P.ar_htail, P.taper_htail, t_c_tail)
    vtail = _surface(S_v, P.ar_vtail, P.taper_vtail, t_c_tail)

    # fuselage: sized by contents
    w_i, h_i, bay = fuselage_bay(battery)
    t = P.fus_skin_t_m
    w_o, h_o = w_i + 2 * t, h_i + 2 * t
    d_eq = math.sqrt(w_o * h_o)
    L_nose = P.nose_length_ratio * d_eq
    x_min = 0.25 * wing.c_root + 0.005                # wing leading edge behind the nose tip
    clamped = False
    if x_qc is None:
        x_qc = L_nose + bay / 2
    elif x_qc < x_min:
        x_qc, clamped = x_min, True
    # the fuselage runs to the furthest tail trailing edge (stabilizer or fin root)
    L_f = max(x_qc + l_t + 0.75 * max(htail.c_root, vtail.c_root), L_nose + bay + 1.5 * d_eq)
    lam = L_f / d_eq
    a_, b_ = w_o / 2, h_o / 2       # elliptic section perimeter (Ramanujan)
    perim = math.pi * (3 * (a_ + b_) - math.sqrt((3 * a_ + b_) * (a_ + 3 * b_)))
    s_wet_f = (perim * L_f * max(1 - 2 / lam, 0.05) ** (2 / 3) * (1 + 1 / lam ** 2))  # Torenbeek

    # nacelles + motor station
    nac_d = motor.od_m + 2 * P.nacelle_wall_gap_m
    nac_L = 1.2 * motor.od_m + P.nacelle_extra_len_m
    nac_S_wet = 0.8 * math.pi * nac_d * nac_L
    y_eng = w_o / 2 + prop.D / 2 + P.prop_fuselage_clearance_m

    # gear: high wing, thrust line ~ at wing mid-thickness above fuselage top
    c_eng = wing.c_root * (1 - (1 - wing.taper) * min(2 * y_eng / b, 1.0))
    z_axis_above_fus_bottom = h_o + 0.5 * wing.t_c * c_eng
    mw, nw = main_wheel(), nose_wheel()
    gear_h = max(mw.d / 2 + 0.02,                                     # main wheels beside the fuselage
                 prop.D / 2 + P.prop_ground_clearance_m - z_axis_above_fus_bottom,
                 nw.d + P.nose_wheel_clearance_m)                       # nose wheel directly under it
    track = w_o + 2 * gear_h
    main_leg = math.sqrt(2) * gear_h
    nose_strut = gear_h + 0.03

    return Geometry(wing, htail, vtail, l_t, y, dy, chord, schrenk,
                    w_o, h_o, d_eq, L_f, s_wet_f, bay,
                    nac_d, nac_L, nac_S_wet, y_eng, prop.D,
                    gear_h, track, main_leg, nose_strut, gear_h + h_o,
                    sv_vol, sv_engine_out, sh_vol, sh_trim, washout_deg, t_tip,
                    x_qc, L_nose, mw, nw, wheel_brake(), clamped, dihedral_deg)
