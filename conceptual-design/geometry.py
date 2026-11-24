"""
Builds the aircraft layout from the sizing variables.

Configuration (README notional concept): high wing on top of the fuselage,
twin wing-mounted tractor props, conventional tail, fixed tricycle gear.

The fuselage is sized by what it must carry (battery + avionics), the motor
station by prop clearance, the gear height by prop ground clearance, and the
tails by Raymer tail-volume coefficients (fin also by engine-out if larger).
"""
import math
from dataclasses import dataclass

import numpy as np

from catalog import components
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


def fuselage_bay(battery):
    """Internal width/height/length needed for the battery + avionics."""
    c = P.internal_clearance_m
    comps = [k for k in components() if k.in_fuselage and k.dims_m[0] > 0]
    w_i = max([battery.dims_m[1]] + [k.dims_m[1] for k in comps]) + 2 * c
    h_i = battery.dims_m[2] + P.pi_stack_height_m + 2 * c
    vol = sum(k.qty * np.prod(k.dims_m) for k in comps) + np.prod(battery.dims_m)
    L = max(P.fuselage_packing_factor * vol / (w_i * h_i),
            battery.dims_m[0] + 2 * c, max(k.dims_m[0] for k in comps) + 2 * c)
    return w_i, h_i, L


def build(S, AR, lt_ratio, wing_secs, motor, prop, battery, sv_engine_out=0.0,
          washout_deg=0.0, sh_trim=0.0):
    """wing_secs: aerodynamics.WingSections (root/tip airfoil). sv_engine_out /
    sh_trim: fin / stabilizer areas required by engine-out / elevator sizing
    (the larger of these and the Raymer tail-volume value is used)."""
    t_root, t_tip = wing_secs.root.t_c, wing_secs.tip.t_c
    wing = _surface(S, AR, P.wing_taper, 0.5 * (t_root + t_tip))
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
    L_f = max(L_nose + bay / 2 + l_t + 0.75 * htail.mac, L_nose + bay + 1.5 * d_eq)
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
    gear_h = max(P.wheel_d_main_m / 2 + 0.02,
                 prop.D / 2 + P.prop_ground_clearance_m - z_axis_above_fus_bottom)
    track = w_o + 2 * gear_h
    main_leg = math.sqrt(2) * gear_h
    nose_strut = gear_h + 0.03

    return Geometry(wing, htail, vtail, l_t, y, dy, chord, schrenk,
                    w_o, h_o, d_eq, L_f, s_wet_f, bay,
                    nac_d, nac_L, nac_S_wet, y_eng, prop.D,
                    gear_h, track, main_leg, nose_strut, gear_h + h_o,
                    sv_vol, sv_engine_out, sh_vol, sh_trim, washout_deg, t_tip)
