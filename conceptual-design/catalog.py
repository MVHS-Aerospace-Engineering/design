"""
Loads the part catalogs in data/ and builds the data-driven models:
  - AirfoilPolar : NeuralFoil polars -> cd(cl, Re), clmax(Re), lift slope
  - Prop         : APC CT/CP(J, RPM) tables
  - Motor, ESC, Battery (real packs + continuous "rubber" families), Components
"""
import csv
import glob
import math
import os
import re
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from parameters import P

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
IN2M = 0.0254


def _rows(fname):
    with open(os.path.join(DATA, fname), newline="") as f:
        return list(csv.DictReader(f))


# ============================================================== airfoils
class AirfoilPolar:
    """cd(cl, Re) on the pre-stall branch, with a smooth penalty past clmax so
    an optimizer is pushed away from stalled operating points."""
    CL_GRID = np.linspace(-0.8, 2.2, 151)

    def __init__(self, name):
        self.name = name
        geo = {r["name"]: r for r in _rows("airfoils/geometry.csv")}[name]
        self.t_c = float(geo["t_c"])
        self.area_ratio = float(geo["area_ratio"])
        rows = _rows(f"airfoils/{name}.csv")
        re_all = np.array([float(r["re"]) for r in rows])
        self.re_grid = np.unique(re_all)
        cd_tab, clmax, cla, a0, cm0 = [], [], [], [], []
        for re_ in self.re_grid:
            sel = [r for r in rows if float(r["re"]) == re_]
            al = np.array([float(r["alpha_deg"]) for r in sel])
            cl = np.array([float(r["cl"]) for r in sel])
            cd = np.array([float(r["cd"]) for r in sel])
            cm = np.array([float(r["cm"]) for r in sel])
            conf = np.array([float(r["confidence"]) for r in sel])
            good = conf >= 0.7
            # clmax: largest trusted cl at positive alpha
            pos = good & (al > 0)
            i_max = int(np.flatnonzero(pos)[np.argmax(cl[pos])])
            # pre-stall branch: from the most-negative trusted cl up to clmax
            neg = good & (al <= 0)
            i_min = int(np.flatnonzero(neg)[np.argmin(cl[neg])])
            b_cl, b_cd = [], []
            for i in range(i_min, i_max + 1):
                if not b_cl or cl[i] > b_cl[-1] + 1e-6:
                    b_cl.append(cl[i]); b_cd.append(cd[i])
            b_cl, b_cd = np.array(b_cl), np.array(b_cd)
            g = self.CL_GRID
            row = np.interp(g, b_cl, b_cd)
            hi, lo = g > b_cl[-1], g < b_cl[0]
            row[hi] = b_cd[-1] + 0.5 * (g[hi] - b_cl[-1]) + 3.0 * (g[hi] - b_cl[-1]) ** 2
            row[lo] = b_cd[0] + 0.5 * (b_cl[0] - g[lo]) + 3.0 * (b_cl[0] - g[lo]) ** 2
            cd_tab.append(row)
            clmax.append(b_cl[-1])
            lin = good & (al >= -2) & (al <= 4)
            slope, icpt = np.polyfit(np.radians(al[lin]), cl[lin], 1)
            cla.append(slope)
            a0.append(-icpt / slope)
            cm0.append(float(np.mean(cm[lin])))     # ~constant pre-stall (moment about c/4)
        self.log_re = np.log(self.re_grid)
        self._cd = RegularGridInterpolator((self.log_re, self.CL_GRID), np.array(cd_tab))
        self._clmax = np.array(clmax)
        self._cla = np.array(cla)
        self._a0 = np.array(a0)
        self._cm0 = np.array(cm0)

    def _lr(self, re):
        return np.clip(np.log(np.maximum(re, 1.0)), self.log_re[0], self.log_re[-1])

    def cd(self, cl, re):
        cl, re = np.broadcast_arrays(np.asarray(cl, float), np.asarray(re, float))
        pts = np.stack([self._lr(re).ravel(),
                        np.clip(cl, self.CL_GRID[0], self.CL_GRID[-1]).ravel()], axis=-1)
        return self._cd(pts).reshape(cl.shape)

    def clmax(self, re):
        return np.interp(self._lr(re), self.log_re, self._clmax)

    def cl_alpha(self, re):
        """2-D lift-curve slope [1/rad]"""
        return np.interp(self._lr(re), self.log_re, self._cla)

    def alpha0(self, re):
        """zero-lift angle [rad]"""
        return np.interp(self._lr(re), self.log_re, self._a0)

    def cm0(self, re):
        """quarter-chord pitching moment in the linear range"""
        return np.interp(self._lr(re), self.log_re, self._cm0)


def wing_airfoil_names():
    return [r["name"] for r in _rows("airfoils/geometry.csv") if r["role"] == "wing"]


@lru_cache(maxsize=None)
def airfoil(name):
    return AirfoilPolar(name)


# ============================================================== props
class Prop:
    """APC published performance data (PER3_*.dat): CT, CP vs (RPM, J)."""

    def __init__(self, path):
        txt = open(path, errors="ignore").read().splitlines()
        self.name = "APC " + txt[0].split()[0]
        m = re.match(r"([\d.]+)x([\d.]+)", txt[0].split()[0])
        self.d_in, self.pitch_in = float(m.group(1)), float(m.group(2))
        self.D = self.d_in * IN2M
        blocks, rpm = {}, None
        for line in txt:
            if "PROP RPM" in line:
                rpm = float(line.split("=")[1])
                blocks[rpm] = []
                continue
            parts = line.split()
            if rpm is None or len(parts) < 5:
                continue
            try:
                vals = [float(p) for p in parts[:5]]
            except ValueError:
                continue
            blocks[rpm].append(vals)   # V(mph), J, Pe, Ct, Cp
        self.rpms = np.array(sorted(blocks))
        jmax = max(np.array(b)[:, 1].max() for b in blocks.values())
        self.j_grid = np.linspace(0.0, jmax * 1.15, 140)
        ct, cp = [], []
        for r in self.rpms:
            b = np.array(blocks[r])
            j, c_t, c_p = b[:, 1], b[:, 3], b[:, 4]
            ct.append(self._extrap(j, c_t))
            cp.append(self._extrap(j, c_p))
        self._ct = RegularGridInterpolator((self.rpms, self.j_grid), np.array(ct))
        self._cp = RegularGridInterpolator((self.rpms, self.j_grid), np.array(cp))
        self.rpm_max = float(self.rpms[-1])
        self.mass_kg = P.prop_mass_10in_kg * (self.d_in / 10.0) ** P.prop_mass_exponent

    def _extrap(self, j, c):
        out = np.interp(self.j_grid, j, c)
        beyond = self.j_grid > j[-1]
        slope = (c[-1] - c[-2]) / (j[-1] - j[-2])
        out[beyond] = c[-1] + slope * (self.j_grid[beyond] - j[-1])
        return out

    def coeffs(self, rpm, J):
        rpm, J = np.broadcast_arrays(np.asarray(rpm, float), np.asarray(J, float))
        pts = np.stack([np.clip(rpm, self.rpms[0], self.rpms[-1]).ravel(),
                        np.clip(J, 0.0, self.j_grid[-1]).ravel()], axis=-1)
        return self._ct(pts).reshape(rpm.shape), self._cp(pts).reshape(rpm.shape)


@lru_cache(maxsize=None)
def props():
    return {p.name: p for p in (Prop(f) for f in sorted(glob.glob(os.path.join(DATA, "props", "PER3_*.dat"))))}


# ============================================================== motors / ESCs
@dataclass(frozen=True)
class Motor:
    name: str
    kv: float        # rpm/V
    io: float        # A
    rm: float        # ohm (phase-to-phase)
    imax: float      # A continuous
    mass_kg: float
    od_m: float
    max_cells: int
    source: str

    @property
    def kv_rad(self):          # rad/s per V
        return self.kv * 2 * math.pi / 60

    @property
    def kt(self):              # N m / A
        return 1.0 / self.kv_rad


@lru_cache(maxsize=None)
def motors():
    return {r["name"]: Motor(r["name"], float(r["kv_rpm_per_v"]), float(r["io_a"]),
                             float(r["rm_ohm"]), float(r["imax_a"]), float(r["mass_g"]) / 1000,
                             float(r["od_mm"]) / 1000, int(r["max_cells"]), r["source"])
            for r in _rows("motors.csv")}


@dataclass(frozen=True)
class ESC:
    name: str
    i_cont: float
    max_cells: int
    mass_kg: float


@lru_cache(maxsize=None)
def escs():
    return sorted((ESC(r["name"], float(r["i_cont_a"]), int(r["max_cells"]), float(r["mass_g"]) / 1000)
                   for r in _rows("escs.csv")), key=lambda e: e.mass_kg)


def pick_esc(current_a, cells):
    for e in escs():
        if e.i_cont >= P.esc_margin * current_a and e.max_cells >= cells:
            return e
    return None


# ============================================================== batteries
@dataclass(frozen=True)
class Battery:
    name: str
    chemistry: str       # "LiPo" | "Li-ion"
    s: int
    p: float             # parallel count (float only for rubber Li-ion)
    v_nom: float         # pack nominal (open-circuit average) voltage
    energy_wh: float
    mass_kg: float
    r_int: float         # pack resistance, ohm
    i_max: float         # continuous current limit, A
    dims_m: tuple        # (length, width, height)
    real: bool = True

    @property
    def k(self):
        return P.k_lipo if self.chemistry == "LiPo" else P.k_liion

    @property
    def u_wh_kg(self):
        return self.energy_wh / self.mass_kg


def _lipo_from_row(r, n_par=1):
    s, ah, c = int(r["cells_s"]), float(r["capacity_mah"]) / 1000, float(r["c_rating"])
    r_cell = P.lipo_sag_at_rated_v / (c * ah)
    l, w, h = (float(r[k]) / 1000 for k in ("l_mm", "w_mm", "h_mm"))
    name = r["name"] if n_par == 1 else f"2x {r['name']} (parallel)"
    return Battery(name, "LiPo", s, n_par, s * P.lipo_v_nom, s * P.lipo_v_nom * ah * n_par,
                   n_par * float(r["mass_g"]) / 1000, s * r_cell / n_par,
                   P.lipo_current_derate * c * ah * n_par, (l, w, h * n_par))


def _liion_pack(cell, s, p, real=True):
    mass = s * p * (float(cell["mass_g"]) / 1000 + P.liion_mass_per_cell_kg) + P.liion_pack_overhead_kg
    dia, ln = float(cell["diameter_mm"]) / 1000, float(cell["length_mm"]) / 1000
    ah = float(cell["capacity_mah"]) / 1000
    v = s * P.liion_v_nom
    name = f"{s}S{p:g}P {cell['name']}" if real else f"{s}S{p:.2f}P {cell['name']}"
    return Battery(name, "Li-ion", s, p, v, v * ah * p, mass,
                   s * float(cell["r_dc_mohm"]) / 1000 / p,
                   P.liion_current_derate * float(cell["imax_a"]) * p,
                   (ln, s * dia, max(p, 1.0) * dia), real)


def real_batteries():
    out = []
    for r in _rows("lipo_packs.csv"):
        out.append(_lipo_from_row(r, 1))
        out.append(_lipo_from_row(r, 2))
    for c in _rows("liion_cells.csv"):
        for s in (3, 4):
            for p in (1, 2, 3, 4):
                out.append(_liion_pack(c, s, p))
    return out


class BatteryFamily:
    """Continuous ('rubber') battery used inside the optimizer, built from the
    real catalog so it stays honest. `make(E)` returns a pack of energy E;
    `snap(E)` returns real packs of the same family with at least E."""

    def __init__(self, chemistry, s, cell=None):
        self.chemistry, self.s, self.cell = chemistry, s, cell
        if chemistry == "LiPo":
            rows = [r for r in _rows("lipo_packs.csv") if int(r["cells_s"]) == s]
            packs = [_lipo_from_row(r) for r in rows]
            E = np.array([b.energy_wh for b in packs])
            m = np.array([b.mass_kg for b in packs])
            self._fit = np.polyfit(E, m, 1)                  # mass = a*E + b
            vol = np.array([np.prod(b.dims_m) for b in packs])
            self._rho = float(np.mean(m / vol))              # pack density
            self._shape = np.mean([np.array(b.dims_m) / np.prod(b.dims_m) ** (1 / 3) for b in packs], axis=0)
            self._c = 30.0                                   # Soaring line C-rating
            self.name = f"LiPo {s}S"
        else:
            self.name = f"Li-ion {s}S {cell['name']}"
            self._cell_e = s * P.liion_v_nom * float(cell["capacity_mah"]) / 1000

    def make(self, energy_wh):
        energy_wh = max(energy_wh, 1.0)
        if self.chemistry == "LiPo":
            mass = float(np.polyval(self._fit, energy_wh))
            ah = energy_wh / (self.s * P.lipo_v_nom)
            dims = tuple(self._shape * (mass / self._rho) ** (1 / 3))
            return Battery(f"{self.name} {ah * 1000:.0f}mAh (continuous)", "LiPo", self.s, 1,
                           self.s * P.lipo_v_nom, energy_wh, mass,
                           self.s * P.lipo_sag_at_rated_v / (self._c * ah),
                           P.lipo_current_derate * self._c * ah, dims, real=False)
        p = max(energy_wh / self._cell_e, 1.0)      # a real pack has at least one full string
        return _liion_pack(self.cell, self.s, p, real=False)

    def options(self):
        if self.chemistry == "LiPo":
            return [b for b in real_batteries() if b.chemistry == "LiPo" and b.s == self.s]
        return [_liion_pack(self.cell, self.s, p) for p in (1, 2, 3, 4, 5, 6)]


def battery_families():
    fams = [BatteryFamily("LiPo", 3), BatteryFamily("LiPo", 4)]
    for c in _rows("liion_cells.csv"):
        fams += [BatteryFamily("Li-ion", 3, c), BatteryFamily("Li-ion", 4, c)]
    return fams


# ============================================================== avionics / materials
@dataclass(frozen=True)
class Component:
    name: str
    qty: int
    mass_kg: float
    power_w: float
    dims_m: tuple
    in_fuselage: bool
    tag: str
    source: str


@lru_cache(maxsize=None)
def components():
    return [Component(r["name"], int(r["qty"]), float(r["mass_g"]) / 1000, float(r["power_w"]),
                      tuple(float(r[k]) / 1000 for k in ("l_mm", "w_mm", "h_mm")),
                      r["in_fuselage"] == "1", r["tag"], r["source"])
            for r in _rows("components.csv")]


@lru_cache(maxsize=None)
def materials():
    out = {}
    for r in _rows("materials.csv"):
        out[r["name"]] = {k: (float(r[k]) if r[k] else None)
                          for k in ("density_kg_m3", "areal_kg_m2", "allowable_pa")}
    return out


def with_battery_mass(b, mass):
    return replace(b, mass_kg=mass)


# ============================================================== wheels + brakes
@dataclass(frozen=True)
class Wheel:
    name: str
    d: float            # m
    w: float            # m
    mass_kg: float
    axle_mm: float
    tag: str
    source: str


@lru_cache(maxsize=None)
def wheels():
    return {r["name"]: Wheel(r["name"], float(r["diameter_mm"]) / 1000, float(r["width_mm"]) / 1000,
                             float(r["mass_g"]) / 1000, float(r["axle_mm"]) if r["axle_mm"] else float("nan"),
                             r["tag"], r["source"])
            for r in _rows("wheels.csv")}


@dataclass(frozen=True)
class Brake:
    name: str
    mass_per_wheel_kg: float
    actuator: str
    actuator_mass_kg: float
    cable_mass_kg: float
    d_min: float        # m, wheel diameters the brake fits
    d_max: float
    includes_wheels: bool
    tag: str
    source: str

    @property
    def active(self):
        return self.name != "none"

    def fits(self, wheel):
        return self.d_min - 1e-6 <= wheel.d <= self.d_max + 1e-6

    def mass_kg(self, n_wheels=2):
        return n_wheels * self.mass_per_wheel_kg + self.actuator_mass_kg + self.cable_mass_kg


@lru_cache(maxsize=None)
def brakes():
    return {r["name"]: Brake(r["name"], float(r["mass_per_wheel_g"]) / 1000, r["actuator"],
                             float(r["actuator_mass_g"]) / 1000, float(r["cable_mass_g"]) / 1000,
                             float(r["wheel_d_min_mm"]) / 1000, float(r["wheel_d_max_mm"]) / 1000,
                             r["includes_wheels"] == "1", r["tag"], r["source"])
            for r in _rows("brakes.csv")}


def main_wheel():
    return wheels()[P.main_wheel]


def nose_wheel():
    return wheels()[P.nose_wheel]


def wheel_brake():
    return brakes()[P.wheel_brake]


@lru_cache(maxsize=None)
def airfoil_shape(name):
    """Section geometry from the airfoil coordinates (AeroSandbox):
    x/c of max thickness, area centroid x/c, and the coordinates (Selig order)."""
    import aerosandbox as asb
    af = asb.Airfoil(name)
    xy = np.asarray(af.coordinates, float)
    xs = np.linspace(0.0, 1.0, 401)
    t = np.asarray(af.local_thickness(x_over_c=xs), float)
    i = int(np.argmax(t))
    x, y = xy[:, 0], xy[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    A = 0.5 * np.sum(cross)
    cx = np.sum((x + np.roll(x, -1)) * cross) / (6 * A) if abs(A) > 1e-9 else 0.4
    camber = float(np.asarray(af.local_camber(x_over_c=xs[i]), float))
    return dict(x_tmax=float(xs[i]), t_max=float(t[i]), x_centroid=float(cx), camber_at_tmax=camber, coords=xy)
