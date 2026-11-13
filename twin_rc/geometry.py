"""AeroSandbox geometry for the twin-engine high-wing RC plane, built from a `Layout`.

Axes follow AeroSandbox geometry axes: x aft from the nose, y out the right
wing, z up.

Control surfaces
----------------
* Ailerons on the main wing (symmetric=False so they deflect opposite).
* One rudder on the single vertical fin (ControlSurface on the fin).
* Two stabilators: the left and right horizontal tail halves are separate
  all-moving wings. There is no fixed stabiliser and no elevator; a stabilator
  "deflection" rotates the whole half about its pivot (quarter chord by
  default). Positive deflection = leading edge up / trailing edge down, the same
  sign convention as an elevator, so +deflection gives a nose-down moment.
"""

from __future__ import annotations

import aerosandbox as asb
import aerosandbox.numpy as np

from .config import PlaneConfig
from .layout import Layout

INCH = 0.0254


def main_wing(cfg: PlaneConfig, lay: Layout, aileron_deg: float = 0.0) -> asb.Wing:
    w = cfg.wing
    semi = lay.span / 2
    dih = np.radians(w.dihedral_deg)
    af = asb.Airfoil(w.airfoil)
    a0, a1 = (semi * f for f in w.aileron_span_frac)

    def chord_at(y):
        if y <= lay.y_break:
            return lay.root_chord
        frac = (y - lay.y_break) / (semi - lay.y_break)
        return lay.root_chord + frac * (lay.tip_chord - lay.root_chord)

    xsecs = []
    for y in sorted({0.0, lay.y_break, a0, a1, semi}):
        chord = chord_at(y)
        surfaces = []
        if (a0 - 1e-9) <= y < (a1 - 1e-9):  # control surface spans xsec i -> i+1
            surfaces.append(asb.ControlSurface(name="aileron", symmetric=False,
                                               hinge_point=w.aileron_hinge, deflection=aileron_deg))
        xsecs.append(asb.WingXSec(
            # straight quarter-chord line: taper on both edges
            xyz_le=[lay.wing_x_le + 0.25 * (lay.root_chord - chord), y, lay.wing_z + y * np.tan(dih)],
            chord=chord, twist=w.incidence_deg, airfoil=af, control_surfaces=surfaces,
        ))
    return asb.Wing(name="Main Wing", symmetric=True, xsecs=xsecs).subdivide_sections(2)


def _stabilator_half(cfg: PlaneConfig, lay: Layout, side: int, deflection_deg: float) -> asb.Wing:
    """One all-moving tail half. side = +1 right, -1 left."""
    s = cfg.stabilator
    af = asb.Airfoil(s.airfoil)
    d = np.radians(deflection_deg)

    def xsec(y_abs, chord, x_le_undeflected):
        # Rotate the section about its pivot; positive = leading edge up.
        px = x_le_undeflected + s.pivot_frac * chord
        return asb.WingXSec(
            xyz_le=[px - s.pivot_frac * chord * np.cos(d), side * y_abs,
                    lay.stab_z + s.pivot_frac * chord * np.sin(d)],
            chord=chord, twist=deflection_deg, airfoil=af,
        )

    tip_y = lay.stab_y_root + lay.stab_semispan
    root = xsec(lay.stab_y_root, lay.stab_root_chord, lay.stab_x_le)
    tip = xsec(tip_y, lay.stab_tip_chord,
               lay.stab_x_le + lay.stab_semispan * np.tan(np.radians(s.sweep_le_deg)))
    # Sections must run in +y so the normal points up: tip first on the left half.
    return asb.Wing(name="Stabilator R" if side > 0 else "Stabilator L", symmetric=False,
                    xsecs=[root, tip] if side > 0 else [tip, root])


def stabilators(cfg: PlaneConfig, lay: Layout, left_deg: float = 0.0, right_deg: float = 0.0):
    return _stabilator_half(cfg, lay, -1, left_deg), _stabilator_half(cfg, lay, +1, right_deg)


def fin(cfg: PlaneConfig, lay: Layout, rudder_deg: float = 0.0) -> asb.Wing:
    f = cfg.fin
    af = asb.Airfoil(f.airfoil)
    rudder = asb.ControlSurface(name="rudder", symmetric=True, hinge_point=f.rudder_hinge, deflection=rudder_deg)
    return asb.Wing(name="Fin", symmetric=False, xsecs=[
        asb.WingXSec(xyz_le=[lay.fin_x_le, 0, lay.fin_z_root], chord=lay.fin_root_chord, airfoil=af,
                     control_surfaces=[rudder]),
        asb.WingXSec(xyz_le=[lay.fin_x_le + lay.fin_height * np.tan(np.radians(f.sweep_le_deg)), 0,
                             lay.fin_z_root + lay.fin_height], chord=lay.fin_tip_chord, airfoil=af),
    ])


def fuselage(cfg: PlaneConfig, lay: Layout) -> asb.Fuselage:
    return asb.Fuselage(name="Fuselage", xsecs=[
        asb.FuselageXSec(xyz_c=[x, 0, zc], width=w, height=h, shape=cfg.fuselage.section_exponent)
        for x, w, h, zc in lay.fus_stations
    ])


def nacelle(cfg: PlaneConfig, lay: Layout, side: int) -> asb.Fuselage:
    y, r, x0, L = side * lay.motor_y, lay.nacelle_radius, lay.prop_x + 0.008, lay.nacelle_length
    pts = [(0.0, 0.55 * r), (0.15 * L, r), (0.65 * L, r), (L, 0.15 * r)]
    return asb.Fuselage(name=f"Nacelle {'R' if side > 0 else 'L'}",
                        xsecs=[asb.FuselageXSec(xyz_c=[x0 + dx, y, lay.motor_z], radius=rr) for dx, rr in pts])


def propulsors(cfg: PlaneConfig, lay: Layout) -> list[asb.Propulsor]:
    p = cfg.propulsion
    return [
        asb.Propulsor(name=f"Prop {'R' if side > 0 else 'L'} ({p.n_blades}-blade)",
                      xyz_c=[lay.prop_x, side * lay.motor_y, lay.motor_z], xyz_normal=[-1, 0, 0],
                      radius=p.prop_diameter_in * INCH / 2)
        for side in (-1, +1)
    ]


def build_airplane(cfg: PlaneConfig, lay: Layout, stab_left_deg: float = 0.0, stab_right_deg: float = 0.0,
                   rudder_deg: float = 0.0, aileron_deg: float = 0.0, xyz_ref=None,
                   lifting_only: bool = False) -> asb.Airplane:
    wing = main_wing(cfg, lay, aileron_deg)
    stab_l, stab_r = stabilators(cfg, lay, stab_left_deg, stab_right_deg)
    return asb.Airplane(
        name="MVHS Twin High-Wing",
        xyz_ref=xyz_ref if xyz_ref is not None else [0, 0, 0],
        wings=[wing, stab_l, stab_r, fin(cfg, lay, rudder_deg)],
        fuselages=[] if lifting_only else [fuselage(cfg, lay), nacelle(cfg, lay, -1), nacelle(cfg, lay, +1)],
        propulsors=propulsors(cfg, lay),
        s_ref=wing.area(), c_ref=wing.mean_aerodynamic_chord(), b_ref=wing.span(),
    )
