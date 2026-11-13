"""AeroSandbox geometry for the twin-engine high-wing RC plane.

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

INCH = 0.0254


def _airfoil(name: str) -> asb.Airfoil:
    return asb.Airfoil(name)


def main_wing(cfg: PlaneConfig, aileron_deg: float = 0.0) -> asb.Wing:
    w = cfg.wing
    semi = w.span / 2
    y_break = semi * w.taper_start_frac
    dih = np.radians(w.dihedral_deg)
    af = _airfoil(w.airfoil)

    def xsec(y, chord, ailerons):
        # Keep the quarter chord line straight (unswept) so taper sits on both edges.
        x_le = w.x_le + 0.25 * (w.root_chord - chord)
        surfaces = []
        if ailerons:
            surfaces.append(
                asb.ControlSurface(
                    name="aileron",
                    symmetric=False,
                    hinge_point=w.aileron_hinge,
                    deflection=aileron_deg,
                )
            )
        return asb.WingXSec(
            xyz_le=[x_le, y, w.z + y * np.tan(dih)],
            chord=chord,
            twist=w.incidence_deg,
            airfoil=af,
            control_surfaces=surfaces,
        )

    a0, a1 = (semi * f for f in w.aileron_span_frac)

    def chord_at(y):
        if y <= y_break:
            return w.root_chord
        frac = (y - y_break) / (semi - y_break)
        return w.root_chord + frac * (w.tip_chord - w.root_chord)

    stations = sorted({0.0, y_break, a0, a1, semi})
    xsecs = []
    for i, y in enumerate(stations):
        # A control surface on xsec i spans to xsec i+1.
        has_aileron = (a0 - 1e-9) <= y < (a1 - 1e-9)
        xsecs.append(xsec(y, chord_at(y), has_aileron))

    return asb.Wing(name="Main Wing", symmetric=True, xsecs=xsecs).subdivide_sections(2)


def _stabilator_half(cfg: PlaneConfig, side: int, deflection_deg: float) -> asb.Wing:
    """One all-moving tail half. side = +1 right, -1 left."""
    s = cfg.stabilator
    af = _airfoil(s.airfoil)
    d = np.radians(deflection_deg)

    def xsec(y_abs, chord, x_le_undeflected):
        # Rotate the section about its pivot point. Positive deflection is
        # leading edge up (twist up), as for an elevator with TE down.
        px = x_le_undeflected + s.pivot_frac * chord
        pz = s.z
        x_le = px - s.pivot_frac * chord * np.cos(d)
        z_le = pz + s.pivot_frac * chord * np.sin(d)
        return asb.WingXSec(
            xyz_le=[x_le, side * y_abs, z_le],
            chord=chord,
            twist=deflection_deg,
            airfoil=af,
        )

    root = xsec(s.y_root, s.root_chord, s.x_le)
    tip_y = s.y_root + s.semispan
    tip = xsec(tip_y, s.tip_chord, s.x_le + s.semispan * np.tan(np.radians(s.sweep_le_deg)))
    # Section order must run root -> tip in +y (right) so the surface normal points up.
    # For the left half that means tip first.
    xsecs = [root, tip] if side > 0 else [tip, root]
    name = "Stabilator R" if side > 0 else "Stabilator L"
    return asb.Wing(name=name, symmetric=False, xsecs=xsecs)


def stabilators(cfg: PlaneConfig, left_deg: float = 0.0, right_deg: float = 0.0):
    return _stabilator_half(cfg, -1, left_deg), _stabilator_half(cfg, +1, right_deg)


def fin(cfg: PlaneConfig, rudder_deg: float = 0.0) -> asb.Wing:
    f = cfg.fin
    af = _airfoil(f.airfoil)
    rudder = asb.ControlSurface(
        name="rudder", symmetric=True, hinge_point=f.rudder_hinge, deflection=rudder_deg
    )
    return asb.Wing(
        name="Fin",
        symmetric=False,
        xsecs=[
            asb.WingXSec(
                xyz_le=[f.x_le, 0, f.z_root],
                chord=f.root_chord,
                airfoil=af,
                control_surfaces=[rudder],
            ),
            asb.WingXSec(
                xyz_le=[f.x_le + f.height * np.tan(np.radians(f.sweep_le_deg)), 0, f.z_root + f.height],
                chord=f.tip_chord,
                airfoil=af,
            ),
        ],
    )


def fuselage(cfg: PlaneConfig) -> asb.Fuselage:
    fz = cfg.fuselage
    # (x/L, width fraction, height fraction, z centre)
    profile = [
        (0.00, 0.30, 0.30, 0.02),
        (0.04, 0.70, 0.70, 0.02),
        (0.12, 0.95, 0.95, 0.03),
        (0.30, 1.00, 1.00, 0.03),
        (0.50, 0.85, 0.80, 0.03),
        (0.75, 0.50, 0.45, 0.03),
        (0.92, 0.32, 0.30, 0.04),
        (1.00, 0.22, 0.22, 0.04),
    ]
    xsecs = [
        asb.FuselageXSec(
            xyz_c=[xf * fz.length, 0, zc],
            width=wf * fz.max_width,
            height=hf * fz.max_height,
            shape=3.0,
        )
        for xf, wf, hf, zc in profile
    ]
    return asb.Fuselage(name="Fuselage", xsecs=xsecs)


def nacelle(cfg: PlaneConfig, side: int) -> asb.Fuselage:
    p = cfg.propulsion
    y = side * p.motor_y
    x0 = p.motor_x + 0.01
    pts = [(0.00, 0.012), (0.03, 0.022), (0.10, 0.022), (0.20, 0.014), (0.26, 0.004)]
    return asb.Fuselage(
        name=f"Nacelle {'R' if side > 0 else 'L'}",
        xsecs=[asb.FuselageXSec(xyz_c=[x0 + dx, y, p.motor_z], radius=r) for dx, r in pts],
    )


def propulsors(cfg: PlaneConfig) -> list[asb.Propulsor]:
    p = cfg.propulsion
    r = p.prop_diameter_in * INCH / 2
    return [
        asb.Propulsor(
            name=f"Prop {'R' if side > 0 else 'L'} ({p.n_blades}-blade)",
            xyz_c=[p.motor_x, side * p.motor_y, p.motor_z],
            xyz_normal=[-1, 0, 0],
            radius=r,
        )
        for side in (-1, +1)
    ]


def build_airplane(
    cfg: PlaneConfig,
    stab_left_deg: float = 0.0,
    stab_right_deg: float = 0.0,
    rudder_deg: float = 0.0,
    aileron_deg: float = 0.0,
    xyz_ref=None,
) -> asb.Airplane:
    stab_l, stab_r = stabilators(cfg, stab_left_deg, stab_right_deg)
    wing = main_wing(cfg, aileron_deg)
    return asb.Airplane(
        name="MVHS Twin High-Wing",
        xyz_ref=xyz_ref if xyz_ref is not None else [0, 0, 0],
        wings=[wing, stab_l, stab_r, fin(cfg, rudder_deg)],
        fuselages=[fuselage(cfg), nacelle(cfg, -1), nacelle(cfg, +1)],
        propulsors=propulsors(cfg),
        s_ref=wing.area(),
        c_ref=wing.mean_aerodynamic_chord(),
        b_ref=wing.span(),
    )
