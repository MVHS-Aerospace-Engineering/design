"""Exports for cross-checking the design in XFLR5 and AVL.

XFLR5 planes have one main wing, one "elevator" and one fin. The two
stabilators are therefore exported as a single symmetric all-moving tail. In
XFLR5, give that tail a tilt angle (Plane -> Edit -> Elevator -> tilt) to model
a collective stabilator deflection, or add it as a control variable in a
Stability (T7) analysis. Differential stabilator can't be modelled in XFLR5;
use the AeroSandbox results for that.
"""

from __future__ import annotations

from pathlib import Path

import aerosandbox as asb
import numpy as np

from .analysis import TwinPlane
from .geometry import fin, main_wing


def symmetric_stabilator(plane: TwinPlane) -> asb.Wing:
    """Both stabilator halves as one symmetric wing, with the centre gap closed."""
    s, lay = plane.cfg.stabilator, plane.layout
    af = asb.Airfoil(s.airfoil)
    tip_x = lay.stab_x_le + lay.stab_semispan * np.tan(np.radians(s.sweep_le_deg))
    return asb.Wing(
        name="Stabilators",
        symmetric=True,
        xsecs=[
            asb.WingXSec(xyz_le=[lay.stab_x_le, 0, lay.stab_z], chord=lay.stab_root_chord, airfoil=af),
            asb.WingXSec(xyz_le=[lay.stab_x_le, lay.stab_y_root, lay.stab_z], chord=lay.stab_root_chord,
                         airfoil=af),
            asb.WingXSec(xyz_le=[tip_x, lay.stab_y_root + lay.stab_semispan, lay.stab_z],
                         chord=lay.stab_tip_chord, airfoil=af),
        ],
    )


def export_all(plane: TwinPlane, out: Path) -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    cfg = plane.cfg
    files = {}

    wing, tail, vfin = main_wing(cfg, plane.layout), symmetric_stabilator(plane), fin(cfg, plane.layout)
    xflr_plane = asb.Airplane(
        name=plane.clean.name, xyz_ref=plane.xyz_cg, wings=[wing, tail, vfin],
        s_ref=plane.clean.s_ref, c_ref=plane.clean.c_ref, b_ref=plane.clean.b_ref,
    )
    files["xflr5"] = out / "twin_rc_xflr5.xml"
    xflr_plane.export_XFLR5_xml(
        files["xflr5"], mass_props=plane.mass, mainwing=wing, elevator=tail, fin=vfin
    )

    files["avl"] = out / "twin_rc.avl"
    plane.clean.export_AVL(files["avl"], include_fuselages=False)

    for name in {cfg.wing.airfoil, cfg.stabilator.airfoil, cfg.fin.airfoil}:
        af = asb.Airfoil(name).repanel(n_points_per_side=100)
        path = out / f"{name}.dat"
        af.write_dat(path)
        files[f"airfoil_{name}"] = path

    return files
