"""Point-mass build-up of the airplane's mass properties."""

from __future__ import annotations

import aerosandbox as asb

from .config import PlaneConfig


def mass_properties(cfg: PlaneConfig) -> asb.MassProperties:
    total = asb.MassProperties(mass=0.0)
    for name, (mass, x, y, z) in cfg.mass.items.items():
        if name.endswith(("_L", "_R")):
            y = (-1 if name.endswith("_L") else 1) * cfg.propulsion.motor_y
        total = total + asb.MassProperties(mass=mass, x_cg=x, y_cg=y, z_cg=z)
    return total


def table(cfg: PlaneConfig) -> list[tuple[str, float, float, float, float]]:
    return [(name, *vals) for name, vals in cfg.mass.items.items()]
