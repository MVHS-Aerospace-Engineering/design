"""Mass properties from the layout's component build-up."""

from __future__ import annotations

import aerosandbox as asb

from .layout import Layout


def mass_properties(lay: Layout) -> asb.MassProperties:
    total = asb.MassProperties(mass=0.0)
    for mass, x, y, z in lay.mass_items.values():
        total = total + asb.MassProperties(mass=mass, x_cg=x, y_cg=y, z_cg=z)
    return total


def table(lay: Layout) -> list[tuple[str, float, float, float, float]]:
    return sorted(((name, *vals) for name, vals in lay.mass_items.items()), key=lambda r: -r[1])
