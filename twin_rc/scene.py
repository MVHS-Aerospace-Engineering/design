"""Export the plane as a render scene for the C++ engine in `render/`.

    python -m twin_rc.scene                       # -> render/scenes/twin_rc.json
    python -m twin_rc.scene -o some/other.json

The scene is plain JSON:

* `parts`: triangle meshes in AeroSandbox geometry axes (x aft, y right, z up, metres).
  Control surfaces are split out as their own parts with a hinge axis, so the
  renderer can deflect them live. Positive deflection about `hinge.axis`
  (right-hand rule) is trailing edge down for horizontal surfaces and trailing
  edge right for the rudder, the same convention the analysis uses.
* `props`: hub position, thrust direction, radius, blade count and spin direction.
  The renderer builds the blades itself so it can spin them.
* `markers`: CG and neutral point.
* `sim`: trim and stability numbers read from `results/results.json`, if it exists.
  The renderer uses them for its "trimmed cruise" pose.

The geometry is generated directly from `twin_rc.config` and is quick to
produce. The aero analysis is not re-run.

A part can also point at a mesh file instead of inline vertices:
`{"name": ..., "file": "cad/wing.stl", "transform": [16 floats, column-major]}`.
That is the hook for swapping in real CAD later (STL or OBJ, exported from
Fusion/Onshape).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import aerosandbox as asb
import numpy as np

from .config import PlaneConfig
from .geometry import fin, fuselage, main_wing, nacelle, stabilators
from .mass import mass_properties

COLORS = {
    "wing": [0.93, 0.93, 0.95],
    "control": [0.95, 0.55, 0.12],
    "stabilator": [0.20, 0.55, 0.95],
    "fin": [0.93, 0.93, 0.95],
    "fuselage": [0.80, 0.12, 0.15],
    "nacelle": [0.25, 0.25, 0.28],
}

N_PER_SIDE = 40  # airfoil points per surface


def _airfoil_loop(af: asb.Airfoil) -> np.ndarray:
    """Closed airfoil loop, TE (upper) -> LE -> TE (lower), 2*N-1 points."""
    return np.asarray(af.repanel(n_points_per_side=N_PER_SIDE).coordinates)


def _split_loop(loop: np.ndarray, hinge: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split an airfoil loop at x/c = hinge.

    Returns (fixed_loop, flap_loop, hinge_point). Both loops are closed polygons
    with the hinge line as one edge.
    """
    n_le = int(np.argmin(loop[:, 0]))
    upper = loop[: n_le + 1][::-1]  # LE -> TE
    lower = loop[n_le:]  # LE -> TE

    def at(side, x):
        return np.array([x, np.interp(x, side[:, 0], side[:, 1])])

    up_h, lo_h = at(upper, hinge), at(lower, hinge)
    up_fwd = upper[upper[:, 0] < hinge]
    lo_fwd = lower[lower[:, 0] < hinge]
    up_aft = upper[upper[:, 0] > hinge]
    lo_aft = lower[lower[:, 0] > hinge]

    # Fixed part: hinge(upper) -> LE along upper (reversed) -> along lower -> hinge(lower).
    fixed = np.vstack([up_h, up_fwd[::-1], lo_fwd[1:], lo_h])
    # Flap: TE(upper) -> hinge(upper) -> hinge(lower) -> TE(lower).
    flap = np.vstack([up_aft[::-1], up_h, lo_h, lo_aft])
    hinge_pt = (up_h + lo_h) / 2
    return fixed, flap, hinge_pt


def _resample(loop: np.ndarray, n: int) -> np.ndarray:
    """Resample a closed-ish polyline to n points by arc length (keeps endpoints)."""
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(loop, axis=0), axis=1))]
    t = np.linspace(0, d[-1], n)
    return np.column_stack([np.interp(t, d, loop[:, 0]), np.interp(t, d, loop[:, 1])])


def _section_points(wing: asb.Wing, i: int, loop2d: np.ndarray) -> np.ndarray:
    xs = wing.xsecs[i]
    xg, _, zg = wing._compute_frame_of_WingXSec(i)
    le = np.asarray(xs.xyz_le, dtype=float)
    return le + xs.chord * (loop2d[:, :1] * np.asarray(xg) + loop2d[:, 1:] * np.asarray(zg))


def _loft(rings: list[np.ndarray], cap_ends=True) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate between consecutive rings (same point count) and cap the ends."""
    n = len(rings[0])
    verts = np.vstack(rings)
    tris = []
    for r in range(len(rings) - 1):
        a, b = r * n, (r + 1) * n
        for k in range(n - 1):
            tris += [[a + k, b + k, a + k + 1], [a + k + 1, b + k, b + k + 1]]
        # close the loop (last point back to first)
        tris += [[a + n - 1, b + n - 1, a], [a, b + n - 1, b]]
    if cap_ends:
        for r, flip in ((0, True), (len(rings) - 1, False)):
            c = len(verts)
            verts = np.vstack([verts, rings[r].mean(axis=0)])
            base = r * n
            for k in range(n):
                t = [c, base + k, base + (k + 1) % n]
                tris.append(t[::-1] if flip else t)
    return verts, np.asarray(tris, dtype=np.int64)


def _orient(tris: np.ndarray, verts: np.ndarray) -> np.ndarray:
    """Make winding consistent with outward normals (centroid test, fine for these convex-ish solids)."""
    c = verts.mean(axis=0)
    out = tris.copy()
    for i, (a, b, d) in enumerate(tris):
        n = np.cross(verts[b] - verts[a], verts[d] - verts[a])
        if np.dot(n, (verts[a] + verts[b] + verts[d]) / 3 - c) < 0:
            out[i] = [a, d, b]
    return out


def _part(name, verts, tris, color, control=None, hinge=None):
    p = {
        "name": name,
        "color": color,
        "vertices": np.round(np.asarray(verts, dtype=float).ravel(), 6).tolist(),
        "indices": np.asarray(tris, dtype=int).ravel().tolist(),
    }
    if control:
        p["control"] = control
        p["hinge"] = hinge
    return p


def _mirror(verts, tris):
    v = np.asarray(verts, dtype=float).copy()
    v[:, 1] *= -1
    return v, np.asarray(tris)[:, ::-1]


def _hinge_axis(p0, p1, horizontal: bool):
    axis = np.asarray(p1) - np.asarray(p0)
    axis /= np.linalg.norm(axis)
    # Orient so positive rotation is TE-down (horizontal) / TE-right (vertical).
    if (horizontal and axis[1] < 0) or (not horizontal and axis[2] < 0):
        axis = -axis
    return {"origin": np.round(p0, 6).tolist(), "axis": np.round(axis, 6).tolist()}


def wing_parts(wing: asb.Wing, base_name: str, color, horizontal=True) -> list[dict]:
    """Loft a wing, splitting out any trailing-edge control surfaces.

    Consecutive sections without a control surface are lofted as one piece. A
    section carrying a control surface becomes a fixed piece plus a flap piece
    with its own hinge.
    """
    loop = _airfoil_loop(wing.xsecs[0].airfoil)
    parts: list[dict] = []
    plain_run: list[int] = []

    def flush_plain():
        if len(plain_run) >= 2:
            rings = [_section_points(wing, i, loop) for i in plain_run]
            v, t = _loft(rings)
            parts.append((f"{base_name}", v, _orient(t, v), None, None))

    for i in range(len(wing.xsecs) - 1):
        cs = wing.xsecs[i].control_surfaces
        if not cs:
            if not plain_run or plain_run[-1] != i:
                plain_run.append(i)
            plain_run.append(i + 1)
            continue
        flush_plain()
        plain_run = []
        surf = cs[0]
        fixed2d, flap2d, hinge2d = _split_loop(loop, surf.hinge_point)
        n_fixed, n_flap = len(fixed2d), len(flap2d)
        fixed2d, flap2d = _resample(fixed2d, n_fixed), _resample(flap2d, n_flap)
        rf = [_section_points(wing, j, fixed2d) for j in (i, i + 1)]
        rc = [_section_points(wing, j, flap2d) for j in (i, i + 1)]
        hp = [_section_points(wing, j, hinge2d[None, :])[0] for j in (i, i + 1)]
        vf, tf = _loft(rf)
        vc, tc = _loft(rc)
        parts.append((base_name, vf, _orient(tf, vf), None, None))
        parts.append((surf.name, vc, _orient(tc, vc), surf, _hinge_axis(hp[0], hp[1], horizontal)))
    flush_plain()

    out = []
    for idx, (name, v, t, surf, hinge) in enumerate(parts):
        side = "R" if wing.symmetric else ""
        ctrl = None
        if surf is not None:
            ctrl = {"id": surf.name, "gain": 1.0}
        out.append(_part(f"{name} {side}{idx}".strip(), v, t,
                         COLORS["control"] if surf else color, ctrl, hinge))
        if wing.symmetric:
            mv, mt = _mirror(v, t)
            mctrl, mh = None, None
            if surf is not None:
                # Symmetric surfaces (flaps) move together; ailerons move opposite.
                mctrl = {"id": surf.name, "gain": 1.0 if surf.symmetric else -1.0}
                o = np.asarray(hinge["origin"]) * [1, -1, 1]
                a = np.asarray(hinge["axis"]) * [-1, 1, -1]  # mirrored + reoriented to keep TE-down positive
                mh = {"origin": np.round(o, 6).tolist(), "axis": np.round(a, 6).tolist()}
            out.append(_part(f"{name} L{idx}", mv, mt,
                             COLORS["control"] if surf else color, mctrl, mh))
    return out


def stabilator_parts(cfg: PlaneConfig) -> list[dict]:
    """Each stabilator half is a single all-moving part, pivoting about its quarter chord."""
    s = cfg.stabilator
    out = []
    for wing, side, cid in zip(stabilators(cfg), ("L", "R"), ("stab_L", "stab_R")):
        loop = _airfoil_loop(wing.xsecs[0].airfoil)
        rings = [_section_points(wing, j, loop) for j in range(len(wing.xsecs))]
        v, t = _loft(rings)
        pivots = [_section_points(wing, j, np.array([[s.pivot_frac, 0.0]]))[0]
                  for j in range(len(wing.xsecs))]
        out.append(_part(f"Stabilator {side}", v, _orient(t, v), COLORS["stabilator"],
                         {"id": cid, "gain": 1.0}, _hinge_axis(pivots[0], pivots[1], True)))
    return out


def body_part(fus: asb.Fuselage, color) -> dict:
    pts, faces = fus.mesh_body(method="tri", tangential_resolution=36)
    pts, faces = np.asarray(pts), np.asarray(faces)
    return _part(fus.name, pts, _orient(faces, pts), color)


def build_scene(cfg: PlaneConfig | None = None, results_path: Path | None = None) -> dict:
    cfg = cfg or PlaneConfig()
    parts = []
    parts += wing_parts(main_wing(cfg), "Main Wing", COLORS["wing"])
    parts += stabilator_parts(cfg)
    parts += wing_parts(fin(cfg), "Fin", COLORS["fin"], horizontal=False)
    parts.append(body_part(fuselage(cfg), COLORS["fuselage"]))
    parts += [body_part(nacelle(cfg, s), COLORS["nacelle"]) for s in (-1, 1)]

    p = cfg.propulsion
    props = [
        {
            "name": f"Prop {'L' if side < 0 else 'R'}",
            "hub": [p.motor_x, side * p.motor_y, p.motor_z],
            "thrust_dir": [-1.0, 0.0, 0.0],
            "radius": p.prop_diameter_in * 0.0254 / 2,
            "blades": p.n_blades,
            "pitch": p.prop_pitch_in * 0.0254,
            # Right-hand rotation sense about thrust_dir. Left prop CW, right CCW
            # seen from behind (config.PropulsionConfig), i.e. +1 / -1.
            "spin": 1 if side < 0 else -1,
        }
        for side in (-1, 1)
    ]

    m = mass_properties(cfg)
    scene = {
        "format": "twin_rc.scene/1",
        "units": "m",
        "axes": "x aft, y right, z up (AeroSandbox geometry axes)",
        "name": "MVHS Twin High-Wing",
        "parts": parts,
        "props": props,
        "markers": [{"name": "CG", "position": [m.x_cg, m.y_cg, m.z_cg], "color": [1.0, 0.85, 0.1]}],
        "controls": {
            "aileron": {"max_deg": 20.0},
            "rudder": {"max_deg": cfg.fin.max_rudder_deg},
            "stab_L": {"max_deg": cfg.stabilator.max_deflection_deg},
            "stab_R": {"max_deg": cfg.stabilator.max_deflection_deg},
        },
    }

    if results_path and results_path.exists():
        r = json.loads(results_path.read_text())
        cruise = r["cruise"]
        x_np = r["vlm_check"]["x_np"]
        scene["markers"].append({"name": "Neutral point (VLM)", "position": [x_np, 0.0, m.z_cg],
                                 "color": [0.2, 0.9, 0.4]})
        scene["sim"] = {
            "source": str(results_path),
            "cruise_speed": cruise["V"],
            "trim_alpha_deg": cruise["alpha"],
            "trim_stab_deg": cruise["stab_deg"],
            "L_over_D": cruise["L_over_D"],
            "static_margin": r["static_margin_vlm"],
            "v_stall": r["v_stall"],
            "vmc": r["vmc"],
            "mass_kg": r["mass_kg"],
        }
    return scene


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default="render/scenes/twin_rc.json")
    ap.add_argument("--results", default="results/results.json")
    args = ap.parse_args(argv)
    scene = build_scene(results_path=Path(args.results))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scene, separators=(",", ":")))
    n_tri = sum(len(p["indices"]) // 3 for p in scene["parts"])
    print(f"Wrote {out}: {len(scene['parts'])} parts, {n_tri} triangles, {len(scene['props'])} props")


if __name__ == "__main__":
    main()
