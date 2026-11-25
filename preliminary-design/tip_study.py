"""
Wing-tip study: flat tip vs. a tip curved up at the end, with AeroSandbox's
vortex-lattice method (VLM), which - unlike the conceptual lifting line - can
model a non-planar wing.

    .venv/bin/python preliminary-design/tip_study.py [design_point.json] [h_m] [r_m]   (from design/)

Cases (wing alone, same airfoils, twist, dihedral and taper as the design point):
  flat          the design point's wing
  curved_in     the outer r of each panel curves up by h (same projected span)
  curved_out    a tip curving up by h is ADDED outboard (projected span + 2 r)
  flat_longer   the same added material laid flat instead (span + 2 x arc length)

Compared at the loiter speed and weight (same lift): induced drag (VLM),
profile drag of the added surface (the design's wing profile-drag level), the
effect on loiter time (power ~ drag at fixed speed), the change in the
wing's roll-due-to-sideslip Cl_beta (and the dihedral it is worth), and the
added printed mass.
"""
import json
import math
import os
import sys

import numpy as np

import aerosandbox as asb

HERE = os.path.dirname(os.path.abspath(__file__))
DESIGN_POINT = os.path.join(HERE, "..", "conceptual-design", "outputs", "design_point.json")
OUT = os.path.join(HERE, "outputs")
SPAN_RES = 3          # spanwise panels per section pair (finer meshes give slivers on the curved tip)


def tip_arc(h, r, n=3):
    """Quarter-ellipse from flat (0, 0) to (r, h): (dy, dz) points, excluding the start."""
    t = np.linspace(0, math.pi / 2, n + 1)[1:]
    return [(r * math.sin(a), h * (1 - math.cos(a))) for a in t]


def make_wing(j, case, h, r):
    g, L, sel = j["geometry"]["wing"], j["layout"]["wing"], j["selection"]
    semi, cr, lam = g["b"] / 2, g["c_root"], g["taper"]
    root = asb.Airfoil(sel["wing_airfoil"])
    tip = asb.Airfoil(sel.get("wing_tip_airfoil") or sel["wing_airfoil"])
    dih = math.radians(L.get("dihedral_deg", 0.0))
    wash = L.get("washout_deg", 0.0)
    x_qc, z0, inc = L["x_quarter_chord"], L["z_root_chord_line"], L["incidence_deg"]

    def chord(y):                                   # planform taper, continued past the tip if needed
        return cr * (1 - (1 - lam) * min(y, semi) / semi) if y <= semi else cr * lam

    def xsec(y, dz):
        eta = min(y / semi, 1.0)
        c = chord(y)
        af = root if tip.name == root.name else root.blend_with_another_airfoil(tip, blend_fraction=eta)
        # only the washout is applied as section twist: the incidence goes into the flight angle
        # (as section twist it would TOE a near-vertical tip and make it push sideways)
        return asb.WingXSec(xyz_le=[x_qc - 0.25 * c, y, z0 + y * math.tan(dih) + dz], chord=c,
                            twist=-wash * eta, airfoil=af)

    pts = []                                        # (y, extra dz)
    if case == "flat":
        pts = [(0.0, 0.0), (semi, 0.0)]
    elif case == "curved_in":
        y0 = semi - r
        pts = [(0.0, 0.0), (y0, 0.0)] + [(y0 + dy, dz) for dy, dz in tip_arc(h, r)]
    elif case == "curved_out":
        pts = [(0.0, 0.0), (semi, 0.0)] + [(semi + dy, dz) for dy, dz in tip_arc(h, r)]
    elif case == "flat_longer":
        arc = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip([(0, 0)] + tip_arc(h, r), tip_arc(h, r)))
        pts = [(0.0, 0.0), (semi + arc, 0.0)]
    xs = [xsec(y, dz) for y, dz in pts]
    added = 0.0                                     # planform area added vs. flat (both panels, along the surface)
    for a, b in zip(pts[:-1], pts[1:]):
        ds = math.hypot(b[0] - a[0], b[1] - a[1])
        added += 2 * ds * 0.5 * (chord(a[0]) + chord(b[0]))
    return asb.Wing(name=case, symmetric=True, xsecs=xs), added - g["S"]


def vlm(wing, S_ref, b_ref, c_ref, V, alpha, beta=0.0):
    plane = asb.Airplane(name="w", xyz_ref=[0, 0, 0], wings=[wing], s_ref=S_ref, b_ref=b_ref, c_ref=c_ref)
    r = asb.VortexLatticeMethod(plane, asb.OperatingPoint(velocity=V, alpha=alpha, beta=beta),
                                spanwise_resolution=SPAN_RES, chordwise_resolution=8).run()
    return float(r["CL"]), float(r["CD"]), float(r["Cl"])


def study(j, h=0.03, r=0.05):
    g = j["geometry"]["wing"]
    S, b, c = g["S"], g["b"], g["mac"]
    lo = j["aero"]["loiter"]
    V, CL_req = lo["V"], lo["CL"]
    rho = 2 * lo["D"] / (lo["CD"] * S * V ** 2)
    q = 0.5 * rho * V ** 2
    cd_prof = j["aero"]["loiter_drag_breakdown"]["wing_profile"]           # wing profile CD (ref S)
    D_tot, t_lo = lo["D"], j["mission_bmf"]["loiter_max_min"]
    P = {p["name"]: p["value"] for p in j["inputs"]}
    skin_kg_m2 = 540 * P["wing_skin_t_m"] * 2 * (1 + P["wing_internal_ratio"])   # LW-PLA, both surfaces
    lam = g["taper"]
    clb_per_deg = -(j["aero"]["CL_alpha_per_rad"] * math.radians(1) / 6) * (1 + 2 * lam) / (1 + lam)
    rows = []
    for case in ("flat", "curved_in", "curved_out", "flat_longer"):
        wing, dS = make_wing(j, case, h, r)
        a1, a2 = 4.0, 10.0                              # flight angle incl. the wing incidence
        CL1 = vlm(wing, S, b, c, V, a1)[0]
        CL2 = vlm(wing, S, b, c, V, a2)[0]
        a = a1 + (CL_req - CL1) * (a2 - a1) / (CL2 - CL1)                 # angle for the loiter lift
        CL, CDi, _ = vlm(wing, S, b, c, V, a)
        _, _, Cl_b = vlm(wing, S, b, c, V, a, beta=2.0)
        Di = q * S * CDi
        Dp_add = q * cd_prof * dS                                           # same profile-drag level per area
        rows.append(dict(case=case, alpha=a, CL=CL, CDi=CDi, D_induced=Di, dS=dS, D_profile_added=Dp_add,
                         Clb=Cl_b / math.radians(2.0), mass_added=skin_kg_m2 * dS / 2))
    base = rows[0]
    for rw in rows:
        dD = (rw["D_induced"] - base["D_induced"]) + rw["D_profile_added"]
        rw["dD"] = dD
        rw["d_loiter_min"] = -t_lo * dD / D_tot                             # power ~ drag at fixed speed
        rw["dClb"] = rw["Clb"] - base["Clb"]
        rw["dihedral_equiv_deg"] = rw["dClb"] / clb_per_deg
        rw["e"] = CL_req ** 2 / (math.pi * b ** 2 / S * rw["CDi"])
    return rows, dict(V=V, CL=CL_req, D_total=D_tot, loiter_min=t_lo, h=h, r=r)


def report(rows, info):
    print(f"\n WING-TIP STUDY (AeroSandbox VLM, wing alone; tip rises {info['h'] * 1000:.0f} mm over "
          f"{info['r'] * 1000:.0f} mm; loiter {info['V']:.1f} m/s, CL {info['CL']:.3f})")
    print(f"  {'case':12s} {'e(ref b)':>8s} {'D_ind N':>8s} {'+prof N':>8s} {'net dD N':>9s} {'loiter':>8s} "
          f"{'dCl_beta':>9s} {'= dihedral':>10s} {'+mass g':>8s}")
    for rw in rows:
        print(f"  {rw['case']:12s} {rw['e']:8.3f} {rw['D_induced']:8.4f} {rw['D_profile_added']:8.4f} "
              f"{rw['dD']:+9.4f} {rw['d_loiter_min']:+7.2f}m {rw['dClb']:+9.4f} {rw['dihedral_equiv_deg']:+9.2f}d "
              f"{rw['mass_added'] * 1000:+8.1f}")
    print(f"  (loiter drag {info['D_total']:.3f} N, loiter time {info['loiter_min']:.1f} min; negative dD = less drag. "
          f"Mass change not fed back into the sizing.)")


def main(argv):
    path = argv[0] if argv else DESIGN_POINT
    h = float(argv[1]) if len(argv) > 1 else 0.03
    r = float(argv[2]) if len(argv) > 2 else 0.05
    with open(path) as f:
        j = json.load(f)
    rows, info = study(j, h, r)
    report(rows, info)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "tip_study.json"), "w") as f:
        json.dump(dict(info=info, rows=rows), f, indent=2)


if __name__ == "__main__":
    main(sys.argv[1:])
