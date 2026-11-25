"""
AeroSandbox model of the conceptual design point + pictures + a first aero
cross-check.

    .venv/bin/python preliminary-design/visualize.py [design_point.json] [out_dir]   (from design/)

Reads  ../conceptual-design/outputs/design_point.json  (geometry + layout)
Writes outputs/three_view.png, outputs/isometric.png, outputs/airplane.html
(interactive 3-D, open in a browser), and prints an AeroBuildup vs.
conceptual-drag comparison at the loiter point.
"""
import json
import os
import sys

import numpy as np

import aerosandbox as asb

HERE = os.path.dirname(os.path.abspath(__file__))
DESIGN_POINT = os.path.join(HERE, "..", "conceptual-design", "outputs", "design_point.json")
OUT = os.path.join(HERE, "outputs")

COLORS = dict(wing="#3b6ea5", tail="#4f8fc0", fuselage="#d9d9d9", nacelle="#8c8c8c",
              prop="#222222", gear="#555555")


def _trapezoid(s, x_qc, z, af, twist=0.0, vertical=False, mid_eta=None, mid_surfaces=None,
               root_surfaces=None):
    """Unswept quarter-chord trapezoidal surface. Wing/h-tail: s['b'] is the full
    span (symmetric half built). Fin: s['b'] is its height."""
    half = s["b"] if vertical else s["b"] / 2
    etas = [0.0] + ([mid_eta] if mid_eta else []) + [1.0]
    xsecs = []
    for eta in etas:
        c = s["c_root"] * (1 - (1 - s["taper"]) * eta)
        pos = eta * half
        xyz = [x_qc - 0.25 * c, 0.0 if vertical else pos, z + (pos if vertical else 0.0)]
        cs = None
        if eta == 0.0 and root_surfaces:
            cs = root_surfaces
        if mid_eta and eta == mid_eta and mid_surfaces:
            cs = mid_surfaces
        xsecs.append(asb.WingXSec(xyz_le=xyz, chord=c, twist=twist, airfoil=af,
                                  control_surfaces=cs or []))
    return xsecs


def _tube(name, p0, p1, radius, color):
    p0, p1 = np.array(p0, float), np.array(p1, float)
    n = (p1 - p0) / np.linalg.norm(p1 - p0)
    return asb.Fuselage(name=name, color=color, xsecs=[
        asb.FuselageXSec(xyz_c=p0, xyz_normal=n, radius=radius),
        asb.FuselageXSec(xyz_c=p1, xyz_normal=n, radius=radius)])


def _wing_xsecs(j):
    """Main wing cross-sections at every control-surface boundary, with washout
    and the root->tip airfoil loft applied per section. A control surface on
    xsec i applies to the panel between xsec i and i+1 (AeroSandbox)."""
    g, L, sel = j["geometry"]["wing"], j["layout"]["wing"], j["selection"]
    ctl = j.get("controls") or {}
    semi = g["b"] / 2
    root = asb.Airfoil(sel["wing_airfoil"])
    tip = asb.Airfoil(sel.get("wing_tip_airfoil") or sel["wing_airfoil"])
    washout = L.get("washout_deg", 0.0)
    panels = []                                   # (y0, y1, ControlSurface or None)
    ail, flp = ctl.get("ailerons"), ctl.get("flaps")
    if ail:
        name = "flaperon" if ail.get("is_flaperon") else "aileron"
        panels.append((ail["y_inboard_m"], ail["y_outboard_m"],
                       asb.ControlSurface(name=name, symmetric=False, hinge_point=1 - ail["chord_fraction"])))
    if flp and not (ail and ail.get("is_flaperon")):
        panels.append((flp["y_inboard_m"], flp["y_outboard_m"],
                       asb.ControlSurface(name="flap", symmetric=True, hinge_point=1 - flp["chord_fraction"])))
    stations = sorted({0.0, semi, *[p[0] for p in panels], *[p[1] for p in panels]})
    xsecs = []
    for y in stations:
        eta = y / semi
        c = g["c_root"] * (1 - (1 - g["taper"]) * eta)
        af = root if tip.name == root.name else root.blend_with_another_airfoil(tip, blend_fraction=eta)
        cs = [p[2] for p in panels if abs(p[0] - y) < 1e-9]
        z_dih = y * np.tan(np.radians(L.get("dihedral_deg", 0.0)))
        xsecs.append(asb.WingXSec(xyz_le=[L["x_quarter_chord"] - 0.25 * c, y, L["z_root_chord_line"] + z_dih],
                                  chord=c, twist=L["incidence_deg"] - washout * eta, airfoil=af,
                                  control_surfaces=cs))
    return xsecs


def build_airplane(j, with_gear=True, with_props=True):
    g, L, sel = j["geometry"], j["layout"], j["selection"]
    tail_af = asb.Airfoil(sel["tail_airfoil"])
    x_qc = L["wing"]["x_quarter_chord"]
    ctl = j.get("controls") or {}

    # ---- lifting surfaces (control surfaces placed where the sizing put them)
    wing = asb.Wing(name="Main wing", symmetric=True, color=COLORS["wing"], xsecs=_wing_xsecs(j))
    el_cf = (ctl.get("elevator") or {}).get("chord_fraction", 0.3)
    ru_cf = (ctl.get("rudder") or {}).get("chord_fraction", 0.4)
    htail = asb.Wing(name="Horizontal stabilizer", symmetric=True, color=COLORS["tail"],
                     xsecs=_trapezoid(g["htail"], L["htail"]["x_quarter_chord"], L["htail"]["z"], tail_af,
                                      twist=L["htail"].get("incidence_deg", 0.0),
                                      root_surfaces=[asb.ControlSurface(name="elevator", symmetric=True,
                                                                        hinge_point=1 - el_cf)]))
    vtail = asb.Wing(name="Vertical stabilizer", symmetric=False, color=COLORS["tail"],
                     xsecs=_trapezoid(g["vtail"], L["vtail"]["x_quarter_chord"], L["vtail"]["z_root"], tail_af,
                                      vertical=True,
                                      root_surfaces=[asb.ControlSurface(name="rudder", symmetric=True,
                                                                        hinge_point=1 - ru_cf)]))

    # ---- fuselage (rounded-box sections sized by the payload bay)
    F = L["fuselage"]
    nose, bay, Lf, W, H = F["nose_length"], F["bay_length"], F["length"], F["width"], F["height"]
    x_bay_end = nose + bay
    stations = [(0.0, 0.20, 0.20), (0.3 * nose, 0.70, 0.72), (0.7 * nose, 0.95, 0.96), (nose, 1, 1),
                (x_bay_end, 1, 1), (x_bay_end + 0.35 * (Lf - x_bay_end), 0.62, 0.70),
                (Lf, F["tail_end_width"] / W, F["tail_end_height"] / H)]
    fus = asb.Fuselage(name="Fuselage", color=COLORS["fuselage"], xsecs=[
        asb.FuselageXSec(xyz_c=[x, 0, 0], width=W * kw, height=H * kh, shape=2.6)
        for x, kw, kh in stations])

    # ---- nacelles + props
    N = L["nacelles"]
    fuselages = [fus]
    propulsors = []
    for side in (+1, -1):
        r, xf, xr = N["diameter"] / 2, N["x_front"], N["x_rear"]
        ln = xr - xf
        fuselages.append(asb.Fuselage(name=f"Nacelle {'R' if side > 0 else 'L'}", color=COLORS["nacelle"], xsecs=[
            asb.FuselageXSec(xyz_c=[x, side * N["y"], N["z_axis"]], radius=r * k)
            for x, k in ((xf, 0.45), (xf + 0.12 * ln, 0.9), (xf + 0.3 * ln, 1.0),
                         (xr - 0.3 * ln, 1.0), (xr, 0.35))]))
        if with_props:
            Pp = L["props"]
            propulsors.append(asb.Propulsor(name=f"Prop {'R' if side > 0 else 'L'}",
                                            xyz_c=[Pp["x"], side * Pp["y"], Pp["z"]],
                                            xyz_normal=[-1, 0, 0], radius=Pp["diameter"] / 2,
                                            color=COLORS["prop"]))

    # ---- landing gear (drawing only - AeroBuildup has no gear model)
    if with_gear:
        G = L["gear"]
        zg = G["z_ground"]
        for side in (+1, -1):
            r, w = G["main_wheel_d"] / 2, G["main_wheel_w"]
            hub = [G["x_main"], side * G["y_main"], zg + r]
            fuselages.append(asb.Fuselage(name="Main wheel", color=COLORS["gear"], xsecs=[
                asb.FuselageXSec(xyz_c=[hub[0], hub[1] - w / 2, hub[2]], xyz_normal=[0, 1, 0], radius=r),
                asb.FuselageXSec(xyz_c=[hub[0], hub[1] + w / 2, hub[2]], xyz_normal=[0, 1, 0], radius=r)]))
            fuselages.append(_tube("Main gear leg", [G["x_main"], side * W * 0.3, -H / 2],
                                   [hub[0], hub[1] - side * w / 2, hub[2]], 0.003, COLORS["gear"]))
        rn, wn = G["nose_wheel_d"] / 2, G["nose_wheel_w"]
        hub = [G["x_nose"], 0.0, zg - G.get("nose_gear_extra_length", 0.0) + rn]
        fuselages.append(asb.Fuselage(name="Nose wheel", color=COLORS["gear"], xsecs=[
            asb.FuselageXSec(xyz_c=[hub[0], -wn / 2, hub[2]], xyz_normal=[0, 1, 0], radius=rn),
            asb.FuselageXSec(xyz_c=[hub[0], +wn / 2, hub[2]], xyz_normal=[0, 1, 0], radius=rn)]))
        fuselages.append(_tube("Nose gear strut", [G["x_nose"], 0, -H / 2], [G["x_nose"], 0, hub[2] + rn * 0.6],
                               0.002, COLORS["gear"]))

    return asb.Airplane(name=f"MVHS twin ({j['takeoff_mass_kg']:.3f} kg)",
                        xyz_ref=[x_qc, 0, 0], wings=[wing, htail, vtail],
                        fuselages=fuselages, propulsors=propulsors)


def crosscheck(j):
    """AeroBuildup (same NeuralFoil section data + its own fuselage/induced models)
    vs. the conceptual build-up at the loiter point. Gear and leakage are
    removed from the conceptual number because AeroBuildup does not model them."""
    lo = j["aero"]["loiter"]
    parts = j["aero"]["loiter_drag_breakdown"]
    cd_conc = lo["CD"] - parts["gear"] - parts["leakage"]
    elev = next(p["value"] for p in j["inputs"] if p["name"] == "field_elevation_m")
    dT = next(p["value"] for p in j["inputs"] if p["name"] == "isa_temp_offset_K")
    plane = build_airplane(j, with_gear=False, with_props=False)
    alphas = np.linspace(-4, 10, 29)
    r = asb.AeroBuildup(plane, op_point=asb.OperatingPoint(
        atmosphere=asb.Atmosphere(altitude=elev, temperature_deviation=dT),
        velocity=lo["V"] * np.ones_like(alphas), alpha=alphas)).run()
    CL, CD = np.ravel(r["CL"]), np.ravel(r["CD"])
    cd_ab = float(np.interp(lo["CL"], CL, CD))
    a_req = float(np.interp(lo["CL"], CL, alphas))
    diff = (cd_ab - cd_conc) / cd_conc
    print("\n AERO CROSS-CHECK at loiter  (gear + leakage excluded)")
    print(f"  CL {lo['CL']:.3f} at V {lo['V']:.2f} m/s -> AeroBuildup body alpha {a_req:+.2f} deg "
          f"(conceptual assumed 0 = fuselage level)")
    print(f"  CD conceptual {cd_conc:.4f}   CD AeroBuildup {cd_ab:.4f}   difference {diff * 100:+.1f}%"
          + ("   <-- investigate (>20%)" if abs(diff) > 0.2 else "   (within 20%)"))
    return dict(cd_conceptual=cd_conc, cd_aerobuildup=cd_ab, alpha_deg=a_req, diff=diff)


def main(argv):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = argv[1] if len(argv) > 1 else OUT
    os.makedirs(out, exist_ok=True)
    with open(argv[0] if argv else DESIGN_POINT) as f:
        j = json.load(f)
    if "layout" not in j:
        raise SystemExit("design_point.json has no 'layout' block - run "
                         "conceptual-design/conceptual.py --evaluate first")
    plane = build_airplane(j)
    title = (f"{plane.name}: span {j['geometry']['wing']['b']:.2f} m, "
             f"{j['selection']['wing_airfoil'].upper()} wing, 2 x {j['selection']['motor']} + "
             f"{j['selection']['prop']}")

    plane.draw_three_view(show=False)
    fig = plt.gcf()
    fig.suptitle(title, fontsize=10)
    fig.savefig(os.path.join(out, "three_view.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)

    plane.draw(backend="matplotlib", show=False, use_preset_view_angle="left_isometric",
               set_axis_visibility=False)
    fig = plt.gcf()
    plt.gca().set_box_aspect(None, zoom=1.6)
    fig.set_size_inches(9, 7)
    fig.suptitle(title, fontsize=10)
    fig.savefig(os.path.join(out, "isometric.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)

    pfig = plane.draw(backend="plotly", show=False)
    pfig.update_layout(title=title)
    pfig.write_html(os.path.join(out, "airplane.html"), include_plotlyjs="cdn")

    crosscheck(j)
    print(f"\n wrote {os.path.relpath(out)}/three_view.png, isometric.png, airplane.html")


if __name__ == "__main__":
    main(sys.argv[1:])
