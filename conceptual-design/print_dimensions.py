"""
CAD hand-off: every dimension of the aircraft, in millimetres and degrees,
grouped by part, for the people modelling it in CAD.

    .venv/bin/python conceptual-design/print_dimensions.py            (prints)
    .venv/bin/python conceptual-design/print_dimensions.py -o out.txt  (also saves)

Reads outputs/design_point.json (written by conceptual.py), which also calls
    print_dimensions.write(json_path, out_path)      -> outputs/dimensions.txt
    print_dimensions.write_airfoils(json_path, dir)  -> outputs/airfoils/<name>.dat
The airfoil files are Selig-format coordinates (x/c, y/c) to import into CAD
and scale to each section's chord.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, "outputs", "design_point.json")
AIRFOIL_DIR = os.path.join(HERE, "outputs", "airfoils")
W = 86


def _mm(v):
    return "-" if v is None else f"{float(v) * 1000:.1f}"


def _deg(v):
    return "-" if v is None else f"{float(v):.2f}"


def _airfoils(sel):
    names = []
    for k in ("wing_airfoil", "wing_tip_airfoil", "tail_airfoil"):
        n = sel.get(k)
        if n and n not in names:
            names.append(n)
    return names


def render(j):
    L = []
    w = L.append
    lay = j["layout"]
    sel = j.get("selection", {})

    def head(t):
        w("")
        w("=" * W)
        w(f" {t}")
        w("=" * W)

    def sub(t):
        w(f" -- {t}")

    def row(label, value, unit="", note=""):
        dots = "." * max(2, 46 - len(label))
        w(f"  {label} {dots} {value:>10s} {unit:4s} {note}".rstrip())

    def mm(label, v, note=""):
        row(label, _mm(v), "mm", note)

    def deg(label, v, note=""):
        row(label, _deg(v), "deg", note)

    def txt(label, v, note=""):
        row(label, "-" if v is None else str(v), "", note)

    fus, wing, spar = lay["fuselage"], lay["wing"], lay.get("spar", {})
    ail, flp = lay.get("ailerons") or {}, lay.get("flaps")
    ht, el = lay["htail"], lay.get("elevator") or {}
    vt, ru = lay["vtail"], lay.get("rudder") or {}
    nac, prop = lay["nacelles"], lay["props"]
    gear, cg = lay["gear"], lay.get("cg") or {}
    bay = lay.get("bay_contents") or {}
    dat = lambda n: f"airfoils/{n}.dat" if n else "-"

    w("AIRCRAFT DIMENSIONS FOR CAD")
    w(f"Source: conceptual-design/outputs/design_point.json   (generated {j.get('generated', '?')})")
    w(f"Takeoff mass {float(j.get('takeoff_mass_kg', 0)):.3f} kg   motor {sel.get('motor', '-')}   "
      f"prop {sel.get('prop', '-')}   battery {sel.get('battery', '-')}")

    # 1 ------------------------------------------------------------------
    head("1. DATUM AND FRAME")
    w("  Origin   : tip of the fuselage nose, on the fuselage centreline")
    w("  Axes     : x aft (toward the tail), y to the right wing, z up")
    w("  z = 0    : fuselage centreline (half way between fuselage top and bottom)")
    w("  Units    : millimetres and degrees. Positions are x/y/z of the point named.")
    w("  DO NOT SCALE - these dimensions govern. Re-generate this file after any design change.")
    w("  Wing and tails are unswept along their quarter-chord (25%) line; chords are")
    w("  measured along x; 'LE' = leading edge, 'TE' = trailing edge.")

    # 2 ------------------------------------------------------------------
    head("2. OVERALL ENVELOPE")
    tail_te = max(float(ht["x_te_root"]), float(vt["x_te_root"]))
    length = max(float(fus["length"]), tail_te)
    fin_top = float(vt.get("z_root", 0)) + float(vt.get("height", vt.get("span", 0)))
    mm("wingspan", wing["span"])
    mm("overall length (nose to furthest tail TE)", length)
    mm("height on the gear (ground to fin tip)", fin_top - float(gear["z_ground"]))
    mm("ground level (z of the ground)", gear["z_ground"])

    # 3 ------------------------------------------------------------------
    head("3. WING")
    sub("planform")
    mm("span (tip to tip)", wing["span"])
    mm("root chord", wing["c_root"])
    mm("tip chord", wing["c_tip"])
    row("taper (tip / root)", f"{float(wing['taper']):.3f}")
    row("area", f"{float(wing['area']) * 1e4:.1f}", "cm^2")
    mm("quarter-chord line x (straight, unswept)", wing["x_quarter_chord"])
    mm("root LE x", wing["x_le_root"])
    mm("root TE x", wing["x_te_root"])
    mm("tip LE x", wing["x_le_tip"])
    mm("tip TE x", wing["x_te_tip"])
    mm("mean aerodynamic chord (MAC)", wing["mac"])
    mm("MAC spanwise station y", wing.get("y_mac"))
    mm("MAC LE x", wing.get("x_le_mac"))
    sub("mounting")
    mm("root chord line z (on top of the fuselage)", wing["z_root_chord_line"])
    deg("incidence (root chord vs fuselage axis)", wing["incidence_deg"], "LE up")
    deg("washout (tip twisted LE-down vs root)", wing["washout_deg"], "linear root->tip")
    deg("dihedral", wing.get("dihedral_deg", 0.0))
    sub("airfoils")
    txt("root airfoil", wing.get("root_airfoil"), dat(wing.get("root_airfoil")))
    txt("tip airfoil", wing.get("tip_airfoil"), dat(wing.get("tip_airfoil")))
    row("root thickness / chord", f"{float(wing.get('t_c_root', wing.get('t_c', 0))):.3f}")
    w("  (sections loft linearly from root to tip; twist about the quarter-chord line)")
    if spar:
        sub("main spar")
        txt("material", spar.get("material", "carbon tube"))
        if spar.get("tube_model"):
            txt("tube (off the shelf)", spar["tube_model"])
        mm("outside diameter", spar["od"])
        mm("wall thickness", spar["wall"])
        mm("length (centred on the fuselage)", spar["length"])
        mm("spar centre x (straight tube)", spar["x"])
        mm("spar centre z at the root", spar.get("z"))
        row("spar position x/c at root", f"{float(spar['xc_root']):.3f}", "", "= max-thickness point")
        row("spar position x/c at tip", f"{float(spar['xc_tip']):.3f}")

    # 4 ------------------------------------------------------------------
    head("4. AILERONS AND FLAPS")
    if ail:
        sub("flaperons" if ail.get("is_flaperon") else "ailerons (each wing)")
        mm("inboard edge y", ail["y_inboard"])
        mm("outboard edge y", ail["y_outboard"])
        row("chord fraction (of local chord)", f"{float(ail['chord_fraction']):.2f}")
        mm("surface chord at inboard edge", ail.get("chord_inboard"))
        mm("surface chord at outboard edge", ail.get("chord_outboard"))
        mm("hinge line x at inboard edge", ail["hinge_x_inboard"])
        mm("hinge line x at outboard edge", ail["hinge_x_outboard"])
        deg("throw (each direction)", ail["throw_deg"])
        if float(ail.get("crow_up_deg") or 0) > 0:
            deg("crow / air-brake (both up, on rollout)", ail["crow_up_deg"])
    if flp:
        sub("flaps (each wing)")
        mm("inboard edge y", flp["y_inboard"])
        mm("outboard edge y", flp["y_outboard"])
        row("chord fraction", f"{float(flp['chord_fraction']):.2f}")
        mm("hinge line x at inboard edge", flp["hinge_x_inboard"])
        mm("hinge line x at outboard edge", flp["hinge_x_outboard"])
        deg("landing deflection (TE down)", flp["landing_deflection_deg"])
    else:
        w("  no separate flaps")

    # 5 ------------------------------------------------------------------
    head("5. HORIZONTAL TAIL AND ELEVATOR")
    mm("span", ht["span"])
    mm("root chord", ht["c_root"])
    mm("tip chord", ht["c_tip"])
    row("area", f"{float(ht['area']) * 1e4:.1f}", "cm^2")
    mm("quarter-chord line x", ht["x_quarter_chord"])
    mm("root LE x", ht["x_le_root"])
    mm("root TE x", ht["x_te_root"])
    mm("tip LE x", ht["x_le_tip"])
    mm("tip TE x", ht["x_te_tip"])
    mm("chord plane z", ht.get("z", 0.0))
    deg("incidence (vs fuselage axis)", ht.get("incidence_deg"), "LE up")
    txt("airfoil", ht.get("airfoil"), dat(ht.get("airfoil")))
    if el:
        sub("elevator")
        row("chord fraction", f"{float(el['chord_fraction']):.2f}")
        row("span fraction (centred)", f"{float(el['span_fraction']):.2f}")
        mm("hinge line x at root", el["hinge_x_root"])
        mm("hinge line x at tip", el["hinge_x_tip"])

    # 6 ------------------------------------------------------------------
    head("6. FIN AND RUDDER")
    h_fin = float(vt.get("height", vt.get("span")))
    z0 = float(vt.get("z_root", 0.0))
    mm("height (root to tip)", h_fin)
    mm("root chord", vt["c_root"])
    mm("tip chord", vt["c_tip"])
    row("area", f"{float(vt['area']) * 1e4:.1f}", "cm^2")
    mm("quarter-chord line x", vt["x_quarter_chord"])
    mm("root z (on the tail cone)", z0)
    mm("tip z", z0 + h_fin)
    mm("root LE x", vt["x_le_root"])
    mm("root TE x", vt["x_te_root"])
    mm("tip LE x", vt["x_le_tip"])
    mm("tip TE x", vt["x_te_tip"])
    txt("airfoil", vt.get("airfoil"), dat(vt.get("airfoil")))
    if ru:
        sub("rudder")
        row("chord fraction", f"{float(ru['chord_fraction']):.2f}")
        row("height fraction (from root)", f"{float(ru['span_fraction']):.2f}")
        mm("rudder top z", z0 + float(ru["span_fraction"]) * h_fin)
        mm("hinge line x at root", ru["hinge_x_root"])
        mm("hinge line x at tip", ru["hinge_x_tip"])

    # 7 ------------------------------------------------------------------
    head("7. FUSELAGE")
    mm("length", fus["length"])
    mm("nose fairing length", fus["nose_length"])
    mm("bay length", fus["bay_length"])
    mm("bay start x", fus.get("bay_x_start"))
    mm("bay end x", fus.get("bay_x_end"))
    mm("outer width (max)", fus["width"])
    mm("outer height (max)", fus["height"])
    mm("inner bay width", fus.get("bay_inner_width"))
    mm("inner bay height", fus.get("bay_inner_height"))
    mm("wall thickness", fus.get("wall_t"))
    mm("tail-end width", fus["tail_end_width"])
    mm("tail-end height", fus["tail_end_height"])
    txt("cross-section", fus.get("section_shape", "rounded box"))
    if fus.get("stations"):
        sub("outer-shape stations (loft through these; centred on z = 0)")
        w(f"  {'x mm':>10s} {'width mm':>10s} {'height mm':>10s}")
        for x, wd, hh in fus["stations"]:
            w(f"  {_mm(x):>10s} {_mm(wd):>10s} {_mm(hh):>10s}")

    # 8 ------------------------------------------------------------------
    head("8. NACELLES, MOTOR MOUNTS AND PROPS")
    sub("nacelles (one under each wing, mirror at -y)")
    mm("centreline y", nac["y"])
    mm("thrust-axis z", nac["z_axis"])
    mm("front x (motor mount face)", nac["x_front"])
    mm("rear x", nac["x_rear"])
    mm("diameter", nac["diameter"])
    mm("length", nac.get("length"))
    txt("motor", sel.get("motor"))
    sub("props (tractor)")
    txt("model", prop.get("model", sel.get("prop")))
    mm("diameter", prop["diameter"])
    mm("prop disc x", prop["x"])
    mm("prop hub y", prop["y"])
    mm("prop hub z", prop["z"])
    mm("prop tip to fuselage side", prop.get("tip_to_fuselage"))
    mm("prop tip to ground", prop.get("tip_to_ground"))

    # 9 ------------------------------------------------------------------
    head("9. LANDING GEAR, WHEELS AND BRAKES")
    sub("main gear (" + ("printed PETG, one piece - print flat" if gear.get("main_gear_material") == "petg_printed"
                         else "aluminium strap") + ")")
    mm("axle x", gear["x_main"])
    mm("wheel centre y (each side)", gear["y_main"])
    mm("track (wheel centre to wheel centre)", 2 * float(gear["y_main"]))
    mm("gear height (fuselage bottom to ground)", gear.get("gear_height"))
    mm("leg length (each)", gear.get("main_leg_length"))
    mm("strap thickness", gear.get("strap_thickness"))
    mm("strap width", gear.get("strap_width"))
    txt("main wheels", gear.get("main_wheel_model"))
    mm("main wheel diameter", gear["main_wheel_d"])
    mm("main wheel width", gear["main_wheel_w"])
    for lbl, k in (("main axle diameter", "main_axle_mm"), ("nose axle diameter", "nose_axle_mm")):
        v = gear.get(k)
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = float("nan")
        row(lbl, "-" if v != v else f"{v:.1f}", "mm", "" if v == v else "check the wheel")
    sub("nose gear (steerable" + (", " + gear["nose_gear_model"] if gear.get("nose_gear_model") else " music wire")
        + ")")
    mm("axle x", gear["x_nose"])
    mm("strut length", gear.get("nose_strut_length"))
    if gear.get("nose_gear_type") == "printed_petg":
        mm("printed leg diameter (PETG, print lying flat)", gear.get("nose_leg_d"))
    else:
        mm("wire diameter", gear.get("nose_wire_d"))
    mm("extra strut length for nose-up stance", gear.get("nose_gear_extra_length"))
    deg("ground stance (nose up)", gear.get("ground_stance_deg"))
    txt("nose wheel", gear.get("nose_wheel_model"))
    mm("nose wheel diameter", gear["nose_wheel_d"])
    mm("nose wheel width", gear["nose_wheel_w"])
    sub("brakes")
    txt("brakes (main wheels)", gear.get("brake_model") or "none")
    txt("actuator", gear.get("brake_actuator") or "-")
    if gear.get("note"):
        w(f"  note: {gear['note']}")

    # 10 -----------------------------------------------------------------
    head("10. INTERNAL LAYOUT AND BALANCE")
    b = bay.get("battery")
    if b:
        sub("battery")
        txt("pack", b.get("name"))
        w(f"  box l x w x h {_mm(b['l'])} x {_mm(b['w'])} x {_mm(b['h'])} mm, {float(b['mass']) * 1000:.0f} g")
        mm("battery tray travel (each way)", cg.get("battery_slide"))
    if bay.get("components"):
        sub("components (l x w x h, mass each)")
        for c in bay["components"]:
            where = "bay" if c.get("in_fuselage") else "outside bay"
            dims = (f"{_mm(c['l'])} x {_mm(c['w'])} x {_mm(c['h'])} mm"
                    if float(c.get("l") or 0) > 0 else "(no box)")
            w(f"  {c['qty']} x {c['name'][:42]:42s} {dims:>26s} {float(c['mass_each']) * 1000:6.1f} g  {where}")
    if bay.get("note"):
        w(f"  note: {bay['note']}")
    sub("CENTRE OF GRAVITY - BALANCE HERE")
    mm("forward CG limit x", cg.get("x_forward_limit"))
    mm("target (nominal) CG x", cg.get("x_nominal"))
    mm("aft CG limit x", cg.get("x_aft_limit"))
    mm("CG band width", cg.get("band"))
    mm("neutral point x (never balance behind this)", cg.get("x_neutral_point"))
    if cg.get("note"):
        w(f"  note: {cg['note']}")

    # 11 -----------------------------------------------------------------
    head("11. CONTROL THROWS AND SERVOS")
    w(f"  {'surface':16s} {'direction':22s} {'throw deg':>9s}  notes")
    rows = []
    if ail:
        nm = "flaperons" if ail.get("is_flaperon") else "ailerons"
        rows.append((nm, "up / down (each)", ail.get("throw_deg"), "differential not assumed"))
        if float(ail.get("crow_up_deg") or 0) > 0:
            rows.append((nm, "both up (crow)", ail["crow_up_deg"], "landing rollout air-brake"))
    if flp:
        rows.append(("flaps", "down", flp.get("landing_deflection_deg"), "landing"))
    if el:
        rows.append(("elevator", "trailing edge up", el.get("throw_up_deg"), "needed"))
        rows.append(("elevator", "servo throw (each way)", el.get("servo_throw_deg"), "design max"))
    if ru:
        rows.append(("rudder", "left / right (needed)", ru.get("throw_needed_deg"), "engine-out / crosswind"))
        rows.append(("rudder", "servo throw (each way)", ru.get("servo_throw_deg"), "design max"))
    for s, d, t, n in rows:
        w(f"  {s:16s} {d:22s} {_deg(t):>9s}  {n}")
    if lay.get("servos"):
        sub("servos")
        for s in lay["servos"]:
            w(f"  {s['count']} x {s['surface']:16s} {s['location']}")
    w("")
    w("Airfoil coordinate files (Selig format, x/c y/c; scale to the local chord):")
    for n in _airfoils(sel):
        w(f"  outputs/{dat(n)}")
    return "\n".join(L)


def write_airfoils(json_path=DEFAULT, out_dir=None):
    """Selig-format .dat for every airfoil the design uses. Returns the paths."""
    sys.path.insert(0, HERE)
    from catalog import airfoil_shape
    with open(json_path) as f:
        j = json.load(f)
    out_dir = out_dir or AIRFOIL_DIR
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for n in _airfoils(j.get("selection", {})):
        p = os.path.join(out_dir, f"{n}.dat")
        with open(p, "w") as f:
            f.write(n + "\n")
            for x, y in airfoil_shape(n)["coords"]:
                f.write(f"{x:.6f} {y:.6f}\n")
        paths.append(p)
    return paths


def write(json_path=DEFAULT, out_path=None):
    with open(json_path) as f:
        text = render(json.load(f))
    if out_path:
        with open(out_path, "w") as f:
            f.write(text + "\n")
    return text


if __name__ == "__main__":
    args = sys.argv[1:]
    out = args[args.index("-o") + 1] if "-o" in args else None
    print(write(DEFAULT, out))
    for p in write_airfoils(DEFAULT):
        print("wrote", os.path.relpath(p))
