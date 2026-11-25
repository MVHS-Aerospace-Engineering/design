"""
Parts list (bill of materials) for the conceptual design point.

    .venv/bin/python conceptual-design/print_parts.py            (prints)
    .venv/bin/python conceptual-design/print_parts.py -o out.txt  (also saves)

Reads outputs/design_point.json (written by conceptual.py) and the part
catalogs in data/. conceptual.py also writes outputs/parts_list.txt and
outputs/parts_list.csv automatically.

Every quantity and size comes from the design point (spar, gear strap, nose
wire, wire lengths, laminate area, printed-part masses) or the catalogs; the
masses add up to the design's takeoff mass (checked at the bottom). No prices:
the catalogs do not carry them.
"""
import csv
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DEFAULT = os.path.join(HERE, "outputs", "design_point.json")


def _inputs(j):
    return {p["name"]: p["value"] for p in j.get("inputs", [])}


def build(j):
    """List of rows: dict(category, item, qty, spec, unit_g, total_g, source)."""
    from catalog import brakes, components, materials, wheels
    P = _inputs(j)
    pa, L, g, mb = j["parts"], j["layout"], j["geometry"], j["mass_breakdown_kg"]
    sel = j["selection"]
    mat = materials()
    rows = []

    def add(cat, item, qty, spec, unit_kg, source, total_kg=None):
        tot = unit_kg * qty if total_kg is None else total_kg
        rows.append(dict(category=cat, item=item, qty=qty, spec=spec, unit_g=unit_kg * 1000,
                         total_g=tot * 1000, source=source))

    n = int(P.get("n_motors", 2))
    # ---------------------------------------------------------------- propulsion
    m = pa["motor"]
    add("propulsion", m["name"], n, f"{m['kv_rpm_per_V']:.0f} Kv, {m['imax_A']:.0f} A max, "
        f"{m['od_m'] * 1000:.0f} mm can", m["mass_kg"], f"DATASHEET: {m['source']}")
    pr = pa["prop"]
    add("propulsion", pr["name"], n, f"{pr['diameter_in']:g} x {pr['pitch_in']:g} in (APC E)", pr["mass_kg_estimate"], "ESTIMATE (mass scaled from a 10 in APC E; WEIGH)")
    add("propulsion", "prop adapter + spinner", n, "to fit the motor shaft", P["spinner_mass_kg"], "ESTIMATE")
    e = pa.get("esc")
    if e:
        add("propulsion", e["name"], n, f"{e['i_cont_A']:.0f} A continuous", e["mass_kg"], "DATASHEET")
    b = pa["battery"]
    add("propulsion", b["name"], 1, f"{b['cells_series']}S{b['parallel']}P {b['chemistry']}, "
        f"{b['energy_Wh']:.1f} Wh, {' x '.join(f'{v * 1000:.0f}' for v in b['dims_m'])} mm",
        b["mass_kg"], "DATASHEET")
    y_eng = g["motor_station_m"]
    wire_len = 2 * n * (y_eng + 0.10)
    add("propulsion", "power wire 16 AWG silicone", 1, f"{wire_len:.2f} m total (2 conductors per motor, "
        f"fuselage to nacelle + 0.10 m)", P["power_wire_kg_m"] * wire_len, "ESTIMATE (mass per metre)",
        total_kg=P["power_wire_kg_m"] * wire_len)
    rows[-1]["qty"] = 1

    # ---------------------------------------------------------------- avionics, servos, hardware
    for c in components():
        add("avionics + servos", c.name, c.qty,
            f"{c.dims_m[0] * 1000:.0f} x {c.dims_m[1] * 1000:.0f} x {c.dims_m[2] * 1000:.0f} mm"
            if c.dims_m[0] > 0 else "", c.mass_kg, f"{c.tag}: {c.source}")
    if mb.get("flap_servos", 0) > 0:
        add("avionics + servos", "flap servo + horn + hinges", 2, "", P["flap_servo_mass_kg"], "ESTIMATE")

    # ---------------------------------------------------------------- landing gear
    G = L["gear"]
    wh = wheels()
    mw, nw = wh[G["main_wheel_model"]], wh[G["nose_wheel_model"]]
    if mw.name == nw.name:
        add("landing gear", mw.name, 3, f"{mw.d * 1000:.1f} x {mw.w * 1000:.1f} mm, {mw.axle_mm:g} mm axle",
            mw.mass_kg, f"{mw.tag}: {mw.source}")
    else:
        add("landing gear", mw.name + " (main)", 2, f"{mw.d * 1000:.1f} x {mw.w * 1000:.1f} mm", mw.mass_kg,
            f"{mw.tag}: {mw.source}")
        add("landing gear", nw.name + " (nose)", 1, f"{nw.d * 1000:.1f} x {nw.w * 1000:.1f} mm", nw.mass_kg,
            f"{nw.tag}: {nw.source}")
    if G.get("brake_model"):
        br = brakes()[G["brake_model"]]
        add("landing gear", br.name, 2, f"fits wheels {br.d_min * 1000:.0f}-{br.d_max * 1000:.0f} mm",
            br.mass_per_wheel_kg, f"{br.tag}: {br.source}")
        add("landing gear", f"brake actuator: {br.actuator}", 1, "drives both brake cables",
            br.actuator_mass_kg, f"{br.tag}: same 12 g servo class as the other servos")
        add("landing gear", "brake cable / pushrod", 1, "", br.cable_mass_kg, "ESTIMATE")
    gmat = G.get("main_gear_material", "aluminum_6061_t6")
    gm = mat[gmat]
    strap_len = 2 * G["main_leg_length"] + g["fuselage_width_m"]
    t, w = G["strap_thickness"], G["strap_width"]
    if gmat == "petg_printed":
        add("printed parts (PETG)", "main landing gear (one piece, legs at 45 deg)", 1,
            f"{t * 1000:.2f} mm thick x {w * 1000:.0f} mm wide x {strap_len * 1000:.0f} mm developed length; "
            f"print FLAT so the layers run along the legs", gm["density_kg_m3"] * t * w * strap_len,
            "DATASHEET (PETG strength/density); thickness sized from the landing load")
    else:
        add("landing gear", "main gear strap, 6061-T6 aluminum", 1,
            f"{t * 1000:.2f} mm thick (minimum - use the next stock size up) x {w * 1000:.0f} mm x "
            f"{strap_len * 1000:.0f} mm, bent to 45 deg legs", gm["density_kg_m3"] * t * w * strap_len,
            "DATASHEET material; thickness sized from the landing load")
    if G.get("nose_gear_type") == "printed_petg":
        pg = mat["petg_printed"]
        d = G["nose_leg_d"]
        fw, ft = P["printed_fork_section_m"]
        r_w, w_w = nw.d / 2, nw.w
        leg = pg["density_kg_m3"] * math.pi * d ** 2 / 4 * (G["nose_strut_length"] + 0.02)
        fork = pg["density_kg_m3"] * fw * ft * (2 * (r_w + 0.010) + (w_w + 2 * (ft + 0.002)))
        add("printed parts (PETG)", "nose leg + wheel fork (one piece)", 1,
            f"{d * 1000:.1f} mm round leg x {(G['nose_strut_length'] + 0.02) * 1000:.0f} mm, fork "
            f"{fw * 1000:.0f} x {ft * 1000:.0f} mm for the {nw.d * 1000:.0f} mm wheel; print lying flat",
            leg + fork, "DATASHEET (PETG); leg sized from the nose load, min diameter is a design choice")
        add("printed parts (PETG)", "nose gear mounting/bearing blocks + steering arm", 1, "",
            P["printed_nose_mount_mass_kg"], "ESTIMATE")
        add("landing gear", "nose gear hardware (axle bolt, 4 x M3 bolts + nuts)", 1, "",
            P["nose_gear_hardware_mass_kg"], "ESTIMATE")
    elif G.get("nose_gear_model"):
        from catalog import nose_gears
        ng = {n.name: n for n in nose_gears()}[G["nose_gear_model"]]
        add("landing gear", ng.name, 1, f"{ng.wire_d * 1000:.2f} mm (5/32 in) wire, {ng.leg_length * 1000:.0f} mm "
            f"leg; nylon blocks, steering arm, hardware", ng.mass_kg, f"{ng.tag}: {ng.source}")
    else:
        mwire = mat["music_wire"]
        d = G["nose_wire_d"]
        wl = G["nose_strut_length"] + 0.04
        add("landing gear", "nose strut, music wire (ASTM A228)", 1, f"{d * 1000:.1f} mm dia x {wl * 1000:.0f} mm",
            mwire["density_kg_m3"] * math.pi * d ** 2 / 4 * wl, "ESTIMATE material; diameter sized from the load")
        add("landing gear", "nose gear steering block + collar", 1, "", P["nose_steer_mass_kg"], "ESTIMATE")

    # ---------------------------------------------------------------- structure (materials)
    sp = L["spar"]
    cf = mat["cfrp_tube"]
    if sp.get("tube_model"):
        from catalog import carbon_tubes
        tb = {x.name: x for x in carbon_tubes()}[sp["tube_model"]]
        n_t = int(sp.get("n_tubes") or 1)
        add("off-the-shelf structure", tb.name, n_t, f"cut to {sp['length'] * 1000:.0f} mm (main spar)"
            + ("; join the pieces with a sleeve" if n_t > 1 else ""),
            tb.kg_per_m * sp["length"] / n_t, f"{tb.tag}: {tb.source}")
    else:
        sp_mass = cf["density_kg_m3"] * math.pi * (sp["od"] ** 2 - max(sp["od"] - 2 * sp["wall"], 0) ** 2) / 4 * sp["length"]
        add("structure", "carbon tube (main spar)", 1, f"{sp['od'] * 1000:.1f} mm OD x {sp['wall'] * 1000:.2f} mm "
            f"wall x {sp['length'] * 1000:.0f} mm", sp_mass, "ESTIMATE density; size from bending at ultimate load")
    S_wet = g["wing"]["S_wet"]
    if sel["wing_construction"] == "foam_glass":
        lam = mat["glass_laminate_25gsm"]["areal_kg_m2"] * S_wet
        add("structure", "wing laminate: 25 g/m^2 glass cloth + epoxy", 1,
            f"{S_wet:.3f} m^2 of cloth (one layer per side, before waste), ~1:1 epoxy by weight",
            lam, "ESTIMATE (areal mass)")
        add("structure", "wing core: EPS foam (hot-wire cut)", 1,
            f"{sel['wing_airfoil']} sections, {g['wing']['b'] * 1000:.0f} mm span",
            mb["wing_shell"] - lam, "ESTIMATE (foam density)")
    else:
        add("printed parts (LW-PLA filament mass)", "wing (both halves)", 1,
            f"{P['wing_skin_t_m'] * 1000:.2f} mm single-wall skin + internal web, {sel['wing_airfoil']}, "
            f"{g['wing']['b'] * 1000:.0f} mm span", mb["wing_shell"], "ESTIMATE (LW-PLA density; WEIGH)")
    motor_mounts = n * P["motor_mount_mass_kg"]
    for key, name in (("fuselage", "fuselage shell + frames"), ("htail", "horizontal tail"),
                      ("vtail", "fin")):
        add("printed parts (LW-PLA filament mass)", name, 1, "", mb[key], "ESTIMATE (LW-PLA density; WEIGH)")
    add("printed parts (LW-PLA filament mass)", "nacelles", n, "", (mb["nacelles"] - motor_mounts) / n,
        "ESTIMATE")
    add("printed parts (LW-PLA filament mass)", "motor mounts", n, "",
        P["motor_mount_mass_kg"], "ESTIMATE")

    # ---------------------------------------------------------------- allowance
    add("allowance", "glue, paint, connectors, build variation", 1,
        f"{P['weight_growth_allowance'] * 100:.0f}% of everything except the battery",
        mb["growth_allowance"], "DESIGN CHOICE (weight_growth_allowance)")
    return rows


def render(j):
    rows = build(j)
    order = list(dict.fromkeys(r["category"] for r in rows))          # group by category, first-seen order
    rows = sorted(rows, key=lambda r: order.index(r["category"]))
    W = []
    w = W.append
    w("PARTS LIST (bill of materials)")
    w(f"Source: conceptual-design/outputs/design_point.json   (generated {j.get('generated', '?')})")
    w(f"Design takeoff mass {j['takeoff_mass_kg']:.3f} kg   {j['selection']['motor']} + {j['selection']['prop']}   "
      f"{j['selection']['battery']}")
    w("Masses: datasheet where tagged DATASHEET, otherwise estimates to WEIGH. No prices (catalogs carry none).")
    w("Sizes of stock (spar, strap, wire) are the computed minimums; dimensions for CAD: outputs/dimensions.txt")
    cat = None
    for r in rows:
        if r["category"] != cat:
            cat = r["category"]
            sub = sum(x["total_g"] for x in rows if x["category"] == cat)
            w("")
            w("=" * 100)
            w(f" {cat.upper()}   ({sub:.1f} g)")
            w("=" * 100)
            w(f"  {'qty':>3s}  {'item':50s} {'each g':>7s} {'total g':>8s}  spec")
        w(f"  {r['qty']:3d}  {r['item'][:50]:50s} {r['unit_g']:7.1f} {r['total_g']:8.1f}  {r['spec']}")
    tot = sum(r["total_g"] for r in rows)
    w("")
    w("=" * 100)
    w(f"  TOTAL {tot:.1f} g   (design takeoff mass {j['takeoff_mass_kg'] * 1000:.1f} g, "
      f"difference {tot - j['takeoff_mass_kg'] * 1000:+.1f} g)")
    w("")
    w(" Sources (full text in the CSV):")
    seen = []
    for r in rows:
        if r["source"] not in seen:
            seen.append(r["source"])
    for s in seen:
        w(f"  - {s}")
    return "\n".join(W)


def write(json_path=DEFAULT, out_path=None, csv_path=None):
    with open(json_path) as f:
        j = json.load(f)
    text = render(j)
    if out_path:
        with open(out_path, "w") as f:
            f.write(text + "\n")
    if csv_path:
        with open(csv_path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=["category", "item", "qty", "spec", "unit_g", "total_g", "source"])
            wr.writeheader()
            for r in build(j):
                wr.writerow({**r, "unit_g": f"{r['unit_g']:.2f}", "total_g": f"{r['total_g']:.2f}"})
    return text


if __name__ == "__main__":
    args = sys.argv[1:]
    out = args[args.index("-o") + 1] if "-o" in args else None
    print(write(DEFAULT, out, os.path.splitext(out)[0] + ".csv" if out else None))
