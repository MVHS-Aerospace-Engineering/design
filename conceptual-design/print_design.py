"""
Print EVERY final sizing variable of the conceptual design point, followed by
every input ratio / coefficient / value used to calculate them (with source).

    .venv/bin/python conceptual-design/print_design.py            (prints)
    .venv/bin/python conceptual-design/print_design.py -o out.txt  (also saves)

Reads outputs/design_point.json (written by conceptual.py). conceptual.py also
saves this printout automatically as outputs/design_parameters.txt.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, "outputs", "design_point.json")

SURFACE_UNITS = {"S": "m^2", "b": "m", "taper": "-", "c_root": "m", "c_tip": "m", "mac": "m",
                 "t_c": "-", "S_wet": "m^2"}
SURFACE_NAMES = {"S": "area", "b": "span (fin: height)", "taper": "taper ratio", "c_root": "root chord",
                 "c_tip": "tip chord", "mac": "mean aerodynamic chord", "t_c": "thickness/chord",
                 "S_wet": "wetted area"}


def _fmt(v):
    if isinstance(v, float):
        if v == 0:
            return "0"
        return f"{v:.6g}"
    if isinstance(v, list):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return str(v)


def render(j):
    L = []
    w = L.append

    def head(t):
        w("")
        w("=" * 86)
        w(f" {t}")
        w("=" * 86)

    def kv(k, v, unit="", note=""):
        w(f"  {k:44s} {_fmt(v):>16s} {unit:8s} {note}")

    def block(title, d, skip=()):
        w(f" -- {title}")
        for k, v in d.items():
            if k in skip or isinstance(v, dict):
                continue
            kv(k, v)

    w(f"CONCEPTUAL DESIGN PARAMETERS   (generated {j.get('generated', '?')})")
    w("Source: conceptual-design/outputs/design_point.json")

    # 1 ---------------------------------------------------------------- summary
    head("1. SUMMARY")
    kv("takeoff mass", j["takeoff_mass_kg"], "kg")
    kv("wing loading W/S", j["wing_loading_N_m2"], "N/m^2")
    kv("installed shaft power / weight (capped)", j["power_loading_W_N"], "W/N")
    if "throttle_cap" in j:
        kv("throttle cap (max ESC duty)", j["throttle_cap"], "-", "set on both motor outputs in the FC")

    # 2 ---------------------------------------------------------------- variables
    head("2. DESIGN VARIABLES")
    w(" -- chosen by the optimizer (SLSQP)")
    bounds = j.get("design_variable_bounds", {})
    for k, v in j["design_variables"].items():
        b = bounds.get(k)
        kv(k, v, "", f"bounds {b}" if b else "")
    w(" -- sized inside the convergence loop")
    if "throttle_cap" in j:
        kv("throttle cap", j["throttle_cap"], "-", "smallest meeting takeoff + climb target x margin")
    g = j["geometry"]
    kv("fin area", g["vtail"]["S"], "m^2", "max(tail-volume, engine-out)")
    kv("battery (snapped to a real pack)", j["selection"]["battery"])
    kv("spar OD / wall", f"{g['spar_od_m'] * 1000:.2f} / {g['spar_wall_m'] * 1000:.2f}", "mm")
    w(" -- discrete choices (searched over the catalogs)")
    for k, v in j["selection"].items():
        kv(k, v)

    # 3 ---------------------------------------------------------------- parts
    if "parts" in j:
        head("3. SELECTED PARTS (datasheet values)")
        for part, d in j["parts"].items():
            if d:
                block(part, d)

    # 4 ---------------------------------------------------------------- geometry
    head("4. GEOMETRY")
    for surf, title in (("wing", "main wing"), ("htail", "horizontal tail"), ("vtail", "vertical tail (fin)")):
        w(f" -- {title}")
        for k, v in g[surf].items():
            kv(f"{SURFACE_NAMES.get(k, k)} ({k})", v, SURFACE_UNITS.get(k, ""))
        if surf == "wing":
            kv("aspect ratio", g["wing"]["b"] ** 2 / g["wing"]["S"], "-")
    block("fuselage, nacelles, props, gear, spar",
          {k: v for k, v in g.items() if k not in ("wing", "htail", "vtail")})
    if "layout" in j:
        w(" -- layout (x aft from nose tip, y right, z up; metres)")
        for part, d in j["layout"].items():
            if isinstance(d, dict):
                for k, v in d.items():
                    if k != "note":
                        kv(f"{part}.{k}", v)

    # 4b --------------------------------------------------------------- controls
    if j.get("controls"):
        head("4b. CONTROL SURFACES & STABILITY")
        ct = j["controls"]
        kv("configuration", ct["configuration"])
        for part in ("ailerons", "flaps", "elevator", "rudder", "stability"):
            if ct.get(part):
                block(part, ct[part])
    if j.get("twist_study"):
        head("4c. TWIST TRADE STUDY (each row re-optimized)")
        w(f"  {'tip airfoil':11s} {'washout':>7s} {'mass':>7s} {'span':>6s} {'loiter':>7s} {'landing':>8s} "
          f"{'CLmax':>6s} {'stall@':>7s} {'e':>6s}  status")
        for r in j["twist_study"]:
            if not r.get("ok"):
                w(f"  {r['tip']:11s} {r['washout']:7.1f}  (did not converge)")
                continue
            st = "ok" if r["feasible"] else "misses " + "+".join(r["misses"])
            w(f"  {r['tip']:11s} {r['washout']:7.1f} {r['m']:7.4f} {r['span']:6.3f} {r['loiter_min']:6.1f}m "
              f"{r['landing_m']:7.1f}m {r['CLmax']:6.3f} {r['stall_eta']:6.0%} {r['e']:6.3f}  {st}")

    # 5 ---------------------------------------------------------------- aero
    head("5. AERODYNAMICS")
    a = j["aero"]
    block("coefficients & speeds", a)
    block("loiter point", a["loiter"])
    block("loiter drag breakdown (CD, referenced to wing area)", a["loiter_drag_breakdown"])
    block("best-rate climb point (at throttle cap)", a["climb"])
    block("best glide", a["best_glide"])

    # 6 ---------------------------------------------------------------- performance
    head("6. PERFORMANCE")
    block("field / currents", j["performance"])
    if "engine_out" in j:
        block("engine-out (one motor failed)", j["engine_out"])
    if "thrust_in_flight" in j:
        w(" -- in-flight thrust (thrust AVAILABLE at the throttle cap vs. level-flight drag)")
        w(f"  {'point':40s} {'V':>6s} {'T_avail':>8s} {'T/W':>6s} {'drag':>7s} {'excess':>7s} {'eta_p':>6s} {'rpm':>6s}")
        for r in j["thrust_in_flight"]:
            f = lambda x, s: (s % x) if x is not None else "-"
            w(f"  {r['point']:40s} {r['V_ms']:6.2f} {r['thrust_avail_N']:8.3f} {r['thrust_per_weight']:6.3f} "
              f"{f(r['drag_level_N'], '%7.3f'):>7s} {f(r['excess_N'], '%7.3f'):>7s} "
              f"{f(r['eta_prop'], '%6.3f'):>6s} {r['rpm']:6.0f}")

    # 7 ---------------------------------------------------------------- mission
    head("7. MISSION BATTERY-MASS-FRACTION BUILD-UP")
    block("segments", j["mission_bmf"])

    # 8 ---------------------------------------------------------------- mass
    head("8. MASS BREAKDOWN")
    tot = j["takeoff_mass_kg"]
    for k, v in sorted(j["mass_breakdown_kg"].items(), key=lambda kv_: -kv_[1]):
        kv(k, v, "kg", f"{v / tot * 100:5.1f}%")

    # 9 ---------------------------------------------------------------- margins
    head("9. CONSTRAINT MARGINS  (>= 0 satisfied; ~0 = ACTIVE, i.e. this requirement sizes the design)")
    for k, v in j["constraint_margins"].items():
        kv(k, v, "", "ACTIVE" if abs(v) < 0.02 else "")

    # 10/11 ------------------------------------------------------------ inputs
    inputs = j.get("inputs", [])
    ratios = [p for p in inputs if p["units"] == "-"]
    others = [p for p in inputs if p["units"] != "-"]
    head("10. INPUT RATIOS & COEFFICIENTS USED TO CALCULATE THE ABOVE (dimensionless)")
    for p in ratios:
        kv(p["name"], p["value"], "-", f"[{p['tag']}{', MEASURE' if p['measure'] else ''}] {p['source']}")
    head("11. OTHER INPUTS (dimensional)")
    for p in others:
        kv(p["name"], p["value"], p["units"], f"[{p['tag']}{', MEASURE' if p['measure'] else ''}] {p['source']}")
    w("")
    w("Parts catalogs: conceptual-design/data/*.csv (motors, lipo_packs, liion_cells, escs, components, materials)")
    return "\n".join(L)


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
