"""Run every analysis and write plots, a Markdown summary, JSON results and
XFLR5/AVL exports to `results/`.

    python -m twin_rc.run_all            # default output folder ./results
    python -m twin_rc.run_all -o out_dir
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict
from pathlib import Path

import aerosandbox as asb
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .analysis import TwinPlane  # noqa: E402
from .export import export_all  # noqa: E402
from .layout import check_packaging  # noqa: E402
from .mass import table as mass_table  # noqa: E402

COMPONENT_COLORS = {"battery": "#e4572e", "pi5": "#2e86de", "flight_controller": "#17b978",
                    "bec": "#a0a0a0", "receiver": "#8854d0"}


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def airfoil_polars(plane: TwinPlane, out: Path, V: float) -> dict:
    """NeuralFoil polars at flight Reynolds numbers (the 'Direct Foil Analysis'
    step of an XFLR5 workflow)."""
    cfg, lay = plane.cfg, plane.layout
    mu = float(np.ravel(plane.atmo.dynamic_viscosity())[0])
    alphas = np.linspace(-8, 16, 97)
    cases = [(cfg.wing.airfoil, lay.root_chord, "wing root"), (cfg.wing.airfoil, lay.tip_chord, "wing tip"),
             (cfg.stabilator.airfoil, lay.stab_root_chord, "stabilator")]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    info = {}
    for name, chord, label in cases:
        Re = plane.rho * V * chord / mu
        a = asb.Airfoil(name).get_aero_from_neuralfoil(alpha=alphas, Re=Re)
        CL, CD = np.ravel(a["CL"]), np.ravel(a["CD"])
        tag = f"{name} {label} (Re {Re / 1e3:.0f}k)"
        ax[0].plot(alphas, CL, label=tag), ax[1].plot(CD, CL, label=tag), ax[2].plot(alphas, CL / CD, label=tag)
        i = int(np.argmax(CL))
        info[label] = {"airfoil": name, "Re": Re, "Clmax": CL[i], "alpha_Clmax": alphas[i],
                       "max_L_over_D": float(np.max(CL / CD))}
    for a_, xl, yl in zip(ax, ("alpha [deg]", "Cd", "alpha [deg]"), ("Cl", "Cl", "Cl/Cd")):
        a_.set_xlabel(xl), a_.set_ylabel(yl), a_.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.suptitle(f"Airfoil polars (NeuralFoil) at {V:.0f} m/s")
    _save(fig, out / "airfoil_polars.png")
    return info


def aircraft_polar(plane: TwinPlane, out: Path, V: float) -> None:
    alphas = np.linspace(-6, 18, 49)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for stab in (-15, -8, 0, 5):
        r = plane.alpha_sweep(V, alphas, stab=stab)
        lbl = f"stabilators {stab:+d} deg"
        ax[0].plot(alphas, r["CL"], label=lbl), ax[1].plot(r["CD"], r["CL"], label=lbl)
        ax[2].plot(alphas, r["Cm"], label=lbl)
    for a_, xl, yl in zip(ax, ("alpha [deg]", "CD", "alpha [deg]"), ("CL", "CL", "Cm (about CG)")):
        a_.set_xlabel(xl), a_.set_ylabel(yl), a_.grid(alpha=0.3)
    ax[2].axhline(0, color="k", lw=0.8)
    ax[0].legend(fontsize=8)
    fig.suptitle(f"Whole-aircraft polars (AeroBuildup) at {V:.0f} m/s")
    _save(fig, out / "aircraft_polars.png")


def trim_and_performance(plane: TwinPlane, out: Path, v_stall: float) -> list:
    pts = plane.trim_sweep(np.arange(np.ceil(v_stall * 1.1 * 2) / 2, 26.0, 0.5))
    V = np.array([p.V for p in pts])
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot(V, [p.alpha for p in pts], label="alpha")
    ax[0, 0].plot(V, [p.stab_deg for p in pts], label="stabilators (collective)")
    ax[0, 0].set_ylabel("deg"), ax[0, 0].set_title("Trim")
    ax[0, 1].plot(V, [p.drag for p in pts], label="thrust required (drag)")
    ax[0, 1].plot(V, [p.thrust_avail for p in pts], label="thrust available (2 motors)")
    ax[0, 1].plot(V, [p.thrust_avail / 2 for p in pts], "--", label="one engine out")
    ax[0, 1].set_ylabel("N"), ax[0, 1].set_title("Thrust")
    ax[1, 0].plot(V, [p.endurance_min or np.nan for p in pts])
    ax[1, 0].set_ylabel("min"), ax[1, 0].set_title("Flight time (80 % of battery, incl. Pi 5 + FC)")
    ax[1, 1].plot(V, [p.total_power or np.nan for p in pts], label="total")
    ax[1, 1].axhline(plane.avionics_power, color="gray", ls=":", label="avionics (Pi 5, FC, servos)")
    ax[1, 1].set_ylabel("W from battery"), ax[1, 1].set_title("Electrical power in level flight")
    for a_ in ax.flat:
        a_.axvline(1.3 * v_stall, color="r", lw=0.8, ls=":")
        a_.set_xlabel("airspeed [m/s]"), a_.grid(alpha=0.3)
        if a_.get_legend_handles_labels()[0]:
            a_.legend(fontsize=8)
    _save(fig, out / "trim_performance.png")
    return pts


def engine_out(plane: TwinPlane, out: Path, speeds) -> tuple[list, float | None, list]:
    rows = [plane.engine_out_rudder(V) for V in speeds]
    vmc = plane.vmc(float(speeds[0]), float(speeds[-1]))
    vmc_bu = plane.vmc(float(speeds[0]), float(speeds[-1]), method="buildup")
    limits = [(float(V), plane.oei_throttle_limit(V)) for V in speeds if vmc is None or V < vmc + 0.5]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(speeds, [r["rudder_req"] for r in rows], label="rudder required (handbook fin, design)")
    ax[0].plot(speeds, [plane.engine_out_rudder(V, method="buildup")["rudder_req"] for V in speeds], "--",
               label="rudder required (AeroBuildup, optimistic)")
    ax[0].axhline(plane.cfg.fin.max_rudder_deg, color="r", ls="--", label="rudder limit")
    if vmc:
        ax[0].axvline(vmc, color="k", ls=":", label=f"Vmc = {vmc:.1f} m/s")
    ax[0].set_xlabel("airspeed [m/s]"), ax[0].set_ylabel("deg"), ax[0].grid(alpha=0.3), ax[0].legend(fontsize=8)
    ax[0].set_title("One engine out")
    if limits:
        ax[1].plot([v for v, _ in limits], [t * 100 for _, t in limits], "o-")
    ax[1].set_xlabel("airspeed [m/s]"), ax[1].set_ylabel("max throttle on live motor [%]")
    ax[1].set_title("Flight-controller throttle limit after a motor failure"), ax[1].grid(alpha=0.3)
    _save(fig, out / "engine_out.png")
    return rows, (vmc, vmc_bu), limits


def packaging_figure(plane: TwinPlane, out: Path) -> list:
    """Side view + cross-section of the avionics bay, drawn to scale."""
    lay, cfg = plane.layout, plane.cfg
    n, wall = cfg.fuselage.section_exponent, cfg.fuselage.wall
    st = np.array(lay.fus_stations)
    fig = plt.figure(figsize=(14, 5.5))
    ax = fig.add_axes([0.04, 0.12, 0.62, 0.8])
    ax.fill_between(st[:, 0], -st[:, 2] / 2, st[:, 2] / 2, color="#d9d9d9", label="fuselage (outer)")
    ax.fill_between(st[:, 0], -st[:, 2] / 2 + wall, st[:, 2] / 2 - wall, color="#f4f4f4")
    for c in lay.components:
        b = c.box
        ax.add_patch(mpatches.Rectangle((c.center[0] - b.length / 2, c.center[2] - b.height / 2), b.length, b.height,
                                        color=COMPONENT_COLORS.get(c.name, "k"), alpha=0.85, label=c.name))
    # wing root section and nacelle/prop
    af = asb.Airfoil(cfg.wing.airfoil).coordinates
    ax.fill(lay.wing_x_le + af[:, 0] * lay.root_chord, lay.wing_z + af[:, 1] * lay.root_chord, color="#bbb",
            alpha=0.6, label="wing root")
    R = cfg.propulsion.prop_diameter_in * 0.0254 / 2
    ax.plot([lay.prop_x, lay.prop_x], [lay.motor_z - R, lay.motor_z + R], color="k", lw=2,
            label=f"prop disc, edge-on (at y = +/-{lay.motor_y * 1000:.0f} mm)")
    cg = plane.xyz_cg
    ax.plot(cg[0], cg[2], "o", color="gold", ms=10, mec="k", label="CG")
    ax.plot(lay.x_np, cg[2], "^", color="limegreen", ms=10, mec="k", label="neutral point")
    ax.set_aspect("equal"), ax.set_xlim(-0.05, lay.wing_x_le + lay.root_chord + 0.1)
    ax.set_ylim(min(-0.10, lay.motor_z - R - 0.01), lay.wing_z + 0.07)
    ax.set_xlabel("x from nose [m]"), ax.set_ylabel("z [m]"), ax.grid(alpha=0.3)
    ax.set_title("Avionics bay, side view (to scale)")
    ax.legend(fontsize=7, loc="lower right", ncol=2)

    # cross-section at the Pi 5
    ax2 = fig.add_axes([0.70, 0.12, 0.28, 0.8])
    pi = next(c for c in lay.components if c.name == "pi5")
    t = np.linspace(0, 2 * np.pi, 400)

    def se(a, b):
        ct, s_ = np.cos(t), np.sin(t)
        return a * np.sign(ct) * np.abs(ct) ** (2 / n), b * np.sign(s_) * np.abs(s_) ** (2 / n)

    W, H = lay.fus_width, lay.fus_height
    ax2.fill(*se(W / 2, H / 2), color="#d9d9d9")
    ax2.fill(*se(W / 2 - wall, H / 2 - wall), color="#f4f4f4")
    for c in lay.components:
        if c.name in ("pi5", "battery", "flight_controller"):
            b = c.box
            ax2.add_patch(mpatches.Rectangle((c.center[1] - b.width / 2, c.center[2] - b.height / 2), b.width,
                                             b.height, fill=c.name == "pi5", alpha=0.85 if c.name == "pi5" else 1,
                                             ec=COMPONENT_COLORS[c.name], fc=COMPONENT_COLORS[c.name], lw=1.5,
                                             label=f"{c.name} {b.width * 1000:.0f}x{b.height * 1000:.0f} mm"))
    ax2.set_aspect("equal"), ax2.grid(alpha=0.3), ax2.legend(fontsize=7, loc="lower center")
    ax2.set_title(f"Cross-section at the Pi 5 (x = {pi.center[0]:.3f} m)\n"
                  f"outer {W * 1000:.0f} x {H * 1000:.0f} mm, 5 mm walls")
    ax2.set_xlabel("y [m]"), ax2.set_ylim(-H / 2 - 0.02, H / 2 + 0.03)
    fig.savefig(out / "packaging.png", dpi=140)
    plt.close(fig)
    return check_packaging(lay, cfg)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", default="results")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore")

    plane = TwinPlane()
    cfg, lay = plane.cfg, plane.layout
    print(f"Mass {plane.mass.mass:.3f} kg, CG x = {plane.xyz_cg[0]:.3f} m, SM {plane.static_margin:.1%}")

    plane.clean.draw_three_view(show=False)
    _save(plt.gcf(), out / "three_view.png")

    clmax = plane.trimmed_clmax()
    v_stall = plane.stall_speed(clmax)
    print(f"Trimmed CLmax {clmax:.2f}, stall {v_stall:.1f} m/s")

    pts = trim_and_performance(plane, out, v_stall)
    usable = [p for p in pts if p.V >= 1.3 * v_stall and p.endurance_min]
    best_end = max(usable, key=lambda p: p.endurance_min)
    cruise = (min(pts, key=lambda p: abs(p.V - cfg.cruise_speed)) if cfg.cruise_speed else best_end)
    level = [p for p in pts if p.thrust_avail >= p.drag]
    one_eng = [p for p in pts if p.thrust_avail / 2 >= p.drag]
    best_ld = max(pts, key=lambda p: p.L_over_D)

    foil = airfoil_polars(plane, out, cruise.V)
    aircraft_polar(plane, out, cruise.V)
    derivs = plane.stability_derivatives(cruise.V, cruise.alpha)
    ctrl = plane.control_derivatives(cruise.V, cruise.alpha)
    eo_rows, (vmc, vmc_buildup), thr_limits = engine_out(plane, out, np.arange(np.floor(v_stall), 22.0, 1.0))
    pack = packaging_figure(plane, out)
    files = export_all(plane, out / "exports")
    static = plane.prop.operating_point(0.1, plane.rho, 1.0)

    results = {
        "motor": cfg.propulsion.motor_name,
        "prop": f"{cfg.propulsion.prop_diameter_in:.0f}x{cfg.propulsion.prop_pitch_in} {cfg.propulsion.n_blades}-blade",
        "design_vars": asdict(cfg.design),
        "mass_kg": plane.mass.mass,
        "cg": plane.xyz_cg,
        "x_np": lay.x_np,
        "static_margin": plane.static_margin,
        "static_margin_buildup": (derivs.get("x_np", np.nan) - plane.xyz_cg[0]) / lay.mac,
        "wing_area_m2": lay.wing_area,
        "wing_loading_g_dm2": plane.mass.mass * 1000 / (lay.wing_area * 100),
        "aspect_ratio": lay.span**2 / lay.wing_area,
        "mac_m": lay.mac,
        "dimensions": {k: getattr(lay, k) for k in (
            "span", "root_chord", "tip_chord", "wing_x_le", "wing_z", "mac", "fus_length", "fus_width",
            "fus_height", "tail_arm", "stab_semispan", "stab_root_chord", "stab_tip_chord", "stab_x_le",
            "stab_area", "fin_height", "fin_root_chord", "fin_tip_chord", "fin_x_le", "fin_area", "motor_y",
            "prop_x", "motor_z", "nacelle_length", "Vh", "Vv")},
        "components": [{"name": c.name, "center": c.center, "size": [c.box.length, c.box.width, c.box.height],
                        "mass": c.box.mass} for c in lay.components],
        "fc_offset_from_cg_mm": lay.fc_offset_from_cg() * 1000,
        "packaging": pack,
        "mass_table": mass_table(lay),
        "airfoils": foil,
        "CLmax_trimmed": clmax,
        "v_stall": v_stall,
        "cruise": asdict(cruise),
        "best_L_over_D": asdict(best_ld),
        "v_max_level": max(p.V for p in level) if level else None,
        "one_engine_level_speeds": [p.V for p in one_eng],
        "static_thrust_per_motor_N": static.thrust,
        "static_current_per_motor_A": static.current,
        "thrust_to_weight": 2 * static.thrust / plane.weight,
        "avionics_power_W": plane.avionics_power,
        "usable_energy_Wh": plane.usable_energy_wh,
        "stability_derivatives_buildup_per_rad": derivs,
        "control_derivatives_per_deg": ctrl,
        "vmc": vmc,
        "vmc_buildup": vmc_buildup,
        "oei_throttle_limits": thr_limits,
        "engine_out": eo_rows,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    write_summary(out, plane, results, cruise, files)
    print(f"Wrote {out}/")


def _mm(x):
    return f"{x * 1000:.0f} mm"


def write_summary(out: Path, plane: TwinPlane, r: dict, cruise, files) -> None:
    cfg, lay = plane.cfg, plane.layout
    d, c = r["stability_derivatives_buildup_per_rad"], r["control_derivatives_per_deg"]
    p, a = cfg.propulsion, cfg.avionics
    sizing = Path("results/sizing/best.json")
    lines = [
        "# Twin high-wing RC plane: simulation results",
        "",
        "Generated by `python -m twin_rc.run_all`. Don't edit by hand.",
        "",
        "![three view](three_view.png)",
        "",
        "## Hardware",
        "",
        f"- **Motors:** 2x {p.motor_name} ({p.motor_mass * 1000:.0f} g), {p.prop_diameter_in:.0f}x{p.prop_pitch_in} "
        f"{p.n_blades}-blade props, counter-rotating, {p.battery_cells}S. "
        f"{p.esc_current_rating:.0f} A ESCs (static draw is {r['static_current_per_motor_A']:.0f} A per motor).",
        f"- **Battery:** 3S {cfg.design.battery_mah} mAh ({r['usable_energy_Wh']:.0f} Wh usable).",
        f"- **Avionics:** Raspberry Pi 5 with active cooler ({_mm(a.pi5.length)} x {_mm(a.pi5.width)} x "
        f"{_mm(a.pi5.height)}), flight controller PCB ({_mm(a.flight_controller.length)} x "
        f"{_mm(a.flight_controller.width)}), 5 V/5 A BEC, receiver. "
        f"These draw {r['avionics_power_W']:.1f} W continuously.",
        "",
        "## Dimensions (sized by `python -m twin_rc.sizing`)",
        "",
        "| | |",
        "|---|---|",
        f"| Wingspan | {_mm(lay.span)} |",
        f"| Wing chord, root / tip | {_mm(lay.root_chord)} / {_mm(lay.tip_chord)} (rectangular to "
        f"{cfg.wing.taper_start_frac:.0%} semispan) |",
        f"| Wing area, aspect ratio, MAC | {lay.wing_area:.3f} m^2, {r['aspect_ratio']:.1f}, {_mm(lay.mac)} |",
        f"| Wing leading edge from nose | {_mm(lay.wing_x_le)} |",
        f"| Fuselage length / width / height | {_mm(lay.fus_length)} / {_mm(lay.fus_width)} / {_mm(lay.fus_height)} |",
        f"| Tail arm (wing AC to stabilator AC) | {_mm(lay.tail_arm)} |",
        f"| Stabilators (each) | {_mm(lay.stab_semispan)} span, chord {_mm(lay.stab_root_chord)} / "
        f"{_mm(lay.stab_tip_chord)}, LE at {_mm(lay.stab_x_le)}; total {lay.stab_area * 1e4:.0f} cm^2, "
        f"Vh {lay.Vh:.2f} |",
        f"| Fin + rudder | height {_mm(lay.fin_height)}, chord {_mm(lay.fin_root_chord)} / "
        f"{_mm(lay.fin_tip_chord)}, {lay.fin_area * 1e4:.0f} cm^2, Vv {lay.Vv:.3f}; rudder = aft "
        f"{1 - cfg.fin.rudder_hinge:.0%} |",
        f"| Motor centrelines | +/-{_mm(lay.motor_y)} from the centreline "
        f"({_mm(cfg.propulsion.prop_tip_clearance)} prop-tip clearance to the fuselage) |",
        f"| Prop plane from nose | {_mm(lay.prop_x)} |",
        f"| Nacelle length | {_mm(lay.nacelle_length)} (motor + {p.esc_current_rating:.0f} A ESC inside) |",
        "",
        "## Avionics bay",
        "",
        "![packaging](packaging.png)",
        "",
        "| Component | Centre x from nose | Size (L x W x H) | Fits |",
        "|---|---|---|---|",
        *[f"| {comp['name']} | {_mm(comp['center'][0])} | "
          f"{' x '.join(_mm(v) for v in comp['size'])} | "
          f"{'yes' if next(pk for pk in r['packaging'] if pk['name'] == comp['name'])['fits'] else '**NO**'} |"
          for comp in r["components"]],
        "",
        f"The flight controller sits **{abs(r['fc_offset_from_cg_mm']):.0f} mm from the CG**, so its IMU "
        "measures close to the true CG accelerations. The battery and Pi 5 are up front and set the "
        "balance. The space between the Pi and the flight controller is free for GPS, a camera, etc. "
        "The Pi needs a cooling-air inlet/outlet in the hatch.",
        "",
        "## Performance",
        "",
        "| | |",
        "|---|---|",
        f"| All-up mass | {r['mass_kg']:.2f} kg, wing loading {r['wing_loading_g_dm2']:.1f} g/dm^2 |",
        f"| Trimmed CLmax | {r['CLmax_trimmed']:.2f} |",
        f"| Stall speed | {r['v_stall']:.1f} m/s |",
        f"| Max flight time ({cruise.V:.1f} m/s) | **{cruise.endurance_min:.1f} min**; "
        f"{cruise.total_power:.0f} W total, of which {r['avionics_power_W']:.0f} W is avionics; "
        f"L/D {cruise.L_over_D:.1f} |",
        f"| Trim at that speed | alpha {cruise.alpha:.1f} deg, stabilators {cruise.stab_deg:+.1f} deg |",
        f"| Best L/D | {r['best_L_over_D']['L_over_D']:.1f} at {r['best_L_over_D']['V']:.1f} m/s |",
        f"| Static thrust | {r['static_thrust_per_motor_N']:.1f} N per motor, T/W {r['thrust_to_weight']:.2f} |",
        f"| Max level speed | {r['v_max_level'] or 'beyond sweep'} m/s |",
        "| Level flight on one engine | "
        + (f"{min(r['one_engine_level_speeds']):.1f}-{max(r['one_engine_level_speeds']):.1f} m/s |"
           if r["one_engine_level_speeds"] else "not possible |"),
        "",
        "![trim and performance](trim_performance.png)",
        "",
        "## Stability (at the max-flight-time speed)",
        "",
        f"- **Static margin {r['static_margin']:.0%} MAC.** Neutral point at {_mm(r['x_np'])} from the "
        f"VLM with a fuselage correction; CG at {_mm(r['cg'][0])}. AeroBuildup gives "
        f"{r['static_margin_buildup']:.0%}, but its downwash model overstates the tail, so the VLM "
        "figure is the design value.",
        "",
        "| Derivative | Value (per rad) | Want |",
        "|---|---|---|",
        f"| Cm_alpha | {d['Cma']:.3f} | < 0 |",
        f"| Cn_beta | {d['Cnb']:.3f} | > 0 |",
        f"| Cl_beta | {d['Clb']:.3f} | < 0 |",
        f"| Cm_q | {d['Cmq']:.2f} | < 0 |",
        f"| Cl_p | {d['Clp']:.3f} | < 0 |",
        f"| Cn_r | {d['Cnr']:.3f} | < 0 |",
        "",
        "## Control power (per degree)",
        "",
        "| Surface | Effect |",
        "|---|---|",
        f"| Stabilators, collective (+ = TE down) | Cm {c['Cm_dstab']:+.4f}, CL {c['CL_dstab']:+.4f} |",
        f"| Stabilators, differential (+ = left TE down) | Cl {c['Cl_dstab_diff']:+.4f} |",
        f"| Rudder (+ = TE right) | Cn {c['Cn_drudder']:+.4f}, CY {c['CY_drudder']:+.4f} |",
        f"| Ailerons | Cl {c['Cl_daileron']:+.4f}, Cn {c['Cn_daileron']:+.5f} |",
        "",
        "![aircraft polars](aircraft_polars.png)",
        "",
        "## One engine out",
        "",
        "The motors sit as close to the fuselage as the props allow, which keeps the asymmetric "
        "thrust moment small.",
        "",
        (f"- **Full-power Vmc = {r['vmc']:.1f} m/s** ({r['vmc'] / r['v_stall']:.2f} x stall), using the "
         "handbook estimate of rudder power for a low-aspect-ratio fin. Below this, the flight controller "
         "should cap the live motor's throttle at the limit in the right-hand plot."
         if r["vmc"] else "- Full rudder holds full power on one engine at every speed in the sweep."),
        f"- AeroBuildup predicts about 3x more rudder power (Vmc "
        f"{r['vmc_buildup'] or 0:.1f} m/s). The fin is sized with the more conservative handbook value.",
        "",
        "![engine out](engine_out.png)",
        "",
        "## Airfoils",
        "",
        "| Section | Airfoil | Re | Clmax | max L/D |",
        "|---|---|---|---|---|",
        *[f"| {k} | {v['airfoil']} | {v['Re']:.0f} | {v['Clmax']:.2f} | {v['max_L_over_D']:.0f} |"
          for k, v in r["airfoils"].items()],
        "",
        "![airfoil polars](airfoil_polars.png)",
        "",
        "## Mass breakdown",
        "",
        "| Item | Mass |",
        "|---|---|",
        *[f"| {name} | {m * 1000:.0f} g |" for name, m, *_ in r["mass_table"]],
        f"| **Total** | **{r['mass_kg'] * 1000:.0f} g** |",
        "",
        "## Cross-check in XFLR5 / AVL",
        "",
        f"- XFLR5: File -> Import -> Import from XML -> `exports/{files['xflr5'].name}`. Load the airfoil "
        "`.dat` files in `exports/` first. The stabilators are exported as one symmetric all-moving "
        "tail; set its tilt angle, or make it the control in a T7 stability analysis.",
        f"- AVL: `exports/{files['avl'].name}`",
        "",
        "## Model limits",
        "",
        "- The motor's winding resistance and no-load current are estimates. The prop "
        "coefficients are calibrated to published 2218 bench data. A thrust-stand run of the "
        "real motor and prop should replace them.",
        "- Structure masses assume foam-board construction (see `config.py` areal masses). "
        "Weigh the parts as you build and update the config.",
        "- The prop wash over the wing and tail isn't modelled.",
    ]
    if sizing.exists():
        lines += ["", "## Sizing study", "", "![sizing](sizing/sizing.png)", "",
                  "Full sweep: `sizing/sweep.csv`. Chosen point: `sizing/best.json`."]
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
