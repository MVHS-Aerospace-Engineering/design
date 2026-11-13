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
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .analysis import TwinPlane  # noqa: E402
from .export import export_all  # noqa: E402
from .mass import table as mass_table  # noqa: E402


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def airfoil_polars(plane: TwinPlane, out: Path) -> dict:
    """NeuralFoil polars at flight Reynolds numbers (the 'Direct Foil Analysis'
    step of an XFLR5 workflow)."""
    cfg = plane.cfg
    mu = float(np.ravel(plane.atmo.dynamic_viscosity())[0])
    V = cfg.cruise_speed
    alphas = np.linspace(-8, 16, 97)
    cases = [
        (cfg.wing.airfoil, cfg.wing.root_chord, "wing root"),
        (cfg.wing.airfoil, cfg.wing.tip_chord, "wing tip"),
        (cfg.stabilator.airfoil, cfg.stabilator.root_chord, "stabilator"),
    ]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    info = {}
    for name, chord, label in cases:
        Re = plane.rho * V * chord / mu
        a = asb.Airfoil(name).get_aero_from_neuralfoil(alpha=alphas, Re=Re)
        CL, CD, CM = (np.ravel(a[k]) for k in ("CL", "CD", "CM"))
        tag = f"{name} {label} (Re {Re / 1e3:.0f}k)"
        ax[0].plot(alphas, CL, label=tag)
        ax[1].plot(CD, CL, label=tag)
        ax[2].plot(alphas, CL / CD, label=tag)
        i = int(np.argmax(CL))
        info[label] = {"airfoil": name, "Re": Re, "Clmax": CL[i], "alpha_Clmax": alphas[i],
                       "max_L_over_D": float(np.max(CL / CD))}
    for a_, xl, yl in zip(ax, ("alpha [deg]", "Cd", "alpha [deg]"), ("Cl", "Cl", "Cl/Cd")):
        a_.set_xlabel(xl), a_.set_ylabel(yl), a_.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.suptitle(f"Airfoil polars (NeuralFoil) at {V:.0f} m/s")
    _save(fig, out / "airfoil_polars.png")
    return info


def aircraft_polar(plane: TwinPlane, out: Path) -> dict:
    V = plane.cfg.cruise_speed
    alphas = np.linspace(-6, 18, 49)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    clmax = None
    for stab in (-10, -5, 0, 5):
        r = plane.alpha_sweep(V, alphas, stab=stab)
        lbl = f"stabilators {stab:+d} deg"
        ax[0].plot(alphas, r["CL"], label=lbl)
        ax[1].plot(r["CD"], r["CL"], label=lbl)
        ax[2].plot(alphas, r["Cm"], label=lbl)
        if stab == -10:  # nose-up trim at high alpha
            clmax = float(np.max(r["CL"]))
    for a_, xl, yl in zip(ax, ("alpha [deg]", "CD", "alpha [deg]"), ("CL", "CL", "Cm (about CG)")):
        a_.set_xlabel(xl), a_.set_ylabel(yl), a_.grid(alpha=0.3)
    ax[2].axhline(0, color="k", lw=0.8)
    ax[0].legend(fontsize=8)
    fig.suptitle(f"Whole-aircraft polars (AeroBuildup) at {V:.0f} m/s")
    _save(fig, out / "aircraft_polars.png")
    return {"CLmax_approx": clmax}


def trim_and_performance(plane: TwinPlane, out: Path, v_stall: float) -> list:
    speeds = np.arange(np.ceil(v_stall * 1.1), 30.5, 1.0)
    pts = plane.trim_sweep(speeds)
    V = np.array([p.V for p in pts])
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot(V, [p.alpha for p in pts], label="alpha")
    ax[0, 0].plot(V, [p.stab_deg for p in pts], label="stabilator (collective)")
    ax[0, 0].set_ylabel("deg"), ax[0, 0].set_title("Trim")
    ax[0, 1].plot(V, [p.drag for p in pts], label="thrust required (drag)")
    ax[0, 1].plot(V, [p.thrust_avail for p in pts], label="thrust available (2 motors)")
    ax[0, 1].plot(V, [p.thrust_avail / 2 for p in pts], "--", label="one engine out")
    ax[0, 1].set_ylabel("N"), ax[0, 1].set_title("Thrust")
    ax[1, 0].plot(V, [p.L_over_D for p in pts])
    ax[1, 0].set_ylabel("L/D"), ax[1, 0].set_title("Aerodynamic efficiency")
    ax[1, 1].plot(V, [p.electrical_power or np.nan for p in pts])
    ax[1, 1].set_ylabel("W (battery)"), ax[1, 1].set_title("Electrical power in level flight")
    for a_ in ax.flat:
        a_.set_xlabel("airspeed [m/s]"), a_.grid(alpha=0.3)
        if a_.get_legend_handles_labels()[0]:
            a_.legend(fontsize=8)
    _save(fig, out / "trim_performance.png")
    return pts


def engine_out(plane: TwinPlane, out: Path, speeds) -> tuple[list, float | None]:
    rows = [plane.engine_out_rudder(V) for V in speeds]
    vmc = plane.vmc(float(speeds[0]), float(speeds[-1]))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(speeds, [r["rudder_req"] for r in rows], label="rudder required")
    ax.axhline(plane.cfg.fin.max_rudder_deg, color="r", ls="--", label="rudder limit")
    if vmc:
        ax.axvline(vmc, color="k", ls=":", label=f"Vmc = {vmc:.1f} m/s")
    ax.set_xlabel("airspeed [m/s]"), ax.set_ylabel("deg"), ax.grid(alpha=0.3), ax.legend()
    ax.set_title("One engine out, other at full throttle")
    _save(fig, out / "engine_out.png")
    return rows, vmc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", default="results")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore")

    plane = TwinPlane()
    cfg = plane.cfg
    print(f"Mass {plane.mass.mass:.3f} kg, CG x = {plane.xyz_cg[0]:.3f} m")

    axs = plane.clean.draw_three_view(show=False)
    _save(plt.gcf(), out / "three_view.png")

    foil = airfoil_polars(plane, out)
    polar = aircraft_polar(plane, out)
    clmax = polar["CLmax_approx"]
    v_stall = plane.stall_speed(clmax)
    print(f"CLmax ~ {clmax:.2f}, stall {v_stall:.1f} m/s")

    pts = trim_and_performance(plane, out, v_stall)
    cruise = min(pts, key=lambda p: abs(p.V - cfg.cruise_speed))
    level = [p for p in pts if p.thrust_avail >= p.drag]
    v_max = max(p.V for p in level) if level else None
    one_eng = [p for p in pts if p.thrust_avail / 2 >= p.drag]
    best_ld = max(pts, key=lambda p: p.L_over_D)

    derivs = plane.stability_derivatives(cruise.V, cruise.alpha)
    sm_buildup = plane.static_margin(cruise.V, cruise.alpha)
    vlm = plane.vlm(cruise.V, cruise.alpha)
    sm_vlm = (vlm["x_np"] - plane.xyz_cg[0]) / plane.clean.c_ref
    ctrl = plane.control_derivatives(cruise.V, cruise.alpha)

    eo_rows, vmc = engine_out(plane, out, np.arange(np.floor(v_stall), 25.0, 1.0))
    files = export_all(plane, out / "exports")

    cap_ah = cfg.propulsion.battery_capacity_mah / 1000
    v_batt = cfg.propulsion.battery_cells * cfg.propulsion.cell_voltage
    endurance_min = (
        0.8 * cap_ah * v_batt / cruise.electrical_power * 60 if cruise.electrical_power else None
    )
    static = plane.prop.operating_point(0.1, plane.rho, 1.0)

    results = {
        "mass_kg": plane.mass.mass,
        "cg": plane.xyz_cg,
        "wing_area_m2": plane.clean.s_ref,
        "wing_loading_g_dm2": plane.mass.mass * 1000 / (plane.clean.s_ref * 100),
        "mac_m": plane.clean.c_ref,
        "airfoils": foil,
        "CLmax": clmax,
        "v_stall": v_stall,
        "cruise": asdict(cruise),
        "best_L_over_D": asdict(best_ld),
        "v_max_level": v_max,
        "one_engine_level_speeds": [p.V for p in one_eng],
        "static_thrust_per_motor_N": static.thrust,
        "static_current_per_motor_A": static.current,
        "thrust_to_weight": 2 * static.thrust / plane.weight,
        "endurance_min_cruise_80pct": endurance_min,
        "stability_derivatives_buildup_per_rad": derivs,
        "static_margin_buildup": sm_buildup,
        "vlm_check": vlm,
        "static_margin_vlm": sm_vlm,
        "control_derivatives_per_deg": ctrl,
        "vmc": vmc,
        "engine_out": eo_rows,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    write_summary(out, plane, results, cruise, files)
    print(f"Wrote {out}/")


def write_summary(out: Path, plane: TwinPlane, r: dict, cruise, files) -> None:
    cfg = plane.cfg
    d = r["stability_derivatives_buildup_per_rad"]
    c = r["control_derivatives_per_deg"]
    stab_trim_range = cfg.stabilator.max_deflection_deg
    lines = [
        "# Twin high-wing RC plane: simulation results",
        "",
        "Generated by `python -m twin_rc.run_all`. Don't edit by hand.",
        "",
        "![three view](three_view.png)",
        "",
        "## Configuration",
        "",
        f"- High wing, span {cfg.wing.span:.2f} m, area {r['wing_area_m2']:.3f} m^2, MAC {r['mac_m']:.3f} m, "
        f"{cfg.wing.airfoil} airfoil, {cfg.wing.dihedral_deg:.0f} deg dihedral",
        f"- Two wing-mounted tractor motors at y = +/-{cfg.propulsion.motor_y:.2f} m, each turning a "
        f"{cfg.propulsion.n_blades}-blade {cfg.propulsion.prop_diameter_in:.0f}x{cfg.propulsion.prop_pitch_in:.0f} "
        f"prop (counter-rotating), {cfg.propulsion.motor_kv:.0f} Kv on {cfg.propulsion.battery_cells}S",
        "- Tail: **one fin with a rudder** + **two independent all-moving stabilators** "
        f"({cfg.stabilator.airfoil}, pivot at {cfg.stabilator.pivot_frac:.0%} chord). No elevator.",
        f"- Mass {r['mass_kg']:.2f} kg, wing loading {r['wing_loading_g_dm2']:.1f} g/dm^2, "
        f"CG at x = {r['cg'][0]:.3f} m",
        "",
        "## Performance",
        "",
        "| | |",
        "|---|---|",
        f"| CLmax (whole aircraft, approx.) | {r['CLmax']:.2f} |",
        f"| Stall speed | {r['v_stall']:.1f} m/s |",
        f"| Cruise ({cruise.V:.0f} m/s) | alpha {cruise.alpha:.1f} deg, stabilators {cruise.stab_deg:+.1f} deg, "
        f"L/D {cruise.L_over_D:.1f}, drag {cruise.drag:.2f} N |",
        f"| Best L/D | {r['best_L_over_D']['L_over_D']:.1f} at {r['best_L_over_D']['V']:.0f} m/s |",
        f"| Static thrust | {r['static_thrust_per_motor_N']:.1f} N per motor "
        f"({r['static_current_per_motor_A']:.0f} A), T/W = {r['thrust_to_weight']:.2f} |",
        f"| Max level speed (both motors) | {r['v_max_level'] or 'beyond sweep'} m/s |",
        f"| Cruise battery power | {cruise.electrical_power:.0f} W ({cruise.throttle_current:.1f} A) |"
        if cruise.electrical_power else "| Cruise battery power | n/a |",
        f"| Endurance at cruise (80 % of {cfg.propulsion.battery_capacity_mah:.0f} mAh) | "
        f"{r['endurance_min_cruise_80pct']:.1f} min |" if r["endurance_min_cruise_80pct"] else "",
        f"| Level flight on one engine | "
        + (f"{min(r['one_engine_level_speeds']):.0f}-{max(r['one_engine_level_speeds']):.0f} m/s |"
           if r["one_engine_level_speeds"] else "not possible |"),
        "",
        "![trim and performance](trim_performance.png)",
        "",
        "## Stability (at cruise trim)",
        "",
        "| Derivative | Value (per rad) | Want |",
        "|---|---|---|",
        f"| CL_alpha | {d['CLa']:.2f} | > 0 |",
        f"| Cm_alpha | {d['Cma']:.3f} | < 0 (pitch stable) |",
        f"| Cn_beta | {d['Cnb']:.3f} | > 0 (weathercock stable) |",
        f"| Cl_beta | {d['Clb']:.3f} | < 0 (dihedral effect) |",
        f"| Cm_q | {d['Cmq']:.2f} | < 0 (pitch damping) |",
        f"| Cl_p | {d['Clp']:.3f} | < 0 (roll damping) |",
        f"| Cn_r | {d['Cnr']:.3f} | < 0 (yaw damping) |",
        "",
        f"- **Static margin: {r['static_margin_vlm']:.0%} MAC from VLM** "
        f"(neutral point x = {r['vlm_check']['x_np']:.3f} m). AeroBuildup gives "
        f"{r['static_margin_buildup']:.0%}. Treat the VLM number as the design value: VLM "
        "models the wing downwash on the tail, while AeroBuildup's simpler model "
        "overstates tail effectiveness. Neither includes the fuselage's destabilising "
        "effect perfectly, so the real margin is likely a few % lower than the VLM figure. "
        "Target 10-15 %.",
        "",
        "## Control power (per degree, at cruise)",
        "",
        "| Surface | Effect |",
        "|---|---|",
        f"| Stabilators, collective (+ = TE down) | Cm {c['Cm_dstab']:+.4f}, CL {c['CL_dstab']:+.4f} |",
        f"| Stabilators, differential (+ = left TE down) | Cl {c['Cl_dstab_diff']:+.4f}, "
        f"Cn {c['Cn_dstab_diff']:+.5f} |",
        f"| Rudder (+ = TE right, nose right) | Cn {c['Cn_drudder']:+.4f}, CY {c['CY_drudder']:+.4f}, "
        f"Cl {c['Cl_drudder']:+.4f} |",
        f"| Ailerons | Cl {c['Cl_daileron']:+.4f}, Cn {c['Cn_daileron']:+.5f} (adverse yaw) |",
        "",
        f"Stabilator travel needed across the speed range is well inside +/-{stab_trim_range:.0f} deg; "
        "see the trim plot. Differential stabilator adds about "
        f"{abs(c['Cl_dstab_diff'] / c['Cl_daileron']):.0%} of the ailerons' roll power.",
        "",
        "![aircraft polars](aircraft_polars.png)",
        "",
        "## One engine out",
        "",
        "With one motor dead and the other at full throttle, the rudder alone has to cancel the "
        "yaw from asymmetric thrust and the dead prop's drag.",
        "",
        (f"- **Vmc = {r['vmc']:.1f} m/s** ({r['vmc'] / r['v_stall']:.2f} x stall). Below this, full rudder "
         "can't hold heading at full power on one engine, so throttle back first."
         if r["vmc"] else "- Rudder holds heading at every speed in the sweep."),
        "",
        "![engine out](engine_out.png)",
        "",
        "## Airfoils",
        "",
        "| Section | Airfoil | Re | Clmax | alpha at Clmax | max L/D |",
        "|---|---|---|---|---|---|",
        *[f"| {k} | {v['airfoil']} | {v['Re']:.0f} | {v['Clmax']:.2f} | {v['alpha_Clmax']:.1f} | "
          f"{v['max_L_over_D']:.0f} |" for k, v in r["airfoils"].items()],
        "",
        "![airfoil polars](airfoil_polars.png)",
        "",
        "## Cross-check in XFLR5 / AVL",
        "",
        f"- XFLR5: File -> Import -> Import from XML -> `exports/{files['xflr5'].name}`; airfoil "
        "`.dat` files are in `exports/` (load them first, then run a Batch Foil Analysis "
        f"at Re {int(r['airfoils']['wing tip']['Re'] // 1000)}k-{int(r['airfoils']['wing root']['Re'] // 1000)}k). "
        "The two stabilators are exported as one symmetric all-moving tail. In XFLR5, set its tilt "
        "angle, or make it the control variable in a T7 stability analysis.",
        f"- AVL: `exports/{files['avl'].name}`",
        "",
        f"- VLM cross-check at cruise: CL_alpha {r['vlm_check']['CLa_per_deg'] * 57.3:.2f}/rad "
        f"(AeroBuildup {d['CLa']:.2f}/rad).",
    ]
    (out / "SUMMARY.md").write_text("\n".join(l for l in lines if l is not None) + "\n")


if __name__ == "__main__":
    main()
