"""Sizing sweep: find the best span, chord, tail arm and battery around the FT
Radial 2218 motors and the Pi 5 + flight-controller payload.

    python -m twin_rc.sizing          # writes results/sizing/ (~5 min)

For every candidate this script:
1. Derives the full layout. Fuselage section from the payload, flight controller
   at the CG, wing placed for the static-margin target, tails sized from volume
   coefficients plus the engine-out need, motors as close in as the props allow.
2. Builds a trimmed polar with AeroBuildup: alpha sweeps at several stabilator
   settings, interpolated to Cm = 0. That gives the trimmed CLmax, stall speed
   and the drag polar.
3. Computes the battery power for level flight at each speed with the motor/prop
   model, plus the Pi 5 / flight-controller / servo load, and finds the best
   flight time (flown no slower than 1.3 x stall).

Constraints: stall <= cfg.max_stall_speed; on one engine at full power, level
flight is possible at 1.3 x stall; the payload fits inside the fuselage.
Objective: maximum flight time.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
import warnings
from dataclasses import asdict, replace
from pathlib import Path

import aerosandbox as asb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import PlaneConfig  # noqa: E402
from .geometry import build_airplane  # noqa: E402
from .layout import build_layout, check_packaging, neutral_point_offset  # noqa: E402
from .propulsion import Propulsion  # noqa: E402

G = 9.81
SPANS = [1.2, 1.4, 1.6, 1.8]
CHORDS = [0.20, 0.23, 0.26, 0.29, 0.32]
TAIL_ARMS = [2.5, 3.0, 3.5]
BATTERIES = [2200, 3000, 4000, 5000]
MAX_SPAN_NOTE = "1.8 m cap: fits a car and builds from 3 foam-board panels"


def evaluate(cfg: PlaneConfig, np_offset: float) -> dict:
    lay = build_layout(cfg, np_offset)
    lay = build_layout(cfg, np_offset, mass_guess=lay.mass)  # fin size depends on the stall speed
    atmo = asb.Atmosphere(altitude=cfg.altitude)
    rho = float(np.ravel(atmo.density())[0])
    cg = list(lay.cg)
    W = lay.mass * G
    S = lay.wing_area
    lim = cfg.stabilator.max_deflection_deg

    # Trimmed polar at a representative speed (Re effects are second order here).
    alphas = np.linspace(-4, 16, 41)
    stabs = (-lim, -lim / 2, 0.0, lim / 4)
    sweeps = []
    for s in stabs:
        ap = build_airplane(cfg, lay, stab_left_deg=s, stab_right_deg=s, xyz_ref=cg)
        r = asb.AeroBuildup(ap, asb.OperatingPoint(atmosphere=atmo, velocity=11.0, alpha=alphas), xyz_ref=cg).run()
        sweeps.append({k: np.ravel(r[k]) for k in ("CL", "CD", "Cm")})
    CLt, CDt = [], []
    for i in range(len(alphas)):
        cm = np.array([sw["Cm"][i] for sw in sweeps])
        if not (cm.min() <= 0 <= cm.max()):
            continue
        o = np.argsort(cm)
        CLt.append(np.interp(0, cm[o], np.array([sw["CL"][i] for sw in sweeps])[o]))
        CDt.append(np.interp(0, cm[o], np.array([sw["CD"][i] for sw in sweeps])[o]))
    CLt, CDt = np.array(CLt), np.array(CDt)
    i_max = int(np.argmax(CLt))
    clmax = float(CLt[i_max])
    cl_branch, cd_branch = CLt[: i_max + 1], CDt[: i_max + 1]  # pre-stall branch
    order = np.argsort(cl_branch)
    cl_branch, cd_branch = cl_branch[order], cd_branch[order]

    v_s = float(np.sqrt(2 * W / (rho * S * clmax)))
    prop = Propulsion(cfg.propulsion)
    a = cfg.avionics
    hotel = (a.pi5_power + a.fc_power + a.servo_receiver_power) / a.bec_efficiency
    p = cfg.propulsion
    energy_wh = cfg.design.battery_mah / 1000 * p.battery_cells * p.cell_voltage * p.usable_battery_fraction

    best = (0.0, None, None)
    oei_ok = False
    for V in np.arange(max(1.3 * v_s, 6.0), 22.0, 0.5):
        CL = W / (0.5 * rho * V**2 * S)
        if CL > cl_branch[-1]:
            continue
        D = float(np.interp(CL, cl_branch, cd_branch)) * 0.5 * rho * V**2 * S
        op = prop.throttle_for_thrust(D / 2, V, rho)
        if op is None:
            continue
        P = 2 * op.electrical_power + hotel
        t = energy_wh / P * 60
        if t > best[0]:
            best = (t, float(V), float(P))
        if V <= 1.3 * v_s + 1.0 and prop.operating_point(V, rho, 1.0).thrust >= D * 1.05:
            oei_ok = True

    pack = check_packaging(lay, cfg)
    return {
        "span": cfg.design.span, "root_chord": cfg.design.root_chord,
        "tail_arm_over_mac": cfg.design.tail_arm_over_mac, "battery_mah": cfg.design.battery_mah,
        "mass": lay.mass, "wing_area": S, "wing_loading_g_dm2": lay.mass * 1000 / (S * 100),
        "aspect_ratio": lay.span**2 / S, "fus_length": lay.fus_length, "fin_area": lay.fin_area,
        "Vv": lay.Vv, "clmax_trim": clmax, "v_stall": v_s, "endurance_min": best[0], "v_endurance": best[1],
        "power_W": best[2], "oei_level_ok": oei_ok, "packaging_ok": all(r["fits"] for r in pack),
        "fc_offset_mm": lay.fc_offset_from_cg() * 1000,
    }


def feasible(r: dict, cfg: PlaneConfig) -> bool:
    return (r["v_stall"] <= cfg.max_stall_speed and r["oei_level_ok"] and r["packaging_ok"]
            and r["endurance_min"] > 0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default="results/sizing")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore")

    base = PlaneConfig()
    rows = []
    t0 = time.time()
    for span, chord, arm in itertools.product(SPANS, CHORDS, TAIL_ARMS):
        d = replace(base.design, span=span, root_chord=chord, tail_arm_over_mac=arm, battery_mah=3000)
        cfg = replace(base, design=d)
        # Neutral point (relative to the MAC) only depends on the wing/tail geometry.
        npo = neutral_point_offset(cfg, build_layout(cfg, 0.45))
        for batt in BATTERIES:
            cfg_b = replace(cfg, design=replace(d, battery_mah=batt))
            r = evaluate(cfg_b, npo)
            r["feasible"] = feasible(r, base)
            rows.append(r)
        print(f"span {span} chord {chord} arm {arm}: done ({time.time() - t0:.0f} s)", flush=True)

    with open(out / "sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r["feasible"]]
    best = max(ok, key=lambda r: r["endurance_min"])
    (out / "best.json").write_text(json.dumps(best, indent=2, default=float))
    print("BEST", json.dumps(best, default=float))

    # Plot: endurance vs span for each battery, best chord/arm per point, feasible only.
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for batt in BATTERIES:
        xs, ys, vs = [], [], []
        for span in SPANS:
            cand = [r for r in ok if r["battery_mah"] == batt and r["span"] == span]
            if cand:
                b = max(cand, key=lambda r: r["endurance_min"])
                xs.append(span), ys.append(b["endurance_min"]), vs.append(b["v_stall"])
        ax[0].plot(xs, ys, "o-", label=f"3S {batt} mAh")
    ax[0].set_xlabel("span [m]"), ax[0].set_ylabel("best flight time [min]")
    ax[0].set_title("Best feasible design per span / battery"), ax[0].grid(alpha=0.3), ax[0].legend()
    sc = ax[1].scatter([r["wing_loading_g_dm2"] for r in rows], [r["endurance_min"] for r in rows],
                       c=[r["v_stall"] for r in rows], cmap="viridis", s=14,
                       marker="o", alpha=0.8)
    ax[1].scatter([best["wing_loading_g_dm2"]], [best["endurance_min"]], s=160, facecolors="none",
                  edgecolors="r", linewidths=2, label="chosen")
    fig.colorbar(sc, ax=ax[1], label="stall speed [m/s]")
    ax[1].set_xlabel("wing loading [g/dm^2]"), ax[1].set_ylabel("flight time [min]")
    ax[1].set_title("All candidates"), ax[1].grid(alpha=0.3), ax[1].legend()
    fig.tight_layout()
    fig.savefig(out / "sizing.png", dpi=130)
    print(f"Wrote {out}/ in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
