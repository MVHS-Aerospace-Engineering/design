"""
Constraint diagram (Raymer Ch.5 / Ch.17, power-based form for a prop aircraft):
installed full-throttle shaft power per weight  P/W [W/N]  vs  wing loading W/S [N/m^2].

Each line holds the converged design's aerodynamics (parabolic polar fit,
CLmax, ground-roll CL) and the SHAPE of its real thrust-vs-speed curve fixed,
and asks how much installed power (thrust curve scaled by k) a given wing
loading needs to meet that requirement:
  - takeoff ground roll <= 250 ft       (numerical integration, grass mu)
  - landing ground roll <= 250 ft       (no brakes -> vertical line, needs no power)
  - climb rate   >= 300 ft/min          (14 CFR 23.65)
  - climb gradient >= 1:12              (14 CFR 23.65)
  - loiter shaft power (reference, not a limit - endurance is an energy constraint)
"""
import math

import numpy as np

from atmosphere import G0
from parameters import P
from propulsion import V_GRID


def compute(res, n=60):
    """Lines use the SAME drag build-up as the sizing (aero.drag), holding the
    design's drag coefficients fixed while W/S varies (standard constraint-
    diagram assumption), so the design point sits exactly on its active lines."""
    d = res.detail
    aero, perf, pt = d["aero"], d["perf"], d["pt"]
    rho = d["atm"].rho
    W = res.m * G0
    CD0, K = aero.polar_fit(perf["loiter"]["V"])          # reported only
    CLmax = perf["CLmax"]
    CL_g = perf["CL_ground"]
    ft = pt.full_throttle()
    shape = ft["T"] / ft["T"][0]                     # thrust lapse with speed
    P0 = ft["P_shaft"][0]                             # installed static shaft power
    T0 = ft["T"][0]
    ws_design = W / aero.S
    ws = np.linspace(0.3 * ws_design, 2.2 * ws_design, n)

    def cd_ground(V):
        return aero.drag(np.full_like(V, CL_g), V, ground=True)[0]

    def roll_to(k, ws_):
        S = W / ws_
        Vs = math.sqrt(2 * ws_ / (rho * CLmax))
        V = np.linspace(0, P.v_liftoff_ratio * Vs, 80)
        q = 0.5 * rho * V ** 2
        T = k * T0 * np.interp(V, V_GRID, shape)
        a = G0 / W * (T - q * S * cd_ground(V) - P.mu_roll_takeoff * np.maximum(W - q * S * CL_g, 0))
        if np.any(a <= 0.05):
            return math.inf
        return np.trapezoid(V / a, V) + V[-1] * P.rotation_time_s

    def roll_ldg(ws_):
        S = W / ws_
        Vs = math.sqrt(2 * ws_ / (rho * CLmax))
        V = np.linspace(0, P.v_touchdown_ratio * Vs, 80)
        q = 0.5 * rho * V ** 2
        a = G0 / W * (q * S * cd_ground(V) + P.mu_roll_landing * np.maximum(W - q * S * CL_g, 0))
        return np.trapezoid(V / a, V)

    out = dict(ws=ws, takeoff=[], climb_rate=[], climb_grad=[], loiter=[])
    eta_p = perf["loiter"]["eta_prop"]
    for w_s in ws:
        S = W / w_s
        Vs = math.sqrt(2 * w_s / (rho * CLmax))
        V = V_GRID[(V_GRID >= P.v_min_ratio * Vs) & (shape > 0.05)]
        q = 0.5 * rho * V ** 2
        CL = W / (q * S)
        D = q * S * aero.drag(CL, V)[0]
        T1 = T0 * np.interp(V, V_GRID, shape)            # k = 1 thrust
        k_roc = np.min((W * P.min_roc_ms + D * V) / (T1 * V))
        k_grad = np.min((W * P.min_climb_gradient + D) / T1)
        lo, hi = 0.01, 20.0
        if roll_to(hi, w_s) > P.takeoff_roll_max_m:
            k_to = math.inf
        else:
            for _ in range(40):
                mid = math.sqrt(lo * hi)
                lo, hi = (mid, hi) if roll_to(mid, w_s) > P.takeoff_roll_max_m else (lo, mid)
            k_to = hi
        out["takeoff"].append(k_to * P0 / W)
        out["climb_rate"].append(k_roc * P0 / W)
        out["climb_grad"].append(k_grad * P0 / W)
        out["loiter"].append(np.min(D * V) / eta_p / W)
    # landing: largest W/S meeting the roll
    lo, hi = ws[0] * 0.2, ws[-1] * 3
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if roll_ldg(mid) <= P.landing_roll_max_m else (lo, mid)
    out["landing_ws_max"] = lo
    out["design"] = (ws_design, P0 / W)
    out["CD0"], out["K"] = CD0, K
    for k in ("takeoff", "climb_rate", "climb_grad", "loiter"):
        out[k] = np.array(out[k])
    return out


def plot(cd, path, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ws = cd["ws"]
    req = np.maximum.reduce([cd["takeoff"], cd["climb_rate"], cd["climb_grad"]])
    top = max(np.nanmax(np.where(np.isfinite(req), req, np.nan)) * 1.3, cd["design"][1] * 1.4)
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=140)
    ax.fill_between(ws, np.where(np.isfinite(req), req, top), top,
                    where=ws <= cd["landing_ws_max"], color="#2a9d8f", alpha=0.12,
                    label="feasible region")
    ax.plot(ws, cd["takeoff"], color="#e76f51", lw=2, label="takeoff roll limit")
    ax.plot(ws, cd["climb_rate"], color="#264653", lw=2, label="climb rate >= 300 ft/min")
    ax.plot(ws, cd["climb_grad"], color="#8d99ae", lw=2, ls="-.", label="climb gradient >= 1:12")
    ax.plot(ws, cd["loiter"], color="#e9c46a", lw=2, ls=":", label="loiter shaft power (reference)")
    ax.axvline(cd["landing_ws_max"], color="#6a4c93", lw=2, label="landing roll limit (no brakes)")
    ax.plot(*cd["design"], marker="*", ms=16, color="#d62828", mec="k", ls="none",
            label=f"design point ({cd['design'][0]:.0f} N/m², {cd['design'][1]:.1f} W/N)")
    ax.set_xlim(ws[0], ws[-1])
    ax.set_ylim(0, top)
    ax.set_xlabel("Wing loading W/S  [N/m²]")
    ax.set_ylabel("Installed shaft power / weight  P/W  [W/N]")
    ax.set_title(title or "Constraint diagram")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
