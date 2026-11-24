"""Verification checks for the conceptual sizing models.
Run from design/:   .venv/bin/python -m pytest conceptual-design/tests -q
"""
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requirements  # noqa: E402,F401
from aerodynamics import cf_flat_plate, oswald_inviscid  # noqa: E402
from atmosphere import G0, isa  # noqa: E402
from catalog import airfoil, battery_families, motors, props, real_batteries  # noqa: E402
from mission import ground_roll_landing, ground_roll_takeoff  # noqa: E402
from parameters import P  # noqa: E402
from propulsion import (V_GRID, Powertrain, bmf_climb, bmf_loiter,  # noqa: E402
                        bmf_specified_endurance)


# ---------------------------------------------------------------- atmosphere
def test_isa_sea_level_and_1000m():
    a = isa(0.0, 0.0)
    assert a.rho == pytest.approx(1.2250, abs=1e-4)
    assert a.T == pytest.approx(288.15)
    assert a.mu == pytest.approx(1.789e-5, rel=1e-3)
    b = isa(1000.0, 0.0)
    assert b.p == pytest.approx(89874.6, rel=1e-4)       # ISA table
    assert b.rho == pytest.approx(1.1117, rel=1e-3)


# ---------------------------------------------------------------- image BMF equations
def test_bmf_equations_reproduce_first_hand_calc():
    """Same inputs as the first version of conceptual.py (hand-checked)."""
    assert bmf_loiter(0.5, 54.0, 10.0, 150.0, 0.52, 0.8) == pytest.approx(0.1179, abs=2e-4)
    assert bmf_climb(120.0, 11.4, 2.5, 8.66, 150.0, 0.52, 0.8) == pytest.approx(0.0080, abs=1e-4)
    assert bmf_specified_endurance(30 / 3600, 0.125, 1.0, 150.0, 0.8, 0.8) == pytest.approx(0.0109, abs=1e-4)


def test_bmf_climb_equals_energy_balance():
    """Climb BMF must equal W h (1 + D V/(W Vv)) / eta, i.e. potential energy plus
    drag work over the climb time, divided by usable battery energy."""
    m, h, V, Vv, LD, U, eta, K = 1.0, 120.0, 12.0, 3.0, 9.0, 160.0, 0.5, 0.8
    E = m * G0 * h * (1 + V / (Vv * LD)) / eta / 3600          # Wh
    assert bmf_climb(h, V, Vv, LD, U, eta, K) == pytest.approx(E / (U * K * m), rel=1e-9)


# ---------------------------------------------------------------- aerodynamics
def test_skin_friction_limits():
    long_L = 100.0      # no roughness cutoff
    assert cf_flat_plate(1e5, long_L) == pytest.approx(1.328 / math.sqrt(1e5), rel=1e-9)
    turb = 0.455 / math.log10(1e7) ** 2.58 - 1700 / 1e7
    assert cf_flat_plate(1e7, long_L) == pytest.approx(turb, rel=1e-9)
    # roughness cutoff (Raymer 12.28) raises Cf on a short, rough part
    with P.override(roughness_k_m=1e-3):
        assert cf_flat_plate(1e7, 0.5) > turb


def test_oswald_inviscid_reasonable():
    e = oswald_inviscid(8.0, 0.45, 0.0, 1.0)
    assert 0.97 < e <= 1.0            # near-elliptic taper -> ~1
    assert oswald_inviscid(8.0, 1.0, 0.0, 1.0) < e      # rectangular is worse


def test_airfoil_polars_sane():
    af = airfoil("e387")
    assert 1.0 < float(af.clmax(2e5)) < 1.45
    assert 0.006 < float(af.cd(0.5, 2e5)) < 0.016
    assert float(af.cd(0.5, 5e4)) > float(af.cd(0.5, 3e5))   # low-Re penalty
    assert 5.5 < float(af.cl_alpha(2e5)) < 7.5


# ---------------------------------------------------------------- propulsion
def test_prop_reproduces_apc_table_point():
    p = props()["APC 10x7E"]
    ct, cp = p.coeffs(1000.0, 0.0)                 # first row of PER3_10x7E.dat
    assert float(ct) == pytest.approx(0.1197, abs=1e-4)
    assert float(cp) == pytest.approx(0.0625, abs=1e-4)


def test_motor_model_matches_three_constant_efficiency():
    m = motors()["Cobra C-2217/20"]
    batt = battery_families()[0].make(30.0)
    pt = Powertrain(m, props()["APC 10x5E"], batt, 1.2, 0.0)
    n = np.array([100.0, 150.0])                   # rev/s
    Q = np.array([0.05, 0.10])                     # N m
    I, Vm = pt._motor_state(n, Q * 2 * np.pi * n)
    eta = Q * 2 * np.pi * n / (Vm * I)
    eta_theory = (1 - m.io / I) * (1 - I * m.rm / Vm)
    assert np.allclose(eta, eta_theory, rtol=1e-9)


def test_powertrain_required_thrust_is_delivered():
    m = motors()["Cobra C-2814/16"]
    b = battery_families()[1].make(60.0)
    pt = Powertrain(m, props()["APC 10x7E"], b, 1.2, 10.0)
    T_req = np.full(V_GRID.shape, 2.0)
    r = pt.required(T_req)
    i = int(np.searchsorted(V_GRID, 12.0))
    assert r["ok"][i]
    # thrust power / shaft power is the prop efficiency, and it is physical
    assert 0.3 < r["eta_prop"][i] < 0.85
    assert 0.4 < r["eta_b2s"][i] < 0.95
    assert r["P_chem"][i] > r["P_shaft"][i] / 0.95


# ---------------------------------------------------------------- ground rolls vs closed form
class _NoDragAero:
    def __init__(self, rho):
        self.atm = SimpleNamespace(rho=rho)
        self.S = 0.3

    def drag(self, CL, V, ground=False, config="clean"):
        return np.zeros_like(np.asarray(V, float)), {}


def test_takeoff_roll_matches_closed_form():
    """Constant thrust, no drag, CL_g = 0: s = V^2 / (2 g (T/W - mu)) + V t_rot."""
    W, T, mu, V_lo = 10.0, 4.0, 0.1, 10.0
    ft = dict(T=np.full(V_GRID.shape, T), P_chem=np.zeros(V_GRID.shape),
              P_shaft=np.zeros(V_GRID.shape), v_term=np.ones(V_GRID.shape))
    pt = SimpleNamespace(full_throttle=lambda: ft, p_av_term=0.0, batt=SimpleNamespace(v_nom=1.0))
    s = ground_roll_takeoff(_NoDragAero(1.2), pt, W, 0.0, V_lo, mu)["s"]
    expected = V_lo ** 2 / (2 * G0 * (T / W - mu)) + V_lo * P.rotation_time_s
    assert s == pytest.approx(expected, rel=2e-3)


def test_landing_roll_matches_closed_form():
    """No brakes, no drag: s = V^2 / (2 mu g)."""
    s = ground_roll_landing(_NoDragAero(1.2), 10.0, 0.0, 11.0, 0.05)
    assert s == pytest.approx(11.0 ** 2 / (2 * 0.05 * G0), rel=2e-3)


# ---------------------------------------------------------------- full sizing loop
def test_sizing_loop_converges_and_balances():
    import sizing
    fam = [f for f in battery_families() if f.name == "LiPo 3S"][0]
    c = sizing.Combo("e387", "foam_glass", "Cobra C-2204/40", "APC 6x4E", family=fam)
    r = sizing.evaluate((0.14, 7.5, 2.6), c)
    assert r.ok and r.cons["converged"] == 0.0
    assert sum(r.detail["masses"].values()) == pytest.approx(r.m, abs=2e-4)
    b = r.detail["bmf"]
    # continuous battery is sized exactly to the mission
    assert r.detail["battery"].mass_kg == pytest.approx(b["battery_mass_required"], rel=2e-3)
    # loiter BMF from the image equation == loiter energy from the powertrain model
    lo, batt = r.detail["perf"]["loiter"], r.detail["battery"]
    e_lo = b["loiter"] * batt.u_wh_kg * batt.k * r.m          # Wh (propulsion chemical)
    assert e_lo == pytest.approx(lo["P_chem_prop"] * P.endurance_min / 60, rel=1e-3)


def test_performance_is_continuous_in_design_variables():
    """Regression: loiter speed used to snap to the 0.25 m/s grid when the
    1.2 V_stall bound was active, so rounding the design variables changed the
    landing roll by ~2.5%. Tiny input changes must give tiny output changes."""
    import sizing
    from catalog import real_batteries
    b = {x.name: x for x in real_batteries()}["Gens Ace Soaring 3S 2200 20C"]
    c = sizing.Combo("sd7062", "foam_glass", "Cobra C-2204/40", "APC 6x4E", battery=b)
    x = (0.08722, 5.64329, 2.51178)
    outs = []
    for k in (1.0, 1 + 3e-6, 1 - 3e-6):
        r = sizing.evaluate((x[0] * k, x[1], x[2]), c)
        p = r.detail["perf"]
        outs.append((p["loiter"]["V"], p["s_landing"], r.detail["bmf"]["loiter_max_min"]))
    for o in outs[1:]:
        assert abs(o[0] - outs[0][0]) < 1e-3
        assert abs(o[1] - outs[0][1]) < 0.05
        assert abs(o[2] - outs[0][2]) < 0.01


# ---------------------------------------------------------------- lifting line
def _ll_config(AR, kind, taper=1.0, twist_deg=0.0):
    from aerodynamics import LiftingLine, WingConfig
    b = 1.0
    S = b * b / AR
    if kind == "ellipse":
        cf = lambda y: 4 * S / (math.pi * b) * np.sqrt(np.maximum(1 - (2 * np.asarray(y) / b) ** 2, 1e-12))
    else:
        cr = 2 * S / (b * (1 + taper))
        cf = lambda y: cr * (1 - (1 - taper) * 2 * np.abs(np.asarray(y)) / b)
    ll = LiftingLine(b, S, cf, N=24)
    tw = -math.radians(twist_deg) * 2 * ll.y / b
    return WingConfig(ll, np.full(ll.N, 2 * math.pi), np.zeros(ll.N), tw, np.linspace(0.01, 0.49, 12))


def test_lifting_line_classical_results():
    w = _ll_config(6, "ellipse")
    assert w.e == pytest.approx(1.0, abs=1e-4)
    assert w.CLa == pytest.approx(2 * math.pi * 6 / 8, rel=1e-4)          # 2 pi AR / (AR + 2)
    w = _ll_config(6, "taper", 1.0)
    assert w.e == pytest.approx(0.95, abs=0.01)                          # Glauert, rectangular AR 6
    assert w.CLa == pytest.approx(4.5, abs=0.1)
    # agrees with the Nita & Scholz / Hoerner span efficiency used before
    for tp in (0.4, 0.6, 1.0):
        assert _ll_config(7.5, "taper", tp).e == pytest.approx(oswald_inviscid(7.5, tp, 0, 1), abs=0.01)


def test_washout_unloads_tip_and_is_only_right_at_one_cl():
    w0, w4 = _ll_config(7, "taper", 0.6, 0.0), _ll_config(7, "taper", 0.6, 4.0)
    assert w4.lb[-1] < w0.lb[-1]                    # less basic lift at the tip
    assert w4.e == pytest.approx(w0.e, rel=1e-9)     # additional loading is unchanged
    # taper 0.6 is tip-heavy vs. elliptic: washout HELPS at high CL ...
    assert w4.cdi(0.8) < w0.cdi(0.8)
    # ... but costs induced drag at low CL (twist is only "right" at one CL)
    assert w4.cdi(0.0) > w0.cdi(0.0) + 1e-5


# ---------------------------------------------------------------- control surfaces
def test_flap_section_physics():
    import controls as C
    re = 110000
    d = C.max_usable_deflection("sd7062", 0.25, re, 40)
    assert 8 <= d <= 30                               # trusted range is limited
    pr = C.FlapSection("sd7062", 0.25, d).props(re)
    assert pr["d_alpha0"] < 0                          # flap down = more camber
    assert pr["d_clmax"] > 0
    assert pr["d_cm0"] < 0                             # more nose-down moment
    tau = -math.degrees(pr["d_alpha0"]) / d           # flap effectiveness
    assert 0.25 < tau < 0.75                          # thin-airfoil theory ~0.6 at cf 0.25, less when large


def test_roll_helix_grows_with_aileron_span():
    import sizing
    b = {x.name: x for x in real_batteries()}["Gens Ace Soaring 3S 2200 20C"]
    r = sizing.evaluate((0.0803, 4.0, 2.5), sizing.Combo("sd7062", "foam_glass", "Cobra C-2204/40",
                                                       "APC 6x6E", battery=b, controls="none"))
    import controls as C
    g, a = r.detail["geom"], r.detail["aero"]
    p1 = C.roll_helix(g, a, 0.4, 0.7 * g.b / 2, 0.95 * g.b / 2)
    p2 = C.roll_helix(g, a, 0.4, 0.4 * g.b / 2, 0.95 * g.b / 2)
    assert p2 > p1 > 0
    assert r.detail["controls"]["aileron"]["pb2v"] >= P.roll_helix_min - 1e-6


def test_all_control_configs_converge_and_stability_is_sane():
    import sizing
    b = {x.name: x for x in real_batteries()}["Gens Ace Soaring 3S 2200 20C"]
    for cfg in ("none", "flaps", "flaperons"):
        r = sizing.evaluate((0.0803, 4.0, 2.5), sizing.Combo("sd7062", "foam_glass", "Cobra C-2204/40",
                                                           "APC 6x6E", battery=b, controls=cfg))
        assert r.ok and r.cons["converged"] == 0.0, cfg
        t = r.detail["tail"]
        assert 0.25 < t["h_n"] < 1.0                     # neutral point aft of the wing AC
        assert t["h_cg_fwd"] < t["h_cg_aft"] < t["h_n"]   # stable CG range
        p = r.detail["perf"]
        assert p["CLmax_landing"] >= p["CLmax"] - 1e-9
        if cfg != "none":
            assert p["CLmax_landing"] > p["CLmax"]


def test_continuous_liion_pack_never_below_one_string():
    """Regression: a continuous Li-ion pack with < 1 parallel string is not
    buildable; it once out-ranked every LiPo and then snapped to a 2x pack."""
    for fam in battery_families():
        if fam.chemistry != "Li-ion":
            continue
        small = fam.make(5.0)
        one_p = fam.options()[0]
        assert small.mass_kg >= one_p.mass_kg - 1e-9
        assert small.energy_wh >= one_p.energy_wh - 1e-9
