"""
Every input the sizing loop does NOT calculate lives here, with a tag that says
where it comes from. Nothing in this project is a bare "[ASSUMED]" value.

Tags
----
REQUIREMENT    from the README mission requirements
STANDARD       regulation / textbook standard (14 CFR Part 23, ISA, ...)
RAYMER         Raymer, "Aircraft Design: A Conceptual Approach" (table/section given)
DATASHEET      manufacturer / vendor data (see data/*.csv for parts)
DESIGN CHOICE  a decision the team makes (not a guess about physics)
SITE           depends on where you fly
ESTIMATE       best available estimate - flagged MEASURE so the team replaces it

Access values as  P.name . Use  P.override(name=value)  (context manager) for
sensitivity studies.
"""
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass(frozen=True)
class Param:
    name: str
    value: object
    units: str
    tag: str
    source: str
    measure: bool = False   # True -> team should measure/verify this


class _Registry:
    def __init__(self):
        object.__setattr__(self, "_defs", {})
        object.__setattr__(self, "_over", {})

    def define(self, name, value, units, tag, source, measure=False):
        self._defs[name] = Param(name, value, units, tag, source, measure)

    def __getattr__(self, name):
        if name in self._over:
            return self._over[name]
        try:
            return self._defs[name].value
        except KeyError:
            raise AttributeError(f"unknown parameter '{name}'") from None

    def __setattr__(self, name, value):
        raise AttributeError("use P.override(...) to change parameters")

    @contextmanager
    def override(self, **kw):
        for k in kw:
            if k not in self._defs:
                raise KeyError(k)
        old = dict(self._over)
        self._over.update(kw)
        try:
            yield
        finally:
            self._over.clear()
            self._over.update(old)

    def all(self):
        return list(self._defs.values())


P = _Registry()
d = P.define

# ---------------------------------------------------------------- site / atmosphere
d("field_elevation_m", 0.0, "m", "SITE", "Set to your flying field elevation (ISA)")
d("isa_temp_offset_K", 15.0, "K", "DESIGN CHOICE",
  "Hot day ISA+15 K (30 C at sea level): lower density -> conservative takeoff/climb")

# ---------------------------------------------------------------- mission / standards
d("climb_altitude_m", 120.0, "m", "DESIGN CHOICE", "400 ft AGL recreational ceiling")
d("min_roc_ms", 1.524, "m/s", "STANDARD",
  "14 CFR 23.65 (pre-amdt 64): all-engine ROC >= 300 ft/min")
d("min_climb_gradient", 1.0 / 12.0, "-", "STANDARD",
  "14 CFR 23.65 (pre-amdt 64): climb gradient >= 1:12")
d("climb_rate_target_ms", 2.5, "m/s", "DESIGN CHOICE",
  "Operational climb rate (~500 ft/min) for go-arounds, gusts and autopilot authority; "
  "stricter than the 300 ft/min regulatory minimum. Sizes the throttle cap")
d("throttle_cap_margin", 1.05, "-", "DESIGN CHOICE",
  "Thrust margin over the binding takeoff/climb requirement when sizing the throttle cap")
d("v_min_ratio", 1.2, "-", "STANDARD",
  "Loiter/climb speeds >= 1.2 V_stall (14 CFR 23 speed margins; Raymer Ch.17)")
d("v_liftoff_ratio", 1.1, "-", "RAYMER", "Raymer Sec. 17.8: V_TO = 1.1 V_stall")
d("rotation_time_s", 1.0, "s", "RAYMER", "Raymer Sec. 17.8: ~1 s rotation for small aircraft")
d("v_touchdown_ratio", 1.15, "-", "RAYMER", "Raymer Sec. 17.9: V_TD = 1.15 V_stall")
d("k_lipo", 0.80, "-", "DESIGN CHOICE", "Use 80% of LiPo capacity (cell health + reserve)")
d("k_liion", 0.85, "-", "DESIGN CHOICE", "Use 85% of Li-ion capacity (cutoff ~3.0 V/cell + reserve)")

# ---------------------------------------------------------------- ground (field survey 2026-11-24)
d("mu_roll_pavement", 0.04, "-", "RAYMER", "Raymer Table 17.1 dry concrete/asphalt rolling friction 0.03-0.05")
d("mu_grass_ref", 0.10, "-", "ESTIMATE",
  "Soft-grass rolling friction of the REFERENCE wheel below (Raymer Table 17.1 soft turf 0.07, "
  "raised for small RC wheels). Other wheels scale by Bekker compaction resistance. MEASURE: tow "
  "test (luggage-scale pull / weight) with TWO wheel sizes -> gives mu_grass_ref AND bekker_n",
  measure=True)
d("grass_ref_wheel_d_m", 0.0508, "m", "DESIGN CHOICE", "Reference wheel for mu_grass_ref: Du-Bro 200SL (2 in)")
d("grass_ref_wheel_w_m", 0.0175, "m", "DESIGN CHOICE", "Reference wheel width (200SL)")
d("grass_ref_wheel_load_n", 3.8, "N", "DESIGN CHOICE", "Reference wheel load (one main wheel of the v1 design)")
d("bekker_n", 0.8, "-", "ESTIMATE",
  "Bekker soil-deformation exponent n (Bekker 1962; Wong, Theory of Ground Vehicles, Table 2.3: "
  "loams ~0.5-1.1). Rolling resistance ~ W^(1/(2n+1)) b^(-1/(2n+1)) D^(-(n+1)/(2n+1))",
  measure=True)
d("mu_brake_pavement", 0.30, "-", "RAYMER",
  "Raymer Table 17.1 braking, dry concrete 0.3-0.5 (low end: small drag brakes may not reach "
  "wheel lock). MEASURE: braked tow test", measure=True)
d("mu_brake_grass", 0.20, "-", "RAYMER", "Raymer Table 17.1 braking, soft turf / wet grass 0.2. MEASURE", measure=True)
d("brake_delay_s", 1.0, "s", "RAYMER", "Raymer Sec. 17.9: free roll before the brakes act (~1 s small aircraft, 3 s large)")
d("rollout_crow", True, "-", "DESIGN CHOICE",
  "Crow on the landing rollout: FC raises both ailerons (flaps stay down) - drag + lift dump "
  "put more weight on the braked wheels")
d("crow_aileron_up_deg", 20.0, "deg", "DESIGN CHOICE",
  "Aileron up-deflection for crow (further limited by NeuralFoil confidence)")

# ---------------------------------------------------------------- aerodynamics
d("n_crit", 9.0, "-", "ESTIMATE",
  "XFOIL e^N transition, 9 = standard clean conditions; printed skins/turbulence may be "
  "lower", measure=True)
d("roughness_k_m", 3.0e-5, "m", "ESTIMATE",
  "Printed LW-PLA surface; cf. Raymer Table 12.4 (camouflage paint 1.0e-5 m)", measure=True)
d("re_transition_A", 1700.0, "-", "STANDARD",
  "Prandtl-Schlichting mixed laminar/turbulent Cf, transition Re_x = 5e5")
d("q_fuselage", 1.0, "-", "RAYMER", "Raymer Sec. 12.5.5 interference factor, fuselage")
d("q_nacelle", 1.5, "-", "RAYMER", "Raymer Sec. 12.5.5: nacelle mounted directly on wing")
d("q_tail", 1.04, "-", "RAYMER", "Raymer Sec. 12.5.5: conventional tail 1.04-1.05")
d("q_gear", 1.2, "-", "ESTIMATE", "Gear mutual/airframe interference")
d("dq_wheel", 0.25, "-", "RAYMER", "Raymer Table 12.5: regular wheel and tire, D/q per frontal area")
d("dq_flat_leg", 1.40, "-", "RAYMER", "Raymer Table 12.5: flat spring gear leg")
d("dq_round_strut", 0.30, "-", "RAYMER", "Raymer Table 12.5: round strut or wire")
d("leak_protub_frac", 0.05, "-", "ESTIMATE",
  "Leakage + protuberances (FPV camera, pitot, antennas), Raymer Sec. 12.5 few % of parasite")
d("re_tip_min", 70e3, "-", "DESIGN CHOICE",
  "Wing-tip chord Reynolds number at loiter >= 70k: below the ~7e4 critical Re, laminar "
  "separation stops reattaching and section data/stall become unreliable (Lissaman 1983, "
  "Annu. Rev. Fluid Mech. 15)")
d("wing_taper", 0.6, "-", "DESIGN CHOICE",
  "Raymer Sec. 4.3.4: 0.45 is near-elliptic; 0.6 keeps tip Reynolds number up and tip stall benign")
d("tail_cht", 0.50, "-", "RAYMER", "Raymer Table 6.4 homebuilt horizontal tail volume coefficient")
d("tail_cvt", 0.04, "-", "RAYMER", "Raymer Table 6.4 homebuilt vertical tail volume coefficient")
d("ar_htail", 4.0, "-", "RAYMER", "Raymer Table 4.3 horizontal tail aspect ratio 3-5")
d("taper_htail", 0.6, "-", "RAYMER", "Raymer Table 4.3 tail taper 0.3-0.6")
d("ar_vtail", 1.6, "-", "RAYMER", "Raymer Table 4.3 vertical tail aspect ratio 1.3-2.0")
d("taper_vtail", 0.5, "-", "RAYMER", "Raymer Table 4.3 tail taper 0.3-0.6")
d("tail_airfoil", "naca0009", "-", "DESIGN CHOICE", "Symmetric 9% section for tails")
d("engine_out_sizing", True, "-", "DESIGN CHOICE",
  "Size the fin/rudder so one motor at full throttle can be held at V_mc")
d("vmc_ratio", 1.2, "-", "STANDARD", "14 CFR 23.149: V_MC <= 1.2 V_stall")
d("rudder_max_deg", 25.0, "deg", "DESIGN CHOICE", "Maximum rudder deflection (servo throw)")
d("rudder_span_frac", 0.9, "-", "DESIGN CHOICE", "Rudder height / fin height")
d("foil_conf_min", 0.7, "-", "ESTIMATE",
  "Minimum NeuralFoil analysis confidence trusted for the deflected-rudder section: beyond it "
  "the flow separates over the rudder and XFOIL-type methods over-predict side force")
# ---------------------------------------------------------------- control surfaces
d("roll_helix_min", 0.07, "-", "STANDARD",
  "Roll performance pb/2V >= 0.07 with full aileron (Gilruth, NACA Report 755, 1943) - "
  "a non-dimensional criterion, so it scales correctly to a small UAV")
d("approach_speed_ratio", 1.3, "-", "RAYMER", "Raymer Ch.17: approach speed = 1.3 V_stall (landing config)")
d("control_cf_grid", (0.20, 0.25, 0.30, 0.35), "-", "DESIGN CHOICE",
  "Aileron / flap chord fractions considered (hinge must stay ahead of the thin trailing edge)")
d("elevator_cf_grid", (0.25, 0.30, 0.35, 0.40, 0.45, 0.50), "-", "DESIGN CHOICE",
  "Elevator chord fractions considered")
d("rudder_cf_grid", (0.30, 0.35, 0.40, 0.45, 0.50), "-", "DESIGN CHOICE", "Rudder chord fractions considered")
d("aileron_max_deg", 20.0, "deg", "DESIGN CHOICE", "Aileron throw (each direction)")
d("aileron_outboard_eta", 0.95, "-", "DESIGN CHOICE", "Aileron/flaperon outboard edge (fraction of semispan)")
d("flap_aileron_gap_eta", 0.02, "-", "DESIGN CHOICE", "Gap between flap and aileron (fraction of semispan)")
d("flap_max_deg", 40.0, "deg", "DESIGN CHOICE", "Maximum landing flap deflection (further limited by analysis confidence)")
d("flaperon_max_droop_deg", 20.0, "deg", "DESIGN CHOICE",
  "Maximum flaperon droop - leaves aileron travel about the drooped position")
d("elevator_max_deg", 25.0, "deg", "DESIGN CHOICE", "Elevator throw (trailing edge up)")
d("elevator_span_frac", 0.95, "-", "DESIGN CHOICE", "Elevator span / stabilizer span")
d("tail_efficiency", 0.9, "-", "RAYMER", "Raymer Ch.16: tail dynamic-pressure ratio ~0.9 (conventional tail)")
d("static_margin_min", 0.05, "-", "DESIGN CHOICE", "Aft CG limit: static margin >= 5% MAC")
d("static_margin_max", 0.20, "-", "DESIGN CHOICE", "Forward CG limit: static margin <= 20% MAC (sizes the elevator)")
d("tipback_deg", 15.0, "deg", "RAYMER", "Raymer Ch.11: main gear >= 15 deg behind the aft CG (tip-back angle)")
d("flap_servo_mass_kg", 0.015, "kg", "ESTIMATE", "Extra flap servo + horn + hinges, each (2 per aircraft)")
d("washout_max_deg", 6.0, "deg", "DESIGN CHOICE", "Upper bound on wing washout considered by the optimizer")
d("vtail_ar_eff_factor", 1.55, "-", "RAYMER",
  "Raymer Ch.16: fin effective aspect ratio ~1.55 x geometric (fuselage/stabilizer end-plate)")
d("htail_sizing", "scissor", "-", "DESIGN CHOICE",
  "'scissor': stabilizer area = smallest meeting static stability at the AFT CG and elevator "
  "control at the FORWARD CG of the computed CG band (Torenbeek/Raymer scissor plot). "
  "'tail_volume': Raymer Table 6.4 c_HT floor + elevator (old behaviour)")
d("cg_coverage_sigma", 2.0, "-", "DESIGN CHOICE", "CG band covers +/- 2 sigma of the as-built CG scatter (95%)")
d("battery_slide_m", 0.010, "m", "DESIGN CHOICE",
  "Battery tray travel each way: re-balances the as-built aircraft on the bench")
d("cg_balance_tol_m", 0.002, "m", "ESTIMATE",
  "Accuracy of finding the CG on a balancer (each way). MEASURE: repeat the balance 5x", measure=True)
d("component_position_tol_m", 0.003, "m", "ESTIMATE", "As-built component placement scatter (1 sigma)")
d("mass_sigma_datasheet", 0.02, "-", "ESTIMATE", "1-sigma mass scatter of datasheet parts (motors, ESCs, cells)")
d("mass_sigma_estimate", 0.20, "-", "ESTIMATE",
  "1-sigma mass scatter of ESTIMATE-tagged parts (until weighed)", measure=True)
d("mass_sigma_structure", 0.15, "-", "ESTIMATE",
  "1-sigma mass scatter of printed/laminated structure (until test panels are weighed)", measure=True)
d("crosswind_ratio", 0.2, "-", "STANDARD",
  "14 CFR 23.233: demonstrated 90-deg crosswind of 0.2 V_SO for takeoff/landing")
d("engine_out_bank_max_deg", 5.0, "deg", "STANDARD",
  "14 CFR 23.149: V_MC with not more than 5 deg of bank toward the operating engine")
d("trim_aileron_max_frac", 0.75, "-", "DESIGN CHOICE",
  "Steady trim (engine-out, crosswind) may use at most 75% of aileron throw - the rest is kept "
  "for maneuvering and gusts")
d("prop_solidity", 0.10, "-", "ESTIMATE",
  "Blade area / disk area of the APC E props (for stopped-prop drag). MEASURE from a blade outline",
  measure=True)
d("stopped_prop_dq_per_solidity", 0.8, "-", "RAYMER",
  "Raymer Sec. 12.5: stopped (not feathered) propeller D/q = 0.8 * solidity * disk area. "
  "Set the ESC to brake the motor (a free-windmilling prop has less drag, but less yaw too)")
d("fus_ki_sideforce", 1.5, "-", "RAYMER",
  "DATCOM/Raymer Ch.16 wing-body interference factor K_i for fuselage side force, high wing ~1.5")

# ---------------------------------------------------------------- objective
d("stall_speed_weight_g_per_ms", 30.0, "g/(m/s)", "DESIGN CHOICE",
  "Team decision 2026-11-24: 1 m/s of (trimmed, landing-config) stall speed is worth 30 g. "
  "The optimizer minimizes  mass + this x V_stall  instead of mass alone (mass alone always "
  "shrinks the wing to the fastest stall speed any constraint allows). See the stall-speed trade study")
d("stall_speed_max_ms", None, "m/s", "DESIGN CHOICE",
  "Optional hard cap on stall speed - None in the main search; used by the stall-speed trade study")
d("balked_landing_gradient_min", 0.033, "-", "STANDARD",
  "14 CFR 23.77(a) (pre-amdt 64): balked-landing climb gradient >= 3.3% in landing configuration")

# ---------------------------------------------------------------- geometry / layout
d("prop_fuselage_clearance_m", 0.025, "m", "DESIGN CHOICE", "Prop tip to fuselage side (1 in)")
d("prop_ground_clearance_m", 0.05, "m", "DESIGN CHOICE", "Prop tip to ground on grass (2 in)")
d("internal_clearance_m", 0.005, "m", "DESIGN CHOICE", "Gap around the payload bay contents, per side")
d("fuselage_packing_factor", 2.0, "-", "DESIGN CHOICE",
  "Bay volume / sum of component volumes (wiring, airflow, access)")
d("pi_stack_height_m", 0.030, "m", "DESIGN CHOICE", "Pi 5 + cooler stacked above the battery tray")
d("nose_length_ratio", 0.6, "-", "DESIGN CHOICE", "Nose fairing length / fuselage equivalent diameter")
d("nacelle_extra_len_m", 0.07, "m", "DESIGN CHOICE", "Nacelle length beyond motor: ESC + fairing")
d("nacelle_wall_gap_m", 0.004, "m", "DESIGN CHOICE", "Nacelle radial clearance around motor")
d("max_motor_station_frac", 0.45, "-", "DESIGN CHOICE", "Motor station y_eng <= 0.45 b (90% semispan)")
d("main_wheel", "Du-Bro 300SL", "-", "DESIGN CHOICE",
  "Main wheels (data/wheels.csv): 3 in for soft grass - lowest rolling resistance and sinkage, "
  "~10 g/wheel more than 2 in, inside the E/Z brake fit range (see wheel trade table)")
d("nose_wheel", "Du-Bro 300SL", "-", "DESIGN CHOICE",
  "Nose wheel = main wheel size so it does not dig in on soft grass")
d("wheel_brake", "Du-Bro E/Z Brake", "-", "DESIGN CHOICE",
  "Main-wheel brakes (data/brakes.csv): cable brakes on one micro servo (~+28 g); JP Hobby "
  "electric brakes rejected (~200 g)")

# ---------------------------------------------------------------- structure / weights
d("wing_constructions", ("printed_lwpla",), "-", "DESIGN CHOICE",
  "Wing constructions the search may use. Team decision 2026-11-24: print most of the aircraft in LW-PLA, "
  "including the wing (only the carbon spar, aluminum gear strap and music-wire nose strut are not printed). "
  "Add 'foam_glass' to let the search compare a hot-wire foam core + glass skin wing again")
d("wing_skin_t_m", 0.00045, "m", "DESIGN CHOICE", "Single-wall LW-PLA wing skin")
d("tail_skin_t_m", 0.00045, "m", "DESIGN CHOICE", "Single-wall LW-PLA tail skin")
d("fus_skin_t_m", 0.0006, "m", "DESIGN CHOICE", "LW-PLA fuselage wall")
d("nacelle_skin_t_m", 0.0006, "m", "DESIGN CHOICE", "LW-PLA nacelle wall")
d("wing_internal_ratio", 0.40, "-", "ESTIMATE",
  "Printed internal ribs/geodesic web area per wetted area. MEASURE on a printed section",
  measure=True)
d("tail_internal_ratio", 0.30, "-", "ESTIMATE", "Printed tail internal web area ratio", measure=True)
d("fus_frames_ratio", 0.25, "-", "ESTIMATE", "Fuselage frames/longerons mass / shell mass", measure=True)
d("spar_span_frac", 0.9, "-", "DESIGN CHOICE", "Main spar tube length / span")
d("spar_min_wall_m", 0.0005, "m", "DESIGN CHOICE", "Thinnest practical carbon tube wall")
d("spar_min_od_m", 0.006, "m", "DESIGN CHOICE", "Smallest spar tube OD (handling stiffness)")
d("spar_depth_frac", 0.85, "-", "DESIGN CHOICE", "Spar OD <= 85% of root airfoil thickness")
d("n_limit_maneuver", 3.8, "-", "STANDARD", "14 CFR 23.337 normal category limit load factor")
d("gust_ude_ms", 15.24, "m/s", "STANDARD", "14 CFR 23.341: 50 ft/s derived gust at V_C")
d("factor_of_safety", 1.5, "-", "STANDARD", "14 CFR 23.303 ultimate / limit")
d("n_landing_limit", 3.0, "-", "ESTIMATE", "Landing limit load factor (14 CFR 23.473 >= 2.67)")
d("main_gear_load_frac", 0.9, "-", "RAYMER", "Raymer Ch.11: nose gear carries 8-15% of weight")
d("gear_plate_width_m", 0.020, "m", "DESIGN CHOICE", "Main-gear strap width")
d("main_gear_material", "petg_printed", "-", "DESIGN CHOICE",
  "Team decision 2026-11-24: every part 3D printed or off-the-shelf -> main gear strap printed in PETG "
  "(data/materials.csv), thickness sized from the landing load. 'aluminum_6061_t6' = bent strap from stock")
d("spar_from_catalog", True, "-", "DESIGN CHOICE",
  "Spar = lightest off-the-shelf carbon tube in data/carbon_tubes.csv that carries the load and fits the root "
  "(False = any diameter/wall, the old idealized sizing)")
d("nose_gear_type", "printed_petg", "-", "DESIGN CHOICE",
  "'printed_petg': printed PETG leg + wheel fork (sized from the nose load, min diameter below), printed "
  "mounting/bearing blocks + steering arm, off-the-shelf bolts and axle. 'catalog': shortest off-the-shelf "
  "steerable assembly in data/nose_gear.csv (Du-Bro, 55-62 g). 'music_wire': bent wire + steering block")
d("printed_nose_leg_min_d_m", 0.008, "m", "DESIGN CHOICE",
  "Minimum printed nose-leg diameter: the nose sizing load is an ESTIMATE and printed legs break at layer "
  "lines (print the leg lying flat so the layers run along it)")
d("printed_fork_section_m", (0.008, 0.004), "m", "DESIGN CHOICE", "Printed nose-wheel fork tine/bridge section (w x t)")
d("printed_nose_mount_mass_kg", 0.006, "kg", "ESTIMATE",
  "Printed PETG mounting/bearing blocks + steering arm for the nose leg. WEIGH", measure=True)
d("nose_gear_hardware_mass_kg", 0.004, "kg", "ESTIMATE", "Off-the-shelf axle bolt + 4 x M3 mounting bolts/nuts")
d("nose_steer_mass_kg", 0.010, "kg", "ESTIMATE", "Nose gear steering block + collar")
d("motor_mount_mass_kg", 0.008, "kg", "ESTIMATE", "Printed/ply motor mount, each")
d("spinner_mass_kg", 0.006, "kg", "ESTIMATE", "Prop adapter + spinner, each")
d("power_wire_kg_m", 0.015, "kg/m", "ESTIMATE", "16 AWG silicone wire, per conductor-metre")
d("prop_mass_10in_kg", 0.024, "kg", "ESTIMATE",
  "APC E 10 in prop mass (APC site blocks scraping). WEIGH props", measure=True)
d("prop_mass_exponent", 2.2, "-", "ESTIMATE", "prop mass ~ D^2.2")
d("weight_growth_allowance", 0.10, "-", "DESIGN CHOICE",
  "Mass growth allowance on everything except the battery (glue, paint, connectors, "
  "build variation) - standard conceptual-design practice; reduce as parts are weighed")
d("avionics_power_factor", 1.0, "-", "ESTIMATE",
  "Multiplier on data/components.csv power draws (Pi 5 workload is the big unknown). "
  "MEASURE the avionics current on the bench", measure=True)

# ---------------------------------------------------------------- propulsion
d("eta_esc", 0.95, "-", "ESTIMATE", "ESC efficiency. MEASURE on thrust stand", measure=True)
d("eta_bec", 0.90, "-", "ESTIMATE", "Switching BEC efficiency for avionics")
d("esc_margin", 1.1, "-", "DESIGN CHOICE", "ESC rating >= 1.1 x static full-throttle current")
d("lipo_v_nom", 3.7, "V", "DATASHEET", "LiPo nominal cell voltage")
d("liion_v_nom", 3.6, "V", "DATASHEET", "Molicel nominal cell voltage")
d("lipo_sag_at_rated_v", 0.4, "V", "ESTIMATE",
  "Cell sag at the rated C current -> R_cell = 0.4/(C*Ah). MEASURE with charger IR",
  measure=True)
d("liion_pack_overhead_kg", 0.020, "kg", "ESTIMATE", "Li-ion pack leads, XT60, wrap, balance lead")
d("liion_mass_per_cell_kg", 0.002, "kg", "ESTIMATE", "Nickel strip + insulator per cell")
d("liion_current_derate", 0.6, "-", "DESIGN CHOICE",
  "Use 60% of the cell's temperature-limited max current as continuous limit")
d("lipo_current_derate", 0.8, "-", "DESIGN CHOICE", "Use 80% of the label C-rating as continuous limit")
del d
