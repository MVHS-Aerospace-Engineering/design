"""Design inputs for the twin-engine high-wing RC plane.

The plane is sized around fixed hardware:

* 2x Flite Test Radial 2218 1180 Kv v2 (76 g, 27.5 x 34 mm, 3S-4S) with
  2-blade 10x4.7 slow-fly props, counter-rotating.
* A Raspberry Pi 5 (with active cooler) and a dedicated flight-controller PCB in
  the fuselage.

Only the numbers in `DesignVars` are "free". `layout.py` derives everything else
from them and the hardware below:
fuselage cross-section from the avionics bay, flight controller at the CG,
wing position for the static-margin target, tail sizes from volume
coefficients plus the one-engine-out rudder requirement, and the motor spacing
from prop clearance. `python -m twin_rc.sizing` sweeps `DesignVars` and
reports the best combination. The defaults below are that result.

Units are SI (m, kg, N, s) unless the name says otherwise.
"""

from dataclasses import dataclass, field


@dataclass
class Box:
    """A rectangular component: length (x) x width (y) x height (z), and mass."""

    length: float
    width: float
    height: float
    mass: float


@dataclass
class DesignVars:
    # Chosen by `python -m twin_rc.sizing` (max flight time; results/sizing/).
    span: float = 1.80  # sweep cap: fits a car, builds from 3 foam-board panels
    root_chord: float = 0.23
    taper: float = 0.70  # tip chord / root chord
    tail_arm_over_mac: float = 3.0  # wing AC -> stabilator AC, in MACs
    battery_mah: int = 5000
    static_margin: float = 0.12  # target, fraction of MAC
    Vh: float = 0.45  # horizontal tail volume (all-moving tails need less than a fixed stab + elevator)
    Vv_min: float = 0.040  # vertical tail volume floor; the engine-out case may need more
    # Full-power engine-out must be holdable with full rudder down to this x stall.
    # Below it, the flight controller caps the live motor's throttle (see
    # results/engine_out.png). Holding full power down to 1.3 x stall would need a
    # fin bigger than both stabilators combined.
    vmc_over_vs: float = 1.50


@dataclass
class WingConfig:
    taper_start_frac: float = 0.45  # fraction of semispan that stays rectangular
    dihedral_deg: float = 2.0  # high wing: pendulum stability means little dihedral is needed
    incidence_deg: float = 1.0
    airfoil: str = "clarky"
    aileron_hinge: float = 0.75
    aileron_span_frac: tuple = (0.55, 0.95)
    areal_mass: float = 1.00  # kg/m^2 of planform, foam-board folded airfoil + spar


@dataclass
class StabilatorConfig:
    """Two independent all-moving tail halves on pivot tubes (no fixed stab + elevator)."""

    aspect_ratio: float = 4.2  # of the pair, tip to tip
    taper: float = 0.70
    sweep_le_deg: float = 8.0
    airfoil: str = "naca0009"
    pivot_frac: float = 0.25
    max_deflection_deg: float = 20.0
    areal_mass: float = 0.70
    pivot_hardware_mass: float = 0.010  # per side: carbon tube + bearing + horn


@dataclass
class FinConfig:
    aspect_ratio: float = 1.5  # height^2 / area
    taper: float = 0.55
    sweep_le_deg: float = 25.0
    airfoil: str = "naca0009"
    rudder_hinge: float = 0.60  # rudder = aft 40 %
    max_rudder_deg: float = 25.0
    areal_mass: float = 0.70


@dataclass
class PropulsionConfig:
    """FT Radial 2218 1180 Kv v2 + 2-blade 10x4.7 slow-fly prop, per side.

    Motor: Kv, mass and size from the Flite Test listing. Winding resistance and
    no-load current aren't published; they are typical values for 2218-size
    motors. Prop: static Ct/Cp calibrated so this model matches published 2218
    bench data (EMAX GT2218 1100 Kv + 10x4.7: 1290 g at 24 A on 3S) to ~4 %.
    Replace these with your own thrust-stand numbers when you have them.
    """

    motor_name: str = "Flite Test Radial 2218 1180Kv v2"
    motor_kv: float = 1180.0
    motor_mass: float = 0.076
    motor_diameter: float = 0.0275
    motor_length: float = 0.034
    motor_resistance: float = 0.115  # ohm (estimated)
    motor_no_load_current: float = 0.7  # A (estimated)
    prop_diameter_in: float = 10.0
    prop_pitch_in: float = 4.7
    n_blades: int = 2
    prop_mass: float = 0.014
    ct_static: float = 0.107
    cp_static: float = 0.0435
    j_zero_thrust: float = 0.62  # advance ratio where thrust reaches zero (~1.3 x pitch/diameter)
    esc_current_rating: float = 40.0  # A. Static draw is ~27 A, too close to a 30 A ESC.
    esc_mass: float = 0.040
    esc_resistance: float = 0.008
    battery_cells: int = 3
    cell_voltage: float = 3.7
    prop_tip_clearance: float = 0.025  # prop tip to fuselage side
    usable_battery_fraction: float = 0.80


# 3S LiPo packs: capacity (mAh) -> Box. Typical sizes and masses.
BATTERIES = {
    2200: Box(0.105, 0.034, 0.025, 0.185),
    3000: Box(0.130, 0.043, 0.022, 0.245),
    4000: Box(0.138, 0.044, 0.030, 0.320),
    5000: Box(0.145, 0.048, 0.032, 0.400),
}


@dataclass
class AvionicsConfig:
    """Fuselage payload. Boxes include connector overhang."""

    pi5: Box = field(default_factory=lambda: Box(0.085, 0.056, 0.028, 0.070))  # board + active cooler
    flight_controller: Box = field(default_factory=lambda: Box(0.036, 0.036, 0.012, 0.015))
    bec: Box = field(default_factory=lambda: Box(0.035, 0.020, 0.010, 0.020))  # 5 V / 5 A for the Pi
    receiver: Box = field(default_factory=lambda: Box(0.035, 0.022, 0.010, 0.010))
    # Continuous electrical draw (W) besides the motors
    pi5_power: float = 6.0
    fc_power: float = 1.0
    servo_receiver_power: float = 1.5
    bec_efficiency: float = 0.88
    servo_mass: float = 0.009  # 9 g micro servos: 2 aileron, 2 stabilator, 1 rudder
    n_servos: int = 5


@dataclass
class FuselageConfig:
    wall: float = 0.005  # 5 mm foam board
    clearance: float = 0.004  # around every component
    floor_gap: float = 0.006  # wiring under the trays
    nose_length: float = 0.05  # taper ahead of the battery
    gap: float = 0.010  # between components along x
    tail_boom_width: float = 0.030
    tail_boom_height: float = 0.034
    section_exponent: float = 4.0  # superellipse: 2 = ellipse, large = box. Foam board is boxy.
    areal_mass: float = 0.60
    structure_extra: float = 0.040  # formers, hatch, magnets
    landing_gear_mass: float = 0.070
    wiring_mass: float = 0.050


@dataclass
class PlaneConfig:
    design: DesignVars = field(default_factory=DesignVars)
    wing: WingConfig = field(default_factory=WingConfig)
    stabilator: StabilatorConfig = field(default_factory=StabilatorConfig)
    fin: FinConfig = field(default_factory=FinConfig)
    propulsion: PropulsionConfig = field(default_factory=PropulsionConfig)
    avionics: AvionicsConfig = field(default_factory=AvionicsConfig)
    fuselage: FuselageConfig = field(default_factory=FuselageConfig)
    cruise_speed: float | None = None  # None -> fly at the max-endurance speed
    altitude: float = 50.0
    max_stall_speed: float = 8.5  # m/s, sizing constraint (hand launch / small field)
