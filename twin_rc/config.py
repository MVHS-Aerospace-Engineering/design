"""Design parameters for the twin-engine high-wing RC plane.

Every dimension is in SI units (m, kg, N, s) unless the name says otherwise.
Change numbers here and re-run `python -m twin_rc.run_all`; nothing else in the
package hard-codes geometry.
"""

from dataclasses import dataclass, field


@dataclass
class WingConfig:
    span: float = 1.60
    root_chord: float = 0.28
    tip_chord: float = 0.20
    taper_start_frac: float = 0.45  # fraction of semispan that stays rectangular
    dihedral_deg: float = 2.0  # high wing: little dihedral needed (pendulum effect)
    incidence_deg: float = 0.5
    airfoil: str = "clarky"
    x_le: float = 0.30  # leading edge station from the nose
    z: float = 0.10  # top of fuselage -> high wing
    aileron_hinge: float = 0.75  # x/c
    aileron_span_frac: tuple = (0.55, 0.95)


@dataclass
class StabilatorConfig:
    """Two independent all-moving horizontal tail halves (no fixed stab + elevator).

    Each half pivots about `pivot_frac` of its root chord. Deflecting both together
    gives pitch control; deflecting them differentially also gives some roll.
    """

    semispan: float = 0.30
    root_chord: float = 0.17
    tip_chord: float = 0.12
    sweep_le_deg: float = 8.0
    airfoil: str = "naca0009"
    x_le: float = 1.00
    z: float = 0.02
    y_root: float = 0.025  # gap from the centreline (half the fuselage width at the tail)
    pivot_frac: float = 0.25  # pivot at quarter chord
    max_deflection_deg: float = 20.0


@dataclass
class FinConfig:
    height: float = 0.24
    root_chord: float = 0.22
    tip_chord: float = 0.12
    sweep_le_deg: float = 25.0
    airfoil: str = "naca0009"
    x_le: float = 0.98
    z_root: float = 0.04
    rudder_hinge: float = 0.60  # x/c; rudder is the aft 40 % of the fin
    max_rudder_deg: float = 25.0


@dataclass
class FuselageConfig:
    length: float = 1.22
    max_width: float = 0.095
    max_height: float = 0.11


@dataclass
class PropulsionConfig:
    """Two wing-mounted tractor motors, each turning a 2-blade propeller.

    The props counter-rotate (left CW, right CCW seen from behind) so the torque
    and P-factor cancel.
    """

    motor_y: float = 0.30  # spanwise station of each nacelle
    motor_x: float = 0.18  # propeller plane station (ahead of the wing LE)
    motor_z: float = 0.07
    prop_diameter_in: float = 10.0
    prop_pitch_in: float = 6.0
    n_blades: int = 2
    motor_kv: float = 1000.0  # rpm / V
    motor_resistance: float = 0.08  # ohm
    motor_no_load_current: float = 0.9  # A
    battery_cells: int = 3  # LiPo S-count
    cell_voltage: float = 3.7
    battery_capacity_mah: float = 2200.0
    esc_resistance: float = 0.01


@dataclass
class MassConfig:
    """Component masses (kg) and CG stations (x from the nose, m)."""

    items: dict = field(
        default_factory=lambda: {
            # name: (mass, x, y, z). For the *_L / *_R motor, prop and ESC items, y is
            # overwritten with -/+ PropulsionConfig.motor_y in mass.py.
            "wing": (0.360, 0.41, 0.0, 0.10),
            "fuselage": (0.300, 0.52, 0.0, 0.0),
            "stabilators": (0.060, 1.05, 0.0, 0.02),
            "fin_rudder": (0.040, 1.07, 0.0, 0.13),
            "motor_L": (0.110, 0.24, -0.36, 0.07),
            "motor_R": (0.110, 0.24, 0.36, 0.07),
            "prop_L": (0.020, 0.18, -0.36, 0.07),
            "prop_R": (0.020, 0.18, 0.36, 0.07),
            "esc_L": (0.035, 0.30, -0.30, 0.07),
            "esc_R": (0.035, 0.30, 0.30, 0.07),
            "battery": (0.190, 0.16, 0.0, -0.01),
            "receiver_servos": (0.110, 0.40, 0.0, 0.0),
            "landing_gear": (0.150, 0.30, 0.0, -0.06),
            "wiring_misc": (0.090, 0.40, 0.0, 0.02),
        }
    )


@dataclass
class PlaneConfig:
    wing: WingConfig = field(default_factory=WingConfig)
    stabilator: StabilatorConfig = field(default_factory=StabilatorConfig)
    fin: FinConfig = field(default_factory=FinConfig)
    fuselage: FuselageConfig = field(default_factory=FuselageConfig)
    propulsion: PropulsionConfig = field(default_factory=PropulsionConfig)
    mass: MassConfig = field(default_factory=MassConfig)
    cruise_speed: float = 14.0  # m/s
    altitude: float = 50.0  # m, field height above sea level
