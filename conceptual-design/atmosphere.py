"""International Standard Atmosphere (troposphere) with a temperature offset."""
import math
from dataclasses import dataclass

R_AIR = 287.053       # J/(kg K)
G0 = 9.80665          # m/s^2
T0 = 288.15           # K
P0 = 101325.0         # Pa
LAPSE = 0.0065        # K/m
RHO0 = P0 / (R_AIR * T0)


@dataclass(frozen=True)
class Atmosphere:
    h: float
    T: float
    p: float
    rho: float
    mu: float
    a: float


def isa(h=0.0, dT=0.0):
    """Pressure follows the standard day; temperature is offset by dT
    (the usual 'ISA + dT' convention). Viscosity from Sutherland's law."""
    t_std = T0 - LAPSE * h
    p = P0 * (t_std / T0) ** (G0 / (R_AIR * LAPSE))
    T = t_std + dT
    rho = p / (R_AIR * T)
    mu = 1.458e-6 * T ** 1.5 / (T + 110.4)
    a = math.sqrt(1.4 * R_AIR * T)
    return Atmosphere(h, T, p, rho, mu, a)
