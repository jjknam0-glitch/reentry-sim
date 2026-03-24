"""
Atmospheric Model Module
========================
Implements the US Standard Atmosphere 1976 model for density, temperature,
and pressure as functions of geometric altitude.

Reference: U.S. Standard Atmosphere, 1976 (NOAA/NASA/USAF)
"""

import numpy as np
from dataclasses import dataclass


# ─── Constants ────────────────────────────────────────────────────────────────

R_EARTH = 6.371e6          # Mean Earth radius [m]
G0 = 9.80665               # Sea-level gravitational acceleration [m/s²]
M_AIR = 0.0289644          # Molar mass of dry air [kg/mol]
R_GAS = 8.31447            # Universal gas constant [J/(mol·K)]
GAMMA = 1.4                # Ratio of specific heats for air


# ─── US Standard Atmosphere 1976 Layer Definitions ────────────────────────────

@dataclass(frozen=True)
class AtmoLayer:
    """Defines a single layer of the US Standard Atmosphere 1976."""
    h_base: float       # Geopotential base altitude [m]
    T_base: float       # Base temperature [K]
    lapse_rate: float   # Temperature lapse rate [K/m]
    p_base: float       # Base pressure [Pa]


# Layers from 0 to 86 km geopotential altitude
ATMO_LAYERS = [
    AtmoLayer(h_base=0,     T_base=288.15,  lapse_rate=-0.0065,  p_base=101325.0),
    AtmoLayer(h_base=11000, T_base=216.65,  lapse_rate=0.0,      p_base=22632.1),
    AtmoLayer(h_base=20000, T_base=216.65,  lapse_rate=0.001,    p_base=5474.89),
    AtmoLayer(h_base=32000, T_base=228.65,  lapse_rate=0.0028,   p_base=868.019),
    AtmoLayer(h_base=47000, T_base=270.65,  lapse_rate=0.0,      p_base=110.906),
    AtmoLayer(h_base=51000, T_base=270.65,  lapse_rate=-0.0028,  p_base=66.9389),
    AtmoLayer(h_base=71000, T_base=214.65,  lapse_rate=-0.002,   p_base=3.95642),
]


def geometric_to_geopotential(z: float) -> float:
    """Convert geometric altitude [m] to geopotential altitude [m]."""
    return (R_EARTH * z) / (R_EARTH + z)


def gravity(z: float) -> float:
    """
    Gravitational acceleration at geometric altitude z [m].
    Uses inverse-square law.
    """
    return G0 * (R_EARTH / (R_EARTH + z)) ** 2


def _find_layer(h: float) -> AtmoLayer:
    """Find the atmospheric layer for a given geopotential altitude."""
    for i in range(len(ATMO_LAYERS) - 1, -1, -1):
        if h >= ATMO_LAYERS[i].h_base:
            return ATMO_LAYERS[i]
    return ATMO_LAYERS[0]


def atmosphere(z: float) -> dict:
    """
    Compute atmospheric properties at geometric altitude z [m].

    Parameters
    ----------
    z : float
        Geometric altitude above sea level [m].

    Returns
    -------
    dict with keys:
        'temperature' : float [K]
        'pressure'    : float [Pa]
        'density'     : float [kg/m³]
        'speed_of_sound' : float [m/s]
    """
    # Clamp altitude
    if z < 0:
        z = 0.0

    # Above 86 km — use exponential extrapolation
    if z > 86000:
        # Exponential decay model for upper atmosphere
        rho_86 = 6.39e-5   # Approximate density at 86 km [kg/m³]
        H_scale = 6500.0    # Scale height [m]
        T = 186.87          # Approximate temperature at 86 km [K]
        rho = rho_86 * np.exp(-(z - 86000) / H_scale)
        p = rho * (R_GAS / M_AIR) * T
        a = np.sqrt(GAMMA * R_GAS * T / M_AIR)
        return {
            'temperature': T,
            'pressure': max(p, 0.0),
            'density': max(rho, 1e-15),
            'speed_of_sound': a,
        }

    h = geometric_to_geopotential(z)
    layer = _find_layer(h)
    dh = h - layer.h_base

    # Temperature
    T = layer.T_base + layer.lapse_rate * dh

    # Pressure
    if abs(layer.lapse_rate) < 1e-10:
        # Isothermal layer
        p = layer.p_base * np.exp(-G0 * M_AIR * dh / (R_GAS * layer.T_base))
    else:
        # Gradient layer
        p = layer.p_base * (layer.T_base / T) ** (G0 * M_AIR / (R_GAS * layer.lapse_rate))

    # Density from ideal gas law
    rho = (p * M_AIR) / (R_GAS * T)

    # Speed of sound
    a = np.sqrt(GAMMA * R_GAS * T / M_AIR)

    return {
        'temperature': T,
        'pressure': p,
        'density': max(rho, 1e-15),
        'speed_of_sound': a,
    }


def density(z: float) -> float:
    """Shortcut: atmospheric density at altitude z [m] → [kg/m³]."""
    return atmosphere(z)['density']


def temperature(z: float) -> float:
    """Shortcut: atmospheric temperature at altitude z [m] → [K]."""
    return atmosphere(z)['temperature']


def speed_of_sound(z: float) -> float:
    """Shortcut: speed of sound at altitude z [m] → [m/s]."""
    return atmosphere(z)['speed_of_sound']
