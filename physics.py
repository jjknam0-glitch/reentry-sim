"""
Physics Module
==============
Core aerodynamic heating and trajectory equations of motion for
atmospheric re-entry simulation.

Heating Models:
    - Sutton-Graves convective heating (stagnation point)
    - Chapman's approximation (simplified convective)
    - Radiative equilibrium wall temperature
    - Lumped-mass thermal model with ablation

Trajectory Model:
    - 2D planar equations of motion (altitude + downrange)
    - Includes gravity, aerodynamic drag, and lift
    - Bank angle modulation for guided re-entry
"""

import numpy as np
from atmosphere import atmosphere, gravity, R_EARTH, GAMMA, M_AIR, R_GAS
from vehicle import Vehicle

# ─── Physical Constants ───────────────────────────────────────────────────────

STEFAN_BOLTZMANN = 5.670374419e-8   # Stefan-Boltzmann constant [W/(m²·K⁴)]


# ─── Heating Models ──────────────────────────────────────────────────────────

def sutton_graves_heating(rho: float, V: float, R_n: float) -> float:
    """
    Sutton-Graves stagnation-point convective heat flux.

    q_conv = k * sqrt(rho / R_n) * V^3

    Parameters
    ----------
    rho : float
        Atmospheric density [kg/m³].
    V : float
        Vehicle velocity [m/s].
    R_n : float
        Effective nose radius [m].

    Returns
    -------
    float
        Convective heat flux at stagnation point [W/m²].
    """
    k = 1.7415e-4   # Sutton-Graves constant for air [kg^0.5 / m]
    if rho <= 0 or V <= 0:
        return 0.0
    return k * np.sqrt(rho / R_n) * V ** 3


def chapman_heating(rho: float, V: float, R_n: float) -> float:
    """
    Chapman's simplified convective heating approximation.

    Parameters
    ----------
    rho : float
        Atmospheric density [kg/m³].
    V : float
        Vehicle velocity [m/s].
    R_n : float
        Effective nose radius [m].

    Returns
    -------
    float
        Approximate convective heat flux [W/m²].
    """
    rho_sl = 1.225  # Sea-level density [kg/m³]
    if rho <= 0 or V <= 0:
        return 0.0
    return 1.83e-4 * np.sqrt(rho / R_n) * V ** 3.05


def radiative_equilibrium_temp(q_conv: float, emissivity: float) -> float:
    """
    Radiative equilibrium wall temperature.

    At equilibrium: q_conv = ε · σ · T_w^4
    → T_w = (q_conv / (ε · σ))^(1/4)

    Parameters
    ----------
    q_conv : float
        Convective heat flux [W/m²].
    emissivity : float
        Surface emissivity [-].

    Returns
    -------
    float
        Equilibrium wall temperature [K].
    """
    if q_conv <= 0:
        return 0.0
    return (q_conv / (emissivity * STEFAN_BOLTZMANN)) ** 0.25


def wall_temperature_rate(q_conv: float, T_wall: float, emissivity: float,
                          thickness: float, density: float,
                          specific_heat: float) -> float:
    """
    Rate of change of wall temperature using lumped-mass thermal model.

    dT_wall/dt = (q_conv - ε·σ·T_wall⁴) / (ρ_w · c_w · δ)

    Parameters
    ----------
    q_conv : float
        Incoming convective heat flux [W/m²].
    T_wall : float
        Current wall temperature [K].
    emissivity : float
        Surface emissivity [-].
    thickness : float
        TPS wall thickness [m].
    density : float
        TPS material density [kg/m³].
    specific_heat : float
        TPS specific heat capacity [J/(kg·K)].

    Returns
    -------
    float
        dT_wall/dt [K/s].
    """
    q_rad_out = emissivity * STEFAN_BOLTZMANN * T_wall ** 4
    q_net = q_conv - q_rad_out
    thermal_mass = density * specific_heat * thickness
    return q_net / thermal_mass


# ─── Aerodynamic Forces ──────────────────────────────────────────────────────

def dynamic_pressure(rho: float, V: float) -> float:
    """Dynamic pressure q = 0.5 · ρ · V² [Pa]."""
    return 0.5 * rho * V ** 2


def drag_force(rho: float, V: float, Cd: float, A: float) -> float:
    """Aerodynamic drag force [N]."""
    return dynamic_pressure(rho, V) * Cd * A


def lift_force(rho: float, V: float, Cl: float, A: float) -> float:
    """Aerodynamic lift force [N]."""
    return dynamic_pressure(rho, V) * Cl * A


def mach_number(V: float, z: float) -> float:
    """Compute Mach number at altitude z for velocity V."""
    atmo = atmosphere(z)
    a = atmo['speed_of_sound']
    if a <= 0:
        return 0.0
    return V / a


# ─── Equations of Motion ─────────────────────────────────────────────────────

def equations_of_motion(t: float, state: np.ndarray, vehicle: Vehicle,
                        bank_angle: float = 0.0) -> np.ndarray:
    """
    2D planar re-entry equations of motion.

    State vector: [altitude, velocity, flight_path_angle, downrange, wall_temp]
        - altitude (z)              [m]       Geometric altitude
        - velocity (V)              [m/s]     Inertial speed
        - flight_path_angle (gamma) [rad]     Below local horizontal (negative = descending)
        - downrange (s)             [m]       Along Earth surface
        - wall_temp (T_w)           [K]       TPS surface temperature

    Parameters
    ----------
    t : float
        Current time [s].
    state : np.ndarray
        State vector [z, V, gamma, s, T_w].
    vehicle : Vehicle
        Vehicle configuration object.
    bank_angle : float
        Bank angle for lift vector rotation [rad]. Default 0 (lift in trajectory plane).

    Returns
    -------
    np.ndarray
        State derivatives [dz/dt, dV/dt, dgamma/dt, ds/dt, dTw/dt].
    """
    z, V, gamma, s, T_w = state

    # Clamp values for numerical stability
    z = max(z, 0.0)
    V = max(V, 1.0)

    # Atmospheric properties
    atmo = atmosphere(z)
    rho = atmo['density']

    # Gravitational acceleration at altitude
    g = gravity(z)

    # Aerodynamic forces
    q = dynamic_pressure(rho, V)
    D = q * vehicle.drag_coefficient * vehicle.reference_area  # Drag [N]
    L = q * vehicle.lift_coefficient * vehicle.reference_area   # Lift [N]

    # Earth radius factor
    r = R_EARTH + z

    # ── Translational equations ──
    # dz/dt = V · sin(γ)
    dz_dt = V * np.sin(gamma)

    # dV/dt = -D/m - g·sin(γ) + (V²/r)·sin(γ)  [centrifugal correction]
    dV_dt = -D / vehicle.mass - g * np.sin(gamma)

    # dγ/dt = (1/V)·[(L·cos(σ)/m) - (g - V²/r)·cos(γ)]
    cos_bank = np.cos(bank_angle)
    if V > 10.0:
        dgamma_dt = (1.0 / V) * (
            (L * cos_bank / vehicle.mass)
            - (g - V**2 / r) * np.cos(gamma)
        )
    else:
        dgamma_dt = 0.0

    # ds/dt = (V · cos(γ) · R_earth) / r
    ds_dt = V * np.cos(gamma) * R_EARTH / r

    # ── Thermal equation ──
    q_conv = sutton_graves_heating(rho, V, vehicle.nose_radius)

    # Ablation check
    if T_w >= vehicle.ablation_temp and vehicle.heat_of_ablation > 0:
        # Ablation regime: excess heat goes to ablation, wall temp held at ablation temp
        dTw_dt = 0.0
    else:
        dTw_dt = wall_temperature_rate(
            q_conv, T_w, vehicle.emissivity,
            vehicle.wall_thickness, vehicle.wall_density,
            vehicle.wall_specific_heat
        )

    return np.array([dz_dt, dV_dt, dgamma_dt, ds_dt, dTw_dt])


# ─── Derived Quantities ──────────────────────────────────────────────────────

def compute_g_load(V: float, dV_dt: float) -> float:
    """
    Compute deceleration g-loading.

    Parameters
    ----------
    V : float
        Current velocity [m/s].
    dV_dt : float
        Rate of change of velocity [m/s²].

    Returns
    -------
    float
        Deceleration in Earth g's.
    """
    from atmosphere import G0
    return abs(dV_dt) / G0


def compute_stagnation_pressure(rho: float, V: float, z: float) -> float:
    """
    Approximate stagnation pressure for hypersonic flow.

    Uses modified Newtonian approximation:
        p_stag ≈ p_inf + q_inf · (Cp_max)
    where Cp_max ≈ 2 for hypersonic blunt body.

    Returns
    -------
    float
        Stagnation pressure [Pa].
    """
    atmo = atmosphere(z)
    p_inf = atmo['pressure']
    q_inf = dynamic_pressure(rho, V)
    return p_inf + 2.0 * q_inf
