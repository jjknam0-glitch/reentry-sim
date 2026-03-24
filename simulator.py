"""
Simulator Engine
================
Core simulation engine that integrates the re-entry equations of motion
using scipy.integrate.solve_ivp with configurable parameters.

Supports:
    - Multiple ODE solvers (RK45, RK23, DOP853, Radau, BDF)
    - Event detection (ground impact, velocity threshold, skip-out)
    - Bank angle scheduling for guided re-entry
    - Comprehensive result packaging with derived quantities
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import Optional, Callable, List
import time as time_module

from vehicle import Vehicle
from atmosphere import atmosphere, gravity, R_EARTH, G0
from physics import (
    equations_of_motion,
    sutton_graves_heating,
    radiative_equilibrium_temp,
    dynamic_pressure,
    mach_number,
    compute_g_load,
    compute_stagnation_pressure,
)


# ─── Simulation Configuration ────────────────────────────────────────────────

@dataclass
class SimConfig:
    """
    Simulation configuration parameters.

    Parameters
    ----------
    initial_altitude : float
        Entry interface altitude [m]. Typically 120-400 km.
    initial_velocity : float
        Entry velocity [m/s]. LEO ~7800, lunar return ~11000.
    initial_fpa : float
        Initial flight path angle [deg]. Negative = descending.
        Typical range: -1° to -15°.
    bank_angle : float
        Constant bank angle [deg]. 0 = max lift-up.
    max_time : float
        Maximum simulation time [s].
    dt_output : float
        Output time step for dense output [s].
    solver : str
        ODE solver method. One of: 'RK45', 'RK23', 'DOP853', 'Radau', 'BDF'.
    rtol : float
        Relative tolerance for ODE solver.
    atol : float
        Absolute tolerance for ODE solver.
    min_altitude : float
        Ground altitude / stop condition [m].
    skip_out_altitude : float
        Altitude above which vehicle is considered to have skipped out [m].
    min_velocity : float
        Minimum velocity threshold [m/s].
    """
    initial_altitude: float = 120_000.0
    initial_velocity: float = 7800.0
    initial_fpa: float = -5.0
    bank_angle: float = 0.0
    max_time: float = 2000.0
    dt_output: float = 0.5
    solver: str = 'RK45'
    rtol: float = 1e-10
    atol: float = 1e-10
    min_altitude: float = 0.0
    skip_out_altitude: float = 200_000.0
    min_velocity: float = 50.0

    def __post_init__(self):
        assert self.initial_altitude > 0, "Altitude must be positive"
        assert self.initial_velocity > 0, "Velocity must be positive"
        assert self.initial_fpa < 0, "FPA must be negative (descending)"


# ─── Simulation Results ──────────────────────────────────────────────────────

@dataclass
class SimResults:
    """
    Container for simulation results with all time histories.
    """
    # Time
    time: np.ndarray = field(default_factory=lambda: np.array([]))

    # Primary state
    altitude: np.ndarray = field(default_factory=lambda: np.array([]))
    velocity: np.ndarray = field(default_factory=lambda: np.array([]))
    flight_path_angle: np.ndarray = field(default_factory=lambda: np.array([]))
    downrange: np.ndarray = field(default_factory=lambda: np.array([]))
    wall_temperature: np.ndarray = field(default_factory=lambda: np.array([]))

    # Derived quantities
    heat_flux: np.ndarray = field(default_factory=lambda: np.array([]))
    heat_load: np.ndarray = field(default_factory=lambda: np.array([]))
    g_load: np.ndarray = field(default_factory=lambda: np.array([]))
    dynamic_pressure: np.ndarray = field(default_factory=lambda: np.array([]))
    mach: np.ndarray = field(default_factory=lambda: np.array([]))
    density: np.ndarray = field(default_factory=lambda: np.array([]))
    equilibrium_temp: np.ndarray = field(default_factory=lambda: np.array([]))
    stagnation_pressure: np.ndarray = field(default_factory=lambda: np.array([]))

    # Metadata
    vehicle_name: str = ""
    solver_status: str = ""
    wall_time: float = 0.0
    termination_reason: str = ""
    n_steps: int = 0

    # Peak values
    peak_heat_flux: float = 0.0
    peak_g_load: float = 0.0
    peak_dynamic_pressure: float = 0.0
    peak_wall_temperature: float = 0.0
    total_heat_load: float = 0.0
    peak_mach: float = 0.0

    def summary(self) -> str:
        """Return a formatted summary of key results."""
        lines = [
            "╔══════════════════════════════════════════════════════════╗",
            f"║  RE-ENTRY SIMULATION RESULTS                            ║",
            f"║  Vehicle: {self.vehicle_name:<46s}║",
            "╠══════════════════════════════════════════════════════════╣",
            f"║  Termination:        {self.termination_reason:<35s}║",
            f"║  Simulation Time:    {self.time[-1]:>12.1f} s                      ║",
            f"║  Wall-clock Time:    {self.wall_time:>12.4f} s                      ║",
            f"║  Solver Steps:       {self.n_steps:>12d}                        ║",
            "╠══════════════════════════════════════════════════════════╣",
            f"║  PEAK VALUES                                            ║",
            f"║  ─────────────────────────────────────────────────────  ║",
            f"║  Heat Flux:          {self.peak_heat_flux / 1e6:>12.3f} MW/m²                ║",
            f"║  G-Loading:          {self.peak_g_load:>12.2f} g                     ║",
            f"║  Dynamic Pressure:   {self.peak_dynamic_pressure / 1e3:>12.2f} kPa                   ║",
            f"║  Wall Temperature:   {self.peak_wall_temperature:>12.1f} K                     ║",
            f"║  Mach Number:        {self.peak_mach:>12.2f}                        ║",
            f"║  Total Heat Load:    {self.total_heat_load / 1e6:>12.3f} MJ/m²                ║",
            "╠══════════════════════════════════════════════════════════╣",
            f"║  FINAL STATE                                            ║",
            f"║  ─────────────────────────────────────────────────────  ║",
            f"║  Altitude:           {self.altitude[-1] / 1e3:>12.2f} km                    ║",
            f"║  Velocity:           {self.velocity[-1]:>12.1f} m/s                   ║",
            f"║  Downrange:          {self.downrange[-1] / 1e3:>12.1f} km                    ║",
            f"║  Wall Temperature:   {self.wall_temperature[-1]:>12.1f} K                     ║",
            "╚══════════════════════════════════════════════════════════╝",
        ]
        return "\n".join(lines)


# ─── Event Functions ──────────────────────────────────────────────────────────

def _ground_impact_event(t, state, vehicle, bank_angle):
    """Event: altitude reaches ground level."""
    return state[0]  # z = 0

_ground_impact_event.terminal = True
_ground_impact_event.direction = -1


def _velocity_threshold_event(t, state, vehicle, bank_angle):
    """Event: velocity drops below threshold."""
    return state[1] - 50.0  # V = 50 m/s

_velocity_threshold_event.terminal = True
_velocity_threshold_event.direction = -1


def _skip_out_event(t, state, vehicle, bank_angle):
    """Event: vehicle flight path angle becomes positive (ascending) above 100km."""
    if state[0] > 100_000 and t > 10:
        return state[2]  # gamma = 0 (ascending)
    return -1.0  # Don't trigger

_skip_out_event.terminal = False
_skip_out_event.direction = 1


# ─── Main Simulator ──────────────────────────────────────────────────────────

class ReentrySimulator:
    """
    Atmospheric re-entry trajectory and heating simulator.

    Usage
    -----
    >>> from vehicle import apollo_cm
    >>> sim = ReentrySimulator(vehicle=apollo_cm(), config=SimConfig())
    >>> results = sim.run()
    >>> print(results.summary())
    """

    def __init__(self, vehicle: Vehicle, config: SimConfig):
        self.vehicle = vehicle
        self.config = config

    def run(self, verbose: bool = True) -> SimResults:
        """
        Execute the re-entry simulation.

        Parameters
        ----------
        verbose : bool
            Print progress messages.

        Returns
        -------
        SimResults
            Complete simulation results.
        """
        cfg = self.config
        v = self.vehicle

        if verbose:
            print(f"\n{'='*60}")
            print(f"  ATMOSPHERIC RE-ENTRY SIMULATION")
            print(f"  Vehicle: {v.name}")
            print(f"  Entry: {cfg.initial_altitude/1e3:.0f} km, "
                  f"{cfg.initial_velocity:.0f} m/s, "
                  f"FPA = {cfg.initial_fpa:.1f}°")
            print(f"{'='*60}\n")

        # Initial state
        gamma0 = np.radians(cfg.initial_fpa)
        bank_rad = np.radians(cfg.bank_angle)
        y0 = np.array([
            cfg.initial_altitude,
            cfg.initial_velocity,
            gamma0,
            0.0,                    # downrange
            v.initial_wall_temp,    # wall temperature
        ])

        # Time span
        t_span = (0.0, cfg.max_time)
        t_eval = np.arange(0.0, cfg.max_time, cfg.dt_output)

        # ODE right-hand side
        def rhs(t, state):
            return equations_of_motion(t, state, v, bank_angle=bank_rad)

        # Event functions with vehicle/bank args
        events = [
            lambda t, s, v=v, b=bank_rad: _ground_impact_event(t, s, v, b),
            lambda t, s, v=v, b=bank_rad: _velocity_threshold_event(t, s, v, b),
        ]
        events[0].terminal = True
        events[0].direction = -1
        events[1].terminal = True
        events[1].direction = -1

        # Run integration
        start_time = time_module.perf_counter()
        sol = solve_ivp(
            rhs,
            t_span,
            y0,
            method=cfg.solver,
            t_eval=t_eval,
            events=events,
            rtol=cfg.rtol,
            atol=cfg.atol,
            max_step=1.0,
        )
        wall_time = time_module.perf_counter() - start_time

        if verbose:
            print(f"  Integration complete: {sol.status}")
            print(f"  Steps: {sol.t.shape[0]}, Wall time: {wall_time:.3f}s\n")

        # Determine termination reason
        if sol.status == 1:
            if sol.t_events[0].size > 0:
                termination = "Ground impact"
            elif sol.t_events[1].size > 0:
                termination = "Velocity below threshold"
            else:
                termination = "Terminal event"
        elif sol.status == 0:
            termination = "Max time reached"
        else:
            termination = f"Solver error (status={sol.status})"

        # Extract state histories
        t = sol.t
        z = sol.y[0]
        V = sol.y[1]
        gamma = sol.y[2]
        s = sol.y[3]
        T_w = sol.y[4]

        # Compute derived quantities
        n = len(t)
        q_conv = np.zeros(n)
        q_load = np.zeros(n)
        g_loads = np.zeros(n)
        q_dyn = np.zeros(n)
        mach_arr = np.zeros(n)
        rho_arr = np.zeros(n)
        T_eq = np.zeros(n)
        p_stag = np.zeros(n)

        for i in range(n):
            atmo = atmosphere(z[i])
            rho = atmo['density']
            rho_arr[i] = rho

            # Heat flux
            q_conv[i] = sutton_graves_heating(rho, V[i], v.nose_radius)

            # Heat load (trapezoidal integration)
            if i > 0:
                dt = t[i] - t[i - 1]
                q_load[i] = q_load[i - 1] + 0.5 * (q_conv[i] + q_conv[i - 1]) * dt

            # G-load from deceleration
            if i > 0:
                dV_dt = (V[i] - V[i - 1]) / max(t[i] - t[i - 1], 1e-10)
                g_loads[i] = abs(dV_dt) / G0
            elif i == 0 and n > 1:
                dV_dt = (V[1] - V[0]) / max(t[1] - t[0], 1e-10)
                g_loads[0] = abs(dV_dt) / G0

            # Dynamic pressure
            q_dyn[i] = dynamic_pressure(rho, V[i])

            # Mach number
            a = atmo['speed_of_sound']
            mach_arr[i] = V[i] / a if a > 0 else 0.0

            # Equilibrium temperature
            T_eq[i] = radiative_equilibrium_temp(q_conv[i], v.emissivity)

            # Stagnation pressure
            p_stag[i] = compute_stagnation_pressure(rho, V[i], z[i])

        # Smooth g-load with moving average (reduce numerical noise)
        if n > 5:
            kernel = 5
            g_loads_smooth = np.convolve(g_loads, np.ones(kernel)/kernel, mode='same')
        else:
            g_loads_smooth = g_loads

        # Package results
        results = SimResults(
            time=t,
            altitude=z,
            velocity=V,
            flight_path_angle=np.degrees(gamma),
            downrange=s,
            wall_temperature=T_w,
            heat_flux=q_conv,
            heat_load=q_load,
            g_load=g_loads_smooth,
            dynamic_pressure=q_dyn,
            mach=mach_arr,
            density=rho_arr,
            equilibrium_temp=T_eq,
            stagnation_pressure=p_stag,
            vehicle_name=v.name,
            solver_status=str(sol.status),
            wall_time=wall_time,
            termination_reason=termination,
            n_steps=len(t),
            peak_heat_flux=float(np.max(q_conv)),
            peak_g_load=float(np.max(g_loads_smooth)),
            peak_dynamic_pressure=float(np.max(q_dyn)),
            peak_wall_temperature=float(np.max(T_w)),
            total_heat_load=float(q_load[-1]) if len(q_load) > 0 else 0.0,
            peak_mach=float(np.max(mach_arr)),
        )

        if verbose:
            print(results.summary())

        return results
