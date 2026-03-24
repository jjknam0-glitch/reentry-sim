"""
Visualization Module
====================
Professional-grade matplotlib plotting for re-entry simulation results.

Generates:
    - Multi-panel trajectory dashboard
    - Individual parameter plots
    - Altitude-velocity map (entry corridor)
    - Heating analysis plot
    - Comparison plots for multiple vehicles
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter, MaxNLocator
import matplotlib.patheffects as path_effects
from typing import List, Optional

from simulator import SimResults


# ─── Style Configuration ─────────────────────────────────────────────────────

COLORS = {
    'primary': '#E85D3A',
    'secondary': '#3A8EE8',
    'accent1': '#45B764',
    'accent2': '#E8C93A',
    'accent3': '#B745E8',
    'dark_bg': '#0D1117',
    'panel_bg': '#161B22',
    'grid': '#21262D',
    'text': '#E6EDF3',
    'text_dim': '#8B949E',
    'border': '#30363D',
}

COMPARE_COLORS = ['#E85D3A', '#3A8EE8', '#45B764', '#E8C93A', '#B745E8']


def _apply_style(fig, axes):
    """Apply consistent dark theme styling to figure and axes."""
    fig.patch.set_facecolor(COLORS['dark_bg'])

    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    for ax in np.array(axes).flat:
        ax.set_facecolor(COLORS['panel_bg'])
        ax.tick_params(colors=COLORS['text_dim'], labelsize=8)
        ax.xaxis.label.set_color(COLORS['text'])
        ax.yaxis.label.set_color(COLORS['text'])
        ax.title.set_color(COLORS['text'])
        ax.spines['bottom'].set_color(COLORS['border'])
        ax.spines['left'].set_color(COLORS['border'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, color=COLORS['grid'], linewidth=0.5, alpha=0.7)


def _km_formatter():
    """Format axis values from meters to km."""
    return FuncFormatter(lambda x, _: f'{x/1e3:.0f}')


def _mw_formatter():
    """Format axis values from W/m² to MW/m²."""
    return FuncFormatter(lambda x, _: f'{x/1e6:.1f}')


def _kpa_formatter():
    """Format axis values from Pa to kPa."""
    return FuncFormatter(lambda x, _: f'{x/1e3:.1f}')


# ─── Main Dashboard ──────────────────────────────────────────────────────────

def plot_dashboard(results: SimResults, save_path: str = "dashboard.png",
                   dpi: int = 200) -> str:
    """
    Generate a comprehensive 6-panel re-entry simulation dashboard.

    Panels:
        1. Altitude vs Time
        2. Velocity vs Time
        3. Heat Flux vs Time
        4. G-Loading vs Time
        5. Wall Temperature vs Time
        6. Altitude vs Velocity (entry corridor)

    Parameters
    ----------
    results : SimResults
        Simulation results to plot.
    save_path : str
        Output file path.
    dpi : int
        Resolution.

    Returns
    -------
    str
        Path to saved figure.
    """
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25,
                  left=0.07, right=0.95, top=0.92, bottom=0.06)

    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]
    _apply_style(fig, axes)

    t = results.time

    # ── Title ──
    fig.suptitle(
        f"Atmospheric Re-entry Simulation — {results.vehicle_name}",
        color=COLORS['text'], fontsize=16, fontweight='bold', y=0.97
    )

    # Subtitle with key info
    subtitle = (
        f"Peak Heat Flux: {results.peak_heat_flux/1e6:.2f} MW/m²  │  "
        f"Peak G-Load: {results.peak_g_load:.1f}g  │  "
        f"Peak Wall Temp: {results.peak_wall_temperature:.0f} K  │  "
        f"Total Heat Load: {results.total_heat_load/1e6:.2f} MJ/m²"
    )
    fig.text(0.5, 0.94, subtitle, ha='center', va='center',
             color=COLORS['text_dim'], fontsize=9)

    # ── Panel 1: Altitude vs Time ──
    ax = axes[0]
    ax.plot(t, results.altitude / 1e3, color=COLORS['primary'], linewidth=1.5)
    ax.fill_between(t, 0, results.altitude / 1e3,
                    color=COLORS['primary'], alpha=0.08)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Altitude [km]")
    ax.set_title("Altitude Profile", fontsize=11, fontweight='bold')
    ax.set_ylim(bottom=0)

    # Mark key altitudes
    for h_km, label in [(80, 'Mesopause'), (50, 'Stratopause'), (12, 'Tropopause')]:
        ax.axhline(y=h_km, color=COLORS['text_dim'], linestyle=':', linewidth=0.5, alpha=0.5)
        ax.text(t[-1]*0.98, h_km+1.5, label, ha='right', fontsize=6,
                color=COLORS['text_dim'], alpha=0.7)

    # ── Panel 2: Velocity vs Time ──
    ax = axes[1]
    ax.plot(t, results.velocity / 1e3, color=COLORS['secondary'], linewidth=1.5)
    ax.fill_between(t, 0, results.velocity / 1e3,
                    color=COLORS['secondary'], alpha=0.08)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Velocity [km/s]")
    ax.set_title("Velocity Profile", fontsize=11, fontweight='bold')
    ax.set_ylim(bottom=0)

    # Mark Mach milestones
    for i_m, m_val in [(np.argmin(np.abs(results.mach - 25)), 'M25'),
                        (np.argmin(np.abs(results.mach - 10)), 'M10'),
                        (np.argmin(np.abs(results.mach - 5)), 'M5'),
                        (np.argmin(np.abs(results.mach - 1)), 'M1')]:
        if i_m > 0 and i_m < len(t) - 1:
            ax.axvline(x=t[i_m], color=COLORS['text_dim'], linestyle=':',
                      linewidth=0.5, alpha=0.4)
            ax.text(t[i_m], ax.get_ylim()[1]*0.95, m_val, ha='center',
                   fontsize=6, color=COLORS['text_dim'])

    # ── Panel 3: Heat Flux vs Time ──
    ax = axes[2]
    ax.plot(t, results.heat_flux / 1e6, color=COLORS['primary'], linewidth=1.5)
    ax.fill_between(t, 0, results.heat_flux / 1e6,
                    color=COLORS['primary'], alpha=0.15)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Heat Flux [MW/m²]")
    ax.set_title("Stagnation-Point Heat Flux (Sutton-Graves)", fontsize=11, fontweight='bold')
    ax.set_ylim(bottom=0)

    # Mark peak
    i_peak = np.argmax(results.heat_flux)
    ax.plot(t[i_peak], results.heat_flux[i_peak]/1e6, 'o',
            color=COLORS['accent2'], markersize=6, zorder=5)
    ax.annotate(f'Peak: {results.peak_heat_flux/1e6:.2f} MW/m²',
                xy=(t[i_peak], results.heat_flux[i_peak]/1e6),
                xytext=(30, 15), textcoords='offset points',
                color=COLORS['accent2'], fontsize=8,
                arrowprops=dict(arrowstyle='->', color=COLORS['accent2'], lw=0.8))

    # ── Panel 4: G-Loading vs Time ──
    ax = axes[3]
    ax.plot(t, results.g_load, color=COLORS['accent3'], linewidth=1.5)
    ax.fill_between(t, 0, results.g_load,
                    color=COLORS['accent3'], alpha=0.1)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Deceleration [g]")
    ax.set_title("G-Loading", fontsize=11, fontweight='bold')
    ax.set_ylim(bottom=0)

    # Human tolerance lines
    for g_val, label, clr in [(9, '9g (fighter pilot limit)', COLORS['accent2']),
                                (3, '3g (crew comfort)', COLORS['accent1'])]:
        if results.peak_g_load > g_val * 0.5:
            ax.axhline(y=g_val, color=clr, linestyle='--', linewidth=0.8, alpha=0.6)
            ax.text(t[0]+5, g_val+0.3, label, fontsize=6, color=clr, alpha=0.8)

    # Mark peak
    i_peak_g = np.argmax(results.g_load)
    ax.plot(t[i_peak_g], results.g_load[i_peak_g], 'o',
            color=COLORS['accent2'], markersize=6, zorder=5)

    # ── Panel 5: Wall Temperature vs Time ──
    ax = axes[4]
    ax.plot(t, results.wall_temperature, color=COLORS['primary'],
            linewidth=1.5, label='Wall Temp (lumped)')
    ax.plot(t, results.equilibrium_temp, color=COLORS['accent2'],
            linewidth=1.0, linestyle='--', alpha=0.7, label='Rad. Equilibrium')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("Surface Temperature", fontsize=11, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right', framealpha=0.3,
              labelcolor=COLORS['text_dim'])

    # Mark peak wall temp
    i_peak_t = np.argmax(results.wall_temperature)
    ax.plot(t[i_peak_t], results.wall_temperature[i_peak_t], 'o',
            color=COLORS['accent2'], markersize=6, zorder=5)
    ax.annotate(f'Peak: {results.peak_wall_temperature:.0f} K',
                xy=(t[i_peak_t], results.wall_temperature[i_peak_t]),
                xytext=(30, -20), textcoords='offset points',
                color=COLORS['accent2'], fontsize=8,
                arrowprops=dict(arrowstyle='->', color=COLORS['accent2'], lw=0.8))

    # ── Panel 6: Altitude vs Velocity (Entry Corridor) ──
    ax = axes[5]
    scatter = ax.scatter(results.velocity / 1e3, results.altitude / 1e3,
                         c=results.heat_flux / 1e6, cmap='inferno',
                         s=2, alpha=0.8, zorder=3)
    ax.plot(results.velocity / 1e3, results.altitude / 1e3,
            color=COLORS['text_dim'], linewidth=0.3, alpha=0.3)
    ax.set_xlabel("Velocity [km/s]")
    ax.set_ylabel("Altitude [km]")
    ax.set_title("Entry Corridor (color = heat flux)", fontsize=11, fontweight='bold')
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)

    cbar = fig.colorbar(scatter, ax=ax, pad=0.02, aspect=30)
    cbar.set_label("Heat Flux [MW/m²]", color=COLORS['text_dim'], fontsize=8)
    cbar.ax.tick_params(colors=COLORS['text_dim'], labelsize=7)

    # Mark entry and peak heating points
    ax.plot(results.velocity[0]/1e3, results.altitude[0]/1e3, '^',
            color=COLORS['accent1'], markersize=8, zorder=5, label='Entry')
    i_qmax = np.argmax(results.heat_flux)
    ax.plot(results.velocity[i_qmax]/1e3, results.altitude[i_qmax]/1e3, 's',
            color=COLORS['accent2'], markersize=8, zorder=5, label='Peak Heating')
    ax.legend(fontsize=7, loc='lower left', framealpha=0.3,
              labelcolor=COLORS['text_dim'])

    # Save
    fig.savefig(save_path, dpi=dpi, facecolor=COLORS['dark_bg'],
                edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    return save_path


# ─── Heating Analysis Plot ───────────────────────────────────────────────────

def plot_heating_analysis(results: SimResults, save_path: str = "heating_analysis.png",
                          dpi: int = 200) -> str:
    """
    Detailed heating analysis: heat flux, heat load, wall temp, and equilibrium temp.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    _apply_style(fig, axes)

    fig.suptitle(f"Aerothermal Heating Analysis — {results.vehicle_name}",
                 color=COLORS['text'], fontsize=14, fontweight='bold', y=0.97)

    t = results.time

    # Heat flux vs altitude
    ax = axes[0, 0]
    ax.plot(results.altitude / 1e3, results.heat_flux / 1e6,
            color=COLORS['primary'], linewidth=1.5)
    ax.set_xlabel("Altitude [km]")
    ax.set_ylabel("Heat Flux [MW/m²]")
    ax.set_title("Heat Flux vs Altitude", fontsize=10, fontweight='bold')
    ax.invert_xaxis()

    # Cumulative heat load
    ax = axes[0, 1]
    ax.plot(t, results.heat_load / 1e6, color=COLORS['accent1'], linewidth=1.5)
    ax.fill_between(t, 0, results.heat_load / 1e6,
                    color=COLORS['accent1'], alpha=0.1)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Heat Load [MJ/m²]")
    ax.set_title("Cumulative Heat Load", fontsize=10, fontweight='bold')

    # Temperature comparison
    ax = axes[1, 0]
    ax.plot(t, results.wall_temperature, color=COLORS['primary'],
            linewidth=1.5, label='Wall (lumped mass)')
    ax.plot(t, results.equilibrium_temp, color=COLORS['accent2'],
            linewidth=1.0, linestyle='--', label='Rad. Equilibrium')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("Temperature Comparison", fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, framealpha=0.3, labelcolor=COLORS['text_dim'])

    # Heating rate vs velocity
    ax = axes[1, 1]
    ax.plot(results.velocity / 1e3, results.heat_flux / 1e6,
            color=COLORS['secondary'], linewidth=1.5)
    ax.set_xlabel("Velocity [km/s]")
    ax.set_ylabel("Heat Flux [MW/m²]")
    ax.set_title("Heat Flux vs Velocity", fontsize=10, fontweight='bold')

    fig.savefig(save_path, dpi=dpi, facecolor=COLORS['dark_bg'],
                edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    return save_path


# ─── Comparison Plot ─────────────────────────────────────────────────────────

def plot_comparison(results_list: List[SimResults],
                    save_path: str = "comparison.png",
                    dpi: int = 200) -> str:
    """
    Compare re-entry trajectories of multiple vehicles or configurations.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    _apply_style(fig, axes)

    fig.suptitle("Re-entry Trajectory Comparison",
                 color=COLORS['text'], fontsize=14, fontweight='bold', y=0.97)

    panels = [
        (0, 0, 'altitude', 1e3, 'Altitude [km]', 'Altitude vs Time'),
        (0, 1, 'velocity', 1e3, 'Velocity [km/s]', 'Velocity vs Time'),
        (0, 2, 'heat_flux', 1e6, 'Heat Flux [MW/m²]', 'Heat Flux vs Time'),
        (1, 0, 'g_load', 1.0, 'G-Load [g]', 'G-Loading vs Time'),
        (1, 1, 'wall_temperature', 1.0, 'Wall Temp [K]', 'Wall Temperature vs Time'),
        (1, 2, 'dynamic_pressure', 1e3, 'Dyn. Pressure [kPa]', 'Dynamic Pressure vs Time'),
    ]

    for row, col, attr, scale, ylabel, title in panels:
        ax = axes[row, col]
        for idx, res in enumerate(results_list):
            color = COMPARE_COLORS[idx % len(COMPARE_COLORS)]
            data = getattr(res, attr) / scale
            ax.plot(res.time, data, color=color, linewidth=1.2,
                    label=res.vehicle_name)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.legend(fontsize=6, framealpha=0.3, labelcolor=COLORS['text_dim'])
        ax.set_ylim(bottom=0)

    fig.savefig(save_path, dpi=dpi, facecolor=COLORS['dark_bg'],
                edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    return save_path


# ─── Atmospheric Profile Plot ────────────────────────────────────────────────

def plot_atmosphere(save_path: str = "atmosphere_profile.png",
                    dpi: int = 200) -> str:
    """Plot the atmospheric model used in the simulation."""
    from atmosphere import atmosphere as atmo_func

    altitudes = np.linspace(0, 120_000, 1000)
    rho = np.array([atmo_func(z)['density'] for z in altitudes])
    T = np.array([atmo_func(z)['temperature'] for z in altitudes])
    p = np.array([atmo_func(z)['pressure'] for z in altitudes])
    a = np.array([atmo_func(z)['speed_of_sound'] for z in altitudes])

    fig, axes = plt.subplots(1, 4, figsize=(18, 7), sharey=True)
    _apply_style(fig, axes)

    fig.suptitle("US Standard Atmosphere 1976 — Model Validation",
                 color=COLORS['text'], fontsize=14, fontweight='bold', y=0.97)

    h_km = altitudes / 1e3

    axes[0].semilogx(rho, h_km, color=COLORS['primary'], linewidth=1.5)
    axes[0].set_xlabel("Density [kg/m³]")
    axes[0].set_ylabel("Altitude [km]")
    axes[0].set_title("Density", fontsize=10, fontweight='bold')

    axes[1].plot(T, h_km, color=COLORS['secondary'], linewidth=1.5)
    axes[1].set_xlabel("Temperature [K]")
    axes[1].set_title("Temperature", fontsize=10, fontweight='bold')

    axes[2].semilogx(p, h_km, color=COLORS['accent1'], linewidth=1.5)
    axes[2].set_xlabel("Pressure [Pa]")
    axes[2].set_title("Pressure", fontsize=10, fontweight='bold')

    axes[3].plot(a, h_km, color=COLORS['accent2'], linewidth=1.5)
    axes[3].set_xlabel("Speed of Sound [m/s]")
    axes[3].set_title("Speed of Sound", fontsize=10, fontweight='bold')

    fig.savefig(save_path, dpi=dpi, facecolor=COLORS['dark_bg'],
                edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    return save_path
