#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  Simulation of Atmospheric Re-entry Trajectory and Aerothermal Heating
  for Blunt Body Vehicles
═══════════════════════════════════════════════════════════════════════════════

  A Python-based simulation tool that models the trajectory and aerothermal
  heating of a blunt-body vehicle during atmospheric re-entry. It enables
  engineers and students to predict critical parameters such as heat flux,
  surface temperature, and g-loading to support early-stage thermal
  protection system design.

  Author:  [Your Name]
  Version: 1.0.0

  Usage:
      python main.py                         # Run default Apollo CM simulation
      python main.py --vehicle orion          # Run with Orion capsule
      python main.py --vehicle dragon --fpa -3.5  # Custom flight path angle
      python main.py --compare                # Compare all preset vehicles

═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import os
import argparse
import numpy as np

# Ensure module imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vehicle import Vehicle, load_preset, list_presets
from simulator import ReentrySimulator, SimConfig, SimResults
from plotting import (
    plot_dashboard,
    plot_heating_analysis,
    plot_comparison,
    plot_atmosphere,
)


def run_single(args) -> SimResults:
    """Run a single vehicle simulation."""
    # Load vehicle
    vehicle = load_preset(args.vehicle)
    print(vehicle.info())

    # Configure simulation
    config = SimConfig(
        initial_altitude=args.altitude * 1e3,   # km → m
        initial_velocity=args.velocity,
        initial_fpa=args.fpa,
        bank_angle=args.bank_angle,
        max_time=args.max_time,
        dt_output=args.dt,
        solver=args.solver,
    )

    # Run simulation
    sim = ReentrySimulator(vehicle=vehicle, config=config)
    results = sim.run(verbose=True)

    # Generate plots
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n  Generating plots...")
    path1 = plot_dashboard(results, save_path=os.path.join(output_dir, "dashboard.png"))
    print(f"    ✓ Dashboard:        {path1}")

    path2 = plot_heating_analysis(results, save_path=os.path.join(output_dir, "heating_analysis.png"))
    print(f"    ✓ Heating Analysis: {path2}")

    path3 = plot_atmosphere(save_path=os.path.join(output_dir, "atmosphere_profile.png"))
    print(f"    ✓ Atmosphere Model: {path3}")

    print(f"\n  All outputs saved to: {output_dir}/")
    return results


def run_comparison(args):
    """Run comparison across all preset vehicles."""
    print("\n" + "=" * 60)
    print("  MULTI-VEHICLE COMPARISON")
    print("=" * 60)

    results_list = []
    presets = list_presets()

    for name in presets:
        vehicle = load_preset(name)
        config = SimConfig(
            initial_altitude=args.altitude * 1e3,
            initial_velocity=args.velocity,
            initial_fpa=args.fpa,
            max_time=args.max_time,
            dt_output=args.dt,
            solver=args.solver,
        )
        sim = ReentrySimulator(vehicle=vehicle, config=config)
        results = sim.run(verbose=False)
        results_list.append(results)
        print(f"  ✓ {vehicle.name}: Peak q={results.peak_heat_flux/1e6:.2f} MW/m², "
              f"Peak g={results.peak_g_load:.1f}g, "
              f"Peak T_w={results.peak_wall_temperature:.0f} K")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    path = plot_comparison(results_list, save_path=os.path.join(output_dir, "comparison.png"))
    print(f"\n  ✓ Comparison plot saved: {path}")

    # Print comparison table
    print(f"\n  {'Vehicle':<25s} {'Peak q [MW/m²]':>15s} {'Peak g [g]':>12s} "
          f"{'Peak T_w [K]':>14s} {'Heat Load [MJ/m²]':>18s}")
    print(f"  {'─'*25} {'─'*15} {'─'*12} {'─'*14} {'─'*18}")
    for r in results_list:
        print(f"  {r.vehicle_name:<25s} {r.peak_heat_flux/1e6:>15.3f} "
              f"{r.peak_g_load:>12.2f} {r.peak_wall_temperature:>14.1f} "
              f"{r.total_heat_load/1e6:>18.3f}")


def run_parametric(args):
    """Run parametric study varying flight path angle."""
    print("\n" + "=" * 60)
    print("  PARAMETRIC STUDY: Flight Path Angle")
    print("=" * 60)

    vehicle = load_preset(args.vehicle)
    fpas = np.linspace(-1.5, -15.0, 8)
    results_list = []

    for fpa in fpas:
        config = SimConfig(
            initial_altitude=args.altitude * 1e3,
            initial_velocity=args.velocity,
            initial_fpa=fpa,
            max_time=args.max_time,
            dt_output=args.dt,
            solver=args.solver,
        )
        sim = ReentrySimulator(vehicle=vehicle, config=config)
        res = sim.run(verbose=False)
        # Tag the vehicle name with FPA for legend
        res.vehicle_name = f"FPA = {fpa:.1f}°"
        results_list.append(res)
        print(f"  ✓ FPA={fpa:>6.1f}°: Peak q={res.peak_heat_flux/1e6:.2f} MW/m², "
              f"Peak g={res.peak_g_load:.1f}g")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    path = plot_comparison(results_list, save_path=os.path.join(output_dir, "parametric_fpa.png"))
    print(f"\n  ✓ Parametric plot saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Atmospheric Re-entry Trajectory & Heating Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                  # Apollo CM, default conditions
  python main.py --vehicle orion --fpa -3.0       # Orion at shallow entry
  python main.py --vehicle dragon --velocity 11000 # Dragon at lunar return speed
  python main.py --compare                         # Compare all vehicles
  python main.py --parametric                      # FPA parametric study
  python main.py --list-vehicles                   # Show available presets
        """,
    )

    # Vehicle selection
    parser.add_argument('--vehicle', type=str, default='apollo',
                        help=f'Preset vehicle name. Options: {list_presets()}')
    parser.add_argument('--list-vehicles', action='store_true',
                        help='List available vehicle presets and exit')

    # Entry conditions
    parser.add_argument('--altitude', type=float, default=120.0,
                        help='Entry altitude [km] (default: 120)')
    parser.add_argument('--velocity', type=float, default=7800.0,
                        help='Entry velocity [m/s] (default: 7800 for LEO)')
    parser.add_argument('--fpa', type=float, default=-5.0,
                        help='Flight path angle [deg] (default: -5.0)')
    parser.add_argument('--bank-angle', type=float, default=0.0,
                        help='Bank angle [deg] (default: 0)')

    # Simulation settings
    parser.add_argument('--max-time', type=float, default=2000.0,
                        help='Maximum simulation time [s] (default: 2000)')
    parser.add_argument('--dt', type=float, default=0.5,
                        help='Output time step [s] (default: 0.5)')
    parser.add_argument('--solver', type=str, default='RK45',
                        choices=['RK45', 'RK23', 'DOP853', 'Radau', 'BDF'],
                        help='ODE solver method (default: RK45)')

    # Run modes
    parser.add_argument('--compare', action='store_true',
                        help='Compare all preset vehicles')
    parser.add_argument('--parametric', action='store_true',
                        help='Run FPA parametric study')

    # Output
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='Output directory (default: ./results)')

    args = parser.parse_args()

    # Handle list-vehicles
    if args.list_vehicles:
        print("\nAvailable vehicle presets:")
        for name in list_presets():
            v = load_preset(name)
            print(f"\n  {name}:")
            print(v.info())
        return

    # Run selected mode
    if args.compare:
        run_comparison(args)
    elif args.parametric:
        run_parametric(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
