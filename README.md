# Simulation of Atmospheric Re-entry Trajectory and Aerothermal Heating for Blunt Body Vehicles

A Python-based simulation tool that models the trajectory and aerothermal heating of a blunt-body vehicle during atmospheric re-entry. It enables engineers and students to predict critical parameters such as heat flux, surface temperature, and g-loading to support early-stage thermal protection system design.

---

## Features

- **US Standard Atmosphere 1976** model (0–120 km) with exponential extrapolation above 86 km
- **Sutton-Graves** stagnation-point convective heating correlation
- **Lumped-mass thermal model** with radiative cooling and ablation support
- **2D planar trajectory** with drag, lift, gravity, and centrifugal correction
- **5 preset vehicles**: Apollo CM, Orion MPCV, SpaceX Dragon, Stardust SRC, Generic Capsule
- **Bank angle modulation** for guided re-entry simulation
- **Professional dark-themed dashboards** with 6-panel trajectory overview
- **Comparison mode** for multi-vehicle analysis
- **Parametric study mode** for flight path angle sweeps
- **Configurable ODE solver** (RK45, RK23, DOP853, Radau, BDF)

## Project Structure

```
reentry_sim/
├── main.py          # CLI entry point with argument parsing
├── atmosphere.py    # US Standard Atmosphere 1976 model
├── vehicle.py       # Vehicle configuration & presets
├── physics.py       # Equations of motion & heating models
├── simulator.py     # ODE integration engine
├── plotting.py      # Matplotlib visualization
├── __init__.py      # Package init
└── README.md        # This file
```

## Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib

```bash
pip install numpy scipy matplotlib
```

## Usage

### Basic Simulation (Apollo CM at LEO re-entry)
```bash
python main.py
```

### Custom Vehicle and Conditions
```bash
python main.py --vehicle orion --fpa -3.0 --velocity 7800
```

### Lunar Return Speed
```bash
python main.py --vehicle apollo --velocity 11000 --fpa -6.5
```

### Compare All Vehicles
```bash
python main.py --compare
```

### Flight Path Angle Parametric Study
```bash
python main.py --parametric --vehicle apollo
```

### List Available Vehicles
```bash
python main.py --list-vehicles
```

### All Options
```
--vehicle       Preset name: apollo, orion, dragon, stardust, generic
--altitude      Entry altitude in km (default: 120)
--velocity      Entry velocity in m/s (default: 7800)
--fpa           Flight path angle in degrees (default: -5.0)
--bank-angle    Bank angle in degrees (default: 0)
--max-time      Max simulation time in seconds (default: 2000)
--dt            Output time step in seconds (default: 0.5)
--solver        ODE solver: RK45, RK23, DOP853, Radau, BDF
--compare       Run multi-vehicle comparison
--parametric    Run FPA parametric study
--output-dir    Output directory (default: ./results)
```

## Physics Models

### Atmospheric Model
US Standard Atmosphere 1976 with 7 layers from sea level to 86 km geopotential altitude. Above 86 km, an exponential decay model is used.

### Heating Model
Sutton-Graves stagnation-point convective heating:
```
q_conv = k × √(ρ / R_n) × V³
```
where k = 1.7415×10⁻⁴ kg⁰·⁵/m for air.

### Thermal Model
Lumped-mass energy balance with radiative cooling:
```
dT_wall/dt = (q_conv − ε·σ·T_wall⁴) / (ρ_w · c_w · δ)
```

### Equations of Motion
2D planar trajectory with state vector [altitude, velocity, flight path angle, downrange, wall temperature]:
```
dz/dt = V·sin(γ)
dV/dt = −D/m − g·sin(γ)
dγ/dt = (1/V)·[(L·cos(σ)/m) − (g − V²/r)·cos(γ)]
ds/dt = V·cos(γ)·R_earth / r
```

## Output

The simulator generates:
1. **dashboard.png** — 6-panel trajectory overview
2. **heating_analysis.png** — Detailed aerothermal analysis
3. **atmosphere_profile.png** — Atmospheric model validation
4. **comparison.png** — Multi-vehicle comparison (if --compare)
5. **parametric_fpa.png** — FPA parametric study (if --parametric)

## License

MIT License — Free for educational and research use.
