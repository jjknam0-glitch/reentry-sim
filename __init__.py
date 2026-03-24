"""
ReEntry Sim — Atmospheric Re-entry Trajectory & Aerothermal Heating Simulator
==============================================================================

Simulation of Atmospheric Re-entry Trajectory and Aerothermal Heating
for Blunt Body Vehicles.

Modules:
    atmosphere  — US Standard Atmosphere 1976 model
    vehicle     — Re-entry vehicle configuration & presets
    physics     — Equations of motion & heating correlations
    simulator   — Core ODE integration engine
    plotting    — Professional visualization
"""

__version__ = "1.0.0"
__author__ = "ReEntry Sim"

from .atmosphere import atmosphere, density, temperature, gravity
from .vehicle import Vehicle, load_preset, list_presets
from .simulator import ReentrySimulator, SimConfig, SimResults
