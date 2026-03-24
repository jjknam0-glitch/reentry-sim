"""
Vehicle Configuration Module
=============================
Defines re-entry vehicle parameters using dataclasses for clean,
validated configuration management.

Includes preset vehicles based on real spacecraft data.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Vehicle:
    """
    Re-entry vehicle configuration.

    Parameters
    ----------
    name : str
        Vehicle identifier.
    mass : float
        Vehicle mass [kg].
    drag_coefficient : float
        Aerodynamic drag coefficient Cd [-].
    lift_coefficient : float
        Aerodynamic lift coefficient Cl [-].
    reference_area : float
        Aerodynamic reference area (cross-section) [m²].
    nose_radius : float
        Effective nose radius for heating calculations [m].
    emissivity : float
        Surface emissivity for radiative cooling [-].
    wall_thickness : float
        TPS wall thickness [m].
    wall_density : float
        TPS material density [kg/m³].
    wall_specific_heat : float
        TPS material specific heat capacity [J/(kg·K)].
    initial_wall_temp : float
        Initial wall/surface temperature [K].
    ablation_temp : float
        Temperature at which ablation begins [K]. Set to np.inf to disable.
    heat_of_ablation : float
        Specific heat of ablation [J/kg]. Energy absorbed per kg of TPS ablated.
    """
    name: str
    mass: float
    drag_coefficient: float
    lift_coefficient: float
    reference_area: float
    nose_radius: float
    emissivity: float = 0.85
    wall_thickness: float = 0.05
    wall_density: float = 1440.0
    wall_specific_heat: float = 1260.0
    initial_wall_temp: float = 300.0
    ablation_temp: float = np.inf
    heat_of_ablation: float = 0.0

    def __post_init__(self):
        """Validate vehicle parameters."""
        assert self.mass > 0, "Mass must be positive"
        assert self.drag_coefficient >= 0, "Cd must be non-negative"
        assert self.reference_area > 0, "Reference area must be positive"
        assert self.nose_radius > 0, "Nose radius must be positive"
        assert 0 < self.emissivity <= 1.0, "Emissivity must be in (0, 1]"

    @property
    def ballistic_coefficient(self) -> float:
        """Ballistic coefficient β = m / (Cd · A) [kg/m²]."""
        return self.mass / (self.drag_coefficient * self.reference_area)

    @property
    def tps_mass_per_area(self) -> float:
        """TPS mass per unit area [kg/m²]."""
        return self.wall_density * self.wall_thickness

    def info(self) -> str:
        """Return a formatted summary of vehicle parameters."""
        lines = [
            f"╔══════════════════════════════════════════════════╗",
            f"║  Vehicle: {self.name:<39s}║",
            f"╠══════════════════════════════════════════════════╣",
            f"║  Mass:                {self.mass:>12.1f} kg           ║",
            f"║  Drag Coefficient:    {self.drag_coefficient:>12.3f}              ║",
            f"║  Lift Coefficient:    {self.lift_coefficient:>12.3f}              ║",
            f"║  Reference Area:      {self.reference_area:>12.3f} m²           ║",
            f"║  Nose Radius:         {self.nose_radius:>12.3f} m            ║",
            f"║  Ballistic Coeff:     {self.ballistic_coefficient:>12.1f} kg/m²        ║",
            f"║  Emissivity:          {self.emissivity:>12.3f}              ║",
            f"║  TPS Thickness:       {self.wall_thickness:>12.4f} m            ║",
            f"║  Initial Wall Temp:   {self.initial_wall_temp:>12.1f} K            ║",
            f"╚══════════════════════════════════════════════════╝",
        ]
        return "\n".join(lines)


# ─── Preset Vehicles ─────────────────────────────────────────────────────────

def apollo_cm() -> Vehicle:
    """NASA Apollo Command Module (approximate)."""
    return Vehicle(
        name="Apollo Command Module",
        mass=5560.0,
        drag_coefficient=1.29,
        lift_coefficient=0.368,
        reference_area=11.63,
        nose_radius=4.694,
        emissivity=0.85,
        wall_thickness=0.04,
        wall_density=1440.0,
        wall_specific_heat=1260.0,
        initial_wall_temp=300.0,
        ablation_temp=3300.0,
        heat_of_ablation=8.0e6,
    )


def orion_capsule() -> Vehicle:
    """NASA Orion MPCV (approximate)."""
    return Vehicle(
        name="Orion MPCV",
        mass=10400.0,
        drag_coefficient=1.30,
        lift_coefficient=0.40,
        reference_area=19.63,
        nose_radius=5.03,
        emissivity=0.85,
        wall_thickness=0.05,
        wall_density=1440.0,
        wall_specific_heat=1260.0,
        initial_wall_temp=300.0,
        ablation_temp=3300.0,
        heat_of_ablation=8.0e6,
    )


def stardust_capsule() -> Vehicle:
    """NASA Stardust Sample Return Capsule (approximate)."""
    return Vehicle(
        name="Stardust SRC",
        mass=45.8,
        drag_coefficient=1.50,
        lift_coefficient=0.0,
        reference_area=0.52,
        nose_radius=0.229,
        emissivity=0.90,
        wall_thickness=0.06,
        wall_density=1830.0,
        wall_specific_heat=710.0,
        initial_wall_temp=300.0,
        ablation_temp=3700.0,
        heat_of_ablation=12.0e6,
    )


def spacex_dragon() -> Vehicle:
    """SpaceX Crew Dragon (approximate)."""
    return Vehicle(
        name="SpaceX Crew Dragon",
        mass=12520.0,
        drag_coefficient=1.27,
        lift_coefficient=0.36,
        reference_area=18.10,
        nose_radius=4.40,
        emissivity=0.85,
        wall_thickness=0.045,
        wall_density=1600.0,
        wall_specific_heat=1100.0,
        initial_wall_temp=300.0,
        ablation_temp=3400.0,
        heat_of_ablation=9.0e6,
    )


def generic_capsule() -> Vehicle:
    """Generic small re-entry capsule for testing."""
    return Vehicle(
        name="Generic Capsule",
        mass=3000.0,
        drag_coefficient=1.20,
        lift_coefficient=0.0,
        reference_area=8.0,
        nose_radius=2.0,
        emissivity=0.85,
        wall_thickness=0.03,
        wall_density=1440.0,
        wall_specific_heat=1260.0,
        initial_wall_temp=300.0,
    )


PRESETS = {
    "apollo": apollo_cm,
    "orion": orion_capsule,
    "stardust": stardust_capsule,
    "dragon": spacex_dragon,
    "generic": generic_capsule,
}


def list_presets() -> list:
    """Return list of available preset vehicle names."""
    return list(PRESETS.keys())


def load_preset(name: str) -> Vehicle:
    """Load a preset vehicle by name."""
    if name.lower() not in PRESETS:
        raise ValueError(
            f"Unknown preset '{name}'. Available: {list_presets()}"
        )
    return PRESETS[name.lower()]()
