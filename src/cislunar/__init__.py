"""
cislunar-sim — high-fidelity cislunar spacecraft simulation.

Public API is re-exported here for convenience:

    from cislunar import Spacecraft, make_lunar_craft, Action
    from cislunar.physics import GravityModel, SolarSailModel
    from cislunar.guidance import GVEPeriapsisGuidance
"""

from importlib.metadata import version as _version

__version__ = _version("cislunar-sim")

from cislunar.physics import (
    Action,
    AtmosphericDragModel,
    AttitudeConfig,
    AttitudeDynamicsModel,
    Checkpoint,
    CheckpointEvent,
    EclipseEvent,
    Event,
    EventResult,
    GravityModel,
    IntegratorConfig,
    IonThrusterModel,
    PowerBudgetModel,
    SolarArrayConfig,
    SolarSailModel,
    Spacecraft,
    SpacecraftState,
    SurfaceEvent,
    make_lunar_craft,
)

__all__ = [
    "__version__",
    "Action",
    "AttitudeConfig",
    "AttitudeDynamicsModel",
    "AtmosphericDragModel",
    "Checkpoint",
    "CheckpointEvent",
    "EclipseEvent",
    "Event",
    "EventResult",
    "GravityModel",
    "IntegratorConfig",
    "IonThrusterModel",
    "PowerBudgetModel",
    "SolarArrayConfig",
    "SolarSailModel",
    "Spacecraft",
    "SpacecraftState",
    "SurfaceEvent",
    "make_lunar_craft",
]
