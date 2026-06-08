"""Public physics-engine exports for :mod:`cislunar.physics`."""

from . import constants
from .forces.attitude import AttitudeConfig, AttitudeDynamicsModel
from .forces.comms import (
    CommsConfig,
    CommsPointingState,
    blackout_fraction_low_lunar_orbit,
    comms_blackout,
    is_occulted_by_moon,
    is_visible_from_dsn,
    link_budget,
)
from .forces.drag import AtmosphericDragModel, SpaceWeatherClient, f107_for_date
from .forces.eclipse import (
    R_EARTH_M,
    R_MOON_M,
    R_SUN_M,
    PowerBudgetModel,
    SolarArrayConfig,
    bisect_eclipse_boundary,
    body_shadow_illumination,
    eclipse_fraction_circular_orbit,
    is_in_eclipse,
    reflected_flux_ratio,
    sail_occultation_fraction,
    shadow_fraction,
)
from .forces.ephemeris import (
    EphemerisOutOfRangeError,
    build_cislunar_ephemeris,
    ephemeris_error_vs_circular,
)
from .forces.events import CheckpointEvent, EclipseEvent, SurfaceEvent
from .forces.gravity import GravityModel
from .forces.mascon import (
    GRAIL_MASCONS,
    LunarHarmonicGravity,
    default_lunar_harmonic_gravity,
    lunar_harmonic_acceleration,
    lunar_mascon_acceleration,
    mascon_acceleration_at_altitude,
)
from .forces.plume import plume_impingement_force
from .forces.propulsion import IonThrusterModel, SolarSailModel
from .spacecraft import (
    Action,
    Event,
    EventResult,
    IntegratorConfig,
    Spacecraft,
    make_lunar_craft,
)
from .state import Checkpoint, SpacecraftState

__all__ = [
    "SpacecraftState",
    "Checkpoint",
    "Spacecraft",
    "Action",
    "IntegratorConfig",
    "Event",
    "EventResult",
    "make_lunar_craft",
    "CheckpointEvent",
    "EclipseEvent",
    "SurfaceEvent",
    "GravityModel",
    "SolarSailModel",
    "IonThrusterModel",
    "AtmosphericDragModel",
    "SpaceWeatherClient",
    "f107_for_date",
    "PowerBudgetModel",
    "SolarArrayConfig",
    "AttitudeDynamicsModel",
    "AttitudeConfig",
    "build_cislunar_ephemeris",
    "ephemeris_error_vs_circular",
    "EphemerisOutOfRangeError",
    "lunar_mascon_acceleration",
    "lunar_harmonic_acceleration",
    "default_lunar_harmonic_gravity",
    "LunarHarmonicGravity",
    "GRAIL_MASCONS",
    "mascon_acceleration_at_altitude",
    "CommsConfig",
    "CommsPointingState",
    "comms_blackout",
    "is_occulted_by_moon",
    "blackout_fraction_low_lunar_orbit",
    "is_visible_from_dsn",
    "link_budget",
    "is_in_eclipse",
    "eclipse_fraction_circular_orbit",
    "shadow_fraction",
    "bisect_eclipse_boundary",
    "body_shadow_illumination",
    "sail_occultation_fraction",
    "reflected_flux_ratio",
    "plume_impingement_force",
    "R_SUN_M",
    "R_EARTH_M",
    "R_MOON_M",
    "constants",
]
