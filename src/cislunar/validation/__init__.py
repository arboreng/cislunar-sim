"""Public validation and telemetry exports for :mod:`cislunar.validation`."""

from .data_paths import (
    DEFAULT_BEACON_FILE,
    DEFAULT_LIGHTSAIL2_ARCHIVE,
    DEFAULT_VALIDATION_DATA_DIR,
    VALIDATION_DATA_ENV_VAR,
)
from .eclipse_validator import (
    assess_transition_windows,
    beacon_cadence_segments,
    compare_eclipse_timing,
    extract_eclipse_transitions,
    summarize_beacon_coverage,
)
from .telemetry_fetcher import (
    IKAROS_SAIL_AREA_M2,
    IKAROS_SAIL_DISTANCE_AU,
    IKAROS_SAIL_THRUST_REF_N,
    available_lightsail2_epochs,
    fetch_lightsail2,
    load_historical_tle_series,
)
from .telemetry_replay import (
    ReplayMetrics,
    TelemetryDataset,
    TelemetryPoint,
    generate_lightsail2_synthetic,
    replay,
)

__all__ = [
    "TelemetryPoint",
    "TelemetryDataset",
    "ReplayMetrics",
    "generate_lightsail2_synthetic",
    "replay",
    "fetch_lightsail2",
    "available_lightsail2_epochs",
    "IKAROS_SAIL_AREA_M2",
    "IKAROS_SAIL_THRUST_REF_N",
    "IKAROS_SAIL_DISTANCE_AU",
    "load_historical_tle_series",
    "DEFAULT_VALIDATION_DATA_DIR",
    "DEFAULT_LIGHTSAIL2_ARCHIVE",
    "DEFAULT_BEACON_FILE",
    "VALIDATION_DATA_ENV_VAR",
    "extract_eclipse_transitions",
    "compare_eclipse_timing",
    "beacon_cadence_segments",
    "assess_transition_windows",
    "summarize_beacon_coverage",
]
