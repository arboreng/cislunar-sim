"""
Communications blackout detection.

A spacecraft loses contact with Earth when the Moon or Earth itself
occludes the line of sight. This affects:
  - obs[33] comms_delay — clamped to 1.0 (maximum) during blackout
  - mission analysis — operations may need to avoid blackout windows

Occultation geometry
─────────────────────
Moon occultation: cylindrical check (exact for a spherical body LOS test).
Earth horizon:    uses three DSN stations; blackout when no station has
                  line-of-sight above the 5° elevation mask.

DSN station positions
──────────────────────
Three-station Deep Space Network (Goldstone CA, Canberra AU, Madrid ES)
spaced ~120° apart in longitude.  Together they provide near-continuous
coverage except during brief Earth horizon occultations.

APPROXIMATION: station ECI positions are pre-computed at J2000.0 (t=0)
and then rotated by Earth's sidereal rotation at the current sim time.
Earth's sidereal rotation rate: ω_E = 7.2921150e-5 rad/s.
Error from ignoring polar motion, nutation, and precession: < 20 m at the
station position — negligible for a 5° elevation mask check.

APPROXIMATION: Earth is treated as a sphere (mean radius 6371 km).
The actual oblate geoid changes the horizon altitude by < 0.1°.

Blackout durations (approximate):
  Low lunar orbit (200 km): ~30 min blackout per ~2-hour orbit (25% of time)
  Trans-lunar coast (midpoint): ~45 min per event, much less frequent
  LEO behind Earth (from single station): ~15–20 min per orbit (~18%)
  LEO with 3 DSN stations: almost never blocked except near-antipodal passes
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

R_MOON_M = 1.7374e6
R_EARTH_M = 6.371e6
OMEGA_EARTH = 7.2921150e-5  # Earth sidereal rotation rate [rad/s]

# Minimum elevation angle for a DSN station to communicate [rad]
DSN_ELEVATION_MASK_RAD = math.radians(5.0)


class StationCandidate(TypedDict):
    """A DSN station candidate that satisfies geometry and link-margin checks."""

    idx: int
    station: NDArray[np.float64]
    target_hat: NDArray[np.float64]
    range_m: float
    elevation_rad: float
    margin_db: float


# ── DSN station ECEF positions ────────────────────────────────────────────────
# Geodetic coordinates → ECEF (assuming spherical Earth for consistency)
def _ecef(lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> NDArray:
    """Approximate ECEF position assuming spherical Earth."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    r = R_EARTH_M + alt_m
    return np.array(
        [
            r * math.cos(lat) * math.cos(lon),
            r * math.cos(lat) * math.sin(lon),
            r * math.sin(lat),
        ]
    )


# DSN complex coordinates (geodetic lat/lon, approximate)
_DSN_ECEF = np.array(
    [
        _ecef(35.43, 243.11, 1_073.0),  # Goldstone, CA  (altitude ~1073 m)
        _ecef(-35.40, 148.98, 688.0),  # Canberra, AU   (altitude ~688 m)
        _ecef(40.43, 355.75, 834.0),  # Madrid, ES     (altitude ~834 m)
    ],
    dtype=np.float64,
)  # shape [3, 3] — three stations × xyz


@dataclass
class CommsConfig:
    """Configuration for antenna pointing and S-band link closure."""

    elevation_mask_rad: float = DSN_ELEVATION_MASK_RAD
    max_slew_rate_deg_s: float = float("inf")
    acquisition_tolerance_deg: float = 0.25
    frequency_hz: float = 2.2e9
    tx_power_w: float = 5.0
    tx_gain_dbi: float = 8.0
    rx_gain_dbi: float = 45.0
    system_loss_db: float = 2.0
    required_rx_power_dbw: float = -161.0
    stations_ecef_m: NDArray[np.float64] = field(default_factory=lambda: _DSN_ECEF.copy())


@dataclass
class CommsPointingState:
    """Persistent antenna pointing state across comms_blackout() calls."""

    boresight_eci: NDArray[np.float64] | None = None
    tracked_station_idx: int | None = None
    acquired: bool = False
    last_time_s: float | None = None
    acquisition_remaining_s: float = 0.0


def _dsn_eci(time_s: float, stations_ecef_m: NDArray[np.float64] | None = None) -> NDArray:
    """
    DSN station positions in ECI at simulation time t [s].

    Applies Earth's sidereal rotation to the J2000.0 ECEF positions.
    """
    theta = OMEGA_EARTH * time_s  # sidereal rotation angle [rad]
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    # Rotation matrix: ECEF → ECI (z-axis rotation by theta)
    R = np.array(
        [
            [cos_t, -sin_t, 0.0],
            [sin_t, cos_t, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    stations = (
        _DSN_ECEF if stations_ecef_m is None else np.asarray(stations_ecef_m, dtype=np.float64)
    )
    return stations @ R.T  # [N, 3] — each row is one station in ECI


def _unit(vec: NDArray[np.float64]) -> NDArray[np.float64]:
    mag = float(np.linalg.norm(vec))
    if mag < 1e-12:
        return vec.copy()
    return vec / mag


def _rotate_toward(
    current_hat: NDArray[np.float64],
    target_hat: NDArray[np.float64],
    max_angle_rad: float,
) -> NDArray[np.float64]:
    """Rotate current_hat toward target_hat by at most max_angle_rad."""
    dot = float(np.clip(np.dot(current_hat, target_hat), -1.0, 1.0))
    angle = math.acos(dot)
    if angle < 1e-12 or max_angle_rad >= angle:
        return target_hat.copy()

    sin_total = math.sin(angle)
    if abs(sin_total) < 1e-12:
        return target_hat.copy()

    w0 = math.sin(angle - max_angle_rad) / sin_total
    w1 = math.sin(max_angle_rad) / sin_total
    return _unit(w0 * current_hat + w1 * target_hat)


def free_space_path_loss_db(range_m: float, frequency_hz: float) -> float:
    """Friis free-space path loss [dB] for one-way range at a given frequency."""
    range_m = max(float(range_m), 1.0)
    wavelength_m = 299_792_458.0 / float(frequency_hz)
    return 20.0 * math.log10(4.0 * math.pi * range_m / wavelength_m)


def link_budget(
    range_m: float,
    tx_power_w: float,
    tx_gain_dbi: float,
    rx_gain_dbi: float,
    *,
    frequency_hz: float = 2.2e9,
    system_loss_db: float = 2.0,
    required_rx_power_dbw: float = -161.0,
) -> float:
    """
    S-band receive-power margin [dB].

    Positive margin means the downlink closes with the configured gains and
    minimum required received power.
    """
    tx_power_dbw = 10.0 * math.log10(max(float(tx_power_w), 1e-12))
    received_power_dbw = (
        tx_power_dbw
        + float(tx_gain_dbi)
        + float(rx_gain_dbi)
        - free_space_path_loss_db(range_m, frequency_hz)
        - float(system_loss_db)
    )
    return received_power_dbw - float(required_rx_power_dbw)


def _visible_station_candidates(
    spacecraft_pos_eci: NDArray[np.float64],
    time_s: float,
    cfg: CommsConfig,
    tx_power_w: float,
) -> list[StationCandidate]:
    """Stations with geometric LOS above mask and positive link margin."""
    stations = _dsn_eci(time_s, cfg.stations_ecef_m)
    candidates: list[StationCandidate] = []

    for idx, station in enumerate(stations):
        rel = spacecraft_pos_eci - station
        rel_mag = float(np.linalg.norm(rel))
        if rel_mag < 1.0:
            margin_db = link_budget(
                1.0,
                tx_power_w,
                cfg.tx_gain_dbi,
                cfg.rx_gain_dbi,
                frequency_hz=cfg.frequency_hz,
                system_loss_db=cfg.system_loss_db,
                required_rx_power_dbw=cfg.required_rx_power_dbw,
            )
            candidates.append(
                {
                    "idx": idx,
                    "station": station,
                    "target_hat": _unit(station - spacecraft_pos_eci),
                    "range_m": 1.0,
                    "elevation_rad": math.pi / 2.0,
                    "margin_db": margin_db,
                }
            )
            continue

        up = station / float(np.linalg.norm(station))
        rel_hat = rel / rel_mag
        sin_el = float(np.dot(rel_hat, up))
        if sin_el < math.sin(cfg.elevation_mask_rad):
            continue

        proj = float(np.dot(-station, rel_hat))
        if 0.0 < proj < rel_mag:
            perp = -station - proj * rel_hat
            if float(np.linalg.norm(perp)) < R_EARTH_M:
                continue

        margin_db = link_budget(
            rel_mag,
            tx_power_w,
            cfg.tx_gain_dbi,
            cfg.rx_gain_dbi,
            frequency_hz=cfg.frequency_hz,
            system_loss_db=cfg.system_loss_db,
            required_rx_power_dbw=cfg.required_rx_power_dbw,
        )
        if margin_db <= 0.0:
            continue

        candidates.append(
            {
                "idx": idx,
                "station": station,
                "target_hat": _unit(station - spacecraft_pos_eci),
                "range_m": rel_mag,
                "elevation_rad": math.asin(max(-1.0, min(1.0, sin_el))),
                "margin_db": margin_db,
            }
        )

    return candidates


def is_visible_from_dsn(
    spacecraft_pos_eci: NDArray[np.float64],
    time_s: float = 0.0,
    elevation_mask_rad: float = DSN_ELEVATION_MASK_RAD,
    tx_power_w: float = 5.0,
    config: CommsConfig | None = None,
) -> bool:
    """
    True if at least one DSN station has line-of-sight to the spacecraft
    above the elevation mask and without Earth blocking the path.

    Args:
        spacecraft_pos_eci: spacecraft position in ECI [m]
        time_s:             simulation elapsed time [s]  (drives ECEF→ECI rotation)
        elevation_mask_rad: minimum elevation angle above horizon [rad]

    Returns:
        True if the spacecraft is contactable from at least one DSN station

    PHASE 1 FIDELITY LIMIT — comms blackout
    ────────────────────────────────────────
    This model is useful for scenario analysis but remains simplified in four ways:

    1. Spherical Earth: the occulting body is modelled as a sphere of radius
       R_EARTH.  Real Earth oblateness (flattening f ≈ 1/298) shifts the
       geometric horizon by up to ~20 km at grazing angles, affecting eclipse
       and horizon contact timings by ~30–60 s per pass.

    2. Frozen station kinematics: DSN station ECI positions are computed by
       rotating fixed ECEF positions using a simple sidereal-rate rotation
       (see _dsn_eci()).  True ECEF coordinates include polar wander and UT1
       corrections at the ~10 m level; negligible for a 5° elevation mask.

    3. No antenna pointing constraints: the model grants contact whenever
       geometric line-of-sight exists above the elevation mask.  Real
       high-gain antennas require explicit pointing commands and have finite
       slew rates; the window during which the antenna has both LOS and
       correct pointing is shorter than geometric LOS alone.

    4. No link margin dynamics: received signal strength is not modelled.
       Real blackout boundaries depend on free-space path loss, atmospheric
       attenuation, transmit power, and receiver noise figure.  At cislunar
       distances path loss reaches ~180 dB and link closure is not guaranteed
       even with clear LOS.

    These simplifications are acceptable for simulation purposes (where blackout
    is a coarse flag, not a telemetry truth).  For operations-quality comms
    modelling, replace this function with a full link budget calculation using
    actual DSN aperture sizes and spacecraft EIRP.
    """
    cfg = config or CommsConfig(elevation_mask_rad=elevation_mask_rad)
    cfg.elevation_mask_rad = elevation_mask_rad
    return bool(_visible_station_candidates(spacecraft_pos_eci, time_s, cfg, tx_power_w))


def is_occulted_by_moon(
    spacecraft_pos_eci: NDArray,
    moon_pos_eci: NDArray,
) -> bool:
    """
    True if the Moon blocks line-of-sight from Earth to the spacecraft.

    Uses a cylindrical shadow model — exact for a spherical occultation
    body when checking a straight line-of-sight.  Earth is at the ECI origin.

    APPROXIMATION: cylindrical (not conical) — error < 0.1% for LOS checks.

    Args:
        spacecraft_pos_eci: spacecraft position [m]
        moon_pos_eci:       Moon position [m]
    """
    sc_mag = float(np.linalg.norm(spacecraft_pos_eci))
    if sc_mag < 1.0:
        return False

    sc_hat = spacecraft_pos_eci / sc_mag

    # Projection of Moon along Earth→spacecraft direction
    proj = float(np.dot(moon_pos_eci, sc_hat))

    # Moon must lie strictly between Earth and spacecraft
    if not (0.0 < proj < sc_mag):
        return False

    # Perpendicular distance from Moon centre to the line
    perp_vec = moon_pos_eci - proj * sc_hat
    perp_dist = float(np.linalg.norm(perp_vec))

    return perp_dist < R_MOON_M


def comms_blackout(
    spacecraft_pos_eci: NDArray[np.float64],
    moon_pos_eci: NDArray[np.float64],
    time_s: float = 0.0,
    tx_power_w: float | None = None,
    config: CommsConfig | None = None,
    pointing_state: CommsPointingState | None = None,
) -> bool:
    """
    Combined blackout check: occulted by Moon, or no DSN station visible.

    Returns True if the spacecraft has no usable link to Earth.

    Args:
        spacecraft_pos_eci: spacecraft position in ECI [m]
        moon_pos_eci:       Moon position in ECI [m]
        time_s:             simulation elapsed time [s]  (for DSN station rotation)
    """
    cfg = config or CommsConfig()
    tx_power = cfg.tx_power_w if tx_power_w is None else tx_power_w

    if is_occulted_by_moon(spacecraft_pos_eci, moon_pos_eci):
        if pointing_state is not None:
            pointing_state.acquired = False
            pointing_state.tracked_station_idx = None
            pointing_state.acquisition_remaining_s = 0.0
            pointing_state.last_time_s = time_s
        return True

    candidates = _visible_station_candidates(spacecraft_pos_eci, time_s, cfg, tx_power)
    if not candidates:
        if pointing_state is not None:
            if pointing_state.boresight_eci is None:
                pointing_state.boresight_eci = _unit(-spacecraft_pos_eci)
            pointing_state.acquired = False
            pointing_state.tracked_station_idx = None
            pointing_state.acquisition_remaining_s = 0.0
            pointing_state.last_time_s = time_s
        return True

    if pointing_state is None:
        return False

    target = None
    if pointing_state.tracked_station_idx is not None:
        for candidate in candidates:
            if candidate["idx"] == pointing_state.tracked_station_idx:
                target = candidate
                break
    if target is None:
        target = max(
            candidates,
            key=lambda cand: (
                float(cand["margin_db"]),
                float(cand["elevation_rad"]),
            ),
        )

    target_hat = np.asarray(target["target_hat"], dtype=np.float64)
    dt_s = 0.0
    if pointing_state.last_time_s is not None and time_s >= pointing_state.last_time_s:
        dt_s = time_s - pointing_state.last_time_s

    max_slew_rate_rad_s = math.radians(cfg.max_slew_rate_deg_s)
    station_changed = pointing_state.tracked_station_idx != int(target["idx"])
    if pointing_state.boresight_eci is None:
        pointing_state.boresight_eci = _unit(-spacecraft_pos_eci)

    if station_changed:
        angle_to_target_rad = math.acos(
            float(np.clip(np.dot(_unit(pointing_state.boresight_eci), target_hat), -1.0, 1.0))
        )
        if math.isfinite(max_slew_rate_rad_s) and max_slew_rate_rad_s > 0.0:
            pointing_state.acquisition_remaining_s = angle_to_target_rad / max_slew_rate_rad_s
        elif math.isfinite(max_slew_rate_rad_s):
            pointing_state.acquisition_remaining_s = float("inf")
        else:
            pointing_state.acquisition_remaining_s = 0.0

    if not math.isfinite(max_slew_rate_rad_s):
        pointing_state.boresight_eci = target_hat.copy()
        pointing_state.acquisition_remaining_s = 0.0
    else:
        pointing_state.acquisition_remaining_s = max(
            0.0, pointing_state.acquisition_remaining_s - dt_s
        )
        if pointing_state.acquisition_remaining_s <= 0.0:
            pointing_state.boresight_eci = target_hat.copy()

    pointing_state.acquired = pointing_state.acquisition_remaining_s <= 0.0
    pointing_state.tracked_station_idx = int(target["idx"])
    pointing_state.last_time_s = time_s
    return not pointing_state.acquired


def blackout_fraction_low_lunar_orbit(altitude_km: float = 200.0) -> float:
    """
    Fraction of a circular low lunar orbit spent in Earth-occultation blackout.

    Analytical formula (analogous to eclipse fraction for solar occultation).
    At 200 km altitude: ~25% of orbit in blackout (~30 min per ~2-hour orbit).

    APPROXIMATION: uses cylindrical occultation, equatorial orbit.
    """
    r = R_MOON_M + altitude_km * 1e3
    # Half-angle of the shadow cone as seen from Moon orbit
    half_angle = math.asin(R_MOON_M / r)
    return half_angle / math.pi
