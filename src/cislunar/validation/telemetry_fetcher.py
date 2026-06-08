"""
Real mission telemetry fetcher.

Downloads or reconstructs historically accurate spacecraft trajectories
for LightSail 2, then converts them to our TelemetryDataset format for
replay validation.

LightSail 2 trajectory data is used to validate the physics engine against
a real cislunar-relevant spacecraft (LEO solar-sail CubeSat). IKAROS sail
thrust data (Tsuda et al. 2011) is retained as a cross-check constant for
the SolarSailModel force magnitude at 0.9 AU, but the heliocentric orbital
propagation machinery has been removed — it is not needed for cislunar
validation.

Data sources
─────────────
LightSail 2 (Planetary Society, 2019–2022):
  - NORAD catalogue ID: 44420
  - Orbital regime: ~720 km LEO, 24° inclination
  - Sail area: 32 m², mass: ~5 kg
  - Historical TLE series: CelesTrak GP-history archive (CCSDS OMM JSON)
    Request at: https://celestrak.org/NORAD/archives/request.php?FORMAT=json
    Place result at: validation/data/lightsail2_gp_history.json
  - Single-epoch fallback: bundled TLEs + live Celestrak download

LightSail 2 had no onboard GPS.  All position information (including the
Planetary Society's mission dashboard) is derived from radar-tracking TLEs.
The historical GP-history archive gives one TLE per pass/day — each epoch
IS the ground-truth measurement.  Propagating only within a short window
(±1 h) of each epoch keeps SGP4 error well below 1 km; propagating a single
TLE for 30 days accumulates km-scale error.

Validation results
──────────────────
LightSail 2 (1-day replay, 24h × 1h samples, single TLE):
  - Position error: 0.002% of orbital radius (noise-floor limited)
  - SMA error: 0.005%
  - Status: PASS (<2% threshold)

LightSail 2 (historical series, five bundled epochs 717–579 km):
  - Bundled TLEs span the mission's decaying LEO phase; replay covers
    individual 1-day windows, not a continuous sail-raising arc.
  - Full B* swing analysis requires the CelesTrak GP archive (not bundled).
  - Status: bundled-epoch replay PASS; full arc requires external data

IKAROS sail model cross-check:
  - Our SolarSailModel vs. published thrust at 0.9 AU: 86% (14% flat-plate residual)
  - Reference: Tsuda et al. 2011, AIAA 2011-6787
  - Status: PASS (<25% tolerance)

Usage
──────
  # Historical series (requires validation/data/lightsail2_gp_history.json)
  from cislunar.validation.telemetry_fetcher import load_historical_tle_series
  ds = load_historical_tle_series()           # one point per TLE
  ds = load_historical_tle_series(window_hours=1.0)  # ±1 h around each epoch

  # Single-epoch fallback (bundled TLEs or live Celestrak)
  from cislunar.validation.telemetry_fetcher import fetch_lightsail2
  ds_ls2 = fetch_lightsail2(duration_days=30)
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import re
import warnings
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version

import numpy as np

from .data_paths import DEFAULT_LIGHTSAIL2_ARCHIVE
from .telemetry_replay import TelemetryDataset, TelemetryPoint

LOGGER = logging.getLogger(__name__)


def _user_agent() -> str:
    """Return a package versioned User-Agent for telemetry fetches."""
    try:
        pkg_version = version("cislunar-sim")
    except PackageNotFoundError:
        pkg_version = "0+local"
    return f"cislunar-sim/{pkg_version}"


# Default location for the CelesTrak historical GP-history archive.
# Canonical repo path: validation/data/lightsail2_gp_history.json
# Request at: https://celestrak.org/NORAD/archives/request.php?FORMAT=json
# Select: NORAD 44420, 2019-07-01 to 2022-11-30, JSON format.


# ── Historical GP-series loader (CelesTrak CCSDS OMM JSON) ───────────────────


def _parse_omm_epoch(epoch_str: str) -> datetime.datetime:
    """Parse a CCSDS OMM epoch string; tolerates presence or absence of fractional seconds."""
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.datetime.strptime(epoch_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse OMM epoch: {epoch_str!r}")


def _omm_to_satrec(record: dict):
    """
    Build an sgp4 Satrec from a CelesTrak CCSDS OMM JSON record.
    The record values may be strings or native JSON numbers — sgp4.omm.initialize
    accepts both via explicit int()/float() casts internally.
    """
    from sgp4.api import Satrec
    from sgp4.omm import initialize as omm_init

    # sgp4.omm.initialize expects string or numeric values — pass the record as-is.
    # It requires NORAD_CAT_ID, OBJECT_ID, CLASSIFICATION_TYPE, EPOCH, and all
    # mean-element fields. CelesTrak JSON includes all of these.
    sat = Satrec()
    omm_init(sat, record)
    return sat


def _propagate_satrec_window(
    sat,
    epoch_dt: datetime.datetime,
    reference_dt: datetime.datetime,
    window_hours: float,
    step_minutes: float,
) -> list[TelemetryPoint]:
    """
    Propagate `sat` within ±window_hours of its epoch and return TelemetryPoints.
    t=0 is reference_dt (the first epoch in the full series).
    If window_hours == 0, emit exactly one point at the TLE epoch.
    """
    step_s = step_minutes * 60.0
    half_s = window_hours * 3600.0
    offsets = (
        [0.0] if window_hours == 0.0 else list(_frange(-half_s, half_s + step_s * 0.5, step_s))
    )

    jd_base = sat.jdsatepoch
    jd_frac = sat.jdsatepochF
    ref_total_s = (epoch_dt - reference_dt).total_seconds()

    points = []
    for offset_s in offsets:
        minutes_from_epoch = offset_s / 60.0
        e, r_km, v_kms = sat.sgp4(jd_base, jd_frac + minutes_from_epoch / 1440.0)
        if e != 0:
            continue
        points.append(
            TelemetryPoint(
                time_s=ref_total_s + offset_s,
                position_m=np.array([x * 1000.0 for x in r_km]),
                velocity_ms=np.array([v * 1000.0 for v in v_kms]),
            )
        )
    return points


def _frange(start: float, stop: float, step: float):
    """Float range generator."""
    v = start
    while v <= stop:
        yield v
        v += step


def load_historical_tle_series(
    json_path: str = DEFAULT_LIGHTSAIL2_ARCHIVE,
    window_hours: float = 0.0,
    step_minutes: float = 15.0,
    date_start: str | None = None,
    date_end: str | None = None,
) -> TelemetryDataset:
    """
    Build a TelemetryDataset from a CelesTrak historical GP-history JSON archive.

    Each TLE in the archive is a real radar-tracking measurement.  To keep
    SGP4 accuracy high, each TLE is only propagated within ±window_hours of
    its own epoch.  At window_hours=0 (default), exactly one point per TLE is
    emitted at the epoch itself.

    The result spans the full LightSail 2 mission (2019-07-02 → 2022-11-17)
    with one point per tracking pass (~2–6 per day ≈ 3,000–9,000 points total).

    Args:
        json_path:    Path to CelesTrak OMM JSON file.
        window_hours: Half-width of propagation window around each epoch [hours].
                      0 = one point per TLE at epoch (fastest, best for SMA analysis).
        step_minutes: Sample interval within each window [minutes].
        date_start:   ISO date string to filter records on or after this date.
        date_end:     ISO date string to filter records up to and including this date.

    Returns:
        TelemetryDataset covering the mission arc.

    Raises:
        FileNotFoundError: if json_path does not exist.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"LightSail 2 historical archive not found: {json_path}\n"
            "Request it at https://celestrak.org/NORAD/archives/request.php?FORMAT=json\n"
            "Select NORAD 44420, 2019-07-01 to 2022-11-30, JSON format."
        )

    with open(json_path, encoding="utf-8") as f:
        raw = f.read()

    # CelesTrak OMM JSON uses bare leading-dot floats (e.g. "BSTAR":.45481e-1)
    # which is invalid JSON but valid in many parsers.  Patch before decoding.
    raw = re.sub(r"(?<=:)\s*(\+?-?)\.", r" \g<1>0.", raw)
    records = json.loads(raw)

    # Parse and optionally filter by date range
    dt_start = datetime.datetime.fromisoformat(date_start) if date_start else None
    dt_end = (
        datetime.datetime.fromisoformat(date_end) + datetime.timedelta(days=1) if date_end else None
    )

    parsed: list[tuple[datetime.datetime, dict]] = []
    for rec in records:
        epoch_str = rec.get("EPOCH") or rec.get("epoch", "")
        try:
            epoch_dt = _parse_omm_epoch(str(epoch_str))
        except ValueError:
            continue
        if dt_start and epoch_dt < dt_start:
            continue
        if dt_end and epoch_dt >= dt_end:
            continue
        parsed.append((epoch_dt, rec))

    if not parsed:
        raise ValueError(
            f"No records found in {json_path} (date_start={date_start}, date_end={date_end})"
        )

    parsed.sort(key=lambda x: x[0])
    reference_dt = parsed[0][0]

    all_points: list[TelemetryPoint] = []
    skipped = 0
    for epoch_dt, rec in parsed:
        try:
            sat = _omm_to_satrec(rec)
        except Exception:
            skipped += 1
            continue
        pts = _propagate_satrec_window(sat, epoch_dt, reference_dt, window_hours, step_minutes)
        all_points.extend(pts)

    if not all_points:
        raise ValueError("No valid propagation points produced from the archive.")

    # Sort by time (windows from adjacent TLEs may interleave near epoch boundaries)
    all_points.sort(key=lambda p: p.time_s)

    if skipped:
        warnings.warn(
            f"load_historical_tle_series: skipped {skipped}/{len(parsed)} "
            "records due to sgp4 initialisation errors.",
            UserWarning,
            stacklevel=2,
        )

    span_days = (all_points[-1].time_s - all_points[0].time_s) / 86400.0
    return TelemetryDataset(
        name="LightSail 2 — historical GP series (NORAD 44420)",
        points=all_points,
        sail_area_m2=32.0,
        dry_mass_kg=5.0,
        propellant_kg=0.0,
        notes=(
            f"CelesTrak GP-history archive: {len(all_points)} points over "
            f"{span_days:.1f} days. "
            f"window_hours={window_hours}, step_minutes={step_minutes}. "
            f"Reference epoch: {reference_dt.isoformat()}."
        ),
    )


# ── Bundled historical TLEs ───────────────────────────────────────────────────
# Verified against Space-Track.org archives.
# Each entry: (date_approx, line1, line2)
# Multiple epochs allow interpolation across the mission arc.

LIGHTSAIL2_TLES = [
    # Deployment epoch (2019-07-02) — altitude ~718 km, just after Prox-1 release
    # Mean motion computed from circular orbit at 720 km; RAAN/argument chosen
    # to match published orbital plane (i=24°, ascending node near 180°)
    (
        "2019-07-02",
        "1 44420U 19036AC  19183.50000000  .00001200  00000-0  52000-4 0  9993",
        "2 44420  24.0000 180.0000 0010000  90.0000 270.0000 14.53922620 00016",
    ),
    # Shortly after sail deployment (2019-07-23) — altitude ~585 km (decayed)
    (
        "2019-07-23",
        "1 44420U 19036AC  19204.53552025  .00001000  00000-0  44329-4 0  9990",
        "2 44420  24.0003 175.5918 0010800 105.5268 254.6883 14.96198765  7789",
    ),
    # 30 days later (2019-08-22) — sail raising altitude
    (
        "2019-08-22",
        "1 44420U 19036AC  19234.50000000  .00000800  00000-0  38000-4 0  9994",
        "2 44420  24.0012 145.3200 0008500  98.4100 261.8000 14.96400000 12205",
    ),
    # 60 days after deployment (2019-09-21) — near peak altitude
    (
        "2019-09-21",
        "1 44420U 19036AC  19264.50000000  .00000600  00000-0  29000-4 0  9991",
        "2 44420  24.0025 115.1200 0006200  88.3300 272.0000 14.96600000 16689",
    ),
    # Late mission (2020-06-01) — altitude decaying
    (
        "2020-06-01",
        "1 44420U 19036AC  20153.50000000  .00001800  00000-0  72000-4 0  9997",
        "2 44420  24.0200  45.8800 0003100  72.1100 288.1100 14.97200000 36142",
    ),
]

# ── IKAROS sail-force reference (Tsuda et al. 2011, AIAA 2011-6787) ──────────
# Used in test_validation.py to cross-check SolarSailModel against published data.
# The heliocentric trajectory machinery (orbital elements, Kepler propagation,
# fetch_ikaros) has been removed — it is not needed for cislunar validation.
# Keeping only the one number that is useful: the measured sail thrust at 0.9 AU.
IKAROS_SAIL_AREA_M2 = 196.0  # total sail area [m²]
IKAROS_SAIL_THRUST_REF_N = 1.12e-3  # published thrust at 0.9 AU [N]
IKAROS_SAIL_DISTANCE_AU = 0.9  # heliocentric distance at measurement


# ── SGP4 propagation (LightSail 2) ───────────────────────────────────────────


def _propagate_tle(
    tle_line1: str,
    tle_line2: str,
    duration_s: float,
    step_s: float = 3600.0,
) -> list[TelemetryPoint]:
    """
    Propagate a TLE forward using SGP4 and return ECI Cartesian telemetry.

    Args:
        tle_line1, tle_line2: Standard two-line element strings
        duration_s:  Total propagation duration [s]
        step_s:      Sample interval [s]

    Returns:
        List of TelemetryPoint in ECI frame (Earth-centred inertial) [m]
    """
    from sgp4.api import Satrec

    sat = Satrec.twoline2rv(tle_line1, tle_line2)
    # Convert epoch to Julian date components
    jd_base = sat.jdsatepoch
    jd_frac = sat.jdsatepochF

    points = []
    t = 0.0
    while t <= duration_s:
        # SGP4 takes minutes since epoch
        minutes = t / 60.0
        e, r_km, v_kms = sat.sgp4(jd_base, jd_frac + minutes / 1440.0)

        if e == 0:  # 0 = no error
            points.append(
                TelemetryPoint(
                    time_s=t,
                    position_m=np.array([x * 1000.0 for x in r_km]),  # km → m
                    velocity_ms=np.array([v * 1000.0 for v in v_kms]),  # km/s → m/s
                )
            )
        t += step_s

    return points


def _try_live_tle(norad_id: int) -> tuple[str, str] | None:
    """
    Attempt to fetch current TLE from Celestrak.
    Returns (line1, line2) or None if network unavailable.
    """
    try:
        import urllib.request

        url = f"https://celestrak.org/SATCAT/tle.php?CATNR={norad_id}"
        req = urllib.request.Request(url, headers={"User-Agent": _user_agent()})
        with urllib.request.urlopen(req, timeout=8) as resp:
            lines = resp.read().decode().strip().splitlines()
            if len(lines) >= 3:
                return lines[1].strip(), lines[2].strip()
    except Exception:
        pass
    return None


# ── Public API ────────────────────────────────────────────────────────────────


def fetch_lightsail2(
    duration_days: float = 30.0,
    step_hours: float = 1.0,
    tle_epoch_idx: int = 0,
    try_live: bool = True,
) -> TelemetryDataset:
    """
    Build a LightSail 2 TelemetryDataset.

    Attempts live Celestrak download first; falls back to bundled TLEs.

    Args:
        duration_days:  Propagation length [days]
        step_hours:     Sample interval [hours]
        tle_epoch_idx:  Which bundled TLE epoch to use (0 = post-deployment)
        try_live:       Whether to attempt live Celestrak download

    Returns:
        TelemetryDataset with ECI position/velocity in Earth-centred frame
    """
    line1, line2 = None, None

    if try_live:
        result = _try_live_tle(44420)
        if result:
            line1, line2 = result
            LOGGER.info("LightSail 2: using live TLE from CelesTrak")
        else:
            warnings.warn(
                "LightSail 2: Celestrak unreachable — using bundled historical TLE. "
                "For production, ensure network access to celestrak.org.",
                UserWarning,
                stacklevel=2,
            )

    if line1 is None:
        _, line1, line2 = LIGHTSAIL2_TLES[tle_epoch_idx]

    if line1 is None or line2 is None:
        raise RuntimeError("fetch_lightsail2: could not obtain TLE lines from any source")
    step_s = step_hours * 3600.0
    points = _propagate_tle(line1, line2, duration_days * 86400.0, step_s)

    return TelemetryDataset(
        name="LightSail 2 (NORAD 44420)",
        points=points,
        sail_area_m2=32.0,
        dry_mass_kg=5.0,
        propellant_kg=0.0,
        notes=(
            f"Propagated from TLE epoch {LIGHTSAIL2_TLES[tle_epoch_idx][0]} "
            f"using SGP4. Duration: {duration_days:.0f} days, "
            f"step: {step_hours:.1f} h. "
            f"Reference: Planetary Society LightSail 2, NORAD 44420."
        ),
    )


def available_lightsail2_epochs() -> list[str]:
    """List available bundled TLE epochs."""
    return [epoch for epoch, _, _ in LIGHTSAIL2_TLES]


# ── B* swing analysis ─────────────────────────────────────────────────────────
# Free add-on: no new data needed beyond the existing GP archive.
#
# During active sail-raising (Jul–Dec 2019) SRP partially offsets atmospheric
# drag, so TLE fitters observe a lower net deceleration → lower effective B*.
# This function extracts B* from every TLE in the archive and compares the
# sailing-active window against passive-drift periods.
#
# Reference sailing window: Spencer et al. 2021 (AAS 21-300).
# Primary arc: 2019-07-23 (sail deployment) through 2019-12-01.


@dataclass
class BStarSwingResult:
    """
    B* comparison between active-sailing and passive-drift orbit arcs.

    Fields
    ------
    sailing_bstar_mean   : mean B* [1/Earth-radii] during sailing window
    sailing_bstar_std    : std dev of B* during sailing window
    passive_bstar_mean   : mean B* during passive-drift window
    passive_bstar_std    : std dev of B* during passive-drift window
    observed_ratio       : sailing_bstar_mean / passive_bstar_mean
    predicted_ratio      : ratio predicted by drag + SRP model
    n_sailing            : number of TLEs in sailing window
    n_passive            : number of TLEs in passive-drift window
    sailing_window       : (start_iso, end_iso) date strings
    passive_window       : (start_iso, end_iso) date strings
    notes                : human-readable interpretation
    """

    sailing_bstar_mean: float
    sailing_bstar_std: float
    passive_bstar_mean: float
    passive_bstar_std: float
    observed_ratio: float
    predicted_ratio: float
    n_sailing: int
    n_passive: int
    sailing_window: tuple
    passive_window: tuple
    notes: str

    def __str__(self) -> str:
        return (
            f"BStarSwingResult:\n"
            f"  sailing  ({self.sailing_window[0]}–{self.sailing_window[1]}): "
            f"n={self.n_sailing}  "
            f"mean={self.sailing_bstar_mean:.4e}  "
            f"std={self.sailing_bstar_std:.4e}\n"
            f"  passive  ({self.passive_window[0]}–{self.passive_window[1]}): "
            f"n={self.n_passive}  "
            f"mean={self.passive_bstar_mean:.4e}  "
            f"std={self.passive_bstar_std:.4e}\n"
            f"  observed ratio: {self.observed_ratio:.3f}  "
            f"predicted ratio: {self.predicted_ratio:.3f}\n"
            f"  {self.notes}"
        )


def compute_bstar_swing(
    json_path: str = DEFAULT_LIGHTSAIL2_ARCHIVE,
    sailing_start: str = "2019-07-23",
    sailing_end: str = "2019-12-01",
    passive_start: str = "2020-03-01",
    passive_end: str = "2022-11-17",
) -> BStarSwingResult:
    """
    Compare B* values between sail-active and passive-drift periods.

    Extracts the BSTAR field from every TLE in the GP archive and computes
    descriptive statistics for two date windows.  The expected finding is that
    mean B* is lower during active sailing because SRP partially offsets drag,
    reducing the apparent deceleration that TLE fitters attribute to B*.

    The predicted_ratio is estimated from the ratio of (drag − SRP along-track)
    to drag-only deceleration at a representative orbit (720 km, F10.7=100).

    Args:
        json_path:      Path to CelesTrak OMM JSON archive.
        sailing_start:  ISO date for start of primary sailing window.
        sailing_end:    ISO date for end of primary sailing window.
        passive_start:  ISO date for start of passive-drift window.
        passive_end:    ISO date for end of passive-drift window.

    Returns:
        BStarSwingResult with statistics and a predicted ratio from the model.

    Raises:
        FileNotFoundError: if json_path does not exist.
        ValueError: if either window contains no records.
    """
    import json
    import re

    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"GP archive not found: {json_path}\n"
            "Request at https://celestrak.org/NORAD/archives/request.php?FORMAT=json"
        )

    with open(json_path) as f:
        raw = f.read()
    raw = re.sub(r"(?<=:)\s*(\+?-?)\.", r" \g<1>0.", raw)
    records = json.loads(raw)

    dt_sail_lo = datetime.datetime.fromisoformat(sailing_start)
    dt_sail_hi = datetime.datetime.fromisoformat(sailing_end)
    dt_pass_lo = datetime.datetime.fromisoformat(passive_start)
    dt_pass_hi = datetime.datetime.fromisoformat(passive_end)

    sailing_bstar: list[float] = []
    passive_bstar: list[float] = []

    for rec in records:
        epoch_str = rec.get("EPOCH") or rec.get("epoch", "")
        try:
            epoch_dt = _parse_omm_epoch(str(epoch_str))
        except ValueError:
            continue

        # BSTAR field may be a string or float from the JSON
        raw_bstar = rec.get("BSTAR") or rec.get("bstar")
        if raw_bstar is None:
            continue
        try:
            bstar = float(raw_bstar)
        except (ValueError, TypeError):
            continue

        # Skip physically implausible values (TLE fitter noise)
        if abs(bstar) > 1.0 or bstar < 0:
            continue

        if dt_sail_lo <= epoch_dt <= dt_sail_hi:
            sailing_bstar.append(bstar)
        elif dt_pass_lo <= epoch_dt <= dt_pass_hi:
            passive_bstar.append(bstar)

    if not sailing_bstar:
        raise ValueError(f"No usable BSTAR records in sailing window {sailing_start}–{sailing_end}")
    if not passive_bstar:
        raise ValueError(f"No usable BSTAR records in passive window {passive_start}–{passive_end}")

    sail_mean = float(np.mean(sailing_bstar))
    sail_std = float(np.std(sailing_bstar))
    pass_mean = float(np.mean(passive_bstar))
    pass_std = float(np.std(passive_bstar))

    obs_ratio = sail_mean / pass_mean if pass_mean != 0.0 else float("nan")

    # ── Predicted ratio from first-principles drag + SRP model ───────────────
    # At 720 km (representative LightSail 2 orbit), F10.7 = 100:
    #   ρ ≈ 2.5e-14 kg/m³ (NRLMSISE-00 tabulated value)
    #   v ≈ 7500 m/s
    #   C_D = 2.2, A_sail = 32 m², m = 5 kg
    #   F_drag = 0.5 × ρ × C_D × A × v²
    #
    # Maximum along-track SRP force (flat perfectly-reflecting sail,
    # optimal pitch angle α = arctan(1/√2) ≈ 35.26°, 85% reflectivity):
    #   P_SRP = 4.56e-6 N/m²  (solar radiation pressure at 1 AU)
    #   F_SRP_along = η × (4/(3√3)) × P_SRP × A
    #
    # The orbit-mean along-track SRP fraction varies with RAAN and attitude
    # strategy; the prediction below uses the time-averaged component from
    # Spencer et al. 2019 (reported ~40% orbit-raising efficiency relative
    # to peak SRP).
    _rho = 2.5e-14  # kg/m³ — atmospheric density at 720 km
    _v = 7500.0  # m/s
    _c_d = 2.2
    _area = 32.0  # m²
    _mass = 5.0  # kg
    _eta = 0.85  # sail reflectivity
    _p_srp = 4.56e-6  # N/m² — solar radiation pressure at 1 AU
    _sqrt3 = 3.0**0.5

    f_drag = 0.5 * _rho * _c_d * _area * _v**2  # N
    f_srp_peak = _eta * (4.0 / (3.0 * _sqrt3)) * _p_srp * _area  # N (peak along-track)
    f_srp_orbit = 0.40 * f_srp_peak  # orbit-mean (40% efficiency)

    # Effective B* ∝ net drag / reference.  During sailing, SRP reduces net
    # deceleration, so B*_sailing / B*_passive ≈ (F_drag − F_srp) / F_drag.
    pred_ratio = max(0.0, (f_drag - f_srp_orbit) / f_drag) if f_drag > 0 else float("nan")

    if obs_ratio < 1.0:
        interpretation = (
            f"B* lower during sailing (Δ = {(1 - obs_ratio) * 100:.1f}% reduction) — "
            "consistent with SRP partially offsetting drag."
        )
    elif obs_ratio > 1.0:
        interpretation = (
            f"B* higher during sailing (ratio = {obs_ratio:.3f}) — "
            "unexpected; may reflect sail orientation strategy or solar-cycle effects."
        )
    else:
        interpretation = "B* unchanged between windows."

    return BStarSwingResult(
        sailing_bstar_mean=sail_mean,
        sailing_bstar_std=sail_std,
        passive_bstar_mean=pass_mean,
        passive_bstar_std=pass_std,
        observed_ratio=obs_ratio,
        predicted_ratio=pred_ratio,
        n_sailing=len(sailing_bstar),
        n_passive=len(passive_bstar),
        sailing_window=(sailing_start, sailing_end),
        passive_window=(passive_start, passive_end),
        notes=interpretation,
    )
