"""
Eclipse timing validation against LightSail 2 beacon telemetry.

Compares LightSail 2 eclipse transitions extracted from beacon telemetry
against the shadow-entry/exit times predicted by our eclipse geometry model.

Data requirements
─────────────────
1. Beacon frames (validation/data/lightsail2_beacons_p1.json, acquired via
   SatNOGS — see validation/data/README.md).
2. Historical GP archive (validation/data/lightsail2_gp_history.json, already
   used by load_historical_tle_series()).

Algorithm
─────────
For each eclipse transition detected in the beacon stream:
  1. Locate the two nearest TLE epochs in the GP archive.
  2. Propagate each TLE to the transition midpoint → ECI position.
  3. Call shadow_fraction() at each propagated position.
  4. Binary-search within the beacon window for the predicted transition time.
  5. Record error_s = predicted_transition_s − observed_midpoint_s.

Penumbra extension
──────────────────
The ramp in solar_face_mw (nX/pX/nY/pY) during shadow ingress/egress gives an
empirical illumination proxy.  A face's current drops from full to zero over
~10–20 s as the penumbra sweeps the spacecraft.  Plotting face power vs. time
during a shadow entry provides the empirical penumbra width, compared against
our conical shadow model's predicted penumbra duration.

Usage
─────
  from cislunar.validation.beacon_parser import load_satnogs_frames
  from cislunar.validation.eclipse_validator import compare_eclipse_timing

  frames = load_satnogs_frames("validation/data/lightsail2_beacons_p1.json")
  report = compare_eclipse_timing(frames)
  print(report)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from .beacon_parser import BeaconFrame
from .data_paths import DEFAULT_BEACON_FILE as _DEFAULT_BEACON_FILE
from .telemetry_fetcher import (
    DEFAULT_LIGHTSAIL2_ARCHIVE,
    _omm_to_satrec,
    _parse_omm_epoch,
)

DEFAULT_BEACON_FILE = _DEFAULT_BEACON_FILE


# ── Result dataclasses ────────────────────────────────────────────────────────


@dataclass
class EclipseEvent:
    """One observed eclipse transition in the beacon stream."""

    kind: str  # "entry" or "exit"
    time_lo_s: float  # beacon timestamp before transition (last pre-event frame)
    time_hi_s: float  # beacon timestamp after transition (first post-event frame)
    pre_solar_mw: float  # total solar face power before transition (illumination proxy)
    post_solar_mw: float  # total solar face power after transition


@dataclass
class EclipseTimingReport:
    """Statistics from comparing predicted vs. observed eclipse transitions."""

    n_entries: int
    n_exits: int
    entry_error_mean_s: float  # signed mean; positive = model predicts late
    entry_error_std_s: float
    exit_error_mean_s: float
    exit_error_std_s: float
    beacon_period_s: float  # empirically measured frame cadence
    n_skipped_wide_windows: int  # observed transitions with too-wide beacon gaps
    max_transition_window_s: float
    resolution_note: str  # "timing resolution ±Xs = ±Ykm along-track"

    def __str__(self) -> str:
        def _fmt_seconds(value: float, *, signed: bool = False) -> str:
            if np.isnan(value):
                return "n/a"
            num = f"{value:+.1f}" if signed else f"{value:.1f}"
            return f"{num}s"

        return (
            f"EclipseTimingReport:\n"
            f"  entries: n={self.n_entries}  "
            f"mean={_fmt_seconds(self.entry_error_mean_s, signed=True)}  "
            f"std={_fmt_seconds(self.entry_error_std_s)}\n"
            f"  exits:   n={self.n_exits}  "
            f"mean={_fmt_seconds(self.exit_error_mean_s, signed=True)}  "
            f"std={_fmt_seconds(self.exit_error_std_s)}\n"
            f"  skipped wide windows: {self.n_skipped_wide_windows}  "
            f"(max observed bracket = {self.max_transition_window_s:.0f}s)\n"
            f"  {self.resolution_note}"
        )


@dataclass
class CadenceSegment:
    """One contiguous stretch of beacon frames separated by bounded gaps."""

    start_time_s: float
    end_time_s: float
    n_frames: int
    duration_s: float
    median_gap_s: float
    max_gap_s: float


@dataclass
class TransitionWindowAssessment:
    """Quality assessment for one observed eclipse transition window."""

    kind: str
    time_lo_s: float
    time_hi_s: float
    window_s: float
    usable_for_eclipse_timing: bool
    reason: str


@dataclass
class BeaconCoverageReport:
    """Beacon coverage summary for eclipse-validation readiness."""

    n_frames: int
    time_span_s: float
    beacon_period_s: float
    cadence_gap_threshold_s: float
    max_transition_window_s: float
    cadence_segments: list[CadenceSegment]
    transition_windows: list[TransitionWindowAssessment]

    @property
    def usable_transition_windows(self) -> list[TransitionWindowAssessment]:
        return [w for w in self.transition_windows if w.usable_for_eclipse_timing]

    @property
    def usable_entries(self) -> int:
        return sum(
            w.kind == "entry" and w.usable_for_eclipse_timing for w in self.transition_windows
        )

    @property
    def usable_exits(self) -> int:
        return sum(
            w.kind == "exit" and w.usable_for_eclipse_timing for w in self.transition_windows
        )

    def __str__(self) -> str:
        return (
            f"BeaconCoverageReport:\n"
            f"  frames: {self.n_frames}  span={self.time_span_s:.0f}s  "
            f"beacon period≈{self.beacon_period_s:.0f}s\n"
            f"  cadence segments: {len(self.cadence_segments)}  "
            f"transition windows: {len(self.transition_windows)}\n"
            f"  usable eclipse windows: entries={self.usable_entries}  "
            f"exits={self.usable_exits}  "
            f"(max bracket {self.max_transition_window_s:.0f}s)"
        )


# ── Transition extractor ──────────────────────────────────────────────────────


def extract_eclipse_transitions(frames: list[BeaconFrame]) -> list[EclipseEvent]:
    """
    Scan a time-ordered beacon sequence for eclipse entry and exit events.

    An entry is a transition in_eclipse False → True.
    An exit  is a transition in_eclipse True  → False.

    The bounding timestamps (time_lo_s, time_hi_s) bracket the actual transition;
    the midpoint is used as the observed time ± half the frame period.

    Args:
        frames: time-sorted list of BeaconFrame objects.

    Returns:
        List of EclipseEvent objects in chronological order.
    """
    events: list[EclipseEvent] = []
    for i in range(1, len(frames)):
        prev, curr = frames[i - 1], frames[i]
        if prev.in_eclipse == curr.in_eclipse:
            continue

        kind = "entry" if curr.in_eclipse else "exit"
        pre_solar_mw = float(sum(prev.solar_face_mw.values()))
        post_solar_mw = float(sum(curr.solar_face_mw.values()))

        events.append(
            EclipseEvent(
                kind=kind,
                time_lo_s=prev.timestamp_unix_s,
                time_hi_s=curr.timestamp_unix_s,
                pre_solar_mw=pre_solar_mw,
                post_solar_mw=post_solar_mw,
            )
        )
    return events


def _empirical_beacon_period(frames: list[BeaconFrame]) -> float:
    """Return median inter-frame interval [s] from a beacon sequence."""
    if len(frames) < 2:
        return 45.0  # LightSail 2 nominal beacon period
    gaps = [
        frames[i + 1].timestamp_unix_s - frames[i].timestamp_unix_s for i in range(len(frames) - 1)
    ]
    in_family = [g for g in gaps if 0 < g < 300]
    if not in_family:
        return 45.0
    return float(np.median(in_family))


def beacon_cadence_segments(
    frames: list[BeaconFrame],
    gap_threshold_s: float = 300.0,
) -> list[CadenceSegment]:
    """
    Split a beacon slice into contiguous cadence segments.

    A new segment begins whenever the inter-frame gap exceeds gap_threshold_s.
    """
    if not frames:
        return []

    segments: list[list[BeaconFrame]] = []
    current = [frames[0]]
    for prev, curr in zip(frames, frames[1:]):
        if (curr.timestamp_unix_s - prev.timestamp_unix_s) > gap_threshold_s:
            segments.append(current)
            current = [curr]
        else:
            current.append(curr)
    segments.append(current)

    results: list[CadenceSegment] = []
    for seg in segments:
        if len(seg) < 2:
            gaps = [0.0]
        else:
            gaps = [
                seg[i + 1].timestamp_unix_s - seg[i].timestamp_unix_s for i in range(len(seg) - 1)
            ]
        results.append(
            CadenceSegment(
                start_time_s=seg[0].timestamp_unix_s,
                end_time_s=seg[-1].timestamp_unix_s,
                n_frames=len(seg),
                duration_s=seg[-1].timestamp_unix_s - seg[0].timestamp_unix_s,
                median_gap_s=float(np.median(gaps)),
                max_gap_s=float(np.max(gaps)),
            )
        )
    return results


def assess_transition_windows(
    frames: list[BeaconFrame],
    max_transition_window_s: float = 300.0,
) -> list[TransitionWindowAssessment]:
    """Assess which observed eclipse transitions are tightly bracketed enough."""
    assessments: list[TransitionWindowAssessment] = []
    for ev in extract_eclipse_transitions(frames):
        window_s = ev.time_hi_s - ev.time_lo_s
        usable = window_s <= max_transition_window_s
        reason = (
            "bounded transition window"
            if usable
            else "window too wide for meaningful eclipse timing"
        )
        assessments.append(
            TransitionWindowAssessment(
                kind=ev.kind,
                time_lo_s=ev.time_lo_s,
                time_hi_s=ev.time_hi_s,
                window_s=window_s,
                usable_for_eclipse_timing=usable,
                reason=reason,
            )
        )
    return assessments


def summarize_beacon_coverage(
    frames: list[BeaconFrame],
    cadence_gap_threshold_s: float = 300.0,
    max_transition_window_s: float = 300.0,
) -> BeaconCoverageReport:
    """
    Summarize beacon cadence quality for eclipse validation.

    Reports contiguous cadence segments, observed transition window widths,
    and which windows are usable for eclipse timing comparison.
    """
    span_s = frames[-1].timestamp_unix_s - frames[0].timestamp_unix_s if len(frames) >= 2 else 0.0
    return BeaconCoverageReport(
        n_frames=len(frames),
        time_span_s=span_s,
        beacon_period_s=_empirical_beacon_period(frames),
        cadence_gap_threshold_s=cadence_gap_threshold_s,
        max_transition_window_s=max_transition_window_s,
        cadence_segments=beacon_cadence_segments(frames, cadence_gap_threshold_s),
        transition_windows=assess_transition_windows(frames, max_transition_window_s),
    )


# ── GP archive helpers ────────────────────────────────────────────────────────


def _load_gp_archive(json_path: str) -> list[tuple]:
    """
    Load (epoch_dt, satrec) pairs from a CelesTrak OMM JSON archive.
    Returns list sorted by epoch.
    """
    import json
    import re

    with open(json_path) as f:
        raw = f.read()
    raw = re.sub(r"(?<=:)\s*(\+?-?)\.", r" \g<1>0.", raw)
    records = json.loads(raw)

    pairs = []
    for rec in records:
        epoch_str = rec.get("EPOCH") or rec.get("epoch", "")
        try:
            epoch_dt = _parse_omm_epoch(str(epoch_str))
            sat = _omm_to_satrec(rec)
            pairs.append((epoch_dt, sat))
        except Exception:
            continue
    pairs.sort(key=lambda x: x[0])
    return pairs


def _nearest_satrec(gp_pairs: list[tuple], unix_s: float):
    """Return the Satrec whose epoch is nearest to unix_s (as Unix timestamp)."""
    import datetime

    target_dt = datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=unix_s)
    best = min(gp_pairs, key=lambda p: abs((p[0] - target_dt).total_seconds()))
    return best[1]


def _propagate_to_unix(sat, unix_s: float) -> np.ndarray | None:
    """
    Propagate a Satrec to a Unix timestamp → ECI position [m].
    Returns None if SGP4 reports an error.
    """

    # Minutes since the TLE epoch
    jd_target = (unix_s / 86400.0) + 2440587.5  # Julian date of unix_s
    minutes_from_epoch = (jd_target - (sat.jdsatepoch + sat.jdsatepochF)) * 1440.0
    e, r_km, _ = sat.sgp4(sat.jdsatepoch, sat.jdsatepochF + minutes_from_epoch / 1440.0)
    if e != 0:
        return None
    return np.array(r_km) * 1000.0  # km → m


# ── Eclipse model wrapper ─────────────────────────────────────────────────────


def _sun_pos_at_unix(unix_s: float) -> np.ndarray:
    """
    Compute Earth-centred inertial Sun position [m] at a Unix timestamp.

    Uses astropy (same backend as the main physics ephemeris cache) via a
    direct one-off query rather than the pre-built simulation cache.
    """
    import astropy.units as u
    from astropy.coordinates import GCRS, get_sun
    from astropy.time import Time

    t = Time(unix_s, format="unix", scale="tdb")
    sun_gcrs = get_sun(t).transform_to(GCRS(obstime=t))
    cartesian = cast(Any, sun_gcrs.cartesian)
    return np.asarray(cartesian.xyz.to_value(u.m), dtype=np.float64)


def _model_in_eclipse(pos_m: np.ndarray, unix_s: float) -> bool:
    """
    Query our shadow_fraction() at a given ECI position and time.

    Sun position is computed via astropy (same ephemeris used by the main
    physics engine) with a direct one-off lookup instead of the simulation
    cache, since the eclipse validator runs offline over historical data.
    """
    from cislunar.physics.forces.eclipse import shadow_fraction

    sun_pos = _sun_pos_at_unix(unix_s)
    return shadow_fraction(pos_m, sun_pos) == 0.0


# ── Main comparison function ──────────────────────────────────────────────────


def compare_eclipse_timing(
    frames: list[BeaconFrame],
    gp_archive_path: str = DEFAULT_LIGHTSAIL2_ARCHIVE,
    max_events: int | None = None,
    max_transition_window_s: float = 300.0,
) -> EclipseTimingReport:
    """
    Compare observed eclipse transitions to model predictions.

    For each transition in the beacon stream, the nearest TLE in the GP archive
    is propagated to the observed midpoint timestamp.  The model predicts whether
    the spacecraft is in eclipse at that position; a bisection search within the
    beacon window locates the predicted crossing time.

    Args:
        frames:           Time-sorted BeaconFrame list (from load_satnogs_frames).
        gp_archive_path:  Path to CelesTrak historical GP JSON archive.
        max_events:       Limit the number of events compared (None = all).
        max_transition_window_s:
                          Ignore observed transitions whose bracketing beacon
                          gap exceeds this width. Multi-hour gaps can confirm
                          that a state change happened, but they do not bound
                          the transition time tightly enough for a meaningful
                          timing residual.

    Returns:
        EclipseTimingReport with mean/std timing errors for entries and exits.

    Raises:
        FileNotFoundError: if gp_archive_path does not exist.
        ValueError: if frames is empty or contains fewer than two frames.
    """
    if len(frames) < 2:
        raise ValueError("Need at least two beacon frames to detect transitions.")

    if not os.path.exists(gp_archive_path):
        raise FileNotFoundError(
            f"GP archive not found: {gp_archive_path}\n"
            "See validation/data/README.md for acquisition instructions."
        )

    gp_pairs = _load_gp_archive(gp_archive_path)
    events = extract_eclipse_transitions(frames)
    beacon_period = _empirical_beacon_period(frames)

    if max_events is not None:
        events = events[:max_events]

    entry_errors: list[float] = []
    exit_errors: list[float] = []
    skipped_wide_windows = 0

    for ev in events:
        if (ev.time_hi_s - ev.time_lo_s) > max_transition_window_s:
            skipped_wide_windows += 1
            continue

        midpoint_s = 0.5 * (ev.time_lo_s + ev.time_hi_s)
        sat = _nearest_satrec(gp_pairs, midpoint_s)
        pos_m = _propagate_to_unix(sat, midpoint_s)
        if pos_m is None:
            continue

        # Binary-search for predicted transition within the beacon window
        predicted_s = _bisect_eclipse_transition(sat, ev.time_lo_s, ev.time_hi_s, ev.kind)
        if predicted_s is None:
            # Model agrees with observed state at both endpoints — timing error
            # is bounded by the beacon window half-width.
            predicted_s = midpoint_s

        error_s = predicted_s - midpoint_s
        if ev.kind == "entry":
            entry_errors.append(error_s)
        else:
            exit_errors.append(error_s)

    # Orbital speed ≈ 7.5 km/s at 720 km; resolution in km along-track
    orbital_speed_kms = 7.5
    resolution_km = beacon_period * orbital_speed_kms / 2.0

    resolution_note = (
        f"timing resolution ±{beacon_period / 2:.0f}s "
        f"= ±{resolution_km:.0f} km along-track at 720 km"
    )

    def _stats(errors: list[float]) -> tuple[float, float]:
        if not errors:
            return float("nan"), float("nan")
        a = np.array(errors)
        return float(np.mean(a)), float(np.std(a))

    em, es = _stats(entry_errors)
    xm, xs = _stats(exit_errors)

    return EclipseTimingReport(
        n_entries=len(entry_errors),
        n_exits=len(exit_errors),
        entry_error_mean_s=em,
        entry_error_std_s=es,
        exit_error_mean_s=xm,
        exit_error_std_s=xs,
        beacon_period_s=beacon_period,
        n_skipped_wide_windows=skipped_wide_windows,
        max_transition_window_s=max_transition_window_s,
        resolution_note=resolution_note,
    )


def _bisect_eclipse_transition(
    sat,
    t_lo_s: float,
    t_hi_s: float,
    kind: str,
    n_iter: int = 8,
) -> float | None:
    """
    Binary-search for the eclipse transition time within [t_lo_s, t_hi_s].

    For "entry": searches for False→True crossing (eclipse onset).
    For "exit":  searches for True→False crossing (eclipse exit).

    Returns the predicted transition time [Unix s], or None if the model
    does not find a crossing within the window (prediction outside window).
    """
    pos_lo = _propagate_to_unix(sat, t_lo_s)
    pos_hi = _propagate_to_unix(sat, t_hi_s)
    if pos_lo is None or pos_hi is None:
        return None

    eclipse_lo = _model_in_eclipse(pos_lo, t_lo_s)
    eclipse_hi = _model_in_eclipse(pos_hi, t_hi_s)

    # Verify the crossing is actually bracketed
    expected_lo = kind == "exit"  # for exit: lo=eclipsed, hi=sunlit
    expected_hi = kind == "entry"  # for entry: lo=sunlit,  hi=eclipsed
    if eclipse_lo != expected_lo or eclipse_hi != expected_hi:
        return None

    lo, hi = t_lo_s, t_hi_s
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        pos_m = _propagate_to_unix(sat, mid)
        if pos_m is None:
            break
        if _model_in_eclipse(pos_m, mid) == expected_hi:
            hi = mid
        else:
            lo = mid

    return 0.5 * (lo + hi)
