"""
Attitude-envelope validation against LightSail 2 beacon telemetry.

Three falsifiable statistics extracted from LightSail 2 beacon telemetry,
compared against our AttitudeDynamicsModel predictions.

Data requirement: beacon frames (typically validation/data/lightsail2_beacons_p1.json).
All functions work on any list[BeaconFrame] and skip gracefully when called
with an empty list, enabling synthetic unit tests without real data.

Statistics
──────────
1. rate_distribution()
   Filter: sail_deployed=True, adcs_mode ≥ 2 (stabilize or sail mode).
   Compute |ω| per frame.  Expected: bimodal — near-zero plateau during
   cruise; ~0.3°/s peak during slews.  Our model's hard cap on slew rate
   is 0.005 rad/s ≈ 0.29°/s; the histogram upper tail must be cut off there.

2. slew_statistics()
   Detect slew events: contiguous runs where |ω| > threshold (default 0.05°/s).
   Our model predicts a 90° half-plane slew takes 90°/0.3°/s ≈ 300 s.
   Compare the observed distribution mean to 300 ± mission tolerance.

3. tumble_decay_timescale()
   During anomaly periods (|ω| > 1°/s sustained ≥ 2 frames), fit an
   exponential decay |ω|(t) = A·exp(−t/τ) to the recovery segment.
   Our PD controller has ωn = sqrt(kp/Izz) = sqrt(1e-5/0.10) ≈ 0.01 rad/s,
   implying τ ≈ 1/ωn = 100 s.  The fitted τ from real data bounds this.

Model reference constants (from cislunar/physics/forces/attitude.py defaults)
───────────────────────────────────────────────────────────────────
  max slew rate   : 0.005 rad/s ≈ 0.286°/s  (spacecraft.py max_attitude_rate)
  PD kp           : 1e-5 N·m/rad
  PD kd           : 6e-4 N·m·s/rad
  Izz             : ~0.10 kg·m²  (12U CubeSat dominant axis)
  ωn              : sqrt(kp/Izz) = 0.01 rad/s  → τ = 100 s
  90° slew target : ~10 min (attitude.py comments, LightSail 2 performance target)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .beacon_parser import BeaconFrame

# ── Model reference constants ─────────────────────────────────────────────────

MODEL_MAX_SLEW_DEG_S = 0.005 * 180.0 / math.pi  # 0.005 rad/s → °/s
MODEL_HALFPLANE_SLEW_S = 90.0 / MODEL_MAX_SLEW_DEG_S  # ≈ 314 s
MODEL_DAMPING_TAU_S = 1.0 / math.sqrt(1e-5 / 0.10)  # 1/ωn ≈ 100 s
MAX_VALID_RATE_DEG_S = 5.0
MAX_CONTIGUOUS_GAP_S = 300.0


# ── Result dataclasses ────────────────────────────────────────────────────────


@dataclass
class RateDistributionResult:
    """Body-rate magnitude distribution during active sailing operations."""

    n_frames: int
    mean_deg_s: float
    p50_deg_s: float
    p95_deg_s: float
    p99_deg_s: float
    max_deg_s: float
    fraction_above_model_max: float  # fraction of frames where |ω| > model cap
    model_max_deg_s: float = field(default=MODEL_MAX_SLEW_DEG_S)

    def __str__(self) -> str:
        return (
            f"RateDistributionResult  (n={self.n_frames} sail-mode frames)\n"
            f"  mean={self.mean_deg_s:.4f}  p50={self.p50_deg_s:.4f}  "
            f"p95={self.p95_deg_s:.4f}  p99={self.p99_deg_s:.4f}  "
            f"max={self.max_deg_s:.4f} °/s\n"
            f"  model cap: {self.model_max_deg_s:.4f} °/s  "
            f"fraction above cap: {self.fraction_above_model_max:.4f}"
        )


@dataclass
class SlewStats:
    """Slew event statistics: contiguous runs where |ω| > threshold."""

    n_slews: int
    durations_s: list[float]
    mean_duration_s: float
    p50_duration_s: float
    p95_duration_s: float
    max_duration_s: float
    threshold_deg_s: float
    model_halfplane_s: float = field(default=MODEL_HALFPLANE_SLEW_S)

    def __str__(self) -> str:
        return (
            f"SlewStats  (n={self.n_slews} slews, threshold={self.threshold_deg_s} °/s)\n"
            f"  mean={self.mean_duration_s:.1f}s  p50={self.p50_duration_s:.1f}s  "
            f"p95={self.p95_duration_s:.1f}s  max={self.max_duration_s:.1f}s\n"
            f"  model half-plane slew: {self.model_halfplane_s:.0f} s"
        )


@dataclass
class TumbleDecayResult:
    """Exponential decay fit from anomaly recovery segments."""

    n_anomaly_periods: int
    fitted_tau_s: float  # mean τ across all recovery fits
    fitted_tau_std_s: float  # std dev of τ across fits
    fitted_A_deg_s: float  # mean initial amplitude at recovery onset
    peak_rate_deg_s: float  # max |ω| seen across all anomaly periods
    anomaly_threshold_deg_s: float
    model_tau_s: float = field(default=MODEL_DAMPING_TAU_S)

    def __str__(self) -> str:
        return (
            f"TumbleDecayResult  (n={self.n_anomaly_periods} anomaly periods, "
            f"threshold={self.anomaly_threshold_deg_s} °/s)\n"
            f"  fitted τ = {self.fitted_tau_s:.1f} ± {self.fitted_tau_std_s:.1f} s\n"
            f"  model τ  = {self.model_tau_s:.1f} s  "
            f"peak |ω| = {self.peak_rate_deg_s:.3f} °/s"
        )


# ── Helper ─────────────────────────────────────────────────────────────────────


def _rate_magnitudes(frames: list[BeaconFrame]) -> NDArray:
    """Return |ω| [°/s] for each frame in the list."""
    if not frames:
        return np.empty(0)
    return np.array([float(np.linalg.norm(f.body_rate_deg_s)) for f in frames])


def _filter_sailing(frames: list[BeaconFrame]) -> list[BeaconFrame]:
    """Return only frames with sail_deployed=True and adcs_mode ≥ 2."""
    return [f for f in frames if f.sail_deployed and f.adcs_mode >= 2]


def _filter_physical_rates(
    frames: list[BeaconFrame],
    max_rate_deg_s: float = MAX_VALID_RATE_DEG_S,
) -> list[BeaconFrame]:
    """Drop frames with implausibly large body-rate magnitudes."""
    return [f for f in frames if float(np.linalg.norm(f.body_rate_deg_s)) <= max_rate_deg_s]


# ── Statistic 1: rate magnitude distribution ──────────────────────────────────


def rate_distribution(
    frames: list[BeaconFrame],
    sailing_only: bool = True,
) -> RateDistributionResult:
    """
    Compute body-rate magnitude statistics during (optionally) sailing operations.

    Args:
        frames:       Time-sorted BeaconFrame list.
        sailing_only: If True, restrict to sail_deployed=True and adcs_mode ≥ 2.

    Returns:
        RateDistributionResult with percentiles and fraction above model cap.
        Returns zero-populated result if no qualifying frames exist.
    """
    subset = _filter_sailing(frames) if sailing_only else frames

    if not subset:
        return RateDistributionResult(
            n_frames=0,
            mean_deg_s=0.0,
            p50_deg_s=0.0,
            p95_deg_s=0.0,
            p99_deg_s=0.0,
            max_deg_s=0.0,
            fraction_above_model_max=0.0,
        )

    mags = _rate_magnitudes(subset)

    return RateDistributionResult(
        n_frames=len(subset),
        mean_deg_s=float(np.mean(mags)),
        p50_deg_s=float(np.percentile(mags, 50)),
        p95_deg_s=float(np.percentile(mags, 95)),
        p99_deg_s=float(np.percentile(mags, 99)),
        max_deg_s=float(np.max(mags)),
        fraction_above_model_max=float(np.mean(mags > MODEL_MAX_SLEW_DEG_S)),
    )


# ── Statistic 2: slew duration distribution ───────────────────────────────────


def slew_statistics(
    frames: list[BeaconFrame],
    threshold_deg_s: float = 0.05,
    sailing_only: bool = True,
    max_gap_s: float = MAX_CONTIGUOUS_GAP_S,
    max_rate_deg_s: float = MAX_VALID_RATE_DEG_S,
) -> SlewStats:
    """
    Detect slew events and compute their duration distribution.

    A slew event is a maximal contiguous run of frames where |ω| > threshold.
    Duration is measured from the timestamp of the first slewing frame to the
    last, so a single frame above threshold has duration = 0.

    Args:
        frames:          Time-sorted BeaconFrame list.
        threshold_deg_s: |ω| threshold to enter/exit a slew [°/s].
        sailing_only:    Restrict to sail_deployed=True, adcs_mode ≥ 2.

    Returns:
        SlewStats with duration distribution and model reference.
    """
    subset = _filter_sailing(frames) if sailing_only else frames
    subset = _filter_physical_rates(subset, max_rate_deg_s=max_rate_deg_s)

    if not subset:
        return SlewStats(
            n_slews=0,
            durations_s=[],
            mean_duration_s=0.0,
            p50_duration_s=0.0,
            p95_duration_s=0.0,
            max_duration_s=0.0,
            threshold_deg_s=threshold_deg_s,
        )

    mags = _rate_magnitudes(subset)
    above = mags > threshold_deg_s

    durations: list[float] = []
    in_slew = False
    slew_start_t = 0.0

    for i, (frame, is_slewing) in enumerate(zip(subset, above)):
        if i > 0:
            gap_s = frame.timestamp_unix_s - subset[i - 1].timestamp_unix_s
            if gap_s > max_gap_s:
                if in_slew:
                    durations.append(subset[i - 1].timestamp_unix_s - slew_start_t)
                    in_slew = False
                if not is_slewing:
                    continue
        if is_slewing and not in_slew:
            in_slew = True
            slew_start_t = frame.timestamp_unix_s
        elif not is_slewing and in_slew:
            in_slew = False
            durations.append(subset[i - 1].timestamp_unix_s - slew_start_t)

    # Close an open slew at the end of the sequence
    if in_slew:
        durations.append(subset[-1].timestamp_unix_s - slew_start_t)

    if not durations:
        return SlewStats(
            n_slews=0,
            durations_s=[],
            mean_duration_s=0.0,
            p50_duration_s=0.0,
            p95_duration_s=0.0,
            max_duration_s=0.0,
            threshold_deg_s=threshold_deg_s,
        )

    d = np.array(durations)
    return SlewStats(
        n_slews=len(durations),
        durations_s=durations,
        mean_duration_s=float(np.mean(d)),
        p50_duration_s=float(np.percentile(d, 50)),
        p95_duration_s=float(np.percentile(d, 95)),
        max_duration_s=float(np.max(d)),
        threshold_deg_s=threshold_deg_s,
    )


# ── Statistic 3: tumble decay timescale ───────────────────────────────────────


def tumble_decay_timescale(
    frames: list[BeaconFrame],
    anomaly_threshold_deg_s: float = 1.0,
    min_anomaly_frames: int = 2,
    max_gap_s: float = MAX_CONTIGUOUS_GAP_S,
    max_rate_deg_s: float = MAX_VALID_RATE_DEG_S,
) -> TumbleDecayResult:
    """
    Fit exponential decay to body-rate recovery after anomaly periods.

    An anomaly period is a contiguous run of ≥ min_anomaly_frames where
    |ω| > anomaly_threshold_deg_s.  The recovery segment is the descending
    tail from the anomaly peak to the next frame below threshold.

    The exponential model |ω|(t) = A · exp(−t/τ) is fitted by linear
    regression on log(|ω|) vs. elapsed time (avoids scipy dependency).

    Args:
        frames:                  Time-sorted BeaconFrame list.
        anomaly_threshold_deg_s: |ω| threshold to identify a tumble [°/s].
        min_anomaly_frames:      Minimum consecutive frames above threshold
                                 to count as an anomaly (filters noise spikes).

    Returns:
        TumbleDecayResult.  If no anomaly periods are found, returns a result
        with n_anomaly_periods=0 and NaN for fitted values.
    """
    frames = _filter_physical_rates(frames, max_rate_deg_s=max_rate_deg_s)

    if not frames:
        return TumbleDecayResult(
            n_anomaly_periods=0,
            fitted_tau_s=float("nan"),
            fitted_tau_std_s=float("nan"),
            fitted_A_deg_s=float("nan"),
            peak_rate_deg_s=0.0,
            anomaly_threshold_deg_s=anomaly_threshold_deg_s,
        )

    mags = _rate_magnitudes(frames)
    above = mags > anomaly_threshold_deg_s

    # Locate contiguous anomaly runs
    anomaly_runs: list[tuple[int, int]] = []  # (start_idx, end_idx inclusive)
    i = 0
    while i < len(frames):
        if above[i]:
            j = i
            while j < len(frames) and above[j]:
                if (
                    j > i
                    and (frames[j].timestamp_unix_s - frames[j - 1].timestamp_unix_s) > max_gap_s
                ):
                    break
                j += 1
            if (j - i) >= min_anomaly_frames:
                anomaly_runs.append((i, j - 1))
            i = j
        else:
            i += 1

    if not anomaly_runs:
        return TumbleDecayResult(
            n_anomaly_periods=0,
            fitted_tau_s=float("nan"),
            fitted_tau_std_s=float("nan"),
            fitted_A_deg_s=float("nan"),
            peak_rate_deg_s=float(np.max(mags)) if len(mags) else 0.0,
            anomaly_threshold_deg_s=anomaly_threshold_deg_s,
        )

    tau_fits: list[float] = []
    A_fits: list[float] = []
    peak_global = 0.0

    for start, end in anomaly_runs:
        # Peak within the anomaly
        peak_idx = int(np.argmax(mags[start : end + 1])) + start
        peak_global = max(peak_global, mags[peak_idx])

        # Recovery segment: from peak to end of anomaly run (inclusive)
        rec_idx = list(range(peak_idx, end + 1))

        # Extend past the anomaly end while |ω| > 0.01 deg/s (still decaying)
        k = end + 1
        while (
            k < len(frames)
            and mags[k] > 0.01
            and (frames[k].timestamp_unix_s - frames[k - 1].timestamp_unix_s) <= max_gap_s
        ):
            rec_idx.append(k)
            k += 1

        if len(rec_idx) < 3:
            continue  # not enough points for a reliable fit

        t_rec = np.array([frames[i].timestamp_unix_s for i in rec_idx])
        om_rec = mags[np.array(rec_idx)]

        # Guard: all omega values must be positive for log transform
        if np.any(om_rec <= 0):
            continue

        t_rel = t_rec - t_rec[0]
        try:
            coeffs = np.polyfit(t_rel, np.log(om_rec), 1)
        except (np.linalg.LinAlgError, ValueError):
            continue

        slope, intercept = coeffs
        if slope >= 0:
            continue  # not decaying — skip (could be a spin-up)

        tau_fits.append(-1.0 / slope)
        A_fits.append(math.exp(intercept))

    if not tau_fits:
        return TumbleDecayResult(
            n_anomaly_periods=len(anomaly_runs),
            fitted_tau_s=float("nan"),
            fitted_tau_std_s=float("nan"),
            fitted_A_deg_s=float("nan"),
            peak_rate_deg_s=peak_global,
            anomaly_threshold_deg_s=anomaly_threshold_deg_s,
        )

    return TumbleDecayResult(
        n_anomaly_periods=len(anomaly_runs),
        fitted_tau_s=float(np.mean(tau_fits)),
        fitted_tau_std_s=float(np.std(tau_fits)),
        fitted_A_deg_s=float(np.mean(A_fits)),
        peak_rate_deg_s=peak_global,
        anomaly_threshold_deg_s=anomaly_threshold_deg_s,
    )
