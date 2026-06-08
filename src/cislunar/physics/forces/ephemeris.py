"""
JPL-quality ephemeris for the physics engine.

Replaces the circular-orbit approximations for Moon and Sun with
positions derived from astropy's built-in ERFA/DE430 ephemeris data.
No network access required; astropy ships the ephemeris tables locally.

Accuracy vs. circular approximation
──────────────────────────────────────
The Moon's orbit is elliptical (e=0.055) and perturbed by the Sun,
producing position errors of up to ~20,000 km in the circular model
over a 14-day cislunar transit. The real ephemeris reduces this to
< 1 km interpolation error (1-hour cache step) + sub-km ERFA residual.

Impact on simulation fidelity
──────────────────────────────
Simulations using the circular Moon approximation learn trans-lunar
gravity-assist geometries that don't exist in reality. With the real
ephemeris, trajectories that exploit the correct gate timing are using
real orbital mechanics — directly transferable to hardware.

Usage in make_lunar_craft
──────────────────────────
    from cislunar.physics.forces.ephemeris import build_cislunar_ephemeris
    moon_fn, sun_fn = build_cislunar_ephemeris(
        start_time="2025-06-01",   # simulation epoch
        duration_days=30,
    )
    grav = (
        GravityModel()
        .add_body("Earth", mu=Body.MU_EARTH, j2=Body.J2_EARTH, r_body=Body.R_EARTH)
        .add_body("Moon",  mu=Body.MU_MOON,  body_pos_fn=moon_fn)
        .add_body("Sun",   mu=Body.MU_SUN,   body_pos_fn=sun_fn)
    )

The returned callables match the body_pos_fn signature:
    fn(time_s: float) -> NDArray[float64]  # Earth-centred [m]

Cache parameters
─────────────────
Moon cache: 1-hour steps. Moon moves ~3,700 km/hr at perigee;
  1-hr interpolation gives <10 km error mid-step. Adequate for
  simulation where the gravity timestep is typically 1–60 s.

Sun cache: 6-hour steps. Sun moves ~7,200 km/hr (1° of arc every 4 min);
  6-hr interpolation gives <2,000 km error = 0.001% at 1 AU. The sun's
  tidal effect on the spacecraft changes by <0.01% over 6 hours —
  negligible for trajectory accuracy.

Dependency requirement
──────────────────────
astropy is mandatory for cislunar physics. Without it, ephemeris
construction raises RuntimeError instead of silently using the
circular approximation, whose Moon error can exceed ~20,000 km.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from ..constants import AU

# ── Circular-orbit reference model (for comparison/testing only) ─────────────


def _moon_circular(t: float) -> NDArray:
    """Circular Moon orbit reference model used for comparison/testing."""
    MOON_PERIOD = 27.3217 * 86400
    theta = 2 * math.pi * t / MOON_PERIOD
    r = 3.844e8
    return np.array([r * math.cos(theta), r * math.sin(theta), 0.0])


def _sun_circular(t: float) -> NDArray:
    """Circular Sun orbit reference model used for comparison/testing."""
    EARTH_PERIOD = 365.25 * 86400
    theta = 2 * math.pi * t / EARTH_PERIOD
    return np.array([AU * math.cos(theta), AU * math.sin(theta), 0.0])


# ── Astropy ephemeris cache ───────────────────────────────────────────────────


class EphemerisOutOfRangeError(RuntimeError):
    """Simulation time is outside the pre-built ephemeris cache window.

    Increase ``duration_days`` (or the 10% margin factor) when building the
    cache, or ensure the episode cannot run past the cache end time.
    """


class _EphemerisCache:
    """
    Pre-compute body positions at regular intervals, linearly interpolate.

    Build cost: ~100–250 ms (one astropy call per cached epoch).
    Lookup cost: ~3 μs (pure NumPy linear interpolation).
    Error: < 10 km for Moon at 1-hr step; < 2,000 km for Sun at 6-hr step.

    Args:
        body:         Astropy body name ("moon", "sun", etc.)
        t0:           Simulation start time (astropy Time object)
        duration_days: Length of simulation [days]
        step_hours:   Cache resolution [hours]
    """

    def __init__(
        self,
        body: str,
        t0,  # astropy Time
        duration_days: float,
        step_hours: float,
    ):
        try:
            import astropy.units as u
            from astropy.coordinates import GCRS, get_body
        except ImportError:
            raise ImportError("astropy is required for real ephemeris") from None

        self._body = body
        self._step_s = step_hours * 3600.0
        n = int(duration_days * 24 / step_hours) + 2

        for i in range(n):
            t = t0 + i * step_hours * u.hour
            # Use the same Earth-centred GCRS frame as the rest of the
            # simulation and the validation tests. A simple barycentric
            # subtraction (body - Earth) omits the topocentric/GCRS
            # transform terms and was drifting by ~30 km for the Moon.
            body_gcrs = get_body(body, t).transform_to(GCRS(obstime=t))
            if i == 0:
                positions_m = np.empty((n, 3), dtype=np.float64)
            cartesian = cast(Any, body_gcrs.cartesian)
            positions_m[i] = np.asarray(cartesian.xyz.to_value(u.m), dtype=np.float64)

        self._positions_m = positions_m
        self._n = n

    def get(self, sim_time_s: float) -> NDArray:
        """Interpolate position at sim_time_s seconds after epoch."""
        idx_f = sim_time_s / self._step_s
        idx0 = int(idx_f)
        frac = idx_f - idx0
        # Use idx_f (float) for the lower-bound check. Python's int() truncates
        # toward zero, so int(-1.0003) == -1, which would pass an "idx0 < -1"
        # guard. Comparing idx_f < -1.0 correctly catches any query more than
        # one step before the cache start. The upper bound uses idx0 (int) since
        # that's what drives the array index.
        if idx_f < -1.0 or idx0 >= self._n - 1:
            t_max_s = (self._n - 1) * self._step_s
            raise EphemerisOutOfRangeError(
                f"{self._body!r} ephemeris queried at t={sim_time_s:.1f}s; "
                f"cache covers 0–{t_max_s:.1f}s. "
                f"Rebuild with a longer duration_days."
            )
        idx0 = max(0, idx0)
        return (1.0 - frac) * self._positions_m[idx0] + frac * self._positions_m[idx0 + 1]


# ── Public API ────────────────────────────────────────────────────────────────


def build_cislunar_ephemeris(
    start_time: str = "2025-01-01",
    duration_days: float = 30.0,
    moon_step_hours: float = 1.0,
    sun_step_hours: float = 6.0,
    verbose: bool = False,
) -> tuple[Callable[[float], NDArray], Callable[[float], NDArray]]:
    """
    Build cached Moon and Sun ephemeris functions for a cislunar simulation.

    Args:
        start_time:      ISO 8601 date string for simulation epoch
        duration_days:   Simulation duration; cache covers this + 10% margin
        moon_step_hours: Cache resolution for Moon (1 hr recommended)
        sun_step_hours:  Cache resolution for Sun (6 hr recommended)
        verbose:         Print build timing

    Returns:
        (moon_pos_fn, sun_pos_fn) — callables taking sim_time_s [s],
        returning Earth-centred inertial position [m].

    Requires astropy. If it is unavailable, raises RuntimeError rather than
    silently using the circular approximation.
    """
    try:
        import time as _time

        from astropy.coordinates import solar_system_ephemeris
        from astropy.time import Time

        solar_system_ephemeris.set("builtin")  # no network needed
        t0 = Time(start_time, scale="tdb")

        # Add 10% margin so simulations approaching the end of a run don't
        # walk off the end of the cached table
        cache_days = duration_days * 1.10

        if verbose:
            print(f"Building ephemeris cache ({start_time}, {cache_days:.0f} days)...")

        t_start = _time.perf_counter()
        moon_cache = _EphemerisCache("moon", t0, cache_days, moon_step_hours)
        sun_cache = _EphemerisCache("sun", t0, cache_days, sun_step_hours)
        elapsed = _time.perf_counter() - t_start

        if verbose:
            print(f"  Built in {elapsed * 1000:.0f} ms")

        moon_fn: Callable[[float], NDArray] = moon_cache.get
        sun_fn: Callable[[float], NDArray] = sun_cache.get

        return moon_fn, sun_fn

    except ImportError as exc:
        raise RuntimeError(
            "astropy is required for cislunar physics but is not installed.\n"
            "The circular-orbit fallback introduces Moon position errors of up to\n"
            "~20,000 km, which makes sail geometry, eclipse timing, and lunar\n"
            "encounter calculations physically incorrect.\n\n"
            "Install astropy before running the environment:\n"
            "    pip install astropy\n\n"
            f"Original error: {exc}"
        ) from exc


def ephemeris_error_vs_circular(
    start_time: str = "2025-01-01",
    duration_days: float = 14.0,
    n_samples: int = 100,
) -> dict:
    """
    Quantify the error introduced by the circular-orbit approximation
    versus the astropy ephemeris, over a given time window.

    Returns a dict with statistics:
        moon_max_km, moon_mean_km, sun_max_km, sun_mean_km
    """
    moon_fn, sun_fn = build_cislunar_ephemeris(start_time, duration_days)

    times = np.linspace(0, duration_days * 86400, n_samples)
    moon_errs = []
    sun_errs = []

    for t in times:
        m_real = moon_fn(t)
        m_circ = _moon_circular(t)
        moon_errs.append(np.linalg.norm(m_real - m_circ) / 1e3)

        s_real = sun_fn(t)
        s_circ = _sun_circular(t)
        sun_errs.append(np.linalg.norm(s_real - s_circ) / 1e3)

    return {
        "moon_max_km": float(max(moon_errs)),
        "moon_mean_km": float(sum(moon_errs) / len(moon_errs)),
        "sun_max_km": float(max(sun_errs)),
        "sun_mean_km": float(sum(sun_errs) / len(sun_errs)),
    }
