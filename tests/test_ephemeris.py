"""Ephemeris cache and cislunar accuracy tests."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import ONE_DAY

from cislunar.physics import EphemerisOutOfRangeError
from cislunar.physics.forces.ephemeris import _EphemerisCache


def _make_ephemeris_cache(
    n: int = 5, step_s: float = 3600.0, body: str = "test_body"
) -> _EphemerisCache:
    """Construct a ready-to-query ``_EphemerisCache`` without an astropy call."""
    cache = _EphemerisCache.__new__(_EphemerisCache)
    cache._body = body
    cache._step_s = step_s
    cache._n = n
    cache._positions_m = np.tile(np.linspace(0, 1, n)[:, None], (1, 3))
    return cache


# ══════════════════════════════════════════════════════════════════════════════
# Section 15 — Ephemeris cache out-of-range boundary
# ══════════════════════════════════════════════════════════════════════════════
class TestEphemerisOutOfRange:
    """Boundary handling in the real _EphemerisCache.get() method."""

    @pytest.fixture
    def cache(self):
        """A 4-hour cache with 1-hour steps — valid query range [0, 14400) s."""
        return _make_ephemeris_cache(n=5, step_s=3600.0)

    def test_in_range_queries_do_not_raise(self, cache):
        cache.get(0.0)
        cache.get(3600.0)
        cache.get(7199.9)

    def test_tiny_negative_is_tolerated(self, cache):
        """Float-arithmetic noise (t ≈ -1 ms) must not blow up."""
        cache.get(-0.001)

    def test_negative_beyond_one_step_raises(self, cache):
        with pytest.raises(EphemerisOutOfRangeError):
            cache.get(-3600.1)

    def test_query_at_cache_end_raises(self, cache):
        with pytest.raises(EphemerisOutOfRangeError):
            cache.get(4 * 3600.0)

    def test_query_beyond_cache_end_raises(self, cache):
        with pytest.raises(EphemerisOutOfRangeError):
            cache.get(9 * 3600.0)

    def test_error_message_includes_body_and_times(self, cache):
        with pytest.raises(EphemerisOutOfRangeError) as exc:
            cache.get(99999.0)
        msg = str(exc.value)
        assert "test_body" in msg and "99999" in msg and "14400" in msg, f"msg={msg!r}"

    def test_exception_reexported_from_physics_package(self):
        from cislunar.physics import EphemerisOutOfRangeError as pkg_exc
        from cislunar.physics.forces.ephemeris import EphemerisOutOfRangeError as src_exc

        assert pkg_exc is src_exc


# ══════════════════════════════════════════════════════════════════════════════
# Section 15b — astropy is a mandatory dependency for cislunar physics
# ══════════════════════════════════════════════════════════════════════════════
class TestEphemerisAstropyRequirement:
    """``build_cislunar_ephemeris`` must raise ``RuntimeError`` when astropy
    is missing — no silent fallback to a circular approximation."""

    def test_raises_runtime_error_mentioning_astropy_when_import_blocked(
        self,
        monkeypatch,
    ):
        import builtins
        import importlib

        from cislunar.physics.forces import ephemeris as eph_mod

        real_import = builtins.__import__

        def blocking_import(name, *args, **kwargs):
            if name == "astropy" or name.startswith("astropy."):
                raise ImportError(f"blocked: {name}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocking_import)
        try:
            eph_fresh = importlib.reload(eph_mod)
            with pytest.raises(RuntimeError) as exc_info:
                eph_fresh.build_cislunar_ephemeris("2025-01-01", duration_days=1)
            assert "astropy" in str(exc_info.value).lower(), (
                f"message does not mention astropy: {exc_info.value!r}"
            )
        finally:
            monkeypatch.undo()
            importlib.reload(eph_mod)


# ══════════════════════════════════════════════════════════════════════════════
# Section 15c — Cislunar ephemeris accuracy vs astropy
# ══════════════════════════════════════════════════════════════════════════════
class TestCislunarEphemerisAccuracy:
    """Real ephemeris callables should stay close to direct astropy queries."""

    def test_moon_matches_astropy_get_body_within_10km_at_three_times(self):
        import astropy.units as u
        from astropy.coordinates import GCRS, get_body, solar_system_ephemeris
        from astropy.time import Time

        from cislunar.physics.forces.ephemeris import build_cislunar_ephemeris

        start_time = "2025-01-01"
        sample_times_s = [0.0, 3.5 * ONE_DAY + 1234.0, 14.0 * ONE_DAY - 777.0]

        solar_system_ephemeris.set("builtin")
        t0 = Time(start_time, scale="tdb")
        moon_fn, _ = build_cislunar_ephemeris(
            start_time=start_time,
            duration_days=15.0,
        )

        for sim_time_s in sample_times_s:
            t = t0 + sim_time_s * u.s
            moon_astropy = get_body("moon", t).transform_to(GCRS(obstime=t))
            moon_ref_m = moon_astropy.cartesian.xyz.to(u.m).value
            moon_ephem_m = moon_fn(sim_time_s)
            err_m = float(np.linalg.norm(moon_ephem_m - moon_ref_m))
            assert err_m <= 10e3, f"t={sim_time_s:.1f}s err_km={err_m / 1e3:.3f}"

    def test_sun_matches_astropy_get_body_within_2000km_at_three_times(self):
        import astropy.units as u
        from astropy.coordinates import GCRS, get_body, solar_system_ephemeris
        from astropy.time import Time

        from cislunar.physics.forces.ephemeris import build_cislunar_ephemeris

        start_time = "2025-01-01"
        sample_times_s = [0.0, 3.5 * ONE_DAY + 1234.0, 14.0 * ONE_DAY - 777.0]

        solar_system_ephemeris.set("builtin")
        t0 = Time(start_time, scale="tdb")
        _, sun_fn = build_cislunar_ephemeris(
            start_time=start_time,
            duration_days=15.0,
        )

        for sim_time_s in sample_times_s:
            t = t0 + sim_time_s * u.s
            sun_astropy = get_body("sun", t).transform_to(GCRS(obstime=t))
            sun_ref_m = sun_astropy.cartesian.xyz.to(u.m).value
            sun_ephem_m = sun_fn(sim_time_s)
            err_m = float(np.linalg.norm(sun_ephem_m - sun_ref_m))
            assert err_m <= 2_000e3, f"t={sim_time_s:.1f}s err_km={err_m / 1e3:.3f}"


# ══════════════════════════════════════════════════════════════════════════════
# Section 15d — Circular reference models
# ══════════════════════════════════════════════════════════════════════════════
class TestCircularModels:
    """_moon_circular and _sun_circular must return plausible reference positions."""

    def test_moon_circular_returns_3vector(self):
        from cislunar.physics.forces.ephemeris import _moon_circular

        pos = _moon_circular(0.0)
        assert pos.shape == (3,)

    def test_moon_circular_at_t0_on_x_axis(self):
        from cislunar.physics.forces.ephemeris import _moon_circular

        pos = _moon_circular(0.0)
        assert abs(pos[0] - 3.844e8) < 1e4
        assert abs(pos[1]) < 1e4
        assert abs(pos[2]) < 1e4

    def test_moon_circular_radius_constant(self):
        from cislunar.physics.forces.ephemeris import _moon_circular

        for t in [0.0, 86400.0, 7 * 86400.0]:
            r = float(np.linalg.norm(_moon_circular(t)))
            assert abs(r - 3.844e8) < 1e4, f"radius drift at t={t}: {r:.0f} m"

    def test_moon_circular_completes_one_orbit(self):
        from cislunar.physics.forces.ephemeris import _moon_circular

        period_s = 27.3217 * 86400.0
        p0 = _moon_circular(0.0)
        p1 = _moon_circular(period_s)
        assert float(np.linalg.norm(p1 - p0)) < 1e5

    def test_sun_circular_returns_3vector(self):
        from cislunar.physics.forces.ephemeris import _sun_circular

        pos = _sun_circular(0.0)
        assert pos.shape == (3,)

    def test_sun_circular_at_t0_near_1au(self):
        from cislunar.physics.constants import AU
        from cislunar.physics.forces.ephemeris import _sun_circular

        pos = _sun_circular(0.0)
        assert abs(float(np.linalg.norm(pos)) - AU) < 1e6

    def test_sun_circular_radius_constant(self):
        from cislunar.physics.constants import AU
        from cislunar.physics.forces.ephemeris import _sun_circular

        for t in [0.0, 30 * 86400.0, 180 * 86400.0]:
            r = float(np.linalg.norm(_sun_circular(t)))
            assert abs(r - AU) / AU < 1e-6, f"radius drift at t={t}"


# ══════════════════════════════════════════════════════════════════════════════
# Section 15e — build_cislunar_ephemeris verbose path
# ══════════════════════════════════════════════════════════════════════════════
class TestBuildEphemerisVerbose:
    def test_verbose_true_does_not_raise(self, capsys):
        from cislunar.physics.forces.ephemeris import build_cislunar_ephemeris

        build_cislunar_ephemeris("2025-01-01", duration_days=1.0, verbose=True)
        captured = capsys.readouterr()
        assert "ms" in captured.out or "Built" in captured.out


# ══════════════════════════════════════════════════════════════════════════════
# Section 15f — ephemeris_error_vs_circular
# ══════════════════════════════════════════════════════════════════════════════
class TestEphemerisErrorVsCircular:
    def test_returns_expected_keys(self):
        from cislunar.physics.forces.ephemeris import ephemeris_error_vs_circular

        result = ephemeris_error_vs_circular("2025-01-01", duration_days=3.0, n_samples=5)
        assert set(result.keys()) == {"moon_max_km", "moon_mean_km", "sun_max_km", "sun_mean_km"}

    def test_moon_error_positive_and_nonzero(self):
        from cislunar.physics.forces.ephemeris import ephemeris_error_vs_circular

        result = ephemeris_error_vs_circular("2025-01-01", duration_days=3.0, n_samples=5)
        assert result["moon_max_km"] > 0
        assert result["moon_mean_km"] > 0

    def test_sun_error_nonzero(self):
        from cislunar.physics.forces.ephemeris import ephemeris_error_vs_circular

        result = ephemeris_error_vs_circular("2025-01-01", duration_days=3.0, n_samples=5)
        assert result["sun_max_km"] > 0

    def test_moon_error_exceeds_1000km_over_two_weeks(self):
        from cislunar.physics.forces.ephemeris import ephemeris_error_vs_circular

        result = ephemeris_error_vs_circular("2025-01-01", duration_days=14.0, n_samples=20)
        assert result["moon_max_km"] > 1000.0, (
            f"Expected Moon circular model error > 1000 km over 14 days; "
            f"got {result['moon_max_km']:.0f} km"
        )
