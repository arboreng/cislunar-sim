"""Atmospheric drag tests: density profile, force model, solar indices, live space weather."""

from __future__ import annotations

import datetime
import math

import numpy as np
import pytest
from conftest import ONE_DAY, R_LEO, V_CIRC_LEO

import cislunar.physics.forces.drag as drag_module
from cislunar.physics import (
    Action,
    AtmosphericDragModel,
    SpaceWeatherClient,
    f107_for_date,
    make_lunar_craft,
)
from cislunar.physics.constants import Body
from cislunar.physics.forces.drag import effective_drag_area


# ══════════════════════════════════════════════════════════════════════════════
# Section 7 — Atmospheric drag
# ══════════════════════════════════════════════════════════════════════════════
class TestAtmosphericDrag:
    """Density profile, force direction, effective area, and trajectory impact."""

    @pytest.fixture
    def drag(self):
        """Default drag model: NRLMSISE-00 at F10.7 = 150, body + 32 m² sail."""
        return AtmosphericDragModel(body_area_m2=0.077, sail_area_m2=32.0)

    def test_density_at_400km_realistic(self, drag):
        rho = drag.density(np.array([Body.R_EARTH + 400e3, 0.0, 0.0]), time_s=0.0)
        assert 1e-16 < rho < 1e-12, f"ρ(400 km)={rho:.2e} kg/m³"

    def test_density_decreases_with_altitude(self, drag):
        rho_200 = drag.density(np.array([Body.R_EARTH + 200e3, 0.0, 0.0]), 0.0)
        rho_400 = drag.density(np.array([Body.R_EARTH + 400e3, 0.0, 0.0]), 0.0)
        rho_600 = drag.density(np.array([Body.R_EARTH + 600e3, 0.0, 0.0]), 0.0)
        assert rho_200 > rho_400 > rho_600, (
            f"ρ(200)={rho_200:.2e}  ρ(400)={rho_400:.2e}  ρ(600)={rho_600:.2e}"
        )

    def test_density_ratio_200_to_600_is_large(self, drag):
        rho_200 = drag.density(np.array([Body.R_EARTH + 200e3, 0.0, 0.0]), 0.0)
        rho_600 = drag.density(np.array([Body.R_EARTH + 600e3, 0.0, 0.0]), 0.0)
        assert rho_200 / rho_600 > 100, f"ratio={rho_200 / rho_600:.0f}×"

    def test_zero_density_above_altitude_limit(self, drag):
        """Drag model returns zero above its 1200 km validity ceiling."""
        rho = drag.density(np.array([Body.R_EARTH + 1500e3, 0.0, 0.0]), 0.0)
        assert rho == 0.0, f"ρ(1500 km)={rho}"

    def test_drag_force_opposes_velocity(self, drag):
        pos_400 = np.array([Body.R_EARTH + 400e3, 0.0, 0.0])
        vel = np.array([0.0, 7668.0, 0.0])
        sail_norm = np.array([-1.0, 0.0, 0.0])
        F = drag.force(pos_400, vel, sail_norm, time_s=0.0)
        assert F[1] < 0, f"F_y={F[1]:.3e} — should be negative (opposes +y velocity)"

    def test_drag_force_is_collinear_with_velocity(self, drag):
        pos_400 = np.array([Body.R_EARTH + 400e3, 0.0, 0.0])
        vel = np.array([0.0, 7668.0, 0.0])
        sail_norm = np.array([-1.0, 0.0, 0.0])
        F = drag.force(pos_400, vel, sail_norm, time_s=0.0)
        assert abs(F[0]) < abs(F[1]) and abs(F[2]) < abs(F[1]), f"F={F}"

    def test_effective_area_sail_perpendicular_to_velocity(self):
        v_unit = np.array([0.0, 1.0, 0.0])
        sail_perp = np.array([1.0, 0.0, 0.0])
        A = effective_drag_area(sail_perp, v_unit, body_area_m2=0.077, sail_area_m2=32.0)
        assert abs(A - 0.077) < 0.01, f"A={A:.4f} m² (expected ~body only = 0.077)"

    def test_effective_area_sail_parallel_to_velocity(self):
        v_unit = np.array([0.0, 1.0, 0.0])
        sail_parallel = np.array([0.0, 1.0, 0.0])
        A = effective_drag_area(sail_parallel, v_unit, 0.077, 32.0)
        assert abs(A - (0.077 + 32.0)) < 0.1, f"A={A:.3f} m² (expected body + sail)"

    SAIL_THRUST_LEO_MS2 = 1.3e-5

    def test_drag_at_400km_below_20pct_of_sail(self, drag):
        pos = np.array([Body.R_EARTH + 400e3, 0.0, 0.0])
        vel = np.array([0.0, 7668.0, 0.0])
        sail_parallel = np.array([0.0, 1.0, 0.0])
        a = drag.acceleration(pos, vel, sail_parallel, total_mass_kg=12.0, time_s=0.0)
        ratio = np.linalg.norm(a) / self.SAIL_THRUST_LEO_MS2
        assert ratio < 0.20, f"drag/sail at 400 km = {ratio:.3f} (expect < 0.2)"

    def test_drag_at_400km_nontrivial(self, drag):
        pos = np.array([Body.R_EARTH + 400e3, 0.0, 0.0])
        vel = np.array([0.0, 7668.0, 0.0])
        sail_parallel = np.array([0.0, 1.0, 0.0])
        a = drag.acceleration(pos, vel, sail_parallel, 12.0, 0.0)
        ratio = np.linalg.norm(a) / self.SAIL_THRUST_LEO_MS2
        assert ratio > 0.001, f"drag/sail at 400 km = {ratio:.4f} (expect > 0.1%)"

    def test_drag_at_200km_dominates_sail(self, drag):
        pos = np.array([Body.R_EARTH + 200e3, 0.0, 0.0])
        vel = np.array([0.0, 7668.0, 0.0])
        sail_parallel = np.array([0.0, 1.0, 0.0])
        a = drag.acceleration(pos, vel, sail_parallel, 12.0, 0.0)
        ratio = np.linalg.norm(a) / self.SAIL_THRUST_LEO_MS2
        assert ratio > 1.0, f"drag/sail at 200 km = {ratio:.2f} (expect > 1)"

    def test_orbital_decay_rate_at_400km_negative_and_bounded(self, drag):
        pos = np.array([Body.R_EARTH + 400e3, 0.0, 0.0])
        vel = np.array([0.0, 7668.0, 0.0])
        decay = drag.orbital_decay_rate(pos, vel, total_mass_kg=12.0, time_s=0.0)
        assert decay < 0, f"decay={decay:.2f} m/day (expect negative)"
        assert abs(decay) < 100, f"|decay|={abs(decay):.2f} m/day (expect < 100)"

    def test_one_day_orbit_decays_with_drag(self, make_leo_spacecraft):
        sc_drag = make_leo_spacecraft()
        sc_nodrag = make_leo_spacecraft()
        sc_drag.attitude = None
        sc_nodrag.attitude = None
        sc_nodrag.drag = None

        coast = Action(attitude_dir_cmd=np.array([-1.0, 0.0, 0.0]), throttle=0.0)
        sc_drag.step(coast, dt_requested=ONE_DAY)
        sc_nodrag.step(coast, dt_requested=ONE_DAY)

        alt_drag = np.linalg.norm(sc_drag.state.position) - Body.R_EARTH
        alt_nodrag = np.linalg.norm(sc_nodrag.state.position) - Body.R_EARTH
        alt_drop_m = alt_nodrag - alt_drag
        assert alt_drop_m > 0.1, f"1-day decay Δalt = {alt_drop_m:.3f} m (expect > 0.1)"


# ══════════════════════════════════════════════════════════════════════════════
# Section 14 — Solar indices for drag
# ══════════════════════════════════════════════════════════════════════════════
class TestSolarIndicesForDrag:
    """F10.7 lookup table and its effect on atmospheric density."""

    def test_lightsail2_epoch_is_solar_minimum(self):
        """2019-08 is a canonical solar-minimum epoch (F10.7 < 80)."""
        assert f107_for_date(2019, 8) < 80, f"F10.7(2019-08)={f107_for_date(2019, 8)}"

    def test_cycle24_peak_is_high(self):
        assert f107_for_date(2014, 7) > 130, f"F10.7(2014-07)={f107_for_date(2014, 7)}"

    def test_all_table_values_are_physically_plausible(self):
        """Every (year, month) in 2005–2025 has F10.7 in the 50–250 SFU range."""
        for y in range(2005, 2026):
            for m in range(1, 13):
                v = f107_for_date(y, m)
                assert 50 < v < 250, f"F10.7({y}-{m:02d})={v}"

    @pytest.fixture
    def drag_at_400(self):
        """Drag model and a 400 km reference position."""
        drag = AtmosphericDragModel()
        pos = np.array([Body.R_EARTH + 400e3, 0.0, 0.0])
        return drag, pos

    def test_solar_min_gives_lower_density(self, drag_at_400):
        drag, pos = drag_at_400
        rho_default = drag.density(pos, 0.0)
        drag.update_indices(68.0)
        rho_min = drag.density(pos, 0.0)
        assert rho_min < rho_default, f"default={rho_default:.3e}  min={rho_min:.3e}"

    def test_solar_max_gives_higher_density(self, drag_at_400):
        drag, pos = drag_at_400
        rho_default = drag.density(pos, 0.0)
        drag.update_indices(200.0)
        rho_max = drag.density(pos, 0.0)
        assert rho_max > rho_default, f"default={rho_default:.3e}  max={rho_max:.3e}"

    def test_solar_max_min_ratio_exceeds_5x(self, drag_at_400):
        """Density at 400 km swings by > 5× across the solar cycle."""
        drag, pos = drag_at_400
        drag.update_indices(68.0)
        rho_min = drag.density(pos, 0.0)
        drag.update_indices(200.0)
        rho_max = drag.density(pos, 0.0)
        ratio = rho_max / rho_min
        assert ratio > 5, f"max/min ratio = {ratio:.1f}×"


# ══════════════════════════════════════════════════════════════════════════════
# Live space-weather drag
# ══════════════════════════════════════════════════════════════════════════════
class TestLiveSpaceWeatherDrag:
    """Latitude-aware density path and live-space-weather fallback logic."""

    @pytest.fixture
    def pos_equator_400(self):
        return np.array([Body.R_EARTH + 400e3, 0.0, 0.0], dtype=float)

    @pytest.fixture
    def pos_high_lat_400(self):
        r = Body.R_EARTH + 400e3
        lat = math.radians(60.0)
        return np.array([r * math.cos(lat), 0.0, r * math.sin(lat)], dtype=float)

    def test_live_density_includes_latitude_correction(
        self, monkeypatch, pos_equator_400, pos_high_lat_400
    ):
        client = SpaceWeatherClient(
            fetch_json_fn=lambda: {
                "data": [
                    {"time": "2026-04-21T00:00:00Z", "metric": "f107", "value": 150.0},
                    {"time": "2026-04-21T00:00:00Z", "metric": "ap", "value": 4.0},
                ]
            }
        )
        drag = AtmosphericDragModel(use_live_indices=True, space_weather_client=client)

        def fake_density(altitude_km, lat_deg, lon_deg, doy, hour_ut, f107, ap):
            base = 4.9e-15 * (f107 / 150.0) ** 1.6
            lat_scale = 1.0 + 0.35 * abs(lat_deg) / 90.0
            return base * lat_scale

        monkeypatch.setattr(drag_module, "_density_nrlmsise", fake_density)

        rho_eq = drag.density(pos_equator_400, 0.0)
        rho_hi = drag.density(pos_high_lat_400, 0.0)
        ratio = rho_hi / rho_eq
        assert 1.20 < ratio < 1.40, f"latitude ratio={ratio:.3f} (expected 1.2–1.4)"

    def test_full_density_f107_300_is_about_double_150(self, monkeypatch, pos_equator_400):
        drag = AtmosphericDragModel(use_live_indices=False)

        def fake_density(altitude_km, lat_deg, lon_deg, doy, hour_ut, f107, ap):
            return 4.9e-15 * (f107 / 150.0) ** 1.0

        monkeypatch.setattr(drag_module, "_density_nrlmsise", fake_density)

        rho_150 = drag._density_nrlmsise_full(pos_equator_400, 0.0, 150.0, 4.0)
        rho_300 = drag._density_nrlmsise_full(pos_equator_400, 0.0, 300.0, 4.0)
        ratio = rho_300 / rho_150
        assert 1.9 < ratio < 2.1, f"ratio={ratio:.3f} (expected ~2x)"

    def test_live_indices_fall_back_to_cache(self, tmp_path):
        cache_path = tmp_path / "space_weather.json"
        payload = {
            "data": [
                {"time": "2026-04-21T00:00:00Z", "metric": "f107", "value": 180.0},
                {"time": "2026-04-21T00:00:00Z", "metric": "ap", "value": 12.0},
            ]
        }
        now = datetime.datetime(2026, 4, 21, 12, 0, tzinfo=datetime.UTC)
        client = SpaceWeatherClient(
            cache_path=cache_path,
            fetch_json_fn=lambda: payload,
            now_fn=lambda: now,
        )
        assert client.get_indices() == (180.0, 12.0)

        failing = SpaceWeatherClient(
            cache_path=cache_path,
            fetch_json_fn=lambda: (_ for _ in ()).throw(RuntimeError("offline")),
            now_fn=lambda: now + datetime.timedelta(hours=7),
        )
        assert failing.get_indices() == (180.0, 12.0)

    def test_make_lunar_craft_exposes_live_drag_flag(self, monkeypatch):
        class StubClient:
            def get_indices(self):
                return (180.0, 12.0)

        monkeypatch.setattr(drag_module, "SpaceWeatherClient", lambda: StubClient())
        sc = make_lunar_craft(
            position_m=[R_LEO, 0.0, 0.0],
            velocity_ms=[0.0, V_CIRC_LEO, 0.0],
            use_live_drag_indices=True,
        )
        assert sc.drag is not None
        assert sc.drag._use_live_indices is True

    @pytest.mark.network
    def test_space_weather_client_live_smoke(self):
        client = SpaceWeatherClient(ttl_s=0.0)
        f107, ap = client.get_indices()
        assert 50.0 < f107 < 400.0, f"f107={f107}"
        assert 0.0 <= ap < 400.0, f"ap={ap}"


# ══════════════════════════════════════════════════════════════════════════════
# Gap-closing additions
# ══════════════════════════════════════════════════════════════════════════════
class TestF107ForDate:
    def test_known_year_returns_table_value(self):
        val = drag_module.f107_for_date(2020, 1)
        assert isinstance(val, float) and val > 0

    def test_year_outside_table_returns_default(self):
        val = drag_module.f107_for_date(1800, 6)
        assert val == pytest.approx(drag_module.F10_7_DEFAULT)


class TestDefaultSpaceWeatherCachePath:
    def test_returns_path(self):
        p = drag_module._default_space_weather_cache_path()
        assert "cislunar" in str(p).lower()

    def test_xdg_cache_home_used_when_set(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
        p = drag_module._default_space_weather_cache_path()
        assert str(tmp_path / "xdg") in str(p)


class TestSpaceWeatherClientParseTimestamp:
    def test_none_returns_none(self):
        assert drag_module.SpaceWeatherClient._parse_timestamp(None) is None

    def test_non_string_returns_none(self):
        assert drag_module.SpaceWeatherClient._parse_timestamp(12345) is None

    def test_invalid_string_returns_none(self):
        assert drag_module.SpaceWeatherClient._parse_timestamp("not-a-date") is None

    def test_valid_iso_returns_datetime(self):
        dt = drag_module.SpaceWeatherClient._parse_timestamp("2025-01-01T00:00:00+00:00")
        assert dt is not None and dt.year == 2025

    def test_z_suffix_accepted(self):
        dt = drag_module.SpaceWeatherClient._parse_timestamp("2025-06-01T12:00:00Z")
        assert dt is not None


class TestSpaceWeatherClientParseIndices:
    import datetime as _dt

    _NOW = __import__("datetime").datetime(
        2025, 1, 2, 12, 0, tzinfo=__import__("datetime").timezone.utc
    )

    def _payload(self, rows):
        return {"data": rows}

    def test_not_dict_raises(self):
        with pytest.raises(ValueError, match="Unexpected"):
            drag_module.SpaceWeatherClient._parse_indices("string", now=self._NOW)

    def test_missing_data_key_raises(self):
        with pytest.raises(ValueError, match="missing data list"):
            drag_module.SpaceWeatherClient._parse_indices({}, now=self._NOW)

    def test_no_usable_rows_raises(self):
        with pytest.raises(ValueError, match="no usable forecast rows"):
            drag_module.SpaceWeatherClient._parse_indices({"data": []}, now=self._NOW)

    def test_missing_ap_raises(self):
        rows = [{"metric": "f107", "time": "2025-01-02T00:00:00Z", "value": 150.0}]
        with pytest.raises(ValueError, match="both F10.7 and Ap"):
            drag_module.SpaceWeatherClient._parse_indices(self._payload(rows), now=self._NOW)

    def test_valid_payload_returns_values(self):
        rows = [
            {"metric": "f107", "time": "2025-01-02T00:00:00Z", "value": 150.0},
            {"metric": "ap", "time": "2025-01-02T00:00:00Z", "value": 7.0},
        ]
        f107, ap = drag_module.SpaceWeatherClient._parse_indices(self._payload(rows), now=self._NOW)
        assert f107 == pytest.approx(150.0) and ap == pytest.approx(7.0)

    def test_non_dict_rows_are_skipped(self):
        rows = [
            "bad_row",
            {"metric": "f107", "time": "2025-01-02T00:00:00Z", "value": 140.0},
            {"metric": "ap", "time": "2025-01-02T00:00:00Z", "value": 5.0},
        ]
        f107, ap = drag_module.SpaceWeatherClient._parse_indices(self._payload(rows), now=self._NOW)
        assert f107 == pytest.approx(140.0)


class TestSpaceWeatherClientLoadCache:
    def test_missing_file_returns_none(self, tmp_path):
        client = drag_module.SpaceWeatherClient(cache_path=tmp_path / "missing.json")
        assert client._load_cache() is None

    def test_corrupted_json_returns_none(self, tmp_path):
        p = tmp_path / "cache.json"
        p.write_text("not valid json")
        client = drag_module.SpaceWeatherClient(cache_path=p)
        assert client._load_cache() is None

    def test_non_dict_json_returns_none(self, tmp_path):
        import json

        p = tmp_path / "cache.json"
        p.write_text(json.dumps([1, 2, 3]))
        client = drag_module.SpaceWeatherClient(cache_path=p)
        assert client._load_cache() is None

    def test_missing_fields_returns_none(self, tmp_path):
        import json

        p = tmp_path / "cache.json"
        p.write_text(json.dumps({"f107": 150.0}))
        client = drag_module.SpaceWeatherClient(cache_path=p)
        assert client._load_cache() is None

    def test_valid_cache_returned(self, tmp_path):
        import json

        p = tmp_path / "cache.json"
        p.write_text(
            json.dumps({"f107": 150.0, "ap": 7.0, "fetched_at": "2025-01-01T00:00:00+00:00"})
        )
        client = drag_module.SpaceWeatherClient(cache_path=p)
        cached = client._load_cache()
        assert cached is not None and cached["f107"] == pytest.approx(150.0)


class TestSpaceWeatherClientGetIndices:
    def test_uses_fresh_cache_without_network(self, tmp_path):
        import datetime
        import json

        now = datetime.datetime(2025, 1, 2, 12, 0, tzinfo=datetime.UTC)
        p = tmp_path / "cache.json"
        p.write_text(json.dumps({"f107": 155.0, "ap": 8.0, "fetched_at": now.isoformat()}))
        client = drag_module.SpaceWeatherClient(
            cache_path=p,
            ttl_s=3600.0,
            now_fn=lambda: now,
        )
        f107, ap = client.get_indices()
        assert f107 == pytest.approx(155.0)

    def test_fetch_failure_raises_when_no_cache(self, tmp_path):
        client = drag_module.SpaceWeatherClient(
            cache_path=tmp_path / "empty.json",
            fetch_json_fn=lambda: (_ for _ in ()).throw(OSError("blocked")),
        )
        with pytest.raises(RuntimeError, match="Unable to fetch"):
            client.get_indices()

    def test_fetch_failure_falls_back_to_stale_cache(self, tmp_path):
        import datetime
        import json

        old_time = datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC)
        now = datetime.datetime(2025, 1, 2, tzinfo=datetime.UTC)
        p = tmp_path / "cache.json"
        p.write_text(json.dumps({"f107": 130.0, "ap": 4.0, "fetched_at": old_time.isoformat()}))
        client = drag_module.SpaceWeatherClient(
            cache_path=p,
            ttl_s=1.0,
            now_fn=lambda: now,
            fetch_json_fn=lambda: (_ for _ in ()).throw(OSError("blocked")),
        )
        f107, _ = client.get_indices()
        assert f107 == pytest.approx(130.0)

    def test_successful_fetch_and_cache_write(self, tmp_path):
        import datetime
        import json

        now = datetime.datetime(2025, 1, 2, 12, 0, tzinfo=datetime.UTC)
        rows = [
            {"metric": "f107", "time": "2025-01-02T00:00:00Z", "value": 145.0},
            {"metric": "ap", "time": "2025-01-02T00:00:00Z", "value": 6.0},
        ]
        p = tmp_path / "cache.json"
        client = drag_module.SpaceWeatherClient(
            cache_path=p,
            now_fn=lambda: now,
            fetch_json_fn=lambda: {"data": rows},
        )
        f107, ap = client.get_indices()
        assert f107 == pytest.approx(145.0) and ap == pytest.approx(6.0)
        assert json.loads(p.read_text())["f107"] == pytest.approx(145.0)


class TestAtmosphericDragModelGaps:
    def test_high_altitude_returns_zero_density(self):
        dm = AtmosphericDragModel()
        pos_above = np.array([R_LEO + 1000e3, 0.0, 0.0])  # far above model ceiling
        assert dm.density(pos_above, 0.0) == pytest.approx(0.0)

    def test_density_nrlmsise_full_above_limit_returns_zero(self):
        dm = AtmosphericDragModel()
        pos_above = np.array([R_LEO + 1000e3, 0.0, 0.0])
        assert dm._density_nrlmsise_full(pos_above, 0.0, 150.0, 9.0) == pytest.approx(0.0)

    def test_density_nrlmsise_full_at_leo_nonzero(self):
        dm = AtmosphericDragModel()
        rho = dm._density_nrlmsise_full(np.array([R_LEO, 0.0, 0.0]), 0.0, 150.0, 9.0)
        assert rho > 0

    def test_geocentric_lat_lon_degenerate_position(self):
        lat, lon = AtmosphericDragModel._geocentric_lat_lon(np.zeros(3), 0.0)
        assert lat == 0.0 and lon == 0.0

    def test_force_zero_velocity_returns_zero(self):
        dm = AtmosphericDragModel()
        f = dm.force(np.array([R_LEO, 0.0, 0.0]), np.zeros(3), np.array([1.0, 0.0, 0.0]), 0.0)
        np.testing.assert_array_equal(f, np.zeros(3))

    def test_force_high_altitude_returns_zero(self):
        dm = AtmosphericDragModel()
        pos_high = np.array([R_LEO + 1000e3, 0.0, 0.0])
        f = dm.force(pos_high, np.array([0.0, 7500.0, 0.0]), np.array([1.0, 0.0, 0.0]), 0.0)
        np.testing.assert_array_equal(f, np.zeros(3))

    def test_update_indices_changes_lut(self):
        dm = AtmosphericDragModel()
        rho_before = dm.density(np.array([R_LEO, 0.0, 0.0]), 0.0)
        dm.update_indices(f107=200.0)
        rho_after = dm.density(np.array([R_LEO, 0.0, 0.0]), 0.0)
        assert rho_after != rho_before  # F10.7 200 vs 150 → different density

    def test_orbital_decay_rate_is_negative(self):
        dm = AtmosphericDragModel()
        dr_dt = dm.orbital_decay_rate(
            np.array([R_LEO, 0.0, 0.0]),
            np.array([0.0, V_CIRC_LEO, 0.0]),
            total_mass_kg=12.5,
            time_s=0.0,
        )
        assert dr_dt < 0.0

    def test_live_indices_path_uses_client(self, tmp_path):
        import datetime
        import json

        now = datetime.datetime(2025, 1, 2, 12, 0, tzinfo=datetime.UTC)
        p = tmp_path / "cache.json"
        p.write_text(json.dumps({"f107": 155.0, "ap": 8.0, "fetched_at": now.isoformat()}))
        sw_client = drag_module.SpaceWeatherClient(cache_path=p, ttl_s=3600.0, now_fn=lambda: now)
        dm = AtmosphericDragModel(use_live_indices=True, space_weather_client=sw_client)
        rho = dm.density(np.array([R_LEO, 0.0, 0.0]), 0.0)
        assert rho > 0
