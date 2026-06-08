"""
Atmospheric drag model — NRLMSISE-00 density + aerodynamic force.

Why this matters for the simulation
─────────────────────────────────────
At 400 km altitude, drag produces ~4-5% of the sail thrust.
That's not enough to dominate mission strategy, but it's enough to:

  - Decay the orbit if a spacecraft lingers in LEO too long (real missions lose
    ~2 km of altitude per day at 400 km in moderate solar conditions)
  - Penalise orientations that present the large sail face into the velocity vector
    (sail drag = 4.5% of sail thrust, so sail attitude affects net acceleration,
    not just solar thrust direction)
  - Enable drag-augmented deorbit manoeuvres
    (a real strategy used by LEO spacecraft to reduce required propellant)

At 200 km, drag is 2.5× the sail thrust — it completely dominates and must
be accounted for explicitly.

NRLMSISE-00 model
──────────────────
NRLMSISE-00 (Naval Research Laboratory Mass Spectrometer and Incoherent
Scatter Radar Extended 2000) is the US standard atmosphere model for
altitudes 0–1000+ km. It accounts for:

  - Solar activity via F10.7 index (10.7 cm solar radio flux, proxy for UV)
  - Geomagnetic activity via Ap index
  - Latitude, longitude, local solar time
  - Seasonal variation

nrlmsise00 is a required dependency. Install cislunar-sim normally and it
will be present.

Usage
──────
  drag = AtmosphericDragModel(
      spacecraft_area_m2=0.077,    # ram-facing body area
      sail_area_m2=32.0,           # sail area (added when sail faces velocity)
      drag_coeff=2.2,
  )

  # In the derivatives function:
  a_drag = drag.acceleration(position_eci, velocity_eci, sail_normal, mass, time_s)
"""

from __future__ import annotations

import datetime
import json
import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict
from urllib.request import urlopen

import numpy as np
from numpy.typing import NDArray

# ── Constants ─────────────────────────────────────────────────────────────────

R_EARTH_M = 6.371e6  # mean radius [m]
J2000_EPOCH = datetime.datetime(2000, 1, 1, 12, 0, 0)  # J2000.0
SIDEREAL_DAY_S = 86164.0905
CACHE_TTL_S = 6 * 3600
# SWPC's current consolidated JSON feed for daily Ap and F10.7 values.
SPACE_WEATHER_URL = "https://services.swpc.noaa.gov/json/45-day-forecast.json"

# Solar / geomagnetic activity indices — moderate average conditions.
# Real operations: replace with daily indices from
#   https://celestrak.org/SpaceData/SW-Last5Years.csv
F10_7_DEFAULT = 150.0  # solar radio flux index [SFU]  (low ~70, high ~250)
AP_DEFAULT = 4.0  # geomagnetic Ap index (quiet ~4, storm ~100)


# ── Bundled monthly F10.7 solar flux index (NOAA/SWPC, cycles 23-25) ─────────
_F107_MONTHLY: dict = {
    2005: [102, 103, 123, 113, 100, 83, 90, 107, 120, 85, 110, 104],
    2006: [86, 81, 80, 81, 82, 80, 75, 72, 75, 77, 80, 80],
    2007: [83, 79, 75, 75, 78, 77, 70, 68, 70, 68, 71, 74],
    2008: [81, 75, 73, 69, 68, 69, 66, 66, 70, 67, 69, 72],
    2009: [72, 70, 69, 70, 72, 68, 68, 68, 68, 73, 73, 76],
    2010: [80, 88, 89, 101, 89, 81, 84, 97, 98, 105, 87, 94],
    2011: [100, 110, 104, 110, 109, 101, 100, 97, 120, 120, 134, 135],
    2012: [145, 136, 123, 122, 113, 102, 97, 117, 120, 109, 115, 113],
    2013: [126, 113, 113, 136, 130, 126, 123, 109, 104, 114, 157, 168],
    2014: [167, 175, 159, 172, 151, 147, 144, 151, 148, 180, 174, 166],
    2015: [140, 148, 136, 117, 120, 108, 115, 100, 114, 124, 108, 93],
    2016: [97, 99, 98, 86, 89, 84, 79, 79, 83, 80, 72, 79],
    2017: [78, 73, 73, 77, 75, 74, 72, 86, 93, 86, 73, 68],
    2018: [76, 72, 70, 71, 72, 70, 73, 70, 69, 72, 70, 72],
    2019: [71, 72, 73, 70, 68, 68, 68, 68, 70, 70, 72, 72],
    2020: [73, 74, 70, 69, 69, 71, 70, 71, 73, 76, 80, 89],
    2021: [83, 82, 80, 89, 93, 88, 87, 105, 104, 85, 107, 100],
    2022: [105, 111, 130, 123, 121, 107, 126, 138, 130, 157, 151, 136],
    2023: [149, 144, 151, 165, 168, 163, 168, 155, 188, 195, 175, 170],
    2024: [185, 218, 192, 185, 195, 195, 185, 195, 200, 190, 175, 165],
    2025: [160, 165, 155, 150, 150, 148, 145, 143, 140, 138, 135, 132],
}


class SpaceWeatherCache(TypedDict):
    """Cached live space-weather indices."""

    f107: float
    ap: float
    fetched_at: str


def f107_for_date(year: int, month: int) -> float:
    """Monthly-averaged F10.7 [SFU] for the given year/month.
    Returns climatological average for dates outside the 2005-2025 table."""
    if year in _F107_MONTHLY:
        return float(_F107_MONTHLY[year][max(0, min(month - 1, 11))])
    return F10_7_DEFAULT


def _default_space_weather_cache_path() -> Path:
    cache_home = os.environ.get("XDG_CACHE_HOME")
    if cache_home:
        base = Path(cache_home)
    else:
        base = Path.home() / ".cache"
    return base / "cislunar-sim" / "space_weather.json"


class SpaceWeatherClient:
    """Fetch and cache live F10.7 / Ap indices from NOAA SWPC."""

    def __init__(
        self,
        forecast_url: str = SPACE_WEATHER_URL,
        cache_path: str | os.PathLike[str] | None = None,
        ttl_s: float = CACHE_TTL_S,
        fetch_json_fn: Callable[[], object] | None = None,
        now_fn: Callable[[], datetime.datetime] | None = None,
    ):
        self.forecast_url = forecast_url
        self.cache_path = (
            Path(cache_path) if cache_path is not None else _default_space_weather_cache_path()
        )
        self.ttl_s = ttl_s
        self._fetch_json_fn = fetch_json_fn or self._fetch_json
        self._now_fn = now_fn or (lambda: datetime.datetime.now(datetime.UTC))

    def get_indices(self) -> tuple[float, float]:
        now = self._now_fn()
        cached = self._load_cache()
        fetched_at = self._parse_timestamp(cached.get("fetched_at")) if cached else None

        if cached and fetched_at is not None and (now - fetched_at).total_seconds() <= self.ttl_s:
            return float(cached["f107"]), float(cached["ap"])

        try:
            f107, ap = self._parse_indices(self._fetch_json_fn(), now=now)
        except Exception as exc:
            if cached:
                return float(cached["f107"]), float(cached["ap"])
            raise RuntimeError("Unable to fetch live space weather indices") from exc

        self._save_cache(
            {
                "f107": float(f107),
                "ap": float(ap),
                "fetched_at": now.isoformat(),
            }
        )
        return float(f107), float(ap)

    def _fetch_json(self) -> object:
        with urlopen(self.forecast_url, timeout=5.0) as response:
            payload = response.read().decode("utf-8")
        return json.loads(payload)

    @staticmethod
    def _parse_timestamp(value: object) -> datetime.datetime | None:
        if not isinstance(value, str):
            return None
        try:
            return datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None

    @classmethod
    def _parse_indices(
        cls,
        payload: object,
        now: datetime.datetime,
    ) -> tuple[float, float]:
        if not isinstance(payload, dict):
            raise ValueError("Unexpected NOAA space weather payload")
        data = payload.get("data")
        if not isinstance(data, list):
            raise ValueError("NOAA space weather payload missing data list")

        by_day: dict[datetime.date, dict[str, float]] = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            metric = item.get("metric")
            stamp = cls._parse_timestamp(item.get("time"))
            value = item.get("value")
            if stamp is None or metric not in {"f107", "ap"} or not isinstance(value, (int, float)):
                continue
            by_day.setdefault(stamp.date(), {})[metric] = float(value)

        if not by_day:
            raise ValueError("NOAA space weather payload had no usable forecast rows")

        today = now.date()
        candidate_days = sorted(
            day for day, metrics in by_day.items() if {"f107", "ap"} <= metrics.keys()
        )
        if not candidate_days:
            raise ValueError("NOAA space weather payload did not contain both F10.7 and Ap")

        chosen_day = next((day for day in candidate_days if day >= today), candidate_days[-1])
        metrics = by_day[chosen_day]
        return metrics["f107"], metrics["ap"]

    def _load_cache(self) -> SpaceWeatherCache | None:
        try:
            with self.cache_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        if not isinstance(payload, dict):
            return None
        if (
            "f107" not in payload
            or "ap" not in payload
            or "fetched_at" not in payload
            or not isinstance(payload["f107"], (int, float))
            or not isinstance(payload["ap"], (int, float))
            or not isinstance(payload["fetched_at"], str)
        ):
            return None
        return {
            "f107": float(payload["f107"]),
            "ap": float(payload["ap"]),
            "fetched_at": payload["fetched_at"],
        }

    def _save_cache(self, payload: SpaceWeatherCache) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle)
        except OSError:
            pass


# ── Density providers ─────────────────────────────────────────────────────────


def _density_nrlmsise(
    altitude_km: float,
    lat_deg: float,
    lon_deg: float,
    doy: int,
    hour_ut: float,
    f107: float,
    ap: float,
) -> float:
    """Total mass density [kg/m³] from NRLMSISE-00."""
    from nrlmsise00 import msise_flat  # pyright: ignore[reportMissingImports]

    dt = J2000_EPOCH + datetime.timedelta(days=doy - 1, hours=hour_ut)
    result = msise_flat(dt, altitude_km, lat_deg, lon_deg, f107, f107, ap)
    rho = float(result[5])  # total mass density [kg/m³]
    return rho if rho > 0 else 0.0


def _density_exponential(altitude_m: float, f107: float = F10_7_DEFAULT) -> float:
    """
    Simple exponential atmosphere (used only for LUT self-test comparisons).

    Calibrated against NRLMSISE-00 at F10.7=150, Ap=4:
        200 km: 2.7e-13 kg/m³
        400 km: 4.9e-15 kg/m³
        600 km: 2.6e-16 kg/m³

    Scale height and base chosen to match at 400 km reference.

    F10.7 scaling: thermospheric density scales as (F10.7/150)^1.6.
    This reproduces the ~5–8× density range between solar min (F10.7≈70)
    and solar max (F10.7≈200) seen in NRLMSISE-00 at LEO altitudes.
    Exponent 1.6 gives a max/min ratio of ~5.5 at 400 km, consistent
    with the NRLMSISE reference values used by the test suite.
    """
    if altitude_m < 0:
        return 1.225  # sea level
    h_ref = 400e3
    rho_ref = 4.9e-15
    H = 60e3  # scale height [m]
    solar_scale = (f107 / F10_7_DEFAULT) ** 1.6
    return rho_ref * solar_scale * math.exp(-(altitude_m - h_ref) / H)


# ── Effective drag area ───────────────────────────────────────────────────────


def effective_drag_area(
    sail_normal: NDArray[np.float64],
    velocity_unit: NDArray[np.float64],
    body_area_m2: float,
    sail_area_m2: float,
) -> float:
    """
    Effective frontal area presented to the ram flow.

    The CubeSat body always contributes its ram-facing cross-section.
    The sail contributes its projected area in the velocity direction
    (|cos θ| where θ is angle between sail normal and velocity vector).

    When the sail is aligned for maximum solar thrust it faces the Sun,
    not the velocity vector, so sail drag is usually < 10% of sail area.
    When a guidance law aligns the sail in the ram
    direction, it pays for it here.

    Args:
        sail_normal:    unit normal to the sail face (inertial)
        velocity_unit:  unit vector of spacecraft velocity (inertial)
        body_area_m2:   CubeSat bus ram cross-section [m²]
        sail_area_m2:   total sail area [m²]

    Returns:
        effective area [m²]
    """
    cos_theta = abs(float(np.dot(sail_normal, velocity_unit)))
    return body_area_m2 + sail_area_m2 * cos_theta


# ── Main model class ──────────────────────────────────────────────────────────


class AtmosphericDragModel:
    """
    Aerodynamic drag force for a solar-sail CubeSat in low Earth orbit.

    NRLMSISE-00 aerodynamic drag for a solar-sail CubeSat in low Earth orbit.

    Args:
        body_area_m2:  CubeSat bus ram-facing cross-section [m²]
                       12U bus: ≈ 0.226 m × 0.341 m = 0.077 m²
        sail_area_m2:  Solar sail total area [m²] — drag contribution
                       is projection onto velocity direction
        drag_coeff:    Aerodynamic drag coefficient (flat plate ≈ 2.2,
                       sphere = 2.0; 2.2 is standard for CubeSats)
        f107:          F10.7 solar radio flux index (daily value in SFU)
        ap:            Geomagnetic Ap index (3-hour value)
        use_live_indices:
                       If True, fetch live F10.7/Ap values and evaluate
                       NRLMSISE per call using the spacecraft latitude and
                       local solar time. Defaults False for deterministic
                       offline/test use.
        altitude_limit_km: Zero drag above this altitude [km].
                           Above ~1200 km density is negligible and the
                           model is extrapolating anyway.
    """

    def __init__(
        self,
        body_area_m2: float = 0.077,
        sail_area_m2: float = 32.0,
        drag_coeff: float = 2.2,
        f107: float = F10_7_DEFAULT,
        ap: float = AP_DEFAULT,
        use_live_indices: bool = False,
        altitude_limit_km: float = 1200.0,
        lut_step_km: float = 5.0,  # altitude LUT resolution [km]
        space_weather_client: SpaceWeatherClient | None = None,
    ):
        self.body_area = body_area_m2
        self.sail_area = sail_area_m2
        self.Cd = drag_coeff
        self.f107 = f107
        self.ap = ap
        self._use_live_indices = bool(use_live_indices)
        self._alt_limit = altitude_limit_km * 1e3
        self._space_weather = space_weather_client or SpaceWeatherClient()

        # Pre-compute density LUT at construction time (one call per 5km bin).
        # This replaces ~427 nrlmsise calls per training step with a single
        # array lookup. Atmospheric density changes < 0.1% over 1 km at LEO.
        #
        # ── PHASE 1 FIDELITY LIMIT — atmospheric drag ─────────────────────────
        # The LUT is altitude-only: density is sampled at lat=0, lon=0 and
        # held constant for the entire mission.  Real NRLMSISE-00 density
        # varies with:
        #   - Geographic latitude and local solar time (±20–40% at LEO)
        #   - Day-to-day solar/geomagnetic index variation (F10.7, Ap)
        #   - Seasonal heating patterns
        #
        # This is acceptable for simulation purposes — the dominant drag signal
        # (altitude trend) is captured.  It is NOT adequate for:
        #   - High-fidelity LEO mission analysis or debris conjunction work
        #   - Accurate orbit-decay rate prediction across an entire LEO season
        #   - Atmosphere-sensitive launch window calculations
        #
        # Upgrade path: replace the LUT with a per-call _density_nrlmsise()
        # lookup that passes the spacecraft's actual sub-satellite latitude,
        # local solar time, and live F10.7/Ap indices fetched from NOAA.
        # The density() method signature already accepts position_eci and
        # time_s; no interface changes are required.
        # ─────────────────────────────────────────────────────────────────────
        alts = list(range(100, int(altitude_limit_km) + 1, int(lut_step_km)))
        self._lut_alts_m = [a * 1e3 for a in alts]
        self._lut_rho = []
        for alt_km in alts:
            self._lut_rho.append(_density_nrlmsise(alt_km, 0, 0, 1, 12, f107, ap))
        self._lut_step_m = lut_step_km * 1e3
        self._lut_base_m = alts[0] * 1e3

    def density(
        self,
        position_eci: NDArray[np.float64],
        time_s: float,
    ) -> float:
        """
        Atmospheric density via pre-computed altitude LUT (fast path).

        Linearly interpolates between 5 km bins built at construction.
        Accuracy: < 1% error vs. per-call NRLMSISE at same conditions.
        """
        altitude_m = float(np.linalg.norm(position_eci)) - R_EARTH_M

        if altitude_m > self._alt_limit or altitude_m < 0:
            return 0.0

        if self._use_live_indices:
            try:
                f107, ap = self._space_weather.get_indices()
            except RuntimeError:
                f107, ap = self.f107, self.ap
            self.f107 = float(f107)
            self.ap = float(ap)
            return self._density_nrlmsise_full(position_eci, time_s, self.f107, self.ap)

        # LUT lookup with linear interpolation
        idx_f = (altitude_m - self._lut_base_m) / self._lut_step_m
        idx0 = int(idx_f)
        frac = idx_f - idx0
        n = len(self._lut_rho)
        idx0 = max(0, min(idx0, n - 2))
        return (1.0 - frac) * self._lut_rho[idx0] + frac * self._lut_rho[idx0 + 1]

    def _density_nrlmsise_full(
        self,
        position_eci: NDArray[np.float64],
        time_s: float,
        f107: float,
        ap: float,
    ) -> float:
        """Per-call NRLMSISE evaluation using latitude and local solar time."""
        altitude_m = float(np.linalg.norm(position_eci)) - R_EARTH_M
        if altitude_m > self._alt_limit or altitude_m < 0:
            return 0.0

        dt = J2000_EPOCH + datetime.timedelta(seconds=float(time_s))
        doy = dt.timetuple().tm_yday
        hour_ut = dt.hour + dt.minute / 60.0 + dt.second / 3600.0 + dt.microsecond / 3.6e9

        lat_deg, lon_deg = self._geocentric_lat_lon(position_eci, time_s)
        return _density_nrlmsise(
            altitude_km=altitude_m / 1e3,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            doy=doy,
            hour_ut=hour_ut,
            f107=f107,
            ap=ap,
        )

    @staticmethod
    def _geocentric_lat_lon(
        position_eci: NDArray[np.float64],
        time_s: float,
    ) -> tuple[float, float]:
        x_eci, y_eci, z_eci = [float(v) for v in position_eci]
        r = math.sqrt(x_eci * x_eci + y_eci * y_eci + z_eci * z_eci)
        if r < 1.0:
            return 0.0, 0.0

        theta = 2.0 * math.pi * (float(time_s) / SIDEREAL_DAY_S)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        x_ecef = cos_t * x_eci + sin_t * y_eci
        y_ecef = -sin_t * x_eci + cos_t * y_eci
        z_ecef = z_eci

        lat_deg = math.degrees(math.asin(max(-1.0, min(1.0, z_ecef / r))))
        lon_deg = math.degrees(math.atan2(y_ecef, x_ecef))
        return lat_deg, lon_deg

    def force(
        self,
        position_eci: NDArray[np.float64],
        velocity_eci: NDArray[np.float64],
        sail_normal: NDArray[np.float64],
        time_s: float,
    ) -> NDArray[np.float64]:
        """
        Drag force vector [N], opposing the velocity direction.

        F_drag = -½ × ρ × Cd × A_eff × v² × v̂

        Args:
            position_eci: position in Earth-centred frame [m]
            velocity_eci: velocity in inertial frame [m/s]
            sail_normal:  unit sail normal (inertial frame)
            time_s:       mission elapsed time [s]

        Returns:
            drag force vector [N]
        """
        v_mag = float(np.linalg.norm(velocity_eci))
        if v_mag < 1e-6:
            return np.zeros(3)

        rho = self.density(position_eci, time_s)
        if rho == 0.0:
            return np.zeros(3)

        v_unit = velocity_eci / v_mag
        A_eff = effective_drag_area(sail_normal, v_unit, self.body_area, self.sail_area)

        F_mag = 0.5 * rho * self.Cd * A_eff * v_mag**2
        return -F_mag * v_unit  # opposes velocity

    def acceleration(
        self,
        position_eci: NDArray[np.float64],
        velocity_eci: NDArray[np.float64],
        sail_normal: NDArray[np.float64],
        total_mass_kg: float,
        time_s: float,
    ) -> NDArray[np.float64]:
        """Drag acceleration [m/s²] = force / mass."""
        return self.force(position_eci, velocity_eci, sail_normal, time_s) / total_mass_kg

    def update_indices(self, f107: float, ap: float = AP_DEFAULT) -> None:
        """Rebuild density LUT with new solar/geomagnetic activity indices.
        Call with f107_for_date(year, month) to use mission-date-appropriate drag."""
        self.f107 = f107
        self.ap = ap
        lut_step_km = int(self._lut_step_m / 1e3)
        alts = range(100, int(self._alt_limit / 1e3) + 1, lut_step_km)
        self._lut_rho = []
        for alt_km in alts:
            self._lut_rho.append(_density_nrlmsise(alt_km, 0, 0, 1, 12, f107, ap))

    def orbital_decay_rate(
        self,
        position_eci: NDArray[np.float64],
        velocity_eci: NDArray[np.float64],
        total_mass_kg: float,
        time_s: float,
    ) -> float:
        """
        Approximate altitude decay rate [m/day] for circular orbit.

        Useful for mission planning: how long can the spacecraft loiter
        at this altitude before re-entry becomes a concern?

        Based on: dh/dt = -2π × a_drag / n  (circular orbit approximation)
        where n is mean motion.
        """
        from ..constants import Body

        r = float(np.linalg.norm(position_eci))
        n = math.sqrt(Body.MU_EARTH / r**3)  # mean motion [rad/s]

        sail_norm_default = np.array([0.0, 0.0, 1.0])  # worst-case broadside
        a_drag = np.linalg.norm(
            self.acceleration(position_eci, velocity_eci, sail_norm_default, total_mass_kg, time_s)
        )
        dh_dt = -2 * math.pi * a_drag / n  # m/s
        return float(dh_dt * 86400)  # convert to m/day
