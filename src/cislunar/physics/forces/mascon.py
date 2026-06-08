"""
Lunar mascon and spherical-harmonic gravity models.

This module provides both a lightweight 8-site mascon approximation and a
GRGM1200A-backed harmonic evaluator for higher-fidelity low-lunar-orbit work.
The harmonic model is packaged as a callable object so it can be passed
directly to ``GravityModel.set_mascon_model(...)`` without changing the
broader interface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

G = 6.674e-11
R_MOON_M = 1.7374e6


@dataclass(frozen=True)
class MasconSite:
    """One lumped lunar mascon site used by the reduced-order gravity model."""

    name: str
    lat_deg: float
    lon_deg: float
    excess_mass_kg: float


# Eight dominant sites — Zuber et al. 2013, Andrews-Hanna et al. 2013
GRAIL_MASCONS = (
    MasconSite("Mare Imbrium", lat_deg=32.8, lon_deg=344.4, excess_mass_kg=5.5e17),
    MasconSite("Mare Serenitatis", lat_deg=28.0, lon_deg=17.5, excess_mass_kg=4.2e17),
    MasconSite("Mare Crisium", lat_deg=17.0, lon_deg=59.1, excess_mass_kg=4.8e17),
    MasconSite("Mare Nectaris", lat_deg=-15.2, lon_deg=35.5, excess_mass_kg=1.8e17),
    MasconSite("Mare Humorum", lat_deg=-24.4, lon_deg=320.7, excess_mass_kg=2.2e17),
    MasconSite("Orientale Basin", lat_deg=-19.4, lon_deg=265.4, excess_mass_kg=1.5e17),
    MasconSite("Mare Marginis", lat_deg=13.3, lon_deg=86.1, excess_mass_kg=1.1e17),
    MasconSite("Mare Smythii", lat_deg=-2.5, lon_deg=87.5, excess_mass_kg=1.3e17),
)


def _build_tables() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    pos = []
    for s in GRAIL_MASCONS:
        lat, lon = math.radians(s.lat_deg), math.radians(s.lon_deg)
        pos.append(
            [
                R_MOON_M * math.cos(lat) * math.cos(lon),
                R_MOON_M * math.cos(lat) * math.sin(lon),
                R_MOON_M * math.sin(lat),
            ]
        )
    return (
        np.array(pos, dtype=np.float64),
        np.array([G * s.excess_mass_kg for s in GRAIL_MASCONS], dtype=np.float64),
    )


_BODY_POS, _G_DM = _build_tables()


def _body_to_eci(moon_pos_eci: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the Moon body-frame basis expressed in ECI coordinates."""
    r = np.linalg.norm(moon_pos_eci)
    if r < 1.0:
        return np.eye(3)

    x_axis = -moon_pos_eci / r  # body +x points toward Earth
    pole = np.array([0.0, 0.0, 1.0])
    z_axis = pole - np.dot(pole, x_axis) * x_axis
    zn = np.linalg.norm(z_axis)
    if zn < 1e-10:
        pole = np.array([1.0, 0.0, 0.0])
        z_axis = pole - np.dot(pole, x_axis) * x_axis
        zn = np.linalg.norm(z_axis)
    z_axis /= zn
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def lunar_mascon_acceleration(
    spacecraft_pos_eci: NDArray[np.float64],
    moon_pos_eci: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Total 8-site mascon acceleration on the spacecraft in ECI [m/s²].

    Returns zero if the spacecraft is more than 2,000 km above the lunar
    surface. This keeps the model cheap during cislunar transfer phases where
    the high-frequency lunar texture is dynamically irrelevant.
    """
    alt = float(np.linalg.norm(moon_pos_eci - spacecraft_pos_eci)) - R_MOON_M
    if alt > 2_000_000:
        return np.zeros(3)

    rotation = _body_to_eci(moon_pos_eci)
    mascon_eci = moon_pos_eci + (_BODY_POS @ rotation.T)
    dr = mascon_eci - spacecraft_pos_eci
    dist2 = np.einsum("ij,ij->i", dr, dr)
    dist3 = dist2 * np.sqrt(dist2)
    scale = _G_DM / dist3
    return np.einsum("n,ni->i", scale, dr)


def mascon_acceleration_at_altitude(altitude_km: float) -> float:
    """Peak 8-site mascon acceleration at lunar nadir [m/s²]."""
    r = R_MOON_M + altitude_km * 1e3
    return sum(G * s.excess_mass_kg / r**2 for s in GRAIL_MASCONS)


_GRGM1200A_DEGREE70_PATH = Path(__file__).resolve().parents[1] / "data" / "grgm1200a_degree70.csv"


@lru_cache(maxsize=1)
def _load_grgm1200a_degree70() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Load degree-70 GRGM1200A coefficients from the vendored NASA PDS slice.

    Source file:
      GGGRX_1200A_SHA.TAB from the NASA PDS Geosciences Node, truncated offline
      to degrees 2..70. ``LunarHarmonicGravity`` defaults to using only the
      residual degrees 4..70 so it composes cleanly with the existing lunar
      point-mass, J2, and J3 path in ``GravityModel``.
    """
    raw = np.loadtxt(_GRGM1200A_DEGREE70_PATH, delimiter=",", dtype=np.float64)
    degree = int(np.max(raw[:, 0]))
    c = np.zeros((degree + 1, degree + 1), dtype=np.float64)
    s = np.zeros_like(c)
    deg = raw[:, 0].astype(np.int64)
    order = raw[:, 1].astype(np.int64)
    c[deg, order] = raw[:, 2]
    s[deg, order] = raw[:, 3]
    return c, s


@lru_cache(maxsize=16)
def _legendre_recurrence_coeffs(
    max_degree: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    a = np.zeros((max_degree + 1, max_degree + 1), dtype=np.float64)
    b = np.zeros_like(a)
    prev = np.zeros_like(a)
    for n in range(2, max_degree + 1):
        m = np.arange(0, n - 1, dtype=np.float64)
        a[n, : n - 1] = np.sqrt(((2.0 * n - 1.0) * (2.0 * n + 1.0)) / ((n - m) * (n + m)))
        b[n, : n - 1] = np.sqrt(
            ((2.0 * n + 1.0) * (n + m - 1.0) * (n - m - 1.0))
            / ((n - m) * (n + m) * (2.0 * n - 3.0))
        )
        m_full = np.arange(0, n + 1, dtype=np.float64)
        mask = m_full <= (n - 1)
        prev[n, : n + 1][mask] = (n + m_full[mask]) * np.sqrt(
            ((2.0 * n + 1.0) * (n - m_full[mask])) / ((2.0 * n - 1.0) * (n + m_full[mask]))
        )
    return a, b, prev


def _normalized_legendre(
    sin_lat: float,
    max_degree: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Fully normalized associated Legendre functions and latitude derivatives.

    The GRGM coefficients are geodesy-normalized (4π normalized), so the
    recurrence follows that normalization convention directly.
    """
    x = float(np.clip(sin_lat, -1.0, 1.0))
    cos_lat = math.sqrt(max(0.0, 1.0 - x * x))
    a_nm, b_nm, prev_nm = _legendre_recurrence_coeffs(max_degree)

    p = np.zeros((max_degree + 1, max_degree + 1), dtype=np.float64)
    dp_dphi = np.zeros_like(p)
    p[0, 0] = 1.0

    if max_degree == 0:
        return p, dp_dphi

    diag_scale = np.sqrt(
        (2.0 * np.arange(1, max_degree + 1) + 1.0) / (2.0 * np.arange(1, max_degree + 1))
    )
    for m in range(1, max_degree + 1):
        p[m, m] = diag_scale[m - 1] * cos_lat * p[m - 1, m - 1]

    for n in range(1, max_degree + 1):
        p[n, n - 1] = math.sqrt(2.0 * n + 1.0) * x * p[n - 1, n - 1]
        if n >= 2:
            p[n, : n - 1] = (
                a_nm[n, : n - 1] * x * p[n - 1, : n - 1] - b_nm[n, : n - 1] * p[n - 2, : n - 1]
            )

    denom = x * x - 1.0
    if abs(denom) < 1e-14:
        denom = -1e-14

    for n in range(1, max_degree + 1):
        prev = prev_nm[n, : n + 1].copy()
        prev[n] = 0.0
        dp_dx = (n * x * p[n, : n + 1] - prev * p[n - 1, : n + 1]) / denom
        dp_dphi[n, : n + 1] = cos_lat * dp_dx

    return p, dp_dphi


class LunarHarmonicGravity:
    """
    Degree/order-truncated GRGM1200A lunar gravity evaluator.

    The default model uses degrees 4..70 from GRGM1200A. Degrees 0..3 are
    intentionally omitted because ``GravityModel`` already contributes the
    point-mass, J2, and J3 terms through its regular body registration path.
    This makes the harmonic model a drop-in replacement for the old mascon
    perturbation without requiring a ``GravityModel`` interface change.
    """

    REFERENCE_RADIUS_M = 1.738_0e6
    GM_M3_S2 = 4.902_800_122_445_3e12
    MAX_PACKAGED_DEGREE = 70

    def __init__(
        self,
        degree: int = 70,
        min_degree: int = 4,
        active_altitude_m: float = 2_000_000.0,
    ) -> None:
        if degree < 2 or degree > self.MAX_PACKAGED_DEGREE:
            raise ValueError(f"degree must be in [2, {self.MAX_PACKAGED_DEGREE}], got {degree}")
        if min_degree < 2 or min_degree > degree:
            raise ValueError(f"min_degree must be in [2, degree], got {min_degree}")

        c_all, s_all = _load_grgm1200a_degree70()
        self.degree = degree
        self.min_degree = min_degree
        self.active_altitude_m = float(active_altitude_m)
        self._c = c_all[: degree + 1, : degree + 1].copy()
        self._s = s_all[: degree + 1, : degree + 1].copy()
        if min_degree > 4:
            self._c[4:min_degree, :] = 0.0
            self._s[4:min_degree, :] = 0.0
        self._orders = np.arange(degree + 1, dtype=np.float64)

    def acceleration_body(
        self,
        spacecraft_pos_body: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Residual lunar acceleration in the Moon body frame [m/s²]."""
        x, y, z = map(float, spacecraft_pos_body)
        r = math.sqrt(x * x + y * y + z * z)
        if r <= self.REFERENCE_RADIUS_M:
            return np.zeros(3, dtype=np.float64)

        altitude_m = r - self.REFERENCE_RADIUS_M
        if altitude_m > self.active_altitude_m:
            return np.zeros(3, dtype=np.float64)

        rho_xy = math.hypot(x, y)
        cos_lat = rho_xy / r
        sin_lat = z / r
        lon = math.atan2(y, x)

        cos_lon = math.cos(lon)
        sin_lon = math.sin(lon)
        cos_m = np.empty(self.degree + 1, dtype=np.float64)
        sin_m = np.empty(self.degree + 1, dtype=np.float64)
        cos_m[0] = 1.0
        sin_m[0] = 0.0
        for m in range(1, self.degree + 1):
            cos_m[m] = cos_m[m - 1] * cos_lon - sin_m[m - 1] * sin_lon
            sin_m[m] = sin_m[m - 1] * cos_lon + cos_m[m - 1] * sin_lon

        p, dp_dphi = _normalized_legendre(sin_lat, self.degree)

        radial_sum = 0.0
        lat_sum = 0.0
        lon_sum = 0.0
        rho = self.REFERENCE_RADIUS_M / r
        rho_n = rho**self.min_degree

        for n in range(self.min_degree, self.degree + 1):
            orders = slice(0, n + 1)
            common = self._c[n, orders] * cos_m[orders] + self._s[n, orders] * sin_m[orders]
            common_lon = self._s[n, orders] * cos_m[orders] - self._c[n, orders] * sin_m[orders]
            pn = p[n, orders]
            radial_sum += (n + 1.0) * rho_n * np.dot(pn, common)
            lat_sum += rho_n * np.dot(dp_dphi[n, orders], common)
            lon_sum += rho_n * np.dot(self._orders[orders] * pn, common_lon)
            rho_n *= rho

        scale = self.GM_M3_S2 / (r * r)
        a_r = -scale * radial_sum
        a_phi = scale * lat_sum
        denom_lon = max(cos_lat, 1e-12)
        a_lon = scale * lon_sum / denom_lon

        e_r = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat], dtype=np.float64)
        e_phi = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat], dtype=np.float64)
        e_lon = np.array([-sin_lon, cos_lon, 0.0], dtype=np.float64)
        return a_r * e_r + a_phi * e_phi + a_lon * e_lon

    def __call__(
        self,
        spacecraft_pos_eci: NDArray[np.float64],
        moon_pos_eci: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Residual lunar acceleration in ECI [m/s²]."""
        rotation = _body_to_eci(moon_pos_eci)
        rel_body = rotation.T @ (spacecraft_pos_eci - moon_pos_eci)
        acc_body = self.acceleration_body(rel_body)
        return rotation @ acc_body


@lru_cache(maxsize=1)
def default_lunar_harmonic_gravity() -> LunarHarmonicGravity:
    """Shared degree-70 GRGM1200A residual model for ``set_mascon_model``."""
    return LunarHarmonicGravity()


def lunar_harmonic_acceleration(
    spacecraft_pos_eci: NDArray[np.float64],
    moon_pos_eci: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convenience function matching the existing mascon callable signature."""
    return default_lunar_harmonic_gravity()(spacecraft_pos_eci, moon_pos_eci)
