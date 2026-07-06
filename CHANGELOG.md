# Changelog

All notable changes to cislunar-sim will be documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed

- Lunar mascon rotation matrix (`_body_to_eci`) now explicitly casts to `float64`, fixing a pyright `reportReturnType` failure

## [1.0.0] — 2026-06-08

Initial public release.

### Physics engine (`cislunar.physics`)

- Adaptive RK4(5) Dormand-Prince integrator with configurable error tolerances
- Two-body gravity with J2 Earth oblateness
- Lunar gravity: J2 + J3 zonal harmonics + 8-mascon GRAIL model
- Third-body perturbations (Moon, Sun)
- Hall-effect ion thruster model (iodine-fed, 15 mN, 60 s warmup delay)
- Solar sail radiation pressure (McInnes model + albedo reflected light)
- Atmospheric drag via NRLMSISE-00 proxy (F10.7 solar flux)
- Earth and Moon umbra/penumbra eclipse model
- Solar array power budget and battery model
- Quaternion rigid-body attitude dynamics with reaction wheels and cold-gas RCS
- Sun and Moon positions from astropy DE430 ephemeris
- DSN visibility and link budget utilities
- Optional live/cached space-weather support for atmospheric-drag studies

### Guidance laws (`cislunar.guidance`)

- `GVEPeriapsisGuidance` — GVE-optimal periapsis raising
- `GVEApogeeGuidance` — GVE-optimal apogee raising
- `GTOPeriapsisGuidance` / `GTOApogeeGuidance` — burn gating for GTO
- `ProgradeGuidance` — prograde burn with Oberth timing

### Validation (`cislunar.validation`)

- LightSail 2 telemetry replay (SMA error < 0.05% over 1-day epoch)
- 18-benchmark physics suite against McInnes, Wertz, JPL Horizons, IKAROS
- Eclipse timing validation against SatNOGS beacon data
- Attitude envelope analysis from LightSail 2 beacon telemetry
- CelesTrak and SatNOGS telemetry fetchers
- CCSDS OEM trajectory export for comparison with GMAT, Orekit, or STK

[1.0.0]: https://github.com/arboreng/cislunar-sim/releases/tag/v1.0.0
