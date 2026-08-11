# Changelog

All notable changes to cislunar-sim will be documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- `ruff` and `pyright` are now version-constrained across the `dev` extra, pre-commit, and CI. A floating `ruff>=0.4` had let ruff 0.16 — which began formatting Python blocks inside Markdown — fail CI on unrelated pull requests. The `dev` extra allows `ruff~=0.15.20`; `pyright` takes an exact pin because it has only ever released 1.1.x, which leaves `~=` unbounded
- `CITATION.cff` now credits Sean O'Hara with an `Arbor Engineering Group` affiliation rather than the organisation alone, so citations name the author; `url` points at the project's landing page instead of duplicating `repository-code`, and the PyPI release is recorded under `identifiers`
- `Homepage` in `[project.urls]` now points at the project's landing page on arboreng.com rather than repeating the `Repository` URL, so the PyPI listing links somewhere the repository does not already cover. Takes effect on the next release

### Fixed

- Lunar mascon rotation matrix (`_body_to_eci`) now explicitly casts to `float64`, fixing a pyright `reportReturnType` failure
- The GP-history archive holds 2,680 records, not the 2,438 previously described as its full size; 2,438 is the number falling inside the two comparison windows. The remaining 242 are 10 records predating the early window, 68 rejected by the B\* plausibility filter, and 164 in an undocumented three-month gap between the windows — the period in which any change in behaviour would appear, whose removal makes a step between the windows look cleaner than the series supports
- The B\* swing analysis described its two windows as "sailing" and "passive", the latter defined as "sail furled". LightSail 2 never furled its sail — it deployed in July 2019 via non-retractable booms and stayed deployed until reentry in November 2022 — so that contrast does not exist. The windows are hardcoded date ranges that inspect no attitude or sail state, and they also differ in solar activity and altitude, both of which move fitted B\* in the same direction as the hypothesised sail effect. The predicted ratio is additionally a single-point calculation at 720 km and F10.7 = 100, compared against an archive running to reentry. The observed and predicted numbers are unchanged; the claim they support is now stated as suggestive rather than confirming, and `VALIDATION_RECORD.md` §2 records what a controlled version would require. The result interpretation string no longer cites confounds only when the ratio falls in the unexpected direction
- The README described cislunar-sim as an alternative to "the XML configuration of GMAT". GMAT is not XML-configured — missions are `.script` files using `Create`/assignment syntax — so the comparison now names the real difference: an importable library versus an application driven by mission scripts. This text is the PyPI long description, so the correction reaches PyPI on the next release

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
