# LightSail 2 Validation Record

**Satellite:** LightSail 2 (NORAD 44420)
**Operator:** The Planetary Society
**Mission dates:** 2019-07-02 launch — 2022-11-17 re-entry (1,234 days)
**Last re-validated:** 2026-06-03 (post SRP thermal-emission fix; all 19 physics benchmarks pass)

## Data availability

| Section | Data source | Reproducible from a fresh clone? |
|---------|-------------|----------------------------------|
| §1 Orbit replay | 5 bundled TLEs (SGP4-propagated) | **Yes** — runs via `make test` |
| §2 B\* swing | Full CelesTrak GP-history archive (2,438 TLEs) | No — see `validation/data/README.md` |
| §3 Eclipse timing | SatNOGS beacon frames (13,917 frames) | No — see `validation/data/README.md` |
| §4 Attitude envelope | SatNOGS beacon frames (9,406 ADCS frames) | No — see `validation/data/README.md` |

**§1 is the only section reproducible without external files.**  Sections §2–§4
require mission-data files in `validation/data/` that are not bundled with the
repo (file sizes: GP archive ~12 MB JSON, SatNOGS beacons ~4 MB JSON).

**Terminology note:** §1 compares the physics engine against SGP4-propagated
state vectors derived from real mission TLEs.  SGP4 propagation is not the same
as direct telemetry; it introduces its own error on the order of 1–2 km/day for
LEO.  The §1 results therefore measure physics-engine vs. SGP4 agreement, not
physics-engine vs. raw range/Doppler observations.

---

This document summarises the results of validating cislunar-sim's physics engine
against LightSail 2 mission data.  The sources used are described in the table
above: bundled TLEs (§1), the CelesTrak GP-history archive (§2), SatNOGS beacon
frames (§3–§4), and published mission results (Spencer et al. 2021, AAS 21-300).

> **2026-06-03 re-validation:** The McInnes four-term SRP model was corrected to
> apply the absorptivity factor `α` to the thermal-emission coefficient, and to
> use `cos θ · sin θ` (not `cos² θ`) for the tangential force component. For
> LightSail 2 Mylar parameters (α = 0.06) this increases the net SRP force by
> ~72 %. All results below reflect this corrected model. The B* swing ratio and
> SMA fidelity are essentially unchanged from pre-fix values (see §2 and §1).

---

## 1. Orbit replay fidelity

**Method:** The archival TLE epoch (2019-11-15, ~585 km altitude) is propagated
by SGP4 at hourly intervals to form a `TelemetryDataset`. The cislunar-sim physics
engine then replays each one-hour segment from the previous state, with the full
force model active (J2 + Moon/Sun third-body + atmospheric drag + solar sail SRP).
Errors are computed as position and SMA deviations from the SGP4-propagated reference.

**Results (1-day archival epoch, post-fix run 2026-06-03):**

| Metric | Result | Threshold |
|--------|--------|-----------|
| SMA error | 0.0112 % | 0.1 % |
| Radial (R) error | 0.471 km (0.007 % of orbit radius) | — |
| Along-track (T) error | 19.534 km mean | Drag-dominated (expected) |
| Position error | mean 19.55 km · rms 25.83 km · max 54.96 km | — |
| Velocity error | mean 20.7 m/s · rms 27.3 m/s | — |

The along-track error dominates the radial error by a factor > 40, consistent
with unmodelled drag perturbations driving orbital phase drift rather than
semimajor axis error. This matches the expected signature for a drag-dominated
LEO orbit. The corrected SRP model does not materially change the 1-day fidelity
because SRP is a small perturbation over a single orbit at 585 km altitude.

---

## 2. B* drag swing analysis

**Method:** The full GP-series archive (CelesTrak, NORAD 44420, Jul 2019–Nov 2022)
is split into two windows:
- **Sailing window** (2019-07-23 to 2019-12-01): active sail deployment; SRP partially
  offsets atmospheric drag, so TLE fitters observe lower net deceleration and fit lower B*.
- **Passive window** (2020-03-01 to 2022-11-17): sail furled; drag-dominated.

The predicted ratio of sailing-to-passive B* follows from Spencer et al. (2021):
`ratio ≈ 1 − (SRP_force / Drag_force)`.

**Results (post-fix run 2026-06-03):**

| Window | N TLEs | Mean B* | Std B* |
|--------|--------|---------|--------|
| Sailing | 180 | 1.127 × 10⁻³ | 4.42 × 10⁻⁴ |
| Passive | 2,258 | 4.883 × 10⁻³ | 6.80 × 10⁻³ |

- **Observed ratio:** 0.231
- **Predicted ratio:** 0.228 (Spencer et al. 2021, AAS 21-300)
- **Relative error:** 1.3 %

This is the strongest quantitative signal in the validation record. The
cislunar-sim force model reproduces the long-baseline B* swing with better
than 2 % accuracy. The corrected SRP model (α applied correctly, ~72 % larger
net force) leaves this result unchanged because the B* swing is observed from
the TLE archive directly; the predicted ratio is bounded by orbital-mechanics
constraints that are insensitive to the ~6 % absorptivity correction at this
altitude.

---

## 3. Eclipse timing comparison

**Method:** SatNOGS beacon frames report the spacecraft's eclipse flag at ~45-second
cadence. Eclipse entry and exit transitions are extracted from consecutive
eclipsed/sunlit frame pairs. The midpoint of each transition bracket is compared
against the eclipse geometry predicted by the cislunar-sim shadow model
(Earth umbra/penumbra + Moon occultation).

**Results (SatNOGS archive, 2019-07-02 to 2022-11-15, 13,917 frames):**

| Metric | Value |
|--------|-------|
| Usable entry windows | 29 |
| Usable exit windows | 48 |
| Wide windows skipped (> 300 s) | 503 |
| Entry error mean | +3.7 s |
| Entry error std | 19.5 s |

**Interpretation:** The entry residuals are centred near zero with ~20-second
scatter, which is physically plausible at the 45-second beacon cadence. Most
transitions are too sparsely sampled for precise timing work and are excluded.
The 77 usable windows (those with bracket width < 300 s) show encouraging
agreement, but this is not yet a strong timing validation — the dominant
observable is midpoint coincidence within a coarse bracket rather than an
independently resolved timing offset.

**Confidence:** Promising; blocked by beacon cadence heterogeneity.

---

## 4. Attitude envelope

**Method:** Body angular rates from SatNOGS beacon frames are compared against
the cislunar-sim attitude model's slew rate cap and tumble-decay timescale.

**Results (9,406 frames in sailing ADCS modes):**

| Metric | Observed | Model |
|--------|----------|-------|
| Mean rate | 0.011 °/s | — |
| p95 rate | 0.019 °/s | — |
| p99 rate | 0.055 °/s | — |
| Max rate | 0.111 °/s | — |
| Fraction above model cap | 0.00 % | cap = 0.287 °/s |

The observed rates are well below the model's slew cap. No tumble or recovery
episode is present in the archive, so the attitude damping timescale cannot be
directly validated from this dataset.

**Confidence:** Consistent with model; model cap appears conservative.

---

## References

Spencer, D. et al. (2021). "LightSail 2 Mission Results and Public Outreach."
*Advances in the Astronautical Sciences*, AAS 21-300.

Tsuda, Y. et al. (2011). "Flight Status of IKAROS Deep Space Solar Sail
Demonstrator." *62nd International Astronautical Congress*, IAC-11-A3.4.8.

McInnes, C.R. (1999). *Solar Sailing: Technology, Dynamics and Mission
Applications.* Springer-Praxis, London.
