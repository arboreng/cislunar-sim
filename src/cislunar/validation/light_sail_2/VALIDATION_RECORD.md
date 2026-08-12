# LightSail 2 Validation Record

**Satellite:** LightSail 2 (NORAD 44420)
**Operator:** The Planetary Society
**Mission dates:** 2019-07-02 launch — 2022-11-17 re-entry (1,234 days)
**Last re-validated:** 2026-06-03 (post SRP thermal-emission fix; all 19 physics benchmarks pass)

## Data availability

| Section | Data source | Reproducible from a fresh clone? |
|---------|-------------|----------------------------------|
| §1 Orbit replay | 5 bundled TLEs (SGP4-propagated) | **Yes** — runs via `make test` |
| §2 B\* swing | Full CelesTrak GP-history archive (2,680 TLEs) | No — see `validation/data/README.md` |
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

**Results (1-day archival epoch, re-run 2026-08-11):**

| Metric | Result | Threshold |
|--------|--------|-----------|
| SMA error | 0.0113 % | 0.1 % |
| Radial (R) error | 0.449 km (0.006 % of orbit radius) | — |
| Along-track (T) error | 19.559 km mean | Drag-dominated (expected) |
| Position error | mean 19.58 km · rms 25.97 km · max 54.84 km | — |
| Velocity error | mean 21.3 m/s · rms 28.3 m/s | — |

Figures move slightly between runs as dependency versions change. The 2026-06-03
run reported SMA 0.0112 %, radial 0.471 km, along-track 19.534 km; the radial
term has since moved by ~5 % and the velocity terms by ~3 %, with the rest inside
1 %. Quote these to the precision the re-run date supports, not beyond it.

The along-track error dominates the radial error by a factor > 40, consistent
with unmodelled drag perturbations driving orbital phase drift rather than
semimajor axis error. This matches the expected signature for a drag-dominated
LEO orbit. The corrected SRP model does not materially change the 1-day fidelity
because SRP is a small perturbation over a single orbit at 585 km altitude.

---

## 2. B* drag swing analysis

> **Status: suggestive, not confirming.** The premise of this comparison was
> wrong and the result is confounded. Both are documented below rather than
> removed, because the numbers are real and the failure mode is instructive.
> Do not cite the ~1 % agreement as validation of the force model.

**Method:** The full GP-series archive (CelesTrak, NORAD 44420, Jul 2019–Nov 2022)
is split into two hardcoded date windows and the fitted BSTAR field is averaged
within each:

- **Early window** (2019-07-23 to 2019-12-01), 180 TLEs
- **Late window** (2020-03-01 to 2022-11-17), 2,258 TLEs

The archive holds 2,680 records; 2,438 fall inside one of the two windows. The
remaining 242 are excluded — 10 before the early window opens, 164 in the gap
between the windows, and 68 discarded for reporting a negative B*.

That last filter is worth naming precisely, because it changes the result.
Its stated purpose is to remove implausible values, but the
`abs(bstar) > 1.0` arm never fires on this archive: every exclusion is a
negative coefficient. A negative B* is not noise — it is what a fitter
reports when an object gains orbital energy rather than loses it. Of the 68,
**none fall in the early window**; 61 are in the late window and 7 predate
the early one, so the filter acts on one side of the comparison only.

Recomputing with those records included:

| | Early mean B* | Late mean B* | Observed ratio | Error vs 0.228 |
|---|---|---|---|---|
| As published (`B* >= 0`) | 1.1268e-03 | 4.8829e-03 | 0.2308 | ~1 % |
| Negatives included | 1.1268e-03 | 4.7474e-03 | 0.2373 | ~4 % |

Discarding the negatives raises the late-window mean, lowers the ratio, and
improves agreement with the prediction by close to a factor of four. The
choice itself is defensible — many negative fits are genuinely noise on short
arcs. What was not defensible was leaving it undocumented, describing it as a
plausibility filter when it only ever removes negatives, and never measuring
its effect on the headline number.

Note that these are date ranges only. Nothing in `compute_bstar_swing` inspects
attitude, sail state, or spacecraft mode; the window boundaries are constants.

**Results (post-fix run 2026-06-03):**

| Window | N TLEs | Mean B* | Std B* |
|--------|--------|---------|--------|
| Early | 180 | 1.127 × 10⁻³ | 4.42 × 10⁻⁴ |
| Late | 2,258 | 4.883 × 10⁻³ | 6.80 × 10⁻³ |

- **Observed ratio:** 0.231
- **Predicted ratio:** 0.228, computed here from `(F_drag − F_SRP) / F_drag`
  using the formulation in Spencer et al. (2021), AAS 21-300 — not a figure
  published by that paper
- **Relative error:** ~1 %, and not meaningfully more precise than that. The
  observed ratio is 0.2308 and the prediction 0.2284 before rounding, so the
  error is 1.0–1.3 % depending on where each is rounded. The prediction is built
  from 2-significant-figure constants — ρ, C_D, sail reflectivity, a hardcoded
  40 % orbit-mean efficiency — and never supported two digits. Earlier versions
  of this record quoted 1.3 %, which implied a resolution the calculation cannot
  deliver.

### Why this is not the validation it appears to be

**The sail was never furled.** These windows were originally labelled *sailing*
and *passive*, on the premise that LightSail 2 stowed its sail after the primary
mission. It did not. The sail deployed on 2019-07-23 via tape-measure booms and
remained deployed until reentry on 2022-11-17; the mechanism was not retractable.
What varied was attitude — the spacecraft slewed between face-on and edge-on, a
cadence regularly interrupted by momentum-wheel desaturation. So the physical
contrast the comparison assumed does not exist, and any real effect would be the
weaker one of time-averaged effective sail area.

**The windows are confounded.** They differ in more than sail behaviour:

| | Early window | Late window |
|---|---|---|
| Solar activity | Cycle 25 minimum | Cycle 25 ramp |
| Altitude | near-initial (~720 km) | decaying to reentry |

Fitted B* is not a clean ballistic coefficient. It absorbs density-model error,
and SGP4's atmosphere is a fixed profile not driven by observed F10.7. As true
density outruns that profile — which is what happens across a solar ramp — fitted
B* rises. Altitude decay pushes the same way. All three candidate causes act in
the same direction, and this analysis cannot separate them.

**The window boundaries discard the transition.** The two windows are not
adjacent: 2019-12-01 to 2020-03-01 falls in neither, dropping 164 TLEs, 6 % of the
archive. Nothing in the code or in this record explains the gap. Whatever it was
originally for, its effect is to remove the period in which any change in
behaviour would actually appear, which makes a step between the two windows look
cleaner than the underlying series supports. An undocumented exclusion sitting on
the boundary a comparison depends on is a defect in its own right, independent of
the confounds above.

**The prediction is a single-point calculation.** `predicted_ratio` is evaluated
at one fixed condition — 720 km, F10.7 = 100, ρ = 2.5 × 10⁻¹⁴ kg/m³ — while the
late window runs to reentry, where density is orders of magnitude higher. The
predicted ratio therefore describes conditions holding only at the very start of
the archive it is compared against.

Under those conditions, agreement of ~1 % is better read as coincidence than as
confirmation.

### What would make this a real test

- Restrict both windows to comparable F10.7 bands and comparable altitude shells,
  so the residual difference is attributable to sail behaviour
- Normalise fitted B* against a contemporaneous non-sailing reference object in a
  similar orbit, absorbing the shared density-model error
- Evaluate the predicted ratio at the actual altitude and flux of each window
  rather than at a single representative point

**The prediction does not exercise the library's force model.** `predicted_ratio`
is computed inline in `compute_bstar_swing` from literal constants — ρ, v, C_D,
area, mass, reflectivity, P_SRP — and never calls `SolarSailModel` or any other
part of `cislunar.physics`. Agreement between observed and predicted therefore
says nothing about whether this library's physics is correct; it compares a
hand-written formula against an observation. This is also why the SRP absorptivity
fix (α applied correctly, ~72 % larger net force) leaves both numbers unchanged:
neither is derived from the model that was fixed.

Until then this is an observation in search of a controlled comparison.

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
