# cislunar-sim

**High-fidelity cislunar spacecraft simulation in pure Python — cross-validated against LightSail 2 mission data.**

[![Tests](https://github.com/arboreng/cislunar-sim/actions/workflows/tests.yml/badge.svg)](https://github.com/arboreng/cislunar-sim/actions/workflows/tests.yml)
[![PyPI version](https://img.shields.io/pypi/v/cislunar-sim.svg)](https://pypi.org/project/cislunar-sim/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

`cislunar-sim` is a Python library for simulating spacecraft trajectories from low Earth orbit through cislunar space. It provides a high-fidelity, validated physics engine and Q-law orbital guidance laws as a library you import — not an application you drive with mission scripts, like GMAT, or a JVM you host from Python, like Orekit.

## Why cislunar-sim?

- **Validated**: physics engine achieves < 0.05% SMA error against SGP4-propagated LightSail 2 states (bundled TLEs, reproducible from a fresh clone)
- **Complete force environment**: J2/J3 + GRAIL mascon lunar gravity, ion thruster, solar sail SRP, atmospheric drag, eclipse shadowing, attitude dynamics
- **JPL-quality ephemeris**: Sun and Moon positions from astropy DE430
- **Minimal dependencies**: `numpy`, `astropy`, and `nrlmsise00` are the only runtime dependencies
- **Fast**: adaptive RK4(5) integrator; 7-day GTO periapsis-raise and 5-day drag-decay demos each run in under 60 seconds

## Installation

```bash
pip install cislunar-sim
```

**Requirements:** Python ≥ 3.11, numpy ≥ 1.24, astropy ≥ 5.3, nrlmsise00 ≥ 0.1

For TLE propagation and LightSail 2 telemetry replay, install the `validation` extra:

```bash
pip install "cislunar-sim[validation]"  # adds sgp4>=2.22
```

## Quick start

```python
import numpy as np
from cislunar.guidance import ProgradeGuidance
from cislunar.physics import Action, make_lunar_craft

R_E = 6.371e6
MU = 3.986004418e14
r = R_E + 400e3
v_c = np.sqrt(MU / r)

sc = make_lunar_craft(
    position_m=[r, 0.0, 0.0],
    velocity_ms=[0.0, v_c, 0.0],
    propellant_kg=0.5,
)
guidance = ProgradeGuidance()

for _ in range(60):  # 1 hour at 60 s control steps
    thrust_dir, throttle = guidance.steer(sc.state.position, sc.state.velocity)
    sc.step(
        Action(attitude_dir_cmd=thrust_dir, thrust_dir=thrust_dir, throttle=throttle),
        dt_requested=60.0,
    )

print(sc.state.time_s, np.linalg.norm(sc.state.position))
```

## Examples

The tested feature demos live in [`examples/`](examples/):

- [`examples/gto_periapsis_raise.py`](examples/gto_periapsis_raise.py) — ion-thruster GTO periapsis raise to a 1,000 km target
- [`examples/solar_sail_leo.py`](examples/solar_sail_leo.py) — solar-sail orbit raising from 585 km LEO
- [`examples/drag_decay_leo.py`](examples/drag_decay_leo.py) — atmospheric drag driven SMA decay at 400 km
- [`examples/eclipse_power_gating.py`](examples/eclipse_power_gating.py) — eclipse-aware power gating for ion thrust
- [`examples/cislunar_transfer.py`](examples/cislunar_transfer.py) — combined ion + sail cislunar orbit raising

## What's included

### Physics engine (`cislunar.physics`)

Adaptive RK4(5) Dormand-Prince integrator with:

| Force | Model |
|---|---|
| Earth gravity | Two-body + J2 oblateness |
| Lunar gravity | J2 + J3 + 8-mascon GRAIL model |
| Solar gravity | Third-body perturbation |
| Ion thruster | Hall-effect, 15 mN, iodine-fed, 60 s warmup |
| Solar sail | McInnes radiation pressure model with albedo |
| Atmospheric drag | NRLMSISE-00 proxy (F10.7 solar flux) |
| Eclipse | Earth/Moon umbra–penumbra; power/battery model |
| Attitude | Quaternion rigid-body + reaction wheels + RCS |
| Ephemeris | astropy DE430 (Sun + Moon positions) |

### Guidance laws (`cislunar.guidance`)

| Class | Description |
|---|---|
| `GVEPeriapsisGuidance` | GVE-optimal: maximise dr_p/dt |
| `GVEApogeeGuidance` | GVE-optimal: maximise dr_apo/dt |
| `GTOPeriapsisGuidance` | Periapsis burn gating for GTO |
| `GTOApogeeGuidance` | Apogee burn gating for GTO |
| `ProgradeGuidance` | Prograde burn with Oberth timing |

### Validation (`cislunar.validation`)

```bash
# Spot-check against McInnes, Wertz, JPL Horizons (19 benchmarks)
python -m cislunar.validation.physics_benchmark --extended

# Full LightSail 2 replay / telemetry validation
# Requires external mission files in validation/data/
make validate-external
```

## Validation record

### Reproducible from a fresh clone

The orbit-replay test compares the physics engine against SGP4-propagated
state vectors derived from bundled LightSail 2 TLEs (5 historical epochs,
NORAD 44420).  These run automatically via `make test`.

| Metric | Result | Reference | Reproducible |
|---|---|---|---|
| SMA error (1-day archival epoch) | < 0.05% | SGP4 propagation | ✓ from clone |
| Radial error | < 0.02% of orbital radius | SGP4 propagation | ✓ from clone |
| Sail force vs. McInnes (IKAROS params) | within 1% | McInnes (1999) Table 2.1 | ✓ from clone |

The along-track drift is drag-dominated and expected; see the validation record for details.

### Requires external mission files

The B\* swing analysis and eclipse/attitude comparisons use the full CelesTrak
GP-history archive and SatNOGS beacon frames, neither of which is bundled in
this repo.

| Metric | Result | Requires |
|---|---|---|
| B\* swing ratio (early / late window) | 0.231 observed vs 0.228 predicted — **confounded, see note** | CelesTrak GP archive |
| Eclipse timing (entry residuals) | mean +3.7 s, std 19.5 s | SatNOGS beacon data |
| Body-rate envelope | 0% of frames above model cap | SatNOGS beacon data |

> **On the B\* swing:** the two windows differ in solar activity and altitude as
> well as sail behaviour, and the sail was never furled — it deployed in July 2019
> and stayed deployed until reentry. The agreement is suggestive, not confirming.
> [VALIDATION_RECORD.md §2](src/cislunar/validation/light_sail_2/VALIDATION_RECORD.md)
> explains what would be needed to make it a controlled test.

See [`src/cislunar/validation/light_sail_2/VALIDATION_RECORD.md`](src/cislunar/validation/light_sail_2/VALIDATION_RECORD.md)
for full methodology and [`validation/data/README.md`](validation/data/README.md)
for acquisition instructions.

```bash
make validate-external  # requires validation/data/ mission files (see that README)
```

## Development setup

```bash
git clone https://github.com/arboreng/cislunar-sim.git
cd cislunar-sim
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
make test          # physics + validation + example tests (~5 min)
make validate-external  # requires validation/data/* mission files
make bench         # spot-check benchmarks
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

See [LICENSE](LICENSE). Copyright © 2026 Arbor Engineering Group.
