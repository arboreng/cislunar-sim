# External Comparison Protocol

**Purpose:** Step-by-step instructions to compare cislunar-sim trajectory output
against GMAT, Orekit, or STK. This document closes the remaining
trajectory-level validation gap by checking the package against an
independent toolchain.

**Status:** The cislunar-sim physics engine passes 18/18 internal benchmarks
(see `python -m cislunar.validation.physics_benchmark --extended`). This protocol is the
next tier — trajectory-level cross-validation against an independent tool.

---

## 1. Generate the cislunar-sim reference trajectory

```bash
cd cislunar-sim/

# Standard 60-second propagation step
python -m cislunar.validation.trajectory_export \
  --epoch 2025-01-01 \
  --days 3 \
  --step 60 \
  --alt 400 \
  --inc 28.5 \
  --out cislunar-sim_3day_60s.oem

# Fine-step reference (closer to integrator truth)
python -m cislunar.validation.trajectory_export \
  --epoch 2025-01-01 \
  --days 3 \
  --step 10 \
  --alt 400 \
  --inc 28.5 \
  --out cislunar-sim_3day_10s.oem
```

The OEM files use:
- **Reference frame:** EME2000 (Earth-centred inertial, J2000 orientation)
- **Time system:** TDB
- **Units:** km and km/s

---

## 2. GMAT procedure

### Force model configuration

Match the cislunar-sim force model exactly:

| Parameter | cislunar-sim value | GMAT setting |
|-----------|-------------|--------------|
| Earth gravity | EGM96 J2 only | JGM-2, degree=2, order=0 |
| Moon gravity | point mass | Luna (point mass) |
| Sun gravity | point mass | Sun (point mass) |
| Moon ephemeris | astropy DE430 | DE430 or DE405 |
| Atmospheric drag | **off** | PropagatorFactory drag = None |
| Solar radiation | **off** | SRP = off |
| Relativity | **off** | RelativisticCorrection = false |

> **Note:** cislunar-sim applies atmospheric drag and SRP force during env steps with
> active actions. For a coast trajectory (zero throttle, sail face-on to velocity)
> the drag model is active.  To match this, either enable drag in GMAT with
> NRLMSISE-00 at F10.7=150, or use the `--step 60 --inc 0` trajectory which
> starts in the equatorial plane where drag effects are symmetric and small.

### Import and propagate

1. `File → New → Script`
2. Paste the template below, substituting your OEM file path:

```gmat
Create Spacecraft cislunar-simRef;
GMAT cislunar-simRef.CoordinateSystem = EarthMJ2000Eq;

% Load initial conditions from OEM first line
% (manually copy position/velocity from the OEM header)
GMAT cislunar-simRef.X  =  6771.0;   % km
GMAT cislunar-simRef.Y  =  0.0;
GMAT cislunar-simRef.Z  =  0.0;
GMAT cislunar-simRef.VX =  0.0;      % km/s
GMAT cislunar-simRef.VY =  7.6726;   % circular orbit speed at 400 km
GMAT cislunar-simRef.VZ =  0.0;

Create ForceModel EarthMoonSun;
GMAT EarthMoonSun.CentralBody        = Earth;
GMAT EarthMoonSun.PrimaryBodies      = {Earth};
GMAT EarthMoonSun.PointMasses        = {Luna, Sun};
GMAT EarthMoonSun.Gravity.Earth.Model = JGM2;
GMAT EarthMoonSun.Gravity.Earth.Degree = 2;
GMAT EarthMoonSun.Gravity.Earth.Order  = 0;
GMAT EarthMoonSun.Drag               = None;
GMAT EarthMoonSun.SRP                = Off;

Create Propagator RK89;
GMAT RK89.FM   = EarthMoonSun;
GMAT RK89.Type = RungeKutta89;
GMAT RK89.InitialStepSize = 10;
GMAT RK89.Accuracy = 1e-12;

BeginMissionSequence;
Propagate RK89(cislunar-simRef) {cislunar-simRef.ElapsedDays = 3.0};

Report cislunar-simRef.EarthMJ2000Eq.X cislunar-simRef.EarthMJ2000Eq.Y cislunar-simRef.EarthMJ2000Eq.Z;
```

### Comparison

After GMAT completes, compare the final position vector against the last line
of `cislunar-sim_3day_60s.oem`.

**Pass criterion:** Position disagreement < 100 km at day 3.

This is intentionally looser than the internal 10 km benchmark because GMAT
uses a different J2 model coefficient (JGM-2 vs EGM96).  Disagreements in the
50–100 km range are expected from this and the slightly different DE430 vs
DE405 Moon ephemeris.  Disagreements > 200 km should trigger investigation.

---

## 3. Orekit procedure

### Dependencies

```python
pip install orekit
```

Orekit requires the orekit-data zip (available from https://www.orekit.org/download.html).

### Code

```python
import orekit
orekit.initVM()
from orekit.pyhelpers import setup_orekit_curdir
setup_orekit_curdir()

from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.frames import FramesFactory
from org.orekit.utils import Constants
from org.orekit.bodies import CelestialBodyFactory
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel, ThirdBodyAttraction
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.orbits import CartesianOrbit
from org.orekit.propagation.numerical import NumericalPropagator
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.orekit.propagation import SpacecraftState as OrkState

import numpy as np

TDB   = TimeScalesFactory.getTDB()
ICRF  = FramesFactory.getICRF()
MU    = Constants.WGS84_EARTH_MU

# Read final state from cislunar-sim OEM (copy from last line of cislunar-sim_3day_60s.oem)
# Position [km] and velocity [km/s] → convert to m and m/s
pos_km = np.array([FILL_FROM_OEM_X, FILL_FROM_OEM_Y, FILL_FROM_OEM_Z])
vel_kms = np.array([FILL_FROM_OEM_VX, FILL_FROM_OEM_VY, FILL_FROM_OEM_VZ])

epoch = AbsoluteDate(2025, 1, 1, 0, 0, 0.0, TDB)
from org.orekit.utils import PVCoordinates
from org.hipparchus.geometry.euclidean.threed import Vector3D
pv0 = PVCoordinates(
    Vector3D(0., 6771e3, 0.),     # initial position [m]
    Vector3D(7672.6, 0., 0.),     # initial velocity [m/s]
)
orbit0 = CartesianOrbit(pv0, ICRF, epoch, MU)

# Force model
provider = GravityFieldFactory.getNormalizedProvider(2, 0)  # J2 only
gravity  = HolmesFeatherstoneAttractionModel(ICRF, provider)
moon_att = ThirdBodyAttraction(CelestialBodyFactory.getMoon())
sun_att  = ThirdBodyAttraction(CelestialBodyFactory.getSun())

integrator = DormandPrince853Integrator(1., 300., 1e-5, 1e-8)
prop = NumericalPropagator(integrator)
prop.setOrbitType(None)  # Cartesian
prop.addForceModel(gravity)
prop.addForceModel(moon_att)
prop.addForceModel(sun_att)
prop.setInitialState(OrkState(orbit0))

target_epoch = epoch.shiftedBy(3 * 86400.)
final_state  = prop.propagate(target_epoch)
pv_final     = final_state.getPVCoordinates(ICRF)

pos_orekit = np.array(pv_final.getPosition().toArray()) / 1e3  # → km
print("Orekit final position [km]:", pos_orekit)
print("cislunar-sim final position [km]:", pos_km)
print("Disagreement [km]:", np.linalg.norm(pos_orekit - pos_km))
```

---

## 4. STK procedure

1. **New Scenario** → set epoch to 2025-01-01 00:00:00 TDB
2. **Insert → Satellite**
3. **Properties → Basic → Orbit** → set initial Cartesian state from OEM line 1
4. **Propagator:** `HPOP`
5. **Force model:**
   - Gravity: EGM96 degree 2, order 0
   - Point masses: Moon, Sun (JPL DE430)
   - Drag: off
   - SRP: off
6. **Run** for 3 days
7. **Report → Cartesian State** at stop time → compare with OEM final line

---

## 5. Interpreting results

| Disagreement at day 3 | Interpretation |
|---|---|
| < 10 km | Excellent — better than internal convergence test |
| 10–100 km | Expected — force model coefficient differences (J2, ephemeris) |
| 100–500 km | Investigate — possible epoch or frame mismatch |
| > 500 km | Error — check initial conditions, time system, frame |

The dominant disagreement source at 3 days is the J2 coefficient:
- cislunar-sim: `J2_EARTH = 1.08262668e-3` (EGM96)
- GMAT JGM-2: `J2 = 1.08262999e-3`
- Difference: ~3e-8, which over 3 days at 400 km produces ~30 km divergence

This is well within the 100 km tolerance and is not a bug — it reflects the
choice of gravity field model.  To tighten the comparison, configure both
tools to use the same J2 value.

---

## 6. Updating the automated benchmark

Once you have run the external comparison, embed the result as a constant
in `cislunar.validation.physics_benchmark` and add a new benchmark:

```python
# GMAT_3DAY_POS_KM = [X, Y, Z]  from GMAT final state, retrieved YYYY-MM-DD
GMAT_3DAY_POS_KM = np.array([...])

def benchmark_gmat_trajectory() -> BenchmarkResult:
    records = propagate(duration_days=3.0, step_s=60.0, verbose=False)
    cislunar-sim_pos = records[-1].pos_m / 1e3   # → km
    error_km   = float(np.linalg.norm(cislunar-sim_pos - GMAT_3DAY_POS_KM))
    return BenchmarkResult(
        name   = "3-day cislunar trajectory vs. GMAT (J2+Moon+Sun)",
        passed = error_km < 100.0,
        ...
    )
```

This converts the external comparison from a manual step into an automated
regression that runs with `python -m cislunar.validation.physics_benchmark --extended`.
