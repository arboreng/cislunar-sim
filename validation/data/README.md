# validation/data/

Canonical repo location for external mission data used by the LightSail 2
validation flow. These files are not committed to the repository because of
size and licensing constraints.

If you are running from a source checkout, the validation loaders look here by
default. You can override the directory with
`CISLUNAR_VALIDATION_DATA_DIR=/path/to/data`.

If you already have local files under the legacy path
`src/cislunar/validation/data/`, move them into this directory.

## LightSail 2 historical GP series

**File:** `lightsail2_gp_history.json`
**Source:** CelesTrak NORAD archive request
**Request URL:** https://celestrak.org/NORAD/archives/request.php?FORMAT=json

Request parameters:
- NORAD catalogue ID: `44420`
- Date range: `2019-07-01` to `2022-11-30`
- Format: `JSON` (CCSDS OMM)

The archive is emailed within 2 hours of the request. Place the received file
here as `lightsail2_gp_history.json`.

### What this enables

`load_historical_tle_series()` in `cislunar.validation.telemetry_fetcher`
reads this file and builds a `TelemetryDataset` with one point per
radar-tracking epoch (typically ~3,000–9,000 points over the 3.5-year
mission).

Each point is propagated only at the TLE's own epoch, keeping SGP4 error well
below 1 km compared to km-scale drift when a single TLE is propagated 30 days.

### Tests

Section 9 of `tests/test_validation.py` covers the loader and the sail-raising
signature. These tests skip automatically when the file is absent.

---

## LightSail 2 beacon telemetry (SatNOGS)

**File:** `lightsail2_beacons_p1.json`
**Source:** SatNOGS network (free account required)
**Satellite ID:** `44420`

### Account setup

1. Create a free account at https://db.satnogs.org/
2. Go to **Account → API Auth Token** to retrieve your token.

### Download

The repo does not currently ship a dedicated SatNOGS downloader script.
Acquire the paginated API responses directly, then merge the pages into
`lightsail2_beacons_p1.json`.

Example first page:

```bash
curl -H "Authorization: Token <YOUR_TOKEN>" \
  "https://db.satnogs.org/api/telemetry/?satellite=44420&limit=1000&format=json" \
  > validation/data/lightsail2_beacons_p1.json
```

If the full history spans multiple pages, save each page into
`validation/data/raw_pages/` and merge them locally:

```python
import glob
import json

frames = []
for path in sorted(glob.glob("validation/data/raw_pages/*.json")):
    payload = json.load(open(path))
    frames.extend(payload["results"] if isinstance(payload, dict) else payload)

json.dump(frames, open("validation/data/lightsail2_beacons_p1.json", "w"))
```

The beacon loader (`cislunar.validation.beacon_parser.load_satnogs_frames`)
accepts either a merged list of records or a paginated response object with a
top-level `results` list.

### What this enables

`load_satnogs_frames()` reads this file and returns a time-sorted list of
`BeaconFrame` objects containing:
- Eclipse flag and sail-deployed flag per frame
- Body attitude quaternion and angular rates
- Battery voltage, current, and power
- Solar face currents (nX/pX/nY/pY/nZ) as an illumination proxy during penumbra

The frames feed into:
1. `eclipse_validator.py` for predicted vs. observed eclipse timing comparison
2. `attitude_envelope.py` for rate distribution, slew statistics, and tumble damping

### Tests

Section 10 of `tests/test_validation.py` covers the eclipse timing comparison.
Tests that require beacon data skip automatically when this file is absent.
Tests that only need synthetic beacon frames always run.

---

## Repo command

Once both files are present, run the full external-data validation flow with:

```bash
make validate-external
```

## Files at a glance

| File | Source | Required by |
|------|--------|-------------|
| `lightsail2_gp_history.json` | CelesTrak email archive | `telemetry_fetcher.py`, `eclipse_validator.py` |
| `lightsail2_beacons_p1.json` | SatNOGS API | `beacon_parser.py`, `eclipse_validator.py` |

The CI fixture (`lightsail2_sail_arc_fixture.json`) is committed at
`tests/fixtures/` rather than here.
