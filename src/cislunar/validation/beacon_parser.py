"""
LightSail 2 beacon packet parser.

Decodes the 228-byte UDP beacon payload transmitted by LightSail 2 on
437.025 MHz using AX.25 framing. Optionally loads SatNOGS telemetry JSON.

The payload layout implemented here follows the official LightSail 2 beacon
definition published by The Planetary Society (`ls2-beacon-info-v01.txt`).
The packet is AX.25 + IPv4 + UDP + a packed `LSB_BeaconData` payload:

  16 bytes  AX.25 header
  20 bytes  IPv4 header
   8 bytes  UDP header
 228 bytes  LightSail 2 beacon payload
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime

import numpy as np
from numpy.typing import NDArray

_PREAMBLE_BYTES = 44
_PAYLOAD_BYTES = 228
_FACE_NAMES = ("nX", "pX", "nY", "pY", "nZ")

_BATT_FLAG_ECLIPSE = 0x08
_BATT_FLAG_SAIL = 0x04


def _decode_packed2x12(b3: bytes) -> tuple[int, int]:
    """Decode two 12-bit signed integers packed into 3 bytes."""
    raw_a = (b3[0] << 4) | (b3[2] >> 4)
    raw_b = (b3[1] << 4) | (b3[2] & 0x0F)
    a = raw_a - 4096 if raw_a >= 2048 else raw_a
    b = raw_b - 4096 if raw_b >= 2048 else raw_b
    return a, b


def _decode_packed2x12_triple(b9: bytes) -> tuple[int, int, int, int, int, int]:
    """Decode 9 bytes (3 × PackedSigned2x12) into six 12-bit signed values."""
    a0, a1 = _decode_packed2x12(b9[0:3])
    b0, b1 = _decode_packed2x12(b9[3:6])
    c0, c1 = _decode_packed2x12(b9[6:9])
    return a0, a1, b0, b1, c0, c1


@dataclass
class BeaconFrame:
    """Key fields extracted from a single LightSail 2 beacon packet."""

    timestamp_unix_s: float
    in_eclipse: bool
    sail_deployed: bool
    adcs_mode: int
    quaternion: NDArray
    body_rate_deg_s: NDArray
    battery_voltage_v: float
    battery_current_a: float
    batt_power_mw: float
    solar_face_mw: dict[str, float]


def _signed_u8(value: int) -> int:
    return value - 256 if value >= 128 else value


def _parse_iso_timestamp(value: object) -> float | None:
    """Parse an ISO-8601 timestamp string to Unix seconds."""
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.astimezone(UTC).timestamp()


def parse_beacon_bytes(payload: bytes) -> BeaconFrame:
    """
    Parse a 228-byte LightSail 2 beacon UDP payload into a BeaconFrame.
    """
    if len(payload) < _PAYLOAD_BYTES:
        raise ValueError(f"payload too short: {len(payload)} < {_PAYLOAD_BYTES} bytes")

    buf = memoryview(payload[:_PAYLOAD_BYTES])
    i = 0

    def u8() -> int:
        nonlocal i
        v = buf[i]
        i += 1
        return int(v)

    def s8() -> int:
        return _signed_u8(u8())

    def be_u16() -> int:
        nonlocal i
        v = int.from_bytes(buf[i : i + 2], "big", signed=False)
        i += 2
        return v

    def be_i16() -> int:
        nonlocal i
        v = int.from_bytes(buf[i : i + 2], "big", signed=True)
        i += 2
        return v

    def be_u32() -> int:
        nonlocal i
        v = int.from_bytes(buf[i : i + 4], "big", signed=False)
        i += 4
        return v

    beacon_type = u8()
    if beacon_type != 1:
        raise ValueError(f"unexpected LightSail 2 beacon type: {beacon_type}")

    # Temperatures (0 == -75 C, LSB = 0.5 C). We only need panel ordering here.
    _daughter_a_tmp = u8()
    _daughter_b_tmp = u8()
    _threev_pl_tmp = u8()
    _rf_amp_tmp = u8()
    _nx_tmp = u8()
    _px_tmp = u8()
    _ny_tmp = u8()
    _py_tmp = u8()
    _nz_tmp = u8()
    _pz_tmp = u8()

    # Power sensors.
    _atmel_curr = u8()
    _atmel_volt = u8()
    _threev_curr = u8()
    _threev_volt = u8()
    _threev_pl_curr = u8()
    _threev_pl_volt = u8()
    _fivev_pl_curr = u8()
    _fivev_pl_volt = u8()
    _daughter_a_curr = u8()
    _daughter_a_volt = u8()
    _daughter_b_curr = u8()
    _daughter_b_volt = u8()

    nx_int_curr_raw = u8()
    nx_int_volt_raw = u8()
    nx_ext_curr_raw = u8()
    nx_ext_volt_raw = u8()
    px_int_curr_raw = u8()
    px_int_volt_raw = u8()
    px_ext_curr_raw = u8()
    px_ext_volt_raw = u8()
    ny_int_curr_raw = u8()
    ny_int_volt_raw = u8()
    ny_ext_curr_raw = u8()
    ny_ext_volt_raw = u8()
    py_int_curr_raw = u8()
    py_int_volt_raw = u8()
    py_ext_curr_raw = u8()
    py_ext_volt_raw = u8()
    nz_ext_curr_raw = u8()
    nz_ext_volt_raw = u8()

    # Software telemetry.
    _user_cpu = be_u32()
    _sys_cpu = be_u32()
    _idle_cpu = be_u32()
    _processes = be_u32()
    _mem_free = be_u32()
    _buffers = be_u32()
    _cached = be_u32()
    _data_free = be_u32()
    _nand_erasures = be_u32()
    _beacon_count = be_u16()
    timestamp_unix_s = float(be_u32())
    _boot_time = be_u32()
    _long_dur_counter = be_u16()

    # Communications telemetry.
    _rx_count = be_u16()
    _tx_count = be_u16()
    _rx_bytes = be_u32()
    _tx_bytes = be_u32()

    batt_voltages: list[float] = []
    batt_currents: list[float] = []
    for _ in range(8):
        batt_currents.append(s8() / 128.0)
        batt_voltages.append(u8() / 32.0)
        _temp = u8()
        _flags = u8()
        _ctl_flags = u8()

    batt_pwr_mw = float(be_i16())
    adcs_mode = u8()
    flags = u8()

    quaternion = np.array(
        [be_i16() / 128.0 for _ in range(4)],
        dtype=float,
    )
    body_rate_deg_s = np.array(
        [be_i16() / 128.0 for _ in range(3)],
        dtype=float,
    )

    _gyro = bytes(buf[i : i + 9])
    i += 9
    _decode_packed2x12_triple(_gyro)

    _sol_nxx = be_u16()
    _sol_nxy = be_u16()
    _sol_nyx = be_u16()
    _sol_nyy = be_u16()
    _sol_nzx = be_u16()
    _sol_nzy = be_u16()
    _sol_pxx = be_u16()
    _sol_pxy = be_u16()
    _sol_pyx = be_u16()
    _sol_pyy = be_u16()

    _mag = bytes(buf[i : i + 15])
    i += 15
    _wheel_rpm = be_i16()

    # Camera state (2 cameras × 4 bytes).
    i += 8

    # Torquers + motor power bytes.
    i += 8

    # Panel/PIC/motor counters.
    i += 7

    assert i == _PAYLOAD_BYTES, f"decoder consumed {i} bytes, expected {_PAYLOAD_BYTES}"

    def solar_power_mw(curr_raw: int, volt_raw: int) -> float:
        curr_a = _signed_u8(curr_raw) / 64.0
        volt_v = volt_raw / 32.0
        return curr_a * volt_v * 1000.0

    solar_face_mw = {
        "nX": solar_power_mw(nx_int_curr_raw, nx_int_volt_raw)
        + solar_power_mw(nx_ext_curr_raw, nx_ext_volt_raw),
        "pX": solar_power_mw(px_int_curr_raw, px_int_volt_raw)
        + solar_power_mw(px_ext_curr_raw, px_ext_volt_raw),
        "nY": solar_power_mw(ny_int_curr_raw, ny_int_volt_raw)
        + solar_power_mw(ny_ext_curr_raw, ny_ext_volt_raw),
        "pY": solar_power_mw(py_int_curr_raw, py_int_volt_raw)
        + solar_power_mw(py_ext_curr_raw, py_ext_volt_raw),
        "nZ": solar_power_mw(nz_ext_curr_raw, nz_ext_volt_raw),
    }

    return BeaconFrame(
        timestamp_unix_s=timestamp_unix_s,
        in_eclipse=bool(flags & _BATT_FLAG_ECLIPSE),
        sail_deployed=bool(flags & _BATT_FLAG_SAIL),
        adcs_mode=adcs_mode,
        quaternion=quaternion,
        body_rate_deg_s=body_rate_deg_s,
        battery_voltage_v=float(np.mean(batt_voltages)),
        battery_current_a=float(np.mean(batt_currents)),
        batt_power_mw=batt_pwr_mw,
        solar_face_mw=solar_face_mw,
    )


def parse_beacon_hex(packet_hex: str) -> BeaconFrame:
    """Parse a full AX.25/IP/UDP/beacon hex string."""
    data = bytes.fromhex(packet_hex.replace(" ", "").upper())
    payload = data[_PREAMBLE_BYTES:]
    return parse_beacon_bytes(payload)


def load_satnogs_frames(json_path: str) -> list[BeaconFrame]:
    """
    Load BeaconFrames from a SatNOGS telemetry JSON export.

    Supported formats:
      - paginated API wrapper: top-level object with a "results" list
      - raw-hex: each record has a "frame" key with a hex-encoded packet
      - pre-decoded: each record has field keys directly
    """
    import os

    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"SatNOGS beacon file not found: {json_path}\n"
            "Download with:\n"
            "  curl -H 'Authorization: Token <YOUR_TOKEN>' \\\n"
            "    'https://db.satnogs.org/api/telemetry/?satellite=44420"
            "&limit=1000&format=json' \\\n"
            "    > validation/data/lightsail2_beacons_p1.json\n"
            "Paginate with ?page=2, 3, … until the response is empty."
        )

    with open(json_path) as f:
        records = json.load(f)

    if isinstance(records, dict):
        results = records.get("results")
        if isinstance(results, list):
            records = results
        else:
            raise ValueError(
                "SatNOGS beacon file must be a list of records or a "
                "paginated response object with a 'results' list."
            )
    elif not isinstance(records, list):
        raise ValueError(
            f"SatNOGS beacon file has unsupported top-level type: {type(records).__name__}"
        )

    frames: list[BeaconFrame] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        if "frame" in rec:
            try:
                frame = parse_beacon_hex(rec["frame"])
            except Exception:
                continue
            satnogs_ts = _parse_iso_timestamp(rec.get("timestamp"))
            if satnogs_ts is not None:
                frame = replace(frame, timestamp_unix_s=satnogs_ts)
            frames.append(frame)
        else:
            frames.append(_decode_kaitai_record(rec))

    frames.sort(key=lambda f: f.timestamp_unix_s)
    return frames


def _decode_kaitai_record(rec: dict) -> BeaconFrame:
    """Map a decoded record to BeaconFrame."""

    def _get(*keys: str, default: object = 0.0) -> object:
        for key in keys:
            if key in rec:
                return rec[key]
        return default

    def _get_int(*keys: str, default: int = 0) -> int:
        value = _get(*keys, default=default)
        return int(value) if isinstance(value, (int, float, str)) else default

    def _get_float(*keys: str, default: float = 0.0) -> float:
        value = _get(*keys, default=default)
        return float(value) if isinstance(value, (int, float, str)) else default

    flags = _get_int("flags", "flag_byte", default=0)
    adcs_mode = _get_int("adcs_mode", "attitude_mode", default=0)

    q = [_get_float(f"q{i}", f"quaternion_{i}", default=0.0) for i in range(4)]
    r = [_get_float(k, default=0.0) for k in ("rate_x", "rate_y", "rate_z")]

    solar: dict[str, float] = {}
    for name in _FACE_NAMES:
        solar[name] = _get_float(f"solar_{name}_mw", f"face_{name}_mw", default=0.0)

    satnogs_ts = _parse_iso_timestamp(_get("timestamp", default=""))
    raw_ts = _get_float("sys_time", "timestamp", default=0.0)

    return BeaconFrame(
        timestamp_unix_s=satnogs_ts if satnogs_ts is not None else raw_ts,
        in_eclipse=bool(flags & _BATT_FLAG_ECLIPSE),
        sail_deployed=bool(flags & _BATT_FLAG_SAIL),
        adcs_mode=adcs_mode,
        quaternion=np.array(q, dtype=float),
        body_rate_deg_s=np.array(r, dtype=float),
        battery_voltage_v=_get_float("batt_voltage_mean", "battery_voltage_v", default=0.0),
        battery_current_a=_get_float("batt_current_mean", "battery_current_a", default=0.0),
        batt_power_mw=_get_float("batt_pwr_draw", "batt_power_mw", default=0.0),
        solar_face_mw=solar,
    )
