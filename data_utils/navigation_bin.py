"""navigation_bin.py: read/write the PufferDrive navigation sidecar (`<map_stem>.navi`).

The sidecar sits next to a 3.0 map `.bin` and carries lane-level navigation:
per goal x lane next-hop, distance-to-goal and left/center/right lane feasibility.
Layout (little-endian), mirrored by `load_navigation_binary` in map_data.h:

    char[8]  magic "PDNAVI01"
    int32    version
    uint32   map_crc32       (zlib CRC-32 of the source map .bin bytes)
    uint64   map_size_bytes
    int32    num_roads, num_lanes, num_goals
    int32    lane_road_idx[num_lanes]
    int32    left_lane[num_lanes], right_lane[num_lanes]
    int32    goal_lane[num_goals]
    float32  goal_x[num_goals], goal_y[num_goals]
    int32    next_lane[num_goals * num_lanes]
    float32  distance_to_goal_m[num_goals * num_lanes]
    float32  feasibility_left_m / center_m / right_m [num_goals * num_lanes]
"""

import struct
import zlib
from pathlib import Path

import numpy as np

NAVI_MAGIC = b"PDNAVI01"
NAVI_VERSION = 1
NAVI_SUFFIX = ".navi"
NAVI_MAX_GOALS = 1024
NAVI_UNREACHABLE_M = 1.0e9
NAVI_FEASIBILITY_CAP_M = 1000.0

_HEADER_FMT = "<8siIQiii"

_PER_GOAL_LANE_FIELDS = (
    ("next_lane", np.int32),
    ("distance_to_goal_m", np.float32),
    ("feasibility_left_m", np.float32),
    ("feasibility_center_m", np.float32),
    ("feasibility_right_m", np.float32),
)


def map_fingerprint(map_path) -> tuple[int, int]:
    data = Path(map_path).read_bytes()
    return zlib.crc32(data) & 0xFFFFFFFF, len(data)


def navigation_path_for(map_path, navigation_dir=None) -> Path:
    map_path = Path(map_path)
    directory = Path(navigation_dir) if navigation_dir else map_path.parent
    return directory / (map_path.stem + NAVI_SUFFIX)


def validate_navigation(navi: dict) -> None:
    num_lanes = navi["num_lanes"]
    num_goals = navi["num_goals"]
    if num_lanes <= 0:
        raise ValueError(f"num_lanes must be positive, got {num_lanes}")
    if not 1 <= num_goals <= NAVI_MAX_GOALS:
        raise ValueError(f"num_goals must be in [1, {NAVI_MAX_GOALS}], got {num_goals}")
    for key in ("lane_road_idx", "left_lane", "right_lane"):
        if navi[key].shape != (num_lanes,):
            raise ValueError(f"{key} must have shape ({num_lanes},)")
    for key in ("goal_lane", "goal_x", "goal_y"):
        if navi[key].shape != (num_goals,):
            raise ValueError(f"{key} must have shape ({num_goals},)")
    for key, _ in _PER_GOAL_LANE_FIELDS:
        if navi[key].shape != (num_goals, num_lanes):
            raise ValueError(f"{key} must have shape ({num_goals}, {num_lanes})")
    if np.any(np.diff(navi["lane_road_idx"]) <= 0) or navi["lane_road_idx"][0] < 0:
        raise ValueError("lane_road_idx must be non-negative and strictly ascending")
    if navi["lane_road_idx"][-1] >= navi["num_roads"]:
        raise ValueError("lane_road_idx out of range")
    for key in ("left_lane", "right_lane", "next_lane"):
        if np.any(navi[key] < -1) or np.any(navi[key] >= num_lanes):
            raise ValueError(f"{key} out of range [-1, {num_lanes})")
    if np.any(navi["goal_lane"] < 0) or np.any(navi["goal_lane"] >= num_lanes):
        raise ValueError("goal_lane out of range")
    for key in (
        "goal_x",
        "goal_y",
        "distance_to_goal_m",
        "feasibility_left_m",
        "feasibility_center_m",
        "feasibility_right_m",
    ):
        if not np.all(np.isfinite(navi[key])):
            raise ValueError(f"{key} contains NaN/Inf")
    if np.any(navi["distance_to_goal_m"] < 0) or np.any(navi["distance_to_goal_m"] > NAVI_UNREACHABLE_M):
        raise ValueError("distance_to_goal_m out of range")
    for key in ("feasibility_left_m", "feasibility_center_m", "feasibility_right_m"):
        if np.any(navi[key] < 0) or np.any(navi[key] > NAVI_FEASIBILITY_CAP_M):
            raise ValueError(f"{key} out of range [0, {NAVI_FEASIBILITY_CAP_M}]")


def write_navigation(navi: dict, path) -> None:
    validate_navigation(navi)
    path = Path(path)
    with open(path, "wb") as f:
        f.write(
            struct.pack(
                _HEADER_FMT,
                NAVI_MAGIC,
                NAVI_VERSION,
                navi["map_crc32"],
                navi["map_size_bytes"],
                navi["num_roads"],
                navi["num_lanes"],
                navi["num_goals"],
            )
        )
        for key in ("lane_road_idx", "left_lane", "right_lane", "goal_lane"):
            f.write(np.ascontiguousarray(navi[key], dtype="<i4").tobytes())
        for key in ("goal_x", "goal_y"):
            f.write(np.ascontiguousarray(navi[key], dtype="<f4").tobytes())
        for key, dtype in _PER_GOAL_LANE_FIELDS:
            f.write(np.ascontiguousarray(navi[key], dtype=np.dtype(dtype).newbyteorder("<")).tobytes())


def read_navigation(path) -> dict:
    raw = Path(path).read_bytes()
    header_size = struct.calcsize(_HEADER_FMT)
    if len(raw) < header_size:
        raise ValueError(f"{path}: truncated header")
    magic, version, map_crc32, map_size_bytes, num_roads, num_lanes, num_goals = struct.unpack_from(_HEADER_FMT, raw, 0)
    if magic != NAVI_MAGIC:
        raise ValueError(f"{path}: bad magic {magic!r}")
    if version != NAVI_VERSION:
        raise ValueError(f"{path}: unsupported version {version}")
    if num_lanes <= 0 or not 1 <= num_goals <= NAVI_MAX_GOALS:
        raise ValueError(f"{path}: invalid counts lanes={num_lanes} goals={num_goals}")
    expected_size = header_size + 4 * (3 * num_lanes + 3 * num_goals + 5 * num_goals * num_lanes)
    if len(raw) != expected_size:
        raise ValueError(f"{path}: size {len(raw)} != expected {expected_size}")

    offset = header_size

    def take(count, dtype):
        nonlocal offset
        arr = np.frombuffer(raw, dtype=dtype, count=count, offset=offset).copy()
        offset += 4 * count
        return arr

    navi = {
        "map_crc32": map_crc32,
        "map_size_bytes": map_size_bytes,
        "num_roads": num_roads,
        "num_lanes": num_lanes,
        "num_goals": num_goals,
        "lane_road_idx": take(num_lanes, "<i4"),
        "left_lane": take(num_lanes, "<i4"),
        "right_lane": take(num_lanes, "<i4"),
        "goal_lane": take(num_goals, "<i4"),
        "goal_x": take(num_goals, "<f4"),
        "goal_y": take(num_goals, "<f4"),
    }
    for key, dtype in _PER_GOAL_LANE_FIELDS:
        navi[key] = take(num_goals * num_lanes, np.dtype(dtype).newbyteorder("<")).reshape(num_goals, num_lanes)
    validate_navigation(navi)
    return navi
