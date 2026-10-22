"""Lane-navigation sidecars (`map_XXX.navi`) for PufferDrive 2.0 map binaries.

Each navigation-free `map_XXX.bin` gets a `map_XXX.navi` next to it holding, per goal x lane,
the longitudinal next hop, the distance to the goal and how many metres the agent can stay in
its left / own / right lane while the goal stays reachable (lane feasibility). Every object's
logged goal is matched to a goal lane so the C env knows which goal each agent drives to.
Loaded by `load_navigation_binary` in drive.h when `navigation = 1` in drive.ini.

Layout (little-endian):
    char[8]  magic "PDNAVI01"
    int32    version (2: PufferDrive 2.0 layout)
    uint32   map_crc32        zlib CRC-32 of the source map .bin bytes
    uint64   map_size_bytes
    int32    num_objects, num_roads, num_lanes, num_goals
    int32    lane_entity_idx[num_lanes]           entity index of each ROAD_LANE (ascending)
    int32    left_lane[num_lanes], right_lane[num_lanes]   navi lane idx of a legal lane change, or -1
    int32    goal_lane[num_goals]                 navi lane whose end point is the goal
    float32  goal_x[num_goals], goal_y[num_goals]
    int32    object_goal_idx[num_objects]         goal of each object's logged goal, or -1
    int32    next_lane[num_goals * num_lanes]     longitudinal next hop toward the goal, -1 if none
    float32  distance_to_goal_m[num_goals * num_lanes]   from lane start; NAVI_UNREACHABLE_M if unreachable
    float32  feasibility_left_m / center_m / right_m [num_goals * num_lanes]
"""

import heapq
import os
import struct
import zlib
from pathlib import Path

import numpy as np

NAVI_MAGIC = b"PDNAVI01"
NAVI_VERSION = 2
NAVI_SUFFIX = ".navi"
NAVI_MAX_GOALS = 1024
NAVI_UNREACHABLE_M = 1.0e9
NAVI_FEASIBILITY_CAP_M = 1000.0
DEFAULT_MAX_GOALS = 64
GOAL_SEED_BASE = 42

VEHICLE = 1
PEDESTRIAN = 2
CYCLIST = 3
ROAD_LANE = 4
ROAD_EDGE = 6
OBJECT_TYPES = (VEHICLE, PEDESTRIAN, CYCLIST)
NAVIGATING_OBJECT_TYPES = (VEHICLE, CYCLIST)
SCENARIO_ID_BYTES = 16
MAX_ENTITY_COUNT = 1_000_000
MAX_POLYLINE_POINTS = 1_000_000

DEFAULT_LANE_WIDTH_M = 3.7
EXIT_MATCH_RADIUS_M = 0.5
EXIT_HEADING_COS_MIN = 0.0
LANE_CHANGE_COST_M = 10.0
MIN_EDGE_COST_M = 1.0e-3
LATERAL_HEADING_COS_MIN = 0.9
LATERAL_OFFSET_TOLERANCE_FRAC = 0.5
LATERAL_PROBE_FRACTIONS = (0.25, 0.5, 0.75)
MARKER_SEARCH_RADIUS_M = 1.0
GOAL_LANE_MATCH_RADIUS_M = 3.0
GOAL_HEADING_COS_MIN = 0.5
DENSIFY_STEP_M = 0.5
FEASIBILITY_MAX_HOPS = 512

_HEADER_FMT = "<8siIQiiii"
_PER_GOAL_LANE_FIELDS = (
    ("next_lane", "<i4"),
    ("distance_to_goal_m", "<f4"),
    ("feasibility_left_m", "<f4"),
    ("feasibility_center_m", "<f4"),
    ("feasibility_right_m", "<f4"),
)


def read_map_binary(path) -> dict:
    """Strict reader mirroring load_map_binary in drive.h; rejects truncated or trailing bytes."""
    raw = Path(path).read_bytes()
    offset = SCENARIO_ID_BYTES

    def take(count, dtype):
        nonlocal offset
        if count < 0 or offset + 4 * count > len(raw):
            raise ValueError(f"{path}: truncated at byte {offset}")
        values = np.frombuffer(raw, dtype=dtype, count=count, offset=offset)
        offset += 4 * count
        return values

    if len(raw) < SCENARIO_ID_BYTES:
        raise ValueError(f"{path}: truncated scenario id")
    take(1, "<i4")  # sdc_track_index
    tracks_to_predict_count = int(take(1, "<i4")[0])
    if not 0 <= tracks_to_predict_count <= MAX_ENTITY_COUNT:
        raise ValueError(f"{path}: invalid tracks_to_predict count {tracks_to_predict_count}")
    take(tracks_to_predict_count, "<i4")
    num_objects, num_roads = (int(v) for v in take(2, "<i4"))
    if not (0 <= num_objects <= MAX_ENTITY_COUNT and 0 <= num_roads <= MAX_ENTITY_COUNT):
        raise ValueError(f"{path}: invalid entity counts objects={num_objects} roads={num_roads}")

    entities = []
    for _ in range(num_objects + num_roads):
        _, entity_type, _, point_count = (int(v) for v in take(4, "<i4"))
        if not 0 <= point_count <= MAX_POLYLINE_POINTS:
            raise ValueError(f"{path}: invalid array_size {point_count}")
        entity = {"type": entity_type, "x": take(point_count, "<f4"), "y": take(point_count, "<f4")}
        take(point_count, "<f4")  # z
        if entity_type in OBJECT_TYPES:
            take(3 * point_count, "<f4")  # vx, vy, vz
            entity["heading"] = take(point_count, "<f4")
            entity["valid"] = take(point_count, "<i4")
        scalars = take(6, "<f4")
        entity["goal_x"], entity["goal_y"] = float(scalars[3]), float(scalars[4])
        take(1, "<i4")  # mark_as_expert
        entities.append(entity)
    if offset != len(raw):
        raise ValueError(f"{path}: {len(raw) - offset} trailing bytes")
    return {"num_objects": num_objects, "num_roads": num_roads, "entities": entities}


def map_fingerprint(map_path) -> tuple[int, int]:
    data = Path(map_path).read_bytes()
    return zlib.crc32(data) & 0xFFFFFFFF, len(data)


def navigation_path_for(map_path) -> Path:
    return Path(map_path).with_suffix(NAVI_SUFFIX)


def validate_navigation(navi: dict) -> None:
    num_lanes, num_goals, num_objects = navi["num_lanes"], navi["num_goals"], navi["num_objects"]
    if num_lanes <= 0:
        raise ValueError(f"num_lanes must be positive, got {num_lanes}")
    if not 1 <= num_goals <= NAVI_MAX_GOALS:
        raise ValueError(f"num_goals must be in [1, {NAVI_MAX_GOALS}], got {num_goals}")
    expected_shapes = {"lane_entity_idx": (num_lanes,), "left_lane": (num_lanes,), "right_lane": (num_lanes,)}
    expected_shapes.update({key: (num_goals,) for key in ("goal_lane", "goal_x", "goal_y")})
    expected_shapes["object_goal_idx"] = (num_objects,)
    expected_shapes.update({key: (num_goals, num_lanes) for key, _ in _PER_GOAL_LANE_FIELDS})
    for key, shape in expected_shapes.items():
        if navi[key].shape != shape:
            raise ValueError(f"{key} must have shape {shape}, got {navi[key].shape}")
    lane_entity_idx = navi["lane_entity_idx"]
    if np.any(np.diff(lane_entity_idx) <= 0) or lane_entity_idx[0] < 0:
        raise ValueError("lane_entity_idx must be non-negative and strictly ascending")
    if lane_entity_idx[-1] >= num_objects + navi["num_roads"]:
        raise ValueError("lane_entity_idx out of range")
    for key in ("left_lane", "right_lane", "next_lane"):
        if np.any(navi[key] < -1) or np.any(navi[key] >= num_lanes):
            raise ValueError(f"{key} out of range [-1, {num_lanes})")
    if np.any(navi["goal_lane"] < 0) or np.any(navi["goal_lane"] >= num_lanes):
        raise ValueError("goal_lane out of range")
    if np.any(navi["object_goal_idx"] < -1) or np.any(navi["object_goal_idx"] >= num_goals):
        raise ValueError(f"object_goal_idx out of range [-1, {num_goals})")
    for key in ("goal_x", "goal_y", "distance_to_goal_m"):
        if not np.all(np.isfinite(navi[key])):
            raise ValueError(f"{key} contains NaN/Inf")
    if np.any(navi["distance_to_goal_m"] < 0) or np.any(navi["distance_to_goal_m"] > NAVI_UNREACHABLE_M):
        raise ValueError("distance_to_goal_m out of range")
    for key in ("feasibility_left_m", "feasibility_center_m", "feasibility_right_m"):
        values = navi[key]
        if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > NAVI_FEASIBILITY_CAP_M):
            raise ValueError(f"{key} must be finite and in [0, {NAVI_FEASIBILITY_CAP_M}]")


def write_navigation(navi: dict, path) -> None:
    validate_navigation(navi)
    header = struct.pack(
        _HEADER_FMT,
        NAVI_MAGIC,
        NAVI_VERSION,
        navi["map_crc32"],
        navi["map_size_bytes"],
        navi["num_objects"],
        navi["num_roads"],
        navi["num_lanes"],
        navi["num_goals"],
    )
    with open(path, "wb") as f:
        f.write(header)
        for key in ("lane_entity_idx", "left_lane", "right_lane", "goal_lane"):
            f.write(np.ascontiguousarray(navi[key], dtype="<i4").tobytes())
        for key in ("goal_x", "goal_y"):
            f.write(np.ascontiguousarray(navi[key], dtype="<f4").tobytes())
        f.write(np.ascontiguousarray(navi["object_goal_idx"], dtype="<i4").tobytes())
        for key, dtype in _PER_GOAL_LANE_FIELDS:
            f.write(np.ascontiguousarray(navi[key], dtype=dtype).tobytes())


def read_navigation(path) -> dict:
    raw = Path(path).read_bytes()
    header_size = struct.calcsize(_HEADER_FMT)
    if len(raw) < header_size:
        raise ValueError(f"{path}: truncated header")
    magic, version, crc32, size_bytes, num_objects, num_roads, num_lanes, num_goals = struct.unpack_from(
        _HEADER_FMT, raw, 0
    )
    if magic != NAVI_MAGIC:
        raise ValueError(f"{path}: bad magic {magic!r}")
    if version != NAVI_VERSION:
        raise ValueError(f"{path}: unsupported version {version}")
    if num_objects < 0 or num_roads < 0 or num_lanes <= 0 or not 1 <= num_goals <= NAVI_MAX_GOALS:
        raise ValueError(f"{path}: invalid counts objects={num_objects} lanes={num_lanes} goals={num_goals}")
    expected_size = header_size + 4 * (3 * num_lanes + 3 * num_goals + num_objects + 5 * num_goals * num_lanes)
    if len(raw) != expected_size:
        raise ValueError(f"{path}: size {len(raw)} != expected {expected_size}")
    offset = header_size

    def take(count, dtype):
        nonlocal offset
        values = np.frombuffer(raw, dtype=dtype, count=count, offset=offset).copy()
        offset += 4 * count
        return values

    navi = {
        "map_crc32": crc32,
        "map_size_bytes": size_bytes,
        "num_objects": num_objects,
        "num_roads": num_roads,
        "num_lanes": num_lanes,
        "num_goals": num_goals,
        "lane_entity_idx": take(num_lanes, "<i4"),
        "left_lane": take(num_lanes, "<i4"),
        "right_lane": take(num_lanes, "<i4"),
        "goal_lane": take(num_goals, "<i4"),
        "goal_x": take(num_goals, "<f4"),
        "goal_y": take(num_goals, "<f4"),
        "object_goal_idx": take(num_objects, "<i4"),
    }
    for key, dtype in _PER_GOAL_LANE_FIELDS:
        navi[key] = take(num_goals * num_lanes, dtype).reshape(num_goals, num_lanes)
    validate_navigation(navi)
    return navi


def _polyline_length(xs, ys) -> float:
    return float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))


def _densify(xs, ys, step_m):
    """Resample a polyline at ~step_m; returns points (N,2), unit directions (N,2) and arc lengths (N,)."""
    pts = np.stack([np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)], axis=1)
    out_pts, out_dirs, out_s = [], [], []
    s_start = 0.0
    for seg_idx in range(len(pts) - 1):
        seg = pts[seg_idx + 1] - pts[seg_idx]
        seg_len = float(np.hypot(*seg))
        if seg_len <= 0.0:
            continue
        sample_count = max(1, int(np.ceil(seg_len / step_m)))
        t = np.arange(sample_count, dtype=np.float64) / sample_count
        out_pts.append(pts[seg_idx] + t[:, None] * seg)
        out_dirs.append(np.repeat((seg / seg_len)[None, :], sample_count, axis=0))
        out_s.append(s_start + t * seg_len)
        s_start += seg_len
    if not out_pts:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)
    out_pts.append(pts[-1:])
    out_dirs.append(out_dirs[-1][-1:])
    out_s.append(np.array([s_start]))
    return np.concatenate(out_pts), np.concatenate(out_dirs), np.concatenate(out_s)


def _end_tangents(xs, ys):
    """Unit direction of the first and last non-degenerate segment (zero vectors if none)."""
    seg = np.stack([np.diff(xs), np.diff(ys)], axis=1).astype(np.float64)
    norms = np.hypot(seg[:, 0], seg[:, 1])
    nonzero = np.nonzero(norms > 0.0)[0]
    if len(nonzero) == 0:
        return np.zeros(2), np.zeros(2)
    first, last = nonzero[0], nonzero[-1]
    return seg[first] / norms[first], seg[last] / norms[last]


def _lane_pose_at(xs, ys, s_m):
    """Point and unit direction at arc length s_m along a polyline."""
    seg_lengths = np.hypot(np.diff(xs), np.diff(ys))
    cum = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    end_idx = int(np.clip(np.searchsorted(cum, s_m), 1, len(cum) - 1))
    start_idx = end_idx - 1
    span = cum[end_idx] - cum[start_idx]
    frac = float(np.clip((s_m - cum[start_idx]) / span, 0.0, 1.0)) if span > 0 else 0.0
    point = np.array(
        [xs[start_idx] + frac * (xs[end_idx] - xs[start_idx]), ys[start_idx] + frac * (ys[end_idx] - ys[start_idx])]
    )
    direction = np.array([xs[end_idx] - xs[start_idx], ys[end_idx] - ys[start_idx]], dtype=np.float64)
    norm = np.hypot(*direction)
    return point, (direction / norm if norm > 0 else np.array([1.0, 0.0]))


class _SpatialIndex:
    """Densified point cloud with per-point owner, direction and arc length; brute-force radius queries."""

    def __init__(self, polylines):
        pts, dirs, arc, owners = [], [], [], []
        for owner, (xs, ys) in polylines:
            p, d, s = _densify(xs, ys, DENSIFY_STEP_M)
            pts.append(p)
            dirs.append(d)
            arc.append(s)
            owners.append(np.full(len(p), owner, dtype=np.int64))
        self.points = np.concatenate(pts) if pts else np.zeros((0, 2))
        self.dirs = np.concatenate(dirs) if dirs else np.zeros((0, 2))
        self.arc_m = np.concatenate(arc) if arc else np.zeros(0)
        self.owners = np.concatenate(owners) if owners else np.zeros(0, dtype=np.int64)

    def query(self, center, radius):
        if len(self.points) == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0)
        dist = np.hypot(self.points[:, 0] - center[0], self.points[:, 1] - center[1])
        idx = np.nonzero(dist <= radius)[0]
        return idx, dist[idx]


def _infer_exits(lane_polylines):
    """Exit lanes per lane: another lane starting at this lane's end (2.0 bins carry no connectivity)."""
    starts = np.array([[xs[0], ys[0]] for xs, ys in lane_polylines], dtype=np.float64)
    ends = np.array([[xs[-1], ys[-1]] for xs, ys in lane_polylines], dtype=np.float64)
    tangents = [_end_tangents(xs, ys) for xs, ys in lane_polylines]
    start_dirs = np.array([t[0] for t in tangents])
    end_dirs = np.array([t[1] for t in tangents])
    exits = []
    for lane in range(len(lane_polylines)):
        gap = np.hypot(starts[:, 0] - ends[lane, 0], starts[:, 1] - ends[lane, 1])
        aligned = start_dirs @ end_dirs[lane] >= EXIT_HEADING_COS_MIN
        candidates = np.nonzero((gap <= EXIT_MATCH_RADIUS_M) & aligned)[0]
        exits.append(sorted(int(c) for c in candidates if c != lane))
    return exits


def _find_lateral_neighbor(lane, xs, ys, length_m, side_sign, lane_index, edge_index):
    """Navi lane on one side (+1 left, -1 right) at every probe with no road edge in between, or -1."""
    neighbor = None
    for frac in LATERAL_PROBE_FRACTIONS:
        point, direction = _lane_pose_at(xs, ys, frac * length_m)
        left_normal = np.array([-direction[1], direction[0]])
        probe = point + side_sign * DEFAULT_LANE_WIDTH_M * left_normal
        idx, dist = lane_index.query(probe, LATERAL_OFFSET_TOLERANCE_FRAC * DEFAULT_LANE_WIDTH_M)
        aligned = (lane_index.owners[idx] != lane) & (lane_index.dirs[idx] @ direction >= LATERAL_HEADING_COS_MIN)
        if not np.any(aligned):
            return -1
        candidate = int(lane_index.owners[idx[aligned][np.argmin(dist[aligned])]])
        if neighbor is not None and candidate != neighbor:
            return -1
        neighbor = candidate
        midpoint = point + side_sign * 0.5 * DEFAULT_LANE_WIDTH_M * left_normal
        if len(edge_index.query(midpoint, MARKER_SEARCH_RADIUS_M)[0]) > 0:
            return -1
    return neighbor


def _dijkstra_to_goal(goal_lane, edge_costs, reverse_edges):
    """Distance from each lane's start to the goal lane's end over reversed weighted edges."""
    dist = np.full(len(edge_costs), np.inf)
    dist[goal_lane] = edge_costs[goal_lane]
    heap = [(dist[goal_lane], goal_lane)]
    while heap:
        d, lane = heapq.heappop(heap)
        if d > dist[lane]:
            continue
        for pred, edge_cost in reverse_edges[lane]:
            candidate = d + edge_cost
            if candidate < dist[pred]:
                dist[pred] = candidate
                heapq.heappush(heap, (candidate, pred))
    return dist


def _best_successor(successors, dist):
    """Successor with minimum finite distance (lowest index on ties), or -1."""
    best, best_dist = -1, np.inf
    for succ in successors:
        if dist[succ] < best_dist:
            best, best_dist = succ, dist[succ]
    return best


def _center_feasibility(lane, goal_lane, lengths, dist_full, stay_next):
    """Metres one can keep following exit lanes from `lane` start while the goal stays reachable."""
    accumulated_m = 0.0
    current = lane
    for _ in range(FEASIBILITY_MAX_HOPS):
        if not np.isfinite(dist_full[current]):
            break
        accumulated_m += lengths[current]
        if current == goal_lane or accumulated_m >= NAVI_FEASIBILITY_CAP_M:
            break
        current = stay_next[current]
        if current == -1:
            break
    return min(accumulated_m, NAVI_FEASIBILITY_CAP_M)


def _farthest_point_goals(endpoints, pool, preselected, count, rng):
    selected = list(preselected)
    pool = np.asarray(pool, dtype=np.int64)
    if len(pool) == 0 or len(selected) >= count:
        return selected
    if not selected:
        selected.append(int(pool[rng.integers(len(pool))]))
    min_dist = np.full(len(pool), np.inf)
    for goal in selected:
        min_dist = np.minimum(min_dist, np.hypot(*(endpoints[pool] - endpoints[goal]).T))
    while len(selected) < count:
        pick = int(np.argmax(min_dist))
        if min_dist[pick] <= 0.0:
            break
        selected.append(int(pool[pick]))
        min_dist = np.minimum(min_dist, np.hypot(*(endpoints[pool] - endpoints[pool[pick]]).T))
    return selected


def _match_object_goal_lane(entity, lane_index, lane_lengths):
    """Navi lane holding the object's logged goal, aligned with its final heading; -1 if none."""
    if entity["type"] not in NAVIGATING_OBJECT_TYPES:
        return -1
    valid_steps = np.nonzero(entity["valid"])[0]
    goal = np.array([entity["goal_x"], entity["goal_y"]], dtype=np.float64)
    if len(valid_steps) == 0 or not np.all(np.isfinite(goal)):
        return -1
    final_heading = float(entity["heading"][valid_steps[-1]])
    heading_dir = np.array([np.cos(final_heading), np.sin(final_heading)])
    idx, dist = lane_index.query(goal, GOAL_LANE_MATCH_RADIUS_M)
    aligned = lane_index.dirs[idx] @ heading_dir >= GOAL_HEADING_COS_MIN
    if not np.any(aligned):
        return -1
    idx, dist = idx[aligned], dist[aligned]
    owners = lane_index.owners[idx]
    remaining_m = lane_lengths[owners] - lane_index.arc_m[idx]
    # Shared endpoints tie on distance: prefer the lane that continues past the goal.
    best = np.lexsort((-remaining_m, np.round(dist, 3)))[0]
    return int(owners[best])


def build_navigation(map_data, max_goals=DEFAULT_MAX_GOALS, seed=GOAL_SEED_BASE) -> dict:
    entities = map_data["entities"]
    num_objects = map_data["num_objects"]
    lane_entity_idx = [i for i, e in enumerate(entities) if e["type"] == ROAD_LANE]
    if not lane_entity_idx:
        raise ValueError("map has no lanes")
    lane_polylines = [
        (entities[i]["x"].astype(np.float64), entities[i]["y"].astype(np.float64)) for i in lane_entity_idx
    ]
    for xs, ys in lane_polylines:
        if len(xs) == 0 or not (np.all(np.isfinite(xs)) and np.all(np.isfinite(ys))):
            raise ValueError("lane geometry must be non-empty and finite")
    num_lanes = len(lane_entity_idx)
    lengths = np.array([_polyline_length(xs, ys) for xs, ys in lane_polylines], dtype=np.float64)
    edge_costs = np.maximum(lengths, MIN_EDGE_COST_M)
    exits = _infer_exits(lane_polylines)

    lane_index = _SpatialIndex(list(enumerate(lane_polylines)))
    edge_index = _SpatialIndex([(i, (e["x"], e["y"])) for i, e in enumerate(entities) if e["type"] == ROAD_EDGE])
    left_lane = np.full(num_lanes, -1, dtype=np.int32)
    right_lane = np.full(num_lanes, -1, dtype=np.int32)
    for lane, (xs, ys) in enumerate(lane_polylines):
        if lengths[lane] <= 0.0:
            continue
        left_lane[lane] = _find_lateral_neighbor(lane, xs, ys, lengths[lane], +1.0, lane_index, edge_index)
        right_lane[lane] = _find_lateral_neighbor(lane, xs, ys, lengths[lane], -1.0, lane_index, edge_index)

    reverse_long = [[] for _ in range(num_lanes)]
    reverse_full = [[] for _ in range(num_lanes)]
    for lane in range(num_lanes):
        for succ in exits[lane]:
            reverse_long[succ].append((lane, edge_costs[lane]))
            reverse_full[succ].append((lane, edge_costs[lane]))
        for side in (left_lane[lane], right_lane[lane]):
            if side != -1:
                reverse_full[side].append((lane, LANE_CHANGE_COST_M))

    object_goal_lane = [_match_object_goal_lane(entities[i], lane_index, lengths) for i in range(num_objects)]
    required = sorted({lane for lane in object_goal_lane if lane >= 0})
    if len(required) > NAVI_MAX_GOALS:
        raise ValueError(f"{len(required)} object goal lanes exceed NAVI_MAX_GOALS={NAVI_MAX_GOALS}")
    goal_budget = min(max(max_goals, len(required)), NAVI_MAX_GOALS)
    endpoints = np.array([[xs[-1], ys[-1]] for xs, ys in lane_polylines], dtype=np.float64)
    required_set = set(required)
    dead_ends = [lane for lane in range(num_lanes) if not exits[lane] and lane not in required_set]
    rng = np.random.default_rng(seed)
    goal_lanes = _farthest_point_goals(endpoints, dead_ends, required, goal_budget, rng)
    goal_lanes = _farthest_point_goals(endpoints, np.arange(num_lanes), goal_lanes, goal_budget, rng)
    goal_of_lane = {lane: goal for goal, lane in enumerate(goal_lanes)}
    object_goal_idx = np.array([goal_of_lane[lane] if lane >= 0 else -1 for lane in object_goal_lane], dtype=np.int32)

    num_goals = len(goal_lanes)
    next_lane = np.full((num_goals, num_lanes), -1, dtype=np.int32)
    distance_to_goal = np.full((num_goals, num_lanes), NAVI_UNREACHABLE_M, dtype=np.float32)
    feas_center = np.zeros((num_goals, num_lanes), dtype=np.float32)
    for goal, goal_lane in enumerate(goal_lanes):
        dist_long = _dijkstra_to_goal(goal_lane, edge_costs, reverse_long)
        dist_full = _dijkstra_to_goal(goal_lane, edge_costs, reverse_full)
        stay_next = [_best_successor(exits[lane], dist_full) for lane in range(num_lanes)]
        for lane in range(num_lanes):
            if lane != goal_lane and np.isfinite(dist_long[lane]):
                next_lane[goal, lane] = _best_successor(exits[lane], dist_long)
            if np.isfinite(dist_full[lane]):
                distance_to_goal[goal, lane] = min(dist_full[lane], NAVI_UNREACHABLE_M)
            feas_center[goal, lane] = _center_feasibility(lane, goal_lane, lengths, dist_full, stay_next)

    feas_left = np.where(left_lane[None, :] >= 0, feas_center[:, np.maximum(left_lane, 0)], 0.0).astype(np.float32)
    feas_right = np.where(right_lane[None, :] >= 0, feas_center[:, np.maximum(right_lane, 0)], 0.0).astype(np.float32)
    return {
        "num_objects": num_objects,
        "num_roads": map_data["num_roads"],
        "num_lanes": num_lanes,
        "num_goals": num_goals,
        "lane_entity_idx": np.asarray(lane_entity_idx, dtype=np.int32),
        "left_lane": left_lane,
        "right_lane": right_lane,
        "goal_lane": np.asarray(goal_lanes, dtype=np.int32),
        "goal_x": endpoints[goal_lanes, 0].astype(np.float32),
        "goal_y": endpoints[goal_lanes, 1].astype(np.float32),
        "object_goal_idx": object_goal_idx,
        "next_lane": next_lane,
        "distance_to_goal_m": distance_to_goal,
        "feasibility_left_m": feas_left,
        "feasibility_center_m": feas_center,
        "feasibility_right_m": feas_right,
    }


def process_map_navigation(map_path, max_goals=DEFAULT_MAX_GOALS, seed=GOAL_SEED_BASE) -> dict:
    """Build the sidecar for one map and write it atomically next to the map as `<stem>.navi`."""
    navi = build_navigation(read_map_binary(map_path), max_goals=max_goals, seed=seed)
    navi["map_crc32"], navi["map_size_bytes"] = map_fingerprint(map_path)
    output_path = navigation_path_for(map_path)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp{os.getpid()}")
    try:
        write_navigation(navi, tmp_path)
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return navi
