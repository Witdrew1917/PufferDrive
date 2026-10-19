#!/usr/bin/env python3
"""process_navigation.py: build lane-level navigation sidecars for PufferDrive 3.0 map binaries.

Reads navigation-free `.bin` maps, derives a goal-conditioned lane navigation table and writes
`<map_stem>.navi` next to each map (or into --output-dir). The C env loads it when
`env.route_source=navigation` and replaces the random-walk route with goal-directed routes.

Per map:
  * lanes        every lane-type road element (types 0-9), indexed in ascending road order
  * lateral      left/right neighbour per lane: co-directional lane one lane-width to the side at
                 every probe point, with no solid line / road edge in between (legal lane change)
  * goals        dead-end lanes + logged route-end lanes, topped up by farthest-point sampling
  * per goal     Dijkstra (lane change cost LANE_CHANGE_COST_M) -> distance_to_goal_m;
                 longitudinal-only Dijkstra -> next_lane (route chains follow exit lanes only);
                 feasibility_*_m = metres one can stay in (left/own/right) lane and still reach the goal

Usage:
    python data_utils/process_navigation.py --input pufferlib/resources/drive/binaries/carla
    python data_utils/process_navigation.py --input maps/ --output-dir navi/ --workers 16 --overwrite
    python data_utils/process_navigation.py --input maps/ --output-dir navi/ --plot-dir navi_plots/
"""

import argparse
import heapq
import multiprocessing
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mirror_map_bin import _is_lane, read_bin  # noqa: E402
from navigation_bin import (  # noqa: E402
    NAVI_FEASIBILITY_CAP_M,
    NAVI_MAX_GOALS,
    NAVI_UNREACHABLE_M,
    map_fingerprint,
    navigation_path_for,
    validate_navigation,
    write_navigation,
)

LANE_FREEWAY = 1
LANE_SURFACE_STREET = 2
DRIVABLE_LANE_TYPES = (LANE_FREEWAY, LANE_SURFACE_STREET)
SOLID_ROAD_LINE_TYPES = (12, 13, 16, 17)
ROAD_LINE_TYPE_RANGE = (10, 19)
ROAD_EDGE_TYPE_RANGE = (20, 29)

DEFAULT_LANE_WIDTH_M = 3.7
LANE_CHANGE_COST_M = 10.0
MIN_EDGE_COST_M = 1.0e-3
LATERAL_HEADING_COS_MIN = 0.9
LATERAL_OFFSET_TOLERANCE_FRAC = 0.5
LATERAL_PROBE_FRACTIONS = (0.25, 0.5, 0.75)
MARKER_SEARCH_RADIUS_M = 1.0
DENSIFY_STEP_M = 0.5
FEASIBILITY_MAX_HOPS = 512
DEFAULT_MAX_GOALS = 64
GOAL_SEED_BASE = 42
DEFAULT_TIMEOUT_SECONDS = 600


def _densify(xs, ys, step_m):
    """Resample a polyline at ~step_m; returns points (N,2) and unit directions (N,2)."""
    pts = np.stack([np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)], axis=1)
    out_pts, out_dirs = [], []
    for seg_idx in range(len(pts) - 1):
        seg = pts[seg_idx + 1] - pts[seg_idx]
        seg_len = float(np.hypot(*seg))
        if seg_len <= 0.0:
            continue
        sample_count = max(1, int(np.ceil(seg_len / step_m)))
        t = np.arange(sample_count, dtype=np.float64)[:, None] / sample_count
        out_pts.append(pts[seg_idx] + t * seg)
        out_dirs.append(np.repeat((seg / seg_len)[None, :], sample_count, axis=0))
    if not out_pts:
        return np.zeros((0, 2)), np.zeros((0, 2))
    out_pts.append(pts[-1:])
    out_dirs.append(out_dirs[-1][-1:])
    return np.concatenate(out_pts), np.concatenate(out_dirs)


def _lane_pose_at(road, s_m):
    """Point, unit direction and lane width at arc-length s_m along a lane."""
    cum = np.asarray(road["cum_lengths"], dtype=np.float64)
    xs = np.asarray(road["x"], dtype=np.float64)
    ys = np.asarray(road["y"], dtype=np.float64)
    end_idx = int(np.clip(np.searchsorted(cum, s_m), 1, len(cum) - 1))
    start_idx = end_idx - 1
    span = cum[end_idx] - cum[start_idx]
    frac = float(np.clip((s_m - cum[start_idx]) / span, 0.0, 1.0)) if span > 0 else 0.0
    point = np.array(
        [xs[start_idx] + frac * (xs[end_idx] - xs[start_idx]), ys[start_idx] + frac * (ys[end_idx] - ys[start_idx])]
    )
    direction = np.array([xs[end_idx] - xs[start_idx], ys[end_idx] - ys[start_idx]])
    norm = np.hypot(*direction)
    direction = direction / norm if norm > 0 else np.array([1.0, 0.0])
    width = DEFAULT_LANE_WIDTH_M
    if "widths" in road:
        widths = road["widths"]
        width = float(widths[start_idx] + frac * (widths[end_idx] - widths[start_idx]))
        if not np.isfinite(width) or width <= 0.0:
            width = DEFAULT_LANE_WIDTH_M
    return point, direction, width


class _SpatialIndex:
    """Flat point cloud with per-point owner and direction; brute-force radius queries."""

    def __init__(self, polylines):
        pts, dirs, owners = [], [], []
        for owner, (xs, ys) in polylines:
            p, d = _densify(xs, ys, DENSIFY_STEP_M)
            pts.append(p)
            dirs.append(d)
            owners.append(np.full(len(p), owner, dtype=np.int64))
        self.points = np.concatenate(pts) if pts else np.zeros((0, 2))
        self.dirs = np.concatenate(dirs) if dirs else np.zeros((0, 2))
        self.owners = np.concatenate(owners) if owners else np.zeros(0, dtype=np.int64)

    def query(self, center, radius):
        if len(self.points) == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0)
        dist = np.hypot(self.points[:, 0] - center[0], self.points[:, 1] - center[1])
        idx = np.nonzero(dist <= radius)[0]
        return idx, dist[idx]


def _lane_change_blocked(marker_index, marker_types, center):
    idx, _ = marker_index.query(center, MARKER_SEARCH_RADIUS_M)
    for road_type in {marker_types[o] for o in marker_index.owners[idx].tolist()}:
        if ROAD_EDGE_TYPE_RANGE[0] <= road_type <= ROAD_EDGE_TYPE_RANGE[1]:
            return True
        if road_type in SOLID_ROAD_LINE_TYPES:
            return True
    return False


def _find_lateral_neighbor(lane_idx, road, side_sign, lane_index, marker_index, marker_types):
    """Navi lane index of the legal neighbour on one side (+1 left, -1 right), or -1."""
    neighbor = None
    for frac in LATERAL_PROBE_FRACTIONS:
        point, direction, width = _lane_pose_at(road, frac * road["length"])
        left_normal = np.array([-direction[1], direction[0]])
        probe = point + side_sign * width * left_normal
        idx, dist = lane_index.query(probe, LATERAL_OFFSET_TOLERANCE_FRAC * width)
        aligned = (lane_index.owners[idx] != lane_idx) & (lane_index.dirs[idx] @ direction >= LATERAL_HEADING_COS_MIN)
        if not np.any(aligned):
            return -1
        candidate = int(lane_index.owners[idx[aligned][np.argmin(dist[aligned])]])
        if neighbor is not None and candidate != neighbor:
            return -1
        neighbor = candidate
        if _lane_change_blocked(marker_index, marker_types, point + side_sign * 0.5 * width * left_normal):
            return -1
    return neighbor


def _dijkstra_to_goal(goal_lane, lane_costs, reverse_edges):
    """Distance from each lane's start to the goal lane's end over reversed weighted edges."""
    dist = np.full(len(lane_costs), np.inf)
    dist[goal_lane] = lane_costs[goal_lane]
    heap = [(dist[goal_lane], goal_lane)]
    while heap:
        d, lane = heapq.heappop(heap)
        if d > dist[lane]:
            continue
        for pred, edge_cost in reverse_edges[lane]:
            nd = d + edge_cost
            if nd < dist[pred]:
                dist[pred] = nd
                heapq.heappush(heap, (nd, pred))
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
    if len(pool) == 0:
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


def select_goal_lanes(roads, lane_road_idx, exits, agents, road_to_navi, max_goals, seed):
    drivable = [n for n, r in enumerate(lane_road_idx) if roads[r]["type"] in DRIVABLE_LANE_TYPES]
    if not drivable:
        raise ValueError("map has no drivable lanes")
    drivable_set = set(drivable)
    priority = {n for n in drivable if not exits[n]}
    for agent in agents:
        if agent["route"] and agent["route"][-1] in road_to_navi:
            route_end = road_to_navi[agent["route"][-1]]
            if route_end in drivable_set:
                priority.add(route_end)
    endpoints = np.array([[roads[r]["x"][-1], roads[r]["y"][-1]] for r in lane_road_idx], dtype=np.float64)
    rng = np.random.default_rng(seed)
    priority = sorted(priority)
    if len(priority) >= max_goals:
        return _farthest_point_goals(endpoints, priority, [], max_goals, rng)
    return _farthest_point_goals(endpoints, drivable, priority, max_goals, rng)


def build_navigation(data, max_goals=DEFAULT_MAX_GOALS, seed=GOAL_SEED_BASE):
    roads = data["roads"]
    lane_road_idx = [i for i, r in enumerate(roads) if _is_lane(r["type"])]
    if not lane_road_idx:
        raise ValueError("map has no lanes")
    road_to_navi = {r: n for n, r in enumerate(lane_road_idx)}
    num_lanes = len(lane_road_idx)
    lengths = np.array([roads[r]["length"] for r in lane_road_idx], dtype=np.float64)
    if not np.all(np.isfinite(lengths)) or np.any(lengths < 0):
        raise ValueError("lane lengths must be finite and non-negative")
    edge_costs = np.maximum(lengths, MIN_EDGE_COST_M)

    exits = []
    for r in lane_road_idx:
        lane_exits = []
        for exit_road in roads[r]["exit_lanes"]:
            if exit_road == -1:
                continue
            if exit_road not in road_to_navi:
                raise ValueError(f"lane {r} exits to non-lane road element {exit_road}")
            lane_exits.append(road_to_navi[exit_road])
        exits.append(sorted(set(lane_exits)))

    lane_index = _SpatialIndex([(n, (roads[r]["x"], roads[r]["y"])) for n, r in enumerate(lane_road_idx)])
    marker_roads = [i for i, r in enumerate(roads) if ROAD_LINE_TYPE_RANGE[0] <= r["type"] <= ROAD_EDGE_TYPE_RANGE[1]]
    marker_index = _SpatialIndex([(i, (roads[i]["x"], roads[i]["y"])) for i in marker_roads])
    marker_types = {i: roads[i]["type"] for i in marker_roads}
    left_lane = np.full(num_lanes, -1, dtype=np.int32)
    right_lane = np.full(num_lanes, -1, dtype=np.int32)
    for n, r in enumerate(lane_road_idx):
        if lengths[n] <= 0.0:
            continue
        left_lane[n] = _find_lateral_neighbor(n, roads[r], +1.0, lane_index, marker_index, marker_types)
        right_lane[n] = _find_lateral_neighbor(n, roads[r], -1.0, lane_index, marker_index, marker_types)

    reverse_long = [[] for _ in range(num_lanes)]
    reverse_full = [[] for _ in range(num_lanes)]
    for n in range(num_lanes):
        for succ in exits[n]:
            reverse_long[succ].append((n, edge_costs[n]))
            reverse_full[succ].append((n, edge_costs[n]))
        for side in (left_lane[n], right_lane[n]):
            if side != -1:
                reverse_full[side].append((n, LANE_CHANGE_COST_M))

    goal_lanes = select_goal_lanes(roads, lane_road_idx, exits, data["agents"], road_to_navi, max_goals, seed)
    num_goals = len(goal_lanes)
    next_lane = np.full((num_goals, num_lanes), -1, dtype=np.int32)
    distance_to_goal = np.full((num_goals, num_lanes), NAVI_UNREACHABLE_M, dtype=np.float32)
    feas_center = np.zeros((num_goals, num_lanes), dtype=np.float32)
    for g, goal_lane in enumerate(goal_lanes):
        dist_long = _dijkstra_to_goal(goal_lane, edge_costs, reverse_long)
        dist_full = _dijkstra_to_goal(goal_lane, edge_costs, reverse_full)
        stay_next = [_best_successor(exits[n], dist_full) for n in range(num_lanes)]
        for n in range(num_lanes):
            if n != goal_lane and np.isfinite(dist_long[n]):
                next_lane[g, n] = _best_successor(exits[n], dist_long)
            if np.isfinite(dist_full[n]):
                distance_to_goal[g, n] = min(dist_full[n], NAVI_UNREACHABLE_M)
            feas_center[g, n] = _center_feasibility(n, goal_lane, lengths, dist_full, stay_next)

    feas_left = np.where(left_lane[None, :] >= 0, feas_center[:, np.maximum(left_lane, 0)], 0.0).astype(np.float32)
    feas_right = np.where(right_lane[None, :] >= 0, feas_center[:, np.maximum(right_lane, 0)], 0.0).astype(np.float32)
    goal_roads = [lane_road_idx[g] for g in goal_lanes]
    return {
        "num_roads": len(roads),
        "num_lanes": num_lanes,
        "num_goals": num_goals,
        "lane_road_idx": np.asarray(lane_road_idx, dtype=np.int32),
        "left_lane": left_lane,
        "right_lane": right_lane,
        "goal_lane": np.asarray(goal_lanes, dtype=np.int32),
        "goal_x": np.array([roads[r]["x"][-1] for r in goal_roads], dtype=np.float32),
        "goal_y": np.array([roads[r]["y"][-1] for r in goal_roads], dtype=np.float32),
        "next_lane": next_lane,
        "distance_to_goal_m": distance_to_goal,
        "feasibility_left_m": feas_left,
        "feasibility_center_m": feas_center,
        "feasibility_right_m": feas_right,
    }


def process_map(map_path, output_path, max_goals=DEFAULT_MAX_GOALS, seed=GOAL_SEED_BASE):
    """Build and atomically write one sidecar. Returns the navigation dict."""
    navi = build_navigation(read_bin(Path(map_path)), max_goals=max_goals, seed=seed)
    navi["map_crc32"], navi["map_size_bytes"] = map_fingerprint(map_path)
    validate_navigation(navi)
    output_path = Path(output_path)
    tmp_path = output_path.with_name(output_path.name + f".tmp{os.getpid()}")
    try:
        write_navigation(navi, tmp_path)
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return navi


def _timeout_handler(signum, frame):
    raise TimeoutError("navigation processing timed out")


def _worker(job):
    map_path, output_path, max_goals, seed, timeout_s, plot_dir = job
    start = time.time()
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_s)
    try:
        navi = process_map(map_path, output_path, max_goals, seed)
        summary = f"lanes={navi['num_lanes']} goals={navi['num_goals']}"
        if plot_dir:
            from plot_navigation import render_map

            render_map(map_path, plot_dir, navigation_dir=Path(output_path).parent)
        return map_path, True, f"{summary} ({time.time() - start:.1f}s)"
    except Exception as exc:  # report per-map failure, keep the batch going
        return map_path, False, f"{type(exc).__name__}: {exc}"
    finally:
        signal.alarm(0)


def _collect_maps(input_path):
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    return sorted(p for p in input_path.iterdir() if p.suffix == ".bin")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Map .bin file or directory of .bin maps")
    parser.add_argument("--output-dir", default=None, help="Sidecar directory (default: next to each map)")
    parser.add_argument("--max-goals", type=int, default=DEFAULT_MAX_GOALS)
    parser.add_argument("--seed", type=int, default=GOAL_SEED_BASE)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--timeout-s", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--overwrite", action="store_true", help="Rebuild sidecars that already exist")
    parser.add_argument(
        "--plot-dir", default=None, help="Also write <map_stem>.html/.svg navigation plots here (plot_navigation.py)"
    )
    args = parser.parse_args(argv)
    if not 1 <= args.max_goals <= NAVI_MAX_GOALS:
        parser.error(f"--max-goals must be in [1, {NAVI_MAX_GOALS}]")

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    jobs, skipped = [], 0
    for map_path in _collect_maps(args.input):
        output_path = navigation_path_for(map_path, args.output_dir)
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        jobs.append((str(map_path), str(output_path), args.max_goals, args.seed, args.timeout_s, args.plot_dir))

    failed = 0
    with multiprocessing.Pool(min(args.workers, max(1, len(jobs)))) as pool:
        for map_path, ok, message in pool.imap_unordered(_worker, jobs):
            failed += not ok
            print(f"[{'OK' if ok else 'FAIL'}] {Path(map_path).name}: {message}", flush=True)
    print(f"done: {len(jobs) - failed} written, {failed} failed, {skipped} skipped (existing)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
