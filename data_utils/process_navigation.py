#!/usr/bin/env python3
"""process_navigation.py: build lane-level navigation sidecars for PufferDrive 3.0 map binaries.

Reads navigation-free `.bin` maps, derives a goal-conditioned lane navigation table and writes
`<map_stem>.navi` next to each map (or into --output-dir). The C env loads it when
`env.obs_navigation=true` and conditions the navigation features on whatever goal the agent has.

Per map:
  * lanes        every lane-type road element (types 0-9), indexed in ascending road order
  * lateral      left/right neighbour per lane: co-directional lane one lane-width to the side at
                 every probe point, with no solid line / road edge in between (legal lane change)
  * per goal     every lane is a goal; Dijkstra (lane change cost NAVI_LANE_CHANGE_COST_M) ->
                 distance_to_goal_m from each lane start to the goal lane end;
                 feasibility_center_m = metres one can keep following exit lanes and still reach the goal
                 (NAVI_FEASIBILITY_CAP_M once the stay chain reaches the goal lane)

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
    NAVI_LANE_CHANGE_COST_M,
    NAVI_MAX_LANES,
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
MIN_EDGE_COST_M = 1.0e-3
LATERAL_HEADING_COS_MIN = 0.9
LATERAL_OFFSET_TOLERANCE_FRAC = 0.5
LATERAL_PROBE_FRACTIONS = (0.25, 0.5, 0.75)
MARKER_SEARCH_RADIUS_M = 1.0
DENSIFY_STEP_M = 0.5
FEASIBILITY_MAX_HOPS = 512
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


def _stay_successors(exits, distance_to_goal):
    """Per (goal, lane): the exit lane with the smallest distance to the goal (lowest index on ties), or -1."""
    num_goals, num_lanes = distance_to_goal.shape
    max_exit_count = max((len(lane_exits) for lane_exits in exits), default=0)
    if max_exit_count == 0:
        return np.full((num_goals, num_lanes), -1, dtype=np.int64)
    padded_exits = np.full((num_lanes, max_exit_count), -1, dtype=np.int64)
    for lane, lane_exits in enumerate(exits):
        padded_exits[lane, : len(lane_exits)] = lane_exits
    exit_distances = np.where(padded_exits[None] >= 0, distance_to_goal[:, np.maximum(padded_exits, 0)], np.inf)
    best_slot = np.argmin(exit_distances, axis=2)
    best_exit = np.take_along_axis(padded_exits[None].repeat(num_goals, 0), best_slot[..., None], axis=2)[..., 0]
    has_reachable_exit = np.isfinite(np.min(exit_distances, axis=2))
    return np.where(has_reachable_exit, best_exit, -1)


def _center_feasibility(lengths, distance_to_goal, stay_next):
    """Metres one can follow stay_next from each lane start with the goal reachable; capped on the goal lane."""
    num_goals, num_lanes = distance_to_goal.shape
    reachable = np.isfinite(distance_to_goal)
    goal_rows = np.arange(num_goals)[:, None]
    is_goal_lane = np.zeros((num_goals, num_lanes), dtype=bool)
    is_goal_lane[np.arange(num_goals), np.arange(num_goals)] = True
    feasibility = np.zeros((num_goals, num_lanes), dtype=np.float64)
    for _ in range(FEASIBILITY_MAX_HOPS):
        successor_feasibility = np.where(stay_next >= 0, feasibility[goal_rows, np.maximum(stay_next, 0)], 0.0)
        updated = np.minimum(lengths[None, :] + successor_feasibility, NAVI_FEASIBILITY_CAP_M)
        updated = np.where(is_goal_lane, NAVI_FEASIBILITY_CAP_M, updated)
        updated = np.where(reachable, updated, 0.0)
        if np.array_equal(updated, feasibility):
            break
        feasibility = updated
    return feasibility


def build_navigation(data):
    roads = data["roads"]
    lane_road_idx = [i for i, r in enumerate(roads) if _is_lane(r["type"])]
    if not lane_road_idx:
        raise ValueError("map has no lanes")
    if len(lane_road_idx) > NAVI_MAX_LANES:
        raise ValueError(f"map has {len(lane_road_idx)} lanes, more than NAVI_MAX_LANES={NAVI_MAX_LANES}")
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

    reverse_edges = [[] for _ in range(num_lanes)]
    for n in range(num_lanes):
        for succ in exits[n]:
            reverse_edges[succ].append((n, edge_costs[n]))
        for side in (left_lane[n], right_lane[n]):
            if side != -1:
                reverse_edges[side].append((n, NAVI_LANE_CHANGE_COST_M))

    distance_to_goal = np.stack(
        [_dijkstra_to_goal(goal_lane, edge_costs, reverse_edges) for goal_lane in range(num_lanes)]
    )
    stay_next = _stay_successors(exits, distance_to_goal)
    feasibility_center = _center_feasibility(lengths, distance_to_goal, stay_next)
    return {
        "num_roads": len(roads),
        "num_lanes": num_lanes,
        "lane_road_idx": np.asarray(lane_road_idx, dtype=np.int32),
        "left_lane": left_lane,
        "right_lane": right_lane,
        "distance_to_goal_m": np.minimum(distance_to_goal, NAVI_UNREACHABLE_M).astype(np.float32),
        "feasibility_center_m": feasibility_center.astype(np.float32),
    }


def process_map(map_path, output_path):
    """Build and atomically write one sidecar. Returns the navigation dict."""
    navi = build_navigation(read_bin(Path(map_path)))
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
    map_path, output_path, timeout_s, plot_dir = job
    start = time.time()
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_s)
    try:
        navi = process_map(map_path, output_path)
        summary = f"lanes={navi['num_lanes']}"
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
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--timeout-s", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--overwrite", action="store_true", help="Rebuild sidecars that already exist")
    parser.add_argument(
        "--plot-dir", default=None, help="Also write <map_stem>.html/.svg navigation plots here (plot_navigation.py)"
    )
    args = parser.parse_args(argv)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    jobs, skipped = [], 0
    for map_path in _collect_maps(args.input):
        output_path = navigation_path_for(map_path, args.output_dir)
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        jobs.append((str(map_path), str(output_path), args.timeout_s, args.plot_dir))

    failed = 0
    with multiprocessing.Pool(min(args.workers, max(1, len(jobs)))) as pool:
        for map_path, ok, message in pool.imap_unordered(_worker, jobs):
            failed += not ok
            print(f"[{'OK' if ok else 'FAIL'}] {Path(map_path).name}: {message}", flush=True)
    print(f"done: {len(jobs) - failed} written, {failed} failed, {skipped} skipped (existing)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
