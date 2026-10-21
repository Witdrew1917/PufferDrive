#!/usr/bin/env python3
"""json_to_bin.py: convert WOMD-style JSON scenarios to PufferDrive 3.0 map binaries (replay mode).

Input JSON (one scenario per file):
  scenario_id                       str (optional, defaults to the file stem)
  metadata.sim_dt / num_timesteps   float / int
  metadata.sdc_track_index          index into objects; written as agent 0 (EGO_IDX) and always controllable
  objects[]                         type, length, width, height, position[T]{x,y,z}, heading[T],
                                    velocity[T]{x,y}, valid[T], mark_as_expert (true -> replayed)
  roads[]                           type (lane/road_line/road_edge/crosswalk/speed_bump/driveway),
                                    map_element_id (WOMD enum), geometry[]{x,y,z}, speed_limit (m/s, lanes)

Derived (the JSON carries no topology):
  * lane entry/exit lanes from coincident end->start points with aligned tangents
  * lane-graph distance matrix (Dijkstra over drivable lanes, edge weight = source lane length)
  * per-vehicle routes: the logged trajectory is matched to an exit-connected lane chain (DP over
    nearby, heading-aligned lane candidates), then extended to a dead end; route_gt_len counts the
    GT-supported prefix. Mirrors 123Drive's route algorithm in simplified form.

Not converted: traffic lights / stop signs (none in the format), lane widths (sim default), lane priority.

Usage:
    python data_utils/json_to_bin.py --input ~/Data/dl2_micro --output ~/Data/dl2_micro_bins
    python data_utils/json_to_bin.py --input scenario.json --output bins/ --dataset-name dl2
"""

import argparse
import heapq
import json
import math
import multiprocessing
import sys
import traceback
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mirror_map_bin import write_bin  # noqa: E402

AGENT_VEHICLE = 1
AGENT_PEDESTRIAN = 2
AGENT_CYCLIST = 3
AGENT_TYPE_BY_NAME = {
    "vehicle": AGENT_VEHICLE,
    "dynamicclassifiedotherobject": AGENT_VEHICLE,
    "trailer": AGENT_VEHICLE,
    "pedestrian": AGENT_PEDESTRIAN,
    "othervulnerablerider": AGENT_PEDESTRIAN,
    "animal": AGENT_PEDESTRIAN,
    "cyclist": AGENT_CYCLIST,
    "bicyclist": AGENT_CYCLIST,
    "motorcyclist": AGENT_CYCLIST,
}

CONTROL_STATE_CONTROLLABLE = 0
CONTROL_STATE_REPLAYED = 1

LANE_FREEWAY = 1
LANE_SURFACE_STREET = 2
DRIVABLE_LANE_TYPES = (LANE_FREEWAY, LANE_SURFACE_STREET)
MISC_UNKNOWN = 30
MISC_CROSSWALK = 31
MISC_SPEED_BUMP = 32
WOMD_ELEMENT_TO_PUFFER_TYPE = {
    "lane": {0: 0, 1: 1, 2: 2, 3: 3},
    "road_line": {5: 10, 6: 11, 7: 12, 8: 13, 9: 14, 10: 15, 11: 16, 12: 17, 13: 18},
    "road_edge": {14: 20, 15: 21, 16: 22},
}
AREA_TYPE_BY_NAME = {"crosswalk": MISC_CROSSWALK, "speed_bump": MISC_SPEED_BUMP, "driveway": MISC_UNKNOWN}

SCENARIO_ID_BYTES = 128
DATASET_NAME_BYTES = 32
UNKNOWN_SPEED_LIMIT_MPS = -1.0
DUPLICATE_POINT_EPS_M = 1.0e-3

LANE_CONNECT_DISTANCE_M = 0.5
LANE_CONNECT_MIN_COS = 0.5
ROUTE_CANDIDATE_MAX_DISTANCE_M = 7.0
ROUTE_CANDIDATE_MIN_COS = 0.3
ROUTE_CANDIDATES_PER_POINT = 7
ROUTE_MAX_CONNECTOR_HOPS = 3
ROUTE_SKIP_POINT_COST = 50.0
ROUTE_HOP_COST = 1.0
ROUTE_OFFROAD_DISTANCE_M = 5.0
ROUTE_STATIONARY_OFFROAD_DISTANCE_M = 0.3
ROUTE_STATIONARY_SPEED_MPS = 0.5
MAX_ROUTE_LENGTH = 64  # matches constants.h; caps the post-GT extension


def _finite(value, what):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{what} must be a finite number, got {value!r}")
    return float(value)


def _xyz_array(points, what):
    if not isinstance(points, list):
        raise ValueError(f"{what} must be a list of {{x,y,z}} points")
    return np.array(
        [[_finite(p["x"], what), _finite(p["y"], what), _finite(p.get("z", 0.0), what)] for p in points],
        dtype=np.float64,
    )


def _polyline_frame(points_xyz):
    """Drop repeated points, then per-point forward headings and cumulative arc length."""
    step_m = np.hypot(*np.diff(points_xyz[:, :2], axis=0).T)
    keep = np.concatenate([[True], step_m > DUPLICATE_POINT_EPS_M])
    points_xyz = points_xyz[keep]
    deltas = np.diff(points_xyz[:, :2], axis=0)
    segment_headings = np.arctan2(deltas[:, 1], deltas[:, 0])
    headings = np.append(segment_headings, segment_headings[-1:]) if len(deltas) else np.zeros(len(points_xyz))
    cum_lengths = np.concatenate([[0.0], np.cumsum(np.hypot(deltas[:, 0], deltas[:, 1]))])
    return points_xyz, headings, cum_lengths


def _road_type(road, road_idx):
    type_name = road.get("type")
    if type_name in AREA_TYPE_BY_NAME:
        return AREA_TYPE_BY_NAME[type_name]
    if type_name not in WOMD_ELEMENT_TO_PUFFER_TYPE:
        raise ValueError(f"road {road_idx} (id={road.get('id')}): unsupported type {type_name!r}")
    element_id = road.get("map_element_id")
    puffer_type = WOMD_ELEMENT_TO_PUFFER_TYPE[type_name].get(element_id)
    if puffer_type is None:
        raise ValueError(f"road {road_idx} (id={road.get('id')}): {type_name} has unknown map_element_id {element_id}")
    return puffer_type


def build_roads(roads_json):
    """Road elements reindexed to their list position; lanes get entry/exit lanes, speed limit and arc length."""
    roads = []
    for road_idx, road in enumerate(roads_json):
        road_type = _road_type(road, road_idx)
        points_xyz, headings, cum_lengths = _polyline_frame(_xyz_array(road.get("geometry"), f"road {road_idx}"))
        if len(points_xyz) < 2:
            raise ValueError(f"road {road_idx} (id={road.get('id')}): needs at least 2 distinct points")
        element = {
            "id": road_idx,
            "type": road_type,
            "S": len(points_xyz),
            "x": points_xyz[:, 0].astype(np.float32).tolist(),
            "y": points_xyz[:, 1].astype(np.float32).tolist(),
            "z": points_xyz[:, 2].astype(np.float32).tolist(),
            "headings": headings.astype(np.float32).tolist(),
        }
        if 0 <= road_type <= 9:
            speed_limit_mps = road.get("speed_limit")
            if speed_limit_mps is None:
                speed_limit_mps = UNKNOWN_SPEED_LIMIT_MPS
            elif _finite(speed_limit_mps, f"road {road_idx} speed_limit") <= 0.0:
                raise ValueError(f"road {road_idx} (id={road.get('id')}): speed_limit must be > 0")
            element.update(
                speed_limit=float(speed_limit_mps),
                length=float(cum_lengths[-1]),
                cum_lengths=cum_lengths.astype(np.float32).tolist(),
                entry_lanes=[],
                exit_lanes=[],
                _points_xy=points_xyz[:, :2],
            )
        roads.append(element)
    _connect_lanes([road for road in roads if "_points_xy" in road])
    return roads


def _connect_lanes(lanes):
    if not lanes:
        return
    ends_xy = np.array([lane["_points_xy"][-1] for lane in lanes])
    starts_xy = np.array([lane["_points_xy"][0] for lane in lanes])
    end_dirs = np.array([_unit(lane["_points_xy"][-1] - lane["_points_xy"][-2]) for lane in lanes])
    start_dirs = np.array([_unit(lane["_points_xy"][1] - lane["_points_xy"][0]) for lane in lanes])
    gap_m = np.linalg.norm(ends_xy[:, None, :] - starts_xy[None, :, :], axis=2)
    aligned = end_dirs @ start_dirs.T >= LANE_CONNECT_MIN_COS
    connected = (gap_m <= LANE_CONNECT_DISTANCE_M) & aligned
    np.fill_diagonal(connected, False)
    for from_pos, to_pos in zip(*np.nonzero(connected)):
        lanes[from_pos]["exit_lanes"].append(lanes[to_pos]["id"])
        lanes[to_pos]["entry_lanes"].append(lanes[from_pos]["id"])


def _unit(vector):
    return vector / np.linalg.norm(vector)


def _lane_distances_from(roads, source_id, drivable_ids):
    best_m = {source_id: 0.0}
    frontier = [(0.0, source_id)]
    while frontier:
        dist_m, lane_id = heapq.heappop(frontier)
        if dist_m > best_m[lane_id]:
            continue
        next_dist_m = dist_m + roads[lane_id]["length"]
        for exit_id in roads[lane_id]["exit_lanes"]:
            if exit_id in drivable_ids and next_dist_m < best_m.get(exit_id, np.inf):
                best_m[exit_id] = next_dist_m
                heapq.heappush(frontier, (next_dist_m, exit_id))
    return best_m


def build_lane_graph(roads):
    """All-pairs lane distances over drivable lanes (edge weight = source lane length; inf = unreachable)."""
    lane_ids = [road["id"] for road in roads if road["type"] in DRIVABLE_LANE_TYPES]
    graph_idx = {lane_id: idx for idx, lane_id in enumerate(lane_ids)}
    distances = np.full((len(lane_ids), len(lane_ids)), np.inf, dtype=np.float32)
    for source_idx, source_id in enumerate(lane_ids):
        for lane_id, dist_m in _lane_distances_from(roads, source_id, graph_idx).items():
            distances[source_idx, graph_idx[lane_id]] = dist_m
    return {"n": len(lane_ids), "lane_ids": lane_ids, "distances": distances.ravel().tolist()}


class LaneMatcher:
    """Projects trajectory points onto every lane at once (all lane segments stacked)."""

    def __init__(self, roads):
        self.lane_ids = [road["id"] for road in roads if "_points_xy" in road]
        segment_starts, segment_vectors, segment_owner = [], [], []
        for lane_pos, lane_id in enumerate(self.lane_ids):
            points_xy = roads[lane_id]["_points_xy"]
            segment_starts.append(points_xy[:-1])
            segment_vectors.append(np.diff(points_xy, axis=0))
            segment_owner.append(np.full(len(points_xy) - 1, lane_pos))
        self.has_lanes = bool(self.lane_ids)
        if not self.has_lanes:
            return
        self.segment_starts = np.concatenate(segment_starts)
        self.segment_vectors = np.concatenate(segment_vectors)
        self.segment_owner = np.concatenate(segment_owner)
        self.lane_first_segment = np.searchsorted(self.segment_owner, np.arange(len(self.lane_ids)))
        segment_lengths = np.linalg.norm(self.segment_vectors, axis=1)
        self.segment_dirs = self.segment_vectors / segment_lengths[:, None]

    def lane_distances(self, points_xy, headings):
        """Per point and lane: distance to the lane and cos between point heading and lane tangent there."""
        offsets = points_xy[:, None, :] - self.segment_starts[None, :, :]
        lengths_sq = np.einsum("ij,ij->i", self.segment_vectors, self.segment_vectors)
        along = np.clip(np.einsum("tij,ij->ti", offsets, self.segment_vectors) / lengths_sq, 0.0, 1.0)
        nearest = self.segment_starts[None] + along[..., None] * self.segment_vectors[None]
        segment_dist_m = np.linalg.norm(points_xy[:, None, :] - nearest, axis=2)
        heading_dirs = np.stack([np.cos(headings), np.sin(headings)], axis=1)
        segment_cos = heading_dirs @ self.segment_dirs.T
        lane_dist_m = np.minimum.reduceat(segment_dist_m, self.lane_first_segment, axis=1)
        is_lane_min = segment_dist_m == lane_dist_m[:, self.segment_owner]
        masked_cos = np.where(is_lane_min, segment_cos, -np.inf)
        lane_cos = np.maximum.reduceat(masked_cos, self.lane_first_segment, axis=1)
        return lane_dist_m, lane_cos


def _connector_paths_from(roads, source_id):
    """Fewest-hop exit-graph paths (lane lists, source included) of at most ROUTE_MAX_CONNECTOR_HOPS hops."""
    reached = {source_id: [source_id]}
    frontier = deque([source_id])
    while frontier:
        lane_id = frontier.popleft()
        if len(reached[lane_id]) > ROUTE_MAX_CONNECTOR_HOPS:
            continue
        for exit_id in roads[lane_id]["exit_lanes"]:
            if exit_id not in reached:
                reached[exit_id] = reached[lane_id] + [exit_id]
                frontier.append(exit_id)
    return reached


def _point_candidates(lane_dist_m, lane_cos, lane_ids):
    candidates = []
    for point_dist_m, point_cos in zip(lane_dist_m, lane_cos):
        eligible = np.nonzero((point_dist_m <= ROUTE_CANDIDATE_MAX_DISTANCE_M) & (point_cos > ROUTE_CANDIDATE_MIN_COS))[
            0
        ]
        ranked = eligible[np.lexsort((eligible, -point_cos[eligible], point_dist_m[eligible]))]
        candidates.append([(lane_ids[pos], float(point_dist_m[pos])) for pos in ranked[:ROUTE_CANDIDATES_PER_POINT]])
    return candidates


def _best_predecessor(lane_id, cost_by_lane, connector_paths, fresh_start_cost):
    """Cheapest way to be on lane_id: start fresh (None) or hop from a lane matched at the previous point."""
    best_cost, best_prev = fresh_start_cost, None
    for prev_id, prev_cost in cost_by_lane.items():
        path = connector_paths[prev_id].get(lane_id)
        if path is None:
            continue
        cost = prev_cost + ROUTE_HOP_COST * (len(path) - 1)
        if cost < best_cost:
            best_cost, best_prev = cost, prev_id
    return best_cost, best_prev


def _match_lane_sequence(candidates, connector_paths):
    """DP over points: each point is matched to a candidate lane reachable from the previous one, or skipped."""
    cost_by_lane = {}
    backpointers = []
    skipped_prefix_cost = 0.0
    for point_candidates in candidates:
        next_cost = {lane_id: cost + ROUTE_SKIP_POINT_COST for lane_id, cost in cost_by_lane.items()}
        step_back = {lane_id: lane_id for lane_id in cost_by_lane}
        for lane_id, dist_m in point_candidates:
            prev_cost, prev_id = _best_predecessor(lane_id, cost_by_lane, connector_paths, skipped_prefix_cost)
            if prev_cost + dist_m < next_cost.get(lane_id, np.inf):
                next_cost[lane_id], step_back[lane_id] = prev_cost + dist_m, prev_id
        skipped_prefix_cost += ROUTE_SKIP_POINT_COST
        cost_by_lane = next_cost
        backpointers.append(step_back)
    if not cost_by_lane:
        return []
    lane_id = min(cost_by_lane, key=lambda candidate: (cost_by_lane[candidate], candidate))
    lane_sequence = []
    for step_back in reversed(backpointers):
        if lane_id is None:
            break
        if not lane_sequence or lane_sequence[-1] != lane_id:
            lane_sequence.append(lane_id)
        lane_id = step_back[lane_id]
    return lane_sequence[::-1]


def _expand_connectors(lane_sequence, connector_paths):
    route = lane_sequence[:1]
    for next_id in lane_sequence[1:]:
        route.extend(connector_paths[route[-1]][next_id][1:])
    return route


def _extend_to_dead_end(route, roads):
    visited = set(route)
    while len(route) < MAX_ROUTE_LENGTH:
        lane = roads[route[-1]]
        end_dir = _unit(lane["_points_xy"][-1] - lane["_points_xy"][-2])
        exits = [exit_id for exit_id in lane["exit_lanes"] if exit_id not in visited]
        if not exits:
            break
        exit_points = [roads[exit_id]["_points_xy"] for exit_id in exits]
        alignment = [float(end_dir @ _unit(points[1] - points[0])) for points in exit_points]
        best_exit = min(zip(exits, alignment), key=lambda pair: (-pair[1], pair[0]))[0]
        route.append(best_exit)
        visited.add(best_exit)
    return route


def compute_route(agent, matcher, connector_paths, roads, is_ego):
    """Lane route for a vehicle and its GT-supported prefix length; ([], 0) when ineligible."""
    if agent["type"] != AGENT_VEHICLE or not matcher.has_lanes:
        return [], 0
    valid_steps = np.nonzero(agent["valid_mask"])[0]
    if len(valid_steps) == 0:
        return [], 0
    points_xy = agent["xy"][valid_steps]
    lane_dist_m, lane_cos = matcher.lane_distances(points_xy, agent["heading"][valid_steps])
    first_speed_mps = float(np.hypot(*agent["velocity_xy"][valid_steps[0]]))
    offroad_limit_m = (
        ROUTE_STATIONARY_OFFROAD_DISTANCE_M
        if first_speed_mps < ROUTE_STATIONARY_SPEED_MPS
        else ROUTE_OFFROAD_DISTANCE_M
    )
    if not is_ego and lane_dist_m[0].min() > offroad_limit_m:
        return [], 0
    lane_sequence = _match_lane_sequence(_point_candidates(lane_dist_m, lane_cos, matcher.lane_ids), connector_paths)
    if not lane_sequence:
        return [], 0
    gt_route = _expand_connectors(lane_sequence, connector_paths)
    return _extend_to_dead_end(list(gt_route), roads), len(gt_route)


def parse_agent(obj, obj_idx, num_timesteps):
    what = f"object {obj_idx} (id={obj.get('id')})"
    type_name = obj.get("type")
    if type_name not in AGENT_TYPE_BY_NAME:
        raise ValueError(f"{what}: unsupported type {type_name!r}")
    for key in ("position", "heading", "velocity", "valid"):
        if not isinstance(obj.get(key), list) or len(obj[key]) != num_timesteps:
            raise ValueError(f"{what}: '{key}' must have num_timesteps={num_timesteps} entries")
    dimensions_m = [_finite(obj.get(key), f"{what} {key}") for key in ("length", "width", "height")]
    if min(dimensions_m) <= 0.0:
        raise ValueError(f"{what}: length/width/height must be > 0")
    if not all(isinstance(flag, bool) for flag in obj["valid"]):
        raise ValueError(f"{what}: 'valid' entries must be booleans")
    if not isinstance(obj.get("mark_as_expert", False), bool):
        raise ValueError(f"{what}: 'mark_as_expert' must be a boolean")
    position_xyz = _xyz_array(obj["position"], f"{what} position")
    velocity = obj["velocity"]
    velocity_xy = np.array(
        [[_finite(v["x"], f"{what} velocity"), _finite(v["y"], f"{what} velocity")] for v in velocity]
    )
    heading = np.array([_finite(h, f"{what} heading") for h in obj["heading"]])
    return {
        "type": AGENT_TYPE_BY_NAME[type_name],
        "xyz": position_xyz,
        "xy": position_xyz[:, :2],
        "heading": heading,
        "velocity_xy": velocity_xy,
        "valid_mask": np.array(obj["valid"], dtype=bool),
        "dimensions_m": dimensions_m,
        "is_expert": obj.get("mark_as_expert", False),
    }


def _agent_record(agent_idx, agent, route, route_gt_len, is_ego):
    num_timesteps = len(agent["valid_mask"])
    valid_steps = np.nonzero(agent["valid_mask"])[0]
    goal_xyz = agent["xyz"][valid_steps[-1]] if len(valid_steps) else np.zeros(3)
    length_m, width_m, height_m = agent["dimensions_m"]
    columns = {
        "x": agent["xyz"][:, 0],
        "y": agent["xyz"][:, 1],
        "z": agent["xyz"][:, 2],
        "h": agent["heading"],
        "vx": agent["velocity_xy"][:, 0],
        "vy": agent["velocity_xy"][:, 1],
        "len": np.full(num_timesteps, length_m),
        "wid": np.full(num_timesteps, width_m),
        "hgt": np.full(num_timesteps, height_m),
    }
    cols = {key: values.astype(np.float32).tolist() for key, values in columns.items()}
    cols["valid"] = agent["valid_mask"].astype(int).tolist()
    replayed = agent["is_expert"] and not is_ego
    return {
        "id": agent_idx,
        "type": agent["type"],
        "T": num_timesteps,
        "cols": cols,
        "route": route,
        "route_gt_len": route_gt_len,
        "goal": tuple(float(v) for v in np.float32(goal_xyz)),
        "mark_as_expert": CONTROL_STATE_REPLAYED if replayed else CONTROL_STATE_CONTROLLABLE,
    }


def _padded_bytes(text, size, what):
    encoded = text.encode("utf-8")
    if len(encoded) >= size:
        raise ValueError(f"{what} {text!r} exceeds {size - 1} bytes")
    return encoded.ljust(size, b"\0")


def json_to_bin_data(scenario, scenario_id, dataset_name):
    """Validate a JSON scenario and build the dict mirror_map_bin.write_bin serializes."""
    metadata = scenario.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("missing 'metadata'")
    sim_dt_s = _finite(metadata.get("sim_dt"), "metadata.sim_dt")
    num_timesteps = metadata.get("num_timesteps")
    if sim_dt_s <= 0.0 or not isinstance(num_timesteps, int) or num_timesteps <= 0:
        raise ValueError("metadata.sim_dt and metadata.num_timesteps must be positive")
    objects = scenario.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError("'objects' must be a non-empty list")
    sdc_idx = metadata.get("sdc_track_index")
    if not isinstance(sdc_idx, int) or not 0 <= sdc_idx < len(objects):
        raise ValueError(f"metadata.sdc_track_index {sdc_idx!r} out of range for {len(objects)} objects")
    if not isinstance(scenario.get("roads"), list):
        raise ValueError("'roads' must be a list")

    roads = build_roads(scenario["roads"])
    lane_graph = build_lane_graph(roads)
    matcher = LaneMatcher(roads)
    connector_paths = {lane_id: _connector_paths_from(roads, lane_id) for lane_id in matcher.lane_ids}

    object_order = [sdc_idx] + [obj_idx for obj_idx in range(len(objects)) if obj_idx != sdc_idx]
    agents = []
    for agent_idx, obj_idx in enumerate(object_order):
        agent = parse_agent(objects[obj_idx], obj_idx, num_timesteps)
        is_ego = agent_idx == 0
        route, route_gt_len = compute_route(agent, matcher, connector_paths, roads, is_ego)
        if is_ego and not route:
            raise ValueError(f"SDC (object {sdc_idx}) could not be matched to a lane route")
        agents.append(_agent_record(agent_idx, agent, route, route_gt_len, is_ego))

    for road in roads:
        road.pop("_points_xy", None)
    return {
        "agents": agents,
        "roads": roads,
        "traffic": [],
        "objects": [],
        "lane_graph": lane_graph,
        "scenario_id": _padded_bytes(scenario_id, SCENARIO_ID_BYTES, "scenario_id"),
        "dataset_name": _padded_bytes(dataset_name, DATASET_NAME_BYTES, "dataset name"),
        "log_length": num_timesteps,
        "log_dt": sim_dt_s,
        "objects_of_interest": (),
        "tracks_to_predict": (),
        "has_phase_section": False,
        "has_width_section": False,
        "has_zone_section": False,
    }


def output_path_for(json_path, output_dir, dataset_name, scenario_id):
    if "/" in scenario_id or not scenario_id:
        raise ValueError(f"{json_path}: scenario_id {scenario_id!r} is not a valid file name")
    return Path(output_dir) / f"{dataset_name}__{scenario_id}.bin"


def convert_file(json_path, output_dir, dataset_name):
    with open(json_path) as file:
        scenario = json.load(file)
    scenario_id = str(scenario.get("scenario_id") or Path(json_path).stem)
    output_path = output_path_for(json_path, output_dir, dataset_name, scenario_id)
    write_bin(json_to_bin_data(scenario, scenario_id, dataset_name), output_path)
    return output_path


def _worker(job):
    json_path, output_dir, dataset_name = job
    try:
        return str(json_path), str(convert_file(json_path, output_dir, dataset_name)), None
    except ValueError as error:
        return str(json_path), None, str(error)
    except Exception:  # noqa: BLE001 - malformed JSON structure; reported per file, run exits non-zero
        return str(json_path), None, traceback.format_exc(limit=2)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert WOMD-style JSON scenarios to PufferDrive 3.0 .bin")
    parser.add_argument("--input", required=True, help="JSON file or directory of *.json")
    parser.add_argument("--output", required=True, help="Output directory for .bin files")
    parser.add_argument("--dataset-name", default="json", help="Dataset tag stored in the bin and file prefix")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)

    input_path = Path(args.input).expanduser()
    json_paths = sorted(input_path.glob("*.json")) if input_path.is_dir() else [input_path]
    if not json_paths or not all(path.is_file() for path in json_paths):
        parser.error(f"no JSON files found at {input_path}")
    _padded_bytes(args.dataset_name, DATASET_NAME_BYTES, "dataset name")
    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(path, output_dir, args.dataset_name) for path in json_paths]
    if args.workers > 1:
        with multiprocessing.Pool(args.workers) as pool:
            results = pool.map(_worker, jobs)
    else:
        results = [_worker(job) for job in jobs]
    failures = [(json_path, error) for json_path, _, error in results if error]
    for json_path, error in failures:
        print(f"[FAILED] {json_path}: {error}", file=sys.stderr)
    print(f"Converted {len(results) - len(failures)}/{len(results)} scenarios -> {output_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
