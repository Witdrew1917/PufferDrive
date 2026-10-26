#!/usr/bin/env python3
"""generate_highway_loop_map.py: procedural 5-lane clockwise highway loop with ramps (PufferDrive 3.0 .bin).

Layout (deterministic in --seed):
  * rounded-rectangle loop driven clockwise (right turns only); the long side of the outer road edge is
    slightly longer than the largest CARLA town, the short side is SHORT_SIDE_SCALE of it
  * corners are sized for CORNER_SPEED_KMH at CORNER_LATERAL_ACCEL_MPS2 and carry that speed limit
  * each straight has 1-2 single-lane ramps; a ramp splits off the leftmost (outside) or rightmost (inside)
    lane, runs beside the highway and merges back into the same lane before the next corner
  * unevenly spaced stretches use solid white dividers, which process_navigation.py treats as
    no-lane-change; elsewhere dividers are broken white
  * all lanes break at shared stations so lateral neighbours line up, and the lane graph is a closed loop
    (no dead ends)

Usage:
    python data_utils/generate_highway_loop_map.py \\
        --output pufferlib/resources/drive/binaries/carla/synthetic__highway_loop.bin --seed 1337
"""

import argparse
import itertools
import math
import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))
from json_to_bin import DATASET_NAME_BYTES, SCENARIO_ID_BYTES, _padded_bytes, build_lane_graph, build_roads
from mirror_map_bin import write_bin


KMH_TO_MPS = 1.0 / 3.6

LANE_COUNT = 5
LANE_WIDTH_M = 3.5
ROAD_WIDTH_M = LANE_COUNT * LANE_WIDTH_M
LEFTMOST_LANE_IDX = 0
RIGHTMOST_LANE_IDX = LANE_COUNT - 1

OUTER_LONG_SIDE_M = 1200.0
SHORT_SIDE_SCALE = 0.7
CORNER_SPEED_KMH = 70.0
CORNER_LATERAL_ACCEL_MPS2 = 2.0
HIGHWAY_SPEED_KMH = 110.0
RAMP_SPEED_KMH = 60.0

RAMP_LATERAL_OFFSET_M = 25.0
RAMP_TAPER_LENGTH_M = 120.0
RAMP_PARALLEL_MIN_M = 40.0
RAMP_PARALLEL_MAX_M = 160.0
RAMP_CLEARANCE_AFTER_CORNER_M = 40.0
RAMP_CLEARANCE_BEFORE_CORNER_M = 60.0
RAMP_MIN_GAP_M = 40.0
MAX_RAMPS_PER_STRAIGHT = 2

SOLID_SEGMENT_COUNT = 6
SOLID_LENGTH_MIN_M = 80.0
SOLID_LENGTH_MAX_M = 320.0
SOLID_MIN_GAP_M = 60.0

# Sections adjacent to a split/merge span the whole taper so navigation's lateral probes clear the overlap.
MAX_LANE_SECTION_M = 125.0
MIN_SECTION_M = 5.0
POINT_SPACING_M = 5.0

# WOMD map_element_id values understood by json_to_bin.build_roads
WOMD_LANE_SURFACE_STREET = 2
WOMD_LINE_BROKEN_SINGLE_WHITE = 6
WOMD_LINE_SOLID_SINGLE_WHITE = 7
WOMD_EDGE_BOUNDARY = 15

PIECE_STRAIGHT = 0
PIECE_CORNER = 1
RAMP_SIDE_LEFT = +1.0
RAMP_SIDE_RIGHT = -1.0

DEFAULT_SEED = 1337
SCENARIO_ID = "highway_loop"
DATASET_NAME = "synthetic"


class LoopReference:
    """Inner road edge of the loop parameterized by station (arc length, metres); left normal points outward."""

    def __init__(self, inner_half_x_m, inner_half_y_m, inner_radius_m):
        self.radius_m = inner_radius_m
        corner_center_x_m = inner_half_x_m - inner_radius_m
        corner_center_y_m = inner_half_y_m - inner_radius_m
        straight_x_m = 2.0 * corner_center_x_m
        straight_y_m = 2.0 * corner_center_y_m
        corner_length_m = inner_radius_m * math.pi / 2.0
        # Clockwise from the top-left: east, south, west, north; each straight is followed by its corner.
        straights = [
            ((-corner_center_x_m, inner_half_y_m), 0.0, straight_x_m, (corner_center_x_m, corner_center_y_m)),
            ((inner_half_x_m, corner_center_y_m), -math.pi / 2, straight_y_m, (corner_center_x_m, -corner_center_y_m)),
            ((corner_center_x_m, -inner_half_y_m), math.pi, straight_x_m, (-corner_center_x_m, -corner_center_y_m)),
            ((-inner_half_x_m, -corner_center_y_m), math.pi / 2, straight_y_m, (-corner_center_x_m, corner_center_y_m)),
        ]
        self.pieces = []
        station_m = 0.0
        for start_xy, heading_rad, length_m, corner_center_xy in straights:
            self.pieces.append(
                {
                    "kind": PIECE_STRAIGHT,
                    "start_m": station_m,
                    "length_m": length_m,
                    "xy": start_xy,
                    "heading": heading_rad,
                }
            )
            station_m += length_m
            self.pieces.append(
                {
                    "kind": PIECE_CORNER,
                    "start_m": station_m,
                    "length_m": corner_length_m,
                    "xy": corner_center_xy,
                    "heading": heading_rad,
                }
            )
            station_m += corner_length_m
        self.total_length_m = station_m
        self.piece_starts_m = np.array([piece["start_m"] for piece in self.pieces])

    def piece_idx_at(self, station_m):
        return int(np.searchsorted(self.piece_starts_m, station_m, side="right") - 1)

    def offset_points(self, stations_m, lateral_offsets_m):
        """Points at the given stations shifted by lateral_offsets_m toward the outside of the loop."""
        stations_m = np.asarray(stations_m, dtype=np.float64)
        lateral_offsets_m = np.broadcast_to(np.asarray(lateral_offsets_m, dtype=np.float64), stations_m.shape)
        xs = np.empty_like(stations_m)
        ys = np.empty_like(stations_m)
        for point_idx, (station_m, offset_m) in enumerate(zip(stations_m, lateral_offsets_m)):
            piece = self.pieces[min(self.piece_idx_at(station_m), len(self.pieces) - 1)]
            local_m = station_m - piece["start_m"]
            if piece["kind"] == PIECE_STRAIGHT:
                heading_rad = piece["heading"]
                base_x = piece["xy"][0] + local_m * math.cos(heading_rad)
                base_y = piece["xy"][1] + local_m * math.sin(heading_rad)
            else:
                heading_rad = piece["heading"] - local_m / self.radius_m
                position_angle_rad = heading_rad + math.pi / 2
                base_x = piece["xy"][0] + self.radius_m * math.cos(position_angle_rad)
                base_y = piece["xy"][1] + self.radius_m * math.sin(position_angle_rad)
            xs[point_idx] = base_x - offset_m * math.sin(heading_rad)
            ys[point_idx] = base_y + offset_m * math.cos(heading_rad)
        return xs, ys


def lane_center_offset_m(lane_idx):
    """Lane 0 is the leftmost (outside) lane; offsets are measured outward from the inner road edge."""
    return (LANE_COUNT - lane_idx - 0.5) * LANE_WIDTH_M


def sample_stations(start_m, end_m):
    point_count = max(1, math.ceil((end_m - start_m) / POINT_SPACING_M))
    return np.linspace(start_m, end_m, point_count + 1)


def ramp_lateral_shift_m(ramp, stations_m):
    """Ramp centreline distance from its parent lane centre: cosine taper out, constant, cosine taper back."""
    stations_m = np.asarray(stations_m, dtype=np.float64)
    diverge_frac = np.clip((stations_m - ramp["split_m"]) / RAMP_TAPER_LENGTH_M, 0.0, 1.0)
    converge_frac = np.clip((ramp["merge_m"] - stations_m) / RAMP_TAPER_LENGTH_M, 0.0, 1.0)
    taper_frac = np.minimum(diverge_frac, converge_frac)
    return RAMP_LATERAL_OFFSET_M * 0.5 * (1.0 - np.cos(math.pi * taper_frac))


def ramp_parent_offset_m(ramp):
    parent_lane_idx = LEFTMOST_LANE_IDX if ramp["side"] == RAMP_SIDE_LEFT else RIGHTMOST_LANE_IDX
    return lane_center_offset_m(parent_lane_idx)


def ramp_gore_tip_taper_m():
    """Taper distance at which the ramp lane no longer overlaps its parent lane (shift == one lane width)."""
    return RAMP_TAPER_LENGTH_M * math.acos(1.0 - 2.0 * LANE_WIDTH_M / RAMP_LATERAL_OFFSET_M) / math.pi


def place_ramps(reference, rng):
    ramp_length_min_m = 2.0 * RAMP_TAPER_LENGTH_M + RAMP_PARALLEL_MIN_M
    ramps = []
    for piece in reference.pieces:
        if piece["kind"] != PIECE_STRAIGHT:
            continue
        usable_start_m = piece["start_m"] + RAMP_CLEARANCE_AFTER_CORNER_M
        usable_length_m = piece["length_m"] - RAMP_CLEARANCE_AFTER_CORNER_M - RAMP_CLEARANCE_BEFORE_CORNER_M
        max_fitting_count = int((usable_length_m + RAMP_MIN_GAP_M) // (ramp_length_min_m + RAMP_MIN_GAP_M))
        if max_fitting_count < 1:
            raise ValueError(f"straight of {piece['length_m']:.1f} m cannot fit a ramp")
        ramp_count = int(rng.integers(1, min(max_fitting_count, MAX_RAMPS_PER_STRAIGHT) + 1))
        parallel_budget_m = (usable_length_m - RAMP_MIN_GAP_M * (ramp_count - 1)) / ramp_count - 2 * RAMP_TAPER_LENGTH_M
        parallel_lengths_m = rng.uniform(RAMP_PARALLEL_MIN_M, min(RAMP_PARALLEL_MAX_M, parallel_budget_m), ramp_count)
        ramp_lengths_m = 2.0 * RAMP_TAPER_LENGTH_M + parallel_lengths_m
        slack_m = usable_length_m - ramp_lengths_m.sum() - RAMP_MIN_GAP_M * (ramp_count - 1)
        gaps_m = rng.dirichlet(np.ones(ramp_count + 1)) * slack_m
        split_m = usable_start_m + gaps_m[0]
        for ramp_idx in range(ramp_count):
            ramps.append({"split_m": split_m, "merge_m": split_m + ramp_lengths_m[ramp_idx]})
            split_m += ramp_lengths_m[ramp_idx] + RAMP_MIN_GAP_M + gaps_m[ramp_idx + 1]
    # Balanced left/right mix in a seeded order so both sides appear intermittently.
    sides = [RAMP_SIDE_LEFT, RAMP_SIDE_RIGHT] * math.ceil(len(ramps) / 2)
    for ramp, side in zip(ramps, rng.permutation(sides[: len(ramps)])):
        ramp["side"] = float(side)
    return ramps


def place_solid_segments(total_length_m, structural_stations_m, taper_windows, rng):
    lengths_m = rng.uniform(SOLID_LENGTH_MIN_M, SOLID_LENGTH_MAX_M, SOLID_SEGMENT_COUNT)
    slack_m = total_length_m - lengths_m.sum() - SOLID_MIN_GAP_M * (SOLID_SEGMENT_COUNT + 1)
    if slack_m <= 0.0:
        raise ValueError("solid segments do not fit on the loop")
    gaps_m = SOLID_MIN_GAP_M + rng.dirichlet(np.ones(SOLID_SEGMENT_COUNT + 1)) * slack_m
    structural_stations_m = np.asarray(structural_stations_m)
    segments = []
    start_m = 0.0
    for segment_idx in range(SOLID_SEGMENT_COUNT):
        start_m += gaps_m[segment_idx]
        end_m = start_m + lengths_m[segment_idx]
        snapped_start_m, snapped_end_m = start_m, end_m
        for window_start_m, window_end_m in taper_windows:
            if window_start_m < snapped_start_m < window_end_m:
                snapped_start_m = window_start_m
            if window_start_m < snapped_end_m < window_end_m:
                snapped_end_m = window_end_m
        snapped = []
        for station_m in (snapped_start_m, snapped_end_m):
            nearest_m = structural_stations_m[np.argmin(np.abs(structural_stations_m - station_m))]
            snapped.append(nearest_m if abs(nearest_m - station_m) < MIN_SECTION_M else station_m)
        segments.append(tuple(snapped))
        start_m = end_m
    return segments


def section_bounds(event_stations_m, total_length_m):
    """Sorted unique stations covering [0, total], subdivided so no section exceeds MAX_LANE_SECTION_M."""
    events_m = np.unique(np.concatenate([[0.0, total_length_m], np.asarray(event_stations_m)]))
    if np.any(np.diff(events_m) < MIN_SECTION_M):
        raise ValueError("section events closer than MIN_SECTION_M; adjust layout constants or seed")
    bounds_m = [0.0]
    for start_m, end_m in itertools.pairwise(events_m):
        piece_count = math.ceil((end_m - start_m) / MAX_LANE_SECTION_M)
        bounds_m.extend(np.linspace(start_m, end_m, piece_count + 1)[1:].tolist())
    return np.array(bounds_m)


def _road(road_type, map_element_id, xs, ys, **extra):
    return {
        "type": road_type,
        "map_element_id": map_element_id,
        "geometry": [{"x": float(x), "y": float(y), "z": 0.0} for x, y in zip(xs, ys)],
        **extra,
    }


def _inside_any(station_m, intervals):
    return any(start_m <= station_m <= end_m for start_m, end_m in intervals)


def highway_roads(reference, bounds_m, solid_segments):
    """Lanes (with speed zone = loop piece index) and lane dividers for every highway section."""
    roads, zones = [], []
    for start_m, end_m in itertools.pairwise(bounds_m):
        mid_m = 0.5 * (start_m + end_m)
        piece_idx = reference.piece_idx_at(mid_m)
        is_corner = reference.pieces[piece_idx]["kind"] == PIECE_CORNER
        speed_mps = (CORNER_SPEED_KMH if is_corner else HIGHWAY_SPEED_KMH) * KMH_TO_MPS
        stations_m = sample_stations(start_m, end_m)
        for lane_idx in range(LANE_COUNT):
            xs, ys = reference.offset_points(stations_m, lane_center_offset_m(lane_idx))
            roads.append(_road("lane", WOMD_LANE_SURFACE_STREET, xs, ys, speed_limit=speed_mps))
            zones.append(piece_idx)
        divider_id = (
            WOMD_LINE_SOLID_SINGLE_WHITE if _inside_any(mid_m, solid_segments) else WOMD_LINE_BROKEN_SINGLE_WHITE
        )
        for divider_idx in range(1, LANE_COUNT):
            xs, ys = reference.offset_points(stations_m, divider_idx * LANE_WIDTH_M)
            roads.append(_road("road_line", divider_id, xs, ys))
    return roads, zones


def ramp_roads(reference, ramps, zone_offset):
    roads, zones = [], []
    speed_mps = RAMP_SPEED_KMH * KMH_TO_MPS
    gore_tip_m = ramp_gore_tip_taper_m()
    for ramp_idx, ramp in enumerate(ramps):
        parent_offset_m = ramp_parent_offset_m(ramp)
        side = ramp["side"]
        events_m = [ramp["split_m"] + RAMP_TAPER_LENGTH_M, ramp["merge_m"] - RAMP_TAPER_LENGTH_M]
        local_bounds_m = section_bounds(np.array(events_m) - ramp["split_m"], ramp["merge_m"] - ramp["split_m"])
        for start_m, end_m in zip(local_bounds_m[:-1] + ramp["split_m"], local_bounds_m[1:] + ramp["split_m"]):
            stations_m = sample_stations(start_m, end_m)
            offsets_m = parent_offset_m + side * ramp_lateral_shift_m(ramp, stations_m)
            xs, ys = reference.offset_points(stations_m, offsets_m)
            roads.append(_road("lane", WOMD_LANE_SURFACE_STREET, xs, ys, speed_limit=speed_mps))
            zones.append(zone_offset + ramp_idx)

        stations_m = sample_stations(ramp["split_m"], ramp["merge_m"])
        far_offsets_m = parent_offset_m + side * (ramp_lateral_shift_m(ramp, stations_m) + 0.5 * LANE_WIDTH_M)
        roads.append(_road("road_edge", WOMD_EDGE_BOUNDARY, *reference.offset_points(stations_m, far_offsets_m)))

        # Gore: highway edge forward, then the ramp's near edge back to the tip, closing the island.
        gore_stations_m = sample_stations(ramp["split_m"] + gore_tip_m, ramp["merge_m"] - gore_tip_m)
        highway_edge_offset_m = parent_offset_m + side * 0.5 * LANE_WIDTH_M
        near_offsets_m = parent_offset_m + side * (
            ramp_lateral_shift_m(ramp, gore_stations_m[::-1]) - 0.5 * LANE_WIDTH_M
        )
        edge_xs, edge_ys = reference.offset_points(gore_stations_m, highway_edge_offset_m)
        near_xs, near_ys = reference.offset_points(gore_stations_m[::-1], near_offsets_m)
        gore_xs = np.concatenate([edge_xs, near_xs[1:]])
        gore_ys = np.concatenate([edge_ys, near_ys[1:]])
        roads.append(_road("road_edge", WOMD_EDGE_BOUNDARY, gore_xs, gore_ys))
    return roads, zones


def highway_edge_roads(reference, ramps):
    """Inner and outer road edges, open wherever a ramp leaves and rejoins on that side."""
    roads = []
    for side, edge_offset_m in ((RAMP_SIDE_RIGHT, 0.0), (RAMP_SIDE_LEFT, ROAD_WIDTH_M)):
        openings = sorted((ramp["split_m"], ramp["merge_m"]) for ramp in ramps if ramp["side"] == side)
        run_start_m = 0.0
        for opening_start_m, opening_end_m in openings + [(reference.total_length_m, reference.total_length_m)]:
            if opening_start_m > run_start_m:
                stations_m = sample_stations(run_start_m, opening_start_m)
                roads.append(
                    _road("road_edge", WOMD_EDGE_BOUNDARY, *reference.offset_points(stations_m, edge_offset_m))
                )
            run_start_m = opening_end_m
    return roads


def build_highway_loop(seed):
    rng = np.random.default_rng(seed)
    outer_half_x_m = 0.5 * OUTER_LONG_SIDE_M
    outer_half_y_m = 0.5 * OUTER_LONG_SIDE_M * SHORT_SIDE_SCALE
    corner_speed_mps = CORNER_SPEED_KMH * KMH_TO_MPS
    innermost_lane_radius_m = corner_speed_mps**2 / CORNER_LATERAL_ACCEL_MPS2
    inner_radius_m = innermost_lane_radius_m - lane_center_offset_m(RIGHTMOST_LANE_IDX)
    reference = LoopReference(outer_half_x_m - ROAD_WIDTH_M, outer_half_y_m - ROAD_WIDTH_M, inner_radius_m)

    ramps = place_ramps(reference, rng)
    taper_windows = [(r["split_m"], r["split_m"] + RAMP_TAPER_LENGTH_M) for r in ramps]
    taper_windows += [(r["merge_m"] - RAMP_TAPER_LENGTH_M, r["merge_m"]) for r in ramps]
    structural_m = np.concatenate([reference.piece_starts_m, [reference.total_length_m], np.ravel(taper_windows)])
    solid_segments = place_solid_segments(reference.total_length_m, structural_m, taper_windows, rng)
    solid_ends_m = [station_m for segment in solid_segments for station_m in segment]
    bounds_m = section_bounds(np.concatenate([structural_m, solid_ends_m]), reference.total_length_m)

    highway, highway_zones = highway_roads(reference, bounds_m, solid_segments)
    ramp_elements, ramp_zones = ramp_roads(reference, ramps, len(reference.pieces))
    roads_json = highway + ramp_elements + highway_edge_roads(reference, ramps)
    roads = build_roads(roads_json)

    lane_zones = iter(highway_zones + ramp_zones)
    for road in roads:
        road.pop("_points_xy", None)
        if 0 <= road["type"] <= 9:
            road["widths"] = [LANE_WIDTH_M] * road["S"]
            road["speed_zone_idx"] = next(lane_zones)
            if not road["entry_lanes"] or not road["exit_lanes"]:
                raise ValueError(f"lane {road['id']} is not part of the closed loop network")

    data = {
        "agents": [],
        "roads": roads,
        "traffic": [],
        "objects": [],
        "lane_graph": build_lane_graph(roads),
        "scenario_id": _padded_bytes(SCENARIO_ID, SCENARIO_ID_BYTES, "scenario_id"),
        "dataset_name": _padded_bytes(DATASET_NAME, DATASET_NAME_BYTES, "dataset name"),
        "log_length": 0,
        "log_dt": 0.0,
        "objects_of_interest": (),
        "tracks_to_predict": (),
        "has_phase_section": True,
        "has_width_section": True,
        "has_zone_section": True,
    }
    layout = {"reference": reference, "ramps": ramps, "solid_segments": solid_segments}
    return data, layout


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, help="Output .bin path")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Seed for ramp and solid-line placement")
    args = parser.parse_args(argv)

    data, layout = build_highway_loop(args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_bin(data, output_path)

    reference = layout["reference"]
    lane_count = sum(1 for road in data["roads"] if 0 <= road["type"] <= 9)
    print(f"loop {reference.total_length_m:.1f} m (inner edge), {lane_count} lanes, {len(data['roads'])} road elements")
    for ramp in layout["ramps"]:
        side_name = "left" if ramp["side"] == RAMP_SIDE_LEFT else "right"
        print(f"  ramp {side_name:5s} split {ramp['split_m']:7.1f} m  merge {ramp['merge_m']:7.1f} m")
    for start_m, end_m in layout["solid_segments"]:
        print(f"  solid dividers {start_m:7.1f} - {end_m:7.1f} m")
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
