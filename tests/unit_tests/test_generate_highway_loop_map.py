"""Procedural highway loop map (data_utils/generate_highway_loop_map.py)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_utils import generate_highway_loop_map as hl
from data_utils.mirror_map_bin import read_bin, write_bin
from data_utils.navigation_bin import NAVI_LANE_CHANGE_COST_M as LANE_CHANGE_COST_M
from data_utils.process_navigation import build_navigation


SOLID_LINE_TYPE = 12
BROKEN_LINE_TYPE = 11


def _highway_map(tmp_path):
    data, layout = hl.build_highway_loop(hl.DEFAULT_SEED)
    bin_path = tmp_path / "highway_loop.bin"
    write_bin(data, bin_path)
    return read_bin(bin_path), layout


def test_ramps_split_and_merge_on_both_sides(tmp_path):
    data, layout = _highway_map(tmp_path)
    lanes = [road for road in data["roads"] if road["type"] <= 9]
    ramp_count = len(layout["ramps"])
    assert {ramp["side"] for ramp in layout["ramps"]} == {hl.RAMP_SIDE_LEFT, hl.RAMP_SIDE_RIGHT}
    assert all(lane["entry_lanes"] and lane["exit_lanes"] for lane in lanes)
    assert sum(len(lane["exit_lanes"]) == 2 for lane in lanes) == ramp_count
    assert sum(len(lane["entry_lanes"]) == 2 for lane in lanes) == ramp_count


def test_lane_changes_follow_divider_types(tmp_path):
    data, _ = _highway_map(tmp_path)
    roads = data["roads"]
    navi = build_navigation(data)
    navi_idx = {road_idx: n for n, road_idx in enumerate(navi["lane_road_idx"])}
    highway_lane_ids = [road["id"] for road in roads if road["type"] <= 9 and road["speed_zone_idx"] < 8]
    divider_types = set()
    for section_start in range(0, len(highway_lane_ids), hl.LANE_COUNT):
        section = [navi_idx[road_id] for road_id in highway_lane_ids[section_start : section_start + hl.LANE_COUNT]]
        divider_type = roads[highway_lane_ids[section_start + hl.LANE_COUNT - 1] + 1]["type"]
        divider_types.add(divider_type)
        legal_changes = sum(navi["left_lane"][n] >= 0 for n in section) + sum(
            navi["right_lane"][n] >= 0 for n in section
        )
        expected = 0 if divider_type == SOLID_LINE_TYPE else 2 * (hl.LANE_COUNT - 1)
        assert legal_changes == expected
    assert divider_types == {SOLID_LINE_TYPE, BROKEN_LINE_TYPE}


def test_ramp_goals_require_lane_changes_from_inner_lanes(tmp_path):
    data, _ = _highway_map(tmp_path)
    roads = data["roads"]
    navi = build_navigation(data)
    navi_idx = {road_idx: n for n, road_idx in enumerate(navi["lane_road_idx"])}
    highway_lane_ids = [road["id"] for road in roads if road["type"] <= 9 and road["speed_zone_idx"] < 8]
    checked_ramp_count = 0
    for section_start in range(0, len(highway_lane_ids), hl.LANE_COUNT):
        section = highway_lane_ids[section_start : section_start + hl.LANE_COUNT]
        divider_type = roads[section[-1] + 1]["type"]
        for parent_position, parent_id in enumerate(section):
            ramp_ids = [exit_id for exit_id in roads[parent_id]["exit_lanes"] if roads[exit_id]["speed_zone_idx"] >= 8]
            if not ramp_ids or divider_type == SOLID_LINE_TYPE:
                continue
            distance_row = navi["distance_to_goal_m"][navi_idx[ramp_ids[0]]]
            parent_distance_m = distance_row[navi_idx[parent_id]]
            for position, lane_id in enumerate(section):
                lane_changes = abs(position - parent_position)
                expected_m = parent_distance_m + lane_changes * LANE_CHANGE_COST_M
                assert distance_row[navi_idx[lane_id]] == pytest.approx(expected_m, abs=1e-2)
            checked_ramp_count += 1
    assert checked_ramp_count > 0
