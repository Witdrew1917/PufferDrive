"""WOMD-style JSON -> PufferDrive 3.0 .bin converter (data_utils/json_to_bin.py)."""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_utils import json_to_bin as j2b
from data_utils.mirror_map_bin import read_bin

NUM_TIMESTEPS = 91
SIM_DT_S = 0.1
SPEED_MPS = 8.0
LANE_LENGTH_M = 60.0


def _points(xs, ys):
    return [{"x": float(x), "y": float(y), "z": 0.0} for x, y in zip(xs, ys)]


def _object(object_type, start_x_m, y_m, speed_mps, mark_as_expert=True):
    xs = start_x_m + speed_mps * SIM_DT_S * np.arange(NUM_TIMESTEPS)
    return {
        "id": len(object_type),
        "type": object_type,
        "length": 4.5,
        "width": 2.0,
        "height": 1.5,
        "position": _points(xs, np.full(NUM_TIMESTEPS, y_m)),
        "heading": [0.0] * NUM_TIMESTEPS,
        "velocity": [{"x": speed_mps, "y": 0.0, "z": 0.0}] * NUM_TIMESTEPS,
        "valid": [True] * NUM_TIMESTEPS,
        "mark_as_expert": mark_as_expert,
    }


def _straight_road(road_type, map_element_id, x0_m, x1_m, y_m, **extra):
    xs = np.linspace(x0_m, x1_m, 13)
    return {
        "type": road_type,
        "map_element_id": map_element_id,
        "geometry": _points(xs, np.full(len(xs), y_m)),
        **extra,
    }


def _scenario():
    """Three chained lanes along y=0 bounded by road edges; the SDC (object 1) drives over the first two."""
    lanes = [
        _straight_road("lane", 2, LANE_LENGTH_M * idx, LANE_LENGTH_M * (idx + 1), 0.0, speed_limit=13.4)
        for idx in range(3)
    ]
    edges = [_straight_road("road_edge", 15, 0.0, 3 * LANE_LENGTH_M, y_m) for y_m in (-2.0, 2.0)]
    return {
        "scenario_id": "synthetic_0",
        "metadata": {"sim_dt": SIM_DT_S, "num_timesteps": NUM_TIMESTEPS, "sdc_track_index": 1},
        "objects": [
            _object("vehicle", 70.0, 0.0, SPEED_MPS),
            _object("vehicle", 5.0, 0.0, SPEED_MPS),
            _object("pedestrian", 10.0, 30.0, 1.0),
        ],
        "roads": lanes + edges,
    }


def test_converts_topology_routes_and_sdc_order(tmp_path):
    json_path = tmp_path / "scenario.json"
    json_path.write_text(json.dumps(_scenario()))
    bin_path = j2b.convert_file(json_path, tmp_path, "synthetic")
    assert bin_path.name == "synthetic__synthetic_0.bin"

    data = read_bin(bin_path)
    lanes = data["roads"][:3]
    assert [tuple(lane["exit_lanes"]) for lane in lanes] == [(1,), (2,), ()]
    assert [tuple(lane["entry_lanes"]) for lane in lanes] == [(), (0,), (1,)]
    assert data["lane_graph"]["lane_ids"] == (0, 1, 2)
    assert data["lane_graph"]["distances"][2] == pytest.approx(2 * LANE_LENGTH_M, rel=1e-4)
    assert np.isinf(data["lane_graph"]["distances"][3])

    sdc, other_vehicle, pedestrian = data["agents"]
    assert sdc["cols"]["x"][0] == pytest.approx(5.0)
    assert sdc["mark_as_expert"] == j2b.CONTROL_STATE_CONTROLLABLE
    assert other_vehicle["mark_as_expert"] == j2b.CONTROL_STATE_REPLAYED
    assert tuple(sdc["route"]) == (0, 1, 2) and sdc["route_gt_len"] == 2
    assert tuple(other_vehicle["route"]) == (1, 2) and other_vehicle["route_gt_len"] == 2
    assert pedestrian["type"] == j2b.AGENT_PEDESTRIAN and not pedestrian["route"]
    assert sdc["goal"][0] == pytest.approx(5.0 + SPEED_MPS * SIM_DT_S * (NUM_TIMESTEPS - 1), rel=1e-5)


def test_converted_bin_runs_in_replay(tmp_path):
    from pufferlib.ocean.drive.drive import Drive

    json_path = tmp_path / "scenario.json"
    json_path.write_text(json.dumps(_scenario()))
    bin_path = j2b.convert_file(json_path, tmp_path, "synthetic")
    env = Drive(
        num_agents=4,
        min_agents_per_env=1,
        max_agents_per_env=4,
        num_maps=1,
        allow_map_subset=True,
        map_dir=str(bin_path),
        simulation_mode="replay",
        control_mode="control_vehicles",
        replay_expert_agents=0,
        goal_source="route",
        action_type="continuous",
        scenario_length=NUM_TIMESTEPS,
    )
    env.reset(seed=0)
    assert env.agent_offsets[1] - env.agent_offsets[0] == 2
    for _ in range(10):
        observations, _, _, _, _ = env.step(np.zeros((env.num_agents, *env.single_action_space.shape), np.float32))
        assert np.isfinite(observations).all()
    env.close()


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda s: s["objects"][0]["heading"].pop(), "num_timesteps"),
        (lambda s: s["objects"][0].update(type="spaceship"), "unsupported type"),
        (lambda s: s["metadata"].update(sdc_track_index=7), "out of range"),
        (lambda s: s["metadata"].update(sdc_track_index=2), "could not be matched"),
        (lambda s: s["roads"][0].update(map_element_id=99), "unknown map_element_id"),
        (lambda s: s["roads"][0].update(geometry=s["roads"][0]["geometry"][:1]), "2 distinct points"),
        (lambda s: s["objects"][0]["position"][3].update(x=float("nan")), "finite"),
    ],
)
def test_invalid_scenarios_are_rejected(mutate, message):
    scenario = _scenario()
    mutate(scenario)
    with pytest.raises(ValueError, match=message):
        j2b.json_to_bin_data(scenario, "synthetic_0", "synthetic")
