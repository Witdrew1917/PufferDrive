"""Navigation sidecar generator (data_utils/process_navigation.py), its IO, and goal-conditioned env navigation."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_utils import navigation_bin as nb
from data_utils import process_navigation as pn

REPO_ROOT = Path(__file__).resolve().parents[2]
TOWN01_BIN = REPO_ROOT / "pufferlib/resources/drive/binaries/carla/opendrive__Town01.bin"
LANE_WIDTH_M = 3.7
LANE_LENGTH_M = 100.0
ROAD_LINE_BROKEN = 10
ROAD_LINE_SOLID = 12
LANE_SURFACE_STREET = 2


def _lane(x0, y0, x1, y1, exits):
    xs = np.linspace(x0, x1, 11)
    ys = np.linspace(y0, y1, 11)
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
    return {
        "type": LANE_SURFACE_STREET,
        "x": xs.tolist(),
        "y": ys.tolist(),
        "exit_lanes": list(exits),
        "length": float(cum[-1]),
        "cum_lengths": cum.tolist(),
    }


def _two_lane_map(separator_type):
    """Lane 0 (y=0) continues into lane 2; lane 1 is its left neighbour (y=3.7), split by a road line."""
    roads = [
        _lane(0.0, 0.0, LANE_LENGTH_M, 0.0, [2]),
        _lane(0.0, LANE_WIDTH_M, LANE_LENGTH_M, LANE_WIDTH_M, []),
        _lane(LANE_LENGTH_M, 0.0, 2 * LANE_LENGTH_M, 0.0, []),
        {"type": separator_type, "x": [0.0, LANE_LENGTH_M], "y": [LANE_WIDTH_M / 2] * 2},
    ]
    return {"roads": roads, "agents": []}


def test_broken_line_allows_lane_change_toward_goal():
    navi = pn.build_navigation(_two_lane_map(ROAD_LINE_BROKEN))
    assert navi["left_lane"].tolist() == [1, -1, -1]
    assert navi["right_lane"].tolist() == [-1, 0, -1]
    goal = 1
    assert navi["distance_to_goal_m"][goal, 0] == pytest.approx(nb.NAVI_LANE_CHANGE_COST_M + LANE_LENGTH_M)
    assert navi["feasibility_center_m"][goal, 0] == pytest.approx(LANE_LENGTH_M)
    assert navi["feasibility_center_m"][goal, goal] == nb.NAVI_FEASIBILITY_CAP_M


def test_solid_line_blocks_lane_change():
    navi = pn.build_navigation(_two_lane_map(ROAD_LINE_SOLID))
    assert navi["left_lane"].tolist() == [-1, -1, -1]
    goal = 1
    assert navi["distance_to_goal_m"][goal, 0] == nb.NAVI_UNREACHABLE_M
    assert navi["feasibility_center_m"][goal, 0] == 0.0


def test_fork_tables_are_conditioned_on_goal_branch():
    roads = [
        _lane(0.0, 0.0, 50.0, 0.0, [1, 2]),
        _lane(50.0, 0.0, 100.0, 40.0, []),
        _lane(50.0, 0.0, 100.0, -40.0, []),
    ]
    navi = pn.build_navigation({"roads": roads, "agents": []})
    for goal, other_branch in ((1, 2), (2, 1)):
        assert navi["distance_to_goal_m"][goal, 0] == pytest.approx(roads[0]["length"] + roads[goal]["length"])
        assert navi["feasibility_center_m"][goal, 0] == nb.NAVI_FEASIBILITY_CAP_M
        assert navi["distance_to_goal_m"][goal, other_branch] == nb.NAVI_UNREACHABLE_M


def test_town01_sidecar_round_trips(tmp_path):
    out = tmp_path / "opendrive__Town01.navi"
    written = pn.process_map(TOWN01_BIN, out)
    navi = nb.read_navigation(out)
    for key in ("lane_road_idx", "left_lane", "right_lane", "distance_to_goal_m", "feasibility_center_m"):
        np.testing.assert_array_equal(navi[key], written[key])
    assert (navi["map_crc32"], navi["map_size_bytes"]) == nb.map_fingerprint(TOWN01_BIN)
    assert np.all(np.diag(navi["feasibility_center_m"]) == nb.NAVI_FEASIBILITY_CAP_M)

    raw = out.read_bytes()
    out.write_bytes(raw + b"\0")
    with pytest.raises(ValueError):
        nb.read_navigation(out)
    out.write_bytes(raw[:-4])
    with pytest.raises(ValueError):
        nb.read_navigation(out)


def test_plot_navigation_writes_html_and_svg_per_goal(tmp_path):
    pytest.importorskip("plotly")
    pytest.importorskip("matplotlib")
    from data_utils import plot_navigation

    navi = pn.process_map(TOWN01_BIN, tmp_path / "opendrive__Town01.navi")
    goals = [3, 7]
    html_path, svg_path = plot_navigation.render_map(
        TOWN01_BIN, tmp_path / "plots", navigation_dir=tmp_path, goals=goals
    )
    html = html_path.read_text()
    assert all(f'"label":"goal {goal}:' in html for goal in goals)
    assert svg_path.read_text().lstrip().startswith("<?xml")
    with pytest.raises(ValueError, match="goal lane"):
        plot_navigation.render_map(TOWN01_BIN, tmp_path / "plots", navigation_dir=tmp_path, goals=[navi["num_lanes"]])


def _town01_navigation_env(tmp_path, goal_source="route", **overrides):
    from pufferlib.ocean.drive.drive import Drive

    sidecar_path = tmp_path / "opendrive__Town01.navi"
    if not sidecar_path.exists():
        pn.process_map(TOWN01_BIN, sidecar_path)
    return Drive(
        num_agents=16,
        min_agents_per_env=1,
        max_agents_per_env=16,
        num_maps=1,
        allow_map_subset=True,
        map_dir=str(TOWN01_BIN),
        simulation_mode="gigaflow",
        action_type="continuous",
        dynamics_model="jerk",
        scenario_length=40,
        resample_frequency=1_000_000,
        termination_mode=0,
        goal_source=goal_source,
        navigation_dir=str(tmp_path),
        seed=3,
        **overrides,
    )


@pytest.mark.parametrize("goal_source", ["route", "map"])
def test_env_navigation_tracks_goals_from_any_source(tmp_path, goal_source):
    env = _town01_navigation_env(tmp_path, goal_source=goal_source, obs_navigation=True)
    env.reset(seed=3)
    logs = []
    actions = np.zeros_like(env.actions)
    for _ in range(env.scenario_length + 1):
        _, _, _, _, info = env.step(actions)
        logs += info
    env.close()
    assert logs and all(log["navi_goal_active_rate"] > 0.5 for log in logs)


def test_obs_navigation_rejects_missing_sidecars(tmp_path):
    from pufferlib.ocean.drive.drive import Drive

    with pytest.raises(ValueError, match="does not exist"):
        Drive(config_only=True, obs_navigation=True, navigation_dir=str(tmp_path / "missing"))
    with pytest.raises(ValueError, match="missing"):
        Drive(
            num_agents=1,
            num_maps=1,
            map_dir=str(TOWN01_BIN),
            simulation_mode="gigaflow",
            obs_navigation=True,
            navigation_dir=str(tmp_path),
        )


@pytest.mark.parametrize("obs_navigation", [False, True])
def test_navigation_frame_telemetry_tracks_final_goal(tmp_path, obs_navigation):
    from pufferlib.ocean.drive import binding

    env = _town01_navigation_env(tmp_path, obs_navigation=obs_navigation)
    assert env._navigation_replay_keys()
    env.reset(seed=3)
    env.step(np.zeros_like(env.actions))
    env_count, agent_capacity = env.num_envs, env.max_agents_per_env
    navi_f32 = np.empty((env_count, agent_capacity, binding.NAVI_F32_FIELDS), dtype=np.float32)
    navi_route_i32 = np.empty((env_count, agent_capacity, binding.NAVI_ROUTE_PREVIEW_LANE_COUNT), dtype=np.int32)
    binding.vec_get_navigation_frame(env.c_envs, navi_f32, navi_route_i32)
    wrong_fields = np.empty((env_count, agent_capacity, binding.NAVI_F32_FIELDS + 1), dtype=np.float32)
    with pytest.raises(ValueError):
        binding.vec_get_navigation_frame(env.c_envs, wrong_fields, navi_route_i32)
    env.close()

    routed = navi_f32[..., 0] >= 0
    distance_to_goal_m = navi_f32[..., 3]
    assert routed.any()
    assert np.all(navi_route_i32[routed][:, 0] >= 0)
    assert np.all(np.isfinite(distance_to_goal_m[routed]) & (distance_to_goal_m[routed] >= 0))
    assert np.all(navi_route_i32[~routed] == -1)


@pytest.mark.parametrize("goal_source", ["route", "map"])
def test_obs_navigation_appends_ego_features_without_changing_others(tmp_path, goal_source):
    from pufferlib.ocean.drive import binding

    observations = {}
    for obs_navigation in (False, True):
        env = _town01_navigation_env(tmp_path, goal_source=goal_source, obs_navigation=obs_navigation)
        env.reset(seed=3)
        env.step(np.zeros_like(env.actions))
        observations[obs_navigation] = env.observations.copy()
        env.close()

    navi_start = binding.EGO_FEATURES
    navi_end = navi_start + binding.NAVIGATION_EGO_FEATURES
    with_navi = observations[True]
    assert with_navi.shape[1] == observations[False].shape[1] + binding.NAVIGATION_EGO_FEATURES
    np.testing.assert_array_equal(np.delete(with_navi, np.s_[navi_start:navi_end], axis=1), observations[False])
    navi_obs = with_navi[:, navi_start:navi_end]
    reachable = navi_obs[:, 0] == 1.0
    assert set(np.unique(navi_obs[:, 0])) <= {0.0, 1.0}
    assert reachable.mean() > 0.5
    assert np.all((navi_obs >= 0.0) & (navi_obs <= 1.0))
    assert np.all(navi_obs[~reachable] == 0.0)
    assert np.any(navi_obs[reachable, 2] > 0.0)
