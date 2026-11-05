"""Navigation sidecar (.navi) generation, IO and env integration on a synthetic 2.0 map.

Running the test: python -m unittest tests/test_drive_navigation.py
"""

import json
import os
import struct
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pufferlib.ocean.drive import navigation as nav

REPO_ROOT = Path(__file__).resolve().parents[1]
LANE_WIDTH_M = 3.7
TRAJECTORY_STEPS = 91
SPEED_MPS = 5.0
DT_SECONDS = 0.1


def _polyline(start_x, end_x, y, step_m=1.0):
    xs = np.arange(start_x, end_x + 1e-6, step_m, dtype=np.float32)
    return xs, np.full_like(xs, y)


def _write_entity(out, entity_type, xs, ys, goal_x=0.0, goal_y=0.0):
    count = len(xs)
    out += struct.pack("<4i", 0, entity_type, 0, count)
    out += np.asarray(xs, "<f4").tobytes() + np.asarray(ys, "<f4").tobytes() + np.zeros(count, "<f4").tobytes()
    if entity_type == nav.VEHICLE:
        out += np.full(count, SPEED_MPS, "<f4").tobytes() + np.zeros(2 * count, "<f4").tobytes()
        out += np.zeros(count, "<f4").tobytes() + np.ones(count, "<i4").tobytes()
    out += struct.pack("<6fi", 2.0, 4.5, 1.5, goal_x, goal_y, 0.0, 0)


def write_synthetic_map(path, with_edge_between_lanes=False, goal_y=0.0):
    """Vehicle drives east on lane A (0..50 m) toward a goal on lane B (50..100 m); lane C runs parallel to the left.

    goal_y=LANE_WIDTH_M moves the goal onto lane C, which is only reachable from A by a lane change.
    """
    vehicle_xs = np.arange(TRAJECTORY_STEPS, dtype=np.float32) * SPEED_MPS * DT_SECONDS
    roads = [_polyline(0, 50, 0.0), _polyline(50, 100, 0.0), _polyline(0, 100, LANE_WIDTH_M)]
    road_types = [nav.ROAD_LANE] * 3
    if with_edge_between_lanes:
        roads.append(_polyline(0, 100, LANE_WIDTH_M / 2))
        road_types.append(nav.ROAD_EDGE)
    out = bytearray(b"synthetic_navi00")
    out += struct.pack("<ii", 0, 0) + struct.pack("<ii", 1, len(roads))
    _write_entity(out, nav.VEHICLE, vehicle_xs, np.zeros_like(vehicle_xs), goal_x=95.0, goal_y=goal_y)
    for road_type, (xs, ys) in zip(road_types, roads):
        _write_entity(out, road_type, xs, ys)
    Path(path).write_bytes(bytes(out))


class TestNavigationGeneration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.map_path = Path(self.tmp.name) / "map_000.bin"
        write_synthetic_map(self.map_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_topology_goal_and_round_trip(self):
        navi = nav.process_map_navigation(self.map_path, max_goals=4, seed=0)
        lane_a, lane_b, lane_c = 0, 1, 2
        goal_idx = navi["object_goal_idx"][0]
        self.assertEqual(navi["goal_lane"][goal_idx], lane_b)
        self.assertEqual(navi["next_lane"][goal_idx, lane_a], lane_b)
        self.assertAlmostEqual(float(navi["distance_to_goal_m"][goal_idx, lane_a]), 100.0, places=3)
        self.assertEqual(navi["left_lane"][lane_a], lane_c)
        self.assertGreater(navi["feasibility_center_m"][goal_idx, lane_a], 0.0)

        loaded = nav.read_navigation(nav.navigation_path_for(self.map_path))
        for key in ("lane_entity_idx", "left_lane", "next_lane", "distance_to_goal_m", "feasibility_left_m"):
            np.testing.assert_array_equal(loaded[key], navi[key])
        self.assertEqual((loaded["map_crc32"], loaded["map_size_bytes"]), nav.map_fingerprint(self.map_path))

    def test_next_lane_chains_terminate_at_goal(self):
        navi = nav.process_map_navigation(self.map_path, max_goals=4, seed=0)
        for goal_idx, goal_lane in enumerate(navi["goal_lane"]):
            for lane in range(navi["num_lanes"]):
                if lane == goal_lane or navi["next_lane"][goal_idx, lane] == -1:
                    continue
                for _ in range(navi["num_lanes"]):
                    lane = navi["next_lane"][goal_idx, lane]
                    if lane == goal_lane or lane == -1:
                        break
                self.assertEqual(lane, goal_lane)

    def test_road_edge_blocks_lane_change(self):
        write_synthetic_map(self.map_path, with_edge_between_lanes=True)
        navi = nav.process_map_navigation(self.map_path, max_goals=4, seed=0)
        self.assertEqual(navi["left_lane"][0], -1)

    def test_reader_rejects_trailing_bytes(self):
        nav.process_map_navigation(self.map_path, max_goals=4, seed=0)
        navi_path = nav.navigation_path_for(self.map_path)
        navi_path.write_bytes(navi_path.read_bytes() + b"\0")
        with self.assertRaises(ValueError):
            nav.read_navigation(navi_path)


ENV_SCRIPT = textwrap.dedent(
    """
    import json
    import sys
    import numpy as np
    import torch
    from pufferlib.ocean.drive.drive import Drive
    from pufferlib.ocean.torch import Drive as DrivePolicy
    env = Drive(num_agents=1, num_maps=1, map_dir=sys.argv[1], navigation=int(sys.argv[2]),
                navigation_waypoints=int(sys.argv[3]), navigation_waypoint_spacing=float(sys.argv[4]),
                control_mode="control_vehicles", episode_length=91, resample_frequency=0)
    obs, _ = env.reset()
    neutral_action = 3 * 13 + 6  # classic dynamics: zero acceleration, zero steering
    for _ in range(5):
        obs, *_ = env.step(np.full_like(env.actions, neutral_action))
    policy = DrivePolicy(env)
    logits, value = policy(torch.as_tensor(obs))
    waypoint_dim = env.max_waypoints * env.waypoint_features
    print("RESULT " + json.dumps({
        "ego": env.ego_features,
        "obs": obs.shape[1],
        "feasibility": obs[0, env.ego_features - 3 : env.ego_features].tolist() if env.navigation else [],
        "waypoints": obs[0, obs.shape[1] - waypoint_dim :].reshape(-1, 3).tolist() if waypoint_dim else [],
        "value_shape": list(value.shape),
        "has_waypoint_encoder": hasattr(policy, "waypoint_encoder"),
    }))
    env.close()
    """
)


class TestNavigationEnv(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.map_path = Path(self.tmp.name) / "map_000.bin"
        write_synthetic_map(self.map_path)

    def tearDown(self):
        self.tmp.cleanup()

    def _run_env(self, navigation, waypoints=16, spacing=5.0):
        return subprocess.run(
            [sys.executable, "-c", ENV_SCRIPT, self.tmp.name, str(navigation), str(waypoints), str(spacing)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )

    def _result(self, completed):
        self.assertEqual(completed.returncode, 0, completed.stderr)
        line = next(line for line in completed.stdout.splitlines() if line.startswith("RESULT "))
        return json.loads(line[len("RESULT ") :])

    def test_navigation_flag_adds_feasibility_and_waypoints(self):
        off = self._result(self._run_env(0))
        nav.process_map_navigation(self.map_path, max_goals=4, seed=0)
        on = self._result(self._run_env(1))
        self.assertEqual(on["ego"] - off["ego"], 3)
        self.assertEqual(on["obs"] - off["obs"], 3 + 16 * 3)
        self.assertGreater(on["feasibility"][1], 0.0)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in on["feasibility"]))
        self.assertFalse(off["has_waypoint_encoder"])
        self.assertTrue(on["has_waypoint_encoder"])
        self.assertEqual(on["value_shape"], [1, 1])

        # Straight route east along lanes A -> B: waypoints every 5 m (x0.02 scale) on the lane centre.
        waypoints = np.array(on["waypoints"])
        np.testing.assert_array_equal(waypoints[:, 2], 1.0)
        np.testing.assert_allclose(np.diff(waypoints[:, 0]), 5.0 * 0.02, atol=1e-4)
        np.testing.assert_allclose(waypoints[:, 1], 0.0, atol=1e-3)

    def test_waypoint_count_and_spacing_are_configurable(self):
        nav.process_map_navigation(self.map_path, max_goals=4, seed=0)
        default = self._result(self._run_env(1))
        custom = self._result(self._run_env(1, waypoints=4, spacing=10.0))
        self.assertEqual(default["obs"] - custom["obs"], (16 - 4) * 3)
        waypoints = np.array(custom["waypoints"])
        self.assertEqual(waypoints.shape, (4, 3))
        np.testing.assert_allclose(np.diff(waypoints[:, 0]), 10.0 * 0.02, atol=1e-4)

    def test_waypoints_follow_lane_change_and_stop_at_goal(self):
        write_synthetic_map(self.map_path, goal_y=LANE_WIDTH_M)
        nav.process_map_navigation(self.map_path, max_goals=4, seed=0)
        waypoints = np.array(self._result(self._run_env(1, spacing=10.0))["waypoints"])
        valid = waypoints[:, 2] == 1.0
        # Lane C ends at x = 100 m, so fewer than 16 waypoints 10 m apart fit ahead of the agent.
        self.assertTrue(5 < valid.sum() < 16, valid)
        # Goal only reachable on the parallel left lane C: route switches to it (ego frame y = +lane width).
        np.testing.assert_allclose(waypoints[valid, 1], LANE_WIDTH_M * 0.02, atol=1e-3)
        # Waypoints stop at the end of lane C (x = 100 m) and the padding is all zeros.
        self.assertTrue(np.all(np.diff(valid.astype(int)) <= 0))
        np.testing.assert_array_equal(waypoints[~valid], 0.0)

    def test_invalid_waypoint_count_is_rejected(self):
        nav.process_map_navigation(self.map_path, max_goals=4, seed=0)
        result = self._run_env(1, waypoints=0)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("navigation_waypoints", result.stderr)

    def test_stale_sidecar_aborts_env_creation(self):
        nav.process_map_navigation(self.map_path, max_goals=4, seed=0)
        map_bytes = bytearray(self.map_path.read_bytes())
        map_bytes[-5] ^= 0x01
        self.map_path.write_bytes(bytes(map_bytes))
        result = self._run_env(1)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("fingerprint mismatch", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
