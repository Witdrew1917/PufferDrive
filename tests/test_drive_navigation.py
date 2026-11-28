"""zpufferdrive navigation built at load time for legacy maps: env integration on a synthetic 2.0 map.

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

REPO_ROOT = Path(__file__).resolve().parents[1]
LANE_WIDTH_M = 3.7
TRAJECTORY_STEPS = 91
SPEED_MPS = 5.0
DT_SECONDS = 0.1
VEHICLE = 1
ROAD_LANE = 4
ROAD_EDGE = 6
EXPERT_ARRAY_COUNT = 5  # per-object expert accel, steering, delta x/y/yaw (load_map_binary)


def _polyline(start_x, end_x, y, step_m=1.0):
    xs = np.arange(start_x, end_x + 1e-6, step_m, dtype=np.float32)
    return xs, np.full_like(xs, y)


def _write_entity(out, entity_type, xs, ys, goal_x=0.0, goal_y=0.0):
    count = len(xs)
    out += struct.pack("<4i", 0, entity_type, 0, count)
    out += np.asarray(xs, "<f4").tobytes() + np.asarray(ys, "<f4").tobytes() + np.zeros(count, "<f4").tobytes()
    if entity_type == VEHICLE:
        out += np.full(count, SPEED_MPS, "<f4").tobytes() + np.zeros(2 * count, "<f4").tobytes()
        out += np.zeros(count, "<f4").tobytes() + np.ones(count, "<i4").tobytes()
        out += np.zeros(EXPERT_ARRAY_COUNT * count, "<f4").tobytes()
    out += struct.pack("<6fi", 2.0, 4.5, 1.5, goal_x, goal_y, 0.0, 0)


def write_synthetic_map(path, with_edge_between_lanes=False, goal_y=0.0):
    """Vehicle drives east on lane A (0..50 m) toward a goal on lane B (50..100 m); lane C runs parallel to the left.

    goal_y=LANE_WIDTH_M moves the goal onto lane C, which is only reachable from A by a lane change.
    """
    vehicle_xs = np.arange(TRAJECTORY_STEPS, dtype=np.float32) * SPEED_MPS * DT_SECONDS
    roads = [_polyline(0, 50, 0.0), _polyline(50, 100, 0.0), _polyline(0, 100, LANE_WIDTH_M)]
    road_types = [ROAD_LANE] * 3
    if with_edge_between_lanes:
        roads.append(_polyline(0, 100, LANE_WIDTH_M / 2))
        road_types.append(ROAD_EDGE)
    out = bytearray(b"synthetic_navi00")
    out += struct.pack("<ii", 0, 0) + struct.pack("<ii", 1, len(roads))
    _write_entity(out, VEHICLE, vehicle_xs, np.zeros_like(vehicle_xs), goal_x=95.0, goal_y=goal_y)
    for road_type, (xs, ys) in zip(road_types, roads):
        _write_entity(out, road_type, xs, ys)
    Path(path).write_bytes(bytes(out))


def write_split_map(path, reverse_lanes=False, turn_goal=False):
    straight = (np.array([0, 10, 25, 60]), np.zeros(4))
    turn = (np.array([0, 10, 25, 25, 60, 60]), np.array([0.2, 0.2, -15, -30, -30, 0]))
    roads = [straight, turn, _polyline(60, 100, 0)]
    if reverse_lanes:
        roads[:2] = roads[1::-1]
    out = bytearray(b"synthetic_navi00")
    out += struct.pack("<4i", 0, 0, 1, len(roads))
    xs = np.arange(TRAJECTORY_STEPS, dtype=np.float32) * SPEED_MPS * DT_SECONDS
    _write_entity(
        out, VEHICLE, xs, np.zeros_like(xs), goal_x=50.0 if turn_goal else 95.0, goal_y=-30.0 if turn_goal else 0.0
    )
    for x, y in roads:
        _write_entity(out, ROAD_LANE, x, y)
    Path(path).write_bytes(out)


def write_curve_map(path, radius):
    goal_angle = 3.0
    road_angles = np.linspace(0.0, goal_angle, 121, dtype=np.float32)
    if radius is None:
        road_xs, road_ys = road_angles * 10.0, np.zeros_like(road_angles)
        goal_x, goal_y = float(road_xs[-1]), 0.0
        vehicle_xs = np.arange(TRAJECTORY_STEPS, dtype=np.float32) * SPEED_MPS * DT_SECONDS
        vehicle_ys = np.zeros_like(vehicle_xs)
    else:
        road_xs = radius * np.sin(road_angles)
        road_ys = radius * (1.0 - np.cos(road_angles))
        goal_x, goal_y = float(road_xs[-1]), float(road_ys[-1])
        vehicle_angles = np.arange(TRAJECTORY_STEPS, dtype=np.float32) * SPEED_MPS * DT_SECONDS / radius
        vehicle_xs = radius * np.sin(vehicle_angles)
        vehicle_ys = radius * (1.0 - np.cos(vehicle_angles))

    out = bytearray(b"synthetic_navi00")
    out += struct.pack("<ii", 0, 0) + struct.pack("<ii", 1, 3)
    _write_entity(out, VEHICLE, vehicle_xs, vehicle_ys, goal_x=goal_x, goal_y=goal_y)
    _write_entity(out, ROAD_LANE, road_xs, road_ys)
    side_xs = np.array([0.0, goal_x], dtype=np.float32)
    _write_entity(out, ROAD_LANE, side_xs, np.full_like(side_xs, 3.7))
    _write_entity(out, ROAD_LANE, side_xs, np.full_like(side_xs, -3.7))
    Path(path).write_bytes(out)


def write_overlapping_goal_map(path, reverse_lanes=False, goal_y=0.1):
    straight = (np.array([0, 100]), np.zeros(2))
    incoming = (np.array([20, 35, 35, 45.3]), np.array([-20, -20, 0.1, 0.1]))
    roads = [straight, incoming, _polyline(-20, 0, 0)]
    if reverse_lanes:
        roads[:2] = roads[1::-1]
    out = bytearray(b"synthetic_navi00")
    out += struct.pack("<4i", 0, 0, 1, len(roads))
    xs = np.arange(TRAJECTORY_STEPS, dtype=np.float32) * SPEED_MPS * DT_SECONDS
    _write_entity(out, VEHICLE, xs, np.zeros_like(xs), goal_x=45.3, goal_y=goal_y)
    for x, y in roads:
        _write_entity(out, ROAD_LANE, x, y)
    Path(path).write_bytes(out)


ENV_SCRIPT = textwrap.dedent(
    """
    import json
    import sys
    import numpy as np
    import torch
    from pufferlib.ocean.drive import binding
    from pufferlib.ocean.drive.drive import Drive
    from pufferlib.ocean.torch import Drive as DrivePolicy
    config = json.loads(sys.argv[6])
    resample_map = config.pop("resample_map", False)
    env = Drive(num_agents=1, num_maps=1, map_dir=sys.argv[1], navigation=int(sys.argv[2]),
                navigation_waypoints=int(sys.argv[3]), control_mode="control_vehicles", episode_length=91,
                resample_frequency=0, goal_obs_dropout_prob=float(sys.argv[4]), **config)
    obs, _ = env.reset()
    initial_obs = obs[0].tolist()
    goal_samples = []
    for seed in range(int(sys.argv[5])):
        obs, _ = env.reset(seed=seed + 1)
        goal_samples.append(obs[0, 4:6].tolist())
    if resample_map:
        env.resample_maps()
        goal_samples.append(env.observations[0, 4:6].tolist())
    # classic dynamics: action = accel_idx * NUM_STEER_BINS + steer_idx; middle bins are zero accel/steer
    neutral_action = (binding.NUM_ACCEL_BINS // 2) * binding.NUM_STEER_BINS + binding.NUM_STEER_BINS // 2
    for _ in range(5):
        obs, *_ = env.step(np.full_like(env.actions, neutral_action))
    policy = DrivePolicy(env)
    logits, value = policy(torch.as_tensor(obs))
    waypoint_dim = env.max_waypoints * env.waypoint_features
    print("RESULT " + json.dumps({
        "ego": env.ego_features,
        "obs": obs.shape[1],
        "goal": obs[0, 4:6].tolist(),
        "initial_obs": initial_obs,
        "final_speed": float(obs[0, 6]),
        "goal_samples": goal_samples,
        "feasibility": obs[0, env.ego_features - 3 : env.ego_features].tolist() if env.navigation else [],
        "waypoints": obs[0, obs.shape[1] - waypoint_dim :].reshape(-1, env.waypoint_features).tolist() if waypoint_dim else [],
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

    def _run_env(self, navigation, waypoints=16, goal_dropout_prob=0.0, reset_count=0, **config):
        return subprocess.run(
            [
                sys.executable,
                "-c",
                ENV_SCRIPT,
                self.tmp.name,
                str(navigation),
                str(waypoints),
                str(goal_dropout_prob),
                str(reset_count),
                json.dumps(config),
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )

    def _result(self, completed):
        self.assertEqual(completed.returncode, 0, completed.stderr)
        line = next(line for line in completed.stdout.splitlines() if line.startswith("RESULT "))
        return json.loads(line[len("RESULT ") :])

    def test_navi_pid_follows_built_navigation_without_policy_navigation(self):
        result = self._result(self._run_env(0, sdc_controller="navi_pid"))
        self.assertGreater(result["final_speed"], result["initial_obs"][6])

    def test_navi_pid_reduces_target_speed_for_curvature(self):
        write_curve_map(self.map_path, radius=None)
        straight = self._result(self._run_env(0, sdc_controller="navi_pid"))
        write_curve_map(self.map_path, radius=10.0)
        self.map_path.with_suffix(".znavi").unlink(missing_ok=True)
        curved = self._result(self._run_env(0, sdc_controller="navi_pid"))
        self.assertLess(curved["final_speed"], straight["final_speed"])

    def test_built_navigation_needs_no_sidecar(self):
        off = self._result(self._run_env(0, zbin_navigation_source=1))
        for source in (1, 0):
            on = self._result(self._run_env(1, zbin_navigation_source=source))
            expected = ["map_000.bin"] if source == 1 else ["map_000.bin", "map_000.znavi"]
            self.assertEqual(sorted(path.name for path in Path(self.tmp.name).iterdir()), expected)
            self.assertEqual(on["ego"] - off["ego"], 3)
            # zpufferdrive layout: ego-frame (x, y) per waypoint
            self.assertEqual(on["obs"] - off["obs"], 3 + 16 * 2)
            self.assertGreater(on["feasibility"][1], 0.0)
            self.assertTrue(all(0.0 <= value <= 1.0 for value in on["feasibility"]))
            self.assertFalse(off["has_waypoint_encoder"])
            self.assertTrue(on["has_waypoint_encoder"])
            self.assertEqual(on["value_shape"], [1, 1])
            # Straight route east along the lane centre towards the goal.
            waypoints = np.array(on["waypoints"])
            valid = waypoints[np.any(waypoints != 0.0, axis=1)]
            self.assertGreater(len(valid), 1)
            self.assertTrue(np.all(np.diff(valid[:, 0]) > 0.0))
            np.testing.assert_allclose(valid[:, 1], 0.0, atol=1e-3)

    def test_znavi_sidecar_round_trip_and_stale_map(self):
        sidecar = self.map_path.with_suffix(".znavi")
        built = self._result(self._run_env(1, zbin_navigation_source=0, reset_count=2))
        self.assertTrue(sidecar.exists())
        auto_written = sidecar.read_bytes()
        loaded = self._result(self._run_env(1, zbin_navigation_source=0, reset_count=2))
        self.assertEqual(loaded, built)
        # The CLI writes the same map-intrinsic sidecar and skips existing files unless asked to overwrite.
        cli = [sys.executable, "-m", "pufferlib.ocean.drive.drive", "navigation", "--map-dir", self.tmp.name]
        completed = subprocess.run(cli, cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
        self.assertIn("Wrote 0, skipped 1", completed.stdout, completed.stderr)
        sidecar.unlink()
        completed = subprocess.run(cli, cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
        self.assertIn("Wrote 1", completed.stdout, completed.stderr)
        self.assertEqual(sidecar.read_bytes(), auto_written)
        # A changed map binary invalidates its sidecar.
        write_synthetic_map(self.map_path, goal_y=1.0)
        stale = self._run_env(1, zbin_navigation_source=0)
        self.assertNotEqual(stale.returncode, 0)
        self.assertIn("map fingerprint mismatch", stale.stdout + stale.stderr)

    def test_waypoint_count_is_configurable(self):
        default = self._result(self._run_env(1))
        custom = self._result(self._run_env(1, waypoints=4))
        self.assertEqual(default["obs"] - custom["obs"], (16 - 4) * 2)
        waypoints = np.array(custom["waypoints"])
        self.assertEqual(waypoints.shape, (4, 2))
        valid = waypoints[np.any(waypoints != 0.0, axis=1)]
        self.assertGreater(len(valid), 0)
        self.assertTrue(np.all(np.diff(valid[:, 0]) > 0.0))

    def test_snap_initial_goal_on_legacy_maps(self):
        xs = np.arange(TRAJECTORY_STEPS, dtype=np.float32) * SPEED_MPS * DT_SECONDS
        out = bytearray(b"synthetic_navi00")
        out += struct.pack("<4i", 0, 0, 1, 2)
        _write_entity(out, VEHICLE, xs, np.zeros_like(xs), goal_x=95.3, goal_y=1.0)
        for y in (0.0, LANE_WIDTH_M):
            lane_x, lane_y = _polyline(0, 500, y, step_m=10.0)
            _write_entity(out, ROAD_LANE, lane_x, lane_y)
        self.map_path.write_bytes(out)
        for navigation in (0, 1):
            off = self._result(self._run_env(navigation, snap_goal_to_navi_endpoint=0))
            on = self._result(
                self._run_env(
                    navigation, snap_goal_to_navi_endpoint=1, goal_behavior=1, reset_count=2, resample_map=True
                )
            )
            np.testing.assert_allclose(off["initial_obs"][4:6], [95.3 * 0.005, 1.0 * 0.005], atol=1e-6)
            # Snapped onto the goal lane centre (route endpoints are lane samples, so x may move by a sample).
            self.assertAlmostEqual(on["initial_obs"][5], 0.0, places=6)
            self.assertAlmostEqual(on["initial_obs"][4], 95.3 * 0.005, delta=10.0 * 0.005)
            np.testing.assert_allclose(on["goal_samples"], [on["initial_obs"][4:6]] * 3, atol=1e-6)
            np.testing.assert_array_equal(np.delete(off["initial_obs"], [4, 5]), np.delete(on["initial_obs"], [4, 5]))

    def test_snap_does_not_change_generated_legacy_goals(self):
        write_synthetic_map(self.map_path, goal_y=1.0)
        config = {"goal_behavior": 1, "goal_radius": 1000.0}
        off = self._result(self._run_env(0, **config))
        on = self._result(self._run_env(0, snap_goal_to_navi_endpoint=1, **config))
        self.assertNotEqual(off["initial_obs"][5], on["initial_obs"][5])
        np.testing.assert_array_equal(off["goal"], on["goal"])

    def test_invalid_waypoint_count_is_rejected(self):
        result = self._run_env(1, waypoints=0)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("navigation_waypoints", result.stderr)

    def test_goal_observation_dropout_resamples_on_reset(self):
        visible = self._result(self._run_env(0, goal_dropout_prob=0.0))
        masked = self._result(self._run_env(0, goal_dropout_prob=1.0))
        self.assertTrue(any(value != 0.0 for value in visible["goal"]))
        self.assertEqual(masked["goal"], [0.0, 0.0])

        samples = self._result(self._run_env(0, goal_dropout_prob=0.5, reset_count=20))["goal_samples"]
        self.assertIn([0.0, 0.0], samples)
        self.assertTrue(any(any(value != 0.0 for value in sample) for sample in samples))


if __name__ == "__main__":
    unittest.main()
