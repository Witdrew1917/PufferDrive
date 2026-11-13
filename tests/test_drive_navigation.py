"""Navigation sidecar (.navi) generation, IO and env integration on a synthetic 2.0 map.

Running the test: python -m unittest tests/test_drive_navigation.py
"""

import json
import os
import shutil
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
        out += np.zeros(nav.EXPERT_ARRAY_COUNT * count, "<f4").tobytes()
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
        out, nav.VEHICLE, xs, np.zeros_like(xs), goal_x=50.0 if turn_goal else 95.0, goal_y=-30.0 if turn_goal else 0.0
    )
    for x, y in roads:
        _write_entity(out, nav.ROAD_LANE, x, y)
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
    _write_entity(out, nav.VEHICLE, xs, np.zeros_like(xs), goal_x=45.3, goal_y=goal_y)
    for x, y in roads:
        _write_entity(out, nav.ROAD_LANE, x, y)
    Path(path).write_bytes(out)


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

    def test_split_prefix_and_longitudinal_distances_round_trip(self):
        write_split_map(self.map_path)
        navi = nav.process_map_navigation(self.map_path, max_goals=4, seed=0)
        np.testing.assert_array_equal(navi["branch_offsets"], [0, 1, 2, 2])
        np.testing.assert_array_equal(navi["branch_lane"], [1, 0])
        self.assertTrue(np.all(navi["branch_end_m"] >= 2))
        self.assertTrue(np.all(navi["branch_end_m"] <= 11))
        goal = navi["object_goal_idx"][0]
        self.assertLess(
            navi["longitudinal_distance_to_goal_m"][goal, 0], navi["longitudinal_distance_to_goal_m"][goal, 1]
        )
        loaded = nav.read_navigation(nav.navigation_path_for(self.map_path))
        for key in (
            "branch_offsets",
            "branch_lane",
            "branch_end_m",
            "branch_other_end_m",
            "longitudinal_distance_to_goal_m",
        ):
            np.testing.assert_array_equal(loaded[key], navi[key])

    def test_split_relations_are_pairwise_and_exclude_unrelated_lanes(self):
        lanes = [_polyline(0, 20, y) for y in (0, 0.8, 1.6, LANE_WIDTH_M)]
        lanes += [(np.array([0, 0]), np.array([0, 20])), (np.array([0, -20]), np.array([0, 0]))]
        branches = nav._infer_split_branches(lanes, [[] for _ in lanes])
        offsets = branches["branch_offsets"]
        siblings = [branches["branch_lane"][offsets[i] : offsets[i + 1]].tolist() for i in range(len(lanes))]
        self.assertEqual(siblings, [[1], [0, 2], [1], [], [], []])

    def test_hidden_goal_copies_do_not_consume_goal_slots_or_move_goal(self):
        write_overlapping_goal_map(self.map_path)
        navi = nav.process_map_navigation(self.map_path, max_goals=1, seed=0)
        self.assertEqual(navi["num_goals"], 1)
        self.assertEqual(navi["object_goal_idx"][0], 0)
        self.assertEqual(navi["goal_lane"][0], 1)
        np.testing.assert_allclose([navi["goal_x"][0], navi["goal_y"][0]], [45.3, 0.1], atol=1e-4)
        self.assertEqual(navi["next_lane"][0, 0], -1)
        self.assertEqual(navi["next_lane"][0, 1], -1)
        self.assertEqual(navi["next_lane"][0, 2], 0)
        for key in ("longitudinal_distance_to_goal_m", "distance_to_goal_m", "feasibility_center_m"):
            self.assertAlmostEqual(float(navi[key][0, 0]), 45.3, places=4)
            self.assertAlmostEqual(float(navi[key][0, 2]), 65.3, places=4)
        loaded = nav.read_navigation(nav.navigation_path_for(self.map_path))
        np.testing.assert_array_equal(loaded["next_lane"], navi["next_lane"])
        np.testing.assert_array_equal(
            loaded["longitudinal_distance_to_goal_m"], navi["longitudinal_distance_to_goal_m"]
        )

    def test_goal_copies_exclude_adjacent_crossing_opposing_and_unshared_lanes(self):
        lanes = [
            (np.array([20, 35, 35, 45.3]), np.array([-20, -20, 0.1, 0.1])),
            _polyline(0, 100, 0),
            _polyline(0, 100, LANE_WIDTH_M),
            (np.array([45.3, 45.3]), np.array([-10, 10])),
            (np.array([100, 0]), np.zeros(2)),
            (np.array([44.8, 45.3]), np.zeros(2)),
        ]
        lengths = np.array([nav._polyline_length(x, y) for x, y in lanes])
        terminals = nav._goal_lane_terminals(0, lanes, lengths, nav._SpatialIndex(list(enumerate(lanes))))
        self.assertEqual(set(terminals), {0, 1})
        self.assertAlmostEqual(terminals[1], 45.3, places=4)

    def test_rejects_old_version_and_invalid_split_metadata(self):
        write_split_map(self.map_path)
        navi = nav.process_map_navigation(self.map_path, max_goals=4, seed=0)
        path = nav.navigation_path_for(self.map_path)
        raw = bytearray(path.read_bytes())
        struct.pack_into("<i", raw, 8, nav.NAVI_VERSION - 1)
        path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "unsupported version.*overwrite"):
            nav.read_navigation(path)
        for field, value in (
            ("branch_offsets", -1),
            ("branch_lane", navi["num_lanes"]),
            ("branch_end_m", np.nan),
            ("longitudinal_distance_to_goal_m", -1),
        ):
            invalid = {**navi, field: navi[field].copy()}
            invalid[field].flat[0] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                nav.validate_navigation(invalid)


class TestNavigationSplits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build_dir = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.build_dir.cleanup)
        compiler = shutil.which("cc")
        if compiler is None:
            raise unittest.SkipTest("C compiler is required for native navigation tests")
        raylib = "raylib-5.5_macos" if sys.platform == "darwin" else "raylib-5.5_linux_amd64"
        cls.harness = str(Path(cls.build_dir.name) / "navigation_harness")
        completed = subprocess.run(
            [
                compiler,
                "-O1",
                "-D_GNU_SOURCE",
                "-ffunction-sections",
                "-fdata-sections",
                "-I" + str(REPO_ROOT / "pufferlib/ocean/drive"),
                "-I" + str(REPO_ROOT / raylib / "include"),
                "-I" + str(REPO_ROOT / "inih-r62"),
                str(REPO_ROOT / "tests/drive_navigation_harness.c"),
                "-Wl,-dead_strip" if sys.platform == "darwin" else "-Wl,--gc-sections",
                "-lm",
                "-o",
                cls.harness,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.map_path = Path(self.tmp.name) / "map_000.bin"
        write_split_map(self.map_path)
        self.navi = nav.process_map_navigation(self.map_path, max_goals=4, seed=0)

    def run_harness(self, x=5.0, y=0.18, heading=0.0, previous=-1, goal=None):
        args = [self.harness, str(self.map_path), str(x), str(y), str(heading), str(previous)]
        if goal is not None:
            args.append(str(goal))
        completed = subprocess.run(args, check=False, capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
        line = next(line for line in completed.stdout.splitlines() if line.startswith("RESULT "))
        return json.loads(line[len("RESULT ") :])

    def run_reward_steps(self, scenario):
        args = [self.harness, str(self.map_path), "5", "0", "0", "-1", str(self.navi["object_goal_idx"][0]), scenario]
        completed = subprocess.run(args, check=False, capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
        line = next(line for line in completed.stdout.splitlines() if line.startswith("RESULT "))
        return json.loads(line[len("RESULT ") :])

    def test_progress_reward_each_forward_step_and_branch_change(self):
        for scenario in ("progress", "branch"):
            with self.subTest(scenario=scenario):
                result = self.run_reward_steps(scenario)
                self.assertEqual(result["rewards"], [0.5, 0.5, 0.0, 0.5, 0.5, 0.0])
                self.assertEqual(result["episode_return"], 2.0)

    def test_progress_reward_stops_on_failure_and_stays_stopped(self):
        for scenario in ("collision", "offroad", "offroute"):
            with self.subTest(scenario=scenario):
                result = self.run_reward_steps(scenario)
                self.assertEqual(result["rewards"], [0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
                self.assertEqual(result["episode_return"], 0.5)

    def test_progress_reward_requires_new_closest_distance(self):
        result = self.run_reward_steps("retrace")
        self.assertEqual(result["rewards"], [0.5, 0.5, 0.0, 0.0, 0.5, 0.5])
        self.assertEqual(result["episode_return"], 2.0)

    def test_progress_baseline_resets_for_new_goal(self):
        result = self.run_reward_steps("newgoal_reset")
        self.assertEqual(result["rewards"][:4], [0.5, 2.0, 0.0, 0.5])

    def test_progress_baseline_and_failure_reset_for_new_episode(self):
        result = self.run_reward_steps("collision_reset")
        self.assertEqual(result["rewards"], [0.5, 0.0, 0.5, 0.5, 0.5, 0.0])

    def test_progress_reward_includes_step_reaching_goal(self):
        for scenario in ("goal", "newgoal", "removegoal"):
            with self.subTest(scenario=scenario):
                result = self.run_reward_steps(scenario)
                self.assertEqual(result["rewards"][:2], [0.5, 2.5])
                if scenario != "newgoal":
                    self.assertEqual(result["rewards"][2:], [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(self.run_reward_steps("collisiongoal")["rewards"][:2], [0.5, 2.0])

    def test_shorter_branch_drives_waypoints_renderers_and_metrics(self):
        result = self.run_harness(previous=1)
        self.assertEqual((result["physical"], result["selected"]), (1, 0))
        self.assertEqual(result["off_route"], 0)
        self.assertAlmostEqual(result["distance"], 95.0, places=4)
        self.assertGreater(result["center_feasibility"], 0)
        np.testing.assert_allclose(np.array(result["waypoints"])[:, 1], 0)
        self.assertEqual(result["gui_min_y"], 0)
        self.assertEqual(result["legacy_min_y"], 0)
        self.assertEqual(result["gui_goals"], 1)
        self.assertEqual(result["legacy_goals"], 1)
        self.assertEqual(result["previous_distance"], result["distance"])
        self.assertEqual(result["stationary_delta"], 0)
        self.assertEqual(result["sampled_goal"], int(self.navi["object_goal_idx"][0]))
        moved = self.run_harness(x=6, previous=0)
        self.assertAlmostEqual(result["distance"] - moved["distance"], 1.0, places=4)

    def test_selection_does_not_depend_on_lane_order(self):
        write_split_map(self.map_path, reverse_lanes=True)
        nav.process_map_navigation(self.map_path, max_goals=4, seed=0)
        result = self.run_harness()
        self.assertEqual((result["physical"], result["selected"]), (0, 1))

    def test_goal_can_select_the_turn_instead(self):
        write_split_map(self.map_path, turn_goal=True)
        nav.process_map_navigation(self.map_path, max_goals=4, seed=0)
        result = self.run_harness(y=0.02)
        self.assertEqual((result["physical"], result["selected"]), (0, 1))
        self.assertLess(result["gui_min_y"], -20)
        self.assertLess(result["legacy_min_y"], -20)
        self.assertEqual(result["off_route"], 0)

    def test_overlapping_goal_is_reachable_without_entering_incoming_turn(self):
        write_overlapping_goal_map(self.map_path)
        nav.process_map_navigation(self.map_path, max_goals=1, seed=0)
        result = self.run_harness(x=15, y=0)
        self.assertEqual((result["physical"], result["selected"]), (0, 0))
        self.assertEqual(result["off_route"], 0)
        self.assertAlmostEqual(result["distance"], 30.3, places=4)
        self.assertAlmostEqual(result["center_feasibility"], 30.3, places=4)
        self.assertEqual((result["gui_goals"], result["legacy_goals"]), (1, 1))
        np.testing.assert_allclose(result["waypoints"], [[x, 0] for x in range(20, 46, 5)], atol=1e-4)
        self.assertAlmostEqual(result["gui_max_x"], 45.3, places=4)
        self.assertAlmostEqual(result["legacy_max_x"], 45.3, places=4)
        self.assertEqual(result["gui_min_y"], 0)
        self.assertEqual(result["legacy_min_y"], 0)
        upstream = self.run_harness(x=-5, y=0)
        self.assertEqual(upstream["selected"], 2)
        self.assertEqual(upstream["off_route"], 0)
        self.assertAlmostEqual(upstream["distance"], 50.3, places=4)
        self.assertAlmostEqual(upstream["gui_max_x"], 45.3, places=4)
        past_goal = self.run_harness(x=50, y=0)
        self.assertEqual(past_goal["waypoints"], [])

    def test_overlapping_goal_routes_survive_lane_reordering(self):
        write_overlapping_goal_map(self.map_path, reverse_lanes=True)
        nav.process_map_navigation(self.map_path, max_goals=1, seed=0)
        result = self.run_harness(x=15, y=0)
        self.assertEqual(result["selected"], 1)
        self.assertEqual(result["off_route"], 0)
        self.assertAlmostEqual(result["distance"], 30.3, places=4)
        self.assertAlmostEqual(result["gui_max_x"], 45.3, places=4)

    def test_native_loader_rejects_goal_terminal_beyond_lane_end(self):
        write_overlapping_goal_map(self.map_path)
        navi = nav.process_map_navigation(self.map_path, max_goals=1, seed=0)
        path = nav.navigation_path_for(self.map_path)
        raw = bytearray(path.read_bytes())
        tables = struct.calcsize(nav._HEADER_FMT) + 4 * (
            3 * navi["num_lanes"] + 3 * navi["num_goals"] + navi["num_objects"]
        )
        longitudinal_table = tables + 4 * 5 * navi["num_lanes"] * navi["num_goals"]
        struct.pack_into("<f", raw, longitudinal_table, 1000.0)
        path.write_bytes(raw)
        completed = subprocess.run(
            [self.harness, str(self.map_path), "15", "0", "0"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("terminal progress", completed.stderr + completed.stdout)

    def test_commit_to_turn_after_divergence(self):
        result = self.run_harness(x=20, y=-10, heading=-0.79, previous=0)
        self.assertEqual((result["physical"], result["selected"]), (1, 1))
        self.assertLess(result["gui_min_y"], -20)
        self.assertLess(np.array(result["waypoints"])[:, 1].min(), -20)

    def test_no_goal_and_outside_shared_prefix_use_physical_lane(self):
        result = self.run_harness(goal=-1)
        self.assertEqual((result["physical"], result["selected"]), (1, 1))
        self.assertEqual(result["waypoints"], [])
        result = self.run_harness(x=12, y=-1.8, heading=-0.79)
        self.assertEqual((result["physical"], result["selected"]), (1, 1))

    def test_equal_cost_prefers_previous_branch_and_unreachable_retains_fallback(self):
        goal = self.navi["object_goal_idx"][0]
        for costs, previous, expected in (([100, 100], 0, 0), ([100, 100], 1, 1), ([nav.NAVI_UNREACHABLE_M] * 2, 0, 1)):
            self.navi["longitudinal_distance_to_goal_m"][goal, :2] = costs
            nav.write_navigation(self.navi, nav.navigation_path_for(self.map_path))
            result = self.run_harness(previous=previous)
            self.assertEqual(result["selected"], expected)

    def test_off_lane_retains_anchor_and_opposing_heading_does_not_choose_a_branch(self):
        result = self.run_harness(x=-10000, previous=0)
        self.assertEqual((result["physical"], result["selected"]), (-1, 0))
        result = self.run_harness(heading=np.pi)
        self.assertEqual((result["physical"], result["selected"]), (1, 1))

    def test_native_loader_rejects_bad_split_metadata_and_old_version(self):
        path = nav.navigation_path_for(self.map_path)
        original = path.read_bytes()
        branch_offset = len(original) - 4 * (3 * self.navi["num_branches"] + self.navi["num_lanes"] + 1)
        branch_lane_offset = branch_offset + 4 * (self.navi["num_lanes"] + 1)
        prefix_offset = branch_lane_offset + 4 * self.navi["num_branches"]
        cases = [
            (8, "<i", nav.NAVI_VERSION - 1, "unsupported version"),
            (struct.calcsize(nav._HEADER_FMT) + 4, "<i", 1000000, "non-lane entity"),
            (branch_offset, "<i", -1, "branch_offsets"),
            (branch_lane_offset, "<i", self.navi["num_lanes"], "branch_lane"),
            (prefix_offset, "<f", float("nan"), "prefix limit"),
            (prefix_offset, "<f", 10000.0, "prefix limit"),
        ]
        for offset, fmt, value, message in cases:
            raw = bytearray(original)
            struct.pack_into(fmt, raw, offset, value)
            path.write_bytes(raw)
            result = subprocess.run(
                [self.harness, str(self.map_path), "5", "0.18", "0"],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(message, result.stderr + result.stdout)
        for raw in (original[:-1], original + b"\0"):
            path.write_bytes(raw)
            result = subprocess.run(
                [self.harness, str(self.map_path), "5", "0.18", "0"],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("file size", result.stderr + result.stdout)


ENV_SCRIPT = textwrap.dedent(
    """
    import json
    import sys
    import numpy as np
    import torch
    from pufferlib.ocean.drive import binding
    from pufferlib.ocean.drive.drive import Drive
    from pufferlib.ocean.torch import Drive as DrivePolicy
    config = json.loads(sys.argv[7])
    resample_map = config.pop("resample_map", False)
    env = Drive(num_agents=1, num_maps=1, map_dir=sys.argv[1], navigation=int(sys.argv[2]),
                navigation_waypoints=int(sys.argv[3]), navigation_waypoint_spacing=float(sys.argv[4]),
                control_mode="control_vehicles", episode_length=91, resample_frequency=0,
                goal_obs_dropout_prob=float(sys.argv[5]), **config)
    obs, _ = env.reset()
    initial_obs = obs[0].tolist()
    goal_samples = []
    for seed in range(int(sys.argv[6])):
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
        "goal_samples": goal_samples,
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

    def _run_env(self, navigation, waypoints=16, spacing=5.0, goal_dropout_prob=0.0, reset_count=0, **config):
        return subprocess.run(
            [
                sys.executable,
                "-c",
                ENV_SCRIPT,
                self.tmp.name,
                str(navigation),
                str(waypoints),
                str(spacing),
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

    def test_snap_initial_goal_on_legacy_maps(self):
        xs = np.arange(TRAJECTORY_STEPS, dtype=np.float32) * SPEED_MPS * DT_SECONDS
        out = bytearray(b"synthetic_navi00")
        out += struct.pack("<4i", 0, 0, 1, 2)
        _write_entity(out, nav.VEHICLE, xs, np.zeros_like(xs), goal_x=95.3, goal_y=1.0)
        for y in (0.0, LANE_WIDTH_M):
            lane_x, lane_y = _polyline(0, 500, y, step_m=10.0)
            _write_entity(out, nav.ROAD_LANE, lane_x, lane_y)
        self.map_path.write_bytes(out)
        nav.process_map_navigation(self.map_path, max_goals=4, seed=0)
        for navigation in (0, 1):
            off = self._result(self._run_env(navigation, snap_goal_to_navi_endpoint=0))
            on = self._result(
                self._run_env(
                    navigation, snap_goal_to_navi_endpoint=1, goal_behavior=1, reset_count=2, resample_map=True
                )
            )
            np.testing.assert_allclose(off["initial_obs"][4:6], [95.3 * 0.005, 1.0 * 0.005], atol=1e-6)
            np.testing.assert_allclose(on["initial_obs"][4:6], [95.3 * 0.005, 0.0], atol=1e-6)
            np.testing.assert_allclose(on["goal_samples"], [[95.3 * 0.005, 0.0]] * 3, atol=1e-6)
            np.testing.assert_array_equal(np.delete(off["initial_obs"], [4, 5]), np.delete(on["initial_obs"], [4, 5]))

    def test_snap_projects_onto_associated_lane_not_hidden_copy(self):
        write_overlapping_goal_map(self.map_path, goal_y=0.03)
        navi = nav.process_map_navigation(self.map_path, max_goals=3, seed=0)
        navi["object_goal_idx"][0] = np.flatnonzero(navi["goal_lane"] == 1)[0]
        nav.write_navigation(navi, nav.navigation_path_for(self.map_path))
        off = self._result(self._run_env(0))
        on = self._result(self._run_env(0, snap_goal_to_navi_endpoint=1))
        np.testing.assert_allclose(off["initial_obs"][4:6], [45.3 * 0.005, 0.03 * 0.005], atol=1e-6)
        np.testing.assert_allclose(on["initial_obs"][4:6], [45.3 * 0.005, 0.1 * 0.005], atol=1e-6)

    def test_snap_does_not_change_legacy_agent_selection(self):
        xs = np.arange(TRAJECTORY_STEPS, dtype=np.float32) * SPEED_MPS * DT_SECONDS
        out = bytearray(b"synthetic_navi00")
        out += struct.pack("<4i", -1, 0, 2, 3)
        _write_entity(out, nav.VEHICLE, xs, np.full_like(xs, 3.0), goal_x=1.0, goal_y=3.0)
        _write_entity(out, nav.VEHICLE, xs + 99.0, np.zeros_like(xs), goal_x=95.0, goal_y=1.0)
        for start, end in ((0, 50), (50, 100)):
            lane_x, lane_y = _polyline(start, end, 0)
            _write_entity(out, nav.ROAD_LANE, lane_x, lane_y)
        lane_x, lane_y = _polyline(0, 100, LANE_WIDTH_M)
        _write_entity(out, nav.ROAD_LANE, lane_x, lane_y)
        self.map_path.write_bytes(out)
        navi = nav.process_map_navigation(self.map_path, max_goals=3, seed=0)
        navi["object_goal_idx"][0] = np.flatnonzero(navi["goal_lane"] == 0)[0]
        nav.write_navigation(navi, nav.navigation_path_for(self.map_path))
        off = self._result(self._run_env(0))
        on = self._result(self._run_env(0, snap_goal_to_navi_endpoint=1))
        np.testing.assert_allclose(off["initial_obs"][4:6], [-4.0 * 0.005, 1.0 * 0.005], atol=1e-6)
        np.testing.assert_allclose(on["initial_obs"][4:6], [-4.0 * 0.005, 0.0], atol=1e-6)
        np.testing.assert_array_equal(np.delete(off["initial_obs"], [4, 5]), np.delete(on["initial_obs"], [4, 5]))

    def test_snap_requires_legacy_sidecar(self):
        result = self._run_env(0, snap_goal_to_navi_endpoint=1)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(".navi", result.stderr)
        self.assertIn("goal snapping", result.stderr)

    def test_snap_does_not_change_generated_legacy_goals(self):
        write_synthetic_map(self.map_path, goal_y=1.0)
        nav.process_map_navigation(self.map_path, max_goals=4, seed=0)
        config = {"goal_behavior": 1, "goal_radius": 1000.0}
        off = self._result(self._run_env(0, **config))
        on = self._result(self._run_env(0, snap_goal_to_navi_endpoint=1, **config))
        self.assertNotEqual(off["initial_obs"][5], on["initial_obs"][5])
        np.testing.assert_array_equal(off["goal"], on["goal"])

    def test_snap_keeps_goal_without_a_matching_route(self):
        navi = nav.process_map_navigation(self.map_path, max_goals=4, seed=0)
        navi["object_goal_idx"][0] = -1
        nav.write_navigation(navi, nav.navigation_path_for(self.map_path))
        off = self._result(self._run_env(0))
        on = self._result(self._run_env(0, snap_goal_to_navi_endpoint=1))
        np.testing.assert_array_equal(off["initial_obs"], on["initial_obs"])

    def test_waypoints_reach_hidden_goal_copy_on_overlapping_straight_lane(self):
        write_overlapping_goal_map(self.map_path)
        nav.process_map_navigation(self.map_path, max_goals=1, seed=0)
        result = self._result(self._run_env(1))
        waypoints = np.array(result["waypoints"])
        valid = waypoints[:, 2] == 1
        self.assertTrue(0 < valid.sum() < 16)
        np.testing.assert_allclose(waypoints[valid, 1], 0, atol=1e-3)
        np.testing.assert_array_equal(waypoints[~valid], 0)
        self.assertGreater(result["feasibility"][1], 0)

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

    def test_goal_observation_dropout_resamples_on_reset(self):
        visible = self._result(self._run_env(0, goal_dropout_prob=0.0))
        masked = self._result(self._run_env(0, goal_dropout_prob=1.0))
        self.assertTrue(any(value != 0.0 for value in visible["goal"]))
        self.assertEqual(masked["goal"], [0.0, 0.0])

        samples = self._result(self._run_env(0, goal_dropout_prob=0.5, reset_count=20))["goal_samples"]
        self.assertIn([0.0, 0.0], samples)
        self.assertTrue(any(any(value != 0.0 for value in sample) for sample in samples))

    def test_stale_sidecar_aborts_env_creation(self):
        nav.process_map_navigation(self.map_path, max_goals=4, seed=0)
        map_bytes = bytearray(self.map_path.read_bytes())
        map_bytes[-5] ^= 0x01
        self.map_path.write_bytes(bytes(map_bytes))
        for navigation, config in ((1, {}), (0, {"snap_goal_to_navi_endpoint": 1})):
            result = self._run_env(navigation, **config)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("fingerprint mismatch", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
