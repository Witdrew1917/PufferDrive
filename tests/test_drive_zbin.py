"""zpufferdrive (zdrive) v14 map binaries: loading, navigation gating and parity with zpufferdrive.

Running the test: python -m unittest tests/test_drive_zbin.py

The parity test needs a zpufferdrive checkout (ZPUFFERDRIVE_ROOT, default ~/Repositories/zpufferdrive) and zdrive
binaries (ZBIN_MAP_DIR, default ~/Data/zpufferdrive/waymo_binaries/interactive_1k); it is skipped without them.
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

from pufferlib.ocean.drive.drive import MAP_FORMAT_LEGACY, MAP_FORMAT_ZBIN, detect_map_format


REPO_ROOT = Path(__file__).resolve().parents[1]
ZPUFFERDRIVE_ROOT = Path(os.environ.get("ZPUFFERDRIVE_ROOT", Path.home() / "Repositories/zpufferdrive"))
ZBIN_MAP_DIR = Path(os.environ.get("ZBIN_MAP_DIR", Path.home() / "Data/zpufferdrive/waymo_binaries/interactive_1k"))
PARITY_MAPS = int(os.environ.get("ZBIN_PARITY_MAPS", "20"))

TRAJECTORY_STEPS = 91
LANE_LENGTH_M = 200.0
SEGMENT_M = 10.0
SUB_LINES = 2
SPEED_LIMIT_MPS = 10.0
START_X_M = 22.0
SPEED_MPS = 5.0
GOAL_X_M = 195.0


def write_zbin(path, num_goals=1):
    """v14 zbin: one vehicle driving east on a straight 200 m lane, routed to the lane end (goal index 0).

    Each 10 m lane segment is one navi line split into two 5 m sub-lines; the route chains every sample to the next
    one, so the navi lookahead follows the lane centre and distance_to_goal is the remaining lane length.
    """
    steps = np.arange(TRAJECTORY_STEPS, dtype=np.float32)
    zeros = np.zeros(TRAJECTORY_STEPS, "<f4")
    section1 = bytearray(b"synthetic_zbin".ljust(512, b"\0") + bytes(128 + 48))
    section1 += struct.pack("<4i", 0, 0, 1, 0)  # static-expert goals, sdc index, tracks_to_predict = [0]
    section1 += struct.pack("<i", 1)  # num_objects
    section1 += struct.pack("<2i3fi", 1, 7, 2.0, 4.5, 1.5, TRAJECTORY_STEPS)  # vehicle id 7
    for values in (START_X_M + steps * SPEED_MPS * 0.1, zeros, zeros, np.full(TRAJECTORY_STEPS, SPEED_MPS), zeros):
        section1 += np.asarray(values, "<f4").tobytes()
    section1 += zeros.tobytes() + zeros.tobytes() + np.ones(TRAJECTORY_STEPS, "<i4").tobytes()  # vz, heading, valid
    section1 += struct.pack("<i4f2i", 0, 0.0, 1.0, GOAL_X_M, 0.0, 0, 0)  # policy agent, goal_index 0, not a zombie
    section1 += zeros.tobytes() + zeros.tobytes() + np.zeros(TRAJECTORY_STEPS, "<i4").tobytes()

    xs = np.arange(0.0, LANE_LENGTH_M + 1e-6, SEGMENT_M, dtype=np.float32)
    section2 = bytearray(struct.pack("<i", 2))
    section2 += struct.pack("<3i", 4, 100, len(xs)) + xs.tobytes() + bytes(8 * len(xs))  # ROAD_LANE
    section2 += struct.pack("<2fi", 3.7, SPEED_LIMIT_MPS, 0)
    section2 += struct.pack("<3i", 7, 101, len(xs)) + xs.tobytes() + np.full(len(xs), -2.0, "<f4").tobytes()
    section2 += bytes(4 * len(xs)) + struct.pack("<2fi", 0.0, 0.0, 0)  # ROAD_EDGE

    num_lines = len(xs) - 1
    section3 = bytearray(struct.pack("<3i", num_lines, 0, SUB_LINES))
    for line in range(num_lines):
        section3 += struct.pack("<3i", 100, line, SUB_LINES)

    section4 = bytearray(struct.pack("<i", num_goals))
    for goal in range(num_goals):
        for line in range(num_lines):
            sample_x = line * SEGMENT_M + np.arange(SUB_LINES) * SEGMENT_M / SUB_LINES
            last = line == num_lines - 1
            next_entity = [goal * num_lines + line] * (SUB_LINES - 1) + [-1 if last else goal * num_lines + line + 1]
            next_point = list(range(1, SUB_LINES)) + [0]
            remaining = (LANE_LENGTH_M - sample_x).astype("<f4")
            section4 += np.asarray(next_entity, "<i4").tobytes() + np.asarray(next_point, "<i4").tobytes()
            section4 += bytes(4 * SUB_LINES) + remaining.tobytes() + bytes(4 * SUB_LINES) + remaining.tobytes()

    offsets = [24]
    for section in (section1, section2, section3):
        offsets.append(offsets[-1] + len(section))
    Path(path).write_bytes(struct.pack("<6I", 14, 4, *offsets) + section1 + section2 + section3 + section4)


def write_lane_selection_zbin(path, distances, lane_ys=(0.0, 0.2), num_goals=1):
    write_zbin(path)
    original = Path(path).read_bytes()
    _, _, section1_start, section2_start, _, _ = struct.unpack_from("<6I", original)
    section1 = original[section1_start:section2_start]
    section2 = bytearray(struct.pack("<i", len(lane_ys)))
    section3 = bytearray(struct.pack("<3i", len(lane_ys), 0, 4))
    for i, y in enumerate(lane_ys):
        section2 += struct.pack("<3i", 4, 100 + i, 2)
        section2 += struct.pack("<2f", 0.0, 20.0) + struct.pack("<2f", y, y) + bytes(8)
        section2 += struct.pack("<2fi", 3.7, 10.0 + i, 0)
        section3 += struct.pack("<3i", 100 + i, 0, 4)
    section4 = bytearray(struct.pack("<i", num_goals))
    for goal in range(num_goals):
        for i, values in enumerate(distances):
            entity = goal * len(lane_ys) + i
            section4 += struct.pack("<4i", entity, entity, entity, entity)
            section4 += struct.pack("<4i", 1, 2, 3, 4)
            section4 += bytes(16) + struct.pack("<4f", *([20.0 + i * 10.0] * 4)) + bytes(16)
            section4 += struct.pack("<4f", *values)
    offsets = [24]
    for section in (section1, section2, section3):
        offsets.append(offsets[-1] + len(section))
    Path(path).write_bytes(struct.pack("<6I", 14, 4, *offsets) + section1 + section2 + section3 + section4)


ENV_SCRIPT = textwrap.dedent(
    """
    import json
    import sys
    import numpy as np
    from pufferlib.ocean.drive.drive import Drive
    config = json.loads(sys.argv[2])
    env = Drive(num_agents=1, num_maps=1, map_dir=sys.argv[1], control_mode="control_vehicles",
                episode_length=91, resample_frequency=0, **config)
    obs, _ = env.reset(seed=0)
    observations, rewards = [obs[0].tolist()], []
    for _ in range(5):
        obs, reward, *_ = env.step(np.zeros_like(env.actions))
        observations.append(obs[0].tolist())
        rewards.append(float(reward[0]))
    print("RESULT " + json.dumps({
        "ego": env.ego_features,
        "waypoint_features": env.waypoint_features,
        "max_waypoints": env.max_waypoints,
        "observations": observations,
        "rewards": rewards,
    }))
    env.close()
    """
)


class TestZbinEnv(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.map_dir = Path(self.tmp.name)
        write_zbin(self.map_dir / "map_000.bin")

    def _run_env(self, map_dir=None, **config):
        return subprocess.run(
            [sys.executable, "-c", ENV_SCRIPT, str(map_dir or self.map_dir), json.dumps(config)],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )

    def _result(self, **config):
        completed = self._run_env(**config)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        line = next(line for line in completed.stdout.splitlines() if line.startswith("RESULT "))
        result = json.loads(line[len("RESULT ") :])
        result["observations"] = np.array(result["observations"])
        return result

    def test_format_detection(self):
        self.assertEqual(detect_map_format(self.map_dir / "map_000.bin"), MAP_FORMAT_ZBIN)
        legacy = self.map_dir / "legacy.bin"
        legacy.write_bytes(b"synthetic_navi00" + bytes(64))
        self.assertEqual(detect_map_format(legacy), MAP_FORMAT_LEGACY)

    def test_navigation_flag_adds_feasibility_and_route_lookahead(self):
        off = self._result(navigation=0)
        on = self._result(navigation=1, navigation_waypoints=4, navigation_lookahead_distance=40.0)
        self.assertEqual(on["ego"] - off["ego"], 3)
        self.assertEqual(on["waypoint_features"], 2)
        self.assertEqual(on["observations"].shape[1] - off["observations"].shape[1], 3 + 4 * 2)

        # Agent at x = 22 m: feasibility index 0 of the sub-line starting at 20 m -> 180 m left at 10 m/s.
        first = on["observations"][0]
        np.testing.assert_allclose(first[on["ego"] - 3 : on["ego"]], [0.0, 180.0 / SPEED_LIMIT_MPS / 30.0, 0.0])
        # Lookahead along the lane centre every 40 m / 4 from the agent, normalised by 150 m, in the ego frame.
        waypoints = first[-8:].reshape(4, 2)
        np.testing.assert_allclose(waypoints[:, 0], np.arange(1, 5) * 10.0 / 150.0, atol=1e-6)
        np.testing.assert_allclose(waypoints[:, 1], 0.0, atol=1e-6)

    def test_lookahead_slots_past_route_end_are_zero(self):
        on = self._result(navigation=1, navigation_waypoints=10, navigation_lookahead_distance=150.0)
        # 178 m of route remain from x = 22 m, so lookahead slots at 15..150 m all fit, and the last row
        # (x = 24.5 m) still has 175.5 m.
        waypoints = on["observations"][0][-20:].reshape(10, 2)
        np.testing.assert_allclose(waypoints[:, 0], np.arange(1, 11) * 15.0 / 150.0, atol=1e-6)
        short = self.map_dir / "short"
        short.mkdir()
        shutil.copy(self.map_dir / "map_000.bin", short / "map_000.bin")
        # Lookahead of N waypoints over D = norm from close to the end of the route is truncated with zeros.
        result = self._result(
            map_dir=short, navigation=1, navigation_waypoints=10, navigation_lookahead_distance=150.0, init_steps=80
        )
        waypoints = result["observations"][0][-20:].reshape(10, 2)
        remaining = LANE_LENGTH_M - (START_X_M + 80 * SPEED_MPS * 0.1)
        valid = np.arange(1, 11) * 15.0 <= remaining
        self.assertTrue(0 < valid.sum() < 10)
        np.testing.assert_array_equal(waypoints[~valid], 0.0)

    def test_navigation_off_ignores_loaded_routes(self):
        # Same scene without any routes: with navigation=0 the agent must see and be rewarded identically.
        no_routes = self.map_dir / "no_routes"
        no_routes.mkdir()
        write_zbin(no_routes / "map_000.bin", num_goals=0)
        config = dict(navigation=0, reward_progress_navi=1.0)
        with_routes = self._result(**config)
        without_routes = self._result(map_dir=no_routes, **config)
        np.testing.assert_array_equal(with_routes["observations"], without_routes["observations"])
        self.assertEqual(with_routes["rewards"], without_routes["rewards"])

    def test_navigation_does_not_move_the_goal(self):
        off = self._result(navigation=0)
        on = self._result(navigation=1)
        np.testing.assert_array_equal(off["observations"][:, 4:6], on["observations"][:, 4:6])
        self.assertGreater(off["observations"][0, 4], 0.0)

    def test_route_snapped_goal_is_replaced_by_last_logged_position(self):
        # The binary goal (GOAL_X_M) is zdrive's route-snapped goal; legacy maps use the last valid logged sample.
        last_logged_x = START_X_M + (TRAJECTORY_STEPS - 1) * SPEED_MPS * 0.1
        first = self._result(navigation=1)["observations"][0]
        np.testing.assert_allclose(first[4:6] / 0.005, [last_logged_x - START_X_M, 0.0], atol=1e-3)

    def test_invalid_lookahead_config_is_rejected(self):
        for config, message in (
            (dict(navigation_waypoints=51), "navigation_waypoints"),
            (dict(navigation_lookahead_distance=200.0), "navigation_lookahead_distance"),
            (dict(navigation_lookahead_distance=0.0), "navigation_lookahead_distance"),
        ):
            completed = self._run_env(navigation=1, **config)
            self.assertNotEqual(completed.returncode, 0, config)
            self.assertIn(message, completed.stdout + completed.stderr)

    def test_truncated_binary_is_rejected(self):
        path = self.map_dir / "map_000.bin"
        path.write_bytes(path.read_bytes()[:-4])
        completed = self._run_env(navigation=0)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("malformed zpufferdrive binary", completed.stdout + completed.stderr)


class TestZbinLaneSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("cc")
        if compiler is None:
            raise unittest.SkipTest("C compiler is required for lane selection tests")
        cls.build_dir = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.build_dir.cleanup)
        cls.harness = str(Path(cls.build_dir.name) / "zbin_navigation_harness")
        raylib = "raylib-5.5_macos" if sys.platform == "darwin" else "raylib-5.5_linux_amd64"
        command = [
            compiler,
            "-O1",
            "-D_GNU_SOURCE",
            "-ffunction-sections",
            "-fdata-sections",
            "-I" + str(REPO_ROOT / "pufferlib/ocean/drive"),
            "-I" + str(REPO_ROOT / raylib / "include"),
            "-I" + str(REPO_ROOT / "inih-r62"),
            str(REPO_ROOT / "tests/zbin_navigation_harness.c"),
            "-Wl,-dead_strip" if sys.platform == "darwin" else "-Wl,--gc-sections",
            "-lm",
            "-o",
            cls.harness,
        ]
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "map_000.bin"

    def probe(self, x=17.0, y=0.01, heading=0.0, goal=0, navigation=1):
        completed = subprocess.run(
            [
                self.harness,
                str(self.path),
                "1",
                "1",
                "2",
                "probe",
                str(x),
                str(y),
                str(heading),
                str(goal),
                str(navigation),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return json.loads(next(line[7:] for line in completed.stdout.splitlines() if line.startswith("RESULT ")))

    def test_projected_sample_selects_route_and_consistent_navigation_state(self):
        write_lane_selection_zbin(self.path, [[10, 10, 200, 200], [100, 100, 30, 30]])
        result = self.probe()
        self.assertEqual((result["lane"], result["entity"], result["point"]), (101, 1, 3))
        self.assertEqual((result["distance"], result["center"], result["speed_limit"]), (30, 30, 11))
        self.assertEqual(result["previous_distance"], result["distance"])
        self.assertGreater(result["waypoint_y"], 0)
        self.assertEqual(result["off_route"], 0)
        self.assertEqual(self.probe(navigation=0)["lane"], 101)
        self.assertEqual(self.probe(x=2)["lane"], 100)
        self.assertEqual(self.probe(x=10)["lane"], 100)

    def test_reachability_is_checked_at_projection_and_falls_back_to_geometry(self):
        write_lane_selection_zbin(self.path, [[10, 10, 2000, 2000], [100, 100, 30, 30]])
        self.assertEqual(self.probe()["lane"], 101)
        write_lane_selection_zbin(self.path, [[10, 10, 2000, 2000], [100, 100, 2000, 2000]])
        fallback = self.probe()
        self.assertEqual((fallback["lane"], fallback["entity"], fallback["off_route"]), (100, 0, 1))
        self.assertEqual(self.probe(goal=-1)["lane"], 100)
        self.assertEqual(self.probe(goal=5)["lane"], 100)
        write_lane_selection_zbin(self.path, [[10] * 4, [30] * 4], num_goals=0)
        self.assertEqual(self.probe(goal=-1)["lane"], 100)

    def test_route_preference_respects_distance_and_heading_limits(self):
        write_lane_selection_zbin(self.path, [[200] * 4, [30] * 4], lane_ys=(0, 1.5))
        self.assertEqual(self.probe(y=0)["lane"], 100)
        write_lane_selection_zbin(self.path, [[2000] * 4, [30] * 4], lane_ys=(0, 4.1))
        self.assertEqual(self.probe(y=0)["lane"], 100)
        write_lane_selection_zbin(self.path, [[200] * 4, [30] * 4])
        self.assertEqual(self.probe(heading=np.pi)["lane"], -1)

    def test_projected_ranking_does_not_depend_on_closer_lane_order(self):
        write_lane_selection_zbin(self.path, [[100, 100, 30, 30], [10, 10, 200, 200]], lane_ys=(0.2, 0))
        self.assertEqual(self.probe()["lane"], 100)


class TestZbinParity(unittest.TestCase):
    """Navigation features of every logged object at every logged pose, PufferDrive vs zpufferdrive itself."""

    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("cc")
        if compiler is None:
            raise unittest.SkipTest("C compiler is required for the zdrive parity test")
        if not (ZPUFFERDRIVE_ROOT / "ocean/zdrive/zdrive.c").exists():
            raise unittest.SkipTest(f"zpufferdrive checkout not found at {ZPUFFERDRIVE_ROOT}")
        cls.maps = sorted(ZBIN_MAP_DIR.glob("map_*.bin"))[:PARITY_MAPS]
        if not cls.maps:
            raise unittest.SkipTest(f"no zdrive binaries in {ZBIN_MAP_DIR}")
        cls.build_dir = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.build_dir.cleanup)
        raylib = "raylib-5.5_macos" if sys.platform == "darwin" else "raylib-5.5_linux_amd64"
        cls.oracle = str(Path(cls.build_dir.name) / "zdrive_navigation_oracle")
        cls.harness = str(Path(cls.build_dir.name) / "zbin_navigation_harness")
        builds = (
            [
                compiler,
                "-O1",
                "-w",
                "-I" + str(ZPUFFERDRIVE_ROOT),
                "-I" + str(ZPUFFERDRIVE_ROOT / "vendor"),
                "-I" + str(ZPUFFERDRIVE_ROOT / "ocean/zdrive"),
                "-I" + str(ZPUFFERDRIVE_ROOT / "vendor" / raylib / "include"),
                str(REPO_ROOT / "tests/zdrive_navigation_oracle.c"),
                "-lm",
                "-lpthread",
                "-o",
                cls.oracle,
            ],
            [
                compiler,
                "-O1",
                "-w",
                "-D_GNU_SOURCE",
                "-ffunction-sections",
                "-fdata-sections",
                "-I" + str(REPO_ROOT / "pufferlib/ocean/drive"),
                "-I" + str(REPO_ROOT / raylib / "include"),
                "-I" + str(REPO_ROOT / "inih-r62"),
                str(REPO_ROOT / "tests/zbin_navigation_harness.c"),
                "-Wl,-dead_strip" if sys.platform == "darwin" else "-Wl,--gc-sections",
                "-lm",
                "-o",
                cls.harness,
            ],
        )
        for command in builds:
            completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=300)
            if completed.returncode:
                raise RuntimeError(completed.stderr)

    def _dump(self, binary, path, waypoints, distance):
        completed = subprocess.run(
            [binary, str(path), str(TRAJECTORY_STEPS), str(waypoints), str(distance)],
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return completed.stdout.splitlines()

    def test_navigation_features_match_zdrive(self):
        for waypoints, distance in ((10, 120.0), (50, 150.0), (3, 7.5)):
            routed = 0
            for path in self.maps:
                with self.subTest(map=path.name, waypoints=waypoints, distance=distance):
                    expected = self._dump(self.oracle, path, waypoints, distance)
                    actual = self._dump(self.harness, path, waypoints, distance)
                    self.assertEqual(len(actual), len(expected))
                    # Row: step, object, id, goal_index, speed_limit, distance_to_goal, 3 feasibility, lookahead.
                    for actual_row, expected_row in zip(actual, expected):
                        self.assertEqual(actual_row, expected_row)
                    routed += sum(float(row.split()[7]) > 0 for row in expected)
            self.assertGreater(routed, 0)


if __name__ == "__main__":
    unittest.main()
