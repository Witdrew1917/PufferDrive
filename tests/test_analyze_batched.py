import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from examples import analyze_batched as batched


RUN = "laveno-chalmers-university-of-technology/pufferlib/ssuuqvs3"


class TestAnalyzeBatched(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.checkpoint = self.directory / "model.pt"
        self.base = {"env": {"dynamics_model": "classic", "navigation": 0}, "load_id": "training-run"}

    def save_config(self, env):
        torch.save({"full_args": {**copy.deepcopy(self.base), "env": env}}, self.checkpoint)

    def test_artifact_resolution(self):
        api = Mock()
        api.artifact.return_value.download.return_value = str(self.directory)
        with self.assertRaisesRegex(ValueError, "expected one model checkpoint, found 0"):
            batched.download_checkpoint(api, RUN)
        api.reset_mock()
        self.checkpoint.touch()
        (self.directory / "trainer_state.pt").touch()
        self.assertEqual(batched.download_checkpoint(api, RUN), self.checkpoint)
        api.artifact.assert_called_once_with(f"{RUN}:v0", type="model")
        (self.directory / "another.pt").touch()
        with self.assertRaisesRegex(ValueError, "expected one model checkpoint, found 2"):
            batched.download_checkpoint(api, RUN)

    def test_checkpoint_dynamics_are_preserved(self):
        self.save_config({"dynamics_model": "jerk", "navigation": 1})
        config = batched.prepare_checkpoint_config(self.checkpoint, self.base, RUN)
        self.assertEqual(config["env"], {"dynamics_model": "jerk", "navigation": 1})
        self.assertIsNone(config["load_id"])
        self.assertEqual(config["load_model_path"], str(self.checkpoint))
        with self.assertRaisesRegex(ValueError, "conflicts"):
            batched.prepare_checkpoint_config(self.checkpoint, self.base, RUN, "classic")

    def test_missing_dynamics_requires_pairing_and_explicit_navigation(self):
        for navigation in (0, 1):
            with self.subTest(navigation=navigation):
                self.save_config({"navigation": navigation})
                with self.assertRaisesRegex(ValueError, "supply --dynamics"):
                    batched.prepare_checkpoint_config(self.checkpoint, self.base, RUN)
                config = batched.prepare_checkpoint_config(self.checkpoint, self.base, RUN, "delta_local")
                self.assertEqual(config["env"], {"navigation": navigation, "dynamics_model": "delta_local"})
        for env in ({}, {"navigation": None}, {"navigation": 2}):
            with self.subTest(env=env):
                self.save_config(env)
                with self.assertRaisesRegex(ValueError, "explicitly set env.navigation"):
                    batched.prepare_checkpoint_config(self.checkpoint, self.base, RUN, "classic")
        torch.save({}, self.checkpoint)
        with self.assertRaisesRegex(ValueError, "explicitly set env.navigation"):
            batched.prepare_checkpoint_config(self.checkpoint, self.base, RUN, "classic")

    def test_cli_pairs_dynamics_with_each_run(self):
        other = "entity/project/other"
        args = batched.parse_args([RUN, other, "--dynamics", RUN, "classic", "--dynamics", other, "jerk"])
        self.assertEqual(args.runs, [RUN, other])
        self.assertEqual(args.dynamics, {RUN: "classic", other: "jerk"})
        for arguments in (
            ["invalid"],
            [RUN, "--dynamics", other, "classic"],
            [RUN, "--dynamics", RUN, "invalid"],
            [RUN, "--dynamics", RUN, "classic", "--dynamics", RUN, "jerk"],
        ):
            with self.subTest(arguments=arguments), patch("sys.stderr"), self.assertRaises(SystemExit):
                batched.parse_args(arguments)

    def test_rollout_resampling_preserves_checkpoint_settings(self):
        config = {"env": {"dynamics_model": "jerk", "navigation": 1, "episode_length": 100}}
        original = copy.deepcopy(config)
        env = Mock()
        env.driver_env.num_envs = 2
        evaluator = Mock()
        evaluator.rollout.return_value = [{}, {"n": 0}, {"n": 2, "score": 0.75}]
        with patch.object(batched, "load_env", return_value=env) as load_env:
            rows = batched.run_mode(
                evaluator,
                Mock(),
                config,
                "validation",
                "control_sdc_only",
                "model.pt",
                "idm_interactive",
                200,
                controller_overrides={"non_sdc_controller": "idm"},
                total_eval_agents=batched.NUM_AGENTS_PER_VECENV + 1,
            )
        self.assertEqual(config, original)
        eval_env = load_env.call_args.args[1]["env"]
        self.assertEqual(eval_env["dynamics_model"], "jerk")
        self.assertEqual(eval_env["navigation"], 1)
        self.assertEqual(eval_env["non_sdc_controller"], "idm")
        self.assertEqual(eval_env["episode_length"], 200)
        self.assertEqual([row["scene_idx"] for row in rows], [0, 2])
        self.assertEqual([row["score"] for row in rows], [0.75, 0.75])
        env.driver_env.resample_maps.assert_called_once()
        env.close.assert_called_once()

    def test_rollout_failure_closes_environment_and_propagates(self):
        evaluator = Mock()
        evaluator.rollout.side_effect = RuntimeError("rollout failed")
        with patch.object(batched, "load_env") as load_env, self.assertRaisesRegex(RuntimeError, "rollout failed"):
            batched.run_mode(evaluator, Mock(), self.base, "validation", "control_vehicles", "model.pt", "sp_val", 10)
        load_env.return_value.close.assert_called_once()

    def test_evaluates_all_modes_for_multiple_runs(self):
        paths = [RUN, "entity/project/other"]
        self.save_config({"dynamics_model": "delta_local", "navigation": 0})

        def load_base_config(env_name):
            self.assertEqual(batched.sys.argv, ["analyze_batched.py"])
            return self.base

        def rollout(*args, **kwargs):
            return [{**dict.fromkeys(batched.METRICS, 0.0), "mode": args[6]}]

        output = self.directory / "results" / "eval.csv"
        original_argv = ["analyze_batched.py", "--foreign-argument"]
        with (
            patch.object(batched.sys, "argv", original_argv),
            patch.object(batched, "load_config", side_effect=load_base_config),
            patch.object(batched.wandb, "Api"),
            patch.object(batched, "download_checkpoint", return_value=self.checkpoint),
            patch.object(batched, "load_env") as load_env,
            patch.object(batched, "load_policy") as load_policy,
            patch.object(batched, "CheckpointEvaluator"),
            patch.object(batched, "run_mode", side_effect=rollout) as run_mode,
        ):
            df = batched.main([*paths, "--output-csv", str(output)])
            self.assertIs(batched.sys.argv, original_argv)
        rows = df.to_dict("records")
        self.assertEqual(len(batched.pd.read_csv(output)), 10)
        self.assertEqual(load_policy.call_count, 2)
        self.assertEqual(load_env.return_value.close.call_count, 2)
        self.assertEqual(run_mode.call_count, 10)
        self.assertEqual({row["run_path"] for row in rows}, set(paths))
        self.assertEqual(
            {row["mode"] for row in rows}, {"sp_train", "sp_val", "hr_val", "hr_interactive", "idm_interactive"}
        )
        idm_call = run_mode.call_args
        self.assertEqual(idm_call.kwargs["controller_overrides"]["non_sdc_controller"], "idm")
        self.assertEqual(idm_call.kwargs["total_eval_agents"], batched.NUM_TOTAL_INTERACTIVE_EVAL_AGENTS)
        self.assertFalse({"sp_maps", "is_regularized", "anchor_maps"} & rows[0].keys())


if __name__ == "__main__":
    unittest.main()
