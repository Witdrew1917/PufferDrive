import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from scripts.analyze import analyze_batched as batched


RUN = "laveno-chalmers-university-of-technology/pufferlib/ssuuqvs3"


class TestAnalyzeBatched(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.artifacts = self.directory / "artifacts"
        artifacts_patch = patch.object(batched, "ARTIFACTS_DIR", self.artifacts)
        artifacts_patch.start()
        self.addCleanup(artifacts_patch.stop)
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

    def test_reuses_verified_cached_artifacts(self):
        for cached in (self.artifacts / f"{RUN}:v0", self.artifacts / "ssuuqvs3:v0"):
            with self.subTest(cached=cached):
                cached.mkdir(parents=True)
                checkpoint = cached / "model.pt"
                checkpoint.touch()
                api = Mock()
                self.assertEqual(batched.download_checkpoint(api, RUN), checkpoint)
                api.artifact.return_value.verify.assert_called_once_with(root=str(cached))
                api.artifact.return_value.download.assert_not_called()
                checkpoint.unlink()
                cached.rmdir()

    def test_invalid_cached_artifact_is_downloaded_again(self):
        cached = self.artifacts / f"{RUN}:v0"
        cached.mkdir(parents=True)
        api = Mock()
        artifact = api.artifact.return_value

        def download(**kwargs):
            checkpoint = cached / "model.pt"
            checkpoint.write_bytes(b"fresh model")
            return str(cached)

        artifact.download.side_effect = download
        for error in (ValueError("digest mismatch"), FileNotFoundError("missing model")):
            with self.subTest(error=error):
                artifact.verify.side_effect = error
                self.assertEqual(batched.download_checkpoint(api, RUN).read_bytes(), b"fresh model")
                artifact.download.assert_called_with(root=str(cached))

    def test_no_cached_forces_fresh_download_and_persists_it(self):
        cached = self.artifacts / f"{RUN}:v0"
        cached.mkdir(parents=True)
        checkpoint = cached / "model.pt"
        checkpoint.write_bytes(b"old model")
        api = Mock()
        artifact = api.artifact.return_value

        def download(root, skip_cache):
            self.assertTrue(skip_cache)
            self.assertNotEqual(Path(root), cached)
            self.assertEqual(list(Path(root).iterdir()), [])
            (Path(root) / "model.pt").write_bytes(b"fresh model")
            return root

        artifact.download.side_effect = download
        result = batched.download_checkpoint(api, RUN, no_cached=True)
        self.assertEqual(result, checkpoint)
        self.assertEqual(result.read_bytes(), b"fresh model")
        artifact.verify.assert_not_called()
        artifact.download.assert_called_once()
        self.assertFalse(Path(artifact.download.call_args.kwargs["root"]).exists())
        self.assertEqual(batched.download_checkpoint(api, RUN), checkpoint)
        self.assertEqual(artifact.download.call_count, 1)

    def test_incomplete_forced_download_preserves_existing_checkpoint(self):
        cached = self.artifacts / f"{RUN}:v0"
        cached.mkdir(parents=True)
        checkpoint = cached / "model.pt"
        checkpoint.write_bytes(b"old model")
        api = Mock()
        artifact = api.artifact.return_value
        artifact.download.side_effect = lambda root, skip_cache: root
        with self.assertRaisesRegex(ValueError, "expected one model checkpoint, found 0"):
            batched.download_checkpoint(api, RUN, no_cached=True)
        self.assertEqual(checkpoint.read_bytes(), b"old model")
        self.assertFalse(Path(artifact.download.call_args.kwargs["root"]).exists())

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
        self.assertEqual(args.output_csv, Path(batched.OUTPUT_CSV))
        self.assertFalse(args.no_cached)
        self.assertTrue(batched.parse_args([RUN, "--no-cached"]).no_cached)
        for arguments in (
            ["invalid"],
            [RUN, "--dynamics", other, "classic"],
            [RUN, "--dynamics", RUN, "invalid"],
            [RUN, "--dynamics", RUN, "classic", "--dynamics", RUN, "jerk"],
        ):
            with self.subTest(arguments=arguments), patch("sys.stderr"), self.assertRaises(SystemExit):
                batched.parse_args(arguments)

    def test_batch_file_parses_comments_and_per_run_overrides(self):
        batch = self.directory / "batch.txt"
        other = "entity/project/other"
        batch.write_text(
            f"# group\n{RUN} --env.goal-obs-dropout-prob=0.0 --env.navigation 1 # inline\n"
            f"\n{other} --env.map-dir='a # b'\n{RUN} --env.navigation=0\n",
            encoding="utf-8",
        )
        args = batched.parse_args([str(batch), "--dynamics", other, "jerk"])
        self.assertEqual(
            args.entries,
            [
                (RUN, {"env.goal_obs_dropout_prob": 0.0, "env.navigation": 1}),
                (other, {"env.map_dir": "a # b"}),
                (RUN, {"env.navigation": 0}),
            ],
        )
        self.assertEqual(args.dynamics, {other: "jerk"})
        self.assertEqual(args.output_csv, Path("results/batch.csv"))
        custom_output = self.directory / "custom.csv"
        self.assertEqual(batched.parse_args([str(batch), "--output-csv", str(custom_output)]).output_csv, custom_output)

    def test_batch_file_reports_invalid_entries(self):
        batch = self.directory / "batch.txt"
        for contents in (
            "# no runs\n",
            "invalid\n",
            f"{RUN} --env.navigation\n",
            f"{RUN} --env.navigation=1 --env.navigation=0\n",
            f"{RUN} unexpected\n",
            f"{RUN} --env.navigation='unterminated\n",
        ):
            with self.subTest(contents=contents):
                batch.write_text(contents, encoding="utf-8")
                with patch("sys.stderr"), self.assertRaises(SystemExit):
                    batched.parse_args([str(batch)])

    def test_batch_overrides_apply_to_policy_and_rollouts(self):
        self.base["env"].update(goal_obs_dropout_prob=0.8, episode_length=91)
        self.save_config({"dynamics_model": "classic", "navigation": 0, "goal_obs_dropout_prob": 0.7})
        batch = self.directory / "batch.txt"
        batch.write_text(
            f"{RUN} --env.goal-obs-dropout-prob=0.0 --env.episode-length=123\nentity/project/other\n",
            encoding="utf-8",
        )
        with (
            patch.object(batched, "load_config", return_value=self.base),
            patch.object(batched.wandb, "Api"),
            patch.object(batched, "download_checkpoint", return_value=self.checkpoint),
            patch.object(batched, "load_env") as load_env,
            patch.object(batched, "load_policy"),
            patch.object(batched, "CheckpointEvaluator"),
            patch.object(batched, "run_mode", return_value=[]) as run_mode,
        ):
            batched.main([str(batch), "--output-csv", str(self.directory / "out.csv")])
        first_policy_config = load_env.call_args_list[0].args[1]["env"]
        second_policy_config = load_env.call_args_list[1].args[1]["env"]
        self.assertEqual(
            (first_policy_config["goal_obs_dropout_prob"], first_policy_config["episode_length"]), (0.0, 123)
        )
        self.assertEqual(
            (second_policy_config["goal_obs_dropout_prob"], second_policy_config["episode_length"]), (0.7, 200)
        )
        first_call = run_mode.call_args_list[0]
        eval_config = batched.make_eval_config(
            first_call.args[2], "maps", "control_vehicles", 10, overrides=first_call.kwargs["overrides"]
        )
        self.assertEqual(
            (eval_config["env"]["goal_obs_dropout_prob"], eval_config["env"]["episode_length"]), (0.0, 123)
        )
        self.assertEqual(run_mode.call_args_list[4].kwargs["overrides"], {})
        self.assertEqual(self.base["env"]["goal_obs_dropout_prob"], 0.8)

    def test_unknown_override_is_rejected(self):
        self.save_config({"dynamics_model": "classic", "navigation": 0, "legacy_setting": 1})
        config = batched.prepare_checkpoint_config(self.checkpoint, self.base, RUN, overrides={"env.legacy_setting": 2})
        self.assertEqual(config["env"]["legacy_setting"], 2)
        with self.assertRaisesRegex(ValueError, "Unknown config override"):
            batched.prepare_checkpoint_config(self.checkpoint, self.base, RUN, overrides={"env.goal_obs_dropout": 0.0})
        with self.assertRaisesRegex(ValueError, "managed by the evaluator"):
            batched.prepare_checkpoint_config(self.checkpoint, self.base, RUN, overrides={"load_id": "other"})

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
            patch.object(batched, "download_checkpoint", return_value=self.checkpoint) as download_checkpoint,
            patch.object(batched, "load_env") as load_env,
            patch.object(batched, "load_policy") as load_policy,
            patch.object(batched, "CheckpointEvaluator"),
            patch.object(batched, "run_mode", side_effect=rollout) as run_mode,
        ):
            df = batched.main([*paths, "--no-cached", "--output-csv", str(output)])
            self.assertIs(batched.sys.argv, original_argv)
        rows = df.to_dict("records")
        self.assertEqual(len(batched.pd.read_csv(output)), 8)
        self.assertEqual(load_policy.call_count, 2)
        self.assertEqual(load_env.return_value.close.call_count, 2)
        self.assertEqual(run_mode.call_count, 8)
        self.assertEqual(download_checkpoint.call_count, 2)
        for call in download_checkpoint.call_args_list:
            self.assertTrue(call.kwargs["no_cached"])
        self.assertEqual({row["run_path"] for row in rows}, set(paths))
        self.assertEqual({row["mode"] for row in rows}, {"sp_train", "sp_val", "hr_val", "hr_interactive"})
        interactive_call = run_mode.call_args
        self.assertIsNone(interactive_call.kwargs["controller_overrides"])
        self.assertEqual(interactive_call.kwargs["total_eval_agents"], batched.NUM_TOTAL_INTERACTIVE_EVAL_AGENTS)
        self.assertFalse({"sp_maps", "is_regularized", "anchor_maps"} & rows[0].keys())


if __name__ == "__main__":
    unittest.main()
