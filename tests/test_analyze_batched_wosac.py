import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import torch

from scripts.analyze import analyze_batched_wosac as batched


RUN = "entity/project/run"


class TestAnalyzeBatchedWosac(unittest.TestCase):
    def test_batch_input_and_options(self):
        with tempfile.TemporaryDirectory() as directory:
            batch = Path(directory) / "batch.txt"
            batch.write_text(
                f"# comment\n{RUN} --env.goal-obs-dropout-prob=0.0\nentity/project/other --env.navigation 1\n",
                encoding="utf-8",
            )
            args = batched.parse_args([str(batch), "--dynamics", RUN, "classic", "--no-cached"])
            self.assertEqual(
                args.entries,
                [
                    (RUN, {"env.goal_obs_dropout_prob": 0.0}),
                    ("entity/project/other", {"env.navigation": 1}),
                ],
            )
            self.assertEqual(args.dynamics, {RUN: "classic"})
            self.assertTrue(args.no_cached)
            self.assertEqual(args.output_csv, Path("results/batch_wosac.csv"))
            self.assertEqual(batched.parse_args([RUN, "--output-csv", "out.csv"]).output_csv, Path("out.csv"))

    def test_evaluates_multiple_runs_and_writes_scenario_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            base = {"env": {"dynamics_model": "classic", "navigation": 0, "goal_obs_dropout_prob": 0.8}}
            torch.save({"full_args": copy.deepcopy(base)}, checkpoint)
            entries = [
                (RUN, {"env.goal_obs_dropout_prob": 0.0}),
                ("entity/project/other", {}),
            ]
            results = pd.DataFrame(
                {"realism_meta_score": [0.25, 0.75], "likelihood_linear_speed": [0.2, 0.8]},
                index=["scene-a", "scene-b"],
            )
            output = Path(directory) / "results.csv"
            with (
                patch.object(batched, "parse_args") as parse_args,
                patch.object(batched, "load_config", return_value=base),
                patch.object(batched.wandb, "Api"),
                patch.object(batched, "download_checkpoint", return_value=checkpoint) as download,
                patch.object(batched, "load_env") as load_env,
                patch.object(batched, "load_policy") as load_policy,
                patch("pufferlib.ocean.benchmark.evaluator.WOSACEvaluator") as evaluator,
            ):
                parse_args.return_value = Mock(entries=entries, dynamics={}, no_cached=True, output_csv=output)
                evaluator.return_value.evaluate.return_value = results
                df = batched.main()

            self.assertEqual(len(pd.read_csv(output)), 4)
            self.assertEqual(df["scenario_id"].tolist(), ["scene-a", "scene-b"] * 2)
            self.assertEqual(df["run_path"].tolist(), [RUN, RUN, "entity/project/other", "entity/project/other"])
            self.assertTrue(df["kinematic_metrics"].isna().all())
            self.assertEqual(load_env.return_value.close.call_count, 2)
            self.assertEqual(download.call_count, 2)
            self.assertTrue(all(call.kwargs["no_cached"] for call in download.call_args_list))
            configs = [call.args[1] for call in load_env.call_args_list]
            self.assertEqual([config["env"]["goal_obs_dropout_prob"] for config in configs], [0.0, 0.8])
            self.assertEqual(configs[0]["env"]["control_mode"], "control_wosac")
            self.assertEqual(configs[0]["eval"]["wosac_num_rollouts"], 32)
            self.assertIs(load_policy.call_args_list[0].args[0], configs[0])
            self.assertIs(evaluator.call_args_list[0].args[0], configs[0])
            self.assertEqual(base["env"]["goal_obs_dropout_prob"], 0.8)

    def test_evaluation_failure_closes_environment(self):
        config = {"env": {"dynamics_model": "classic", "navigation": 0}}
        with (
            patch.object(batched, "download_checkpoint"),
            patch.object(batched, "prepare_checkpoint_config", return_value=config),
            patch.object(batched, "load_env") as load_env,
            patch.object(batched, "load_policy"),
            patch("pufferlib.ocean.benchmark.evaluator.WOSACEvaluator") as evaluator,
        ):
            evaluator.return_value.evaluate.side_effect = RuntimeError("evaluation failed")
            with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                batched.evaluate_runs(Mock(), [(RUN, {})], config, {})
            load_env.return_value.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
