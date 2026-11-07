import json

from pufferlib import pufferl


def test_eval_runs_human_replay_and_returns_aggregate_metrics(monkeypatch, capsys):
    class FakeEnv:
        def __init__(self):
            self.driver_env = self
            self.recorder_stopped = False
            self.closed = False

        def stop_recorder(self, env_idx):
            assert env_idx == 0
            self.recorder_stopped = True

        def close(self):
            self.closed = True

    class FakeEvaluator:
        def __init__(self, configs):
            assert configs["eval"]["render_human_replay_eval"] is False
            self.hr_eval_config = {"env": {"control_mode": "control_sdc_only"}}
            self.hr_env = None
            self.rollout_mode = None

        def rollout(self, policy, mode):
            assert policy is policy_marker
            self.rollout_mode = mode

        def collect_stats(self):
            return {"eval/hr_score": 0.75}

    env = FakeEnv()
    policy_marker = object()
    evaluator_holder = {}

    def create_evaluator(configs):
        evaluator = FakeEvaluator(configs)
        evaluator_holder["evaluator"] = evaluator
        return evaluator

    monkeypatch.setattr(
        "pufferlib.ocean.benchmark.evaluator.Evaluator",
        create_evaluator,
    )
    monkeypatch.setattr(pufferl, "load_env", lambda name, config: env)

    args = {
        "env": {},
        "eval": {
            "wosac_realism_eval": False,
            "human_replay_eval": True,
            "render_human_replay_eval": False,
        },
    }

    results = pufferl.eval("puffer_drive", args=args, policy=policy_marker)

    assert results == {"eval/hr_score": 0.75}
    assert evaluator_holder["evaluator"].rollout_mode == "human_replay"
    assert env.recorder_stopped
    assert env.closed
    output = capsys.readouterr().out
    assert json.dumps(results) in output


def test_hr_mask_goal_overrides_environment_dropout():
    from pufferlib.ocean.benchmark.evaluator import Evaluator

    configs = {
        "env": {"lambda_value": 0.0, "goal_obs_dropout_prob": 0.6},
        "eval": {
            "map_dir": "validation",
            "num_eval_agents": 1,
            "render_select_mode": "first",
            "render_self_play_eval": False,
            "render_human_replay_eval": False,
            "hr_mask_goal": True,
        },
    }
    evaluator = Evaluator(configs)

    assert evaluator.hr_eval_config["env"]["goal_obs_dropout_prob"] == 1.0
    assert evaluator.sp_eval_config["env"]["goal_obs_dropout_prob"] == 0.6

    configs["eval"]["hr_mask_goal"] = False
    evaluator = Evaluator(configs)
    assert evaluator.hr_eval_config["env"]["goal_obs_dropout_prob"] == 0.0
