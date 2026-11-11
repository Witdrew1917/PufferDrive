r"""Evaluate W&B model artifacts without checkpoint filename conventions.

Usage:
    python examples/analyze_batched.py ENTITY/PROJECT/RUN_ID [ENTITY/PROJECT/RUN_ID ...]
    python examples/analyze_batched.py ENTITY/PROJECT/RUN_ID \
        --dynamics ENTITY/PROJECT/RUN_ID delta_local --output-csv results/batched.csv

Each run resolves to its model artifact at RUN_PATH:v0. Dynamics come from the
checkpoint's full_args; --dynamics supplies them only when absent. In that case,
full_args must explicitly set env.navigation to 0 or 1.
Verified local artifacts are reused unless --no-cached is supplied.
"""

import argparse
import copy
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import wandb
from pufferlib.ocean.benchmark.evaluator_minimal import CheckpointEvaluator
from pufferlib.pufferl import load_config, load_env, load_policy


ENV_NAME = "puffer_drive"
DETERMINISTIC = False
TRAIN_MAP_DIR = "resources/drive/binaries/training"
VAL_MAP_DIR = "resources/drive/binaries/validation"
INTERACTIVE_MAP_DIR = "resources/drive/binaries/wosac_val_clean"
INTERACTIVE_MAP_DIR_MAPS = 200
NUM_TOTAL_EVAL_AGENTS = 1024 * 10
NUM_TOTAL_INTERACTIVE_EVAL_AGENTS = 2048
NUM_AGENTS_PER_VECENV = 1024
DATASET = "womd"
DYNAMICS_MODELS = ("delta_local", "classic", "jerk")
OUTPUT_CSV = "results/checkpoint_eval_batched.csv"
ARTIFACTS_DIR = Path("artifacts")
METRICS = [
    "n",
    "score",
    "collision_rate",
    "at_fault_collision_rate",
    "rear_collision_rate",
    "delta_v_sum",
    "delta_v_max",
    "delta_v_count",
    "delta_v_under_1mph",
    "collisions_per_agent",
    "offroad_rate",
    "offroad_per_agent",
    "completion_rate",
    "route_progress",
    "episode_length",
    "episode_return",
    "perc_controlled",
    "lateral_error_avg",
    "longitudinal_error_avg",
    "displacement_error_avg",
]


def load_checkpoint_config(checkpoint_path, fallback_config):
    """Load full_args from the checkpoint without inferring settings from its name."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "full_args" in checkpoint:
        print("  Using env config stored in checkpoint.")
        return copy.deepcopy(checkpoint["full_args"]), True
    print("  No config found in checkpoint, falling back to ini config.")
    return copy.deepcopy(fallback_config), False


def make_eval_config(cpt_config, map_dir, control_mode, num_maps, episode_length=200, controller_overrides=None):
    config = copy.deepcopy(cpt_config)
    config["env"].update(
        map_dir=map_dir,
        control_mode=control_mode,
        num_maps=num_maps,
        num_agents=NUM_AGENTS_PER_VECENV,
        async_resets=False,
        goal_behavior=0,
        render_mode=1,
        termination_mode=1,
        fix_lambdas=True,
        fix_rewards=True,
        obs_partner_noise_speed=0.0,
        obs_partner_noise_pos=0.0,
    )
    if episode_length is not None:
        config["env"]["episode_length"] = episode_length
    if controller_overrides is not None:
        config["env"].update(controller_overrides)
    config["vec"] = {"backend": "PufferEnv", "num_envs": 1}
    return config


def process_rollout_data(info_list, checkpoint, mode, scene_offset=0, dataset=DATASET):
    populated = [log for log in info_list if log and log.get("n", 0) > 0]
    rows = []
    for i, log in enumerate(populated):
        row = {
            "checkpoint": checkpoint,
            "dataset": dataset,
            "mode": mode,
            "scene_idx": scene_offset + i,
        }
        row.update({key: float(log.get(key, 0.0)) for key in METRICS})
        rows.append(row)
    return rows


def run_mode(
    evaluator,
    policy,
    cpt_config,
    map_dir,
    control_mode,
    checkpoint,
    mode_name,
    num_maps,
    controller_overrides=None,
    total_eval_agents=NUM_TOTAL_EVAL_AGENTS,
):
    config = make_eval_config(cpt_config, map_dir, control_mode, num_maps, controller_overrides=controller_overrides)
    env = load_env(ENV_NAME, config)
    rows = []
    n_rounds = max(1, (total_eval_agents + NUM_AGENTS_PER_VECENV - 1) // NUM_AGENTS_PER_VECENV)
    try:
        for round_idx in range(n_rounds):
            if round_idx > 0:
                env.driver_env.resample_maps()
            rollout_stats = evaluator.rollout(env=env, policy=policy, deterministic=DETERMINISTIC)
            scene_offset = round_idx * env.driver_env.num_envs
            rows.extend(process_rollout_data(rollout_stats, checkpoint, mode_name, scene_offset))
    finally:
        env.close()

    if rows:
        mean_score = np.mean([r["score"] for r in rows])
        mean_coll = np.mean([r["collision_rate"] for r in rows])
        mean_coll_fault = np.mean([r["at_fault_collision_rate"] for r in rows])
        print(
            f"  {mode_name}: {len(rows)} scenes, score={mean_score:.3f}, "
            f"collision_rate={mean_coll:.3f}, at_fault_collision_rate={mean_coll_fault:.3f}"
        )
    else:
        print(f"  {mode_name}: no populated scenes")
    return rows


def delta_v_summary(group):
    """Severity stats conditional on collision."""
    coll = group[group["delta_v_count"] > 0]
    if coll.empty:
        return pd.Series({"n_collisions": 0, "mean_dv_per_event": np.nan, "max_dv": np.nan, "frac_under_1mph": np.nan})
    total_count = coll["delta_v_count"].sum()
    return pd.Series(
        {
            "n_collisions": int(total_count),
            "mean_dv_per_event": coll["delta_v_sum"].sum() / total_count,
            "max_dv": coll["delta_v_max"].max(),
            "frac_under_1mph": coll["delta_v_under_1mph"].mean(),
        }
    )


def run_path(value):
    if len(value.split("/")) != 3 or not all(value.split("/")) or ":" in value:
        raise argparse.ArgumentTypeError("Expected a W&B run path: ENTITY/PROJECT/RUN_ID")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=run_path, help="W&B run paths (model artifact version v0)")
    parser.add_argument(
        "--dynamics",
        nargs=2,
        action="append",
        default=[],
        metavar=("RUN_PATH", "MODEL"),
        help=f"Pair a run missing checkpoint dynamics with one of: {', '.join(DYNAMICS_MODELS)}; repeat as needed",
    )
    parser.add_argument("--output-csv", type=Path, default=Path(OUTPUT_CSV))
    parser.add_argument(
        "--no-cached", action="store_true", help="Force a fresh download, bypassing local and W&B caches"
    )
    args = parser.parse_args(argv)
    dynamics = {}
    for path, model in args.dynamics:
        if path not in args.runs:
            parser.error(f"--dynamics run {path!r} is not among the requested runs")
        if model not in DYNAMICS_MODELS:
            parser.error(f"Unknown dynamics model {model!r}; choose from {', '.join(DYNAMICS_MODELS)}")
        if path in dynamics:
            parser.error(f"Multiple --dynamics pairings for {path}")
        dynamics[path] = model
    args.dynamics = dynamics
    return args


def find_checkpoint(directory, path):
    checkpoints = sorted(p for p in directory.rglob("*.pt") if p.name != "trainer_state.pt")
    if len(checkpoints) != 1:
        raise ValueError(f"{path}:v0: expected one model checkpoint, found {len(checkpoints)} in {directory}")
    return checkpoints[0]


def download_checkpoint(api, path, no_cached=False):
    artifact = api.artifact(f"{path}:v0", type="model")
    directory = ARTIFACTS_DIR / f"{path}:v0"
    if not no_cached:
        # Also recognize downloads made with W&B's default, unqualified root.
        for cached in (directory, ARTIFACTS_DIR / f"{path.split('/')[-1]}:v0"):
            if not cached.is_dir():
                continue
            try:
                artifact.verify(root=str(cached))
            except (ValueError, FileNotFoundError) as error:
                print(f"  Ignoring invalid cached artifact at {cached}: {error}")
                continue
            print(f"  Using cached artifact at {cached}")
            return find_checkpoint(cached, path)

    if no_cached:
        directory.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="artifact-download-", dir=directory.parent) as temporary:
            downloaded = Path(artifact.download(root=temporary, skip_cache=True))
            find_checkpoint(downloaded, path)
            shutil.copytree(downloaded, directory, dirs_exist_ok=True)
    else:
        directory = Path(artifact.download(root=str(directory)))
    return find_checkpoint(directory, path)


def prepare_checkpoint_config(checkpoint_path, base_config, path, dynamics=None):
    config, stored = load_checkpoint_config(checkpoint_path, base_config)
    checkpoint_env = config.get("env", {}) if stored else {}
    configured_dynamics = checkpoint_env.get("dynamics_model")
    if not configured_dynamics:
        navigation = checkpoint_env.get("navigation")
        if navigation not in (0, 1):
            raise ValueError(f"{path}: checkpoint missing dynamics_model must explicitly set env.navigation to 0 or 1")
        if dynamics is None:
            raise ValueError(f"{path}: checkpoint missing dynamics_model; supply --dynamics {path} MODEL")
        configured_dynamics = dynamics
    elif dynamics is not None and dynamics != configured_dynamics:
        raise ValueError(f"{path}: --dynamics {dynamics!r} conflicts with checkpoint dynamics {configured_dynamics!r}")

    if configured_dynamics not in DYNAMICS_MODELS:
        raise ValueError(f"{path}: unsupported checkpoint dynamics_model {configured_dynamics!r}")
    config["env"]["dynamics_model"] = configured_dynamics
    config["load_model_path"] = str(checkpoint_path)
    config["load_id"] = None
    return config


def evaluate_runs(api, paths, base_config, dynamics, no_cached=False):
    all_rows = []
    for path in paths:
        print(f"\nEvaluating {path}")
        checkpoint_path = download_checkpoint(api, path, no_cached=no_cached)
        config = prepare_checkpoint_config(checkpoint_path, base_config, path, dynamics.get(path))

        policy_config = make_eval_config(config, TRAIN_MAP_DIR, control_mode="control_vehicles", num_maps=50_000)
        env = load_env(ENV_NAME, policy_config)
        try:
            policy = load_policy(config, env, ENV_NAME)
            policy.eval()
        finally:
            env.close()

        evaluator = CheckpointEvaluator(config)
        modes = [
            ("sp_train", TRAIN_MAP_DIR, "control_vehicles", 50_000, None, NUM_TOTAL_EVAL_AGENTS),
            ("sp_val", VAL_MAP_DIR, "control_vehicles", 10_000, None, NUM_TOTAL_EVAL_AGENTS),
            ("hr_val", VAL_MAP_DIR, "control_sdc_only", 10_000, None, NUM_TOTAL_EVAL_AGENTS),
            (
                "hr_interactive",
                INTERACTIVE_MAP_DIR,
                "control_sdc_only",
                INTERACTIVE_MAP_DIR_MAPS,
                None,
                NUM_TOTAL_INTERACTIVE_EVAL_AGENTS,
            ),
        ]
        for name, map_dir, control_mode, num_maps, controllers, total_agents in modes:
            rows = run_mode(
                evaluator,
                policy,
                config,
                map_dir,
                control_mode,
                str(checkpoint_path),
                name,
                num_maps,
                controller_overrides=controllers,
                total_eval_agents=total_agents,
            )
            for row in rows:
                row["run_path"] = path
                row["dynamics"] = config["env"]["dynamics_model"]
            all_rows.extend(rows)
    return all_rows


def main(argv=None):
    args = parse_args(argv)
    # load_config parses sys.argv itself; keep this script's CLI out of it.
    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0]]
        base_config = load_config(ENV_NAME)
    finally:
        sys.argv = original_argv

    rows = evaluate_runs(wandb.Api(), args.runs, base_config, args.dynamics, no_cached=args.no_cached)
    df = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"\nResults saved to {args.output_csv} ({len(df)} rows)")
    if not df.empty:
        summary = df.groupby(["run_path", "mode"]).agg(
            scenes=("score", "count"),
            score=("score", "mean"),
            collision_rate=("collision_rate", "mean"),
            at_fault_collision_rate=("at_fault_collision_rate", "mean"),
            offroad_rate=("offroad_rate", "mean"),
        )
        print(f"\n{summary}")
        dv_columns = ["delta_v_count", "delta_v_sum", "delta_v_max", "delta_v_under_1mph"]
        print(df.groupby(["run_path", "mode"])[dv_columns].apply(delta_v_summary))
    return df


if __name__ == "__main__":
    main()
