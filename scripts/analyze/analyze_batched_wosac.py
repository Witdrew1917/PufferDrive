r"""Evaluate W&B model artifacts with WOSAC, without checkpoint filename conventions.

Usage:
    python scripts/analyze/analyze_batched_wosac.py scripts/analyze/batch_descriptions/goal_dropout.txt
    python scripts/analyze/analyze_batched_wosac.py ENTITY/PROJECT/RUN_ID [ENTITY/PROJECT/RUN_ID ...]
    python scripts/analyze/analyze_batched_wosac.py ENTITY/PROJECT/RUN_ID \
        --dynamics ENTITY/PROJECT/RUN_ID delta_local --output-csv results/batched_wosac.csv

The batch file contains one run per line, followed by optional config overrides,
such as --env.goal-obs-dropout-prob=0.0. Blank lines and # comments are ignored.
Overrides use config section and key names and take precedence over evaluation defaults.
By default, batch results are saved to results/<batch-file-stem>_wosac.csv.
Each run resolves to its model artifact at RUN_PATH:v0. Dynamics come from the
checkpoint's full_args; --dynamics supplies them only when absent. In that case,
full_args must explicitly set env.navigation to 0 or 1.
Verified local artifacts are reused unless --no-cached is supplied.
"""

import argparse
import ast
import copy
import shlex
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch

import wandb
from pufferlib.pufferl import load_config, load_env, load_policy


ENV_NAME = "puffer_drive"
WOSAC_MAP_DIR = "resources/drive/binaries/validation"
WOSAC_BATCH_SIZE = 256
WOSAC_SCENARIO_POOL_SIZE = 10000
WOSAC_MAX_BATCHES = 100
WOSAC_NUM_ROLLOUTS = 32
WOSAC_INIT_STEPS = 10
WOSAC_CONTROL_MODE = "control_wosac"
WOSAC_INIT_MODE = "create_all_valid"
WOSAC_GOAL_BEHAVIOR = 2
WOSAC_GOAL_RADIUS = 2.5
DYNAMICS_MODELS = ("delta_local", "classic", "jerk")
OUTPUT_CSV = "results/checkpoint_wosac_batched.csv"
ARTIFACTS_DIR = Path("artifacts")
WOSAC_METRICS = [
    "realism_meta_score",
    "kinematic_metrics",
    "interactive_metrics",
    "map_based_metrics",
    "num_agents_per_scene",
    "likelihood_linear_speed",
    "likelihood_linear_acceleration",
    "likelihood_angular_speed",
    "likelihood_angular_acceleration",
    "likelihood_collision_indication",
    "likelihood_distance_to_nearest_object",
    "likelihood_time_to_collision",
    "likelihood_distance_to_road_edge",
    "likelihood_offroad_indication",
]


def load_checkpoint_config(checkpoint_path, fallback_config):
    """Load full_args from the checkpoint without inferring settings from its name."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "full_args" in checkpoint:
        print("  Using env config stored in checkpoint.")
        return copy.deepcopy(checkpoint["full_args"]), True
    print("  No config found in checkpoint, falling back to ini config.")
    return copy.deepcopy(fallback_config), False


def apply_overrides(config, overrides, reference_config):
    for key, value in overrides.items():
        if key in ("load_model_path", "load_id"):
            raise ValueError(f"Config override --{key.replace('_', '-')} is managed by the evaluator")
        parts = key.split(".")
        target = config
        reference = reference_config
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                raise ValueError(f"Config section {part!r} is missing or not a mapping")
            target = target[part]
            reference = reference.get(part, {}) if isinstance(reference, dict) else {}
        if parts[-1] not in target and (not isinstance(reference, dict) or parts[-1] not in reference):
            raise ValueError(f"Unknown config override --{key.replace('_', '-')}")
        target[parts[-1]] = value


def make_wosac_config(cpt_config, overrides=None):
    config = copy.deepcopy(cpt_config)
    config["env"].update(
        map_dir=WOSAC_MAP_DIR,
        num_maps=WOSAC_SCENARIO_POOL_SIZE,
        num_agents=WOSAC_BATCH_SIZE * 10,
        termination_mode=0,
        async_resets=False,
        init_mode=WOSAC_INIT_MODE,
        control_mode=WOSAC_CONTROL_MODE,
        init_steps=WOSAC_INIT_STEPS,
        goal_behavior=WOSAC_GOAL_BEHAVIOR,
        goal_radius=WOSAC_GOAL_RADIUS,
        episode_length=91,
    )
    config["vec"] = {"backend": "PufferEnv", "num_envs": 1}
    config.setdefault("eval", {}).update(
        wosac_realism_eval=True,
        wosac_batch_size=WOSAC_BATCH_SIZE,
        wosac_scenario_pool_size=WOSAC_SCENARIO_POOL_SIZE,
        wosac_max_batches=WOSAC_MAX_BATCHES,
        wosac_num_rollouts=WOSAC_NUM_ROLLOUTS,
        wosac_init_steps=WOSAC_INIT_STEPS,
        wosac_control_mode=WOSAC_CONTROL_MODE,
        wosac_init_mode=WOSAC_INIT_MODE,
        wosac_goal_behavior=WOSAC_GOAL_BEHAVIOR,
        wosac_goal_radius=WOSAC_GOAL_RADIUS,
        wosac_target_scenarios=1000,
        wosac_aggregate_results=False,
        wosac_filter_out_post_done=False,
        wosac_sanity_check=False,
        wosac_eval_mode="policy",
        backend="PufferEnv",
        map_dir=WOSAC_MAP_DIR,
    )
    if overrides:
        apply_overrides(config, overrides, cpt_config)
    return config


def run_path(value):
    if len(value.split("/")) != 3 or not all(value.split("/")) or ":" in value:
        raise argparse.ArgumentTypeError("Expected a W&B run path: ENTITY/PROJECT/RUN_ID")
    return value


def load_batch_descriptions(filename):
    entries = []
    with filename.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            try:
                tokens = shlex.split(line, comments=True)
                if not tokens:
                    continue
                path = run_path(tokens[0])
                overrides = {}
                index = 1
                while index < len(tokens):
                    token = tokens[index]
                    if not token.startswith("--"):
                        raise ValueError(f"Expected a config override, got {token!r}")
                    key, separator, value = token[2:].partition("=")
                    if not separator:
                        index += 1
                        if index >= len(tokens) or tokens[index].startswith("--"):
                            raise ValueError(f"Missing value for --{key}")
                        value = tokens[index]
                    key = key.replace("-", "_")
                    if not key or any(not part for part in key.split(".")):
                        raise ValueError(f"Invalid config override {token!r}")
                    if key in overrides:
                        raise ValueError(f"Duplicate config override --{key.replace('_', '-')}")
                    try:
                        value = ast.literal_eval(value)
                    except (ValueError, SyntaxError):
                        pass
                    overrides[key] = value
                    index += 1
                entries.append((path, overrides))
            except (ValueError, argparse.ArgumentTypeError) as error:
                raise ValueError(f"{filename}:{line_number}: {error}") from error
    if not entries:
        raise ValueError(f"{filename}: no runs found")
    return entries


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="W&B run paths or a batch description .txt file")
    parser.add_argument(
        "--dynamics",
        nargs=2,
        action="append",
        default=[],
        metavar=("RUN_PATH", "MODEL"),
        help=f"Pair a run missing checkpoint dynamics with one of: {', '.join(DYNAMICS_MODELS)}; repeat as needed",
    )
    parser.add_argument("--output-csv", type=Path, help="Override the default results CSV path")
    parser.add_argument(
        "--no-cached", action="store_true", help="Force a fresh download, bypassing local and W&B caches"
    )
    args = parser.parse_args(argv)
    if len(args.runs) == 1 and args.runs[0].endswith(".txt"):
        batch_file = Path(args.runs[0])
        try:
            args.entries = load_batch_descriptions(batch_file)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        args.runs = [path for path, _ in args.entries]
        if args.output_csv is None:
            args.output_csv = Path(OUTPUT_CSV).parent / f"{batch_file.stem}_wosac.csv"
    else:
        try:
            args.runs = [run_path(path) for path in args.runs]
        except argparse.ArgumentTypeError as error:
            parser.error(str(error))
        args.entries = [(path, {}) for path in args.runs]
        if args.output_csv is None:
            args.output_csv = Path(OUTPUT_CSV)
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


def prepare_checkpoint_config(checkpoint_path, base_config, path, dynamics=None, overrides=None):
    config, stored = load_checkpoint_config(checkpoint_path, base_config)
    if overrides:
        apply_overrides(config, overrides, base_config)
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


def evaluate_runs(api, entries, base_config, dynamics, no_cached=False):
    from pufferlib.ocean.benchmark.evaluator import WOSACEvaluator

    all_rows = []
    for path, overrides in entries:
        print(f"\nEvaluating {path}")
        checkpoint_path = download_checkpoint(api, path, no_cached=no_cached)
        config = prepare_checkpoint_config(checkpoint_path, base_config, path, dynamics.get(path), overrides)
        wosac_config = make_wosac_config(config, overrides)
        env = load_env(ENV_NAME, wosac_config)
        try:
            policy = load_policy(wosac_config, env, ENV_NAME)
            policy.eval()
            evaluator = WOSACEvaluator(wosac_config)
            results = evaluator.evaluate(wosac_config, env, policy)
        finally:
            env.close()

        for scenario_id, row in results.iterrows():
            entry = {
                "run_path": path,
                "checkpoint": str(checkpoint_path),
                "scenario_id": scenario_id,
                "dynamics": config["env"]["dynamics_model"],
            }
            entry.update({metric: float(row.get(metric, float("nan"))) for metric in WOSAC_METRICS})
            all_rows.append(entry)
        if not results.empty:
            print(f"  WOSAC: {len(results)} scenarios, realism_meta_score={results['realism_meta_score'].mean():.4f}")
        else:
            print("  WOSAC: no scenario results returned.")
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

    rows = evaluate_runs(wandb.Api(), args.entries, base_config, args.dynamics, no_cached=args.no_cached)
    df = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"\nResults saved to {args.output_csv} ({len(df)} rows)")
    if not df.empty:
        summary = df.groupby("run_path")["realism_meta_score"].agg(["mean", "sem", "count"])
        summary = summary.rename(columns={"mean": "realism_mean", "sem": "realism_sem", "count": "n_scenarios"})
        print(f"\n{summary}")
    return df


if __name__ == "__main__":
    main()
