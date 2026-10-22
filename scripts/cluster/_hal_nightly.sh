#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${1:?GPU count required}"
shift
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

CONTAINER="${PUFFERDRIVE_IMAGE:-$REPO_ROOT/container/pufferdrive.sif}"
PROGRAM_CONFIG="${PROGRAM_CONFIG:-$REPO_ROOT/scripts/cluster_configs/nightly_navigation.yaml}"
MAP_DIR="${MAP_DIR:-$REPO_ROOT/pufferlib/resources/drive/binaries/carla}"
RUNS_DIR="${RUNS_DIR:-$REPO_ROOT/experiments}"
SEED="${SEED:-${SLURM_ARRAY_TASK_ID:-0}}"
DATE_STAMP="$(date +%Y-%m-%d)"
RUN_NAME="${RUN_NAME:-${DATE_STAMP}_hal_${SLURM_JOB_ID:-manual}_seed${SEED}}"

if [[ ! -f "$CONTAINER" ]]; then
    echo "error: container image not found: $CONTAINER (build it with container/build.sh)" >&2
    exit 1
fi
if [[ ! -f "$REPO_ROOT/.venv/bin/activate" ]]; then
    echo "error: missing repo .venv; create it with:" >&2
    echo "  singularity exec --nv --bind \"$REPO_ROOT:$REPO_ROOT\" --pwd \"$REPO_ROOT\" \\" >&2
    echo "    \"$CONTAINER\" /opt/pufferdrive/setup-venv.sh" >&2
    exit 1
fi
if [[ ! -d "$MAP_DIR" ]]; then
    echo "error: map directory not found: $MAP_DIR" >&2
    exit 1
fi

NAVI_DIR="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/pufferdrive-navigation.XXXXXX")"
trap 'rm -rf "$NAVI_DIR"' EXIT
mkdir -p "$RUNS_DIR"
RUN_DIR="$RUNS_DIR/$RUN_NAME"
mkdir -p "$RUN_DIR"

CONTAINER_ARGS=(--nv --bind "$REPO_ROOT:$REPO_ROOT" --pwd "$REPO_ROOT")
singularity run "${CONTAINER_ARGS[@]}" --bind "$NAVI_DIR:$NAVI_DIR" "$CONTAINER" \
    python data_utils/process_navigation.py \
    --input "$MAP_DIR" \
    --output-dir "$NAVI_DIR" \
    --workers "${NAVI_WORKERS:-4}"

CONFIG_OVERRIDES="$(
    singularity run "${CONTAINER_ARGS[@]}" "$CONTAINER" python -c '
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)
for key, value in config.items():
    if isinstance(value, bool):
        value = str(value).lower()
    elif value is None:
        value = "null"
    print(f"{key}={value}")
' "$PROGRAM_CONFIG"
)"
mapfile -t NIGHTLY_ARGS <<< "$CONFIG_OVERRIDES"

singularity run "${CONTAINER_ARGS[@]}" --bind "$NAVI_DIR:$NAVI_DIR" "$CONTAINER" \
    torchrun --standalone --nnodes=1 --nproc-per-node="$NUM_GPUS" \
    -m pufferlib.pufferl train puffer_drive \
    "${NIGHTLY_ARGS[@]}" \
    "$@" \
    "env.map_dir=$MAP_DIR" \
    "env.goal_source=route" \
    "env.route_source=navigation" \
    "env.navigation_dir=$NAVI_DIR" \
    "env.obs_navigation=true" \
    "train.seed=$SEED" \
    "train.data_dir=$RUN_DIR" \
    "run_name=$RUN_NAME" \
    "wandb_group=$DATE_STAMP"
