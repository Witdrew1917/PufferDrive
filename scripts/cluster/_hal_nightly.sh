#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${1:?GPU count required}"
shift
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

CONTAINER="${PUFFERDRIVE_IMAGE:-/staging/agp/containers/pufferdrive_20261022.sif}"
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

CONTAINER_COMMAND=(
    singularity exec --nv --bind "$REPO_ROOT:$REPO_ROOT" --pwd "$REPO_ROOT"
    --bind "$NAVI_DIR:$NAVI_DIR" "$CONTAINER"
    /bin/bash -euc '
        source "$PWD/.venv/bin/activate"
        if [[ ! -x "$PWD/.venv/bin/python" ]]; then
            echo "error: repo .venv/bin/python is missing or broken inside the container; recreate the venv with /opt/pufferdrive/setup-venv.sh inside this image" >&2
            exit 1
        fi
        exec "$PWD/.venv/bin/python" "$@"
    ' pufferdrive-hal
)
"${CONTAINER_COMMAND[@]}" data_utils/process_navigation.py \
    --input "$MAP_DIR" \
    --output-dir "$NAVI_DIR" \
    --workers "${NAVI_WORKERS:-4}"

CONFIG_OVERRIDES="$(
    "${CONTAINER_COMMAND[@]}" -c '
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

"${CONTAINER_COMMAND[@]}" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node="$NUM_GPUS" \
    -m pufferlib.pufferl train puffer_drive \
    "${NIGHTLY_ARGS[@]}" \
    "$@" \
    "env.map_dir=$MAP_DIR" \
    "env.goal_source=route" \
    "env.navigation_dir=$NAVI_DIR" \
    "env.obs_navigation=true" \
    "train.seed=$SEED" \
    "train.data_dir=$RUN_DIR" \
    "run_name=$RUN_NAME" \
    "wandb_group=$DATE_STAMP"
