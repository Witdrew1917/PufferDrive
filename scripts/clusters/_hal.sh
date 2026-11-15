#!/bin/bash

mode=${1:?"No mode specified (e.g. train)"}
dataset=${2:?"No dataset specified (zbin or spicy)"}

source "$(dirname "$0")/_git_state.sh" || exit 1
zpuffer_snapshot || exit 1
zpuffer_prepare_workdir "$ZPUFFER_COMMIT" "$ZPUFFER_DIFF_FILE" || exit 1

CONTAINER="/staging/agp/containers/spicy_cuda12.2_261107.sif"
SPICY_DATASET_ROOT="/open_datasets/waymo/data/gpudrive/spicy_bins"
ZBIN_DATASET_ROOT="/open_datasets/waymo/data/gpudrive/bin"
OUTPUT_DIR="/staging/agp/projects/pufferdrive/experiments"

output_dir=$OUTPUT_DIR
mkdir -p "$output_dir"
zpuffer_record_git_state "$output_dir" || exit 1

singularity exec --nv \
  --bind "$ZPUFFER_WORKDIR":/workspace \
  --bind "$DATASET_ROOT":/workspace/resources/drive/binaries \
  --bind "$ZBIN_DATASET_ROOT":/workspace/resources/drive/zbin \
  --bind "$SPICY_DATASET_ROOT":/workspace/resources/drive/spicy \
  --bind "$ZPUFFER_GIT_STATE_DIR":/git-state:ro \
  --bind "$output_dir":/workspace/experiments \
  --bind /staging:/staging \
  --bind /workspaces:/workspaces \
  --bind "$(realpath "$(git rev-parse --git-common-dir)")" \
  --env ZPUFFER_GIT_STATE_DIR=/git-state \
  --pwd /workspace \
  "$CONTAINER" \
  bash -c '
    /opt/pufferdrive-venv/bin/python setup.py build_ext --inplace --force &&
      exec /opt/pufferdrive-venv/bin/python -m pufferlib.pufferl "$1" puffer_drive --env.map-dir=/workspace/resources/drive/"$3"/training --eval.map-dir=/workspace/resources/drive/"$3"/validation "${@:4}" --wandb --wandb-group=pufferdrive --tag="$2"
  ' \
  _ "$mode" "slurm-${SLURM_JOB_ID:-local-$$}" "${@:2}"
