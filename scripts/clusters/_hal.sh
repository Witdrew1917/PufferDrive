#!/bin/bash

mode=${1:?"No mode specified (e.g. train)"}

source "$(dirname "$0")/_git_state.sh" || exit 1
zpuffer_snapshot || exit 1
zpuffer_prepare_workdir "$ZPUFFER_COMMIT" "$ZPUFFER_DIFF_FILE" || exit 1

CONTAINER="/staging/agp/containers/spicy_cuda12.2_261105.sif"
DATASET_ROOT="/open_datasets/waymo/data/gpudrive/spicy_bins"
ZPUFFER_VENV_PATH="$PWD/.venv"

zpuffer_record_git_state "$PWD/experiments" || exit 1

singularity exec --nv \
  --bind "$ZPUFFER_WORKDIR":/workspace \
  --bind "$ZPUFFER_VENV_PATH":/workspace/.venv \
  --bind "$DATASET_ROOT":/workspace/resources/drive/binaries \
  --bind "$ZPUFFER_GIT_STATE_DIR":/git-state:ro \
  --bind /staging:/staging \
  --bind "$(realpath "$(git rev-parse --git-common-dir)")" \
  --env ZPUFFER_GIT_STATE_DIR=/git-state \
  --pwd /workspace \
  "$CONTAINER" \
  bash -c 'source ".venv/bin/activate" && python setup.py build_ext --inplace --force && exec python -m pufferlib.pufferl "$2" puffer_drive "${@:3}" --wandb --wandb-group=pufferdrive' \
  _ "$ZPUFFER_VENV_PATH" "$mode" "${@:2}"
