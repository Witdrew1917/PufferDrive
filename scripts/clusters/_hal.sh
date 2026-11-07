#!/bin/bash

mode=${1:?"No mode specified (e.g. train)"}

source "$(dirname "$0")/_git_state.sh" || exit 1
zpuffer_snapshot || exit 1
zpuffer_prepare_workdir "$ZPUFFER_COMMIT" "$ZPUFFER_DIFF_FILE" || exit 1

CONTAINER="/staging/agp/containers/spicy_cuda12.2_261107.sif"
DATASET_ROOT="/open_datasets/waymo/data/gpudrive/spicy_bins"

zpuffer_record_git_state "$PWD/experiments" || exit 1

singularity exec --nv \
  --bind "$ZPUFFER_WORKDIR":/workspace \
  --bind "$DATASET_ROOT":/workspace/resources/drive/binaries \
  --bind "$ZPUFFER_GIT_STATE_DIR":/git-state:ro \
  --bind /staging:/staging \
  --bind /workspaces:/workspaces \
  --bind "$(realpath "$(git rev-parse --git-common-dir)")" \
  --env ZPUFFER_GIT_STATE_DIR=/git-state \
  --pwd /workspace \
  "$CONTAINER" \
  bash -c '/opt/pufferdrive-venv/bin/python setup.py build_ext --inplace --force && exec /opt/pufferdrive-venv/bin/python -m pufferlib.pufferl "$1" puffer_drive "${@:2}" --wandb --wandb-group=pufferdrive' \
  _ "$mode" "${@:2}"
