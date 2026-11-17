#!/bin/bash

CONTAINER="/staging/agp/containers/spicy_cuda12.2_261107.sif"
SPICY_DATASET_ROOT="/open_datasets/waymo/data/gpudrive/spicy_bins"
ZBIN_DATASET_ROOT="/open_datasets/waymo/data/gpudrive/bin"
OUTPUT_DIR="/staging/agp/projects/pufferdrive/experiments"

output_dir=$OUTPUT_DIR
mkdir -p "$output_dir"

singularity exec --nv \
  --bind "$PWD":/workspace \
  --bind "$ZBIN_DATASET_ROOT":/zbin \
  --bind "$SPICY_DATASET_ROOT":/spicy \
  --bind "$output_dir":/workspace/experiments \
  --bind /staging:/staging \
  --bind /workspaces:/workspaces \
  --pwd /workspace \
  "$CONTAINER" \
  bash -c "exec bash"
