#!/bin/bash

CONTAINER="/staging/agp/containers/spicy_cuda12.2_261107.sif"
DATASET_ROOT="/open_datasets/waymo/data/gpudrive/spicy_bins"
OUTPUT_DIR="/staging/agp/projects/pufferdrive/experiments"

output_dir=$OUTPUT_DIR
mkdir -p "$output_dir"

singularity exec --nv \
  --bind "$PWD":/workspace \
  --bind "$DATASET_ROOT":/workspace/resources/drive/binaries \
  --bind "$output_dir":/workspace/experiments \
  --bind /staging:/staging \
  --bind /workspaces:/workspaces \
  --pwd /workspace \
  "$CONTAINER" \
  bash -c "exec bash"
