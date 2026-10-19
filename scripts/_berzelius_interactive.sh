#!/bin/bash
mode=${1:?"No mode given"}

CONTAINER="/proj/ag_plan/containers/pufferdrive_090126.sif"
WAYMO_JSON_ROOT="/proj/ag_plan/projects/zpufferdrive/feature_cache/waymo_jsons"
WAYMO_TARGET_DIR="/proj/ag_plan/projects/pufferdrive/feature_cache/waymo_binaries"
BUILD_DIR="/proj/ag_plan/projects/pufferdrive/build"

output_dir="$WAYMO_TARGET_DIR/${mode}"
mkdir -p "$output_dir"

singularity exec --nv \
  --bind "$WAYMO_JSON_ROOT/${mode}":/pufferdrive/data/jsons \
  --bind "$BUILD_DIR":/build \
  --bind $output_dir:/pufferdrive/resources/drive/binaries/jsons \
  --bind /proj:/proj \
  --bind "$PWD":/pufferdrive \
  --env BUILD_DIR=/build/.docker_temp/build \
  --env UV_CACHE_DIR=/build/.docker_temp/.uv_cache \
  --env UV_PROJECT_ENVIRONMENT="/build/.venv_docker" \
  --pwd /pufferdrive \
  $CONTAINER \
  bash -c "exec bash"
