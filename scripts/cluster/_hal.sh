#!/bin/bash
# Run a command inside the PufferDrive container on HAL, from the repo root.
#
#   scripts/cluster/_hal.sh                                   # interactive shell
#   scripts/cluster/_hal.sh python data_utils/process_navigation.py --input pufferlib/resources/drive/binaries/carla --output-dir navi/carla --workers 16
#   scripts/cluster/_hal.sh puffer train puffer_drive env.goal_source=map env.obs_navigation=true env.navigation_dir=navi/carla
#   scripts/cluster/_hal.sh bash -c "python setup.py build_ext --inplace --force && puffer train puffer_drive"

CONTAINER="/staging/agp/containers/pufferdrive_20261022.sif"
OUTPUT_DIR="/staging/agp/projects/pufferdrive/experiments"
CARLA_MAPS_DIR="$PWD/pufferlib/resources/drive/binaries/carla"
CARLA_NAVI_DIR="/staging/agp/projects/pufferdrive/binaries/carla_navi"

mkdir -p "$OUTPUT_DIR" "$CARLA_NAVI_DIR" "$PWD/navi/carla"

if [ ! -f "$PWD/.venv/bin/activate" ]; then
  echo "error: no .venv in $PWD; create it inside the container with:" >&2
  echo "  singularity exec --nv --bind \"\$PWD\":/Pufferdrive --pwd /Pufferdrive $CONTAINER /opt/pufferdrive/setup-venv.sh" >&2
  exit 1
fi

if [ $# -eq 0 ]; then
  set -- bash
fi

singularity exec --nv \
  --bind "$PWD":/Pufferdrive \
  --bind "$CARLA_MAPS_DIR":/Pufferdrive/pufferlib/resources/drive/binaries/carla \
  --bind "$CARLA_NAVI_DIR":/Pufferdrive/navi/carla \
  --bind "$OUTPUT_DIR":/Pufferdrive/experiments \
  --bind /staging:/staging \
  --bind "$(realpath "$(git rev-parse --git-common-dir)")" \
  --pwd /Pufferdrive \
  "$CONTAINER" \
  bash -c 'source .venv/bin/activate && exec "$@"' _hal "$@"
