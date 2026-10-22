#!/usr/bin/env bash
# Render a policy rollout with the lane-navigation overlay (navigation = 1).
# Usage: scripts/render_navigation.sh [MAP_DIR] [MODEL_PATH|latest|none] [RENDER_MODE 0=window|1=mp4]
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

MAP_DIR="${1:-resources/drive/binaries/interactive_1k}"
MODEL_PATH="${2:-none}"
RENDER_MODE="${3:-0}"
NUM_MAPS="$(find "$MAP_DIR" -maxdepth 1 -name 'map_*.bin' | wc -l)"

python -m pufferlib.ocean.drive.drive navigation --map-dir "$MAP_DIR"

MODEL_ARGS=()
if [ "$MODEL_PATH" != "none" ]; then
    MODEL_ARGS=(--load-model-path "$MODEL_PATH")
fi

puffer eval puffer_drive \
    --env.navigation 1 \
    --env.map-dir "$MAP_DIR" \
    --env.num-maps "$NUM_MAPS" \
    --env.num-agents 64 \
    --env.render-mode "$RENDER_MODE" \
    --vec.backend PufferEnv \
    --train.device cpu \
    "${MODEL_ARGS[@]}"
