#!/usr/bin/env bash
#
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH --gres=gpu:1
#SBATCH --mem 100G
#SBATCH --time 15-00:00:00
#SBATCH --output /staging/agp/users/%u/logs/zpufferdrive/%j.out
#SBATCH --partition zprod
#SBATCH --job-name=pufferdrive-nightly
#

NUM_GPUS="${NUM_GPUS:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/_hal_nightly.sh" "$NUM_GPUS" "$@"
