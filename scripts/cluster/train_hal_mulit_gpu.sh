#!/usr/bin/env bash
#
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 128
#SBATCH --gres=gpu:8
#SBATCH --mem 480G
#SBATCH --time 3-00:00:00
#SBATCH --output /staging/agp/users/%u/logs/zpufferdrive/%j.out
#SBATCH --partition zprod
#SBATCH --job-name=pufferdrive-nightly
#SBATCH --reservation=ctps2
#

NUM_GPUS="${NUM_GPUS:-8}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/_hal_nightly.sh" "$NUM_GPUS" "$@"
