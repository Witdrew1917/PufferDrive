#!/bin/bash
#
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH --gres=gpu:1
#SBATCH --mem 100G
#SBATCH --time 3-00:00:00
#SBATCH --output /workspaces/%u/logs/pufferdrive/%j.out
#SBATCH --partition zprodlow
#SBATCH --job-name=zpufferdrive
#SBATCH --reservation=ctps2
#

source scripts/clusters/_git_state.sh || exit 1
zpuffer_submit_with_snapshot "$0" "$@"
zpuffer_snapshot || exit 1

NUM_GPUS=${NUM_GPUS:-1}

scripts/clusters/_hal.sh train $@

#
#EOF
