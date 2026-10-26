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
#SBATCH --job-name=pufferdrive
#SBATCH --reservation=ctps2
#

NUM_GPUS="${NUM_GPUS:-8}"
exec scripts/cluster/_hal.sh python -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node="$NUM_GPUS" \
    -m pufferlib.pufferl train puffer_drive "$@"
