#!/usr/bin/env bash
#
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH --gres=gpu:1
#SBATCH --mem 100G
#SBATCH --time 3-00:00:00
#SBATCH --output /staging/agp/users/%u/logs/zpufferdrive/%j.out
#SBATCH --partition zprod
#SBATCH --job-name=pufferdrive
#SBATCH --reservation=ctps2
#

exec scripts/cluster/_hal.sh puffer train puffer_drive "$@"
