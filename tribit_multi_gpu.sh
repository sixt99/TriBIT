#!/bin/bash
#SBATCH --account bsc03
#SBATCH --qos=acc_debug
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=80
#SBATCH --time 01:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1

srun singularity exec --nv tribit.sif /app/src/multi_gpu/target/release/rs artifact/sc26/data/multi_gpu/gsh-2015-host/gsh-2015-host 128 

