#!/bin/bash
#SBATCH --account bsc03
#SBATCH --qos=acc_debug
#SBATCH --cpus-per-task=80
#SBATCH --time 01:00:00
#SBATCH --ntasks-per-node=1

DATA_PATH=artifact/sc26/data/multi_gpu/gsh-2015-host/gsh-2015-host 
N_PARTITIONS=64

# Run natively
srun src/multi_gpu/target/release/rs $DATA_PATH $N_PARTITIONS 

# Run under tribit.sif
srun --mpi=pmix singularity exec --nv tribit.sif /app/src/multi_gpu/target/release/rs $DATA_PATH $N_PARTITIONS 
