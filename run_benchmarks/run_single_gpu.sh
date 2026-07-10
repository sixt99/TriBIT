#!/bin/bash
#SBATCH --account bsc03
#SBATCH --qos=acc_bsccs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time 05:00:00

#set -euo pipefail # Exit the bash script if something fails
#module purge; module load mkl intel impi boost cuda/12.8 python/3.12.1 gcc/13.2.0 sqlite3
module purge; module load cuda/12.8 intel mkl python/3.12.1 sqlite3

export CUDA_VISIBLE_DEVICES=0

timestamp=$(date +"%Y-%m-%d %H:%M:%S")
echo "Begin: $timestamp"

python3 run_single_gpu.py \
	--exe_path ./tribit \
	--mtx_path ../data \
	--csv_path result_$SLURM_JOB_ID.csv \
	--dry_run true \
	--n_repetitions 5 \
	--get_memory_consumption true

timestamp=$(date +"%Y-%m-%d %H:%M:%S")
echo "End: $timestamp"
