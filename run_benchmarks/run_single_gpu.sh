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
	--data_path ../data \
	--denyfile_path denylist.txt \
	--out_path result_$SLURM_JOB_ID.csv \
	--n_repetitions 1 \
	--dry_run \
	--get_memory_consumption

timestamp=$(date +"%Y-%m-%d %H:%M:%S")
echo "End: $timestamp"
