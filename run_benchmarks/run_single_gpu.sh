#!/bin/bash
#SBATCH --account bsc03
#SBATCH --qos=acc_bsccs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time 02:00:00

timestamp=$(date +"%Y-%m-%d %H:%M:%S")
echo "Begin: $timestamp"

export CUDA_VISIBLE_DEVICES=0
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
