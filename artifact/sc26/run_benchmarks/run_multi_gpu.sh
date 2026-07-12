#!/bin/bash
#SBATCH --account bsc03
#SBATCH --qos=acc_debug
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=80
#SBATCH --ntasks-per-node=1
#SBATCH --time 02:00:00
#SBATCH --nodes=1

timestamp=$(date +"%Y-%m-%d %H:%M:%S")
echo "Begin: $timestamp"

mkdir -p ../results
python3 run_multi_gpu.py \
	--exe_path ../../../src/multi_gpu/target/release/rs \
	--data_path ../data/multi_gpu/gsh-2015-host \
	--denyfile_path denylist.txt \
	--partition_file partitions.json \
	--out_path ../results/results_multi.csv \
	--n_repetitions 1 
	#--dry_run \

timestamp=$(date +"%Y-%m-%d %H:%M:%S")
echo "End: $timestamp"
