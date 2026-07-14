#!/bin/bash
#SBATCH --account bsc03
#SBATCH --qos=acc_debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time 02:00:00

timestamp=$(date +"%Y-%m-%d %H:%M:%S")
echo "Begin: $timestamp"

export CUDA_VISIBLE_DEVICES=0
mkdir -p ../results


# TODO REMOVE THE ENVIRONMENT PATH AFTER FIXING DOCKERFILE



# Set variables depending on docker context
if [[ -n "$SIF_PATH" ]]; then
	echo "Running inside container: $SIF_PATH"
    cmd=(singularity exec --nv \
        --env "PATH=/opt/nvidia/nsight-compute/2025.1.0/host/target-linux-x64:\$PATH" \
        --bind "$(pwd)/../data:/data" \
        "$SIF_PATH")
	exe_path=/app/./tribit
    data_path=/data
else
	echo "Running natively (no SIF_PATH set)"
    cmd=()
    exe_path=../../.././tribit
    data_path=../data
fi

"${cmd[@]}" python3 run_single_gpu.py \
    --exe_path $exe_path \
    --data_path $data_path \
    --denyfile_path denylist.txt \
    --out_path ../results/result_${SLURM_JOB_ID}.csv \
    --n_repetitions 1 \
    --get_memory_consumption
    #--dry_run

timestamp=$(date +"%Y-%m-%d %H:%M:%S")
echo "End: $timestamp"
