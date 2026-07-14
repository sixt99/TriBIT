#!/bin/bash
#SBATCH --account bsc03
#SBATCH --qos=acc_debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time 02:00:00

timestamp=$(date +"%Y-%m-%d %H:%M:%S")
echo "Begin: $timestamp"

export CUDA_VISIBLE_DEVICES=0


# TODO REMOVE THE ENVIRONMENT PATH AFTER FIXING DOCKERFILE
# TODO 


NATIVE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$NATIVE_SCRIPT_DIR/../results/raw"

# USING SINGULARITY .SIF FILE
if [[ -n "$SIF_PATH" ]]; then
	echo "Running inside container: $SIF_PATH"
	WORKFOLDER="/app"
    cmd=(singularity exec \
	    --nv \
		--contain \
        --env "PATH=/opt/nvidia/nsight-compute/2025.1.0/host/target-linux-x64:\$PATH" \
		--env "TMPDIR=/tmp" \
        --bind "$NATIVE_SCRIPT_DIR/../data:/app/artifact/sc26/data" \
		--bind "$NATIVE_SCRIPT_DIR/../results:/app/artifact/sc26/results" \
		--bind "$NATIVE_SCRIPT_DIR/../run_benchmarks:/app/artifact/sc26/run_benchmarks" \
        "$SIF_PATH")
# RUNNING NATIVELY 
else
	echo "Running natively (no SIF_PATH set)"
	WORKFOLDER="$NATIVE_SCRIPT_DIR/../../.."
    cmd=()
fi

exe_path="$WORKFOLDER/./tribit"
data_path="$WORKFOLDER/artifact/sc26/data"
denyfile_path="$WORKFOLDER/artifact/sc26/run_benchmarks/denylist.txt"
results_path="$WORKFOLDER/artifact/sc26/results/raw"

"${cmd[@]}" python3 $WORKFOLDER/artifact/sc26/run_benchmarks/run_single_gpu.py \
    --exe_path "$exe_path" \
    --data_path "$data_path" \
    --denyfile_path "$denyfile_path" \
    --out_path "$results_path/results_single${SLURM_JOB_ID}.csv" \
    --n_repetitions 1 \
    --get_memory_consumption 
    #--dry_run

timestamp=$(date +"%Y-%m-%d %H:%M:%S")
echo "End: $timestamp"
