#!/bin/bash
#SBATCH --account bsc03
#SBATCH --qos=acc_debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time 02:00:00

timestamp=$(date +"%Y-%m-%d %H:%M:%S")
echo "Begin: $timestamp"

export CUDA_VISIBLE_DEVICES=0

# Make sure tmpdir exists
if [[ -n "$TMPDIR" ]]; then
	echo "TMPDIR is $TMPDIR"
else
	echo "Please do 'export TMPDIR=<tmpdir_path>'"
	exit 1
fi

# USING SINGULARITY .SIF FILE
if [[ -n "$SIF_PATH" ]]; then
	echo "Running inside container: $SIF_PATH"
	WORKFOLDER="/app"
	ARTIFACT_DIR="$WORKFOLDER/artifact/sc26"
    cmd=(singularity exec \
	    --nv \
		--contain \
		--bind "$TMPDIR:/tmp" \
        --bind "data:$ARTIFACT_DIR/data" \
		--bind "results:$ARTIFACT_DIR/results" \
		--bind "run_benchmarks/denylist.txt:$ARTIFACT_DIR/run_benchmarks/denylist.txt" \
        "$SIF_PATH")
# RUNNING NATIVELY 
else
	echo "Running natively (no SIF_PATH set)"
	WORKFOLDER="../.."
	ARTIFACT_DIR="."
    cmd=()
fi

exe_path="$WORKFOLDER/./tribit"
data_path="$ARTIFACT_DIR/data"
denyfile_path="$ARTIFACT_DIR/run_benchmarks/denylist.txt"
results_path="$ARTIFACT_DIR/results/raw"

# Count triangles
"${cmd[@]}" python3 $ARTIFACT_DIR/run_benchmarks/run_single_gpu.py \
    --exe_path "$exe_path" \
    --data_path "$data_path" \
    --denyfile_path "$denyfile_path" \
    --out_path "$results_path/results_single.csv" \
    --n_repetitions 1 \
    --get_memory_consumption \
    #--dry_run

# Plot results
"${cmd[@]}" python3 $ARTIFACT_DIR/analysis/analyse_single.py --input "$results_path/results_single.csv" --output "$results_path/../plot_single_gpu.png" 

timestamp=$(date +"%Y-%m-%d %H:%M:%S")
echo "End: $timestamp"
