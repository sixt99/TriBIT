#!/bin/bash
#SBATCH --account bsc03
#SBATCH --qos=acc_debug
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=80
#SBATCH --ntasks-per-node=1
#SBATCH --time 02:00:00
#SBATCH --nodes=2

timestamp=$(date +"%Y-%m-%d %H:%M:%S")
echo "Begin: $timestamp"

# USING SINGULARITY .SIF FILE
if [[ -n "$SIF_PATH" ]]; then
    WORKFOLDER="/app"
	echo "Running inside container: $SIF_PATH"
# RUNNING NATIVELY 
else
	WORKFOLDER="../.."
    echo "Running natively (no SIF_PATH set)"
fi

exe_path="$WORKFOLDER/src/multi_gpu/target/release/rs"
# No need to bind the following paths if "--contain" is not added to singularity run
data_path="data/multi_gpu/gsh-2015-host"
denyfile_path="run_benchmarks/denylist.txt"
partition_path="run_benchmarks/partitions.json"
results_path="results/raw"

python3 run_benchmarks/run_multi_gpu.py \
    --exe_path "$exe_path" \
    --data_path "$data_path" \
    --denyfile_path "$denyfile_path" \
	--partition_file "$partition_path" \
    --out_path "$results_path/results_multi.csv" \
    --n_repetitions 1
    #--dry_run

# Plot results
singularity exec $SIF_PATH python3 analysis/analyse_multi.py --input "$results_path/results_multi.csv" --output "$results_path/../plot_multi_gpu.txt" 

timestamp=$(date +"%Y-%m-%d %H:%M:%S")
echo "End: $timestamp"
