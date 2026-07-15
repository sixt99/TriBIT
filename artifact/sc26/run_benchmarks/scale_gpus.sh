#!/bin/bash
#SBATCH --account bsc03
#SBATCH --qos=acc_bsccs
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=80
#SBATCH --ntasks-per-node=1
#SBATCH --time 02:00:00

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

# Scalability only makes sense for graphs that can be run with 1 GPU
files=(
	gsh-2015-host
	gsh-2015
	twitter-2010
	uk-2006-09
	uk-2007-01
	uk-2007-02
	uk-2014
	uk-union-2006-06-2007-05
)

if [ "$SLURM_JOB_NUM_NODES" -eq 1 ]; then
        GPUS_PER_NODE=(1 2 3 4)
else
        GPUS_PER_NODE=(4)
fi

exe_path="$WORKFOLDER/src/multi_gpu/target/release/rs"
denyfile_path="run_benchmarks/denylist.txt"
partition_path="run_benchmarks/partitions.json"
results_path="results/raw"

for FILE in "${files[@]}"; do
	data_path="data/multi_gpu/$FILE"
    for N_GPUS in "${GPUS_PER_NODE[@]}"; do
        python3 run_benchmarks/run_multi_gpu.py \
			--exe_path $exe_path \
			--data_path "$data_path" \
			--denyfile_path "$denyfile_path" \
			--partition_file "$partition_path" \
			--out_path "$results_path/results_multi.csv" \
			--n_repetitions 1 \
			--gpus_per_node "$N_GPUS"
    done
done

timestamp=$(date +"%Y-%m-%d %H:%M:%S")
echo "End: $timestamp"





