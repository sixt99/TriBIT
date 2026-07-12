#!/bin/bash
#SBATCH --account bsc03
#SBATCH --qos=acc_debug
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=80
#SBATCH --ntasks-per-node=1
#SBATCH --time 02:00:00

timestamp=$(date +"%Y-%m-%d %H:%M:%S")
echo "Begin: $timestamp"
mkdir -p ../results

# Scalability only makes sense for graphs that can be run with 1 GPU
files=(
        gsh-2015
        gsh-2015-host
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

for FILE in "${files[@]}"; do
        for N_GPUS in "${GPUS_PER_NODE[@]}"; do
                python3 run_multi_gpu.py \
                        --exe_path ../src/multi_gpu/target/release/rs \
                        --data_path "../data/multi_gpu/test/$FILE" \
                        --denyfile_path denylist.txt \
                        --partition_file partitions.json \
                        --out_path ../results/results_multi.csv \
                        --n_repetitions 1 \
                        --gpus_per_node "$N_GPUS"
        done
done

timestamp=$(date +"%Y-%m-%d %H:%M:%S")
echo "End: $timestamp"
