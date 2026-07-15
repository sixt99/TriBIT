#for N_NODES in 1 2 3 4 5 6 7; do # 8 nodes is already included in multi_gpu experiments
for N_NODES in 2; do # 8 nodes is already included in multi_gpu experiments
    sbatch --nodes=$N_NODES run_benchmarks/scale_gpus.sh
done

