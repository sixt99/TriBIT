for N_NODES in 1 2 3 4 5 6 7; do # 8 nodes is already included in multi_gpu experiments
    sbatch --nodes=$N_NODES run_benchmarks/scale_gpus.sh
done

# Plot results
if [[ -n "$SIF_PATH" ]]; then
	plot_cmd=(singularity exec "$SIF_PATH")
else
	plot_cmd=()
fi

results_path="results/raw"
# ATTENTION: run this command only when jobs finish 
# "${plot_cmd[@]}" python3 analysis/analyse_scalability.py --input "$results_path/results_multi.csv" --output "$results_path/../plot_scalability.png" 
