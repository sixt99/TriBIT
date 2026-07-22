# SC26 Artifact

This folder contains the artifact for the SC26 submission: data-download utilities, benchmarking scripts, and analysis/plotting scripts for single-GPU and multi-GPU triangle-counting experiments, plus a multi-node scalability study.

## 1. Prerequisites

- For single-GPU experiments, an H100 GPU or similar.
- For multi-GPU and scalability experiments, access to a SLURM cluster with up to 32 H100 GPUs (4 per node, up to 8 nodes).
- The experiments below can be run natively (no container required) or by using a Singularity/Apptainer image (`tribit.sif`), which already contains the built executables and Python dependencies (recommended).
- You can install Singularity/Apptainer by running
```bash
sudo apt update
sudo apt install singularity-container
```
- Run `export SIF_PATH=/path/to/tribit.sif` to run benchmark + plotting inside the Singularity container for full reproducibility; leave it unset to run natively against local builds.

## 2. Building a Singularity Image (recommended)

**Option 1: Pull `.sif`**
```bash
singularity pull tribit.sif library://sixte99/tribit/tribit:v1
```

**Option 2: Build directly with Singularity/Apptainer**

```bash
sudo singularity build tribit.sif tribit.def
```

**Option 3: Build with Docker, then convert to `.sif`**

```bash
docker build -t tribit .
docker save tribit -o tribit.tar
singularity build tribit.sif docker-archive://tribit.tar
```

Once built, `tribit.sif` can be pointed to via `export SIF_PATH=/path/to/tribit.sif` for all benchmark scripts.

## 3. Building TriBIT From Scratch

If NO container is used, TriBIT can be built natively. This requires:

- **CUDA 12.8** (`nvcc`) with an H100-compatible architecture (`compute_90`/`sm_90`).
- **CCCL v3.3.0**, installed under the CUDA targets directory.
- **Rust** (via `rustup`, toolchain `1.92.0`) for the multi-GPU components.
- An **MPI implementation** (e.g. OpenMPI with libopenmpi-dev, openmpi-bin, and PMIx via libpmix-dev, libpmix2; or Intel MPI v2021.16, bundled with Intel oneAPI 2025.2) for multi-node/multi-GPU communication.
- Standard build tools: `build-essential`, `cmake`, `git`, `pkg-config`, `libssl-dev`, `libclang-dev`.
- **Python 3** with `tabulate`, `pandas`, `matplotlib`, and `ssgetpy` installed (for benchmarking/analysis scripts).

Once dependencies are installed, build both the single-GPU and multi-GPU executables with:

```bash
make            # builds both single-GPU (tribit) and multi-GPU (Rust + CUDA) targets
make single     # builds only the single-GPU executable
make multi      # builds only the multi-GPU executable
make clean      # removes all build artifacts
```

This produces:
- `tribit`: the single-GPU binary.
- `src/multi_gpu/target/release/rs`: the multi-GPU executable.

## 4. Downloading Data

### Single-GPU (SuiteSparse Matrix Collection)
If using `tribit.sif`, 
```bash
export SIF_PATH=/path/to/tribit.sif
cd artifact/sc26
singularity exec "$SIF_PATH" python3 download/download_data_single.py --input download/graphs.txt --output data
```
If NOT using `tribit.sif`,
```bash
cd artifact/sc26
python3 download/download_data_single.py --input download/graphs.txt --output data # Requires ssgetpy
```
- Downloads 174.57 GiB.

### Multi-GPU (Large Web Graphs)

```bash
cd artifact/sc26
chmod +x download/download_data_multi.sh
./download/download_data_multi.sh --output data/multi_gpu
```
- Downloads 35.62 GiB of compressed data.
- `Hyperlink-2012` and `Hyperlink-2014` have been omitted but can be included by uncommenting them in `download/download_data_multi.sh`.

## 5. Running Benchmarks

### Single-GPU Benchmark

```bash
export SIF_PATH=/path/to/tribit.sif   # optional; omit to run natively
export TMPDIR=/path/to/tmpdir
cd artifact/sc26
./run_single_gpu.sh           # directly
#sbatch run_single_gpu.sh     # or via SLURM
```

- Requests 1 H100 GPU (or similar), 2 hours.
- Runs `run_benchmarks/run_single_gpu.py`, which drives the `tribit` executable over each dataset in `data/`, skipping anything in `run_benchmarks/denylist.txt`, and also records memory consumption (`--get_memory_consumption`).
- Writes raw results to `results/raw/results_single.csv`.
- Automatically generates `results/plot_single_gpu.png` via `analysis/analyse_single.py`.

### Multi-GPU Benchmark

```bash
export SIF_PATH=/path/to/tribit.sif   # optional; omit to run natively
cd artifact/sc26
sbatch run_multi_gpu.sh # SLURM is needed for 32-GPU experiments
```

- Requests 8 nodes, 4 GPUs/node, 80 CPUs/task.
- Runs `run_benchmarks/run_multi_gpu.py`, which drives the multi-GPU executable (`target/release/rs`) using `run_benchmarks/partitions.json` for partitioning and `run_benchmarks/denylist.txt` to skip datasets.
- Writes raw results to `results/raw/results_multi.csv`.
- Automatically generates `results/plot_multi_gpu.txt` via `analysis/analyse_multi.py`.

### Scalability Sweep (1–7 nodes)

```bash
export SIF_PATH=/path/to/tribit.sif   # optional; omit to run natively
cd artifact/sc26
chmod +x run_scalability.sh
./run_scalability.sh
```

- Submits `run_benchmarks/scale_gpus.sh` once per node count, for `N_NODES` in `1..7` (8-node results are already covered by the multi-GPU experiment above).
- **Important:** the final plotting step (`analysis/analyse_scalability.py`, producing `results/plot_scalability.png`) is commented out in `run_scalability.sh` by design, since it must only be run *after* all submitted jobs have completed. Once all scalability jobs finish, uncomment and run that line manually.

## 6. Outputs

| File | Produced by | Description |
|---|---|---|
| `results/raw/results_single.csv` | `run_single_gpu.sh` | Per-dataset timing/memory results, single GPU |
| `results/raw/results_multi.csv` | `run_multi_gpu.sh` / `run_scalability.sh` | Per-dataset timing results, multi-GPU / multi-node |
| `results/plot_single_gpu.png` | `analysis/analyse_single.py` | Single-GPU performance plot |
| `results/plot_multi_gpu.txt` | `analysis/analyse_multi.py` | Multi-GPU performance summary |
| `results/plot_scalability.png` | `analysis/analyse_scalability.py` (manual step) | Strong/weak scaling plot across 1–8 nodes |

## 7. Notes
- Dry runs are available: uncomment the `#--dry_run` flag in `run_single_gpu.sh` / `run_multi_gpu.sh` to validate the pipeline without launching full benchmarks.
- The flag `--max_matrices` in `run_benchmarks/run_single_gpu.py` allows to limit the maximum processed `.mtx` files in single-GPU experiments.
