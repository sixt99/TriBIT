# TriBIT

This repository contains instructions to deploy and run TriBIT, an efficient triangle-counting algorithm that leverages binary TCUs, on both single-GPU and multi-GPU setups. For SC26 reproducibility, please see `artifact/sc26/README.md`.

## 1. Prerequisites

- For single-GPU experiments, an H100 GPU or similar.
- For multi-GPU and scalability experiments, access to a multi-GPU / multi-node SLURM cluster.
- The experiments below can be run natively (no container required) or by using a Singularity/Apptainer image (`tribit.sif`), which already contains the built executables and Python dependencies (recommended).

## 2. Building a Singularity Image (recommended)

**Option 1: Pull `.sif`**
```bash
singularity pull tribit.sif library://sixte99/tribit/tribit:v1
```

**Option 2: Build with Docker, then convert to `.sif`**

```bash
docker build -t tribit .
docker save tribit -o tribit.tar
singularity build tribit.sif docker-archive://tribit.tar
```

**Option 3: Build directly with Singularity/Apptainer**

```bash
sudo singularity build tribit.sif tribit.def
```
Once built, `tribit.sif` can be pointed to via `export SIF_PATH=/path/to/tribit.sif` for all benchmark scripts.

## 3. Building TriBIT From Scratch

If no container is used, TriBIT can be built natively. This requires:

- **CUDA 12.8** (`nvcc`) with an H100-compatible architecture (`compute_90`/`sm_90`).
- **CCCL v3.3.0**, installed under the CUDA targets directory.
- **Rust** (via `rustup`, toolchain `1.92.0`) for the multi-GPU components.
- An **MPI implementation** (e.g. OpenMPI with libopenmpi-dev, openmpi-bin, and PMIx via libpmix-dev, libpmix2; or Intel MPI v2021.16, bundled with Intel oneAPI 2025.2) for multi-node/multi-GPU communication.
- Standard build tools: `build-essential`, `cmake`, `git`, `pkg-config`, `libssl-dev`, `libclang-dev`.

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

## 4. Run Single-GPU TriBIT
```bash
./tribit -i nemeth23.mtx
```

If using the `.sif` file:

```bash
singularity exec --nv tribit.sif /app/./tribit -i nemeth23.mtx
```

## 5. Run Multi-GPU TriBIT

An example is provided in `tribit_multi_gpu.sh`. Launch it with
```
sbatch --gres=gpu:<n_gpus_per_node> --nodes=<n_nodes> tribit_multi_gpu.sh
```
Feel free to modify `tribit_multi_gpu.sh` to fit your setup. Note: depending on the MPI implementation, the extra flag `--mpi=pmix` may have to be modified.
