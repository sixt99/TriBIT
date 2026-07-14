# =============================================
# Dockerfile for tribit.cu
# =============================================
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Install build dependencies needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential ca-certificates pkg-config libssl-dev libclang-dev \
    cmake git vim \
    software-properties-common \
    libopenmpi-dev openmpi-bin libpmix-dev libpmix2 \
    python3 python3-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Rust via rustup
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --default-toolchain 1.92.0 --profile minimal
ENV CUDA_ARCH_DIR="$NVARCH-linux"
ENV LIBRARY_PATH="/usr/local/cuda-12.8/targets/${CUDA_ARCH_DIR}/lib:${LIBRARY_PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda-12.8/targets/${CUDA_ARCH_DIR}/lib:${LD_LIBRARY_PATH}"

# Install CCCL v3.3.0
RUN curl -L https://github.com/NVIDIA/cccl/archive/refs/tags/v3.3.0.tar.gz -o cccl.tar.gz && \
    tar -xzf cccl.tar.gz && \
    cmake -S cccl-3.3.0 -B /tmp/cccl-build \
        -DCMAKE_INSTALL_PREFIX=/usr/local/cuda-12.8/targets/$CUDA_ARCH_DIR \
        -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /tmp/cccl-build --target install -j2 && \
    rm -rf cccl.tar.gz cccl-3.3.0 /tmp/cccl-build

# Install python packages
RUN pip3 install --no-cache-dir tabulate pandas matplotlib
# Run PMIX by default
ENV PMIX_MCA_psec=native
# Add the profiler to PATH
ENV PATH="/opt/nvidia/nsight-compute/2025.1.0/host/target-linux-x64:${PATH}"

WORKDIR /app
COPY . .
RUN make
