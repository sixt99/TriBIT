# =============================================
# Makefile for tribit.cu
# =============================================

CUDA_VERSION      ?= 12.8

# ------------------ Single GPU ------------------
SINGLE_TARGET     := tribit
SINGLE_SOURCE     := src/single_gpu/tribit.cu

SINGLE_NVCC_FLAGS := -std=c++20 \
                     --extended-lambda \
                     -O3 \
                     -arch=compute_90 \
                     -code=sm_90
# ------------------ Single GPU ------------------

# ------------------ Multi GPU ------------------
RUST_DIR          := src/multi_gpu 
MULTI_DIR         := src/multi_gpu/cuda
MULTI_KERNELS     := main5_multi main6_multi
MULTI_OBJECTS     := $(addprefix $(MULTI_DIR)/lib,$(addsuffix .o,$(MULTI_KERNELS)))
MULTI_LIBS        := $(addprefix $(MULTI_DIR)/lib,$(addsuffix .a,$(MULTI_KERNELS)))

MULTI_NVCC_FLAGS  := -std=c++20 \
                     --extended-lambda \
                     -O3 \
                     -arch=compute_90 \
                     -code=sm_90 \
                     -Xcompiler -fPIC
# ------------------ Multi GPU ------------------

.PHONY: all single multi clean

all: single multi

single: $(SINGLE_TARGET)

$(SINGLE_TARGET): $(SINGLE_SOURCE)
	nvcc $(SINGLE_NVCC_FLAGS) $< -o $@

multi: $(MULTI_LIBS) cargo

# .cu -> lib*.o
$(MULTI_DIR)/lib%.o: $(MULTI_DIR)/%.cu
	nvcc $(MULTI_NVCC_FLAGS) -c $< -o $@

# lib*.o -> lib*.a
$(MULTI_DIR)/lib%.a: $(MULTI_DIR)/lib%.o
	ar rcs $@ $<

cargo: 
	cd $(RUST_DIR) && \
	cargo clean --release && \
	cargo build --release

clean:
	rm -f $(SINGLE_TARGET) $(MULTI_OBJECTS) $(MULTI_LIBS) && \
	cd $(RUST_DIR) && \
	cargo clean --release
