# CU Benchmark Suite v1.0
# Standardized GPU/accelerator benchmarking environment.
#
# NVIDIA:
#   docker build -t cu-bench:1.0 .
#   docker run --gpus all --cap-add=SYS_ADMIN -v ./results:/results cu-bench:1.0
#
# With model for inference:
#   docker run --gpus all --cap-add=SYS_ADMIN \
#     -v ./results:/results -v /path/to/models:/models cu-bench:1.0
#
# --cap-add=SYS_ADMIN: required for nvidia-smi clock locking & persistence mode
# --gpus all: exposes all GPUs via nvidia-container-toolkit

ARG CUDA_VERSION=12.8.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV RESULTS_DIR=/results
ENV CU_BENCH_MODEL_DIR=/models

# ─── System dependencies ───
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    git wget curl jq numactl \
    && rm -rf /var/lib/apt/lists/*

# ─── Python packages ───
RUN pip3 install --no-cache-dir --break-system-packages \
    rich

# PyTorch — let pip resolve CUDA-compatible version
RUN pip3 install --no-cache-dir --break-system-packages \
    torch --index-url https://download.pytorch.org/whl/cu128

# vLLM for inference benchmark (optional)
RUN pip3 install --no-cache-dir --break-system-packages \
    vllm || echo "WARNING: vLLM install failed — inference benchmark will be skipped"

# ─── NCCL tests for interconnect benchmark ───
RUN git clone --depth 1 https://github.com/NVIDIA/nccl-tests.git /opt/nccl-tests && \
    cd /opt/nccl-tests && \
    make MPI=0 CUDA_HOME=/usr/local/cuda -j$(nproc) || \
    echo "WARNING: nccl-tests build failed — interconnect benchmark will be skipped"

# ─── Benchmark scripts ───
COPY bench/ /bench/
RUN chmod +x /bench/*.sh /bench/*.py

# ─── Mount points ───
VOLUME ["/models", "/results"]

WORKDIR /bench
ENTRYPOINT ["bash", "/bench/run_all.sh"]
