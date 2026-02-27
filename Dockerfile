# CU Benchmark Suite v1.0
# Standardized GPU/accelerator benchmarking environment.
#
# Recommended: use run.sh wrapper (bakes in all required flags):
#   ./run.sh
#   CU_QUICK=1 ./run.sh
#   ./run.sh -v /path/to/models:/models -e CU_BENCH_MODEL=llama-70b
#
# With upload to home server (results sent before container exits):
#   ./run.sh \
#     -v ~/.ssh/id_ed25519:/root/.ssh/id_bench:ro \
#     -e CU_UPLOAD_DEST=user@homeserver:/data/cu-results/
#
# Manual docker run (all flags required):
#   docker run --rm --gpus all --cap-add=SYS_ADMIN \
#     --ipc=host --ulimit memlock=-1:-1 \
#     -v ./results:/results ghcr.io/yggdrasil-technologies/cu-bench:latest
#
# --cap-add=SYS_ADMIN:     required for nvidia-smi clock locking & persistence mode
# --gpus all:              exposes all GPUs via nvidia-container-toolkit
# --ipc=host:              required for NCCL shared memory (multi-GPU interconnect)
# --ulimit memlock=-1:-1:  allow pinned memory for NCCL all-reduce operations

ARG CUDA_VERSION=12.8.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV RESULTS_DIR=/results
ENV CU_BENCH_MODEL_DIR=/models
# Upload configuration (set at runtime — do NOT bake credentials into image)
# hadolint ignore=DL3048
ENV CU_UPLOAD_DEST="" \
    CU_UPLOAD_WEBHOOK="" \
    CU_UPLOAD_KEY="" \
    CU_UPLOAD_KEY_TYPE="ed25519"

# ─── System dependencies ───
RUN apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update \
    && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    git wget curl jq numactl \
    openssh-client rsync \
    && rm -rf /var/lib/apt/lists/*

# ─── Python packages ───
RUN pip3 install --no-cache-dir \
    rich matplotlib numpy plotext

# PyTorch — let pip resolve CUDA-compatible version
RUN pip3 install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu128

# vLLM for inference benchmark (optional)
RUN pip3 install --no-cache-dir \
    vllm || echo "WARNING: vLLM install failed — inference benchmark will be skipped"

# ─── NCCL tests for interconnect benchmark ───
RUN git clone --depth 1 https://github.com/NVIDIA/nccl-tests.git /opt/nccl-tests && \
    cd /opt/nccl-tests && \
    make MPI=0 CUDA_HOME=/usr/local/cuda -j$(nproc) || \
    echo "WARNING: nccl-tests build failed — interconnect benchmark will be skipped"

# ─── SSH dir (for upload) ───
RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh

# ─── Benchmark scripts ───
COPY bench/ /bench/
RUN chmod +x /bench/*.sh /bench/*.py

# ─── Mount points ───
VOLUME ["/models", "/results"]

WORKDIR /bench
ENTRYPOINT ["bash", "/bench/run_all.sh"]
