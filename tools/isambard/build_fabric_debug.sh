#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/s6p/fergus.s6p/pplx-garden-cxi/pplx-garden}"
CONTAINER="${CONTAINER:-/lus/lfs1aip2/scratch/s6p/fergus.s6p/containers/vllm-u2404.sif}"

apptainer exec --nv \
  --bind /home/s6p/fergus.s6p/pplx-garden-cxi:/home/s6p/fergus.s6p/pplx-garden-cxi \
  --bind /lus/lfs1aip2/scratch/s6p/fergus.s6p/nvhpc/Linux_aarch64/26.3/cuda/13.1/compat:/opt/cuda-compat \
  --bind /opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6:/opt/cuda12 \
  --bind /opt/cray/libfabric/1.22.0/include:/opt/libfabric/include \
  --bind /opt/cray/libfabric/1.22.0/lib64:/opt/libfabric/lib \
  --bind /usr/include/gdrapi.h:/usr/include/gdrapi.h \
  --bind /opt/cray/pe/cce/19.0.0:/opt/cray/pe/cce/19.0.0 \
  --bind /usr/lib64:/hostusr/lib64 \
  --bind /dev/cxi0:/dev/cxi0 \
  --bind /dev/cxi1:/dev/cxi1 \
  --bind /dev/cxi2:/dev/cxi2 \
  --bind /dev/cxi3:/dev/cxi3 \
  --env TMPDIR=/tmp \
  --env ROOT="$ROOT" \
  --env CUDA_HOME=/opt/cuda12 \
  --env LIBFABRIC_HOME=/opt/libfabric \
  --env GDRAPI_HOME=/usr \
  --env LIBIBVERBS_HOME=/usr \
  --env LIBCLANG_PATH=/opt/cray/pe/cce/19.0.0/cce-clang/aarch64/lib \
  --env "BINDGEN_EXTRA_CLANG_ARGS=-isystem /opt/cray/pe/cce/19.0.0/cce-clang/aarch64/lib/clang/19/include" \
  --env LD_LIBRARY_PATH=/opt/cuda-compat:/hostusr/lib64:/opt/cray/pe/cce/19.0.0/cce-clang/aarch64/lib:/opt/libfabric/lib:/opt/cuda12/lib64:/usr/lib/aarch64-linux-gnu \
  "$CONTAINER" bash -lc '
    set -euo pipefail
    export LIBRARY_PATH=/hostusr/lib64:/opt/libfabric/lib
    cd "$ROOT"
    cargo build -p fabric-debug --release
  '
