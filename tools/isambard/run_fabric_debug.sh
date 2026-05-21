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
  --env ROOT="$ROOT" \
  --env CUDA_HOME=/opt/cuda12 \
  --env LD_LIBRARY_PATH=/opt/cuda-compat:/hostusr/lib64:/opt/libfabric/lib:/opt/cuda12/lib64:/usr/lib/aarch64-linux-gnu \
  --env PPLX_GARDEN_LIBFABRIC_PROVIDER=cxi \
  --env PPLX_GARDEN_TOPOLOGY_NUMA=1 \
  --env FI_PROVIDER=cxi \
  --env FI_CXI_DEFAULT_CQ_SIZE=131072 \
  --env FI_CXI_DEFAULT_TX_SIZE=2048 \
  --env FI_CXI_RDZV_PROTO=alt_read \
  --env FI_CXI_RDZV_THRESHOLD=0 \
  --env FI_CXI_RDZV_GET_MIN=0 \
  --env FI_CXI_RDZV_EAGER_SIZE=0 \
  --env FI_CXI_RX_MATCH_MODE=hybrid \
  --env FI_CXI_DISABLE_NON_INJECT_MSG_IDC=1 \
  --env FI_CXI_DISABLE_HOST_REGISTER=1 \
  --env FI_HMEM_CUDA_USE_GDRCOPY=1 \
  --env FI_MR_CACHE_MONITOR=userfaultfd \
  "$CONTAINER" bash -lc '
    set -euo pipefail
    cd "$ROOT"
    exec target/release/fabric-debug "$@"
  ' fabric-debug "$@"
