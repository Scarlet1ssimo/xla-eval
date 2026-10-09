#!/usr/bin/env bash
set -euo pipefail
xla_root=$(cd "$(dirname "$0")/.." && pwd)
source "$LMOD_PKG/init/bash"
module load gcc/12.3.0-ubuntu cuda-toolkit/13.0
python_lib=$("$xla_root/.venv/bin/python" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')
# Keep cuDNN/CUDA runtime versions from the pinned wheel environment.
export LD_LIBRARY_PATH="$python_lib"
export PJRT_DEVICE=CUDA CUDA_VISIBLE_DEVICES=0
dump_dir="$xla_root/outputs/hlo"
args=("$@")
for ((i=0; i<$#; i++)); do
  if [[ "${args[$i]}" == "--directory" ]]; then dump_dir="${args[$((i+1))]}/hlo"; fi
done
for ((i=0; i<$#; i++)); do
  if [[ "${args[$i]}" == "--result-name" && "${args[$((i+1))]}" == "event-result.json" ]]; then dump_dir="${dump_dir}-events"; fi
done
mkdir -p "$dump_dir"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/software/cuda-13.0 --xla_dump_to=$dump_dir --xla_dump_hlo_as_text"
exec "$xla_root/.venv/bin/python" "$@"
