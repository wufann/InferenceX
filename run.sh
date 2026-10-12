#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

INFX_MODEL_DIR=/shared/amdgpu/home/fan_wu2_7kq/models/quark/Qwen3.8-Flash-Next-Quark-MXFP4-PLEFP8
test -f "$INFX_MODEL_DIR/config.json"
git submodule update --init inferencex-e2e/utils/aiperf
mkdir -p "$PWD/local-agentx-results" "$HOME/.cache/huggingface"

podman run --rm --network host --ipc host \
  --device /dev/kfd --device /dev/dri --group-add keep-groups \
  -e ROCR_VISIBLE_DEVICES=0 \
  -v "$PWD:/work/InferenceX:ro" \
  -v "$INFX_MODEL_DIR:/models/qwen38:ro" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$PWD/local-agentx-results:/results" \
  -w /work/InferenceX/inferencex-e2e \
  aigmkt/qwen3.8-flash-next:v0.5.21-rocm10-mi35x-20261009-nikolai-opt \
  bash -lc 'PYTHONPATH=. python3 -m infx.bench.local_qwen38 run \
    --model amd/Qwen3.8-Flash-Next-Quark-MXFP4-PLEFP8 \
    --model-path /models/qwen38 \
    --image aigmkt/qwen3.8-flash-next:v0.5.21-rocm10-mi35x-20261009-nikolai-opt \
    --output-dir /results/qwen38-002 --concurrencies 16 \
    --duration 3600 --port 8000'
