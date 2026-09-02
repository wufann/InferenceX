#!/usr/bin/env bash
# 纯 SGLang Decode server · DSv4 FP4 · B300 · TP-only · fake-prefill。
# 用于纯 Decode kernel trace：只启动一个 Decode 实例，用 fake transfer backend
# 直接给 128K/256K 分配 KV 并立即把 transfer 标记完成，不做真实 Prefill、不搬真实 KV。
# 输出 token 无正确性意义，但稳态 Decode 的 batch/KV length/attention/MoE/MTP kernel
# shape 可用于性能 profiling。不适合测 P→D transfer、首 token 延迟、真实 KV 或 prefix 命中。
#
# 与 server_tponly.sh 的差异（仅新增，其余保持一致以贴合目标 Decode 配置）：
#   + --disaggregation-mode decode
#   + --disaggregation-transfer-backend fake
#   + --disable-radix-cache        # fake prefill 时不要开 decode radix cache
#
# 用法: MODEL_PATH=/data/models/DeepSeek-V4-Pro TP=8 CONC=16 PORT=8888 bash server_tponly_fakeprefill.sh
# 客户端须用 run_case.sh ... --fake-prefill，并把 URL 直接指向本 server（不经 PD router）。
set -euo pipefail

MODEL="${MODEL:-deepseek-ai/DeepSeek-V4-Pro}"       # 对外 served-model-name
MODEL_PATH="${MODEL_PATH:-/data/models/DeepSeek-V4-Pro}"
TP="${TP:-8}"
CONC="${CONC:-16}"
PORT="${PORT:-8888}"

# ---- server 相关 env（性能/正确性，不是 InferenceX 动作）----
export PYTHONNOUSERSITE=1
export TORCH_CUDA_ARCH_LIST=10.0
export SGLANG_TIMEOUT_KEEP_ALIVE=900
export SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
export SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS=1
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
export SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT=1
export SGLANG_OPT_USE_JIT_NORM=1
export SGLANG_OPT_USE_JIT_INDEXER_METADATA=1
export SGLANG_OPT_USE_TOPK_V2=1
export SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=1
# 要抓 profiler trace 就在这里 export（起 server 前设才生效）：
export SGLANG_TORCH_PROFILER_DIR=/home/fanwu103/traces
export SGLANG_PROFILE_RECORD_SHAPES=True

MAX_RUNNING=$((2 * CONC))
CGBS=$((CONC * 4)); [ "$CGBS" -gt 64 ] && CGBS=64

exec python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" --served-model-name "$MODEL" \
  --host 0.0.0.0 --port "$PORT" --trust-remote-code \
  --tp "$TP" \
  --moe-runner-backend flashinfer_mxfp4 --disable-flashinfer-autotune \
  --mem-fraction-static 0.88 --swa-full-tokens-ratio 0.1 \
  --max-running-requests "$MAX_RUNNING" --cuda-graph-max-bs "$CGBS" \
  --allow-auto-truncate --chunked-prefill-size 8192 \
  --tool-call-parser deepseekv4 --reasoning-parser deepseek-v4 \
  --watchdog-timeout 1800 \
  --attention-backend compressed --page-size 256 --disable-shared-experts-fusion \
  --enable-metrics \
  --disaggregation-mode decode --disaggregation-transfer-backend fake \
  --disable-radix-cache \
  ${CHAT_TEMPLATE:+--chat-template "$CHAT_TEMPLATE"}
