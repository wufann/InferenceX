#!/usr/bin/env bash
# 纯 SGLang server 启动 · DSv4 FP4 · B300 · DEP8 (dp-attn + megamoe + hicache + router)。
# 无任何 InferenceX 依赖。backend 在 PORT+1，router 在 PORT（对外打 PORT）。
# 用法: MODEL_PATH=/data/models/DeepSeek-V4-Pro TP=8 EP_SIZE=8 CONC=128 PORT=8888 \
#       HICACHE_RATIO=2 bash server_dep8.sh
set -euo pipefail

MODEL="${MODEL:-deepseek-ai/DeepSeek-V4-Pro}"
MODEL_PATH="${MODEL_PATH:-/data/models/DeepSeek-V4-Pro}"
TP="${TP:-8}"
EP="${EP_SIZE:-8}"
CONC="${CONC:-128}"
PORT="${PORT:-8888}"
BACKEND=$((PORT + 1))
HICACHE_RATIO="${HICACHE_RATIO:-3}"      # 你这台 RAM 小建议 1~2

# ---- server 相关 env ----
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
# MegaMoE 的 FP4/MXF4 激活路径（opt-in，否则跑的是另一套 kernel）
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS=1
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND=1
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320
# 要抓 profiler trace 就在这里 export：
# export SGLANG_TORCH_PROFILER_DIR=/home/fanwu103/traces

# mem-fraction 随 CONC（MegaMoE workspace 在静态分配之外，headroom 随 batch 增长）
MF=0.93
if   [ "$CONC" -ge 512 ]; then MF=0.86
elif [ "$CONC" -ge 384 ]; then MF=0.88
elif [ "$CONC" -ge 256 ]; then MF=0.9
fi
MAX_RUNNING=$((2 * CONC))
CHUNK=$((8192 * TP))

# ---- 1) backend（后台）----
python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" --served-model-name "$MODEL" \
  --host 0.0.0.0 --port "$BACKEND" --trust-remote-code \
  --tp "$TP" --dp "$TP" --tokenizer-worker-num "$TP" \
  --enable-prefill-delayer --prefill-decode-interval 20 \
  --enable-dp-attention --enable-dp-attention-local-control-broadcast \
  --incremental-streaming-output --stream-interval 20 \
  --dist-init-addr "127.0.0.1:$((PORT + 2000))" --ep-size "$EP" \
  --moe-a2a-backend megamoe --enable-deepseek-v4-fp4-indexer --disable-flashinfer-autotune \
  --mem-fraction-static "$MF" --swa-full-tokens-ratio 0.075 \
  --max-running-requests "$MAX_RUNNING" --cuda-graph-max-bs-decode 544 \
  --allow-auto-truncate --chunked-prefill-size "$CHUNK" \
  --tool-call-parser deepseekv4 --reasoning-parser deepseek-v4 \
  --watchdog-timeout 1800 \
  --speculative-algorithm EAGLE --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --attention-backend compressed --page-size 256 --disable-shared-experts-fusion \
  --enable-metrics --enable-cache-report \
  --enable-hierarchical-cache --hicache-ratio "$HICACHE_RATIO" \
  --hicache-write-policy write_back --hicache-io-backend direct --hicache-mem-layout page_first_direct \
  ${CHAT_TEMPLATE:+--chat-template "$CHAT_TEMPLATE"} &
BACKEND_PID=$!

trap 'kill $BACKEND_PID ${ROUTER_PID:-} 2>/dev/null || true' EXIT INT TERM

# 等 backend 健康
echo "waiting backend :$BACKEND ..."
until curl -sf "http://localhost:$BACKEND/health" >/dev/null 2>&1; do
  kill -0 "$BACKEND_PID" 2>/dev/null || { echo "backend died"; exit 1; }
  sleep 5
done
echo "backend ready on :$BACKEND"

# ---- 2) router（后台，带重试）----
python3 -m sglang_router.launch_router \
  --worker-urls "http://localhost:$BACKEND" \
  --policy consistent_hashing --request-id-headers x-correlation-id --dp-aware \
  --host 0.0.0.0 --port "$PORT" \
  --prometheus-host 127.0.0.1 --prometheus-port "$((PORT + 10000))" \
  --connect-timeout-secs 900 --request-timeout-secs 14400 --disable-health-check \
  --retry-max-retries 8 --retry-initial-backoff-ms 500 \
  --retry-max-backoff-ms 10000 --retry-backoff-multiplier 2 &
ROUTER_PID=$!

echo "waiting router :$PORT ..."
until curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; do
  kill -0 "$ROUTER_PID" 2>/dev/null || { echo "router died"; exit 1; }
  sleep 3
done
echo "READY  ->  router :$PORT   backend :$BACKEND"

wait      # 保持前台，Ctrl-C 收掉两个进程
