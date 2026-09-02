#!/usr/bin/env bash
# 纯 SGLang Decode server · DSv4 FP4 · B300 · TP(MoE) + DP-attention · fake-prefill。
# 单 server 直连（无 router）+ --load-balance-method follow_bootstrap_room。
#
# 为什么不用 dp-aware router：disaggregation-decode + DP-attention 下，8 个 dp rank
# 每步要锁步做 MLP/MoE sync 的 all_gather；dp-aware router 把请求定向到单个 dp rank，
# 会打破这个跨 rank 同步 → all_gather 崩溃 → 整个 backend 级联挂掉（实测）。
# 正确做法（README 推荐）：run_case --fake-prefill 给每 lane 设 bootstrap_room=lane，
# 内置 follow_bootstrap_room 按 room % dp_size 均匀派发并由 DP 控制器正确协调锁步。
#
# 用于纯 Decode kernel trace：fake transfer 直接给 128K 分配 KV 并立即标记 transfer 完成，
# 不做真实 Prefill、不搬真实 KV。输出 token 无正确性意义，但稳态 Decode 的 batch/KV
# length/attention/MoE/MTP kernel shape 可用于性能 profiling。MTP(EAGLE) 保留。
#
# 用法: MODEL_PATH=/data/models/DeepSeek-V4-Pro TP=8 CONC=128 PORT=8888 bash server_tpdp_fakeprefill.sh
# 客户端 run_case.sh ... --fake-prefill，URL 直接指向本 server（PORT，不经 router）。
set -euo pipefail

MODEL="${MODEL:-deepseek-ai/DeepSeek-V4-Pro}"       # 对外 served-model-name
MODEL_PATH="${MODEL_PATH:-/data/models/DeepSeek-V4-Pro}"
TP="${TP:-8}"
CONC="${CONC:-128}"
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
CHUNK=$((16384 * 8))   # chunked-prefill-size = 16384*8 = 131072

exec python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" --served-model-name "$MODEL" \
  --host 0.0.0.0 --port "$PORT" --trust-remote-code \
  --tp "$TP" --dp "$TP" \
  --dist-init-addr "127.0.0.1:$((PORT + 2000))" \
  --enable-dp-attention --enable-dp-attention-local-control-broadcast \
  --load-balance-method follow_bootstrap_room \
  --moe-runner-backend flashinfer_mxfp4 --enable-deepseek-v4-fp4-indexer --disable-flashinfer-autotune \
  --mem-fraction-static 0.88 --swa-full-tokens-ratio 0.1 \
  --max-running-requests "$MAX_RUNNING" --cuda-graph-max-bs "$CGBS" \
  --allow-auto-truncate --chunked-prefill-size "$CHUNK" \
  --tool-call-parser deepseekv4 --reasoning-parser deepseek-v4 \
  --watchdog-timeout 1800 \
  --speculative-algorithm EAGLE --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --attention-backend compressed --page-size 256 --disable-shared-experts-fusion \
  --enable-metrics \
  --disaggregation-mode decode --disaggregation-transfer-backend fake \
  --disable-radix-cache \
  ${CHAT_TEMPLATE:+--chat-template "$CHAT_TEMPLATE"}
