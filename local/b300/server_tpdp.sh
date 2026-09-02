#!/usr/bin/env bash
# 纯 SGLang server · DSv4 FP4 · B300 · TP(MoE) + DP-attention · prefill。
# 单 server 直连（无 router）+ --load-balance-method follow_bootstrap_room。
#
# 为什么不用 dp-aware router：DP-attention 下 8 个 dp rank 每步要锁步做 MLP/MoE sync 的
# all_gather；dp-aware router 把请求定向到单个 dp rank，高并发下打破跨 rank 同步 →
# scheduler 崩溃（prefill/decode 都会崩，实测）。
# 为什么不用 follow_bootstrap_room：它在非 disaggregation server 上对任何没有
# bootstrap_room 的请求直接 assert → SIGQUIT 杀整个 server；SGLang 自己的 /health
# 内部 generate 就没有 room，一探测就死（实测 dp_ctrl:778）。
# prefill 的 cache 亲和改用 round_robin + 显式 routed_dp_rank：客户端给同一 lane 的
# warmup+formal 注入相同 routed_dp_rank=lane%dp_size（run_case 的 INJECT_BOOTSTRAP_ROOM=1
# → select_concurrency --inject-bootstrap-room 现改为注入 routed_dp_rank），
# data_parallel_controller.maybe_external_dp_rank_routing 直接路由到该 rank → 同 rank
# 命中 cache；而无 rank 的 /health/内部 warmup 走 round_robin 正常处理、不 assert。
# radix cache 保持开启（prefill 需要 warmup 填 cache、formal 命中）。
#
# 用法: MODEL_PATH=/data/models/DeepSeek-V4-Pro TP=8 CONC=128 PORT=8888 bash server_tpdp.sh
# 客户端 aiperf 直接打 PORT；profiler 控制也打 PORT（单 server，无 8889）。
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
# conc128 warmup 边缘 OOM(差 ~160MiB,14GiB reserve 因碎片用不上)。expandable_segments
# 让分配器回收碎片化的 reserved 段,覆盖 prefill 激活峰值，且不减 KV 容量。
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 要抓 profiler trace 就在这里 export（起 server 前设才生效）：
export SGLANG_TORCH_PROFILER_DIR=/data/trace
export SGLANG_PROFILE_RECORD_SHAPES=True
# torch 导出 .gz 时先把未压缩 JSON 写到 TMPDIR（默认 /tmp = 容器根盘，会被写爆）。
# 指到 /data（49T）上，暂存和最终 gz 都落大盘，根分区永不被碰。
export TMPDIR=/data/trace/staging
mkdir -p "$TMPDIR"

MAX_RUNNING=$((2 * CONC))
CGBS=$((CONC * 4)); [ "$CGBS" -gt 64 ] && CGBS=64
CHUNK=$((16384 * 8))   # chunked-prefill-size = 16384*8 = 131072

exec python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" --served-model-name "$MODEL" \
  --host 0.0.0.0 --port "$PORT" --trust-remote-code \
  --tp "$TP" --dp "$TP" \
  --dist-init-addr "127.0.0.1:$((PORT + 2000))" \
  --enable-dp-attention --enable-dp-attention-local-control-broadcast \
  --load-balance-method round_robin \
  --moe-runner-backend flashinfer_mxfp4 --enable-deepseek-v4-fp4-indexer --disable-flashinfer-autotune \
  --mem-fraction-static 0.88 --swa-full-tokens-ratio 0.1 \
  --max-running-requests "$MAX_RUNNING" --cuda-graph-max-bs "$CGBS" \
  --allow-auto-truncate --chunked-prefill-size "$CHUNK" \
  --tool-call-parser deepseekv4 --reasoning-parser deepseek-v4 \
  --watchdog-timeout 1800 \
  --skip-server-warmup \
  --speculative-algorithm EAGLE --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --attention-backend compressed --page-size 256 --disable-shared-experts-fusion \
  --enable-metrics --enable-cache-report \
  ${CHAT_TEMPLATE:+--chat-template "$CHAT_TEMPLATE"}
