#!/usr/bin/env bash
# ============================================================================
# DeepSeek-V4-Pro FP4 · B300 · SGLang · AgentX · DEP8 路 (dp-attn + hicache)
# 从 benchmarks/single_node/agentic/dsv4_fp4_b300_sglang_mtp.sh 抽出的
# dp-attn=true 分支，写死、无分支。对应官方 recipe 的第二条 arm：
#   { tp: 8, ep: 8, dp-attn: true, kv-offloading: dram, kv-offload-backend: hicache,
#     spec-decoding: mtp, conc-list: [32,64,128,256,384,512,576], router: sglang-router }
# 特点：DP-attention + MegaMoE + prefill-decode-interval 20（#2701 +28%）
#       + HiCache 主机 KV 层 + sglang_router（带重试）。就是你 conc384/128 跑的这条。
#
# 需要的 env：MODEL TP CONC EP_SIZE RESULT_DIR DURATION
#            KV_OFFLOADING=dram KV_OFFLOAD_BACKEND=hicache TOTAL_CPU_DRAM_GB
# 可选覆盖：HICACHE_RATIO（默认 3，你这台 RAM 小建议 1~2）
# ============================================================================
set -eo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERENCEX_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"          # local/b300 -> repo root
export INFMAX_CONTAINER_WORKSPACE="${INFMAX_CONTAINER_WORKSPACE:-$INFERENCEX_ROOT}"
source "$INFERENCEX_ROOT/benchmarks/benchmark_lib.sh"

export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="sglang:"
export KV_OFFLOADING="${KV_OFFLOADING:-dram}"
export KV_OFFLOAD_BACKEND="${KV_OFFLOAD_BACKEND:-hicache}"
export DP_ATTENTION="true"
check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE

# ---- 模型权重 ----
if [[ -n "${MODEL_PATH:-}" ]]; then
    [[ -d "$MODEL_PATH" && -n "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]] || hf download "$MODEL" --local-dir "$MODEL_PATH"
else
    hf download "$MODEL"; export MODEL_PATH="$MODEL"
fi
nvidia-smi

# ---- AgentX trace + 依赖（隔离 venv）----
resolve_trace_source
SGLANG_PYTHON="$(command -v python3)"
if [[ "$INFMAX_CONTAINER_WORKSPACE" != /workspace ]]; then
    AGENTIC_VENV="${AGENTIC_VENV:-/tmp/inferencex-agentic-venv}"
    "$SGLANG_PYTHON" -m venv "$AGENTIC_VENV"
    export PATH="$AGENTIC_VENV/bin:$PATH"
fi
install_agentic_deps

SERVER_LOG="$RESULT_DIR/server.log"
ROUTER_LOG="$RESULT_DIR/router.log"
mkdir -p "$RESULT_DIR"

# ---- 引擎公共 env ----
export SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
export SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS=1
export PYTHONNOUSERSITE=1
export TORCH_CUDA_ARCH_LIST=10.0
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
export SGLANG_TIMEOUT_KEEP_ALIVE=900
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
export SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT=1
export SGLANG_OPT_USE_JIT_NORM=1
export SGLANG_OPT_USE_JIT_INDEXER_METADATA=1
export SGLANG_OPT_USE_TOPK_V2=1
export SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=1
# MegaMoE 的 FP4/MXF4 激活路径是 opt-in，光 --moe-a2a-backend megamoe 跑的是另一套 kernel
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS=1
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND=1
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320   # 覆盖 per-rank prefill 8192
if [ "${EVAL_ONLY:-false}" != "true" ]; then
    export SGLANG_SIMULATE_ACC_LEN=2.49
    export SGLANG_SIMULATE_ACC_METHOD=match-expected
    export SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
fi
export AIPERF_HTTP_X_SMG_ROUTING_KEY_FROM_CORRELATION_ID=true      # router 一致性哈希用
TRITON_PTXAS_PATH=$(find /usr/local/cuda* /usr/local/lib/python*/dist-packages/nvidia \
    /usr/local/lib/python*/site-packages/nvidia -type f -name ptxas -perm -u+x -print -quit 2>/dev/null || true)
[ -n "$TRITON_PTXAS_PATH" ] && export TRITON_PTXAS_PATH

# ---- HiCache 主机 KV 层（DEP8 专属）----
if [ "$TP" -ge 8 ]; then DEFAULT_HICACHE_RATIO=3; else DEFAULT_HICACHE_RATIO=8; fi
HICACHE_RATIO="${HICACHE_RATIO:-$DEFAULT_HICACHE_RATIO}"
if [ "$HICACHE_RATIO" -gt "$DEFAULT_HICACHE_RATIO" ]; then
    echo "Error: HICACHE_RATIO=$HICACHE_RATIO exceeds limit $DEFAULT_HICACHE_RATIO" >&2; exit 1
fi
CACHE_ARGS=(
    --enable-hierarchical-cache
    --hicache-ratio "$HICACHE_RATIO"
    --hicache-write-policy "${HICACHE_WRITE_POLICY:-write_back}"
    --hicache-io-backend "${HICACHE_IO_BACKEND:-direct}"
    --hicache-mem-layout "${HICACHE_MEM_LAYOUT:-page_first_direct}"
)
WARMUP_ARGS=(--skip-server-warmup)        # AgentX 自己做代表性 warmup，跳过 SGLang 的

# ---- DEP8 专属取值（写死）----
SGLANG_BACKEND_PORT=$((PORT + 1))         # backend 在 PORT+1，router 占 PORT
PARALLEL_ARGS=(
    --tp "$TP"
    --dp "$TP"
    --tokenizer-worker-num "$TP"
    --enable-prefill-delayer
    --prefill-decode-interval 20          # ← #2701 +28% 的关键：破 DP 全局 decode 同步
    --enable-dp-attention
    --enable-dp-attention-local-control-broadcast
    --incremental-streaming-output
    --stream-interval 20
    --dist-init-addr "127.0.0.1:$((PORT + 2000))"
    --ep-size "$EP_SIZE"
    --moe-a2a-backend megamoe
    --enable-deepseek-v4-fp4-indexer
    --disable-flashinfer-autotune
)
# mem-fraction 随并发下调（MegaMoE 瞬时 workspace 在静态分配之外，headroom 要随 batch 增长）
MEM_FRACTION_STATIC=0.93
if   [ "$CONC" -ge 512 ]; then MEM_FRACTION_STATIC=0.86
elif [ "$CONC" -ge 384 ]; then MEM_FRACTION_STATIC=0.88
elif [ "$CONC" -ge 256 ]; then MEM_FRACTION_STATIC=0.9
fi
CHUNKED_PREFILL_SIZE=$((8192 * TP))       # 全局预算被 dp_size(=TP) 除 → 每 rank 8192
SWA_FULL_TOKENS_RATIO=0.075
MAX_RUNNING_REQUESTS=$((2 * CONC))
CUDA_GRAPH_ARGS=(--cuda-graph-max-bs-decode 544)   # 覆盖跨 DP rank 的 padded MTP batch
MODEL_ARGS=(--attention-backend compressed --page-size 256 --disable-shared-experts-fusion)
METRICS_ARGS=(--enable-metrics --enable-cache-report)

# ---- 组装 & 起 backend ----
SGLANG_CMD=(
    "$SGLANG_PYTHON" -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$SGLANG_BACKEND_PORT"
    --trust-remote-code
    "${PARALLEL_ARGS[@]}"
    --mem-fraction-static "$MEM_FRACTION_STATIC"
    --swa-full-tokens-ratio "$SWA_FULL_TOKENS_RATIO"
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    "${CUDA_GRAPH_ARGS[@]}"
    --allow-auto-truncate
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
    --tool-call-parser deepseekv4
    --reasoning-parser deepseek-v4
    --chat-template "$SCRIPT_DIR/../../benchmarks/single_node/chat_templates/deepseek_v4_thinking.jinja"
    --watchdog-timeout 1800
    --speculative-algorithm EAGLE
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
    "${MODEL_ARGS[@]}"
    "${METRICS_ARGS[@]}"
    "${CACHE_ARGS[@]}"
    "${WARMUP_ARGS[@]}"
)
write_command "$RESULT_DIR/sglang_command.txt" "${SGLANG_CMD[@]}"
{ echo "=== SGLANG_* env ==="; env | grep -E '^SGLANG_' | sort; } | tee "$SERVER_LOG"

echo "Starting SGLang backend (DEP8) for B300..."
"${SGLANG_CMD[@]}" >> "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
wait_for_ready --endpoint "http://localhost:$SGLANG_BACKEND_PORT/health" --log "$SERVER_LOG" --pid "$SERVER_PID"

# ---- 起 router（DEP8 专属，带 #2701 重试）----
echo "Starting SGLang router on port $PORT for $TP DP ranks..."
"$SGLANG_PYTHON" -m sglang_router.launch_router \
    --worker-urls "http://localhost:$SGLANG_BACKEND_PORT" \
    --policy consistent_hashing \
    --request-id-headers x-correlation-id \
    --dp-aware \
    --host 0.0.0.0 \
    --port "$PORT" \
    --prometheus-host 127.0.0.1 \
    --prometheus-port "$((PORT + 10000))" \
    --connect-timeout-secs 900 \
    --request-timeout-secs 14400 \
    --disable-health-check \
    --retry-max-retries 8 \
    --retry-initial-backoff-ms 500 \
    --retry-max-backoff-ms 10000 \
    --retry-backoff-multiplier 2 > "$ROUTER_LOG" 2>&1 &
ROUTER_PID=$!
wait_for_ready --endpoint "http://localhost:$PORT/health" --log "$ROUTER_LOG" --pid "$ROUTER_PID"

# ---- 跑回放 / eval（都打 router 的 PORT）----
if [ "${EVAL_ONLY:-false}" = "true" ]; then
    git config --global --add safe.directory "$INFMAX_CONTAINER_WORKSPACE"
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --server-metrics http://localhost:$SGLANG_BACKEND_PORT/metrics"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
