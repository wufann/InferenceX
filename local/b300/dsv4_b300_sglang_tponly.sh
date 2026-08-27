#!/usr/bin/env bash
# ============================================================================
# DeepSeek-V4-Pro FP4 · B300 · SGLang · AgentX · TP-ONLY 路
# 从 benchmarks/single_node/agentic/dsv4_fp4_b300_sglang_mtp.sh 抽出的
# dp-attn=false 分支，写死、无分支。对应官方 recipe 的第一条 arm：
#   { tp: 8, kv-offloading: none, spec-decoding: mtp, conc-list: [1,4,8,16,32] }
# 特点：纯 TP，无 dp/ep/megamoe/hicache/router，flashinfer_mxfp4 MoE。
#
# 需要的 env：MODEL TP CONC RESULT_DIR DURATION (KV_OFFLOADING=none)
# ============================================================================
set -eo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERENCEX_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"          # local/b300 -> repo root
export INFMAX_CONTAINER_WORKSPACE="${INFMAX_CONTAINER_WORKSPACE:-$INFERENCEX_ROOT}"
source "$INFERENCEX_ROOT/benchmarks/benchmark_lib.sh"

export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="sglang:"
export KV_OFFLOADING="${KV_OFFLOADING:-none}"              # TP-only 不用 offload
export EP_SIZE="${EP_SIZE:-1}"
export DP_ATTENTION="false"
check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION

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
if [ "${EVAL_ONLY:-false}" != "true" ]; then
    export SGLANG_SIMULATE_ACC_LEN=2.49
    export SGLANG_SIMULATE_ACC_METHOD=match-expected
    export SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
fi
TRITON_PTXAS_PATH=$(find /usr/local/cuda* /usr/local/lib/python*/dist-packages/nvidia \
    /usr/local/lib/python*/site-packages/nvidia -type f -name ptxas -perm -u+x -print -quit 2>/dev/null || true)
[ -n "$TRITON_PTXAS_PATH" ] && export TRITON_PTXAS_PATH

# ---- TP-only 专属取值（写死，无分支）----
SGLANG_BACKEND_PORT="$PORT"                 # 无 router，backend 直接在 PORT
PARALLEL_ARGS=(
    --tp "$TP"
    --moe-runner-backend flashinfer_mxfp4
    --disable-flashinfer-autotune
)
MEM_FRACTION_STATIC=0.88
CHUNKED_PREFILL_SIZE=8192
SWA_FULL_TOKENS_RATIO=0.1
MAX_RUNNING_REQUESTS=$((2 * CONC))          # AgentX 按会话树计，留 subagent fan-out
CUDA_GRAPH_MAX_BS=$((CONC * 4)); [ "$CUDA_GRAPH_MAX_BS" -gt 64 ] && CUDA_GRAPH_MAX_BS=64
CUDA_GRAPH_ARGS=(--cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS")
MODEL_ARGS=(--attention-backend compressed --page-size 256 --disable-shared-experts-fusion)
METRICS_ARGS=(--enable-metrics --enable-cache-report)

# ---- 组装 & 起 server ----
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
    # TP-only：无 CACHE_ARGS / WARMUP_ARGS
)
write_command "$RESULT_DIR/sglang_command.txt" "${SGLANG_CMD[@]}"
{ echo "=== SGLANG_* env ==="; env | grep -E '^SGLANG_' | sort; } | tee "$SERVER_LOG"

echo "Starting SGLang server (TP-only) for B300..."
"${SGLANG_CMD[@]}" >> "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
wait_for_ready --endpoint "http://localhost:$SGLANG_BACKEND_PORT/health" --log "$SERVER_LOG" --pid "$SERVER_PID"

# ---- 跑回放 / eval ----
if [ "${EVAL_ONLY:-false}" = "true" ]; then
    git config --global --add safe.directory "$INFMAX_CONTAINER_WORKSPACE"
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --server-metrics http://localhost:$SGLANG_BACKEND_PORT/metrics"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
