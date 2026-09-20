#!/usr/bin/env bash
set -euo pipefail
set -x

# AgentX trace replay for Qwen3.8-Flash-Next MXFP4 on MI355X (gfx950) using
# SGLang with native EAGLE MTP. Throughput uses the committed golden synthetic
# acceptance length; evaluation retains real target-model verification.
#
# This is the AMD-native FP4 arm: the checkpoint is the Quark OCP-MXFP4 quant
# (Qwen3.8-Flash-Next-Quark-MXFP4, per_group group_size 32, e8m0 shared scale),
# the microscaling counterpart to the NVIDIA arms' NVFP4/modelopt checkpoint.
# SGLang reads the quant scheme from the checkpoint, so no --quantization flag.
#
# Topology is single-GPU TP1/EP1, the verified serving config for this model on
# this node. (Plain TP would slice the MoE gate/up output dim 640/TP; MXFP4's
# group_size is 32, so TP1/2/4 divide cleanly but TP8 -> 80 is not a multiple of
# 32. TP1 sidesteps this and every rank-crossing collective disappears; 288 GB
# of MI355X HBM holds the ~90 GiB MXFP4 weights plus the conc 8..16 KV pools.)
#
# Launch flags are the verified single-node serving config for this exact model
# on this node (aiter attention + aiter MoE runner, page-size 32, kv-cache-dtype
# auto, EAGLE 3-step MTP) -- identical to the FP8 arm's server config except the
# checkpoint -- extended with the agentic framework's env contract, trace
# replay, metrics, and golden-AL simulation. The NVIDIA arms' flashinfer
# linear-attn backends and --mamba-ssm-dtype gymnastics do NOT apply on
# ROCm/aiter and are intentionally omitted.
#
# Required env vars:
#   MODEL, TP, CONC, EP_SIZE, KV_OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR, DURATION
# KV_OFFLOADING=dram requires KV_OFFLOAD_BACKEND=hicache.

source "$(dirname "$0")/../../benchmark_lib.sh"

# Lightweight GSM8K eval instead of the AgentX SWE-bench default (EVAL_ONLY).
export EVAL_FRAMEWORK="lm-eval"

check_env_vars \
    MODEL TP CONC EP_SIZE KV_OFFLOADING \
    TOTAL_CPU_DRAM_GB RESULT_DIR DURATION

SCHEDULER_RECV_INTERVAL=${SCHEDULER_RECV_INTERVAL:-30}

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

# ROCR/HIP visibility under slurm cgroups.
if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

# `hf download` is idempotent and creates the target dir. When MODEL_PATH points
# at a populated local checkpoint dir (the usual MI355X case -- the Quark MXFP4
# checkpoint is local-only), skip the network.
if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    if [[ -d "$MODEL" ]]; then
        export MODEL_PATH="$MODEL"
    else
        hf download "$MODEL"
        export MODEL_PATH="$MODEL"
    fi
fi
rocm-smi || true
amd-smi || true

# A server killed on this node moments earlier (a previous conc point in the
# sweep) can still be draining its HBM: KFD reclaim takes time, and launching
# into a half-drained GPU fails RCCL/HIP init with 'unhandled cuda error' /
# 'invalid argument'. Wait for the GPU(s) to come back before launching. Idle
# nodes hold a small driver/firmware VRAM baseline (~4%); a draining or occupied
# GPU sits far higher. Require every GPU <= 10%, up to 15 min.
GPU_CLEAN=false
for i in $(seq 1 90); do
    VRAM_MAX=$(rocm-smi --showmemuse 2>/dev/null | grep -oE "GPU Memory Allocated \(VRAM%\): [0-9]+" | awk '{if ($NF > m) m = $NF} END {print m+0}')
    if [ "${VRAM_MAX:-0}" -le 10 ]; then echo "GPUs clean (vram%max=$VRAM_MAX after $((i*10))s)"; GPU_CLEAN=true; break; fi
    echo "waiting for prior-job GPU memory reclaim: vram%max=$VRAM_MAX"; sleep 10
done
[ "$GPU_CLEAN" = "true" ] || { echo "Error: GPUs still draining prior job's memory after 15min" >&2; exit 1; }

export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126_256k
resolve_trace_source
install_agentic_deps

export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="sglang:"

SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

SERVER_PID=""
cleanup_agentic_services() {
    local exit_code=$?
    trap - EXIT INT TERM
    set +e
    stop_background_process_tree "$SERVER_PID" "SGLang server" 60
    exit "$exit_code"
}
trap cleanup_agentic_services EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ---- HiCache (host DRAM KV tier) -- only when KV_OFFLOADING=dram -------------
CACHE_ARGS=()
if require_agentic_kv_offload_backend hicache; then
    HICACHE_RATIO="${HICACHE_RATIO:-1.5}"
    HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through}"
    HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-direct}"
    HICACHE_MEM_LAYOUT="${HICACHE_MEM_LAYOUT:-page_first_direct}"
    echo "HiCache CPU tier: ratio=$HICACHE_RATIO, write_policy=$HICACHE_WRITE_POLICY, io_backend=$HICACHE_IO_BACKEND, mem_layout=$HICACHE_MEM_LAYOUT, dram_budget=${TOTAL_CPU_DRAM_GB} GB, tp=$TP"
    CACHE_ARGS=(
        --enable-hierarchical-cache
        --hicache-ratio "$HICACHE_RATIO"
        --hicache-write-policy "$HICACHE_WRITE_POLICY"
        --hicache-io-backend "$HICACHE_IO_BACKEND"
        --hicache-mem-layout "$HICACHE_MEM_LAYOUT"
    )
fi

PARALLEL_ARGS=(
    --tp "$TP"
    --dp 1
    --ep-size "$EP_SIZE"
)

# TP1 stays on the single-worker default; TP>=4 would need parallel tokenization.
TOKENIZER_ARGS=()
if [ "$TP" -ge 4 ]; then
    TOKENIZER_ARGS=(--tokenizer-worker-num 6)
fi

# AgentX concurrency counts live session trees, not individual HTTP requests;
# subagent fan-out can burst instantaneous request concurrency above CONC, so
# leave 2x headroom. EAGLE silently caps --max-running-requests at 48 when it is
# unset, so keep it explicit and sized to the concurrency.
MAX_RUNNING_REQUESTS=$((2 * CONC))
CUDA_GRAPH_MAX_BS=$MAX_RUNNING_REQUESTS
[ "$CUDA_GRAPH_MAX_BS" -gt 64 ] && CUDA_GRAPH_MAX_BS=64

MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.9}"

export PYTHONNOUSERSITE=1
# AIPerf pins one pooled keep-alive connection per session; outlast its 300s
# client pool so bursty inter-turn gaps cannot reuse a socket as it closes.
export SGLANG_TIMEOUT_KEEP_ALIVE=1800
# Honor the explicit --mem-fraction-static below instead of letting aiter recompute it.
export SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION=1

# Throughput runs pin acceptance to the committed golden AL for this model,
# thinking mode, and draft length so submissions compare on system performance
# at a fixed acceptance target rather than draft-head quality. 2.32 is the
# qwen3.8-flash-next thinking_on curve at num_speculative_tokens=3
# (golden_al_distribution/qwen3.8next_mtp.yaml, measured on the FP8 target).
# No MXFP4-specific AL is committed; the FP8 value is the closest proxy since
# the MTP draft head is shared and only the target logits shift slightly.
# EVAL_ONLY leaves this off and keeps real target verification.
if [ "${EVAL_ONLY:-false}" != "true" ]; then
    export SGLANG_SIMULATE_ACC_LEN=2.32
    export SGLANG_SIMULATE_ACC_METHOD=match-expected
    export SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
fi

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    "${PARALLEL_ARGS[@]}"
    # Verified single-node serving flags for Qwen3.8-Flash-Next MXFP4 on MI355X.
    --attention-backend aiter
    --moe-runner-backend aiter
    --page-size 32
    --kv-cache-dtype auto
    --chunked-prefill-size 16384
    --watchdog-timeout 1200
    --mem-fraction-static "$MEM_FRACTION_STATIC"
    --model-loader-extra-config '{"enable_multithread_load": true}'
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --scheduler-recv-interval "$SCHEDULER_RECV_INTERVAL"
    --stream-interval 50
    "${TOKENIZER_ARGS[@]}"
    --tokenizer-path "$MODEL_PATH"
    # Reasoning parser on auto, matching the NVIDIA same-model arms (B300/H200/H100).
    # The verified server.sh/server_fp4.sh set no parser at all.
    --reasoning-parser auto
    # Native MTP through the EAGLE spec path (eagle-topk 1 = single MTP chain);
    # 3 steps / 4 draft tokens = 3 speculative tokens per verification step.
    --speculative-algorithm EAGLE
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
    --enable-metrics
    --enable-cache-report
    "${CACHE_ARGS[@]}"
)

printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"
printf '\n' | tee -a "$RESULT_DIR/sglang_command.txt"
"${SGLANG_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "${EVAL_ONLY:-false}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --apply-chat-template"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
