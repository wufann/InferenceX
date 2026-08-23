#!/usr/bin/env bash
# Aggregate a finished agentic run's raw artifacts into the InferenceX agg JSON,
# then print the headline metrics. Wraps:
#   cd <repo-root>
#   <env from the run> python3 -m utils.agentic.aggregation.process_agentic_result
#   python3 local/show_agentic.py <agg.json>
#
# It auto-derives the run's metadata from the command files the benchmark left
# in RESULT_DIR (benchmark_command.txt = aiperf, *_command.txt = server), so you
# don't have to hand-type CONC/MODEL/TP/EP/KV/spec. Every derived value can be
# overridden by exporting the matching env var before calling.
#
# Usage:
#   local/aggregate_agentic.sh [RESULT_DIR]      # default: /workspace/results
#   TP=8 PRECISION=fp4 local/aggregate_agentic.sh /workspace/results
#
# Must be run from inside the serving container (or anywhere the repo + RESULT_DIR
# are visible). It cd's to the repo root itself, so `python -m utils...` resolves.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

RESULT_DIR="${1:-${RESULT_DIR:-/workspace/results}}"
if [[ ! -d "$RESULT_DIR" ]]; then
    echo "ERROR: RESULT_DIR '$RESULT_DIR' not found" >&2
    exit 1
fi

ARTIFACTS="$RESULT_DIR/aiperf_artifacts"
if [[ ! -f "$ARTIFACTS/profile_export.jsonl" ]]; then
    echo "ERROR: $ARTIFACTS/profile_export.jsonl not found — is this a finished agentic run?" >&2
    exit 1
fi

BENCH_CMD="$RESULT_DIR/benchmark_command.txt"   # aiperf profile ... (always present)
# Server command file: sglang_command.txt / vllm_command.txt / atom... — anything
# that is not the aiperf benchmark command.
SRV_CMD="$(ls "$RESULT_DIR"/*_command.txt 2>/dev/null | grep -v benchmark_command.txt | head -1)"

# ---- derive from aiperf benchmark_command.txt ----
CONC="${CONC:-$(grep -oE -- '--concurrency +[0-9]+' "$BENCH_CMD" 2>/dev/null | head -1 | grep -oE '[0-9]+$')}"
MODEL="${MODEL:-$(grep -oE -- '--model +[^ ]+' "$BENCH_CMD" 2>/dev/null | head -1 | awk '{print $2}')}"

# ---- derive from the server command file ----
FRAMEWORK="${FRAMEWORK:-}"
if [[ -z "$FRAMEWORK" && -n "$SRV_CMD" ]]; then
    case "$(basename "$SRV_CMD")" in
        sglang_command.txt) FRAMEWORK=sglang ;;
        vllm_command.txt)   FRAMEWORK=vllm ;;
        *)                  FRAMEWORK=sglang ;;
    esac
fi
FRAMEWORK="${FRAMEWORK:-sglang}"

_tp_from_srv() {
    [[ -n "$SRV_CMD" ]] || return
    grep -oE -- '--(tp|tensor-parallel-size|tp-size) +[0-9]+' "$SRV_CMD" 2>/dev/null | head -1 | grep -oE '[0-9]+$'
}
TP="${TP:-$(_tp_from_srv)}"
TP="${TP:-1}"
EP_SIZE="${EP_SIZE:-$( [[ -n "$SRV_CMD" ]] && grep -oE -- '--ep-size +[0-9]+' "$SRV_CMD" 2>/dev/null | head -1 | grep -oE '[0-9]+$')}"
EP_SIZE="${EP_SIZE:-1}"

if [[ -z "${DP_ATTENTION:-}" ]]; then
    if [[ -n "$SRV_CMD" ]] && grep -q -- '--enable-dp-attention' "$SRV_CMD" 2>/dev/null; then
        DP_ATTENTION=true
    else
        DP_ATTENTION=false
    fi
fi

# spec-decoding: EAGLE/MTP flag in the server command
if [[ -z "${SPEC_DECODING:-}" ]]; then
    if [[ -n "$SRV_CMD" ]] && grep -qiE -- '--speculative-algorithm|--speculative-config|speculative-num' "$SRV_CMD" 2>/dev/null; then
        SPEC_DECODING=mtp
    else
        SPEC_DECODING=none
    fi
fi

# KV offloading: hierarchical cache => dram; backend hicache unless mooncake set
if [[ -z "${KV_OFFLOADING:-}" ]]; then
    if [[ -n "$SRV_CMD" ]] && grep -q -- '--enable-hierarchical-cache' "$SRV_CMD" 2>/dev/null; then
        KV_OFFLOADING=dram
        if grep -q -- '--hicache-storage-backend mooncake' "$SRV_CMD" 2>/dev/null; then
            KV_OFFLOAD_BACKEND="${KV_OFFLOAD_BACKEND:-mooncake}"
        else
            KV_OFFLOAD_BACKEND="${KV_OFFLOAD_BACKEND:-hicache}"
        fi
    else
        KV_OFFLOADING=none
        KV_OFFLOAD_BACKEND="${KV_OFFLOAD_BACKEND:-}"
    fi
fi
KV_OFFLOAD_BACKEND="${KV_OFFLOAD_BACKEND:-}"

# model-prefix from the model id
if [[ -z "${MODEL_PREFIX:-}" ]]; then
    case "$MODEL" in
        *DeepSeek-V4*) MODEL_PREFIX=dsv4 ;;
        *Kimi-K3*)     MODEL_PREFIX=kimik3 ;;
        *GLM-5.2*)     MODEL_PREFIX=glm5.2 ;;
        *DeepSeek-R1*) MODEL_PREFIX=dsr1 ;;
        *)             MODEL_PREFIX="${MODEL##*/}" ;;
    esac
fi

PRECISION="${PRECISION:-fp4}"
RUNNER_TYPE="${RUNNER_TYPE:-mi355x}"
DISAGG="${DISAGG:-false}"
IMAGE="${IMAGE:-}"

# Build the result filename the same way local/expand_recipe.py does, unless
# the caller pinned RESULT_FILENAME explicitly.
if [[ -z "${RESULT_FILENAME:-}" ]]; then
    if [[ "$KV_OFFLOADING" == "none" ]]; then kv_tag="kvnone"; else kv_tag="kv${KV_OFFLOADING}-${KV_OFFLOAD_BACKEND}"; fi
    RESULT_FILENAME="${MODEL_PREFIX}_tp${TP}_conc${CONC}_${kv_tag}"
    [[ "$SPEC_DECODING" != "none" ]] && RESULT_FILENAME+="_spec-${SPEC_DECODING}"
    RESULT_FILENAME+="_${PRECISION}_${FRAMEWORK}"
fi

# process_agentic_result requires KV_OFFLOAD_BACKEND_METADATA.name to match
# KV_OFFLOAD_BACKEND when offloading is on; synthesize it if not provided.
if [[ "$KV_OFFLOADING" != "none" && -z "${KV_OFFLOAD_BACKEND_METADATA:-}" ]]; then
    KV_OFFLOAD_BACKEND_METADATA="{\"name\":\"${KV_OFFLOAD_BACKEND}\",\"version\":\"0\"}"
fi

AGENTIC_OUTPUT_DIR="${AGENTIC_OUTPUT_DIR:-$RESULT_DIR}"

echo "== derived run metadata =="
printf '  %-22s %s\n' RESULT_DIR "$RESULT_DIR" MODEL "$MODEL" MODEL_PREFIX "$MODEL_PREFIX" \
    FRAMEWORK "$FRAMEWORK" PRECISION "$PRECISION" TP "$TP" EP_SIZE "$EP_SIZE" \
    DP_ATTENTION "$DP_ATTENTION" CONC "$CONC" SPEC_DECODING "$SPEC_DECODING" \
    KV_OFFLOADING "$KV_OFFLOADING" KV_OFFLOAD_BACKEND "${KV_OFFLOAD_BACKEND:-<none>}" \
    RESULT_FILENAME "$RESULT_FILENAME" OUTPUT_DIR "$AGENTIC_OUTPUT_DIR"

if [[ -z "$CONC" || -z "$MODEL" ]]; then
    echo "ERROR: could not derive CONC/MODEL from $BENCH_CMD; export them and retry" >&2
    exit 1
fi

export RESULT_DIR AGENTIC_OUTPUT_DIR RESULT_FILENAME RUNNER_TYPE MODEL MODEL_PREFIX \
    FRAMEWORK PRECISION SPEC_DECODING DISAGG TP EP_SIZE DP_ATTENTION CONC \
    KV_OFFLOADING KV_OFFLOAD_BACKEND IMAGE
[[ -n "${KV_OFFLOAD_BACKEND_METADATA:-}" ]] && export KV_OFFLOAD_BACKEND_METADATA
[[ -n "${TOTAL_CPU_DRAM_GB:-}" ]] && export TOTAL_CPU_DRAM_GB

echo ""
echo "== aggregating (cd $REPO_ROOT) =="
( cd "$REPO_ROOT" && python3 -m utils.agentic.aggregation.process_agentic_result )
rc=$?
if [[ $rc -ne 0 ]]; then
    echo "aggregation failed (rc=$rc)" >&2
    exit $rc
fi

AGG="$AGENTIC_OUTPUT_DIR/$RESULT_FILENAME.json"
echo ""
echo "== metrics ($AGG) =="
python3 "$SCRIPT_DIR/show_agentic.py" "$AGG"
