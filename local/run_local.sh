#!/usr/bin/env bash
# Local, CI-free MI355X benchmark driver for InferenceX.
#
# Replaces the CI chain (generate_sweep_configs -> benchmark-tmpl.yml ->
# launch_mi355x-amds.sh). Assumes you are ALREADY inside the recipe's inference
# container (sglang / vllm / atom image). It does NOT manage SLURM, enroot, or
# docker: it just expands a recipe into jobs, exports each job's env-var
# contract, and runs the matching benchmarks/single_node/*/*.sh in-process.
#
# Usage:
#   local/run_local.sh --list
#   local/run_local.sh <recipe-name> [expander flags...]
#   local/run_local.sh dsv4-fp4-mi355x-sglang --max-conc 128
#   local/run_local.sh glm5.2-fp4-mi355x-sglang-agentic-mtp --min-conc 4
#   DRY_RUN=1 local/run_local.sh dsv4-fp4-mi355x-vllm     # print jobs, run nothing
#
# Flags after the recipe name are forwarded to expand_recipe.py
# (--step-size, --min-conc, --max-conc, --total-cpu-dram-gb, --duration, --port).
#
# Env knobs:
#   DRY_RUN=1     expand and print jobs, do not launch
#   RESULT_ROOT   where result JSONs are collected (default: ./local_results)
#   KEEP_GOING=1  continue to the next job if one fails (default: stop)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RESULT_ROOT="${RESULT_ROOT:-$REPO_ROOT/local_results}"

# --list is a pass-through to the expander.
if [[ "${1:-}" == "--list" ]]; then
    exec python3 "$SCRIPT_DIR/expand_recipe.py" --list
fi

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <recipe-name> [expander flags]   (or --list)" >&2
    exit 2
fi

RECIPE="$1"; shift

# Fixed-seq-len scripts hardcode --result-dir /workspace/. Make sure it exists
# and, when the repo is not mounted at /workspace, point /workspace results back
# into RESULT_ROOT so nothing is silently lost.
mkdir -p "$RESULT_ROOT" /workspace/results 2>/dev/null || true

echo "== expanding recipe: $RECIPE =="
mapfile -t JOBS < <(python3 "$SCRIPT_DIR/expand_recipe.py" "$RECIPE" "$@")
rc=$?
if [[ $rc -ne 0 || ${#JOBS[@]} -eq 0 ]]; then
    echo "expander failed or produced no jobs" >&2
    exit 1
fi
echo "== ${#JOBS[@]} job(s) =="

job_num=0
fail=0
for line in "${JOBS[@]}"; do
    job_num=$((job_num + 1))
    SCRIPT=""
    # Reset the full contract each job so a prior arm never leaks (e.g. an
    # agentic KV_OFFLOADING bleeding into a fixed-seq-len job).
    unset MODEL MODEL_PREFIX PRECISION FRAMEWORK IMAGE EXP_NAME ISL OSL \
          MAX_MODEL_LEN TP PP_SIZE DCP_SIZE PCP_SIZE EP_SIZE DP_ATTENTION \
          CONC SPEC_DECODING RANDOM_RANGE_RATIO DISAGG RUN_EVAL EVAL_ONLY \
          RESULT_DIR RESULT_FILENAME PORT GPU_COUNT SCENARIO_TYPE \
          SCENARIO_SUBDIR IS_AGENTIC KV_OFFLOADING KV_OFFLOAD_BACKEND \
          KV_OFFLOAD_BACKEND_METADATA TOTAL_CPU_DRAM_GB DURATION 2>/dev/null

    # Split the tab-separated KEY=VALUE fields and export them.
    IFS=$'\t' read -r -a FIELDS <<< "$line"
    for kv in "${FIELDS[@]}"; do
        key="${kv%%=*}"
        val="${kv#*=}"
        if [[ "$key" == "__SCRIPT__" ]]; then
            SCRIPT="$val"
        else
            export "$key=$val"
        fi
    done

    if [[ -z "$SCRIPT" || ! -f "$SCRIPT" ]]; then
        echo "[$job_num/${#JOBS[@]}] ERROR: benchmark script not found: '$SCRIPT'" >&2
        fail=1
        [[ "${KEEP_GOING:-0}" == "1" ]] && continue || exit 1
    fi

    echo ""
    echo "===================================================================="
    echo "[$job_num/${#JOBS[@]}] $SCRIPT"
    echo "  MODEL=$MODEL  TP=$TP  EP=$EP_SIZE  DPA=${DP_ATTENTION:-}  CONC=$CONC  SPEC=${SPEC_DECODING:-none}"
    [[ -n "${KV_OFFLOADING:-}" ]] && echo "  KV_OFFLOADING=$KV_OFFLOADING  BACKEND=${KV_OFFLOAD_BACKEND:-}  DRAM_GB=${TOTAL_CPU_DRAM_GB:-}"
    echo "  RESULT_FILENAME=$RESULT_FILENAME"
    echo "===================================================================="

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        continue
    fi

    bash "$SCRIPT"
    job_rc=$?
    if [[ $job_rc -ne 0 ]]; then
        echo "[$job_num/${#JOBS[@]}] FAILED (rc=$job_rc)" >&2
        fail=1
        if [[ "${KEEP_GOING:-0}" != "1" ]]; then
            echo "stopping (set KEEP_GOING=1 to continue past failures)" >&2
            exit $job_rc
        fi
    fi

    # Collect result JSONs written to /workspace/ (fixed) or RESULT_DIR (agentic).
    for src in /workspace "${RESULT_DIR:-/workspace/results}"; do
        [[ -d "$src" ]] || continue
        find "$src" -maxdepth 2 -name "${RESULT_FILENAME}*.json" -type f 2>/dev/null \
            | while read -r f; do
                cp -f "$f" "$RESULT_ROOT/" 2>/dev/null && \
                    echo "  collected $(basename "$f") -> $RESULT_ROOT/"
            done
    done
done

echo ""
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "== DRY_RUN: no jobs executed =="
else
    echo "== done. results in $RESULT_ROOT =="
fi
exit $fail
