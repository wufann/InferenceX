# Run AgentX-Harness Standalone

Run the AgentX client against an already-running OpenAI-compatible server.

## Install

With Git and [uv](https://docs.astral.sh/uv/getting-started/installation/) installed,
clone the harness at the pinned `agentx-v1.0.6` release commit. Its CLI is named `aiperf`.

```bash
git clone --filter=blob:none --no-checkout https://github.com/SemiAnalysisAI/agentx-harness.git
cd agentx-harness
git checkout --detach 89b21867872a5bbc4b0676bf5005c404da5e9f94
uv venv --python 3.11 .venv-agentx
uv pip install --python .venv-agentx/bin/python -e . 'datasets>=4.7.0'
source .venv-agentx/bin/activate
```

## Run

```bash
set -eo pipefail
SERVER_URL="http://127.0.0.1:8000"
SERVED_MODEL_NAME="moonshotai/Kimi-K3"
TOKENIZER="moonshotai/Kimi-K3"
DATASET="semianalysis_cc_traces_weka_062126"
CONC=8
OUTPUT_DIR="$PWD/results/agentx-c${CONC}-$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "$OUTPUT_DIR"
aiperf profile \
  --scenario agentx \
  --url "$SERVER_URL" --endpoint /v1/chat/completions \
  --model "$SERVED_MODEL_NAME" --tokenizer "$TOKENIZER" \
  --tokenizer-trust-remote-code \
  --concurrency "$CONC" \
  --public-dataset "$DATASET" \
  --output-artifact-dir "$OUTPUT_DIR/aiperf_artifacts" \
  2>&1 | tee "$OUTPUT_DIR/aiperf.log"
```

The `agentx` preset supplies a one-hour profile, ten warmup requests per lane,
and shared reporting and runtime settings. Explicit CLI arguments and `AIPERF_*`
environment variables override preset defaults. Model, tokenizer, dataset,
concurrency, and output remain caller inputs; `--tokenizer-trust-remote-code` is
needed only for tokenizers with custom code.

The default random seed is **42**. Keep this seed when reproducing AgentX results,
or pass `--random-seed 123` to use a different seed.

Allow additional time for dataset preparation, warmup, and draining requests.
To sweep, restart the server for every `CONC` value and use a fresh output
directory. Do not flush caches or reuse a live server across concurrency points. Match the target
recipe's duration and warmup settings when reproducing a result.

Add `--server-metrics "${SERVER_URL}/metrics"` if the server exposes metrics.

## Optional session-affinity flags

For DP-attention runs requiring router affinity, export these before
`aiperf profile`. They add headers without replacing `X-Correlation-ID`.

```bash
# Adds X-Dynamo-Session-ID; subagents also get X-Dynamo-Parent-Session-ID.
# Only this option sends the parent ID, preserving forked-agent lineage.
export AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=true
# Also sends X-Session-ID, if a front-end router needs it.
export AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=true
```

## Local Qwen3.8 MI355X sweep and Pareto charts

`python3 -m infx.bench.local_qwen38` runs the Qwen3.8 Quark MXFP4 SGLang recipe
inside one ROCm Docker container, without CI or Slurm. It starts a **fresh server
for every concurrency**, delegates trace replay and aggregation to the same
`infx.bench agentic` client used by CI, and writes local CSV/JSON/SVG charts.
The command is specific to one MI355X GPU, TP=EP=1, native NEXTN MTP, no KV
offload, and the 256k-capped Weka trace corpus. The default chart uses P90 E2E
normalized interactivity (output tokens/s/user) against total tokens/s/GPU;
an output-only throughput chart is also produced. Both axes are higher-is-better.

### 1. Prepare on the Docker host

Run these commands from the **root of the `local_qwen38` InferenceX checkout**.
Set `INFX_MODEL_DIR` to the existing, complete local checkpoint directory (the
directory containing `config.json` and safetensors files). The AIPerf submodule
is pinned by this checkout. The run directory `/results/qwen38-001` must be new
or empty; use a new name for another sweep.

```bash
cd /apps/fanwu103/git/InferenceX
INFX_MODEL_DIR=/absolute/path/to/Qwen3.8-Flash-Next-Quark-MXFP4-PLEFP8
test -f "$INFX_MODEL_DIR/config.json"
git submodule update --init inferencex-e2e/utils/aiperf
mkdir -p "$PWD/local-agentx-results" "$HOME/.cache/huggingface"
```

The container needs `uv` (or network access to install it), Python 3.11 for
AIPerf (which `uv` can fetch), and the packages used by the existing InferenceX
result processor. The first point prepares the AIPerf venv under the mounted
result directory; later points reuse it. Dataset/tokenizer downloads use the
mounted Hugging Face cache. A first run therefore needs network access or a
cache already containing those files.

### 2. Execute the benchmark in Docker

Still on the **Docker host**, run the following command. It mounts the checkout
read-only at `/work/InferenceX`, the model at `/models/qwen38`, and results at
`/results`. `ROCR_VISIBLE_DEVICES=0` selects one MI355X GPU. The Python command
inside the container starts SGLang, waits for `/health` and the served model,
profiles C=1/4/8/12/16 for 3600 seconds **each**, validates each point, and
shuts that server down before starting the next concurrency.

```bash
docker run --rm -it --network host --ipc host --shm-size 64g \
  --device /dev/kfd --device /dev/dri --group-add video \
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
    --output-dir /results/qwen38-001 --concurrencies 1 4 8 12 16 \
    --duration 3600 --port 8000'
```

For a command preview without loading the model, append `--dry-run` to the
container's Python command. A dry run prints all five server commands and
does not write result files. Do not reuse `/results/qwen38-001` for another
full run after it has data; choose a new directory name.

The server flags mirror the MI355X recipe: AITER attention/MoE, BF16 SSM,
NEXTN 3 steps, `max-running-requests=2*CONC`, and graph batch `CONC`. The
Nikolai environment switches are set before each server starts. The MTP
acceptance length is read from the committed `qwen3.8next_mtp.yaml` golden
curve for thinking-on and K=3; it is not supplied by the caller.

### 3. Find results and draw the curves

The `run` command **draws charts automatically after each passing point**. On
the Docker host, open these files after the container finishes:

```text
local-agentx-results/qwen38-001/run_config.json
local-agentx-results/qwen38-001/points/c1/server.log
local-agentx-results/qwen38-001/points/c1/aiperf_artifacts/
local-agentx-results/qwen38-001/aggregates/qwen38_mi355x_c1.json
local-agentx-results/qwen38-001/charts/total_pareto.svg
local-agentx-results/qwen38-001/charts/output_pareto.svg
```

The same per-point layout repeats for C=4/8/12/16. `points/cN/` also contains
`server_command.txt`, `server_env.json`, and benchmark logs. A passing point
gets `validated.ok`. `aggregates/` may retain diagnostic JSON from a failed
point; the plotter only uses points with `validated.ok`. `charts/` includes
`total_points.csv` (all valid measured points), `total_pareto.csv` (the
non-dominated subset), `total_pareto.json`, and the SVG. `output_*` files use
output tokens/s/GPU instead of total tokens/s/GPU on the y-axis. The `x` column
in each CSV is P90 E2E normalized interactivity in output tokens/s/user; `y`
is the named throughput per GPU.

To **regenerate charts from existing aggregates without rerunning the
benchmark**, run this on the Docker host. It starts a short-lived container
without GPU devices and writes updated files into the same result directory:

```bash
docker run --rm \
  -v "$PWD:/work/InferenceX:ro" \
  -v "$PWD/local-agentx-results:/results" \
  -w /work/InferenceX/inferencex-e2e \
  aigmkt/qwen3.8-flash-next:v0.5.21-rocm10-mi35x-20261009-nikolai-opt \
  python3 -m infx.bench.local_qwen38 plot /results/qwen38-001
```

The local frontier retains points that no other measured point exceeds on
both displayed axes. It is a local diagnostic visualization, not an
InferenceX-app ingest or published dashboard result. Compare runs only when
their model, image, trace corpus, duration, GPU count, and client version are
matched. Short runs can be useful for debugging but do not replace the
one-hour AgentX measurement.
