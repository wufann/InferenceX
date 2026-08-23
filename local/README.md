# 本地 MI355X Benchmark 工具（CI-free）

把 InferenceX 的 MI355X 单机 benchmark 从 GitHub Actions CI 里摘出来，做成一套本地可复现的工具。
**前提：你已经在目标 recipe 对应的推理容器内**（sglang / vllm / atom 镜像），本工具不管
SLURM / enroot / docker，只负责「展开搜索空间 → 导出环境变量 → 调用官方 benchmark 脚本 → 收集结果」。

## 它替代了 CI 的哪一段

官方 CI 链路：

```
configs/amd-master.yaml
  → utils/matrix_logic/generate_sweep_configs.py   (展开搜索空间)
  → .github/workflows/benchmark-tmpl.yml           (matrix entry → 环境变量)
  → runners/launch_mi355x-amds.sh                  (salloc/srun + enroot 起容器)
  → benchmarks/single_node/{fixed_seq_len,agentic}/<model>.sh   (起 server + 压测)
```

本工具用两个文件替代中间三步（不含容器编排，因为你已在容器内）：

- `local/expand_recipe.py` —— 解析 `amd-master.yaml` + `runners.yaml`，把一个 recipe 展开成
  一条条 job，算好每个 job 的完整环境变量，并用和 `launch_mi355x-amds.sh` **完全一致**的规则
  解析出要跑的脚本路径（含 `_mtp` / `_atom` / framework 回退逻辑）。
- `local/run_local.sh` —— 遍历 job，导出环境变量，`bash` 对应脚本，收集结果 JSON。

最终跑的仍然是仓库里**未经改动的官方脚本**（`benchmarks/single_node/...`），所以性能口径与官方一致。

## 三个目标模型在 MI355X 上的可用 recipe

| 模型 | recipe | 场景 | 官方口径？ |
| --- | --- | --- | --- |
| DeepSeek-V4-Pro | `dsv4-fp4-mi355x-sglang` / `-vllm` / `-atom`（及 `-mtp`） | fixed-seq-len 8k/1k 纯吞吐 | ✅ 官方发布口径 |
| DeepSeek-V4-Pro | `dsv4-fp4-mi355x-*-agentic-mtp` | agentic-coding | ⚠️ 实验性 |
| Kimi-K3 | `kimik3-fp4-mi355x-vllm-agentic-mtp` / `-atom-agentic-mtp` | agentic-coding | ⚠️ 实验性 |
| GLM-5.2 | `glm5.2-fp4-mi355x-sglang-agentic-mtp` / `-atom-agentic-mtp` | agentic-coding | ⚠️ 实验性 |

> **重要**：`benchmarks/single_node/agentic/README.md` 声明 agentic 目录是 MVP/实验性，结果
> **不发布在 inferencex.com、不应被引用**。在 MI355X 上只有 **dsv4 的 fixed-seq-len** 是官方
> 可复现的纯性能口径；kimi-k3 / glm5.2 在 MI355X 上只有 agentic 跑法，且依赖 AgentX trace 数据源。

## 用法

```bash
# 列出所有 MI355X 单机 recipe
local/run_local.sh --list

# 只看会跑哪些 job，不实际启动（强烈建议先跑一遍）
DRY_RUN=1 local/run_local.sh dsv4-fp4-mi355x-sglang

# 跑 dsv4 fixed-seq-len（sglang），只测到并发 128
local/run_local.sh dsv4-fp4-mi355x-sglang --max-conc 128

# 跑 glm5.2 agentic（sglang），并发 4~8
local/run_local.sh glm5.2-fp4-mi355x-sglang-agentic-mtp --min-conc 4 --max-conc 8

# 跑 kimi-k3 agentic（vllm）—— 自动回退到 kimik3_fp4_mi355x_mtp.sh
local/run_local.sh kimik3-fp4-mi355x-vllm-agentic-mtp
```

recipe 名之后的参数会透传给 `expand_recipe.py`：

| 参数 | 作用 |
| --- | --- |
| `--config PATH` | 从指定配置文件读 recipe（默认 `configs/amd-master.yaml`），用于跑自定义/裁剪的独立 recipe |
| `--min-conc N` / `--max-conc N` | 只跑落在 `[min,max]` 的并发点 |
| `--step-size N` | 并发几何倍增步长（默认 2，和 CI 一致） |
| `--total-cpu-dram-gb N` | 覆盖 agentic 自动算出的 DRAM 预算（GB） |
| `--duration N` | 覆盖 agentic 回放时长（秒，默认 3600） |
| `--port N` | server 端口（默认 8888） |

`run_local.sh` 的环境变量开关：

| 变量 | 作用 |
| --- | --- |
| `DRY_RUN=1` | 只展开并打印 job，不启动 |
| `RESULT_ROOT=<dir>` | 结果收集目录（默认 `./local_results`） |
| `KEEP_GOING=1` | 单个 job 失败后继续跑下一个（默认失败即停） |

## 只跑单个并发点（自定义独立 recipe）

想只测某一个并发点（例如 dsv4 sglang agentic 的 conc 32），有两种方式：

**方式 A —— 在完整 recipe 上用并发过滤（最省事）**

```bash
local/run_local.sh dsv4-fp4-mi355x-sglang-agentic-mtp --min-conc 32 --max-conc 32
```

`--min-conc/--max-conc` 会把该 recipe 所有 arm 里落在 `[32,32]` 的并发点筛出来。注意
conc 32 只存在于官方 recipe 的第二条 DRAM-offload arm，所以最终正好 1 个 job。

**方式 B —— 用一个自包含的独立 recipe 文件（推荐，可复现、可版本化）**

仓库自带示例 `local/recipes/dsv4-fp4-mi355x-sglang-agentic-conc32.yaml`，从官方
`dsv4-fp4-mi355x-sglang-agentic-mtp` 裁剪而来，镜像/模型/精度/框架/所有 flag 不变，
只保留 `conc-list: [32]`。用 `--config` 指向它即可，**无需改动 `configs/amd-master.yaml`**：

```bash
# 预览（不启动）：确认拓扑、脚本、DRAM 预算
DRY_RUN=1 local/run_local.sh dsv4-fp4-mi355x-sglang-agentic-conc32 \
  --config local/recipes/dsv4-fp4-mi355x-sglang-agentic-conc32.yaml

# 实跑（需在 SGLang ROCm 容器内）
local/run_local.sh dsv4-fp4-mi355x-sglang-agentic-conc32 \
  --config local/recipes/dsv4-fp4-mi355x-sglang-agentic-conc32.yaml

# 跑完聚合结果
python3 local/aggregate_results.py local_results --csv local_results/summary.csv
```

预览会解析出唯一的 job：

```
[1/1] benchmarks/single_node/agentic/dsv4_fp4_mi355x_sglang_mtp.sh
  MODEL=deepseek-ai/DeepSeek-V4-Pro  TP=8  EP=1  DPA=false  CONC=32  SPEC=mtp
  KV_OFFLOADING=dram  BACKEND=hicache  DRAM_GB=2399
  RESULT_FILENAME=dsv4_tp8_conc32_kvdram-hicache_spec-mtp_fp4_sglang_local
```

其中 `TOTAL_CPU_DRAM_GB=2399` 由官方公式算出（TP8 @ dram-util 0.80：
`min(3_095_781, 2_861_022) MiB × 1MiB × 0.80 × 8/8 / 1e9`）。

要自己做别的单点，复制那个 yaml 改 `conc-list`（以及需要的 `tp`/`kv-offloading` 等）即可。
每个独立 recipe 文件的顶层 key 就是传给 `run_local.sh` 的 recipe 名。

## 每个 job 导出的环境变量契约

这套契约来自 `.github/workflows/benchmark-tmpl.yml` 的 `env:` 块与各脚本的 `check_env_vars`。

**fixed-seq-len**（dsv4）：
`MODEL TP EP_SIZE DP_ATTENTION CONC ISL OSL MAX_MODEL_LEN RANDOM_RANGE_RATIO(=0.8)
RESULT_FILENAME PORT` + 元数据（`PRECISION FRAMEWORK EXP_NAME PP_SIZE/DCP_SIZE/PCP_SIZE=1 …`）。

**agentic-coding**（dsv4 / kimi-k3 / glm5.2）：
`MODEL TP EP_SIZE DP_ATTENTION CONC KV_OFFLOADING KV_OFFLOAD_BACKEND TOTAL_CPU_DRAM_GB
DURATION RESULT_DIR RESULT_FILENAME SCENARIO_TYPE=agentic-coding IS_AGENTIC=1 PORT`。
`TOTAL_CPU_DRAM_GB` 由官方公式算得：
`floor(min(available_mib, 2_861_022) * 1MiB * dram_util * tp / gpus_per_node / 1e9)`，
其中 `available_mib` / `gpus_per_node` 取自 `configs/runners.yaml` 的 `cluster:mi355x-amds`
（MI355X = 3_095_781 MiB → 被 3TB 上限截到 2_861_022，8 GPU/node）。

## 前置条件与注意事项

- **PyYAML**：`expand_recipe.py` 需要它。若容器内缺失：
  `python3 -m pip install --break-system-packages pyyaml`
- **/workspace**：fixed-seq-len 脚本把 server 日志和结果 JSON 硬编码写到 `/workspace/`。请确保
  容器内 `/workspace` 存在且可写（官方 CI 里它是仓库挂载点）。若你的仓库不在 `/workspace`，
  建议 `ln -s <repo> /workspace` 或在 `/workspace` 里放一份仓库。脚本用 `$(dirname "$0")/../../`
  定位 `benchmark_lib.sh`，从仓库根目录跑即可。
- **模型权重**：脚本会 `hf download "$MODEL"`。请确保 `HF_TOKEN` 和 HF cache（`HF_HUB_CACHE`）
  已配好，或提前把权重放到 cache 里。agentic 脚本支持 `MODEL_PATH` 指向本地权重目录。
- **agentic trace 数据**：agentic 脚本通过 `resolve_trace_source` 拉取 AgentX 回放 trace，并用
  `install_agentic_deps` 装 AIPerf。首次运行需联网；这也是 agentic 比 fixed-seq-len 复杂、
  更易失败的原因。建议先用 dsv4 fixed-seq-len 打通链路。
- **GPU 独占**：agentic 脚本开头会轮询 `rocm-smi` 等上一个 job 的 HBM 释放（最多 15 分钟）。
  串行跑多个 job 时这是正常等待。
- **改并发/搜索空间**：不要改脚本，改 `configs/amd-master.yaml` 里对应 recipe 的 `search-space`，
  或用 `--min-conc/--max-conc/--step-size` 现场裁剪。

## 结果聚合

两种场景产物不同，用不同工具：

| 场景 | 原始产物 | 聚合工具 |
| --- | --- | --- |
| fixed-seq-len | `benchmark_serving.py` 的单文件 JSON | `local/aggregate_results.py` |
| agentic-coding | `RESULT_DIR/aiperf_artifacts/`（逐请求 + 服务器指标） | `local/aggregate_agentic.sh` + `local/show_agentic.py` |

### fixed-seq-len（纯吞吐）：aggregate_results.py

每个 job 的原始结果 JSON（`benchmark_serving.py` 产出，文件名以 `RESULT_FILENAME` 开头）会被
收集到 `RESULT_ROOT`（默认 `./local_results`）。用 `local/aggregate_results.py` 把整个目录汇总成
一张对比表 + CSV：

```bash
# 汇总 ./local_results 下所有结果，打印表格并导出 CSV
python3 local/aggregate_results.py local_results --csv local_results/summary.csv

# 也可直接指定文件
python3 local/aggregate_results.py a.json b.json --hw mi355x
```

或让驱动器跑完自动聚合：`AGGREGATE=1 local/run_local.sh dsv4-fp4-mi355x-sglang`。

`aggregate_results.py` 是 `utils/process_result.py` 的**本地无依赖等价物**。官方脚本在 CI 里
逐文件、从环境变量读拓扑；本地版从 `RESULT_FILENAME`（`…_tp8-pp1-…-dpatrue_…_conc64_local`）
里反解出拓扑，因此事后对一整个目录聚合无需任何环境变量。派生字段与官方**完全一致**：

- `tput_per_gpu` = `total_token_throughput / num_gpus`
- `output_tput_per_gpu` = `output_throughput / num_gpus`
- `input_tput_per_gpu` = `(total - output) / num_gpus`
- 所有 `*_ms` 指标 → 秒（去掉 `_ms` 后缀）
- 所有 `tpot` 指标 → 交互性 `intvty` = `1000 / tpot_ms`（tokens/s/user）
- `num_gpus = tp * pp * pcp`（单机 MI355X 下 pp=pcp=1，即 `num_gpus = tp`）

每个输入文件旁生成一份 `agg_<name>.json`（与官方 `agg_*.json` 同 schema），并打印按
`(tp, dpa, conc)` 排序的表格，列包括：并发、TP、DP-attention、投机解码、
每卡输出吞吐、每卡总吞吐、中位交互性、中位 TTFT、中位 E2E 时延——正好是复现官方
Pareto 性能曲线所需的维度。

> 注意：官方 `agg_*.json` 还含功耗字段（`aggregate_power.py` 从 `gpu_metrics.csv` 积分得到）。
> 本地聚合器不做功耗积分（那需要 `benchmark_lib.sh` 的 GPU 监控采样和额外模块）。若需要功耗，
> 直接用 `utils/process_result.py` 配合各 job 留下的 `gpu_metrics.csv`。

### agentic-coding：aggregate_agentic.sh + show_agentic.py

agentic 场景不产出 `benchmark_serving.py` 那种单 JSON，而是把 AIPerf 回放的原始产物写到
`RESULT_DIR/aiperf_artifacts/`（`profile_export.jsonl` 逐请求 + `server_metrics_export.json`）。
必须再跑一步 `process_agentic_result` 把它聚合成一份指标 JSON——**这一步在纯脚本直跑时经常没自动
完成**（`INFMAX_CONTAINER_WORKSPACE` 没指向仓库根时会静默失败），所以 `RESULT_DIR` 里只剩原始产物、
没有聚合 JSON 是常见情况。用 `local/aggregate_agentic.sh` 一条命令补上：

```bash
# 在推理容器内，从仓库根运行
cd <仓库根>                                  # 例如 /shared/.../InferenceX
local/aggregate_agentic.sh /workspace/results   # 参数是 RESULT_DIR，默认就是 /workspace/results
```

它会：
1. 从 `RESULT_DIR/benchmark_command.txt` 解析 `CONC`、`MODEL`；
2. 从服务器命令文件（`sglang_command.txt` / `vllm_command.txt`）解析
   `TP`/`EP`/`dp-attn`/spec/KV-offload（`--enable-hierarchical-cache` → `dram`+`hicache`/`mooncake`）；
3. 由 model 推 `MODEL_PREFIX`，拼出与 `expand_recipe.py` 一致的 `RESULT_FILENAME`，
   并自动补 `KV_OFFLOAD_BACKEND_METADATA`（否则 dram 档会报 `KV_OFFLOAD_BACKEND is required`）；
4. `cd` 仓库根跑 `python3 -m utils.agentic.aggregation.process_agentic_result`，
   把聚合 JSON 写到 `AGENTIC_OUTPUT_DIR`（默认 = `RESULT_DIR`）；
5. 调 `show_agentic.py` 打印指标表。

任何自动值都能用环境变量覆盖（跑完 run 的实际配置和命令文件不符时）：

```bash
PRECISION=fp8 TP=4 RESULT_FILENAME=my_run local/aggregate_agentic.sh /path/to/results
```
可覆盖：`CONC MODEL MODEL_PREFIX FRAMEWORK PRECISION TP EP_SIZE DP_ATTENTION SPEC_DECODING
KV_OFFLOADING KV_OFFLOAD_BACKEND TOTAL_CPU_DRAM_GB RESULT_FILENAME RUNNER_TYPE AGENTIC_OUTPUT_DIR`。

单独读一份（或一批）已聚合的 agentic JSON，用 `show_agentic.py`：

```bash
python3 local/show_agentic.py /workspace/results/dsv4_tp8_conc32_*.json
python3 local/show_agentic.py /workspace/results --csv agentic_summary.csv   # 扫目录里所有 *_local.json
python3 local/show_agentic.py <json> --full                                  # 打印完整 request_metrics/server_metrics
```

表格列（正是 AgentX dashboard 曲线所需维度）：

```
conc  tp  ep  spec            kv   ok/total  out/gpu   tot/gpu  e2eNI.p50  e2eNI.p90  intvty.p50  ttft.p50
  32   8   1   mtp  dram/hicache  3079/3433   102.53  15196.22      45.05      27.66       50.59      0.62
```

- **`out/gpu`** = 每卡输出吞吐（`throughput.per_gpu.output_tput_tps`，dashboard 纵轴）
- **`e2eNI.p90`** = E2E Normalized Interactivity P90（`request_metrics.latency.e2e_norm_intvty.p90`，
  dashboard 横轴，含 TTFT/排队的真实体感，通常远低于 `intvty=1/tpot`）
- `ok/total` = 请求成功/总数；CSV 里字段更全（含 `e2eNI.p95`、聚合吞吐、`ttft/e2el` 各分位）。

> `process_agentic_result` 只用标准库，不需要 agentic 那个 uv/aiperf venv，系统 `python3` 即可重算。
