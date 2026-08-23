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

## 结果

每个 job 的结果 JSON（文件名以 `RESULT_FILENAME` 开头）会被收集到 `RESULT_ROOT`（默认
`./local_results`）。如需与官方一样做聚合，可参考 `utils/process_result.py`。
