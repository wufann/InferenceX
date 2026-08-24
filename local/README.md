# 本地 MI355X Benchmark 工具（CI-free）

把 InferenceX 的 MI355X 单机 benchmark 从 GitHub Actions CI 里摘出来，在**推理容器内**直接、可复现地跑。
不碰 SLURM / enroot / docker，只做：**展开配置 → 跑官方脚本 → 聚合 → 读数**。

---

## 0. 心智模型（四步）

```
配置(recipe)  ──①展开&跑──▶  原始产物  ──②聚合──▶  指标 JSON  ──③读──▶  表格/CSV
  amd-master.yaml            /workspace/...        agg JSON            对比曲线
  或 local/recipes/*.yaml
```

跑的始终是仓库里**未改动的官方脚本**（`benchmarks/single_node/...`），所以性能口径与官方一致。
根据**场景**分两条 lane，用不同的聚合工具：

| 场景 | 例子 | 原始产物 | 聚合工具 |
| --- | --- | --- | --- |
| **fixed-seq-len**（纯吞吐 8k/1k） | dsv4 | 单个 `*_local.json` | `aggregate_results.py` |
| **agentic-coding**（AgentX 回放） | dsv4 / kimi-k3 / glm5.2 | `RESULT_DIR/aiperf_artifacts/` | `aggregate_agentic.sh` + `show_agentic.py` |

---

## 1. 工具速查

| 文件 | 角色 | 你直接敲吗 |
| --- | --- | --- |
| `run_local.sh` | **主入口**：展开 recipe → 跑 → 收集 | ✅ 跑测就用它 |
| `recipes/*.yaml` | 单点/裁剪过的独立 recipe（配 `--config` 用） | 选用 |
| `expand_recipe.py` | 把 recipe 展开成逐并发 job（`run_local.sh` 内部调用；也能 `--list`） | 偶尔 `--list` |
| `aggregate_results.py` | **fixed-seq-len** 聚合：一堆 `*_local.json` → 表 + CSV | fixed 场景 |
| `aggregate_agentic.sh` | **agentic** 聚合：`RESULT_DIR` → agg JSON（自动填 env）+ 打印 | agentic 场景 |
| `show_agentic.py` | 读 agentic agg JSON → 表 / CSV | agentic 场景 |

---

## 2. 前提（跑之前确认）

- **在推理容器内**，且 recipe 对应的镜像（sglang / vllm / atom）就是当前容器。
- **从仓库根运行**：`cd /shared/.../InferenceX`（`run_local.sh` 会自己 `cd`，但 `--config` 等相对路径按仓库根算）。
- `/workspace` 存在且可写（fixed 脚本把日志/结果硬编码写这里；agentic 用 `RESULT_DIR`）。
- **PyYAML**：`python3 -m pip install --break-system-packages pyyaml`（缺了才装）。
- **HF 权重**：脚本会 `hf download "$MODEL"`；配好 `HF_HUB_CACHE` / `HF_TOKEN`，或 `HF_HUB_OFFLINE=1` 用本地缓存。
  agentic 脚本支持 `export MODEL_PATH=/本地/权重目录`；fixed 脚本则把 `MODEL` 直接当路径（`MODEL` 以 `/` 开头就跳过下载）。
- **8 卡空闲**：agentic 脚本开头会轮询 `rocm-smi`，要求每卡 VRAM ≤10%（~4% 是正常驱动基线）。

---

## 3. 场景 A：fixed-seq-len 纯吞吐（dsv4，官方可复现口径）

```bash
# 列出所有 MI355X 单机 recipe
local/run_local.sh --list

# 预览会跑哪些 job（强烈建议先跑）
DRY_RUN=1 local/run_local.sh dsv4-fp4-mi355x-sglang

# 跑（可用 --max-conc / --min-conc 裁剪并发点）
local/run_local.sh dsv4-fp4-mi355x-sglang --max-conc 128

# 聚合：目录里所有 *_local.json → 表 + CSV
python3 local/aggregate_results.py local_results --csv local_results/summary.csv
# 或让驱动器跑完自动聚合：
AGGREGATE=1 local/run_local.sh dsv4-fp4-mi355x-sglang
```

结果 JSON 收集到 `RESULT_ROOT`（默认 `./local_results`）。`aggregate_results.py` 从文件名反解拓扑，
派生字段与官方 `utils/process_result.py` 完全一致（`tput_per_gpu`、`*_ms`→秒、`tpot`→`intvty` 等）。

---

## 4. 场景 B：agentic-coding（dsv4 / kimi-k3 / glm5.2）

agentic 是 AgentX 真实 trace 回放，一次跑一个并发点、约 1 小时、占满 8 卡。**属实验性口径**（见 §6）。

### 4.1 跑一个并发点（推荐：独立 recipe + `--config`）

`recipes/` 下已有 dsv4 sglang 的 conc32 / conc48 示例（从官方 `dsv4-fp4-mi355x-sglang-agentic-mtp` 裁剪，
只留一个 `conc-list`）。跑法：

```bash
cd /shared/.../InferenceX

# 预览
DRY_RUN=1 local/run_local.sh dsv4-fp4-mi355x-sglang-agentic-conc48 \
  --config local/recipes/dsv4-fp4-mi355x-sglang-agentic-conc48.yaml

# 实跑（≈1 小时）
local/run_local.sh dsv4-fp4-mi355x-sglang-agentic-conc48 \
  --config local/recipes/dsv4-fp4-mi355x-sglang-agentic-conc48.yaml
```

`run_local.sh` 已替你处理好两个历史坑：
- 自动 `export INFMAX_CONTAINER_WORKSPACE=<仓库根>`（否则 trace 加载/依赖安装/聚合会静默失败）。
- 每个并发点用**独立** `RESULT_DIR=/workspace/results/<RESULT_FILENAME>/`，**不会互相覆盖**。

跑别的点：复制 `recipes/*conc48.yaml` 改 `conc-list`，或直接 `--min-conc N --max-conc N` 裁官方完整 recipe。

### 4.2 聚合（若跑完没自动出 agg JSON，用这个补）

正常情况下 §4.1 跑完脚本会自带聚合。但若 `RESULT_DIR` 里只有 `aiperf_artifacts/` 而没有 agg JSON
（`INFMAX_CONTAINER_WORKSPACE` 当时没对时会这样），一条命令补上——它自动从 `benchmark_command.txt` /
`sglang_command.txt` 解析 `CONC/MODEL/TP/EP/KV/spec`，无需手敲 env：

```bash
cd /shared/.../InferenceX
local/aggregate_agentic.sh /workspace/results/dsv4_tp8_conc48_kvdram-hicache_spec-mtp_fp4_sglang_local
```

底层调的是官方 `python3 -m utils.agentic.aggregation.process_agentic_result`（纯标准库，不需 GPU / aiperf venv）。

### 4.3 读数 / 对比

```bash
# 单个
python3 local/show_agentic.py <RESULT_DIR>/dsv4_tp8_conc48_*.json

# 多点对比 + CSV（画 Pareto 曲线用）
python3 local/show_agentic.py \
  local_results/dsv4_tp8_conc32_*.json \
  local_results/dsv4_tp8_conc48_*.json --csv local_results/dsv4_curve.csv
```

表格列：`out/gpu`（每卡输出吞吐，dashboard 纵轴）、`e2eNI.p90`（E2E Normalized Interactivity P90，
dashboard 横轴，含 TTFT/排队的真实体感）、`intvty.p50`（=1/tpot，参考）、`ttft.p50`、`ok/total`。

---

## 5. 三个目标模型在 MI355X 上的 recipe

| 模型 | recipe | 场景 | 官方口径？ |
| --- | --- | --- | --- |
| DeepSeek-V4-Pro | `dsv4-fp4-mi355x-sglang` / `-vllm` / `-atom`（及 `-mtp`） | fixed-seq-len | ✅ |
| DeepSeek-V4-Pro | `dsv4-fp4-mi355x-*-agentic-mtp` | agentic | ⚠️ 实验性 |
| Kimi-K3 | `kimik3-fp4-mi355x-vllm-agentic-mtp` / `-atom-agentic-mtp` | agentic | ⚠️ 实验性 |
| GLM-5.2 | `glm5.2-fp4-mi355x-sglang-agentic-mtp` / `-atom-agentic-mtp` | agentic | ⚠️ 实验性 |

> `benchmarks/single_node/agentic/README.md` 声明 agentic 目录是 MVP/实验性、结果不发布、不应被引用。
> 在 MI355X 上只有 **dsv4 fixed-seq-len** 是官方可复现的纯性能口径；kimi-k3 / glm5.2 只有 agentic 跑法。

---

## 6. 常见坑

| 症状 | 原因 / 解决 |
| --- | --- |
| `No module named utils` | 没在仓库根跑聚合。`cd <仓库根>` 再跑。 |
| agentic 跑完 `RESULT_DIR` 里没有 agg JSON | `INFMAX_CONTAINER_WORKSPACE` 没指向仓库根 → 用 `aggregate_agentic.sh` 补聚合（§4.2）。 |
| `KV_OFFLOAD_BACKEND is required when KV_OFFLOADING is enabled` | dram 档手动聚合时漏设 `KV_OFFLOAD_BACKEND`；用 `aggregate_agentic.sh` 会自动补。 |
| `File not found: .../workspace/utils/...` | `INFMAX_CONTAINER_WORKSPACE` 多带了 `/workspace` 一层。它应 = 仓库根本身。 |
| conc 点互相覆盖 | 已修：agentic 每点独立 `RESULT_DIR`。老数据在扁平 `/workspace/results/` 是历史遗留。 |
| GPU 显示 ~4% 不释放 | 正常驱动/固件基线（0 进程时）；脚本以 ≤10% 判定「干净」，可直接开跑。 |

---

## 7. 参考：env 契约与 DRAM 公式

**fixed-seq-len** 每 job：`MODEL TP EP_SIZE DP_ATTENTION CONC ISL OSL MAX_MODEL_LEN
RANDOM_RANGE_RATIO(=0.8) RESULT_FILENAME PORT` + 元数据。

**agentic** 每 job：`MODEL TP EP_SIZE DP_ATTENTION CONC KV_OFFLOADING KV_OFFLOAD_BACKEND
TOTAL_CPU_DRAM_GB DURATION RESULT_DIR AGENTIC_OUTPUT_DIR RESULT_FILENAME
SCENARIO_TYPE=agentic-coding IS_AGENTIC=1 PORT`。

`TOTAL_CPU_DRAM_GB`（dram 档）由官方公式算得：
`floor(min(available_mib, 2_861_022) * 1MiB * dram_util * tp / gpus_per_node / 1e9)`，
`available_mib`/`gpus_per_node` 取自 `configs/runners.yaml` 的 `cluster:mi355x-amds`
（MI355X = 3_095_781 MiB → 截到 3TB 上限 2_861_022，8 GPU/node）。例：TP8 @ 0.80 → 2399 GB。

**`run_local.sh` 参数**（recipe 名后透传给 `expand_recipe.py`）：
`--config PATH`、`--min-conc N`、`--max-conc N`、`--step-size N`、`--total-cpu-dram-gb N`、`--duration N`、`--port N`。
**env 开关**：`DRY_RUN=1`（只展开不跑）、`RESULT_ROOT=<dir>`、`KEEP_GOING=1`、`AGGREGATE=1`（fixed 跑完自动聚合）。
