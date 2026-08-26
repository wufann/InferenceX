#!/usr/bin/env python3
"""Expand one amd-master.yaml recipe into per-job env for local (CI-free) runs.

This is the local replacement for the CI chain
    generate_sweep_configs.py  ->  benchmark-tmpl.yml (env mapping)  ->  launch_mi355x-amds.sh

It parses configs/amd-master.yaml + configs/runners.yaml, expands a recipe's
search-space into one job per (arm, concurrency), computes the exact env-var
contract each benchmarks/single_node/*/*.sh consumes, and resolves the script
path with the same fallback rule as launch_mi355x-amds.sh.

Output: one job per line, tab-separated `KEY=VALUE` fields. The first field is
always `__SCRIPT__=<path relative to repo root>`. run_local.sh consumes this.

It deliberately does NOT touch SLURM/enroot/docker: the caller is assumed to be
inside the recipe's container already (per the task constraint).
"""

from __future__ import annotations

import argparse
import sys
from decimal import Decimal
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit(
        "PyYAML is required. Inside the container run:\n"
        "  python3 -m pip install --break-system-packages pyyaml"
    )

REPO_ROOT = Path(__file__).resolve().parent.parent
AMD_MASTER = REPO_ROOT / "configs" / "amd-master.yaml"
RUNNERS = REPO_ROOT / "configs" / "runners.yaml"

# Mirrors generate_sweep_configs.py.
SEQ_LEN_ITOS = {(1024, 1024): "1k1k", (8192, 1024): "8k1k"}
MAX_AGENTIC_AVAILABLE_CPU_DRAM_MIB = 2_861_022  # 3 TB decimal cap
BYTES_PER_MIB = 1024 * 1024
BYTES_PER_GB = 1_000_000_000
DEFAULT_AGENTIC_DURATION_SECONDS = 3600
DEFAULT_STEP_SIZE = 2  # geometric doubling of concurrency


def seq_len_str(isl: int, osl: int) -> str:
    return SEQ_LEN_ITOS.get((isl, osl), f"{isl}_{osl}")


def _yaml_int(value) -> int:
    # runners.yaml uses digit-group underscores (e.g. 3_095_781); PyYAML 1.1
    # already parses these as ints, but be defensive about str inputs.
    if isinstance(value, str):
        return int(value.replace("_", ""))
    return int(value)


def expand_conc_range(start: int, end: int, step: int) -> list[int]:
    """Geometric expansion identical to generate_sweep_configs.py."""
    out: list[int] = []
    conc = start
    while conc <= end:
        out.append(conc)
        if conc == end:
            break
        conc *= step
        if conc > end:
            conc = end
    return out


def conc_values_for_arm(arm: dict, step: int) -> list[int]:
    conc_list = arm.get("conc-list")
    if conc_list:
        return list(conc_list)
    return expand_conc_range(arm["conc-start"], arm["conc-end"], step)


def agentic_dram_gb(dram_util: float, tp: int, runner: str, runners_cfg: dict) -> int:
    """Reproduce agentic_dram_offload_gb() for single-node entries.

    floor(min(available_mib, CAP) * MiB * utilization * gpus_used / gpus_per_node / 1e9)
    where gpus_used == tp for a single-node (pp=pcp=1) entry.
    """
    hw = runners_cfg.get("hardware", {}).get(runner)
    if hw is None:
        sys.exit(
            f"runner '{runner}' has no hardware entry in configs/runners.yaml; "
            "cannot size agentic DRAM budget. Pass --total-cpu-dram-gb to override."
        )
    available_mib = min(
        _yaml_int(hw["available-cpu-dram-mib"]), MAX_AGENTIC_AVAILABLE_CPU_DRAM_MIB
    )
    gpus_per_node = int(hw["gpus-per-node"])
    utilization = Decimal(str(dram_util))
    proportional = (
        Decimal(available_mib) * BYTES_PER_MIB * utilization * tp / gpus_per_node
    )
    return int(proportional / BYTES_PER_GB)


# SKU tokens as they appear in benchmark script names. Longer/compound tokens
# first so 'gb300' is not shadowed by the 'b300' substring test.
HW_TOKENS = ("mi355x", "mi325x", "mi300x", "gb300", "gb200",
             "b300", "b200", "h200", "h100", "rtx6000pro")


def hw_from_runner(runner: str) -> str:
    """Derive the hardware SKU token (mi355x / b300 / ...) from a runner label.

    Runners look like 'mi355x', 'cluster:mi355x-amds', 'cluster:b300-nv', 'b300'.
    The SKU is what the benchmark script name embeds (dsv4_fp4_b300_sglang_mtp.sh).
    """
    r = runner.lower()
    for tok in HW_TOKENS:
        if tok in r:
            return tok
    sys.exit(f"cannot derive hardware SKU from runner '{runner}' "
             f"(known: {', '.join(HW_TOKENS)})")


def resolve_script(model_prefix: str, precision: str, framework: str,
                   scenario_subdir: str, spec: str, hw: str) -> str:
    """Reproduce the script-resolution logic in launch_<hw>-*.sh.

    SCRIPT_BASE = <prefix>_<precision>_<hw>
    Try <base>_<framework>[_mtp].sh first; else <base>[_atom][_mtp].sh
    (framework suffix is '_atom' only for the atom framework, else '').
    """
    spec_suffix = "_mtp" if spec == "mtp" else ""
    framework_suffix = "_atom" if framework == "atom" else ""
    base = f"{model_prefix}_{precision}_{hw}"
    subdir = f"benchmarks/single_node/{scenario_subdir}"
    primary = f"{subdir}{base}_{framework}{spec_suffix}.sh"
    fallback = f"{subdir}{base}{framework_suffix}{spec_suffix}.sh"
    if (REPO_ROOT / primary).is_file():
        return primary
    return fallback


def build_jobs(recipe_name: str, recipe: dict, runners_cfg: dict,
               step: int, min_conc, max_conc, dram_override,
               duration_override, port: int) -> list[dict]:
    model = recipe["model"]
    model_prefix = recipe["model-prefix"]
    precision = recipe["precision"]
    framework = recipe["framework"]
    runner = recipe["runner"]
    hw = hw_from_runner(runner)
    image = recipe.get("image", "")

    if recipe.get("multinode"):
        sys.exit(
            f"recipe '{recipe_name}' is multinode/disaggregated; this local tool "
            "targets single-node recipes only."
        )

    scenarios = recipe["scenarios"]
    jobs: list[dict] = []

    def conc_filter(values: list[int]) -> list[int]:
        if min_conc is not None:
            values = [c for c in values if c >= min_conc]
        if max_conc is not None:
            values = [c for c in values if c <= max_conc]
        return values

    # ---- fixed-seq-len ----
    for sc in scenarios.get("fixed-seq-len", []):
        isl, osl = sc["isl"], sc["osl"]
        exp_name = f"{model_prefix}_{seq_len_str(isl, osl)}"
        max_model_len = isl + osl + 256
        script = resolve_script(model_prefix, precision, framework,
                                "fixed_seq_len/", "none", hw)
        for arm in sc["search-space"]:
            tp = arm["tp"]
            ep = arm.get("ep", 1)
            dp_attn = bool(arm.get("dp-attn", False))
            spec = arm.get("spec-decoding", "none")
            for conc in conc_filter(conc_values_for_arm(arm, step)):
                rf = (f"{exp_name}_{precision}_{framework}_tp{tp}-pp1-dcp1-pcp1"
                      f"-ep{ep}-dpa{str(dp_attn).lower()}_disagg-false"
                      f"_spec-{spec}_conc{conc}_local")
                jobs.append({
                    "__SCRIPT__": script,
                    "MODEL": model,
                    "MODEL_PREFIX": model_prefix,
                    "PRECISION": precision,
                    "FRAMEWORK": framework,
                    "IMAGE": image,
                    "EXP_NAME": exp_name,
                    "ISL": str(isl),
                    "OSL": str(osl),
                    "MAX_MODEL_LEN": str(max_model_len),
                    "TP": str(tp),
                    "PP_SIZE": "1",
                    "DCP_SIZE": "1",
                    "PCP_SIZE": "1",
                    "EP_SIZE": str(ep),
                    "DP_ATTENTION": str(dp_attn).lower(),
                    "CONC": str(conc),
                    "SPEC_DECODING": spec,
                    "RANDOM_RANGE_RATIO": "0.8",
                    "DISAGG": "false",
                    "RUN_EVAL": "false",
                    "EVAL_ONLY": "false",
                    "RESULT_DIR": "/workspace/results",
                    "RESULT_FILENAME": rf,
                    "PORT": str(port),
                    "GPU_COUNT": str(tp),
                })

    # ---- agentic-coding ----
    for sc in scenarios.get("agentic-coding", []):
        dram_util = sc.get("dram-utilization")
        duration = duration_override or DEFAULT_AGENTIC_DURATION_SECONDS
        for arm in sc["search-space"]:
            tp = arm["tp"]
            ep = arm.get("ep", 1)
            dp_attn = bool(arm.get("dp-attn", False))
            spec = arm.get("spec-decoding", "none")
            kv_offloading = arm.get("kv-offloading", "none")
            kv_backend = arm.get("kv-offload-backend") or {}
            kv_backend_name = kv_backend.get("name", "")
            kv_backend_version = kv_backend.get("version", "")
            if kv_offloading == "dram":
                if dram_override is not None:
                    total_dram = dram_override
                else:
                    total_dram = agentic_dram_gb(dram_util, tp, runner, runners_cfg)
            else:
                total_dram = 0
            script = resolve_script(model_prefix, precision, framework,
                                    "agentic/", spec, hw)
            for conc in conc_filter(conc_values_for_arm(arm, step)):
                kv_tag = ("kvnone" if kv_offloading == "none"
                          else f"kv{kv_offloading}-{kv_backend_name}")
                exp_name = f"{model_prefix}_tp{tp}_conc{conc}_{kv_tag}"
                if spec != "none":
                    exp_name += f"_spec-{spec}"
                rf = f"{exp_name}_{precision}_{framework}_local"
                jobs.append({
                    "__SCRIPT__": script,
                    "MODEL": model,
                    "MODEL_PREFIX": model_prefix,
                    "PRECISION": precision,
                    "FRAMEWORK": framework,
                    "IMAGE": image,
                    "EXP_NAME": exp_name,
                    "TP": str(tp),
                    "PP_SIZE": "1",
                    "DCP_SIZE": "1",
                    "PCP_SIZE": "1",
                    "EP_SIZE": str(ep),
                    "DP_ATTENTION": str(dp_attn).lower(),
                    "CONC": str(conc),
                    "SPEC_DECODING": spec,
                    "SCENARIO_TYPE": "agentic-coding",
                    "SCENARIO_SUBDIR": "agentic/",
                    "IS_AGENTIC": "1",
                    "KV_OFFLOADING": kv_offloading,
                    "KV_OFFLOAD_BACKEND": kv_backend_name,
                    "KV_OFFLOAD_BACKEND_METADATA": (
                        f'{{"name":"{kv_backend_name}","version":"{kv_backend_version}"}}'
                        if kv_backend_name else ""),
                    "TOTAL_CPU_DRAM_GB": str(total_dram),
                    "DURATION": str(duration),
                    "DISAGG": "false",
                    "RUN_EVAL": "false",
                    "EVAL_ONLY": "false",
                    # Per-point dir so concurrency points never overwrite each
                    # other's raw artifacts; agg JSON lands in the same dir.
                    "RESULT_DIR": f"/workspace/results/{rf}",
                    "AGENTIC_OUTPUT_DIR": f"/workspace/results/{rf}",
                    "RESULT_FILENAME": rf,
                    "PORT": str(port),
                    "GPU_COUNT": str(tp),
                })

    if not jobs:
        sys.exit(f"recipe '{recipe_name}' produced no jobs after filtering.")
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recipe", nargs="?", help="recipe key from the config file")
    ap.add_argument("--config", default=str(AMD_MASTER),
                    help="master config to read recipes from "
                         "(default: configs/amd-master.yaml)")
    ap.add_argument("--list", action="store_true",
                    help="list all single-node mi355x recipe names and exit")
    ap.add_argument("--step-size", type=int, default=DEFAULT_STEP_SIZE)
    ap.add_argument("--min-conc", type=int, default=None)
    ap.add_argument("--max-conc", type=int, default=None)
    ap.add_argument("--total-cpu-dram-gb", type=int, default=None,
                    help="override the computed agentic DRAM budget (GB)")
    ap.add_argument("--duration", type=int, default=None,
                    help="override agentic replay duration (seconds)")
    ap.add_argument("--port", type=int, default=8888)
    args = ap.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        sys.exit(f"config not found: {config_path}")
    master = yaml.safe_load(config_path.read_text())
    runners_cfg = yaml.safe_load(RUNNERS.read_text())

    if args.list:
        for name, rec in master.items():
            if not isinstance(rec, dict):
                continue
            if "mi355x" in str(rec.get("runner", "")) and not rec.get("multinode"):
                scen = ",".join((rec.get("scenarios") or {}).keys())
                print(f"{name}\t[{rec.get('framework')}] {scen}")
        return

    if not args.recipe:
        ap.error("recipe name required (or use --list)")
    if args.recipe not in master:
        sys.exit(f"recipe '{args.recipe}' not found in {config_path}")

    jobs = build_jobs(args.recipe, master[args.recipe], runners_cfg,
                      args.step_size, args.min_conc, args.max_conc,
                      args.total_cpu_dram_gb, args.duration, args.port)

    for job in jobs:
        fields = [f"__SCRIPT__={job.pop('__SCRIPT__')}"]
        fields += [f"{k}={v}" for k, v in job.items()]
        print("\t".join(fields))


if __name__ == "__main__":
    main()
