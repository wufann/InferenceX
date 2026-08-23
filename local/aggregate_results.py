#!/usr/bin/env python3
"""Aggregate local benchmark result JSONs into per-file agg JSON + a combined table.

Local, dependency-free counterpart to utils/process_result.py. That script runs
once per raw result file inside CI (reading topology from env vars) and emits
`agg_<RESULT_FILENAME>.json`. Here we instead recover the topology from the
RESULT_FILENAME that run_local.sh bakes in (…_tp8-pp1-…-dpatrue_…_conc64_local),
so you can aggregate a whole directory of results after the fact with no env.

Derived fields match process_result.py exactly:
  * tput_per_gpu        = total_token_throughput / num_gpus
  * output_tput_per_gpu = output_throughput      / num_gpus
  * input_tput_per_gpu  = (total - output)        / num_gpus
  * every `*_ms` metric  -> seconds (key loses the `_ms` suffix)
  * every `tpot` metric  -> interactivity `intvty` = 1000 / tpot_ms  (tok/s/user)
    (num_gpus = tp * pp * pcp; pp/pcp default to 1, matching single-node MI355X.)

Usage:
  local/aggregate_results.py [dir]                 # default: ./local_results
  local/aggregate_results.py ./local_results --csv summary.csv
  local/aggregate_results.py f1.json f2.json --hw mi355x
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

# Fields process_result.py drops from the arrays-heavy raw result; irrelevant to
# scalar aggregation but listed so we never accidentally treat them as metrics.
_ARRAY_KEYS = {"input_lens", "output_lens", "ttfts", "itls",
               "generated_texts", "errors"}


def parse_filename_meta(stem: str) -> dict:
    """Recover topology/labels from a RESULT_FILENAME stem. All best-effort."""
    def _find(pattern, cast=str, default=None):
        m = re.search(pattern, stem)
        return cast(m.group(1)) if m else default

    return {
        "tp": _find(r"tp(\d+)", int),
        "pp": _find(r"-pp(\d+)", int, 1),
        "pcp": _find(r"-pcp(\d+)", int, 1),
        "ep": _find(r"-ep(\d+)", int, 1),
        "dp_attention": _find(r"dpa(true|false)"),
        "spec_decoding": _find(r"spec-([a-z0-9]+)"),
        "conc_name": _find(r"conc(\d+)", int),
        "framework": _find(r"_(sglang|vllm|atom)_", str),
        "precision": _find(r"_(fp4|fp8|fp16|bf16)_", str),
        "seq": _find(r"_(\d+k\d+k)_", str),
    }


def is_raw_result(path: Path, blob: dict) -> bool:
    if path.name.startswith(("agg_", "power_", "power_validation_")):
        return False
    return "total_token_throughput" in blob and "output_throughput" in blob


def aggregate_one(path: Path, hw: str, gpus_override: int | None) -> dict | None:
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  skip {path.name}: {exc}", file=sys.stderr)
        return None
    if not is_raw_result(path, raw):
        return None

    meta = parse_filename_meta(path.stem)
    tp = meta["tp"] or 1
    num_gpus = gpus_override or (tp * (meta["pp"] or 1) * (meta["pcp"] or 1))
    if num_gpus <= 0:
        num_gpus = 1

    total_tput = float(raw["total_token_throughput"])
    out_tput = float(raw["output_throughput"])

    data: dict = {
        "hw": hw,
        "conc": int(raw.get("max_concurrency") or meta["conc_name"] or 0),
        "model": raw.get("model_id", ""),
        "framework": meta["framework"] or "",
        "precision": meta["precision"] or "",
        "spec_decoding": meta["spec_decoding"] or "none",
        "seq": meta["seq"] or "",
        "tp": tp,
        "pp": meta["pp"],
        "pcp": meta["pcp"],
        "ep": meta["ep"],
        "dp_attention": meta["dp_attention"] or "false",
        "num_gpus": num_gpus,
        "tput_per_gpu": total_tput / num_gpus,
        "output_tput_per_gpu": out_tput / num_gpus,
        "input_tput_per_gpu": (total_tput - out_tput) / num_gpus,
        "source_file": path.name,
    }

    # Same *_ms -> seconds and tpot -> interactivity transform as process_result.
    for key, value in raw.items():
        if key in _ARRAY_KEYS:
            continue
        if key.endswith("ms"):
            try:
                data[key.replace("_ms", "")] = float(value) / 1000.0
                if "tpot" in key:
                    data[key.replace("_ms", "").replace("tpot", "intvty")] = (
                        1000.0 / float(value) if float(value) else 0.0)
            except (TypeError, ValueError):
                pass
    return data


def write_agg_json(data: dict, out_dir: Path) -> None:
    stem = Path(data["source_file"]).stem
    (out_dir / f"agg_{stem}.json").write_text(json.dumps(data, indent=2))


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("no benchmark result files found.")
        return
    rows = sorted(rows, key=lambda r: (r["tp"], r["dp_attention"], r["conc"]))
    cols = [
        ("conc", "conc", "{:>5}"),
        ("tp", "tp", "{:>3}"),
        ("dpa", "dp_attention", "{:>5}"),
        ("spec", "spec_decoding", "{:>5}"),
        ("out_tok/s/gpu", "output_tput_per_gpu", "{:>13.1f}"),
        ("tot_tok/s/gpu", "tput_per_gpu", "{:>13.1f}"),
        ("intvty(med)", "median_intvty", "{:>11.2f}"),
        ("ttft_med(s)", "median_ttft", "{:>11.3f}"),
        ("e2el_med(s)", "median_e2el", "{:>11.3f}"),
    ]
    header = "  ".join(f"{h:>{len(fmt.format(0))}}" if False else h
                       for h, _, fmt in cols)
    # Simpler fixed-width header aligned to the format widths.
    widths = [max(len(h), len(fmt.format(0))) for h, _, fmt in cols]
    hdr = "  ".join(f"{h:>{w}}" for (h, _, _), w in zip(cols, widths))
    print(hdr)
    print("  ".join("-" * w for w in widths))
    for r in rows:
        cells = []
        for (h, key, fmt), w in zip(cols, widths):
            v = r.get(key)
            cells.append(f"{fmt.format(v):>{w}}" if isinstance(v, (int, float))
                         else f"{str(v if v is not None else '-'):>{w}}")
        print("  ".join(cells))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", default=["local_results"],
                    help="result dir(s) or JSON file(s) (default: local_results)")
    ap.add_argument("--hw", default="mi355x", help="hardware label (default mi355x)")
    ap.add_argument("--gpus", type=int, default=None,
                    help="override num_gpus (else derived from filename tp*pp*pcp)")
    ap.add_argument("--csv", default=None, help="also write a combined CSV here")
    ap.add_argument("--out-dir", default=None,
                    help="where to write per-file agg_*.json (default: alongside input)")
    args = ap.parse_args()

    files: list[Path] = []
    for p in args.paths:
        path = Path(p)
        if path.is_dir():
            files += sorted(path.glob("*.json"))
        elif path.is_file():
            files.append(path)
        else:
            print(f"warning: '{p}' not found", file=sys.stderr)

    rows: list[dict] = []
    for f in files:
        data = aggregate_one(f, args.hw, args.gpus)
        if data is None:
            continue
        out_dir = Path(args.out_dir) if args.out_dir else f.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        write_agg_json(data, out_dir)
        rows.append(data)

    print_table(rows)

    if args.csv and rows:
        keys: list[str] = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows -> {args.csv}")
    elif rows:
        print(f"\n{len(rows)} result(s); per-file agg_*.json written alongside inputs")


if __name__ == "__main__":
    main()
