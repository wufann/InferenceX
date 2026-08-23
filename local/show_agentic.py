#!/usr/bin/env python3
"""Print key metrics from agentic aggregated result JSON(s).

The agentic path writes one aggregated JSON per run (built by
utils/agentic/aggregation/process_agentic_result.py). This pulls out the
dashboard-relevant numbers: per-GPU output throughput (curve Y axis),
e2e_norm_intvty percentiles (curve X axis), plus ttft / success / cache.

Usage:
  python3 local/show_agentic.py <agg.json> [more.json ...]
  python3 local/show_agentic.py <dir>            # scans *_local.json in dir
  python3 local/show_agentic.py <dir> --csv out.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def is_agentic_agg(blob: dict) -> bool:
    return (blob.get("scenario_type") == "agentic-coding"
            or "request_metrics" in blob)


def pct(d: dict, key: str, p: str):
    v = d.get(key, {})
    return v.get(p) if isinstance(v, dict) else None


def row_for(path: Path, blob: dict) -> dict:
    rm = blob.get("request_metrics", {})
    lat = rm.get("latency", {})
    tput = rm.get("throughput", {})
    pg = tput.get("per_gpu", {})
    # kv_offload_backend may be a plain string or a {name, version} dict.
    backend = blob.get("kv_offload_backend")
    if isinstance(backend, dict):
        backend = backend.get("name", "")
    # Aggregate throughput is nested as throughput.<in|out|total>.tokens_per_second.
    def agg_tps(kind: str):
        return (tput.get(kind) or {}).get("tokens_per_second")
    return {
        "file": path.name,
        "hw": blob.get("hw", ""),
        "tp": blob.get("tp", ""),
        "ep": blob.get("ep", ""),
        "conc": blob.get("conc", ""),
        "spec": blob.get("spec_decoding", ""),
        "kv": f"{blob.get('kv_offloading','')}/{backend or '-'}",
        "dram_gb": blob.get("allocated_cpu_dram_gb", ""),
        "ok": blob.get("num_requests_successful", ""),
        "total": blob.get("num_requests_total", ""),
        # throughput (tokens/s)
        "out_tps_per_gpu": pg.get("output_tput_tps"),
        "tot_tps_per_gpu": pg.get("total_tput_tps"),
        "out_tps_agg": agg_tps("output"),
        "tot_tps_agg": agg_tps("total"),
        # e2e normalized interactivity (dashboard X axis), tok/s/user
        "e2e_norm_intvty_p50": pct(lat, "e2e_norm_intvty", "p50"),
        "e2e_norm_intvty_p90": pct(lat, "e2e_norm_intvty", "p90"),
        "e2e_norm_intvty_p95": pct(lat, "e2e_norm_intvty", "p95"),
        # plain 1/tpot interactivity for reference
        "intvty_p50": pct(lat, "intvty", "p50"),
        "ttft_p50_s": pct(lat, "ttft", "p50"),
        "e2el_p50_s": pct(lat, "e2el", "p50"),
    }


def fmt(v):
    if isinstance(v, float):
        return f"{v:.2f}"
    return "-" if v is None else str(v)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="agg JSON file(s) or dir(s)")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--full", action="store_true",
                    help="also dump full request_metrics + server_metrics for each file")
    args = ap.parse_args()

    files: list[Path] = []
    for p in args.paths:
        path = Path(p)
        if path.is_dir():
            files += sorted(path.glob("*_local.json"))
        elif path.is_file():
            files.append(path)
        else:
            print(f"warning: '{p}' not found", file=sys.stderr)

    rows = []
    for f in files:
        try:
            blob = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"skip {f.name}: {e}", file=sys.stderr)
            continue
        if not is_agentic_agg(blob):
            continue
        rows.append(row_for(f, blob))
        if args.full:
            print(f"\n===== {f.name} =====")
            print(json.dumps({
                "request_metrics": blob.get("request_metrics", {}),
                "server_metrics": blob.get("server_metrics", {}),
            }, indent=2, ensure_ascii=False))

    if not rows:
        print("no agentic aggregated JSON found.")
        return

    rows.sort(key=lambda r: (r["tp"], r["conc"]))
    cols = [
        ("conc", "conc"), ("tp", "tp"), ("ep", "ep"), ("spec", "spec"),
        ("kv", "kv"), ("ok/total", None),
        ("out/gpu", "out_tps_per_gpu"), ("tot/gpu", "tot_tps_per_gpu"),
        ("e2eNI.p50", "e2e_norm_intvty_p50"),
        ("e2eNI.p90", "e2e_norm_intvty_p90"),
        ("intvty.p50", "intvty_p50"),
        ("ttft.p50", "ttft_p50_s"),
    ]
    cells_all = []
    for r in rows:
        line = []
        for _, key in cols:
            line.append(f"{r['ok']}/{r['total']}" if key is None else fmt(r.get(key)))
        cells_all.append(line)
    widths = [max(len(h), *(len(c[i]) for c in cells_all)) for i, (h, _) in enumerate(cols)]
    print("  ".join(f"{h:>{w}}" for (h, _), w in zip(cols, widths)))
    print("  ".join("-" * w for w in widths))
    for line in cells_all:
        print("  ".join(f"{c:>{w}}" for c, w in zip(line, widths)))

    if args.csv:
        keys = list(rows[0].keys())
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows -> {args.csv}")


if __name__ == "__main__":
    main()
