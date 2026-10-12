"""Run Qwen3.8 AgentX on one MI355X container and plot local Pareto points.

This is a local counterpart of the single-node AgentX recipe. It delegates replay,
aggregation, and validity checks to the existing ``infx.bench agentic`` command.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
from contextlib import suppress
from html import escape
from pathlib import Path
from typing import Any

from infx.bench import proc, server
from infx.golden_al_distribution import golden_length

SERVER_FLAGS = {
    "SGLANG_USE_AITER": "1",
    "SGLANG_AITER_GDN_DECODE": "1",
    "SGLANG_AITER_GDN_CONV": "1",
    "SGLANG_AITER_HC_MIX": "1",
    "SGLANG_AITER_QSA_PA_DECODE": "1",
    "SGLANG_ROCM_USE_AITER_TOPK_GATING": "1",
    "SGLANG_AITER_GDN_PREFILL": "1",
    "AITER_GDR_FLYDSL": "1",
    "SGLANG_QSA_PAGED_DECODE": "1",
    "SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION": "1",
    "SGLANG_TIMEOUT_KEEP_ALIVE": "1800",
    "PYTHONNOUSERSITE": "1",
}
LOADER = "semianalysis_cc_traces_weka_062126_256k"
METRICS = {
    "total": ("total_tput_tps", "Total tokens/s/GPU"),
    "output": ("output_tput_tps", "Output tokens/s/GPU"),
}


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def server_argv(args: argparse.Namespace, conc: int) -> list[str]:
    return [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", args.model_path,
        "--served-model-name", args.model,
        "--host", "0.0.0.0", "--port", str(args.port),
        "--trust-remote-code", "--tp-size", "1", "--ep-size", "1",
        "--attention-backend", "aiter", "--moe-runner-backend", "aiter",
        "--mamba-ssm-dtype", "bfloat16", "--page-size", "32",
        "--kv-cache-dtype", "auto", "--chunked-prefill-size", "16384",
        "--watchdog-timeout", "1200", "--mem-fraction-static", "0.9",
        "--model-loader-extra-config", '{"enable_multithread_load": true}',
        "--max-running-requests", str(2 * conc),
        "--cuda-graph-max-bs-decode", str(min(conc, 64)),
        "--speculative-algorithm", "NEXTN", "--speculative-num-steps", "3",
        "--speculative-eagle-topk", "1", "--speculative-num-draft-tokens", "4",
        "--reasoning-parser", "qwen3-thinking", "--stream-interval", "50",
        "--scheduler-recv-interval", "30", "--tokenizer-path", args.model,
        "--enable-metrics", "--enable-cache-report",
    ]


def point_env(args: argparse.Namespace, conc: int, root: Path) -> dict[str, str]:
    url = f"http://127.0.0.1:{args.port}"
    golden = golden_length(
        "qwen3.8next", {"method": "NEXTN", "num_speculative_tokens": 3}, "thinking_on"
    )
    return {
        **os.environ,
        **SERVER_FLAGS,
        "SGLANG_SIMULATE_ACC_LEN": f"{golden:g}",
        "SGLANG_SIMULATE_ACC_METHOD": "match-expected",
        "SGLANG_SIMULATE_ACC_TOKEN_MODE": "real-draft-token",
        "PYTHONPATH": proc.pythonpath(),
        "RESULT_DIR": str(root / "points" / f"c{conc}"),
        "RESULT_FILENAME": f"qwen38_mi355x_c{conc}",
        "AGENTIC_OUTPUT_DIR": str(root / "aggregates"),
        "AIPERF_RUNTIME_DIR": str(root / "aiperf-runtime"),
        "AIPERF_SERVER_URL": url,
        "AIPERF_SERVER_METRICS_URLS": f"{url}/metrics",
        "AIPERF_REQUIRED_SERVER_METRIC_PREFIX": "sglang:",
        "AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS": "300",
        "AIPERF_EXPERIMENTAL_FAST": "0",
        "AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID": "false",
        "AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING": "0",
        "WEKA_LOADER_OVERRIDE": LOADER,
        "MODEL": args.model,
        "MODEL_PREFIX": "qwen3.8next",
        "SERVED_MODEL_NAME": args.model,
        "FRAMEWORK": "sglang",
        "PRECISION": "fp4",
        "SPEC_DECODING": "mtp",
        "RUNNER_TYPE": "mi355x",
        "IMAGE": args.image,
        "CONC": str(conc),
        "DURATION": str(args.duration),
        "EVAL_ONLY": "false",
        "IS_AGENTIC": "1",
        "IS_MULTINODE": "false",
        "SCENARIO_TYPE": "agentic-coding",
        "THINKING_MODE": "thinking_on",
        "KV_OFFLOADING": "none",
        "TP": "1",
        "EP_SIZE": "1",
    }


def wait_for_server(url: str, model: str, child: subprocess.Popen[bytes], timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError(f"SGLang exited before ready (code {child.returncode})")
        models = server.http_json(f"{url}/v1/models")
        if isinstance(models, dict) and any(
            isinstance(item, dict) and item.get("id") == model
            for item in models.get("data", [])
        ):
            status = server.http_status(f"{url}/health")
            if status is not None and status < 400:
                return
        time.sleep(5)
    raise TimeoutError(f"SGLang did not serve {model} within {timeout}s")


def stop_server(child: subprocess.Popen[bytes]) -> None:
    with suppress(ProcessLookupError):
        os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=60)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def run_point(args: argparse.Namespace, conc: int, root: Path) -> None:
    env = point_env(args, conc, root)
    point_dir = Path(env["RESULT_DIR"])
    point_dir.mkdir(parents=True)
    command = server_argv(args, conc)
    (point_dir / "server_command.txt").write_text(shlex.join(command) + "\n")
    recorded_env = set(SERVER_FLAGS) | {
        "SGLANG_SIMULATE_ACC_LEN",
        "SGLANG_SIMULATE_ACC_METHOD",
        "SGLANG_SIMULATE_ACC_TOKEN_MODE",
    }
    (point_dir / "server_env.json").write_text(
        json.dumps({key: env[key] for key in sorted(recorded_env)}, indent=2) + "\n"
    )
    url = env["AIPERF_SERVER_URL"]
    if server.http_status(f"{url}/health") is not None:
        raise RuntimeError(f"port {args.port} already has an HTTP server; use a free port")
    with (point_dir / "server.log").open("wb") as log:
        child = subprocess.Popen(
            command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            wait_for_server(url, args.model, child, args.startup_timeout)
            runtime_python = root / "aiperf-runtime" / "venv" / "bin" / "python"
            python = runtime_python if runtime_python.is_file() else Path(sys.executable)
            client = [str(python), "-m", "infx.bench", "agentic"]
            rc = proc.tee(client, point_dir / "local_benchmark.log", env)
            if rc:
                raise RuntimeError(f"AgentX point C={conc} failed with exit code {rc}")
            (point_dir / "validated.ok").write_text(
                "AgentX replay, aggregation and validation passed\n"
            )
        except (OSError, ValueError, RuntimeError) as error:
            raise RuntimeError(f"{error}; inspect {point_dir / 'server.log'}") from error
        finally:
            stop_server(child)


def _number(value: Any) -> float | None:
    if type(value) not in (int, float):
        return None
    result = float(value)
    return result if math.isfinite(result) and result > 0 else None


def load_points(root: Path, metric: str) -> list[dict[str, Any]]:
    field, _ = METRICS[metric]
    points: list[dict[str, Any]] = []
    managed_run = (root / "run_config.json").is_file()
    for path in sorted((root / "aggregates").glob("*.json")):
        row = json.loads(path.read_text())
        if managed_run and not (root / "points" / f"c{row.get('conc')}" / "validated.ok").is_file():
            continue
        req = row.get("request_metrics", {})
        x = _number(req.get("latency", {}).get("e2e_norm_intvty", {}).get("p90"))
        y = _number(req.get("throughput", {}).get("per_gpu", {}).get(field))
        if x is None or y is None or row.get("num_requests_successful", 0) <= 0:
            continue
        points.append({"conc": row["conc"], "x": x, "y": y, "source": str(path)})
    return points


def frontier(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep nondominated points when both interactivity and throughput are larger-is-better."""
    return sorted(
        [
            point for point in points
            if not any(
                other is not point
                and other["x"] >= point["x"]
                and other["y"] >= point["y"]
                and (other["x"] > point["x"] or other["y"] > point["y"])
                for other in points
            )
        ],
        key=lambda point: point["x"],
    )


def svg_chart(points: list[dict[str, Any]], selected: list[dict[str, Any]], ylabel: str) -> str:
    width, height, left, top, right, bottom = 900, 560, 95, 50, 35, 85
    x_max = max(point["x"] for point in points) * 1.1
    y_max = max(point["y"] for point in points) * 1.1
    plot_w, plot_h = width - left - right, height - top - bottom

    def xy(point: dict[str, Any]) -> tuple[float, float]:
        return left + plot_w * point["x"] / x_max, top + plot_h * (1 - point["y"] / y_max)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<path d="M{left} {top}V{height-bottom}H{width-right}" fill="none" stroke="#555"/>',
    ]
    for index in range(6):
        x = left + plot_w * index / 5
        y = top + plot_h * (1 - index / 5)
        lines.append(f'<path d="M{x:.1f} {top}V{height-bottom}" stroke="#eee"/>')
        lines.append(f'<path d="M{left} {y:.1f}H{width-right}" stroke="#eee"/>')
        lines.append(
            f'<text x="{x:.1f}" y="{height-bottom+23}" text-anchor="middle" '
            f'font-size="13">{x_max*index/5:.1f}</text>'
        )
        lines.append(
            f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" '
            f'font-size="13">{y_max*index/5:.0f}</text>'
        )
    if selected:
        coordinates = " ".join(f"{x:.1f},{y:.1f}" for x, y in map(xy, selected))
        lines.append(
            f'<polyline points="{coordinates}" fill="none" stroke="#d65a18" '
            'stroke-width="3"/>'
        )
    chosen = {id(point) for point in selected}
    for point in points:
        x, y = xy(point)
        color = "#d65a18" if id(point) in chosen else "#65758b"
        label = escape(str(point["conc"]))
        lines.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}">'
            f'<title>C={label}: {point["x"]:.2f}, {point["y"]:.1f}</title></circle>'
        )
        lines.append(f'<text x="{x+9:.1f}" y="{y-8:.1f}" font-size="13">C={label}</text>')
    lines += [
        f'<text x="{left+plot_w/2:.1f}" y="{height-20}" text-anchor="middle" '
        'font-size="16">P90 E2E normalized interactivity (output tokens/s/user)</text>',
        f'<text x="23" y="{top+plot_h/2:.1f}" '
        f'transform="rotate(-90 23,{top+plot_h/2:.1f})" text-anchor="middle" '
        f'font-size="16">{escape(ylabel)}</text>',
        '</svg>',
    ]
    return "\n".join(lines) + "\n"


def plot(root: Path) -> None:
    charts = root / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    for metric, (_, ylabel) in METRICS.items():
        points = load_points(root, metric)
        if not points:
            raise ValueError(f"No valid AgentX aggregate points in {root / 'aggregates'}")
        selected = frontier(points)
        for name, rows in (("points", points), ("pareto", selected)):
            with (charts / f"{metric}_{name}.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=("conc", "x", "y", "source"))
                writer.writeheader()
                writer.writerows(rows)
        (charts / f"{metric}_pareto.json").write_text(json.dumps(selected, indent=2) + "\n")
        (charts / f"{metric}_pareto.svg").write_text(svg_chart(points, selected, ylabel))
        print(f"{metric}: {len(points)} measured, {len(selected)} Pareto points -> {charts}")


def run(args: argparse.Namespace) -> int:
    root = args.output_dir.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Output directory must be new or empty: {root}")
    if len(args.concurrencies) != len(set(args.concurrencies)):
        raise ValueError("Concurrency values must be unique")
    if not args.model_path:
        raise ValueError("--model-path must be nonempty")
    if args.port > 65535:
        raise ValueError("--port must be at most 65535")
    if not args.dry_run and not Path(args.model_path).is_dir():
        raise ValueError(f"Model directory does not exist: {args.model_path}")
    if not args.dry_run and not (proc.REPO_ROOT / "utils" / "aiperf" / "pyproject.toml").is_file():
        raise ValueError(
            "utils/aiperf submodule is empty; run git submodule update --init "
            "inferencex-e2e/utils/aiperf"
        )
    if args.dry_run:
        for conc in args.concurrencies:
            print(f"C={conc}: {shlex.join(server_argv(args, conc))}")
        return 0
    root.mkdir(parents=True, exist_ok=True)
    (root / "aggregates").mkdir(exist_ok=True)
    manifest = vars(args).copy()
    manifest["output_dir"] = str(root)
    manifest["golden_al"] = golden_length(
        "qwen3.8next", {"method": "NEXTN", "num_speculative_tokens": 3}, "thinking_on"
    )
    (root / "run_config.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for conc in args.concurrencies:
        print(f"Starting fresh SGLang server for C={conc}", flush=True)
        run_point(args, conc, root)
        plot(root)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="Launch SGLang afresh for each AgentX concurrency")
    for name in ("model", "model-path", "image"):
        run_parser.add_argument(f"--{name}", required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--concurrencies", nargs="+", type=positive_int, required=True)
    run_parser.add_argument("--duration", type=positive_int, required=True)
    run_parser.add_argument("--port", type=positive_int, required=True)
    run_parser.add_argument("--startup-timeout", type=positive_int, default=1800)
    run_parser.add_argument("--dry-run", action="store_true")
    plot_parser = sub.add_parser("plot", help="Regenerate local CSV, JSON, and SVG charts")
    plot_parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return run(args)
        plot(args.output_dir)
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
