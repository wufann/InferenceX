"""Behavioral tests for the local Qwen3.8 AgentX runner and plotter."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

from infx.bench.local_qwen38 import plot, point_env, server_argv


class LocalQwen38Tests(unittest.TestCase):
    def test_point_settings_follow_concurrency_and_committed_golden_curve(self) -> None:
        args = argparse.Namespace(
            model="amd/Qwen3.8-Flash-Next-Quark-MXFP4-PLEFP8",
            model_path="/models/qwen38",
            image="aigmkt/qwen38:test",
            port=8123,
            duration=3600,
        )
        with tempfile.TemporaryDirectory() as tmp:
            env = point_env(args, 8, Path(tmp))
        command = server_argv(args, 8)
        self.assertEqual(command[command.index("--max-running-requests") + 1], "16")
        self.assertEqual(command[command.index("--cuda-graph-max-bs") + 1], "8")
        self.assertEqual(env["SGLANG_SIMULATE_ACC_LEN"], "2.32")
        self.assertEqual(env["CONC"], "8")
        self.assertEqual(env["AIPERF_SERVER_URL"], "http://127.0.0.1:8123")
        self.assertEqual(env["SGLANG_AITER_QSA_PA_DECODE"], "1")

    def test_plot_filters_dominated_point_and_writes_numeric_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            aggregate_dir = root / "aggregates"
            aggregate_dir.mkdir()
            for conc, x, total, output in (
                (1, 60.0, 90.0, 50.0),
                (4, 45.0, 180.0, 100.0),
                (8, 40.0, 120.0, 75.0),
            ):
                row = {
                    "conc": conc,
                    "num_requests_successful": 25,
                    "request_metrics": {
                        "latency": {"e2e_norm_intvty": {"p90": x}},
                        "throughput": {
                            "per_gpu": {
                                "total_tput_tps": total,
                                "output_tput_tps": output,
                            }
                        },
                    },
                }
                (aggregate_dir / f"c{conc}.json").write_text(json.dumps(row))

            plot(root)
            with (root / "charts" / "total_pareto.csv").open() as stream:
                chosen = list(csv.DictReader(stream))
            self.assertEqual([int(point["conc"]) for point in chosen], [4, 1])
            self.assertEqual([float(point["x"]) for point in chosen], [45.0, 60.0])
            svg = (root / "charts" / "total_pareto.svg").read_text()
            self.assertIn("P90 E2E normalized interactivity", svg)
            self.assertIn("C=4", svg)
            output_frontier = json.loads((root / "charts" / "output_pareto.json").read_text())
            self.assertEqual([point["conc"] for point in output_frontier], [4, 1])

            (root / "run_config.json").write_text("{}")
            for conc in (1, 4):
                point_dir = root / "points" / f"c{conc}"
                point_dir.mkdir(parents=True)
                (point_dir / "validated.ok").write_text("passed\n")
            plot(root)
            with (root / "charts" / "total_points.csv").open() as stream:
                valid_points = list(csv.DictReader(stream))
            self.assertEqual({int(point["conc"]) for point in valid_points}, {1, 4})


if __name__ == "__main__":
    unittest.main()
