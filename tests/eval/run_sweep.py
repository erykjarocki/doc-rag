#!/usr/bin/env python3
"""Run eval with multiple configurations and collect results for comparison.

Usage:
    python tests/eval/run_sweep.py                    # Run default sweep
    python tests/eval/run_sweep.py --configs chunk256 chunk384 chunk512
    python tests/eval/run_sweep.py --dry-run           # Show what would run

Each config is a separate pytest subprocess with its own env vars.
Results are saved to tests/eval/eval-sweep/<config-name>/.
"""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).parent
PROJECT_ROOT = EVAL_DIR.parent.parent
PYTEST_BIN = PROJECT_ROOT / "venv" / "bin" / "pytest"
SWEEP_DIR = EVAL_DIR / "eval-sweep"

# ---------------------------------------------------------------------------
# Default configurations to sweep
# ---------------------------------------------------------------------------

DEFAULT_CONFIGS = {
    "chunk256": {
        "description": "Small chunks (256 tokens, 30 overlap)",
        "env": {"CHUNK_SIZE": "256", "CHUNK_OVERLAP": "30"},
    },
    "chunk384": {
        "description": "Default chunks (384 tokens, 50 overlap)",
        "env": {"CHUNK_SIZE": "384", "CHUNK_OVERLAP": "50"},
    },
    "chunk512": {
        "description": "Large chunks (512 tokens, 80 overlap)",
        "env": {"CHUNK_SIZE": "512", "CHUNK_OVERLAP": "80"},
    },
    "overlap100": {
        "description": "Default size, high overlap (384 tokens, 100 overlap)",
        "env": {"CHUNK_SIZE": "384", "CHUNK_OVERLAP": "100"},
    },
    "no-rerank": {
        "description": "No cross-encoder reranking",
        "env": {"CHUNK_SIZE": "384", "CHUNK_OVERLAP": "50", "RERANK_ENABLED": "false"},
    },
}


def run_config(name: str, cfg: dict, verbose: bool = True) -> dict:
    """Run eval for a single configuration. Returns result metadata."""
    out_dir = SWEEP_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "eval-report.json"
    report_html = out_dir / "eval-report.html"

    env = os.environ.copy()
    env.update(cfg["env"])
    # Ensure the sweep report path is used
    env["EVAL_REPORT_PATH"] = str(report_path)
    env["EVAL_REPORT_HTML"] = str(report_html)

    cmd = [
        str(PYTEST_BIN),
        str(EVAL_DIR),
        "-v",
        "-m", "eval or rerank",
        "--tb=short",
        "-q",
    ]

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  Config: {name}")
        print(f"  {cfg['description']}")
        print(f"  Env: {cfg['env']}")
        print(f"{'=' * 60}")

    t0 = time.time()
    result = subprocess.run(
        cmd,
        env=env,
        cwd=str(PROJECT_ROOT),
        capture_output=not verbose,
        text=True,
    )
    elapsed = time.time() - t0

    status = "pass" if result.returncode == 0 else "fail"

    # Load metrics from the report if it was generated
    metrics = {}
    num_queries = 0
    if report_path.exists():
        try:
            data = json.loads(report_path.read_text())
            metrics = data.get("metrics", {})
            num_queries = len(data.get("queries", []))
        except (json.JSONDecodeError, KeyError):
            pass

    summary = {
        "name": name,
        "description": cfg["description"],
        "env": cfg["env"],
        "status": status,
        "elapsed_s": round(elapsed, 1),
        "metrics": metrics,
        "num_queries": num_queries,
        "returncode": result.returncode,
    }

    # Save per-config summary
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if verbose:
        if metrics:
            print(f"\n  Results: Recall@2={metrics.get('recall_at_2', '?'):.2f}  "
                  f"Precision@2={metrics.get('precision_at_2', '?'):.2f}  "
                  f"MRR={metrics.get('mrr', '?'):.2f}")
        print(f"  Status: {status} ({elapsed:.1f}s)")

    return summary


def generate_comparison_html(results: list[dict]) -> Path:
    """Generate a comparison HTML report from all config results."""
    # Lazy import to avoid loading report gen when just running eval
    sys.path.insert(0, str(PROJECT_ROOT))
    from tests.eval.sweep_report import generate_sweep_report

    out_path = SWEEP_DIR / "sweep-comparison.html"
    generate_sweep_report(results, out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Run eval sweep across configurations")
    parser.add_argument(
        "--configs", nargs="*",
        help="Config names to run (default: all)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show configs without running")
    parser.add_argument("--quiet", action="store_true", help="Suppress pytest output")
    parser.add_argument("--parallel", "-p", action="store_true",
                        help="Run configs in parallel (default: sequential)")
    parser.add_argument("--jobs", "-j", type=int, default=0,
                        help="Max parallel jobs (default: number of configs)")
    args = parser.parse_args()

    configs = DEFAULT_CONFIGS
    if args.configs:
        configs = {k: v for k, v in DEFAULT_CONFIGS.items() if k in args.configs}
        if not configs:
            print(f"No matching configs found. Available: {list(DEFAULT_CONFIGS.keys())}")
            sys.exit(1)

    print(f"Sweep: {len(configs)} configurations")
    for name, cfg in configs.items():
        print(f"  {name}: {cfg['description']}")

    if args.dry_run:
        print("\n(dry run — no tests executed)")
        return

    if args.parallel:
        max_workers = args.jobs or len(configs)
        print(f"\nRunning {len(configs)} configs in parallel (max {max_workers} workers)")
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(run_config, name, cfg, verbose=False): name
                for name, cfg in configs.items()
            }
            results = []
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    print(f"  {name}: FAILED — {e}")
                    result = {
                        "name": name, "description": configs[name]["description"],
                        "env": configs[name]["env"], "status": "error",
                        "elapsed_s": 0, "metrics": {}, "num_queries": 0, "returncode": -1,
                    }
                results.append(result)
                m = result["metrics"]
                print(f"  {result['name']}: Recall@2={m.get('recall_at_2', '?'):.2f}  "
                      f"Precision@2={m.get('precision_at_2', '?'):.2f}  "
                      f"MRR={m.get('mrr', '?'):.2f}  ({result['elapsed_s']:.1f}s)")
        # Sort results to match config order
        name_order = list(configs.keys())
        results.sort(key=lambda r: name_order.index(r["name"]) if r["name"] in name_order else 99)
    else:
        results = []
        for name, cfg in configs.items():
            result = run_config(name, cfg, verbose=not args.quiet)
            results.append(result)

    # Print summary table
    print(f"\n{'=' * 70}")
    print("  SWEEP SUMMARY")
    print(f"{'=' * 70}")
    header = (
        f"  {'Config':<18} {'Recall@2':>10} {'Prec@2':>10}"
        f" {'MRR':>10} {'Time':>8} {'Status':>8}"
    )
    print(header)
    print(f"  {'-' * 64}")
    for r in results:
        m = r["metrics"]
        print(f"  {r['name']:<18} "
              f"{m.get('recall_at_2', 0):>10.2f} "
              f"{m.get('precision_at_2', 0):>10.2f} "
              f"{m.get('mrr', 0):>10.2f} "
              f"{r['elapsed_s']:>7.1f}s "
              f"{r['status']:>8}")
    print(f"{'=' * 70}")

    # Generate comparison report
    html_path = generate_comparison_html(results)
    print(f"\nComparison report: {html_path}")

    # Write sweep summary
    summary_path = SWEEP_DIR / "sweep-summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"Sweep summary:     {summary_path}")


if __name__ == "__main__":
    main()
