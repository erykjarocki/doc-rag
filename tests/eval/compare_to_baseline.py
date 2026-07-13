"""Compare eval-report.json against eval-baseline.json.

Outputs a markdown table and exits non-zero if any metric regressed.
"""

import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).parent
BASELINE_PATH = EVAL_DIR / "eval-baseline.json"
REPORT_PATH = EVAL_DIR / "eval-report.json"

REGRESSION_THRESHOLD = 0.05  # fail if metric drops by more than this


def load_json(path):
    if not path.exists():
        print(f"ERROR: {path} not found")
        sys.exit(1)
    return json.loads(path.read_text())


def delta_icon(delta, threshold):
    if abs(delta) < 0.001:
        return "—"
    if delta < -threshold:
        return f"{delta:+.4f} **FAIL**"
    if delta < 0:
        return f"{delta:+.4f} ⚠"
    return f"+{delta:.4f}"


def main():
    baseline_raw = load_json(BASELINE_PATH)
    current = load_json(REPORT_PATH)

    # Handle both simple {recall_at_2: ...} and CI format {metrics: {...}}
    if "metrics" in baseline_raw:
        baseline = baseline_raw["metrics"]
    else:
        baseline = baseline_raw

    metrics = ["recall_at_2", "precision_at_2", "mrr"]
    labels = ["Recall@2", "Precision@2", "MRR"]

    has_regression = False
    rows = []

    for metric, label in zip(metrics, labels):
        b = baseline.get(metric, 0)
        c = current["metrics"].get(metric, 0)
        delta = c - b
        icon = delta_icon(delta, REGRESSION_THRESHOLD)
        if "FAIL" in icon:
            has_regression = True
        rows.append((label, f"{b:.4f}", f"{c:.4f}", icon))

    # Build markdown
    lines = [
        "## Eval Report — Baseline Comparison\n",
        "| Metric | Baseline | Current | Delta |",
        "|--------|----------|---------|-------|",
    ]
    for label, b, c, icon in rows:
        lines.append(f"| {label} | {b} | {c} | {icon} |")

    lines.append("")
    if has_regression:
        lines.append(
            f"**Regression detected** (threshold: {REGRESSION_THRESHOLD}). "
            "Investigate before merging."
        )
    else:
        lines.append("No regressions detected.")

    md = "\n".join(lines)
    print(md)

    # Write to file for CI to pick up
    (EVAL_DIR / "comparison.md").write_text(md)

    if has_regression:
        sys.exit(1)


if __name__ == "__main__":
    main()
