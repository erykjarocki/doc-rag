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
        return "\u2014"
    if delta < -threshold:
        return f"{delta:+.4f} **FAIL**"
    if delta < 0:
        return f"{delta:+.4f} \u26a0"
    return f"+{delta:.4f}"


def main():
    baseline_raw = load_json(BASELINE_PATH)
    current = load_json(REPORT_PATH)

    # Handle both simple {recall_at_2: ...} and CI format {metrics: {...}}
    if "metrics" in baseline_raw:
        baseline = baseline_raw["metrics"]
    else:
        baseline = baseline_raw

    metrics = ["recall_at_2", "recall_at_4", "precision_at_2", "precision_at_4", "mrr"]
    labels = ["Recall@2", "Recall@4", "Precision@2", "Precision@4", "MRR"]

    has_regression = False
    lines = []

    # Standard baseline comparison
    rows = []
    for metric, label in zip(metrics, labels):
        b = baseline.get(metric, 0)
        c = current["metrics"].get(metric, 0)
        delta = c - b
        icon = delta_icon(delta, REGRESSION_THRESHOLD)
        if "FAIL" in icon:
            has_regression = True
        rows.append((label, f"{b:.4f}", f"{c:.4f}", icon))

    lines.append("## Eval Report \u2014 Baseline Comparison\n")
    lines.append("| Metric | Baseline | Current | Delta |")
    lines.append("|--------|----------|---------|-------|")
    for label, b, c, icon in rows:
        lines.append(f"| {label} | {b} | {c} | {icon} |")

    # Pipeline comparison (two-stage)
    pipeline = current.get("pipeline_comparison")
    if pipeline and "before" in pipeline and "after" in pipeline:
        before = pipeline["before"]
        after = pipeline["after"]
        lines.append("")
        lines.append("## Pipeline Comparison: Bi-Encoder \u2192 Cross-Encoder\n")
        lines.append("| Metric | Bi-Encoder (before) | +Rerank (after) | Delta |")
        lines.append("|--------|--------------------:|----------------:|------:|")
        for metric, label in zip(metrics, labels):
            b = before.get(metric, 0)
            a = after.get(metric, 0)
            d = a - b
            if abs(d) < 0.001:
                icon = "\u2014"
            elif d > 0:
                icon = f"+{d:.4f}"
            else:
                icon = f"{d:.4f}"
            lines.append(f"| {label} | {b:.4f} | {a:.4f} | {icon} |")

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
