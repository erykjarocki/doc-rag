"""Generate a detailed HTML report from eval-report.json."""

import json
from pathlib import Path

REPORT_JSON = Path(__file__).parent / "eval-report.json"
REPORT_HTML = Path(__file__).parent / "eval-report.html"
BASELINE_JSON = Path(__file__).parent / "eval-baseline.json"

CSS = """
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  max-width: 960px; margin: 0 auto; padding: 20px;
  background: #f8f9fa; color: #1a1a1a;
}
h1 { border-bottom: 2px solid #dee2e6; padding-bottom: 10px; }
.metrics { display: flex; gap: 16px; margin: 20px 0; flex-wrap: wrap; }
.metric {
  background: white; border: 1px solid #dee2e6;
  border-radius: 8px; padding: 16px 24px; text-align: center;
  min-width: 120px;
}
.metric .value { font-size: 28px; font-weight: 700; }
.metric .label { font-size: 13px; color: #666; margin-top: 4px; }
.metric.pass .value { color: #28a745; }
.metric.warn .value { color: #ffc107; }
.delta { font-size: 13px; font-weight: 600; margin-top: 2px; }
.delta.up { color: #28a745; }
.delta.down { color: #dc3545; }
.delta.flat { color: #6c757d; }
.delta.none { color: #999; font-style: italic; }
.query-card {
  background: white; border: 1px solid #dee2e6;
  border-radius: 8px; margin: 16px 0; overflow: hidden;
}
.query-header { padding: 12px 16px; background: #e9ecef; font-weight: 600; }
.query-meta { padding: 8px 16px; font-size: 13px; color: #555; }
.fragment { padding: 12px 16px; border-top: 1px solid #eee; }
.fragment-header {
  display: flex; justify-content: space-between;
  margin-bottom: 6px; font-size: 13px;
}
.rank { font-weight: 700; }
.score { color: #555; }
.page { color: #555; }
.relevant { background: #e6ffe6; }
.irrelevant { background: #fff0f0; }
.tag {
  display: inline-block; padding: 1px 6px; border-radius: 4px;
  font-size: 11px; font-weight: 600; margin-left: 6px;
}
.tag.relevant-tag { background: #28a745; color: white; }
.tag.irrelevant-tag { background: #dc3545; color: white; }
.text {
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
  font-size: 12px; line-height: 1.5; white-space: pre-wrap; color: #333;
}
.relevant-pages { font-size: 13px; color: #555; }
"""


def _load_baseline():
    """Load baseline metrics if available.

    Handles both simple format {recall_at_2: ...} and CI format
    {queries: [...], metrics: {recall_at_2: ...}}.
    """
    if not BASELINE_JSON.exists():
        return None
    try:
        data = json.loads(BASELINE_JSON.read_text())
        if "metrics" in data:
            return data["metrics"]
        if "recall_at_2" in data:
            return data
        return None
    except (json.JSONDecodeError, KeyError):
        return None


def _delta_html(current, baseline_val, key):
    """Return HTML string showing delta from baseline."""
    if baseline_val is None:
        return '<div class="delta none">no baseline</div>'
    diff = current - baseline_val
    if abs(diff) < 0.005:
        return '<div class="delta flat">=&nbsp;0.00</div>'
    sign = "+" if diff > 0 else ""
    cls = "up" if diff > 0 else "down"
    return f'<div class="delta {cls}">{sign}{diff:.2f}</div>'


def generate():
    if not REPORT_JSON.exists():
        print(f"No {REPORT_JSON} found — skipping report generation.")
        return

    data = json.loads(REPORT_JSON.read_text())
    metrics = data["metrics"]
    queries = data["queries"]
    baseline = _load_baseline()

    def metric_class(key, threshold):
        val = metrics[key]
        if val >= threshold:
            return "pass"
        return "warn"

    recall_cls = metric_class("recall_at_2", 0.8)
    precision_cls = metric_class("precision_at_2", 0.5)
    mrr_cls = metric_class("mrr", 0.7)

    bl_recall = baseline.get("recall_at_2") if baseline else None
    bl_precision = baseline.get("precision_at_2") if baseline else None
    bl_mrr = baseline.get("mrr") if baseline else None

    html = (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        "<title>Eval Report — PDF RAG</title>\n"
        f"<style>{CSS}</style>\n"
        "</head>\n<body>\n"
        "<h1>PDF RAG — Eval Report</h1>\n\n"
        '<div class="metrics">\n'
        f'  <div class="metric {recall_cls}">\n'
        f'    <div class="value">{metrics["recall_at_2"]:.2f}</div>\n'
        f'    {_delta_html(metrics["recall_at_2"], bl_recall, "recall_at_2")}\n'
        '    <div class="label">Recall@2</div>\n  </div>\n'
        f'  <div class="metric {precision_cls}">\n'
        f'    <div class="value">{metrics["precision_at_2"]:.2f}</div>\n'
        f'    {_delta_html(metrics["precision_at_2"], bl_precision, "precision_at_2")}\n'
        '    <div class="label">Precision@2</div>\n  </div>\n'
        f'  <div class="metric {mrr_cls}">\n'
        f'    <div class="value">{metrics["mrr"]:.2f}</div>\n'
        f'    {_delta_html(metrics["mrr"], bl_mrr, "mrr")}\n'
        '    <div class="label">MRR</div>\n  </div>\n'
        '  <div class="metric pass">\n'
        f'    <div class="value">{len(queries)}</div>\n'
        '    <div class="label">Queries</div>\n  </div>\n'
        "</div>\n"
    )

    for q in queries:
        relevant_pages = q["relevant_pages"]
        recall = q.get("recall_at_k", 0)
        precision = q.get("precision_at_k", 0)

        html += (
            '<div class="query-card">\n'
            f'  <div class="query-header">&ldquo;{q["query"]}&rdquo;</div>\n'
            "  <div class=\"query-meta\">"
            f"Relevant pages: {relevant_pages} &nbsp;|&nbsp; "
            f"Recall@2: {recall:.2f} &nbsp;|&nbsp; "
            f'Precision@2: {precision:.2f}</div>\n'
        )

        for frag in q["retrieved_fragments"]:
            is_rel = frag["is_relevant"]
            cls = "relevant" if is_rel else "irrelevant"
            tag_cls = "relevant-tag" if is_rel else "irrelevant-tag"
            tag_text = "RELEVANT" if is_rel else "NOT RELEVANT"
            text = frag["text"][:300]
            if len(frag["text"]) > 300:
                text += "…"

            html += (
                f'  <div class="fragment {cls}">\n'
                '    <div class="fragment-header">\n'
                "      <span>\n"
                f'        <span class="rank">#{frag["rank"]}</span>\n'
                f'        <span class="page">p.{frag["start_page"]}</span>\n'
                f'        <span class="tag {tag_cls}">{tag_text}</span>\n'
                "      </span>\n"
                f'      <span class="score">score: {frag["score"]:.4f}</span>\n'
                "    </div>\n"
                f'    <div class="text">{text}</div>\n'
                "  </div>\n"
            )

        html += "</div>\n"

    html += "</body></html>"
    REPORT_HTML.write_text(html)
    # Also write index.html for GitHub Pages
    (REPORT_HTML.parent / "index.html").write_text(html)
    print(f"Report written to {REPORT_HTML}")


if __name__ == "__main__":
    generate()
