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
.comparison {
  background: white; border: 2px solid #1a1a2e;
  border-radius: 8px; margin: 24px 0; overflow: hidden;
}
.comparison-header {
  padding: 14px 16px; background: #1a1a2e; color: white;
  font-weight: 600; font-size: 15px;
  display: flex; justify-content: space-between; align-items: center;
}
.comparison-header .badge {
  background: #4caf50; color: white; padding: 2px 8px;
  border-radius: 4px; font-size: 12px; font-weight: 600;
}
.comparison table {
  width: 100%; border-collapse: collapse; font-size: 14px;
}
.comparison th {
  padding: 10px 16px; text-align: left; background: #e9ecef;
  font-weight: 600; border-bottom: 2px solid #dee2e6;
}
.comparison td {
  padding: 10px 16px; border-bottom: 1px solid #eee;
}
.comparison tr:last-child td { border-bottom: none; }
.comparison .delta-up { color: #28a745; font-weight: 600; }
.comparison .delta-down { color: #dc3545; font-weight: 600; }
.comparison .delta-flat { color: #6c757d; }
.comparison .summary {
  padding: 12px 16px; background: #f0f7ff; border-top: 1px solid #dee2e6;
  font-size: 13px; color: #333;
}
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
.source { color: #555; font-family: monospace; font-size: 12px; }
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
.relevant-docs { font-size: 13px; color: #555; }
.rerank-detail { margin: 16px 0; }
.rerank-detail-header {
  padding: 10px 16px; background: #f0f7ff; border: 1px solid #b8daff;
  border-radius: 8px 8px 0 0; font-weight: 600; font-size: 14px;
  color: #004085;
}
.rerank-columns {
  display: grid; grid-template-columns: 1fr 1fr; gap: 0;
  border: 1px solid #dee2e6; border-top: none; border-radius: 0 0 8px 8px;
  overflow: hidden;
}
.rerank-col { padding: 0; }
.rerank-col-header {
  padding: 8px 12px; font-weight: 600; font-size: 12px;
  text-transform: uppercase; letter-spacing: 0.5px;
}
.rerank-col:first-child .rerank-col-header { background: #e9ecef; color: #495057; }
.rerank-col:last-child .rerank-col-header { background: #d4edda; color: #155724; }
.rerank-item {
  padding: 6px 12px; border-top: 1px solid #f0f0f0;
  font-size: 12px; display: flex; align-items: baseline; gap: 8px;
}
.rerank-item:first-of-type { border-top: none; }
.rerank-rank { font-weight: 700; min-width: 18px; color: #333; }
.rerank-source { color: #555; min-width: 28px; font-family: monospace; font-size: 11px; }
.rerank-score { color: #666; font-family: monospace; font-size: 11px; }
.rerank-arrow { font-weight: 700; min-width: 22px; text-align: center; }
.rerank-arrow.up { color: #28a745; }
.rerank-arrow.down { color: #dc3545; }
.rerank-arrow.flat { color: #6c757d; }
.rerank-text {
  flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  color: #555; font-family: monospace; font-size: 11px;
}
.rerank-item.relevant-row { background: #f0fff0; }
.rerank-item.irrelevant-row { background: #fff5f5; }
"""


def _load_baseline():
    """Load baseline metrics if available."""
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


def _pipeline_comparison_html(pipeline):
    """Build the two-stage pipeline comparison section."""
    if not pipeline or "before" not in pipeline or "after" not in pipeline:
        return ""

    before = pipeline["before"]
    after = pipeline["after"]

    improvements = []
    for key, label in [
        ("recall_at_2", "Recall@2"),
        ("precision_at_2", "Precision@2"),
        ("mrr", "MRR"),
    ]:
        b = before.get(key, 0)
        a = after.get(key, 0)
        d = a - b
        if d > 0.01:
            improvements.append(f"{label} +{d:.0%}")
        elif d < -0.01:
            improvements.append(f"{label} {d:.0%}")

    summary_text = (
        "Reranking improves retrieval precision by rescoring candidates with a cross-encoder."
    )
    if improvements:
        summary_text = f"Improvements: {', '.join(improvements)}."

    html = (
        '<div class="comparison">\n'
        '  <div class="comparison-header">\n'
        "    <span>Pipeline: Bi-Encoder \u2192 Cross-Encoder Reranking</span>\n"
        '    <span class="badge">RERANK ENABLED</span>\n'
        "  </div>\n"
        "  <table>\n"
        "    <thead><tr>"
        "<th>Metric</th>"
        "<th>Stage 1: Bi-Encoder</th>"
        "<th>Stage 2: +Rerank</th>"
        "<th>Delta</th>"
        "</tr></thead>\n"
        "    <tbody>\n"
    )

    for key, label in [
        ("recall_at_2", "Recall@2"),
        ("precision_at_2", "Precision@2"),
        ("mrr", "MRR"),
    ]:
        b = before.get(key, 0)
        a = after.get(key, 0)
        d = a - b
        if abs(d) < 0.005:
            delta_cls, delta_text = "delta-flat", "\u2014"
        elif d > 0:
            delta_cls, delta_text = "delta-up", f"+{d:.2f}"
        else:
            delta_cls, delta_text = "delta-down", f"{d:.2f}"
        html += (
            f"      <tr><td><strong>{label}</strong></td>"
            f"<td>{b:.2f}</td><td>{a:.2f}</td>"
            f'<td class="{delta_cls}">{delta_text}</td></tr>\n'
        )

    html += f'    </tbody>\n  </table>\n  <div class="summary">{summary_text}</div>\n</div>\n\n'
    return html


def _rerank_detail_html(detail):
    """Build the per-query reranking detail section."""
    if not detail:
        return ""

    bi_top8 = detail.get("bi_encoder_top8", [])
    reranked_top8 = detail.get("reranked_top8", [])
    rank_changes = detail.get("rank_changes", [])

    if not bi_top8 and not reranked_top8:
        return ""

    rc_by_source = {}
    for rc in rank_changes:
        rc_by_source[rc.get("source_file", rc.get("page", "?"))] = rc

    html = (
        '<div class="rerank-detail">\n'
        '  <div class="rerank-detail-header">'
        "Reranking Detail \u2014 Bi-Encoder top 8 \u2192 Cross-Encoder top 8"
        "</div>\n"
        '  <div class="rerank-columns">\n'
    )

    # Left column: Before (bi-encoder)
    html += '    <div class="rerank-col">\n'
    html += '      <div class="rerank-col-header">Before (Bi-Encoder)</div>\n'
    for frag in bi_top8:
        cls = "relevant-row" if frag["is_relevant"] else "irrelevant-row"
        tag_cls = "relevant-tag" if frag["is_relevant"] else "irrelevant-tag"
        tag_text = "REL" if frag["is_relevant"] else "NOT REL"
        html += (
            f'      <div class="rerank-item {cls}">'
            f'<span class="rerank-rank">#{frag["rank"]}</span>'
            f'<span class="rerank-source">{_escape_html(frag["source_file"])}</span>'
            f'<span class="rerank-score">{frag["bi_score"]:.4f}</span>'
            f'<span class="tag {tag_cls}">{tag_text}</span>'
            f'<span class="rerank-text">{_escape_html(frag["text_preview"])}</span>'
            f"</div>\n"
        )
    if not bi_top8:
        html += '      <div class="rerank-item" style="color:#999">No data</div>\n'
    html += "    </div>\n"

    # Right column: After (cross-encoder)
    html += '    <div class="rerank-col">\n'
    html += '      <div class="rerank-col-header">After (Cross-Encoder)</div>\n'
    for frag in reranked_top8:
        cls = "relevant-row" if frag["is_relevant"] else "irrelevant-row"
        tag_cls = "relevant-tag" if frag["is_relevant"] else "irrelevant-tag"
        tag_text = "REL" if frag["is_relevant"] else "NOT REL"

        source = frag.get("source_file", "?")
        rc = rc_by_source.get(source)
        if rc and rc["delta"] > 0:
            arrow = f'<span class="rerank-arrow up">\u2191{rc["delta"]}</span>'
        elif rc and rc["delta"] < 0:
            arrow = f'<span class="rerank-arrow down">\u2193{abs(rc["delta"])}</span>'
        else:
            arrow = '<span class="rerank-arrow flat">\u2014</span>'

        html += (
            f'      <div class="rerank-item {cls}">'
            f'<span class="rerank-rank">#{frag["rank"]}</span>'
            f"{arrow}"
            f'<span class="rerank-source">{_escape_html(frag["source_file"])}</span>'
            f'<span class="rerank-score">bi:{frag["bi_score"]:.4f} ce:{frag["ce_score"]:.4f}</span>'
            f'<span class="tag {tag_cls}">{tag_text}</span>'
            f'<span class="rerank-text">{_escape_html(frag["text_preview"])}</span>'
            f"</div>\n"
        )
    if not reranked_top8:
        html += '      <div class="rerank-item" style="color:#999">No data</div>\n'
    html += "    </div>\n"

    html += "  </div>\n</div>\n\n"
    return html


def _escape_html(text):
    """Escape HTML special characters."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def generate():
    if not REPORT_JSON.exists():
        print(f"No {REPORT_JSON} found \u2014 skipping report generation.")
        return

    data = json.loads(REPORT_JSON.read_text())
    metrics = data["metrics"]
    queries = data["queries"]
    baseline = _load_baseline()
    rerank_detail_list = data.get("rerank_detail", [])
    rerank_detail_by_query = {d["query"]: d for d in rerank_detail_list}

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
        "<title>Eval Report \u2014 DOC RAG</title>\n"
        f"<style>{CSS}</style>\n"
        "</head>\n<body>\n"
        "<h1>DOC RAG \u2014 Eval Report</h1>\n\n"
    )

    # Pipeline comparison (prominent, at the top)
    pipeline = data.get("pipeline_comparison")
    html += _pipeline_comparison_html(pipeline)

    # Standard metrics cards
    html += (
        '<div class="metrics">\n'
        f'  <div class="metric {recall_cls}">\n'
        f'    <div class="value">{metrics["recall_at_2"]:.2f}</div>\n'
        f"    {_delta_html(metrics['recall_at_2'], bl_recall, 'recall_at_2')}\n"
        '    <div class="label">Recall@2</div>\n  </div>\n'
        f'  <div class="metric {precision_cls}">\n'
        f'    <div class="value">{metrics["precision_at_2"]:.2f}</div>\n'
        f"    {_delta_html(metrics['precision_at_2'], bl_precision, 'precision_at_2')}\n"
        '    <div class="label">Precision@2</div>\n  </div>\n'
        f'  <div class="metric {mrr_cls}">\n'
        f'    <div class="value">{metrics["mrr"]:.2f}</div>\n'
        f"    {_delta_html(metrics['mrr'], bl_mrr, 'mrr')}\n"
        '    <div class="label">MRR</div>\n  </div>\n'
        '  <div class="metric pass">\n'
        f'    <div class="value">{len(queries)}</div>\n'
        '    <div class="label">Queries</div>\n  </div>\n'
        "</div>\n"
    )

    for q in queries:
        relevant_docs = q.get("relevant_documents", [])
        recall = q.get("recall_at_k", 0)
        precision = q.get("precision_at_k", 0)

        html += (
            '<div class="query-card">\n'
            f'  <div class="query-header">&ldquo;{q["query"]}&rdquo;</div>\n'
            '  <div class="query-meta">'
            f"Relevant: {', '.join(relevant_docs)} &nbsp;|&nbsp; "
            f"Recall@2: {recall:.2f} &nbsp;|&nbsp; "
            f"Precision@2: {precision:.2f}</div>\n"
        )

        for frag in q["retrieved_fragments"]:
            is_rel = frag["is_relevant"]
            cls = "relevant" if is_rel else "irrelevant"
            tag_cls = "relevant-tag" if is_rel else "irrelevant-tag"
            tag_text = "RELEVANT" if is_rel else "NOT RELEVANT"
            text = frag["text"][:300]
            if len(frag["text"]) > 300:
                text += "\u2026"

            html += (
                f'  <div class="fragment {cls}">\n'
                '    <div class="fragment-header">\n'
                "      <span>\n"
                f'        <span class="rank">#{frag["rank"]}</span>\n'
                f'        <span class="source">{_escape_html(frag["source_file"])}</span>\n'
                f'        <span class="tag {tag_cls}">{tag_text}</span>\n'
                "      </span>\n"
                f'      <span class="score">score: {frag["score"]:.4f}</span>\n'
                "    </div>\n"
                f'    <div class="text">{_escape_html(text)}</div>\n'
                "  </div>\n"
            )

        # Add per-query reranking detail if available
        detail = rerank_detail_by_query.get(q["query"])
        html += _rerank_detail_html(detail)

        html += "</div>\n"

    html += "</body></html>"
    REPORT_HTML.write_text(html)
    # Also write index.html for GitHub Pages
    (REPORT_HTML.parent / "index.html").write_text(html)
    print(f"Report written to {REPORT_HTML}")


if __name__ == "__main__":
    generate()
