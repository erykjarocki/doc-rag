"""Generate an HTML comparison report from sweep results.

Reads per-config eval-report.json files and produces a side-by-side
comparison with bar charts, per-query breakdown, and winner highlights.
"""

import json
from pathlib import Path

CSS = """
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  max-width: 1100px; margin: 0 auto; padding: 20px;
  background: #f8f9fa; color: #1a1a1a;
}
h1 { border-bottom: 2px solid #dee2e6; padding-bottom: 10px; }
h2 { margin-top: 32px; color: #333; }
.subtitle { color: #666; font-size: 0.9em; margin-top: -12px; }

/* Summary table */
.summary-table {
  width: 100%; border-collapse: collapse; margin: 20px 0;
  background: white; border-radius: 8px; overflow: hidden;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
.summary-table th {
  background: #343a40; color: white; padding: 12px 16px;
  text-align: left; font-weight: 600;
}
.summary-table td {
  padding: 10px 16px; border-bottom: 1px solid #eee;
}
.summary-table tr:last-child td { border-bottom: none; }
.summary-table tr:hover { background: #f1f3f5; }
.winner { background: #d4edda !important; font-weight: 600; }
.worst { background: #fff3cd !important; }
.status-pass { color: #28a745; font-weight: 600; }
.status-fail { color: #dc3545; font-weight: 600; }

/* Metric bars */
.metric-section { margin: 24px 0; }
.metric-bar-row {
  display: flex; align-items: center; margin: 8px 0; gap: 12px;
}
.metric-bar-label {
  width: 140px; font-size: 0.85em; font-weight: 500; text-align: right;
  flex-shrink: 0;
}
.metric-bar-track {
  flex: 1; height: 28px; background: #e9ecef; border-radius: 4px;
  overflow: hidden; position: relative;
}
.metric-bar-fill {
  height: 100%; border-radius: 4px; transition: width 0.3s;
  display: flex; align-items: center; padding-left: 8px;
  font-size: 0.8em; font-weight: 600; color: white;
  min-width: 40px;
}
.metric-bar-fill.best { box-shadow: 0 0 0 2px #28a745; }
.bar-recall { background: #0d6efd; }
.bar-precision { background: #198754; }
.bar-mrr { background: #6f42c1; }

/* Per-query comparison */
.query-compare {
  background: white; border: 1px solid #dee2e6; border-radius: 8px;
  padding: 16px; margin: 16px 0;
}
.query-compare h3 {
  font-size: 0.95em; margin: 0 0 8px 0; color: #333;
}
.query-compare .meta { font-size: 0.8em; color: #666; margin-bottom: 8px; }
.query-grid {
  display: grid; gap: 8px;
}
.query-grid.cols-2 { grid-template-columns: 1fr 1fr; }
.query-grid.cols-3 { grid-template-columns: 1fr 1fr 1fr; }
.query-grid.cols-4 { grid-template-columns: 1fr 1fr 1fr 1fr; }
.query-grid.cols-5 { grid-template-columns: 1fr 1fr 1fr 1fr 1fr; }
.query-cell {
  padding: 8px 12px; border-radius: 6px; font-size: 0.85em;
  border: 1px solid #e9ecef;
}
.query-cell.best { border-color: #28a745; background: #d4edda; }
.query-cell .config-name { font-weight: 600; margin-bottom: 4px; font-size: 0.8em; color: #555; }
.query-cell .scores { display: flex; gap: 12px; }
.query-cell .score-item { display: flex; flex-direction: column; align-items: center; }
.query-cell .score-val { font-weight: 700; font-size: 1.1em; }
.query-cell .score-label { font-size: 0.7em; color: #888; }

/* Config details */
.config-cards {
  display: flex; gap: 12px; flex-wrap: wrap; margin: 16px 0;
}
.config-card {
  background: white; border: 1px solid #dee2e6; border-radius: 8px;
  padding: 12px 16px; flex: 1; min-width: 180px;
}
.config-card h4 { margin: 0 0 6px 0; font-size: 0.9em; }
.config-card .env-tag {
  display: inline-block; background: #e9ecef; padding: 2px 8px;
  border-radius: 4px; font-size: 0.75em; margin: 2px 4px 2px 0;
  font-family: monospace;
}
.config-card .desc { font-size: 0.8em; color: #666; }

.footer {
  margin-top: 40px; padding-top: 16px; border-top: 1px solid #dee2e6;
  font-size: 0.8em; color: #888; text-align: center;
}
"""


def _esc(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _bar(metric_key, results, max_val=1.0):
    """Build metric bar rows for a given metric."""
    rows = []
    values = []
    for r in results:
        v = r.get("metrics", {}).get(metric_key, 0)
        values.append(v)
    best_val = max(values) if values else 0

    for r, v in zip(results, values):
        pct = (v / max_val * 100) if max_val > 0 else 0
        is_best = v == best_val and v > 0
        cls = "metric-bar-fill best" if is_best else "metric-bar-fill"
        color_cls = {
            "recall_at_2": "bar-recall",
            "precision_at_2": "bar-precision",
            "mrr": "bar-mrr",
        }.get(metric_key, "bar-recall")
        rows.append(
            f'<div class="metric-bar-row">\n'
            f'  <div class="metric-bar-label">{_esc(r["name"])}</div>\n'
            f'  <div class="metric-bar-track">\n'
            f'    <div class="{cls} {color_cls}" style="width: {pct:.0f}%">{v:.2f}</div>\n'
            f'  </div>\n'
            f"</div>"
        )
    return "\n".join(rows)


def generate_sweep_report(results: list[dict], out_path: Path):
    """Generate the HTML comparison report.

    Args:
        results: List of sweep result dicts from run_sweep.py.
        out_path: Path to write the HTML file.
    """
    configs_html = ""
    for r in results:
        tags = " ".join(
            f'<span class="env-tag">{k}={v}</span>'
            for k, v in r.get("env", {}).items()
        )
        configs_html += (
            f'<div class="config-card">\n'
            f"  <h4>{_esc(r['name'])}</h4>\n"
            f'  <div class="desc">{_esc(r.get("description", ""))}</div>\n'
            f'  <div style="margin-top: 6px">{tags}</div>\n'
            f"</div>\n"
        )

    # Summary table
    metric_keys = [("recall_at_2", "Recall@2"), ("precision_at_2", "Precision@2"), ("mrr", "MRR")]
    best_per_metric = {}
    for key, _ in metric_keys:
        vals = [(r["name"], r.get("metrics", {}).get(key, 0)) for r in results]
        if vals:
            best_per_metric[key] = max(vals, key=lambda x: x[1])[0]

    worst_per_metric = {}
    for key, _ in metric_keys:
        vals = [(r["name"], r.get("metrics", {}).get(key, 0)) for r in results]
        if vals:
            worst_per_metric[key] = min(vals, key=lambda x: x[1])[0]

    table_rows = ""
    for r in results:
        m = r.get("metrics", {})
        cells = []
        for key, label in metric_keys:
            v = m.get(key, 0)
            cls = ""
            if r["name"] == best_per_metric.get(key):
                cls = "winner"
            elif r["name"] == worst_per_metric.get(key):
                cls = "worst"
            cells.append(f'<td class="{cls}">{v:.2f}</td>')

        status_cls = "status-pass" if r["status"] == "pass" else "status-fail"
        table_rows += (
            f"<tr>\n"
            f"  <td><strong>{_esc(r['name'])}</strong></td>\n"
            f"  {''.join(cells)}\n"
            f'  <td>{r.get("num_queries", 0)}</td>\n'
            f'  <td>{r.get("elapsed_s", 0):.1f}s</td>\n'
            f'  <td class="{status_cls}">{r["status"]}</td>\n'
            f"</tr>\n"
        )

    # Metric bar charts
    bar_sections = ""
    for key, label in metric_keys:
        bar_sections += (
            f'<div class="metric-section">\n'
            f"  <h3>{label}</h3>\n"
            f"  {_bar(key, results)}\n"
            f"</div>\n"
        )

    # Per-query comparison
    all_queries = {}  # query_text -> {config_name: query_data}
    for r in results:
        report_path = out_path.parent / r["name"] / "eval-report.json"
        if not report_path.exists():
            continue
        try:
            data = json.loads(report_path.read_text())
        except (json.JSONDecodeError, KeyError):
            continue
        for q in data.get("queries", []):
            qt = q["query"]
            if qt not in all_queries:
                all_queries[qt] = {"relevant_docs": q.get("relevant_documents", [])}
            all_queries[qt][r["name"]] = q

    n_configs = len(results)
    grid_cls = f"cols-{min(n_configs, 5)}"

    query_sections = ""
    for qt, qdata in all_queries.items():
        relevant = qdata.get("relevant_docs", [])
        cells = []
        best_recall = 0
        best_name = None
        for r in results:
            if r["name"] in qdata:
                q = qdata[r["name"]]
                rec = q.get("recall_at_k", 0)
                if rec > best_recall:
                    best_recall = rec
                    best_name = r["name"]

        for r in results:
            if r["name"] not in qdata:
                cell = (
                    f'<div class="query-cell">'
                    f'<div class="config-name">{_esc(r["name"])}</div>'
                    f'<div class="scores">No data</div></div>'
                )
                cells.append(cell)
                continue
            q = qdata[r["name"]]
            rec = q.get("recall_at_k", 0)
            prec = q.get("precision_at_k", 0)
            rr = q.get("reciprocal_rank", 0)
            is_best = r["name"] == best_name and best_recall > 0
            cls = "query-cell best" if is_best else "query-cell"
            cells.append(
                f'<div class="{cls}">\n'
                f'  <div class="config-name">{_esc(r["name"])}</div>\n'
                f'  <div class="scores">\n'
                f'    <div class="score-item">'
                f'<span class="score-val">{rec:.1f}</span>'
                f'<span class="score-label">Rec</span></div>\n'
                f'    <div class="score-item">'
                f'<span class="score-val">{prec:.1f}</span>'
                f'<span class="score-label">Prec</span></div>\n'
                f'    <div class="score-item">'
                f'<span class="score-val">{rr:.2f}</span>'
                f'<span class="score-label">MRR</span></div>\n'
                f"  </div>\n"
                f"</div>"
            )

        query_sections += (
            f'<div class="query-compare">\n'
            f'  <h3>&ldquo;{_esc(qt)}&rdquo;</h3>\n'
            f'  <div class="meta">Relevant: {", ".join(_esc(d) for d in relevant)}</div>\n'
            f'  <div class="query-grid {grid_cls}">\n'
            f"    {''.join(cells)}\n"
            f"  </div>\n"
            f"</div>\n"
        )

    # Winner determination
    overall_scores = {}
    for r in results:
        m = r.get("metrics", {})
        # Simple sum of metrics as overall score
        overall_scores[r["name"]] = (
            m.get("recall_at_2", 0) + m.get("precision_at_2", 0) + m.get("mrr", 0)
        )
    winner = max(overall_scores, key=overall_scores.get) if overall_scores else "N/A"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Eval Sweep Comparison — DOC RAG</title>
<style>{CSS}</style>
</head>
<body>
<h1>DOC RAG — Eval Sweep Comparison</h1>
<p class="subtitle">Compared {len(results)} configurations across {len(all_queries)} queries</p>

<h2>Configurations</h2>
<div class="config-cards">{configs_html}</div>

<h2>Summary</h2>
<table class="summary-table">
<tr>
  <th>Config</th>
  <th>Recall@2</th>
  <th>Precision@2</th>
  <th>MRR</th>
  <th>Queries</th>
  <th>Time</th>
  <th>Status</th>
</tr>
{table_rows}
</table>

<h2>Metric Comparison</h2>
{bar_sections}

<h2>Per-Query Breakdown</h2>
{query_sections}

<div class="footer">
  Generated by <code>make eval-sweep</code> &mdash;
  Best overall: <strong>{_esc(winner)}</strong>
</div>
</body>
</html>"""

    out_path.write_text(html)
    print(f"Sweep comparison written to {out_path}")
