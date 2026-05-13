from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any


def build_scorecard_html(
    *,
    scorecard: dict[str, Any],
    experiment_comparisons: list[dict[str, Any]] | None = None,
    title: str | None = None,
) -> str:
    comparisons = experiment_comparisons or []
    page_title = title or f"Model Fit Scorecard: {scorecard['machine']['label']}"
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{_text(page_title)}</title>",
            _style(),
            "</head>",
            "<body>",
            '<main class="page">',
            _header(scorecard, page_title),
            _summary(scorecard),
            _ranking_table(scorecard),
            _comparison_table(comparisons),
            _script(),
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def write_scorecard_html(
    *,
    scorecard_path: Path,
    output_path: Path,
    experiment_comparison_paths: list[Path] | None = None,
    title: str | None = None,
) -> str:
    scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
    comparisons = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in experiment_comparison_paths or []
    ]
    html = build_scorecard_html(scorecard=scorecard, experiment_comparisons=comparisons, title=title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html + "\n", encoding="utf-8")
    return html


def _header(scorecard: dict[str, Any], title: str) -> str:
    machine = scorecard["machine"]
    platform = machine["platform"]
    return f"""
<section class="header">
  <div>
    <p class="eyebrow">AppLens-LLM</p>
    <h1>{_text(title)}</h1>
    <p class="subtle">{_text(machine["label"])} | {_text(platform["cpu"])} | {_text(platform["gpu"])} | {_text(platform["ram_gb"])} GB RAM</p>
  </div>
  <div class="meta">
    <div><span>Scorecard</span><strong>{_text(scorecard["scorecard_id"])}</strong></div>
    <div><span>Created</span><strong>{_text(scorecard["created_at"])}</strong></div>
  </div>
</section>
""".strip()


def _summary(scorecard: dict[str, Any]) -> str:
    evidence = scorecard["evidence"]
    rankings = scorecard["rankings"]
    observed = sum(1 for row in rankings if row.get("confidence") == "observed")
    inferred = sum(1 for row in rankings if row.get("confidence") == "inferred")
    top = rankings[0]
    return f"""
<section class="metrics">
  {_metric("Top model", top["display_name"], f"{top['fit_score']}/100")}
  {_metric("Observed models", observed, "benchmark-backed")}
  {_metric("Capability records", evidence.get("capability_record_count", 0), "applens-local-v1")}
  {_metric("Evidence", evidence["benchmark_record_count"], f"{evidence['experiment_summary_count']} experiment summaries")}
</section>
""".strip()


def _metric(label: str, value: Any, detail: Any) -> str:
    return f"""
<article class="metric">
  <span>{_text(label)}</span>
  <strong>{_text(value)}</strong>
  <small>{_text(detail)}</small>
</article>
""".strip()


def _ranking_table(scorecard: dict[str, Any]) -> str:
    rows = []
    for ranking in scorecard["rankings"]:
        lane = ranking["best_lane"]
        evidence = ranking["evidence"]
        rows.append(
            "<tr>"
            f"<td data-sort-value=\"{ranking['rank']}\">{_text(ranking['rank'])}</td>"
            f"<td>{_text(ranking['display_name'])}<small>{_text(ranking['model_id'])}</small></td>"
            f"<td data-sort-value=\"{ranking['fit_score']}\"><strong>{_text(ranking['fit_score'])}</strong><small>{_text(ranking['score_band'])}</small></td>"
            f"<td>{_text(ranking['recommended_role'])}</td>"
            f"<td>{_text(lane['lane_id'])}<small>{_text(lane['backend'])} / {_text(_join(lane.get('accelerator_ids', [])))}</small></td>"
            f"<td>{_text(ranking['confidence'])}<small>{_text(evidence.get('source', 'unknown'))}</small></td>"
            f"<td data-sort-value=\"{evidence.get('avg_latency_ms', 0)}\">{_text(evidence.get('avg_latency_ms', 'n/a'))}</td>"
            f"<td data-sort-value=\"{evidence.get('capability_score_pct', 0)}\">{_text(evidence.get('capability_score_pct', 'n/a'))}<small>{_text(_join(evidence.get('thinking_modes', []), empty='unknown'))}</small></td>"
            f"<td data-sort-value=\"{evidence.get('recommended_context_tokens', 0)}\">{_text(_context_label(evidence))}<small>{_text(_context_detail(evidence))}</small></td>"
            f"<td>{_text(_join(ranking.get('blockers', []), empty='none'))}</td>"
            f"<td>{_text(ranking['next_benchmark'])}</td>"
            "</tr>"
        )
    return f"""
<section class="section">
  <div class="section-head">
    <div>
      <h2>Model Fit Ranking</h2>
      <p>Click headers to sort. Use the filter for model, role, backend, blockers, or confidence.</p>
    </div>
    <input id="tableFilter" type="search" placeholder="Filter rows" oninput="filterTables()">
  </div>
  <div class="table-wrap">
    <table data-sort-table>
      <thead>
        <tr>
          <th onclick="sortTable(this)">Rank</th>
          <th onclick="sortTable(this)">Model</th>
          <th onclick="sortTable(this)">Score</th>
          <th onclick="sortTable(this)">Role</th>
          <th onclick="sortTable(this)">Lane</th>
          <th onclick="sortTable(this)">Confidence</th>
          <th onclick="sortTable(this)">Avg latency ms</th>
          <th onclick="sortTable(this)">Capability</th>
          <th onclick="sortTable(this)">Recommended context</th>
          <th onclick="sortTable(this)">Blockers</th>
          <th>Next benchmark</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
  </div>
</section>
""".strip()


def _comparison_table(comparisons: list[dict[str, Any]]) -> str:
    if not comparisons:
        return """
<section class="section">
  <h2>Experiment Comparisons</h2>
  <p class="subtle">No experiment comparison files were attached.</p>
</section>
""".strip()
    rows = []
    for comparison in comparisons:
        baseline = comparison["baseline"]
        candidate = comparison["candidate"]
        deltas = comparison.get("deltas", {})
        fast = deltas.get("fast", {})
        deep = deltas.get("deep", {})
        warnings = (comparison.get("comparability") or {}).get("warnings", [])
        rows.append(
            "<tr>"
            f"<td>{_text(baseline['experiment_id'])}<small>{_text((baseline.get('driver') or {}).get('branch', 'unknown'))}</small></td>"
            f"<td>{_text(candidate['experiment_id'])}<small>{_text((candidate.get('driver') or {}).get('branch', 'unknown'))}</small></td>"
            f"<td>{_text(comparison.get('verdict', 'unknown'))}</td>"
            f"<td data-sort-value=\"{fast.get('latency_ms_delta', 0)}\">{_text(fast.get('latency_ms_delta', 'n/a'))}<small>{_text(fast.get('latency_ms_delta_pct', 'n/a'))}%</small></td>"
            f"<td data-sort-value=\"{deep.get('latency_ms_delta', 0)}\">{_text(deep.get('latency_ms_delta', 'n/a'))}<small>{_text(deep.get('latency_ms_delta_pct', 'n/a'))}%</small></td>"
            f"<td>{_text(_join(warnings, empty='none'))}</td>"
            "</tr>"
        )
    return f"""
<section class="section">
  <h2>Experiment Comparisons</h2>
  <div class="table-wrap">
    <table data-sort-table>
      <thead>
        <tr>
          <th onclick="sortTable(this)">Baseline</th>
          <th onclick="sortTable(this)">Candidate</th>
          <th onclick="sortTable(this)">Verdict</th>
          <th onclick="sortTable(this)">Fast delta ms</th>
          <th onclick="sortTable(this)">Deep delta ms</th>
          <th onclick="sortTable(this)">Warnings</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
  </div>
</section>
""".strip()


def _style() -> str:
    return """
<style>
:root {
  color-scheme: light;
  --bg: #f7f8fa;
  --text: #20242a;
  --muted: #5d6673;
  --line: #d9dee7;
  --panel: #ffffff;
  --accent: #2364aa;
  --ok: #18794e;
  --warn: #9a6700;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: "Segoe UI", Arial, sans-serif;
  line-height: 1.45;
}
.page {
  width: min(1360px, calc(100vw - 32px));
  margin: 0 auto;
  padding: 24px 0 40px;
}
.header {
  display: flex;
  justify-content: space-between;
  gap: 24px;
  align-items: flex-end;
  border-bottom: 1px solid var(--line);
  padding-bottom: 18px;
}
.eyebrow {
  margin: 0 0 6px;
  color: var(--accent);
  font-weight: 700;
  text-transform: uppercase;
  font-size: 12px;
}
h1, h2, p { margin-top: 0; }
h1 { margin-bottom: 8px; font-size: 30px; }
h2 { margin-bottom: 4px; font-size: 20px; }
.subtle, .section-head p { color: var(--muted); }
.meta {
  display: grid;
  gap: 8px;
  min-width: 260px;
}
.meta div, .metric {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px;
}
.meta span, .metric span, .metric small, td small {
  display: block;
  color: var(--muted);
  font-size: 12px;
}
.metrics {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin: 20px 0;
}
.metric strong { display: block; font-size: 22px; margin: 4px 0; }
.section {
  margin-top: 22px;
}
.section-head {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: end;
}
input[type="search"] {
  width: min(360px, 100%);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 10px 12px;
  font: inherit;
}
.table-wrap {
  overflow-x: auto;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
}
table {
  width: 100%;
  border-collapse: collapse;
  min-width: 980px;
}
th, td {
  border-bottom: 1px solid var(--line);
  padding: 10px 12px;
  text-align: left;
  vertical-align: top;
}
th {
  background: #eef2f7;
  color: #243244;
  cursor: pointer;
  user-select: none;
  white-space: nowrap;
}
tbody tr:last-child td { border-bottom: 0; }
tbody tr:hover { background: #f5f8fc; }
td strong { color: var(--ok); }
@media (max-width: 820px) {
  .header, .section-head { display: block; }
  .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .meta { min-width: 0; }
}
</style>
""".strip()


def _script() -> str:
    return """
<script>
function sortTable(header) {
  const table = header.closest("table");
  const index = Array.from(header.parentNode.children).indexOf(header);
  const body = table.tBodies[0];
  const direction = header.dataset.direction === "asc" ? "desc" : "asc";
  Array.from(header.parentNode.children).forEach((cell) => cell.dataset.direction = "");
  header.dataset.direction = direction;
  const rows = Array.from(body.rows);
  rows.sort((a, b) => {
    const left = sortValue(a.cells[index]);
    const right = sortValue(b.cells[index]);
    if (typeof left === "number" && typeof right === "number") {
      return direction === "asc" ? left - right : right - left;
    }
    return direction === "asc"
      ? String(left).localeCompare(String(right))
      : String(right).localeCompare(String(left));
  });
  rows.forEach((row) => body.appendChild(row));
}
function sortValue(cell) {
  const raw = cell.dataset.sortValue || cell.innerText.trim();
  const numeric = Number(raw);
  return Number.isFinite(numeric) && raw !== "" ? numeric : raw.toLowerCase();
}
function filterTables() {
  const query = document.getElementById("tableFilter").value.toLowerCase();
  document.querySelectorAll("tbody tr").forEach((row) => {
    row.style.display = row.innerText.toLowerCase().includes(query) ? "" : "none";
  });
}
</script>
""".strip()


def _join(values: list[Any], *, empty: str = "unknown") -> str:
    cleaned = [str(value) for value in values if value not in {None, ""}]
    return ", ".join(cleaned) if cleaned else empty


def _text(value: Any) -> str:
    return escape(str(value), quote=True)


def _format_tokens(value: Any) -> str:
    if not isinstance(value, (int, float)) or int(value) <= 0:
        return "n/a"
    return f"{int(value):,}"


def _context_label(evidence: dict[str, Any]) -> str:
    status = evidence.get("context_evidence_status")
    if status == "advertised_unproven":
        return "unproven"
    if status == "observed_limited":
        return "quality needed"
    return _format_tokens(evidence.get("recommended_context_tokens", 0))


def _context_detail(evidence: dict[str, Any]) -> str:
    status = evidence.get("context_evidence_status", "unknown")
    tested = _format_tokens(evidence.get("max_tested_context_tokens", 0))
    advertised = _format_tokens(evidence.get("advertised_context_tokens", 0))
    if status == "advertised_unproven":
        return f"advertised {advertised}; taper needed"
    if status == "observed_limited":
        return f"load-tested {tested}; quality needed"
    return f"tested {tested} / advertised {advertised}"
