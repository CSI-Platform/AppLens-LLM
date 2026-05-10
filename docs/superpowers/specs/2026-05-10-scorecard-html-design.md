# Scorecard HTML Design

## Purpose

The JSON `model-fit-scorecard` remains the source of truth, but users need a better inspection surface than Markdown. A static HTML report gives AppLens-LLM a sortable local dashboard without adding a server or database.

## Inputs

- Required scorecard JSON.
- Optional experiment comparison JSON files.

## Output

`model-fit-html` writes one static HTML file under `out/scorecards/` by default. The file contains:

- machine and scorecard header
- summary metrics
- sortable/filterable model ranking table
- sortable experiment comparison table

The report must not embed local model paths unless they are already present in the input scorecard. Committed examples should stay JSON-only unless a future sanitized HTML fixture is useful.

## Data Ownership

HTML is a view, not a contract. AppLens integrations should read scorecard JSON and generate UI from that. The static HTML file is for local review and human comparison.

## Testing

Tests cover direct HTML generation and the CLI command. The tests assert that the ranking table, experiment comparison verdicts, warnings, and client-side sort function are present.
