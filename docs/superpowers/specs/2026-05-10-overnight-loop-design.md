# Overnight Loop Design

## Purpose

`overnight-loop` turns the one-shot fast-to-deep handoff into a bounded run the user can leave unattended. It uses the existing runtime lanes and blackboard ledger instead of introducing a new orchestration framework.

## Behavior

- Resolve a fast lane and deep lane from a runtime lane config.
- Start lanes unless `--skip-start` is set.
- Cycle through inline prompts and/or a prompt file.
- For each iteration, append a `task`, run the fast lane, append a `handoff`, run the deep lane, and append a `verdict`.
- Stop on `max_iterations`, `max_runtime_minutes`, or the first lane failure.
- Optionally continue after failures with `--continue-on-failure`.
- Stop started lanes unless `--keep-running` is set.
- Write a summary JSON with attempted iterations, completed iterations, stop reason, and per-iteration response summaries.

## Safety

The loop does not change drivers, services, firewall rules, startup entries, firmware, or user data. It only starts user-configured local model runtimes, calls local endpoints, and writes ignored evidence files under `out/`.

## Data Ownership

The blackboard JSONL is the detailed audit trail. The summary JSON is the quick status artifact. Both remain local-only unless explicitly sanitized.
