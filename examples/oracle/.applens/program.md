# Oracle AutoResearch Program

Goal:
Find one testable market hypothesis, run only approved fake or read-only checks, and produce evidence a human can review.

Allowed:
- read committed workload files
- run allowlisted commands
- write artifacts under `.applens/artifacts`
- propose memory under `.applens/memory/proposed`

Blocked:
- live trades
- broker orders
- credentials
- system changes
- downloads or driver changes

Loop:
1. Read the workload profile, metrics, probes, and last blackboard events.
2. Propose one small experiment.
3. Select one allowlisted command only when needed.
4. Record the command result, failure, or blocked action.
5. Write a short critique and next step.

