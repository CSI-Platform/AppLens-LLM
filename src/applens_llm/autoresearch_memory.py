from __future__ import annotations

import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from applens_llm.autoresearch_layout import workload_paths


def write_memory_proposal(root: Path, *, run_id: str, title: str, body: str) -> Path:
    paths = workload_paths(root)
    paths.memory_proposed_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{_slug(title)}-{run_id}.md"
    proposal = paths.memory_proposed_dir / filename
    proposal.write_text(
        f"# {title}\n\n"
        f"Run: `{run_id}`\n\n"
        f"Created: {_utc_now()}\n\n"
        f"{body.strip()}\n",
        encoding="utf-8",
    )
    return proposal


def promote_memory(root: Path, proposal: Path) -> Path:
    paths = workload_paths(root)
    proposal_resolved = proposal.resolve()
    proposed_root = paths.memory_proposed_dir.resolve()
    if proposal_resolved != proposed_root and proposed_root not in proposal_resolved.parents:
        raise ValueError("memory proposal must live under .applens/memory/proposed")
    if not proposal.exists():
        raise ValueError(f"memory proposal does not exist: {proposal}")
    paths.memory_wiki_dir.mkdir(parents=True, exist_ok=True)
    target = paths.memory_wiki_dir / proposal.name
    shutil.copy2(proposal, target)
    return target


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug[:64] or "memory"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

