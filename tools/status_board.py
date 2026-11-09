#!/usr/bin/env python3
"""
Render the README status table from results/<tier>/by_model/*.csv counts.

Updates the block between <!-- status:start --> and <!-- status:end --> in
README.md. Run from project root: python3 tools/status_board.py
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import sys
from pathlib import Path

# Long chain-of-thought traces in the `thinking` column can exceed Python's
# default 128 KB per-field cap, so bump it before any DictReader/reader call.
csv.field_size_limit(sys.maxsize)

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
RESULTS = ROOT / "results"

sys.path.insert(0, str(ROOT / "src"))
from veltharrow.config import (  # noqa: E402
    DOCUMENTS,
    model_specs_for,
    retrieval_conditions,
)

START_MARKER = "<!-- status:start -->"
END_MARKER = "<!-- status:end -->"


def expected_rows(tier: str) -> int:
    """Rows expected for a full run: specs × conditions × questions."""
    n_specs = len(model_specs_for(tier))
    n_conditions = len(retrieval_conditions(tier))
    n_questions = 0
    for doc in DOCUMENTS:
        try:
            n_questions += len(json.loads(doc.questions_path.read_text(encoding="utf-8")))
        except Exception:
            n_questions += 100  # fall back to nominal
    return n_specs * n_conditions * n_questions


def collected_rows(tier: str) -> tuple[int, str]:
    """Count data rows across results/<tier>/by_model/*.csv, plus newest mtime."""
    by_model = RESULTS / tier / "by_model"
    if not by_model.exists():
        return 0, " - "
    total = 0
    newest = 0.0
    for csv_path in sorted(by_model.glob("*.csv")):
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            next(reader, None)  # header
            total += sum(1 for _ in reader)
        newest = max(newest, csv_path.stat().st_mtime)
    last = dt.date.fromtimestamp(newest).isoformat() if newest else " - "
    return total, last


def render_table(tiers: list[str]) -> str:
    """Build the markdown status table (specs, rows expected/collected, last update) for the
    given tiers."""
    lines = [
        "| Tier   | Specs | Rows expected | Rows collected | Last update |",
        "|--------|------:|--------------:|---------------:|-------------|",
    ]
    for tier in tiers:
        n_specs = len(model_specs_for(tier))
        exp = expected_rows(tier)
        got, last = collected_rows(tier)
        lines.append(f"| {tier:<6} | {n_specs:>5} | {exp:>13,} | {got:>14,} | {last:<11} |")
    return "\n".join(lines)


def main() -> int:
    """Render the status table and splice it into the marked block of README.md (printing it
    instead if the README or markers are missing); returns a process exit code."""
    table = render_table(["medium", "small"])
    if not README.exists():
        print(table)
        return 0
    text = README.read_text(encoding="utf-8")
    if START_MARKER not in text or END_MARKER not in text:
        print(f"README is missing {START_MARKER} / {END_MARKER} markers; printing table only:\n")
        print(table)
        return 1
    before, _, rest = text.partition(START_MARKER)
    _, _, after = rest.partition(END_MARKER)
    new = f"{before}{START_MARKER}\n{table}\n{END_MARKER}{after}"
    if new != text:
        README.write_text(new, encoding="utf-8")
        print(f"Updated status table in {README.relative_to(ROOT)}:\n\n{table}")
    else:
        print("No change.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
