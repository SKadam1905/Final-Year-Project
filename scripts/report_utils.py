"""Shared helpers for the Phase 5 result scripts: tables as CSV + Markdown, figures as 200-dpi PNG."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # no window; works headless


def write_table(stem: Path, rows: Sequence[dict], title: str, notes: Sequence[str] = ()) -> None:
    """Write `<stem>.csv` and `<stem>.md` from a list of dicts with the same keys."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0]) if rows else []
    with open(stem.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = [f"# {title}", ""]
    if rows:
        lines += ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
        lines += ["| " + " | ".join(str(row[c]) for c in columns) + " |" for row in rows]
    if notes:
        lines += [""] + [f"- {note}" for note in notes]
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_figure(fig, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
