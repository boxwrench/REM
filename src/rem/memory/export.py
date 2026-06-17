from pathlib import Path
from rem.memory.tiers import MemoryState


def export_episodes(state: MemoryState, out_dir: str | Path) -> None:
    """Exports SpanSummary objects as timestamped markdown files.

    Writes one markdown file per summary in state.summaries.
    Includes YAML frontmatter with covers_turn_ids, created_at, and source.
    Idempotent: skips writing if the target file already exists.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for summary in state.summaries:
        if not summary.covers_turn_ids:
            continue

        min_turn = min(summary.covers_turn_ids)
        max_turn = max(summary.covers_turn_ids)
        filename = f"episode_{min_turn}_{max_turn}.md"
        filepath = out_path / filename

        if filepath.exists():
            continue

        # Write file with YAML frontmatter
        frontmatter = (
            "---\n"
            f"covers_turn_ids: {summary.covers_turn_ids}\n"
            f"created_at: {summary.created_at}\n"
            "source: path-a\n"
            "---\n\n"
            f"{summary.text}\n"
        )

        # Atomic write
        tmp_filepath = filepath.with_suffix(".tmp")
        try:
            tmp_filepath.write_text(frontmatter, encoding="utf-8")
            tmp_filepath.replace(filepath)
        except Exception as e:
            if tmp_filepath.exists():
                tmp_filepath.unlink()
            raise e

