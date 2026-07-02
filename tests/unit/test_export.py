"""Unit tests for the episodic summary export component."""

import time
from rem.memory.tiers import MemoryState, SpanSummary
from rem.memory.export import export_episodes


def test_export_episodes_idempotence(tmp_path):
    """Asserts that exporting summaries produces files with correct frontmatter and is idempotent."""
    out_dir = tmp_path / "episodes"

    # Construct memory state with summaries
    summaries = [
        SpanSummary(
            covers_turn_ids=[1, 2, 3],
            text="Summary of turns 1 to 3",
            tokens=10,
            created_at=123.45,
        ),
        SpanSummary(
            covers_turn_ids=[4, 5],
            text="Summary of turns 4 to 5",
            tokens=5,
            created_at=128.9,
        ),
    ]
    state = MemoryState(summaries=summaries)

    # First export
    export_episodes(state, out_dir)

    file1 = out_dir / "episode_1_3.md"
    file2 = out_dir / "episode_4_5.md"

    assert file1.exists()
    assert file2.exists()

    # Verify content and YAML frontmatter structure
    content1 = file1.read_text(encoding="utf-8")
    assert content1.startswith("---\n")
    assert "covers_turn_ids: [1, 2, 3]\n" in content1
    assert "created_at: 123.45\n" in content1
    assert "source: path-a\n" in content1
    assert "---\n\nSummary of turns 1 to 3\n" in content1

    content2 = file2.read_text(encoding="utf-8")
    assert "covers_turn_ids: [4, 5]\n" in content2
    assert "created_at: 128.9\n" in content2
    assert "source: path-a\n" in content2
    assert "---\n\nSummary of turns 4 to 5\n" in content2

    # Record initial modification times
    mtime1_initial = file1.stat().st_mtime
    mtime2_initial = file2.stat().st_mtime

    # Wait briefly to ensure file modification times would change if rewritten
    time.sleep(0.01)

    # Second export (re-export)
    export_episodes(state, out_dir)

    # Verify modification times are unchanged (demonstrating idempotence)
    assert file1.stat().st_mtime == mtime1_initial
    assert file2.stat().st_mtime == mtime2_initial
