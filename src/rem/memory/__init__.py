"""Path A: Memory compaction channel (working + episodic memory) for REM."""

from rem.memory.sidecar import MemorySidecar, MemorySidecarServer
from rem.memory.selector import (
    LexicalSelector,
    PackedLexicalSelector,
    RecencySelector,
    SparseChronologicalSelector,
)

__all__ = [
    "MemorySidecar", "MemorySidecarServer", "RecencySelector",
    "LexicalSelector", "PackedLexicalSelector", "SparseChronologicalSelector",
]
