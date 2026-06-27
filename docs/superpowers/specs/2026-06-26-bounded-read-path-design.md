# Bounded Read Path — Step 0 Design (fit-to-budget selection)

**Date:** 2026-06-26
**Status:** Design, awaiting approval
**Parent spec:** [`docs/REM-memory-architecture-spec.md`](../../REM-memory-architecture-spec.md) §3 (reserved memory region)
**Battery context:** [`bench/battery/FINDINGS.md`](../../../bench/battery/FINDINGS.md) "Verdict and next" — option A

## 1. Goal and scope

REM has no bounded read path. `src/rem/memory/assembler.py` renders the facts
ledger "in full, never truncated/omitted" (line 80) and appends every verbatim
turn; its only size mechanism is a hard cap that **raises** `ContextLimitExceeded`,
not one that **fits** memory to a budget. On the five oldest-gold knowledge-update
items, assembled memory is 37k–58k tokens, exceeding both the assemble ceiling and
the answering model's own ~32–40k window (`gemma4-it:e2b` returns HTTP 400 "Max
length reached!"). REM scores 0/5 as `context_overflow` — it cannot read its own
memory, so the battery cannot yet produce a memory-quality verdict.

This spec covers **Step 0 only**: build a bounded, fit-to-budget read path and
validate it NPU-free against the one already-persisted compacted state
(`bench/battery/diag_031748ae_w64k_state.json`, 836K). Step 0 turns
`context_overflow` into an actual answer and proves the fit mechanism before any
NPU is spent. Later steps (per-item state persistence in the battery; one 6h
ingest; iterate read-path variants against saved states to get the failure mix)
are out of scope here and follow once Step 0 passes.

## 2. Non-goals (explicitly deferred)

- **Embeddings / vector retrieval.** No embedding or similarity machinery exists in
  the tree today; `semantic_block` in the assembler is a reserved-but-empty hook.
  A query-relevant retrieval selector is a later variant, not Step 0.
- **Write/ingest path changes.** Step 0 is read-side only. The system-under-test
  compaction is untouched.
- **The 950-entry ledger bloat.** Acknowledged growth problem (ledger + summaries
  are 93% of memory, both unbounded with conversation length). Addressed separately;
  the read-path fit makes the system answerable in the meantime.
- **Judged-correctness pass on 031748ae.** The item's gold rests on a dataset
  inference (the "4 engineers" is an outing headcount), so correctness here is
  uninterpretable at n=1. Deferred to the five-item mix.

## 3. Architecture

### 3.1 Selector interface (load-bearing decision)

```python
class MemorySelector(Protocol):
    def select(self, state: MemoryState, question: str, budget_tokens: int) -> MemoryState: ...
```

The selector returns a **filtered `MemoryState`** (a subset: fewer summaries, fewer
ledger entries, verbatim retained), which flows through the **existing**
`assemble()` / `assemble_messages()`. The selector decides *what is included*; the
assembler still decides *how it is rendered*, so all quarantine / stale-value /
slot-suppression logic in `assembler.py` is reused unchanged.

Variants (lexical, structure-aware) are new classes implementing the same protocol.
Because they consume a persisted `MemoryState`, every variant after Step 0 can be
evaluated NPU-free against saved states.

### 3.2 First implementation: `RecencySelector`

Fit order, stopping before any item would exceed `budget_tokens`:

1. Always keep **current-state ledger slots** — the newest active value per slot key
   (small; REM's "what is true now").
2. Add **episodic summaries newest→oldest** until the next would exceed budget.
3. Add remaining **ledger entries by recency** with leftover budget.
4. Keep **verbatim turns** (already bounded, ~2.5k).

No scoring against the question. Deterministic. It is the honest baseline the
five-item failure mix is measured against.

### 3.3 Budget

New setting `REM_READ_FIT_TOKENS`, default **28000** — headroom under the ~32k
point where the answering model returns HTTP 400. Distinct from `max_context_tokens`
(the assemble *ceiling*, which stays as a safety raise); the fit budget is the
active target the selector aims at.

### 3.4 Harness: `--load-state` on `diagnose_memory.py`

`diagnose_memory.py` currently always re-ingests (~75 min NPU). Add
`--load-state <path>` to deserialize a persisted `MemoryState` JSON and skip ingest.
Step 0 runs entirely from `diag_031748ae_w64k_state.json` with zero NPU.

## 4. Data flow

```
persisted MemoryState JSON
   └─(--load-state, NPU-free)→ MemoryState
        └─ RecencySelector.select(state, question, 28000) → fitted MemoryState
             └─ assemble_messages(fitted, …) → messages (≤ 28k, no raise)
                  └─ answering model → answer (no HTTP 400)
```

## 5. Step 0 success criterion (agreed bar)

Against the loaded 031748ae state, Step 0 PASSES when:

- `fitted_tokens <= REM_READ_FIT_TOKENS` (no overflow), and
- the model returns an answer (no `ContextLimitExceeded`, no HTTP 400), and
- both `"4 engineers"` and `"5 engineers"` are present in the fitted slice (gold
  survives the eviction).

Judged correctness is **not** required (n=1 ambiguity, see §2).

## 6. Testing (TDD, NPU-free)

- `RecencySelector` on synthetic `MemoryState`s: fits to budget; always keeps newest
  slot per key; deterministic ordering; degrades gracefully when a single tier alone
  exceeds budget.
- `--load-state` round-trip: serialize → load → identical assembled tokens.
- Gold-survival assertion on the real persisted state fixture.

All runnable under `PYTHONPATH=.:src python3 -m pytest -m 'not npu'`.

## 7. Decision log (for backtracking)

Recorded so a later failure-mix result can cheaply reverse a choice.

| # | Decision | Chosen | Rejected alternative | Reverse if… |
|---|---|---|---|---|
| D1 | Selector return type | Filtered `MemoryState`, reuse `assemble()` | Selector emits final string | Quarantine/rendering needs to differ per-selector (then push rendering into the selector). |
| D2 | First strategy | `RecencySelector` (no question scoring) | Lexical retrieval first | Five-item mix shows misses are retrieval-recall (right facts not selected), not size/structure → promote LexicalSelector. |
| D3 | Selection is pluggable | Strategy protocol from the start | Hardcode recency, generalize later | Only ever one strategy is needed (unlikely; whole point is to swap variants NPU-free). |
| D4 | Fit budget | `REM_READ_FIT_TOKENS=28000`, separate from assemble ceiling | Reuse `max_context_tokens` as both | Model window changes, or a single knob proves simpler in practice. |
| D5 | Step 0 bar | Fit + gold-survives + answers; correctness deferred | Require correct judgement on 031748ae | A token-matched, unambiguous item replaces 031748ae as the Step 0 fixture. |
| D6 | Scope | Read-side only; ledger bloat deferred | Bound the ledger in the same change | Ledger growth alone blocks fitting even after eviction (i.e. current-state slots + verbatim already exceed budget). |

## 8. Open question carried forward

This unblocks measurement; it does not choose the architecture. After Step 0, the
five-item failure mix decides between continuing to tune the read path (retrieval/
eviction variants) and committing to the graph-resident read path (Phase 1 on the
existing Phase 0 store). The mix — size vs. retrieval-recall vs. temporal/structure
— is the evidence that choice waits on.
