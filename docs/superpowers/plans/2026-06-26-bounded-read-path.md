# Bounded Read Path — Step 0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give REM a bounded, fit-to-budget read path (a pluggable `MemorySelector`, recency-first) and validate it NPU-free against the one already-persisted compacted state, turning `context_overflow` into an actual answer.

**Architecture:** A `MemorySelector` protocol returns a *filtered* `MemoryState` (fewer summaries/ledger entries, verbatim kept) that flows through the existing `assemble()`, so all quarantine/rendering logic is reused. `RecencySelector` protects the current-state slots (newest active entry per slot key) plus verbatim, then fills the remaining budget with summaries newest→oldest and remaining active ledger entries newest-first. `diagnose_memory.py` gains a `--load-state` flag so Step 0 runs from the persisted state with no 75-min ingest.

**Tech Stack:** Python 3.12, pydantic / pydantic-settings, pytest. No new dependencies.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-06-26-bounded-read-path-design.md`. Decision log §7 governs; do not silently deviate.
- Token heuristic is `len(text) // 4` (`rem.memory.tiers.count_tokens`); all budgeting uses it, never a real tokenizer.
- Settings use env prefix `REM_` (`pydantic_settings`); the new knob is `REM_READ_FIT_TOKENS`, default `28000`.
- Selector returns a `MemoryState`; it MUST NOT render strings or duplicate quarantine logic (spec D1).
- First strategy is recency-only, no question-scoring (spec D2). Lexical/structure selectors are out of scope.
- Read-side only; do not touch ingest/compaction or attempt to bound the ledger writer (spec D6).
- Unit tests run NPU-free: `PYTHONPATH=.:src python3 -m pytest -m 'not npu'`.
- NPU discipline: the only NPU call in Step 0 is a single brief answer inference; never run it while another NPU job (e.g. a battery ingest) is active. No 75-min ingest is performed.

---

### Task 1: `read_fit_tokens` setting + `RecencySelector`

> **STATUS: COMPLETE** (commits `b62eda5..e453946`, diff verified). 6 selector tests pass; full NPU-free suite 149 passed. Resume execution at Task 2.

**Files:**
- Modify: `src/rem/config.py:35-40` (add one setting)
- Create: `src/rem/memory/selector.py`
- Test: `tests/unit/test_selector.py`

**Interfaces:**
- Consumes: `MemoryState`, `SpanSummary`, `count_tokens` from `rem.memory.tiers`; `FactEntry`, `FactsLedger` from `rem.memory.facts_ledger`; `Settings` from `rem.config`.
- Produces:
  - `Settings.read_fit_tokens: int = 28000`
  - `rem.memory.selector.MemorySelector` (Protocol) with `select(self, state: MemoryState, question: str, budget_tokens: int) -> MemoryState`
  - `rem.memory.selector.RecencySelector` implementing it
  - `rem.memory.selector.SELECTOR_RESERVE_TOKENS: int = 512`

- [ ] **Step 1: Add the setting**

In `src/rem/config.py`, in the "Compaction parameters (Path A)" block, after `max_context_tokens: int = 32000` (line 39) add:

```python
    # Read path (bounded assembly). Target for the fitted read budget; kept under
    # the answering model's ~32-40k window so it never returns HTTP 400. Distinct
    # from max_context_tokens, which is the assemble safety ceiling.
    read_fit_tokens: int = 28000
```

- [ ] **Step 2: Write the failing tests**

Create `tests/unit/test_selector.py`:

```python
"""Unit tests for the bounded read-path selector (NPU-free)."""

from rem.config import Settings
from rem.memory.assembler import assemble
from rem.memory.facts_ledger import FactEntry, FactsLedger
from rem.memory.selector import RecencySelector
from rem.memory.tiers import MemoryState, SpanSummary, Turn, count_tokens


def _big_state(n_summaries: int = 200, n_free_facts: int = 200) -> MemoryState:
    """A state whose summaries + ledger far exceed any small budget."""
    turns = [Turn(role="user", content=f"recent turn {i}", turn_id=900 + i, tokens=4)
             for i in range(4)]
    summaries = [
        SpanSummary(covers_turn_ids=[i], text=f"summary number {i} " * 12, tokens=40)
        for i in range(n_summaries)
    ]
    entries = [
        FactEntry(kind="entity", text=f"free fact {i} about the system " * 3,
                  source_turn_id=i, status="active")
        for i in range(n_free_facts)
    ]
    return MemoryState(turns=turns, summaries=summaries, ledger=FactsLedger(entries=entries))


def test_recency_selector_fits_budget():
    state = _big_state()
    question = "what is the current state?"
    budget = 3000
    fitted = RecencySelector().select(state, question, budget)
    assembled = assemble(fitted, system="", task=question)
    assert count_tokens(assembled) <= budget


def test_recency_selector_keeps_newest_per_slot_and_both_distinct_slots():
    # Two DIFFERENT slot keys must both survive (the 031748ae gold shape):
    # team.size newest = "5 engineers"; team_members.count newest = "4 engineers".
    entries = [
        FactEntry(kind="number", text="team size is 5 engineers", source_turn_id=5,
                  status="active", slot_key="team.size", slot_value="5 engineers"),
        FactEntry(kind="number", text="team size is 3 engineers", source_turn_id=2,
                  status="active", slot_key="team.size", slot_value="3 engineers"),
        FactEntry(kind="number", text="outing had 4 engineers", source_turn_id=12,
                  status="active", slot_key="team_members.count", slot_value="4 engineers"),
    ]
    state = MemoryState(turns=[], summaries=[],
                        ledger=FactsLedger(entries=entries))
    fitted = RecencySelector().select(state, "headcount?", budget_tokens=2000)
    text = assemble(fitted, system="", task="headcount?")
    assert "5 engineers" in text          # newest of team.size kept
    assert "4 engineers" in text          # newest of team_members.count kept
    assert "3 engineers" not in text      # older same-slot value dropped


def test_recency_selector_excludes_stale():
    entries = [
        FactEntry(kind="entity", text="active fact alpha", source_turn_id=10, status="active"),
        FactEntry(kind="entity", text="stale fact beta", source_turn_id=3, status="stale"),
    ]
    state = MemoryState(turns=[], summaries=[], ledger=FactsLedger(entries=entries))
    fitted = RecencySelector().select(state, "q", budget_tokens=2000)
    text = assemble(fitted, system="", task="q")
    assert "active fact alpha" in text
    assert "stale fact beta" not in text


def test_recency_selector_is_deterministic():
    state = _big_state()
    a = RecencySelector().select(state, "q", 3000)
    b = RecencySelector().select(state, "q", 3000)
    assert assemble(a, system="", task="q") == assemble(b, system="", task="q")


def test_recency_selector_prefers_newest_summaries():
    state = _big_state(n_summaries=200, n_free_facts=0)
    fitted = RecencySelector().select(state, "q", budget_tokens=1500)
    kept_ids = {min(s.covers_turn_ids) for s in fitted.summaries}
    # Newest summary (id 199) kept; oldest (id 0) dropped under a tight budget.
    assert 199 in kept_ids
    assert 0 not in kept_ids


def test_read_fit_tokens_default():
    assert Settings().read_fit_tokens == 28000
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `PYTHONPATH=.:src python3 -m pytest tests/unit/test_selector.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'rem.memory.selector'` (and `test_read_fit_tokens_default` fails until Step 1 is in).

- [ ] **Step 4: Write the selector**

Create `src/rem/memory/selector.py`:

```python
"""Bounded read path: fit a compacted MemoryState to a token budget.

The assembler renders the facts ledger in full and every summary unbounded, so on
long conversations the assembled memory exceeds the answering model's window. A
MemorySelector chooses which summaries / ledger entries to keep so the assembled
memory fits a budget, returning a FILTERED MemoryState that flows through the
existing assemble() (the selector decides what is in; the assembler still decides
how it is rendered, reusing quarantine/stale handling).
"""
from __future__ import annotations

from typing import Protocol

from rem.memory.facts_ledger import FactEntry, FactsLedger
from rem.memory.tiers import MemoryState, SpanSummary, count_tokens

# Reserve inside the budget for section headers + the answer the model must still
# generate. The question's own tokens are reserved separately (it is variable).
SELECTOR_RESERVE_TOKENS = 512


class MemorySelector(Protocol):
    """Chooses which tiers of a compacted state to keep so it fits a budget."""

    def select(self, state: MemoryState, question: str, budget_tokens: int) -> MemoryState:
        ...


def _summary_cost(s: SpanSummary) -> int:
    rendered = s.rendered_text if s.rendered_text is not None else s.text
    return count_tokens(f"- {rendered}")


def _entry_cost(e: FactEntry) -> int:
    status = "" if e.status == "active" else " stale"
    return count_tokens(f"- [{e.kind}{status}] {e.text} (Turn {e.source_turn_id})")


class RecencySelector:
    """Keeps current-state slots + verbatim, then fills the budget newest-first.

    Priority (highest first):
      1. verbatim turns (already bounded) and the newest active entry per slot_key
         (the current-state facts) -- always kept;
      2. episodic summaries, newest -> oldest;
      3. remaining active ledger entries (no slot_key), newest-first.
    Stale entries are never included. No scoring against the question. Deterministic.
    """

    def select(self, state: MemoryState, question: str, budget_tokens: int) -> MemoryState:
        budget = budget_tokens - count_tokens(question) - SELECTOR_RESERVE_TOKENS

        # --- Protected tier: verbatim + newest active entry per slot_key ---
        kept_turns = list(state.turns)
        verbatim_cost = (
            count_tokens("\n".join(f"{t.role.upper()}: {t.content}" for t in kept_turns))
            if kept_turns else 0
        )

        newest_by_slot: dict[str, FactEntry] = {}
        free_actives: list[FactEntry] = []
        for e in state.ledger.entries:
            if e.status != "active":
                continue
            if e.slot_key:
                cur = newest_by_slot.get(e.slot_key)
                if cur is None or e.source_turn_id > cur.source_turn_id:
                    newest_by_slot[e.slot_key] = e
            else:
                free_actives.append(e)

        kept_entries: list[FactEntry] = list(newest_by_slot.values())
        used = verbatim_cost + sum(_entry_cost(e) for e in kept_entries)

        # --- Tier 2: summaries newest-first ---
        kept_summaries: list[SpanSummary] = []
        for s in sorted(
            state.summaries,
            key=lambda s: max(s.covers_turn_ids) if s.covers_turn_ids else 0,
            reverse=True,
        ):
            c = _summary_cost(s)
            if used + c > budget:
                continue
            kept_summaries.append(s)
            used += c

        # --- Tier 3: remaining active ledger entries newest-first ---
        for e in sorted(free_actives, key=lambda e: e.source_turn_id, reverse=True):
            c = _entry_cost(e)
            if used + c > budget:
                continue
            kept_entries.append(e)
            used += c

        return MemoryState(
            schema_version=state.schema_version,
            turns=kept_turns,
            summaries=kept_summaries,
            ledger=FactsLedger(entries=kept_entries),
        )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `PYTHONPATH=.:src python3 -m pytest tests/unit/test_selector.py -q`
Expected: PASS (6 passed).

- [ ] **Step 6: Run the full NPU-free suite (no regressions)**

Run: `PYTHONPATH=.:src python3 -m pytest -m 'not npu' -q`
Expected: PASS — prior 143 passed plus the 6 new = 149 passed.

- [ ] **Step 7: Commit**

```bash
git add src/rem/config.py src/rem/memory/selector.py tests/unit/test_selector.py
git commit -m "REM(read): bounded read-path selector (RecencySelector) + REM_READ_FIT_TOKENS

Pluggable MemorySelector returns a filtered MemoryState reused by assemble();
RecencySelector protects current-state slots + verbatim, fills budget newest-first.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `--load-state` on `diagnose_memory.py` (skip the 75-min ingest)

> **STATUS: COMPLETE** (commit `5bad116`, diff verified). 1 load-state test passes; full NPU-free suite 150 passed. Resume execution at Task 3.

**Files:**
- Modify: `evals/battery/diagnose_memory.py` (`run` signature, state acquisition, `main` argparse)
- Test: `tests/unit/test_diagnose_load_state.py`

**Interfaces:**
- Consumes: `MemoryState.load` from `rem.memory.tiers`; `RecencContextManager` state field `cm._state`.
- Produces: `run(data, max_gold_recency, out, load_state=None, answer=True)`; new CLI flags `--load-state PATH`, `--no-answer`. When `load_state` is set, ingest is NOT called and `ingest_secs == 0.0`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_diagnose_load_state.py`:

```python
"""--load-state must skip ingest entirely (NPU-free state acquisition)."""

from pathlib import Path

from rem.memory.facts_ledger import FactEntry, FactsLedger
from rem.memory.tiers import MemoryState, Turn
from evals.battery.diagnose_memory import acquire_state


class _ExplodingCM:
    """Stands in for RemContextManager; ingest must never be called."""
    def __init__(self):
        self._state = None
    def ingest(self, *a, **k):
        raise AssertionError("ingest() must not be called when --load-state is set")


def test_acquire_state_loads_without_ingest(tmp_path: Path):
    saved = MemoryState(
        turns=[Turn(role="user", content="hi", turn_id=1, tokens=1)],
        summaries=[],
        ledger=FactsLedger(entries=[FactEntry(kind="entity", text="x", source_turn_id=1)]),
    )
    path = tmp_path / "state.json"
    saved.save(path)

    cm = _ExplodingCM()
    state, ingest_secs = acquire_state(cm, load_state=str(path), item=None, budget_tokens=1000)

    assert ingest_secs == 0.0
    assert len(state.turns) == 1
    assert cm._state is state          # cm now answers from the loaded state
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=.:src python3 -m pytest tests/unit/test_diagnose_load_state.py -q`
Expected: FAIL — `ImportError: cannot import name 'acquire_state'`.

- [ ] **Step 3: Factor state acquisition into `acquire_state`**

In `evals/battery/diagnose_memory.py`, add the import near the other `rem.memory` imports (after line 36 `from rem.memory.tiers import count_tokens`):

```python
from rem.memory.tiers import MemoryState
```

Add this function above `run` (before line 123):

```python
def acquire_state(cm, load_state, item, budget_tokens):
    """Returns (state, ingest_secs). With --load-state, loads NPU-free and skips ingest.

    Otherwise runs the real ~75-min compaction on the item's sessions.
    """
    if load_state:
        state = MemoryState.load(load_state)
        cm._state = state
        return state, 0.0
    t0 = time.time()
    cm.ingest(item.sessions, budget_tokens=budget_tokens)
    return cm._state, round(time.time() - t0, 1)
```

- [ ] **Step 4: Use it in `run` and add flags**

In `run`, replace the ingest block (current lines 139-144):

```python
    reset_extraction_stats()
    t0 = time.time()
    cm.ingest(it.sessions, budget_tokens=1000)
    assembled = cm.assemble()
    ingest_secs = round(time.time() - t0, 1)
    extraction = get_extraction_stats()
```

with:

```python
    reset_extraction_stats()
    state, ingest_secs = acquire_state(cm, load_state, it, budget_tokens=1000)
    assembled = cm.assemble()
    extraction = get_extraction_stats()  # empty/zeros when state was loaded
```

Update `run`'s signature (line 123) and `main` to thread the flag:

```python
def run(data: str, max_gold_recency: float, out: str,
        load_state: str | None = None, answer: bool = True) -> int:
```

Delete the now-duplicate `state = cm._state` line (current line 153; `state` is already bound from `acquire_state`).

In `main` (after line 245 `--out`):

```python
    ap.add_argument("--load-state", default=None,
                    help="Load a persisted MemoryState JSON and skip the 75-min ingest.")
    ap.add_argument("--no-answer", dest="answer", action="store_false",
                    help="Skip the NPU answer calls (pure-Python size/gold checks only).")
```

and change the final call (line 247):

```python
    return run(args.data, args.max_gold_recency, args.out,
               load_state=args.load_state, answer=args.answer)
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `PYTHONPATH=.:src python3 -m pytest tests/unit/test_diagnose_load_state.py -q`
Expected: PASS (1 passed).

- [ ] **Step 6: Commit**

```bash
git add evals/battery/diagnose_memory.py tests/unit/test_diagnose_load_state.py
git commit -m "REM(diag): --load-state to diagnose_memory (NPU-free state acquisition)

acquire_state() loads a persisted MemoryState and skips the ~75-min ingest, so
read-path changes are evaluated against saved states without NPU.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Selector-based fitted answer + gold-in-fitted-slice check

> **STATUS: COMPLETE** (commit `ef3358c`, diff verified). 2 fit tests pass; full NPU-free suite 152 passed. Resume execution at Task 4.

**Files:**
- Modify: `evals/battery/diagnose_memory.py` (the "Answer attempt 2 / fitted" block, ~lines 220-236; payload fields ~lines 187-191)
- Test: `tests/unit/test_diagnose_fit.py`

**Interfaces:**
- Consumes: `RecencySelector` from `rem.memory.selector`; `assemble` from `rem.memory.assembler`; `Settings.read_fit_tokens`.
- Produces: a `fit_with_selector(state, question, settings) -> tuple[str, int]` helper returning `(fitted_text, fitted_tokens)`; `gold_in_fitted(fitted_text, needles) -> dict[str, bool]`; payload gains `rem_fitted_tokens`, `gold_in_fitted`. `run` gains `gold_needles: list[str] | None = None`; new repeatable CLI flag `--gold-needle`. The fitted answer step uses the selector instead of a raw head-slice.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_diagnose_fit.py`:

```python
"""The fitted read path must fit the budget and preserve distinct-slot gold."""

from rem.config import Settings
from rem.memory.facts_ledger import FactEntry, FactsLedger
from rem.memory.tiers import MemoryState, SpanSummary, Turn, count_tokens
from evals.battery.diagnose_memory import fit_with_selector, gold_in_fitted


def _state_with_gold() -> MemoryState:
    summaries = [SpanSummary(covers_turn_ids=[i], text=f"noise summary {i} " * 20, tokens=60)
                 for i in range(300)]
    entries = [
        FactEntry(kind="number", text="team size is 5 engineers", source_turn_id=74,
                  status="active", slot_key="team.size", slot_value="5 engineers"),
        FactEntry(kind="number", text="the outing had 4 engineers", source_turn_id=12,
                  status="active", slot_key="team_members.count", slot_value="4 engineers"),
    ]
    return MemoryState(turns=[Turn(role="user", content="now?", turn_id=900, tokens=2)],
                       summaries=summaries, ledger=FactsLedger(entries=entries))


def test_fit_with_selector_fits_budget():
    settings = Settings(read_fit_tokens=4000)
    fitted_text, fitted_tokens = fit_with_selector(_state_with_gold(), "headcount?", settings)
    assert fitted_tokens == count_tokens(fitted_text)
    assert fitted_tokens <= settings.read_fit_tokens


def test_gold_survives_fit():
    settings = Settings(read_fit_tokens=4000)
    fitted_text, _ = fit_with_selector(_state_with_gold(), "headcount?", settings)
    hits = gold_in_fitted(fitted_text, ["4 engineers", "5 engineers"])
    assert hits == {"4 engineers": True, "5 engineers": True}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=.:src python3 -m pytest tests/unit/test_diagnose_fit.py -q`
Expected: FAIL — `ImportError: cannot import name 'fit_with_selector'`.

- [ ] **Step 3: Add the helpers**

In `evals/battery/diagnose_memory.py`, add imports near the other `rem.memory` imports:

```python
from rem.memory.assembler import assemble
from rem.memory.selector import RecencySelector
```

Add these functions above `run`:

```python
def fit_with_selector(state, question, settings):
    """Fit the compacted state to settings.read_fit_tokens via RecencySelector.

    Returns (fitted_text, fitted_tokens) using the same assemble() the real read
    path uses, so the size reflects what the model would receive.
    """
    fitted_state = RecencySelector().select(state, question, settings.read_fit_tokens)
    fitted_text = assemble(fitted_state, system="", task=question)
    return fitted_text, count_tokens(fitted_text)


def gold_in_fitted(fitted_text: str, needles: list[str]) -> dict[str, bool]:
    """Whether each gold needle survives into the fitted slice (case-insensitive)."""
    low = fitted_text.lower()
    return {n: n.lower() in low for n in needles}
```

- [ ] **Step 4: Replace the head-slice fitted block in `run`**

Replace the current "Answer attempt 2" block (lines 220-236) with:

```python
    # Answer attempt 2: the REAL bounded read path. Fit via the selector, check the
    # gold survives the fit, then (optionally) answer. Step 0 PASS = fits budget +
    # gold present + an answer returned (judged correctness deferred; see spec).
    settings = Settings(summarizer_model=GEMMA)
    fitted, fitted_tokens = fit_with_selector(state, it.question, settings)
    # Faithful two-word gold needles come from --gold-needle; fall back to the
    # (looser, single-token) salient scan only if none were given.
    needles = gold_needles or sorted(survival["salient_keyword_hits"] or {})
    gold_hits = gold_in_fitted(fitted, needles)
    payload["rem_fitted_tokens"] = fitted_tokens
    payload["gold_in_fitted"] = gold_hits
    _write()
    print(f"fitted_tokens: {fitted_tokens} (budget {settings.read_fit_tokens})  "
          f"gold_in_fitted: {gold_hits}")

    if answer:
        try:
            ans = answer_question(npu, context=fitted, question=it.question).strip()
            payload["rem_fitted_answer"] = ans
        except Exception as e:  # noqa: BLE001
            payload["rem_fitted_judge_reason"] = f"answer failed: {type(e).__name__}: {e}"
        _write()
        print(f"fitted answer: {(payload['rem_fitted_answer'] or '')[:200]}")
```

Add the two new payload keys to the payload dict initializer (after line 190, alongside the other `rem_fitted_*` fields):

```python
        "rem_fitted_tokens": None,
        "gold_in_fitted": None,
```

Extend `run`'s signature (from Task 2) to accept the needles:

```python
def run(data: str, max_gold_recency: float, out: str,
        load_state: str | None = None, answer: bool = True,
        gold_needles: list[str] | None = None) -> int:
```

In `main`, add the repeatable flag (after `--no-answer`):

```python
    ap.add_argument("--gold-needle", dest="gold_needles", action="append", default=None,
                    help="Exact gold substring that must survive the fit (repeatable, "
                         "e.g. --gold-needle '4 engineers' --gold-needle '5 engineers').")
```

and pass it in the final `run(...)` call:

```python
    return run(args.data, args.max_gold_recency, args.out,
               load_state=args.load_state, answer=args.answer,
               gold_needles=args.gold_needles)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `PYTHONPATH=.:src python3 -m pytest tests/unit/test_diagnose_fit.py -q`
Expected: PASS (2 passed).

- [ ] **Step 6: Run the full NPU-free suite**

Run: `PYTHONPATH=.:src python3 -m pytest -m 'not npu' -q`
Expected: PASS — 152 passed (143 baseline + 6 selector + 1 load-state + 2 fit; no regressions).

- [ ] **Step 7: Commit**

```bash
git add evals/battery/diagnose_memory.py tests/unit/test_diagnose_fit.py
git commit -m "REM(diag): selector-based fitted read + gold-in-fitted check

Replaces the naive head-slice with the RecencySelector read path; records
rem_fitted_tokens and gold_in_fitted so Step 0's PASS bar is checkable.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Run Step 0 against the persisted state + record the result

> **STATUS: COMPLETE** (commit `b3deb3d`). Step 0 verdict: **FAIL** on the §5 bar by the budget criterion alone — fitted 28,121 vs 28,000 (+121, 0.43% over). Gold survived (both needles) and the model returned an answer, so the fit turns the 40,626-token overflow into an answerable read. Recorded unpatched (spec D2); `SELECTOR_RESERVE_TOKENS` calibration noted as the follow-up lever. Plan complete.

**Files:**
- Read: `bench/battery/diag_031748ae_w64k_state.json` (the 836K persisted state)
- Create: `bench/battery/step0_031748ae.json` (the run output)
- Modify: `bench/battery/FINDINGS.md` (record Step 0 outcome)

**Interfaces:**
- Consumes: Task 1-3 code; the LongMemEval data path (same `--data` the diagnostic used).

- [ ] **Step 1: Confirm no other NPU job is running**

Run: `ps aux | grep -i -E 'battery|ingest|diagnose' | grep -v grep`
Expected: no running battery/ingest process. (The single answer inference must not overlap another NPU job.)

- [ ] **Step 2: Run Step 0 from the persisted state (no 75-min ingest)**

The dataset is the local LongMemEval-S file `/home/keith/datasets/longmemeval/longmemeval_s` (278 MB, JSON; the same `--data` the diagnostic used). Then:

```bash
PYTHONPATH=.:src python3 evals/battery/diagnose_memory.py \
  --data /home/keith/datasets/longmemeval/longmemeval_s \
  --max-gold-recency 0.33 \
  --load-state bench/battery/diag_031748ae_w64k_state.json \
  --gold-needle "4 engineers" \
  --gold-needle "5 engineers" \
  --out bench/battery/step0_031748ae.json
```

Expected stdout includes `item=031748ae`, `assembled_total_tokens: ~40000`, a
`fitted_tokens:` line `<= 28000`, and `gold_in_fitted: {'4 engineers': True, '5 engineers': True}`.

Note the gold-answer phrasing for 031748ae before trusting the needles: if the
dataset states the counts differently (e.g. "four"/"five" spelled out, or a
different headcount), pass the matching `--gold-needle` values instead. Confirm
against `it.answer` printed in the run (`gold answer: …`).

- [ ] **Step 3: Evaluate the PASS bar (spec §5)**

Open `bench/battery/step0_031748ae.json`. Step 0 PASSES when all hold:
- `rem_fitted_tokens <= 28000`
- `rem_fitted_answer` is non-null (model returned an answer; no `ContextLimitExceeded`, no HTTP 400)
- both headcount needles in `gold_in_fitted` are `true`

If gold is `false` under recency, that is a real, informative result (recency drops
early gold) — record it as a FAIL with the per-needle detail; per spec D2 it argues
for the lexical selector. Do not patch the selector to force a pass.

- [ ] **Step 4: Record the outcome in FINDINGS.md**

Append a `## Step 0: bounded read path` section to `bench/battery/FINDINGS.md` stating: fitted tokens (vs 40,626 unfitted), whether each gold needle survived, whether an answer was produced, and the PASS/FAIL verdict against the §5 bar. Reference `step0_031748ae.json`. Keep the house voice (concise, factual, no "not X but Y").

- [ ] **Step 5: Commit**

```bash
git add bench/battery/step0_031748ae.json bench/battery/FINDINGS.md
git commit -m "REM(bench): Step 0 result — bounded read path on 031748ae (NPU-free)

<one line: PASS/FAIL + fitted_tokens + gold survival>

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Post-plan: what Step 0 unblocks (not in this plan)

Step 0 proves the read-path mechanism on one item. It does NOT choose the
architecture. The next increment (separate plan) persists per-item MemoryState in
the battery, pays one ~6h ingest to capture all five oldest-gold states, then
iterates selector variants NPU-free against them to get the failure *mix* — the
evidence the graph-vs-tune decision (spec §8) waits on.
