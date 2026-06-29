# Slot-Key Canonicalization — Implementation Plan

> **STATUS: EXECUTED (2026-06-28) — string-first insufficient; do not promote.** The
> experiment ran via a parallel build. The actual implementation lives in
> `src/rem/memory/canonicalize.py` (`canonical_slot_key`, `recanonicalize`) and the
> audit in `evals/battery/canonicalize_audit.py` (`bench/battery/canonicalize_audit.json`)
> — these supersede the inline code below, which differs in structure. Result: full-key
> granularity buys ~1%, subject-only buys ~25% but creates ~592 merge-risk groups
> (unsafe for the writer); the residual fragmentation is semantic. **Decision: neither
> string strategy is promoted into `_apply_supersession`; escalate to embedding-based
> identity (Qwen baseline, then DREAM).** See `bench/battery/FINDINGS.md`
> "## Slot-key canonicalization" and roadmap Gate 0. The 031748ae paid rerun was
> skipped (both candidates already failed the promotion threshold). The inline tasks
> below are retained as the SDD trail of what was designed; they are not live work.

**Goal:** Test string-first slot-key canonicalization as the fix for near-absent
supersession, NPU-free against the five captured states; measure its ceiling so the
embedding decision (Thread 2) rests on evidence.

**Spec:** `docs/superpowers/specs/2026-06-28-slot-key-canonicalization-design.md`. Decision log §7 governs.

**Tech stack:** Python 3.12, pydantic v2 (`FactEntry.model_copy`), pytest. No new deps.

## Global constraints

- Tests run NPU-free: `PYTHONPATH=.:src python3 -m pytest -m 'not npu'`. Baseline
  before this plan = **164 passed**.
- Post-hoc only: do NOT modify `FactsLedger._apply_supersession`, the extractor, or
  re-ingest. Operate on captured `MemoryState`s.
- Canonicalization is general and domain-agnostic: no per-item alias tables (spec F5/D2).
- Retain history: superseded priors are kept (marked stale, ordered), not dropped (spec F4).
- House voice on commits; banned construction "not X but Y".

---

### Task 1: `canonical_slot_key` + tokenization (string normalization)

**Files:**
- Create: `src/rem/memory/canonicalize.py`
- Test: `tests/unit/test_canonicalize.py`

- [ ] **Step 1: Failing tests** — `tests/unit/test_canonicalize.py`:

```python
"""String-first slot-key canonicalization (NPU-free)."""
from rem.memory.canonicalize import canonical_slot_key, _tokens


def test_merges_trivial_key_variants_under_full():
    # the model split the same concept across subject/attribute differently
    assert canonical_slot_key("team.size", "full") == canonical_slot_key("team size.size", "full")


def test_does_not_merge_semantically_distinct_keys_under_full():
    # string-only cannot see that "group size.number of engineers" == team size
    assert canonical_slot_key("team.size", "full") != \
        canonical_slot_key("group size.number of engineers", "full")


def test_subject_granularity_merges_attributes_of_one_subject():
    a = canonical_slot_key("coding exercises.frequency", "subject")
    b = canonical_slot_key("coding exercises.duration", "subject")
    assert a == b


def test_singularizes_and_drops_stopwords():
    assert _tokens("number of engineers") == {"engineer"}
    assert _tokens("cups") == {"cup"}
    assert canonical_slot_key("morning.cups", "full") == canonical_slot_key("morning.cup", "full")


def test_deterministic_token_order():
    assert canonical_slot_key("size.team", "full") == canonical_slot_key("team.size", "full")
```

- [ ] **Step 2: Run → fail** (`ModuleNotFoundError: rem.memory.canonicalize`).

- [ ] **Step 3: Implement** — `src/rem/memory/canonicalize.py`:

```python
"""String-first slot-key canonicalization for fact supersession.

Supersession only collapses entries with identical slot_key strings, but the
extractor emits a fresh `subject.attribute` per mention, so one fact fragments
across many keys and never supersedes. canonical_slot_key() maps fragmented keys to
a shared token signature so updates can collapse. String-only by design: the gap it
cannot close is the evidence for embedding-based matching (Thread 2).
"""
from __future__ import annotations

import re

_STOPWORDS = {
    "a", "an", "the", "of", "to", "for", "in", "on", "per", "and", "or",
    "number", "count", "total", "amount", "value", "range", "type", "new", "current",
}
_IRREGULAR = {"people": "person", "children": "child", "men": "man", "women": "woman"}


def _singular(tok: str) -> str:
    if tok in _IRREGULAR:
        return _IRREGULAR[tok]
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 4 and tok.endswith("es") and not tok.endswith(("ses", "zes", "ches", "shes")):
        return tok[:-2]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def _tokens(text: str) -> set[str]:
    raw = re.split(r"[^a-z0-9]+", text.lower())
    return {_singular(t) for t in raw if t and t not in _STOPWORDS}


def canonical_slot_key(slot_key: str, granularity: str = "full") -> str:
    """Canonical token signature of a slot_key.

    granularity 'full' = subject ∪ attribute tokens (merges trivial key variants);
    'subject' = subject tokens only (merges differing attributes of one subject, at
    the risk of over-merging). Returns a sorted, space-joined token string.
    """
    subject, _, attribute = slot_key.rpartition(".")
    if not subject:  # no '.', whole string is the subject
        subject = slot_key
        attribute = ""
    toks = _tokens(subject)
    if granularity == "full":
        toks |= _tokens(attribute)
    return " ".join(sorted(toks))
```

- [ ] **Step 4: Run → pass** (5 passed). **Step 5: Full suite → 169 passed** (164 + 5).
- [ ] **Step 6: Commit** — `REM(memory): canonical_slot_key — string-first slot key signature`.

---

### Task 2: `recanonicalize` transform (re-supersede a captured state)

**Files:**
- Modify: `src/rem/memory/canonicalize.py` (add `recanonicalize`)
- Test: `tests/unit/test_recanonicalize.py`

- [ ] **Step 1: Failing tests** — `tests/unit/test_recanonicalize.py`:

```python
"""recanonicalize collapses fragmented slots, keeping ordered history (NPU-free)."""
from rem.memory.facts_ledger import FactEntry, FactsLedger
from rem.memory.tiers import MemoryState, Turn
from rem.memory.canonicalize import recanonicalize


def _state():
    entries = [
        # same concept, two key strings, two values, two turns -> should collapse
        FactEntry(kind="number", text="team size is 4 engineers", source_turn_id=12,
                  status="active", slot_key="team members.count", slot_value="4 engineers"),
        FactEntry(kind="number", text="team size is 5 engineers", source_turn_id=74,
                  status="active", slot_key="team.size", slot_value="5 engineers"),
        # unrelated active slot, must be untouched
        FactEntry(kind="entity", text="camera is a Sony", source_turn_id=20,
                  status="active", slot_key="camera.model", slot_value="Sony"),
    ]
    return MemoryState(turns=[Turn(role="user", content="now?", turn_id=900, tokens=2)],
                       summaries=[], ledger=FactsLedger(entries=entries))


def test_collapses_fragmented_group_keeps_newest_active():
    # NB: team.size vs team members.count merge only under 'subject' here
    #     ({team} vs {team, member}) — use 'full' which merges via shared {team,size}?
    #     They do NOT share tokens under full either, so this test uses matching keys.
    entries = [
        FactEntry(kind="number", text="team size 4", source_turn_id=12, status="active",
                  slot_key="team.size", slot_value="4 engineers"),
        FactEntry(kind="number", text="team size 5", source_turn_id=74, status="active",
                  slot_key="team size.size", slot_value="5 engineers"),
    ]
    state = MemoryState(turns=[], summaries=[], ledger=FactsLedger(entries=entries))
    out = recanonicalize(state, "full")
    actives = [e for e in out.ledger.entries if e.status == "active"]
    stales = [e for e in out.ledger.entries if e.status == "stale"]
    assert len(actives) == 1 and actives[0].slot_value == "5 engineers"   # newest current
    assert len(stales) == 1 and stales[0].slot_value == "4 engineers"     # prior retained
    assert stales[0].superseded_by_turn_id == 74                          # ordered link
    assert len(out.ledger.entries) == 2                                   # history kept, not dropped


def test_leaves_singletons_and_unrelated_slots_untouched():
    out = recanonicalize(_state(), "full")
    cam = [e for e in out.ledger.entries if e.slot_key == "camera.model"]
    assert len(cam) == 1 and cam[0].status == "active"


def test_does_not_mutate_input_state():
    state = _state()
    before = [(e.slot_key, e.status) for e in state.ledger.entries]
    recanonicalize(state, "subject")
    after = [(e.slot_key, e.status) for e in state.ledger.entries]
    assert before == after          # input untouched (deep copy)
```

- [ ] **Step 2: Run → fail** (`ImportError: cannot import name 'recanonicalize'`).

- [ ] **Step 3: Implement** — append to `src/rem/memory/canonicalize.py`:

```python
from rem.memory.facts_ledger import FactsLedger      # noqa: E402
from rem.memory.tiers import MemoryState             # noqa: E402


def recanonicalize(state: MemoryState, granularity: str = "full") -> MemoryState:
    """Re-supersede a captured state by canonical slot key, NPU-free.

    Groups ACTIVE slotted entries by canonical_slot_key; within each multi-member
    group, the newest (max source_turn_id) stays active and older members are marked
    stale with superseded_by_turn_id set to the newest — retained as ordered history,
    not dropped (spec §4.3). Non-slotted entries and singletons are untouched. The
    input state is not mutated.
    """
    entries = [e.model_copy(deep=True) for e in state.ledger.entries]
    groups: dict[str, list] = {}
    for e in entries:
        if e.status == "active" and e.slot_key:
            ck = canonical_slot_key(e.slot_key, granularity)
            groups.setdefault(ck, []).append(e)
    for members in groups.values():
        if len(members) < 2:
            continue
        members.sort(key=lambda e: e.source_turn_id)
        newest = members[-1]
        for older in members[:-1]:
            older.status = "stale"
            older.superseded_by_turn_id = newest.source_turn_id
    return MemoryState(
        schema_version=state.schema_version,
        turns=list(state.turns),
        summaries=list(state.summaries),
        ledger=FactsLedger(entries=entries),
    )
```

(If `FactEntry` is not pydantic v2, replace `model_copy(deep=True)` with `copy.deepcopy`.)

- [ ] **Step 4: Run → pass** (3 passed). **Step 5: Full suite → 172 passed** (169 + 3).
- [ ] **Step 6: Commit** — `REM(memory): recanonicalize — collapse fragmented slots, keep ordered history`.

---

### Task 3: validation harness over the five states (NPU-free + one answer)

**Files:**
- Create: `evals/battery/canonicalize_audit.py`
- Test: `tests/unit/test_canonicalize_audit.py` (small: asserts before/after deltas on a synthetic state)

**Behavior:** for each state in the manifest and each granularity in (`full`,
`subject`): run `write_recall_audit.audit_state` before and after `recanonicalize`,
and record `active_before/after`, `superseded_before/after`,
`fragmented_values_before/after`, plus an over-merge probe for `subject` (canonical
groups that absorbed ≥2 distinct current values whose normalized forms differ). Write
`bench/battery/canonicalize_audit.json` and print a per-item table.

- [ ] Reuse `audit_state` (Task-prior tool) and `recanonicalize`. Gold survival
  after recanonicalize is checked by reloading needles from `mix_report.GOLD_NEEDLES`
  and confirming the current value still resolves to an active slot.
- [ ] Full suite green after the new unit test.
- [ ] Commit — `REM(bench): canonicalize_audit — fragmentation collapse over the 5 states`.

### Task 4: run + the 031748ae answer + FINDINGS (the decision)

- [ ] Run `canonicalize_audit.py` over `bench/battery/states/` (NPU-free); record the
  fragmentation collapse and over-merge per granularity.
- [ ] One paid inference: recanonicalize 031748ae (best granularity), assemble the
  fitted slice with the canonical ordered-history team slot, and take a single brief
  answer — does the model now give "started 4 → now 5"? (Confirm no other NPU job first.)
- [ ] Append `## Slot-key canonicalization` to `bench/battery/FINDINGS.md`: per
  granularity, fragmentation reduction + ledger shrink + over-merge cost + the
  031748ae outcome; then the decision — wire string canonicalization into the write
  path, or escalate the residual to Thread 2 (embeddings) with the measured gap.
- [ ] Commit.

---

## Notes

- The harder team-size fragmentation (`team.size` vs `group size.number of
  engineers`) will NOT merge under string rules — that is the expected, measured
  ceiling, and the reason Thread 2 exists. The point of this increment is to quantify
  how much string-first buys and whether it alone resolves 031748ae.
- If `full` resolves little and `subject` over-merges, that is itself the result:
  string normalization is insufficient, escalate to embeddings.
