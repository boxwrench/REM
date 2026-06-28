# Failure Mix — Post-Step-0 Implementation Plan

**Goal:** Capture the five oldest-gold compacted `MemoryState`s once (the only NPU
ingest), then label each item's read-path miss NPU-free as size / retrieval-recall
/ temporal-structure / pass — producing the failure *mix* spec §8 waits on.

**Spec:** `docs/superpowers/specs/2026-06-27-failure-mix-design.md`. Decision log §7 governs.

**Tech stack:** Python 3.12, pytest. No new dependencies. Reuses
`RemContextManager.ingest`, `MemoryState.save/load`, `fit_with_selector`,
`gold_in_fitted`, `RecencySelector`.

## Global constraints

- Tests run NPU-free: `PYTHONPATH=.:src python3 -m pytest -m 'not npu'`. Current
  baseline before this plan = **153 passed**.
- No NPU calls in unit tests: inject the context manager / answerer via factory
  args (`make_cm`, `answerer`). The real NPU is touched only by the launch run (Task C).
- Capture settings match the existing 031748ae diagnostic: `budget_tokens=1000`,
  `max_context_tokens=64000`, `summarizer_model=gemma4-it:e2b` (spec E2).
- Capture is idempotent: skip any item whose `<id>_state.json` exists (spec E3).
- Do not modify `selector.py` or the Step-0 read path. This increment is additive.
- House voice on commits: concise, factual; banned construction "not X but Y".

---

### Task A: `capture_states.py` (per-item ingest + save, idempotent)

> NPU-free CODE; the multi-hour ingest is run separately in Task C.

**Files:**
- Create: `evals/battery/capture_states.py`
- Test: `tests/unit/test_capture_states.py`

**Interfaces:**
- `capture_item(it, out_dir, make_cm, budget_tokens=1000) -> dict` — ingest one
  item, save `<out_dir>/<question_id>_state.json`, return the manifest record.
- `run(data, out_dir, max_gold_recency=0.33, limit=None, budget_tokens=1000, make_cm=None) -> int`
  — select items, capture missing ones, write `manifest.json` incrementally.
- CLI: `--data --out-dir --max-gold-recency --limit --budget`.

- [ ] **Step 1: Write the failing tests** — `tests/unit/test_capture_states.py`:

```python
"""capture_states saves per-item state + manifest, idempotent and NPU-free."""
import json

from rem.memory.facts_ledger import FactEntry, FactsLedger
from rem.memory.tiers import MemoryState, Turn
from evals.battery.models import QAItem, Session
from evals.battery import capture_states


class _StubCM:
    """Stands in for RemContextManager; no NPU."""
    def __init__(self):
        self._state = None
        self.ingested = False
    def ingest(self, sessions, budget_tokens):
        self.ingested = True
        self._state = MemoryState(
            turns=[Turn(role="user", content="hi", turn_id=1, tokens=1)],
            summaries=[],
            ledger=FactsLedger(entries=[FactEntry(kind="entity", text="x", source_turn_id=1)]),
        )
    def assemble(self):
        return "ASSEMBLED CONTEXT HERE"


def _item(qid="aaa", recency=0.2):
    return QAItem(question_id=qid, question="q?", answer="a",
                  question_type="knowledge-update",
                  sessions=[Session(session_id="s1", turns=[{"role": "user", "content": "hi"}])],
                  answer_session_ids=["s1"], gold_recency=recency)


def test_capture_item_saves_state_and_record(tmp_path):
    cm = _StubCM()
    rec = capture_states.capture_item(_item(), tmp_path, lambda: cm, budget_tokens=1000)
    assert cm.ingested
    assert (tmp_path / "aaa_state.json").exists()
    assert rec["question_id"] == "aaa"
    assert rec["assembled_total_tokens"] > 0
    assert rec["ingest_secs"] is not None
    assert len(MemoryState.load(tmp_path / "aaa_state.json").turns) == 1


def test_run_skips_existing_and_writes_manifest(tmp_path, monkeypatch):
    items = [_item("aaa", 0.1), _item("bbb", 0.2)]
    monkeypatch.setattr(capture_states, "load_knowledge_update", lambda *a, **k: items)
    # Pre-seed aaa's state so it is skipped (resumable case).
    MemoryState(turns=[Turn(role="user", content="pre", turn_id=1, tokens=1)]).save(
        tmp_path / "aaa_state.json")

    made = []
    def make_cm():
        cm = _StubCM(); made.append(cm); return cm

    rc = capture_states.run("ignored", str(tmp_path), make_cm=make_cm)
    assert rc == 0
    assert len(made) == 1                       # only bbb ingested; aaa skipped
    assert (tmp_path / "bbb_state.json").exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    ids = {r["question_id"] for r in manifest}
    assert ids == {"aaa", "bbb"}                # skipped item still recorded
    bbb = next(r for r in manifest if r["question_id"] == "bbb")
    assert bbb["ingest_secs"] is not None
    aaa = next(r for r in manifest if r["question_id"] == "aaa")
    assert aaa["ingest_secs"] is None           # not re-ingested
```

- [ ] **Step 2: Run to verify failure** — `ModuleNotFoundError: evals.battery.capture_states`.

- [ ] **Step 3: Implement** — `evals/battery/capture_states.py`:

```python
"""Capture per-item compacted MemoryState for the oldest-gold battery items.

Runs the real ~75-min compaction once per item and persists each MemoryState so
the read-path / selector analysis can iterate NPU-free. Idempotent: skips items
whose state file already exists, so the multi-hour ingest is resumable.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from evals.battery.context_managers import RemContextManager
from evals.battery.longmemeval_loader import load_knowledge_update
from rem.config import Settings
from rem.memory.tiers import count_tokens

GEMMA = "gemma4-it:e2b"
DIAG_WINDOW_TOKENS = 64000  # match diagnose_memory so all states are comparable


def _meta(it, state_path: Path) -> dict:
    return {
        "question_id": it.question_id,
        "question": it.question,
        "answer": it.answer,
        "answer_session_ids": it.answer_session_ids,
        "gold_recency": it.gold_recency,
        "n_sessions": len(it.sessions),
        "n_turns": sum(len(s.turns) for s in it.sessions),
        "state_file": str(state_path),
    }


def _upsert(records: list[dict], rec: dict) -> list[dict]:
    out = [r for r in records if r["question_id"] != rec["question_id"]]
    out.append(rec)
    out.sort(key=lambda r: r.get("gold_recency", 1.0))
    return out


def capture_item(it, out_dir: Path, make_cm, budget_tokens: int = 1000) -> dict:
    """Ingest one item, save its state, return the manifest record.

    make_cm() is a zero-arg context-manager factory (injected for tests).
    """
    out_dir = Path(out_dir)
    state_path = out_dir / f"{it.question_id}_state.json"
    t0 = time.time()
    cm = make_cm()
    cm.ingest(it.sessions, budget_tokens=budget_tokens)
    ingest_secs = round(time.time() - t0, 1)
    assembled = cm.assemble()
    cm._state.save(state_path)
    rec = _meta(it, state_path)
    rec.update(assembled_total_tokens=count_tokens(assembled),
               ingest_secs=ingest_secs, captured_at=time.time())
    return rec


def run(data: str, out_dir: str, max_gold_recency: float = 0.33,
        limit: int | None = None, budget_tokens: int = 1000, make_cm=None) -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    records = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []

    items = load_knowledge_update(data, limit=limit, max_gold_recency=max_gold_recency)
    if not items:
        print("No matching knowledge-update items.", file=sys.stderr)
        return 2
    print(f"selected {len(items)} items; recency="
          f"{[round(it.gold_recency, 3) for it in items]}", flush=True)

    if make_cm is None:
        from rem.npu_client import NpuClient
        npu = NpuClient(Settings(summarizer_model=GEMMA))

        def make_cm():
            return RemContextManager(
                client=npu,
                settings=Settings(summarizer_model=GEMMA,
                                  max_context_tokens=DIAG_WINDOW_TOKENS))

    def _flush():
        manifest_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    for it in items:
        state_path = out / f"{it.question_id}_state.json"
        if state_path.exists():
            if not any(r["question_id"] == it.question_id for r in records):
                rec = _meta(it, state_path)
                rec.update(assembled_total_tokens=None, ingest_secs=None, captured_at=None)
                records = _upsert(records, rec)
                _flush()
            print(f"[{it.question_id}] state exists, skipping ingest", flush=True)
            continue
        print(f"[{it.question_id}] ingesting "
              f"(recency={it.gold_recency:.3f}, "
              f"{sum(len(s.turns) for s in it.sessions)} turns)…", flush=True)
        rec = capture_item(it, out, make_cm, budget_tokens=budget_tokens)
        records = _upsert(records, rec)
        _flush()
        print(f"[{it.question_id}] saved {rec['assembled_total_tokens']} tok in "
              f"{rec['ingest_secs']}s -> {rec['state_file']}", flush=True)

    print(f"manifest: {manifest_path} ({len(records)} records)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Capture per-item compacted MemoryState")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out-dir", default="bench/battery/states")
    ap.add_argument("--max-gold-recency", type=float, default=0.33)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--budget", type=int, default=1000)
    args = ap.parse_args()
    return run(args.data, args.out_dir, max_gold_recency=args.max_gold_recency,
               limit=args.limit, budget_tokens=args.budget)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests** — expect 2 passed.
- [ ] **Step 5: Full suite** — `PYTHONPATH=.:src python3 -m pytest -m 'not npu' -q` → 155 passed (153 + 2).
- [ ] **Step 6: Commit** — `REM(bench): capture_states — per-item MemoryState capture (idempotent, NPU-free code)`.

---

### Task B: `mix_report.py` (NPU-free failure-mode labelling)

> NPU-free for fit/gold/tier; an optional single brief answer per item is the only NPU and is off by default.

**Files:**
- Create: `evals/battery/mix_report.py`
- Test: `tests/unit/test_mix_report.py`

**Interfaces:**
- `GOLD_NEEDLES: dict[str, list[str]]` — curated per-item needles (spec E5).
- `needle_tier(state, question, settings, needle) -> str` — "slot"/"summary"/"free"/"absent" in the fitted state.
- `label_item(state, question, answer, needles, settings, answerer=None) -> dict` —
  returns `{fitted_tokens, fits_budget, gold_in_fitted, needle_tiers, brief_answer,
  answer_contains_gold, failure_mode}`.
- `run(states_dir, out, settings=None, answerer=None) -> int` — label every item in
  the manifest, write `mix_report.json` + print a table.

Failure-mode rule (spec §5):
- `not fits_budget` → **size**
- else any needle absent → **retrieval-recall**
- else `answerer` given and `answer_contains_gold` → **pass**
- else `answerer` given → **temporal-structure**
- else → **needs-answer** (fits + present, no brief answer taken)

- [ ] **Step 1: Write the failing tests** — `tests/unit/test_mix_report.py`:

```python
"""mix_report labels each item's read-path miss NPU-free."""
from rem.config import Settings
from rem.memory.facts_ledger import FactEntry, FactsLedger
from rem.memory.tiers import MemoryState, SpanSummary, Turn
from evals.battery.mix_report import label_item


def _slot_gold():
    entries = [
        FactEntry(kind="number", text="team size is 5 engineers", source_turn_id=74,
                  status="active", slot_key="team.size", slot_value="5 engineers"),
        FactEntry(kind="number", text="outing had 4 engineers", source_turn_id=12,
                  status="active", slot_key="team_members.count", slot_value="4 engineers"),
    ]
    return MemoryState(turns=[Turn(role="user", content="now?", turn_id=900, tokens=2)],
                       summaries=[], ledger=FactsLedger(entries=entries))


def test_label_retrieval_recall_when_needle_absent():
    settings = Settings(read_fit_tokens=4000)
    # gold "5 engineers" present; "9 engineers" never in memory -> absent -> recall miss
    out = label_item(_slot_gold(), "headcount?", "5 engineers",
                     ["5 engineers", "9 engineers"], settings)
    assert out["fits_budget"] is True
    assert out["gold_in_fitted"]["9 engineers"] is False
    assert out["failure_mode"] == "retrieval-recall"


def test_label_needs_answer_when_all_present_no_answerer():
    settings = Settings(read_fit_tokens=4000)
    out = label_item(_slot_gold(), "headcount?", "5 engineers",
                     ["4 engineers", "5 engineers"], settings)
    assert out["failure_mode"] == "needs-answer"
    assert out["needle_tiers"]["5 engineers"] == "slot"


def test_label_size_when_protected_floor_exceeds_budget():
    # 200 distinct protected slots blow a tiny budget; cannot fit even minimal.
    entries = [FactEntry(kind="number", text=f"metric {i} is forty two units here",
                         source_turn_id=i, status="active",
                         slot_key=f"m.{i}", slot_value=str(i)) for i in range(200)]
    entries.append(FactEntry(kind="number", text="team size is 5 engineers",
                             source_turn_id=999, status="active",
                             slot_key="team.size", slot_value="5 engineers"))
    state = MemoryState(turns=[], summaries=[], ledger=FactsLedger(entries=entries))
    settings = Settings(read_fit_tokens=800)
    out = label_item(state, "headcount?", "5 engineers", ["5 engineers"], settings)
    assert out["fits_budget"] is False
    assert out["failure_mode"] == "size"


def test_label_pass_and_temporal_with_injected_answerer():
    settings = Settings(read_fit_tokens=4000)
    good = label_item(_slot_gold(), "headcount?", "5 engineers",
                      ["4 engineers", "5 engineers"], settings,
                      answerer=lambda ctx, q: "you lead 5 engineers")
    assert good["failure_mode"] == "pass"
    bad = label_item(_slot_gold(), "headcount?", "5 engineers",
                     ["4 engineers", "5 engineers"], settings,
                     answerer=lambda ctx, q: "the memory does not say")
    assert bad["failure_mode"] == "temporal-structure"
```

- [ ] **Step 2: Run to verify failure** — `ImportError: cannot import name 'label_item'`.

- [ ] **Step 3: Implement** — `evals/battery/mix_report.py`:

```python
"""NPU-free failure-mix analysis over captured MemoryStates.

For each state in the capture manifest, run the Step-0 bounded read path
(RecencySelector via fit_with_selector) and label the item's miss as
size / retrieval-recall / temporal-structure / pass (spec §5). An optional single
brief answer per item (the only NPU) separates temporal-structure from pass.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from rem.config import Settings
from rem.memory.selector import RecencySelector
from rem.memory.tiers import MemoryState
from evals.battery.diagnose_memory import fit_with_selector, gold_in_fitted

GEMMA = "gemma4-it:e2b"

# Curated two-word faithful gold needles per item (spec E5).
GOLD_NEEDLES = {
    "031748ae": ["4 engineers", "5 engineers"],
    "3ba21379": ["F-150"],
    "cc5ded98": ["two hours"],
    "c6853660": ["two cups", "increased"],
    "9bbe84a2": ["level 100"],
}


def needle_tier(state, question, settings, needle) -> str:
    """Which tier of the FITTED state carries the needle."""
    fitted = RecencySelector().select(state, question, settings.read_fit_tokens)
    low = needle.lower()
    for e in fitted.ledger.entries:
        if low in e.text.lower():
            return "slot" if e.slot_key else "free"
    for s in fitted.summaries:
        txt = s.rendered_text if s.rendered_text is not None else s.text
        if low in txt.lower():
            return "summary"
    return "absent"


def label_item(state, question, answer, needles, settings, answerer=None) -> dict:
    fitted_text, fitted_tokens = fit_with_selector(state, question, settings)
    fits = fitted_tokens <= settings.read_fit_tokens
    hits = gold_in_fitted(fitted_text, needles)
    tiers = {n: needle_tier(state, question, settings, n) for n in needles}

    brief_answer = None
    answer_contains_gold = None
    if answerer is not None:
        brief_answer = (answerer(fitted_text, question) or "").strip()
        low = brief_answer.lower()
        answer_contains_gold = any(n.lower() in low for n in needles)

    if not fits:
        mode = "size"
    elif not all(hits.values()):
        mode = "retrieval-recall"
    elif answerer is None:
        mode = "needs-answer"
    elif answer_contains_gold:
        mode = "pass"
    else:
        mode = "temporal-structure"

    return {
        "fitted_tokens": fitted_tokens, "fits_budget": fits,
        "gold_in_fitted": hits, "needle_tiers": tiers,
        "brief_answer": brief_answer, "answer_contains_gold": answer_contains_gold,
        "failure_mode": mode,
    }


def run(states_dir: str, out: str, settings=None, answerer=None) -> int:
    settings = settings or Settings(summarizer_model=GEMMA)
    sdir = Path(states_dir)
    manifest_path = sdir / "manifest.json"
    if not manifest_path.exists():
        print(f"no manifest at {manifest_path}", file=sys.stderr)
        return 2
    records = json.loads(manifest_path.read_text(encoding="utf-8"))

    rows = []
    counts: dict[str, int] = {}
    for r in records:
        state = MemoryState.load(r["state_file"])
        needles = GOLD_NEEDLES.get(r["question_id"], [])
        lab = label_item(state, r["question"], r["answer"], needles, settings,
                         answerer=answerer)
        lab["question_id"] = r["question_id"]
        lab["gold_recency"] = r.get("gold_recency")
        rows.append(lab)
        counts[lab["failure_mode"]] = counts.get(lab["failure_mode"], 0) + 1
        print(f"[{r['question_id']}] mode={lab['failure_mode']:18s} "
              f"fitted={lab['fitted_tokens']:6d} fits={lab['fits_budget']} "
              f"tiers={lab['needle_tiers']}", flush=True)

    payload = {"states_dir": str(sdir), "n_items": len(rows),
               "mix": counts, "items": rows}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nMIX: {counts}")
    print(f"Written to {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Failure-mix analysis over captured states")
    ap.add_argument("--states-dir", default="bench/battery/states")
    ap.add_argument("--out", default="bench/battery/mix_report.json")
    ap.add_argument("--answer", action="store_true",
                    help="Take one brief NPU answer per item (separates pass from "
                         "temporal-structure). Off by default = fully NPU-free.")
    args = ap.parse_args()
    answerer = None
    if args.answer:
        from rem.npu_client import NpuClient
        from evals.battery.answerer import answer_question
        npu = NpuClient(Settings(summarizer_model=GEMMA))
        def answerer(ctx, q):
            return answer_question(npu, context=ctx, question=q)
    return run(args.states_dir, args.out, answerer=answerer)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests** — expect 4 passed.
- [ ] **Step 5: Full suite** → 159 passed (155 + 4).
- [ ] **Step 6: Commit** — `REM(bench): mix_report — NPU-free failure-mode labelling over captured states`.

---

### Task C: launch the ~6h NPU ingest (run separately, gated)

> Driven directly by the operator, not a subagent. The single XDNA2 NPU is busy for the whole run; no other NPU job may overlap.

- [ ] Pre-seed `bench/battery/states/031748ae_state.json` from
  `bench/battery/diag_031748ae_w64k_state.json` (already captured; idempotent skip).
- [ ] Confirm no other NPU job is running (`ps`, and nothing on localhost:13306 mid-job).
- [ ] Launch backgrounded with logging:

```bash
PYTHONPATH=.:src nohup python3 evals/battery/capture_states.py \
  --data /home/keith/datasets/longmemeval/longmemeval_s \
  --max-gold-recency 0.33 \
  --out-dir bench/battery/states \
  > /tmp/capture_states.log 2>&1 &
```

- [ ] Verify the first missing item begins ingesting; then let it run (~5h for the
  four remaining). States + manifest are written incrementally, so a crash keeps
  completed items and re-launching resumes.

### Task D: mix analysis + FINDINGS (deferred — after the ingest completes)

- [ ] When `manifest.json` has all five states, run `mix_report.py` (optionally with
  `--answer` for the one brief inference per item), then append a `## Failure mix`
  section to `bench/battery/FINDINGS.md` with the distribution and the architecture
  read (spec §8). Not in this session's window — the ingest runs ~5h.

---

## Notes

- States are large (~0.8–1.2 MB each). They are tracked like the existing
  `diag_031748ae_w64k_state.json` so the NPU-free iteration is reproducible.
- `mix_report` correctness labels (pass vs temporal-structure) rest on a substring
  check of a brief answer; judged correctness stays deferred (spec §2). The label is
  a signal, the per-needle tier provenance is the harder evidence.
