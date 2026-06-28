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
