"""capture_states saves per-item state + manifest, idempotent and NPU-free."""
import hashlib
import json

from evals.battery import capture_states
from evals.battery.context_managers import RemContextManager
from evals.battery.models import QAItem, Session
from evals.memory_methods import capture_states as development_capture
from rem.config import Settings
from rem.memory.facts_ledger import FactEntry, FactsLedger
from rem.memory.tiers import MemoryState, Turn


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
    assert rec["extraction"]["failures"] == 0
    assert len(MemoryState.load(tmp_path / "aaa_state.json").turns) == 1


def test_capture_item_persists_session_provenance(tmp_path):
    item = _item()
    item.sessions[0].timestamp = "2023/06/11 (Sun) 14:05"

    def make_cm():
        return RemContextManager(
            client=object(),
            settings=Settings(max_context_tokens=64000),
        )

    capture_states.capture_item(item, tmp_path, make_cm, budget_tokens=1000)

    turn = MemoryState.load(tmp_path / "aaa_state.json").turns[0]
    assert turn.session_id == "s1"
    assert turn.timestamp == "2023/06/11 (Sun) 14:05"


def test_run_skips_existing_and_writes_manifest(tmp_path, monkeypatch):
    items = [_item("aaa", 0.1), _item("bbb", 0.2)]
    monkeypatch.setattr(capture_states, "load_knowledge_update", lambda *a, **k: items)
    # Pre-seed aaa's state so it is skipped (resumable case).
    MemoryState(turns=[Turn(role="user", content="pre", turn_id=1, tokens=1)]).save(
        tmp_path / "aaa_state.json")

    made = []
    def make_cm():
        cm = _StubCM()
        made.append(cm)
        return cm

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


def test_development_capture_persists_extraction_telemetry(tmp_path, monkeypatch):
    data_path = tmp_path / "dataset.json"
    data_path.write_text("fixture", encoding="utf-8")
    state_path = tmp_path / "states" / "aaa_state.json"
    manifest_path = tmp_path / "development_manifest.json"
    manifest_path.write_text(json.dumps({
        "source_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "items": [{
            "question_id": "aaa",
            "category": "knowledge-update",
            "state_file": str(state_path),
        }],
    }), encoding="utf-8")
    monkeypatch.setattr(
        development_capture,
        "load_categories",
        lambda *args, **kwargs: [_item()],
    )
    monkeypatch.setattr(development_capture, "capture_item", lambda *args, **kwargs: {
        "state_file": str(state_path),
        "ingest_secs": 12.5,
        "assembled_total_tokens": 321,
        "captured_at": 456.0,
        "extraction": {"attempts": 4, "failures": 1},
    })

    assert development_capture.run(
        str(data_path), str(manifest_path), make_cm=lambda: object()
    ) == 0

    record = json.loads(manifest_path.read_text(encoding="utf-8"))["items"][0]
    assert record["capture"]["extraction"] == {"attempts": 4, "failures": 1}
