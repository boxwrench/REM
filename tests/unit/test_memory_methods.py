from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from evals.battery.models import Session
from evals.memory_methods.artifacts import ItemRun, MemoryMethodArtifact
from evals.memory_methods.external import HindsightArm, SupermemoryArm
from evals.memory_methods.freeze_manifest import build_manifest
from evals.memory_methods.native import NativeSelectorArm
from evals.memory_methods.promotion import promotion_decision
from rem.memory.facts_ledger import FactEntry, FactsLedger
from rem.memory.selector import PackedLexicalSelector
from rem.memory.tiers import MemoryState


def _session(sid="s1", content="User works at Acme"):
    return Session(sid, [{"role": "user", "content": content}])


def _dataset_entry(category, qid, answer_positions=(0,)):
    ids = ["old", "middle", "new"]
    return {
        "question_id": qid,
        "question_type": category,
        "question": f"question {qid}",
        "answer": "answer",
        "haystack_session_ids": ids,
        "haystack_sessions": [[{"role": "user", "content": sid}] for sid in ids],
        "answer_session_ids": [ids[index] for index in answer_positions],
    }


def test_manifest_balances_categories_and_prefers_distributed_old_evidence():
    raw = []
    for category in ("knowledge-update", "temporal-reasoning", "multi-session"):
        raw.extend([
            _dataset_entry(category, f"{category}-new", (2,)),
            _dataset_entry(category, f"{category}-old", (0,)),
            _dataset_entry(category, f"{category}-distributed", (0, 1)),
        ])
    manifest = build_manifest(raw, per_category=2, excluded=set())
    assert len(manifest) == 6
    for category in {item["category"] for item in manifest}:
        selected = [item for item in manifest if item["category"] == category]
        assert selected[0]["distributed_evidence"] is True
        assert selected[1]["question_id"].endswith("-old")
        assert selected[0]["gold_source_turn_groups"] == [[1], [2]]


def test_native_arm_isolates_and_cleans_namespaces():
    states = {
        "q1": MemoryState(ledger=FactsLedger(entries=[
            FactEntry(kind="entity", text="works at Acme", source_turn_id=1)
        ])),
        "q2": MemoryState(ledger=FactsLedger(entries=[
            FactEntry(kind="entity", text="works at Globex", source_turn_id=2)
        ])),
    }
    arm = NativeSelectorArm(
        "packed", PackedLexicalSelector(), lambda namespace, sessions: states[namespace]
    )
    arm.ingest("q1", [])
    arm.ingest("q2", [])
    assert "Acme" in arm.recall("q1", "where works", 1000).rendered_context
    assert "Globex" not in arm.recall("q1", "where works", 1000).rendered_context
    arm.reset("q1")
    with pytest.raises(KeyError):
        arm.recall("q1", "where works", 1000)
    assert "Globex" in arm.recall("q2", "where works", 1000).rendered_context


class _HindsightClient:
    def __init__(self):
        self.banks = {}

    def retain(self, **kwargs):
        self.banks.setdefault(kwargs["bank_id"], []).append(kwargs)

    def recall(self, **kwargs):
        rows = self.banks[kwargs["bank_id"]]
        return SimpleNamespace(results=[SimpleNamespace(
            id="m1", text=rows[0]["content"], type="world",
            document_id=rows[0]["document_id"], metadata=rows[0]["metadata"],
        )])

    def delete_bank(self, bank_id):
        self.banks.pop(bank_id, None)


def test_hindsight_adapter_namespace_isolation_and_cleanup():
    client = _HindsightClient()
    arm = HindsightArm(client)
    arm.ingest("q1", [_session("s1", "Acme")])
    arm.ingest("q2", [_session("s2", "Globex")])
    result = arm.recall("q1", "employer", 1000)
    assert "Acme" in result.rendered_context
    assert result.source_references[0].source_id == "s1"
    arm.reset("q1")
    assert "q1" not in client.banks and "q2" in client.banks


def test_hindsight_adapter_rejects_malformed_recall_result():
    client = _HindsightClient()
    arm = HindsightArm(client)
    arm.ingest("q1", [_session()])
    client.recall = lambda **kwargs: {"results": [{"id": "broken"}]}
    with pytest.raises(ValueError, match="without text"):
        arm.recall("q1", "employer", 1000)


class _Documents:
    def __init__(self):
        self.status = "processing"
        self.deleted = []

    def get(self, document_id):
        return {"status": self.status}

    def delete_bulk(self, container_tags):
        self.deleted.extend(container_tags)


class _Search:
    def memories(self, **kwargs):
        return {"results": [{
            "id": "m1", "memory": f"memory for {kwargs['container_tag']}",
            "metadata": {"session_id": "s1"},
        }]}


class _SupermemoryClient:
    def __init__(self):
        self.documents = _Documents()
        self.search = _Search()
        self.added = []

    def add(self, **kwargs):
        self.added.append(kwargs)
        return {"id": f"d{len(self.added)}"}


def test_supermemory_readiness_timeout_retry_and_cleanup():
    client = _SupermemoryClient()
    arm = SupermemoryArm(client, poll_interval=0.001)
    arm.ingest("q1", [_session()])
    with pytest.raises(TimeoutError):
        arm.await_ready("q1", timeout=0.002)
    client.documents.status = "done"
    arm.await_ready("q1", timeout=0.01)
    result = arm.recall("q1", "employer", 1000)
    assert result.source_references[0].source_id == "s1"
    arm.reset("q1")
    assert client.documents.deleted == ["q1"]


def test_supermemory_readiness_surfaces_failed_ingestion():
    client = _SupermemoryClient()
    arm = SupermemoryArm(client, poll_interval=0)
    arm.ingest("q1", [_session()])
    client.documents.status = "failed"
    with pytest.raises(RuntimeError, match="failed"):
        arm.await_ready("q1", timeout=0.01)


def test_artifact_schema_enforces_token_and_provenance_accounting():
    with pytest.raises(ValidationError, match="context_overflow"):
        ItemRun(
            question_id="q", category="multi-session", arm="a",
            budget_tokens=8000, memory_tokens=8001, candidate_count=0,
            read_latency_ms=1,
        )
    with pytest.raises(ValidationError, match="provenance_lost"):
        ItemRun(
            question_id="q", category="multi-session", arm="a",
            budget_tokens=8000, memory_tokens=20, candidate_count=1,
            read_latency_ms=1,
        )
    artifact = MemoryMethodArtifact(
        repository_revision="abc", source_manifest="manifest.json",
        source_dataset_sha256="deadbeef",
    )
    assert artifact.models.answer == "gemma4-it:e2b"
    assert artifact.models.judge == "claude-haiku-4-5"


def _run(qid, category, correct, arm="candidate", latency=10):
    return ItemRun(
        question_id=qid, category=category, arm=arm,
        budget_tokens=8000, memory_tokens=100, candidate_count=0,
        read_latency_ms=latency, judged_correct=correct,
    )


def test_promotion_requires_three_net_wins_and_no_category_regression():
    baseline = [
        _run("q1", "knowledge-update", False, "baseline"),
        _run("q2", "knowledge-update", False, "baseline"),
        _run("q3", "temporal-reasoning", False, "baseline"),
        _run("q4", "multi-session", True, "baseline"),
    ]
    candidate = [
        _run("q1", "knowledge-update", True),
        _run("q2", "knowledge-update", True),
        _run("q3", "temporal-reasoning", True),
        _run("q4", "multi-session", True),
    ]
    decision = promotion_decision(baseline, candidate)
    assert decision["promote"] is True
    assert decision["wins"] == 3


def test_promotion_rejects_slow_or_provenance_losing_candidate():
    baseline = [_run(f"q{i}", "knowledge-update", False, "baseline") for i in range(3)]
    candidate = [_run(f"q{i}", "knowledge-update", True, latency=1001) for i in range(3)]
    candidate[0].provenance_lost = True
    decision = promotion_decision(baseline, candidate)
    assert decision["promote"] is False
    assert decision["checks"]["recall_p95_at_most_one_second"] is False
    assert decision["checks"]["no_provenance_loss"] is False
