"""Thin evaluation-only adapters for pinned Hindsight and Supermemory clients.

These adapters intentionally use each system's native SDK contract. They do not
become REM runtime providers and do not substitute unsupported configurations.
"""
from __future__ import annotations

import time
from time import perf_counter
from typing import Any

from evals.battery.models import Session
from evals.memory_methods.contracts import RecallResult, SourceReference
from rem.memory.tiers import count_tokens


def _session_text(session: Session) -> str:
    return "\n".join(
        f"{turn.get('role', 'user').upper()}: {turn.get('content', '')}"
        for turn in session.turns
    )


def _field(value: Any, name: str, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _pack_texts(items: list[tuple[str, SourceReference]], budget_tokens: int):
    rendered, refs, used = [], [], 0
    for text, ref in items:
        cost = count_tokens(f"- {text}")
        if used + cost > budget_tokens:
            continue
        rendered.append(f"- {text}")
        refs.append(ref)
        used += cost
    return "\n".join(rendered), tuple(refs)


class HindsightArm:
    """Hindsight banks provide one isolated namespace per question."""

    name = "hindsight"

    def __init__(self, client: Any) -> None:
        self.client = client
        self._stats: dict[str, dict[str, Any]] = {}

    def ingest(self, namespace: str, sessions: list[Session]) -> None:
        started = perf_counter()
        for session in sessions:
            self.client.retain(
                bank_id=namespace,
                content=_session_text(session),
                context=session.session_id,
                document_id=session.session_id,
                metadata={"session_id": session.session_id},
            )
        self._stats[namespace] = {
            "sessions_ingested": len(sessions),
            "ingest_latency_ms": round((perf_counter() - started) * 1000, 3),
            "read_latencies_ms": [],
        }

    def await_ready(self, namespace: str, timeout: float) -> None:
        # Recommended native retain is synchronous. Observation consolidation is
        # background work and is not silently treated as part of this pass.
        if namespace not in self._stats:
            raise KeyError(f"unknown namespace: {namespace}")

    def recall(self, namespace: str, query: str, budget_tokens: int) -> RecallResult:
        if namespace not in self._stats:
            raise KeyError(f"unknown namespace: {namespace}")
        started = perf_counter()
        response = self.client.recall(
            bank_id=namespace, query=query, max_tokens=budget_tokens
        )
        results = _field(response, "results", [])
        normalized = []
        for result in results:
            text = _field(result, "text", "")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Hindsight returned a memory without text")
            metadata = _field(result, "metadata", {}) or {}
            source_id = _field(result, "document_id") or metadata.get("session_id")
            if source_id is None:
                source_id = _field(result, "id", "unknown")
                metadata = {**metadata, "provenance": "memory_only"}
            normalized.append((text, SourceReference(
                source_id=str(source_id), kind=str(_field(result, "type", "memory")),
                metadata={str(key): str(value) for key, value in metadata.items()},
            )))
        context, refs = _pack_texts(normalized, budget_tokens)
        latency = round((perf_counter() - started) * 1000, 3)
        self._stats[namespace]["read_latencies_ms"].append(latency)
        return RecallResult(
            context, count_tokens(context), refs, len(results), latency
        )

    def stats(self, namespace: str) -> dict[str, Any]:
        return dict(self._stats[namespace])

    def reset(self, namespace: str) -> None:
        delete = getattr(self.client, "delete_bank", None)
        if delete is None:
            raise NotImplementedError(
                "the pinned Hindsight client must expose delete_bank for cleanup"
            )
        delete(bank_id=namespace)
        self._stats.pop(namespace, None)


class SupermemoryArm:
    """Supermemory container tags provide one isolated namespace per question."""

    name = "supermemory"

    def __init__(self, client: Any, poll_interval: float = 0.25) -> None:
        self.client = client
        self.poll_interval = poll_interval
        self._documents: dict[str, list[str]] = {}
        self._stats: dict[str, dict[str, Any]] = {}

    def ingest(self, namespace: str, sessions: list[Session]) -> None:
        started = perf_counter()
        document_ids = []
        for session in sessions:
            response = self.client.add(
                content=_session_text(session),
                custom_id=f"{namespace}:{session.session_id}",
                container_tag=namespace,
                metadata={"session_id": session.session_id},
            )
            document_ids.append(str(_field(response, "id")))
        self._documents[namespace] = document_ids
        self._stats[namespace] = {
            "sessions_ingested": len(sessions),
            "ingest_latency_ms": round((perf_counter() - started) * 1000, 3),
            "read_latencies_ms": [],
        }

    def await_ready(self, namespace: str, timeout: float) -> None:
        ids = self._documents.get(namespace)
        if ids is None:
            raise KeyError(f"unknown namespace: {namespace}")
        deadline = time.monotonic() + timeout
        pending = set(ids)
        while pending:
            for document_id in list(pending):
                document = self.client.documents.get(document_id)
                status = str(_field(document, "status", "")).lower()
                if status in {"done", "completed"}:
                    pending.remove(document_id)
                elif status in {"failed", "error"}:
                    raise RuntimeError(f"document {document_id} failed: {status}")
            if not pending:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Supermemory namespace {namespace} not ready; pending={sorted(pending)}"
                )
            time.sleep(min(self.poll_interval, max(0.0, deadline - time.monotonic())))

    def recall(self, namespace: str, query: str, budget_tokens: int) -> RecallResult:
        if namespace not in self._stats:
            raise KeyError(f"unknown namespace: {namespace}")
        started = perf_counter()
        response = self.client.search.memories(
            q=query, container_tag=namespace, search_mode="memories", limit=100
        )
        results = _field(response, "results", [])
        normalized = []
        for result in results:
            text = _field(result, "memory") or _field(result, "chunk", "")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Supermemory returned a result without memory or chunk text")
            metadata = _field(result, "metadata", {}) or {}
            source_id = metadata.get("session_id")
            if source_id is None:
                source_id = _field(result, "id", "unknown")
                metadata = {**metadata, "provenance": "memory_only"}
            normalized.append((text, SourceReference(
                source_id=str(source_id), kind="memory",
                metadata={str(key): str(value) for key, value in metadata.items()},
            )))
        context, refs = _pack_texts(normalized, budget_tokens)
        latency = round((perf_counter() - started) * 1000, 3)
        self._stats[namespace]["read_latencies_ms"].append(latency)
        return RecallResult(context, count_tokens(context), refs, len(results), latency)

    def stats(self, namespace: str) -> dict[str, Any]:
        return dict(self._stats[namespace])

    def reset(self, namespace: str) -> None:
        self.client.documents.delete_bulk(container_tags=[namespace])
        self._documents.pop(namespace, None)
        self._stats.pop(namespace, None)
