"""Unit tests for the memory sidecar service (Task S1)."""

import json
import socket
import threading
import time
import pytest
import respx
import httpx

from rem.config import Settings
from rem.memory.sidecar import MemorySidecar, MemorySidecarServer
from rem.memory.tiers import MemoryState


def _find_free_port() -> int:
    """Finds a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_sidecar_request_processing(tmp_path, mock_npu):
    """Asserts that process_chat_request ingests turns, saves state, and assembles the context."""
    settings = Settings(
        vault_dir=str(tmp_path),
        litellm_port=4000,
        npu_server_port=13306,
    )
    sidecar = MemorySidecar(settings=settings)

    # Initial request
    request_payload = {
        "model": "qwen",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello sidecar!"},
        ],
        "user": "test_session",
    }

    modified_payload, state_path = sidecar.process_chat_request(request_payload)

    # Verify state was saved
    state = MemoryState.load(state_path)
    assert len(state.turns) == 1
    assert state.turns[0].role == "user"
    assert state.turns[0].content == "Hello sidecar!"
    assert state.turns[0].turn_id == 1

    # Verify payload was modified with assembled context
    assert len(modified_payload["messages"]) == 2  # system prompt + recent turn
    assert modified_payload["messages"][0]["role"] == "system"
    assert "=== SYSTEM ===" in modified_payload["messages"][0]["content"]
    assert "Hello sidecar!" in modified_payload["messages"][1]["content"]

    # Record the assistant reply
    sidecar.record_response(state_path, "Response from assistant")

    # Verify assistant turn was recorded
    state = MemoryState.load(state_path)
    assert len(state.turns) == 2
    assert state.turns[1].role == "assistant"
    assert state.turns[1].content == "Response from assistant"
    assert state.turns[1].turn_id == 2


def test_sidecar_turn_deduplication(tmp_path):
    """Asserts that the sidecar aligns and appends only new turns, preventing duplication."""
    settings = Settings(vault_dir=str(tmp_path))
    sidecar = MemorySidecar(settings=settings)

    # 1. First request
    payload_1 = {
        "model": "qwen",
        "messages": [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Turn A"},
        ],
        "user": "test_dedup",
    }
    _, state_path = sidecar.process_chat_request(payload_1)
    sidecar.record_response(state_path, "Reply B")

    state = MemoryState.load(state_path)
    assert [t.content for t in state.turns] == ["Turn A", "Reply B"]

    # 2. Second request with full history + new message
    payload_2 = {
        "model": "qwen",
        "messages": [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Turn A"},
            {"role": "assistant", "content": "Reply B"},
            {"role": "user", "content": "Turn C"},
        ],
        "user": "test_dedup",
    }
    _, _ = sidecar.process_chat_request(payload_2)

    state = MemoryState.load(state_path)
    # Turn C should be appended, but Turn A and Reply B must not be duplicated!
    assert [t.content for t in state.turns] == ["Turn A", "Reply B", "Turn C"]


@respx.mock
def test_sidecar_server_routing(tmp_path, mock_npu):
    """Integration test verifying that the Sidecar HTTP server receives, forwards, and records requests."""
    # Find free port for testing
    test_port = _find_free_port()
    
    # Mock the downstream LiteLLM/completions endpoint
    downstream_port = 4000
    respx.post(f"http://localhost:{downstream_port}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Server Response",
                        }
                    }
                ]
            },
        )
    )
    # Allow the client requests to actually hit the running test sidecar server
    respx.post(f"http://127.0.0.1:{test_port}/v1/chat/completions").pass_through()

    settings = Settings(
        vault_dir=str(tmp_path),
        sidecar_port=test_port,
        litellm_port=downstream_port,
    )
    sidecar = MemorySidecar(settings=settings)
    server = MemorySidecarServer(sidecar=sidecar)

    # Start server in daemon thread
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    # Wait briefly for server startup
    time.sleep(0.2)

    # Send client request to sidecar
    client_payload = {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "Test system prompt"},
            {"role": "user", "content": "Hello Server!"},
        ],
        "user": "server_session",
    }

    try:
        response = httpx.post(
            f"http://127.0.0.1:{test_port}/v1/chat/completions",
            json=client_payload,
            timeout=5.0,
        )
        assert response.status_code == 200
        res_data = response.json()
        assert res_data["choices"][0]["message"]["content"] == "Server Response"

        # Verify that the response was recorded in the state file
        state_path = tmp_path / "server_session_memory_state.json"
        assert state_path.exists()
        state = MemoryState.load(state_path)
        assert len(state.turns) == 2
        assert state.turns[0].content == "Hello Server!"
        assert state.turns[1].content == "Server Response"
    finally:
        server.shutdown()
        server_thread.join(timeout=1.0)
