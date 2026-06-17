"""Unit tests for the NpuClient utilizing the respx mock NPU server."""

import pytest
import httpx
from rem.npu_client import NpuClient, NpuUnavailable


def test_npu_client_default_chat(mock_npu):
    """Assert chat completion parses the response correctly."""
    client = NpuClient()
    response = client.chat([{"role": "user", "content": "Hello"}])
    assert response == "Default mock chat response"
    assert mock_npu.calls.call_count == 1
    
    # Verify the request payload details
    last_request = mock_npu.calls.last.request
    assert last_request.method == "POST"
    assert last_request.url.path == "/v1/chat/completions"


def test_npu_client_custom_chat(mock_npu):
    """Assert custom chat response injection works."""
    client = NpuClient()
    mock_npu.post("/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Custom response text"}}]},
        )
    )
    response = client.chat([{"role": "user", "content": "Ping"}], model="phi3")
    assert response == "Custom response text"


def test_npu_client_embed(mock_npu):
    """Assert embeddings parse correctly."""
    client = NpuClient()
    embeddings = client.embed(["test text"])
    assert embeddings == [[0.1, 0.2, 0.3]]
    assert mock_npu.calls.call_count == 1
    
    last_request = mock_npu.calls.last.request
    assert last_request.url.path == "/v1/embeddings"


def test_npu_client_health(mock_npu):
    """Assert health check parses true/false states correctly."""
    client = NpuClient()
    assert client.health() is True
    
    # Mock a health check failure
    mock_npu.get("/v1/models").mock(return_value=httpx.Response(500))
    assert client.health() is False


def test_npu_client_retry_success(mock_npu):
    """Assert that a request succeeds if the first try fails but retry succeeds."""
    client = NpuClient()
    
    # Sequence of responses: first fails with connect error, second succeeds
    mock_npu.post("/v1/chat/completions").mock(
        side_effect=[
            httpx.ConnectError("Connection failed"),
            httpx.Response(200, json={"choices": [{"message": {"content": "Succeeded on retry"}}]})
        ]
    )
    
    response = client.chat([{"role": "user", "content": "Ping"}])
    assert response == "Succeeded on retry"
    # Verify it was tried twice
    assert mock_npu.calls.call_count == 2


def test_npu_client_retry_failure(mock_npu):
    """Assert that NpuUnavailable is raised if both try and retry fail."""
    client = NpuClient()
    
    # Both calls fail
    mock_npu.post("/v1/chat/completions").mock(
        side_effect=httpx.ConnectError("Connection failed")
    )
    
    with pytest.raises(NpuUnavailable) as exc_info:
        client.chat([{"role": "user", "content": "Ping"}])
        
    assert "NPU service request failed" in str(exc_info.value)
    # Verify it attempted once and then retried once (total 2 attempts)
    assert mock_npu.calls.call_count == 2
