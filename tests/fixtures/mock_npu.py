"""Reusable mock NPU server fixture for unit tests."""

import pytest
import respx
import httpx


from rem.config import Settings


@pytest.fixture
def mock_npu():
    """Mocks the NPU server endpoints using respx.

    Tests can use this fixture to inspect requests or override mock responses.
    """
    settings = Settings()
    # Create a respx mock context with the configured NPU server base url
    with respx.mock(base_url=f"http://localhost:{settings.npu_server_port}", assert_all_called=False) as respx_mock:
        # Default mock for chat completions
        respx_mock.post("/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "Default mock chat response",
                            },
                            "finish_reason": "stop",
                        }
                    ]
                },
            )
        )

        # Default mock for embeddings
        respx_mock.post("/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "embedding": [0.1, 0.2, 0.3],
                            "index": 0,
                            "object": "embedding",
                        }
                    ]
                },
            )
        )

        # Default mock for models (health check)
        respx_mock.get("/v1/models").mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": [
                        {"id": "phi3.5", "object": "model"},
                        {"id": "npu-embeds", "object": "model"},
                    ]
                },
            )
        )

        yield respx_mock
