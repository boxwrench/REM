"""Integration tests for NPU chat completions using the real hardware."""

import pytest
import time
from rem.config import Settings
from rem.npu_client import NpuClient


@pytest.mark.npu
def test_real_npu_chat():
    """Verify that chat completions actually run on the NPU via the real NpuClient."""
    settings = Settings()
    client = NpuClient(settings)
    
    # Assert NPU server health
    assert client.health() is True, "NPU server health check failed."
    
    # Measure latency
    start_time = time.perf_counter()
    response = client.chat(
        messages=[{"role": "user", "content": "Reply with the single word READY"}],
        temperature=0.0
    )
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    # Assert response contains text
    assert len(response.strip()) > 0, "NPU returned an empty response."
    
    # Check that latency is captured
    print(f"\n[INTEGRATION] NPU completions call completed in {duration:.4f}s")
    print(f"[INTEGRATION] Model response: {response.strip()}")
    assert duration > 0
