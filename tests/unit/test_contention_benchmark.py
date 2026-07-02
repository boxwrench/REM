"""Unit tests for the contention benchmark harness (Task S2)."""

import json
from evals.contention.run_contention_benchmark import (
    generate_markdown_table,
    calculate_stats
)


def test_calculate_stats():
    """Asserts that calculate_stats correctly calculates mean, stddev, min, and max."""
    samples = [10.0, 12.0, 14.0]
    stats = calculate_stats(samples)
    assert stats["samples"] == [10.0, 12.0, 14.0]
    assert stats["mean"] == 12.0
    assert stats["stddev"] == 2.0
    assert stats["min"] == 10.0
    assert stats["max"] == 14.0

    # Test single sample
    single = calculate_stats([10.0])
    assert single["stddev"] == 0.0


def test_generate_markdown_table_full(tmp_path):
    """Asserts that generate_markdown_table correctly extracts and formats all metrics from JSON files."""
    m1_path = tmp_path / "m1_contention.json"
    m2_path = tmp_path / "m2_cpu_arm.json"

    m1_data = {
        "baseline": {
            "prefill": {
                "igpu_throughput_tok_s": {"samples": [1000.0], "mean": 1000.0, "stddev": 0.0, "min": 1000.0, "max": 1000.0},
                "avg_power_w": {"samples": [10.0], "mean": 10.0, "stddev": 0.0, "min": 10.0, "max": 10.0}
            },
            "decode": {
                "igpu_throughput_tok_s": {"samples": [50.0], "mean": 50.0, "stddev": 0.0, "min": 50.0, "max": 50.0},
                "avg_power_w": {"samples": [20.0], "mean": 20.0, "stddev": 0.0, "min": 20.0, "max": 20.0}
            }
        },
        "concurrent_npu": {
            "prefill": {
                "igpu_throughput_tok_s": {"samples": [950.0], "mean": 950.0, "stddev": 0.0, "min": 950.0, "max": 950.0},
                "npu_throughput_tok_s": {"samples": [15.0], "mean": 15.0, "stddev": 0.0, "min": 15.0, "max": 15.0},
                "avg_power_w": {"samples": [30.0], "mean": 30.0, "stddev": 0.0, "min": 30.0, "max": 30.0},
                "contention_loss_pct": None
            },
            "decode": {
                "igpu_throughput_tok_s": {"samples": [45.0, 50.0], "mean": 47.5, "stddev": 3.536, "min": 45.0, "max": 50.0},
                "npu_throughput_tok_s": {"samples": [14.0, 16.0], "mean": 15.0, "stddev": 1.414, "min": 14.0, "max": 16.0},
                "avg_power_w": {"samples": [33.0, 37.0], "mean": 35.0, "stddev": 2.828, "min": 33.0, "max": 37.0},
                "contention_loss_pct": {"samples": [10.0, 0.0], "mean": 5.0, "stddev": 7.071, "min": 0.0, "max": 10.0}
            }
        }
    }

    m2_data = {
        "concurrent_cpu": {
            "prefill": {
                "igpu_throughput_tok_s": {"samples": [990.0], "mean": 990.0, "stddev": 0.0, "min": 990.0, "max": 990.0},
                "cpu_throughput_tok_s": {"samples": [5.0], "mean": 5.0, "stddev": 0.0, "min": 5.0, "max": 5.0},
                "avg_power_w": {"samples": [40.0], "mean": 40.0, "stddev": 0.0, "min": 40.0, "max": 40.0},
                "contention_loss_pct": None
            },
            "decode": {
                "igpu_throughput_tok_s": {"samples": [49.0, 50.0], "mean": 49.5, "stddev": 0.707, "min": 49.0, "max": 50.0},
                "cpu_throughput_tok_s": {"samples": [4.0, 6.0], "mean": 5.0, "stddev": 1.414, "min": 4.0, "max": 6.0},
                "avg_power_w": {"samples": [44.0, 46.0], "mean": 45.0, "stddev": 1.414, "min": 44.0, "max": 46.0},
                "contention_loss_pct": {"samples": [2.0, 0.0], "mean": 1.0, "stddev": 1.414, "min": 0.0, "max": 2.0}
            }
        },
        "comparison": {
            "marginal_decode_power_w": {
                "npu": 15.0,
                "cpu": 25.0
            },
            "perf_watt": {
                "npu": 0.429,
                "cpu": 0.111
            }
        }
    }

    m1_path.write_text(json.dumps(m1_data), encoding="utf-8")
    m2_path.write_text(json.dumps(m2_data), encoding="utf-8")

    markdown = generate_markdown_table(m1_path, m2_path)

    # Assert correct header and formatting
    assert "# REM Contention & Placement Benchmark Results" in markdown
    assert "Marginal Power (W)" in markdown
    assert "Generation-tok/s per Total Board Watt" in markdown
    
    # Assert Baseline values
    assert "| **Baseline** (iGPU Only) | 1000.00 | 50.00 | 0.00% | - | 20.00 | - | - |" in markdown
    # Assert NPU values
    assert "| **NPU (Concurrent)** | 950.00 | 47.50 ± 3.54 | 5.00% ± 7.07% | 15.00 ± 1.41 | 35.00 ± 2.83 | +15.00 W | 0.429 |" in markdown
    # Assert CPU values
    assert "| **CPU (Concurrent)** | 990.00 | 49.50 ± 0.71 | 1.00% ± 1.41% | 5.00 ± 1.41 | 45.00 ± 1.41 | +25.00 W | 0.111 |" in markdown


def test_generate_markdown_table_partial_fallback(tmp_path):
    """Asserts that generate_markdown_table handles missing files or fields gracefully without crashing."""
    m1_path = tmp_path / "non_existent_m1.json"
    m2_path = tmp_path / "non_existent_m2.json"

    markdown = generate_markdown_table(m1_path, m2_path)

    # Table should still generate with safe fallback characters
    assert "| **Baseline** (iGPU Only) | 0.00 | 0.00 | 0.00% | - | 0.00 | - | - |" in markdown
    assert "| **NPU (Concurrent)** | 0.00 | 0.00 | - | - | 0.00 | - | - |" in markdown
    assert "| **CPU (Concurrent)** | 0.00 | 0.00 | - | - | 0.00 | - | - |" in markdown
