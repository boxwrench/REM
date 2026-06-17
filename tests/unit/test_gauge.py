"""Unit tests for the C1 telemetry gauge."""

import json
from unittest.mock import patch, MagicMock
import pytest
from rem.config import Settings
from rem.scheduler.gauge import (
    GpuState,
    GaugeReading,
    classify_state,
    get_stable_state,
    parse_xrt_smi,
    HardwareGauge,
)


def test_classify_state():
    settings = Settings(gpu_idle_busy_pct=10, gpu_prefill_power_w=35.0)

    # IDLE
    assert classify_state(5, 5.0, settings) == GpuState.IDLE
    assert classify_state(10, 30.0, settings) == GpuState.IDLE

    # ACTIVE (busy, power below prefill)
    assert classify_state(15, 20.0, settings) == GpuState.ACTIVE
    assert classify_state(50, 30.0, settings) == GpuState.ACTIVE

    # PREFILL_BURST (busy, power at/above prefill)
    assert classify_state(15, 35.0, settings) == GpuState.PREFILL_BURST
    assert classify_state(90, 50.0, settings) == GpuState.PREFILL_BURST


def test_get_stable_state():
    # Single element
    assert get_stable_state([GpuState.IDLE]) == GpuState.IDLE

    # Majority vote
    assert get_stable_state([GpuState.IDLE, GpuState.ACTIVE, GpuState.IDLE]) == GpuState.IDLE
    assert get_stable_state([GpuState.PREFILL_BURST, GpuState.ACTIVE, GpuState.ACTIVE]) == GpuState.ACTIVE

    # Tie break (priority: PREFILL_BURST > ACTIVE > IDLE)
    assert get_stable_state([GpuState.IDLE, GpuState.ACTIVE]) == GpuState.ACTIVE
    assert get_stable_state([GpuState.PREFILL_BURST, GpuState.ACTIVE]) == GpuState.PREFILL_BURST


def test_parse_xrt_smi():
    canned_idle = """
------------------------------
[0000:c6:00.1] : RyzenAI-npu5
------------------------------
AIE Partitions
  Total Memory Usage: N/A
  Partition Index   : 0
    Columns: [0, 1, 2, 3, 4, 5, 6, 7]
    HW Contexts:
      |PID                 |Ctx ID     |Submissions |Migrations  |Err  |Priority |
      |Process Name        |Status     |Completions |Suspensions |     |GOPS     |
      |Memory Usage        |Instr BO   |            |            |     |FPS      |
      |                    |           |            |            |     |Latency  |
      |====================|===========|============|============|=====|=========|
      |93941               |1          |15399       |0           |0    |N/A      |
      |N/A                 |Active     |15399       |0           |     |N/A      |
      |--------------------|-----------|------------|------------|-----|---------|
      |93941               |2          |2800        |0           |0    |N/A      |
      |N/A                 |Active     |2800        |0           |     |N/A      |
"""
    res = parse_xrt_smi(canned_idle)
    assert len(res) == 2
    assert res[0]["pid"] == 93941
    assert res[0]["ctx_id"] == 1
    assert res[0]["submissions"] == 15399
    assert res[0]["completions"] == 15399
    assert res[0]["status"] == "Active"


@patch("xdna_top.gauge.run_xrt_smi")
@patch("xdna_top.gauge.read_igpu")
def test_hardware_gauge_direct_read(mock_read_igpu, mock_run_xrt_smi, tmp_path):
    mock_read_igpu.return_value = (50, 25.0, False)  # active, not degraded
    mock_run_xrt_smi.return_value = """
|====================|===========|============|============|=====|=========|
|93941               |1          |15000       |0           |0    |N/A      |
|N/A                 |Active     |14999       |0           |     |N/A      |
"""  # in-flight submissions > completions => npu_active
    
    settings = Settings(
        gpu_idle_busy_pct=10,
        gpu_prefill_power_w=35.0,
        bench_dir=str(tmp_path),
    )
    gauge = HardwareGauge(settings)
    reading = gauge.read_direct()
    
    assert reading.gpu_busy_pct == 50
    assert reading.gpu_power_w == 25.0
    assert reading.npu_active is True
    assert reading.state == GpuState.ACTIVE


@patch("xdna_top.gauge.load_sysfs_paths")
@patch("builtins.open")
@patch("os.path.exists")
def test_igpu_degradation_and_pessimism(mock_exists, mock_open, mock_load_paths, tmp_path):
    # Simulate both paths missing
    mock_exists.return_value = False
    mock_load_paths.return_value = (None, None)
    
    settings = Settings(gpu_idle_busy_pct=10, bench_dir=str(tmp_path))
    gauge = HardwareGauge(settings)
    reading = gauge.read_direct()
    
    # Should degrade to active-pessimistic
    assert reading.gpu_busy_pct == 100
    assert reading.gpu_power_w == 45.0
    assert reading.state == GpuState.PREFILL_BURST  # because power 45 >= prefill 35
