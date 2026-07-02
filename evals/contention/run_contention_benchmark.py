#!/usr/bin/env python3
"""M1/M2 Contention and CPU Control Arm Benchmark.

Measures shared bandwidth contention and performance/watt on Strix Halo:
1. Baseline: iGPU only (no NPU/CPU active).
2. NPU Concurrent: iGPU + background NPU generation job (Gemma 4-2B on NPU).
3. CPU Concurrent: iGPU + background CPU generation job (Gemma 4-12B on CPU).

Saves results to:
- bench/m1_contention.json
- bench/m2_cpu_arm.json
- bench/RESULTS.md
"""

import os
import sys
import json
import time
import httpx
import threading
import subprocess
import argparse
import statistics
from pathlib import Path
from rem.config import Settings
from rem.scheduler.gauge import HardwareGauge


def calculate_stats(samples: list[float]) -> dict:
    """Calculates mean, stddev, min, and max for a list of numeric samples."""
    if not samples:
        return {
            "samples": [],
            "mean": 0.0,
            "stddev": 0.0,
            "min": 0.0,
            "max": 0.0
        }
    mean = sum(samples) / len(samples)
    stddev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return {
        "samples": [round(x, 3) for x in samples],
        "mean": round(mean, 3),
        "stddev": round(stddev, 3),
        "min": round(min(samples), 3),
        "max": round(max(samples), 3)
    }


# Centralized default port assignments
DEFAULT_IGPU_PORT = 8094
DEFAULT_NPU_PORT = 13306
DEFAULT_CPU_PORT = 8095


class PowerSampler:
    """Samples iGPU/Package power in a background thread."""
    def __init__(self, interval_s: float = 0.1, power_path: str = "/sys/class/hwmon/hwmon5/power1_input"):
        self.interval_s = interval_s
        self.power_path = power_path
        self.powers = []
        self.running = False
        self.thread = None

    def _loop(self):
        while self.running:
            try:
                if os.path.exists(self.power_path):
                    with open(self.power_path, "r") as f:
                        val = int(f.read().strip())
                        self.powers.append(val / 1_000_000.0)
            except Exception:
                pass
            time.sleep(self.interval_s)

    def start(self):
        self.powers = []
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> float:
        self.running = False
        if self.thread:
            self.thread.join()
        return sum(self.powers) / len(self.powers) if self.powers else 0.0


class BackgroundLoad:
    """Generates continuous LLM completion requests to NPU/CPU server."""
    def __init__(self, port: int, model: str, max_tokens: int = 50):
        self.url = f"http://localhost:{port}"
        self.model = model
        self.max_tokens = max_tokens
        self.running = False
        self.tokens_generated = 0
        self.requests_sent = 0
        self.start_time = None
        self.elapsed_time = 0.0
        self.thread = None
        self.samples = []

    def _loop(self):
        self.start_time = time.time()
        client = httpx.Client()
        while self.running:
            try:
                start_req = time.perf_counter()
                r = client.post(
                    f"{self.url}/v1/completions",
                    json={
                        "model": self.model,
                        "prompt": "Explain the step-by-step process of sorting an array of strings in C++.",
                        "max_tokens": self.max_tokens,
                        "temperature": 0.0
                    },
                    timeout=120.0
                )
                duration = time.perf_counter() - start_req
                if r.status_code == 200:
                    data = r.json()
                    tokens = data["usage"]["completion_tokens"]
                    self.tokens_generated += tokens
                    self.requests_sent += 1
                    if duration > 0:
                        self.samples.append(tokens / duration)
                else:
                    print(f"BackgroundLoad HTTP error on port {self.url[-5:]}: {r.status_code} {r.text}", file=sys.stderr)
            except Exception as e:
                print(f"BackgroundLoad Exception on port {self.url[-5:]}: {e}", file=sys.stderr)
                time.sleep(0.1)
        self.elapsed_time = time.time() - self.start_time

    def start(self):
        self.running = True
        self.tokens_generated = 0
        self.requests_sent = 0
        self.samples = []
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> float:
        self.running = False
        if self.thread:
            self.thread.join()
        return self.tokens_generated / self.elapsed_time if self.elapsed_time > 0 else 0.0

    def get_stats(self) -> dict:
        return calculate_stats(self.samples)


def cleanup_ports(ports=(DEFAULT_NPU_PORT, DEFAULT_CPU_PORT)):
    """Kills any process bound to the NPU/CPU benchmark ports."""
    for port in ports:
        try:
            out = subprocess.check_output(["fuser", f"{port}/tcp"], text=True, stderr=subprocess.DEVNULL)
            pids = [int(p) for p in out.strip().split() if p.isdigit()]
            for pid in pids:
                print(f"Cleaning up port {port} by terminating PID {pid}")
                subprocess.run(["kill", "-9", str(pid)], stderr=subprocess.DEVNULL)
        except Exception:
            pass


def wait_for_server(port: int, timeout_s: float = 30.0) -> bool:
    """Polls the server /v1/models endpoint until responsive."""
    start = time.time()
    client = httpx.Client()
    while time.time() - start < timeout_s:
        try:
            r = client.get(f"http://localhost:{port}/v1/models")
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def run_igpu_prefill(port: int, prompt: str) -> tuple[float, int]:
    """Runs a single iGPU prefill benchmark and returns (throughput, prompt_tokens)."""
    start = time.perf_counter()
    r = httpx.post(
        f"http://localhost:{port}/v1/completions",
        json={
            "prompt": prompt,
            "max_tokens": 1
        },
        timeout=30.0
    )
    end = time.perf_counter()
    assert r.status_code == 200, f"iGPU prefill failed: {r.status_code}"
    data = r.json()
    prompt_tokens = data["usage"]["prompt_tokens"]
    duration = end - start
    return prompt_tokens / duration, prompt_tokens


def run_igpu_decode(port: int, max_tokens: int = 150) -> float:
    """Runs a single iGPU decode benchmark by streaming and returns decode throughput."""
    client = httpx.Client()
    first_token_time = None
    last_token_time = None
    num_tokens = 0

    with client.stream(
        "POST",
        f"http://localhost:{port}/v1/completions",
        json={
            "prompt": "Write a highly detailed explanation of quantum computing principles and superposition.",
            "max_tokens": max_tokens,
            "stream": True
        },
        timeout=30.0
    ) as r:
        assert r.status_code == 200, f"iGPU decode failed: {r.status_code}"
        for line in r.iter_lines():
            if line.startswith("data:"):
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    json.loads(data_str)
                    num_tokens += 1
                    ts = time.perf_counter()
                    if num_tokens == 1:
                        first_token_time = ts
                    else:
                        last_token_time = ts
                except Exception:
                    pass

    if first_token_time and last_token_time and num_tokens > 1:
        duration = last_token_time - first_token_time
        return (num_tokens - 1) / duration
    return 0.0


def generate_markdown_table(m1_path: Path, m2_path: Path) -> str:
    """Generates a concise Markdown results table comparing Baseline, NPU, and CPU states."""
    m1_data = {}
    if m1_path.exists():
        try:
            m1_data = json.loads(m1_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Warning: failed to read {m1_path}: {e}", file=sys.stderr)

    m2_data = {}
    if m2_path.exists():
        try:
            m2_data = json.loads(m2_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Warning: failed to read {m2_path}: {e}", file=sys.stderr)

    def get_val_and_stddev(data, path_keys) -> tuple[float, float | None]:
        curr = data
        for k in path_keys:
            if isinstance(curr, dict) and k in curr:
                curr = curr[k]
            else:
                return 0.0, None
        if isinstance(curr, dict):
            return curr.get("mean", 0.0), curr.get("stddev", None)
        elif isinstance(curr, (int, float)):
            return float(curr), None
        return 0.0, None

    def format_val_stddev(val: float, stddev: float | None, precision: int = 2, is_pct: bool = False) -> str:
        suffix = "%" if is_pct else ""
        if stddev is not None and stddev > 0.0:
            return f"{val:.{precision}f}{suffix} ± {stddev:.{precision}f}{suffix}"
        return f"{val:.{precision}f}{suffix}"

    # Extract metrics, with safe fallbacks
    base_prefill_tput, base_prefill_tput_sd = get_val_and_stddev(m1_data, ["baseline", "prefill", "igpu_throughput_tok_s"])
    base_decode_tput, base_decode_tput_sd = get_val_and_stddev(m1_data, ["baseline", "decode", "igpu_throughput_tok_s"])
    base_decode_pwr, base_decode_pwr_sd = get_val_and_stddev(m1_data, ["baseline", "decode", "avg_power_w"])

    npu_prefill_tput, npu_prefill_tput_sd = get_val_and_stddev(m1_data, ["concurrent_npu", "prefill", "igpu_throughput_tok_s"])
    npu_decode_tput, npu_decode_tput_sd = get_val_and_stddev(m1_data, ["concurrent_npu", "decode", "igpu_throughput_tok_s"])
    npu_decode_pwr, npu_decode_pwr_sd = get_val_and_stddev(m1_data, ["concurrent_npu", "decode", "avg_power_w"])
    npu_decode_loss, npu_decode_loss_sd = get_val_and_stddev(m1_data, ["concurrent_npu", "decode", "contention_loss_pct"])
    
    npu_tput, npu_tput_sd = get_val_and_stddev(m1_data, ["concurrent_npu", "decode", "npu_throughput_tok_s"])

    cpu_prefill_tput, cpu_prefill_tput_sd = get_val_and_stddev(m2_data, ["concurrent_cpu", "prefill", "igpu_throughput_tok_s"])
    cpu_decode_tput, cpu_decode_tput_sd = get_val_and_stddev(m2_data, ["concurrent_cpu", "decode", "igpu_throughput_tok_s"])
    cpu_decode_pwr, cpu_decode_pwr_sd = get_val_and_stddev(m2_data, ["concurrent_cpu", "decode", "avg_power_w"])
    cpu_decode_loss, cpu_decode_loss_sd = get_val_and_stddev(m2_data, ["concurrent_cpu", "decode", "contention_loss_pct"])
    
    cpu_tput, cpu_tput_sd = get_val_and_stddev(m2_data, ["concurrent_cpu", "decode", "cpu_throughput_tok_s"])

    comparison = m2_data.get("comparison", {})
    npu_perf_watt = comparison.get("perf_watt", {}).get("npu", 0.0)
    cpu_perf_watt = comparison.get("perf_watt", {}).get("cpu", 0.0)
    
    npu_marginal_pwr = comparison.get("marginal_decode_power_w", {}).get("npu", 0.0)
    cpu_marginal_pwr = comparison.get("marginal_decode_power_w", {}).get("cpu", 0.0)

    if not npu_perf_watt and npu_tput and npu_decode_pwr:
        npu_perf_watt = npu_tput / npu_decode_pwr
    if not cpu_perf_watt and cpu_tput and cpu_decode_pwr:
        cpu_perf_watt = cpu_tput / cpu_decode_pwr
    if not npu_marginal_pwr and npu_decode_pwr and base_decode_pwr:
        npu_marginal_pwr = npu_decode_pwr - base_decode_pwr
    if not cpu_marginal_pwr and cpu_decode_pwr and base_decode_pwr:
        cpu_marginal_pwr = cpu_decode_pwr - base_decode_pwr

    b_prefill_str = format_val_stddev(base_prefill_tput, base_prefill_tput_sd)
    b_decode_str = format_val_stddev(base_decode_tput, base_decode_tput_sd)
    b_pwr_str = format_val_stddev(base_decode_pwr, base_decode_pwr_sd)

    n_prefill_str = format_val_stddev(npu_prefill_tput, npu_prefill_tput_sd)
    n_decode_str = format_val_stddev(npu_decode_tput, npu_decode_tput_sd)
    n_loss_str = format_val_stddev(npu_decode_loss, npu_decode_loss_sd, is_pct=True) if npu_decode_tput > 0.0 else "-"
    n_tput_str = format_val_stddev(npu_tput, npu_tput_sd) if npu_tput > 0.0 else "-"
    n_pwr_str = format_val_stddev(npu_decode_pwr, npu_decode_pwr_sd)
    n_marg_str = f"{npu_marginal_pwr:+.2f} W" if npu_decode_pwr > 0.0 else "-"
    n_pw_str = f"{npu_perf_watt:.3f}" if npu_perf_watt > 0.0 else "-"

    c_prefill_str = format_val_stddev(cpu_prefill_tput, cpu_prefill_tput_sd)
    c_decode_str = format_val_stddev(cpu_decode_tput, cpu_decode_tput_sd)
    c_loss_str = format_val_stddev(cpu_decode_loss, cpu_decode_loss_sd, is_pct=True) if cpu_decode_tput > 0.0 else "-"
    c_tput_str = format_val_stddev(cpu_tput, cpu_tput_sd) if cpu_tput > 0.0 else "-"
    c_pwr_str = format_val_stddev(cpu_decode_pwr, cpu_decode_pwr_sd)
    c_marg_str = f"{cpu_marginal_pwr:+.2f} W" if cpu_decode_pwr > 0.0 else "-"
    c_pw_str = f"{cpu_perf_watt:.3f}" if cpu_perf_watt > 0.0 else "-"

    table = [
        "# REM Contention & Placement Benchmark Results",
        "",
        "| Condition | iGPU Prefill (tok/s) | iGPU Decode (tok/s) | iGPU Decode Loss % | Background Throughput (tok/s) | Avg Decode Power (W) | Marginal Power (W) | Generation-tok/s per Total Board Watt |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **Baseline** (iGPU Only) | {b_prefill_str} | {b_decode_str} | 0.00% | - | {b_pwr_str} | - | - |",
        f"| **NPU (Concurrent)** | {n_prefill_str} | {n_decode_str} | {n_loss_str} | {n_tput_str} | {n_pwr_str} | {n_marg_str} | {n_pw_str} |",
        f"| **CPU (Concurrent)** | {c_prefill_str} | {c_decode_str} | {c_loss_str} | {c_tput_str} | {c_pwr_str} | {c_marg_str} | {c_pw_str} |",
        "",
        "*Note: Prefill contention is below the measurement resolution of this HTTP-based harness.*",
        ""
    ]

    return "\n".join(table)


def main():
    parser = argparse.ArgumentParser(description="REM M1/M2 Contention and Placement Benchmark.")
    parser.add_argument("--igpu-port", type=int, default=DEFAULT_IGPU_PORT, help="Port for the main iGPU server")
    parser.add_argument("--npu-port", type=int, default=DEFAULT_NPU_PORT, help="Port for the NPU server")
    parser.add_argument("--cpu-port", type=int, default=DEFAULT_CPU_PORT, help="Port for the CPU server")
    parser.add_argument("--cpu-model-path", type=str, default=None, help="Path to CPU model GGUF (required for the --cpu control arm)")
    parser.add_argument("--llama-server-path", type=str, default=None, help="Path to the llama-server binary (required for the --cpu control arm)")
    parser.add_argument("--npu-model", type=str, default="gemma4-it:e2b", help="NPU model name")
    parser.add_argument("--background-max-tokens", type=int, default=50, help="Background-load generation length per request (default 50 = MP1 baseline; ~150 ≈ REM's real compaction output length)")
    parser.add_argument("--trials", type=int, default=5, help="Number of trials for benchmarks")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save JSON results")
    parser.add_argument("--results-md", type=str, default=None, help="Markdown file path to save results table")
    parser.add_argument("--power-path", type=str, default=None, help="Custom path to power input sysfs file")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode using simulated telemetry")
    parser.add_argument("--skip-npu", action="store_true", help="Skip NPU concurrency benchmark")
    parser.add_argument("--skip-cpu", action="store_true", help="Skip CPU concurrency benchmark")
    parser.add_argument("--generate-table-only", action="store_true", help="Only generate the markdown table from existing JSON files")

    args = parser.parse_args()

    # Determine default paths relative to script location
    script_dir = Path(__file__).resolve().parent
    default_bench_dir = script_dir.parent.parent / "bench"
    output_dir = Path(args.output_dir) if args.output_dir else default_bench_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select filenames based on dry-run state to protect committed measurements
    if args.dry_run:
        m1_filename = "m1_contention_dryrun.json"
        m2_filename = "m2_cpu_arm_dryrun.json"
        results_md_filename = "RESULTS_dryrun.md"
    else:
        m1_filename = "m1_contention.json"
        m2_filename = "m2_cpu_arm.json"
        results_md_filename = "RESULTS.md"

    m1_path = output_dir / m1_filename
    m2_path = output_dir / m2_filename
    results_md_path = Path(args.results_md) if args.results_md else output_dir / results_md_filename

    # Handler for markdown generation only
    if args.generate_table_only:
        print(f"Generating Markdown table from existing files: {m1_filename}, {m2_filename}")
        table_str = generate_markdown_table(m1_path, m2_path)
        results_md_path.write_text(table_str, encoding="utf-8")
        print(f"Markdown table updated at {results_md_path}")
        print("\n" + table_str)
        return

    # Dry-run handler
    if args.dry_run:
        print("Dry run mode: generating simulated benchmark results...")
        simulated_m1 = {
            "baseline": {
                "prefill": {
                    "igpu_throughput_tok_s": {"samples": [40000.0], "mean": 40000.0, "stddev": 0.0, "min": 40000.0, "max": 40000.0},
                    "avg_power_w": {"samples": [15.0], "mean": 15.0, "stddev": 0.0, "min": 15.0, "max": 15.0}
                },
                "decode": {
                    "igpu_throughput_tok_s": {"samples": [46.0], "mean": 46.0, "stddev": 0.0, "min": 46.0, "max": 46.0},
                    "avg_power_w": {"samples": [73.0], "mean": 73.0, "stddev": 0.0, "min": 73.0, "max": 73.0}
                }
            },
            "concurrent_npu": {
                "prefill": {
                    "igpu_throughput_tok_s": {"samples": [46000.0], "mean": 46000.0, "stddev": 0.0, "min": 46000.0, "max": 46000.0},
                    "npu_throughput_tok_s": {"samples": [16.0], "mean": 16.0, "stddev": 0.0, "min": 16.0, "max": 16.0},
                    "avg_power_w": {"samples": [32.0], "mean": 32.0, "stddev": 0.0, "min": 32.0, "max": 32.0},
                    "contention_loss_pct": None
                },
                "decode": {
                    "igpu_throughput_tok_s": {"samples": [44.5], "mean": 44.5, "stddev": 0.0, "min": 44.5, "max": 44.5},
                    "npu_throughput_tok_s": {"samples": [16.0], "mean": 16.0, "stddev": 0.0, "min": 16.0, "max": 16.0},
                    "avg_power_w": {"samples": [83.0], "mean": 83.0, "stddev": 0.0, "min": 83.0, "max": 83.0},
                    "contention_loss_pct": {"samples": [3.1], "mean": 3.1, "stddev": 0.0, "min": 3.1, "max": 3.1}
                }
            },
            "telemetry": {
                "before": {"gpu_busy_pct": 1, "gpu_power_w": 28.0, "npu_active": False, "state": "IDLE", "ts": time.time()},
                "during": {"gpu_busy_pct": 0, "gpu_power_w": 29.0, "npu_active": True, "state": "IDLE", "ts": time.time() + 1},
                "after": {"gpu_busy_pct": 1, "gpu_power_w": 17.0, "npu_active": False, "state": "IDLE", "ts": time.time() + 2}
            },
            "xrt_attribution_evidence": "Simulated AIE Partition activity"
        }
        
        simulated_m2 = {
            "concurrent_cpu": {
                "prefill": {
                    "igpu_throughput_tok_s": {"samples": [45000.0], "mean": 45000.0, "stddev": 0.0, "min": 45000.0, "max": 45000.0},
                    "cpu_throughput_tok_s": {"samples": [4.5], "mean": 4.5, "stddev": 0.0, "min": 4.5, "max": 4.5},
                    "avg_power_w": {"samples": [79.0], "mean": 79.0, "stddev": 0.0, "min": 79.0, "max": 79.0},
                    "contention_loss_pct": None
                },
                "decode": {
                    "igpu_throughput_tok_s": {"samples": [45.5], "mean": 45.5, "stddev": 0.0, "min": 45.5, "max": 45.5},
                    "cpu_throughput_tok_s": {"samples": [4.5], "mean": 4.5, "stddev": 0.0, "min": 4.5, "max": 4.5},
                    "avg_power_w": {"samples": [89.0], "mean": 89.0, "stddev": 0.0, "min": 89.0, "max": 89.0},
                    "contention_loss_pct": {"samples": [0.68], "mean": 0.68, "stddev": 0.0, "min": 0.68, "max": 0.68}
                }
            },
            "comparison": {
                "decode_loss_pct": {
                    "npu": {"samples": [3.1], "mean": 3.1, "stddev": 0.0, "min": 3.1, "max": 3.1},
                    "cpu": {"samples": [0.68], "mean": 0.68, "stddev": 0.0, "min": 0.68, "max": 0.68}
                },
                "prefill_loss_pct": {
                    "npu": None,
                    "cpu": None
                },
                "background_throughput_tok_s": {
                    "npu": {"samples": [16.0], "mean": 16.0, "stddev": 0.0, "min": 16.0, "max": 16.0},
                    "cpu": {"samples": [4.5], "mean": 4.5, "stddev": 0.0, "min": 4.5, "max": 4.5}
                },
                "avg_decode_power_w": {
                    "baseline": {"samples": [73.0], "mean": 73.0, "stddev": 0.0, "min": 73.0, "max": 73.0},
                    "npu": {"samples": [83.0], "mean": 83.0, "stddev": 0.0, "min": 83.0, "max": 83.0},
                    "cpu": {"samples": [89.0], "mean": 89.0, "stddev": 0.0, "min": 89.0, "max": 89.0}
                },
                "marginal_decode_power_w": {
                    "npu": 10.0,
                    "cpu": 16.0
                },
                "perf_watt": {
                    "npu": 0.143,
                    "cpu": 0.049
                }
            }
        }
        
        m1_path.write_text(json.dumps(simulated_m1, indent=2), encoding="utf-8")
        m2_path.write_text(json.dumps(simulated_m2, indent=2), encoding="utf-8")
        table_str = generate_markdown_table(m1_path, m2_path)
        results_md_path.write_text(table_str, encoding="utf-8")
        print(f"Dry run completed. Output files generated under {output_dir}:")
        print(f"  - {m1_filename}")
        print(f"  - {m2_filename}")
        print(f"  - {results_md_path.name}")
        print("\nGenerated Markdown Table:")
        print(table_str)
        return

    # Real hardware execution
    print("==================================================")
    print("REM M1/M2 Contention and Placement Benchmark")
    print("==================================================")

    # 1. Clean up ports and prepare telemetry
    cleanup_ports(ports=(args.npu_port, args.cpu_port))
    settings = Settings()
    settings.bench_dir = str(output_dir)
    gauge = HardwareGauge(settings)

    # Resolve power sysfs path
    power_path = args.power_path
    if not power_path:
        try:
            from rem.platform.discover_sysfs import discover_sysfs
            discovered = discover_sysfs()
            power_path = discovered.get("gpu_power_path")
        except Exception as e:
            print(f"Warning: sysfs discovery failed: {e}", file=sys.stderr)
    if not power_path:
        power_path = "/sys/class/hwmon/hwmon5/power1_input"
    print(f"Using iGPU power path: {power_path}")

    # Generate a ~4,000 token prompt for prefill tests
    prefill_prompt = "The quick brown fox jumps over the lazy dog. " * 350
    trials = args.trials

    # ==================================================
    # PHASE 1: Baseline (iGPU only)
    # ==================================================
    print("\nRunning iGPU baseline measurements...")
    prefill_tok_s_list = []
    prefill_power_list = []
    for i in range(trials):
        sampler = PowerSampler(power_path=power_path)
        sampler.start()
        tput, _ = run_igpu_prefill(args.igpu_port, prefill_prompt)
        pwr = sampler.stop()
        prefill_tok_s_list.append(tput)
        prefill_power_list.append(pwr)
        time.sleep(0.2)

    decode_tok_s_list = []
    decode_power_list = []
    for i in range(trials):
        sampler = PowerSampler(power_path=power_path)
        sampler.start()
        tput = run_igpu_decode(args.igpu_port)
        pwr = sampler.stop()
        decode_tok_s_list.append(tput)
        decode_power_list.append(pwr)
        time.sleep(0.2)

    baseline_prefill_tput_stats = calculate_stats(prefill_tok_s_list)
    baseline_prefill_pwr_stats = calculate_stats(prefill_power_list)
    baseline_decode_tput_stats = calculate_stats(decode_tok_s_list)
    baseline_decode_pwr_stats = calculate_stats(decode_power_list)

    baseline_prefill_tput = baseline_prefill_tput_stats["mean"]
    baseline_prefill_pwr = baseline_prefill_pwr_stats["mean"]
    baseline_decode_tput = baseline_decode_tput_stats["mean"]
    baseline_decode_pwr = baseline_decode_pwr_stats["mean"]

    print(f"  iGPU Baseline Prefill: {baseline_prefill_tput:.2f} tok/s | Avg Power: {baseline_prefill_pwr:.2f} W")
    print(f"  iGPU Baseline Decode : {baseline_decode_tput:.2f} tok/s  | Avg Power: {baseline_decode_pwr:.2f} W")

    # ==================================================
    # PHASE 2: NPU Concurrent (M1)
    # ==================================================
    npu_tput_measured = 0.0
    npu_decode_pwr_measured = 0.0
    npu_decode_loss_measured = 0.0
    npu_prefill_loss_measured = 0.0

    if not args.skip_npu:
        print("\nStarting NPU model server...")
        npu_proc = subprocess.Popen(
            ["flm", "serve", args.npu_model, "--port", str(args.npu_port)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        if not wait_for_server(args.npu_port):
            print("ERROR: NPU server failed to start.")
            npu_proc.terminate()
            sys.exit(1)
        print("NPU server ready.")

        # Capture idle telemetry evidence
        telemetry_before = gauge.read_direct().to_dict()

        # Start background NPU load
        npu_load = BackgroundLoad(args.npu_port, args.npu_model, max_tokens=args.background_max_tokens)
        npu_load.start()
        time.sleep(1.0) # Let NPU load initialize

        # Verify NPU active and collect xrt-smi attribution evidence
        telemetry_during = gauge.read_direct().to_dict()
        xrt_attribution_evidence = ""
        try:
            xrt_attribution_evidence = subprocess.check_output(
                ["xrt-smi", "examine", "--device", "0000:c6:00.1", "--report", "aie-partitions"],
                text=True, stderr=subprocess.DEVNULL
            )
        except Exception:
            pass

        # Run prefill contention
        npu_prefill_tok_s_list = []
        npu_prefill_power_list = []
        for i in range(trials):
            sampler = PowerSampler(power_path=power_path)
            sampler.start()
            tput, _ = run_igpu_prefill(args.igpu_port, prefill_prompt)
            pwr = sampler.stop()
            npu_prefill_tok_s_list.append(tput)
            npu_prefill_power_list.append(pwr)
            time.sleep(0.2)

        # Run decode contention
        npu_decode_tok_s_list = []
        npu_decode_power_list = []
        for i in range(trials):
            sampler = PowerSampler(power_path=power_path)
            sampler.start()
            tput = run_igpu_decode(args.igpu_port)
            pwr = sampler.stop()
            npu_decode_tok_s_list.append(tput)
            npu_decode_power_list.append(pwr)
            time.sleep(0.2)

        npu_tput_measured = npu_load.stop()
        npu_tput_stats = npu_load.get_stats()
        npu_proc.terminate()
        npu_proc.wait()
        time.sleep(1.0) # Let NPU cool down

        # Capture after telemetry
        telemetry_after = gauge.read_direct().to_dict()

        npu_prefill_tput_stats = calculate_stats(npu_prefill_tok_s_list)
        npu_prefill_pwr_stats = calculate_stats(npu_prefill_power_list)
        npu_decode_tput_stats = calculate_stats(npu_decode_tok_s_list)
        npu_decode_pwr_stats = calculate_stats(npu_decode_power_list)

        npu_prefill_tput = npu_prefill_tput_stats["mean"]
        npu_prefill_pwr = npu_prefill_pwr_stats["mean"]
        npu_decode_tput = npu_decode_tput_stats["mean"]
        npu_decode_pwr_measured = npu_decode_pwr_stats["mean"]

        npu_decode_loss_samples = [
            (baseline_decode_tput - x) / baseline_decode_tput * 100.0
            for x in npu_decode_tok_s_list
        ] if baseline_decode_tput else []
        npu_decode_loss_stats = calculate_stats(npu_decode_loss_samples)
        npu_decode_loss_measured = npu_decode_loss_stats["mean"]

        print(f"  iGPU Prefill (w/ NPU): {npu_prefill_tput:.2f} tok/s | Avg Power: {npu_prefill_pwr:.2f} W")
        print(f"  iGPU Decode  (w/ NPU): {npu_decode_tput:.2f} tok/s (Loss: {npu_decode_loss_measured:.2f}%)  | Avg Power: {npu_decode_pwr_measured:.2f} W")
        print(f"  NPU Background Job Throughput: {npu_tput_stats['mean']:.2f} ± {npu_tput_stats['stddev']:.2f} tok/s")

        # Save m1_contention.json
        m1_data = {
            "baseline": {
                "prefill": {
                    "igpu_throughput_tok_s": baseline_prefill_tput_stats,
                    "avg_power_w": baseline_prefill_pwr_stats
                },
                "decode": {
                    "igpu_throughput_tok_s": baseline_decode_tput_stats,
                    "avg_power_w": baseline_decode_pwr_stats
                }
            },
            "concurrent_npu": {
                "prefill": {
                    "igpu_throughput_tok_s": npu_prefill_tput_stats,
                    "npu_throughput_tok_s": npu_tput_stats,
                    "avg_power_w": npu_prefill_pwr_stats,
                    "contention_loss_pct": None  # Prefill contention loss % is below harness resolution
                },
                "decode": {
                    "igpu_throughput_tok_s": npu_decode_tput_stats,
                    "npu_throughput_tok_s": npu_tput_stats,
                    "avg_power_w": npu_decode_pwr_stats,
                    "contention_loss_pct": npu_decode_loss_stats
                }
            },
            "telemetry": {
                "before": telemetry_before,
                "during": telemetry_during,
                "after": telemetry_after
            },
            "xrt_attribution_evidence": xrt_attribution_evidence
        }
        m1_path.write_text(json.dumps(m1_data, indent=2), encoding="utf-8")
    else:
        print("\nSkipped NPU phase.")
        # Load from existing if available for comparisons
        if m1_path.exists():
            try:
                old_m1 = json.loads(m1_path.read_text(encoding="utf-8"))
                
                def get_mean_val(node):
                    if isinstance(node, dict):
                        return node.get("mean", 0.0)
                    return float(node or 0.0)

                old_npu_tput = old_m1.get("concurrent_npu", {}).get("decode", {}).get("npu_throughput_tok_s", 0.0)
                if isinstance(old_npu_tput, dict):
                    npu_tput_stats = old_npu_tput
                    npu_tput_measured = npu_tput_stats.get("mean", 0.0)
                else:
                    npu_tput_measured = float(old_npu_tput)
                    npu_tput_stats = {"samples": [npu_tput_measured], "mean": npu_tput_measured, "stddev": 0.0, "min": npu_tput_measured, "max": npu_tput_measured}
                
                npu_decode_pwr_measured = get_mean_val(old_m1.get("concurrent_npu", {}).get("decode", {}).get("avg_power_w", 0.0))
                npu_decode_loss_measured = get_mean_val(old_m1.get("concurrent_npu", {}).get("decode", {}).get("contention_loss_pct", 0.0))
                # Create a mock stats block for loaded values
                npu_decode_loss_stats = old_m1.get("concurrent_npu", {}).get("decode", {}).get("contention_loss_pct", {})
                if not isinstance(npu_decode_loss_stats, dict):
                    npu_decode_loss_stats = {"samples": [npu_decode_loss_measured], "mean": npu_decode_loss_measured, "stddev": 0.0, "min": npu_decode_loss_measured, "max": npu_decode_loss_measured}
                npu_decode_pwr_stats = old_m1.get("concurrent_npu", {}).get("decode", {}).get("avg_power_w", {})
                if not isinstance(npu_decode_pwr_stats, dict):
                    npu_decode_pwr_stats = {"samples": [npu_decode_pwr_measured], "mean": npu_decode_pwr_measured, "stddev": 0.0, "min": npu_decode_pwr_measured, "max": npu_decode_pwr_measured}
            except Exception:
                pass

    # ==================================================
    # PHASE 3: CPU Concurrent (M2)
    # ==================================================
    if not args.skip_cpu:
        print("\nStarting CPU model server...")
        cpu_proc = subprocess.Popen([
            args.llama_server_path,
            "--host", "127.0.0.1",
            "--port", str(args.cpu_port),
            "--model", args.cpu_model_path,
            "--gpu-layers", "0",
            "-t", "4",
            "--no-mmap"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if not wait_for_server(args.cpu_port):
            print("ERROR: CPU server failed to start.")
            cpu_proc.terminate()
            sys.exit(1)
        print("CPU server ready.")

        # Start background CPU load
        cpu_load = BackgroundLoad(args.cpu_port, "cpu_background_load", max_tokens=args.background_max_tokens)
        cpu_load.start()
        time.sleep(1.0) # Let CPU load initialize

        # Run prefill contention
        cpu_prefill_tok_s_list = []
        cpu_prefill_power_list = []
        for i in range(trials):
            sampler = PowerSampler(power_path=power_path)
            sampler.start()
            tput, _ = run_igpu_prefill(args.igpu_port, prefill_prompt)
            pwr = sampler.stop()
            cpu_prefill_tok_s_list.append(tput)
            cpu_prefill_power_list.append(pwr)
            time.sleep(0.2)

        # Run decode contention
        cpu_decode_tok_s_list = []
        cpu_decode_power_list = []
        for i in range(trials):
            sampler = PowerSampler(power_path=power_path)
            sampler.start()
            tput = run_igpu_decode(args.igpu_port)
            pwr = sampler.stop()
            cpu_decode_tok_s_list.append(tput)
            cpu_decode_power_list.append(pwr)
            time.sleep(0.2)

        cpu_tput = cpu_load.stop()
        cpu_tput_stats = cpu_load.get_stats()
        cpu_proc.terminate()
        cpu_proc.wait()

        cpu_prefill_tput_stats = calculate_stats(cpu_prefill_tok_s_list)
        cpu_prefill_pwr_stats = calculate_stats(cpu_prefill_power_list)
        cpu_decode_tput_stats = calculate_stats(cpu_decode_tok_s_list)
        cpu_decode_pwr_stats = calculate_stats(cpu_decode_power_list)

        cpu_prefill_tput = cpu_prefill_tput_stats["mean"]
        cpu_prefill_pwr = cpu_prefill_pwr_stats["mean"]
        cpu_decode_tput = cpu_decode_tput_stats["mean"]
        cpu_decode_pwr = cpu_decode_pwr_stats["mean"]

        cpu_decode_loss_samples = [
            (baseline_decode_tput - x) / baseline_decode_tput * 100.0
            for x in cpu_decode_tok_s_list
        ] if baseline_decode_tput else []
        cpu_decode_loss_stats = calculate_stats(cpu_decode_loss_samples)
        cpu_decode_loss = cpu_decode_loss_stats["mean"]

        print(f"  iGPU Prefill (w/ CPU): {cpu_prefill_tput:.2f} tok/s | Avg Power: {cpu_prefill_pwr:.2f} W")
        print(f"  iGPU Decode  (w/ CPU): {cpu_decode_tput:.2f} tok/s (Loss: {cpu_decode_loss:.2f}%)  | Avg Power: {cpu_decode_pwr:.2f} W")
        print(f"  CPU Background Job Throughput: {cpu_tput_stats['mean']:.2f} ± {cpu_tput_stats['stddev']:.2f} tok/s")

        # Performance / Watt calculations
        npu_perf_watt = npu_tput_measured / npu_decode_pwr_measured if npu_decode_pwr_measured else 0.0
        cpu_perf_watt = cpu_tput / cpu_decode_pwr if cpu_decode_pwr else 0.0

        # Marginal added watts
        npu_marginal_power = npu_decode_pwr_measured - baseline_decode_pwr if baseline_decode_pwr else 0.0
        cpu_marginal_power = cpu_decode_pwr - baseline_decode_pwr if baseline_decode_pwr else 0.0

        # Save m2_cpu_arm.json
        m2_data = {
            "concurrent_cpu": {
                "prefill": {
                    "igpu_throughput_tok_s": cpu_prefill_tput_stats,
                    "cpu_throughput_tok_s": cpu_tput_stats,
                    "avg_power_w": cpu_prefill_pwr_stats,
                    "contention_loss_pct": None  # Prefill contention loss % is below harness resolution
                },
                "decode": {
                    "igpu_throughput_tok_s": cpu_decode_tput_stats,
                    "cpu_throughput_tok_s": cpu_tput_stats,
                    "avg_power_w": cpu_decode_pwr_stats,
                    "contention_loss_pct": cpu_decode_loss_stats
                }
            },
            "comparison": {
                "decode_loss_pct": {
                    "npu": npu_decode_loss_stats if 'npu_decode_loss_stats' in locals() else {"samples": [npu_decode_loss_measured], "mean": npu_decode_loss_measured, "stddev": 0.0, "min": npu_decode_loss_measured, "max": npu_decode_loss_measured},
                    "cpu": cpu_decode_loss_stats
                },
                "prefill_loss_pct": {
                    "npu": None,
                    "cpu": None
                },
                "background_throughput_tok_s": {
                    "npu": npu_tput_stats if 'npu_tput_stats' in locals() else {"samples": [npu_tput_measured], "mean": npu_tput_measured, "stddev": 0.0, "min": npu_tput_measured, "max": npu_tput_measured},
                    "cpu": cpu_tput_stats
                },
                "avg_decode_power_w": {
                    "baseline": baseline_decode_pwr_stats,
                    "npu": npu_decode_pwr_stats if 'npu_decode_pwr_stats' in locals() else {"samples": [npu_decode_pwr_measured], "mean": npu_decode_pwr_measured, "stddev": 0.0, "min": npu_decode_pwr_measured, "max": npu_decode_pwr_measured},
                    "cpu": cpu_decode_pwr_stats
                },
                "marginal_decode_power_w": {
                    "npu": round(npu_marginal_power, 3),
                    "cpu": round(cpu_marginal_power, 3)
                },
                "perf_watt": {
                    "npu": round(npu_perf_watt, 3),
                    "cpu": round(cpu_perf_watt, 3)
                }
            }
        }
        m2_path.write_text(json.dumps(m2_data, indent=2), encoding="utf-8")
    else:
        print("\nSkipped CPU phase.")

    # 4. Generate Markdown Results table at the end
    print("\nUpdating Markdown results table...")
    table_str = generate_markdown_table(m1_path, m2_path)
    results_md_path.write_text(table_str, encoding="utf-8")
    print(f"Results table updated at {results_md_path}")
    print("\n" + table_str)


if __name__ == "__main__":
    main()
