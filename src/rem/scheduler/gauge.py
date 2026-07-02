"""Shim for importing gauge from xdna_top.gauge to avoid breaking existing imports."""

try:
    from xdna_top.gauge import (  # noqa: F401  # re-export shim: names are the public API
        GpuState,
        GaugeReading,
        classify_state as _classify_state,
        get_stable_state,
        load_sysfs_paths as _load_sysfs_paths,
        read_igpu,
        run_xrt_smi,
        parse_xrt_smi,
        HardwareGauge as _HardwareGauge,
        run_daemon as _run_daemon,
    )
except ImportError as exc:  # pragma: no cover - exercised only without the optional dep
    raise ImportError(
        "rem.scheduler.gauge requires the 'xdna-top' NPU/iGPU monitor, which is not "
        "installed. Install it with `pip install -e \".[scheduler]\"` or directly from "
        "https://github.com/boxwrench/xdna-top. The core REM memory/compaction library "
        "does not need it."
    ) from exc
from rem.config import Settings

# Wrap classes/functions to inject Settings where needed to preserve existing signatures in REM

def classify_state(gpu_busy_pct: int, gpu_power_w: float, settings: Settings) -> GpuState:
    return _classify_state(
        gpu_busy_pct,
        gpu_power_w,
        gpu_idle_busy_pct=settings.gpu_idle_busy_pct,
        gpu_prefill_power_w=settings.gpu_prefill_power_w,
    )


def load_sysfs_paths(settings: Settings) -> tuple[str | None, str | None]:
    return _load_sysfs_paths(settings.bench_dir)


class HardwareGauge(_HardwareGauge):
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        super().__init__(
            gpu_idle_busy_pct=self.settings.gpu_idle_busy_pct,
            gpu_prefill_power_w=self.settings.gpu_prefill_power_w,
            gauge_hysteresis_samples=self.settings.gauge_hysteresis_samples,
            bench_dir=self.settings.bench_dir,
            pessimistic_fallback=True,
        )


def run_daemon(settings: Settings) -> None:
    _run_daemon(
        gpu_idle_busy_pct=settings.gpu_idle_busy_pct,
        gpu_prefill_power_w=settings.gpu_prefill_power_w,
        gauge_hysteresis_samples=settings.gauge_hysteresis_samples,
        bench_dir=settings.bench_dir,
        pessimistic_fallback=True,
    )


def main() -> None:
    import argparse
    import logging
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    ap = argparse.ArgumentParser(description="REM Telemetry Gauge Daemon (C1)")
    ap.parse_args()

    settings = Settings()
    try:
        run_daemon(settings)
    except KeyboardInterrupt:
        pass
    except Exception:
        sys.exit(1)
