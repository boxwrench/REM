"""Centralized configuration for REM using pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings for the REM system.

    All settings can be overridden by environment variables prefixed with REM_.
    """
    model_config = SettingsConfigDict(
        env_prefix="REM_",
        case_sensitive=False,
    )

    # Ports
    litellm_port: int = 4000
    npu_server_port: int = 13306
    scheduler_port: int = 4100
    sidecar_port: int = 4200

    # Paths
    vault_dir: str = "projects/rem/vault"
    bench_dir: str = "projects/rem/bench"

    # Model names (filled in later by discovery/gates)
    summarizer_model: str = "llama3.2:1b"
    embedding_model: str = ""

    # NPU Client & Generation parameters
    npu_connect_timeout_s: float = 5.0
    npu_request_timeout_s: float = 60.0
    npu_max_tokens: int = 800

    # Compaction parameters (Path A)
    compact_trigger_tokens: int = 8000
    compact_span_turns: int = 6
    keep_recent_turns: int = 8
    max_context_tokens: int = 32000
    # Read path (bounded assembly). Target for the fitted read budget; kept under
    # the answering model's ~32-40k window so it never returns HTTP 400. Distinct
    # from max_context_tokens, which is the assemble safety ceiling.
    read_fit_tokens: int = 28000
    deterministic_fact_capture: bool = True
    # Embedding-based slot identity for supersession (Gate 4 follow-up). When on and
    # an embedder is wired into the ledger, supersession treats two entries whose
    # full-fact embeddings ("natural key: value") reach the threshold as the same
    # slot, collapsing semantically-equivalent keys that exact string matching leaves
    # fragmented. Off by default (needs an embedder + chosen threshold).
    embedding_supersession: bool = False
    embedding_supersession_threshold: float = 0.80

    # Wiki parameters (Path B)
    compile_max_pages: int = 15
    budget_tokens: int = 2000

    # Scheduler parameters (Path C)
    gpu_idle_busy_pct: int = 10  # UNCALIBRATED (E2 will calibrate)
    gpu_prefill_power_w: float = 35.0  # UNCALIBRATED (E2 will calibrate)
    gauge_hysteresis_samples: int = 3
