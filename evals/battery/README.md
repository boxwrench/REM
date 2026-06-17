# REM Battery Spike

Thin-slice comparative battery: REM-compaction vs naive truncation on the
LongMemEval `knowledge-update` subset, answerer fixed at gemma-on-NPU, graded by
an independent Claude judge. Spec: `docs/superpowers/specs/2026-06-16-rem-standardized-battery-spike-design.md`.

## Prerequisites
- FLM serving `gemma4-it:e2b` on the NPU at `:13306` (the project default).
- `ANTHROPIC_API_KEY` in the environment (the Claude judge). Put it in `~/.bashrc`
  (`export ANTHROPIC_API_KEY=...`) or `! export ...` for this session. Never commit it.
- The `anthropic` Python package installed (`pip install anthropic`).
- A LongMemEval JSON downloaded locally, e.g. `longmemeval_s.json` from the HF
  dataset `xiaowu0162/longmemeval`.

## Smoke first (mandatory)
```
PYTHONPATH=src python3 evals/battery/run_battery_spike.py \
  --data /path/to/longmemeval_s.json --limit 3 --budget 8000 \
  --out bench/battery/smoke.json
```
Confirm: `valid: true` (truncation drops the gold session), the answerer used gemma,
and judge verdicts are parseable. THEN run the full subset (drop `--limit`).

## Validity
If `valid: false` with a "budget too generous" reason, lower `--budget` until
truncation drops the gold evidence session — otherwise the comparison is trivial.
