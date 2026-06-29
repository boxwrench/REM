# Normalized memory-method evaluations

This directory contains the frozen 30-item development manifest, once-captured
REM states, normalized run artifacts, and three-item comparator smokes. The
LongMemEval-S source dataset is intentionally external and is identified in every
manifest by SHA-256.

Required sequence:

1. Run `freeze_manifest.py` against the local dataset. Do not hand-pick IDs.
2. Adjudicate any additional ambiguous items and repeat the freeze with
   `--exclude QUESTION_ID`; commit the resulting manifest before evaluating.
3. Run `capture_states.py` once. Existing state files are skipped.
4. Run `run_development.py` at the fixed 8k and 28k budgets.
5. Run each external provider's three-item smoke with an explicit source/image
   revision. Unsupported configuration is a recorded result; do not replace it
   with a hosted service or different model.

Gate 0 activated the embedding-identity audit. Run the Qwen baseline against
`diagnostic_fixtures.json` first; run DREAM with the same command and fixtures
only after the baseline artifact exists. The model argument is required so an
unsupported or substituted model cannot be mistaken for the preregistered arm.

Hindsight uses the pinned revision's recommended local server and its native
Python `retain`/`recall` client contract. Each question is a separate bank and
the caller passes `max_tokens`.

Supermemory uses the pinned self-hosted server, one container tag per question,
the native `/v3/documents` ingestion lifecycle, and memory-only search. Local
model configuration must be recorded in the artifact because hosted extraction
quality is not evidence about the self-hosted pass.

Example smokes:

```bash
PYTHONPATH=.:src python3 evals/memory_methods/embedding_identity.py \
  --model PINNED_QWEN_EMBEDDING_MODEL \
  --out bench/memory_methods/qwen-embedding-identity.json

PYTHONPATH=.:src python3 evals/memory_methods/run_comparator_smoke.py \
  --provider hindsight --revision COMMIT_OR_IMAGE_DIGEST \
  --base-url http://127.0.0.1:8888 --data /path/to/longmemeval_s.json \
  --out bench/memory_methods/hindsight-smoke.json

PYTHONPATH=.:src python3 evals/memory_methods/run_comparator_smoke.py \
  --provider supermemory --revision COMMIT_OR_IMAGE_DIGEST \
  --base-url http://127.0.0.1:3000 --data /path/to/longmemeval_s.json \
  --out bench/memory_methods/supermemory-smoke.json
```
