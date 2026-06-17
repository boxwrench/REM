# Contributing to REM

REM is **eval-led and artifact-led**: a promising idea isn't progress until it's a
bounded change, run against an explicit check, and recorded with the result. If you
contribute, please keep that discipline — it's what makes the numbers trustworthy.

## Working method

1. State the hypothesis.
2. Make the smallest change that tests it.
3. Run the relevant local tests or hardware eval.
4. Record the exact command and outcome.
5. Decide the next step from evidence, not from vibes.

## Evidence standard

A claim should ship with the artifact that proves it:

- **Unit behavior** — the `pytest` command and its output.
- **NPU execution** — command, model, port, runtime, verdict, and a telemetry note.
- **Performance** — the raw benchmark artifact (`bench/*.json` / `.csv`) plus a summary table.
- **Failure** — record it as a *result*, not as an absence of one.
- **Regression fix** — include a test that is **red without the fix and green with it** (show both). A fix with no guarding test invites the same break to return silently.
- **Reproducibility** — produce eval/bench results from a **committed** state, not a dirty tree: commit the code under test, run, then commit the artifact, so every result maps to a known commit.

**Numbers carry dispersion** (mean ± stddev), and signals below measurement resolution are labelled as such rather than published with false precision. **Don't silently rewrite old conclusions** — if new evidence changes a result, add a new entry and point to the one it supersedes. (This repo's own contention number was corrected this way: a small N=5 run didn't reproduce, and the canonical figure is now the pooled 3×N=20 result.)

## Real hardware only

NPU claims must come from the physical NPU, not a mock. Mocks are for unit tests of
logic, never as evidence for a hardware claim. The default test run excludes
hardware tests (`-m 'not npu'`).

## Running the tests

```bash
pip install -e ".[dev]"
pytest            # unit suite (NPU tests excluded by default)
```

## Where help is most useful

See the **"Where we want help"** section of the [README](README.md) — small-model
JSON robustness in fact extraction, and embedding-based fact identity are the open
problems we'd most welcome contributions on.
