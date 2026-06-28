# Battery findings: tight-budget recall gate

Status as of 2026-06-21. Artifacts in this directory; numbers below come straight
from them.

## The gate

Does REM's compaction preserve a user-relevant fact better than naive truncation,
at a context budget tight enough that truncation drops the gold evidence? A run
only counts if truncation actually drops the gold (validity guard in
`evals/battery/aggregate.py`).

## Runs

| Artifact | items | budget | valid | truncation kept gold | REM acc | dominant miss | extraction |
|---|---|---|---|---|---|---|---|
| `tight_smoke_b3000_limit1.json` | 1 | 3000 | no | 100% | 0/1 | — | 181 att / 0 fail |
| `tight_sweep_b2000_limit3.json` | 3 | 2000 | no | 67% | 0/3 | budget_invalid | 957 att / 2 fail |
| `tight_sweep_b1000_limit3.json` | 3 | 1000 | no | 67% | 0/3 | budget_invalid | 1311 att / 4 fail |
| `valid_b1000_oldgold.json` | 5 | 1000 | **yes** | 0% | 0/5 | context_overflow | 2220 att / 1 fail |

## What the budget sweep showed

The first three runs are invalid: truncation kept the gold, so the comparison is
trivial. Tightening the budget from 2000 to 1000 did not move truncation
retention (67% both times). Cause: for `knowledge-update`, the gold is the latest
update, which sits in recent sessions — exactly what truncation preserves. Across
all 78 knowledge-update items the latest gold session has median normalized
timeline position 0.86, and 53 of 78 have it in the newest third. The budget knob
cannot manufacture a valid comparison on recent-gold items.

## The selection fix

`--max-gold-recency 0.33` selects the 5 items whose latest gold is in the oldest
third (`gold_recency` 0.16–0.28). Truncation drops the gold on all 5 (0%
retention), so `valid_b1000_oldgold.json` is the first valid run.

## What the valid run did and did not prove

- It is valid: truncation retains 0% and scores 0/5, as intended.
- It does **not** yet give a memory-quality verdict. REM also scored 0/5, and the
  classifier attributes all 5 misses to `context_overflow`: REM raised
  `ContextLimitExceeded` during assembly and never produced an answer. These items
  are large (431–494 turns); REM's compacted memory (summaries + facts ledger +
  recent window) exceeds the assembly ceiling `max_context_tokens = budget×4 =
  4000`.
- Extraction held up: 2220 attempts, 1 hard failure. The JSON-robustness and
  malformed-entry recovery work is doing its job. One new mangling shape still
  slips through (`"source_turn_id":443,subject"...`).

The classifier's recommendation stands: fix the harness before diagnosing memory.

## The overflow: root cause, and why the 16k "repair" does not fix it

Root cause: the assembler renders the facts ledger in full and every episodic
summary with no size bound, so REM's compacted memory grows with conversation
length. The verbatim tier is bounded but summaries + ledger are not. This is a
scaling gap, not a small config ceiling.

The actual magnitude (from the artifact's own overflow messages, corroborated by
the canary and `diag_031748ae_w64k.json`): on the five oldest-gold items REM's
assembled memory is **36,977 – 58,150 tokens**:

| item | assembled tokens |
|---|---|
| 9bbe84a2 | 36,977 |
| 031748ae | 40,565 |
| 3ba21379 | 50,529 |
| cc5ded98 | 54,790 |
| c6853660 | 58,150 |

An earlier note here estimated ~6.6k from a synthetic reproduction; that
under-counted by 6×. Raising the assemble ceiling from `budget×4 = 4000` to
`REM_MEMORY_WINDOW_TOKENS = 16000` (spec §3 reserved region) **does not unblock
assembly** — all five still exceed 16k, so REM still raises
`ContextLimitExceeded` and answers nothing. The 16k change only moves the
ceiling; it does not bound the memory.

Worse, the larger items (50k–58k) exceed the **answering model's own context
window** (~32–40k): `gemma4-it:e2b` returns HTTP 400 `"Max length reached!"`
above that. So even with an unbounded assemble ceiling, the model cannot read
REM's compacted memory on long items. Compaction here re-encodes the history at
similar scale rather than compressing it to a consumable size.

## What this means for the gate

REM cannot answer these items until it has a **bounded read path** — retrieval
or eviction that fits the assembled memory to the model window (~28k after
question/answer headroom), selecting which summaries/facts to include. That is a
design change, not a ceiling tweak, and is the next architecture gate (roadmap
item 7). A token-matched REM-vs-truncation comparison is only meaningful once
that exists.

## Diagnostic result (item 031748ae)

`evals/battery/diagnose_memory.py` ran the real compaction with a 64k assemble
window so the memory rendered, then inspected it. Artifacts:
`diag_031748ae_w64k.json` (summary) and `diag_031748ae_w64k_state.json` (the full
compacted `MemoryState`, 817K — re-analyzable NPU-free). Ingest took 4,542s
(~75 min) for this one item at budget 1000.

Tier breakdown of the 40,626-token assembled memory:

| tier | tokens | share | size |
|---|---|---|---|
| facts ledger | 20,095 | 49% | 950 entries (940 active) |
| episodic summaries | 17,940 | 44% | 460 summaries (= 460 compactions) |
| verbatim transcript | 2,555 | 6% | 8 turns |

The ledger and summaries are 93% of memory and both grow with conversation
length; verbatim is correctly bounded. Extraction was clean: 460 attempts, 0
failures, 429 strict-parse, 30 repaired, 11 truncations.

**Write recall worked.** Both target values survived compaction:
- "4 engineers" — ledger turn12 `team members.count='4 engineers plus manager
  Rachel'`; summaries turn11/12 ("consist of 4 engineers and Rachel").
- "5 engineers" — ledger turns 5/67/74 (`team.size='5 engineers'`); summaries
  turn5/67.

**The miss is read-path, two ways:**
1. Size — 40,626 tokens exceeds the 16k ceiling and the answering model's own
   ~32–40k window. The full-memory answer attempt returned HTTP 400 "Max length
   reached!".
2. Structure — fitted to a 28k head-slice (which *contained* "4 engineers" via
   the summaries), REM still answered only "five engineers" and was judged wrong:
   the question asks for the starting count *and* now, but the flat ledger +
   prose summaries scatter 4 and 5 across unrelated contexts with no temporal
   "started → now" link.

Caveat (why n=1 is not decisive): this item's source is itself ambiguous. In the
conversation "4 engineers" is the team-*outing* headcount ("4 engineers + manager
Rachel"), which LongMemEval's gold treats as the "when you started" count. REM
recorded the conversation faithfully; the "started with 4" framing is an
inference the dataset expects, not a stated fact. A bitemporal graph helps with
genuine then/now updates but would not by itself resolve this item's ambiguity.

## Verdict and next

Status: documented and paused (no five-item rerun). The gate has produced a clear
read-path finding but not a memory-quality verdict, and a single dissected item is
too messy to choose an architecture on. Outstanding decision (deferred to a
focused session): either add a bounded read path (retrieval/eviction that fits
memory to the model window) and rerun the five-item battery to get the failure
*mix* before choosing — vs. starting the graph-resident read path on the strength
of the read-path size finding alone. The 950-entry ledger bloat needs addressing
either way.

## Step 0: bounded read path

Built the fit-to-budget read path (spec `2026-06-26-bounded-read-path-design.md`)
and ran it NPU-free against the persisted 031748ae state via `--load-state` (one
brief answer inference; no 75-min ingest). Artifacts: `step0_031748ae.json`,
`step0_031748ae_state.json`.

`RecencySelector` cuts the assembled memory from **40,626 tokens** (unfitted — the
size that returned HTTP 400) toward the 28,000 budget while keeping both gold
needles. The first run landed at **28,121 (+121, 0.43% over)** and was recorded as
a FAIL: the selector estimates per-item costs without rendering (spec D1), so
section scaffolding tipped the assembled text just past the budget the selector had
already cleared internally.

The fix enforces the budget where the assembled cost is actually measured, leaving
the selection strategy alone. `fit_with_selector` now renders the fitted state, and
when it overshoots, trims the lowest-priority kept tiers — no-slot ledger entries
oldest-first, then summaries — re-rendering until it fits. The protected floor
(newest active entry per slot_key + verbatim) is never trimmed, so the
current-state gold is preserved. `selector.py` is unchanged: the recency strategy
keeps its D1/D2 contract, and only the budget enforcement moved into the
render-aware consumer (commits `0e15ccd`, re-run result in `step0_031748ae.json`).
The re-run fits at **27,996**.

Result against the §5 PASS bar (re-run):

| criterion | target | observed | verdict |
|---|---|---|---|
| fitted tokens | ≤ 28,000 | 27,996 | PASS |
| answer returned | non-null, no HTTP 400 | returned (see caveat) | PASS |
| gold survives fit | both needles present | `4 engineers`: true, `5 engineers`: true | PASS |

**Verdict: PASS** against the §5 bar. The fit turns the 40,626-token overflow into
a budget-fitting read that preserves the gold and draws a response with no
`ContextLimitExceeded` / HTTP 400.

Honest caveat the bar does not capture. The §5 answer criterion asks only that the
model respond without a context error, and judged correctness is deferred (spec §2,
n=1 ambiguous). The returned answer is now **"The memory does not contain
information on how many engineers you lead…"** — a refusal. The pre-fix
28,121-token slice, 125 tokens larger, had drawn a substantive (judged-wrong)
"4 engineers plus manager Rachel." Trimming to fit kept the gold *substrings* in the
slice yet dropped context the model had used to produce a substantive reply. So a
needle present in the slice does not guarantee the model can use it, and pushing
under budget can cost answer substance. That is a signal for the failure mix, and
it leaves the §5 verdict (mechanism fits + preserves gold + answers without error)
intact.

Open, as before: the architecture choice still waits on the five-item failure *mix*
(per-item states + the one ~6h ingest, post-Step-0 plan). The 950-entry ledger
bloat is unaddressed.

## Failure mix (five oldest-gold items)

The post-Step-0 increment captured all five compacted states once (~4.7h NPU,
`bench/battery/states/`, manifest + `mix_report.json`), then labelled each item's
read-path miss NPU-free (RecencySelector fit + gold/structure needles) with one
brief answer per item. Spec: `docs/superpowers/specs/2026-06-27-failure-mix-design.md`.

The headline: **the bounded read path fits all five** (fitted 27,891–27,999, every
one ≤ 28,000). Size is not a failure mode for any item — the trim-enforced read
path holds across the set, not only on 031748ae. Per item:

| item | recency | fitted | gold in slice | brief answer | outcome |
|---|---|---|---|---|---|
| 031748ae | 0.16 | 27,996 | 4 + 5 engineers (both, slots) | refusal | **temporal-structure** |
| 3ba21379 | 0.18 | 27,999 | F-150 (slot); Mustang distractor also present | "Ford F-150 pickup truck" | pass |
| cc5ded98 | 0.26 | 27,891 | two hours (slot); prior "an hour" dropped | "Two hours each day" | pass |
| c6853660 | 0.28 | 27,912 | one cup + two cups (both, slots) | "...two cups" | pass |
| 9bbe84a2 | 0.28 | 27,979 | prior goal 100 retained (slot `level goal.target level: 100`) | "previous goal... was 100" | pass* |

Corrected mix: **4 pass, 1 temporal-structure, 0 retrieval-recall, 0 size.**

\* `mix_report` auto-labelled 9bbe84a2 retrieval-recall, a measurement artifact:
the gold value 100 is retained in the fitted slice as the slot
`level goal.target level: 100` (turn 61), distinct from the current `goal.level: 150`
(turn 144), and the model answered correctly. The exact-substring needle "level 100"
matched neither the slot rendering nor the answer phrasing ("level was 100"). The
true outcome is a pass; the automated label is recorded as-is in `mix_report.json`
with this correction noted. Lesson: substring needles are brittle for bare-number
gold; a slot-value-aware match is the follow-up.

The one genuine miss, 031748ae, is temporal-structure: both "4 engineers" and
"5 engineers" survive in distinct slots and fit the budget, yet the model refuses
("the memory does not contain…"). The flat ledger holds the values with no
started→now link, and this item's gold rests on a dataset inference (the "4" is the
team-outing headcount; see the diagnostic caveat above), so it is the weakest of
the five as evidence.

### What the mix says about the architecture (read-path spec §8)

The graph-resident store's defining feature is temporal edges (valid_from/valid_to),
which directly target the started→now linking that 031748ae fails. But the
architecture spec (`docs/REM-memory-architecture-spec.md`, decision rows 36 and
42–45) is explicit: validate read recall before assuming the graph helps, and if
extraction misses dominate, graph work must begin with write-recall, not a rebuild.

This mix argues against committing to the graph now:

- Read recall is already good — 4/5 produce a correct answer through recency + the
  trim fix, with the gold fitting the budget every time. The read path is not the
  bottleneck for four of five items.
- The single failure is a then→now linking miss on the most ambiguous item (n=1,
  dataset-inferred gold). That is thin evidence to justify replacing the memory path.
- The capture surfaced write-side noise instead: heavy "malformed fact entry"
  skipping during ingest, the known `"source_turn_id":443,subject"` mangling shape
  (9bbe84a2, fell back to verbatim), and slot-key fragmentation — the Apex goal
  alone split across `level goal.target level`, `goal.level`, and
  `level.target level for goal setting`. That fragmentation is why both then/now
  goals survived for 9bbe84a2, but it is a write-recall quality problem.

Recommendation: keep the tuned read path; do not start the graph rebuild on this
mix. The cheapest informative next step is a larger, token-matched, unambiguous item
set to confirm whether temporal-structure failures recur on clean items before
paying for the graph — and, in parallel, instrument write recall / extraction
quality, which this capture flags as the more pressing lever. Fix the bare-number
needle methodology before the next mix run so the automated labels match outcomes.
