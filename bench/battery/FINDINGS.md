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

## Write-recall audit (the lever the mix pointed at)

`evals/battery/write_recall_audit.py` (NPU-free; `bench/battery/write_recall_audit.json`)
separates write recall from read recall over the five states and measures two
write-side signals. Result: **write recall is good; write quality is the defect.**

Write recall holds. Every gold value is extracted into the compacted state and sits
in a ledger slot — `4/5 engineers`, `F-150`, `two hours`, `one cup`/`two cups` all
resolve to `slot` in the full (unfitted) state, and the read path keeps them. The
one apparent absence, 9bbe84a2 "level 100", is the bare-number needle artifact: the
prior goal is written as the slot `level goal.target level: 100`, just not as the
literal string. No gold fact was dropped on the write side.

The defect is supersession. Slot supersession barely fires — **10–11 of ~850–940
entries per item, ~1.1–1.3%** — because the extractor assigns a fresh slot key for
nearly every mention. The same value lands under many distinct keys (32–55
fragmented values per item, one value under up to 7 keys), so genuine updates never
collapse and every mention persists as an "active current" fact:

| concept (item) | value | distinct active slot keys |
|---|---|---|
| team size (031748ae) | "5 engineers" | `group size.number of engineers`, `team size.size`, `team.size` (+ "4 engineers" under `team members.count`) |
| coding time (cc5ded98) | "two hours" | `coding exercises.frequency` / `.time per day` / `.time spent per day` / `.duration` |
| coffee limit (c6853660) | one→two cups | `morning routine.quantity…` (one cup) vs `morning coffee limit.new limit` (two cups) — never collapse |
| Apex goal (9bbe84a2) | 100→150 | `level goal.target level` (100) vs `goal.level`, `level.target level for goal setting`, `user.goal` (150) |

This single defect explains both open problems. It is the 950-entry ledger bloat
(updates do not collapse, so the ledger grows with every mention), and it is the
lone temporal-structure miss: 031748ae's "4 engineers" and "5 engineers" are both
active under different keys with no supersession and no order, so the model cannot
resolve started→now. The flat ledger is not the obstacle here — the fragmented
keying is, and a graph would inherit the same bad keys from the same extractor.

Extraction also drops entries: the ingest skipped 6–22 malformed fact entries per
item (missing-`text` shapes), plus one hard extraction failure on 9bbe84a2 (the
`"source_turn_id":443,subject"` mangling) that fell back to keeping the span
verbatim. These do not touch the gold here, but they are unmeasured fact loss.

Lever: **slot-key canonicalization** — normalize keys (or add a supersession step
that matches semantically-equivalent keys) so updates collapse (4→5, one→two,
100→150). That shrinks the ledger and hands the read path an ordered current-state
set, which is the most likely single fix for the 031748ae temporal-structure miss —
without a graph rebuild. This is the concrete write-recall work the mix called for.

## Slot-key canonicalization

The post-hoc string-first experiment is complete over all five captured states
(`canonicalize_audit.json`). It retained every existing gold needle and did not
mutate the captures.

| granularity | active entries | fragmented values | superseded entries | merge-risk groups |
|---|---:|---:|---:|---:|
| full key | 4,469 → 4,426 (-0.96%) | 212 → 205 (-3.30%) | 53 → 96 | 36 |
| subject only | 4,469 → 3,349 (-25.06%) | 212 → 130 (-38.68%) | 53 → 1,173 | 592 |

The conservative key barely changes the observed fragmentation. Subject-only
grouping is more aggressive but still misses the required 50% reduction and
groups hundreds of distinct current values under one subject signature. Synthetic
same-slot/different-slot fixtures pass, but the real-state merge-risk count makes
subject-only identity unsafe for the writer.

Decision: do **not** promote either string strategy into `_apply_supersession`.
The residual fragmentation is semantic, which activates the embedding-identity
experiment: establish a Qwen embedding baseline before testing DREAM. The planned
`031748ae` paid answer rerun was skipped because both quantitative candidates had
already failed the promotion threshold, and that item remains diagnostic rather
than gating due to its ambiguous source inference.

## Gate 2 — temporal LexicalSelector vs RecencySelector (the skipped paid run)

The canonicalization detour skipped the one paid answer that closes Gate 2: does
the query-aware `LexicalSelector` (temporal history + ranked fill) resolve
`031748ae`'s then→now miss that `RecencySelector` produced? Run with the
selector-parameterized mix runner `evals/battery/mix_report_selector.py` over the
five captured states. Artifacts: `mix_report_selectors_npufree.json` (recency /
lexical / lexical-packed, NPU-free, deterministic), `mix_report_lexical.json`
(lexical, one brief NPU answer per item), and
`mix_report_recency_031748ae.json` (same-code recency baseline for the headline
item).

**NPU-free structure (the load-bearing, deterministic evidence).** All five items
fit the 28k budget under all three selectors (fitted 27,647–27,890). On the
temporal query the lexical arm correctly flips `include_stale_on_render=True` — but
on `031748ae` `any_stale=False`. The three engineer facts are all *active* under
distinct slot keys and render in the same order under recency and lexical:

| turn | status | slot_key | text |
|---|---|---|---|
| 74 | active | `team.size` | team size: 5 engineers |
| 67 | active | `team size.size` | team size size: five engineers |
| 12 | active | `team members.count` | team members count: 4 engineers plus manager Rachel |

Query-aware ranking reorders the surrounding ~27k of context but cannot create a
started→now order the write side never recorded. There is no stale edge to surface
because supersession never fired (the values sit under four different keys; see the
write-recall audit above).

**Paid answers (one brief Gemma answer per item).** No selector produces the
correct two-part gold (started = 4, now = 5):

| selector (run) | `031748ae` brief answer | reading |
|---|---|---|
| recency (earlier `mix_report.json`) | "The memory does not contain information…" | refusal |
| recency (same-code rerun) | "You lead five engineers when you just started…" | wrong: 5 attributed to *start*; *now* omitted |
| lexical | "You lead 5 engineers." | *now* = 5 correct; *then* = 4 omitted |

The runner's `any()`-substring rule labelled lexical's `031748ae` a **pass** — a
measurement artifact (only "5 engineers" matched). Under the two-part gold it is
still a **temporal-structure** miss. `9bbe84a2` remains the documented
retrieval-recall artifact: the brief answer "previous goal … was 100" is correct,
but the bare-number needle "level 100" does not substring-match. Corrected, the
lexical mix is the same shape as recency — **4 pass / 1 temporal-structure** — with
`031748ae` unresolved.

**Two findings, one weak and one strong.** (1) Weak: the brief answerer is *not*
reproducible run-to-run at temperature 0 — three runs over the same `031748ae`
state gave a refusal, a wrong-attribution, and a now-only answer. Single brief-answer
labels are therefore weak evidence; the deterministic NPU-free structural check is
load-bearing. Needle methodology to fix before the next mix: normalize spelled
numbers ("five" ≡ "5") and require *all* gold needles for multi-part questions.
(2) Strong: across every selector and every run the model never reconstructs
then = 4 / now = 5, because the read path has no ordered then→now structure to
retrieve.

**Verdict.** The temporal `LexicalSelector` does **not** resolve `031748ae`. Read-path
query-awareness is exhausted for this item; the residual is write-side slot
fragmentation (no supersession → no order), which is exactly the Gate 4 trigger. No
promotion claim is made: these five oldest-gold states are diagnostic, not the
frozen 30-item development suite (Gate 1), which must still be materialized before
any Gate 2 promotion verdict. `031748ae` stays diagnostic, not gating (its "4" is
the team-outing headcount, a dataset inference).

**Next:** Gate 4 — the Qwen embedding-identity baseline, so semantically equivalent
keys (`team.size` / `team size.size` / `group size.number of engineers`) collapse,
supersession can order the team-size updates, and an ordered current-state set
reaches the read path. Then the DREAM challenger.

## Gate 4 — Qwen embedding-identity baseline

The string-first canonicalization residual is semantic, so this is the activated
embedding baseline DREAM must beat. The FLM NPU server returns null embeddings for
its generative models, so the baseline runs `Qwen/Qwen3-Embedding-0.6B` locally
(sentence-transformers, CPU) through the existing `evaluate_pairs` harness. Runner:
`evals/memory_methods/run_embedding_identity_local.py`; artifact:
`bench/memory_methods/embedding_identity_qwen.json`. Slot keys are embedded
symmetrically (no asymmetric query prompt), cosine over L2-normalized vectors.
Deterministic (re-run reproduces every similarity to 1e-6).

**Canonical 6-pair fixtures.** Best zero-false-merge threshold = **0.827**, same-slot
recall **0.667 (2/3)**.

| same? | cosine | left ↔ right |
|---|---:|---|
| merge | 0.9325 | morning routine.coffee cup limit ↔ morning coffee.maximum quantity |
| merge | 0.8266 | coding exercises.time spent per day ↔ daily coding practice.duration |
| merge | **0.6729** | team.size ↔ group size.number of engineers |
| keep apart | 0.7920 | game.current level ↔ game.target level |
| keep apart | 0.7551 | camera.model ↔ camera.capacity |
| keep apart | 0.6190 | team.size ↔ team outing.attendees |

**Extended real fragmented keys (9 pairs, from the write-recall audit).** Best
zero-false-merge threshold = **0.8125**, same-slot recall **0.5 (3/6)**.

| same? | cosine | left ↔ right |
|---|---:|---|
| merge | 0.8782 | team.size ↔ team size.size |
| merge | 0.8362 | coding exercises.time per day ↔ coding exercises.duration |
| merge | 0.8125 | goal.level ↔ level.target level for goal setting |
| merge | 0.7543 | team size.size ↔ group size.number of engineers |
| merge | 0.7344 | coding exercises.time spent per day ↔ coding exercises.frequency |
| merge | 0.6735 | goal.level ↔ user.goal |
| keep apart | **0.8093** | team.size ↔ team members.count  *(031748ae: 5-engineer team vs 4+Rachel outing)* |
| keep apart | 0.7909 | goal.level ↔ level goal.target level  *(current goal 150 vs prior goal 100)* |
| keep apart | 0.7057 | morning coffee.maximum quantity ↔ morning routine.quantity |

**Verdict: insufficient as a global-threshold merge — and it fails hardest exactly
where it must succeed.** The same-slot and different-slot cosine distributions
overlap. The hardest genuine paraphrase, `team.size ↔ group size.number of engineers`
(0.673 — the 031748ae fragmentation), scores *below* two must-not-merge traps. Worse,
on the extended set the `team.size ↔ team members.count` trap scores **0.809, above
most genuine merges**, so a threshold high enough to reject it recovers only half the
real fragments. Qwen3-Embedding-0.6B does reach merges string-first could not (coffee
0.93, coding-duration 0.83, `team.size ↔ team size.size` 0.88), but a single cosine
threshold cannot give the writer a clean, zero-false-merge supersession signal.

This is the baseline of record. Levers before/at the DREAM challenger: (a) a stronger
embedder (Qwen3-Embedding-4B/8B); (b) richer keys — embed subject+value or the full
fact text rather than the bare slot key, since "team members count: 4 engineers plus
manager Rachel" vs "team size: 5 engineers" carries the distinguishing signal the bare
keys drop; (c) per-subject local thresholds instead of one global cut. DREAM is now
unblocked (roadmap: it activates only after this baseline is materialized and scored),
but the 031748ae trap shows it must clear a genuinely hard separation, not just raise
average similarity.

### Key-composition sweep (the richer-key lever)

Lever (b) tested directly: the same Qwen-0.6B embedder, but embedding richer key
compositions. Runner `evals/memory_methods/run_embedding_identity_richkey.py`;
artifact `bench/memory_methods/embedding_identity_keycomp.json`. Strategies: `bare_key`
("team.size"), `natural_key` ("team size"), `subject` ("team"), `full_fact`
("team size: 5 engineers"), `subject_value` ("team: 5 engineers"). Evaluated on a
10-pair real-entry set built from actual captured-state entries (key + value),
labelled by *supersession intent* — the same underlying attribute over time is the
same slot, even when the value changed (coffee one→two, goal 100→150); genuinely
different concepts are traps. Deterministic.

| strategy | best zero-false-merge thr | same-slot recall |
|---|---:|---:|
| bare_key | 0.836 | 0.333 (2/6) |
| natural_key | 0.799 | 0.500 (3/6) |
| subject | 0.785 | 0.500 (3/6) |
| subject_value | 0.824 | 0.667 (4/6) |
| **full_fact** | **0.784** | **1.000 (6/6, 0 false)** |

**`full_fact` (natural key + value) is the first composition to cleanly separate the
real fragmented slots.** Lowest same-slot similarity 0.784 (coffee one→two) sits above
the highest trap 0.720 — a small but real margin. The two pivotal pairs invert versus
bare keys:

| pair | bare_key | full_fact | label |
|---|---:|---:|---|
| team.size ↔ group size.number of engineers | 0.673 | **0.942** | merge (031748ae paraphrase) |
| team.size="5 engineers" ↔ team members.count="4 engineers plus manager Rachel" | 0.809 | **0.720** | keep apart (031748ae trap) |

The value supplies disambiguating tokens the bare keys drop: "...4 engineers plus
manager Rachel" reads as a different fact from "team size: 5 engineers", while the
shared numerals/units pull genuine paraphrases together. Cross-value same-slot updates
survive (coffee one→two 0.784, goal 100→150 0.807, both ≥ threshold), so a full_fact
identity check would let "5 engineers" supersede "4…+Rachel"'s sibling keys and give
the read path the ordered then→now set 031748ae needs.

**Caveats.** (1) Small n (6 same / 4 trap); recall 1.0 here is encouraging, not
conclusive — confirm on a larger pair set before promoting. (2) The separating margin
(~0.06) is thin for a single global threshold; per-subject thresholds add robustness.
(3) full_fact owes its win partly to value tokens, which is double-edged for
supersession across *large* value changes — the cross-value pairs cleared the bar only
narrowly. (4) On the key-only canonical fixtures, value-free strategies do not improve
(natural_key 0.667 = bare; subject fails), so the gain is specifically from value
inclusion on real entries. Recommendation: adopt `full_fact` (or natural-key+value) as
the embedding-identity key, re-measure on a larger labelled set, and treat this as the
bar DREAM must beat — bare-key embedding is not it, but a richer key may close most of
the gap without a graph rebuild.

### Larger within-state set (n=67) — global threshold breaks, per-subject holds

Caveat (1) tested. Runner `evals/memory_methods/run_embedding_identity_largeset.py`;
artifact `bench/memory_methods/embedding_identity_largeset.json`. The set is built only
from real captured-state entries, all pairs within a single state (how supersession
actually compares), with auditable labels: 19 SAME (within explicit concept clusters),
16 HARD negatives (same-subject / same-`.model` traps), 32 EASY negatives (seeded
cross-concept sample). Cluster and trap definitions live in the runner; every resolved
key→value is saved in the artifact.

Global zero-false-merge recall by strategy: bare_key **0.21**, natural_key 0.21,
subject 0.32, subject_value 0.42, **full_fact 0.79**. full_fact still dominates by ~4×,
and it keeps the 031748ae behaviour (team.size ↔ group size.number of engineers 0.942;
trap team.size ↔ team members.count 0.717). But the clean global separation from the
10-pair run does **not** survive: margin vs the hardest negative is **−0.029**. One
collision causes it — `vehicle.model` "Ford F-150 pickup truck" ↔ `model car.type`
"Ford Mustang Shelby GT350R" at **0.795** (two Ford models) — and that negative sits in
the vehicle concept, which has *no* same-slot pairs to protect, yet it raises the single
global bar above genuine merges in other concepts (coffee one→two 0.784; the vague
"small" marketing-team value 0.766).

Per-subject thresholds remove that cross-concept contamination. Scoring each concept's
same-slot pairs against only the hard negatives that share its subject, full_fact gives
**5/5 clean separations**:

| concept | min same-slot | max relevant hard-neg | result |
|---|---:|---:|---|
| team size (031748ae) | 0.8463 | 0.7173 | clean |
| coding time (cc5ded98) | 0.8322 | 0.7461 | clean |
| marketing team size (c6853660) | 0.7664 | 0.4532 | clean |
| coffee limit (c6853660) | 0.7837 | 0.6889 | clean |
| Apex goal (9bbe84a2) | 0.8065 | 0.5811 | clean |

**Conclusion.** The richer key holds up at larger n *for the mechanism that matters*:
full_fact embedding identity, scoped per subject (compare a new entry only against
existing entries sharing its subject, or use a subject-local threshold), cleanly
separates same-slot from different-slot across all five real concepts — including the
031748ae team-size fragmentation that bare keys, string-first canonicalization, and the
recency/lexical read path all failed. A single *global* cosine threshold is the wrong
knob (one unrelated hard collision poisons it); subject-scoped comparison is the right
one and matches how supersession already works (per slot/subject).

Remaining honest gaps before promotion: the coffee cross-value pair (0.784) is the
thinnest real merge; the vehicle concept contributed only negatives (no same-slot recall
measured there); n is tens, not hundreds; and "small" vs "3"/"three people" shows
value-text quality varies. Next concrete step is to wire a subject-scoped full_fact
identity check into `_apply_supersession` behind a flag and re-run the five-state mix to
see whether 031748ae resolves end-to-end — the cheapest path to the then→now fix without
a graph rebuild, and the baseline DREAM would have to beat.

### End-to-end — embedding supersession over the full states (the curated benchmark lied)

Wired a pluggable slot-identity matcher into `_apply_supersession` behind a flag
(`Settings.embedding_supersession`, default off; `_slot_matcher` PrivateAttr on
`FactsLedger`; matcher in `rem/memory/semantic_identity.py`). With the matcher off the
write path is byte-for-byte the old behavior (NPU-free suite 194/194). Then replayed each
captured state's ledger through the *flagged* supersession (`full_fact` identity, local
Qwen, threshold 0.80) and re-ran the temporal LexicalSelector + one brief answer per item.
Runner `evals/memory_methods/run_supersession_endtoend.py`; artifact
`bench/battery/supersession_endtoend.json`.

**The target ordering does work.** On 031748ae the three "5 engineers" keys collapse and
order (`group size.number of engineers` t5 → `team size.size` t67 → `team.size` t74), and
`team members.count`="4 engineers plus manager Rachel" correctly stays a *separate active*
fact (0.717 < 0.80 — not merged). On 9bbe84a2 the Apex goal orders 100 → 150 (t61 "100"
marked stale, superseded by the t144 "150"). So the mechanism produces the ordered
then→now the read path wanted.

**But on real states a single global threshold over-merges badly.** Across the five states
it fired **1,583 value-changing merges** (vs only 145 safe same-value dedups) and ~20–25%
of active entries collapsed — the same unsafe profile as string-first subject-only. Sampled
false merges:

| sim | wrongly merged |
|---:|---|
| 0.856 | `dessert.name`="Poffertjes"  →  `dessert.name apple pie`="Dutch apple pie" (two different desserts) |
| 0.929 | `corporate team building package.capacity`="20 people"  →  `.size`="6 people" (capacity vs size) |
| 0.859 | `meets.offering`="vegan banana bread…"  →  `stach.offering`="vegan muffins, bars, cakes" (two different shops) |

The decisive point: genuine merges (team-size 0.87–0.94, apex 0.81) and false merges
(distinct desserts/shops 0.80–0.86, capacity-vs-size 0.93) **overlap in cosine space**, so
no global threshold separates them on real data. `full_fact` similarity measures "same kind
of fact," not "same slot" — it cannot tell *another value for this slot* from *a different
instance of the same attribute type* (Poffertjes vs apple pie). The curated 67-pair set
showed 0 false merges only because it never sampled distinct-instance/same-attribute
negatives; real states are full of them.

**End-to-end mix is unchanged: {4 pass, 1 retrieval-recall}, identical to baseline lexical.**
031748ae is still a real miss — its auto-`pass` is the substring artifact again (the model
answers "a team lead with 4 engineers plus manager Rachel, totaling 5 people", i.e. the
*outing* breakdown, not started=4/now=5). 9bbe84a2 is still wrong ("previous goal … level
150 eventually") **even though 100 is now correctly ordered as stale** — the small answerer
does not reliably read ordered stale history out of flat rendered text. Gold needles for the
three genuine passes (F-150, two hours, two cups) survived the over-merge.

**Verdict.** The cheapest path — richer-key embedding supersession with a global threshold —
does **not** safely resolve the then→now problem. It confirms three things: (1) 031748ae is
unresolvable by any *faithful* identity (the "4" is genuinely the outing count; merging it
would itself be a false merge); (2) global-threshold `full_fact` identity is unsafe on real
states (instance-vs-update collisions); (3) even when history is ordered, the gemma answerer
doesn't exploit it, so part of the gap is the answerer, not memory. What a safe mechanism
needs next: **attribute- and instance-aware identity** — gate merges on matching attribute
head-noun and value compatibility, not raw fact similarity (cheap), or a typed-claim /
graph store (heavy). DREAM is justified only if it supplies instance-aware identity, not
just higher average similarity; that is now the concrete bar. Also: fix the eval needle
methodology (multi-part/all-needles, spelled numbers) and consider an answerer that is
prompted to read stale→active ordering. Flag stays off; nothing promoted into the writer.

### Option (a) prototype — instance-aware (value-type) gate

Cheapest fix for the over-merge: gate the merge on value compatibility. A slot *update*
changes a value to a *compatible* one (quantity→quantity); two distinct *named* values are
different instances, not an update. `FullFactEmbeddingMatcher(value_aware=True)` blocks a
merge when the two slot_values differ and are not *both* quantity-like (digit or spelled
number; `quantity_like` in `rem/memory/semantic_identity.py`). Default off. Runner
`evals/memory_methods/run_supersession_instanceaware.py`; artifact
`bench/battery/supersession_instanceaware.json`. Deterministic.

Merge taxonomy over the five states (plain global-threshold matcher vs value-aware):

| class | plain | instance-aware |
|---|---:|---:|
| safe_dedup (values equal) | 145 | 147 |
| numeric_update (differ, both quantity) | 233 | 233 |
| **textual_distinct (differ, named values)** | **1350** | **0** |

**The value gate removes all 1,350 textual-distinct merges — the entire egregious
false-merge class** (Poffertjes vs Dutch apple pie, distinct vegan shops, `…type`
vs `…capacity`) — while preserving every numeric update and the two target collapses:
031748ae still collapses to a single active `team.size`="5 engineers" with the sibling
keys ordered stale, and 9bbe84a2 still orders the Apex goal 100 (stale) → 150 (active).
Active counts go *up* vs plain (e.g. 031748ae 704 → 891), i.e. ~150–190 distinct facts per
state are correctly kept instead of wrongly collapsed.

**Residual the value gate cannot catch.** Within the 233 surviving numeric merges, some are
genuine (the target collapses; `number of engineers`="5" ↔ `team.size`="5 engineers" at
0.94) and some are false: same-subject, *different numeric attributes* — `linkedin posts.likes`
="20" ↔ `…comments`="5" (0.895), `package.capacity`="20 people" ↔ `package.size`="6 people"
(0.93). A value-type gate can't separate these (both sides are quantities), and a naive
attribute-token match can't either, because the genuine merges *also* cross attribute tokens
(`number of engineers` ↔ `size`). Distinguishing "likes vs comments" from "number of
engineers vs size" needs *semantic attribute typing*, not string rules — which is exactly the
typed-claim / graph capability the heavier path provides. (Minor known limit: `quantity_like`
treats alphanumeric model names like "F-150" as quantity; harmless here since those pairs sit
below the cosine threshold anyway.)

**Verdict.** Option (a) is a real, cheap safety win — it eliminates the dominant false-merge
class and preserves all target behavior, turning the unsafe global matcher into one whose
only remaining errors are a bounded class of same-subject numeric-attribute collisions. It is
not a complete fix; closing the residual requires attribute-semantic / typed-claim identity,
which is the sharpened justification for DREAM or a typed store. Flag stays off; this is a
measured prototype, not a promotion.

### Option (a2) — attribute-similarity gate does NOT work; typed identity does

Tested whether attribute-embedding similarity can close the numeric residual (block
`likes`↔`comments`, `capacity`↔`size` while keeping `number of engineers`↔`size`). It cannot —
the genuine and false distributions are *inverted*:

| pair | attr cosine | needed |
|---|---:|---|
| number of engineers ↔ size | **0.5496** | ALLOW (same slot) |
| number of people ↔ size | 0.6481 | ALLOW |
| capacity ↔ size | 0.7149 | block |
| likes ↔ comments | **0.7510** | block |

`min(genuine)=0.55 < max(false)=0.75`: any threshold that allows the genuine cross-attribute
merge admits every false one, and any threshold that blocks the false ones kills the genuine
ones. Embedding similarity — of the fact (Gate 4), the key (sweep), or now the attribute —
fundamentally measures *resemblance*, not *identity*: "number of engineers" **is** team size
yet is embedding-distant, while "likes" and "comments" are different attributes yet embedding-
close. A2-as-a-similarity-knob is a dead end. (Probe in
`run_typed_identity_probe.py`'s sibling measurement; values above.)

A reasoning judge separates what similarity cannot. Asking the local gemma "SAME or DIFFERENT
slot?" on the eight decisive pairs scored **7/8** (`bench/memory_methods/typed_identity_probe.json`):
SAME for `number of engineers`↔`size`, coding duration↔time-per-day, coffee one↔two; DIFFERENT
for `likes`↔`comments`, `capacity`↔`size`, Poffertjes↔apple pie, and the 031748ae trap
(team size vs 4+Rachel). The one miss is the genuinely ambiguous apex pair
(`target level: 100` vs `level: 150`) — the same item that has been ambiguous throughout. So
typed/reasoning identity is the mechanism class that fits the residual; this is the concrete,
evidence-backed justification for a typed-claim store / DREAM (or an LLM-judge supersession
step), not "more similarity." Open costs before that path: an NPU judge call per candidate
merge at write time is expensive (the per-item ingests are already ~75 min), and the judge
itself needs held-out validation.

### Typed-judge write-cost — the band is large on real states

The typed/reasoning judge is the right mechanism class (above), but a write-time NPU
judge call per candidate pair is expensive, so `TypedIdentityMatcher`
(`rem/memory/semantic_identity.py`) judges **only** the cosine-ambiguous band
`[low, high)` — clear pairs stay on cosine (>=high SAME, <low DIFFERENT, no call).
`run_typed_band_cost.py` measures how big that band is = judge calls per ingest
(NPU-free: counting stub judge over `resupersede_state`, two bracketing policies).
Artifact: `bench/memory_methods/typed_band_cost.json`.

On the captured KU states the default band `[0.70, 0.88]` needs **~1467–1491 judge
calls per ingest** (upper/lower policy bracket within ~2%, so the band size is stable
regardless of merge decisions). That is prohibitive on top of the ~75-min ingest. A
narrower band `[0.80, 0.86]` cut observed calls **~6–8×** (3ba21379 1142→180,
cc5ded98 1701→266). So the cost lever is **band width + a candidate pre-filter**
(e.g. only judge same-subject or same-attribute-head pairs), not "judge the whole
band" — and that pre-filter should be chosen before any real NPU judge is wired in.

Caveat: measured on the 6 captured states (4 are the original overfit dev states + 2
fresh KU); deterministic and NPU-free, but the specific band thresholds are
illustrative, not tuned. Same overfitting caveat as below.

### Methodology — overfitting / benchmaxing risk (read before promoting anything)

Every Gate 4 result above was developed AND measured on the **same five captured states**
(the oldest-gold battery items): the 0.80 threshold, the concept clusters, the n=67 labels,
the value-gate, and these probe pairs. The gates were designed *after* inspecting these
states' own false merges. That is test-set peeking, so all of it is **diagnostic /
hypothesis-generating, not validated**. Concretely:

- No number here is a generalization estimate. "textual_distinct 1350→0" and "typed-judge
  7/8" are dev-set descriptions; they may not hold on unseen states.
- The mechanisms must be validated on held-out data before any writer promotion: the Gate-1
  **frozen 30-item development suite** (not yet materialized — the LongMemEval-S source isn't
  in the repo) for development numbers, and the Gate-5 **held-out LongMemEval-S** run for the
  promotion verdict. Until then the flag stays off and nothing ships.
- Deliberately NOT done: tuning a threshold to drive the dev residual to zero. The a2 probe
  reports the trade-off/overlap as-is precisely to avoid manufacturing a dev-set win.

What is reasonably *general* (mechanism-class claims, lower overfit risk): similarity ≠
identity (shown three independent ways), and typed/reasoning identity separates cases
similarity cannot. What is *not* general yet (specific thresholds, counts, the value-gate's
exact effect): pending the frozen suite.
