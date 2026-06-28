# Slot-Key Canonicalization — Design (string-first, NPU-free validation)

**Date:** 2026-06-28
**Status:** Design, awaiting approval
**Parent:** `bench/battery/FINDINGS.md` "Write-recall audit"; roadmap item 7 next lever.
**Predecessor:** the failure mix (read path holds 4/5) and the write-recall audit
(supersession fires on ~1.1–1.3% of entries; one value lands under up to 7 keys).

## 1. Goal and scope

The write-recall audit located the one internal defect: slot supersession barely
fires. `FactsLedger._apply_supersession` only collapses entries whose `slot_key`
strings are **exactly equal** (`existing.slot_key != new_entry.slot_key → skip`),
and the model emits a fresh `subject.attribute` key for nearly every mention, so the
same fact accumulates under many keys and never supersedes. `infer_slot_key` covers
only infra-domain keywords (rate_limit, region, host, …) and returns `None` on
ordinary conversation. The result is the 950-entry ledger bloat and the lone
temporal-structure miss (031748ae's "4 engineers" and "5 engineers" both active
under different keys, unordered).

This increment tests **string-first slot-key canonicalization** as the fix, and
measures its ceiling, entirely NPU-free against the five captured states. It does
**not** modify the write path or re-ingest. If canonicalization shows value here, a
follow-up wires it into `_apply_supersession` (write path) and pays one re-ingest to
confirm — the same Step-0-then-integrate discipline.

## 2. Non-goals (deferred)

- **Write-path / extractor changes.** This is a post-hoc transform over captured
  `MemoryState`s. Integration into `_apply_supersession` is the next increment.
- **Embedding / semantic key matching.** Thread 2. This increment is string-only on
  purpose, so its measured ceiling calibrates whether embeddings are warranted.
- **Per-item alias tables.** No hand rules mapping "group size" → "team size". That
  would game the five known items (cf. read-path spec D2). Canonicalization must be
  general and domain-agnostic.
- **Bounding the ledger writer.** Addressed indirectly (supersession shrinks it);
  not a separate mechanism here.

## 3. The defect, precisely

`slot_key = f"{subject}.{attribute}"` from model extraction. Observed fragmentation
for one concept (031748ae, team headcount):

```
group size.number of engineers = 5        team members.count = 4 engineers…
team size.size = five engineers           team.size = 5 engineers
team outing.number of people = 5          team outing.attendees = 6
```

Exact-string supersession cannot collapse any of these. Across the five states,
supersession touched 10–11 of ~850–940 entries (~1.2%).

## 4. Canonicalization

### 4.1 Canonical key function

`canonical_slot_key(slot_key, granularity) -> str`, pure and deterministic:

1. Split into `subject` and `attribute` on the last `.`.
2. Tokenize each on non-alphanumerics; lowercase; drop stopwords (a small closed
   list: of, the, per, for, to, in, on, number, count, total, …); singularize
   (suffix `-s`/`-es` strip with a short irregular map).
3. Canonical key = sorted unique token set, joined.

Two granularities, both implemented and measured:

- **`full`** (conservative): token set of `subject ∪ attribute`. Merges
  `team.size` ≈ `team size.size`; will not merge `team.size` with
  `group size.number of engineers`.
- **`subject`** (aggressive): token set of `subject` only. Merges all attributes of
  a subject (`coding exercises.frequency` / `.duration` → one key) at the risk of
  over-merging genuinely distinct attributes (`coffee maker.model` vs `.capacity`).

No per-item rules. The gap between `full` and `subject` recall — and `subject`'s
over-merge rate — is the headline measurement.

### 4.2 Re-keying transform (NPU-free)

`recanonicalize(state, granularity) -> MemoryState`:

1. Compute `canonical_slot_key` for every ledger entry that has a `slot_key`.
2. Group active entries by canonical key. Within a group, order by `source_turn_id`.
3. The newest is current (active); older members become **ordered history** —
   retained, marked stale, with `superseded_by_turn_id` set to the newest. History
   is kept (not discarded) so then→now questions remain answerable.
4. Entries without a `slot_key` are untouched.

Output is a new `MemoryState` whose current-state slots are de-fragmented and whose
prior values are retained in turn order per canonical slot.

### 4.3 Why retain history (then→now)

Keep-newest alone serves "what is true now" but destroys "what was it before"
(031748ae wants both). Grouping by canonical key with an **ordered** value sequence
gives the read path "previous = 4 (turn 12) → current = 5 (turn 74)" inside the flat
ledger — the lighter-than-graph temporal ordering, without temporal edges.

## 5. Validation (NPU-free, against the five states)

For each state and each granularity:

- **Fragmentation collapse:** `fragmented_values` and supersession rate from
  `write_recall_audit` before vs after recanonicalize.
- **Ledger size:** active-entry count before vs after.
- **Over-merge check (`subject` only):** count canonical slots that absorbed
  entries with materially different values that are *not* a plausible update
  (heuristic: > 1 distinct current value of different type) — the cost of aggression.
- **Outcome:** re-run `mix_report` on the recanonicalized states. The one paid
  inference: a single brief answer on 031748ae, to see whether the ordered
  then→now canonical slot lets the model give "started 4 → now 5" (the current
  temporal-structure miss).

## 6. Success criterion

Not a pass/fail bar — a measurement. The increment succeeds when it reports, per
granularity: fragmentation reduction, supersession rate, ledger shrink, over-merge
cost, and the 031748ae outcome. The decision it feeds: does string-first
canonicalization collapse enough fragmentation (and resolve 031748ae) to wire into
the write path, or is the residual fragmentation semantic — justifying Thread 2
(embedding key-matching)?

## 7. Decision log

| # | Decision | Chosen | Rejected | Reverse if… |
|---|---|---|---|---|
| F1 | Where | Post-hoc transform over captured states | Patch `_apply_supersession` now | Validation needs write-time effects a transform can't model; then integrate + re-ingest. |
| F2 | Key signature | Sorted token set (bag) | Ordered string normalize | Token order proves meaningful for keys; unlikely. |
| F3 | Granularity | Measure both `full` and `subject` | Pick one upfront | One clearly dominates after measurement. |
| F4 | History | Retain prior values ordered per slot | Keep-newest, drop priors | then→now questions prove rare; then keep-newest to shrink more. |
| F5 | No aliases | General normalization only | Curated synonym table | Over-fits the five known items (D2); never. |
| F6 | Scope | String-only this increment | Add embeddings now | String ceiling is measured insufficient → Thread 2 with that evidence. |

## 8. What this unblocks

A measured answer to "is the write-side defect string-fixable?" If yes →
write-path integration + one confirming re-ingest. If the residual is semantic →
Thread 2 (embedding-based key matching) starts with concrete evidence of what string
normalization could not merge. Either way the ledger bloat and the 031748ae miss get
a quantified, non-graph remedy or a justified escalation.
