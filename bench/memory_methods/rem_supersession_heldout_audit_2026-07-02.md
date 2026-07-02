# REM held-out supersession audit — fresh states (2026-07-02)

**VERDICT (lead): The ce6d2d27 numeric over-merge problem reproduces on every one of the six fresh held-out states — and is worst on 22d2cb42 (78 numeric merges).** The value-gate's headline is real but misleading: it wipes out the textual_distinct class (→0 everywhere), yet the `numeric_update` bucket it keeps is dominated by FALSE merges of distinct facts that merely share a numeric attribute (start↔end dates, min↔max, sets↔reps, different instances/shops/accounts/sectors/demographics). Genuine then→now temporal updates are essentially absent in these states; the few non-false pairs are benign same-fact restatements (same value, different units/wording, sim ≈0.9–0.99), not true updates.

Recapture status: **complete** — all six fresh ids captured (batch limit of 6 reached; no capture process running). NPU-free throughout (local Qwen CPU, no answerer). Nothing committed.

## Headline numbers (textual_distinct plain → instance_aware; numeric_update count)

| state | plain textual | instance_aware textual | numeric_update | false-merge read |
|-------|--------------:|-----------------------:|---------------:|------------------|
| ce6d2d27 | 304 | **0** | 46 | ~40 false / ~6 benign / ~0 genuine |
| 945e3d21 | 273 | **0** | 45 | ~34 false / ~11 benign / ~0 genuine |
| 6071bd76 | 318 | **0** | 47 | ~42 false / ~4 benign / ~1 genuine |
| 22d2cb42 | 329 | **0** | 78 | ~72 false / ~4 benign / ~2 borderline |
| dfde3500 | 230 | **0** | 45 | ~35 false / ~9 benign / ~0 genuine |
| affe2881 | 276 | **0** | 30 | ~25 false / ~5 benign / ~0 genuine |
| **TOTAL** | **1730** | **0** | **291** | overwhelmingly false in every state |

(false/benign/genuine are my by-eye read of the dumped pairs, not a token heuristic.)

## Representative false merges per state (read, not counted)

**ce6d2d27** — start date May 26 ↔ end date May 28; price min 40 ↔ max 50; capacity min 200 ↔ max 1,500; fridge 5 days ↔ freezer 3 months; two cafés' walk distances (10-min ↔ 12-min ↔ 15-min); geonames 11M ↔ getty 1M place names; kowloon peak pop 50,000 ↔ density 3,250,000; women-in-energy 27% ↔ women-in-construction 12%; four different meal protein values (30g/35g/40g) collapsed together; grooming kit pieces 7 ↔ price $75.

**945e3d21** — techxpo start Feb 10 ↔ end Feb 12; market start 9AM ↔ end 1PM; spartan start age 7 ↔ end age 30; residents under-18 16,667 ↔ over-65 4,017; three people's birth ↔ death dates merged (Oct 1858↔Jan 1932, Aug 1848↔Feb 1921); followers 200 ↔ likes 50 ↔ comments 10 ↔ users 5 all cross-merged; two accounts' follower counts (1.4M ↔ 2.5M); white 34.3% ↔ black 57.4% population.

**6071bd76** — six distinct fitness accounts' follower counts (200K/500K/1M/1.5M/2M/2.5M/4.5M) merged pairwise; the whole probability-definition ladder collapsed (very-unlikely 30%, unlikely 30–40%, even 50%, probable 60–80%, very-probable 80%+); undergrad 17% ↔ grad 36% intl; chicken fridge 3–4 days ↔ frozen 3 months; succulent spring/summer ↔ fall/winter watering. *One plausibly genuine:* CNN accuracy initial 75% ↔ after-tuning 93% (before/after of one model).

**22d2cb42 (worst)** — DV-2024 registration start-time ↔ end-time (both EDT/EST stamps); exhibition start ↔ end dates; data-scientist salary across five sectors (finance/tech/healthcare/govt/retail) merged into one; nearly every exercise's sets ↔ reps merged (russian twists sets 3 ↔ reps 12-15, plus calf raises, plank, glute bridges, donkey kicks, hip thrusts, jump/box squats…); three ski resorts' terrain sizes (1,500 ↔ 2,300 ↔ 3,000 acres); galápagos/iguazú minimum ↔ recommended stay; google 24wk ↔ facebook 26wk parental leave; score min 1 ↔ max 5. *Borderline-genuine:* posting-frequency change story (was 3×/week → starting once a week), but merged across too many distinct slots to count as clean.

**dfde3500** — calvin & hobbes start 1985 ↔ end 1995; program phase weeks 1–4 (20–30 min) ↔ weeks 9–12 (25–35 min); small ↔ medium ↔ larger fish feeding amounts (all three instances); meditation daily-duration ↔ weekly-frequency; verizon monthly $70–130 ↔ pay-as-you-go 2.05; whatsapp member limit 256 ↔ group-**name** limit 25 chars ↔ description limit 512 chars; morning ↔ evening rush hour; four trip-option total costs ($4,984/$5,188/$5,372).

**affe2881** — refrigerated broth 5–7 days ↔ frozen 3–4 months; HD 1080p ↔ 4K 2160p; total-cost min $10k ↔ max $20k; multiple smartwatches' battery-life ↔ water-resistance and cross-device battery lives (fitbit 5d/7d, apple 36h, polar 30h); NYC spring ↔ summer ↔ winter temperatures; fridge 40°F ↔ freezer 0°F; one-way $250–400 ↔ round-trip $400–600; "number of wives twice" ↔ "number of children three".

## Bottom line

Confirmed on the newer fresh states: the instance-aware value-gate solves the textual_distinct false-merge class but does **not** solve the numeric one. `numeric_update` is a false-merge sink — the gate cannot tell a real then→now update from two distinct facts sharing a numeric attribute, and in this held-out suite almost all of that bucket is the latter. The count-only headline ("numeric_update = N, textual = 0") should not be trusted as evidence of safety; the pairs must be read (or run through a typed/semantic judge). 22d2cb42 shows the failure scales with fact density.

Outputs: `supersession_instanceaware_freshcheck.json`, `numeric_merge_audit_fresh.json` (both in `bench/memory_methods/`).
