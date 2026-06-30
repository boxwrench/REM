# REM — Glossary

Project-local terms. General repo terms live in the main docs.

**REM** — **Resident Externalized Memory**: this project. The agent-memory layer
externalized off the model/context and made resident on the NPU. Also the sleep
phase where memory consolidates; the guiding metaphor (NPU consolidates memory while
the iGPU is "awake").

**NPU** — Neural Processing Unit. On Strix Halo this is the **XDNA 2** engine.

**XDNA 2 / AI Engine (AIE)** — AMD's spatial **dataflow** accelerator: a 2D array
of tiles, each with a vector processor, a scalar RISC core, ~32KB local SRAM, and
streaming tile-to-tile interconnects. Lineage: Xilinx **Versal** adaptive SoCs,
where the same fabric does DSP (radar, 5G, signal analysis) and AI. ~50 TOPS class.

**Dataflow** — compute model where data *streams* through compute tiles rather than
being repeatedly fetched from cache. Efficient for continuous linear algebra.

**iGPU** — the integrated GPU (gfx1151) that runs our big MoE model lanes via
ROCm/Vulkan llama.cpp. Shares the ~212–215 GB/s unified memory bus with the NPU.

**FLM / FastFlowLM** — NPU inference runtime for Ryzen AI XDNA2 chips ("Ollama for
the NPU"). Provides the actual NPU LLM/Whisper execution.

**Lemonade / Lemonade Server** — AMD's open-source local AI server; orchestrates
backends (llama.cpp, FastFlowLM, whisper.cpp, etc.) behind OpenAI/Anthropic/Ollama
-compatible APIs. The contest is built around it.

**XRT / IRON** — AMD XDNA runtime (XRT) and the IRON/MLIR-AIE compiler toolchain
needed to target the NPU on Linux. `amdxdna` is the kernel driver.

**IRON / MLIR-AIE** — the framework (Python/MLIR) for writing *custom* kernels that
place compute + dataflow on the NPU tile array. The "deep path"; REM's A/B paths do
not need it. Repo: `Xilinx/mlir-aie`.

**llvm-aie / Peano** — an LLVM/Clang fork that compiles to the AI Engine's VLIW
instruction set; the backend IRON uses. Target triple for XDNA2/Strix is
`aie2p-none-unknown-elf` (XDNA1 = `aie2`). "Experimental" maturity. Repo:
`Xilinx/llvm-aie`. See `research/toolchain-npu-stack.md`.

**VLIW** — "Very Long Instruction Word": a CPU style where one instruction bundle
drives several functional units at once, and the *compiler* (not the hardware)
schedules around the pipeline. The AI Engine tiles are exposed-pipeline VLIW cores,
which is why a specialized compiler (Peano) is needed.

**Triton** — a Python language (from OpenAI) for writing GPU compute kernels at a
high level; widely used in the ML world. You write `@triton.jit` functions instead
of raw CUDA/assembly.

**Triton-XDNA** — official AMD project that compiles Triton kernels to the XDNA NPU
(Triton → triton-shared/Linalg → MLIR-AIR/AIE → llvm-aie → XRT binary). Supports
AIE2 and AIE2P (Strix). The approachable front-end for custom NPU kernels; supports
matmul/softmax/layernorm/reductions — i.e. the math behind embeddings + similarity.
Repo: `amd/Triton-XDNA`. See `research/toolchain-npu-stack.md`.

**MLIR-AIR** — a dataflow-placement compiler layer (Xilinx/mlir-air) that maps tiled
compute onto the NPU core array; sits between Triton-XDNA's front-end and MLIR-AIE.

**GAIA** — AMD's open-source Ryzen AI **agent framework** (`amd/gaia`): runs local LLMs
on NPU+iGPU via Lemonade, with a LlamaIndex **vector-RAG** memory pipeline. REM's prior
art and foil — REM uses wiki/graph + NPU compaction instead of vector RAG.

**xrt-smi** — the NPU management CLI (`xrt-smi examine --device 0000:c6:00.1 --report aie-partitions`); REM's
NPU-side telemetry signal for Path C. (`amd-smi`, the GPU equivalent, is broken on
Strix Halo gfx1151 — use `amdgpu_top`/sysfs for the iGPU side.)

**amdgpu_top** — community tool that reads Strix Halo iGPU utilization/VRAM/power from
sysfs (works where `amd-smi` reports N/A). REM's iGPU-side telemetry for Path C.

**Compaction** — replacing older verbatim transcript with shorter summaries to fit
the context window. Standard but lossy and (normally) blocking.

**Freeform summary** — prose memory written by a model. Useful as a compact view, but
not reliable enough to be the only source of truth for current facts.

**Fact ledger** — REM's structured list of load-bearing facts extracted from compacted
turns. The ledger is rendered in full during assembly so important facts do not depend
only on prose summaries.

**Stale ghost** — an old value that remains visible as if it were current after a later
correction. A7a exposed this with `ratelimit`: the old `1,200 requests per minute`
value survived beside the corrected `950 requests per minute` value.

**Supersession** — the rule that a newer correction for the same fact slot wins. Good
memory systems keep provenance for the old value, but do not render it as active state.

**Write-time adjudication** — deciding whether a newly extracted fact should keep,
replace, stale, or conflict with an existing ledger entry when memory is written, rather
than hoping the answer model resolves contradictions later.

**Slot fragmentation** — the core write defect: the extractor's exact-string slot keys put
the same attribute under many keys (`team.size`, `team size.size`, `group size.number of
engineers`), so genuine updates never collapse, the ledger bloats, and there is no ordered
then→now state. The lever the supersession work targets.

**full_fact identity** — deciding "same slot" from the cosine similarity of two entries'
`"natural key: value"` text rather than the bare key. Separates same-slot from
different-slot far better than bare keys, but a single *global* cosine threshold
over-merges on real states (distinct instances at the same similarity as real updates).

**Value-gate (instance-aware identity)** — block an embedding merge when the two values
differ and are not *both* quantity-like, so a slot UPDATE (5→5, one→two, 100→150) collapses
while two distinct *named* instances (Poffertjes vs Dutch apple pie) stay separate. Removes
the whole textual-distinct false-merge class; residual is same-subject different-numeric
attributes (likes vs comments).

**Typed-judge identity (`TypedIdentityMatcher`)** — using an LLM SAME/DIFFERENT judgment
instead of a cosine threshold to decide slot identity. Separates what similarity cannot
(number-of-engineers vs size = SAME; likes vs comments = DIFFERENT). Called *only* for
cosine-ambiguous **band** pairs to bound write-time NPU cost.

**Band cost** — judge calls per ingest = how many candidate pairs fall in the typed-judge's
ambiguous cosine band `[low, high)`. ~1,490 on real states at the default band (prohibitive
on top of the ~75-min ingest); the lever is band width + a candidate pre-filter.

**Frozen development suite** — the Gate-1 30-item LongMemEval-S manifest (10 knowledge-update
/ 10 temporal / 10 multi-session, SHA-pinned to the dataset, `031748ae` excluded) used to
validate write-side mechanisms on held-out data instead of the 5 overfit dev states.

**Render-time suppression** — hiding an older active ledger value during context
assembly when the recent verbatim window contains a newer, different value for the
same known slot. The old value can remain as audit state, but it should not be
presented as current memory.

**Render quarantine** — a render-time mechanism that builds an index of known stale values from the facts ledger and suppresses them from appearing in both the active ledger and episodic summaries during context assembly, preventing them from leaking as stale ghosts.

**Episode preservation** — keeping the original source turns or blocks as audit
evidence even after summaries/facts are generated. The compact view can be repaired
because the ground truth was not destroyed.

**Premise resistance** — refusing an outdated assumption embedded in a later question
or task. Example: if the current ratelimit is 950 rpm, a query that assumes 1,200 rpm
should be corrected rather than obeyed.

**Working / episodic / semantic memory** — the agent-memory hierarchy. Working =
recent verbatim turns; episodic = session summaries; semantic = durable
entity/concept knowledge across sessions. REM: A=working+episodic, B=semantic.

**LLM Wiki (Karpathy)** — pattern (Apr 2026) of compiling knowledge into a
maintained markdown wiki (pages + backlinks + schema) instead of RAG retrieval;
knowledge *compounds* rather than being re-derived. Obsidian variant skips vectors.

**agentmemory** — open-source implementation of the LLM-Wiki pattern (Rohit
Ghumare) adding confidence scoring, lifecycle, knowledge graph, hybrid search.

**Obsidian** — markdown PKM app; a vault is plain `.md` files with `[[backlinks]]`
and a graph view. Used here as the on-disk substrate for the semantic store (B).

**RSB-3** — our Reasoning Stress Battery context-window probe: ~26K-token CA
drinking-water regulations + a 3-part TTHM needle question. REM's memory-fidelity
test rig (does compacted/filed memory still answer it correctly?).

**Keep-up rate** — NPU compactor throughput (input tokens/sec through the summarizer)
divided by the transcript's growth rate (e.g. 30 tokens/sec). A ratio > 1.0x means
compaction can run in conversational pauses without lagging behind the dialog.

**Re-prefill tax** — the time latency cost of performing the first prompt prefill on the
host iGPU LLM after a memory compaction swap. Because the prompt prefix has changed
(due to compaction), the KV cache is invalidated and must be entirely re-prefilled.

**MTP / speculative decoding** — draft small/cheap tokens, verify with the big
model in one pass. P1 explores splitting draft (NPU) from verify (iGPU).

**Contention budget** — the measured cost of running NPU + iGPU work together. REM's
M1/M2 run (3 runs × N=20) measured **~3.8%** iGPU decode loss with concurrent NPU
compaction on this box (vs **~4.8%** for the CPU control), with **12.15 ± 1.20 tok/s**
compaction throughput and **0.143 tok/s/W**. The older
external ~5.8% figure is historical motivation only unless a citable source is added.

---

## Plain-language ML basics (for `TEACHING.md` readers)

Everyday definitions of the machine-learning terms used in the teaching doc. These
trade precision for clarity on purpose.

**CPU / GPU / NPU** — three kinds of processor. CPU = generalist. GPU = big parallel
number-cruncher (does AI's heavy math). NPU = small power-sipping specialist for AI
math.

**TOPS** — "trillions of operations per second." A rough speed rating for AI chips.
The XDNA 2 NPU is ~50 TOPS.

**APU** — a single chip that puts CPU + GPU (+ NPU) together. Strix Halo is an APU.

**Unified memory** — one pool of RAM that all the processors (CPU/GPU/NPU) share,
instead of each having separate memory.

**Memory bandwidth** — how fast a chip can move data to/from RAM (measured in GB/s).
Often the real speed limit for AI, more than raw math.

**Bandwidth-bound** — a task whose speed is capped by memory bandwidth, not compute.
LLM text generation is bandwidth-bound (it must stream the model's weights from RAM
for every token).

**von Neumann architecture** — the classic computer design where data and
instructions are fetched from central memory, processed, and written back. The
back-and-forth to memory is the bottleneck for AI math.

**Dataflow architecture** — an alternative where data streams *through* a grid of
small compute units (like a conveyor belt), minimizing trips to central memory. The
NPU works this way.

**SIMD** — "Single Instruction, Multiple Data": apply one operation to many numbers
at once. What vector processors (and GPUs/NPUs) are built for.

**DSP (Digital Signal Processor)** — a chip specialized for continuous real-time data
streams (radio, audio, radar). The NPU shares this heritage.

**Token** — a chunk of text (roughly ¾ of a word) that a model reads/writes one at a
time. "Context length" is counted in tokens.

**Context window** — the span of text (in tokens) a model can "see" at once. Fill it
up and old content must be dropped or summarized.

**Prefill vs decode** — two phases of running a model. *Prefill* = reading your whole
prompt to produce the first token (compute/bandwidth-heavy, bursty). *Decode* =
generating the rest one token at a time (steady, bandwidth-bound).

**KV cache** — the model's short-term scratchpad in RAM that stores what it has
already read this conversation, so it doesn't reprocess everything each token. This is
where "working memory" physically lives.

**Re-prefill tax (plain-language)** — the delay penalty when a model's scratchpad (KV cache) is thrown out because the conversation history was summarized. The model has to re-read the entire summarized context from scratch on the first turn.

**Keep-up rate (plain-language)** — how fast the summarizer (NPU) works compared to how fast you talk. If you generate 30 tokens of dialog a second and the NPU processes 120 tokens a second, your keep-up rate is 4x, meaning it will easily finish tidying up during your brief pauses.

**Parameters / weights** — the billions of numbers a model learned during training.
"8B" = 8 billion parameters. Bigger usually = smarter but slower and heavier.

**Quantization / INT8** — storing those weights with fewer bits to save memory and
speed up math. INT8 = 8-bit integers — much smaller than full precision, with a small
accuracy cost.

**Matmul (matrix multiplication)** — the core math operation of neural networks.
Almost everything (embeddings, attention) reduces to matmul.

**Transformer / self-attention** — the architecture behind modern LLMs. Self-attention
lets every token "look at" every other token — powerful, but its cost grows
quadratically with length.

**O(n²) (quadratic scaling)** — shorthand for "cost grows with the square of the
size." Double the context (n), ~4× the attention work. Why long chats get slow.

**Embedding** — a list of numbers (a vector) that represents the meaning of a piece of
text, so similar meanings sit near each other numerically.

**Cosine similarity** — a way to measure how "close in meaning" two embeddings are by
the angle between their vectors. Used to find relevant past context.

**RAG (Retrieval-Augmented Generation)** — the older memory approach: store text as
embeddings in a vector database and fetch the closest matches at query time. REM
deliberately avoids this in favor of the wiki/graph approach.

**MoE (Mixture-of-Experts)** — a big model split into many "expert" sub-networks where
only a few activate per token. Notation like "35B-A3B" = 35B total parameters, ~3B
*active* per token: big-model quality at small-model speed. Our iGPU lanes are MoE.

**Needle in a haystack** — a test that hides a fact in a huge context and checks the
model can still retrieve it. RSB-3 is a harder, multi-part version (see main glossary).

**Compaction / summarization** — shrinking old conversation into short summaries so the
context window doesn't overflow. Phase A does this on the NPU.
