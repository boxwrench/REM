"""Throughput probe: measure the NPU compactor's *drain rate* and per-call latency.

This answers the gate question — "does compaction keep up in wall-clock?" — by
treating REM as a producer/consumer queue. The agent (producer) adds context
tokens; the NPU compactor (consumer) drains them. The queue is stable, and REM
reduces burden, only if:

    service_rate (tok/s drained)  >=  arrival_rate (tok/s the agent adds)

We measure the service_rate by streaming real LongMemEval haystacks turn-by-turn
(steady-state deltas, NOT a cold backlog dump) and timing every `compact_once`
call — which is two NPU round-trips: fact extraction + span summarization.

This probe measures COST/SPEED only. It says nothing about answer quality; that
is the separate battery axis.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass, field

from evals.battery.context_managers import _flatten, _render
from evals.battery.longmemeval_loader import load_knowledge_update
from rem.config import Settings
from rem.memory.compactor import compact_once, should_compact
from rem.memory.tiers import MemoryState, Turn, count_tokens
from rem.npu_client import NpuClient

ANSWERER_MODEL = "gemma4-it:e2b"

# Reference arrival rates (tokens/sec the agent adds to context) to test the
# stability gate against. A local agentic model on the iGPU emits output plus
# tool results; net new context grows at roughly these rates.
REFERENCE_ARRIVAL_RATES = {
    "slow_agent_10tps": 10.0,
    "typical_agent_30tps": 30.0,
    "fast_agent_60tps": 60.0,
}


@dataclass
class CompactionSample:
    absorbed_tokens: int  # verbatim tokens drained from the queue this call
    wall_s: float  # wall-clock for the compact_once call (2 NPU round-trips)
    summary_tokens: int  # tokens produced by summarization
    facts: int  # facts extracted into the ledger
    compacted: bool  # False = extraction/summarize fell back (no drain)


@dataclass
class ProbeResult:
    samples: list[CompactionSample] = field(default_factory=list)

    def drain_tok_per_s(self) -> float:
        """Aggregate service rate: total tokens drained / total compaction wall."""
        total_wall = sum(s.wall_s for s in self.samples)
        total_absorbed = sum(s.absorbed_tokens for s in self.samples)
        return (total_absorbed / total_wall) if total_wall > 0 else 0.0

    def summary(self) -> dict:
        ok = [s for s in self.samples if s.compacted]
        walls = [s.wall_s for s in self.samples]
        absorbed = [s.absorbed_tokens for s in ok]
        drain = self.drain_tok_per_s()
        produced = sum(s.summary_tokens for s in ok)
        total_absorbed = sum(absorbed)
        return {
            "n_compaction_calls": len(self.samples),
            "n_compacted": len(ok),
            "n_fallback": len(self.samples) - len(ok),
            "drain_tok_per_s": round(drain, 2),
            "per_call_wall_s": {
                "mean": round(statistics.mean(walls), 3) if walls else 0.0,
                "median": round(statistics.median(walls), 3) if walls else 0.0,
                "max": round(max(walls), 3) if walls else 0.0,
            },
            "tokens_absorbed_total": total_absorbed,
            "tokens_produced_total": produced,
            "compression_ratio": round(total_absorbed / produced, 2) if produced else 0.0,
            "stability_gate": {
                name: {
                    "arrival_tok_per_s": rate,
                    "keeps_up": drain >= rate,
                    "headroom_x": round(drain / rate, 2) if rate else 0.0,
                }
                for name, rate in REFERENCE_ARRIVAL_RATES.items()
            },
        }


def probe_item(item, client: NpuClient, settings: Settings, max_turns: int | None) -> list[CompactionSample]:
    """Stream one haystack's turns and time every compaction that fires."""
    state = MemoryState()
    samples: list[CompactionSample] = []
    turn_id = 0
    for sid, turn in _flatten(item.sessions):
        if max_turns is not None and turn_id >= max_turns:
            break
        turn_id += 1
        text = _render(turn)
        state.turns.append(
            Turn(role=turn.get("role", "user"), content=text, turn_id=turn_id, tokens=count_tokens(text))
        )
        while should_compact(state, settings):
            pre = [t.tokens for t in state.turns]  # snapshot before drain
            t0 = time.perf_counter()
            res = compact_once(state, client, settings)
            dt = time.perf_counter() - t0
            absorbed = sum(pre[: res.turns_compacted])
            samples.append(
                CompactionSample(
                    absorbed_tokens=absorbed,
                    wall_s=dt,
                    summary_tokens=count_tokens(res.new_summary or ""),
                    facts=res.new_facts_count,
                    compacted=res.compacted,
                )
            )
            if not res.compacted:
                break  # extraction/summarize fell back; stop to avoid spin
    return samples


def main() -> None:
    ap = argparse.ArgumentParser(description="REM NPU compaction throughput probe")
    ap.add_argument("--data", required=True, help="Path to LongMemEval JSON")
    ap.add_argument("--budget", type=int, default=4000, help="compact_trigger_tokens (when compaction fires)")
    ap.add_argument("--span", type=int, default=None, help="compact_span_turns (turns drained per compaction); default = Settings default")
    ap.add_argument("--items", type=int, default=2, help="number of knowledge-update items to stream")
    ap.add_argument("--max-turns", type=int, default=120, help="cap turns streamed per item (bounds runtime)")
    ap.add_argument("--out", default="bench/battery/throughput_probe.json")
    args = ap.parse_args()

    settings = Settings(
        summarizer_model=ANSWERER_MODEL,
        compact_trigger_tokens=args.budget,
        max_context_tokens=max(args.budget * 4, 32000),
    )
    if args.span is not None:
        settings.compact_span_turns = args.span
    client = NpuClient(settings)
    items = load_knowledge_update(args.data, limit=args.items)

    result = ProbeResult()
    t_start = time.perf_counter()
    for i, item in enumerate(items, 1):
        print(f"[{i}/{len(items)}] {item.question_id}: streaming up to {args.max_turns} turns ...", flush=True)
        samples = probe_item(item, client, settings, args.max_turns)
        result.samples.extend(samples)
        print(
            f"    fired {len(samples)} compaction(s); "
            f"running drain={result.drain_tok_per_s():.1f} tok/s",
            flush=True,
        )
    wall = time.perf_counter() - t_start

    out = {
        "config": {
            "model": ANSWERER_MODEL,
            "budget": args.budget,
            "compact_span_turns": settings.compact_span_turns,
            "keep_recent_turns": settings.keep_recent_turns,
            "npu_max_tokens": settings.npu_max_tokens,
            "items": len(items),
            "max_turns_per_item": args.max_turns,
        },
        "wall_total_s": round(wall, 1),
        "summary": result.summary(),
        "samples": [asdict(s) for s in result.samples],
    }
    from pathlib import Path

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    s = out["summary"]
    print("\n=== THROUGHPUT PROBE ===")
    print(f"compaction calls: {s['n_compaction_calls']} ({s['n_fallback']} fallback)")
    print(f"DRAIN RATE: {s['drain_tok_per_s']} tok/s")
    print(f"per-call wall: mean {s['per_call_wall_s']['mean']}s, median "
          f"{s['per_call_wall_s']['median']}s, max {s['per_call_wall_s']['max']}s")
    print(f"compression: {s['compression_ratio']}x  ({s['tokens_absorbed_total']} -> {s['tokens_produced_total']} tok)")
    print("stability gate (service >= arrival):")
    for name, g in s["stability_gate"].items():
        verdict = "KEEPS UP" if g["keeps_up"] else "FALLS BEHIND"
        print(f"  {name:24s} arrival={g['arrival_tok_per_s']:>5} tok/s  {verdict}  ({g['headroom_x']}x)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
