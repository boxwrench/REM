"""Post-generation robustness pipeline for REM fact extraction."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from rem.memory.facts_ledger import FactEntry, validate_and_repair_items, clean_and_check_truncation
from rem.memory.tiers import Turn
from rem.npu_client import NpuClient
from rem.memory.prompts import (
    FACT_EXTRACTION_RETRY_MESSAGE,
    FACT_EXTRACTION_TRUNCATION_RETRY_MESSAGE,
)

logger = logging.getLogger(__name__)


def strip_markdown_fences(text: str) -> str:
    """Strips markdown code fences (balanced or unbalanced) from the text."""
    text = text.strip()
    # Find matching ```json ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    
    # Handle unbalanced fences starting with ```json or ```
    start_fence = re.search(r"^```(?:json)?\s*", text)
    if start_fence:
        text = text[start_fence.end():].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def find_balanced_json(text: str) -> tuple[str, bool]:
    """Locates the first balanced JSON array or object in the text.
    
    If the structure is unbalanced due to truncation at the end (e.g. starts with '['
    and ends with '}'), it heals it (e.g. by appending ']') and returns the healed string.
    
    Returns (extracted_json, is_truncated).
    """
    text = text.strip()
    if not text:
        return "", False

    first_bracket = text.find("[")
    first_brace = text.find("{")
    if first_bracket == -1 and first_brace == -1:
        return text, True  # No JSON start found, treat as truncated

    start_char = "[" if (first_bracket != -1 and (first_brace == -1 or first_bracket < first_brace)) else "{"
    start_idx = first_bracket if start_char == "[" else first_brace

    stack = []
    in_string = False
    escaped = False
    string_char = None
    last_balanced_end = start_idx

    for i in range(start_idx, len(text)):
        c = text[i]
        if escaped:
            escaped = False
            continue
        if c == "\\":
            if in_string:
                escaped = True
            continue
        if c in ('"', "'"):
            if not in_string:
                in_string = True
                string_char = c
            elif string_char == c:
                in_string = False
                string_char = None
            continue

        if in_string:
            continue

        if c in ("[", "{"):
            stack.append((c, i))
        elif c in ("]", "}"):
            if stack:
                top_char, top_idx = stack[-1]
                if (c == "]" and top_char == "[") or (c == "}" and top_char == "{"):
                    stack.pop()
                    if not stack:
                        last_balanced_end = i
                        # Look ahead: skip whitespace/commas and check if there's another starting bracket/brace
                        # indicating sibling objects.
                        has_sibling = False
                        for j in range(i + 1, len(text)):
                            if text[j].isspace() or text[j] == ",":
                                continue
                            if text[j] in ("[", "{"):
                                has_sibling = True
                            break
                        if not has_sibling:
                            return text[start_idx : i + 1], False
                else:
                    stack.pop()
                    if not stack:
                        last_balanced_end = i
                        has_sibling = False
                        for j in range(i + 1, len(text)):
                            if text[j].isspace() or text[j] == ",":
                                continue
                            if text[j] in ("[", "{"):
                                has_sibling = True
                            break
                        if not has_sibling:
                            return text[start_idx : i + 1], False

    # If we scanned to the end and stack was not empty (unbalanced/truncated)
    # check if we had a valid balanced prefix earlier (e.g. in {"a": 1} {"b": (truncated))
    if last_balanced_end > start_idx:
        return text[start_idx : last_balanced_end + 1], True

    candidate = text[start_idx:].strip()
    if candidate.startswith("[") and candidate.endswith("}"):
        return candidate + "]", True

    return candidate, True


def coerce_sibling_objects_to_list(text: str) -> str:
    """Converts a sequence of JSON objects into a list if not already wrapped."""
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        if re.search(r'\}\s*\{', text):
            converted = re.sub(r'\}\s*\{', '},{', text)
            return f"[{converted}]"
    return text


def detect_token_loop(text: str, min_window: int = 8, max_window: int = 16) -> bool:
    """Checks for consecutive repeating windows of tokens (8-16 tokens)."""
    tokens = text.split()
    n = len(tokens)
    for w in range(min_window, min(max_window + 1, n // 2 + 1)):
        for i in range(n - 2 * w + 1):
            w1 = tokens[i : i + w]
            w2 = tokens[i + w : i + 2 * w]
            if w1 == w2:
                joined = " ".join(w1)
                # Avoid triggering on boilerplate formatting/punctuation sequences
                if sum(c.isalnum() for c in joined) > 5:
                    return True
    return False


def detect_loops(raw_text: str, parsed_items: list[Any] | None = None, is_truncated: bool = False) -> bool:
    """Heuristic loop/degeneracy detector."""
    # 1. Consecutive token window repetition (8-16 tokens)
    if detect_token_loop(raw_text):
        return True

    # 2. Max tokens hit with unbalanced brackets. Only check if response is long
    # (>= 500 chars or >= 100 tokens), which prevents false positives on small mock unit tests.
    if is_truncated and (len(raw_text) >= 500 or len(raw_text.split()) >= 100):
        braces = raw_text.count("{") - raw_text.count("}")
        brackets = raw_text.count("[") - raw_text.count("]")
        if braces != 0 or brackets != 0:
            return True

    # 3. One item repeated N times (N=3)
    if parsed_items:
        seen = {}
        for item in parsed_items:
            if isinstance(item, FactEntry):
                item_str = json.dumps(item.model_dump(), sort_keys=True)
            elif isinstance(item, dict):
                item_str = json.dumps(item, sort_keys=True)
            else:
                item_str = str(item)
            seen[item_str] = seen.get(item_str, 0) + 1
            if seen[item_str] >= 3:
                return True

    # 4. Raw substring repetition (sequence of 4+ words repeating 3+ times containing content keys)
    tokens = raw_text.split()
    if len(tokens) > 12:
        for i in range(len(tokens) - 4):
            seq = " ".join(tokens[i : i + 4])
            if len(seq) > 20 and any(k in seq for k in ("subject", "value", "text")) and raw_text.count(seq) >= 3:
                return True

    return False


def attempt_salvage(
    raw_text: str,
    turns: list[Turn],
    diagnostics: dict[str, Any]
) -> tuple[list[FactEntry], bool]:
    """Attempts to salvage a looping response by isolating the first balanced array,
    collapsing degeneration, and validating the remaining facts.

    Keep-first, not latest-wins: salvage is degeneration-recovery, NOT supersession.
    Real corrections are handled downstream by the ledger; do not try to supersede here.
    """
    from json_repair import repair_json
    try:
        stripped_text = strip_markdown_fences(raw_text)
        isolated_text, _ = find_balanced_json(stripped_text)
        isolated_text = coerce_sibling_objects_to_list(isolated_text)
        repaired_text = repair_json(isolated_text)
        parsed = json.loads(repaired_text)
        
        if isinstance(parsed, dict):
            parsed = [parsed]
            
        if not isinstance(parsed, list):
            return [], False

        # Collapse degeneration: keep first occurrence of each (subject, attribute) slot
        # and drop later repeats; also exact-dedup identical objects.
        collapsed = []
        seen_slots = set()
        seen_items = []
        
        for item in parsed:
            if not isinstance(item, dict):
                continue
                
            # Defensively coerce fields to string
            for field in ("subject", "attribute", "value", "text"):
                if field in item and item[field] is not None:
                    item[field] = str(item[field])

            # Exact-dedup identical objects
            if item in seen_items:
                continue
                
            subj = item.get("subject")
            attr = item.get("attribute")
            
            # Try to resolve subject/attribute from text if missing
            if (not subj or not attr) and item.get("text"):
                from rem.memory.facts_ledger import parse_general_fact, infer_slot_key
                p_subj, p_attr, _ = parse_general_fact(item["text"])
                if p_subj and p_attr:
                    subj = p_subj
                    attr = p_attr
                else:
                    slot_key = infer_slot_key(item["text"])
                    if slot_key and "." in slot_key:
                        subj, attr = slot_key.split(".", 1)
                        
            slot_key_tuple = None
            if subj and attr:
                slot_key_tuple = (str(subj).strip().lower(), str(attr).strip().lower())
                
            if slot_key_tuple:
                if slot_key_tuple in seen_slots:
                    continue
                    
            collapsed.append(item)
            seen_items.append(item)
            if slot_key_tuple:
                seen_slots.add(slot_key_tuple)
                
        if not collapsed:
            return [], False
            
        fact_entries = validate_and_repair_items(collapsed, turns)
        if len(fact_entries) >= 1:
            diagnostics["loop_salvaged"] = True
            diagnostics["success"] = True
            diagnostics["loop_detected"] = True
            return fact_entries, True
            
    except Exception as e:
        logger.warning(f"Loop salvage failed with error: {e}")
        
    return [], False


def robust_extract_json(
    raw_text: str,
    turns: list[Turn],
    client: NpuClient,
    messages: list[dict[str, str]],
) -> tuple[list[FactEntry], dict[str, Any]]:
    """Robust extraction pipeline with repair, isolation, retry, and loop detection."""
    from json_repair import repair_json

    diagnostics = {
        "raw_parse_success": False,
        "fence_stripped": False,
        "repaired": False,
        "repair_success": False,
        "loop_detected": False,
        "loop_salvaged": False,
        "truncated": False,
        "retried": False,
        "retry_success": False,
        "success": False,
        "error": None,
        "stage": 0,
    }

    # Stage 1: strict json.loads
    try:
        parsed = json.loads(raw_text.strip())
        fact_entries = validate_and_repair_items(parsed, turns)
        
        # Check loop detector even if it parses cleanly
        if detect_loops(raw_text, parsed_items=fact_entries, is_truncated=False):
            diagnostics["loop_detected"] = True
            salvaged, success = attempt_salvage(raw_text, turns, diagnostics)
            if success:
                diagnostics["stage"] = 1
                return salvaged, diagnostics
            diagnostics["success"] = False
            diagnostics["error"] = "Loop detected in raw clean parse"
            return [], diagnostics

        diagnostics["raw_parse_success"] = True
        diagnostics["success"] = True
        diagnostics["stage"] = 1
        return fact_entries, diagnostics
    except Exception:
        pass

    # First attempt check for loop before repair
    _, is_trunc = clean_and_check_truncation(raw_text)
    diagnostics["truncated"] = is_trunc
    if detect_loops(raw_text, parsed_items=None, is_truncated=is_trunc):
        diagnostics["loop_detected"] = True
        salvaged, success = attempt_salvage(raw_text, turns, diagnostics)
        if success:
            diagnostics["stage"] = 2
            return salvaged, diagnostics
        diagnostics["success"] = False
        diagnostics["error"] = "Loop detected in raw first attempt"
        return [], diagnostics

    # Stage 2 & 3: Fence-strip, balanced isolation, and repair
    first_error = None
    try:
        stripped_text = strip_markdown_fences(raw_text)
        if stripped_text != raw_text.strip():
            diagnostics["fence_stripped"] = True

        isolated_text, is_trunc_isolated = find_balanced_json(stripped_text)
        diagnostics["truncated"] = is_trunc_isolated
        
        isolated_text = coerce_sibling_objects_to_list(isolated_text)

        repaired_text = repair_json(isolated_text)
        diagnostics["repaired"] = True

        parsed = json.loads(repaired_text)
        fact_entries = validate_and_repair_items(parsed, turns)

        # Stage 5: Loop detector
        if detect_loops(raw_text, parsed_items=fact_entries, is_truncated=is_trunc_isolated):
            diagnostics["loop_detected"] = True
            salvaged, success = attempt_salvage(raw_text, turns, diagnostics)
            if success:
                diagnostics["stage"] = 4
                return salvaged, diagnostics
            diagnostics["success"] = False
            diagnostics["error"] = "Loop detected in repaired output"
            return [], diagnostics

        diagnostics["repair_success"] = True
        diagnostics["success"] = True
        diagnostics["stage"] = 4
        return fact_entries, diagnostics
    except Exception as e:
        first_error = str(e)

    # Stage 6: Validator-guided retry (bounded <= 1)
    diagnostics["retried"] = True
    retry_messages = list(messages)
    if raw_text:
        retry_messages.append({"role": "assistant", "content": raw_text})

    retry_message = (
        FACT_EXTRACTION_TRUNCATION_RETRY_MESSAGE
        if diagnostics["truncated"]
        else f"{FACT_EXTRACTION_RETRY_MESSAGE}\nError details: {first_error}"
    )
    retry_messages.append({"role": "user", "content": retry_message})

    retry_max_tokens = max(client.settings.npu_max_tokens * 2, 1024)
    try:
        response_text_retry = client.chat(retry_messages, max_tokens=retry_max_tokens)
        
        # Check loop detector on retry raw output
        _, is_trunc_retry = clean_and_check_truncation(response_text_retry)
        if detect_loops(response_text_retry, parsed_items=None, is_truncated=is_trunc_retry):
            diagnostics["loop_detected"] = True
            salvaged, success = attempt_salvage(response_text_retry, turns, diagnostics)
            if success:
                diagnostics["stage"] = 6
                return salvaged, diagnostics
            diagnostics["success"] = False
            diagnostics["error"] = "Loop detected in retry raw output"
            return [], diagnostics

        # Try strict parse on retry first
        try:
            parsed_retry = json.loads(response_text_retry.strip())
            fact_entries_retry = validate_and_repair_items(parsed_retry, turns)
            diagnostics["retry_success"] = True
            diagnostics["success"] = True
            diagnostics["stage"] = 6
            return fact_entries_retry, diagnostics
        except Exception:
            pass

        # Try fence-strip, balanced isolation, and repair on retry
        stripped_retry = strip_markdown_fences(response_text_retry)
        isolated_retry, is_trunc_isolated_retry = find_balanced_json(stripped_retry)
        isolated_retry = coerce_sibling_objects_to_list(isolated_retry)
        repaired_retry = repair_json(isolated_retry)

        parsed_retry = json.loads(repaired_retry)
        fact_entries_retry = validate_and_repair_items(parsed_retry, turns)

        if detect_loops(response_text_retry, parsed_items=fact_entries_retry, is_truncated=is_trunc_isolated_retry):
            diagnostics["loop_detected"] = True
            salvaged, success = attempt_salvage(response_text_retry, turns, diagnostics)
            if success:
                diagnostics["stage"] = 6
                return salvaged, diagnostics
            diagnostics["success"] = False
            diagnostics["error"] = "Loop detected in repaired retry output"
            return [], diagnostics

        diagnostics["retry_success"] = True
        diagnostics["success"] = True
        diagnostics["stage"] = 6
        return fact_entries_retry, diagnostics
    except Exception as retry_exc:
        diagnostics["success"] = False
        diagnostics["error"] = f"Retry failed: {retry_exc}"
        return [], diagnostics
