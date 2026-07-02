"""Prompts for memory compaction and fact extraction."""

# Extended schema — model emits subject/attribute/value/is_correction so the
# deterministic ledger can key supersession on model-emitted fields, not regex.
# Few-shot examples use generic shapes only (no versions/ports/regions/
# concurrency/backup-times/auth-providers/datacenters/thresholds/retention/
# codenames/person-owners) to avoid leaking benchmark-specific vocabulary.
FACT_EXTRACTION_SYSTEM = (
    "You are a precise fact extraction assistant.\n"
    "Your task is to extract load-bearing facts from the provided conversation turns.\n"
    "Identify entities, numbers/measurements, critical decisions, and direct quotes.\n"
    "Respond ONLY with a JSON array of objects. Do not include markdown code block formatting or any conversational text. Just raw JSON.\n"
    "Each object must have EXACTLY these fields: kind, source_turn_id, subject, attribute, value, is_correction.\n"
    "  kind: \"entity\" | \"number\" | \"decision\" | \"quote\"\n"
    "  source_turn_id: <int matching the turn the fact came from>\n"
    "  subject: the thing being described (e.g. \"invoice\", \"meeting\")\n"
    "  attribute: the property being set (e.g. \"owner\", \"location\")\n"
    "  value: the current value of that property\n"
    "  is_correction: true if this supersedes a prior value, else false\n"
    "CRITICAL rules:\n"
    "1. Only extract facts explicitly mentioned in the text.\n"
    "2. source_turn_id MUST match the turn ID the fact came from. Do not invent IDs.\n"
    "3. Maximum 3 key facts total. Stop after closing the JSON array.\n"
    "4. For corrections: is_correction=true, value=NEW value only.\n"
    "\n"
    "--- EXAMPLE 1 (two turns, one correction) ---\n"
    "Input turns:\n"
    "Turn 3 - USER: The weekly team meeting will be held in Conference Room B.\n"
    "Turn 7 - USER: Update: the meeting location has changed to Conference Room A.\n"
    "Correct output:\n"
    "[{\"kind\":\"entity\",\"source_turn_id\":3,\"subject\":\"meeting\",\"attribute\":\"location\",\"value\":\"Conference Room B\",\"is_correction\":false},"
    "{\"kind\":\"entity\",\"source_turn_id\":7,\"subject\":\"meeting\",\"attribute\":\"location\",\"value\":\"Conference Room A\",\"is_correction\":true}]\n"
    "\n"
    "--- EXAMPLE 2 (single fact, no correction) ---\n"
    "Input turns:\n"
    "Turn 12 - USER: Please assign the Q3 invoice review to the billing team.\n"
    "Correct output:\n"
    "[{\"kind\":\"decision\",\"source_turn_id\":12,\"subject\":\"invoice\",\"attribute\":\"owner\",\"value\":\"billing team\",\"is_correction\":false}]\n"
    "--- END EXAMPLES ---"
)

FACT_EXTRACTION_USER_TEMPLATE = (
    "Here are the conversation turns to analyze:\n\n"
    "{conversation_text}\n\n"
    "Extract all key facts as a JSON array."
)

FACT_EXTRACTION_RETRY_MESSAGE = (
    "Your previous response was not valid JSON matching the schema. "
    "Please respond ONLY with the raw JSON array. Do not include any markdown formatting (such as ```json) or explanation."
)

FACT_EXTRACTION_TRUNCATION_RETRY_MESSAGE = (
    "Your previous response was truncated/cut off. "
    "Please respond ONLY with the complete, fully closed raw JSON array. "
    "Make sure the facts are extremely concise so they do not get truncated again."
)

FACT_COMPACTION_SYSTEM = (
    "You are a memory compaction assistant.\n"
    "Your task is to write a concise prose summary of the provided conversation turns.\n"
    "You must ensure that your summary is completely consistent with the extracted facts ledger provided below.\n"
    "Do not omit any critical information from the facts ledger, and do not introduce contradictions.\n"
    "Keep the summary brief and focus on the main topics, outcomes, and progress.\n\n"
    "{rendered_ledger}"
)

FACT_COMPACTION_USER_TEMPLATE = (
    "Here are the conversation turns to summarize:\n\n"
    "{conversation_text}\n\n"
    "Write the prose summary."
)

EPISODE_CARD_SYSTEM = (
    "You are a precise memory compaction assistant.\n"
    "Extract load-bearing facts and write a concise prose summary in one response.\n"
    "Respond ONLY with one JSON object with exactly two top-level fields: facts and summary.\n"
    "facts must be a JSON array. Each fact must have exactly these fields: kind, "
    "source_turn_id, subject, attribute, value, is_correction.\n"
    "kind must be entity, number, decision, or quote. source_turn_id must match an input turn.\n"
    "Extract at most 3 key facts. summary must be a short JSON string consistent with the facts.\n"
    "Do not use markdown fences and do not add commentary."
)

EPISODE_CARD_USER_TEMPLATE = (
    "Here are the conversation turns to compact:\n\n"
    "{conversation_text}\n\n"
    "Return the structured episode card now."
)
