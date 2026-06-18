"""OpenAI-compatible memory sidecar service (Task S1)."""

import json
import logging
import threading
from pathlib import Path
from typing import Callable, Any
import http.server
import httpx
from filelock import FileLock

from rem.config import Settings
from rem.npu_client import NpuClient
from rem.memory.tiers import MemoryState, Turn, count_tokens
from rem.memory.assembler import assemble_messages
from rem.memory.compactor import should_compact, run_background, state_lock_path

logger = logging.getLogger("rem.memory.sidecar")


class MemorySidecar:
    """Core memory sidecar logic with plug-in points.

    Exposes hooks for summarizer model, memory policy, and scheduler.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        summarizer_model: str | None = None,
        memory_policy: Callable[[MemoryState, Settings], bool] | None = None,
        scheduler: Callable[[str, NpuClient, Settings], None] | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.client = NpuClient(self.settings)
        self.summarizer_model = summarizer_model or self.settings.summarizer_model
        self.memory_policy = memory_policy or should_compact
        self.scheduler = scheduler or self._default_scheduler

    def _default_scheduler(self, state_path: str, client: NpuClient, settings: Settings) -> None:
        """Default scheduler: spawns a background thread to run compaction."""
        thread = threading.Thread(
            target=run_background,
            args=(state_path, client, settings),
            daemon=True,
        )
        thread.start()

    def process_chat_request(self, request_data: dict[str, Any]) -> tuple[dict[str, Any], str]:
        """Processes an incoming chat completion request.

        1. Ingests new turns into the session's MemoryState.
        2. Assembles the stability-first context prompt.
        3. Prepares the modified request for the downstream model server.
        Returns the modified request payload and the path to the state file for compaction.
        """
        # Identify session
        session_id = request_data.get("user", "default")
        # Sanitize session_id for file safety
        session_id = "".join(c for c in str(session_id) if c.isalnum() or c in ("-", "_")).strip()
        if not session_id:
            session_id = "default"

        state_path = f"{self.settings.vault_dir}/{session_id}_memory_state.json"

        # Parse messages (pure — no shared state, so done outside the lock)
        messages = request_data.get("messages", [])
        system_content = ""
        incoming_turns = []
        for msg in messages:
            if msg.get("role") == "system":
                system_content = msg.get("content", "")
            else:
                incoming_turns.append(msg)

        # Load -> ingest -> save under the short state lock, so a concurrent
        # background compaction (or another request thread) cannot clobber this
        # write. The lock is never held across the slow NPU compaction.
        with FileLock(state_lock_path(Path(state_path))):
            try:
                state = MemoryState.load(state_path)
            except Exception:
                state = MemoryState()

            new_messages = self._sync_new_turns(state, incoming_turns)
            for msg in new_messages:
                next_turn_id = (state.turns[-1].turn_id + 1) if state.turns else 1
                if not state.turns and state.summaries:
                    all_covered = [t_id for s in state.summaries for t_id in s.covers_turn_ids]
                    if all_covered:
                        next_turn_id = max(all_covered) + 1

                turn = Turn(
                    role=msg["role"],
                    content=msg["content"],
                    turn_id=next_turn_id,
                    tokens=count_tokens(msg["content"]),
                )
                state.turns.append(turn)

            state.save(state_path)

        # Assemble stability-first prompt messages
        assembled_messages = assemble_messages(
            state=state,
            system=system_content,
            task="",
            settings=self.settings,
        )

        # Create modified payload
        modified_payload = dict(request_data)
        modified_payload["messages"] = assembled_messages

        return modified_payload, state_path

    def record_response(self, state_path: str, response_content: str) -> None:
        """Appends the assistant's reply to the conversation state and triggers compaction."""
        # Load -> append -> save under the short state lock (see process_chat_request).
        with FileLock(state_lock_path(Path(state_path))):
            try:
                state = MemoryState.load(state_path)
            except Exception as exc:
                logger.error(f"Failed to load state for recording response: {exc}")
                return

            next_turn_id = (state.turns[-1].turn_id + 1) if state.turns else 1
            if not state.turns and state.summaries:
                all_covered = [t_id for s in state.summaries for t_id in s.covers_turn_ids]
                if all_covered:
                    next_turn_id = max(all_covered) + 1

            assistant_turn = Turn(
                role="assistant",
                content=response_content,
                turn_id=next_turn_id,
                tokens=count_tokens(response_content),
            )
            state.turns.append(assistant_turn)
            state.save(state_path)

        # Trigger compaction via scheduler plug-in if policy says so
        if self.memory_policy(state, self.settings):
            logger.info(f"Triggering compaction for state {state_path}")
            # Override settings to use our configured summarizer model
            compaction_settings = Settings(**self.settings.model_dump())
            compaction_settings.summarizer_model = self.summarizer_model
            self.scheduler(state_path, self.client, compaction_settings)

    def _sync_new_turns(self, state: MemoryState, incoming_turns: list[dict]) -> list[dict]:
        """Aligns incoming messages with existing turns and returns new ones."""
        if not state.turns:
            return incoming_turns

        match_idx = -1
        for i in range(len(incoming_turns) - 1, -1, -1):
            match = True
            for j in range(len(state.turns)):
                incoming_pos = i - j
                state_pos = len(state.turns) - 1 - j
                if incoming_pos < 0:
                    break

                inc_msg = incoming_turns[incoming_pos]
                st_turn = state.turns[state_pos]
                if inc_msg.get("role") != st_turn.role or inc_msg.get("content") != st_turn.content:
                    match = False
                    break
            if match:
                match_idx = i
                break

        if match_idx != -1:
            return incoming_turns[match_idx + 1:]
        return incoming_turns


class SidecarHTTPHandler(http.server.BaseHTTPRequestHandler):
    """HTTP Request Handler for the OpenAI-compatible sidecar proxy."""

    def __init__(self, *args, sidecar: MemorySidecar | None = None, **kwargs) -> None:
        self.sidecar = sidecar or MemorySidecar()
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
        # Route standard logs through logging module instead of stderr print
        logger.info(format % args)

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
            return

        # Read body
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        try:
            request_data = json.loads(body.decode("utf-8"))
        except Exception as exc:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(f"Malformed JSON: {exc}".encode("utf-8"))
            return

        # 1. Process request and get modified payload + state path
        try:
            modified_payload, state_path = self.sidecar.process_chat_request(request_data)
        except Exception as exc:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(f"Context assembly failed: {exc}".encode("utf-8"))
            return

        # 2. Forward request to downstream host LLM server
        downstream_url = f"http://localhost:{self.sidecar.settings.litellm_port}/v1/chat/completions"
        logger.info(f"Forwarding chat completions request to {downstream_url}")
        
        try:
            with httpx.Client(timeout=self.sidecar.settings.npu_request_timeout_s) as client:
                response = client.post(downstream_url, json=modified_payload)
                response.raise_for_status()
                response_data = response.json()
        except Exception as exc:
            self.send_response(502)
            self.end_headers()
            self.wfile.write(f"Downstream connection error: {exc}".encode("utf-8"))
            return

        # 3. Extract assistant's reply and record it
        try:
            assistant_reply = response_data["choices"][0]["message"]["content"]
            self.sidecar.record_response(state_path, assistant_reply)
        except Exception as exc:
            logger.warning(f"Could not extract assistant reply to record turn: {exc}")

        # 4. Return downstream response back to client
        response_bytes = json.dumps(response_data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)


class MemorySidecarServer:
    """Manages the lifecycle of the sidecar HTTP proxy server."""

    def __init__(self, sidecar: MemorySidecar | None = None) -> None:
        self.sidecar = sidecar or MemorySidecar()
        self.settings = self.sidecar.settings
        
        # Helper factory to pass sidecar instance to handler
        def handler_factory(*args, **kwargs):
            return SidecarHTTPHandler(*args, sidecar=self.sidecar, **kwargs)

        self.server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", self.settings.sidecar_port),
            handler_factory,
        )

    def serve_forever(self) -> None:
        logger.info(f"Starting REM Sidecar server on port {self.settings.sidecar_port}")
        self.server.serve_forever()

    def shutdown(self) -> None:
        logger.info("Shutting down REM Sidecar server")
        self.server.shutdown()
        self.server.server_close()
