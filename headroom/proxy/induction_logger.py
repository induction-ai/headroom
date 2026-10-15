"""Induction payload-event logging for this Headroom fork.

For each HTTP POST to a Responses endpoint, the proxy writes one staging
record per upstream request and response payload. The Induction benchmark
harness folds those records into a run's server log. When the client sends
``x-induction-trajectory-id`` (and optionally ``x-induction-group-id``),
each record is stamped with those ids so the harness can attribute payloads
to a trajectory without relying on body hashes alone. The proxy removes
those headers before forwarding the request upstream.
"""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from headroom import fileperms

logger = logging.getLogger("headroom.proxy")


INDUCTION_HEADER_PREFIX = "x-induction-"


def strip_induction_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Return a copy of ``headers`` with every ``x-induction-*`` entry removed."""
    return {
        key: value
        for key, value in headers.items()
        if not key.lower().startswith(INDUCTION_HEADER_PREFIX)
    }


def _uuid7() -> str:
    value = bytearray(secrets.token_bytes(16))
    value[0:6] = int(time.time() * 1000).to_bytes(6, "big")
    value[6] = (value[6] & 0x0F) | 0x70
    value[8] = (value[8] & 0x3F) | 0x80
    return str(uuid.UUID(bytes=bytes(value)))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def client_body_hash(value: Any) -> str:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


@dataclass
class InductionLogContext:
    request_id: str
    client_body_hash: str
    provider: str
    model: str
    module: str
    trajectory_id: str | None = None
    group_id: str | None = None
    _next_call: int = 1
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def next_llm_call_id(self) -> str:
        with self._lock:
            call_number = self._next_call
            self._next_call += 1
        return f"{self.request_id}:{call_number}"


class InductionPayloadLogger:
    """Append-only JSONL writer for upstream request and response payloads."""

    def __init__(self, log_file: str | None):
        self.log_file = Path(log_file).expanduser() if log_file else None
        self._lock = threading.Lock()
        if self.log_file is not None:
            try:
                self.log_file.parent.mkdir(parents=True, exist_ok=True)
            except OSError as error:
                logger.warning(
                    "Induction payload log directory %s is unavailable: %s",
                    self.log_file.parent,
                    error,
                )
                self.log_file = None

    def begin_request(
        self,
        *,
        request_id: str,
        client_body: Any,
        provider: str,
        model: str,
        module: str,
        trajectory_id: str | None = None,
        group_id: str | None = None,
    ) -> InductionLogContext:
        try:
            body_hash = client_body_hash(client_body)
        except (TypeError, ValueError):
            body_hash = "sha256:" + hashlib.sha256(repr(client_body).encode()).hexdigest()
        return InductionLogContext(
            request_id=request_id,
            client_body_hash=body_hash,
            provider=provider,
            model=model,
            module=module,
            trajectory_id=trajectory_id or None,
            group_id=group_id or None,
        )

    def log_to(
        self,
        context: InductionLogContext,
        payload: Any,
        *,
        internal: bool = False,
    ) -> str:
        llm_call_id = context.next_llm_call_id()
        self._log_payload(
            context,
            name="payload_internal_to_llm" if internal else "payload_to_llm",
            llm_call_id=llm_call_id,
            payload=payload,
        )
        return llm_call_id

    def log_from(
        self,
        context: InductionLogContext,
        llm_call_id: str,
        payload: Any,
        *,
        internal: bool = False,
    ) -> None:
        self._log_payload(
            context,
            name="payload_internal_from_llm" if internal else "payload_from_llm",
            llm_call_id=llm_call_id,
            payload=payload,
        )

    def _log_payload(
        self,
        context: InductionLogContext,
        *,
        name: str,
        llm_call_id: str,
        payload: Any,
    ) -> None:
        if self.log_file is None:
            return
        model = context.model
        if isinstance(payload, dict) and isinstance(payload.get("model"), str):
            model = payload["model"]
        record = {
            "level": 20,
            "time": _now(),
            "module": context.module,
            "name": name,
            "event_id": _uuid7(),
            "headroom_request_id": context.request_id,
            "llm_call_id": llm_call_id,
            "client_body_hash": context.client_body_hash,
            **({"trajectory_id": context.trajectory_id} if context.trajectory_id else {}),
            **({"group_id": context.group_id} if context.group_id else {}),
            "provider": context.provider,
            "model": model,
            "payload": payload,
        }
        try:
            line = json.dumps(record, ensure_ascii=False, default=str)
            with self._lock:
                with fileperms.open_owner_only(self.log_file, "a", encoding="utf-8") as file:
                    file.write(line + "\n")
        except Exception:
            logger.debug("Induction payload logging failed", exc_info=True)

    @staticmethod
    def payload_from_bytes(content: bytes, content_type: str | None = None) -> Any:
        text = content.decode("utf-8", errors="replace")
        if (
            content_type and "text/event-stream" in content_type.lower()
        ) or text.lstrip().startswith(("event:", "data:")):
            parsed_sse = InductionPayloadLogger.response_from_sse(text)
            if parsed_sse is not None:
                return parsed_sse
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError):
            return {"raw": text}
        if isinstance(parsed, (dict, list)):
            return parsed
        return {"value": parsed}

    @staticmethod
    def response_from_sse(sse_data: str) -> dict[str, Any] | None:
        terminal: dict[str, Any] | None = None
        fallback: dict[str, Any] | None = None
        for raw_event in sse_data.replace("\r\n", "\n").split("\n\n"):
            data = "\n".join(
                line[5:].lstrip() for line in raw_event.splitlines() if line.startswith("data:")
            )
            if not data or data == "[DONE]":
                continue
            try:
                event = json.loads(data)
            except (TypeError, ValueError):
                continue
            if not isinstance(event, dict):
                continue
            response = event.get("response")
            if isinstance(response, dict):
                fallback = response
                if event.get("type") in {
                    "response.completed",
                    "response.incomplete",
                    "response.failed",
                }:
                    terminal = response
            elif event.get("type") == "error":
                fallback = event
        return terminal or fallback
