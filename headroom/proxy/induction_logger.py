"""Induction payload-event logging for this Headroom fork.

For each HTTP POST to a Responses endpoint, the proxy writes one staging
record for the client's request, one per upstream request and response
payload, and one for what the client received. The Induction benchmark
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

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response

from headroom import fileperms

logger = logging.getLogger("headroom.proxy")


INDUCTION_HEADER_PREFIX = "x-induction-"
MAX_CLIENT_RESPONSE_LOG_BYTES = 10 * 1024 * 1024


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
        context = InductionLogContext(
            request_id=request_id,
            client_body_hash=body_hash,
            provider=provider,
            model=model,
            module=module,
            trajectory_id=trajectory_id or None,
            group_id=group_id or None,
        )
        # Written now, before any transform edits the body in place.
        self._log_payload(
            context,
            name="payload_from_client",
            llm_call_id=None,
            payload=client_body,
        )
        return context

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

    def log_to_client(self, context: InductionLogContext, payload: Any) -> None:
        self._log_payload(
            context,
            name="payload_to_client",
            llm_call_id=None,
            payload=payload,
        )

    def _log_payload(
        self,
        context: InductionLogContext,
        *,
        name: str,
        llm_call_id: str | None,
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
            **({"llm_call_id": llm_call_id} if llm_call_id is not None else {}),
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


class InductionClientResponseMiddleware:
    """Write ``payload_to_client`` from the bytes the client actually receives.

    The Responses handler has many exits (JSON and SSE replies, re-synthesized
    CCR streams, error bodies, a 413 raised as an exception), so the record is
    taken off the wire instead of at each return. Only requests whose handler
    set ``request.state.induction_context`` are logged.
    """

    def __init__(self, app: Any, *, induction_logger: InductionPayloadLogger) -> None:
        self.app = app
        self.induction_logger = induction_logger

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # The same dict backs the handler's request.state.
        state = scope.setdefault("state", {})
        context: InductionLogContext | None = None
        content_type: str | None = None
        body: bytearray | None = bytearray()
        logged = False

        def log_once() -> None:
            nonlocal logged
            if context is None or logged:
                return
            logged = True
            try:
                payload = (
                    None
                    if body is None
                    else InductionPayloadLogger.payload_from_bytes(bytes(body), content_type)
                )
                self.induction_logger.log_to_client(context, payload)
            except Exception:
                logger.debug("Induction client response logging failed", exc_info=True)

        async def tapped_send(message: Any) -> None:
            nonlocal context, content_type, body
            if message["type"] == "http.response.start":
                # Claimed so the unhandled-error reply cannot log a second
                # record for a reply that has already started.
                found = state.pop("induction_context", None)
                context = found if isinstance(found, InductionLogContext) else None
                for key, value in message.get("headers", []):
                    if key.lower() == b"content-type":
                        content_type = value.decode("latin-1")
            await send(message)
            if message["type"] == "http.response.body" and context is not None:
                if body is not None:
                    body.extend(message.get("body", b""))
                    if len(body) > MAX_CLIENT_RESPONSE_LOG_BYTES:
                        body = None
                if not message.get("more_body", False):
                    log_once()

        try:
            await self.app(scope, receive, tapped_send)
        finally:
            # A stream that broke or lost its client part-way still records
            # what reached the client.
            log_once()


def install_client_response_logging(
    app: Starlette, induction_logger: InductionPayloadLogger
) -> None:
    """Log ``payload_to_client`` for every reply, including unhandled-error 500s.

    Starlette sends the 500 for an unhandled exception from outside every added
    middleware, so the tap never sees it. The installed handler builds that same
    default reply itself and logs it, unless a reply had already started.
    """

    async def reply_to_unhandled_error(request: Request, exc: Exception) -> Response:
        response = PlainTextResponse("Internal Server Error", status_code=500)
        context = getattr(request.state, "induction_context", None)
        if isinstance(context, InductionLogContext):
            induction_logger.log_to_client(
                context,
                InductionPayloadLogger.payload_from_bytes(bytes(response.body), "text/plain"),
            )
        return response

    app.add_middleware(InductionClientResponseMiddleware, induction_logger=induction_logger)
    app.add_exception_handler(Exception, reply_to_unhandled_error)
