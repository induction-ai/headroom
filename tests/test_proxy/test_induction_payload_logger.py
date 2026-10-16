"""Tests for the Induction payload-event logger used by this fork."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import types
from unittest.mock import patch

import httpx
import pytest
from click.testing import CliRunner
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient

from headroom.cache.compression_store import get_compression_store, reset_compression_store
from headroom.ccr.tool_injection import CCR_TOOL_NAME
from headroom.cli.main import main
from headroom.proxy.induction_logger import (
    InductionPayloadLogger,
    install_client_response_logging,
    strip_induction_headers,
)
from headroom.proxy.loopback_guard import require_loopback
from headroom.proxy.server import HeadroomProxy, ProxyConfig, create_app
from headroom.proxy.turn_hooks import clear_turn_hooks, register_turn_hook

_UUID7 = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_logger_writes_a_payload_pair_per_upstream_call(tmp_path):
    log_file = tmp_path / "induction.jsonl"
    logger = InductionPayloadLogger(str(log_file))
    context = logger.begin_request(
        request_id="hr_1",
        client_body={"b": "é", "a": 1},
        provider="openai",
        model="gpt-test",
        module="openai_responses",
    )

    main_call_id = logger.log_to(context, {"model": "gpt-test", "input": "compressed"})
    logger.log_from(context, main_call_id, {"id": "resp_1", "output": []})
    resend_call_id = logger.log_to(
        context,
        {"model": "gpt-test", "input": "continuation"},
        kind="ccr",
    )
    logger.log_from(context, resend_call_id, {"id": "resp_2", "output": []})

    records = _records(log_file)
    assert [record["name"] for record in records] == [
        "payload_from_client",
        "payload_to_llm",
        "payload_from_llm",
        "payload_to_llm",
        "payload_from_llm",
    ]
    expected_hash = "sha256:" + hashlib.sha256('{"a":1,"b":"é"}'.encode()).hexdigest()
    assert {record["client_body_hash"] for record in records} == {expected_hash}
    assert records[1]["llm_call_id"] == records[2]["llm_call_id"] == main_call_id
    assert records[3]["llm_call_id"] == records[4]["llm_call_id"] == resend_call_id
    assert main_call_id != resend_call_id
    assert all(record["headroom_request_id"] == "hr_1" for record in records)
    assert all(_UUID7.match(record["event_id"]) for record in records)
    assert records[1]["payload"] == {"model": "gpt-test", "input": "compressed"}
    assert records[2]["payload"] == {"id": "resp_1", "output": []}


def _turns_by_call(records):
    turns = {}
    for record in records:
        if "llm_call_id" in record:
            turns.setdefault(record["llm_call_id"], []).append(
                (record["name"], record["proxy_turn"], record.get("recursion"))
            )
    return turns


def test_logger_numbers_calls_as_proxy_turns_and_names_why_each_resend_happened(tmp_path):
    log_file = tmp_path / "induction.jsonl"
    logger = InductionPayloadLogger(str(log_file))
    context = logger.begin_request(
        request_id="hr_turns",
        client_body={"model": "gpt-test", "input": "original"},
        provider="openai",
        model="gpt-test",
        module="openai_responses",
    )

    # Phase 1: the client's turn, then every kind of re-send Headroom makes,
    # one of them twice.
    calls = [
        logger.log_to(context, {"input": "turn"}),
        logger.log_to(context, {"input": "retrieve"}, kind="ccr"),
        logger.log_to(context, {"input": "retrieve again"}, kind="ccr"),
        logger.log_to(context, {"input": "remember"}, kind="memory"),
        logger.log_to(context, {"input": "re-drive"}, kind="turn_hook"),
    ]
    # Phase 2: the replies come back in the opposite order.
    for call_id in reversed(calls):
        logger.log_from(context, call_id, {"output": []})
    logger.log_to_client(context, {"output": []})

    records = _records(log_file)
    turns = _turns_by_call(records)
    assert [turns[call_id] for call_id in calls] == [
        [("payload_to_llm", 0, None), ("payload_from_llm", 0, None)],
        [("payload_to_llm", 1, "ccr"), ("payload_from_llm", 1, "ccr")],
        [("payload_to_llm", 2, "ccr"), ("payload_from_llm", 2, "ccr")],
        [("payload_to_llm", 3, "memory"), ("payload_from_llm", 3, "memory")],
        [("payload_to_llm", 4, "turn_hook"), ("payload_from_llm", 4, "turn_hook")],
    ]
    client_turn = [record for record in records if record.get("llm_call_id") == calls[0]]
    assert all("recursion" not in record for record in client_turn)
    client_records = [
        record
        for record in records
        if record["name"] in ("payload_from_client", "payload_to_client")
    ]
    assert len(client_records) == 2
    assert all(
        "proxy_turn" not in record and "recursion" not in record for record in client_records
    )


def test_proxy_turns_start_over_for_each_request(tmp_path):
    log_file = tmp_path / "induction.jsonl"
    logger = InductionPayloadLogger(str(log_file))
    first, second = (
        logger.begin_request(
            request_id=request_id,
            client_body={"model": "gpt-test", "input": request_id},
            provider="openai",
            model="gpt-test",
            module="openai_responses",
        )
        for request_id in ("hr_first", "hr_second")
    )

    logger.log_to(first, {"input": "turn"})
    logger.log_to(first, {"input": "retrieve"}, kind="ccr")
    logger.log_to(second, {"input": "turn"})
    logger.log_to(second, {"input": "retrieve"}, kind="ccr")

    records = [record for record in _records(log_file) if "llm_call_id" in record]
    assert [
        (record["headroom_request_id"], record["proxy_turn"], record.get("recursion"))
        for record in records
    ] == [
        ("hr_first", 0, None),
        ("hr_first", 1, "ccr"),
        ("hr_second", 0, None),
        ("hr_second", 1, "ccr"),
    ]


def test_logger_records_what_the_client_sent_and_received_outside_any_llm_call(tmp_path):
    log_file = tmp_path / "induction.jsonl"
    logger = InductionPayloadLogger(str(log_file))
    client_body = {"model": "gpt-client", "input": [{"role": "user", "content": "hi"}]}
    context = logger.begin_request(
        request_id="hr_client",
        client_body=client_body,
        provider="openai",
        model="gpt-client",
        module="openai_responses",
    )

    call_id = logger.log_to(context, {"model": "gpt-client", "input": "compressed"})
    logger.log_from(context, call_id, {"id": "resp_1", "model": "gpt-upstream", "output": []})
    logger.log_to_client(context, {"id": "resp_1", "model": "gpt-upstream", "output": []})

    records = _records(log_file)
    assert [record["name"] for record in records] == [
        "payload_from_client",
        "payload_to_llm",
        "payload_from_llm",
        "payload_to_client",
    ]
    client_in, client_out = records[0], records[3]
    assert client_in["payload"] == client_body
    assert client_out["payload"] == {"id": "resp_1", "model": "gpt-upstream", "output": []}
    assert "llm_call_id" not in client_in
    assert "llm_call_id" not in client_out
    assert client_in["model"] == "gpt-client"
    assert client_out["model"] == "gpt-upstream"
    assert {record["client_body_hash"] for record in records} == {context.client_body_hash}
    assert all(record["headroom_request_id"] == "hr_client" for record in records)
    assert all(_UUID7.match(record["event_id"]) for record in records)


def test_logger_stamps_trajectory_and_group_ids_on_every_record(tmp_path):
    log_file = tmp_path / "induction.jsonl"
    logger = InductionPayloadLogger(str(log_file))
    context = logger.begin_request(
        request_id="hr_traj",
        client_body={"model": "gpt-test", "input": "original"},
        provider="openai",
        model="gpt-test",
        module="openai_responses",
        trajectory_id="traj_1",
        group_id="grp_1",
    )

    main_call_id = logger.log_to(context, {"model": "gpt-test", "input": "compressed"})
    logger.log_from(context, main_call_id, {"id": "resp_1", "output": []})
    resend_call_id = logger.log_to(
        context,
        {"model": "gpt-test", "input": "continuation"},
        kind="ccr",
    )
    logger.log_from(context, resend_call_id, {"id": "resp_2", "output": []})
    logger.log_to_client(context, {"id": "resp_2", "output": []})

    records = _records(log_file)
    assert [record["name"] for record in records] == [
        "payload_from_client",
        "payload_to_llm",
        "payload_from_llm",
        "payload_to_llm",
        "payload_from_llm",
        "payload_to_client",
    ]
    assert all(record["trajectory_id"] == "traj_1" for record in records)
    assert all(record["group_id"] == "grp_1" for record in records)


def test_logger_omits_trajectory_ids_when_the_client_did_not_send_them(tmp_path):
    log_file = tmp_path / "induction.jsonl"
    logger = InductionPayloadLogger(str(log_file))
    context = logger.begin_request(
        request_id="hr_plain",
        client_body={"model": "gpt-test", "input": "original"},
        provider="openai",
        model="gpt-test",
        module="openai_responses",
    )

    call_id = logger.log_to(context, {"model": "gpt-test", "input": "compressed"})
    logger.log_from(context, call_id, {"id": "resp_1", "output": []})
    logger.log_to_client(context, {"id": "resp_1", "output": []})

    records = _records(log_file)
    assert len(records) == 4
    assert all("trajectory_id" not in record for record in records)
    assert all("group_id" not in record for record in records)


def test_strip_induction_headers_removes_only_induction_headers():
    headers = {
        "Authorization": "Bearer test-key",
        "content-type": "application/json",
        "x-headroom-bypass": "true",
        "x-induction-trajectory-id": "traj_1",
        "X-Induction-Group-Id": "grp_1",
        "x-induction-agent": "agent-1",
    }

    stripped = strip_induction_headers(headers)

    assert stripped == {
        "Authorization": "Bearer test-key",
        "content-type": "application/json",
        "x-headroom-bypass": "true",
    }
    assert headers["x-induction-trajectory-id"] == "traj_1"


def test_induction_headers_are_removed_even_when_headroom_headers_pass_through(monkeypatch):
    from headroom.proxy.helpers import _strip_internal_headers

    monkeypatch.setenv("HEADROOM_STRIP_INTERNAL_HEADERS", "disabled")

    stripped = _strip_internal_headers(
        {
            "Authorization": "Bearer test-key",
            "x-headroom-mode": "passthrough",
            "x-induction-trajectory-id": "traj_1",
        }
    )

    # The diagnostic disabled mode keeps x-headroom-* headers, but Induction
    # correlation headers are consumed by the proxy and never go upstream.
    assert stripped == {
        "Authorization": "Bearer test-key",
        "x-headroom-mode": "passthrough",
    }


class _FakeClient:
    def __init__(self, response):
        self._response = response
        self.calls = 0
        self.headers_seen = []

    async def post(self, _url, content=None, headers=None, timeout=None):
        self.calls += 1
        self.headers_seen.append(dict(headers or {}))
        return self._response


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["ccr", "memory", "turn_hook"])
async def test_buffered_resend_is_logged_with_the_reason_headroom_made_it(tmp_path, kind):
    log_file = tmp_path / "induction.jsonl"
    logger = InductionPayloadLogger(str(log_file))
    context = logger.begin_request(
        request_id="hr_2",
        client_body={"model": "gpt-test", "input": "original"},
        provider="openai",
        model="gpt-test",
        module="openai_responses",
    )
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(
        200,
        request=request,
        json={"id": "resp_1", "output": []},
    )
    proxy = HeadroomProxy.__new__(HeadroomProxy)
    proxy.http_client = _FakeClient(response)
    proxy.config = types.SimpleNamespace(
        retry_enabled=True,
        retry_max_attempts=1,
        retry_base_delay_ms=1,
        retry_max_delay_ms=2,
    )
    proxy.induction_logger = logger

    result = await proxy._retry_request(
        "POST",
        "https://api.openai.com/v1/responses",
        {},
        {"model": "gpt-test", "input": "compressed"},
        request_id="hr_2",
        induction_context=context,
        induction_call_kind=kind,
    )

    assert result.status_code == 200
    records = [record for record in _records(log_file) if "llm_call_id" in record]
    assert [record["name"] for record in records] == [
        "payload_to_llm",
        "payload_from_llm",
    ]
    assert records[0]["payload"] == {"model": "gpt-test", "input": "compressed"}
    assert records[1]["payload"] == {"id": "resp_1", "output": []}
    assert records[0]["llm_call_id"] == records[1]["llm_call_id"]
    assert records[0]["recursion"] == records[1]["recursion"] == kind


def test_streamed_response_payload_is_the_completed_response():
    terminal = {
        "type": "response.completed",
        "response": {"id": "resp_1", "output": [], "usage": {"input_tokens": 3}},
    }
    sse = (
        'event: response.created\ndata: {"type":"response.created","response":{"id":"resp_1"}}\n\n'
        f"event: response.completed\ndata: {json.dumps(terminal)}\n\n"
    )

    assert InductionPayloadLogger.response_from_sse(sse) == terminal["response"]


@pytest.fixture
def _reset_store():
    reset_compression_store()
    yield
    reset_compression_store()


class _CcrFakeClient:
    def __init__(self, hash_key):
        self.hash_key = hash_key
        self.bodies = []

    async def aclose(self):
        pass

    async def post(self, url, content=None, headers=None, timeout=None):
        body = json.loads(content)
        self.bodies.append(body)
        request = httpx.Request("POST", url)
        if len(self.bodies) == 1:
            return httpx.Response(
                200,
                request=request,
                json={
                    "id": "resp_1",
                    "object": "response",
                    "status": "completed",
                    "output": [
                        {"type": "reasoning", "id": "rs_1", "summary": []},
                        {
                            "type": "function_call",
                            "id": "fc_1",
                            "call_id": "call_1",
                            "name": CCR_TOOL_NAME,
                            "arguments": json.dumps({"hash": self.hash_key}),
                        },
                    ],
                    "usage": {"input_tokens": 50, "output_tokens": 10},
                },
            )
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "resp_2",
                "object": "response",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Resolved"}],
                    }
                ],
                "usage": {"input_tokens": 60, "output_tokens": 5},
            },
        )


def test_ccr_continuation_is_logged_as_the_next_proxy_turn(tmp_path, _reset_store):
    store = get_compression_store()
    original = json.dumps([{"id": i} for i in range(30)])
    hash_key = store.store(original=original, compressed="[]", original_item_count=30)
    log_file = tmp_path / "induction.jsonl"
    app = create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            induction_log_file=str(log_file),
        )
    )
    app.dependency_overrides[require_loopback] = lambda: None

    with TestClient(app) as client:
        fake = _CcrFakeClient(hash_key)
        app.state.proxy.http_client = fake
        response = client.post(
            "/v1/responses",
            json={
                "model": "gpt-test",
                "input": [{"role": "user", "content": "get the data"}],
                "tools": [
                    {
                        "type": "function",
                        "name": CCR_TOOL_NAME,
                        "description": "Retrieve original content.",
                        "parameters": {
                            "type": "object",
                            "properties": {"hash": {"type": "string"}},
                            "required": ["hash"],
                        },
                    }
                ],
                "stream": False,
            },
            headers={"Authorization": "Bearer test-key"},
        )

    assert response.status_code == 200, response.text
    assert len(fake.bodies) == 2
    outputs = [
        item
        for item in fake.bodies[1]["input"]
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    ]
    assert outputs
    assert json.loads(outputs[0]["output"])["original_content"] == original

    records = _records(log_file)
    assert [record["name"] for record in records] == [
        "payload_from_client",
        "payload_to_llm",
        "payload_from_llm",
        "payload_to_llm",
        "payload_from_llm",
        "payload_to_client",
    ]
    assert records[1]["llm_call_id"] == records[2]["llm_call_id"]
    assert records[3]["llm_call_id"] == records[4]["llm_call_id"]
    assert records[1]["llm_call_id"] != records[3]["llm_call_id"]
    assert records[3]["payload"] == fake.bodies[1]
    assert [(record.get("proxy_turn"), record.get("recursion")) for record in records] == [
        (None, None),
        (0, None),
        (0, None),
        (1, "ccr"),
        (1, "ccr"),
        (None, None),
    ]


_INDUCTION_HEADERS = {
    "x-induction-trajectory-id": "traj_1",
    "x-induction-group-id": "grp_1",
    "x-induction-agent": "agent-1",
}


def _make_induction_app(log_file):
    app = create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            induction_log_file=str(log_file) if log_file is not None else None,
        )
    )
    app.dependency_overrides[require_loopback] = lambda: None
    return app


def _assert_no_induction_headers(headers):
    assert not [key for key in headers if key.lower().startswith("x-induction-")]


def test_responses_request_stamps_trajectory_ids_and_strips_induction_headers(tmp_path):
    log_file = tmp_path / "induction.jsonl"
    app = _make_induction_app(log_file)
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    fake = _FakeClient(
        httpx.Response(
            200,
            request=request,
            json={
                "id": "resp_1",
                "object": "response",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Done"}],
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 2},
            },
        )
    )

    with TestClient(app) as client:
        app.state.proxy.http_client = fake
        response = client.post(
            "/v1/responses",
            json={
                "model": "gpt-test",
                "input": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
            headers={"Authorization": "Bearer test-key", **_INDUCTION_HEADERS},
        )

    assert response.status_code == 200, response.text
    assert len(fake.headers_seen) == 1
    _assert_no_induction_headers(fake.headers_seen[0])
    seen = {key.lower(): value for key, value in fake.headers_seen[0].items()}
    assert seen.get("authorization") == "Bearer test-key"

    records = _records(log_file)
    assert [record["name"] for record in records] == [
        "payload_from_client",
        "payload_to_llm",
        "payload_from_llm",
        "payload_to_client",
    ]
    assert all(record["trajectory_id"] == "traj_1" for record in records)
    assert all(record["group_id"] == "grp_1" for record in records)


class _AsyncSseStream(httpx.AsyncByteStream):
    def __init__(self, data: bytes):
        self._data = data

    async def __aiter__(self):
        yield self._data


class _StreamingFakeClient:
    def __init__(self, sse: bytes, *, status_code=200, content_type="text/event-stream"):
        self._sse = sse
        self._status_code = status_code
        self._content_type = content_type
        self.headers_seen = []

    async def aclose(self):
        pass

    def build_request(self, method, url, content=None, headers=None):
        self.headers_seen.append(dict(headers or {}))
        return httpx.Request(method, url, content=content, headers=headers)

    async def send(self, request, stream=False):
        return httpx.Response(
            self._status_code,
            headers={"content-type": self._content_type},
            stream=_AsyncSseStream(self._sse),
            request=request,
        )


def test_streaming_responses_request_stamps_trajectory_ids_and_strips_induction_headers(tmp_path):
    log_file = tmp_path / "induction.jsonl"
    app = _make_induction_app(log_file)
    terminal = {
        "type": "response.completed",
        "response": {
            "id": "resp_1",
            "object": "response",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Done"}],
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 2},
        },
    }
    sse = (
        'event: response.created\ndata: {"type":"response.created","response":{"id":"resp_1"}}\n\n'
        f"event: response.completed\ndata: {json.dumps(terminal)}\n\n"
    ).encode()
    fake = _StreamingFakeClient(sse)

    with TestClient(app) as client:
        app.state.proxy.http_client = fake
        response = client.post(
            "/v1/responses",
            json={
                "model": "gpt-test",
                "input": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
            headers={"Authorization": "Bearer test-key", **_INDUCTION_HEADERS},
        )

    assert response.status_code == 200, response.text
    assert len(fake.headers_seen) == 1
    _assert_no_induction_headers(fake.headers_seen[0])
    seen = {key.lower(): value for key, value in fake.headers_seen[0].items()}
    assert seen.get("authorization") == "Bearer test-key"

    records = _records(log_file)
    assert [record["name"] for record in records] == [
        "payload_from_client",
        "payload_to_llm",
        "payload_from_llm",
        "payload_to_client",
    ]
    assert all(record["trajectory_id"] == "traj_1" for record in records)
    assert all(record["group_id"] == "grp_1" for record in records)
    assert records[1]["proxy_turn"] == records[2]["proxy_turn"] == 0
    assert "recursion" not in records[1] and "recursion" not in records[2]


_COMPLETED_RESPONSE = {
    "id": "resp_1",
    "object": "response",
    "status": "completed",
    "output": [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Done"}],
        }
    ],
    "usage": {"input_tokens": 10, "output_tokens": 2},
}


def _client_records(path):
    records = _records(path)
    return (
        [record for record in records if record["name"] == "payload_from_client"],
        [record for record in records if record["name"] == "payload_to_client"],
    )


def test_responses_request_logs_what_the_client_sent_and_received(tmp_path):
    log_file = tmp_path / "induction.jsonl"
    app = _make_induction_app(log_file)
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    fake = _FakeClient(httpx.Response(200, request=request, json=_COMPLETED_RESPONSE))
    first_body = {
        "model": "gpt-test",
        "input": [{"role": "user", "content": "hello"}],
        "stream": False,
    }
    second_body = {**first_body, "input": [{"role": "user", "content": "hello again"}]}

    with TestClient(app) as client:
        app.state.proxy.http_client = fake
        first = client.post(
            "/v1/responses",
            json=first_body,
            headers={"Authorization": "Bearer test-key", **_INDUCTION_HEADERS},
        )
        second = client.post(
            "/v1/responses",
            json=second_body,
            headers={"Authorization": "Bearer test-key", **_INDUCTION_HEADERS},
        )

    assert first.status_code == second.status_code == 200
    client_in, client_out = _client_records(log_file)
    assert [record["payload"] for record in client_in] == [first_body, second_body]
    assert [record["payload"] for record in client_out] == [first.json(), second.json()]
    assert client_out[0]["payload"]["usage"] == {"input_tokens": 10, "output_tokens": 2}
    # Each reply is filed under the request it answered.
    for sent, received in zip(client_in, client_out, strict=True):
        assert received["headroom_request_id"] == sent["headroom_request_id"]
        assert received["client_body_hash"] == sent["client_body_hash"]
        assert received["trajectory_id"] == sent["trajectory_id"] == "traj_1"
        assert received["group_id"] == sent["group_id"] == "grp_1"
        assert "llm_call_id" not in sent
        assert "llm_call_id" not in received
    assert client_in[0]["headroom_request_id"] != client_in[1]["headroom_request_id"]


def test_streaming_responses_request_logs_the_completed_response_the_client_received(tmp_path):
    log_file = tmp_path / "induction.jsonl"
    app = _make_induction_app(log_file)
    terminal = {"type": "response.completed", "response": _COMPLETED_RESPONSE}
    fake = _StreamingFakeClient(
        (
            'event: response.created\ndata: {"type":"response.created","response":{"id":"resp_1"}}\n\n'
            f"event: response.completed\ndata: {json.dumps(terminal)}\n\n"
        ).encode()
    )
    sent = {
        "model": "gpt-test",
        "input": [{"role": "user", "content": "hello"}],
        "stream": True,
    }

    with TestClient(app) as client:
        app.state.proxy.http_client = fake
        response = client.post(
            "/v1/responses",
            json=sent,
            headers={"Authorization": "Bearer test-key"},
        )

    assert response.status_code == 200, response.text
    client_in, client_out = _client_records(log_file)
    assert [record["payload"] for record in client_in] == [sent]
    assert [record["payload"] for record in client_out] == [_COMPLETED_RESPONSE]


@pytest.mark.parametrize("stream", [False, True])
def test_upstream_error_is_logged_as_the_reply_the_client_received(tmp_path, stream):
    log_file = tmp_path / "induction.jsonl"
    app = _make_induction_app(log_file)
    error_body = {"error": {"message": "bad input", "type": "invalid_request_error"}}
    if stream:
        fake = _StreamingFakeClient(
            json.dumps(error_body).encode(),
            status_code=400,
            content_type="application/json",
        )
    else:
        request = httpx.Request("POST", "https://api.openai.com/v1/responses")
        fake = _FakeClient(httpx.Response(400, request=request, json=error_body))

    with TestClient(app) as client:
        app.state.proxy.http_client = fake
        response = client.post(
            "/v1/responses",
            json={
                "model": "gpt-test",
                "input": [{"role": "user", "content": "hello"}],
                "stream": stream,
            },
            headers={"Authorization": "Bearer test-key"},
        )

    assert response.status_code == 400
    _, client_out = _client_records(log_file)
    assert [record["payload"] for record in client_out] == [error_body]


def _run_buffered_ccr_turn(tmp_path):
    store = get_compression_store()
    hash_key = store.store(
        original=json.dumps([{"id": i} for i in range(30)]),
        compressed="[]",
        original_item_count=30,
    )
    log_file = tmp_path / "induction.jsonl"
    app = _make_induction_app(log_file)
    fake = _CcrFakeClient(hash_key)
    sent = {
        "model": "gpt-test",
        "input": [{"role": "user", "content": "get the data"}],
        "tools": [
            {
                "type": "function",
                "name": CCR_TOOL_NAME,
                "description": "Retrieve original content.",
                "parameters": {
                    "type": "object",
                    "properties": {"hash": {"type": "string"}},
                    "required": ["hash"],
                },
            }
        ],
        "stream": True,
    }

    with TestClient(app) as client:
        app.state.proxy.http_client = fake
        response = client.post(
            "/v1/responses",
            json=sent,
            headers={"Authorization": "Bearer test-key"},
        )

    assert response.status_code == 200, response.text
    assert len(fake.bodies) == 2
    return sent, _records(log_file)


def test_client_request_is_logged_as_sent_before_the_proxy_rewrites_it(tmp_path, _reset_store):
    sent, records = _run_buffered_ccr_turn(tmp_path)

    assert records[0]["name"] == "payload_from_client"
    assert records[0]["payload"] == sent
    # The proxy asked upstream for a buffered reply so it could resolve the
    # retrieval itself; the client's own request still says stream.
    assert records[1]["name"] == "payload_to_llm"
    assert records[1]["payload"]["stream"] is False


def test_ccr_turn_logs_the_final_reply_the_client_received(tmp_path, _reset_store):
    _, records = _run_buffered_ccr_turn(tmp_path)

    assert [record["name"] for record in records] == [
        "payload_from_client",
        "payload_to_llm",
        "payload_from_llm",
        "payload_to_llm",
        "payload_from_llm",
        "payload_to_client",
    ]
    client_reply = records[-1]["payload"]
    assert client_reply["id"] == "resp_2"
    assert client_reply["usage"] == {"input_tokens": 60, "output_tokens": 5}
    assert client_reply["output"][0]["content"][0]["text"] == "Resolved"


class _RedriveOnce:
    name = "test_redrive"
    stream_safe = False

    def __init__(self):
        self.calls = 0

    def on_request(self, ctx):
        return None

    async def on_response(self, ctx, response, call_model):
        if self.calls:
            return None
        self.calls += 1
        return await call_model(ctx.messages)


@pytest.fixture
def _redrive_once_hook():
    clear_turn_hooks()
    register_turn_hook(_RedriveOnce())
    yield
    clear_turn_hooks()


def test_turn_hook_redrive_is_logged_as_the_next_proxy_turn(tmp_path, _redrive_once_hook):
    log_file = tmp_path / "induction.jsonl"
    app = _make_induction_app(log_file)
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    fake = _FakeClient(httpx.Response(200, request=request, json=_COMPLETED_RESPONSE))

    with TestClient(app) as client:
        app.state.proxy.http_client = fake
        response = client.post(
            "/v1/responses",
            json={
                "model": "gpt-test",
                "input": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
            headers={"Authorization": "Bearer test-key"},
        )

    assert response.status_code == 200, response.text
    assert fake.calls == 2
    records = _records(log_file)
    assert [
        (record["name"], record.get("proxy_turn"), record.get("recursion")) for record in records
    ] == [
        ("payload_from_client", None, None),
        ("payload_to_llm", 0, None),
        ("payload_from_llm", 0, None),
        ("payload_to_llm", 1, "turn_hook"),
        ("payload_from_llm", 1, "turn_hook"),
        ("payload_to_client", None, None),
    ]


def _tapped_app(log_file, respond):
    logger = InductionPayloadLogger(str(log_file))
    app = FastAPI()

    @app.post("/v1/responses")
    async def endpoint(request: Request):
        request.state.induction_context = logger.begin_request(
            request_id="hr_tap",
            client_body=await request.json(),
            provider="openai",
            model="gpt-test",
            module="openai_responses",
        )
        return respond()

    @app.post("/unlogged")
    async def unlogged():
        return respond()

    install_client_response_logging(app, logger)
    return app


def test_request_that_crashes_the_proxy_logs_the_error_reply_the_client_got(tmp_path):
    log_file = tmp_path / "induction.jsonl"

    def crash():
        raise RuntimeError("handler bug")

    with TestClient(_tapped_app(log_file, crash), raise_server_exceptions=False) as client:
        response = client.post("/v1/responses", json={"model": "gpt-test"})

    assert response.status_code == 500
    assert response.text == "Internal Server Error"
    _, client_out = _client_records(log_file)
    assert [record["payload"] for record in client_out] == [{"raw": "Internal Server Error"}]


def test_refused_request_logs_the_error_reply_the_client_got(tmp_path):
    log_file = tmp_path / "induction.jsonl"

    def refuse():
        raise HTTPException(status_code=413, detail={"error": {"type": "compression_refused"}})

    with TestClient(_tapped_app(log_file, refuse)) as client:
        response = client.post("/v1/responses", json={"model": "gpt-test"})

    assert response.status_code == 413
    _, client_out = _client_records(log_file)
    assert [record["payload"] for record in client_out] == [response.json()]


def test_reply_over_the_size_cap_reaches_the_client_and_is_logged_without_its_payload(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("headroom.proxy.induction_logger.MAX_CLIENT_RESPONSE_LOG_BYTES", 64)
    log_file = tmp_path / "induction.jsonl"
    big = {"id": "resp_1", "output": [{"text": "x" * 500}]}

    with TestClient(_tapped_app(log_file, lambda: JSONResponse(big))) as client:
        response = client.post("/v1/responses", json={"model": "gpt-test"})

    assert response.json() == big
    _, client_out = _client_records(log_file)
    assert len(client_out) == 1
    assert client_out[0]["payload"] is None


def test_stream_that_breaks_part_way_logs_what_reached_the_client(tmp_path):
    log_file = tmp_path / "induction.jsonl"
    created = {"type": "response.created", "response": {"id": "resp_1", "status": "in_progress"}}

    async def breaking_stream():
        yield f"event: response.created\ndata: {json.dumps(created)}\n\n".encode()
        raise RuntimeError("upstream went away")

    def respond():
        return StreamingResponse(breaking_stream(), media_type="text/event-stream")

    with TestClient(_tapped_app(log_file, respond)) as client:
        with pytest.raises(RuntimeError, match="upstream went away"):
            client.post("/v1/responses", json={"model": "gpt-test"})

    _, client_out = _client_records(log_file)
    assert [record["payload"] for record in client_out] == [created["response"]]


@pytest.mark.asyncio
async def test_stream_whose_client_disconnects_logs_only_what_was_delivered(tmp_path):
    log_file = tmp_path / "induction.jsonl"
    created = {"type": "response.created", "response": {"id": "resp_1", "status": "in_progress"}}
    completed = {"type": "response.completed", "response": _COMPLETED_RESPONSE}

    async def full_stream():
        yield f"event: response.created\ndata: {json.dumps(created)}\n\n".encode()
        yield f"event: response.completed\ndata: {json.dumps(completed)}\n\n".encode()

    def respond():
        return StreamingResponse(full_stream(), media_type="text/event-stream")

    app = _tapped_app(log_file, respond)
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/responses",
        "raw_path": b"/v1/responses",
        "root_path": "",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 50000),
        "server": ("testserver", 80),
    }

    request_sent = False

    async def receive():
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": b'{"model": "gpt-test"}', "more_body": False}
        # Like a server, wait for a disconnect message that never comes.
        await asyncio.Event().wait()

    async def send(message):
        # The client is gone by the time the completed event goes out.
        if b"response.completed" in message.get("body", b""):
            raise OSError("client disconnected")

    with pytest.raises(OSError, match="client disconnected"):
        await asyncio.wait_for(app(scope, receive, send), timeout=5)

    _, client_out = _client_records(log_file)
    assert [record["payload"] for record in client_out] == [created["response"]]


def test_requests_without_a_log_context_reach_the_client_unlogged(tmp_path):
    log_file = tmp_path / "induction.jsonl"
    reply = {"id": "resp_1", "output": []}

    with TestClient(_tapped_app(log_file, lambda: JSONResponse(reply))) as client:
        response = client.post("/unlogged", json={"model": "gpt-test"})

    assert response.json() == reply
    assert not log_file.exists()


@pytest.fixture
def _isolated_home_and_cwd(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    (tmp_path / "cwd").mkdir()
    monkeypatch.chdir(tmp_path / "cwd")
    monkeypatch.delenv("HEADROOM_INDUCTION_LOG_FILE", raising=False)
    return tmp_path


def _files_with_payload_records(root):
    return [
        path
        for path in root.rglob("*")
        if path.is_file()
        and any(
            name in path.read_bytes()
            for name in (
                b"payload_from_client",
                b"payload_to_llm",
                b"payload_from_llm",
                b"payload_to_client",
            )
        )
    ]


def test_responses_request_without_a_log_file_writes_no_records(_isolated_home_and_cwd):
    app = _make_induction_app(None)
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    fake = _FakeClient(httpx.Response(200, request=request, json=_COMPLETED_RESPONSE))

    with TestClient(app) as client:
        app.state.proxy.http_client = fake
        response = client.post(
            "/v1/responses",
            json={
                "model": "gpt-test",
                "input": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
            headers={"Authorization": "Bearer test-key", **_INDUCTION_HEADERS},
        )

    assert response.status_code == 200, response.text
    assert len(fake.headers_seen) == 1
    _assert_no_induction_headers(fake.headers_seen[0])
    assert _files_with_payload_records(_isolated_home_and_cwd) == []


def test_streaming_responses_request_without_a_log_file_writes_no_records(
    _isolated_home_and_cwd,
):
    app = _make_induction_app(None)
    terminal = {"type": "response.completed", "response": _COMPLETED_RESPONSE}
    fake = _StreamingFakeClient(
        f"event: response.completed\ndata: {json.dumps(terminal)}\n\n".encode()
    )

    with TestClient(app) as client:
        app.state.proxy.http_client = fake
        response = client.post(
            "/v1/responses",
            json={
                "model": "gpt-test",
                "input": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
            headers={"Authorization": "Bearer test-key", **_INDUCTION_HEADERS},
        )

    assert response.status_code == 200, response.text
    assert len(fake.headers_seen) == 1
    _assert_no_induction_headers(fake.headers_seen[0])
    assert _files_with_payload_records(_isolated_home_and_cwd) == []


def _proxy_config_from_cli(args, env=None):
    captured = {}

    def fake_run_server(config, **kwargs):
        captured["config"] = config

    # Keep the developer's own shell settings out of the result.
    base_env = {"HEADROOM_INDUCTION_LOG_FILE": None, "HEADROOM_STATELESS": None}
    with patch("headroom.proxy.server.run_server", fake_run_server):
        result = CliRunner().invoke(
            main,
            ["proxy", *args],
            env={**base_env, **(env or {})},
            catch_exceptions=False,
        )

    assert result.exit_code == 0, result.output
    return captured["config"]


def test_proxy_command_leaves_payload_logging_off_unless_a_log_file_is_set():
    assert _proxy_config_from_cli([]).induction_log_file is None


@pytest.mark.parametrize("source", ["flag", "env"])
def test_proxy_command_logs_payloads_to_the_configured_file(tmp_path, source):
    log_file = str(tmp_path / "payloads.jsonl")
    if source == "flag":
        config = _proxy_config_from_cli(["--induction-log-file", log_file])
    else:
        config = _proxy_config_from_cli([], {"HEADROOM_INDUCTION_LOG_FILE": log_file})

    assert config.induction_log_file == log_file


def test_proxy_command_turns_payload_logging_off_in_stateless_mode(tmp_path):
    config = _proxy_config_from_cli(
        ["--stateless", "--induction-log-file", str(tmp_path / "payloads.jsonl")]
    )

    assert config.induction_log_file is None
