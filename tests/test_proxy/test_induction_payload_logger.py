"""Tests for the Induction payload-event logger used by this fork."""

from __future__ import annotations

import hashlib
import json
import re
import types
from unittest.mock import patch

import httpx
import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from headroom.cache.compression_store import get_compression_store, reset_compression_store
from headroom.ccr.tool_injection import CCR_TOOL_NAME
from headroom.cli.main import main
from headroom.proxy.induction_logger import InductionPayloadLogger, strip_induction_headers
from headroom.proxy.loopback_guard import require_loopback
from headroom.proxy.server import HeadroomProxy, ProxyConfig, create_app

_UUID7 = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_logger_writes_main_and_internal_payload_pairs(tmp_path):
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
    internal_call_id = logger.log_to(
        context,
        {"model": "gpt-test", "input": "continuation"},
        internal=True,
    )
    logger.log_from(
        context,
        internal_call_id,
        {"id": "resp_2", "output": []},
        internal=True,
    )

    records = _records(log_file)
    assert [record["name"] for record in records] == [
        "payload_to_llm",
        "payload_from_llm",
        "payload_internal_to_llm",
        "payload_internal_from_llm",
    ]
    expected_hash = "sha256:" + hashlib.sha256('{"a":1,"b":"é"}'.encode()).hexdigest()
    assert {record["client_body_hash"] for record in records} == {expected_hash}
    assert records[0]["llm_call_id"] == records[1]["llm_call_id"] == main_call_id
    assert records[2]["llm_call_id"] == records[3]["llm_call_id"] == internal_call_id
    assert main_call_id != internal_call_id
    assert all(record["headroom_request_id"] == "hr_1" for record in records)
    assert all(_UUID7.match(record["event_id"]) for record in records)
    assert records[0]["payload"] == {"model": "gpt-test", "input": "compressed"}
    assert records[1]["payload"] == {"id": "resp_1", "output": []}


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
    internal_call_id = logger.log_to(
        context,
        {"model": "gpt-test", "input": "continuation"},
        internal=True,
    )
    logger.log_from(context, internal_call_id, {"id": "resp_2", "output": []}, internal=True)

    records = _records(log_file)
    assert [record["name"] for record in records] == [
        "payload_to_llm",
        "payload_from_llm",
        "payload_internal_to_llm",
        "payload_internal_from_llm",
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

    records = _records(log_file)
    assert len(records) == 2
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
async def test_buffered_continuation_call_is_logged_as_internal_payload_pair(tmp_path):
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
        induction_internal=True,
    )

    assert result.status_code == 200
    records = _records(log_file)
    assert [record["name"] for record in records] == [
        "payload_internal_to_llm",
        "payload_internal_from_llm",
    ]
    assert records[0]["payload"] == {"model": "gpt-test", "input": "compressed"}
    assert records[1]["payload"] == {"id": "resp_1", "output": []}
    assert records[0]["llm_call_id"] == records[1]["llm_call_id"]


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


def test_ccr_continuation_is_logged_as_internal_payload_pair(tmp_path, _reset_store):
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
        "payload_to_llm",
        "payload_from_llm",
        "payload_internal_to_llm",
        "payload_internal_from_llm",
    ]
    assert records[0]["llm_call_id"] == records[1]["llm_call_id"]
    assert records[2]["llm_call_id"] == records[3]["llm_call_id"]
    assert records[0]["llm_call_id"] != records[2]["llm_call_id"]
    assert records[2]["payload"] == fake.bodies[1]


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
        "payload_to_llm",
        "payload_from_llm",
    ]
    assert all(record["trajectory_id"] == "traj_1" for record in records)
    assert all(record["group_id"] == "grp_1" for record in records)


class _AsyncSseStream(httpx.AsyncByteStream):
    def __init__(self, data: bytes):
        self._data = data

    async def __aiter__(self):
        yield self._data


class _StreamingFakeClient:
    def __init__(self, sse: bytes):
        self._sse = sse
        self.headers_seen = []

    async def aclose(self):
        pass

    def build_request(self, method, url, content=None, headers=None):
        self.headers_seen.append(dict(headers or {}))
        return httpx.Request(method, url, content=content, headers=headers)

    async def send(self, request, stream=False):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
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
        "payload_to_llm",
        "payload_from_llm",
    ]
    assert all(record["trajectory_id"] == "traj_1" for record in records)
    assert all(record["group_id"] == "grp_1" for record in records)


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
        and any(name in path.read_bytes() for name in (b"payload_to_llm", b"payload_from_llm"))
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
