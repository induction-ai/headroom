"""ProxyConfig must reject a 0 requests-per-minute limit when rate limiting is on
(it would divide by zero in the token-bucket wait computation and 500 every
request), while leaving it inert when limiting is off. This fork also never
turns rate limiting on, whatever the config asks for."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from headroom.proxy.models import ProxyConfig


def test_zero_rpm_with_limiting_enabled_is_rejected():
    with pytest.raises(ValueError, match="rate_limit_requests_per_minute must be >= 1"):
        ProxyConfig(rate_limit_enabled=True, rate_limit_requests_per_minute=0)


def test_negative_rpm_with_limiting_enabled_is_rejected():
    with pytest.raises(ValueError, match="rate_limit_requests_per_minute must be >= 1"):
        ProxyConfig(rate_limit_enabled=True, rate_limit_requests_per_minute=-5)


def test_zero_rpm_is_inert_when_limiting_disabled():
    # Limiting off -> the bucket is never consulted, so a 0 limit is harmless.
    config = ProxyConfig(rate_limit_enabled=False, rate_limit_requests_per_minute=0)
    assert config.rate_limit_requests_per_minute == 0


def test_valid_rpm_is_accepted():
    config = ProxyConfig(rate_limit_enabled=True, rate_limit_requests_per_minute=60)
    assert config.rate_limit_requests_per_minute == 60


def test_rate_limiting_is_off_by_default():
    assert ProxyConfig().rate_limit_enabled is False


def test_rate_limiting_stays_off_even_when_the_config_turns_it_on():
    config = ProxyConfig(
        rate_limit_enabled=True,
        rate_limit_requests_per_minute=1,
        rate_limit_tokens_per_minute=1,
    )
    assert config.rate_limit_enabled is False


def test_requests_over_the_configured_limits_still_reach_upstream(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from headroom.proxy.server import create_app

    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    # Phase 1: a proxy asked for one request and one token a minute.
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=True,
        rate_limit_requests_per_minute=1,
        rate_limit_tokens_per_minute=1,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
    )

    with TestClient(create_app(config), raise_server_exceptions=False) as client:
        proxy = client.app.state.proxy
        proxy._retry_request = AsyncMock(side_effect=RuntimeError("upstream stub"))

        # Phase 2: three requests in a row, each well past both limits.
        statuses = [
            client.post(
                "/v1/responses",
                headers={"authorization": "Bearer test-key"},
                json={"model": "gpt-4o-mini", "input": "large request", "stream": False},
            ).status_code
            for _ in range(3)
        ]

    assert proxy.rate_limiter is None
    assert 429 not in statuses
    assert proxy._retry_request.await_count == 3
