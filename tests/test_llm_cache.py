"""Test cho backend LLM: cache, retry, xoay key, fallback (T2.3).

Tất cả test dùng transport giả (session giả) — không gọi mạng, không cần API key.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from artrag.config import Config
from artrag.errors import BackendError, RetryableBackendError
from artrag.llm import get_backend
from artrag.llm.base import LLMResult
from artrag.llm.cache import CachedBackend, cache_key
from artrag.llm.fallback import FallbackBackend
from artrag.llm.openai_compat import OpenAICompatBackend, resolve_api_keys
from artrag.llm.retry import RateLimiter, RetryBackend


# ---------------------------------------------------------------------------
# Tiện ích test: session giả
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status_code: int = 200, payload: dict | None = None,
                 text: str = "", headers: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.headers = headers or {}

    def json(self) -> dict:
        if self._payload is None:
            raise ValueError("không phải JSON")
        return self._payload


class FakeSession:
    """Session giả trả về lần lượt các response đã định nghĩa."""

    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append(
            {"url": url, "body": json, "headers": headers or {}, "timeout": timeout}
        )
        if not self.responses:
            raise AssertionError("session giả đã hết response")
        return self.responses.pop(0)

    @property
    def call_count(self) -> int:
        return len(self.calls)


def ok_payload(text: str = "Xin chào", model: str = "test-model") -> dict:
    return {
        "id": "chatcmpl-1",
        "model": model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text},
             "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


def make_backend(session, **kwargs) -> OpenAICompatBackend:
    defaults = dict(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        model="test-model",
        api_keys=["key-1"],
        timeout_s=5.0,
    )
    defaults.update(kwargs)
    return OpenAICompatBackend(session=session, **defaults)


# ---------------------------------------------------------------------------
# (1)(2) Cache
# ---------------------------------------------------------------------------


def test_cache_hit_avoids_second_http_call(tmp_path: Path):
    session = FakeSession([FakeResponse(200, ok_payload("Câu trả lời"))])
    backend = CachedBackend(make_backend(session), tmp_path / "llm")
    messages = [{"role": "user", "content": "Xin chào"}]

    first = backend.generate(messages)
    second = backend.generate(messages)

    assert session.call_count == 1
    assert first.cached is False
    assert second.cached is True
    assert second.text == first.text == "Câu trả lời"
    assert second.model == "test-model"
    assert second.prompt_tokens == 11
    assert second.completion_tokens == 7


def test_cache_miss_when_gen_params_change(tmp_path: Path):
    session = FakeSession(
        [FakeResponse(200, ok_payload("A")), FakeResponse(200, ok_payload("B"))]
    )
    backend = CachedBackend(make_backend(session), tmp_path / "llm")
    messages = [{"role": "user", "content": "Xin chào"}]

    first = backend.generate(messages, max_new_tokens=64)
    second = backend.generate(messages, max_new_tokens=128)

    assert session.call_count == 2
    assert first.cached is False and second.cached is False
    assert first.text == "A" and second.text == "B"


def test_cache_key_depends_on_backend_model_messages_and_gen():
    base = cache_key("groq", "m", [{"role": "user", "content": "x"}], {"max_new_tokens": 1})
    assert base != cache_key("nim", "m", [{"role": "user", "content": "x"}], {"max_new_tokens": 1})
    assert base != cache_key("groq", "m2", [{"role": "user", "content": "x"}], {"max_new_tokens": 1})
    assert base != cache_key("groq", "m", [{"role": "user", "content": "y"}], {"max_new_tokens": 1})
    assert base != cache_key("groq", "m", [{"role": "user", "content": "x"}], {"max_new_tokens": 2})
    assert base == cache_key("groq", "m", [{"role": "user", "content": "x"}], {"max_new_tokens": 1})


def test_cached_backend_can_be_disabled(tmp_path: Path):
    session = FakeSession(
        [FakeResponse(200, ok_payload("A")), FakeResponse(200, ok_payload("B"))]
    )
    backend = CachedBackend(make_backend(session), tmp_path / "llm", enabled=False)
    messages = [{"role": "user", "content": "Xin chào"}]

    backend.generate(messages)
    backend.generate(messages)

    assert session.call_count == 2


# ---------------------------------------------------------------------------
# (3)(4) Retry
# ---------------------------------------------------------------------------


def test_retry_on_429_then_success():
    session = FakeSession(
        [FakeResponse(429, {"error": {"message": "rate limit"}}), FakeResponse(200, ok_payload())]
    )
    slept: list[float] = []
    backend = RetryBackend(
        make_backend(session), max_retries=5, backoff_s=1.0, sleep=slept.append
    )

    result = backend.generate([{"role": "user", "content": "Xin chào"}])

    assert session.call_count == 2
    assert result.text == "Xin chào"
    assert slept == [1.0]


def test_retry_exhausted_raises_backend_error():
    session = FakeSession([FakeResponse(429, {"error": {"message": "rate limit"}}) for _ in range(6)])
    backend = RetryBackend(
        make_backend(session), max_retries=2, backoff_s=0.0, sleep=lambda _s: None
    )

    with pytest.raises(BackendError) as excinfo:
        backend.generate([{"role": "user", "content": "Xin chào"}])

    assert not isinstance(excinfo.value, RetryableBackendError)
    assert session.call_count == 3  # 1 lần đầu + 2 retry
    # mã HTTP của provider phải được giữ lại để lớp trên đọc được
    assert excinfo.value.status_code == 429


def test_non_retryable_error_is_not_retried():
    session = FakeSession([FakeResponse(400, {"error": {"message": "bad request"}})])
    backend = RetryBackend(
        make_backend(session), max_retries=5, backoff_s=0.0, sleep=lambda _s: None
    )

    with pytest.raises(BackendError):
        backend.generate([{"role": "user", "content": "Xin chào"}])

    assert session.call_count == 1


def test_retry_respects_retry_after_header():
    session = FakeSession(
        [
            FakeResponse(429, {"error": {"message": "slow down"}}, headers={"Retry-After": "7"}),
            FakeResponse(200, ok_payload()),
        ]
    )
    slept: list[float] = []
    backend = RetryBackend(
        make_backend(session), max_retries=3, backoff_s=1.0, sleep=slept.append
    )

    backend.generate([{"role": "user", "content": "Xin chào"}])

    assert slept == [7.0]


# ---------------------------------------------------------------------------
# (5) Xoay API key
# ---------------------------------------------------------------------------


def test_rotates_to_second_key_on_429():
    session = FakeSession(
        [FakeResponse(429, {"error": {"message": "rate limit"}}), FakeResponse(200, ok_payload())]
    )
    backend = make_backend(session, api_keys=["key-1", "key-2"])

    result = backend.generate([{"role": "user", "content": "Xin chào"}])

    assert session.call_count == 2
    assert session.calls[0]["headers"]["Authorization"] == "Bearer key-1"
    assert session.calls[1]["headers"]["Authorization"] == "Bearer key-2"
    assert result.text == "Xin chào"


def test_rotates_to_second_key_on_401():
    session = FakeSession(
        [FakeResponse(401, {"error": {"message": "invalid key"}}), FakeResponse(200, ok_payload())]
    )
    backend = make_backend(session, api_keys=["key-1", "key-2"])

    backend.generate([{"role": "user", "content": "Xin chào"}])

    assert session.calls[1]["headers"]["Authorization"] == "Bearer key-2"


def test_all_keys_exhausted_raises_backend_error():
    session = FakeSession(
        [FakeResponse(429, {"error": {"message": "rate limit"}}) for _ in range(4)]
    )
    backend = make_backend(session, api_keys=["key-1", "key-2"])

    with pytest.raises(BackendError):
        backend.generate([{"role": "user", "content": "Xin chào"}])

    assert session.call_count == 2  # mỗi key một lần


def test_resolve_api_keys_skips_missing(monkeypatch):
    monkeypatch.setenv("ART_TEST_KEY_A", "value-a")
    monkeypatch.delenv("ART_TEST_KEY_B", raising=False)

    keys = resolve_api_keys(
        {"api_key_envs": ["ART_TEST_KEY_A", "ART_TEST_KEY_B"], "api_key_env": "ART_TEST_KEY_A"}
    )

    assert keys == ["value-a"]


# ---------------------------------------------------------------------------
# (6) Fallback
# ---------------------------------------------------------------------------


class FailingBackend:
    name = "primary"

    def count_tokens(self, text: str) -> int:
        return 1

    def generate(self, messages, **gen):
        raise BackendError("hỏng rồi")


class WorkingBackend:
    name = "backup"

    def count_tokens(self, text: str) -> int:
        return 1

    def generate(self, messages, **gen):
        return LLMResult(text="dự phòng OK", model="m", backend=self.name)


def test_fallback_uses_backup_when_primary_fails():
    backend = FallbackBackend([FailingBackend(), WorkingBackend()])

    result = backend.generate([{"role": "user", "content": "Xin chào"}])

    assert result.text == "dự phòng OK"
    assert result.backend == "backup"
    assert backend.failures and "primary" in backend.failures[0]


def test_fallback_raises_when_all_fail():
    backend = FallbackBackend([FailingBackend(), FailingBackend()])

    with pytest.raises(BackendError):
        backend.generate([{"role": "user", "content": "Xin chào"}])


# ---------------------------------------------------------------------------
# (7) Rate limiter
# ---------------------------------------------------------------------------


def test_rate_limiter_waits_between_calls():
    now = {"t": 0.0}
    slept: list[float] = []

    def fake_sleep(seconds: float) -> None:
        slept.append(seconds)
        now["t"] += seconds

    limiter = RateLimiter(min_interval_s=2.0, sleep=fake_sleep, monotonic=lambda: now["t"])

    limiter.wait()  # lần đầu: không chờ
    limiter.wait()  # lần hai: chờ 2 giây

    assert slept == [2.0]


def test_rate_limiter_from_rpm():
    limiter = RateLimiter.from_rpm(30)
    assert limiter is not None
    assert limiter.min_interval_s == pytest.approx(2.0)
    assert RateLimiter.from_rpm(0) is None
    assert RateLimiter.from_rpm(None) is None


def test_retry_backend_applies_rate_limiter():
    session = FakeSession([FakeResponse(200, ok_payload())])
    now = {"t": 0.0}
    slept: list[float] = []

    def fake_sleep(seconds: float) -> None:
        slept.append(seconds)
        now["t"] += seconds

    limiter = RateLimiter(min_interval_s=1.0, sleep=fake_sleep, monotonic=lambda: now["t"])
    backend = RetryBackend(
        make_backend(session), max_retries=1, backoff_s=0.0, rate_limiter=limiter,
        sleep=fake_sleep,
    )

    backend.generate([{"role": "user", "content": "Xin chào"}])

    assert session.call_count == 1


# ---------------------------------------------------------------------------
# (8)(9) count_tokens + get_backend
# ---------------------------------------------------------------------------


def test_count_tokens_returns_positive_int():
    backend = make_backend(FakeSession([]))

    assert backend.count_tokens("Xin chào") > 0
    assert isinstance(backend.count_tokens("Xin chào"), int)


def test_get_backend_without_api_key_raises_with_hint(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("ART_TEST_MISSING_KEY", raising=False)
    cfg = Config(
        {
            "llm": {"backend": "groq", "gen": {}, "cache_dir": "llm"},
            "backends": {
                "groq": {
                    "type": "openai_compat",
                    "base_url": "https://example.invalid/v1",
                    "model": "m",
                    "api_key_envs": ["ART_TEST_MISSING_KEY"],
                }
            },
            "paths": {"data_dir": str(tmp_path), "results_dir": str(tmp_path / "results")},
        }
    )

    with pytest.raises(BackendError) as excinfo:
        get_backend(cfg)

    assert excinfo.value.hint
    assert "ART_TEST_MISSING_KEY" in str(excinfo.value)


def test_get_backend_wraps_with_cache_and_retry(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ART_TEST_KEY_OK", "secret")
    cfg = Config(
        {
            "llm": {"backend": "groq", "gen": {}, "cache_dir": "llm", "max_retries": 3},
            "backends": {
                "groq": {
                    "type": "openai_compat",
                    "base_url": "https://example.invalid/v1",
                    "model": "m",
                    "api_key_envs": ["ART_TEST_KEY_OK"],
                }
            },
            "paths": {"data_dir": str(tmp_path), "results_dir": str(tmp_path / "results")},
        }
    )

    backend = get_backend(cfg)

    assert isinstance(backend, RetryBackend)
    assert isinstance(backend.inner, CachedBackend)
    assert backend.max_retries == 3
    assert backend.inner.cache_dir == tmp_path / "cache" / "llm"


def test_get_backend_builds_fallback_chain(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ART_TEST_KEY_OK", "secret")
    cfg = Config(
        {
            "llm": {
                "backend": "groq",
                "fallback": "nim",
                "gen": {},
                "cache_dir": "llm",
            },
            "backends": {
                "groq": {
                    "type": "openai_compat",
                    "base_url": "https://a.invalid/v1",
                    "model": "m1",
                    "api_key_envs": ["ART_TEST_KEY_OK"],
                },
                "nim": {
                    "type": "openai_compat",
                    "base_url": "https://b.invalid/v1",
                    "model": "m2",
                    "api_key_envs": ["ART_TEST_KEY_OK"],
                },
            },
            "paths": {"data_dir": str(tmp_path), "results_dir": str(tmp_path / "results")},
        }
    )

    backend = get_backend(cfg)

    assert isinstance(backend, FallbackBackend)
    assert len(backend.backends) == 2


def test_get_backend_rejects_unknown_backend(tmp_path: Path):
    cfg = Config(
        {
            "llm": {"backend": "khong_ton_tai", "gen": {}, "cache_dir": "llm"},
            "backends": {},
            "paths": {"data_dir": str(tmp_path), "results_dir": str(tmp_path / "results")},
        }
    )

    with pytest.raises(BackendError):
        get_backend(cfg)


def test_request_body_shape():
    session = FakeSession([FakeResponse(200, ok_payload())])
    backend = make_backend(session)

    backend.generate(
        [{"role": "user", "content": "Xin chào"}], max_new_tokens=32, temperature=0.0
    )

    call = session.calls[0]
    assert call["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert call["body"]["model"] == "test-model"
    assert call["body"]["messages"] == [{"role": "user", "content": "Xin chào"}]
    assert call["body"]["max_tokens"] == 32
    assert call["body"]["temperature"] == 0.0
    assert json.dumps(call["body"])  # body phải serialize được
