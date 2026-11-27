"""Retry + rate limit cho backend LLM (T2.3, SYSTEM_DESIGN 6.3).

`RetryBackend` retry khi gặp 429/5xx/timeout với exponential backoff
1-2-4-8-16 giây, tối đa 5 lần, kèm rate limiter theo `rpm_limit`.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from artrag.errors import BackendError, RetryableBackendError
from artrag.llm.base import LLMBackend, LLMResult


class RateLimiter:
    """Đảm bảo giữa hai request liền nhau có ít nhất ``min_interval_s`` giây."""

    def __init__(
        self,
        min_interval_s: float = 0.0,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.min_interval_s = float(min_interval_s)
        self._sleep = sleep
        self._monotonic = monotonic
        self._last: float | None = None

    @classmethod
    def from_rpm(
        cls,
        rpm_limit: int | float | None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> "RateLimiter | None":
        """Tạo limiter từ `rpm_limit` (request/phút). None nếu không giới hạn."""
        if not rpm_limit or float(rpm_limit) <= 0:
            return None
        return cls(min_interval_s=60.0 / float(rpm_limit), sleep=sleep, monotonic=monotonic)

    def wait(self) -> None:
        if self.min_interval_s <= 0:
            return
        now = self._monotonic()
        if self._last is not None:
            delta = self.min_interval_s - (now - self._last)
            if delta > 0:
                self._sleep(delta)
                now = self._monotonic()
        self._last = now


class RetryBackend:
    """Bọc retry/backoff + rate limit quanh một backend LLM khác."""

    def __init__(
        self,
        inner: LLMBackend,
        max_retries: int = 5,
        backoff_s: float = 1.0,
        rate_limiter: RateLimiter | None = None,
        sleep: Callable[[float], None] = time.sleep,
        name: str | None = None,
    ) -> None:
        self.inner = inner
        self.max_retries = int(max_retries)
        self.backoff_s = float(backoff_s)
        self.rate_limiter = rate_limiter
        self._sleep = sleep
        self.name = name or getattr(inner, "name", "llm")
        #: số lần đã gọi thật (để trace/debug)
        self.attempts = 0

    # -- interface LLMBackend -------------------------------------------
    def count_tokens(self, text: str) -> int:
        return self.inner.count_tokens(text)

    def generate(self, messages: list[dict], **gen: Any) -> LLMResult:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            if self.rate_limiter is not None:
                self.rate_limiter.wait()
            try:
                self.attempts += 1
                return self.inner.generate(messages, **gen)
            except RetryableBackendError as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                delay = self.backoff_s * (2**attempt)
                retry_after = getattr(exc, "retry_after", None)
                if isinstance(retry_after, (int, float)) and retry_after > 0:
                    delay = float(retry_after)
                self._sleep(delay)
            except (TimeoutError, ConnectionError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                self._sleep(self.backoff_s * (2**attempt))

        raise BackendError(
            f"Backend {self.name!r} thất bại sau {self.max_retries + 1} lần thử: {last_error}",
            hint="kiểm tra mạng, hạn mức rate limit và API key (biến môi trường).",
            status_code=getattr(last_error, "status_code", None),
        ) from last_error


__all__ = ["RateLimiter", "RetryBackend"]
