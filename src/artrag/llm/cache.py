"""Cache đĩa cho backend LLM (T2.3, SYSTEM_DESIGN 6.3/6.4).

`CachedBackend` bọc wrap mọi backend và ghi kết quả xuống đĩa bằng `diskcache`.
Khóa cache là `sha256(backend|model|messages|gen)` nên chạy lại eval không tốn
quota API và kết quả tái lập được.

Vì sinh greedy nên kết quả đã cache luôn hợp lệ.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from artrag.llm.base import LLMBackend, LLMResult


def cache_key(
    backend: str,
    model: str,
    messages: list[dict],
    gen: dict[str, Any] | None = None,
) -> str:
    """Khóa cache 16 hex của một lần gọi LLM."""
    payload = json.dumps(
        {
            "backend": backend,
            "model": model,
            "messages": messages,
            "gen": gen or {},
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class CachedBackend:
    """Bọc cache đĩa quanh một backend LLM khác."""

    def __init__(
        self,
        inner: LLMBackend,
        cache_dir: str | Path,
        enabled: bool = True,
        name: str | None = None,
    ) -> None:
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.enabled = bool(enabled)
        self.name = name or getattr(inner, "name", "llm")
        self._cache: Any = None

    # -- số liệu dữ liệu ------------------------------------------------
    @property
    def cache(self) -> Any:
        """Mở (và tạo) cache đĩa; trả None nếu cache bị tắt."""
        if not self.enabled:
            return None
        if self._cache is None:
            import diskcache  # noqa: PLC0415

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache = diskcache.Cache(str(self.cache_dir))
        return self._cache

    def clear(self) -> None:
        """Xóa toàn bộ cache (dùng cho test hoặc khi đổi mô hình)."""
        cache = self.cache
        if cache is not None:
            cache.clear()

    def close(self) -> None:
        if self._cache is not None:
            self._cache.close()
            self._cache = None

    # -- interface LLMBackend -------------------------------------------
    def count_tokens(self, text: str) -> int:
        return self.inner.count_tokens(text)

    def generate(self, messages: list[dict], **gen: Any) -> LLMResult:
        cache = self.cache
        key = cache_key(self.name, self._model_name(), messages, gen)
        if cache is not None:
            started = time.perf_counter()
            hit = cache.get(key)
            if hit is not None:
                result = LLMResult.from_dict(hit)
                result.cached = True
                result.backend = result.backend or self.name
                # latency của lần đọc cache (rất nhỏ), không phải của lần gọi API gốc.
                result.latency_ms = int((time.perf_counter() - started) * 1000)
                return result

        result = self.inner.generate(messages, **gen)
        if cache is not None and result is not None:
            payload = result.to_dict()
            payload["cached"] = False
            payload.pop("total_tokens", None)
            cache.set(key, payload)
        return result

    def _model_name(self) -> str:
        return str(getattr(self.inner, "model", "") or "")


__all__ = ["CachedBackend", "cache_key"]
