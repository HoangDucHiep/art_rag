"""Backend dự phòng (T2.3, SYSTEM_DESIGN 5.10/6.3).

Khi backend chính lỗi (mất mạng, hết hạn mức API, model lỗi) thì chuyển sang
backend dự phòng khai báo trong `llm.fallback` (ví dụ profile `api`:
groq -> nim).
"""

from __future__ import annotations

from typing import Any, Sequence

from artrag.errors import BackendError
from artrag.llm.base import LLMBackend, LLMResult


class FallbackBackend:
    """Thử lần lượt nhiều backend cho tới khi có backend trả lời thành công."""

    def __init__(self, backends: Sequence[LLMBackend]) -> None:
        self.backends = list(backends)
        if not self.backends:
            raise BackendError(
                "FallbackBackend cần ít nhất một backend.",
                hint="kiểm tra `llm.backend` và `llm.fallback` trong config.",
            )
        self.name = getattr(self.backends[0], "name", "llm")
        #: danh sách lỗi của các backend đã thử (để báo cáo)
        self.failures: list[str] = []

    def count_tokens(self, text: str) -> int:
        return self.backends[0].count_tokens(text)

    def generate(self, messages: list[dict], **gen: Any) -> LLMResult:
        self.failures = []
        last_error: Exception | None = None
        for backend in self.backends:
            name = getattr(backend, "name", "llm")
            try:
                result = backend.generate(messages, **gen)
            except Exception as exc:  # noqa: BLE001 - thử backend tiếp theo
                last_error = exc
                self.failures.append(f"{name}: {type(exc).__name__}: {exc}")
                continue
            if result is not None:
                result.backend = result.backend or name
                return result
            last_error = BackendError(f"Backend {name!r} trả về kết quả rỗng.")
            self.failures.append(f"{name}: kết quả rỗng")

        raise BackendError(
            "Không backend nào trả lời được: " + " | ".join(self.failures),
            hint="kiểm tra mạng, API key và tên model của các backend.",
        ) from last_error


__all__ = ["FallbackBackend"]
