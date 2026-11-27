"""Interface chung cho backend sinh câu trả lời (T2.3, SYSTEM_DESIGN 6.3).

Mọi backend đều trả về một `LLMResult` chuẩn, để pipeline (T5) và eval (T6)
không phải biết backend đang dùng là gì.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

from artrag.errors import BackendError, RetryableBackendError

#: Số token xấp xỉ cho một chuỗi (heuristic ~ 4 ký tự / token).
CHARS_PER_TOKEN = 4


@dataclass
class LLMResult:
    """Kết quả của một lần gọi backend LLM."""

    text: str
    model: str = ""
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: bool = False
    backend: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return int(self.prompt_tokens) + int(self.completion_tokens)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["total_tokens"] = self.total_tokens
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLMResult":
        known = {
            "text",
            "model",
            "latency_ms",
            "prompt_tokens",
            "completion_tokens",
            "cached",
            "backend",
            "raw",
        }
        payload = {k: v for k, v in (data or {}).items() if k in known}
        return cls(**payload)


@runtime_checkable
class LLMBackend(Protocol):
    """Protocol của một backend LLM (SYSTEM_DESIGN 6.3)."""

    name: str

    def generate(self, messages: list[dict], **gen: Any) -> LLMResult:
        ...

    def count_tokens(self, text: str) -> int:
        ...


def estimate_tokens(text: str) -> int:
    """Ước lượng số token của chuỗi (heuristic, không cần tokenizer thật)."""
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def count_message_tokens(messages: list[dict]) -> int:
    """Ước lượng số token của cả danh sách messages."""
    total = 0
    for message in messages or []:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    total += estimate_tokens(str(part.get("text", "")))
                else:
                    total += estimate_tokens(str(part))
    return total


__all__ = [
    "CHARS_PER_TOKEN",
    "LLMBackend",
    "LLMResult",
    "BackendError",
    "RetryableBackendError",
    "count_message_tokens",
    "estimate_tokens",
]
