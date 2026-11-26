"""Ngoại lệ riêng của hệ thống (SYSTEM_DESIGN 6.5)."""

from __future__ import annotations


class ArtragError(Exception):
    """Base cho các lỗi của artrag; mang `hint` để UI hiển thị thông báo thân thiện."""

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        if self.hint:
            return f"{self.message} (gợi ý: {self.hint})"
        return self.message


class DataNotReadyError(ArtragError):
    """Thiếu KB hoặc index; thông báo kèm lệnh cần chạy."""


class StaleIndexError(ArtragError):
    """Index không khớp KB hiện tại (kb_hash lệch)."""


class ModelLoadError(ArtragError):
    """Không nạp được model."""


class BackendError(ArtragError):
    """Lỗi gọi backend LLM."""


class InvalidInputError(ArtragError):
    """Input người dùng không hợp lệ."""


class ConfigError(ArtragError):
    """Lỗi config: thiếu file, sai cú pháp, ghi đè không hợp lệ."""
