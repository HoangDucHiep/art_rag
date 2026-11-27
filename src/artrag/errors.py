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
    """Lỗi gọi backend LLM.

    `status_code` là mã HTTP của provider (nếu có) để lớp trên quyết định
    có xoay API key hay không.
    """

    def __init__(
        self,
        message: str,
        hint: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message, hint)
        self.status_code = status_code


class RetryableBackendError(BackendError):
    """Lỗi tạm thời khi gọi backend LLM (429, 5xx, timeout).

    `RetryBackend` chỉ retry loại lỗi này; lỗi vĩnh viễn (400, 401 hết key)
    được ném thẳng ra ngoài.

    `retry_after` là số giây provider yêu cầu chờ (header `Retry-After`); nếu có
    giá trị này thì `RetryBackend` dùng nó thay cho backoff mặc định.
    """

    def __init__(
        self,
        message: str,
        hint: str | None = None,
        retry_after: float | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message, hint, status_code=status_code)
        self.retry_after = retry_after


class InvalidInputError(ArtragError):
    """Input người dùng không hợp lệ."""


class ConfigError(ArtragError):
    """Lỗi config: thiếu file, sai cú pháp, ghi đè không hợp lệ."""
