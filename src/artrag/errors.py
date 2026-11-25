"""Ngoai le rieng cua he thong (SYSTEM_DESIGN 6.5)."""

from __future__ import annotations


class ArtragError(Exception):
    """Base cho cac loi cua artrag; mang `hint` de UI hien thi thong bao than thien."""

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        if self.hint:
            return f"{self.message} (goi y: {self.hint})"
        return self.message


class DataNotReadyError(ArtragError):
    """Thieu KB hoac index; thong bao kem lenh can chay."""


class StaleIndexError(ArtragError):
    """Index khong khop KB hien tai (kb_hash lech)."""


class ModelLoadError(ArtragError):
    """Khong nap duoc model."""


class BackendError(ArtragError):
    """Loi goi backend LLM."""


class InvalidInputError(ArtragError):
    """Input nguoi dung khong hop le."""


class ConfigError(ArtragError):
    """Loi config: thieu file, sai cu phap, ghi de khong hop le."""
