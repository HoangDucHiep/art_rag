"""Trace pipeline: do thoi gian theo buoc, phan loai step bat buoc / tuy chon (SYSTEM_DESIGN 6.6)."""

from __future__ import annotations

import time
import traceback
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

STATUS_OK = "ok"
STATUS_DEGRADED = "degraded"
STATUS_ERROR = "error"


class Step:
    """Handle cua mot buoc, dung trong `with trace.step(...) as s:` de them thong tin."""

    __slots__ = ("name", "info")

    def __init__(self, name: str) -> None:
        self.name = name
        self.info: dict[str, Any] = {}

    def set_info(self, **kwargs: Any) -> "Step":
        self.info.update(kwargs)
        return self

    def update_info(self, mapping: dict[str, Any]) -> "Step":
        self.info.update(mapping)
        return self


class Trace:
    """Thu thong tin truy xuat cua mot request (xem mau trong §6.6)."""

    def __init__(
        self,
        request_id: str | None = None,
        config_hash: str | None = None,
        profile: str | None = None,
    ) -> None:
        self.request_id = request_id or uuid.uuid4().hex[:8]
        self.config_hash = config_hash
        self.profile = profile
        self.query: dict[str, Any] = {}
        self.steps: list[dict[str, Any]] = []
        self.models: dict[str, Any] = {}
        self.vram_peak_mb: float | None = None
        self.abstain: dict[str, Any] = {"gate": None, "post": None}
        self.invalid_citations: list[str] = []
        self.uncited_sentences: int = 0
        self.raw_llm_answer: str | None = None
        self._start = time.perf_counter()

    # -- buoc ------------------------------------------------------------
    @contextmanager
    def step(self, name: str, required: bool = True) -> Iterator[Step]:
        """Context manager bao quanh mot buoc pipeline.

        - ``required=True`` (mac dinh): loi duoc ghi lai va nem lai.
        - ``required=False``: loi bi ghi ``status="degraded"`` va luong tiep tuc.
        """
        handle = Step(name)
        started = time.perf_counter()
        try:
            yield handle
        except Exception as exc:  # noqa: BLE001 - xu ly chung cho ca hai nhanh
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            info: dict[str, Any] = dict(handle.info)
            info["error"] = f"{type(exc).__name__}: {exc}"[:500]
            info["traceback"] = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )[-2000:]
            self.steps.append(
                {
                    "name": name,
                    "ms": elapsed_ms,
                    "status": STATUS_ERROR if required else STATUS_DEGRADED,
                    "info": info,
                }
            )
            if required:
                raise
            return
        else:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            self.steps.append(
                {
                    "name": name,
                    "ms": elapsed_ms,
                    "status": STATUS_OK,
                    "info": dict(handle.info),
                }
            )

    # -- tong hop --------------------------------------------------------
    @property
    def total_ms(self) -> int:
        return int((time.perf_counter() - self._start) * 1000)

    def to_dict(self) -> dict[str, Any]:
        """Dict JSON-serializable dung cho predictions.jsonl va UI."""
        return {
            "request_id": self.request_id,
            "config_hash": self.config_hash,
            "profile": self.profile,
            "query": dict(self.query),
            "steps": [dict(step) for step in self.steps],
            "models": dict(self.models),
            "vram_peak_mb": self.vram_peak_mb,
            "total_ms": self.total_ms,
            "abstain": dict(self.abstain),
            "invalid_citations": list(self.invalid_citations),
            "uncited_sentences": self.uncited_sentences,
            "raw_llm_answer": self.raw_llm_answer,
        }

    def summary(self) -> str:
        """Bang tom tat: name | ms | status."""
        lines = ["step | ms | status"]
        for step in self.steps:
            lines.append(f"{step['name']} | {step['ms']} | {step['status']}")
        lines.append(f"total | {self.total_ms} |")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Trace(request_id={self.request_id!r}, steps={len(self.steps)})"
