"""Tien ich doc/ghi file."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """Tao thu muc (va cha) neu chua ton tai; tra ve Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | os.PathLike[str]) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: str | os.PathLike[str], obj: Any, indent: int = 2) -> Path:
    """Ghi JSON (atomic: ghi file tam roi replace)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=indent, default=str)
    tmp.replace(p)
    return p


def read_jsonl(path: str | os.PathLike[str]) -> list[Any]:
    """Doc file jsonl; dong rong bi bo qua."""
    p = Path(path)
    if not p.exists():
        return []
    rows: list[Any] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[Any]) -> Path:
    """Ghi file jsonl (atomic: ghi file tam roi replace)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    tmp.replace(p)
    return p
