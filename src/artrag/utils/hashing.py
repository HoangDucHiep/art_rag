"""Tien ich hashing."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256_json(obj: Any) -> str:
    """Hash 16 hex cua object (sap xep key, on dinh dua tren noi dung)."""
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def sha256_text(text: str) -> str:
    """Hash 16 hex cua chuoi."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Hash 16 hex cua noi dung file (doc theo khoi de khong nham ram)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]
