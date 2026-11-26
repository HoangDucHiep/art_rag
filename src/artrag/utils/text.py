"""Tiện ích xử lý văn bản (HTML, chuẩn hóa, tách câu)."""

from __future__ import annotations

import html
import re
import unicodedata

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_SENTENCE_RE = re.compile(r"[^.!?…]+[.!?…]*")
_DIACRITIC_RE = re.compile(r"[\u0300-\u036f]")
# "đ" (U+0111) không decompose dưới NFD, phải thay thế tay.
_STROKE_D = {"đ": "d", "Đ": "D"}


def strip_html(text: str) -> str:
    """Bỏ thẻ HTML, giải thực thể, gộp khoảng trắng, cắt đầu/cuối."""
    if not text:
        return ""
    out = _TAG_RE.sub(" ", text)
    out = html.unescape(out)
    out = out.replace("\xa0", " ")
    out = _WS_RE.sub(" ", out)
    return out.strip()


def normalize_text(text: str) -> str:
    """Chuẩn hóa: bỏ HTML, lowercase, bỏ dấu tiếng Việt, gộp khoảng trắng."""
    if not text:
        return ""
    out = strip_html(text).lower()
    for src, dst in _STROKE_D.items():
        out = out.replace(src.lower(), dst)
    out = unicodedata.normalize("NFD", out)
    out = _DIACRITIC_RE.sub("", out)
    out = _WS_RE.sub(" ", out)
    return out.strip(" .,;:!?")


def split_sentences(text: str) -> list[str]:
    """Tách câu, giữ dấu câu ở cuối, bỏ câu rỗng."""
    if not text:
        return []
    out: list[str] = []
    for chunk in _SENTENCE_RE.findall(text):
        sent = chunk.strip()
        if sent:
            out.append(sent)
    return out
