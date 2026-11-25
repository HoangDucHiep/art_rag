"""Tien ich xu ly van ban (HTML, chuan hoa, tach cau)."""

from __future__ import annotations

import html
import re
import unicodedata

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_SENTENCE_RE = re.compile(r"[^.!?…]+[.!?…]*")
_DIACRITIC_RE = re.compile(r"[\u0300-\u036f]")
# "đ" (U+0111) khong decompose duoi NFD, phai thay the tay.
_STROKE_D = {"đ": "d", "Đ": "D"}


def strip_html(text: str) -> str:
    """Bo the HTML, giai thuc the, gop khoang trang, cat dau/cuoi."""
    if not text:
        return ""
    out = _TAG_RE.sub(" ", text)
    out = html.unescape(out)
    out = out.replace("\xa0", " ")
    out = _WS_RE.sub(" ", out)
    return out.strip()


def normalize_text(text: str) -> str:
    """Chuan hoa: bo HTML, lowercase, bo dau tieng Viet, gop khoang trang."""
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
    """Tach cau, giu dau cau o cuoi, bo cau rong."""
    if not text:
        return []
    out: list[str] = []
    for chunk in _SENTENCE_RE.findall(text):
        sent = chunk.strip()
        if sent:
            out.append(sent)
    return out
