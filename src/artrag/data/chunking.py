"""T1.3: quy tắc chunking của KB (SYSTEM_DESIGN 2.3).

Mọi hàm ở đây đều thuần: không cần dữ liệu, model hay mạng.

- ``description``/``did_you_know``/``artist_bio``: tách theo câu, gộp tới
  ``sentence_max_words`` từ, overlap ``overlap_sentences`` câu.
- ``provenance``/``exhibitions``: mỗi field một chunk, cắt ở ``field_max_words``.
- ``overview``: đúng một chunk cho mỗi tác phẩm.
- Mọi chunk đều bị hard-cap bởi ``max_words_per_chunk`` và không được rỗng.
"""

from __future__ import annotations

from typing import Any

from artrag.data.chunk_strategies import (
    STRATEGY_SENTENCE,
    apply_strategy,
)
from artrag.data.cma_parse import overview_text
from artrag.utils.text import split_sentences

# field nào tách theo câu (có overlap), field nào giữ nguyên một chunk
SENTENCE_FIELDS = ("description", "did_you_know", "artist_bio")
SINGLE_FIELDS = ("provenance", "exhibitions")

CHUNK_FIELDS = ("overview",) + SENTENCE_FIELDS + SINGLE_FIELDS

CHUNK_COLUMNS = [
    "chunk_id",
    "artwork_id",
    "artist_id",
    "field",
    "text",
    "source_url",
    "n_words",
    "strategy",
]


def count_words(text: str) -> int:
    """Số từ (tách theo khoảng trắng)."""
    if not text:
        return 0
    return len(text.split())


def chunk_sentences(
    text: str,
    max_words: int,
    overlap_sentences: int = 1,
) -> list[str]:
    """Tách văn bản thành các chunk câu với ghép đến ``max_words`` từ.

    Các chunk liên tiếp chia sẻ ``overlap_sentences`` câu cuối với chunk sau.
    Không sinh chunk rỗng; câu đơn lẻ quá dài bị cắt cứng.
    """
    if not text or not text.strip():
        return []
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        sentence_words = count_words(sentence)
        # câu quá dài -> cắt thành nhiều đoạn nhỏ
        if sentence_words > max_words:
            if current:
                chunks.append(" ".join(current))
                current, current_words = [], 0
            words = sentence.split()
            for start in range(0, len(words), max_words):
                piece = " ".join(words[start : start + max_words]).strip()
                if piece:
                    chunks.append(piece)
            continue

        if current_words + sentence_words > max_words and current:
            chunks.append(" ".join(current))
            tail = current[-overlap_sentences:] if overlap_sentences > 0 else []
            current = list(tail)
            current_words = sum(count_words(s) for s in current)

        current.append(sentence)
        current_words += sentence_words

    if current:
        text_chunk = " ".join(current).strip()
        if text_chunk:
            chunks.append(text_chunk)
    return [c for c in chunks if c.strip()]


def chunk_single(text: str, max_words: int) -> list[str]:
    """Cắt văn bản thành một chunk (theo từ), dùng cho provenance/exhibitions."""
    if not text or not text.strip():
        return []
    words = text.split()
    if len(words) <= max_words:
        return [text.strip()]
    return [" ".join(words[start : start + max_words]) for start in range(0, len(words), max_words)]


def _field_value(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if field == "provenance":
        value = row.get("provenance_text")
    elif field == "exhibitions":
        value = row.get("exhibitions_text")
    elif field == "overview":
        return overview_text(row)
    return str(value) if value else ""


def make_chunk_rows(
    row: dict[str, Any],
    sentence_max_words: int = 180,
    field_max_words: int = 250,
    overlap_sentences: int = 1,
    max_words_per_chunk: int = 250,
    strategies: dict[str, str] | None = None,
    item_separator: str = "; ",
    semantic_target_words: int = 120,
    semantic_max_words: int = 250,
    semantic_min_words: int = 40,
) -> list[dict[str, Any]]:
    """Sinh các dòng chunk cho một tác phẩm (theo thứ tự field).

    ``chunk_id`` = ``{artwork_id}#{field}#{i}`` với ``i`` bắt đầu từ 0.

    ``strategies`` cho phép gán chiến lược chunking **riêng cho từng field**
    (xem ``data.chunking.strategy_<field>`` trong config). Field không có
    trong dict thì dùng chiến lược mặc định ``sentence`` (C1 baseline), nhờ
    vậy hành vi hiện tại được giữ nguyên.
    """
    artwork_id = row.get("artwork_id")
    artist_id = row.get("artist_id")
    source_url = row.get("source_url")
    rows: list[dict[str, Any]] = []
    strategies = strategies or {}

    for field in CHUNK_FIELDS:
        text = _field_value(row, field)
        if not text or not text.strip():
            continue

        strategy = strategies.get(field) or STRATEGY_SENTENCE
        if strategy == STRATEGY_SENTENCE:
            # Baseline (C1): giữ đúng hành vi gốc của T1.3.
            # - `overview`: một chunk (cắt cứng theo từ).
            # - field văn xuôi: tách câu, gộp 180 từ, overlap 1 câu.
            # - field danh sách (provenance/exhibitions): cắt cứng theo từ ở
            #   `field_max_words` (KHÔNG cắt theo câu - đây chính là lỗi
            #   "cắt giữa một mục" mà chiến lược C2 `item` sẽ sửa).
            if field == "overview":
                pieces = chunk_single(text, max_words_per_chunk)
            elif field in SENTENCE_FIELDS:
                pieces = chunk_sentences(text, sentence_max_words, overlap_sentences)
            else:
                pieces = chunk_single(text, field_max_words)
        else:
            pieces = apply_strategy(
                text,
                strategy=strategy,
                sentence_max_words=sentence_max_words,
                overlap_sentences=overlap_sentences,
                field_max_words=field_max_words,
                item_separator=item_separator,
                semantic_target_words=semantic_target_words,
                semantic_max_words=semantic_max_words,
                semantic_min_words=semantic_min_words,
            )

        for index, piece in enumerate(pieces):
            piece = piece.strip()
            if not piece:
                continue
            # hard-cap mọi chunk (kể cả overview)
            words = piece.split()
            if len(words) > max_words_per_chunk:
                piece = " ".join(words[:max_words_per_chunk])
            rows.append(
                {
                    "chunk_id": f"{artwork_id}#{field}#{index}",
                    "artwork_id": artwork_id,
                    "artist_id": artist_id,
                    "field": field,
                    "text": piece,
                    "source_url": source_url,
                    "n_words": count_words(piece),
                    "strategy": strategy,
                }
            )
    return rows
