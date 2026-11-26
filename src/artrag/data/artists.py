"""T1.5: dựng ``artists.parquet`` + chunk tiểu sử họa sĩ (SYSTEM_DESIGN 2.3).

Mọi hàm ở đây đều thuần hoặc chỉ đọc file đã có sẵn trên đĩa — **không gọi
mạng**:

- ``collect_artists``   : lấy họa sĩ duy nhất từ các dòng tác phẩm (dedup theo
  ``artist_id``, KHÔNG theo tên — 3 cặp trùng tên là 3 người khác nhau).
- ``load_wiki_records`` : đọc cache ``raw/wiki/{artist_id}.json`` của T1.4.
- ``build_artist_rows`` : ghép thông tin họa sĩ, chọn nội dung tiểu sử theo
  thứ tự ưu tiên ``cma_bio`` -> ``wiki_summary`` -> không có.
- ``make_artist_bio_chunks`` : tách tiểu sử thành chunk ``artist_bio``
  (``artwork_id`` null, có ``artist_id``).

Lệnh (qua ``build_kb``):
    uv run python -m artrag.data.build_kb --with-bio --check
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from artrag.data.chunk_strategies import (
    STRATEGY_SENTENCE,
    apply_strategy,
)
from artrag.data.chunking import CHUNK_COLUMNS, count_words

# nguồn của nội dung tiểu sử
BIO_SOURCE_CMA = "cma"
BIO_SOURCE_WIKIPEDIA = "wikipedia"
BIO_SOURCE_NONE = "none"
BIO_SOURCES = (BIO_SOURCE_CMA, BIO_SOURCE_WIKIPEDIA, BIO_SOURCE_NONE)

# ngưỡng mặc định: bio quá ngắn thì coi như không có
DEFAULT_MIN_BIO_CHARS = 40

ARTIST_COLUMNS = [
    "artist_id",
    "name",
    "birth",
    "death",
    "nationality",
    "cma_bio",
    "wikidata_qid",
    "wiki_title",
    "wiki_url",
    "wiki_summary",
    "bio",
    "bio_source",
]


# ---------------------------------------------------------------- helpers
def _clean(value: Any) -> str | None:
    """Chuỗi đã cắt khoảng trắng; None nếu rỗng."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if isinstance(value, float) and value != value:  # NaN
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- inputs
def collect_artists(artwork_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Họa sĩ duy nhất trong KB, dedup theo ``artist_id``.

    Dòng tác phẩm không có ``artist_id`` bị bỏ qua. Với cùng một
    ``artist_id``, giữ lần xuất hiện đầu tiên (thứ tự ``artworks.parquet``).
    """
    artists: dict[str, dict[str, Any]] = {}
    for row in artwork_rows:
        artist_id = _clean(row.get("artist_id"))
        if not artist_id:
            continue
        if artist_id in artists:
            continue
        artists[artist_id] = {
            "artist_id": artist_id,
            "name": _clean(row.get("artist_name")),
            "birth": _as_int(row.get("artist_birth")),
            "death": _as_int(row.get("artist_death")),
            "nationality": _clean(row.get("artist_nationality")),
        }
    return list(artists.values())


def load_wiki_records(raw_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Đọc cache ``raw/wiki/{artist_id}.json`` -> map ``artist_id -> record``.

    File hỏng (JSON lỗi) bị bỏ qua, không làm hỏng cả lô.
    """
    wiki_dir = Path(raw_dir) / "wiki"
    records: dict[str, dict[str, Any]] = {}
    if not wiki_dir.is_dir():
        return records
    for path in sorted(wiki_dir.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        artist_id = _clean(payload.get("artist_id"))
        if not artist_id:
            continue
        records[artist_id] = payload
    return records


# ---------------------------------------------------------------- rows
def _pick_bio(
    cma_bio: str | None,
    wiki_summary: str | None,
    prefer_cma: bool,
    min_chars: int,
) -> tuple[str | None, str]:
    """Chọn nội dung tiểu sử + nhãn nguồn.

    Ưu tiên ``cma_bio`` (tiểu sử do chính bảo tàng viết) rồi tới
    ``wiki_summary``. Bio quá ngắn (< ``min_chars`` ký tự) bị bỏ qua và chuyển
    sang nguồn kế tiếp.
    """
    candidates: list[tuple[str, str | None]] = []
    if prefer_cma:
        candidates.append((BIO_SOURCE_CMA, cma_bio))
        candidates.append((BIO_SOURCE_WIKIPEDIA, wiki_summary))
    else:
        candidates.append((BIO_SOURCE_WIKIPEDIA, wiki_summary))
        candidates.append((BIO_SOURCE_CMA, cma_bio))

    for source, text in candidates:
        cleaned = _clean(text)
        if cleaned and len(cleaned) >= min_chars:
            return cleaned, source
    return None, BIO_SOURCE_NONE


def build_artist_rows(
    artwork_rows: Iterable[dict[str, Any]],
    wiki_records: dict[str, dict[str, Any]] | None = None,
    cma_bios: dict[str, str] | None = None,
    prefer_cma: bool = True,
    min_chars: int = DEFAULT_MIN_BIO_CHARS,
) -> list[dict[str, Any]]:
    """Ghép dòng ``artists.parquet`` cho từng họa sĩ.

    ``cma_bios`` là map ``artist_id -> creators[].biography`` (đã strip HTML).
    ``wiki_records`` là map ``artist_id -> record`` của T1.4.
    """
    wiki_records = wiki_records or {}
    cma_bios = cma_bios or {}

    rows: list[dict[str, Any]] = []
    for artist in collect_artists(artwork_rows):
        artist_id = artist["artist_id"]
        record = wiki_records.get(artist_id) or {}

        cma_bio = _clean(cma_bios.get(artist_id)) or _clean(record.get("cma_bio"))
        wiki_summary = _clean(record.get("wiki_summary"))
        bio, bio_source = _pick_bio(cma_bio, wiki_summary, prefer_cma, min_chars)

        rows.append(
            {
                "artist_id": artist_id,
                "name": artist.get("name"),
                "birth": artist.get("birth"),
                "death": artist.get("death"),
                "nationality": artist.get("nationality"),
                "cma_bio": cma_bio,
                "wikidata_qid": _clean(record.get("wikidata_qid")),
                "wiki_title": _clean(record.get("wiki_title")),
                "wiki_url": _clean(record.get("wiki_url")),
                "wiki_summary": wiki_summary,
                "bio": bio,
                "bio_source": bio_source,
            }
        )
    return rows


# ---------------------------------------------------------------- chunks
def make_artist_bio_chunks(
    artist_rows: Iterable[dict[str, Any]],
    source_url_by_artist: dict[str, str] | None = None,
    sentence_max_words: int = 180,
    overlap_sentences: int = 1,
    max_words_per_chunk: int = 250,
    strategy: str = STRATEGY_SENTENCE,
    item_separator: str = "; ",
    semantic_target_words: int = 120,
    semantic_max_words: int = 250,
    semantic_min_words: int = 40,
) -> list[dict[str, Any]]:
    """Tách tiểu sử họa sĩ thành các chunk ``artist_bio``.

    ``chunk_id`` = ``{artist_id}#artist_bio#{i}``; ``artwork_id`` luôn là
    ``None`` (chunk không gắn với tác phẩm nào), ``artist_id`` bắt buộc có.

    ``source_url`` lấy theo nguồn của tiểu sử: nếu nguồn là Wikipedia thì dùng
    URL bài viết, ngược lại dùng URL tác phẩm CMA đầu tiên của họa sĩ đó.

    ``strategy`` chọn chiến lược chunking (``sentence`` / ``item`` /
    ``semantic``) - xem ``artrag.data.chunk_strategies``.
    """
    source_url_by_artist = source_url_by_artist or {}
    rows: list[dict[str, Any]] = []

    for artist in artist_rows:
        artist_id = artist.get("artist_id")
        bio = _clean(artist.get("bio"))
        if not artist_id or not bio:
            continue

        if artist.get("bio_source") == BIO_SOURCE_WIKIPEDIA:
            source_url = _clean(artist.get("wiki_url"))
        else:
            source_url = None
        if not source_url:
            source_url = _clean(source_url_by_artist.get(artist_id))

        pieces = apply_strategy(
            bio,
            strategy=strategy,
            sentence_max_words=sentence_max_words,
            overlap_sentences=overlap_sentences,
            item_separator=item_separator,
            semantic_target_words=semantic_target_words,
            semantic_max_words=semantic_max_words,
            semantic_min_words=semantic_min_words,
        )
        for index, piece in enumerate(pieces):
            piece = piece.strip()
            if not piece:
                continue
            words = piece.split()
            if len(words) > max_words_per_chunk:
                piece = " ".join(words[:max_words_per_chunk])
            rows.append(
                {
                    "chunk_id": f"{artist_id}#artist_bio#{index}",
                    "artwork_id": None,
                    "artist_id": artist_id,
                    "field": "artist_bio",
                    "text": piece,
                    "source_url": source_url,
                    "n_words": count_words(piece),
                    "strategy": strategy,
                }
            )
    return rows


def chunk_rows_for_artists(
    artist_rows: Iterable[dict[str, Any]],
    source_url_by_artist: dict[str, str] | None = None,
    sentence_max_words: int = 180,
    overlap_sentences: int = 1,
    max_words_per_chunk: int = 250,
    strategy: str = STRATEGY_SENTENCE,
    item_separator: str = "; ",
    semantic_target_words: int = 120,
    semantic_max_words: int = 250,
    semantic_min_words: int = 40,
) -> list[dict[str, Any]]:
    """Wrapper có đúng thứ tự cột ``CHUNK_COLUMNS``."""
    rows = make_artist_bio_chunks(
        artist_rows,
        source_url_by_artist=source_url_by_artist,
        sentence_max_words=sentence_max_words,
        overlap_sentences=overlap_sentences,
        max_words_per_chunk=max_words_per_chunk,
        strategy=strategy,
        item_separator=item_separator,
        semantic_target_words=semantic_target_words,
        semantic_max_words=semantic_max_words,
        semantic_min_words=semantic_min_words,
    )
    return [{column: row.get(column) for column in CHUNK_COLUMNS} for row in rows]
