"""T1.3: build KB từ raw CMA thành parquet (SYSTEM_DESIGN 2.3, 4 bước 2).

- ``kb/artworks.parquet``: một dòng mỗi tác phẩm, cột đúng §2.3.
- ``kb/chunks.parquet``: đơn vị truy xuất văn bản, theo quy tắc chunking.

Lệnh:
    uv run python -m artrag.data.build_kb --check
    uv run python -m artrag.data.build_kb --limit 50 --check
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from artrag.config import Config, load_config, paths
from artrag.data.artists import (
    ARTIST_COLUMNS,
    BIO_SOURCE_CMA,
    BIO_SOURCE_NONE,
    BIO_SOURCE_WIKIPEDIA,
    build_artist_rows,
    chunk_rows_for_artists,
    load_wiki_records,
)
from artrag.data.chunking import CHUNK_COLUMNS, make_chunk_rows
from artrag.data.cma_parse import ARTWORK_COLUMNS, parse_artwork
from artrag.data.enrich_wiki import load_cma_bios

DEFAULT_MIN_DESCRIPTION_CHARS = 50


# ---------------------------------------------------------------- reading
def raw_pages(raw_dir: str | Path) -> list[Path]:
    """Danh sách file trang raw, sắp xếp theo ``skip``."""
    cma_dir = Path(raw_dir) / "cma"
    if not cma_dir.exists():
        return []
    pages = sorted(cma_dir.glob("page_*.json"))
    return sorted(pages, key=lambda p: _page_skip(p))


def _page_skip(path: Path) -> int:
    stem = path.stem  # page_00000
    try:
        return int(stem.split("_")[-1])
    except ValueError:
        return 0


def load_records(raw_dir: str | Path) -> list[dict[str, Any]]:
    """Đọc tất cả bản ghi từ các trang raw (theo thứ tự skip)."""
    records: list[dict[str, Any]] = []
    for path in raw_pages(raw_dir):
        try:
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, ValueError):
            continue
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            continue
        for record in data:
            if isinstance(record, dict):
                records.append(record)
    return records


def _cfg_get(cfg: Config, key: str, default: Any) -> Any:
    value = cfg.get(f"data.{key}", None)
    return default if value is None else value


def build_rows(
    records: list[dict[str, Any]],
    min_description_chars: int = DEFAULT_MIN_DESCRIPTION_CHARS,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Parse bản ghi -> dòng của ``artworks``; lọc theo description."""
    rows: list[dict[str, Any]] = []
    for record in records:
        row = parse_artwork(record)
        description = row.get("description")
        if not description or len(description) < min_description_chars:
            continue
        rows.append(row)
        if limit is not None and len(rows) >= limit:
            break
    return rows


def make_chunk_rows_for_all(
    rows: list[dict[str, Any]],
    cfg: Config,
) -> list[dict[str, Any]]:
    """Sinh chunk cho tất cả tác phẩm theo tham số chunking trong config."""
    chunking = cfg.get("data.chunking", {}) or {}
    kwargs = {
        "sentence_max_words": int(chunking.get("sentence_max_words", 180)),
        "field_max_words": int(chunking.get("field_max_words", 250)),
        "overlap_sentences": int(chunking.get("overlap_sentences", 1)),
        "max_words_per_chunk": int(chunking.get("max_words_per_chunk", 250)),
    }
    chunks: list[dict[str, Any]] = []
    for row in rows:
        chunks.extend(make_chunk_rows(row, **kwargs))
    return chunks


def source_url_by_artist(rows: list[dict[str, Any]]) -> dict[str, str]:
    """URL của tác phẩm CMA đầu tiên của mỗi họa sĩ (nguồn cho bio CMA)."""
    urls: dict[str, str] = {}
    for row in rows:
        artist_id = row.get("artist_id")
        source_url = row.get("source_url")
        if not artist_id or not source_url:
            continue
        if artist_id not in urls:
            urls[artist_id] = str(source_url)
    return urls


def build_artists(
    rows: list[dict[str, Any]],
    raw_dir: str | Path,
    prefer_cma: bool = True,
    min_bio_chars: int = 40,
) -> list[dict[str, Any]]:
    """Dựng dòng ``artists.parquet`` từ KB + cache wiki + tiểu sử CMA."""
    wiki_records = load_wiki_records(raw_dir)
    cma_bios = load_cma_bios(raw_dir)
    return build_artist_rows(
        rows,
        wiki_records=wiki_records,
        cma_bios=cma_bios,
        prefer_cma=prefer_cma,
        min_chars=min_bio_chars,
    )


def print_artist_stats(artist_rows: list[dict[str, Any]]) -> None:
    """In số họa sĩ và số bio theo từng nguồn (``cma``/``wikipedia``/``none``)."""
    total = len(artist_rows)
    print(f"\n-- artist ({total}) --")
    if not total:
        return
    by_source = {BIO_SOURCE_CMA: 0, BIO_SOURCE_WIKIPEDIA: 0, BIO_SOURCE_NONE: 0}
    for row in artist_rows:
        source = row.get("bio_source")
        if source in by_source:
            by_source[source] += 1
        else:
            by_source[BIO_SOURCE_NONE] += 1
    for source in (BIO_SOURCE_CMA, BIO_SOURCE_WIKIPEDIA, BIO_SOURCE_NONE):
        count = by_source[source]
        print(f"{source:>16}: {count:>5} / {total} ({count / total:.1%})")
    with_bio = by_source[BIO_SOURCE_CMA] + by_source[BIO_SOURCE_WIKIPEDIA]
    print(f"{'có bio':>16}: {with_bio:>5} / {total} ({with_bio / total:.1%})")


def write_parquet(rows: list[dict[str, Any]], columns: list[str], path: Path) -> Path:
    """Ghi parquet với đúng thứ tự cột (cột thiếu -> null)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_parquet(path, engine="pyarrow", index=False)
    return path


# ---------------------------------------------------------------- check
def print_stats(
    rows: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    max_words_per_chunk: int,
) -> None:
    """In thống kê KB: số tác phẩm, độ phủ trường, phân bố department, chunk."""
    total = len(rows)
    print(f"KB: {total} artworks")

    if total:
        print("\n-- độ phủ trường --")
        for field in (
            "description",
            "did_you_know",
            "provenance_text",
            "exhibitions_text",
            "current_location",
            "prev_owner",
        ):
            covered = sum(1 for row in rows if row.get(field))
            print(f"{field:>18}: {covered:>5} / {total} ({covered / total:.1%})")

        print("\n-- phân bố department --")
        for name, count in Counter(row.get("department") or "(không rõ)" for row in rows).most_common():
            print(f"{name:>42}: {count}")

    print("\n-- chunk theo field --")
    by_field = Counter(chunk.get("field") for chunk in chunks)
    for field, count in by_field.most_common():
        print(f"{field:>16}: {count}")
    print(f"{'TOTAL':>16}: {len(chunks)}")

    empty = sum(1 for chunk in chunks if not chunk.get("text", "").strip())
    over = sum(1 for chunk in chunks if chunk.get("n_words", 0) > max_words_per_chunk)
    print(f"\nchunk rỗng: {empty} | chunk vượt {max_words_per_chunk} từ: {over}")
    if empty or over:
        print("CẢNH BÁO: có chunk không hợp lệ")


# ---------------------------------------------------------------- CLI
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m artrag.data.build_kb",
        description="Dựng KB parquet từ dữ liệu CMA đã tải (raw/cma).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="in thống kê KB (số tác phẩm, độ phủ, department, chunk)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="chỉ xử lý N tác phẩm đầu (dùng để thử nhanh)",
    )
    parser.add_argument(
        "--with-bio",
        action="store_true",
        help="dựng thêm artists.parquet và nối chunk artist_bio vào chunks.parquet",
    )
    parser.add_argument("--profile", default=None, help="profile config (mặc định: local)")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="ghi đè config, ví dụ --set data.min_description_chars=10",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(profile=args.profile, overrides=args.overrides)
    p = paths(cfg, ensure=True)

    records = load_records(p.raw)
    if not records:
        print("Không có bản ghi raw nào. Chạy: python -m artrag.data.fetch_cma")
        return 1

    min_chars = int(_cfg_get(cfg, "min_description_chars", DEFAULT_MIN_DESCRIPTION_CHARS))
    rows = build_rows(records, min_description_chars=min_chars, limit=args.limit)
    if not rows:
        print("Không còn tác phẩm nào qua bộ lọc description.")
        return 1

    chunking = cfg.get("data.chunking", {}) or {}
    max_words_per_chunk = int(chunking.get("max_words_per_chunk", 250))
    chunks = make_chunk_rows_for_all(rows, cfg)

    artist_rows: list[dict[str, Any]] = []
    if args.with_bio:
        prefer_cma = bool(_cfg_get(cfg, "bio_prefer_cma", True))
        min_bio_chars = int(_cfg_get(cfg, "bio_min_chars", 40))
        artist_rows = build_artists(
            rows,
            p.raw,
            prefer_cma=prefer_cma,
            min_bio_chars=min_bio_chars,
        )
        bio_chunks = chunk_rows_for_artists(
            artist_rows,
            source_url_by_artist=source_url_by_artist(rows),
            sentence_max_words=int(chunking.get("sentence_max_words", 180)),
            overlap_sentences=int(chunking.get("overlap_sentences", 1)),
            max_words_per_chunk=max_words_per_chunk,
        )
        chunks = chunks + bio_chunks

    if not args.check:
        write_parquet(rows, ARTWORK_COLUMNS, p.kb / "artworks.parquet")
        write_parquet(chunks, CHUNK_COLUMNS, p.kb / "chunks.parquet")
        print(f"Đã ghi {p.kb / 'artworks.parquet'}")
        print(f"Đã ghi {p.kb / 'chunks.parquet'}")
        if args.with_bio:
            write_parquet(artist_rows, ARTIST_COLUMNS, p.kb / "artists.parquet")
            print(f"Đã ghi {p.kb / 'artists.parquet'}")

    print_stats(rows, chunks, max_words_per_chunk)
    if args.with_bio:
        print_artist_stats(artist_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
