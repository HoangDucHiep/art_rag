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
from artrag.data.chunk_strategies import (
    STRATEGY_ITEM,
    STRATEGY_SEMANTIC,
    STRATEGY_SENTENCE,
    apply_strategy,
)
from artrag.data.chunking import (
    CHUNK_COLUMNS,
    SENTENCE_FIELDS,
    _field_value,
    chunk_sentences,
    chunk_single,
    count_words,
    make_chunk_rows,
)
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


def field_strategies(cfg: Config) -> dict[str, str]:
    """Chiến lược chunking của từng field, đọc từ config.

    Config khoá ``data.chunking.strategy_<field>`` (ví dụ
    ``data.chunking.strategy_provenance: item``). Field không có trong config
    dùng mặc định ``sentence`` (C1 baseline).
    """
    chunking = cfg.get("data.chunking", {}) or {}
    strategies: dict[str, str] = {}
    for key, value in chunking.items():
        if not key.startswith("strategy_"):
            continue
        field = key[len("strategy_") :]
        if not field or not value:
            continue
        strategies[field] = str(value)
    return strategies


def chunking_kwargs(cfg: Config) -> dict[str, Any]:
    """Tham số chunking lấy từ config (dùng chung cho build + variants)."""
    chunking = cfg.get("data.chunking", {}) or {}
    return {
        "sentence_max_words": int(chunking.get("sentence_max_words", 180)),
        "field_max_words": int(chunking.get("field_max_words", 250)),
        "overlap_sentences": int(chunking.get("overlap_sentences", 1)),
        "max_words_per_chunk": int(chunking.get("max_words_per_chunk", 250)),
        "item_separator": str(chunking.get("item_separator", "; ")),
        "semantic_target_words": int(chunking.get("semantic_target_words", 120)),
        "semantic_max_words": int(chunking.get("semantic_max_words", 250)),
        "semantic_min_words": int(chunking.get("semantic_min_words", 40)),
    }


def make_chunk_rows_for_all(
    rows: list[dict[str, Any]],
    cfg: Config,
) -> list[dict[str, Any]]:
    """Sinh chunk cho tất cả tác phẩm theo tham số chunking trong config."""
    kwargs = chunking_kwargs(cfg)
    strategies = field_strategies(cfg)
    chunks: list[dict[str, Any]] = []
    for row in rows:
        chunks.extend(make_chunk_rows(row, strategies=strategies, **kwargs))
    return chunks


# ------------------------------------------------------- bảng biến thể
#: Chiến lược thử nghiệm cho từng field (dùng để sinh ``chunks_variants``).
VARIANT_FIELD_STRATEGIES: dict[str, tuple[str, ...]] = {
    "description": (STRATEGY_SENTENCE, STRATEGY_SEMANTIC),
    "did_you_know": (STRATEGY_SENTENCE,),
    "overview": (STRATEGY_SENTENCE,),
    "provenance": (STRATEGY_SENTENCE, STRATEGY_ITEM),
    "exhibitions": (STRATEGY_SENTENCE, STRATEGY_ITEM),
    "artist_bio": (STRATEGY_SENTENCE, STRATEGY_SEMANTIC),
}


def make_variant_chunk_rows(
    rows: list[dict[str, Any]],
    artist_rows: list[dict[str, Any]],
    cfg: Config,
) -> list[dict[str, Any]]:
    """Sinh bảng chunk **đầy đủ các biến thể** để so sánh trong báo cáo.

    Mỗi chunk có cột ``strategy`` và ``chunk_id`` kèm hậu tố ``#<strategy>``
    để không trùng với bảng ``chunks.parquet`` (giữ nguyên chiến lược đang
    dùng, để không phá pipeline).

    Vì C3 chỉ áp cho ``artist_bio``/``description`` và C2 chỉ áp cho
    ``provenance``/``exhibitions``, hai bảng cho phép so sánh trực tiếp trên
    cùng một artist / cùng một tác phẩm.
    """
    kwargs = chunking_kwargs(cfg)
    max_words_per_chunk = int(kwargs["max_words_per_chunk"])
    urls = source_url_by_artist(rows)
    variants: list[dict[str, Any]] = []

    for field, strategies in VARIANT_FIELD_STRATEGIES.items():
        for strategy in strategies:
            for row in rows:
                if field == "artist_bio":
                    continue
                text = _field_value(row, field)
                if not text or not text.strip():
                    continue
                if strategy == STRATEGY_SENTENCE:
                    # baseline C1: dùng đúng hành vi gốc của T1.3
                    if field == "overview":
                        pieces = chunk_single(text, int(kwargs["max_words_per_chunk"]))
                    elif field in SENTENCE_FIELDS:
                        pieces = chunk_sentences(
                            text,
                            int(kwargs["sentence_max_words"]),
                            int(kwargs["overlap_sentences"]),
                        )
                    else:
                        pieces = chunk_single(text, int(kwargs["field_max_words"]))
                else:
                    pieces = apply_strategy(
                        text,
                        strategy=strategy,
                        sentence_max_words=int(kwargs["sentence_max_words"]),
                        overlap_sentences=int(kwargs["overlap_sentences"]),
                        field_max_words=int(kwargs["field_max_words"]),
                        item_separator=str(kwargs["item_separator"]),
                        semantic_target_words=int(kwargs["semantic_target_words"]),
                        semantic_max_words=int(kwargs["semantic_max_words"]),
                        semantic_min_words=int(kwargs["semantic_min_words"]),
                    )
                for index, piece in enumerate(pieces):
                    piece = piece.strip()
                    if not piece:
                        continue
                    words = piece.split()
                    if len(words) > max_words_per_chunk:
                        piece = " ".join(words[:max_words_per_chunk])
                    variants.append(
                        {
                            "chunk_id": f"{row.get('artwork_id')}#{field}#{index}#{strategy}",
                            "artwork_id": row.get("artwork_id"),
                            "artist_id": row.get("artist_id"),
                            "field": field,
                            "text": piece,
                            "source_url": row.get("source_url"),
                            "n_words": count_words(piece),
                            "strategy": strategy,
                        }
                    )

    # chunk tiểu sử họa sĩ (C1 sentence vs C3 semantic)
    for strategy in VARIANT_FIELD_STRATEGIES["artist_bio"]:
        bio_chunks = chunk_rows_for_artists(
            artist_rows,
            source_url_by_artist=urls,
            max_words_per_chunk=max_words_per_chunk,
            strategy=strategy,
            item_separator=str(kwargs["item_separator"]),
            semantic_target_words=int(kwargs["semantic_target_words"]),
            semantic_max_words=int(kwargs["semantic_max_words"]),
            semantic_min_words=int(kwargs["semantic_min_words"]),
            sentence_max_words=int(kwargs["sentence_max_words"]),
            overlap_sentences=int(kwargs["overlap_sentences"]),
        )
        for chunk in bio_chunks:
            chunk = dict(chunk)
            chunk["chunk_id"] = f"{chunk['chunk_id']}#{strategy}"
            variants.append(chunk)

    return [{column: row.get(column) for column in CHUNK_COLUMNS} for row in variants]


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


def print_variant_stats(
    variants: list[dict[str, Any]],
    max_words_per_chunk: int,
) -> None:
    """In thống kê bảng biến thể chunking (theo chiến lược và theo field)."""
    if not variants:
        return
    print(f"\n-- chunk biến thể: {len(variants)} --")
    by_strategy = Counter(chunk.get("strategy") for chunk in variants)
    for strategy, count in sorted(by_strategy.items(), key=lambda kv: str(kv[0])):
        lengths = [chunk.get("n_words", 0) for chunk in variants if chunk.get("strategy") == strategy]
        avg = sum(lengths) / len(lengths) if lengths else 0.0
        print(f"{str(strategy):>16}: {count:>6} chunk | độ dài TB {avg:6.1f} từ")
    by_field = Counter(chunk.get("field") for chunk in variants)
    print("  theo field:")
    for field, count in by_field.most_common():
        print(f"{str(field):>16}: {count:>6}")
    over = sum(1 for chunk in variants if chunk.get("n_words", 0) > max_words_per_chunk)
    print(f"  vượt {max_words_per_chunk} từ: {over}")


def write_parquet(rows: list[dict[str, Any]], columns: list[str], path: Path) -> Path:
    """Ghi parquet với đúng thứ tự cột (cột thiếu -> null)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_parquet(path, engine="pyarrow", index=False)
    return path


def preserve_image_paths(
    rows: list[dict[str, Any]],
    kb_path: str | Path,
    images_dir: str | Path,
) -> int:
    """Giữ cột ``image_path`` cũ khi rebuild KB (nếu file ảnh còn tồn tại).

    T1.6 tải ảnh và điền ``image_path`` vào ``artworks.parquet``. Nếu chạy lại
    ``build_kb`` mà không có bước này thì cột đó bị mất (ghi đè toàn bộ).
    Ở đây ta giữ lại giá trị cũ **chỉ khi** file ảnh tương ứng vẫn còn trên
    đĩa; ngược lại đặt ``None``.
    """
    p = Path(kb_path)
    if not p.exists():
        return 0
    try:
        old = pd.read_parquet(p)
    except (OSError, ValueError):
        return 0
    if "artwork_id" not in old.columns or "image_path" not in old.columns:
        return 0

    images_dir = Path(images_dir)
    previous: dict[str, str] = {}
    for artwork_id, image_path in zip(old["artwork_id"], old["image_path"]):
        if artwork_id is None or image_path is None:
            continue
        previous[str(artwork_id)] = str(image_path)

    kept = 0
    for row in rows:
        artwork_id = row.get("artwork_id")
        if artwork_id is None:
            row["image_path"] = None
            continue
        artwork_id = str(artwork_id)
        image_path = previous.get(artwork_id)
        if image_path and (images_dir / Path(image_path).name).exists():
            row["image_path"] = image_path
            kept += 1
        else:
            row["image_path"] = None
    return kept


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
    parser.add_argument(
        "--with-variants",
        action="store_true",
        help="sinh thêm chunks_variants.parquet (C1 sentence + C2 item + C3 semantic)",
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
    strategies = field_strategies(cfg)
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
            strategy=strategies.get("artist_bio") or STRATEGY_SENTENCE,
            item_separator=str(chunking.get("item_separator", "; ")),
            semantic_target_words=int(chunking.get("semantic_target_words", 120)),
            semantic_max_words=int(chunking.get("semantic_max_words", 250)),
            semantic_min_words=int(chunking.get("semantic_min_words", 40)),
        )
        chunks = chunks + bio_chunks

    variants: list[dict[str, Any]] = []
    if args.with_variants or bool(chunking.get("write_variants", False)):
        variants = make_variant_chunk_rows(rows, artist_rows, cfg)

    if not args.check:
        # giữ image_path đã tải ở T1.6 nếu file ảnh còn tồn tại
        kept = preserve_image_paths(rows, p.kb / "artworks.parquet", p.images)
        write_parquet(rows, ARTWORK_COLUMNS, p.kb / "artworks.parquet")
        write_parquet(chunks, CHUNK_COLUMNS, p.kb / "chunks.parquet")
        print(f"Đã ghi {p.kb / 'artworks.parquet'}")
        if kept:
            print(f"Giữ {kept} image_path (file ảnh còn tồn tại)")
        print(f"Đã ghi {p.kb / 'chunks.parquet'}")
        if args.with_bio:
            write_parquet(artist_rows, ARTIST_COLUMNS, p.kb / "artists.parquet")
            print(f"Đã ghi {p.kb / 'artists.parquet'}")
        if variants:
            write_parquet(variants, CHUNK_COLUMNS, p.kb / "chunks_variants.parquet")
            print(f"Đã ghi {p.kb / 'chunks_variants.parquet'}")

    print_stats(rows, chunks, max_words_per_chunk)
    if args.with_bio:
        print_artist_stats(artist_rows)
    if variants:
        print_variant_stats(variants, max_words_per_chunk)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
