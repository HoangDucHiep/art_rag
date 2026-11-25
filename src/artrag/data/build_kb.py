"""T1.3: build KB tu raw CMA thanh parquet (SYSTEM_DESIGN 2.3, 4 buoc 2).

- ``kb/artworks.parquet``: mot dong moi tac pham, cot dung §2.3.
- ``kb/chunks.parquet``: don vi truy xuat van ban, theo quy tac chunking.

Lenh:
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
from artrag.data.chunking import CHUNK_COLUMNS, make_chunk_rows
from artrag.data.cma_parse import ARTWORK_COLUMNS, parse_artwork

DEFAULT_MIN_DESCRIPTION_CHARS = 50


# ---------------------------------------------------------------- reading
def raw_pages(raw_dir: str | Path) -> list[Path]:
    """Danh sach file trang raw, sap xep theo ``skip``."""
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
    """Doc tat ca ban ghi tu cac trang raw (theo thu tu skip)."""
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
    """Parse ban ghi -> dong cua ``artworks``; loc theo description."""
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
    """Sinh chunk cho tat ca tac pham theo tham so chunking trong config."""
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


def write_parquet(rows: list[dict[str, Any]], columns: list[str], path: Path) -> Path:
    """Ghi parquet voi dung thu tu cot (cot thieu -> null)."""
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
    """In thong ke KB: so tac pham, do phu truong, phan bo department, chunk."""
    total = len(rows)
    print(f"KB: {total} artworks")

    if total:
        print("\n-- do phu truong --")
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

        print("\n-- phan bo department --")
        for name, count in Counter(row.get("department") or "(khong ro)" for row in rows).most_common():
            print(f"{name:>42}: {count}")

    print("\n-- chunk theo field --")
    by_field = Counter(chunk.get("field") for chunk in chunks)
    for field, count in by_field.most_common():
        print(f"{field:>16}: {count}")
    print(f"{'TOTAL':>16}: {len(chunks)}")

    empty = sum(1 for chunk in chunks if not chunk.get("text", "").strip())
    over = sum(1 for chunk in chunks if chunk.get("n_words", 0) > max_words_per_chunk)
    print(f"\nchunk rong: {empty} | chunk vuot {max_words_per_chunk} tu: {over}")
    if empty or over:
        print("CANH BAO: co chunk khong hop le")


# ---------------------------------------------------------------- CLI
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m artrag.data.build_kb",
        description="Dung KB parquet tu du lieu CMA da tai (raw/cma).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="in thong ke KB (so tac pham, do phu, department, chunk)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="chi xu ly N tac pham dau (dung de thu nhanh)",
    )
    parser.add_argument("--profile", default=None, help="profile config (mac dinh: local)")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="ghi de config, vi du --set data.min_description_chars=10",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(profile=args.profile, overrides=args.overrides)
    p = paths(cfg, ensure=True)

    records = load_records(p.raw)
    if not records:
        print("Khong co ban ghi raw nao. Chay: python -m artrag.data.fetch_cma")
        return 1

    min_chars = int(_cfg_get(cfg, "min_description_chars", DEFAULT_MIN_DESCRIPTION_CHARS))
    rows = build_rows(records, min_description_chars=min_chars, limit=args.limit)
    if not rows:
        print("Khong con tac pham nao qua bo loc description.")
        return 1

    chunking = cfg.get("data.chunking", {}) or {}
    max_words_per_chunk = int(chunking.get("max_words_per_chunk", 250))
    chunks = make_chunk_rows_for_all(rows, cfg)

    if not args.check:
        write_parquet(rows, ARTWORK_COLUMNS, p.kb / "artworks.parquet")
        write_parquet(chunks, CHUNK_COLUMNS, p.kb / "chunks.parquet")
        print(f"Da ghi {p.kb / 'artworks.parquet'}")
        print(f"Da ghi {p.kb / 'chunks.parquet'}")

    print_stats(rows, chunks, max_words_per_chunk)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
