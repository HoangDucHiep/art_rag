"""C3: đo đạc tầng 1 (proxy) cho ablation chunking — KHÔNG cần GPU/model.

Các chỉ số ở đây đo **chất lượng cấu trúc** của bộ chunk, không cần chạy
model truy xuất. Chúng trả lời câu hỏi: "chiến lược chunking nào cắt văn
bản ít bị vỡ nhất, ít lặp nhất, và giữ đủ thông tin nhất?".

Tầng 2 (Recall@K / MRR của nhánh BGE-M3) **deferred** tới T6/T8 vì cần
text index (T3.2) và bộ câu hỏi (T6.1).

Dùng:
    uv run python -m artrag.eval.chunk_eval --variant sentence --variant item --variant semantic
    uv run python -m artrag.eval.chunk_eval --csv results/chunk_eval.csv
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from artrag.config import load_config, paths
from artrag.utils.text import split_sentences

#: Ngưỡng hard-cap của hệ thống (mọi chunk không được vượt).
MAX_WORDS = 250

#: Dấu kết thúc câu hợp lệ (chunk kết thúc bằng một trong các dấu này được
#: coi là "không bị cắt giữa câu").
SENTENCE_END_CHARS = ".!?…\"'”’)]}"

#: Field có cấu trúc danh sách - dùng `"; "` làm mốc "mục tự nhiên".
LIST_FIELDS = ("provenance", "exhibitions")


# ------------------------------------------------------------------ helpers
def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:  # NaN
        return ""
    return str(value).strip()


def _words(text: str) -> list[str]:
    return _text(text).split()


def _norm(text: str) -> str:
    """Chuẩn hóa khoảng trắng để so sánh nguyên văn."""
    return " ".join(_text(text).split())


# ------------------------------------------------------------------ metrics
def chunk_profile(chunks: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Tổng quan bộ chunk: số lượng, độ dài, phân bố theo field.

    Trả về dict với các khoá:
    - ``n_chunks``, ``n_empty``, ``n_over_max``, ``pct_over_max``
    - ``avg_words``, ``median_words``, ``min_words``, ``max_words``
    - ``by_field``: {field: số chunk}
    - ``avg_words_by_field``: {field: độ dài trung bình}
    """
    chunks = list(chunks)
    lengths = [int(chunk.get("n_words") or len(_words(chunk.get("text", "")))) for chunk in chunks]
    by_field: Counter[str] = Counter(_text(chunk.get("field")) for chunk in chunks)

    lengths_sorted = sorted(lengths)
    if lengths_sorted:
        mid = len(lengths_sorted) // 2
        median = (
            float(lengths_sorted[mid])
            if len(lengths_sorted) % 2
            else (lengths_sorted[mid - 1] + lengths_sorted[mid]) / 2
        )
    else:
        median = 0.0

    avg_by_field: dict[str, float] = {}
    for field in by_field:
        field_lengths = [
            int(chunk.get("n_words") or len(_words(chunk.get("text", ""))))
            for chunk in chunks
            if _text(chunk.get("field")) == field
        ]
        avg_by_field[field] = (
            sum(field_lengths) / len(field_lengths) if field_lengths else 0.0
        )

    n_over = sum(1 for length in lengths if length > MAX_WORDS)
    return {
        "n_chunks": len(chunks),
        "n_empty": sum(1 for chunk in chunks if not _text(chunk.get("text"))),
        "n_over_max": n_over,
        "pct_over_max": (n_over / len(chunks) * 100) if chunks else 0.0,
        "avg_words": (sum(lengths) / len(lengths)) if lengths else 0.0,
        "median_words": median,
        "min_words": min(lengths) if lengths else 0,
        "max_words": max(lengths) if lengths else 0,
        "by_field": dict(by_field),
        "avg_words_by_field": avg_by_field,
    }


def mid_item_cut_rate(
    chunks: Iterable[dict[str, Any]],
    source_text: str,
    separator: str = "; ",
) -> float:
    """% chunk có **ranh giới nằm giữa một mục** tự nhiên (mốc ``"; "``).

    Cách đo (theo offset):
    1. Chuẩn hóa khoảng trắng văn bản gốc và xác định các mục tự nhiên
       (tách theo ``separator``) cùng khoảng bắt đầu/kết thúc của từng mục.
    2. Với mỗi chunk (theo đúng thứ tự), tìm vị trí của nó trong văn bản gốc
       bằng cách quét tiếp (chunk của cùng một field đi theo thứ tự nguồn).
    3. Nếu **đầu** hoặc **cuối** chunk rơi vào giữa một mục (không trùng với
       đầu/cuối mục) thì chunk bị cắt giữa mục.

    Chiến lược ``item`` (C2) gom trọn mục nên tỉ lệ này phải bằng 0.

    Chỉ áp dụng cho field danh sách (``provenance``, ``exhibitions``).
    """
    chunks = list(chunks)
    if not chunks:
        return 0.0

    source = _norm(source_text)
    if not source:
        return 0.0

    # --- bảng đồ offset -> biên mục trong văn bản đã chuẩn hóa ---
    item_bounds: list[tuple[int, int]] = []
    cursor = 0
    for raw_item in source_text.split(separator):
        item = _norm(raw_item)
        if not item:
            cursor += len(_norm(raw_item))
            continue
        position = source.find(item, cursor)
        if position < 0:
            continue
        item_bounds.append((position, position + len(item)))
        cursor = position + len(item)

    if not item_bounds:
        return 0.0

    def _inside_item(offset: int) -> bool:
        """Offset có nằm GIỮA một mục (không phải đúng đầu/cuối mục)?"""
        for start, end in item_bounds:
            if start < offset < end:
                return True
        return False

    cut = 0
    counted = 0
    cursor = 0
    for chunk in chunks:
        text = _norm(chunk.get("text", ""))
        if not text:
            continue
        position = source.find(text, cursor)
        if position < 0:
            continue
        cursor = position + len(text)
        counted += 1
        start_cut = _inside_item(position)
        end_cut = _inside_item(position + len(text))
        if start_cut or end_cut:
            cut += 1

    return cut / counted * 100 if counted else 0.0


def mid_sentence_cut_rate(chunks: Iterable[dict[str, Any]]) -> float:
    """% chunk **không** kết thúc bằng dấu câu (tức bị cắt giữa câu).

    Chunk cuối cùng của một đoạn văn cũng được tính, vì nó có thể bị cắt
    cứng bởi hard-cap.
    """
    chunks = list(chunks)
    if not chunks:
        return 0.0
    cut = sum(
        1
        for chunk in chunks
        if _text(chunk.get("text")) and _text(chunk.get("text"))[-1] not in SENTENCE_END_CHARS
    )
    return cut / len(chunks) * 100


def redundancy(chunks: Iterable[dict[str, Any]], ngram: int = 3) -> float:
    """% token bị lặp giữa các chunk liền kề (do overlap).

    Với mỗi cặp chunk liền kề, tính phần token của chunk sau nằm trong một
    **đoạn n-gram trùng lặp** với chunk trước; trung bình trên toàn bộ cặp.

    Dùng **n-gram** (mặc định 3 từ), không dùng từ đơn: nếu đếm từ đơn thì
    các từ thông dụng hoặc tên riêng lặp lại (``Cleveland Museum of Art``)
    bị tính nhầm là overlap. Với n-gram, chỉ đo được **đoạn văn thực sự
    trùng lặp** - đúng thứ mà overlap 1 câu gây ra.
    """
    chunks = list(chunks)
    if len(chunks) < 2:
        return 0.0

    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for chunk in chunks:
        key = (
            _text(chunk.get("artwork_id") or chunk.get("artist_id")),
            _text(chunk.get("field")),
        )
        groups[key].append(_text(chunk.get("text")))

    ratios: list[float] = []
    for pieces in groups.values():
        for previous, current in zip(pieces, pieces[1:]):
            prev_tokens = previous.split()
            curr_tokens = current.split()
            if len(curr_tokens) < ngram:
                continue
            prev_grams = {
                tuple(prev_tokens[i : i + ngram])
                for i in range(len(prev_tokens) - ngram + 1)
            }
            if not prev_grams:
                continue
            # đánh dấu các vị trí bị trùng (để tránh đếm lặp khi nhiều n-gram
            # chồng lấn)
            duplicated = [False] * len(curr_tokens)
            for i in range(len(curr_tokens) - ngram + 1):
                if tuple(curr_tokens[i : i + ngram]) in prev_grams:
                    for j in range(i, i + ngram):
                        duplicated[j] = True
            ratio = sum(duplicated) / len(curr_tokens)
            ratios.append(ratio)

    return sum(ratios) / len(ratios) * 100 if ratios else 0.0


def info_coverage(chunks: Iterable[dict[str, Any]], source_text: str) -> float:
    """% token của văn bản gốc xuất hiện trong **ít nhất một** chunk.

    Đo mức độ "giữ đủ thông tin" của bộ chunk: nếu chiến lược nào làm mất
    token thì coverage giảm.
    """
    chunks = list(chunks)
    source_tokens = source_text.split()
    if not source_tokens:
        return 0.0

    seen: set[str] = set()
    for chunk in chunks:
        seen.update(_text(chunk.get("text")).split())

    hit = sum(1 for token in source_tokens if token in seen)
    return hit / len(source_tokens) * 100


def verbatim_rate(chunks: Iterable[dict[str, Any]], source_text: str) -> float:
    """% chunk là **đoạn con nguyên văn** của văn bản gốc.

    Đây là chỉ số quan trọng nhất: nó phát hiện ngay lỗi "cắt vỡ" (chữ viết
    tắt bị ngắt giữa chừng, mất ký tự, thêm/bớt từ).
    """
    chunks = list(chunks)
    if not chunks:
        return 0.0
    source_norm = _norm(source_text)
    verbatim = sum(
        1
        for chunk in chunks
        if _norm(chunk.get("text", "")) in source_norm
    )
    return verbatim / len(chunks) * 100


# ------------------------------------------------------------------ aggregate
def evaluate_variant(
    chunks: list[dict[str, Any]],
    sources: dict[str, str] | None = None,
    separator: str = "; ",
) -> dict[str, Any]:
    """Tính toàn bộ proxy metrics cho một bộ chunk của một chiến lược.

    ``sources`` là dict ``{f"{field}|{id}": văn bản gốc}`` với ``id`` =
    ``artist_id`` (chunk tiểu sử) hoặc ``artwork_id`` (field khác). Nếu không
    truyền, các chỉ số cần văn bản gốc sẽ bằng 0.
    """
    sources = sources or {}
    result: dict[str, Any] = {}
    result.update(chunk_profile(chunks))
    result["mid_sentence_cut_rate"] = mid_sentence_cut_rate(chunks)
    result["redundancy"] = redundancy(chunks)

    # nhóm chunk theo chủ thể (artist_id cho artist_bio, artwork_id cho field khác)
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        key = (
            _text(chunk.get("artwork_id") or chunk.get("artist_id")),
            _text(chunk.get("field")),
        )
        by_key[key].append(chunk)

    mid_item_rates: list[float] = []
    coverage_rates: list[float] = []
    verbatim_rates: list[float] = []
    for (key, field), group in by_key.items():
        source = sources.get(f"{field}|{key}")
        if not source:
            continue
        if field in LIST_FIELDS:
            mid_item_rates.append(mid_item_cut_rate(group, source, separator=separator))
        coverage_rates.append(info_coverage(group, source))
        verbatim_rates.append(verbatim_rate(group, source))

    result["mid_item_cut_rate"] = (
        sum(mid_item_rates) / len(mid_item_rates) if mid_item_rates else 0.0
    )
    result["info_coverage"] = (
        sum(coverage_rates) / len(coverage_rates) if coverage_rates else 0.0
    )
    result["verbatim_rate"] = (
        sum(verbatim_rates) / len(verbatim_rates) if verbatim_rates else 0.0
    )
    return result


def evaluate_variants(
    variants: list[dict[str, Any]],
    sources: dict[str, str] | None = None,
    separator: str = "; ",
) -> list[dict[str, Any]]:
    """Đánh giá một bảng chunk (có cột ``strategy``) theo từng chiến lược."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in variants:
        grouped[_text(chunk.get("strategy")) or "sentence"].append(chunk)

    rows: list[dict[str, Any]] = []
    for strategy in sorted(grouped):
        metrics = evaluate_variant(grouped[strategy], sources=sources, separator=separator)
        metrics["strategy"] = strategy
        rows.append(metrics)
    return rows


# ------------------------------------------------------------------ CLI
def _load_sources(cfg: Any) -> dict[str, str]:
    """Nạp văn bản gốc của từng field để tính coverage / mid-item cut.

    Key = ``{field}:{id}`` để không lẫn giữa các field:
    - ``artist_bio:{artist_id}`` -> cột ``bio`` của ``artists.parquet``
    - ``description:{artwork_id}`` -> cột ``description``
    - ``provenance:{artwork_id}`` -> cột ``provenance_text``
    - ``exhibitions:{artwork_id}`` -> cột ``exhibitions_text``
    """
    import pandas as pd

    p = paths(cfg)
    sources: dict[str, str] = {}

    artists_path = p.kb / "artists.parquet"
    if artists_path.exists():
        artists = pd.read_parquet(artists_path)
        for row in artists.itertuples(index=False):
            bio = _text(getattr(row, "bio", ""))
            if bio:
                sources[f"artist_bio:{row.artist_id}"] = bio

    artworks_path = p.kb / "artworks.parquet"
    if artworks_path.exists():
        artworks = pd.read_parquet(artworks_path)
        for row in artworks.itertuples(index=False):
            for field, column in (
                ("description", "description"),
                ("provenance", "provenance_text"),
                ("exhibitions", "exhibitions_text"),
            ):
                value = _text(getattr(row, column, ""))
                if value:
                    sources[f"{field}:{row.artwork_id}"] = value
    return sources


def _sources_for_chunks(chunks: list[dict[str, Any]], sources: dict[str, str]) -> dict[str, str]:
    """Chọn đúng nguồn cho từng chunk, theo cặp ``(field, id)``.

    ``evaluate_variant`` dùng key là ``artwork_id`` (hoặc ``artist_id`` với
    chunk tiểu sử), nên ở đây ta gộp nguồn của TẤT CẢ field của một chủ thể
    vào cùng một key - hàm đánh giá chỉ dùng nguồn tương ứng với group của
    nó qua ``sources`` nội bộ.
    """
    picked: dict[str, str] = {}
    for chunk in chunks:
        artist_id = _text(chunk.get("artist_id"))
        artwork_id = _text(chunk.get("artwork_id"))
        field = _text(chunk.get("field"))
        if field == "artist_bio" and artist_id:
            key = f"artist_bio:{artist_id}"
        elif artwork_id:
            key = f"{field}:{artwork_id}"
        else:
            continue
        if key in sources:
            picked[f"{field}|{artwork_id or artist_id}"] = sources[key]
    return picked


def format_markdown(rows: list[dict[str, Any]]) -> str:
    """Bảng so sánh Markdown cho báo cáo."""
    header = (
        "| Chiến lược | Số chunk | Độ dài TB | % cắt giữa mục | % cắt giữa câu "
        "| Redundancy % | Coverage % | Nguyên văn % |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for row in rows:
        lines.append(
            f"| C {row['strategy']} | {row['n_chunks']} | {row['avg_words']:.1f} "
            f"| {row.get('mid_item_cut_rate', 0.0):.2f} "
            f"| {row.get('mid_sentence_cut_rate', 0.0):.2f} "
            f"| {row.get('redundancy', 0.0):.2f} "
            f"| {row.get('info_coverage', 0.0):.2f} "
            f"| {row.get('verbatim_rate', 0.0):.2f} |"
        )
    return header + "\n".join(lines)


def format_csv(rows: list[dict[str, Any]]) -> str:
    """Bảng so sánh CSV (để đưa vào báo cáo/spreadsheet)."""
    columns = [
        "strategy",
        "n_chunks",
        "avg_words",
        "median_words",
        "min_words",
        "max_words",
        "n_empty",
        "n_over_max",
        "pct_over_max",
        "mid_item_cut_rate",
        "mid_sentence_cut_rate",
        "redundancy",
        "info_coverage",
        "verbatim_rate",
    ]
    lines = [",".join(columns)]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append(str(value))
        lines.append(",".join(values))
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m artrag.eval.chunk_eval",
        description="Đo proxy metrics cho ablation chunking (không cần GPU/model).",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        choices=["sentence", "item", "semantic"],
        help="chỉ đánh giá các chiến lược này (mặc định: tất cả có trong bảng)",
    )
    parser.add_argument(
        "--variants-file",
        default=None,
        help="đường dẫn bảng biến thể (mặc định: data/kb/chunks_variants.parquet)",
    )
    parser.add_argument("--csv", default=None, help="ghi kết quả ra file CSV")
    parser.add_argument("--markdown", default=None, help="ghi bảng Markdown ra file")
    parser.add_argument("--profile", default=None, help="profile config")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="ghi đè config, ví dụ --set data.chunking.item_separator='; '",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(profile=args.profile, overrides=args.overrides)
    p = paths(cfg)

    variants_path = Path(args.variants_file) if args.variants_file else p.kb / "chunks_variants.parquet"
    if not variants_path.exists():
        print(f"Không có bảng biến thể: {variants_path}")
        print("Chạy: uv run python -m artrag.data.build_kb --with-bio --with-variants")
        return 1

    import pandas as pd

    variants = pd.read_parquet(variants_path).to_dict("records")
    if args.variant:
        wanted = set(args.variant)
        variants = [
            chunk for chunk in variants if _text(chunk.get("strategy")) in wanted
        ]
    if not variants:
        print("Không có chunk nào khớp chiến lược yêu cầu.")
        return 1

    sources = _load_sources(cfg)
    # gom nguồn theo chủ thể để evaluate_variant dùng đúng văn bản gốc
    per_strategy_sources: dict[str, dict[str, str]] = {}
    for chunk in variants:
        strategy = _text(chunk.get("strategy")) or "sentence"
        bucket = per_strategy_sources.setdefault(strategy, {})
        bucket.update(_sources_for_chunks([chunk], sources))

    rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in variants:
        grouped[_text(chunk.get("strategy")) or "sentence"].append(chunk)
    for strategy in sorted(grouped):
        metrics = evaluate_variant(
            grouped[strategy],
            sources=per_strategy_sources.get(strategy, {}),
            separator=str(cfg.get("data.chunking.item_separator", "; ")),
        )
        metrics["strategy"] = strategy
        rows.append(metrics)

    print("=" * 78)
    print("Ablation chunking - proxy metrics (tầng 1, không cần GPU)")
    print("=" * 78)
    print(format_markdown(rows))
    print()
    print(
        "Ghi chú: Recall@10 / MRR (tầng 2) để trống - đo ở M6/M8 khi có "
        "text index (T3.2) và bộ câu hỏi (T6.1)."
    )

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        Path(args.csv).write_text(format_csv(rows), encoding="utf-8")
        print(f"Đã ghi {args.csv}")
    if args.markdown:
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown).write_text(format_markdown(rows) + "\n", encoding="utf-8")
        print(f"Đã ghi {args.markdown}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
