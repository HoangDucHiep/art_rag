"""B1: các chiến lược chunking văn bản (T1.5 - ablation chunking).

Ba chiến lược, cùng một interface để so sánh công bằng:

| Mã | Tên | Cách cắt | Field áp dụng (thử nghiệm) |
|---|---|---|---|
| C1 | ``sentence`` | theo câu, gộp 180 từ, overlap 1 câu | tất cả (mặc định) |
| C2 | ``item`` | theo mục tự nhiên (``; ``) | ``provenance``, ``exhibitions`` |
| C3 | ``semantic`` | theo đoạn văn 120 từ, không overlap | ``artist_bio``, ``description`` |

Mọi hàm ở đây đều thuần: không cần dữ liệu, model hay mạng.

**Bất biến quan trọng**: với mọi chiến lược, ghép lại các chunk phải bảo toàn
nguyên văn văn bản gốc (sau khi chuẩn hóa khoảng trắng). Chiến lược chỉ
thay đổi **vị trí ranh giới**, không thay đổi nội dung.
"""

from __future__ import annotations

import re
from typing import Any

from artrag.utils.text import split_sentences

# Tên chiến lược (dùng làm giá trị cột ``strategy`` trong bảng biến thể).
STRATEGY_SENTENCE = "sentence"
STRATEGY_ITEM = "item"
STRATEGY_SEMANTIC = "semantic"

STRATEGIES = (STRATEGY_SENTENCE, STRATEGY_ITEM, STRATEGY_SEMANTIC)


def count_words(text: str) -> int:
    """Số từ (tách theo khoảng trắng).

    Cố tình định nghĩa cục bộ thay vì import từ ``artrag.data.chunking`` để
    tránh vòng import hai chiều (``chunking`` cần ``apply_strategy`` của
    module này).
    """
    if not text:
        return 0
    return len(text.split())


def split_paragraphs(text: str) -> list[str]:
    """Tách văn bản thành các đoạn theo ranh giới đoạn văn của nguồn.

    Văn bản nguồn (Wikipedia/CMA) dùng ``\\n\\n`` để tách đoạn văn; đây là
    ranh giới "có nghĩa" tự nhiên nhất, tốt hơn việc cắt theo câu.

    Nếu văn bản không có ``\\n\\n`` nào thì trả về một đoạn duy nhất (văn bản
    liệt kê, ví dụ ``provenance``), để chiến lược ``semantic`` có thể tự
    tách tiếp theo câu.
    """
    if not text or not text.strip():
        return []
    blocks = [block.strip() for block in text.replace("\r\n", "\n").split("\n\n")]
    return [block for block in blocks if block]


def chunk_sentences(
    text: str,
    max_words: int = 180,
    overlap_sentences: int = 1,
) -> list[str]:
    """C1 (baseline): tách theo câu, gộp tới ``max_words`` từ.

    Chunk liên tiếp chia sẻ ``overlap_sentences`` câu cuối với chunk sau.
    Đây là hàm cũ (T1.3) được đặt lại ở đây để ba chiến lược dùng chung
    interface; ``artrag.data.chunking.chunk_sentences`` giữ nguyên để không
    phá code cũ.
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


def _split_keep_separator(text: str, separator: str) -> list[tuple[str, str]]:
    """Tách văn bản theo ``separator``, **giữ lại** dấu phân cách.

    Trả về danh sách cặp ``(nội dung, dấu_phân_cách_đứng_sau)``. Nhờ vậy
    ghép lại các mục (hoặc các chunk) luôn khớp **nguyên văn** với văn bản
    gốc, kể cả khi ranh giới chunk nằm đúng ở một dấu phân cách.
    """
    core = separator.rstrip()
    if not core:
        return [(text, "")]

    # Cho phép khoảng trắng linh hoạt sau dấu phân cách ("; ", ";  ", ";\n").
    pattern = re.compile(f"({re.escape(core)}\\s*)")
    parts = pattern.split(text)

    cells: list[tuple[str, str]] = []
    for index in range(0, len(parts), 2):
        content = parts[index]
        sep = parts[index + 1] if index + 1 < len(parts) else ""
        cells.append((content, sep))
    return cells


def chunk_items(
    text: str,
    max_words: int = 250,
    separator: str = "; ",
) -> list[str]:
    """C2: tách theo **mục tự nhiên**, không cắt giữa một mục.

    Dùng cho các field có cấu trúc danh sách (``provenance``, ``exhibitions``),
    vốn được nối bằng ``"; "``. Mỗi mục là một đơn vị nguyên tử; chỉ khi
    một mục đơn lẻ vượt ``max_words`` mới bị cắt cứng theo từ.

    Với chiến lược này, **không có mảnh nào kết thúc giữa một mục** (trừ
    trường hợp một mục đơn lẻ quá dài).

    Bất biến: nối lại các chunk bằng ``" "`` phải cho ra đúng văn bản gốc
    (sau khi chuẩn hóa khoảng trắng). Dấu phân cách nằm ở ranh giới giữa
    hai chunk được giữ lại (kèm khoảng trắng gốc) ở cuối chunk trước.
    """
    if not text or not text.strip():
        return []

    cells = _split_keep_separator(text, separator)

    # Đơn vị cấp thành chunk: (nội dung thô, dấu_phân_cách_sau, là_đầu_mục).
    # "Nội dung thô" giữ nguyên khoảng trắng gốc để nối lại được nguyên văn.
    units: list[tuple[str, str, bool]] = []
    for content, sep in cells:
        if not content.strip():
            # Ô rỗng: chỉ giữ dấu phân cách, gộp vào ô trước.
            if sep and units:
                previous = units[-1]
                units[-1] = (previous[0], previous[1] + sep, previous[2])
            continue

        if count_words(content) > max_words:
            # Mục đơn lẻ quá dài -> cắt cứng theo từ (không có cách nào khác).
            # Tách theo "từ + khoảng trắng đi kèm" để không mất khoảng trắng gốc.
            tokens = re.findall(r"\S+\s*", content)
            total = len(tokens)
            for start in range(0, total, max_words):
                piece = "".join(tokens[start : start + max_words])
                is_last = start + max_words >= total
                units.append((piece, sep if is_last else "", start == 0))
            continue

        units.append((content, sep, True))

    if not units:
        return []

    groups: list[list[tuple[str, str, bool]]] = []
    current: list[tuple[str, str, bool]] = []
    current_words = 0

    for unit in units:
        unit_words = count_words(unit[0])
        if current_words + unit_words > max_words and current:
            groups.append(current)
            current, current_words = [], 0
        current.append(unit)
        current_words += unit_words
        if not unit[2]:
            # Mục này đã bị cắt cứng -> buộc phải ngắt chunk tại đây.
            groups.append(current)
            current, current_words = [], 0

    if current:
        groups.append(current)

    chunks: list[str] = []
    for group in groups:
        chunk = "".join(f"{content}{sep}" for content, sep, _ in group).strip()
        if chunk:
            chunks.append(chunk)

    return [chunk for chunk in chunks if chunk.strip()]


def chunk_semantic(
    text: str,
    target_words: int = 120,
    max_words: int = 250,
    min_words: int = 40,
) -> list[str]:
    """C3: tách theo **đoạn văn có nghĩa**, không overlap.

    - Gom các đoạn văn (theo ``\\n\\n``) vào chunk cho tới ``target_words``.
    - **Không cắt giữa đoạn**; nếu một đoạn quá lớn thì tách tiếp theo câu.
    - ``min_words`` là ngưỡng bỏ qua: đoạn quá ngắn được gộp vào đoạn kế
      (không tạo chunk riêng quá nhỏ).
    """
    if not text or not text.strip():
        return []

    # đơn vị cơ bản: đoạn văn; đoạn quá dài thì tách theo câu
    units: list[str] = []
    for block in split_paragraphs(text):
        if count_words(block) <= max_words:
            units.append(block)
            continue
        units.extend(chunk_sentences(block, max_words=max_words, overlap_sentences=0))

    if not units:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for unit in units:
        unit_words = count_words(unit)
        if current_words + unit_words > target_words and current:
            chunks.append("\n\n".join(current))
            current, current_words = [], 0
        current.append(unit)
        current_words += unit_words

    if current:
        chunk = "\n\n".join(current).strip()
        if chunk:
            chunks.append(chunk)

    # gộp các chunk quá ngắn vào chunk trước (tránh chunk 1-2 từ)
    if min_words <= 0:
        return [c for c in chunks if c.strip()]

    merged: list[str] = []
    for chunk in chunks:
        if merged and count_words(merged[-1]) < min_words:
            merged[-1] = f"{merged[-1]}\n\n{chunk}".strip()
        else:
            merged.append(chunk)
    return [c for c in merged if c.strip()]


def apply_strategy(
    text: str,
    strategy: str = STRATEGY_SENTENCE,
    sentence_max_words: int = 180,
    overlap_sentences: int = 1,
    field_max_words: int = 250,
    item_separator: str = "; ",
    semantic_target_words: int = 120,
    semantic_max_words: int = 250,
    semantic_min_words: int = 40,
) -> list[str]:
    """Áp dụng một chiến lược lên văn bản (interface chung cho C1/C2/C3)."""
    if strategy == STRATEGY_ITEM:
        return chunk_items(text, max_words=field_max_words, separator=item_separator)
    if strategy == STRATEGY_SEMANTIC:
        return chunk_semantic(
            text,
            target_words=semantic_target_words,
            max_words=semantic_max_words,
            min_words=semantic_min_words,
        )
    if strategy == STRATEGY_SENTENCE:
        return chunk_sentences(
            text,
            max_words=sentence_max_words,
            overlap_sentences=overlap_sentences,
        )
    raise ValueError(f"chiến lược chunking không hợp lệ: {strategy!r}")


def make_chunk_rows_for_strategy(
    row: dict[str, Any],
    strategy: str,
    field: str,
    text: str,
    *,
    sentence_max_words: int = 180,
    overlap_sentences: int = 1,
    field_max_words: int = 250,
    item_separator: str = "; ",
    semantic_target_words: int = 120,
    semantic_max_words: int = 250,
    semantic_min_words: int = 40,
    max_words_per_chunk: int = 250,
    chunk_id_prefix: str | None = None,
) -> list[dict[str, Any]]:
    """Sinh các dòng chunk cho một field theo một chiến lược.

    ``chunk_id`` = ``{chunk_id_prefix}#{field}#{i}``; mặc định ``chunk_id_prefix``
    là ``artwork_id`` (với chunk tác phẩm) hoặc ``artist_id`` (với chunk tiểu sử).
    """
    artwork_id = row.get("artwork_id")
    artist_id = row.get("artist_id")
    source_url = row.get("source_url")
    if chunk_id_prefix is None:
        chunk_id_prefix = str(artwork_id) if artwork_id else str(artist_id)

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

    rows: list[dict[str, Any]] = []
    for index, piece in enumerate(pieces):
        piece = piece.strip()
        if not piece:
            continue
        words = piece.split()
        if len(words) > max_words_per_chunk:
            # Giữ dấu phân cách ở cuối nếu có (chiến lược `item`).
            tail = piece[-1] if piece.endswith(";") else ""
            piece = " ".join(words[:max_words_per_chunk]) + tail
        rows.append(
            {
                "chunk_id": f"{chunk_id_prefix}#{field}#{index}",
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
