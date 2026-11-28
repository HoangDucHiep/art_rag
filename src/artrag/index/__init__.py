"""Index offline: vector FAISS cho CLIP (bước 6) và BGE-M3 (bước 7).

Module `artrag.index.vector_index` chứa:
- `VectorIndex`: lưu vector float32 (`.npy`) + danh sách id (`ids.json`) + metadata
  (`meta.json`), tìm kiếm bằng FAISS `IndexFlatIP` (vectors đã L2-normalize nên
  điểm là cosine similarity).
- Các hàm thuần: `l2_normalize`, `compute_kb_hash` — test được **không cần GPU**.

Lệnh:
    uv run python -m artrag.index.clip_index build --limit 100
    uv run python -m artrag.index.clip_index check
    uv run python -m artrag.index.clip_index search "two men rowing a boat" --top 5
"""

from __future__ import annotations

from artrag.index.vector_index import (
    VectorIndex,
    compute_kb_hash,
    l2_normalize,
)

__all__ = ["VectorIndex", "compute_kb_hash", "l2_normalize"]
