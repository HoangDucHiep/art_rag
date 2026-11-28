"""Test cho T3.2 ``artrag.index.text_index``.

Tất cả test chạy thuần **không cần GPU, không cần model thật**: dùng dữ liệu
mô phỏng hoặc mock loader/inference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from artrag.config import Config
from artrag.index.text_index import (
    batch_size_for,
    build_index,
    check_index,
    chunks_path,
    format_snippet,
    index_dir,
    parse_ids,
    search_text,
    select_chunks,
)
from artrag.index.vector_index import (
    IDS_FILENAME,
    META_FILENAME,
    VECTORS_FILENAME,
    build_meta,
)


# ---------------------------------------------------------------------------
# Tiện ích test
# ---------------------------------------------------------------------------


def _dummy_config(tmp_path: Path) -> Config:
    """Tạo Config trỏ vào thư mục tạm `tmp_path`."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return Config(
        {
            "paths": {
                "data_dir": str(data_dir),
                "results_dir": str(tmp_path / "results"),
            },
            "index": {
                "text_model": "text_embed",
                "batch_size": 64,
                "batch_size_cpu": 8,
            },
        }
    )


def _make_dummy_chunks(path: Path, count: int = 5) -> pd.DataFrame:
    """Tạo file chunk mẫu."""
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(count):
        records.append(
            {
                "chunk_id": f"cma-{100 + i}#overview#0",
                "artwork_id": f"cma-{100 + i}",
                "artist_id": f"artist-{i}",
                "field": "overview",
                "text": f"Mô tả tác phẩm số {i} với nội dung mẫu dài hơn một chút.",
                "source_url": f"https://example.com/{100 + i}",
                "n_words": 12,
                "strategy": "sentence",
            }
        )
    # Thêm một chunk có text rỗng
    records.append(
        {
            "chunk_id": "cma-empty#overview#0",
            "artwork_id": "cma-empty",
            "artist_id": "artist-empty",
            "field": "overview",
            "text": "   ",
            "source_url": "https://example.com/empty",
            "n_words": 0,
            "strategy": "sentence",
        }
    )
    df = pd.DataFrame(records)
    df.to_parquet(path, index=False)
    return df


@dataclass
class DummyHandle:
    name: str = "text_embed"
    task: str = "sbert"
    device: str = "cpu"
    dtype: str = "fp32"
    model: Any = None
    processor: Any = None


# ---------------------------------------------------------------------------
# Tiện ích thuần
# ---------------------------------------------------------------------------


def test_index_dir_and_chunks_path(tmp_path: Path):
    cfg = _dummy_config(tmp_path)
    idx_dir = index_dir(cfg)
    assert idx_dir.name == "text"
    assert idx_dir.parent.name == "index"

    base_chunks = chunks_path(cfg, variants=False)
    assert base_chunks.name == "chunks.parquet"

    var_chunks = chunks_path(cfg, variants=True)
    assert var_chunks.name == "chunks_variants.parquet"


def test_batch_size_for(tmp_path: Path):
    cfg = _dummy_config(tmp_path)
    # CPU: lấy từ `index.batch_size_cpu` trong config (8 trong config mẫu).
    assert batch_size_for(cfg, "cpu") == 8
    # GPU: lấy giá trị trong config (64)
    assert batch_size_for(cfg, "cuda") == 64
    # Override
    assert batch_size_for(cfg, "cpu", override=32) == 32
    assert batch_size_for(cfg, "cuda", override=8) == 8


def test_parse_ids():
    assert parse_ids(None) is None
    assert parse_ids([]) is None
    assert parse_ids("") is None
    assert parse_ids("cma-1") == ["cma-1"]
    assert parse_ids("cma-1, cma-2") == ["cma-1", "cma-2"]
    assert parse_ids(["cma-1", "cma-2, cma-3"]) == ["cma-1", "cma-2", "cma-3"]


def test_select_chunks(tmp_path: Path):
    parquet_path = tmp_path / "chunks.parquet"
    df = _make_dummy_chunks(parquet_path, count=5)

    # 1. Bỏ qua chunk rỗng (count=5 thay vì 6)
    all_valid = select_chunks(df)
    assert len(all_valid) == 5

    # 2. Giới hạn limit
    limited = select_chunks(df, limit=3)
    assert len(limited) == 3

    # 3. Lọc theo artwork_ids
    filtered = select_chunks(df, artwork_ids=["cma-102"])
    assert len(filtered) == 1
    assert filtered[0]["artwork_id"] == "cma-102"


def test_format_snippet():
    assert format_snippet("  Hello   world \n test  ") == "Hello world test"
    short_text = "Ngắn"
    assert format_snippet(short_text, max_chars=10) == "Ngắn"
    long_text = "a" * 150
    snippet = format_snippet(long_text, max_chars=120)
    assert len(snippet) == 120
    assert snippet.endswith("...")


# ---------------------------------------------------------------------------
# check_index
# ---------------------------------------------------------------------------


def test_check_index_missing(tmp_path: Path):
    cfg = _dummy_config(tmp_path)
    stats = check_index(cfg)
    assert stats["status"] == "missing"


def test_check_index_ok_and_stale(tmp_path: Path):
    cfg = _dummy_config(tmp_path)
    target = index_dir(cfg)
    target.mkdir(parents=True, exist_ok=True)

    # Mock kb_fingerprint trả về hash cố định
    with patch("artrag.index.text_index.compute_kb_hash", return_value="hash-123"):
        meta = build_meta(
            model="text_embed",
            revision="rev1",
            dim=1024,
            n=10,
            kb_hash="hash-123",
            build_seconds=1.0,
        )
        (target / META_FILENAME).write_text(json.dumps(meta), encoding="utf-8")

        stats = check_index(cfg)
        assert stats["status"] == "ok"
        assert stats["n"] == 10
        assert stats["dim"] == 1024

    # Hash lệch -> stale
    with patch("artrag.index.text_index.compute_kb_hash", return_value="hash-different"):
        stats = check_index(cfg)
        assert stats["status"] == "stale"


# ---------------------------------------------------------------------------
# build_index & search_text (với mock model)
# ---------------------------------------------------------------------------


def test_build_and_search_index(tmp_path: Path):
    cfg = _dummy_config(tmp_path)
    file_path = chunks_path(cfg, variants=False)
    _make_dummy_chunks(file_path, count=4)

    mock_registry = MagicMock()
    mock_handle = DummyHandle()
    mock_registry.get.return_value = mock_handle
    mock_registry.specs = {"text_embed": MagicMock(hf_id="BAAI/bge-m3")}

    # Mock embed_text trả về vector unit cố định (dim 4)
    def fake_embed(handle, text):
        return np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)

    with (
        patch("artrag.registry.create_registry", return_value=mock_registry),
        patch("artrag.models.inference.embed_text", side_effect=fake_embed),
        patch("artrag.index.text_index.compute_kb_hash", return_value="test-kb-hash"),
    ):
        # Build index
        stats = build_index(cfg, limit=3, force=True, batch_size=2)
        assert stats["status"] == "built"
        assert stats["n"] == 3
        assert stats["dim"] == 4

        target = index_dir(cfg)
        assert (target / VECTORS_FILENAME).exists()
        assert (target / IDS_FILENAME).exists()
        assert (target / META_FILENAME).exists()

        # Idempotent: chạy lại mà không có --force thì trả "exists"
        again = build_index(cfg, limit=3, force=False)
        assert again["status"] == "exists"

        # Search
        results = search_text(cfg, "câu hỏi mẫu", top_k=2)
        assert len(results) == 2
        assert results[0]["rank"] == 1
        assert "chunk_id" in results[0]
        assert "artwork_id" in results[0]
        assert "field" in results[0]
        assert "score" in results[0]
        assert "text" in results[0]
        assert results[0]["score"] == pytest.approx(1.0, abs=1e-4)
