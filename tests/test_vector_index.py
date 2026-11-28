"""Test cho T3.1 ``artrag.index.vector_index``.

Tất cả test chạy **không cần GPU, không cần model**: chỉ dùng vector numpy
ngẫu nhiên và FAISS ``IndexFlatIP``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from artrag.errors import StaleIndexError
from artrag.index.vector_index import (
    IDS_FILENAME,
    META_FILENAME,
    VECTORS_FILENAME,
    VectorIndex,
    build_meta,
    compute_kb_hash,
    l2_normalize,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _random_unit_vectors(n: int = 12, dim: int = 8, seed: int = 1234) -> np.ndarray:
    """Ma trận vector ngẫu nhiên đã L2-normalize (float32)."""
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(n, dim)).astype(np.float32)
    return l2_normalize(raw)


def _meta(kb_hash: str = "hash-abc", **extra) -> dict:
    payload = build_meta(
        model="clip",
        revision="rev-1",
        dim=8,
        n=12,
        kb_hash=kb_hash,
        build_seconds=1.25,
    )
    payload.update(extra)
    return payload


# ---------------------------------------------------------------------------
# l2_normalize
# ---------------------------------------------------------------------------


def test_l2_normalize_rows_have_unit_norm():
    raw = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    out = l2_normalize(raw)
    assert out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)


def test_l2_normalize_keeps_zero_row_without_nan():
    raw = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
    out = l2_normalize(raw)
    assert not np.isnan(out).any()
    assert np.allclose(out[0], [0.0, 0.0])


def test_l2_normalize_accepts_1d_vector():
    out = l2_normalize(np.array([3.0, 4.0], dtype=np.float32))
    assert out.shape == (1, 2)
    assert np.isclose(float(np.linalg.norm(out[0])), 1.0, atol=1e-6)


def test_l2_normalize_rejects_3d_array():
    with pytest.raises(ValueError):
        l2_normalize(np.zeros((2, 2, 2), dtype=np.float32))


# ---------------------------------------------------------------------------
# build validation
# ---------------------------------------------------------------------------


def test_build_rejects_1d_vectors():
    with pytest.raises(ValueError):
        VectorIndex.build(np.zeros(4, dtype=np.float32), ["a"])


def test_build_rejects_mismatched_ids():
    with pytest.raises(ValueError):
        VectorIndex.build(np.zeros((3, 4), dtype=np.float32), ["a", "b"])


def test_build_fills_dim_and_n_in_meta():
    index = VectorIndex.build(
        np.zeros((5, 7), dtype=np.float32), [f"i{i}" for i in range(5)], meta={}
    )
    assert index.meta["dim"] == 7
    assert index.meta["n"] == 5
    assert index.n == 5
    assert index.dim == 7
    assert len(index) == 5


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------


def test_save_writes_expected_files(tmp_path):
    vectors = _random_unit_vectors()
    ids = [f"cma-{i}" for i in range(vectors.shape[0])]
    index = VectorIndex.build(vectors, ids, meta=_meta())
    target = index.save(tmp_path / "clip")

    assert (target / VECTORS_FILENAME).exists()
    assert (target / IDS_FILENAME).exists()
    assert (target / META_FILENAME).exists()

    saved_vectors = np.load(target / VECTORS_FILENAME)
    assert saved_vectors.dtype == np.float32
    assert np.allclose(saved_vectors, vectors, atol=1e-6)


def test_save_load_round_trip_preserves_vectors_ids_and_meta(tmp_path):
    vectors = _random_unit_vectors()
    ids = [f"cma-{i}" for i in range(vectors.shape[0])]
    meta = _meta()
    index = VectorIndex.build(vectors, ids, meta=meta)
    target = index.save(tmp_path / "clip")

    loaded = VectorIndex.load(target)
    assert loaded.ids == ids
    assert loaded.n == index.n
    assert loaded.dim == index.dim
    assert np.allclose(loaded.vectors, vectors, atol=1e-6)
    assert loaded.meta["kb_hash"] == meta["kb_hash"]
    assert loaded.meta["model"] == "clip"
    assert loaded.meta["revision"] == "rev-1"
    assert loaded.meta["created_at"]
    assert loaded.meta["build_seconds"] == pytest.approx(1.25)


def test_ids_file_keeps_exact_order(tmp_path):
    ids = ["cma-9", "cma-1", "cma-5", "cma-3"]
    index = VectorIndex.build(_random_unit_vectors(len(ids)), ids, meta=_meta())
    target = index.save(tmp_path / "clip")
    payload = json.loads((target / IDS_FILENAME).read_text(encoding="utf-8"))
    assert payload == ids

    loaded = VectorIndex.load(target)
    assert loaded.ids == ids


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        VectorIndex.load(tmp_path / "khong-ton-tai")


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def test_search_top1_is_itself_for_normalized_vector():
    vectors = _random_unit_vectors()
    ids = [f"cma-{i}" for i in range(vectors.shape[0])]
    index = VectorIndex.build(vectors, ids, meta=_meta())

    for position in (0, 3, 7):
        hits = index.search(vectors[position], k=1)
        assert len(hits) == 1
        assert hits[0][0] == ids[position]
        assert hits[0][1] == pytest.approx(1.0, abs=1e-4)


def test_search_returns_expected_id_and_score():
    vectors = _random_unit_vectors()
    ids = [f"cma-{i}" for i in range(vectors.shape[0])]
    index = VectorIndex.build(vectors, ids, meta=_meta())

    query = vectors[2]
    hits = index.search(query, k=3)
    assert len(hits) == 3
    assert hits[0][0] == ids[2]
    assert hits[0][1] == pytest.approx(1.0, abs=1e-4)

    # Điểm là cosine similarity -> bằng tích vô hướng.
    for artwork_id, score in hits:
        position = ids.index(artwork_id)
        expected = float(np.dot(vectors[position], query))
        assert score == pytest.approx(expected, abs=1e-4)


def test_search_orders_by_descending_score():
    vectors = _random_unit_vectors()
    ids = [f"cma-{i}" for i in range(vectors.shape[0])]
    index = VectorIndex.build(vectors, ids, meta=_meta())

    hits = index.search(vectors[1], k=5)
    scores = [score for _, score in hits]
    assert scores == sorted(scores, reverse=True)


def test_search_with_k_greater_than_n_does_not_fail():
    vectors = _random_unit_vectors(n=4)
    ids = [f"cma-{i}" for i in range(4)]
    index = VectorIndex.build(vectors, ids, meta=_meta())

    hits = index.search(vectors[0], k=100)
    assert len(hits) == 4
    assert {artwork_id for artwork_id, _ in hits} == set(ids)
    assert hits[0][0] == ids[0]


def test_search_with_k_zero_returns_empty():
    vectors = _random_unit_vectors()
    index = VectorIndex.build(vectors, [f"cma-{i}" for i in range(vectors.shape[0])], meta=_meta())
    assert index.search(vectors[0], k=0) == []


def test_search_on_empty_index_returns_empty():
    index = VectorIndex.build(np.zeros((0, 4), dtype=np.float32), [], meta=_meta())
    assert index.search(np.zeros(4, dtype=np.float32), k=5) == []


def test_search_rejects_wrong_dimension():
    vectors = _random_unit_vectors()
    index = VectorIndex.build(vectors, [f"cma-{i}" for i in range(vectors.shape[0])], meta=_meta())
    with pytest.raises(ValueError):
        index.search(np.zeros(3, dtype=np.float32), k=3)


# ---------------------------------------------------------------------------
# StaleIndexError
# ---------------------------------------------------------------------------


def test_load_with_mismatched_kb_hash_raises_stale(tmp_path):
    vectors = _random_unit_vectors()
    index = VectorIndex.build(vectors, [f"cma-{i}" for i in range(vectors.shape[0])], meta=_meta("hash-a"))
    target = index.save(tmp_path / "clip")

    with pytest.raises(StaleIndexError):
        VectorIndex.load(target, expected_kb_hash="hash-b")


def test_search_with_mismatched_kb_hash_raises_stale(tmp_path):
    vectors = _random_unit_vectors()
    index = VectorIndex.build(vectors, [f"cma-{i}" for i in range(vectors.shape[0])], meta=_meta("hash-a"))
    target = index.save(tmp_path / "clip")
    loaded = VectorIndex.load(target)

    with pytest.raises(StaleIndexError):
        loaded.search(vectors[0], k=3, expected_kb_hash="hash-b")


def test_search_with_matching_kb_hash_is_fine(tmp_path):
    vectors = _random_unit_vectors()
    index = VectorIndex.build(vectors, [f"cma-{i}" for i in range(vectors.shape[0])], meta=_meta("hash-a"))
    target = index.save(tmp_path / "clip")
    loaded = VectorIndex.load(target)

    hits = loaded.search(vectors[0], k=3, expected_kb_hash="hash-a")
    assert hits[0][0] == "cma-0"


def test_stale_error_message_mentions_both_hashes():
    index = VectorIndex.build(
        np.zeros((2, 3), dtype=np.float32), ["a", "b"], meta=_meta("hash-a")
    )
    with pytest.raises(StaleIndexError) as excinfo:
        index.check_freshness("hash-b")
    message = str(excinfo.value)
    assert "hash-a" in message
    assert "hash-b" in message


# ---------------------------------------------------------------------------
# kb_hash
# ---------------------------------------------------------------------------


def test_compute_kb_hash_returns_none_without_kb(tmp_path):
    assert compute_kb_hash(tmp_path) is None


def test_compute_kb_hash_is_stable_and_matches_artifacts(tmp_path):
    import pandas as pd

    from artrag.tools.artifacts import kb_fingerprint

    data_dir = tmp_path / "data"
    kb = data_dir / "kb"
    kb.mkdir(parents=True)
    pd.DataFrame(
        [
            {"artwork_id": "cma-1", "image_path": "images/cma-1.jpg"},
            {"artwork_id": "cma-2", "image_path": None},
        ]
    ).to_parquet(kb / "artworks.parquet", engine="pyarrow", index=False)

    first = compute_kb_hash(data_dir)
    second = compute_kb_hash(data_dir)
    assert first is not None
    assert first == second
    # Phải khớp đúng hash mà `artifacts pack` ghi vào manifest.json.
    assert first == kb_fingerprint(data_dir)


# ---------------------------------------------------------------------------
# build_meta
# ---------------------------------------------------------------------------


def test_build_meta_contains_required_keys():
    meta = build_meta(
        model="clip",
        revision="abc",
        dim=512,
        n=2210,
        kb_hash="hash-1",
        build_seconds=12.3456,
    )
    for key in (
        "model",
        "revision",
        "dim",
        "n",
        "kb_hash",
        "created_at",
        "build_seconds",
    ):
        assert key in meta
    assert meta["dim"] == 512
    assert meta["n"] == 2210
    assert meta["kb_hash"] == "hash-1"
    assert meta["build_seconds"] == pytest.approx(12.346, abs=1e-3)
