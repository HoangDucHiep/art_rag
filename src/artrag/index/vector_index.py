"""T3.1: `VectorIndex` — lưu và tìm kiếm vector FAISS (SYSTEM_DESIGN §4 bước 6–7).

Bố cục một index trên đĩa (khớp §2.4):

    <dir>/emb.npy        # float32 (n, dim) — vectors đã L2-normalize
    <dir>/ids.json       # list[str] — id theo **đúng thứ tự** vector
    <dir>/meta.json      # model, revision, dim, n, kb_hash, created_at, build_seconds

Tìm kiếm dùng FAISS `IndexFlatIP`: vì vector đã L2-normalize nên điểm inner
product chính là **cosine similarity** (không cần chia lại norm).

Vì sao lưu `emb.npy` bên cạnh file FAISS: `IndexFlatIP` với ~2.210 vector tìm
kiếm chính xác dưới 1 ms nên không cần tối ưu; giữ `.npy` giúp việc nạp lại
nhanh, kiểm tra `dim`/`n` và dùng lại được cho ablation (§4 ghi chú mở rộng).

Quy ước bất biến:
- `ids.json` **giữ nguyên thứ tự** của `emb.npy` (id thứ i ứng với vector thứ i).
- `meta["kb_hash"]` phải khớp `kb_fingerprint()` của KB hiện tại, nếu lệch thì
  `search`/`load` ném `StaleIndexError` kèm lệnh build lại.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from artrag.errors import StaleIndexError

#: Tên file chuẩn trong một thư mục index.
VECTORS_FILENAME = "emb.npy"
IDS_FILENAME = "ids.json"
META_FILENAME = "meta.json"

#: Các khoá bắt buộc có trong `meta.json` (theo §2.4).
REQUIRED_META_KEYS: tuple[str, ...] = (
    "model",
    "revision",
    "dim",
    "n",
    "kb_hash",
    "created_at",
    "build_seconds",
)

#: Thông báo gợi ý khi index bị cũ.
_STALE_HINT = (
    "Chạy lại: uv run python -m artrag.index.clip_index build --force"
)


# ---------------------------------------------------------------------------
# Hàm thuần
# ---------------------------------------------------------------------------


def l2_normalize(array: np.ndarray) -> np.ndarray:
    """Chuẩn hoá L2 mỗi hàng của ma trận (hoặc một vector 1D).

    - Input phải là `float32`/`float64`; kết quả trả về `float32`.
    - Hàng có độ dài bằng 0 được giữ nguyên (không chia cho 0).
    """
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"l2_normalize cần mảng 1D hoặc 2D, nhận {arr.ndim} chiều.")

    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    # Hàng toàn 0 -> để nguyên (tránh NaN do chia cho 0).
    safe = np.where(norms > 0.0, norms, 1.0)
    return (arr / safe).astype(np.float32, copy=False)


def compute_kb_hash(data_dir: str | Path) -> str | None:
    """Hash ổn định của KB hiện tại.

    Tái sử dụng `artrag.tools.artifacts.kb_fingerprint` để **không viết lại**
    logic hash (và để `kb_hash` trong `meta.json` khớp đúng hash mà
    `artifacts pack` ghi vào `manifest.json`).

    Trả ``None`` nếu chưa có ``kb/artworks.parquet``.
    """
    from artrag.tools.artifacts import kb_fingerprint  # noqa: PLC0415 - tránh vòng import

    return kb_fingerprint(data_dir)


# ---------------------------------------------------------------------------
# VectorIndex
# ---------------------------------------------------------------------------


@dataclass
class VectorIndex:
    """Một index vector: ma trận float32 + danh sách id + metadata.

    Dùng như sau::

        index = VectorIndex.build(vectors, ids, meta={"model": ..., "kb_hash": ...})
        index.save(dir_path)
        index = VectorIndex.load(dir_path)
        hits = index.search(query, k=5)
    """

    vectors: np.ndarray
    ids: list[str]
    meta: dict[str, Any] = field(default_factory=dict)
    _index: Any = field(default=None, repr=False, compare=False)

    # -- dựng ------------------------------------------------------------
    @classmethod
    def build(
        cls,
        vectors: np.ndarray,
        ids: Sequence[str],
        *,
        meta: dict[str, Any] | None = None,
    ) -> "VectorIndex":
        """Dựng index từ ma trận vector và danh sách id.

        - `vectors` phải là 2D `(n, dim)`, được ép về `float32`.
        - `ids` phải có **đúng** `n` phần tử.
        - `meta` sẽ được bổ sung `dim`/`n` nếu chưa có.
        """
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(
                f"vectors phải là ma trận 2D (n, dim), nhận {arr.ndim} chiều."
            )
        id_list = [str(item) for item in ids]
        if len(id_list) != arr.shape[0]:
            raise ValueError(
                f"Số id ({len(id_list)}) không khớp số vector ({arr.shape[0]})."
            )

        payload: dict[str, Any] = dict(meta or {})
        payload.setdefault("dim", int(arr.shape[1]))
        payload.setdefault("n", int(arr.shape[0]))

        return cls(vectors=arr, ids=id_list, meta=payload)

    # -- lưu / đọc -------------------------------------------------------
    def save(self, directory: str | Path) -> Path:
        """Ghi index vào `directory` (tạo thư mục nếu chưa có)."""
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)

        np.save(target / VECTORS_FILENAME, self.vectors.astype(np.float32, copy=False))

        ids_path = target / IDS_FILENAME
        ids_path.write_text(
            json.dumps(self.ids, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        meta = dict(self.meta)
        meta.setdefault("dim", int(self.vectors.shape[1]))
        meta.setdefault("n", int(self.vectors.shape[0]))
        meta_path = target / META_FILENAME
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        expected_kb_hash: str | None = None,
    ) -> "VectorIndex":
        """Đọc index từ `directory`.

        - Ném `FileNotFoundError` nếu thiếu file bắt buộc.
        - Nếu `expected_kb_hash` được truyền và khác với `meta["kb_hash"]`
          thì ném `StaleIndexError`.
        """
        source = Path(directory)
        vectors_path = source / VECTORS_FILENAME
        ids_path = source / IDS_FILENAME
        meta_path = source / META_FILENAME

        missing = [
            str(path)
            for path in (vectors_path, ids_path, meta_path)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Index thiếu file: " + ", ".join(missing) + f" (thư mục {source})"
            )

        vectors = np.load(vectors_path).astype(np.float32, copy=False)
        ids = json.loads(ids_path.read_text(encoding="utf-8"))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(ids, list):
            raise ValueError(f"{ids_path} phải chứa list id.")
        if not isinstance(meta, dict):
            raise ValueError(f"{meta_path} phải chứa object.")

        index = cls(
            vectors=vectors,
            ids=[str(item) for item in ids],
            meta=meta,
        )
        if expected_kb_hash is not None:
            index.check_freshness(expected_kb_hash)
        return index

    # -- tìm kiếm --------------------------------------------------------
    def _faiss_index(self) -> Any:
        """Tạo (hoặc dùng lại) FAISS `IndexFlatIP` cho ma trận vector."""
        if self._index is not None:
            return self._index

        import faiss  # noqa: PLC0415 - import cục bộ để test không cần faiss

        index = faiss.IndexFlatIP(int(self.vectors.shape[1]))
        if self.vectors.shape[0]:
            index.add(np.ascontiguousarray(self.vectors, dtype=np.float32))
        self._index = index
        return index

    def search(
        self,
        query: np.ndarray,
        k: int = 5,
        *,
        expected_kb_hash: str | None = None,
    ) -> list[tuple[str, float]]:
        """Tìm `k` vector gần nhất với `query`.

        - `query` là vector 1D đã L2-normalize (điểm trả về là cosine).
        - Trả về list `(id, score)` sắp theo điểm giảm dần.
        - `k` lớn hơn số vector vẫn trả về toàn bộ (không lỗi).
        """
        if expected_kb_hash is not None:
            self.check_freshness(expected_kb_hash)

        if self.vectors.shape[0] == 0 or k <= 0:
            return []

        vec = np.asarray(query, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.vectors.shape[1]:
            raise ValueError(
                f"Query có dim {vec.shape[0]} nhưng index dùng dim {self.vectors.shape[1]}."
            )

        index = self._faiss_index()
        top_k = min(int(k), int(self.vectors.shape[0]))
        scores, positions = index.search(np.ascontiguousarray(vec.reshape(1, -1)), top_k)
        results: list[tuple[str, float]] = []
        for score, position in zip(scores[0].tolist(), positions[0].tolist()):
            if position < 0:
                continue
            results.append((self.ids[position], float(score)))
        return results

    # -- kiểm tra độ tươi (kb_hash) -------------------------------------
    def check_freshness(self, expected_kb_hash: str | None) -> None:
        """Ném `StaleIndexError` nếu `kb_hash` của index lệch với KB hiện tại."""
        if not expected_kb_hash:
            return
        current = str(self.meta.get("kb_hash") or "")
        if current == expected_kb_hash:
            return
        raise StaleIndexError(
            f"Index không khớp KB hiện tại (kb_hash={current or '<rỗng>'} "
            f"≠ {expected_kb_hash}).",
            hint=_STALE_HINT,
        )

    # -- tiện ích --------------------------------------------------------
    @property
    def n(self) -> int:
        """Số vector trong index."""
        return int(self.vectors.shape[0]) if self.vectors.ndim == 2 else 0

    @property
    def dim(self) -> int:
        """Số chiều của mỗi vector."""
        return int(self.vectors.shape[1]) if self.vectors.ndim == 2 else 0

    @property
    def kb_hash(self) -> str:
        """`kb_hash` lưu trong `meta.json` (chuỗi rỗng nếu không có)."""
        return str(self.meta.get("kb_hash") or "")

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        return (
            f"VectorIndex(n={self.n}, dim={self.dim}, "
            f"kb_hash={self.kb_hash or '<rỗng>'})"
        )


# ---------------------------------------------------------------------------
# Meta builder
# ---------------------------------------------------------------------------


def build_meta(
    *,
    model: str,
    revision: str | None,
    dim: int,
    n: int,
    kb_hash: str | None,
    build_seconds: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Dựng nội dung `meta.json` theo §2.4.

    Gồm: `model`, `revision`, `dim`, `n`, `kb_hash`, `created_at`, `build_seconds`.
    """
    meta: dict[str, Any] = {
        "model": model,
        "revision": revision or "",
        "dim": int(dim),
        "n": int(n),
        "kb_hash": kb_hash or "",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "build_seconds": round(float(build_seconds), 3),
    }
    if extra:
        meta.update(extra)
    return meta
