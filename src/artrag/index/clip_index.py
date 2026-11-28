"""T3.1: dựng và tra cứu index ảnh CLIP (SYSTEM_DESIGN §4 bước 6).

Nguồn: ``data/kb/artworks.parquet`` (các dòng có ``image_path`` không null và
file ảnh tồn tại) → mỗi ảnh embed bằng CLIP ViT-B/32 → vector L2-normalize →
FAISS ``IndexFlatIP``.

Kết quả nằm ở ``data/index/clip/``:
    emb.npy | ids.json | meta.json

Idempotent: nếu index đã tồn tại và ``kb_hash`` khớp KB hiện tại thì lệnh
``build`` bỏ qua (in "đã tồn tại"), trừ khi dùng ``--force``.

Lệnh:
    uv run python -m artrag.index.clip_index build --limit 100
    uv run python -m artrag.index.clip_index check
    uv run python -m artrag.index.clip_index search "two men rowing a boat" --top 5
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from artrag.config import Config, load_config, paths
from artrag.errors import ArtragError, StaleIndexError
from artrag.index.vector_index import (
    META_FILENAME,
    VectorIndex,
    build_meta,
    compute_kb_hash,
    l2_normalize,
)
from artrag.models.templates import DEFAULT_CLIP_TEMPLATE, render_clip_text

#: Thư mục con của ``data/index/`` chứa index CLIP.
CLIP_INDEX_SUBDIR = "clip"

#: Batch size mặc định cho GPU / CPU (đọc từ config `index.batch_size`).
DEFAULT_BATCH_SIZE = 64
DEFAULT_CPU_BATCH_SIZE = 16

#: Số kết quả mặc định của lệnh `search`.
DEFAULT_TOP_K = 5


# ---------------------------------------------------------------------------
# Tiện ích
# ---------------------------------------------------------------------------


def index_dir(cfg: Config) -> Path:
    """Thư mục chứa index CLIP (``data/index/clip/``)."""
    return paths(cfg).index / CLIP_INDEX_SUBDIR


def _cfg_get(cfg: Config, key: str, default: Any) -> Any:
    value = cfg.get(f"index.{key}", None)
    return default if value is None else value


def batch_size_for(cfg: Config, device: str) -> int:
    """Batch size theo thiết bị: `index.batch_size` cho GPU, 16 cho CPU."""
    configured = int(_cfg_get(cfg, "batch_size", DEFAULT_BATCH_SIZE) or DEFAULT_BATCH_SIZE)
    if str(device).startswith("cpu"):
        return max(1, min(configured, DEFAULT_CPU_BATCH_SIZE))
    return max(1, configured)


def model_revision(handle: Any) -> str:
    """Lấy revision (commit hash) của checkpoint nếu biết; nếu không thì rỗng."""
    config = getattr(handle.model, "config", None)
    for attribute in ("_commit_hash", "commit_hash", "_name_or_path"):
        value = getattr(config, attribute, None) if config is not None else None
        if value:
            return str(value)
    return ""


def select_artworks(
    frame: pd.DataFrame,
    data_dir: Path,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Chọn các dòng có `image_path` không null **và** file ảnh thật sự tồn tại."""
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict("records"):
        image_path = record.get("image_path")
        artwork_id = record.get("artwork_id")
        if not artwork_id or not image_path:
            continue
        absolute = Path(data_dir) / str(image_path)
        if not absolute.exists():
            continue
        rows.append(record)
        if limit is not None and len(rows) >= int(limit):
            break
    return rows


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_index(
    cfg: Config,
    limit: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Dựng index ảnh CLIP; trả về thống kê.

    - Bỏ qua (idempotent) nếu index đã có và `kb_hash` khớp, trừ khi `force`.
    - Batch size theo thiết bị: `index.batch_size` (GPU) / 16 (CPU).
    """
    from artrag.models import loaders  # noqa: PLC0415,F401 - đăng ký loader
    from artrag.models.inference import embed_image
    from artrag.registry import create_registry

    p = paths(cfg)
    kb_path = p.kb / "artworks.parquet"
    if not kb_path.exists():
        raise ArtragError(
            f"Không tìm thấy {kb_path}.",
            hint="Chạy: uv run python -m artrag.data.build_kb",
        )

    kb_hash = compute_kb_hash(p.data_dir)
    target = index_dir(cfg)

    if target.exists() and not force:
        existing = _try_load_meta(target)
        if existing and existing.get("kb_hash") == (kb_hash or ""):
            return {
                "status": "exists",
                "dir": str(target),
                "n": int(existing.get("n", 0)),
                "dim": int(existing.get("dim", 0)),
                "kb_hash": kb_hash or "",
                "model": str(existing.get("model", "")),
                "build_seconds": float(existing.get("build_seconds", 0.0)),
            }

    frame = pd.read_parquet(kb_path)
    rows = select_artworks(frame, p.data_dir, limit=limit)
    if not rows:
        raise ArtragError(
            "Không có tác phẩm nào có file ảnh trong KB.",
            hint="Chạy: uv run python -m artrag.data.images",
        )

    registry = create_registry(cfg)
    model_name = str(_cfg_get(cfg, "clip_model", "clip"))
    handle = registry.get(model_name)

    started = time.perf_counter()
    vectors: list[np.ndarray] = []
    ids: list[str] = []
    batch = batch_size_for(cfg, handle.device)

    for offset in range(0, len(rows), batch):
        chunk = rows[offset : offset + batch]
        for record in chunk:
            image_path = Path(p.data_dir) / str(record["image_path"])
            try:
                from PIL import Image  # noqa: PLC0415 - import cục bộ

                with Image.open(image_path) as image:
                    rgb = image.convert("RGB")
                    vectors.append(np.asarray(embed_image(handle, rgb), dtype=np.float32))
                    ids.append(str(record["artwork_id"]))
            except Exception as exc:  # noqa: BLE001 - ảnh hỏng thì bỏ qua
                print(f"Bỏ qua ảnh lỗi {record.get('artwork_id')}: {exc}")

    if not vectors:
        raise ArtragError(
            "Không embed được ảnh nào.",
            hint="Kiểm tra lại file trong data/images/",
        )

    matrix = l2_normalize(np.vstack(vectors))
    build_seconds = time.perf_counter() - started

    meta = build_meta(
        model=model_name,
        revision=model_revision(handle),
        dim=int(matrix.shape[1]),
        n=int(matrix.shape[0]),
        kb_hash=kb_hash,
        build_seconds=build_seconds,
        extra={
            "device": handle.device,
            "dtype": handle.dtype,
            "batch_size": batch,
            "hf_id": registry.specs[model_name].hf_id,
            "index_type": "IndexFlatIP",
        },
    )

    index = VectorIndex.build(matrix, ids, meta=meta)
    index.save(target)

    return {
        "status": "built",
        "dir": str(target),
        "n": int(matrix.shape[0]),
        "dim": int(matrix.shape[1]),
        "kb_hash": kb_hash or "",
        "model": model_name,
        "build_seconds": build_seconds,
        "device": handle.device,
        "batch_size": batch,
    }


def _try_load_meta(directory: Path) -> dict[str, Any] | None:
    """Đọc `meta.json` nếu có (trả ``None`` nếu thiếu/hỏng)."""
    meta_path = Path(directory) / META_FILENAME
    if not meta_path.exists():
        return None
    import json

    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


# ---------------------------------------------------------------------------
# Check
# ---------------------------------------------------------------------------


def check_index(cfg: Config) -> dict[str, Any]:
    """Kiểm tra trạng thái index CLIP so với KB hiện tại."""
    p = paths(cfg)
    target = index_dir(cfg)
    kb_hash = compute_kb_hash(p.data_dir)

    meta = _try_load_meta(target)
    if meta is None:
        return {
            "status": "missing",
            "dir": str(target),
            "kb_hash_current": kb_hash or "",
            "message": "Chưa có index CLIP. Chạy: python -m artrag.index.clip_index build",
        }

    index_kb_hash = str(meta.get("kb_hash") or "")
    fresh = bool(kb_hash) and index_kb_hash == kb_hash
    return {
        "status": "ok" if fresh else "stale",
        "dir": str(target),
        "n": int(meta.get("n", 0)),
        "dim": int(meta.get("dim", 0)),
        "model": str(meta.get("model", "")),
        "revision": str(meta.get("revision", "")),
        "kb_hash": index_kb_hash,
        "kb_hash_current": kb_hash or "",
        "created_at": str(meta.get("created_at", "")),
        "build_seconds": float(meta.get("build_seconds", 0.0)),
    }


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def search_text(
    cfg: Config,
    text: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """Tìm tác phẩm gần nhất với câu văn bản (nhánh CLIP của pipeline)."""
    from artrag.models import loaders  # noqa: PLC0415,F401 - đăng ký loader
    from artrag.models.inference import embed_text
    from artrag.registry import create_registry

    p = paths(cfg)
    target = index_dir(cfg)
    kb_hash = compute_kb_hash(p.data_dir)
    index = VectorIndex.load(target, expected_kb_hash=kb_hash)

    registry = create_registry(cfg)
    model_name = str(_cfg_get(cfg, "clip_model", "clip"))
    handle = registry.get(model_name)

    template = str(cfg.get("query.clip_template", DEFAULT_CLIP_TEMPLATE) or DEFAULT_CLIP_TEMPLATE)
    rendered = render_clip_text(text, template)
    query = l2_normalize(np.asarray(embed_text(handle, rendered), dtype=np.float32))

    hits = index.search(query, k=top_k)

    kb_path = p.kb / "artworks.parquet"
    info: dict[str, dict[str, Any]] = {}
    if kb_path.exists():
        frame = pd.read_parquet(kb_path)
        columns = [c for c in ("artwork_id", "title", "artist_name") if c in frame.columns]
        if columns:
            subset = frame[columns].to_dict("records")
            info = {str(rec.get("artwork_id")): rec for rec in subset}

    results: list[dict[str, Any]] = []
    for rank, (artwork_id, score) in enumerate(hits, start=1):
        record = info.get(artwork_id, {})
        results.append(
            {
                "rank": rank,
                "artwork_id": artwork_id,
                "title": record.get("title") or "",
                "artist": record.get("artist_name") or "",
                "score": float(score),
            }
        )
    return results


# ---------------------------------------------------------------------------
# In kết quả
# ---------------------------------------------------------------------------


def print_build(stats: dict[str, Any]) -> None:
    """In thống kê sau khi build."""
    if stats.get("status") == "exists":
        print(
            f"Index CLIP đã tồn tại ({stats['dir']}) — bỏ qua. "
            f"n={stats['n']} | dim={stats['dim']} | kb_hash={stats['kb_hash']}"
        )
        print("Dùng --force để dựng lại.")
        return
    print(
        f"Đã dựng index CLIP: {stats['n']} vector | dim={stats['dim']} | "
        f"model={stats['model']} | device={stats.get('device', '')} | "
        f"batch={stats.get('batch_size', '')}"
    )
    print(f"  thời gian build: {stats['build_seconds']:.1f} giây")
    print(f"  thư mục: {stats['dir']}")
    print(f"  kb_hash: {stats['kb_hash']}")


def print_check(stats: dict[str, Any]) -> None:
    """In trạng thái index (OK / STALE / MISSING)."""
    print(f"Thư mục index: {stats['dir']}")
    if stats.get("status") == "missing":
        print(f"Trạng thái: CHƯA CÓ INDEX — {stats.get('message', '')}")
        return
    print(f"Trạng thái: {'OK' if stats['status'] == 'ok' else 'STALE'}")
    print(f"  n: {stats['n']} | dim: {stats['dim']}")
    print(f"  model: {stats['model']} | revision: {stats.get('revision', '')}")
    print(f"  kb_hash index   : {stats['kb_hash']}")
    print(f"  kb_hash hiện tại: {stats['kb_hash_current']}")
    if stats["status"] != "ok":
        print("  -> Index cũ, cần build lại: python -m artrag.index.clip_index build --force")


def print_search(results: list[dict[str, Any]], query_text: str) -> None:
    """In bảng kết quả top-k."""
    print(f"Truy vấn: {query_text}")
    if not results:
        print("Không có kết quả.")
        return
    print(f"{'#':>3}  {'artwork_id':<16}  {'score':>7}  {'title':<40}  artist")
    for item in results:
        title = str(item["title"])[:40]
        print(
            f"{item['rank']:>3}  {item['artwork_id']:<16}  "
            f"{item['score']:>7.4f}  {title:<40}  {item['artist']}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Bộ tham số cho `python -m artrag.index.clip_index`."""
    parser = argparse.ArgumentParser(
        prog="python -m artrag.index.clip_index",
        description="Dựng / kiểm tra / tra cứu index ảnh CLIP (FAISS IndexFlatIP).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(target: argparse.ArgumentParser) -> None:
        target.add_argument("--profile", default=None, help="profile config (mặc định: local)")
        target.add_argument(
            "--set",
            dest="overrides",
            action="append",
            default=[],
            help="ghi đè config, ví dụ --set index.batch_size=32",
        )

    build_parser = sub.add_parser("build", help="dựng index ảnh CLIP")
    build_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="chỉ dùng N tác phẩm đầu (dùng để thử nhanh)",
    )
    build_parser.add_argument(
        "--force",
        action="store_true",
        help="dựng lại dù index đã tồn tại",
    )
    add_common(build_parser)

    check_parser = sub.add_parser("check", help="kiểm tra trạng thái index")
    add_common(check_parser)

    search_parser = sub.add_parser("search", help="tra cứu index bằng câu văn bản")
    search_parser.add_argument("text", help="câu truy vấn (tiếng Anh hoặc tiếng Việt)")
    search_parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"số kết quả (mặc định {DEFAULT_TOP_K})",
    )
    add_common(search_parser)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Điểm vào của `python -m artrag.index.clip_index`."""
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(profile=args.profile, overrides=args.overrides)

    if args.command == "build":
        print_build(build_index(cfg, limit=args.limit, force=args.force))
        return 0

    if args.command == "check":
        print_check(check_index(cfg))
        return 0

    if args.command == "search":
        try:
            results = search_text(cfg, args.text, top_k=args.top)
        except StaleIndexError as exc:
            print(f"LỖI: {exc}")
            return 1
        print_search(results, args.text)
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
