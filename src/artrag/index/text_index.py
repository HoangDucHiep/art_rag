"""T3.2: Dựng và tra cứu index văn bản BGE-M3 (SYSTEM_DESIGN §4 bước 7).

Nguồn: ``data/kb/chunks.parquet`` (hoặc ``chunks_variants.parquet`` nếu dùng cờ
``--variants``) -> mỗi chunk được embed bằng BGE-M3 (task ``sbert``, profile
``local`` chạy CPU fp32) qua ``artrag.models.inference.embed_text(handle, text)``
-> vector L2-normalize -> FAISS ``IndexFlatIP``.

Kết quả nằm ở ``data/index/text/``:
    emb.npy | ids.json | meta.json

Idempotent: nếu index đã tồn tại và ``kb_hash`` khớp KB hiện tại thì lệnh
``build`` bỏ qua (in "đã tồn tại"), trừ khi dùng ``--force``.

Lệnh:
    uv run python -m artrag.index.text_index build --limit 500
    uv run python -m artrag.index.text_index check
    uv run python -m artrag.index.text_index search "ai vẽ tranh hai anh em chèo thuyền" --top 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

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

#: Thư mục con của ``data/index/`` chứa index văn bản.
TEXT_INDEX_SUBDIR = "text"

#: Batch size mặc định cho GPU / CPU.
DEFAULT_BATCH_SIZE = 64
DEFAULT_CPU_BATCH_SIZE = 16

#: Số kết quả mặc định của lệnh `search`.
DEFAULT_TOP_K = 5

#: Độ dài tối đa của trích đoạn text hiển thị khi search.
MAX_SNIPPET_CHARS = 120


# ---------------------------------------------------------------------------
# Tiện ích đường dẫn & config
# ---------------------------------------------------------------------------


def index_dir(cfg: Config) -> Path:
    """Thư mục chứa index văn bản (``data/index/text/``)."""
    return paths(cfg).index / TEXT_INDEX_SUBDIR


def chunks_path(cfg: Config, variants: bool = False) -> Path:
    """Đường dẫn tới file parquet chứa các chunk (baseline hoặc variants)."""
    filename = "chunks_variants.parquet" if variants else "chunks.parquet"
    return paths(cfg).kb / filename


def _cfg_get(cfg: Config, key: str, default: Any) -> Any:
    value = cfg.get(f"index.{key}", None)
    return default if value is None else value


def batch_size_for(cfg: Config, device: str, override: int | None = None) -> int:
    """Batch size theo thiết bị: `index.batch_size` cho GPU, `index.batch_size_cpu` cho CPU."""
    if override is not None and override > 0:
        return int(override)
    configured = int(_cfg_get(cfg, "batch_size", DEFAULT_BATCH_SIZE) or DEFAULT_BATCH_SIZE)
    if str(device).startswith("cpu"):
        cpu_batch = int(
            _cfg_get(cfg, "batch_size_cpu", DEFAULT_CPU_BATCH_SIZE) or DEFAULT_CPU_BATCH_SIZE
        )
        return max(1, min(configured, cpu_batch))
    return max(1, configured)


def model_revision(handle: Any) -> str:
    """Lấy revision (commit hash hoặc checkpoint) của model nếu biết; nếu không thì rỗng."""
    candidates: list[Any] = [
        getattr(handle, "model", None),
        getattr(getattr(handle, "model", None), "config", None),
        getattr(getattr(handle, "model", None), "_model_card", None),
    ]
    if hasattr(handle, "model") and hasattr(handle.model, "__getitem__"):
        try:
            cand = handle.model[0]
            candidates.append(cand)
            auto_model = getattr(cand, "auto_model", None)
            if auto_model is not None:
                candidates.append(auto_model)
                candidates.append(getattr(auto_model, "config", None))
        except Exception:
            pass
    for cand in candidates:
        if cand is None:
            continue
        for attr in ("_commit_hash", "commit_hash", "_name_or_path"):
            val = getattr(cand, attr, None)
            if val:
                return str(val)
    return ""


def parse_ids(raw_ids: Sequence[str] | str | None) -> list[str] | None:
    """Chuẩn hoá danh sách ID từ CLI (hỗ trợ nhiều phần tử hoặc phân tách bằng dấu phẩy)."""
    if not raw_ids:
        return None
    if isinstance(raw_ids, str):
        items = [raw_ids]
    else:
        items = list(raw_ids)
    result: list[str] = []
    for item in items:
        for part in str(item).split(","):
            cleaned = part.strip()
            if cleaned:
                result.append(cleaned)
    return result or None


def select_chunks(
    frame: pd.DataFrame,
    limit: int | None = None,
    artwork_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Lọc và chọn các chunk hợp lệ để embed.

    - Bỏ qua các chunk có text rỗng hoặc null.
    - Nếu `artwork_ids` được chỉ định: chỉ giữ chunk có `artwork_id` khớp
      hoặc `chunk_id` bắt đầu bằng một trong các `artwork_id` đó.
    - Giới hạn tối đa `limit` chunk nếu được truyền.
    """
    target_ids = set(artwork_ids) if artwork_ids else None
    rows: list[dict[str, Any]] = []

    for record in frame.to_dict("records"):
        text = str(record.get("text") or "").strip()
        if not text:
            continue

        chunk_id = str(record.get("chunk_id") or "").strip()
        artwork_id = str(record.get("artwork_id") or "").strip()

        if target_ids is not None:
            matches = (
                artwork_id in target_ids
                or chunk_id in target_ids
                or any(chunk_id.startswith(f"{aid}#") for aid in target_ids)
            )
            if not matches:
                continue

        rows.append(record)
        if limit is not None and len(rows) >= int(limit):
            break

    return rows


def format_snippet(text: str, max_chars: int = MAX_SNIPPET_CHARS) -> str:
    """Cắt ngắn trích đoạn văn bản trên một dòng (~120 ký tự)."""
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3] + "..."


def _try_load_meta(directory: Path) -> dict[str, Any] | None:
    """Đọc `meta.json` nếu có (trả ``None`` nếu thiếu hoặc hỏng)."""
    meta_path = Path(directory) / META_FILENAME
    if not meta_path.exists():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_index(
    cfg: Config,
    limit: int | None = None,
    artwork_ids: Sequence[str] | None = None,
    force: bool = False,
    variants: bool = False,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Dựng index văn bản BGE-M3; trả về thống kê.

    - Bỏ qua (idempotent) nếu index đã có và `kb_hash` khớp, trừ khi `force`.
    - Batch size theo thiết bị: `index.batch_size` (GPU) / 16 (CPU).
    """
    from artrag.models import loaders  # noqa: PLC0415,F401 - đăng ký loader
    from artrag.models.inference import embed_text
    from artrag.registry import create_registry

    p = paths(cfg)
    file_path = chunks_path(cfg, variants=variants)
    if not file_path.exists():
        raise ArtragError(
            f"Không tìm thấy file {file_path}.",
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

    frame = pd.read_parquet(file_path)
    rows = select_chunks(frame, limit=limit, artwork_ids=artwork_ids)
    if not rows:
        raise ArtragError(
            f"Không có chunk nào hợp lệ trong {file_path.name}.",
            hint="Kiểm tra lại dữ liệu hoặc điều kiện lọc --ids / --limit.",
        )

    registry = create_registry(cfg)
    model_name = str(_cfg_get(cfg, "text_model", "text_embed"))
    handle = registry.get(model_name)

    started = time.perf_counter()
    vectors: list[np.ndarray] = []
    ids: list[str] = []
    batch = batch_size_for(cfg, handle.device, override=batch_size)
    total = len(rows)

    print(
        f"Bắt đầu embed {total} chunk bằng model {model_name!r} "
        f"(device={handle.device}, batch={batch})..."
    )

    for offset in range(0, total, batch):
        batch_slice = rows[offset : offset + batch]
        for record in batch_slice:
            text = str(record.get("text") or "").strip()
            if not text:
                continue
            vec = embed_text(handle, text)
            vectors.append(np.asarray(vec, dtype=np.float32))
            ids.append(str(record["chunk_id"]))

        processed = min(offset + batch, total)
        percent = (processed / total) * 100
        print(f"  Tiến trình: {processed}/{total} chunks ({percent:.1f}%)")

    if not vectors:
        raise ArtragError(
            "Không embed được chunk nào.",
            hint="Kiểm tra lại trường text trong file chunk parquet.",
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
            "source_file": file_path.name,
            "variants": bool(variants),
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


# ---------------------------------------------------------------------------
# Check
# ---------------------------------------------------------------------------


def check_index(cfg: Config) -> dict[str, Any]:
    """Kiểm tra trạng thái index văn bản so với KB hiện tại."""
    p = paths(cfg)
    target = index_dir(cfg)
    kb_hash = compute_kb_hash(p.data_dir)

    meta = _try_load_meta(target)
    if meta is None:
        return {
            "status": "missing",
            "dir": str(target),
            "kb_hash_current": kb_hash or "",
            "message": "Chưa có index văn bản. Chạy: python -m artrag.index.text_index build",
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
    query: str,
    top_k: int = DEFAULT_TOP_K,
    variants: bool = False,
) -> list[dict[str, Any]]:
    """Tìm chunk gần nhất với câu truy vấn bằng BGE-M3 (nhánh BGE-M3 của pipeline)."""
    from artrag.models import loaders  # noqa: PLC0415,F401 - đăng ký loader
    from artrag.models.inference import embed_text
    from artrag.registry import create_registry

    p = paths(cfg)
    target = index_dir(cfg)
    kb_hash = compute_kb_hash(p.data_dir)
    index = VectorIndex.load(target, expected_kb_hash=kb_hash)

    registry = create_registry(cfg)
    model_name = str(_cfg_get(cfg, "text_model", "text_embed"))
    handle = registry.get(model_name)

    # Embed câu hỏi trực tiếp bằng BGE-M3 (không cần dịch, không cần instruction riêng)
    q_vec = l2_normalize(np.asarray(embed_text(handle, query), dtype=np.float32))
    hits = index.search(q_vec, k=top_k)

    # Đọc metadata từ file chunk tương ứng
    source_filename = str(
        index.meta.get("source_file")
        or ("chunks_variants.parquet" if variants else "chunks.parquet")
    )
    file_path = p.kb / source_filename
    if not file_path.exists():
        file_path = chunks_path(cfg, variants=variants)

    chunk_info: dict[str, dict[str, Any]] = {}
    if file_path.exists():
        frame = pd.read_parquet(file_path)
        for rec in frame.to_dict("records"):
            cid = str(rec.get("chunk_id") or "").strip()
            if cid:
                chunk_info[cid] = rec

    results: list[dict[str, Any]] = []
    for rank, (chunk_id, score) in enumerate(hits, start=1):
        rec = chunk_info.get(chunk_id, {})
        artwork_id = str(rec.get("artwork_id") or "-")
        field = str(rec.get("field") or "-")
        raw_text = str(rec.get("text") or "")
        snippet = format_snippet(raw_text)

        results.append(
            {
                "rank": rank,
                "chunk_id": chunk_id,
                "artwork_id": artwork_id,
                "field": field,
                "score": float(score),
                "text": snippet,
            }
        )
    return results


# ---------------------------------------------------------------------------
# In kết quả
# ---------------------------------------------------------------------------


def print_build(stats: dict[str, Any]) -> None:
    """In thống kê sau khi dựng index."""
    if stats.get("status") == "exists":
        print(
            f"Index văn bản đã tồn tại ({stats['dir']}) — bỏ qua. "
            f"n={stats['n']} | dim={stats['dim']} | kb_hash={stats['kb_hash']}"
        )
        print("Dùng --force để dựng lại.")
        return

    print(
        f"Đã dựng index văn bản: {stats['n']} vector | dim={stats['dim']} | "
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
        print("  -> Index cũ, cần build lại: python -m artrag.index.text_index build --force")


def print_search(results: list[dict[str, Any]], query_text: str) -> None:
    """In bảng kết quả top-k theo đúng định dạng."""
    print(f"Truy vấn: {query_text}")
    if not results:
        print("Không có kết quả.")
        return
    print(f"{'#':>3}  {'chunk_id':<28}  {'artwork_id':<14}  {'field':<14}  {'score':>7}  text")
    for item in results:
        print(
            f"{item['rank']:>3}  {item['chunk_id']:<28}  "
            f"{item['artwork_id']:<14}  {item['field']:<14}  "
            f"{item['score']:>7.4f}  {item['text']}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Bộ tham số cho `python -m artrag.index.text_index`."""
    parser = argparse.ArgumentParser(
        prog="python -m artrag.index.text_index",
        description="Dựng / kiểm tra / tra cứu index văn bản BGE-M3 (FAISS IndexFlatIP).",
    )
    sub = parser.add_subparsers(dest="command")

    def add_common(target: argparse.ArgumentParser) -> None:
        target.add_argument("--profile", default=None, help="profile config (mặc định: local)")
        target.add_argument(
            "--set",
            dest="overrides",
            action="append",
            default=[],
            help="ghi đè config, ví dụ --set index.batch_size=16",
        )

    # Subcommand: build
    build_parser = sub.add_parser("build", help="dựng index văn bản BGE-M3")
    build_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="chỉ dùng N chunk đầu (dùng để thử nhanh)",
    )
    build_parser.add_argument(
        "--ids",
        nargs="*",
        default=None,
        help="chỉ lấy chunk của các artwork_id cho trước (ví dụ --ids cma-151904)",
    )
    build_parser.add_argument(
        "--variants",
        action="store_true",
        help="dùng file chunks_variants.parquet thay cho chunks.parquet baseline",
    )
    build_parser.add_argument(
        "--force",
        action="store_true",
        help="dựng lại dù index đã tồn tại",
    )
    build_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="kích thước batch khi embed (mặc định: 16 trên CPU, 64 trên GPU)",
    )
    add_common(build_parser)

    # Subcommand: check
    check_parser = sub.add_parser("check", help="kiểm tra trạng thái index")
    add_common(check_parser)

    # Subcommand: search
    search_parser = sub.add_parser("search", help="tra cứu index văn bản bằng câu truy vấn")
    search_parser.add_argument("text", help="câu truy vấn (tiếng Việt hoặc tiếng Anh)")
    search_parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"số kết quả (mặc định {DEFAULT_TOP_K})",
    )
    search_parser.add_argument(
        "--variants",
        action="store_true",
        help="tra cứu cùng với dữ liệu biến thể",
    )
    add_common(search_parser)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Điểm vào của `python -m artrag.index.text_index`."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    command = args.command or "build"
    overrides = getattr(args, "overrides", [])
    profile = getattr(args, "profile", None)
    cfg = load_config(profile=profile, overrides=overrides)

    if command == "build":
        ids_list = parse_ids(getattr(args, "ids", None))
        stats = build_index(
            cfg,
            limit=getattr(args, "limit", None),
            artwork_ids=ids_list,
            force=getattr(args, "force", False),
            variants=getattr(args, "variants", False),
            batch_size=getattr(args, "batch_size", None),
        )
        print_build(stats)
        return 0

    if command == "check":
        print_check(check_index(cfg))
        return 0

    if command == "search":
        try:
            results = search_text(
                cfg,
                args.text,
                top_k=args.top,
                variants=getattr(args, "variants", False),
            )
        except StaleIndexError as exc:
            print(f"LỖI: {exc}")
            return 1
        print_search(results, args.text)
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
