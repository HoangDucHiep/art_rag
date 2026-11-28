"""T3.3: sinh và quản lý caption BLIP cho ảnh trong KB."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from tqdm import tqdm

from artrag.config import Config, load_config, paths
from artrag.errors import ArtragError

CAPTION_COLUMNS = ("artwork_id", "model", "params_hash", "caption", "ms")
CAPTION_FILE_PREFIX = "captions"


def generation_params(cfg: Config) -> tuple[str, dict[str, int]]:
    """Lấy model và tham số sinh caption từ config."""
    model = str(cfg.get("index.caption_model", "blip_cap"))
    params = {
        "max_new_tokens": int(cfg.get("index.caption_max_new_tokens", 30)),
        "num_beams": int(cfg.get("index.caption_num_beams", 3)),
    }
    if not model or min(params.values()) < 1:
        raise ArtragError("Cấu hình caption không hợp lệ.")
    return model, params


def caption_params_hash(params: dict[str, Any]) -> str:
    """Hash ổn định của tham số sinh caption."""
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def caption_file(cache_dir: str | Path, model: str, params_hash: str) -> Path:
    """Đường dẫn parquet riêng cho một model và bộ tham số."""
    tag = hashlib.sha256(f"{model}|{params_hash}".encode("utf-8")).hexdigest()[:12]
    return Path(cache_dir) / f"{CAPTION_FILE_PREFIX}-{tag}.parquet"


def _read_caption_files(cache_dir: Path) -> pd.DataFrame:
    files = sorted(cache_dir.glob(f"{CAPTION_FILE_PREFIX}*.parquet"))
    if not files:
        return pd.DataFrame(columns=CAPTION_COLUMNS)
    frames = [pd.read_parquet(file) for file in files]
    return pd.concat(frames, ignore_index=True)


def _caption_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    missing = set(CAPTION_COLUMNS) - set(frame.columns)
    if missing:
        raise ArtragError(f"Parquet caption thiếu cột: {', '.join(sorted(missing))}.")
    return frame.loc[:, list(CAPTION_COLUMNS)].to_dict("records")


def _has_caption(value: Any) -> bool:
    return value is not None and not pd.isna(value) and bool(str(value).strip())


def _merge_records(
    existing: list[dict[str, Any]],
    additions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Gộp theo artwork/model/params; ưu tiên kết quả mới cùng bộ tham số."""
    replaced = {
        (str(row["artwork_id"]), str(row["model"]), str(row["params_hash"]))
        for row in additions
    }
    merged = [
        row
        for row in existing
        if (
            str(row["artwork_id"]),
            str(row["model"]),
            str(row["params_hash"]),
        )
        not in replaced
    ]
    merged.extend(additions)
    return merged


def _write_captions(target: Path, records: list[dict[str, Any]]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records, columns=CAPTION_COLUMNS)
    if not frame.empty:
        frame["artwork_id"] = frame["artwork_id"].astype(str)
        frame["model"] = frame["model"].astype(str)
        frame["params_hash"] = frame["params_hash"].astype(str)
        frame["caption"] = frame["caption"].astype(str)
        frame["ms"] = frame["ms"].astype(float)
    temporary = target.with_suffix(target.suffix + ".tmp")
    frame.to_parquet(temporary, engine="pyarrow", index=False)
    temporary.replace(target)


def _select_artworks(
    frame: pd.DataFrame,
    data_dir: Path,
    limit: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        image_path = row.get("image_path")
        artwork_id = row.get("artwork_id")
        if (
            artwork_id is None
            or pd.isna(artwork_id)
            or not str(artwork_id).strip()
            or image_path is None
            or pd.isna(image_path)
            or not str(image_path).strip()
        ):
            continue
        if (data_dir / str(image_path)).is_file():
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def build_captions(
    cfg: Config,
    limit: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Tạo caption cho ảnh chưa có caption của model/tham số hiện hành."""
    if limit is not None and limit < 1:
        raise ArtragError("--limit phải lớn hơn 0.")

    p = paths(cfg)
    kb_path = p.kb / "artworks.parquet"
    if not kb_path.exists():
        raise ArtragError(
            f"Không tìm thấy {kb_path}.",
            hint="Chạy: uv run python -m artrag.data.build_kb",
        )

    model_name, gen_params = generation_params(cfg)
    params_hash = caption_params_hash(gen_params)
    target = caption_file(p.cache, model_name, params_hash)
    cache = _read_caption_files(p.cache)
    all_records = _caption_records(cache)
    target_records = (
        _caption_records(pd.read_parquet(target)) if target.exists() else []
    )
    matching_records = [
        record
        for record in all_records
        if str(record["model"]) == model_name
        and str(record["params_hash"]) == params_hash
        and _has_caption(record["caption"])
    ]
    done_ids = {str(record["artwork_id"]) for record in matching_records}

    frame = pd.read_parquet(kb_path)
    candidates = _select_artworks(frame, p.data_dir, limit)
    todo = [
        row
        for row in candidates
        if force or str(row["artwork_id"]) not in done_ids
    ]

    generated: list[dict[str, Any]] = []
    total_started = time.perf_counter()
    if todo:
        from artrag.models import loaders  # noqa: PLC0415,F401 - đăng ký loader
        from artrag.models.inference import caption
        from artrag.registry import create_registry

        registry = create_registry(cfg)
        handle = registry.get(model_name)
        progress = tqdm(todo, desc="Sinh caption", unit="ảnh")
        for row in progress:
            image_started = time.perf_counter()
            image_path = p.data_dir / str(row["image_path"])
            with Image.open(image_path) as image:
                result = caption(handle, image.convert("RGB"), **gen_params)
            elapsed_ms = (time.perf_counter() - image_started) * 1000
            generated.append(
                {
                    "artwork_id": str(row["artwork_id"]),
                    "model": model_name,
                    "params_hash": params_hash,
                    "caption": result,
                    "ms": elapsed_ms,
                }
            )
            progress.set_postfix(ms_per_image=f"{elapsed_ms:.0f}")

    total_seconds = time.perf_counter() - total_started
    merged = _merge_records(target_records, matching_records)
    merged = _merge_records(merged, generated)
    _write_captions(target, merged)

    average_ms = (
        sum(float(row["ms"]) for row in generated) / len(generated)
        if generated
        else 0.0
    )
    return {
        "file": str(target),
        "model": model_name,
        "params_hash": params_hash,
        "selected": len(candidates),
        "new": len(generated),
        "captions": len(merged),
        "generated": generated,
        "total_seconds": total_seconds,
        "average_ms": average_ms,
    }


def check_captions(cfg: Config) -> dict[str, Any]:
    """Thống kê caption cache và độ phủ trên các ảnh hiện có trong KB."""
    p = paths(cfg)
    cache = _read_caption_files(p.cache)
    records = _caption_records(cache)
    kb_path = p.kb / "artworks.parquet"
    if not kb_path.exists():
        raise ArtragError(
            f"Không tìm thấy {kb_path}.",
            hint="Chạy: uv run python -m artrag.data.build_kb",
        )
    frame = pd.read_parquet(kb_path)
    total_images = len(_select_artworks(frame, p.data_dir, None))
    captioned_ids = {
        str(row["artwork_id"])
        for row in records
        if _has_caption(row["caption"])
    }
    by_model: dict[str, dict[str, float | int]] = {}
    for record in records:
        model = str(record["model"])
        stats = by_model.setdefault(model, {"count": 0, "ms_total": 0.0})
        stats["count"] += 1
        stats["ms_total"] += float(record["ms"])
    for stats in by_model.values():
        count = int(stats["count"])
        stats["average_ms"] = float(stats["ms_total"]) / count if count else 0.0

    return {
        "total_captions": len(records),
        "unique_captioned": len(captioned_ids),
        "total_images": total_images,
        "average_ms": (
            sum(float(row["ms"]) for row in records) / len(records)
            if records
            else 0.0
        ),
        "by_model": by_model,
    }


def print_build(stats: dict[str, Any]) -> None:
    print(
        f"Caption mới: {stats['new']} | đã chọn: {stats['selected']} | "
        f"tổng trong file: {stats['captions']}"
    )
    print(f"Model: {stats['model']} | params_hash: {stats['params_hash']}")
    print(f"Parquet: {stats['file']}")
    print(
        f"Thời gian: {stats['total_seconds']:.2f} giây | "
        f"trung bình: {stats['average_ms']:.1f} ms/ảnh"
    )
    for index, record in enumerate(stats["generated"][:3], start=1):
        print(
            f"  {index}. {record['artwork_id']} "
            f"({record['ms']:.1f} ms): {record['caption']}"
        )


def print_check(stats: dict[str, Any]) -> None:
    print(f"Tổng caption: {stats['total_captions']}")
    print(f"Thời gian trung bình: {stats['average_ms']:.1f} ms/ảnh")
    for model, values in sorted(stats["by_model"].items()):
        print(
            f"  {model}: {values['count']} | "
            f"{values['average_ms']:.1f} ms/ảnh"
        )
    total = int(stats["total_images"])
    coverage = int(stats["unique_captioned"])
    ratio = coverage / total if total else 0.0
    print(f"Độ phủ: {coverage}/{total} ảnh ({ratio:.1%})")
    print("OK")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("giá trị phải lớn hơn 0")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m artrag.index.captions",
        description="Sinh, kiểm tra caption BLIP trong data/cache/.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(target: argparse.ArgumentParser) -> None:
        target.add_argument("--profile", default=None, help="profile config (mặc định: local)")
        target.add_argument(
            "--set",
            dest="overrides",
            action="append",
            default=[],
            help="ghi đè config, ví dụ --set index.caption_num_beams=3",
        )

    build_parser = sub.add_parser("build", help="sinh caption cho ảnh chưa có")
    build_parser.add_argument("--limit", type=positive_int, default=None, help="giới hạn số ảnh")
    build_parser.add_argument("--force", action="store_true", help="sinh lại caption đã có")
    add_common(build_parser)

    check_parser = sub.add_parser("check", help="thống kê cache caption")
    add_common(check_parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(profile=args.profile, overrides=args.overrides)
    if args.command == "build":
        print_build(build_captions(cfg, limit=args.limit, force=args.force))
    else:
        print_check(check_captions(cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
