"""T1.6: tải ảnh tác phẩm từ CMA về ``data/images/`` (SYSTEM_DESIGN §4 bước 5).

Quy tắc (chi tiết ở SYSTEM_DESIGN §2.4 "Quy tắc tải ảnh"):
- Nguồn danh sách là ``kb/artworks.parquet`` (cột ``image_url``).
- Tải **song song** bằng ``ThreadPoolExecutor`` (``data.image_workers`` luồng).
- Chuẩn hóa ảnh: Pillow → ``RGB`` → resize theo **cạnh dài** (``data.image_max_side``)
  → JPEG ``quality=data.image_quality``.
- File tại ``images/{artwork_id}.jpg``; bỏ qua file đã có (idempotent).
- Lỗi ghi vào ``data/images_failed.json``; ``--retry-failed`` chỉ thử lại nhóm lỗi.
- Điền ``image_path`` (tương đối so với ``data/``) vào ``artworks.parquet``.

Lệnh:
    uv run python -m artrag.data.images --limit 20
    uv run python -m artrag.data.images --check
"""

from __future__ import annotations

import argparse
import io
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import requests
from PIL import Image

from artrag.config import Config, load_config, paths
from artrag.utils.io import read_json, write_json

DEFAULT_MAX_SIDE = 512
DEFAULT_QUALITY = 90
DEFAULT_WORKERS = 8
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_S = 2.0
DEFAULT_USER_AGENT = "artrag/0.1 (educational RAG project; contact: local)"

#: Tên file lưu danh sách ảnh lỗi (nằm ở gốc ``data/``, ngoài thư mục ``images/``).
FAILED_FILENAME = "images_failed.json"

#: Thư mục con của ``data/`` chứa ảnh.
IMAGES_SUBDIR = "images"


# ---------------------------------------------------------------- helpers
def image_path_for(artwork_id: str, images_dir: str | Path) -> Path:
    """Đường dẫn file ảnh của một tác phẩm trong thư mục ``images``."""
    return Path(images_dir) / f"{artwork_id}.jpg"


def relative_image_path(artwork_id: str) -> str:
    """Đường dẫn tương đối so với ``data/`` (vd ``images/cma-151904.jpg``)."""
    return f"{IMAGES_SUBDIR}/{artwork_id}.jpg"


def failed_path(data_dir: str | Path) -> Path:
    """Đường dẫn file ``images_failed.json`` (ở gốc ``data/``)."""
    return Path(data_dir) / FAILED_FILENAME


def _cfg_get(cfg: Config, key: str, default: Any) -> Any:
    value = cfg.get(f"data.{key}", None)
    return default if value is None else value


def convert_image(
    src_bytes: bytes,
    max_side: int = DEFAULT_MAX_SIDE,
    quality: int = DEFAULT_QUALITY,
) -> bytes:
    """Bytes ảnh thô → bytes JPEG RGB, cạnh dài tối đa ``max_side``.

    - Ép ``RGB`` (ảnh CMA có thể là CMYK / grayscale / có alpha).
    - Resize theo **cạnh dài**, giữ tỷ lệ.
    - Lưu lại dạng JPEG với ``quality``.
    """
    with Image.open(io.BytesIO(src_bytes)) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        longest = max(width, height)
        if longest > max_side:
            scale = max_side / float(longest)
            new_size = (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            )
            rgb = rgb.resize(new_size, Image.LANCZOS)
        buffer = io.BytesIO()
        rgb.save(buffer, format="JPEG", quality=int(quality))
        return buffer.getvalue()


def load_failed(path: str | Path) -> dict[str, str]:
    """Đọc ``images_failed.json`` → map ``artwork_id`` → lý do lỗi."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        payload = read_json(p)
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(key): str(value) for key, value in payload.items()}


def save_failed(path: str | Path, failed: dict[str, str]) -> Path:
    """Ghi ``images_failed.json`` (atomic)."""
    return write_json(path, dict(sorted(failed.items())))


def plan_downloads(
    rows: list[dict[str, Any]],
    images_dir: str | Path,
    failed_ids: set[str] | None = None,
    force: bool = False,
    retry_failed: bool = False,
) -> list[dict[str, Any]]:
    """Danh sách việc cần tải.

    - Bỏ qua tác phẩm đã có file ảnh (trừ khi ``force``).
    - Tác phẩm không có ``image_url`` → bỏ qua hoàn toàn.
    - ``retry_failed=True`` → chỉ giữ các ``artwork_id`` trong ``failed_ids``.
    """
    plan: list[dict[str, Any]] = []
    for row in rows:
        artwork_id = row.get("artwork_id")
        url = row.get("image_url")
        if not artwork_id or not url:
            continue
        artwork_id = str(artwork_id)
        if retry_failed and (not failed_ids or artwork_id not in failed_ids):
            continue
        target = image_path_for(artwork_id, images_dir)
        if target.exists() and not force:
            continue
        plan.append(
            {
                "artwork_id": artwork_id,
                "url": str(url),
                "path": target,
            }
        )
    return plan


def update_artworks_parquet(
    kb_path: str | Path,
    images_dir: str | Path,
    failed: dict[str, str] | None = None,
) -> int:
    """Điền cột ``image_path`` trong ``artworks.parquet``; trả số dòng được điền.

    Giá trị là đường dẫn tương đối so với ``data/`` (``images/cma-151904.jpg``).
    Tác phẩm không có file ảnh giữ ``null``.
    """
    p = Path(kb_path)
    if not p.exists():
        return 0
    frame = pd.read_parquet(p)
    if "artwork_id" not in frame.columns:
        return 0

    images_dir = Path(images_dir)
    filled = 0
    values: list[str | None] = []
    for artwork_id in frame["artwork_id"].tolist():
        if artwork_id is None:
            values.append(None)
            continue
        artwork_id = str(artwork_id)
        if image_path_for(artwork_id, images_dir).exists():
            values.append(relative_image_path(artwork_id))
            filled += 1
        else:
            values.append(None)

    frame["image_path"] = values
    tmp = p.with_suffix(p.suffix + ".tmp")
    frame.to_parquet(tmp, engine="pyarrow", index=False)
    tmp.replace(p)
    return filled


# ---------------------------------------------------------------- fetching
def download_one(
    url: str,
    target: str | Path,
    session: requests.Session,
    max_side: int = DEFAULT_MAX_SIDE,
    quality: int = DEFAULT_QUALITY,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_s: float = DEFAULT_BACKOFF_S,
    user_agent: str = DEFAULT_USER_AGENT,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[bool, str | None]:
    """Tải một ảnh → ``target``. Trả ``(ok, lý do lỗi)``."""
    last_error: str | None = None
    for attempt in range(max_retries + 1):
        try:
            response = session.get(
                url,
                timeout=timeout_s,
                headers={"User-Agent": user_agent},
            )
            status = getattr(response, "status_code", 200)
            if status == 429 or (isinstance(status, int) and status >= 500):
                raise requests.HTTPError(f"HTTP {status}")
            response.raise_for_status()
            payload = convert_image(response.content, max_side=max_side, quality=quality)
            path = Path(target)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            return True, None
        except (requests.RequestException, ValueError, OSError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt >= max_retries:
                break
            sleep(backoff_s * (2**attempt))
    return False, last_error or "unknown error"


def _download_worker(task: dict[str, Any], cfg: Config) -> tuple[str, str | None]:
    """Tải một ảnh (dùng cho ThreadPoolExecutor)."""
    session = requests.Session()
    try:
        ok, error = download_one(
            str(task["url"]),
            task["path"],
            session,
            max_side=int(_cfg_get(cfg, "image_max_side", DEFAULT_MAX_SIDE)),
            quality=int(_cfg_get(cfg, "image_quality", DEFAULT_QUALITY)),
            timeout_s=float(_cfg_get(cfg, "image_timeout_s", DEFAULT_TIMEOUT_S)),
            max_retries=int(_cfg_get(cfg, "image_max_retries", DEFAULT_MAX_RETRIES)),
            backoff_s=float(_cfg_get(cfg, "image_backoff_s", DEFAULT_BACKOFF_S)),
            user_agent=str(_cfg_get(cfg, "image_user_agent", DEFAULT_USER_AGENT)),
        )
    finally:
        session.close()
    return str(task["artwork_id"]), error


def download_images(
    cfg: Config,
    limit: int | None = None,
    force: bool = False,
    retry_failed: bool = False,
) -> dict[str, Any]:
    """Tải ảnh cho các tác phẩm trong KB; trả thống kê."""
    p = paths(cfg, ensure=True)
    kb_path = p.kb / "artworks.parquet"
    if not kb_path.exists():
        return {"error": f"Không tìm thấy {kb_path}. Chạy: python -m artrag.data.build_kb"}

    frame = pd.read_parquet(kb_path)
    rows = frame.to_dict("records")
    if limit is not None:
        rows = rows[:limit]

    failed = load_failed(failed_path(p.data_dir))
    failed_ids = set(failed.keys())
    plan = plan_downloads(
        rows,
        p.images,
        failed_ids=failed_ids,
        force=force,
        retry_failed=retry_failed,
    )

    workers = int(_cfg_get(cfg, "image_workers", DEFAULT_WORKERS))
    downloaded = 0
    if plan:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for artwork_id, error in pool.map(lambda t: _download_worker(t, cfg), plan):
                if error is None:
                    downloaded += 1
                    failed.pop(artwork_id, None)
                else:
                    failed[artwork_id] = error

    save_failed(failed_path(p.data_dir), failed)
    filled = update_artworks_parquet(kb_path, p.images, failed)

    return {
        "total_artworks": len(rows),
        "planned": len(plan),
        "downloaded": downloaded,
        "failed": len(failed),
        "image_path_filled": filled,
        "images_dir": str(p.images),
        "failed_file": str(failed_path(p.data_dir)),
    }


# ---------------------------------------------------------------- check
def check_images(cfg: Config) -> dict[str, Any]:
    """Thống kê ảnh hiện có (không tải mạng)."""
    p = paths(cfg)
    kb_path = p.kb / "artworks.parquet"
    total = 0
    if kb_path.exists():
        frame = pd.read_parquet(kb_path)
        total = len(frame)

    files = sorted(p.images.glob("*.jpg")) if p.images.exists() else []
    total_bytes = sum(f.stat().st_size for f in files)
    failed = load_failed(failed_path(p.data_dir))
    missing = total - len(files)

    return {
        "total_artworks": total,
        "images": len(files),
        "missing": max(missing, 0),
        "failed": len(failed),
        "total_bytes": total_bytes,
        "images_dir": str(p.images),
    }


def print_check(stats: dict[str, Any]) -> None:
    """In thống kê ảnh."""
    if "error" in stats:
        print(stats["error"])
        return
    total = stats["total_artworks"]
    images = stats["images"]
    print(f"KB: {total} artworks")
    if total:
        print(f"Ảnh có: {images} / {total} ({images / total:.1%})")
    else:
        print(f"Ảnh có: {images}")
    print(f"Thiếu: {stats['missing']} | Lỗi (images_failed.json): {stats['failed']}")
    size_mb = stats["total_bytes"] / (1024 * 1024)
    print(f"Tổng dung lượng: {size_mb:.1f} MB ({stats['total_bytes']} bytes)")


# ---------------------------------------------------------------- CLI
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m artrag.data.images",
        description="Tải ảnh tác phẩm CMA về data/images/ và điền image_path vào KB.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="chỉ xử lý N tác phẩm đầu (dùng để thử nhanh)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="in thống kê ảnh hiện có, không tải",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="tải lại cả những ảnh đã có file",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="chỉ thử lại các ảnh bị lỗi (đọc images_failed.json)",
    )
    parser.add_argument("--profile", default=None, help="profile config (mặc định: local)")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="ghi đè config, ví dụ --set data.image_workers=4",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(profile=args.profile, overrides=args.overrides)

    if args.check:
        print_check(check_images(cfg))
        return 0

    stats = download_images(
        cfg,
        limit=args.limit,
        force=args.force,
        retry_failed=args.retry_failed,
    )
    if "error" in stats:
        print(stats["error"])
        return 1

    print(
        f"images: {stats['downloaded']} đã tải | "
        f"{stats['planned']} kế hoạch | {stats['failed']} lỗi | "
        f"image_path: {stats['image_path_filled']} dòng"
    )
    print_check(check_images(cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())