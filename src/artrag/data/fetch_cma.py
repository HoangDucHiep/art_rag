"""T1.1: tải dữ liệu từ CMA Open Access API theo trang, cache vào raw/cma/.

Quy tắc (SYSTEM_DESIGN 2.4): mỗi trang lưu nguyên body API trong
``raw/cma/page_{skip:05d}.json`` kèm khóa ``_meta`` = {skip, limit, count,
total, fetched_at}. Một trang được coi là cache hợp lệ khi ``_meta.skip ==
skip`` và ``_meta.limit >= limit`` yêu cầu (và số bản ghi lưu khớp); nếu không
thì phải tải lại. Hệ quả: chạy ``--limit 50`` rồi chạy đầy đủ sẽ tải lại
trang 0, vì trang đó chỉ chứa 50 bản ghi.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests

from artrag.config import Config, load_config, paths
from artrag.utils.io import write_json

DEFAULT_PAGE_SIZE = 1000
DEFAULT_TIMEOUT_S = 60.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_S = 2.0
DEFAULT_USER_AGENT = "artrag/0.1 (educational RAG project; contact: local)"

META_KEY = "_meta"


# ---------------------------------------------------------------- helpers
def page_path(raw_dir: str | Path, skip: int) -> Path:
    """Đường dẫn file cache của trang có ``skip`` bắt đầu từ 0."""
    return Path(raw_dir) / "cma" / f"page_{skip:05d}.json"


def build_params(
    cma_type: str,
    skip: int,
    limit: int,
    cc0: bool = True,
    has_image: bool = True,
) -> dict[str, Any]:
    """Tham số truy vấn CMA API (chỉ lấy tranh CC0 có ảnh)."""
    params: dict[str, Any] = {"type": cma_type, "limit": limit, "skip": skip}
    if cc0:
        params["cc0"] = 1
    if has_image:
        params["has_image"] = 1
    return params


def _cfg_get(cfg: Config, key: str, default: Any) -> Any:
    value = cfg.get(f"data.{key}", None)
    return default if value is None else value


def read_cached_page(path: str | Path) -> dict[str, Any] | None:
    """Đọc trang cache; trả None nếu không tồn tại hoặc hỏng."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        return None
    return payload


def is_cache_valid(
    payload: dict[str, Any] | None,
    skip: int,
    limit: int,
) -> bool:
    """Trang cache hợp lệ khi ``_meta.skip == skip`` và ``_meta.limit >= limit``.

    ``limit`` là số bản ghi **yêu cầu** cho trang đó; trang đã lấy đầy đủ
    (1000) vẫn hợp lệ khi hiện tại chỉ cần 50 bản ghi.
    """
    if payload is None:
        return False
    meta = payload.get(META_KEY)
    if not isinstance(meta, dict):
        return False
    if meta.get("skip") != skip:
        return False
    cached_limit = meta.get("limit")
    if not isinstance(cached_limit, int):
        return False
    if cached_limit < limit:
        return False
    count = meta.get("count")
    if isinstance(count, int) and count != len(payload.get("data", [])):
        return False
    return True


def with_meta(
    payload: dict[str, Any],
    skip: int,
    limit: int,
    total: int | None,
) -> dict[str, Any]:
    """Gắn khóa ``_meta`` vào body API (không sửa phần ``info``/``data``)."""
    payload[META_KEY] = {
        "skip": skip,
        "limit": limit,
        "count": len(payload.get("data", [])),
        "total": total,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return payload


def page_total(payload: dict[str, Any]) -> int | None:
    """Tổng số bản ghi theo ``info.total`` (None nếu API không trả)."""
    info = payload.get("info")
    if isinstance(info, dict):
        total = info.get("total")
        if isinstance(total, int):
            return total
    return None


def page_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Danh sách bản ghi của một trang."""
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    return [rec for rec in data if isinstance(rec, dict)]


# ---------------------------------------------------------------- fetching
def fetch_page(
    session: requests.Session,
    api_base: str,
    params: dict[str, Any],
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_s: float = DEFAULT_BACKOFF_S,
    user_agent: str = DEFAULT_USER_AGENT,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Gọi API một trang; retry có backoff khi lỗi mạng hoặc HTTP 5xx/429."""
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = session.get(
                api_base,
                params=params,
                timeout=timeout_s,
                headers={"User-Agent": user_agent},
            )
            status = getattr(response, "status_code", 200)
            if status == 429 or (isinstance(status, int) and status >= 500):
                raise requests.HTTPError(f"HTTP {status}", response=response)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            sleep(backoff_s * (2**attempt))
    raise RuntimeError(f"Không tải được trang skip={params.get('skip')}: {last_error}")


def fetch_all(
    cfg: Config,
    force: bool = False,
    limit: int | None = None,
    page_size: int | None = None,
) -> dict[str, Any]:
    """Tải các trang cần thiết vào ``raw/cma/``; trả về thống kê."""
    p = paths(cfg, ensure=True)
    raw_cma = p.raw / "cma"

    api_base = str(
        _cfg_get(cfg, "cma_api_base", "https://openaccess-api.clevelandart.org/api/artworks/")
    )
    cma_type = str(_cfg_get(cfg, "cma_type", "Painting"))
    timeout_s = float(_cfg_get(cfg, "cma_timeout_s", DEFAULT_TIMEOUT_S))
    max_retries = int(_cfg_get(cfg, "cma_max_retries", DEFAULT_MAX_RETRIES))
    backoff_s = float(_cfg_get(cfg, "cma_backoff_s", DEFAULT_BACKOFF_S))
    user_agent = str(_cfg_get(cfg, "cma_user_agent", DEFAULT_USER_AGENT))
    size = int(page_size or _cfg_get(cfg, "cma_page_size", DEFAULT_PAGE_SIZE))

    session = requests.Session()
    fetched = 0
    cached = 0
    records = 0
    total: int | None = None
    used = 0
    skip = 0

    try:
        while True:
            remaining = None if limit is None else limit - used
            if remaining is not None and remaining <= 0:
                break

            path = page_path(p.raw, skip)
            payload = None if force else read_cached_page(path)

            # tổng số bản ghi: ưu tiên từ trang đang cache, rồi đến trang đã tải
            if total is None and payload is not None:
                total = page_total(payload)
            if total is not None and skip >= total:
                break  # đã lấy đủ số bản ghi

            # số bản ghi yêu cầu cho trang này: trang cuối có thể ít hơn page_size
            if remaining is not None:
                page_limit = min(size, remaining)
            elif total is not None:
                page_limit = min(size, max(total - skip, 0))
            else:
                page_limit = size
            page_limit = max(page_limit, 1)

            if not force and is_cache_valid(payload, skip, page_limit):
                cached += 1
                page_recs = page_records(payload)
                take = len(page_recs) if remaining is None else min(len(page_recs), remaining)
                records += take
                used += take
                if remaining is not None and used >= limit:  # type: ignore[operator]
                    break
                if not page_recs:
                    break
                skip += size
                continue

            params = build_params(cma_type, skip, page_limit)
            payload = fetch_page(
                session,
                api_base,
                params,
                timeout_s=timeout_s,
                max_retries=max_retries,
                backoff_s=backoff_s,
                user_agent=user_agent,
            )
            page_total_value = page_total(payload)
            if total is None and page_total_value is not None:
                total = page_total_value

            page_recs = page_records(payload)
            if remaining is not None and len(page_recs) > remaining:
                payload = dict(payload)
                payload["data"] = page_recs[:remaining]
                page_recs = payload["data"]

            payload = with_meta(payload, skip, page_limit, total)
            write_json(path, payload)
            fetched += 1
            records += len(page_recs)
            used += len(page_recs)

            if remaining is not None and used >= limit:  # type: ignore[operator]
                break
            if not page_recs:
                break
            if total is not None and used >= total:
                break
            skip += size
    finally:
        session.close()

    return {
        "fetched_pages": fetched,
        "cached_pages": cached,
        "records": records,
        "total": total,
        "raw_dir": str(raw_cma),
    }


# ---------------------------------------------------------------- CLI
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m artrag.data.fetch_cma",
        description="Tải dữ liệu từ CMA Open Access API (phân trang, cache, retry).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="chỉ lấy N bản ghi đầu (dùng để thử nhanh)",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=None,
        help="số bản ghi mỗi trang (mặc định lấy từ config data.cma_page_size)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="tải lại tất cả trang, bỏ qua cache",
    )
    parser.add_argument("--profile", default=None, help="profile config (mặc định: local)")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="ghi đè config, ví dụ --set data.cma_page_size=500",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(profile=args.profile, overrides=args.overrides)
    stats = fetch_all(
        cfg,
        force=args.force,
        limit=args.limit,
        page_size=args.page_size,
    )
    parts = [f"fetch_cma: {stats['records']} records"]
    if stats["fetched_pages"]:
        parts.append(f"{stats['fetched_pages']} page(s) fetched")
    if stats["cached_pages"]:
        parts.append(f"{stats['cached_pages']} page(s) cached")
    if stats["total"] is not None:
        parts.append(f"total {stats['total']}")
    print(" | ".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
