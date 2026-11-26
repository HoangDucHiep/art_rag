"""T1.7: đóng gói artefact để phân phối (SYSTEM_DESIGN §2.4 "Phân phối artefact").

``pack`` nén ``kb/``, ``images/``, ``index/`` và **chỉ** ``cache/captions*`` thành
một file zip kèm ``manifest.json`` ở gối. ``unpack`` giải nén và đối chiếu số file
theo từng part với manifest.

Lệnh:
    uv run python -m artrag.tools.artifacts pack --only kb,images
    uv run python -m artrag.tools.artifacts unpack artifacts.zip --dest ./tmp
"""

from __future__ import annotations

import argparse
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from artrag.utils.hashing import sha256_json

#: Các phần được đóng gói (khớp §2.4).
PACK_TARGETS: tuple[str, ...] = ("kb", "images", "index", "cache")

#: Tên file manifest ở gối zip.
MANIFEST_NAME = "manifest.json"

#: Tiền tố của file caption trong ``cache/`` (chỉ phần này được đóng gói).
CAPTION_PREFIX = "captions"


# ---------------------------------------------------------------- helpers
def _part_of(relative: Path) -> str:
    """Tên part (kb/images/index/cache) của một đường dẫn tương đối."""
    return relative.parts[0] if relative.parts else ""


def _is_cache_member(relative: Path) -> bool:
    """Chỉ nhận file ``cache/captions*`` (bỏ qua ``cache/llm``, ``cache/vqa``...)."""
    if _part_of(relative) != "cache":
        return False
    if len(relative.parts) < 2:
        return False
    return relative.parts[1].startswith(CAPTION_PREFIX)


def collect_members(
    data_dir: str | Path,
    only: Iterable[str] | None = None,
) -> list[tuple[Path, str]]:
    """Danh sách ``(đường dẫn thật, tên trong zip)`` cần đóng gói.

    - Chỉ lấy các part trong ``only`` (mặc định: tất cả ``PACK_TARGETS``).
    - Trong ``cache/`` chỉ lấy file ``captions*``.
    """
    root = Path(data_dir)
    wanted = set(only) if only else set(PACK_TARGETS)
    members: list[tuple[Path, str]] = []
    for part in PACK_TARGETS:
        if part not in wanted:
            continue
        part_dir = root / part
        if not part_dir.exists():
            continue
        for path in sorted(part_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            if part == "cache" and not _is_cache_member(relative):
                continue
            members.append((path, relative.as_posix()))
    return members


def kb_fingerprint(data_dir: str | Path) -> str | None:
    """Hash ổn định của KB (tập ``(artwork_id, image_path)`` trong artworks.parquet).

    Trả ``None`` nếu không có ``kb/artworks.parquet``.
    """
    path = Path(data_dir) / "kb" / "artworks.parquet"
    if not path.exists():
        return None
    try:
        import pandas as pd

        frame = pd.read_parquet(path, columns=["artwork_id", "image_path"])
    except Exception:  # noqa: BLE001 - parquet hỏng -> không có fingerprint
        return None
    pairs = [
        [str(a) if a is not None else "", str(p) if p is not None else ""]
        for a, p in zip(frame["artwork_id"], frame["image_path"])
    ]
    pairs.sort()
    return sha256_json(pairs)


def build_manifest(
    data_dir: str | Path,
    members: list[tuple[Path, str]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Dựng nội dung ``manifest.json`` từ danh sách file.

    Gồm: ``created_at``, ``parts`` (số file + dung lượng mỗi part),
    ``total_files``, ``total_bytes``, ``kb_hash``.
    """
    parts: dict[str, dict[str, int]] = {}
    total_bytes = 0
    for path, arcname in members:
        part = _part_of(Path(arcname))
        size = path.stat().st_size
        entry = parts.setdefault(part, {"files": 0, "bytes": 0})
        entry["files"] += 1
        entry["bytes"] += size
        total_bytes += size

    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "parts": dict(sorted(parts.items())),
        "total_files": len(members),
        "total_bytes": total_bytes,
        "kb_hash": kb_fingerprint(data_dir),
    }
    if extra:
        manifest.update(extra)
    return manifest


def pack(
    data_dir: str | Path,
    out_path: str | Path,
    only: Iterable[str] | None = None,
    manifest_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Nén artefact thành zip; trả thống kê (số file, dung lượng, manifest)."""
    root = Path(data_dir)
    out = Path(out_path)
    members = collect_members(root, only=only)
    manifest = build_manifest(root, members, extra=manifest_extra)

    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(MANIFEST_NAME, _manifest_bytes(manifest))
        for path, arcname in members:
            archive.write(path, arcname)

    return {
        "zip_path": str(out),
        "zip_bytes": out.stat().st_size,
        "total_files": manifest["total_files"],
        "total_bytes": manifest["total_bytes"],
        "parts": manifest["parts"],
        "kb_hash": manifest["kb_hash"],
        "manifest": manifest,
    }


def _manifest_bytes(manifest: dict[str, Any]) -> bytes:
    """Serialize manifest thành bytes JSON (UTF-8, thứ tự khoá ổn định)."""
    import json

    return json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")


def read_manifest(zip_path: str | Path) -> dict[str, Any] | None:
    """Đọc ``manifest.json`` từ zip (None nếu không có)."""
    with zipfile.ZipFile(zip_path, "r") as archive:
        if MANIFEST_NAME not in archive.namelist():
            return None
        payload = archive.read(MANIFEST_NAME)
    import json

    return json.loads(payload.decode("utf-8"))


def unpack(
    zip_path: str | Path,
    dest_dir: str | Path,
    verify: bool = True,
    strict: bool = False,
) -> dict[str, Any]:
    """Giải nén zip vào ``dest_dir``; đối chiếu số file với manifest.

    - ``verify=True`` → in báo cáo khớp/lệch theo từng part.
    - ``strict=True`` → ném lỗi nếu có sai lệch.
    """
    archive_path = Path(zip_path)
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path, "r") as archive:
        names = [n for n in archive.namelist() if not n.endswith("/")]
        archive.extractall(dest)

    manifest = read_manifest(archive_path)
    actual: dict[str, dict[str, int]] = {}
    for name in names:
        if name == MANIFEST_NAME:
            continue
        part = _part_of(Path(name))
        entry = actual.setdefault(part, {"files": 0, "bytes": 0})
        entry["files"] += 1
        target = dest / name
        if target.exists():
            entry["bytes"] += target.stat().st_size

    expected = (manifest or {}).get("parts", {})
    mismatches: list[str] = []
    if manifest is not None:
        for part in sorted(set(expected) | set(actual)):
            exp_files = int(expected.get(part, {}).get("files", 0))
            got_files = int(actual.get(part, {}).get("files", 0))
            if exp_files != got_files:
                mismatches.append(
                    f"{part}: manifest {exp_files} file, thực tế {got_files} file"
                )

    result = {
        "zip_path": str(archive_path),
        "dest_dir": str(dest),
        "total_files": len([n for n in names if n != MANIFEST_NAME]),
        "parts": dict(sorted(actual.items())),
        "expected_parts": expected,
        "mismatches": mismatches,
        "manifest": manifest,
        "verified": verify,
    }
    if strict and mismatches:
        raise ValueError("Sai lệch artefact: " + "; ".join(mismatches))
    return result


# ---------------------------------------------------------------- CLI
def _parse_only(raw: str | None) -> list[str] | None:
    """Parse ``--only kb,images`` -> ``["kb", "images"]``."""
    if not raw:
        return None
    items = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [part for part in items if part not in PACK_TARGETS]
    if unknown:
        raise ValueError(
            f"Phần không hợp lệ: {', '.join(unknown)}. Chỉ có: {', '.join(PACK_TARGETS)}"
        )
    return items


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m artrag.tools.artifacts",
        description="Đóng gói / giải nén artefact (kb, images, index, cache/captions*).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pack_parser = sub.add_parser("pack", help="nén artefact thành zip")
    pack_parser.add_argument(
        "--data-dir",
        default=None,
        help="thư mục data (mặc định: lấy từ config paths.data_dir)",
    )
    pack_parser.add_argument(
        "--out",
        default="artifacts.zip",
        help="đường dẫn file zip đầu ra",
    )
    pack_parser.add_argument(
        "--only",
        default=None,
        help="chỉ đóng gói các phần, ví dụ --only kb,images",
    )
    pack_parser.add_argument("--profile", default=None, help="profile config")

    unpack_parser = sub.add_parser("unpack", help="giải nén zip")
    unpack_parser.add_argument("zip", help="đường dẫn file zip")
    unpack_parser.add_argument(
        "--dest",
        default=".",
        help="thư mục đích giải nén (mặc định: thư mục hiện tại)",
    )
    unpack_parser.add_argument(
        "--no-verify",
        action="store_true",
        help="không đối chiếu với manifest",
    )
    unpack_parser.add_argument(
        "--strict",
        action="store_true",
        help="ném lỗi nếu số file lệch với manifest",
    )
    return parser


def _print_pack(stats: dict[str, Any]) -> None:
    print(f"Đã đóng gói: {stats['zip_path']}")
    print(
        f"  file: {stats['total_files']} | "
        f"dung lượng gốc: {stats['total_bytes'] / (1024 * 1024):.1f} MB | "
        f"zip: {stats['zip_bytes'] / (1024 * 1024):.1f} MB"
    )
    for part, info in stats["parts"].items():
        print(f"  {part:>10}: {info['files']:>6} file | {info['bytes'] / (1024 * 1024):8.1f} MB")
    print(f"  kb_hash: {stats['kb_hash']}")


def _print_unpack(stats: dict[str, Any]) -> None:
    print(f"Đã giải nén: {stats['zip_path']} -> {stats['dest_dir']}")
    print(f"  file: {stats['total_files']}")
    for part, info in stats["parts"].items():
        print(f"  {part:>10}: {info['files']:>6} file | {info['bytes'] / (1024 * 1024):8.1f} MB")
    if stats["mismatches"]:
        print("  SAI LỆCH so với manifest:")
        for item in stats["mismatches"]:
            print(f"    - {item}")
    elif stats["verified"]:
        print("  Khớp manifest: OK")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.command == "pack":
        from artrag.config import load_config, paths

        cfg = load_config(profile=args.profile)
        data_dir = Path(args.data_dir) if args.data_dir else paths(cfg).data_dir
        try:
            only = _parse_only(args.only)
        except ValueError as exc:
            print(str(exc))
            return 1
        stats = pack(data_dir, args.out, only=only)
        _print_pack(stats)
        return 0

    if args.command == "unpack":
        try:
            stats = unpack(
                args.zip,
                args.dest,
                verify=not args.no_verify,
                strict=args.strict,
            )
        except ValueError as exc:
            print(str(exc))
            return 1
        _print_unpack(stats)
        return 1 if stats["mismatches"] and args.strict else 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
