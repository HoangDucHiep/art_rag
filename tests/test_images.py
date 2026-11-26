"""Test cho T1.6 ``artrag.data.images``.

Tất cả test chạy **không cần mạng**: dùng ảnh giả tạo bằng Pillow.
"""

from __future__ import annotations

import io
import json

import pandas as pd
import pytest
from PIL import Image

from artrag.data.build_kb import preserve_image_paths
from artrag.data.images import (
    FAILED_FILENAME,
    convert_image,
    failed_path,
    image_path_for,
    load_failed,
    plan_downloads,
    relative_image_path,
    save_failed,
    update_artworks_parquet,
)


def _make_image(
    size: tuple[int, int] = (800, 600),
    mode: str = "RGB",
    color: tuple[int, ...] | int | None = None,
) -> bytes:
    """Tạo bytes ảnh giả để test (màu mặc định theo mode)."""
    if color is None:
        color = {
            "RGB": (200, 40, 40),
            "RGBA": (200, 40, 40, 255),
            "L": 128,
            "CMYK": (10, 80, 90, 5),
        }.get(mode, 0)
    # PNG không hỗ trợ CMYK -> dùng JPEG cho mode đó
    fmt = "JPEG" if mode == "CMYK" else "PNG"
    buffer = io.BytesIO()
    Image.new(mode, size, color).save(buffer, format=fmt)
    return buffer.getvalue()


def _artwork_row(**kwargs):
    row = {
        "artwork_id": "cma-1",
        "image_url": "https://example.org/1.jpg",
        "image_path": None,
    }
    row.update(kwargs)
    return row


# ---------------------------------------------------------------- convert
def test_convert_image_resizes_longest_side_to_512():
    payload = convert_image(_make_image((1200, 800)), max_side=512, quality=90)
    with Image.open(io.BytesIO(payload)) as img:
        assert img.format == "JPEG"
        assert img.mode == "RGB"
        assert max(img.size) == 512
        # tỷ lệ giữ nguyên
        assert img.size == (512, 341)


def test_convert_image_keeps_small_image_as_is():
    payload = convert_image(_make_image((200, 100)), max_side=512, quality=90)
    with Image.open(io.BytesIO(payload)) as img:
        assert img.size == (200, 100)


def test_convert_image_converts_rgba_and_grayscale_to_rgb():
    for mode in ("RGBA", "L", "CMYK"):
        payload = convert_image(_make_image((300, 300), mode=mode), max_side=512)
        with Image.open(io.BytesIO(payload)) as img:
            assert img.mode == "RGB", mode
            assert img.format == "JPEG", mode


def test_convert_image_rejects_broken_bytes():
    with pytest.raises(Exception):
        convert_image(b"khong phai anh")


# ---------------------------------------------------------------- paths
def test_relative_image_path_is_relative_to_data_dir():
    assert relative_image_path("cma-151904") == "images/cma-151904.jpg"


def test_image_path_for_uses_jpg_extension(tmp_path):
    path = image_path_for("cma-1", tmp_path)
    assert path.name == "cma-1.jpg"
    assert path.parent == tmp_path


def test_failed_path_is_at_data_root(tmp_path):
    path = failed_path(tmp_path)
    assert path.name == FAILED_FILENAME
    assert path.parent == tmp_path


# ---------------------------------------------------------------- plan
def test_plan_downloads_skips_existing_files(tmp_path):
    image_path_for("cma-1", tmp_path).write_bytes(b"x")
    rows = [_artwork_row(artwork_id="cma-1"), _artwork_row(artwork_id="cma-2")]
    plan = plan_downloads(rows, tmp_path)
    assert [item["artwork_id"] for item in plan] == ["cma-2"]


def test_plan_downloads_force_includes_existing(tmp_path):
    image_path_for("cma-1", tmp_path).write_bytes(b"x")
    rows = [_artwork_row(artwork_id="cma-1")]
    plan = plan_downloads(rows, tmp_path, force=True)
    assert len(plan) == 1


def test_plan_downloads_skips_rows_without_url(tmp_path):
    rows = [_artwork_row(artwork_id="cma-1", image_url=None)]
    assert plan_downloads(rows, tmp_path) == []


def test_plan_downloads_retry_failed_only_keeps_failed_ids(tmp_path):
    rows = [_artwork_row(artwork_id="cma-1"), _artwork_row(artwork_id="cma-2")]
    plan = plan_downloads(rows, tmp_path, failed_ids={"cma-2"}, retry_failed=True)
    assert [item["artwork_id"] for item in plan] == ["cma-2"]


def test_plan_downloads_retry_failed_with_empty_set_is_empty(tmp_path):
    rows = [_artwork_row(artwork_id="cma-1")]
    plan = plan_downloads(rows, tmp_path, failed_ids=set(), retry_failed=True)
    assert plan == []


# ---------------------------------------------------------------- failed json
def test_save_and_load_failed_roundtrip(tmp_path):
    path = failed_path(tmp_path)
    failed = {"cma-2": "HTTPError: 404", "cma-1": "Timeout: quá thời gian"}
    save_failed(path, failed)
    assert load_failed(path) == failed


def test_load_failed_missing_file_returns_empty(tmp_path):
    assert load_failed(failed_path(tmp_path)) == {}


def test_load_failed_invalid_json_returns_empty(tmp_path):
    path = failed_path(tmp_path)
    path.write_text("{khong phai json", encoding="utf-8")
    assert load_failed(path) == {}


def test_save_failed_writes_valid_utf8_json(tmp_path):
    path = failed_path(tmp_path)
    save_failed(path, {"cma-1": "Lỗi mạng"})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["cma-1"] == "Lỗi mạng"


# ---------------------------------------------------------------- parquet
def test_update_artworks_parquet_fills_existing_and_keeps_null(tmp_path):
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    kb_path = kb_dir / "artworks.parquet"
    frame = pd.DataFrame(
        [
            {"artwork_id": "cma-1", "title": "A"},
            {"artwork_id": "cma-2", "title": "B"},
        ]
    )
    frame.to_parquet(kb_path, engine="pyarrow", index=False)

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    image_path_for("cma-1", images_dir).write_bytes(b"x")

    filled = update_artworks_parquet(kb_path, images_dir)
    assert filled == 1

    updated = pd.read_parquet(kb_path)
    assert len(updated) == 2
    assert updated.loc[0, "image_path"] == "images/cma-1.jpg"
    assert pd.isna(updated.loc[1, "image_path"])
    # các cột khác không bị mất
    assert list(updated["title"]) == ["A", "B"]


def test_update_artworks_parquet_missing_file_returns_zero(tmp_path):
    assert update_artworks_parquet(tmp_path / "khong-co.parquet", tmp_path) == 0


def test_update_artworks_parquet_creates_column_if_missing(tmp_path):
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    kb_path = kb_dir / "artworks.parquet"
    pd.DataFrame([{"artwork_id": "cma-1"}]).to_parquet(
        kb_path, engine="pyarrow", index=False
    )
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    image_path_for("cma-1", images_dir).write_bytes(b"x")
    assert update_artworks_parquet(kb_path, images_dir) == 1
    assert "image_path" in pd.read_parquet(kb_path).columns


# ---------------------------------------------------------------- build_kb giữ image_path
def _write_old_kb(kb_path, entries):
    kb_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(entries).to_parquet(kb_path, engine="pyarrow", index=False)


def test_preserve_image_paths_keeps_value_when_file_exists(tmp_path):
    kb_path = tmp_path / "kb" / "artworks.parquet"
    _write_old_kb(
        kb_path,
        [{"artwork_id": "cma-1", "image_path": "images/cma-1.jpg"}],
    )
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    image_path_for("cma-1", images_dir).write_bytes(b"x")

    rows = [{"artwork_id": "cma-1", "image_path": None}]
    kept = preserve_image_paths(rows, kb_path, images_dir)
    assert kept == 1
    assert rows[0]["image_path"] == "images/cma-1.jpg"


def test_preserve_image_paths_drops_value_when_file_missing(tmp_path):
    kb_path = tmp_path / "kb" / "artworks.parquet"
    _write_old_kb(
        kb_path,
        [{"artwork_id": "cma-1", "image_path": "images/cma-1.jpg"}],
    )
    rows = [{"artwork_id": "cma-1", "image_path": None}]
    kept = preserve_image_paths(rows, kb_path, tmp_path / "images")
    assert kept == 0
    assert rows[0]["image_path"] is None


def test_preserve_image_paths_without_old_kb_returns_zero(tmp_path):
    rows = [{"artwork_id": "cma-1", "image_path": None}]
    assert preserve_image_paths(rows, tmp_path / "khong-co.parquet", tmp_path) == 0
    assert rows[0]["image_path"] is None
