"""Test cho T1.7 ``artrag.tools.artifacts``.

Tất cả test chạy **không cần mạng, không cần GPU**: dùng cây thư mục giả.
"""

from __future__ import annotations

import json
import zipfile

import pandas as pd
import pytest

from artrag.tools.artifacts import (
    MANIFEST_NAME,
    PACK_TARGETS,
    build_manifest,
    collect_members,
    kb_fingerprint,
    main,
    pack,
    read_manifest,
    unpack,
)


def _make_data_dir(tmp_path):
    """Dựng cây ``data/`` giả: kb, images, index, cache (captions + llm)."""
    data_dir = tmp_path / "data"

    kb = data_dir / "kb"
    kb.mkdir(parents=True)
    pd.DataFrame(
        [
            {"artwork_id": "cma-1", "image_path": "images/cma-1.jpg"},
            {"artwork_id": "cma-2", "image_path": None},
        ]
    ).to_parquet(kb / "artworks.parquet", engine="pyarrow", index=False)

    images = data_dir / "images"
    images.mkdir(parents=True)
    (images / "cma-1.jpg").write_bytes(b"jpeg-bytes-1")
    (images / "cma-2.jpg").write_bytes(b"jpeg-bytes-2")

    index = data_dir / "index" / "clip-test"
    index.mkdir(parents=True)
    (index / "emb.npy").write_bytes(b"npy")
    (index / "meta.json").write_text("{}", encoding="utf-8")

    cache = data_dir / "cache"
    cache.mkdir(parents=True)
    (cache / "captions-base.parquet").write_bytes(b"captions")
    (cache / "llm").mkdir()
    (cache / "llm" / "response.bin").write_bytes(b"llm-cache")
    (cache / "vqa").mkdir()
    (cache / "vqa" / "answer.bin").write_bytes(b"vqa-cache")

    return data_dir


# ---------------------------------------------------------------- collect
def test_collect_members_includes_kb_images_index(tmp_path):
    data_dir = _make_data_dir(tmp_path)
    names = {arcname for _, arcname in collect_members(data_dir)}
    assert "kb/artworks.parquet" in names
    assert "images/cma-1.jpg" in names
    assert "index/clip-test/emb.npy" in names


def test_collect_members_only_includes_captions_from_cache(tmp_path):
    data_dir = _make_data_dir(tmp_path)
    names = {arcname for _, arcname in collect_members(data_dir)}
    assert "cache/captions-base.parquet" in names
    assert "cache/llm/response.bin" not in names
    assert "cache/vqa/answer.bin" not in names


def test_collect_members_respects_only_filter(tmp_path):
    data_dir = _make_data_dir(tmp_path)
    names = {arcname for _, arcname in collect_members(data_dir, only=["kb"])}
    assert names == {"kb/artworks.parquet"}


def test_collect_members_on_missing_dir_returns_empty(tmp_path):
    assert collect_members(tmp_path / "khong-co") == []


def test_pack_targets_are_the_design_set():
    assert PACK_TARGETS == ("kb", "images", "index", "cache")


# ---------------------------------------------------------------- manifest
def test_kb_fingerprint_is_stable(tmp_path):
    data_dir = _make_data_dir(tmp_path)
    assert kb_fingerprint(data_dir) == kb_fingerprint(data_dir)


def test_kb_fingerprint_changes_when_artwork_changes(tmp_path):
    data_dir = _make_data_dir(tmp_path)
    before = kb_fingerprint(data_dir)
    pd.DataFrame(
        [{"artwork_id": "cma-9", "image_path": None}]
    ).to_parquet(data_dir / "kb" / "artworks.parquet", engine="pyarrow", index=False)
    assert kb_fingerprint(data_dir) != before


def test_kb_fingerprint_missing_kb_is_none(tmp_path):
    assert kb_fingerprint(tmp_path / "khong-co") is None


def test_build_manifest_counts_parts_and_bytes(tmp_path):
    data_dir = _make_data_dir(tmp_path)
    members = collect_members(data_dir)
    manifest = build_manifest(data_dir, members)
    assert manifest["total_files"] == len(members)
    assert manifest["parts"]["kb"]["files"] == 1
    assert manifest["parts"]["images"]["files"] == 2
    assert manifest["parts"]["cache"]["files"] == 1
    assert manifest["total_bytes"] > 0
    assert manifest["kb_hash"] == kb_fingerprint(data_dir)
    assert "created_at" in manifest


# ---------------------------------------------------------------- pack
def test_pack_creates_zip_with_manifest(tmp_path):
    data_dir = _make_data_dir(tmp_path)
    out = tmp_path / "out" / "artifacts.zip"
    stats = pack(data_dir, out)
    assert out.exists()
    assert stats["zip_bytes"] > 0
    with zipfile.ZipFile(out) as archive:
        names = archive.namelist()
    assert MANIFEST_NAME in names
    assert "kb/artworks.parquet" in names
    assert "cache/captions-base.parquet" in names
    assert "cache/llm/response.bin" not in names


def test_pack_only_limits_contents(tmp_path):
    data_dir = _make_data_dir(tmp_path)
    out = tmp_path / "only.zip"
    stats = pack(data_dir, out, only=["kb", "images"])
    assert set(stats["parts"]) == {"kb", "images"}
    with zipfile.ZipFile(out) as archive:
        names = set(archive.namelist())
    assert "index/clip-test/emb.npy" not in names


def test_read_manifest_roundtrip(tmp_path):
    data_dir = _make_data_dir(tmp_path)
    out = tmp_path / "artifacts.zip"
    pack(data_dir, out)
    manifest = read_manifest(out)
    assert manifest is not None
    assert manifest["parts"]["images"]["files"] == 2
    assert manifest["kb_hash"] == kb_fingerprint(data_dir)


# ---------------------------------------------------------------- unpack
def test_pack_unpack_roundtrip_content_matches(tmp_path):
    data_dir = _make_data_dir(tmp_path)
    out = tmp_path / "artifacts.zip"
    stats = pack(data_dir, out)
    dest = tmp_path / "dest"
    result = unpack(out, dest)
    assert result["mismatches"] == []
    assert result["total_files"] == stats["total_files"]
    assert (dest / "kb" / "artworks.parquet").exists()
    assert (dest / "images" / "cma-1.jpg").read_bytes() == b"jpeg-bytes-1"
    assert (dest / "cache" / "captions-base.parquet").read_bytes() == b"captions"


def test_unpack_reports_mismatch(tmp_path):
    data_dir = _make_data_dir(tmp_path)
    out = tmp_path / "artifacts.zip"
    pack(data_dir, out)
    # thêm 1 file lạ vào zip để tạo sai lệch số file
    with zipfile.ZipFile(out, "a") as archive:
        archive.writestr("images/cma-99.jpg", b"tam")
    dest = tmp_path / "dest2"
    result = unpack(out, dest)
    assert result["mismatches"], "phải phát hiện sai lệch số file"
    assert any("images" in item for item in result["mismatches"])


def test_unpack_strict_raises_on_mismatch(tmp_path):
    data_dir = _make_data_dir(tmp_path)
    out = tmp_path / "artifacts.zip"
    pack(data_dir, out)
    with zipfile.ZipFile(out, "a") as archive:
        archive.writestr("images/cma-99.jpg", b"tam")
    with pytest.raises(ValueError):
        unpack(out, tmp_path / "dest3", strict=True)


def test_unpack_without_manifest_still_works(tmp_path):
    data_dir = _make_data_dir(tmp_path)
    out = tmp_path / "khong-manifest.zip"
    with zipfile.ZipFile(out, "w") as archive:
        archive.writestr("kb/x.txt", "x")
    result = unpack(out, tmp_path / "dest4")
    assert result["manifest"] is None
    assert result["total_files"] == 1
    assert result["mismatches"] == []


def test_manifest_json_is_valid_utf8(tmp_path):
    data_dir = _make_data_dir(tmp_path)
    out = tmp_path / "artifacts.zip"
    pack(data_dir, out)
    with zipfile.ZipFile(out) as archive:
        payload = json.loads(archive.read(MANIFEST_NAME).decode("utf-8"))
    assert payload["total_files"] > 0


# ---------------------------------------------------------------- CLI
def test_cli_pack_and_unpack_roundtrip(tmp_path, capsys):
    data_dir = _make_data_dir(tmp_path)
    out = tmp_path / "cli.zip"
    code = main(["pack", "--data-dir", str(data_dir), "--out", str(out)])
    assert code == 0
    assert out.exists()

    dest = tmp_path / "cli_dest"
    code = main(["unpack", str(out), "--dest", str(dest)])
    assert code == 0
    assert "Khớp manifest: OK" in capsys.readouterr().out


def test_cli_unpack_strict_returns_error_code_not_traceback(tmp_path, capsys):
    data_dir = _make_data_dir(tmp_path)
    out = tmp_path / "cli_strict.zip"
    pack(data_dir, out)
    with zipfile.ZipFile(out, "a") as archive:
        archive.writestr("images/cma-99.jpg", b"tam")
    code = main(["unpack", str(out), "--dest", str(tmp_path / "d"), "--strict"])
    assert code == 1
    captured = capsys.readouterr().out
    assert "Sai lệch artefact" in captured
    assert "Traceback" not in captured


def test_cli_pack_rejects_unknown_only_part(tmp_path, capsys):
    data_dir = _make_data_dir(tmp_path)
    code = main(
        ["pack", "--data-dir", str(data_dir), "--out", str(tmp_path / "x.zip"), "--only", "kb,xyz"]
    )
    assert code == 1
    assert "xyz" in capsys.readouterr().out
