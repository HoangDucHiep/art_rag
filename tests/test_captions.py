"""Test cho cache caption T3.3, không cần GPU hay model."""

from __future__ import annotations

import pandas as pd

from artrag.index.captions import (
    _merge_records,
    _select_artworks,
    caption_file,
    caption_params_hash,
)


def test_caption_tag_changes_with_model_or_generation_params(tmp_path):
    params_hash = caption_params_hash({"num_beams": 3, "max_new_tokens": 30})
    other_hash = caption_params_hash({"num_beams": 5, "max_new_tokens": 30})

    assert params_hash == caption_params_hash(
        {"max_new_tokens": 30, "num_beams": 3}
    )
    assert params_hash != other_hash
    assert caption_file(tmp_path, "blip_cap", params_hash) != caption_file(
        tmp_path, "other_model", params_hash
    )


def test_merge_records_replaces_matching_variant_and_preserves_old_captions():
    existing = [
        {
            "artwork_id": "cma-1",
            "model": "blip_cap",
            "params_hash": "old-params",
            "caption": "old variant",
            "ms": 10.0,
        },
        {
            "artwork_id": "cma-1",
            "model": "blip_cap",
            "params_hash": "current",
            "caption": "previous caption",
            "ms": 11.0,
        },
    ]
    replacement = {
        "artwork_id": "cma-1",
        "model": "blip_cap",
        "params_hash": "current",
        "caption": "updated caption",
        "ms": 12.0,
    }

    result = _merge_records(existing, [replacement])

    assert result == [existing[0], replacement]


def test_select_artworks_filters_missing_paths_and_applies_limit(tmp_path):
    first_image = tmp_path / "first.jpg"
    second_image = tmp_path / "second.jpg"
    first_image.touch()
    second_image.touch()
    frame = pd.DataFrame(
        [
            {"artwork_id": "cma-1", "image_path": "first.jpg"},
            {"artwork_id": "cma-2", "image_path": None},
            {"artwork_id": "cma-3", "image_path": "missing.jpg"},
            {"artwork_id": "cma-4", "image_path": "second.jpg"},
        ]
    )

    rows = _select_artworks(frame, tmp_path, limit=1)

    assert [row["artwork_id"] for row in rows] == ["cma-1"]
