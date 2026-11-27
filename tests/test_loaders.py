"""Test cho loader thật của model thị giác và embedding (T2.2).

Các test ở đây **không tải model thật** (không cần mạng, không cần GPU) —
chúng kiểm tra:
  - đủ 7 task đã đăng ký loader (clip, sbert, blip_itm, blip_cap, blip_vqa,
    seq2seq, nli);
  - hàm thuần trong `templates.py`;
  - `inference.py` ném lỗi đúng khi task không khớp;
  - registry vẫn import được khi **chưa cài torch**.

Smoke test thật (nạp + suy luận) chạy bằng:
    uv run python -m artrag.registry --smoke clip text_embed blip_itm \
        blip_cap blip_vqa vi2en nli
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from artrag.errors import ModelLoadError
from artrag.registry import LOADERS, ModelHandle, ModelSpec
from artrag.models import templates
from artrag.models.loaders import (
    REGISTERED_TASKS,
    device_str,
    resolve_device,
    torch_dtype,
)


# ---------------------------------------------------------------------------
# (a) Đăng ký loader
# ---------------------------------------------------------------------------


def test_all_seven_tasks_have_loaders():
    for task in REGISTERED_TASKS:
        assert task in LOADERS, f"thiếu loader cho task {task!r}"
        assert callable(LOADERS[task])


def test_registered_tasks_is_exactly_the_seven():
    assert set(REGISTERED_TASKS) == {
        "clip",
        "sbert",
        "blip_itm",
        "blip_cap",
        "blip_vqa",
        "seq2seq",
        "nli",
    }


def test_importing_loaders_module_populates_registry():
    """Import `artrag.models.loaders` phải tự đăng ký vào LOADERS (idempotent)."""
    import importlib

    from artrag.models import loaders as loaders_module

    importlib.reload(loaders_module)
    for task in REGISTERED_TASKS:
        assert task in LOADERS


# ---------------------------------------------------------------------------
# (b) Tiện ích dtype / device
# ---------------------------------------------------------------------------


def test_device_str_normalises_cuda_variants():
    assert device_str("cuda") == "cuda"
    assert device_str("CUDA") == "cuda"
    assert device_str("cuda:0") == "cuda"
    assert device_str("cpu") == "cpu"
    assert device_str("") == "cpu"


def test_torch_dtype_rejects_unknown():
    with pytest.raises(ModelLoadError):
        torch_dtype("int4")


def test_resolve_device_maps_cuda_to_index():
    import torch

    assert resolve_device("cuda") == torch.device("cuda:0")
    assert resolve_device("cpu") == torch.device("cpu")


def test_torch_dtype_returns_expected_dtypes():
    import torch

    assert torch_dtype("fp16") is torch.float16
    assert torch_dtype("fp32") is torch.float32
    assert torch_dtype("bf16") is torch.bfloat16


# ---------------------------------------------------------------------------
# (c) Template CLIP (T0/T1/T2)
# ---------------------------------------------------------------------------


def test_clip_templates_render():
    assert templates.render_clip_text("a boat", "T0") == "a boat"
    assert templates.render_clip_text("a boat", "T1") == "a painting of a boat"
    assert templates.render_clip_text("a boat", "T2") == "a museum artwork depicting a boat"


def test_clip_template_default_is_t1():
    assert templates.DEFAULT_CLIP_TEMPLATE == "T1"
    assert templates.render_clip_text("a boat") == "a painting of a boat"


def test_clip_template_strips_whitespace():
    assert templates.render_clip_text("  a boat  ", "T1") == "a painting of a boat"


def test_clip_template_invalid_raises():
    with pytest.raises(ValueError):
        templates.render_clip_text("a boat", "T9")


def test_clip_templates_cover_t0_t1_t2():
    assert set(templates.CLIP_TEMPLATES) == {"T0", "T1", "T2"}


# ---------------------------------------------------------------------------
# (d) Dispatch của inference theo task
# ---------------------------------------------------------------------------


def make_handle(task: str, name: str = "m") -> ModelHandle:
    return ModelHandle(
        name=name,
        model=object(),
        processor=object(),
        device="cpu",
        dtype="fp32",
        task=task,
    )


@pytest.mark.parametrize(
    ("func", "wrong_task", "args"),
    [
        ("embed_image", "sbert", (None,)),
        ("embed_text", "nli", ("x",)),
        ("itm_score", "clip", (None, "x")),
        ("caption", "blip_vqa", (None,)),
        ("vqa", "blip_cap", (None, "x")),
        ("translate", "nli", ("x",)),
        ("nli_label", "seq2seq", ("a", "b")),
    ],
)
def test_inference_rejects_wrong_task(func, wrong_task, args):
    from artrag.models import inference

    handle = make_handle(wrong_task)
    with pytest.raises(ModelLoadError):
        getattr(inference, func)(handle, *args)


def test_inference_error_message_mentions_task_and_name():
    from artrag.models import inference

    handle = make_handle("nli", name="model_nli")
    with pytest.raises(ModelLoadError) as exc:
        inference.embed_image(handle, None)
    assert "nli" in str(exc.value)
    assert "model_nli" in str(exc.value)


def test_embed_text_accepts_clip_and_sbert():
    """Hàm embed_text phục vụ được cả CLIP lẫn BGE-M3."""
    from artrag.models import inference

    assert inference._require is not None
    # _require chỉ ném lỗi khi task không thuộc danh sách cho phép.
    inference._require(make_handle("clip"), "embed_text", "clip", "sbert")
    inference._require(make_handle("sbert"), "embed_text", "clip", "sbert")


# ---------------------------------------------------------------------------
# (e) Registry import được khi chưa cài torch
# ---------------------------------------------------------------------------


def test_registry_importable_without_torch():
    """`artrag.registry` phải import được khi torch chưa có (thiết kế §6.2)."""
    code = (
        "import sys, types\n"
        "sys.modules['torch'] = None\n"
        "import artrag.registry as r\n"
        "assert 'clip' in r.ModelSpec.from_dict('x', {'task':'clip','hf_id':'a'}).to_dict()['task']\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_templates_module_has_no_torch_dependency():
    """templates.py là hàm thuần -> import được mà không cần torch."""
    code = (
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "from artrag.models.templates import render_clip_text\n"
        "assert render_clip_text('x', 'T1') == 'a painting of x'\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


# ---------------------------------------------------------------------------
# (f) Spec của model dịch có src_lang/tgt_lang
# ---------------------------------------------------------------------------


def test_models_yaml_has_lang_codes_for_translators():
    from artrag.registry import load_model_specs

    specs = load_model_specs()
    assert specs["vi2en"].src_lang == "vi_VN"
    assert specs["vi2en"].tgt_lang == "en_XX"
    assert specs["en2vi"].src_lang == "en_XX"
    assert specs["en2vi"].tgt_lang == "vi_VN"


def test_model_spec_roundtrip_keeps_lang_codes():
    spec = ModelSpec.from_dict(
        "vi2en",
        {"task": "seq2seq", "hf_id": "x", "src_lang": "vi_VN", "tgt_lang": "en_XX"},
    )
    data = spec.to_dict()
    assert data["src_lang"] == "vi_VN"
    assert data["tgt_lang"] == "en_XX"
