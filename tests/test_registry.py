"""Test cho ModelRegistry (T2.1) — loader giả + GPU giả, không cần torch."""

from __future__ import annotations

import threading
import time

import pytest

from artrag.config import load_config
from artrag.errors import ModelLoadError
from artrag.registry import (
    LOADERS,
    FakeDeviceManager,
    ModelRegistry,
    ModelSpec,
    create_registry,
    is_oom,
    load_model_specs,
    register_loader,
)


# ---------------------------------------------------------------------------
# Tiện ích test
# ---------------------------------------------------------------------------


def make_spec(name: str, **kwargs) -> ModelSpec:
    data = {
        "task": "fake",
        "hf_id": f"fake/{name}",
        "dtype": "fp16",
        "est_vram_mb": 100,
        "devices": ("cuda", "cpu"),
    }
    data.update(kwargs)
    return ModelSpec.from_dict(name, data)


class RecordingLoader:
    """Loader giả: ghi lại lần gọi, có thể ném lỗi theo yêu cầu.

    - `fail_on[(name, device)]`: lỗi luôn xảy ra với cặp (model, thiết bị).
    - `fail_once[(name, device)]`: lỗi chỉ xảy ra ở lần gọi đầu tiên.
    """

    def __init__(
        self,
        fail_on: dict[tuple[str, str], BaseException] | None = None,
        fail_once: dict[tuple[str, str], BaseException] | None = None,
    ):
        self.calls: list[tuple[str, str, str]] = []
        self.fail_on = dict(fail_on or {})
        self.fail_once = dict(fail_once or {})
        self.lock = threading.Lock()

    def __call__(self, spec: ModelSpec, device: str, dtype: str):
        with self.lock:
            self.calls.append((spec.name, device, dtype))
            key = (spec.name, device)
            if key in self.fail_once:
                exc = self.fail_once.pop(key)
                raise exc
            if key in self.fail_on:
                raise self.fail_on[key]
        return {"model": spec.name, "device": device}, {"processor": spec.name}


def make_registry(specs, loader, *, budget=0, devices=("cuda", "cpu"), evict_to="free", on_load=None):
    manager = FakeDeviceManager(available_devices=devices, on_load=on_load)
    registry = ModelRegistry(
        specs={s.name: s for s in specs},
        device_manager=manager,
        vram_budget_mb=budget,
        evict_to=evict_to,
        loaders={"fake": loader},
    )
    return registry, manager


# ---------------------------------------------------------------------------
# (a) Chỉ nạp một lần
# ---------------------------------------------------------------------------


def test_loader_called_only_once():
    loader = RecordingLoader()
    registry, _ = make_registry([make_spec("a")], loader)

    first = registry.get("a")
    second = registry.get("a")

    assert len(loader.calls) == 1
    assert first is second
    assert first.model == {"model": "a", "device": "cuda"}
    assert first.processor == {"processor": "a"}


def test_preload_loads_each_model_once():
    loader = RecordingLoader()
    registry, _ = make_registry([make_spec("a"), make_spec("b")], loader)

    handles = registry.preload(["a", "b", "a"])

    assert [h.name for h in handles] == ["a", "b", "a"]
    assert len(loader.calls) == 2


def test_concurrent_get_loads_once():
    """Hai thread gọi `get` cùng lúc -> loader chỉ chạy một lần (§6.2 hành vi 1)."""
    loader = RecordingLoader()
    registry, _ = make_registry([make_spec("a")], loader)

    barrier = threading.Barrier(2)
    results = []

    def worker():
        barrier.wait()
        results.append(registry.get("a"))

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(loader.calls) == 1
    assert results[0] is results[1]


# ---------------------------------------------------------------------------
# (b) LRU + pinned
# ---------------------------------------------------------------------------


def test_lru_evicts_oldest_when_budget_exceeded():
    loader = RecordingLoader()
    specs = [
        make_spec("a", est_vram_mb=100),
        make_spec("b", est_vram_mb=100),
        make_spec("c", est_vram_mb=100),
    ]
    registry, _ = make_registry(specs, loader, budget=200)

    registry.get("a")
    registry.get("b")
    # Chạm lại "a" -> "b" trở thành LRU
    registry.get("a")
    registry.get("c")

    assert "b" not in registry
    assert "a" in registry
    assert "c" in registry


def test_pinned_model_is_never_evicted():
    """Model ghim không bao giờ bị giải phóng; model khác phải nhường chỗ."""
    loader = RecordingLoader()
    specs = [
        make_spec("pinned_one", est_vram_mb=100, pinned=True),
        make_spec("other", est_vram_mb=100),
    ]
    registry, _ = make_registry(specs, loader, budget=100)

    handle = registry.get("pinned_one")
    assert handle.device == "cuda"
    assert "pinned_one" in registry

    # Không còn chỗ trên GPU (pinned không được giải phóng) -> "other" xuống CPU
    other = registry.get("other")
    assert other.device == "cpu"

    # Model ghim vẫn nguyên trên cuda
    assert registry.get("pinned_one").device == "cuda"
    assert registry.get("pinned_one").model["device"] == "cuda"


def test_pinned_stays_on_gpu_when_other_evicted():
    loader = RecordingLoader()
    specs = [
        make_spec("pin", est_vram_mb=100, pinned=True),
        make_spec("a", est_vram_mb=100),
        make_spec("b", est_vram_mb=100),
    ]
    registry, _ = make_registry(specs, loader, budget=200)

    registry.get("pin")
    registry.get("a")
    registry.get("b")

    assert registry.get("pin").device == "cuda"
    assert "a" not in registry


def test_unload_frees_budget():
    loader = RecordingLoader()
    specs = [make_spec("a", est_vram_mb=100), make_spec("b", est_vram_mb=100)]
    registry, _ = make_registry(specs, loader, budget=100)

    registry.get("a")
    assert registry.unload("a") is True
    assert "a" not in registry
    assert registry.unload("a") is False

    # Chỗ đã trống -> "b" nạp được trên cuda
    assert registry.get("b").device == "cuda"


def test_evict_to_cpu_moves_lru_model_to_cpu():
    """`evict_to="cpu"`: model bị giải phóng được nạp lại trên CPU, vẫn dùng được."""
    loader = RecordingLoader()
    specs = [
        make_spec("a", est_vram_mb=100),
        make_spec("b", est_vram_mb=100),
    ]
    registry, _ = make_registry(specs, loader, budget=100, evict_to="cpu")

    registry.get("a")
    registry.get("b")

    assert "a" in registry
    assert registry.get("a").device == "cpu"
    assert registry.get("a").dtype == "fp32"


def test_evict_to_free_removes_lru_model_entirely():
    loader = RecordingLoader()
    specs = [
        make_spec("a", est_vram_mb=100),
        make_spec("b", est_vram_mb=100),
    ]
    registry, _ = make_registry(specs, loader, budget=100, evict_to="free")

    registry.get("a")
    registry.get("b")

    assert "a" not in registry
    assert registry.get("b").device == "cuda"


# ---------------------------------------------------------------------------
# (c) OOM -> fallback CPU
# ---------------------------------------------------------------------------


def test_oom_falls_back_to_cpu():
    loader = RecordingLoader(fail_on={("a", "cuda"): RuntimeError("CUDA out of memory")})
    registry, _ = make_registry([make_spec("a")], loader, devices=("cuda", "cpu"))

    handle = registry.get("a")

    assert handle.device == "cpu"
    assert handle.dtype == "fp32"
    # đã thử cuda (2 lần: lần đầu + retry) rồi mới tới cpu
    assert [c[1] for c in loader.calls] == ["cuda", "cuda", "cpu"]


def test_oom_without_cpu_raises_model_load_error():
    loader = RecordingLoader(fail_on={("a", "cuda"): RuntimeError("CUDA out of memory")})
    registry, _ = make_registry(
        [make_spec("a", devices=("cuda",))], loader, devices=("cuda",)
    )

    with pytest.raises(ModelLoadError):
        registry.get("a")


def test_oom_evicts_unpinned_then_retries_on_same_device():
    loader = RecordingLoader(fail_once={("a", "cuda"): RuntimeError("CUDA out of memory")})
    specs = [make_spec("other", est_vram_mb=100), make_spec("a", est_vram_mb=100)]
    registry, _ = make_registry(specs, loader, devices=("cuda", "cpu"))

    registry.get("other")
    handle = registry.get("a")

    # Retry lần 2 trên cuda thành công -> vẫn ở cuda
    assert handle.device == "cuda"
    assert "other" not in registry  # đã bị giải phóng để nhả VRAM


def test_non_oom_error_raises_model_load_error():
    loader = RecordingLoader(fail_on={("a", "cuda"): ValueError("hỏng model")})
    registry, _ = make_registry([make_spec("a")], loader)

    with pytest.raises(ModelLoadError):
        registry.get("a")


def test_is_oom_detects_cuda_errors():
    assert is_oom(RuntimeError("CUDA out of memory"))
    assert not is_oom(ValueError("lỗi khác"))


# ---------------------------------------------------------------------------
# (d) stats()
# ---------------------------------------------------------------------------


def test_stats_reflects_loaded_models():
    loader = RecordingLoader()
    specs = [
        make_spec("a", est_vram_mb=100, pinned=True),
        make_spec("b", est_vram_mb=200, dtype="fp16"),
    ]
    registry, _ = make_registry(specs, loader, budget=1000)

    registry.get("a")
    registry.get("b")
    stats = registry.stats()

    assert stats["budget_mb"] == 1000
    assert stats["loaded"] == 2
    assert stats["vram_est_mb"] == 300
    assert set(stats["devices"]) == {"cuda"}
    assert stats["models"]["a"]["pinned"] is True
    assert stats["models"]["a"]["est_vram_mb"] == 100
    assert stats["models"]["a"]["device"] == "cuda"
    assert stats["models"]["a"]["dtype"] == "fp16"
    assert stats["models"]["b"]["est_vram_mb"] == 200
    assert stats["models"]["b"]["load_ms"] >= 0
    assert stats["models"]["a"]["task"] == "fake"
    # §6.2: stats() phải có "lần dùng cuối"
    assert stats["models"]["a"]["last_used_at"] > 0
    assert stats["models"]["b"]["last_used_at"] >= stats["models"]["a"]["last_used_at"]


def test_stats_empty_registry():
    loader = RecordingLoader()
    registry, _ = make_registry([], loader)
    stats = registry.stats()
    assert stats["loaded"] == 0
    assert stats["models"] == {}


# ---------------------------------------------------------------------------
# Lỗi khác
# ---------------------------------------------------------------------------


def test_unknown_model_name_raises():
    loader = RecordingLoader()
    registry, _ = make_registry([make_spec("a")], loader)
    with pytest.raises(ModelLoadError):
        registry.get("khong_ton_tai")


def test_unknown_task_raises():
    registry, _ = make_registry([make_spec("a", task="chua_dang_ky")], RecordingLoader())
    with pytest.raises(ModelLoadError):
        registry.get("a")


def test_no_available_device_raises():
    loader = RecordingLoader()
    registry, _ = make_registry([make_spec("a")], loader, devices=())
    with pytest.raises(ModelLoadError):
        registry.get("a")


def test_cpu_uses_fp32_even_if_spec_says_fp16():
    loader = RecordingLoader()
    registry, _ = make_registry(
        [make_spec("a", dtype="fp16")], loader, devices=("cpu",)
    )
    handle = registry.get("a")
    assert handle.device == "cpu"
    assert handle.dtype == "fp32"


# ---------------------------------------------------------------------------
# Đăng ký loader + spec từ config
# ---------------------------------------------------------------------------


def test_register_loader_decorator():
    calls = []

    @register_loader("task_thu_nghiem")
    def loader_fn(spec, device, dtype):
        calls.append((spec.name, device, dtype))
        return "model", "processor"

    try:
        assert "task_thu_nghiem" in LOADERS
        registry = ModelRegistry(
            specs={"a": make_spec("a", task="task_thu_nghiem")},
            device_manager=FakeDeviceManager(),
        )
        handle = registry.get("a")
        assert handle.model == "model"
        assert calls == [("a", "cuda", "fp16")]
    finally:
        LOADERS.pop("task_thu_nghiem", None)


def test_load_model_specs_reads_models_yaml():
    specs = load_model_specs()
    assert "clip" in specs
    assert specs["clip"].task == "clip"
    assert specs["clip"].hf_id == "openai/clip-vit-base-patch32"
    assert specs["clip"].devices == ("cuda", "cpu")
    assert specs["text_embed"].est_vram_mb == 1150
    assert specs["qwen3b"].quant == "4bit"


def test_load_model_specs_applies_profile_overrides():
    cfg = load_config(profile="local")
    specs = load_model_specs(cfg)
    assert specs["clip"].pinned is True
    assert specs["text_embed"].devices == ("cpu",)
    assert specs["vi2en"].devices == ("cpu",)


def test_create_registry_uses_profile_budget():
    cfg = load_config(profile="local")
    registry = create_registry(cfg, device_manager=FakeDeviceManager())
    assert registry.vram_budget_mb == 3600
    assert "clip" in registry.specs
    assert registry.specs["clip"].pinned is True


def test_create_registry_reads_evict_to_from_config():
    cfg = load_config(profile="local", overrides=["models.evict_to=free"])
    registry = create_registry(cfg, device_manager=FakeDeviceManager())
    assert registry.evict_to == "free"


def test_model_spec_roundtrip_dict():
    spec = make_spec("a", pinned=True)
    data = spec.to_dict()
    assert data["name"] == "a"
    assert data["pinned"] is True
    assert data["devices"] == ["cuda", "cpu"]
