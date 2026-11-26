"""ModelRegistry: lazy load, quản lý thiết bị và ngân sách VRAM (SYSTEM_DESIGN 6.2).

Tất cả model trong hệ thống đều phải đi qua `ModelRegistry` (không gọi
`from_pretrained` rải rác). Registry chịu trách nhiệm:

- **Lazy**: chỉ nạp model ở lần `get()` đầu tiên.
- **Khoá theo tên**: mỗi model có một `threading.Lock` riêng (Gradio có thể gọi
  đồng thời) → không bao giờ nạp trùng.
- **Ngân sách VRAM**: trước khi nạp lên cuda, nếu tổng VRAM ước tính vượt
  `vram_budget_mb` thì giải phóng model **LRU, không pinned** cho tới khi đủ chỗ.
- **OOM**: giải phóng mọi model không pinned → thử lại một lần → còn lỗi thì
  fallback sang CPU (nếu `devices` cho phép) → không còn cách nào thì ném
  `ModelLoadError`.
- **Đo đạc**: thời gian nạp, VRAM thực đo, thời điểm dùng cuối cùng (đưa vào
  `stats()` và hiển thị trong tab Hệ thống của Gradio).

Module này **không import torch ở thời điểm import** (torch chỉ được cài ở T2.2),
nên unit test chạy được với loader giả và device manager giả.
"""

from __future__ import annotations

import abc
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml

from artrag.config import REPO_ROOT, Config
from artrag.errors import ModelLoadError

MODELS_YAML = REPO_ROOT / "configs" / "models.yaml"

#: Đăng ký loader theo task: `LOADERS[task](spec, device, dtype) -> (model, processor)`.
LOADERS: dict[str, Callable[["ModelSpec", str, str], tuple[Any, Any]]] = {}

#: Ngưỡng nhận biết lỗi hết VRAM (torch chưa cài nên không import được).
_OOM_MARKER = "OutOfMemoryError"


class _TryNextDevice(Exception):
    """Nội bộ: thiết bị hiện tại không dùng được (hết VRAM), thử thiết bị kế tiếp."""


def register_loader(
    task: str,
) -> Callable[
    [Callable[["ModelSpec", str, str], tuple[Any, Any]]],
    Callable[["ModelSpec", str, str], tuple[Any, Any]],
]:
    """Decorator đăng ký hàm nạp model cho một task.

    ```python
    @register_loader("clip")
    def load_clip(spec, device, dtype):
        ...
        return model, processor
    ```
    """

    def decorator(fn: Callable[["ModelSpec", str, str], tuple[Any, Any]]):
        LOADERS[task] = fn
        return fn

    return decorator


def is_oom(exc: BaseException) -> bool:
    """True nếu lỗi là OOM của CUDA (không phụ thuộc torch đã cài hay chưa)."""
    if type(exc).__name__ == _OOM_MARKER:
        return True
    # torch.cuda.OutOfMemoryError là lớp con của RuntimeError; tên lớp vẫn là
    # "OutOfMemoryError" nên nhánh trên đã đủ. Kiểm tra thêm chuỗi thông báo cho
    # trường hợp lỗi được bọc lại.
    message = str(exc).lower()
    return "out of memory" in message or "cuda oom" in message


# ---------------------------------------------------------------------------
# Spec / handle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """Mô tả tĩnh của một model trong `configs/models.yaml`."""

    name: str
    task: str
    hf_id: str
    dtype: str = "fp32"
    quant: str | None = None
    est_vram_mb: int = 0
    devices: tuple[str, ...] = ("cpu",)
    pinned: bool = False

    @classmethod
    def from_dict(cls, name: str, data: Mapping[str, Any]) -> "ModelSpec":
        devices = data.get("devices") or ["cpu"]
        if isinstance(devices, str):
            devices = [devices]
        return cls(
            name=name,
            task=str(data.get("task", "")),
            hf_id=str(data.get("hf_id", "")),
            dtype=str(data.get("dtype", "fp32")),
            quant=(str(data["quant"]) if data.get("quant") else None),
            est_vram_mb=int(data.get("est_vram_mb", 0) or 0),
            devices=tuple(str(d) for d in devices),
            pinned=bool(data.get("pinned", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "task": self.task,
            "hf_id": self.hf_id,
            "dtype": self.dtype,
            "quant": self.quant,
            "est_vram_mb": self.est_vram_mb,
            "devices": list(self.devices),
            "pinned": self.pinned,
        }


@dataclass
class ModelHandle:
    """Handle của một model đã nạp: `(model, processor, device)` + thông tin đo."""

    name: str
    model: Any
    processor: Any
    device: str
    dtype: str
    est_vram_mb: int = 0
    vram_measured_mb: float | None = None
    load_ms: int = 0
    last_used_at: float = 0.0
    last_used_seq: int = 0
    pinned: bool = False
    task: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "task": self.task,
            "device": self.device,
            "dtype": self.dtype,
            "est_vram_mb": self.est_vram_mb,
            "vram_measured_mb": self.vram_measured_mb,
            "load_ms": self.load_ms,
            "last_used_at": self.last_used_at,
            "pinned": self.pinned,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Device manager
# ---------------------------------------------------------------------------


class DeviceManager(abc.ABC):
    """Lớp trừu tượng cho quản lý thiết bị — để unit test dùng bản giả."""

    @abc.abstractmethod
    def available(self, device: str) -> bool:
        """Thiết bị có dùng được không."""

    @abc.abstractmethod
    def mem_allocated(self) -> float:
        """VRAM đang cấp phát (MB). CPU trả 0.0."""

    @abc.abstractmethod
    def empty_cache(self) -> None:
        """Giải phóng bộ nhớ cache của thiết bị."""

    @abc.abstractmethod
    def is_cuda(self, device: str) -> bool:
        """Thiết bị có phải GPU không."""

    # -- tiện ích dùng chung ---------------------------------------------
    def cuda_devices(self) -> list[str]:
        return ["cuda", "cuda:0"]


class TorchDeviceManager(DeviceManager):
    """Bản thật, dùng torch. Torch được import **lazy** để module vẫn import
    được khi chưa cài torch (T2.2 mới thêm dep)."""

    def __init__(self) -> None:
        self._torch: Any = None

    def _torch_mod(self) -> Any:
        if self._torch is None:
            try:
                import torch  # noqa: PLC0415 - import lazy có chủ đích
            except ImportError as exc:  # pragma: no cover - phụ thuộc môi trường
                raise ModelLoadError(
                    "Chưa cài torch; chạy `uv sync` hoặc `uv pip install torch`.",
                    hint="torch được thêm ở T2.2",
                ) from exc
            self._torch = torch
        return self._torch

    def available(self, device: str) -> bool:
        if self.is_cuda(device):
            torch = self._torch_mod()
            try:
                return bool(torch.cuda.is_available())
            except Exception:  # noqa: BLE001 - torch có thể lỗi driver
                return False
        return device == "cpu"

    def mem_allocated(self) -> float:
        torch = self._torch_mod()
        if not torch.cuda.is_available():
            return 0.0
        return float(torch.cuda.memory_allocated()) / (1024.0**2)

    def empty_cache(self) -> None:
        torch = self._torch_mod()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def is_cuda(self, device: str) -> bool:
        return str(device).startswith("cuda")


class FakeDeviceManager(DeviceManager):
    """Bản giả cho unit test: không cần torch/GPU thật.

    - `available_devices`: danh sách thiết bị "đang dùng được".
    - `mem_allocated_mb`: VRAM đang cấp phát (giả).
    - `on_load(name, device)`: hook gọi **trước** khi loader thật chạy, dùng
      để giả lập OOM (ném `RuntimeError("CUDA out of memory")`).
    """

    def __init__(
        self,
        available_devices: Sequence[str] = ("cuda", "cpu"),
        mem_allocated_mb: float = 0.0,
        on_load: Callable[[str, str], None] | None = None,
    ) -> None:
        self.available_devices = list(available_devices)
        self.mem_allocated_mb = float(mem_allocated_mb)
        self.on_load = on_load
        self.empty_cache_calls = 0
        self.load_events: list[tuple[str, str]] = []

    def available(self, device: str) -> bool:
        return device in self.available_devices

    def mem_allocated(self) -> float:
        return self.mem_allocated_mb

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1

    def is_cuda(self, device: str) -> bool:
        return str(device).startswith("cuda")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ModelRegistry:
    """Kho model trung tâm của hệ thống (xem SYSTEM_DESIGN 6.2)."""

    def __init__(
        self,
        specs: Mapping[str, ModelSpec],
        device_manager: DeviceManager | None = None,
        vram_budget_mb: int = 0,
        evict_to: str = "cpu",
        loaders: Mapping[str, Callable[[ModelSpec, str, str], tuple[Any, Any]]] | None = None,
    ) -> None:
        self.specs: dict[str, ModelSpec] = dict(specs)
        self.device_manager = device_manager or TorchDeviceManager()
        self.vram_budget_mb = int(vram_budget_mb or 0)
        self.evict_to = evict_to or "cpu"
        self._loaders: dict[str, Callable[[ModelSpec, str, str], tuple[Any, Any]]] = (
            dict(loaders) if loaders is not None else LOADERS
        )

        self._handles: dict[str, ModelHandle] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._registry_lock = threading.Lock()
        self.warnings: list[str] = []
        self._counter = 0

    # -- tiện ích nội bộ -------------------------------------------------
    def _lock_for(self, name: str) -> threading.Lock:
        with self._registry_lock:
            lock = self._locks.get(name)
            if lock is None:
                lock = threading.Lock()
                self._locks[name] = lock
            return lock

    def _next_counter(self) -> int:
        with self._registry_lock:
            self._counter += 1
            return self._counter

    def _touch(self, handle: ModelHandle) -> None:
        """Đánh dấu vừa dùng (cho LRU) và cập nhật thời điểm dùng cuối."""
        handle.last_used_seq = self._next_counter()
        handle.last_used_at = time.time()

    def _spec(self, name: str) -> ModelSpec:
        try:
            return self.specs[name]
        except KeyError as exc:
            raise ModelLoadError(
                f"Không có model nào tên {name!r} trong registry.",
                hint="kiểm tra configs/models.yaml",
            ) from exc

    def _loader_for(self, spec: ModelSpec) -> Callable[[ModelSpec, str, str], tuple[Any, Any]]:
        loader = self._loaders.get(spec.task)
        if loader is None:
            raise ModelLoadError(
                f"Chưa có loader cho task {spec.task!r} (model {spec.name!r}).",
                hint="đăng ký bằng @register_loader(...) trong src/artrag/models/loaders.py",
            )
        return loader

    @staticmethod
    def _dtype_for(spec: ModelSpec, device: str) -> str:
        """Trên CPU luôn dùng fp32 (§6.2 hành vi 3)."""
        if str(device).startswith("cpu"):
            return "fp32"
        return spec.dtype or "fp16"

    def _is_gpu(self, device: str) -> bool:
        return self.device_manager.is_cuda(device)

    def _gpu_used_mb(self) -> int:
        return sum(
            handle.est_vram_mb
            for handle in self._handles.values()
            if self._is_gpu(handle.device)
        )

    def _lru_gpu_candidates(self) -> list[ModelHandle]:
        """Các model GPU có thể giải phóng, theo thứ tự LRU (cũ nhất trước)."""
        candidates = [
            handle
            for handle in self._handles.values()
            if self._is_gpu(handle.device) and not handle.pinned
        ]
        candidates.sort(key=lambda h: (h.last_used_seq, h.name))
        return candidates

    # -- nạp / giải phóng -------------------------------------------------
    def get(self, name: str) -> ModelHandle:
        """Lấy handle của model; nạp lazy ở lần gọi đầu tiên."""
        spec = self._spec(name)

        # fast path: đã nạp -> chỉ cập nhật thời điểm dùng
        handle = self._handles.get(name)
        if handle is not None:
            self._touch(handle)
            return handle

        lock = self._lock_for(name)
        with lock:
            # double-checked locking: thread khác có thể đã nạp xong
            handle = self._handles.get(name)
            if handle is not None:
                self._touch(handle)
                return handle
            handle = self._load(spec)
            self._handles[name] = handle
            return handle

    def _load(self, spec: ModelSpec) -> ModelHandle:
        loader = self._loader_for(spec)
        candidates = [d for d in spec.devices if self.device_manager.available(d)]
        if not candidates:
            raise ModelLoadError(
                f"Không có thiết bị nào khả dụng cho model {spec.name!r} "
                f"(devices={list(spec.devices)}).",
                hint="kiểm tra CUDA hoặc sửa `devices` trong configs/models.yaml",
            )

        last_error: BaseException | None = None
        for index, device in enumerate(candidates):
            try:
                return self._load_on(spec, loader, device)
            except _TryNextDevice:
                # Hết chỗ VRAM theo ngân sách -> thử thiết bị kế tiếp (CPU).
                continue
            except Exception as exc:  # noqa: BLE001 - phân nhánh theo loại lỗi
                if not is_oom(exc):
                    raise ModelLoadError(
                        f"Không nạp được model {spec.name!r} trên {device}: {exc}"
                    ) from exc
                last_error = exc
                # OOM: giải phóng mọi model không pinned rồi thử lại 1 lần trên
                # cùng thiết bị; sau đó mới chuyển sang thiết bị kế tiếp (CPU).
                # Ở nhánh OOM ta **xoá hẳn** model (không chuyển sang CPU) vì cần
                # nhả VRAM thật sự trước khi thử lại.
                self._evict_all_unpinned(move_to_cpu=False)
                self.device_manager.empty_cache()
                try:
                    return self._load_on(spec, loader, device)
                except _TryNextDevice:
                    continue
                except Exception as retry_exc:  # noqa: BLE001
                    if not is_oom(retry_exc):
                        raise ModelLoadError(
                            f"Không nạp được model {spec.name!r} trên {device}: {retry_exc}"
                        ) from retry_exc
                    last_error = retry_exc
                    self.warnings.append(
                        f"OOM khi nạp {spec.name!r} trên {device}; chuyển sang "
                        f"{candidates[index + 1] if index + 1 < len(candidates) else 'không có'}"
                    )
                    continue
        raise ModelLoadError(
            f"Không nạp được model {spec.name!r}: hết VRAM trên mọi thiết bị "
            f"({', '.join(candidates)}).",
            hint="giảm `vram_budget_mb` hoặc dùng model nhỏ hơn (vd qwen1_5b)",
        ) from last_error

    def _load_on(
        self,
        spec: ModelSpec,
        loader: Callable[[ModelSpec, str, str], tuple[Any, Any]],
        device: str,
    ) -> ModelHandle:
        dtype = self._dtype_for(spec, device)

        # Chừa chỗ VRAM theo ngân sách trước khi nạp.
        if self.vram_budget_mb > 0 and self._is_gpu(device):
            while self._gpu_used_mb() + spec.est_vram_mb > self.vram_budget_mb:
                candidates = self._lru_gpu_candidates()
                if not candidates:
                    # Không còn model nào để giải phóng -> thử thiết bị kế tiếp (CPU).
                    raise _TryNextDevice(
                        f"Không đủ VRAM cho {spec.name!r} "
                        f"(cần {spec.est_vram_mb} MB, ngân sách {self.vram_budget_mb} MB)."
                    )
                self._evict(candidates[0])

        before = self.device_manager.mem_allocated()
        started = time.perf_counter()
        model, processor = loader(spec, device, dtype)
        load_ms = int((time.perf_counter() - started) * 1000)
        after = self.device_manager.mem_allocated()

        handle = ModelHandle(
            name=spec.name,
            model=model,
            processor=processor,
            device=device,
            dtype=dtype,
            est_vram_mb=spec.est_vram_mb,
            vram_measured_mb=max(0.0, after - before),
            load_ms=load_ms,
            last_used_at=time.time(),
            last_used_seq=0,
            pinned=spec.pinned,
            task=spec.task,
        )
        self._touch(handle)
        return handle

    def _evict_all_unpinned(self, move_to_cpu: bool = True) -> list[str]:
        evicted: list[str] = []
        for handle in list(self._handles.values()):
            if handle.pinned:
                continue
            if self._evict(handle, move_to_cpu=move_to_cpu):
                evicted.append(handle.name)
        return evicted

    def _evict(self, handle: ModelHandle, move_to_cpu: bool = True) -> bool:
        """Giải phóng một model khỏi registry.

        - `move_to_cpu=False`: xoá hẳn (dùng cho `unload` và khi OOM cần nhả VRAM
          thật sự).
        - `move_to_cpu=True` và `evict_to == "cpu"`: giải phóng GPU rồi nạp lại trên
          CPU (nếu CPU khả dụng) — model vẫn dùng được, chỉ chậm hơn.
        """
        self._handles.pop(handle.name, None)
        model, processor = handle.model, handle.processor
        handle.model = None
        handle.processor = None
        del model, processor
        self.device_manager.empty_cache()

        if move_to_cpu and self.evict_to == "cpu":
            spec = self.specs.get(handle.name)
            if spec is not None and self.device_manager.available("cpu"):
                try:
                    loader = self._loader_for(spec)
                    cpu_dtype = self._dtype_for(spec, "cpu")
                    cpu_model, cpu_processor = loader(spec, "cpu", cpu_dtype)
                    moved = ModelHandle(
                        name=spec.name,
                        model=cpu_model,
                        processor=cpu_processor,
                        device="cpu",
                        dtype=cpu_dtype,
                        est_vram_mb=spec.est_vram_mb,
                        vram_measured_mb=0.0,
                        load_ms=handle.load_ms,
                        last_used_at=time.time(),
                        last_used_seq=0,
                        pinned=spec.pinned,
                        task=spec.task,
                        notes=[*handle.notes, "chuyển sang cpu (evict)"],
                    )
                    self._touch(moved)
                    self._handles[handle.name] = moved
                except Exception as exc:  # noqa: BLE001 - giải phóng không được phép làm hỏng luồng
                    self.warnings.append(
                        f"Không chuyển {handle.name!r} sang CPU được: {exc}"
                    )
        return True

    def unload(self, name: str) -> bool:
        """Giải phóng hẳn một model đang nạp. Trả True nếu có thật sự nạp."""
        handle = self._handles.get(name)
        if handle is None:
            return False
        self._evict(handle, move_to_cpu=False)
        return True

    def preload(self, names: Iterable[str]) -> list[ModelHandle]:
        """Nạp sẵn một danh sách model."""
        return [self.get(name) for name in names]

    # -- thống kê ----------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        """Thông tin các model đang nạp (đưa vào tab Hệ thống của Gradio)."""
        models = {name: handle.to_dict() for name, handle in self._handles.items()}
        return {
            "budget_mb": self.vram_budget_mb,
            "loaded": len(self._handles),
            "vram_est_mb": self._gpu_used_mb(),
            "vram_measured_mb": float(self.device_manager.mem_allocated()),
            "devices": sorted({handle.device for handle in self._handles.values()}),
            "warnings": list(self.warnings),
            "models": models,
        }

    def __contains__(self, name: object) -> bool:
        return name in self._handles

    def __repr__(self) -> str:
        return (
            f"ModelRegistry(specs={len(self.specs)}, loaded={len(self._handles)}, "
            f"budget={self.vram_budget_mb}MB)"
        )


# ---------------------------------------------------------------------------
# Nạp spec từ config
# ---------------------------------------------------------------------------


def _read_models_yaml(path: Path | None = None) -> dict[str, dict[str, Any]]:
    path = path or MODELS_YAML
    if not path.exists():
        raise ModelLoadError(
            f"Không tìm thấy file danh mục model: {path}",
            hint="tạo configs/models.yaml theo SYSTEM_DESIGN 6.2",
        )
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ModelLoadError(f"File {path} phải là mapping.")
    return {str(k): dict(v or {}) for k, v in data.items()}


def load_model_specs(cfg: Config | Mapping[str, Any] | None = None) -> dict[str, ModelSpec]:
    """Nạp `configs/models.yaml` rồi ghi đè bằng khoá `models:` trong config.

    Thứ tự ưu tiên: `configs/models.yaml` < `cfg["models"]` (profile/experiment/--set).
    """
    raw = _read_models_yaml()
    overrides: Mapping[str, Any] = {}
    if cfg is not None:
        if isinstance(cfg, Config):
            node = cfg.raw.get("models", {}) or {}
            overrides = node if isinstance(node, Mapping) else {}
        elif isinstance(cfg, Mapping):
            node = cfg.get("models", {}) or {}
            overrides = dict(node) if isinstance(node, Mapping) else {}

    specs: dict[str, ModelSpec] = {}
    for name, data in raw.items():
        merged = dict(data)
        override = overrides.get(name)
        if isinstance(override, Mapping):
            merged.update({k: v for k, v in override.items() if v is not None})
        specs[name] = ModelSpec.from_dict(name, merged)
    return specs

def create_registry(
    cfg: Config | Mapping[str, Any] | None = None,
    device_manager: DeviceManager | None = None,
    loaders: Mapping[str, Callable[[ModelSpec, str, str], tuple[Any, Any]]] | None = None,
) -> ModelRegistry:
    """Tạo registry từ config (dùng `vram_budget_mb` của profile)."""
    specs = load_model_specs(cfg)
    budget = 0
    evict_to = "cpu"
    if cfg is not None:
        if isinstance(cfg, Config):
            budget = int(cfg.get("vram_budget_mb", 0) or 0)
            evict_to = str(cfg.get("models.evict_to", "cpu") or "cpu")
        elif isinstance(cfg, Mapping):
            budget = int(cfg.get("vram_budget_mb", 0) or 0)
            evict_to = str(cfg.get("models", {}).get("evict_to", "cpu") or "cpu")
    return ModelRegistry(
        specs=specs,
        device_manager=device_manager,
        vram_budget_mb=budget,
        evict_to=evict_to,
        loaders=loaders,
    )
