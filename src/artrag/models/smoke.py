"""Smoke test cho các loader thật (T2.2).

Chạy: `uv run python -m artrag.registry --smoke clip blip_itm text_embed`

Với mỗi model, in: device, dtype, VRAM ước tính/thực đo, thời gian nạp và
**một lần suy luận thử**. Model được **giải phóng trước** model kế tiếp vì
máy local chỉ còn ~1,5 GB RAM trống.

Exit code = 0 nếu tất cả model nạp và suy luận được, ngược lại 1.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from artrag.config import Config, load_config, paths
from artrag.errors import ArtragError
from artrag.registry import ModelHandle, ModelRegistry, create_registry

#: Ảnh dùng cho smoke test (tác phẩm Eakins, có sẵn trong kho ảnh).
DEFAULT_IMAGE_NAME = "cma-151904.jpg"

#: Câu hỏi mặc định cho CLIP/BLIP.
DEFAULT_QUERY = "a painting"


def _resolve_image(cfg: Any, image_arg: str | None) -> Any:
    """Trả về ảnh PIL để dùng cho inference (có ảnh thật, không thì tạo ảnh giả)."""
    from PIL import Image  # noqa: PLC0415

    candidate: Path | None = None
    if image_arg:
        candidate = Path(image_arg)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
    elif cfg is not None:
        candidate = paths(cfg).data_dir / "images" / DEFAULT_IMAGE_NAME

    if candidate is not None and candidate.exists():
        return Image.open(candidate).convert("RGB")

    # Không có ảnh -> tạo ảnh tổng hợp để smoke test vẫn chạy trên clone mới.
    return Image.new("RGB", (512, 512), color=(120, 90, 70))


def _run_inference(handle: ModelHandle, image: Any, query: str) -> str:
    """Chạy một lần suy luận đúng theo task của model, trả về mô tả kết quả."""
    from artrag.models import inference  # noqa: PLC0415

    task = handle.task
    if task == "clip":
        vec_img = inference.embed_image(handle, image)
        vec_txt = inference.embed_text(handle, query)
        cos = float(vec_img @ vec_txt)
        return f"dim={vec_img.shape[0]} cos(img, '{query}')={cos:.4f}"
    if task == "sbert":
        vec = inference.embed_text(handle, query)
        return f"dim={vec.shape[0]} text='{query}'"
    if task == "blip_itm":
        score = inference.itm_score(handle, image, query)
        return f"itm_score={score:.4f} text='{query}'"
    if task == "blip_cap":
        return f"caption='{inference.caption(handle, image)}'"
    if task == "blip_vqa":
        answer = inference.vqa(handle, image, "how many people are in the picture?")
        return f"answer='{answer}'"
    if task == "seq2seq":
        source = "Xin chào" if "vi2en" in handle.name else "Hello"
        return f"translate='{inference.translate(handle, source)}' (from '{source}')"
    if task == "nli":
        label = inference.nli_label(
            handle,
            "A painting of two men rowing a boat.",
            "This artwork shows two men rowing a boat.",
        )
        return f"nli_label={label}"
    if task == "causal_lm":
        # Sinh thử một câu ngắn để kiểm tra model causal LM hoạt động.
        source = "Xin chào" if "vi2en" in handle.name else "What is the capital of Vietnam?"
        return f"generate='{inference.generate_text(handle, source)}' (from '{source}')"
    raise ArtragError(f"Không có logic suy luận cho task {task!r}")


def smoke_one(
    registry: ModelRegistry,
    name: str,
    image: Any,
    query: str,
    keep: bool = False,
) -> bool:
    """Nạp + suy luận thử một model. Trả True nếu thành công."""
    handle = registry.get(name)
    print(
        f"{name} | task={handle.task} | device={handle.device} | dtype={handle.dtype} "
        f"| vram_est_mb={handle.est_vram_mb} "
        f"| vram_measured_mb={handle.vram_measured_mb or 0:.1f} "
        f"| load_ms={handle.load_ms}"
    )
    started = time.perf_counter()
    result = _run_inference(handle, image, query)
    elapsed = int((time.perf_counter() - started) * 1000)
    print(f"  -> {result} | infer_ms={elapsed}")

    if not keep:
        registry.unload(name)
        print("  -> unloaded")
    return True


def run_smoke(
    names: list[str],
    cfg: Config | None = None,
    image_arg: str | None = None,
    query: str = DEFAULT_QUERY,
    keep: bool = False,
) -> int:
    """Chạy smoke test cho danh sách model. Trả exit code (0 = tất cả thành công)."""
    # Import loader để đăng ký vào LOADERS (phải làm trước khi tạo registry).
    from artrag.models import loaders  # noqa: PLC0415,F401

    if cfg is None:
        cfg = load_config()
    registry = create_registry(cfg)
    image = _resolve_image(cfg, image_arg)

    failures: list[str] = []
    for name in names:
        try:
            smoke_one(registry, name, image, query, keep=keep)
        except Exception as exc:  # noqa: BLE001 - lỗi của một model không chặn các model sau
            print(f"{name} | ERROR {type(exc).__name__}: {exc}")
            failures.append(name)
            try:
                registry.unload(name)
            except Exception:  # noqa: BLE001 - dọn dẹp không được phép làm hỏng luồng
                pass

    print("--- registry.stats() ---")
    stats = registry.stats()
    print(
        f"loaded={stats['loaded']} vram_est_mb={stats['vram_est_mb']} "
        f"budget_mb={stats['budget_mb']}"
    )
    for model_name, info in stats["models"].items():
        print(f"  {model_name}: {info['device']} {info['dtype']} est={info['est_vram_mb']}MB")
    for warning in stats["warnings"]:
        print(f"  WARNING: {warning}")

    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("OK: tat ca model nap va suy luan duoc.")
    return 0
