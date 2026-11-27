"""Loader cho các model thị giác và embedding (T2.2, SYSTEM_DESIGN 6.2).

Mỗi loader đăng ký qua `@register_loader(task)` với chữ ký:

    loader(spec: ModelSpec, device: str, dtype: str) -> (model, processor)

`spec.hf_id` cho biết checkpoint cụ thể, nên cùng một task có thể phục vụ nhiều
model khác nhau (ví dụ `seq2seq` dùng cho cả vinai dịch lẫn flan-t5).

Module này **import torch ở thời điểm import** — khác với `artrag.registry`
(vốn phải import được khi chưa cài torch). Vì vậy chỉ import module này khi
thật sự cần nạp model.
"""

from __future__ import annotations

from typing import Any

from artrag.errors import ModelLoadError
from artrag.registry import ModelSpec, register_loader

# ---------------------------------------------------------------------------
# Tiện ích dùng chung
# ---------------------------------------------------------------------------


def torch_dtype(name: str) -> Any:
    """Chuyển chuỗi dtype của registry ('fp16'/'fp32') sang `torch.dtype`."""
    import torch  # noqa: PLC0415 - import cục bộ có chủ đích

    key = (name or "fp32").lower()
    if key in {"fp16", "float16", "half"}:
        return torch.float16
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp32", "float32", "float"}:
        return torch.float32
    raise ModelLoadError(f"Dtype không hợp lệ: {name!r}")


def resolve_device(device: str) -> Any:
    """Chuẩn hoá tên thiết bị ('cuda' -> 'cuda:0')."""
    import torch  # noqa: PLC0415 - import cục bộ có chủ đích

    name = (device or "cpu").strip().lower()
    if name.startswith("cuda") and ":" not in name:
        return torch.device("cuda:0")
    return torch.device(name)


def device_str(device: str) -> str:
    """Tên thiết bị dạng chuỗi mà `sentence_transformers` mong đợi.

    `SentenceTransformer(..., device=...)` nhận chuỗi ('cuda'/'cpu'), không nhận
    `torch.device`.
    """
    name = (device or "cpu").strip().lower()
    if name.startswith("cuda"):
        return "cuda"
    return "cpu"


def _check_quant(spec: ModelSpec) -> None:
    """Lượng tử hoá chỉ dành cho causal_lm (T2.4), chưa hỗ trợ ở đây."""
    if spec.quant:
        raise ModelLoadError(
            f"Model {spec.name!r} yêu cầu quant={spec.quant!r} nhưng loader này "
            "chưa hỗ trợ lượng tử hoá.",
            hint="lượng tử hoá 4bit chỉ dùng cho causal_lm (T2.4)",
        )


def _load_hf(
    spec: ModelSpec,
    device: str,
    dtype: str,
    cls: Any,
    *,
    processor_cls: Any = None,
    processor_kwargs: dict[str, Any] | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> tuple[Any, Any]:
    """Nạp model (và processor) từ HuggingFace rồi chuyển sang thiết bị.

    `device_map` không dùng: ta tự `.to(device)` để `DeviceManager` của
    registry tiếp tục là nguồn sự thật duy nhất về thiết bị.
    """
    import torch  # noqa: PLC0415 - import cục bộ có chủ đích

    _check_quant(spec)
    torch_dtype_value = torch_dtype(dtype)
    target = resolve_device(device)

    kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype_value,
        "low_cpu_mem_usage": True,
    }
    kwargs.update(model_kwargs or {})

    try:
        model = cls.from_pretrained(spec.hf_id, **kwargs)
    except Exception as exc:  # noqa: BLE001 - bọc lại thành lỗi riêng của hệ thống
        raise ModelLoadError(
            f"Không nạp được {spec.hf_id!r} ({spec.task}): {exc}"
        ) from exc

    model = model.to(target)
    model.eval()

    processor = None
    if processor_cls is not None:
        try:
            processor = processor_cls.from_pretrained(
                spec.hf_id, **(processor_kwargs or {})
            )
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(
                f"Không nạp được processor cho {spec.hf_id!r} ({spec.task}): {exc}"
            ) from exc

    del torch
    return model, processor


# ---------------------------------------------------------------------------
# Loader theo task
# ---------------------------------------------------------------------------


@register_loader("clip")
def load_clip(spec: ModelSpec, device: str, dtype: str) -> tuple[Any, Any]:
    """CLIP ViT-B/32: encode ảnh và text về cùng không gian vector (thiết kế §5.3)."""
    from transformers import CLIPModel, CLIPProcessor  # noqa: PLC0415

    return _load_hf(
        spec,
        device,
        dtype,
        CLIPModel,
        processor_cls=CLIPProcessor,
        processor_kwargs={"use_fast": True},
    )


@register_loader("sbert")
def load_sbert(spec: ModelSpec, device: str, dtype: str) -> tuple[Any, Any]:
    """Sentence encoder đa ngôn ngữ (BGE-M3): embed văn bản, kể cả tiếng Việt.

    `sentence_transformers.SentenceTransformer` nhận `device` dạng **chuỗi**.
    """
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    _check_quant(spec)
    try:
        model = SentenceTransformer(
            spec.hf_id,
            device=device_str(device),
            model_kwargs={"torch_dtype": torch_dtype(dtype)},
        )
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(
            f"Không nạp được {spec.hf_id!r} ({spec.task}): {exc}"
        ) from exc
    model.eval()
    # SentenceTransformer không có processor riêng -> None.
    return model, None


@register_loader("blip_itm")
def load_blip_itm(spec: ModelSpec, device: str, dtype: str) -> tuple[Any, Any]:
    """BLIP-ITM: rerank ứng viên theo mức khớp nội dung ảnh (thiết kế §5.4).

    `use_itm_head=True` được truyền lúc **gọi model** (xem
    `artrag.models.inference.itm_score`), không phải lúc nạp processor.
    """
    from transformers import BlipForImageTextRetrieval, BlipProcessor  # noqa: PLC0415

    return _load_hf(
        spec,
        device,
        dtype,
        BlipForImageTextRetrieval,
        processor_cls=BlipProcessor,
    )


@register_loader("blip_cap")
def load_blip_cap(spec: ModelSpec, device: str, dtype: str) -> tuple[Any, Any]:
    """BLIP captioning: sinh mô tả tự động cho ảnh (thiết kế §5.6)."""
    from transformers import BlipForConditionalGeneration, BlipProcessor  # noqa: PLC0415

    return _load_hf(
        spec,
        device,
        dtype,
        BlipForConditionalGeneration,
        processor_cls=BlipProcessor,
    )


@register_loader("blip_vqa")
def load_blip_vqa(spec: ModelSpec, device: str, dtype: str) -> tuple[Any, Any]:
    """BLIP VQA: trả lời câu hỏi có mục tiêu về ảnh (thiết kế §5.6)."""
    from transformers import BlipForQuestionAnswering, BlipProcessor  # noqa: PLC0415

    return _load_hf(
        spec,
        device,
        dtype,
        BlipForQuestionAnswering,
        processor_cls=BlipProcessor,
    )


@register_loader("seq2seq")
def load_seq2seq(spec: ModelSpec, device: str, dtype: str) -> tuple[Any, Any]:
    """Model seq2seq: dịch (vinai vi2en/en2vi) hoặc sinh câu trả lời (flan-t5).

    Các checkpoint vinai là MBart thuần, không cần `trust_remote_code`.

    Với model dịch kiểu MBart, **bắt buộc** set `src_lang`/`tgt_lang` cho
    tokenizer, nếu không model sẽ sinh ra chuỗi vô nghĩa (vd. "Good Good
    Good..."). Giá trị lấy từ `configs/models.yaml` (`src_lang`, `tgt_lang`).
    """
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: PLC0415

    model, tokenizer = _load_hf(
        spec,
        device,
        dtype,
        AutoModelForSeq2SeqLM,
        processor_cls=AutoTokenizer,
        model_kwargs={"trust_remote_code": False},
    )
    if tokenizer is not None and spec.src_lang:
        try:
            tokenizer.src_lang = spec.src_lang
        except (AttributeError, ValueError):
            pass  # tokenizer không hỗ trợ ngôn ngữ (ví dụ flan-t5) -> bỏ qua
    if tokenizer is not None and spec.tgt_lang:
        # Token ngôn ngữ đích phải được đặt làm token bắt đầu decoder, nếu không
        # MBart sinh ra chuỗi vô nghĩa lặp đi lặp lại ("Xin Xin Xin ...").
        # Với checkpoint vinai, `decoder_start_token_id` là cách đúng
        # (`forced_bos_token_id` cho kết quả lặp với en2vi).
        try:
            tgt_id = tokenizer.convert_tokens_to_ids(spec.tgt_lang)
        except (AttributeError, ValueError, KeyError):
            tgt_id = None
        if tgt_id is not None and tgt_id != tokenizer.unk_token_id:
            try:
                model.generation_config.decoder_start_token_id = tgt_id
                model.generation_config.forced_bos_token_id = tgt_id
            except AttributeError:  # pragma: no cover - model không có generation_config
                pass
    return model, tokenizer


@register_loader("nli")
def load_nli(spec: ModelSpec, device: str, dtype: str) -> tuple[Any, Any]:
    """Model NLI đa ngôn ngữ: kiểm tra mâu thuẫn giữa các đoạn bằng chứng (§10.2)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: PLC0415

    return _load_hf(
        spec,
        device,
        dtype,
        AutoModelForSequenceClassification,
        processor_cls=AutoTokenizer,
        model_kwargs={"trust_remote_code": False},
    )


# ---------------------------------------------------------------------------
# LLM chạy local (causal LM)
# ---------------------------------------------------------------------------


def _bitsandbytes_config(dtype: str) -> Any:
    """Tạo `BitsAndBytesConfig` cho lượng tử hoá 4-bit (NF4).

    Bắt lỗi import của `bitsandbytes` ngay tại đây để thông báo lỗi có ý nghĩa
    (bitsandbytes thường lỗi trên Windows với GPU không hỗ trợ đầy đủ).
    """
    try:
        from transformers import BitsAndBytesConfig  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - transformers luôn có sẵn
        raise ModelLoadError(
            "transformers không có BitsAndBytesConfig.",
            hint="cài lại transformers>=4.46,<4.47",
        ) from exc

    compute_dtype = torch_dtype(dtype)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )


def _load_causal_lm_4bit(spec: ModelSpec, device: str, dtype: str) -> tuple[Any, Any]:
    """Nạp model causal LM ở chế độ 4-bit (bitsandbytes).

    Với 4-bit, bitsandbytes tự quản lý vị trí tham số nên ta **không** dùng
    `device_map="auto"` (registry là nguồn sự thật duy nhất về thiết bị) mà
    dùng `device_map={"": device}` để đặt toàn bộ model lên đúng thiết bị.

    `low_cpu_mem_usage` bị bỏ qua vì bitsandbytes yêu cầu `accelerate` với
    cấu hình khác.
    """
    import torch  # noqa: PLC0415 - import cục bộ có chủ đích
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    if not str(device).startswith("cuda"):
        raise ModelLoadError(
            f"Model {spec.name!r} dùng lượng tử hoá 4-bit chỉ chạy được trên GPU, "
            f"nhưng registry chọn thiết bị {device!r}.",
            hint=(
                "đặt `devices: [cuda]` cho model này trong configs/profiles/local.yaml, "
                "hoặc dùng model fp16 (qwen1_5b) cho thiết bị CPU."
            ),
        )

    quant_config = _bitsandbytes_config(dtype)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            spec.hf_id,
            quantization_config=quant_config,
            device_map={"": device},
            low_cpu_mem_usage=True,
        )
    except ImportError as exc:
        raise ModelLoadError(
            f"Không nạp được {spec.hf_id!r} ở chế độ 4-bit: bitsandbytes chưa được "
            f"cài đặt ({exc}).",
            hint=(
                "cài bitsandbytes (`uv sync`) hoặc dùng model fp16 "
                "`qwen1_5b` (đặt `llm.model: qwen1_5b` trong config)."
            ),
        ) from exc
    except Exception as exc:  # noqa: BLE001 - bọc lại thành lỗi riêng của hệ thống
        raise ModelLoadError(
            f"Không nạp được {spec.hf_id!r} ở chế độ 4-bit: {exc}",
            hint=(
                "bitsandbytes có thể không hỗ trợ GPU/driver này; dùng model fp16 "
                "`qwen1_5b` thay thế (đặt `llm.model: qwen1_5b` trong config)."
            ),
        ) from exc

    model.eval()

    try:
        tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(
            f"Không nạp được tokenizer cho {spec.hf_id!r}: {exc}"
        ) from exc

    # Chat template + generate cần pad bên trái.
    if tokenizer is not None:
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

    del torch
    return model, tokenizer


@register_loader("causal_lm")
def load_causal_lm(spec: ModelSpec, device: str, dtype: str) -> tuple[Any, Any]:
    """Model sinh văn bản (causal LM) chạy local: Qwen2.5 cho backend `hf_local`.

    - `quant: 4bit` -> dùng bitsandbytes (NF4), model phải nằm trên GPU.
    - còn lại -> nạp thường theo `_load_hf` (fp16 trên cuda, fp32 trên CPU).
    """
    if spec.quant:
        if str(spec.quant).lower() not in {"4bit", "4-bit", "nf4"}:
            raise ModelLoadError(
                f"Model {spec.name!r} yêu cầu quant={spec.quant!r} không hợp lệ.",
                hint="chỉ hỗ trợ `4bit` cho causal_lm.",
            )
        return _load_causal_lm_4bit(spec, device, dtype)

    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    return _load_hf(
        spec,
        device,
        dtype,
        AutoModelForCausalLM,
        processor_cls=AutoTokenizer,
        model_kwargs={"trust_remote_code": False},
    )


#: Các task đã có loader trong module này.
REGISTERED_TASKS = (
    "clip",
    "sbert",
    "blip_itm",
    "blip_cap",
    "blip_vqa",
    "seq2seq",
    "nli",
    "causal_lm",
)
