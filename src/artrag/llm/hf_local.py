"""Backend LLM chạy local trên máy (T2.4, SYSTEM_DESIGN 6.3).

Backend `hf_local` sinh câu trả lời bằng model **causal LM** lấy từ
`ModelRegistry` (Qwen2.5-3B 4-bit qua bitsandbytes, hoặc Qwen2.5-1.5B fp16
nếu bitsandbytes lỗi trên Windows).

Điểm khác biệt so với `openai_compat`:

- Model được nạp **lazy** qua registry, nên `get_backend()` rất nhanh và không
  tốn VRAM nếu không hỏi gì.
- **Không retry**: lỗi cục bộ (OOM, thiếu bitsandbytes) sẽ không hết khi thử
  lại, và mỗi lần thử lại tốn hàng chục giây nạp model. Vì vậy backend này
  chỉ được bọc `CachedBackend`, không bọc `RetryBackend`.
- Số token đếm **chính xác** bằng tokenizer của model (không dùng heuristic
  ~4 ký tự/token như backend API).
"""

from __future__ import annotations

import time
from typing import Any

from artrag.errors import BackendError, ModelLoadError
from artrag.llm.base import LLMBackend, LLMResult, estimate_tokens
from artrag.registry import ModelHandle, ModelRegistry, create_registry

#: Tham số sinh mặc định khi không có gì trong config.
DEFAULT_GEN: dict[str, Any] = {
    "max_new_tokens": 256,
    "temperature": 0.0,
}

#: Fallback khi model không có chat template.
_PLAIN_PROMPT_TEMPLATE = "{system}\n{user}\nassistant:"


def _as_dict(value: Any) -> dict[str, Any] | None:
    """Chuyển `Config`/dict về dict thuần; None nếu không phải mapping."""
    from artrag.config import Config  # noqa: PLC0415 - tránh vòng import

    if isinstance(value, Config):
        return value.raw
    if isinstance(value, dict):
        return value
    return None


class HFLocalBackend:
    """Backend sinh câu trả lời bằng model causal LM chạy local."""

    def __init__(
        self,
        name: str = "hf_local",
        registry: ModelRegistry | None = None,
        model: str = "qwen3b",
        default_gen: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.registry = registry
        #: Tên model trong registry (khoá của `configs/models.yaml`).
        self.model = model
        self.default_gen = dict(default_gen or DEFAULT_GEN)
        self._handle: ModelHandle | None = None

    # -- model -----------------------------------------------------------
    def set_model(self, name: str) -> None:
        """Đổi model sẽ dùng (tên trong `configs/models.yaml`)."""
        self.model = str(name)

    def _get_handle(self) -> ModelHandle:
        """Lấy handle của model, nạp lazy nếu chưa có."""
        if self.registry is None:
            raise BackendError(
                f"Backend {self.name!r} chưa có registry nên không nạp được model.",
                hint="dựng backend qua `get_backend(cfg)` hoặc truyền registry.",
            )
        try:
            handle = self.registry.get(self.model)
        except ModelLoadError as exc:
            raise BackendError(
                f"Không nạp được model {self.model!r} cho backend {self.name!r}: {exc}",
                hint=(
                    "kiểm tra tên model trong `llm.model` và `configs/models.yaml`; "
                    "nếu lỗi bitsandbytes thì dùng `qwen1_5b` (fp16)."
                ),
            ) from exc
        self._handle = handle
        return handle

    # -- interface LLMBackend -------------------------------------------
    def count_tokens(self, text: str) -> int:
        """Đếm token chính xác nếu model đã nạp, ngược lại dùng heuristic."""
        handle = self._handle
        tokenizer = getattr(handle, "processor", None) if handle else None
        if tokenizer is None:
            return estimate_tokens(text)
        try:
            return len(tokenizer.encode(text))
        except Exception:  # noqa: BLE001 - tokenizer lỗi thì lùi về heuristic
            return estimate_tokens(text)

    def generate(self, messages: list[dict], **gen: Any) -> LLMResult:
        """Sinh câu trả lời từ model local."""
        handle = self._get_handle()
        model = handle.model
        tokenizer = handle.processor
        if model is None or tokenizer is None:
            raise BackendError(
                f"Model {self.model!r} không có model/tokenizer để sinh câu trả lời.",
                hint="kiểm tra loader `causal_lm` trong src/artrag/models/loaders.py.",
            )

        params = dict(self.default_gen)
        params.update({k: v for k, v in gen.items() if v is not None})
        max_new_tokens = int(params.get("max_new_tokens", 256))
        temperature = float(params.get("temperature", 0.0))

        prompt = self._build_prompt(tokenizer, messages)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(handle.device) for k, v in inputs.items()}
        prompt_tokens = int(inputs["input_ids"].shape[-1])

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self._pad_token_id(tokenizer),
        }
        if temperature > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = temperature
            if params.get("top_p") is not None:
                generation_kwargs["top_p"] = float(params["top_p"])
        else:
            # Greedy: không được để lại `temperature`/`top_p`/`top_k` trong
            # `generation_config` của model, nếu không transformers cảnh báo
            # "do_sample is set to False. However, temperature is set to ...".
            generation_kwargs["do_sample"] = False
            for key in ("temperature", "top_p", "top_k"):
                generation_kwargs[key] = None

        started = time.perf_counter()
        outputs = model.generate(**inputs, **generation_kwargs)
        latency_ms = int((time.perf_counter() - started) * 1000)

        generated = outputs[0][prompt_tokens:]
        completion_tokens = int(generated.shape[-1])
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()

        tokens_per_s = (
            round(completion_tokens / (latency_ms / 1000.0), 2)
            if latency_ms > 0 and completion_tokens > 0
            else 0.0
        )

        spec = self.registry.specs.get(self.model) if self.registry else None
        return LLMResult(
            text=text,
            model=spec.hf_id if spec is not None else self.model,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached=False,
            backend=self.name,
            raw={
                "device": handle.device,
                "dtype": handle.dtype,
                "quant": spec.quant if spec is not None else None,
                "vram_mb": handle.vram_measured_mb or 0.0,
                "est_vram_mb": handle.est_vram_mb,
                "load_ms": handle.load_ms,
                "tokens_per_s": tokens_per_s,
                "registry_model": self.model,
            },
        )

    # -- nội bộ ----------------------------------------------------------
    @staticmethod
    def _pad_token_id(tokenizer: Any) -> int:
        """Token pad để dùng cho `generate` (fallback sang eos)."""
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(tokenizer, "eos_token_id", None)
        if pad_id is None:
            pad_id = 0
        return int(pad_id)

    @staticmethod
    def _build_prompt(tokenizer: Any, messages: list[dict]) -> str:
        """Dựng prompt cho model.

        Ưu tiên `apply_chat_template` của model (Qwen2.5-Instruct có sẵn).
        Nếu model không có template -> dùng định dạng phẳng.
        """
        conversation = [
            m for m in (messages or []) if isinstance(m, dict) and m.get("content")
        ]
        apply_template = getattr(tokenizer, "apply_chat_template", None)
        if apply_template is not None and getattr(tokenizer, "chat_template", None):
            try:
                return str(
                    apply_template(
                        conversation, tokenize=False, add_generation_prompt=True
                    )
                )
            except Exception:  # noqa: BLE001 - template lỗi thì lùi về định dạng phẳng
                pass

        system = "\n".join(
            str(m.get("content", "")) for m in conversation if m.get("role") == "system"
        )
        user = "\n".join(
            str(m.get("content", "")) for m in conversation if m.get("role") == "user"
        )
        return _PLAIN_PROMPT_TEMPLATE.format(system=system, user=user)


# ---------------------------------------------------------------------------
# Dùng từ config
# ---------------------------------------------------------------------------


def build_hf_local(
    cfg: Any,
    node: Any = None,
    name: str = "hf_local",
    registry: ModelRegistry | None = None,
) -> HFLocalBackend:
    """Dựng backend `hf_local` từ config.

    Thứ tự ưu tiên tên model: `llm.model` -> `backends.hf_local.model` ->
    `qwen3b`. `llm.model` được ưu tiên vì profile ghi đè khoá này (đổi profile
    là đổi model), còn `backends.hf_local.model` chỉ là giá trị mặc định.
    Tham số sinh lấy từ `llm.gen`.
    """
    node_dict = _as_dict(node) or {}
    getter = getattr(cfg, "get", None)

    model = str(getter("llm.model", "") or "").strip() if getter else ""
    if not model:
        model = str(node_dict.get("model") or "").strip()
    if not model:
        model = "qwen3b"

    default_gen = _as_dict(getter("llm.gen", {}) if getter else {}) or {}

    if registry is None:
        # Import loader để đăng ký `causal_lm` vào LOADERS trước khi tạo registry.
        from artrag.models import loaders  # noqa: PLC0415,F401

        registry = create_registry(cfg)

    return HFLocalBackend(
        name=name,
        registry=registry,
        model=model,
        default_gen=default_gen,
    )


__all__ = ["DEFAULT_GEN", "HFLocalBackend", "build_hf_local"]
