"""Backend LLM `flan_t5` (T2.5, SYSTEM_DESIGN 6.3).

FLAN-T5 chỉ hiểu tiếng Anh, nên backend này chạy pipeline EN:

    câu hỏi (VI -> EN) + context (EN)  ->  FLAN-T5  ->  câu trả lời (EN)  ->  dịch EN->VI

Citation `[IMG-01]` có thể xuất hiện trong context (header của mỗi evidence) và
trong câu trả lời của model. Để máy dịch không làm hỏng nhãn citation, văn bản
được **chia đoạn theo citation**: chỉ đoạn thuần mới được dịch, đoạn chứa
nhãn citation được giữ nguyên (xem `artrag.pipeline.translate`).

Model được nạp **lazy** qua `ModelRegistry` (task `seq2seq`), giống `hf_local`:
dựng backend không tốn VRAM, chỉ nạp khi `generate()` được gọi lần đầu.

Backend này chạy local nên **không** bọc `RetryBackend` (xem `LOCAL_BACKENDS`).
"""

from __future__ import annotations

import time
from typing import Any

from artrag.errors import BackendError, ModelLoadError
from artrag.llm.base import LLMBackend, LLMResult, estimate_tokens
from artrag.registry import ModelHandle, ModelRegistry, create_registry

#: Tham số sinh mặc định (greedy, ngắn gọn).
DEFAULT_GEN: dict[str, Any] = {
    "max_new_tokens": 128,
    "temperature": 0.0,
}

#: Tên model trong registry cho phần sinh (FLAN-T5) và phần dịch (EN -> VI).
DEFAULT_GEN_MODEL = "flan_t5"
DEFAULT_EN2VI_MODEL = "en2vi"
DEFAULT_VI2EN_MODEL = "vi2en"

#: Chỉ dẫn mặc định cho FLAN-T5 (tiếng Anh) ở **lượt 1** (sinh câu trả lời).
#:
#: Có thể ghi đè qua `backends.flan_t5.instruction` (chuỗi rỗng = tắt).
#:
#: **Không được** nhắc model trích dẫn nhãn ở lượt 1: đo thực tế trên
#: FLAN-T5-base cho thấy câu "cite the evidence labels exactly as written" làm
#: model **chỉ phát nhãn** (`[IMG-01]. [IMG-01]`) thay vì trả lời — 8/15 mẫu
#: rỗng nội dung. Nhãn được gắn ở **lượt 2** (xem `DEFAULT_LABEL_INSTRUCTION`).
#:
#: Cũng không nhắc "từ chối khi thiếu bằng chứng": FLAN-T5-base không từ chối
#: được, nó **bịa** (vd. đấu giá "$500.000"). Logic abstain thuộc tầng quy tắc
#: của pipeline (§5.9), không thuộc prompt.
DEFAULT_INSTRUCTION = "Answer the question in a short sentence, using only the evidence."

#: Chỉ dẫn của **lượt 2**: hỏi model nhãn bằng chứng đã dùng.
#:
#: FLAN-T5-base **không** tự phát lại nhãn `[IMG-xx]` trong câu trả lời (đã
#: đo thực tế: 0/11 mẫu prompt trả lời có nhãn). Vì vậy backend dùng cơ chế
#: **hai lượt**: lượt 1 sinh câu trả lời, lượt 2 hỏi "câu trả lời này đến từ
#: nhãn bằng chứng nào" -> nhãn được chèn lại vào câu trả lời. Mẫu này đo được
#: 6/6 đúng trên 6 cặp câu hỏi/bằng chứng mẫu.
DEFAULT_LABEL_INSTRUCTION = (
    "Which evidence label does this answer come from? "
    "Answer with the label only, for example [IMG-01]."
)


def _as_dict(value: Any) -> dict[str, Any] | None:
    """Chuyển `Config`/dict về dict thuần; None nếu không phải mapping."""
    from artrag.config import Config  # noqa: PLC0415 - tránh vòng import

    if isinstance(value, Config):
        return value.raw
    if isinstance(value, dict):
        return value
    return None


class FlanT5Backend:
    """Backend sinh câu trả lời bằng FLAN-T5 (EN) rồi dịch sang tiếng Việt."""

    def __init__(
        self,
        name: str = "flan_t5",
        registry: ModelRegistry | None = None,
        model: str = DEFAULT_GEN_MODEL,
        en2vi_model: str = DEFAULT_EN2VI_MODEL,
        vi2en_model: str = DEFAULT_VI2EN_MODEL,
        default_gen: dict[str, Any] | None = None,
        translator: Any = None,
        instruction: str = DEFAULT_INSTRUCTION,
        label_instruction: str = DEFAULT_LABEL_INSTRUCTION,
    ) -> None:
        self.name = name
        self.registry = registry
        #: Tên model sinh câu trả lời (khoá trong `configs/models.yaml`).
        self.model = model
        #: Tên model dịch EN -> VI trong registry.
        self.en2vi_model = en2vi_model
        #: Tên model dịch VI -> EN trong registry (dùng cho câu hỏi tiếng Việt).
        self.vi2en_model = vi2en_model
        self.default_gen = dict(default_gen or DEFAULT_GEN)
        #: Chỉ dẫn lượt 1 (chuỗi rỗng = không thêm chỉ dẫn nào).
        self.instruction = str(instruction or "")
        #: Chỉ dẫn lượt 2 (chọn nhãn bằng chứng); chuỗi rỗng = tắt lượt 2.
        self.label_instruction = str(label_instruction or "")
        self._handle: ModelHandle | None = None
        self._translator = translator
        #: Số token sinh ở lần gọi gần nhất (đo bởi `_generate_text`).
        self._last_completion_tokens = 0

    # -- model -----------------------------------------------------------
    def set_model(self, name: str) -> None:
        """Đổi model sinh câu trả lời (tên trong `configs/models.yaml`)."""
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
                    "kiểm tra tên model trong `backends.flan_t5.model` và "
                    "`configs/models.yaml`."
                ),
            ) from exc
        self._handle = handle
        return handle

    # -- translator ------------------------------------------------------
    @property
    def translator(self) -> Any:
        """Bộ dịch dùng chung (tạo lazily vì cần registry)."""
        if self._translator is None:
            from artrag.pipeline.translate import Translator  # noqa: PLC0415

            self._translator = Translator(
                registry=self.registry,
                vi2en_model=self.vi2en_model,
                en2vi_model=self.en2vi_model,
            )
        return self._translator

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
        """Sinh câu trả lời bằng FLAN-T5 rồi dịch sang tiếng Việt.

        Quy trình (hai lượt, xem `DEFAULT_LABEL_INSTRUCTION`):

        1. Câu hỏi / system prompt tiếng Việt -> dịch sang tiếng Anh.
        2. **Lượt 1**: FLAN-T5 sinh câu trả lời (tiếng Anh).
        3. **Lượt 2** (tuỳ chọn): hỏi model nhãn bằng chứng đã dùng, chèn nhãn
           `[IMG-xx]` vào cuối câu trả lời.
        4. Dịch câu trả lời EN -> VI; các đoạn chứa citation được giữ nguyên.
        """
        handle = self._get_handle()
        model = handle.model
        tokenizer = handle.processor
        if model is None or tokenizer is None:
            raise BackendError(
                f"Model {self.model!r} không có model/tokenizer để sinh câu trả lời.",
                hint="kiểm tra loader `seq2seq` trong src/artrag/models/loaders.py.",
            )

        params = dict(self.default_gen)
        params.update({k: v for k, v in gen.items() if v is not None})
        max_new_tokens = int(params.get("max_new_tokens", 128))
        temperature = float(params.get("temperature", 0.0))

        # -- 1. Câu hỏi / system prompt: VI -> EN -------------------------
        started = time.perf_counter()
        messages_en, question_en = self._prepare_messages_en(messages)
        question_ms = int((time.perf_counter() - started) * 1000)

        # -- 2. Lượt 1: sinh câu trả lời (tiếng Anh) ----------------------
        # Đuôi "Answer:" là bắt buộc về mặt định dạng: đo thực tế trên
        # FLAN-T5-base cho thấy prompt có `Question: … / Answer:` đạt 15/15
        # câu hỏi có nội dung đúng, còn các biến thể khác rớt 10–14/15.
        prompt = self._build_prompt(messages_en, self.instruction)
        prompt = self._with_answer_cue(prompt)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(handle.device) for k, v in inputs.items()}
        prompt_tokens = int(inputs["input_ids"].shape[-1])

        started = time.perf_counter()
        text_en = self._generate_text(model, tokenizer, inputs, max_new_tokens, temperature, params)
        latency_ms = int((time.perf_counter() - started) * 1000)

        completion_tokens = self._last_completion_tokens
        tokens_per_s = (
            round(completion_tokens / (latency_ms / 1000.0), 2)
            if latency_ms > 0 and completion_tokens > 0
            else 0.0
        )

        # -- 3. Lượt 2: chọn nhãn bằng chứng đã dùng ----------------------
        label_ms = 0
        labels: list[str] = []
        if self.label_instruction and text_en:
            started = time.perf_counter()
            labels = self._pick_evidence_labels(
                model, tokenizer, messages_en, text_en, max_new_tokens
            )
            label_ms = int((time.perf_counter() - started) * 1000)
            if labels:
                text_en = f"{text_en} {' '.join(labels)}"

        # -- 4. Dịch EN -> VI (bảo vệ citation bên trong `Translator`) ----
        started = time.perf_counter()
        text_vi = self.translator.en_to_vi(text_en) if text_en else ""
        translate_ms = int((time.perf_counter() - started) * 1000)

        spec = self.registry.specs.get(self.model) if self.registry else None
        return LLMResult(
            text=text_vi,
            model=spec.hf_id if spec is not None else self.model,
            latency_ms=question_ms + latency_ms + label_ms + translate_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached=False,
            backend=self.name,
            raw={
                "device": handle.device,
                "dtype": handle.dtype,
                "vram_mb": handle.vram_measured_mb or 0.0,
                "est_vram_mb": handle.est_vram_mb,
                "load_ms": handle.load_ms,
                "tokens_per_s": tokens_per_s,
                "generate_ms": latency_ms,
                "translate_ms": translate_ms,
                "question_ms": question_ms,
                "label_ms": label_ms,
                "text_en": text_en,
                "question_en": question_en,
                "labels": labels,
                "registry_model": self.model,
                "en2vi_model": self.en2vi_model,
                "vi2en_model": self.vi2en_model,
            },
        )

    # -- nội bộ ----------------------------------------------------------
    def _generate_text(
        self,
        model: Any,
        tokenizer: Any,
        inputs: dict[str, Any],
        max_new_tokens: int,
        temperature: float,
        params: dict[str, Any],
    ) -> str:
        """Sinh văn bản với model seq2seq; trả về chuỗi đã giải mã.

        Với model **encoder-decoder** (FLAN-T5), output có dạng
        `[decoder_start_token] + token_sinh_moi` — KHÔNG phải `prompt + token`.
        Vì vậy phải cắt từ vị trí 1, nếu không sẽ mất token sinh đầu tiên
        (với prompt ngắn, output dài bằng prompt -> sinh ra 0 token).
        """
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
            generation_kwargs["do_sample"] = False
            for key in ("temperature", "top_p", "top_k"):
                generation_kwargs[key] = None

        outputs = model.generate(**inputs, **generation_kwargs)

        is_encoder_decoder = bool(
            getattr(getattr(model, "config", None), "is_encoder_decoder", False)
        )
        prompt_len = int(inputs["input_ids"].shape[-1])
        if is_encoder_decoder:
            generated = outputs[0][1:]
        else:
            generated = outputs[0][prompt_len:]
        self._last_completion_tokens = int(generated.shape[-1])
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _pick_evidence_labels(
        self,
        model: Any,
        tokenizer: Any,
        messages_en: list[dict],
        answer_en: str,
        max_new_tokens: int,
    ) -> list[str]:
        """Lượt 2: hỏi model nhãn bằng chứng đã dùng cho câu trả lời.

        FLAN-T5-base không tự phát lại nhãn `[IMG-xx]` trong câu trả lời, nên
        ta hỏi riêng: "câu trả lời này đến từ nhãn bằng chứng nào?". Chỉ giữ
        những nhãn **có thật trong context** để không chèn nhãn bịa.

        Trả về danh sách nhãn (ví dụ `["[IMG-01]"]`); rỗng nếu không xác định
        được.
        """
        from artrag.pipeline.translate import CITATION_RE  # noqa: PLC0415

        context = self._build_prompt(messages_en)
        available = list(dict.fromkeys(CITATION_RE.findall(context)))
        if not available:
            return []

        prompt = (
            f"{context}\n\n"
            f"Answer: {answer_en}\n"
            f"{self.label_instruction}\n"
            "Evidence label:"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        device = self._handle.device if self._handle else "cpu"
        inputs = {k: v.to(device) for k, v in inputs.items()}

        try:
            text = self._generate_text(
                model,
                tokenizer,
                inputs,
                max_new_tokens=min(32, max_new_tokens),
                temperature=0.0,
                params={},
            )
        except Exception:  # noqa: BLE001 - lượt 2 hỏng thì vẫn giữ câu trả lời
            return []

        found = [label for label in CITATION_RE.findall(text) if label in available]
        # Giữ đúng thứ tự xuất hiện trong context, bỏ trùng.
        ordered = [label for label in available if label in found]
        return ordered[:1] if ordered else []

    def _prepare_messages_en(self, messages: list[dict]) -> tuple[list[dict], str]:
        """Dịch messages sang tiếng Anh nếu cần.

        FLAN-T5 chỉ hiểu tiếng Anh nên mọi nội dung tiếng Việt (câu hỏi của
        người dùng, system prompt) đều được dịch trước. Nội dung đã là tiếng
        Anh được giữ nguyên.

        Trả về `(messages_en, question_en)` — `question_en` là nội dung message
        role `user` sau khi dịch (dùng để ghi vào `LLMResult.raw`).
        """
        from artrag.pipeline.translate import looks_vietnamese  # noqa: PLC0415

        conversation = [
            m for m in (messages or []) if isinstance(m, dict) and m.get("content")
        ]
        if not conversation:
            return [], ""

        result: list[dict] = []
        question_en = ""
        for message in conversation:
            content = str(message.get("content", ""))
            role = str(message.get("role", "user"))
            if looks_vietnamese(content):
                content = self.translator.vi_to_en(content)
            if role == "user":
                question_en = content
            result.append({"role": role, "content": content})
        return result, question_en

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
    def _with_answer_cue(prompt: str) -> str:
        """Thêm đuôi `Answer:` cho lượt 1 nếu chưa có.

        FLAN-T5 được huấn luyện theo mẫu "Question: … / Answer:", nên prompt
        phải kết thúc bằng `Answer:` để model sinh phần trả lời (nếu không nó
        có xu hướng sinh tiếp một câu hỏi nữa).
        """
        text = (prompt or "").rstrip()
        if not text:
            return "Answer:"
        if text.lower().endswith("answer:"):
            return text
        return f"{text}\n\nAnswer:"

    @staticmethod
    def _build_prompt(messages: list[dict], instruction: str = "") -> str:
        """Dựng prompt phẳng cho FLAN-T5 (model này không có chat template).

        System prompt và user prompt được nối lại; nếu có nhiều message user
        thì nối bằng dấu xuống dòng. `instruction` (nếu có) đặt trước cả hai.
        """
        conversation = [
            m for m in (messages or []) if isinstance(m, dict) and m.get("content")
        ]
        system = "\n".join(
            str(m.get("content", "")) for m in conversation if m.get("role") == "system"
        )
        user = "\n".join(
            str(m.get("content", "")) for m in conversation if m.get("role") == "user"
        )
        if system and user:
            body = f"{system}\n\n{user}"
        else:
            body = system or user
        if not instruction:
            return body
        return f"{instruction}\n\n{body}" if body else instruction


# ---------------------------------------------------------------------------
# Dùng từ config
# ---------------------------------------------------------------------------


def build_flan_t5(
    cfg: Any,
    node: Any = None,
    name: str = "flan_t5",
    registry: ModelRegistry | None = None,
) -> FlanT5Backend:
    """Dựng backend `flan_t5` từ config.

    Thứ tự ưu tiên tên model: `backends.flan_t5.model` -> `flan_t5`; tên model
    dịch lấy từ `backends.flan_t5.en2vi_model` / `vi2en_model`, mặc định là
    `en2vi` / `vi2en` trong `configs/models.yaml`.

    Khác với `hf_local`, `llm.model` **không** được dùng ở đây: khoá đó điều
    khiển model của backend `hf_local` và sẽ ghi đè khi đổi profile.
    """
    node_dict = _as_dict(node) or {}

    model = str(node_dict.get("model") or "").strip() or DEFAULT_GEN_MODEL
    en2vi = str(node_dict.get("en2vi_model") or "").strip() or DEFAULT_EN2VI_MODEL
    vi2en = str(node_dict.get("vi2en_model") or "").strip() or DEFAULT_VI2EN_MODEL
    instruction = str(node_dict.get("instruction") or DEFAULT_INSTRUCTION)
    label_instruction = str(
        node_dict.get("label_instruction") or DEFAULT_LABEL_INSTRUCTION
    )

    default_gen = _as_dict(getattr(cfg, "get", lambda *_: None)("llm.gen", {}) or {}) or {}

    if registry is None:
        # Import loader để đăng ký `seq2seq` vào LOADERS trước khi tạo registry.
        from artrag.models import loaders  # noqa: PLC0415,F401

        registry = create_registry(cfg)

    translator = None
    if registry is not None:
        from artrag.config import paths  # noqa: PLC0415
        from artrag.pipeline.translate import (  # noqa: PLC0415
            DEFAULT_CACHE_DIR,
            Translator,
        )

        try:
            cache_dir = paths(cfg).cache / DEFAULT_CACHE_DIR
        except Exception:  # noqa: BLE001 - config lỗi thì bỏ cache, không chặn backend
            cache_dir = None
        translator = Translator(
            registry=registry,
            vi2en_model=vi2en,
            en2vi_model=en2vi,
            cache_dir=cache_dir,
        )

    return FlanT5Backend(
        name=name,
        registry=registry,
        model=model,
        en2vi_model=en2vi,
        vi2en_model=vi2en,
        default_gen=default_gen,
        translator=translator,
        instruction=instruction,
        label_instruction=label_instruction,
    )


__all__ = [
    "DEFAULT_EN2VI_MODEL",
    "DEFAULT_GEN",
    "DEFAULT_GEN_MODEL",
    "DEFAULT_INSTRUCTION",
    "DEFAULT_LABEL_INSTRUCTION",
    "DEFAULT_VI2EN_MODEL",
    "FlanT5Backend",
    "build_flan_t5",
]
