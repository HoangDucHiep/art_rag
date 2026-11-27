"""Test cho backend LLM chạy local `hf_local` (T2.4).

Tất cả test dùng model/tokenizer **giả** — không nạp model thật, không cần GPU,
không cần mạng. Mục tiêu là kiểm tra logic điều phối: dựng prompt, đếm token,
giải mã, số đo và việc nối vào factory `get_backend`.
"""

from __future__ import annotations

import pytest

from artrag.config import Config
from artrag.errors import BackendError, ModelLoadError
from artrag.llm import get_backend
from artrag.llm.base import LLMResult
from artrag.llm.hf_local import HFLocalBackend, build_hf_local
from artrag.registry import ModelHandle, ModelRegistry, ModelSpec


# ---------------------------------------------------------------------------
# Bản giả: tokenizer + model causal LM
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """Tokenizer giả: encode theo độ dài chuỗi, có chat template."""

    def __init__(self, with_template: bool = True) -> None:
        self.with_template = with_template
        self.chat_template = "fake-template" if with_template else None
        self.padding_side = "right"
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.pad_token = "<pad>"

    def __call__(self, text, return_tensors=None, truncation=False):
        class _Batch(dict):
            def to(self, device):  # noqa: ANN001, D102 - giả lập torch.Tensor.to
                return self

        batch = _Batch()
        # Mỗi "word" -> 1 token, để token count dễ kiểm chứng.
        n_tokens = max(1, len(str(text).split()))
        batch["input_ids"] = _FakeTensor(n_tokens)
        return batch

    def encode(self, text):
        return list(range(max(1, len(str(text).split()))))

    def decode(self, tokens, skip_special_tokens=True):
        # Mỗi token -> một từ "tok".
        return " ".join("tok" for _ in range(len(tokens)))

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        if not self.with_template:
            raise ValueError("không có chat template")
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


class _FakeTensor:
    """Tensor giả 1 chiều với `shape`, `__len__`, `to()` và cắt lát."""

    def __init__(self, length: int) -> None:
        self.length = int(length)
        self.shape = (int(length),)
        self.device = "cpu"

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, key):
        # Hỗ trợ `tensor[a:b]` (cắt lát) và `tensor[0]` (phần tử đầu).
        if isinstance(key, slice):
            start = key.start or 0
            stop = key.stop if key.stop is not None else self.length
            return _FakeTensor(max(0, stop - start))
        if key == 0:
            return self
        raise IndexError(key)

    def to(self, device):  # noqa: ANN001, D102 - giả lập torch.Tensor.to
        return self


class FakeModel:
    """Model causal LM giả: `generate` trả về prompt + N token mới."""

    def __init__(self, new_tokens: int = 5) -> None:
        self.new_tokens = new_tokens
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        prompt_len = int(kwargs["input_ids"].shape[-1])
        total = prompt_len + self.new_tokens
        return [_FakeTensor(total)]


def _make_handle(
    model: object | None = None,
    tokenizer: object | None = None,
    device: str = "cuda",
    dtype: str = "fp16",
    vram_mb: float = 2400.0,
) -> ModelHandle:
    return ModelHandle(
        name="qwen3b",
        model=model if model is not None else FakeModel(),
        processor=tokenizer if tokenizer is not None else FakeTokenizer(),
        device=device,
        dtype=dtype,
        est_vram_mb=2400,
        vram_measured_mb=vram_mb,
        load_ms=1234,
        pinned=False,
        task="causal_lm",
    )


class FakeRegistry:
    """Registry giả: trả về handle cố định, `specs` lấy từ `configs/models.yaml`."""

    def __init__(self, handle: ModelHandle | None = None) -> None:
        self.handle = handle or _make_handle()
        self.requested: list[str] = []
        self.specs = {
            "qwen3b": ModelSpec(
                name="qwen3b",
                task="causal_lm",
                hf_id="Qwen/Qwen2.5-3B-Instruct",
                dtype="fp16",
                quant="4bit",
                est_vram_mb=2400,
                devices=("cuda",),
            )
        }

    def get(self, name):
        self.requested.append(name)
        return self.handle


# ---------------------------------------------------------------------------
# (a) Sinh câu trả lời
# ---------------------------------------------------------------------------


def test_generate_returns_text_and_token_counts():
    registry = FakeRegistry()
    backend = HFLocalBackend(registry=registry, model="qwen3b")

    result = backend.generate([{"role": "user", "content": "xin chao ban"}])

    assert isinstance(result, LLMResult)
    assert result.text == "tok tok tok tok tok"
    assert result.completion_tokens == 5
    assert result.prompt_tokens > 0
    assert result.backend == "hf_local"
    assert result.model == "Qwen/Qwen2.5-3B-Instruct"
    assert result.cached is False
    assert result.latency_ms >= 0


def test_generate_uses_chat_template():
    tokenizer = FakeTokenizer(with_template=True)
    registry = FakeRegistry(_make_handle(tokenizer=tokenizer))
    backend = HFLocalBackend(registry=registry, model="qwen3b")

    backend.generate([{"role": "user", "content": "hello"}])

    # `apply_chat_template` được dùng -> prompt chứa tiền tố vai trò.
    prompt_len = len(tokenizer.encode("user: hello"))
    assert registry.handle.processor is tokenizer


def test_generate_falls_back_to_plain_prompt_without_chat_template():
    tokenizer = FakeTokenizer(with_template=False)
    registry = FakeRegistry(_make_handle(tokenizer=tokenizer))
    backend = HFLocalBackend(registry=registry, model="qwen3b")

    result = backend.generate(
        [
            {"role": "system", "content": "ban la tro ly"},
            {"role": "user", "content": "hello"},
        ]
    )

    assert result.text
    assert result.prompt_tokens > 0


def test_generate_uses_greedy_when_temperature_is_zero():
    model = FakeModel()
    registry = FakeRegistry(_make_handle(model=model))
    backend = HFLocalBackend(registry=registry, model="qwen3b")

    backend.generate([{"role": "user", "content": "hi"}], temperature=0.0)

    assert model.calls[0]["do_sample"] is False


def test_generate_uses_sampling_when_temperature_positive():
    model = FakeModel()
    registry = FakeRegistry(_make_handle(model=model))
    backend = HFLocalBackend(registry=registry, model="qwen3b")

    backend.generate([{"role": "user", "content": "hi"}], temperature=0.7)

    assert model.calls[0]["do_sample"] is True
    assert model.calls[0]["temperature"] == pytest.approx(0.7)


def test_generate_raw_has_vram_and_tokens_per_second():
    registry = FakeRegistry()
    backend = HFLocalBackend(registry=registry, model="qwen3b")

    result = backend.generate([{"role": "user", "content": "hi"}])

    raw = result.raw
    assert raw["device"] == "cuda"
    assert raw["dtype"] == "fp16"
    assert raw["quant"] == "4bit"
    assert raw["vram_mb"] == pytest.approx(2400.0)
    assert raw["tokens_per_s"] >= 0
    assert raw["registry_model"] == "qwen3b"


def test_generate_raises_backend_error_when_model_load_fails():
    class FailingRegistry(FakeRegistry):
        def get(self, name):
            raise ModelLoadError("khong nap duoc model")

    backend = HFLocalBackend(registry=FailingRegistry(), model="qwen3b")

    with pytest.raises(BackendError) as exc:
        backend.generate([{"role": "user", "content": "hi"}])

    assert "qwen3b" in str(exc.value)


def test_generate_without_registry_raises_backend_error():
    backend = HFLocalBackend(registry=None, model="qwen3b")

    with pytest.raises(BackendError):
        backend.generate([{"role": "user", "content": "hi"}])


# ---------------------------------------------------------------------------
# (b) count_tokens
# ---------------------------------------------------------------------------


def test_count_tokens_falls_back_to_heuristic_before_load():
    backend = HFLocalBackend(registry=FakeRegistry(), model="qwen3b")

    # Chưa nạp model -> dùng heuristic (~4 ký tự/token).
    assert backend.count_tokens("a" * 40) == 10


def test_count_tokens_uses_tokenizer_after_load():
    registry = FakeRegistry()
    backend = HFLocalBackend(registry=registry, model="qwen3b")

    backend.generate([{"role": "user", "content": "hai ba"}])

    # Tokenizer giả encode theo số từ -> 2 token.
    assert backend.count_tokens("hai ba") == 2


# ---------------------------------------------------------------------------
# (c) set_model / đổi model
# ---------------------------------------------------------------------------


def test_set_model_changes_registry_model():
    registry = FakeRegistry()
    backend = HFLocalBackend(registry=registry, model="qwen3b")

    backend.set_model("qwen1_5b")
    backend.generate([{"role": "user", "content": "hi"}])

    assert backend.model == "qwen1_5b"
    assert registry.requested == ["qwen1_5b"]


# ---------------------------------------------------------------------------
# (d) Dựng backend từ config
# ---------------------------------------------------------------------------


def test_build_hf_local_reads_model_from_llm_config():
    cfg = Config({"llm": {"model": "qwen1_5b", "gen": {"max_new_tokens": 64}}})
    registry = FakeRegistry()

    backend = build_hf_local(cfg, node=None, registry=registry)

    assert isinstance(backend, HFLocalBackend)
    assert backend.model == "qwen1_5b"
    assert backend.default_gen["max_new_tokens"] == 64


def test_build_hf_local_falls_back_to_backend_node_model():
    cfg = Config({"llm": {}})
    registry = FakeRegistry()

    backend = build_hf_local(cfg, node={"model": "qwen3b"}, registry=registry)

    assert backend.model == "qwen3b"


def test_build_hf_local_default_model():
    cfg = Config({"llm": {}})
    registry = FakeRegistry()

    backend = build_hf_local(cfg, node=None, registry=registry)

    assert backend.model == "qwen3b"


def test_llm_model_overrides_backend_node_model():
    """`llm.model` (profile ghi đè) phải thắng `backends.hf_local.model`."""
    cfg = Config({"llm": {"model": "qwen1_5b"}})
    registry = FakeRegistry()

    backend = build_hf_local(cfg, node={"model": "qwen3b"}, registry=registry)

    assert backend.model == "qwen1_5b"


# ---------------------------------------------------------------------------
# (e) Nối vào factory `get_backend`
# ---------------------------------------------------------------------------


def test_get_backend_builds_hf_local_without_error():
    cfg = Config({"llm": {"backend": "hf_local", "model": "qwen3b"}})

    # `use_cache=False` -> không bọc cache, backend trả về trực tiếp.
    backend = get_backend(cfg, use_cache=False)

    assert isinstance(backend, HFLocalBackend)
    assert backend.model == "qwen3b"


def test_get_backend_wraps_hf_local_in_cache():
    cfg = Config({"llm": {"backend": "hf_local", "model": "qwen3b"}})

    backend = get_backend(cfg, use_cache=True)

    inner = getattr(backend, "inner", None)
    assert isinstance(inner, HFLocalBackend)


def test_hf_local_is_not_wrapped_in_retry_backend():
    """Lỗi cục bộ không hết khi thử lại -> không được bọc retry."""
    from artrag.llm.retry import RetryBackend

    cfg = Config({"llm": {"backend": "hf_local", "model": "qwen3b"}})

    backend = get_backend(cfg, use_cache=False)

    layers = []
    current = backend
    while current is not None:
        layers.append(type(current).__name__)
        current = getattr(current, "inner", None)

    assert "RetryBackend" not in layers


def test_openai_compat_still_wrapped_in_retry():
    from artrag.llm.retry import RetryBackend

    cfg = Config(
        {
            "llm": {"backend": "groq"},
            "backends": {
                "groq": {
                    "type": "openai_compat",
                    "base_url": "https://example.invalid/v1",
                    "model": "m",
                    "api_key_envs": ["GROQ_API_KEY"],
                }
            },
        }
    )

    backend = get_backend(cfg, use_cache=False)

    layers = []
    current = backend
    while current is not None:
        layers.append(type(current).__name__)
        current = getattr(current, "inner", None)

    assert "RetryBackend" in layers


def test_flan_t5_backend_is_supported():
    """T2.5 đã làm -> `flan_t5` phải dựng được backend."""
    cfg = Config(
        {
            "llm": {"backend": "flan_t5"},
            "backends": {"flan_t5": {"type": "flan_t5"}},
        }
    )

    backend = get_backend(cfg, use_cache=False)

    assert type(backend).__name__ == "FlanT5Backend"


# ---------------------------------------------------------------------------
# (f) Loader `causal_lm` được đăng ký
# ---------------------------------------------------------------------------


def test_causal_lm_loader_is_registered():
    from artrag.models.loaders import REGISTERED_TASKS
    from artrag.registry import LOADERS

    assert "causal_lm" in REGISTERED_TASKS
    assert "causal_lm" in LOADERS
    assert callable(LOADERS["causal_lm"])


def test_causal_lm_4bit_rejects_cpu_device():
    """4-bit của bitsandbytes chỉ chạy được trên GPU."""
    from artrag.models.loaders import load_causal_lm

    spec = ModelSpec(
        name="qwen3b",
        task="causal_lm",
        hf_id="Qwen/Qwen2.5-3B-Instruct",
        quant="4bit",
        devices=("cuda",),
    )

    with pytest.raises(ModelLoadError) as exc:
        load_causal_lm(spec, "cpu", "fp32")

    assert "4-bit" in str(exc.value) or "GPU" in str(exc.value)


def test_causal_lm_rejects_unknown_quant():
    from artrag.models.loaders import load_causal_lm

    spec = ModelSpec(
        name="qwen3b",
        task="causal_lm",
        hf_id="Qwen/Qwen2.5-3B-Instruct",
        quant="8bit",
        devices=("cuda",),
    )

    with pytest.raises(ModelLoadError):
        load_causal_lm(spec, "cuda", "fp16")


def test_real_registry_has_causal_lm_specs():
    """`configs/models.yaml` phải có model cho task `causal_lm`."""
    from artrag.registry import load_model_specs

    specs = load_model_specs()
    causal = {name: s for name, s in specs.items() if s.task == "causal_lm"}

    assert causal, "configs/models.yaml phải có ít nhất một model causal_lm"
    for name, spec in causal.items():
        assert spec.hf_id, f"model {name} thiếu hf_id"
        assert spec.devices, f"model {name} thiếu devices"
