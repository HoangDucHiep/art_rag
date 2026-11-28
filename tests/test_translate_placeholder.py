"""Test cho cơ chế bảo vệ citation khi dịch và backend `flan_t5` (T2.5).

Tất cả test dùng model/tokenizer **giả** — không nạp model thật, không cần GPU,
không cần mạng. Mục tiêu là kiểm tra:

- hàm thuần tách/ghép đoạn (bất biến khớp nguyên văn);
- đoạn chứa citation **không** được đưa qua máy dịch;
- cache đĩa của bộ dịch;
- backend `flan_t5` dịch câu hỏi VI->EN và câu trả lời EN->VI.
"""

from __future__ import annotations

import pytest

from artrag.config import Config
from artrag.errors import BackendError
from artrag.llm import get_backend
from artrag.llm.base import LLMResult
from artrag.llm.flan_t5 import FlanT5Backend, build_flan_t5
from artrag.pipeline.translate import (
    Translator,
    join_segments,
    looks_vietnamese,
    protect_citations,
    protect_names,
    restore_names,
    split_segments,
    translate_segments,
)
from artrag.registry import ModelHandle, ModelSpec


# ---------------------------------------------------------------------------
# (a) Hàm thuần: tách / ghép đoạn
# ---------------------------------------------------------------------------


def test_split_segments_marks_citation_segments():
    text = "The painting shows two men rowing [IMG-01] and a crowd."

    segments = split_segments(text)

    assert [s.text for s in segments] == [
        "The painting shows two men rowing ",
        "[IMG-01]",
        " and a crowd.",
    ]
    assert [s.has_citation for s in segments] == [False, True, False]


def test_split_segments_without_citation_is_single_plain_segment():
    segments = split_segments("Không có citation ở đây.")

    assert len(segments) == 1
    assert segments[0].has_citation is False


def test_split_segments_handles_multiple_citations():
    text = "A [IMG-01] B [IMG-02] C"

    segments = split_segments(text)

    assert [s.text for s in segments] == [
        "A ",
        "[IMG-01]",
        " B ",
        "[IMG-02]",
        " C",
    ]
    assert sum(1 for s in segments if s.has_citation) == 2


def test_split_segments_empty_text():
    assert split_segments("") == []
    assert protect_citations("") == []


def test_join_segments_restores_original_text():
    """Bất biến: tách rồi ghép phải khớp nguyên văn."""
    samples = [
        "The painting shows two men rowing [IMG-01] and a crowd.",
        "A [IMG-01] B [IMG-02] C",
        "Không có citation ở đây.",
        "[IMG-01]",
        "Kết thúc bằng citation [IMG-03]",
    ]

    for text in samples:
        assert join_segments(split_segments(text)) == text


def test_protect_citations_returns_segments_in_order():
    text = "A [IMG-01] B"

    assert protect_citations(text) == ["A ", "[IMG-01]", " B"]


def test_citation_regex_accepts_variants():
    from artrag.pipeline.translate import CITATION_RE

    for label in ("[IMG-01]", "[IMG-1]", "[IMG-U]", "[IMG-A1]"):
        assert CITATION_RE.search(label)

    for label in ("[IMAGE-01]", "[img-01]", "[IMG_01]", "[IMG-]"):
        assert not CITATION_RE.search(label)


# ---------------------------------------------------------------------------
# (b) Dịch có bảo vệ citation
# ---------------------------------------------------------------------------


def test_translate_segments_keeps_citation_untouched():
    """Đoạn chứa citation phải được giữ nguyên, không qua máy dịch."""
    text = "The painting shows two men rowing [IMG-01] and a crowd."
    translated_calls: list[str] = []

    def fake_translate(core: str) -> str:
        translated_calls.append(core)
        return f"[{core}]"

    out = translate_segments(text, fake_translate)

    # Chỉ đoạn thuần mới được dịch.
    assert translated_calls == ["The painting shows two men rowing", "and a crowd."]
    # Citation giữ nguyên.
    assert "[IMG-01]" in out
    assert out == (
        "The painting shows two men rowing [IMG-01] and a crowd."
        .replace("The painting shows two men rowing ", "[The painting shows two men rowing] ")
        .replace(" and a crowd.", " [and a crowd.]")
    )


def test_translate_segments_identity_preserves_text():
    """Dịch là hàm nhận dạng -> văn bản phải y hệt bản gốc."""
    text = "Một câu tiếng Việt [IMG-01] và phần còn lại [IMG-02]."

    assert translate_segments(text, lambda core: core) == text


def test_translate_segments_empty_text():
    assert translate_segments("", lambda core: core) == ""


def test_translate_segments_preserves_surrounding_spaces():
    text = "  Hello  [IMG-01]  World  "

    out = translate_segments(text, lambda core: core.upper())

    assert out == "  HELLO  [IMG-01]  WORLD  "


# ---------------------------------------------------------------------------
# (c) Bộ dịch giả + cache
# ---------------------------------------------------------------------------


class FakeRegistry:
    """Registry giả: trả handle cố định, đếm số lần nạp model."""

    def __init__(self) -> None:
        self.requested: list[str] = []
        self.specs = {
            "vi2en": ModelSpec(
                name="vi2en",
                task="seq2seq",
                hf_id="vinai/vinai-translate-vi2en-v2",
                devices=("cpu",),
            ),
            "en2vi": ModelSpec(
                name="en2vi",
                task="seq2seq",
                hf_id="vinai/vinai-translate-en2vi-v2",
                devices=("cpu",),
            ),
        }

    def get(self, name):
        self.requested.append(name)
        return ModelHandle(
            name=name,
            model=object(),
            processor=object(),
            device="cpu",
            dtype="fp32",
            task="seq2seq",
        )


def test_translator_uses_registry_and_counts_calls(monkeypatch):
    from artrag.models import inference

    calls: list[tuple[str, str]] = []

    def fake_translate(handle, text, **gen):
        calls.append((handle.name, text))
        return f"VI({text})" if handle.name == "en2vi" else f"EN({text})"

    monkeypatch.setattr(inference, "translate", fake_translate)

    registry = FakeRegistry()
    translator = Translator(registry=registry, use_cache=False)

    assert translator.en_to_vi("Hello") == "VI(Hello)"
    assert translator.vi_to_en("Xin chào") == "EN(Xin chào)"
    assert translator.model_calls == 2
    assert registry.requested == ["en2vi", "vi2en"]


def test_translator_keeps_citation_untouched(monkeypatch):
    from artrag.models import inference

    seen: list[str] = []

    def fake_translate(handle, text, **gen):
        seen.append(text)
        return f"<<{text}>>"

    monkeypatch.setattr(inference, "translate", fake_translate)

    translator = Translator(registry=FakeRegistry(), use_cache=False)

    out = translator.en_to_vi("Two men rowing [IMG-01] here")

    assert "[IMG-01]" in out
    # Đoạn chứa citation không được đưa qua model.
    assert "[IMG-01]" not in seen
    assert seen == ["Two men rowing", "here"]


def test_translator_cache_avoids_second_model_call(monkeypatch, tmp_path):
    from artrag.models import inference

    calls: list[str] = []

    def fake_translate(handle, text, **gen):
        calls.append(text)
        return f"<<{text}>>"

    monkeypatch.setattr(inference, "translate", fake_translate)

    translator = Translator(
        registry=FakeRegistry(), cache_dir=tmp_path / "translate", use_cache=True
    )

    first = translator.en_to_vi("Hello")
    second = translator.en_to_vi("Hello")

    assert first == second
    assert calls == ["Hello"]
    assert translator.model_calls == 1


def test_translator_without_registry_raises_backend_error():
    translator = Translator(registry=None, use_cache=False)

    with pytest.raises(BackendError):
        translator.en_to_vi("Hello")


# ---------------------------------------------------------------------------
# (d) Nhận diện ngôn ngữ
# ---------------------------------------------------------------------------


def test_looks_vietnamese_detects_diacritics():
    assert looks_vietnamese("Xin chào, bạn khỏe không?")
    assert looks_vietnamese("Đây là một bức tranh")
    assert not looks_vietnamese("This is a painting")


def test_looks_vietnamese_detects_unaccented_words():
    assert looks_vietnamese("ban co khoe khong")
    assert not looks_vietnamese("who painted this painting")


def test_looks_vietnamese_empty():
    assert not looks_vietnamese("")


# ---------------------------------------------------------------------------
# (d2) Bảo vệ danh tính (tên riêng)
# ---------------------------------------------------------------------------


def test_protect_names_replaces_proper_noun_with_placeholder():
    protected, names = protect_names("The painter is Thomas Eakins.")

    assert names == ["Thomas Eakins"]
    assert protected == "The painter is §0§."


def test_protect_names_keeps_multiword_name_together():
    """Tên có từ nối viết thường ("van Gogh") phải giữ trọn."""
    protected, names = protect_names("It was painted by Vincent van Gogh.")

    assert names == ["Vincent van Gogh"]
    assert protected == "It was painted by §0§."


def test_protect_names_ignores_common_words_at_sentence_start():
    """Từ thường gặp viết hoa ở đầu câu không phải tên riêng."""
    protected, names = protect_names("Two men rowing a boat.")

    assert names == []
    assert protected == "Two men rowing a boat."


def test_protect_names_ignores_single_common_noun():
    protected, names = protect_names("Gallery 207")

    assert names == []
    assert protected == "Gallery 207"


def test_protect_names_restore_roundtrip_is_identity():
    """Bất biến: bảo vệ rồi khôi phục phải khớp nguyên văn."""
    samples = [
        "Thomas Eakins",
        "Thomas Eakins painted it in 1873.",
        "It was painted by Vincent van Gogh.",
        "Claude Monet painted the water lilies.",
        "Two men rowing a boat.",
        "The painting shows two men rowing a boat.",
    ]

    for text in samples:
        protected, names = protect_names(text)
        assert restore_names(protected, names) == text


def test_restore_names_puts_original_text_back():
    assert restore_names("§0§ painted it.", ["Thomas Eakins"]) == (
        "Thomas Eakins painted it."
    )


def test_protect_names_empty_text():
    assert protect_names("") == ("", [])


def test_protect_names_roundtrip_with_existing_placeholder_text():
    """Văn bản đã chứa placeholder `[[n]]` không được làm hỏng round-trip."""
    text = "See [[0]] for details about Thomas Eakins."

    protected, names = protect_names(text)

    assert restore_names(protected, names) == text


def test_protect_names_ignores_img_token_without_brackets():
    """Chuỗi "IMG-01" không có ngoặc vuông không phải tên riêng."""
    protected, names = protect_names("IMG-01")

    assert names == []
    assert protected == "IMG-01"


def test_translator_en_to_vi_protects_names(monkeypatch):
    """Tên riêng phải đi qua máy dịch dưới dạng placeholder, rồi khôi phục."""
    from artrag.models import inference

    seen: list[str] = []

    def fake_translate(handle, text, **gen):
        seen.append(text)
        return f"<<{text}>>"

    monkeypatch.setattr(inference, "translate", fake_translate)

    translator = Translator(registry=FakeRegistry(), use_cache=False)
    out = translator.en_to_vi("The painter is Thomas Eakins.")

    assert out == "<<The painter is Thomas Eakins.>>"
    assert seen == ["The painter is §0§."]


def test_translator_vi_to_en_does_not_protect_names(monkeypatch):
    """Hướng VI->EN không dùng placeholder (vi2en xoá placeholder ở đầu đoạn)."""
    from artrag.models import inference

    seen: list[str] = []

    def fake_translate(handle, text, **gen):
        seen.append(text)
        return "TRANSLATED"

    monkeypatch.setattr(inference, "translate", fake_translate)

    translator = Translator(registry=FakeRegistry(), use_cache=False)
    translator.vi_to_en("Tranh này do Thomas Eakins vẽ năm 1873.")

    assert seen == ["Tranh này do Thomas Eakins vẽ năm 1873."]


def test_translator_vi_to_en_keeps_english_evidence_untouched(monkeypatch):
    """Evidence tiếng Anh không được đưa qua vi2en (dịch làm mất thông tin)."""
    from artrag.models import inference

    seen: list[str] = []

    def fake_translate(handle, text, **gen):
        seen.append(text)
        return "TRANSLATED"

    monkeypatch.setattr(inference, "translate", fake_translate)

    translator = Translator(registry=FakeRegistry(), use_cache=False)
    evidence = " Biglin Brothers - Thomas Eakins, 1873. Oil on canvas."

    out = translator.vi_to_en(evidence)

    assert seen == []
    assert out == evidence


def test_translator_vi_to_en_keeps_evidence_header_intact(monkeypatch):
    """Đoạn evidence có citation phải giữ nguyên title + artist + năm."""
    from artrag.models import inference

    seen: list[str] = []

    def fake_translate(handle, text, **gen):
        seen.append(text)
        return "TRANSLATED"

    monkeypatch.setattr(inference, "translate", fake_translate)

    translator = Translator(registry=FakeRegistry(), use_cache=False)
    out = translator.vi_to_en(
        "[IMG-01] Biglin Brothers - Thomas Eakins, 1873. Oil on canvas."
    )

    assert seen == []
    assert out == "[IMG-01] Biglin Brothers - Thomas Eakins, 1873. Oil on canvas."


def test_protect_names_does_not_break_citation_labels():
    """Nhãn citation có cơ chế bảo vệ riêng, không được đụng tới."""
    protected, names = protect_names("[IMG-01] Thomas Eakins")

    assert "[IMG-01]" in protected
    assert "IMG-01" not in names


def test_protect_names_splits_title_and_artist():
    """Định dạng evidence `{title} - {artist}` không được gộp làm một tên."""
    protected, names = protect_names(
        " Biglin Brothers - Thomas Eakins, 1873. Oil on canvas."
    )

    assert names == ["Biglin Brothers", "Thomas Eakins"]
    assert protected == " §0§ - §1§, 1873. Oil on canvas."


# ---------------------------------------------------------------------------
# (e) Backend flan_t5
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """Tokenizer giả: encode theo số từ, decode trả về chuỗi cố định.

    `outputs` là danh sách chuỗi được trả lần lượt theo thứ tự gọi `decode`
    (lượt 1 -> câu trả lời, lượt 2 -> nhãn bằng chứng).
    """

    def __init__(self, output: str | list[str] = "The painter was Thomas Eakins. [IMG-01]") -> None:
        self.outputs = [output] if isinstance(output, str) else list(output)
        self.pad_token_id = 0
        self.eos_token_id = 2
        self._decode_index = 0

    def __call__(self, text, return_tensors=None, truncation=False):
        class _Batch(dict):
            def to(self, device):  # noqa: ANN001, D102 - giả lập torch.Tensor.to
                return self

        batch = _Batch()
        batch["input_ids"] = _FakeTensor(max(1, len(str(text).split())))
        return batch

    def encode(self, text):
        return list(range(max(1, len(str(text).split()))))

    def decode(self, tokens, skip_special_tokens=True):
        index = min(self._decode_index, len(self.outputs) - 1)
        self._decode_index += 1
        return self.outputs[index]


class _FakeTensor:
    """Tensor giả 1 chiều với `shape`, `__len__`, `to()` và cắt lát."""

    def __init__(self, length: int) -> None:
        self.length = int(length)
        self.shape = (int(length),)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, key):
        if isinstance(key, slice):
            start = key.start or 0
            stop = key.stop if key.stop is not None else self.length
            return _FakeTensor(max(0, stop - start))
        if key == 0:
            return self
        raise IndexError(key)

    def to(self, device):  # noqa: ANN001, D102 - giả lập torch.Tensor.to
        return self


class _FakeConfig:
    """Config giả của model seq2seq."""

    def __init__(self, is_encoder_decoder: bool = True) -> None:
        self.is_encoder_decoder = is_encoder_decoder


class FakeSeq2SeqModel:
    """Model seq2seq giả: `generate` trả về output dài `prompt + N token`.

    Với `is_encoder_decoder=True` (giống FLAN-T5), output có dạng
    `[decoder_start_token] + token_sinh_moi` -> dài bằng `1 + new_tokens`.
    """

    def __init__(self, new_tokens: int = 5, is_encoder_decoder: bool = True) -> None:
        self.new_tokens = new_tokens
        self.is_encoder_decoder = is_encoder_decoder
        self.config = _FakeConfig(is_encoder_decoder)
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        prompt_len = int(kwargs["input_ids"].shape[-1])
        total = prompt_len + self.new_tokens
        if self.is_encoder_decoder:
            total = 1 + self.new_tokens
        return [_FakeTensor(total)]


class FakeFlanRegistry:
    """Registry giả cho backend `flan_t5` (model sinh + 2 model dịch)."""

    def __init__(self, tokenizer: FakeTokenizer | None = None) -> None:
        self.tokenizer = tokenizer or FakeTokenizer()
        self.requested: list[str] = []
        self.models: dict[str, FakeSeq2SeqModel] = {}
        self.specs = {
            "flan_t5": ModelSpec(
                name="flan_t5",
                task="seq2seq",
                hf_id="google/flan-t5-base",
                devices=("cpu",),
            ),
            "vi2en": ModelSpec(
                name="vi2en",
                task="seq2seq",
                hf_id="vinai/vinai-translate-vi2en-v2",
                devices=("cpu",),
            ),
            "en2vi": ModelSpec(
                name="en2vi",
                task="seq2seq",
                hf_id="vinai/vinai-translate-en2vi-v2",
                devices=("cpu",),
            ),
        }

    def get(self, name):
        self.requested.append(name)
        model = self.models.setdefault(name, FakeSeq2SeqModel())
        return ModelHandle(
            name=name,
            model=model,
            processor=self.tokenizer,
            device="cpu",
            dtype="fp32",
            est_vram_mb=1000,
            task="seq2seq",
        )


@pytest.fixture()
def patched_translate(monkeypatch):
    """Dịch giả: VI->EN thêm tiền tố `EN:`, EN->VI thêm tiền tố `VI:`."""
    from artrag.models import inference

    def fake_translate(handle, text, **gen):
        if handle.name == "vi2en":
            return f"EN: {text}"
        return f"VI: {text}"

    monkeypatch.setattr(inference, "translate", fake_translate)


def test_flan_t5_generate_returns_vietnamese_text(patched_translate):
    registry = FakeFlanRegistry()
    translator = Translator(registry=registry, use_cache=False)
    backend = FlanT5Backend(registry=registry, translator=translator)

    result = backend.generate([{"role": "user", "content": "Who painted this?"}])

    assert isinstance(result, LLMResult)
    assert result.backend == "flan_t5"
    assert result.text.startswith("VI: ")
    assert result.model == "google/flan-t5-base"
    assert result.completion_tokens == 5
    assert result.prompt_tokens > 0


def test_flan_t5_translates_vietnamese_question_to_english(patched_translate):
    registry = FakeFlanRegistry()
    translator = Translator(registry=registry, use_cache=False)
    backend = FlanT5Backend(registry=registry, translator=translator)

    result = backend.generate([{"role": "user", "content": "Ai đã vẽ bức tranh này?"}])

    # Câu hỏi tiếng Việt được dịch sang EN trước khi sinh câu trả lời.
    assert result.raw["question_en"].startswith("EN: ")
    assert "Ai đã vẽ bức tranh này?" in result.raw["question_en"]


def test_flan_t5_keeps_english_question_untouched(patched_translate):
    registry = FakeFlanRegistry()
    translator = Translator(registry=registry, use_cache=False)
    backend = FlanT5Backend(registry=registry, translator=translator)

    result = backend.generate([{"role": "user", "content": "Who painted this?"}])

    assert result.raw["question_en"] == "Who painted this?"


def test_flan_t5_preserves_citation_in_final_answer(patched_translate):
    """Citation trong câu trả lời phải còn nguyên sau khi dịch EN->VI."""
    registry = FakeFlanRegistry(FakeTokenizer("The painter was Thomas Eakins. [IMG-01]"))
    translator = Translator(registry=registry, use_cache=False)
    backend = FlanT5Backend(registry=registry, translator=translator)

    result = backend.generate([{"role": "user", "content": "Who painted this?"}])

    assert "[IMG-01]" in result.text
    assert result.raw["text_en"].endswith("[IMG-01]")


def test_flan_t5_prompt_ends_with_answer_cue():
    """FLAN-T5 cần đuôi `Answer:` để sinh phần trả lời."""
    prompt = FlanT5Backend._with_answer_cue("ctx\n\nQuestion: Who painted this?")

    assert prompt.endswith("Answer:")

    # Không thêm hai lần nếu đã có sẵn.
    assert FlanT5Backend._with_answer_cue(prompt) == prompt
    assert FlanT5Backend._with_answer_cue("") == "Answer:"


def test_flan_t5_instruction_mentions_no_citation():
    """Chỉ dẫn lượt 1 không được nhắc trích nhãn (FLAN-T5 sẽ chỉ phát nhãn)."""
    from artrag.llm.flan_t5 import DEFAULT_INSTRUCTION

    assert "label" not in DEFAULT_INSTRUCTION.lower()
    assert "cite" not in DEFAULT_INSTRUCTION.lower()


def test_flan_t5_second_pass_adds_evidence_label(patched_translate):
    """Lượt 2 chọn nhãn bằng chứng và chèn nhãn đó vào câu trả lời."""
    # Lượt 1 -> câu trả lời (không nhãn); lượt 2 -> nhãn bằng chứng.
    registry = FakeFlanRegistry(FakeTokenizer(["Thomas Eakins", "[IMG-01]"]))
    translator = Translator(registry=registry, use_cache=False)
    backend = FlanT5Backend(registry=registry, translator=translator)

    result = backend.generate(
        [
            {
                "role": "user",
                "content": "[IMG-01] Biglin Brothers - Thomas Eakins, 1873\n"
                "[IMG-02] Rowing - unknown artist, 1900\n\n"
                "Question: Who painted the Biglin Brothers?",
            }
        ]
    )

    assert result.raw["labels"] == ["[IMG-01]"]
    assert result.raw["text_en"].endswith("[IMG-01]")
    assert "[IMG-01]" in result.text


def test_flan_t5_second_pass_ignores_label_not_in_context(patched_translate):
    """Nhãn không có trong context không được chèn (tránh bịa citation)."""
    registry = FakeFlanRegistry(FakeTokenizer(["Thomas Eakins", "[IMG-07]"]))
    translator = Translator(registry=registry, use_cache=False)
    backend = FlanT5Backend(registry=registry, translator=translator)

    result = backend.generate(
        [
            {
                "role": "user",
                "content": "[IMG-01] Biglin Brothers - Thomas Eakins, 1873\n\n"
                "Question: Who painted the Biglin Brothers?",
            }
        ]
    )

    assert result.raw["labels"] == []
    assert "[IMG-07]" not in result.text


def test_flan_t5_second_pass_can_be_disabled(patched_translate):
    """Đặt `label_instruction` rỗng -> không chạy lượt 2."""
    registry = FakeFlanRegistry(FakeTokenizer(["Thomas Eakins", "[IMG-01]"]))
    translator = Translator(registry=registry, use_cache=False)
    backend = FlanT5Backend(
        registry=registry, translator=translator, label_instruction=""
    )

    result = backend.generate(
        [
            {
                "role": "user",
                "content": "[IMG-01] Biglin Brothers - Thomas Eakins, 1873\n\n"
                "Question: Who painted the Biglin Brothers?",
            }
        ]
    )

    assert result.raw["labels"] == []
    assert "[IMG-01]" not in result.text


def test_flan_t5_second_pass_skipped_without_citation_in_context(patched_translate):
    """Context không có nhãn nào -> bỏ qua lượt 2 (không tốn thêm lượt sinh)."""
    registry = FakeFlanRegistry(FakeTokenizer(["Thomas Eakins", "[IMG-01]"]))
    translator = Translator(registry=registry, use_cache=False)
    backend = FlanT5Backend(registry=registry, translator=translator)

    result = backend.generate([{"role": "user", "content": "Who painted this?"}])

    assert result.raw["labels"] == []
    # Chỉ một lần gọi `generate` (lượt 1).
    assert len(registry.models["flan_t5"].calls) == 1


def test_flan_t5_encoder_decoder_output_is_sliced_correctly(patched_translate):
    """Với model encoder-decoder, output phải cắt từ vị trí 1 (bỏ decoder_start).

    Nếu cắt theo độ dài prompt thì token sinh đầu tiên bị mất -> prompt ngắn
    cho ra chuỗi rỗng.
    """
    registry = FakeFlanRegistry(FakeTokenizer(["Thomas Eakins"]))
    translator = Translator(registry=registry, use_cache=False)
    backend = FlanT5Backend(registry=registry, translator=translator)

    result = backend.generate([{"role": "user", "content": "hi"}])

    # Model giả sinh 5 token -> completion_tokens = 5 (không phải 0).
    assert result.completion_tokens == 5


def test_flan_t5_generate_uses_greedy_when_temperature_zero(patched_translate):
    registry = FakeFlanRegistry()
    translator = Translator(registry=registry, use_cache=False)
    backend = FlanT5Backend(registry=registry, translator=translator)

    backend.generate([{"role": "user", "content": "hi"}], temperature=0.0)

    model = registry.models["flan_t5"]
    assert model.calls[0]["do_sample"] is False
    assert model.calls[0].get("temperature") is None


def test_flan_t5_count_tokens_uses_tokenizer(patched_translate):
    registry = FakeFlanRegistry()
    translator = Translator(registry=registry, use_cache=False)
    backend = FlanT5Backend(registry=registry, translator=translator)

    backend.generate([{"role": "user", "content": "hai ba"}])

    # Tokenizer giả encode theo số từ -> 2 token.
    assert backend.count_tokens("hai ba") == 2


def test_build_flan_t5_reads_model_names_from_config():
    cfg = Config({"llm": {"gen": {"max_new_tokens": 64}}})
    registry = FakeFlanRegistry()

    backend = build_flan_t5(
        cfg,
        node={"model": "flan_t5", "en2vi_model": "en2vi", "vi2en_model": "vi2en"},
        registry=registry,
    )

    assert isinstance(backend, FlanT5Backend)
    assert backend.model == "flan_t5"
    assert backend.en2vi_model == "en2vi"
    assert backend.vi2en_model == "vi2en"
    assert backend.default_gen["max_new_tokens"] == 64


def test_build_flan_t5_defaults():
    cfg = Config({"llm": {}})
    registry = FakeFlanRegistry()

    backend = build_flan_t5(cfg, node=None, registry=registry)

    assert backend.model == "flan_t5"
    assert backend.en2vi_model == "en2vi"
    assert backend.vi2en_model == "vi2en"


def test_get_backend_builds_flan_t5():
    cfg = Config(
        {
            "llm": {"backend": "flan_t5"},
            "backends": {"flan_t5": {"type": "flan_t5", "model": "flan_t5"}},
        }
    )

    backend = get_backend(cfg, use_cache=False)

    assert isinstance(backend, FlanT5Backend)


def test_flan_t5_is_not_wrapped_in_retry_backend():
    """Lỗi cục bộ không hết khi thử lại -> không được bọc retry."""
    from artrag.llm.retry import RetryBackend

    cfg = Config(
        {
            "llm": {"backend": "flan_t5"},
            "backends": {"flan_t5": {"type": "flan_t5"}},
        }
    )

    backend = get_backend(cfg, use_cache=False)

    layers = []
    current = backend
    while current is not None:
        layers.append(type(current).__name__)
        current = getattr(current, "inner", None)

    assert "RetryBackend" not in layers


def test_flan_t5_is_wrapped_in_cache():
    cfg = Config(
        {
            "llm": {"backend": "flan_t5"},
            "backends": {"flan_t5": {"type": "flan_t5"}},
        }
    )

    backend = get_backend(cfg, use_cache=True)

    inner = getattr(backend, "inner", None)
    assert isinstance(inner, FlanT5Backend)
