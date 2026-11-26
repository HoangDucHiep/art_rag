import json

import pytest

from artrag.trace import Trace
from artrag.utils.hashing import sha256_json, sha256_text
from artrag.utils.io import read_jsonl, write_jsonl
from artrag.utils.text import normalize_text, split_sentences, strip_html


def test_step_ok_records_status_and_timing():
    trace = Trace()
    with trace.step("retrieve_clip") as step:
        step.set_info(top1_raw=0.29)
    assert len(trace.steps) == 1
    entry = trace.steps[0]
    assert entry["name"] == "retrieve_clip"
    assert entry["status"] == "ok"
    assert entry["ms"] >= 0
    assert entry["info"]["top1_raw"] == 0.29


def test_optional_step_failure_is_degraded_and_does_not_raise():
    trace = Trace()
    with trace.step("rerank_itm", required=False) as step:
        step.set_info(partial=True)
        raise RuntimeError("OOM -> skipped")
    entry = trace.steps[0]
    assert entry["status"] == "degraded"
    assert "OOM" in entry["info"]["error"]
    # bước tiếp theo vẫn chạy được
    with trace.step("generate") as step:
        step.set_info(cached=True)
    assert trace.steps[1]["status"] == "ok"


def test_required_step_failure_reraises_and_records_error():
    trace = Trace()
    with pytest.raises(RuntimeError):
        with trace.step("generate", required=True):
            raise RuntimeError("backend down")
    entry = trace.steps[0]
    assert entry["status"] == "error"
    assert "backend down" in entry["info"]["error"]
    assert "traceback" in entry["info"]


def test_to_dict_is_json_serializable():
    trace = Trace(config_hash="abc123", profile="local")
    trace.query = {"lang": "vi", "qtype": "ARTIST"}
    trace.models = {"clip": "cuda:fp16"}
    with trace.step("translate") as step:
        step.set_info(cached=True)
    payload = trace.to_dict()
    json.dumps(payload)
    assert payload["config_hash"] == "abc123"
    assert payload["profile"] == "local"
    assert payload["query"]["qtype"] == "ARTIST"
    assert payload["steps"][0]["name"] == "translate"
    assert payload["total_ms"] >= 0
    assert payload["abstain"] == {"gate": None, "post": None}


def test_trace_has_request_id_and_summary():
    trace = Trace()
    assert trace.request_id
    with trace.step("a"):
        pass
    summary = trace.summary()
    assert "a" in summary
    assert "total" in summary


def test_strip_html():
    assert strip_html("<p>X&nbsp;Y</p>") == "X Y"
    assert strip_html("") == ""
    assert strip_html("<div>a</div><div>b</div>") == "a b"


def test_normalize_text():
    assert normalize_text("Bức Tranh Đẹp") == "buc tranh dep"
    assert normalize_text("Nguyễn Văn A.") == "nguyen van a"


def test_split_sentences():
    assert split_sentences("A dog barks. B! C?") == ["A dog barks.", "B!", "C?"]
    assert split_sentences("") == []
    assert len(split_sentences("Mot. Hai! Ba?")) == 3


def test_split_sentences_keeps_abbreviations_intact():
    """Chữ viết tắt KHÔNG được tách thành mảnh vỡ (bug T1.5 phát hiện)."""
    text = (
        "Van Gogh copied works by Millet (q. v.), the painter of peasant life. "
        "He then joined the Academy."
    )
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert sentences[0].endswith("the painter of peasant life.")
    assert "(q. v.)," in sentences[0]
    # không còn mảnh rác "v. )"
    assert not any(s.strip().startswith("v.") for s in sentences)

    assert len(split_sentences("J. M. W. Turner was a painter. He lived in London.")) == 2
    assert len(split_sentences("He was born in the U.S.A. Then he moved.")) == 2
    assert len(split_sentences("Dr. Smith went home. He slept.")) == 2


def test_split_sentences_no_content_is_lost():
    """Ghép lại các câu phải cho lại toàn bộ nội dung (sau khi bỏ khoảng trắng)."""
    text = (
        "Van Gogh copied works by Millet (q. v.), the painter of peasant life. "
        "He then joined the Academy in 1886."
    )
    joined = " ".join(split_sentences(text))
    assert joined == text


def test_split_sentences_handles_unspaced_dotted_abbreviations():
    """Dữ liệu thật viết ``(q.v.)`` không có khoảng trắng - không được cắt vỡ."""
    text = (
        "Van Gogh copied works by Millet (q.v.), the painter of peasant life. "
        "He then joined the Academy in 1886."
    )
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert "(q.v.)," in sentences[0]
    assert not any(s.strip().startswith("v.") for s in sentences)


def test_split_sentences_keeps_latin_abbreviations_intact():
    """``i.e.`` / ``e.g.`` là chữ viết tắt, dấu chấm không phải hết câu."""
    text = "He sold many works, i.e. landscapes. Then he stopped painting."
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert "i.e. landscapes." in sentences[0]


def test_split_sentences_absorbs_closing_quote():
    """Dấu trích dán ngay sau dấu kết thúc thuộc về câu hiện tại."""
    text = 'He called it "The Sower." Then he left Paris.'
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert sentences[0] == 'He called it "The Sower."'


def test_split_sentences_ignores_footnote_marker():
    """Số đánh dấu trích dẫn chân bài (``."1``) không phải bắt đầu câu mới."""
    text = 'Van Gogh painted many works."1 He sold few.2 They were rare.'
    sentences = split_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == 'Van Gogh painted many works."1'
    assert sentences[1] == "He sold few.2"


def test_split_sentences_handles_version_number():
    """``v.15.1`` là số phiên bản, dấu chấm không phải hết câu."""
    text = "See volume v.15.1 for details. The catalogue lists everything."
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert "v.15.1" in sentences[0]


def test_split_sentences_ignores_period_inside_brackets():
    """Dấu kết thúc nằm trong ngoặc không phải hết câu (phiên âm, năm sinh)."""
    text = "Claude Lorrain (French: [klod lɔ.ʁɛ̃]) was a painter. He was called Gellée."
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert "[klod lɔ.ʁɛ̃])" in sentences[0]
    assert sentences[1].startswith("He was called")


def test_split_sentences_ignores_numbered_list_markers():
    """Số đánh dấu mục (``2.``) không phải hết câu."""
    text = "See notes. 2. Reynolds 1996, no. 15.1; Reynolds 1984, no. 17.4."
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert sentences[1].startswith("2. Reynolds")


def test_split_sentences_handles_period_before_comma():
    """``Jansz.,`` / ``etc.,`` -> dấu chấm thuộc về từ, không phải hết câu."""
    text = "His brother Jan Jansz., nephew Jan Jansz. II, and son David Coecke."
    sentences = split_sentences(text)
    assert len(sentences) == 1
    assert sentences[0] == text


def test_split_sentences_handles_ellipsis():
    """Dấu chấm lửng ``. . .`` được gộp vào câu."""
    text = "It was, thus, simplifying the painter's métier, . . . a clear break. He agreed."
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert ". . ." in sentences[0]


def test_split_sentences_matches_original_text_exactly():
    """Bất biến quan trọng: ghép lại các câu phải BẰNG CHÍNH XÁC chuỗi gốc."""
    text = (
        'Van Gogh copied works by Millet (q.v.), the painter of peasant life."1 '
        "He then joined the Academy in 1886. Dr. Smith visited J. M. W. Turner "
        "in the U.S.A. They discussed v.15.1 of the catalogue."
    )
    assert " ".join(split_sentences(text)) == text


def test_sha256_json_is_stable_and_order_insensitive():
    assert sha256_json({"a": 1, "b": 2}) == sha256_json({"b": 2, "a": 1})
    assert sha256_json({"a": 1}) != sha256_json({"a": 2})
    assert sha256_text("abc") == sha256_text("abc")
    assert sha256_text("abc") != sha256_text("abd")


def test_jsonl_roundtrip(tmp_path):
    path = tmp_path / "out" / "rows.jsonl"
    rows = [{"i": 1, "text": "một"}, {"i": 2, "text": "hai"}]
    write_jsonl(path, rows)
    assert read_jsonl(path) == rows
