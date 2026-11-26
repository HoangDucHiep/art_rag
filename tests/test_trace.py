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
    assert split_sentences("A. B! C?") == ["A.", "B!", "C?"]
    assert split_sentences("") == []
    assert len(split_sentences("Mot. Hai! Ba?")) == 3


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
