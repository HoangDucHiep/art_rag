"""Test cho các chiến lược chunking và proxy metrics (T1.5 - ablation chunking).

Mọi hàm ở đây đều thuần: không cần GPU, model hay dữ liệu thật.
"""

from __future__ import annotations

import pytest

from artrag.data.chunk_strategies import (
    STRATEGY_ITEM,
    STRATEGY_SEMANTIC,
    STRATEGY_SENTENCE,
    apply_strategy,
    chunk_items,
    chunk_semantic,
    chunk_sentences,
    make_chunk_rows_for_strategy,
    split_paragraphs,
)
from artrag.eval.chunk_eval import (
    chunk_profile,
    info_coverage,
    mid_item_cut_rate,
    mid_sentence_cut_rate,
    redundancy,
    verbatim_rate,
)


# ------------------------------------------------------------- split_paragraphs
def test_split_paragraphs_splits_on_blank_line():
    text = "Đoạn một.\n\nĐoạn hai.\n\n\nĐoạn ba."
    assert split_paragraphs(text) == ["Đoạn một.", "Đoạn hai.", "Đoạn ba."]


def test_split_paragraphs_without_blank_line_returns_single_block():
    assert split_paragraphs("một dòng duy nhất") == ["một dòng duy nhất"]
    assert split_paragraphs("") == []


# ------------------------------------------------------------- chunk_items (C2)
def test_chunk_items_never_cuts_inside_an_item():
    """Mỗi mục tự nhiên phải nằm trọn trong một chunk."""
    text = "A: first item; B: second item; C: third item"
    chunks = chunk_items(text, max_words=250)
    assert len(chunks) == 1
    assert "A: first item" in chunks[0]
    assert "B: second item" in chunks[0]


def test_chunk_items_groups_until_max_words():
    text = "; ".join(f"item {index} with some words here" for index in range(20))
    chunks = chunk_items(text, max_words=20, separator="; ")
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.split()) <= 20
        # mọi mục phải trọn vẹn
        for item in chunk.split("; "):
            assert item in text


def test_chunk_items_preserves_text_verbatim():
    text = "A: one; B: two; C: three"
    rebuilt = "; ".join(chunk_items(text))
    assert rebuilt == text


def test_chunk_items_keeps_separator_at_chunk_boundary():
    """Dấu phân cách ở ranh giới giữa hai chunk không được bị mất.

    Nối lại các chunk bằng " " phải khớp đúng văn bản gốc - nếu mất dấu
    ";" thì nội dung của văn bản nguồn bị thay đổi.
    """
    text = "; ".join(f"item {index} with several words to fill" for index in range(20))
    chunks = chunk_items(text, max_words=20, separator="; ")
    assert len(chunks) > 1
    rebuilt = " ".join(chunks)
    assert " ".join(rebuilt.split()) == " ".join(text.split())
    # dấu ";" vẫn còn đủ số lần như văn bản gốc
    assert rebuilt.count(";") == text.count(";")


def test_chunk_items_keeps_separator_when_item_too_long():
    """Mục đơn lẻ quá dài bị cắt cứng vẫn phải bảo toàn nguyên văn."""
    text = "; ".join(["short one", "w " * 300, "short two"])
    chunks = chunk_items(text, max_words=250, separator="; ")
    assert len(chunks) > 1
    rebuilt = " ".join(chunks)
    assert " ".join(rebuilt.split()) == " ".join(text.split())


def test_chunk_items_handles_trailing_separator():
    """Văn bản kết thúc bằng dấu phân cách vẫn phải bảo toàn."""
    text = "A: one; B: two;"
    chunks = chunk_items(text, max_words=250)
    assert " ".join(" ".join(chunks).split()) == " ".join(text.split())


# --------------------------------------------------------- chunk_semantic (C3)
def test_chunk_semantic_groups_paragraphs():
    text = ("Para one has five words.\n\n" "Para two has four words.\n\n" "Para three has three words.")
    chunks = chunk_semantic(text, target_words=20, max_words=60, min_words=0)
    assert len(chunks) == 1
    assert "Para one" in chunks[0]
    assert "Para three" in chunks[0]


def test_chunk_semantic_never_cuts_inside_a_paragraph():
    text = "\n\n".join(f"Doan van so {index} co vai tu" for index in range(6))
    chunks = chunk_semantic(text, target_words=15, max_words=40, min_words=0)
    for chunk in chunks:
        for paragraph in chunk.split("\n\n"):
            assert paragraph in text


def test_chunk_semantic_has_no_overlap():
    """C3 không overlap -> không có đoạn n-gram trùng lặp giữa chunk liền kề."""
    paragraphs = [
        "Alpha beta gamma delta epsilon zeta.",
        "Kappa lambda mu nu xi omicron pi.",
        "Rho sigma tau upsilon phi chi psi.",
        "Omega alpha beta gamma delta epsilon.",
    ]
    text = "\n\n".join(paragraphs)
    chunks = chunk_semantic(text, target_words=8, max_words=20, min_words=0)
    assert len(chunks) > 1
    for previous, current in zip(chunks, chunks[1:]):
        prev_grams = {
            tuple(previous.split()[i : i + 3])
            for i in range(len(previous.split()) - 2)
        }
        cur_grams = {
            tuple(current.split()[i : i + 3])
            for i in range(len(current.split()) - 2)
        }
        assert not (prev_grams & cur_grams)


def test_chunk_semantic_preserves_text_verbatim():
    text = "Doan mot.\n\nDoan hai.\n\nDoan ba."
    rebuilt = "\n\n".join(chunk_semantic(text, target_words=5, max_words=20, min_words=0))
    assert rebuilt == text


def test_chunk_semantic_splits_long_paragraph_by_sentence():
    text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here."
    chunks = chunk_semantic(text, target_words=8, max_words=10, min_words=0)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk in text


# ------------------------------------------------------------- interface chung
@pytest.mark.parametrize(
    "strategy",
    [STRATEGY_SENTENCE, STRATEGY_ITEM, STRATEGY_SEMANTIC],
)
def test_apply_strategy_dispatches(strategy):
    text = "First item here; Second item here."
    chunks = apply_strategy(text, strategy=strategy)
    assert chunks
    assert all(chunk.strip() for chunk in chunks)


def test_apply_strategy_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        apply_strategy("abc", strategy="khong_hop_le")


def test_make_chunk_rows_for_strategy_sets_metadata():
    row = {
        "artwork_id": "cma-1",
        "artist_id": "cma-artist-1",
        "source_url": "http://example.org",
    }
    rows = make_chunk_rows_for_strategy(
        row,
        strategy=STRATEGY_SENTENCE,
        field="description",
        text="One sentence. Two sentence.",
    )
    assert rows
    for chunk in rows:
        assert chunk["artwork_id"] == "cma-1"
        assert chunk["artist_id"] == "cma-artist-1"
        assert chunk["field"] == "description"
        assert chunk["strategy"] == STRATEGY_SENTENCE
        assert chunk["chunk_id"].startswith("cma-1#description#")
        assert chunk["n_words"] == len(chunk["text"].split())


# ------------------------------------------------------------- proxy metrics
def test_chunk_profile_basic_stats():
    chunks = [
        {"field": "a", "text": "one two three", "n_words": 3},
        {"field": "a", "text": "four five", "n_words": 2},
        {"field": "b", "text": "six", "n_words": 1},
    ]
    profile = chunk_profile(chunks)
    assert profile["n_chunks"] == 3
    assert profile["avg_words"] == 2.0
    assert profile["min_words"] == 1
    assert profile["max_words"] == 3
    assert profile["by_field"] == {"a": 2, "b": 1}
    assert profile["n_empty"] == 0
    assert profile["n_over_max"] == 0


def test_chunk_profile_counts_over_max_and_empty():
    chunks = [
        {"field": "a", "text": " ", "n_words": 0},
        {"field": "a", "text": "w " * 300, "n_words": 300},
    ]
    profile = chunk_profile(chunks)
    assert profile["n_empty"] == 1
    assert profile["n_over_max"] == 1
    assert profile["pct_over_max"] == 50.0


def test_mid_item_cut_rate_detects_cut_inside_item():
    source = "A: first item; B: second item; C: third item"
    # chunk bị cắt giữa mục "B: second item"
    cut = [{"text": "A: first item; B: second it"}, {"text": "em; C: third item"}]
    assert mid_item_cut_rate(cut, source) > 0


def test_mid_item_cut_rate_zero_when_items_are_whole():
    source = "A: first item; B: second item; C: third item"
    whole = [{"text": "A: first item; B: second item"}, {"text": "C: third item"}]
    assert mid_item_cut_rate(whole, source) == 0.0


def test_mid_sentence_cut_rate():
    chunks = [
        {"text": "A complete sentence."},
        {"text": "A broken sentence without punctuation"},
        {"text": "Another complete one!"},
    ]
    rate = mid_sentence_cut_rate(chunks)
    assert 0 < rate < 100


def test_redundancy_detects_overlap():
    overlapping = [
        {"artwork_id": "a", "field": "f", "text": "one two three four five six"},
        {"artwork_id": "a", "field": "f", "text": "three four five six seven eight"},
    ]
    assert redundancy(overlapping) > 0


def test_redundancy_zero_without_overlap():
    disjoint = [
        {"artwork_id": "a", "field": "f", "text": "one two three four five"},
        {"artwork_id": "a", "field": "f", "text": "six seven eight nine ten"},
    ]
    assert redundancy(disjoint) == 0.0


def test_info_coverage_full_and_partial():
    source = "one two three four five"
    full = [{"text": source}]
    assert info_coverage(full, source) == 100.0
    partial = [{"text": "one two"}]
    assert info_coverage(partial, source) == 40.0


def test_verbatim_rate_detects_mangled_text():
    source = "Millet (q.v.), the painter of peasant life."
    good = [{"text": "Millet (q.v.), the painter of peasant life."}]
    bad = [{"text": "Millet (q. v.), the painter of peasant life."}]
    assert verbatim_rate(good, source) == 100.0
    assert verbatim_rate(bad, source) == 0.0


def test_chunk_sentences_strategy_keeps_overlap():
    chunks = chunk_sentences(
        "aaa bbb ccc. ddd eee fff. ggg hhh iii.",
        max_words=5,
        overlap_sentences=1,
    )
    assert len(chunks) > 1
    # chunk sau phai chua lai mot cau cua chunk truoc
    assert "ccc." in chunks[1]
