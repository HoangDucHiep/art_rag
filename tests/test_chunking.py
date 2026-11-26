"""Test T1.3: quy tắc chunking (SYSTEM_DESIGN 2.3)."""

from __future__ import annotations

import pytest

from artrag.data.chunking import (
    CHUNK_COLUMNS,
    chunk_sentences,
    chunk_single,
    count_words,
    make_chunk_rows,
)


def _sentence(text: str) -> str:
    return text


# ---------------------------------------------------------------- helpers
def test_count_words():
    assert count_words("") == 0
    assert count_words("   ") == 0
    assert count_words("one two three") == 3
    assert count_words("mot   hai   ba") == 3


def test_chunk_sentences_empty_input():
    assert chunk_sentences("", 180) == []
    assert chunk_sentences("   ", 180) == []


def test_chunk_sentences_single_chunk_when_short():
    text = "First sentence. Second sentence. Third sentence."
    chunks = chunk_sentences(text, max_words=100, overlap_sentences=1)
    assert len(chunks) == 1
    assert "First sentence." in chunks[0]
    assert "Third sentence." in chunks[0]


def test_chunk_sentences_overlap_one_sentence():
    # mỗi câu 3 từ -> max 5 từ -> mỗi chunk gồm 1 câu, overlap 1 câu
    text = "aaa bbb ccc. ddd eee fff. ggg hhh iii. jjj kkk lll."
    chunks = chunk_sentences(text, max_words=5, overlap_sentences=1)
    assert len(chunks) > 1
    # chunk 2 phải chứa lại câu cuối của chunk 1
    assert chunks[0].endswith("aaa bbb ccc.")
    assert chunks[1].startswith("aaa bbb ccc.")
    assert "ddd eee fff." in chunks[1]


def test_chunk_sentences_no_overlap():
    text = "aaa bbb ccc. ddd eee fff. ggg hhh iii."
    chunks = chunk_sentences(text, max_words=5, overlap_sentences=0)
    assert chunks[0] == "aaa bbb ccc."
    assert chunks[1] == "ddd eee fff."


def test_chunk_sentences_respects_max_words():
    text = " ".join(f"w{i}" for i in range(100)) + "."
    chunks = chunk_sentences(text, max_words=20, overlap_sentences=1)
    assert chunks
    for chunk in chunks:
        assert count_words(chunk) <= 20


def test_chunk_sentences_never_empty():
    text = "One. Two. Three."
    chunks = chunk_sentences(text, max_words=180, overlap_sentences=1)
    assert all(chunk.strip() for chunk in chunks)


def test_chunk_sentences_splits_long_sentence():
    text = " ".join(f"w{i}" for i in range(50)) + "."
    chunks = chunk_sentences(text, max_words=10, overlap_sentences=1)
    assert len(chunks) > 1
    for chunk in chunks:
        assert count_words(chunk) <= 10


def test_chunk_single_returns_one_when_short():
    assert chunk_single("a b c", 250) == ["a b c"]
    assert chunk_single("", 250) == []
    assert chunk_single("   ", 250) == []


def test_chunk_single_splits_by_words():
    text = " ".join(f"w{i}" for i in range(100))
    chunks = chunk_single(text, 30)
    assert len(chunks) == 4
    assert all(count_words(chunk) <= 30 for chunk in chunks)


# ---------------------------------------------------------------- make_chunk_rows
@pytest.fixture
def row() -> dict:
    return {
        "artwork_id": "cma-151904",
        "artist_id": "cma-artist-4037",
        "title": "The Biglin Brothers Turning the Stake",
        "artist_name": "Thomas Eakins",
        "artist_nationality": "American",
        "artist_birth": 1844,
        "artist_death": 1916,
        "creation_date": "1873",
        "technique": "oil on canvas",
        "culture": "United States",
        "department": "American Painting and Sculpture",
        "current_location": "207 American Realism",
        "credit_line": "Hinman B. Hurlbut Collection",
        "source_url": "https://clevelandart.org/art/1927.1984",
        "description": "A description sentence. Another description sentence here.",
        "did_you_know": "Rowing was popular.",
        "provenance_text": "1916-1927: Susan MacDowell Eakins",
        "exhibitions_text": "Inaugural Exhibition (1916)",
    }


def test_make_chunk_rows_has_expected_columns(row):
    chunks = make_chunk_rows(row)
    assert chunks
    for chunk in chunks:
        assert set(chunk.keys()) == set(CHUNK_COLUMNS)


def test_make_chunk_rows_overview_is_single(row):
    chunks = make_chunk_rows(row)
    overview = [c for c in chunks if c["field"] == "overview"]
    assert len(overview) == 1
    assert overview[0]["chunk_id"] == "cma-151904#overview#0"
    assert "Artist: Thomas Eakins (American, 1844–1916)" in overview[0]["text"]
    assert "Date: 1873" in overview[0]["text"]


def test_make_chunk_rows_chunk_id_format(row):
    for chunk in make_chunk_rows(row):
        artwork_id, field, index = chunk["chunk_id"].split("#")
        assert artwork_id == "cma-151904"
        assert field in ("overview", "description", "did_you_know", "provenance", "exhibitions")
        assert index.isdigit()


def test_make_chunk_rows_field_order(row):
    fields = [c["field"] for c in make_chunk_rows(row)]
    # overview luôn đầu, các field theo thứ tự §2.3
    assert fields[0] == "overview"
    assert fields.index("description") < fields.index("did_you_know")
    assert fields.index("did_you_know") < fields.index("provenance")
    assert fields.index("provenance") < fields.index("exhibitions")


def test_make_chunk_rows_n_words_matches_count(row):
    for chunk in make_chunk_rows(row):
        assert chunk["n_words"] == count_words(chunk["text"])


def test_make_chunk_rows_no_empty_chunks(row):
    for chunk in make_chunk_rows(row):
        assert chunk["text"].strip()
        assert chunk["n_words"] > 0


def test_make_chunk_rows_respects_hard_cap(row):
    row["description"] = " ".join(f"w{i}" for i in range(900))
    chunks = make_chunk_rows(row, max_words_per_chunk=250)
    for chunk in chunks:
        assert chunk["n_words"] <= 250


def test_make_chunk_rows_skips_missing_fields(row):
    row["did_you_know"] = None
    row["provenance_text"] = None
    row["exhibitions_text"] = None
    fields = {c["field"] for c in make_chunk_rows(row)}
    assert "overview" in fields
    assert "description" in fields
    assert "did_you_know" not in fields
    assert "provenance" not in fields
    assert "exhibitions" not in fields


def test_make_chunk_rows_description_is_split(row):
    row["description"] = " ".join(f"w{i}" for i in range(500))
    chunks = make_chunk_rows(row, sentence_max_words=100, overlap_sentences=1)
    description_chunks = [c for c in chunks if c["field"] == "description"]
    assert len(description_chunks) > 1


def test_make_chunk_rows_carries_metadata(row):
    chunk = make_chunk_rows(row)[0]
    assert chunk["artwork_id"] == "cma-151904"
    assert chunk["artist_id"] == "cma-artist-4037"
    assert chunk["source_url"] == "https://clevelandart.org/art/1927.1984"


def test_make_chunk_rows_handles_row_without_artist(row):
    row["artist_name"] = None
    row["artist_id"] = None
    chunks = make_chunk_rows(row)
    assert chunks
    assert all(c["artist_id"] is None for c in chunks)
