"""Test cho T1.5 ``artrag.data.artists``.

Tất cả test chạy **không cần mạng, không cần GPU**: chỉ dùng dữ liệu giả.
"""

from __future__ import annotations

import json

from artrag.data.artists import (
    ARTIST_COLUMNS,
    BIO_SOURCE_CMA,
    BIO_SOURCE_NONE,
    BIO_SOURCE_WIKIPEDIA,
    build_artist_rows,
    chunk_rows_for_artists,
    collect_artists,
    load_wiki_records,
    make_artist_bio_chunks,
)


def _artwork_row(**kwargs):
    row = {
        "artwork_id": "cma-1",
        "artist_id": "cma-artist-1",
        "artist_name": "Test Artist",
        "artist_birth": 1800,
        "artist_death": 1880,
        "artist_nationality": "American",
        "source_url": "https://clevelandart.org/art/1",
    }
    row.update(kwargs)
    return row


# ---------------------------------------------------------------- collect
def test_collect_artists_dedup_by_artist_id():
    rows = [
        _artwork_row(artwork_id="cma-1"),
        _artwork_row(artwork_id="cma-2"),
    ]
    artists = collect_artists(rows)
    assert len(artists) == 1
    assert artists[0]["artist_id"] == "cma-artist-1"
    assert artists[0]["name"] == "Test Artist"
    assert artists[0]["birth"] == 1800
    assert artists[0]["death"] == 1880
    assert artists[0]["nationality"] == "American"


def test_collect_artists_same_name_different_id():
    """Hai người trùng tên nhưng khác artist_id -> 2 dòng."""
    rows = [
        _artwork_row(artwork_id="cma-1", artist_id="cma-artist-1", artist_name="Wang Yuan"),
        _artwork_row(artwork_id="cma-2", artist_id="cma-artist-2", artist_name="Wang Yuan"),
    ]
    artists = collect_artists(rows)
    assert len(artists) == 2
    assert {a["artist_id"] for a in artists} == {"cma-artist-1", "cma-artist-2"}


def test_collect_artists_skips_null_artist_id():
    rows = [_artwork_row(artist_id=None), _artwork_row(artist_id="")]
    assert collect_artists(rows) == []


# ---------------------------------------------------------------- wiki cache
def test_load_wiki_records_reads_and_skips_broken(tmp_path):
    wiki_dir = tmp_path / "raw" / "wiki"
    wiki_dir.mkdir(parents=True)
    (wiki_dir / "cma-artist-1.json").write_text(
        json.dumps({"artist_id": "cma-artist-1", "status": "found"}),
        encoding="utf-8",
    )
    (wiki_dir / "cma-artist-2.json").write_text("{khong phai json", encoding="utf-8")
    (wiki_dir / "cma-artist-3.json").write_text(
        json.dumps({"no_artist_id": True}), encoding="utf-8"
    )

    records = load_wiki_records(tmp_path / "raw")
    assert set(records) == {"cma-artist-1"}


def test_load_wiki_records_missing_dir(tmp_path):
    assert load_wiki_records(tmp_path / "raw") == {}


# ---------------------------------------------------------------- build rows
def test_build_artist_rows_prefers_cma_bio():
    rows = [_artwork_row()]
    wiki = {
        "cma-artist-1": {
            "artist_id": "cma-artist-1",
            "wikidata_qid": "Q1",
            "wiki_title": "Test Artist",
            "wiki_url": "https://en.wikipedia.org/wiki/Test_Artist",
            "wiki_summary": "Wikipedia summary long enough to be used as a bio.",
        }
    }
    cma = {"cma-artist-1": "CMA biography that is long enough to be picked first."}

    out = build_artist_rows(rows, wiki_records=wiki, cma_bios=cma)
    assert len(out) == 1
    artist = out[0]
    assert artist["bio_source"] == BIO_SOURCE_CMA
    assert artist["bio"] == cma["cma-artist-1"]
    # thông tin wikipedia vẫn được giữ
    assert artist["wikidata_qid"] == "Q1"
    assert artist["wiki_url"].startswith("https://en.wikipedia.org/")


def test_build_artist_rows_fallback_to_wikipedia():
    rows = [_artwork_row()]
    wiki = {
        "cma-artist-1": {
            "artist_id": "cma-artist-1",
            "wikidata_qid": "Q1",
            "wiki_title": "Test Artist",
            "wiki_url": "https://en.wikipedia.org/wiki/Test_Artist",
            "wiki_summary": "Wikipedia summary long enough to be used as a bio.",
        }
    }
    out = build_artist_rows(rows, wiki_records=wiki, cma_bios={})
    artist = out[0]
    assert artist["bio_source"] == BIO_SOURCE_WIKIPEDIA
    assert artist["bio"].startswith("Wikipedia summary")


def test_build_artist_rows_none_when_no_bio():
    rows = [_artwork_row()]
    out = build_artist_rows(rows, wiki_records={}, cma_bios={})
    artist = out[0]
    assert artist["bio_source"] == BIO_SOURCE_NONE
    assert artist["bio"] is None


def test_build_artist_rows_short_bio_falls_back():
    """Bio quá ngắn bị bỏ qua -> dùng nguồn kế tiếp."""
    rows = [_artwork_row()]
    wiki = {
        "cma-artist-1": {
            "artist_id": "cma-artist-1",
            "wiki_url": "https://en.wikipedia.org/wiki/Test_Artist",
            "wiki_summary": "Short.",
        }
    }
    cma = {"cma-artist-1": "Tiny"}
    out = build_artist_rows(rows, wiki_records=wiki, cma_bios=cma, min_chars=40)
    assert out[0]["bio_source"] == BIO_SOURCE_NONE
    assert out[0]["bio"] is None


def test_build_artist_rows_prefer_cma_false_uses_wikipedia_first():
    rows = [_artwork_row()]
    wiki = {
        "cma-artist-1": {
            "artist_id": "cma-artist-1",
            "wiki_url": "https://en.wikipedia.org/wiki/Test_Artist",
            "wiki_summary": "Wikipedia summary long enough to be used as a bio.",
        }
    }
    cma = {"cma-artist-1": "CMA biography that is long enough to be picked first."}
    out = build_artist_rows(rows, wiki_records=wiki, cma_bios=cma, prefer_cma=False)
    assert out[0]["bio_source"] == BIO_SOURCE_WIKIPEDIA


def test_build_artist_rows_columns():
    rows = [_artwork_row()]
    out = build_artist_rows(rows, wiki_records={}, cma_bios={})
    assert list(out[0].keys()) == ARTIST_COLUMNS


# ---------------------------------------------------------------- chunks
def test_make_artist_bio_chunks_shape():
    artists = [
        {
            "artist_id": "cma-artist-1",
            "bio": "First sentence about the artist. Second sentence about the artist.",
            "bio_source": BIO_SOURCE_WIKIPEDIA,
            "wiki_url": "https://en.wikipedia.org/wiki/Test_Artist",
        }
    ]
    chunks = make_artist_bio_chunks(artists)
    assert chunks
    for index, chunk in enumerate(chunks):
        assert chunk["chunk_id"] == f"cma-artist-1#artist_bio#{index}"
        assert chunk["artwork_id"] is None
        assert chunk["artist_id"] == "cma-artist-1"
        assert chunk["field"] == "artist_bio"
        assert chunk["text"].strip()
        assert chunk["source_url"] == "https://en.wikipedia.org/wiki/Test_Artist"


def test_make_artist_bio_chunks_cma_uses_artwork_url():
    artists = [
        {
            "artist_id": "cma-artist-1",
            "bio": "A CMA biography that is definitely long enough to be chunked.",
            "bio_source": BIO_SOURCE_CMA,
            "wiki_url": None,
        }
    ]
    chunks = make_artist_bio_chunks(
        artists, source_url_by_artist={"cma-artist-1": "https://clevelandart.org/art/1"}
    )
    assert chunks[0]["source_url"] == "https://clevelandart.org/art/1"


def test_make_artist_bio_chunks_respects_max_words():
    long_text = " ".join(f"word{i}" for i in range(900))
    artists = [
        {
            "artist_id": "cma-artist-1",
            "bio": long_text,
            "bio_source": BIO_SOURCE_CMA,
            "wiki_url": None,
        }
    ]
    chunks = make_artist_bio_chunks(
        artists, sentence_max_words=180, max_words_per_chunk=250
    )
    assert len(chunks) > 1
    assert all(chunk["n_words"] <= 250 for chunk in chunks)


def test_make_artist_bio_chunks_skips_no_bio():
    artists = [
        {"artist_id": "cma-artist-1", "bio": None, "bio_source": BIO_SOURCE_NONE},
        {"artist_id": "cma-artist-2", "bio": "  ", "bio_source": BIO_SOURCE_NONE},
    ]
    assert make_artist_bio_chunks(artists) == []


def test_chunk_rows_for_artists_has_chunk_columns():
    from artrag.data.chunking import CHUNK_COLUMNS

    artists = [
        {
            "artist_id": "cma-artist-1",
            "bio": "Some bio text long enough.",
            "bio_source": BIO_SOURCE_CMA,
            "wiki_url": None,
        }
    ]
    rows = chunk_rows_for_artists(artists)
    assert rows
    assert list(rows[0].keys()) == CHUNK_COLUMNS
