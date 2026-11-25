"""Test T1.2: parse ban ghi CMA thanh schema artworks (SYSTEM_DESIGN 2.3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from artrag.data.cma_parse import (
    ARTWORK_COLUMNS,
    exhibition_items,
    exhibitions_text,
    overview_text,
    parse_artist_description,
    parse_artwork,
    prev_owner,
    provenance_text,
)

FIXTURE = Path(__file__).parent / "fixtures" / "cma_sample.json"


@pytest.fixture(scope="module")
def records() -> list[dict]:
    with FIXTURE.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def eakins(records) -> dict:
    return parse_artwork(records[0])


@pytest.fixture(scope="module")
def indian(records) -> dict:
    # ban ghi thu 2: tranh India, khong co provenance va khong co creator
    return parse_artwork(records[1])


@pytest.fixture(scope="module")
def chinese(records) -> dict:
    # ban ghi thu 3: tranh Trung Quoc, co creator
    return parse_artwork(records[2])


# ---------------------------------------------------------------- fixture
def test_fixture_has_three_real_records(records):
    assert len(records) == 3
    ids = {r["id"] for r in records}
    assert 151904 in ids  # Eakins
    assert all(r.get("title") for r in records)


# ---------------------------------------------------------------- Eakins
def test_eakins_identity(eakins):
    assert eakins["artwork_id"] == "cma-151904"
    assert eakins["cma_id"] == 151904
    assert eakins["accession_number"] == "1927.1984"
    assert eakins["title"] == "The Biglin Brothers Turning the Stake"
    assert eakins["artist_id"] == "cma-artist-4037"
    assert eakins["artist_role"] == "artist"


def test_eakins_artist_fields(eakins):
    assert eakins["artist_name"] == "Thomas Eakins"
    assert eakins["artist_nationality"] == "American"
    assert eakins["artist_birth"] == 1844
    assert eakins["artist_death"] == 1916


def test_eakins_dates(eakins):
    assert eakins["creation_date"] == "1873"
    assert eakins["date_begin"] == 1873
    assert eakins["date_end"] == 1873


def test_eakins_prev_owner(eakins):
    assert eakins["prev_owner"] is not None
    assert "Susan MacDowell Eakins" in eakins["prev_owner"]


def test_eakins_html_is_stripped(eakins):
    assert "<" not in eakins["description"]
    assert ">" not in eakins["description"]
    assert eakins["description"].startswith("Eakins's painting celebrates")


def test_eakins_overview_text(eakins):
    overview = overview_text(eakins)
    assert "Title: The Biglin Brothers Turning the Stake" in overview
    assert "Artist: Thomas Eakins (American, 1844–1916)" in overview
    assert "Date: 1873" in overview
    assert "Medium: oil on canvas" in overview
    assert "Location: 207 American Realism (on view)" in overview
    assert "Credit line: Hinman B. Hurlbut Collection" in overview


def test_eakins_provenance_text(eakins):
    text = eakins["provenance_text"]
    assert "1916-1927" in text
    assert "Susan MacDowell Eakins" in text
    assert text.count(";") >= 1


def test_eakins_source_and_image(eakins):
    assert eakins["source_url"] == "https://clevelandart.org/art/1927.1984"
    assert eakins["image_url"].endswith(".jpg")
    assert eakins["image_path"] is None


# ---------------------------------------------------------------- bien the
def test_record_without_provenance(indian):
    assert indian["provenance_text"] is None
    assert indian["prev_owner"] is None


def test_asian_painting_is_null_safe(indian):
    # khong co creator -> khong duoc nem loi
    assert indian["artist_name"] is None
    assert indian["artist_id"] is None
    assert indian["current_location"] is None
    assert indian["department"] == "Indian and Southeast Asian Art"
    assert indian["culture"]
    assert indian["artwork_id"].startswith("cma-")


def test_chinese_painting_with_artist(chinese):
    # ban ghi thu 3: tranh Trung Quoc, co creator
    assert chinese["artist_name"] == "Mi Youren"
    assert chinese["artist_nationality"] == "Chinese"
    assert chinese["artist_birth"] == 1072
    assert chinese["artist_death"] == 1151
    assert chinese["department"] == "Chinese Art"


def test_exhibitions_text_capped_and_stripped(eakins, records):
    text = eakins["exhibitions_text"]
    assert text
    assert "<i>" not in text
    assert "</i>" not in text
    # so muc gioi han 10 (mo ta trien lam co the chua dau phay)
    assert len(exhibition_items(records[0]["exhibitions"])) <= 10


def test_exhibitions_items_sorted_newest_first():
    items = exhibition_items(
        {
            "current": [
                {"title": "Old", "description": "d", "opening_date": "1900-01-01T00:00:00"},
                {"title": "New", "description": "d", "opening_date": "2010-01-01T00:00:00"},
                {"title": "Mid", "description": "d", "opening_date": "2005-01-01T00:00:00"},
            ]
        }
    )
    assert [item["title"] for item in items] == ["New", "Mid", "Old"]


# ---------------------------------------------------------------- helpers
def test_prev_owner_none_when_cma_is_first_entry():
    entries = [{"date": "1927-", "description": "The Cleveland Museum of Art, Cleveland, OH"}]
    assert prev_owner(entries) is None


def test_prev_owner_none_for_empty_input():
    assert prev_owner([]) is None
    assert prev_owner(None) is None


def test_prev_owner_takes_entry_before_last_cma_entry():
    entries = [
        {"date": "1916-1927", "description": "Susan MacDowell Eakins [1851-1938], Philadelphia"},
        {"date": "1927-", "description": "The Cleveland Museum of Art, Cleveland, OH"},
    ]
    assert prev_owner(entries) == "Susan MacDowell Eakins"


def test_prev_owner_uses_last_cma_entry():
    entries = [
        {"date": "1915", "description": "(Copley Gallery, Boston, MA, sold to the Cleveland Museum of Art)"},
        {"date": "1915-", "description": "The Cleveland Museum of Art, Cleveland, Ohio"},
    ]
    assert prev_owner(entries) == "Copley Gallery"


def test_prev_owner_strips_bracket_part():
    entries = [
        {"date": "1916-1927", "description": "Walter Wesselhoeft [1838-1920], Cambridge, MA"},
        {"date": "1916-", "description": "The Cleveland Museum of Art, Cleveland, OH"},
    ]
    assert prev_owner(entries) == "Walter Wesselhoeft"


def test_provenance_text_joins_entries():
    entries = [
        {"date": "1915", "description": "Sold to the museum"},
        {"date": "1915-", "description": "The Cleveland Museum of Art"},
    ]
    text = provenance_text(entries)
    assert text == "1915: Sold to the museum; 1915-: The Cleveland Museum of Art"


def test_provenance_text_empty_when_missing():
    assert provenance_text([]) == ""
    assert provenance_text(None) == ""


# ---------------------------------------------------------------- artist desc
@pytest.mark.parametrize(
    "description,name,nationality,birth,death",
    [
        ("Thomas Eakins (American, 1844–1916)", "Thomas Eakins", "American", 1844, 1916),
        ("Claude Monet (French, 1840–1926)", "Claude Monet", "French", 1840, 1926),
        ("Vincent van Gogh (Dutch, 1853–1890)", "Vincent van Gogh", "Dutch", 1853, 1890),
        ("John Singleton Copley (American, born The Thirteen Colonies, 1738–1815)",
         "John Singleton Copley", "American", 1738, 1815),
        ("Thomas Cole (American, born England,1801–1848)", "Thomas Cole", "American", 1801, 1848),
    ],
)
def test_parse_artist_description(description, name, nationality, birth, death):
    parsed = parse_artist_description(description)
    assert parsed["name"] == name
    assert parsed["nationality"] == nationality
    assert parsed["birth"] == birth
    assert parsed["death"] == death


def test_parse_artist_description_with_approximate_year():
    parsed = parse_artist_description("Willem Kalf (Dutch, c. 1629–1684)")
    assert parsed["name"] == "Willem Kalf"
    assert parsed["nationality"] == "Dutch"
    # "c." -> khong chắc nam sinh -> birth None, death van lay duoc
    assert parsed["birth"] is None
    assert parsed["death"] == 1684


def test_parse_artist_description_without_parentheses():
    parsed = parse_artist_description("Anonymous")
    assert parsed["name"] == "Anonymous"
    assert parsed["nationality"] is None
    assert parsed["birth"] is None


def test_parse_artist_description_empty():
    parsed = parse_artist_description("")
    assert parsed["name"] is None


# ---------------------------------------------------------------- schema
def test_parse_artwork_has_all_columns(eakins):
    assert set(eakins.keys()) == set(ARTWORK_COLUMNS)


def test_overview_text_skips_missing_fields():
    row = {
        "title": "T",
        "artist_name": None,
        "creation_date": None,
        "technique": None,
        "culture": None,
        "department": None,
        "current_location": None,
        "credit_line": None,
    }
    assert overview_text(row) == "Title: T"


def test_exhibitions_text_handles_missing_input():
    assert exhibitions_text(None) == ""
    assert exhibitions_text({}) == ""
    assert exhibitions_text({"current": []}) == ""
