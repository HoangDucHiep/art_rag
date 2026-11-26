"""Test cho T1.4 ``artrag.data.enrich_wiki``.

Tất cả test đều chạy **không cần mạng**: chỉ dùng hàm thuần hoặc session giả.
"""

from __future__ import annotations

import json

import pytest

from artrag.data.enrich_wiki import (
    STATUS_ERROR,
    STATUS_FOUND,
    STATUS_NOT_FOUND,
    HttpNotFound,
    RateLimiter,
    birth_year_agrees,
    build_extract_query,
    build_record,
    build_sparql_query,
    candidate_name_variants,
    candidate_years,
    entity_years,
    fetch_extracts_batch,
    fetch_enwiki_by_titles,
    fetch_occupations_batch,
    fetch_summary,
    first_birth_year,
    is_artist_description,
    is_artist_qid,
    is_cache_valid,
    is_human,
    label_score,
    load_cma_bios,
    looks_like_disambiguation,
    normalize_name,
    parse_extract_pages,
    parse_sparql_rows,
    pick_batch_candidate,
    pick_candidate,
    qid_from_uri,
    resolve_names_batch,
    search_entities,
    sparql_escape,
    title_from_article_url,
    wiki_cache_path,
    year_from_time,
)


# ---------------------------------------------------------------- helpers
def make_entity(
    qid: str,
    label: str,
    *,
    human: bool = True,
    enwiki: str | None = None,
    birth_years: list[int] | None = None,
) -> dict:
    """Dựng entity Wikidata giả cho test."""
    claims: dict = {}
    if human:
        claims["P31"] = [
            {"mainsnak": {"snaktype": "value", "datavalue": {"value": {"id": "Q5"}}}}
        ]
    if birth_years:
        claims["P569"] = [
            {
                "mainsnak": {
                    "snaktype": "value",
                    "datavalue": {"value": {"time": f"+{year}-01-01T00:00:00Z"}},
                }
            }
            for year in birth_years
        ]
    entity: dict = {
        "id": qid,
        "labels": {"en": {"value": label}},
        "claims": claims,
        "sitelinks": {},
    }
    if enwiki:
        entity["sitelinks"]["enwiki"] = {"site": "enwiki", "title": enwiki}
    return entity


def make_candidate(qid: str, label: str) -> dict:
    return {"id": qid, "label": label, "description": "painter"}


# ---------------------------------------------------------------- names
def test_normalize_name_strips_diacritics_and_case():
    assert normalize_name("Kaihō Yūshō") == "kaiho yusho"
    assert normalize_name("  Thomas   Eakins ") == "thomas eakins"
    assert normalize_name(None) == ""
    assert normalize_name("") == ""


def test_normalize_name_handles_d_with_stroke():
    assert normalize_name("Đông") == "dong"


def test_label_score_exact_and_partial():
    assert label_score("Thomas Eakins", "Thomas Eakins") == 100.0
    assert label_score("Thomas Eakins", "Thomas Eakins Jr.") > 80
    assert label_score("Thomas Eakins", "Someone Else") < 50
    assert label_score("Thomas Eakins", "") == 0.0
    assert label_score("", "Thomas Eakins") == 0.0


def test_label_score_tolerates_name_reordering():
    assert label_score("Kaihō Yūshō", "Yusho Kaiho") == 100.0


def test_label_score_rejects_different_people_with_shared_tokens():
    """Hồi quy: `Shen Zhou` có alias `Zhou Chen` -> phải điểm thấp.

    `token_set_ratio` cho 88.9 vì chung token `zhou` — đủ để qua ngưỡng 88
    và match nhầm họa sĩ khác.
    """
    assert label_score("Zhou Chen", "Shen Zhou") < 88
    assert label_score("Zhou Chen", "Zhou Chen") == 100.0


def test_label_score_handles_initials():
    """`Joseph Mallord William Turner` phải khớp `J. M. W. Turner`."""
    assert label_score("Joseph Mallord William Turner", "J. M. W. Turner") == 100.0


def test_label_score_rejects_prefix_names():
    """`Wang Yuan` và `Wang Yuanqi` là hai họa sĩ khác."""
    assert label_score("Wang Yuan", "Wang Yuanqi") < 88
    assert label_score("Wang Yuan", "Wang Yuan") == 100.0


def test_label_score_handles_curly_apostrophes():
    assert label_score("Kano Tan’yū", "Kanō Tan’yū") == 100.0


def test_label_score_allows_shorter_name_subset():
    """`Agnolo Bronzino` chứa `Bronzino` -> cùng một người."""
    assert label_score("Agnolo Bronzino", "Bronzino") == 100.0


# ---------------------------------------------------------------- entity
def test_year_from_time():
    assert year_from_time("+1844-07-25T00:00:00Z") == 1844
    assert year_from_time("+1916-06-25T00:00:00Z") == 1916
    assert year_from_time("garbage") is None
    assert year_from_time(None) is None
    assert year_from_time("") is None


def test_year_from_time_accepts_sparql_format_without_plus():
    """SPARQL trả ``1844-07-25T00:00:00Z`` (không có dấu ``+``)."""
    assert year_from_time("1844-07-25T00:00:00Z") == 1844
    assert year_from_time("1254-00-00T00:00:00Z") == 1254


def test_entity_years():
    entity = make_entity("Q1", "A", birth_years=[1844, 1850])
    assert entity_years(entity) == [1844, 1850]
    assert entity_years(make_entity("Q2", "B")) == []
    assert entity_years({}) == []


def test_is_human():
    assert is_human(make_entity("Q1", "A")) is True
    assert is_human(make_entity("Q2", "B", human=False)) is False
    assert is_human({}) is False
    assert is_human(None) is False


def test_is_human_requires_exact_q5():
    entity = make_entity("Q1", "A", human=False)
    entity["claims"]["P31"] = [
        {"mainsnak": {"snaktype": "value", "datavalue": {"value": {"id": "Q483501"}}}}
    ]
    assert is_human(entity) is False


# ---------------------------------------------------------------- matching
def test_pick_candidate_rejects_non_human():
    candidates = [make_candidate("Q1", "Thomas Eakins")]
    entities = {"Q1": make_entity("Q1", "Thomas Eakins", human=False, enwiki="Thomas Eakins")}
    assert pick_candidate(candidates, entities, "Thomas Eakins", 1844) is None


def test_pick_candidate_rejects_without_enwiki_sitelink():
    candidates = [make_candidate("Q1", "Thomas Eakins")]
    entities = {"Q1": make_entity("Q1", "Thomas Eakins", enwiki=None)}
    assert pick_candidate(candidates, entities, "Thomas Eakins", 1844) is None


def test_pick_candidate_matches_exact():
    candidates = [make_candidate("Q214905", "Thomas Eakins")]
    entities = {
        "Q214905": make_entity(
            "Q214905", "Thomas Eakins", enwiki="Thomas Eakins", birth_years=[1844]
        )
    }
    best = pick_candidate(candidates, entities, "Thomas Eakins", 1844)
    assert best is not None
    assert best["qid"] == "Q214905"
    assert best["wiki_title"] == "Thomas Eakins"
    assert best["birth_matched"] is True
    assert best["match_score"] == 100.0


def test_pick_candidate_prefers_birth_year_match():
    candidates = [
        make_candidate("Q1", "Sesson Shūkei"),
        make_candidate("Q2", "Sesson Shūkei"),
    ]
    entities = {
        "Q1": make_entity("Q1", "Sesson Shūkei", enwiki="Sesson Shukei", birth_years=[1492]),
        "Q2": make_entity("Q2", "Sesson Shūkei", enwiki="Sesson Shukei (1504)", birth_years=[1504]),
    }
    best = pick_candidate(candidates, entities, "Sesson Shūkei", 1504)
    assert best is not None
    assert best["qid"] == "Q2"
    assert best["birth_matched"] is True


def test_pick_candidate_tolerates_missing_birth_year():
    candidates = [make_candidate("Q1", "Some Unknown Artist")]
    entities = {"Q1": make_entity("Q1", "Some Unknown Artist", enwiki="Some Unknown Artist")}
    best = pick_candidate(candidates, entities, "Some Unknown Artist", None)
    assert best is not None
    assert best["birth_matched"] is False


def test_pick_candidate_accepts_label_below_threshold_when_birth_matches():
    candidates = [make_candidate("Q1", "Jacopo da Sellaio")]
    entities = {
        "Q1": make_entity("Q1", "Jacopo da Sellaio", enwiki="Jacopo da Sellaio", birth_years=[1442])
    }
    best = pick_candidate(candidates, entities, "Jacopo del Sellaio", 1441)
    assert best is not None
    assert best["birth_matched"] is True


def test_pick_candidate_returns_none_when_nothing_qualifies():
    candidates = [make_candidate("Q1", "Totally Different")]
    entities = {"Q1": make_entity("Q1", "Totally Different", enwiki="Totally Different")}
    assert pick_candidate(candidates, entities, "Thomas Eakins", 1844) is None
    assert pick_candidate([], {}, "Thomas Eakins", 1844) is None


def test_pick_candidate_skips_candidate_without_entity():
    candidates = [make_candidate("Q1", "Thomas Eakins")]
    assert pick_candidate(candidates, {}, "Thomas Eakins", 1844) is None


# ---------------------------------------------------------------- aliases
def test_candidate_years_from_description():
    candidate = {"id": "Q1", "label": "X", "description": "English painter (c. 1775-1851)"}
    assert candidate_years(candidate) == [1775, 1851]
    assert candidate_years({"id": "Q1", "label": "X"}) == []
    assert candidate_years(None) == []


def test_candidate_name_variants_includes_match_text_and_aliases():
    candidate = {
        "id": "Q159758",
        "label": "J. M. W. Turner",
        "match": {"type": "alias", "language": "en", "text": "Joseph Mallord William Turner"},
        "aliases": ["Joseph Mallord William Turner"],
    }
    variants = candidate_name_variants(candidate)
    assert "J. M. W. Turner" in variants
    assert "Joseph Mallord William Turner" in variants
    # không trùng lặp
    assert len(variants) == len(set(variants))


def test_pick_candidate_matches_via_alias_when_label_is_initialism():
    """Turner: nhãn canonical là ``J. M. W. Turner`` nhưng CMA gọi tên đầy đủ."""
    candidate = {
        "id": "Q159758",
        "label": "J. M. W. Turner",
        "match": {"type": "alias", "language": "en", "text": "Joseph Mallord William Turner"},
        "aliases": ["Joseph Mallord William Turner"],
    }
    entities = {
        "Q159758": make_entity(
            "Q159758", "J. M. W. Turner", enwiki="J. M. W. Turner", birth_years=[1775]
        )
    }
    best = pick_candidate([candidate], entities, "Joseph Mallord William Turner", 1775)
    assert best is not None
    assert best["qid"] == "Q159758"
    assert best["wiki_title"] == "J. M. W. Turner"
    assert best["match_score"] == 100.0
    assert best["birth_matched"] is True


def test_pick_candidate_falls_back_to_description_years():
    """Khi entity không có P569, năm trong description của search result vẫn dùng."""
    candidate = {
        "id": "Q1",
        "label": "Some Painter",
        "description": "Dutch painter (1882-1925)",
    }
    entities = {"Q1": make_entity("Q1", "Some Painter", enwiki="Some Painter")}
    best = pick_candidate([candidate], entities, "Some Painter", 1882)
    assert best is not None
    assert best["birth_matched"] is True


# ---------------------------------------------------------------- cache
def test_wiki_cache_path():
    assert wiki_cache_path("data/raw", "cma-artist-4037").as_posix() == (
        "data/raw/wiki/cma-artist-4037.json"
    )


def test_is_cache_valid_accepts_all_three_statuses():
    for status in (STATUS_FOUND, STATUS_NOT_FOUND, STATUS_ERROR):
        payload = {"artist_id": "cma-artist-1", "status": status}
        assert is_cache_valid(payload, "cma-artist-1") is True


def test_is_cache_valid_rejects_other_artist():
    payload = {"artist_id": "cma-artist-1", "status": STATUS_FOUND}
    assert is_cache_valid(payload, "cma-artist-2") is False


def test_is_cache_valid_rejects_unknown_status():
    payload = {"artist_id": "cma-artist-1", "status": "weird"}
    assert is_cache_valid(payload, "cma-artist-1") is False
    assert is_cache_valid(None, "cma-artist-1") is False
    assert is_cache_valid("not-a-dict", "cma-artist-1") is False


def test_build_record_shape():
    record = build_record(
        {"artist_id": "cma-artist-4037", "artist_name": "Thomas Eakins", "artist_birth": 1844},
        STATUS_FOUND,
        qid="Q214905",
        wiki_title="Thomas Eakins",
        wiki_url="https://en.wikipedia.org/wiki/Thomas_Eakins",
        wiki_summary="Thomas Eakins was an American painter.",
        matched_label="Thomas Eakins",
        match_score=100.0,
        birth_matched=True,
        cma_bio="bio from CMA",
    )
    assert record["artist_id"] == "cma-artist-4037"
    assert record["status"] == STATUS_FOUND
    assert record["wikidata_qid"] == "Q214905"
    assert record["wiki_url"].startswith("https://en.wikipedia.org/wiki/")
    assert record["cma_bio"] == "bio from CMA"
    assert record["fetched_at"]
    assert is_cache_valid(record, "cma-artist-4037") is True


def test_build_record_not_found_has_no_wiki_fields():
    record = build_record({"artist_id": "cma-artist-1", "artist_name": "X"}, STATUS_NOT_FOUND)
    assert record["wiki_url"] is None
    assert record["wiki_summary"] is None
    assert record["status"] == STATUS_NOT_FOUND


# ---------------------------------------------------------------- rate limiter
def test_rate_limiter_enforces_interval():
    slept: list[float] = []
    clock = {"t": 100.0}

    def fake_sleep(duration: float) -> None:
        slept.append(duration)
        clock["t"] += duration

    limiter = RateLimiter(min_interval_s=0.5, sleep=fake_sleep, monotonic=lambda: clock["t"])
    limiter.wait()
    assert slept == []  # request đầu không cần đợi
    limiter.wait()
    assert slept and slept[0] == pytest.approx(0.5)


def test_rate_limiter_noop_when_interval_zero():
    slept: list[float] = []
    limiter = RateLimiter(min_interval_s=0, sleep=slept.append)
    limiter.wait()
    limiter.wait()
    assert slept == []


# ---------------------------------------------------------------- network (fake)
class FakeResponse:
    def __init__(self, status_code: int, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.headers = headers or {}

    def json(self):
        if not isinstance(self._payload, (dict, list)):
            raise ValueError("not json")
        return self._payload


class FakeSession:
    """Session giả: trả các response theo thứ tự."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params, "headers": headers})
        if not self.responses:
            raise AssertionError("hết response giả")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_http_get_json_retries_on_429_then_succeeds():
    from artrag.data.enrich_wiki import http_get_json

    session = FakeSession(
        [
            FakeResponse(429, headers={"Retry-After": "1"}),
            FakeResponse(200, {"ok": True}),
        ]
    )
    slept: list[float] = []
    payload = http_get_json(
        session,
        "https://example.test/api",
        params={"a": 1},
        sleep=slept.append,
        backoff_s=3.0,
    )
    assert payload == {"ok": True}
    assert len(session.calls) == 2
    # backoff có tôn trọng Retry-After
    assert slept == [3.0]


def test_http_get_json_raises_on_404():
    from artrag.data.enrich_wiki import http_get_json

    session = FakeSession([FakeResponse(404)])
    with pytest.raises(HttpNotFound):
        http_get_json(session, "https://example.test/api", sleep=lambda d: None)


def test_http_get_json_raises_after_exhausting_retries():
    from artrag.data.enrich_wiki import http_get_json

    session = FakeSession([FakeResponse(500), FakeResponse(500), FakeResponse(500)])
    with pytest.raises(RuntimeError):
        http_get_json(session, "https://example.test/api", sleep=lambda d: None, max_retries=2)


def test_search_entities_parses_payload():
    session = FakeSession(
        [FakeResponse(200, {"search": [make_candidate("Q1", "Thomas Eakins"), "junk"]})]
    )
    results = search_entities(session, "Thomas Eakins", sleep=lambda d: None)
    assert len(results) == 1
    assert results[0]["id"] == "Q1"
    assert session.calls[0]["params"]["action"] == "wbsearchentities"
    assert session.calls[0]["params"]["search"] == "Thomas Eakins"


def test_search_entities_handles_missing_search_key():
    session = FakeSession([FakeResponse(200, {"info": {}})])
    assert search_entities(session, "X", sleep=lambda d: None) == []


def test_fetch_summary_returns_extract():
    session = FakeSession(
        [
            FakeResponse(
                200,
                {
                    "extract": "Thomas Eakins was an American painter.",
                    "content_urls": {
                        "desktop": {"page": "https://en.wikipedia.org/wiki/Thomas_Eakins"}
                    },
                },
            )
        ]
    )
    payload = fetch_summary(session, "Thomas Eakins", sleep=lambda d: None)
    assert payload["extract"].startswith("Thomas Eakins")


# ---------------------------------------------------------------- batch mode
def test_sparql_escape():
    assert sparql_escape('Thomas "Big" Cole') == 'Thomas \\"Big\\" Cole'
    assert sparql_escape("back\\slash") == "back\\\\slash"
    assert sparql_escape("line\nbreak") == "line break"


def test_build_sparql_query_contains_values_and_filters():
    query = build_sparql_query(["Thomas Eakins", "Edgar Degas"])
    assert '"Thomas Eakins"@en' in query
    assert '"Edgar Degas"@en' in query
    assert "wdt:P31 wd:Q5" in query
    assert "rdfs:label|skos:altLabel" in query
    # p:P569/ps:P569 lấy cả năm sinh ở mọi rank (không chỉ `preferred`)
    assert "p:P569/ps:P569" in query
    assert "en.wikipedia.org" in query


def test_build_sparql_query_supports_language_tagged_terms():
    """Tên theo ngôn ngữ khác `en` phải giữ nguyên nhãn để tìm đúng entity."""
    query = build_sparql_query([("Lin Liang", "zh"), ("月岡雪鼎", "ja")])
    assert '"Lin Liang"@zh' in query
    assert '"月岡雪鼎"@ja' in query
    assert '"Lin Liang"@en' not in query
    # nhãn chuẩn phải được lấy riêng để loại nhầm (Shen Zhou có alias "Zhou Chen")
    assert "rdfs:label ?canon" in query


def test_parse_sparql_rows():
    payload = {
        "results": {
            "bindings": [
                {
                    "name": {"value": "Thomas Eakins"},
                    "item": {"value": "http://www.wikidata.org/entity/Q214905"},
                    "birth": {"value": "+1844-07-25T00:00:00Z"},
                    "article": {"value": "https://en.wikipedia.org/wiki/Thomas_Eakins"},
                },
                {"name": {"value": "bad"}},
            ]
        }
    }
    rows = parse_sparql_rows(payload)
    # parser giữ mọi binding có ít nhất một giá trị; việc loại row thiếu
    # `item`/`article` do resolve_names_batch đảm nhận
    assert len(rows) == 2
    assert rows[0]["name"] == "Thomas Eakins"
    assert rows[0]["item"].endswith("Q214905")
    assert parse_sparql_rows({}) == []
    assert parse_sparql_rows(None) == []


def test_qid_from_uri():
    assert qid_from_uri("http://www.wikidata.org/entity/Q214905") == "Q214905"
    assert qid_from_uri("https://www.wikidata.org/entity/Q5") == "Q5"
    assert qid_from_uri("http://example.org/x") is None
    assert qid_from_uri(None) is None


def test_title_from_article_url():
    assert title_from_article_url("https://en.wikipedia.org/wiki/Thomas_Eakins") == "Thomas Eakins"
    assert title_from_article_url("https://en.wikipedia.org/wiki/J._M._W._Turner") == "J. M. W. Turner"
    assert title_from_article_url("https://example.org/x") is None
    assert title_from_article_url(None) is None


def test_title_from_article_url_decodes_percent_encoding():
    """URL Wikipedia giải mã ký tự: ``Ogata K%C5%8Drin`` -> ``Ogata Kōrin``."""
    assert title_from_article_url("https://en.wikipedia.org/wiki/Ogata_K%C5%8Drin") == "Ogata Kōrin"
    assert (
        title_from_article_url("https://en.wikipedia.org/wiki/Sessh%C5%AB_T%C5%8Dy%C5%8D")
        == "Sesshū Tōyō"
    )


def test_is_artist_qid_rejects_non_artists():
    """Lớp bảo vệ thứ hai cho đường dẫn dự phòng (theo nghề P106)."""
    assert is_artist_qid({"occupations": ["Q11338576"], "descriptions": ["American boxer"]}) is False
    assert is_artist_qid({"occupations": [], "descriptions": ["Chinese politician"]}) is False
    assert (
        is_artist_qid(
            {"occupations": ["Q15980158"], "descriptions": ["American professional wrestler"]}
        )
        is False
    )


def test_is_artist_qid_accepts_artists():
    assert is_artist_qid({"occupations": ["Q1028181"], "descriptions": ["Chinese painter"]}) is True
    # họa sĩ + thiếu nghề -> dùng mô tả
    assert is_artist_qid({"occupations": [], "descriptions": ["Japanese painter"]}) is True
    # không có dữ liệu -> không loại
    assert is_artist_qid(None) is True
    assert is_artist_qid({}) is True


def test_birth_year_agrees():
    assert birth_year_agrees({"years": [1254]}, 1254) is True
    assert birth_year_agrees({"years": [1582, 1591]}, 1581) is True
    assert birth_year_agrees({"years": [1942]}, 1590) is False
    # không có bằng chứng phủ định -> chấp nhận
    assert birth_year_agrees({"years": []}, 1590) is True
    assert birth_year_agrees({"years": [1942]}, None) is True
    assert birth_year_agrees(None, 1590) is True


def test_fetch_occupations_batch_parses_occupations_and_years():
    payload = {
        "results": {
            "bindings": [
                {
                    "item": {"value": "http://www.wikidata.org/entity/Q36107"},
                    "desc": {"value": "American boxer (1942-2016)"},
                    "occ": {"value": "http://www.wikidata.org/entity/Q11338576"},
                    "birth": {"value": "1942-01-17T00:00:00Z"},
                }
            ]
        }
    }
    session = FakeSession([FakeResponse(200, payload)])
    info = fetch_occupations_batch(session, ["Q36107"], sleep=lambda d: None)
    assert info["Q36107"]["occupations"] == ["Q11338576"]
    assert info["Q36107"]["descriptions"] == ["American boxer (1942-2016)"]
    assert info["Q36107"]["years"] == [1942]


def test_first_birth_year_handles_empty_years():
    assert first_birth_year(None) is None
    assert first_birth_year({"years": []}) is None
    assert first_birth_year({}) is None
    assert first_birth_year({"years": [1844]}) == 1844


def test_looks_like_disambiguation():
    assert looks_like_disambiguation("Hashim may refer to:") is True
    assert looks_like_disambiguation("Tao Hong may refer to:") is True
    assert (
        looks_like_disambiguation("Hashim (Arabic: هاشم) is a common male Arabic given name.")
        is True
    )
    assert (
        looks_like_disambiguation(
            "Thomas Eakins was an American realist painter and teacher."
        )
        is False
    )
    assert looks_like_disambiguation("") is True
    assert looks_like_disambiguation(None) is True


def test_looks_like_disambiguation_only_checks_first_sentence():
    """Hồi quy: bài hợp lệ có câu "may refer to" ở giữa văn bản không bị loại.

    Bài về họa sĩ Bronzino có câu "His sobriquet, Bronzino, may refer to his
    relatively dark skin" — bài này HOÀN TOÀN hợp lệ.
    """
    bronzino = (
        "Agnolo di Cosimo (Italian: [ˈaɲɲolo di ˈkɔːzimo]; 17 November 1503 "
        "– 23 November 1572), usually known as Bronzino (Italian: Il Bronzino "
        "[il bronˈdziːno]) or Agnolo Bronzino, was an Italian Mannerist painter "
        "from Florence. His sobriquet, Bronzino, may refer to his relatively "
        "dark skin or reddish hair."
    )
    assert looks_like_disambiguation(bronzino) is False


def test_looks_like_disambiguation_detects_leading_markers():
    assert (
        looks_like_disambiguation(
            "Sravana, Shravana or Shravan may refer to: Shravana (hearing)."
        )
        is True
    )
    assert (
        looks_like_disambiguation("Liu Jie, may refer to: Liu Jie (politician).")
        is True
    )
    assert (
        looks_like_disambiguation("Manohar is an Indian surname and given name.")
        is True
    )


def test_is_artist_description():
    assert is_artist_description("Chinese painter") is True
    assert is_artist_description("English painter (1734-1802)") is True
    assert is_artist_description("Japanese painter, craftsman") is True
    assert is_artist_description("Chinese mathematician (1930-2021)") is False
    assert is_artist_description("Governor of Michigan") is False
    # không có mô tả -> không loại
    assert is_artist_description(None) is True
    assert is_artist_description("") is True


def test_pick_batch_candidate_rejects_non_artist_same_name():
    """``Wang Yuan`` họa sĩ, không phải nhà toán học cùng tên."""
    candidates = [
        {
            "qid": "Q2601552",
            "wiki_title": "Wang Yuan (mathematician)",
            "wiki_url": "u",
            "birth_year": 1930,
            "label": "Wang Yuan",
            "description": "Chinese mathematician (1930-2021)",
        },
        {
            "qid": "Q2889034",
            "wiki_title": "Wang Yuan (painter)",
            "wiki_url": "u2",
            "birth_year": 1283,
            "label": "Wang Yuan",
            "description": "Chinese painter",
        },
    ]
    best = pick_batch_candidate(candidates, "Wang Yuan", birth_year=1283)
    assert best is not None
    assert best["qid"] == "Q2889034"
    assert best["wiki_title"] == "Wang Yuan (painter)"


def test_pick_batch_candidate_rejects_all_non_artists():
    candidates = [
        {
            "qid": "Q2601552",
            "wiki_title": "Wang Yuan (mathematician)",
            "wiki_url": "u",
            "birth_year": 1930,
            "label": "Wang Yuan",
            "description": "Chinese mathematician",
        }
    ]
    assert pick_batch_candidate(candidates, "Wang Yuan", 1930) is None


def test_parse_extract_pages():
    payload = {
        "query": {
            "pages": [
                {
                    "title": "Thomas Eakins",
                    "extract": "Thomas Eakins was an American painter.",
                    "fullurl": "https://en.wikipedia.org/wiki/Thomas_Eakins",
                    "pageprops": {"wikibase_item": "Q214905"},
                },
                {"title": "No Extract"},
            ]
        }
    }
    pages = parse_extract_pages(payload)
    assert pages["Thomas Eakins"]["extract"].startswith("Thomas Eakins")
    assert pages["Thomas Eakins"]["fullurl"].startswith("https://en.wikipedia.org/wiki/")
    assert pages["Thomas Eakins"]["qid"] == "Q214905"
    assert pages["No Extract"]["extract"] is None
    assert pages["No Extract"]["qid"] is None
    assert parse_extract_pages({}) == {}


def test_parse_extract_pages_maps_redirect_names():
    """API trả tên chuẩn; tên đã hỏi phải được gán lại để khớp."""
    payload = {
        "query": {
            "redirects": [{"from": "Katsushika Hokusai", "to": "Hokusai"}],
            "pages": [
                {
                    "title": "Hokusai",
                    "extract": "Katsushika Hokusai was a Japanese ukiyo-e artist.",
                    "fullurl": "https://en.wikipedia.org/wiki/Hokusai",
                    "pageprops": {"wikibase_item": "Q5586"},
                }
            ],
        }
    }
    pages = parse_extract_pages(payload)
    assert "Hokusai" in pages
    assert "Katsushika Hokusai" in pages
    assert pages["Katsushika Hokusai"]["qid"] == "Q5586"
    assert pages["Katsushika Hokusai"]["fullurl"].endswith("Hokusai")


def test_fetch_enwiki_by_titles_returns_qid():
    payload = {
        "query": {
            "pages": [
                {
                    "title": "Zhao Mengfu",
                    "extract": "Zhao Mengfu was a Chinese painter.",
                    "fullurl": "https://en.wikipedia.org/wiki/Zhao_Mengfu",
                    "pageprops": {"wikibase_item": "Q197461"},
                }
            ]
        }
    }
    session = FakeSession([FakeResponse(200, payload)])
    pages = fetch_enwiki_by_titles(session, ["Zhao Mengfu"], sleep=lambda d: None)
    assert pages["Zhao Mengfu"]["qid"] == "Q197461"
    assert pages["Zhao Mengfu"]["extract"].startswith("Zhao Mengfu")


def test_pick_batch_candidate_uses_canonical_label_not_matched_name():
    """Hồi quy: `Shen Zhou` có alias `Zhou Chen` -> phải loại, không match nhầm.

    Trước đây `label` được gán bằng giá trị đã khớp nên mọi candidate đều đạt
    100 điểm, không phân biệt được họa sĩ nào đúng nhất.
    """
    candidates = [
        {
            "qid": "Q558863",
            "wiki_title": "Shen Zhou",
            "wiki_url": "https://en.wikipedia.org/wiki/Shen_Zhou",
            "birth_year": 1427,
            "label": "Shen Zhou",
            "canonical_label": "Shen Zhou",
            "matched_name": "Zhou Chen",
            "description": "Chinese painter (1427-1509)",
        },
        {
            "qid": "Q3575430",
            "wiki_title": "Zhou Chen",
            "wiki_url": "https://en.wikipedia.org/wiki/Zhou_Chen",
            "birth_year": 1460,
            "label": "Zhou Chen",
            "canonical_label": "Zhou Chen",
            "matched_name": "Zhou Chen",
            "description": "Chinese painter (1460-1535)",
        },
    ]
    best = pick_batch_candidate(candidates, "Zhou Chen", birth_year=1445)
    assert best is not None
    assert best["qid"] == "Q3575430"
    assert best["wiki_title"] == "Zhou Chen"


def test_pick_batch_candidate_rejects_wrong_person_by_birth_year():
    """Không khớp tên -> không được chấp nhận dù entity là họa sĩ.

    `Shen Zhou` (1427) khác `Zhou Chen` (1460); KB ghi 1445 -> sai người.
    """
    candidates = [
        {
            "qid": "Q558863",
            "wiki_title": "Shen Zhou",
            "wiki_url": "https://en.wikipedia.org/wiki/Shen_Zhou",
            "birth_year": 1427,
            "label": "Shen Zhou",
            "canonical_label": "Shen Zhou",
            "description": "Chinese painter (1427-1509)",
        }
    ]
    assert pick_batch_candidate(candidates, "Zhou Chen", birth_year=1445) is None


def test_pick_batch_candidate_prefers_birth_match():
    candidates = [
        {
            "qid": "Q1",
            "wiki_title": "Thomas Cole",
            "wiki_url": "https://en.wikipedia.org/wiki/Thomas_Cole",
            "birth_year": 1949,
            "label": "Thomas Cole",
        },
        {
            "qid": "Q2",
            "wiki_title": "Thomas Cole (painter)",
            "wiki_url": "https://en.wikipedia.org/wiki/Thomas_Cole_(painter)",
            "birth_year": 1801,
            "label": "Thomas Cole",
        },
    ]
    best = pick_batch_candidate(candidates, "Thomas Cole", birth_year=1801)
    assert best is not None
    assert best["qid"] == "Q2"
    assert best["birth_matched"] is True


def test_pick_batch_candidate_returns_none_when_no_match():
    candidates = [
        {"qid": "Q1", "wiki_title": "Someone Else", "wiki_url": "u", "birth_year": 1800, "label": "X"}
    ]
    assert pick_batch_candidate(candidates, "Thomas Eakins", 1844) is None
    assert pick_batch_candidate([], "Thomas Eakins", 1844) is None


def test_pick_batch_candidate_matches_exact():
    candidates = [
        {
            "qid": "Q214905",
            "wiki_title": "Thomas Eakins",
            "wiki_url": "https://en.wikipedia.org/wiki/Thomas_Eakins",
            "birth_year": 1844,
            "label": "Thomas Eakins",
        }
    ]
    best = pick_batch_candidate(candidates, "Thomas Eakins", 1844)
    assert best is not None
    assert best["qid"] == "Q214905"
    assert best["match_score"] == 100.0


def test_resolve_names_batch_groups_by_name():
    payload = {
        "results": {
            "bindings": [
                {
                    "name": {"value": "Thomas Eakins"},
                    "item": {"value": "http://www.wikidata.org/entity/Q214905"},
                    "article": {"value": "https://en.wikipedia.org/wiki/Thomas_Eakins"},
                }
            ]
        }
    }
    session = FakeSession([FakeResponse(200, payload)])
    grouped = resolve_names_batch(session, ["Thomas Eakins"], sleep=lambda d: None)
    assert "Thomas Eakins" in grouped
    assert grouped["Thomas Eakins"][0]["qid"] == "Q214905"


def test_fetch_extracts_batch_returns_map():
    payload = {
        "query": {
            "pages": [
                {
                    "title": "Thomas Eakins",
                    "extract": "A painter.",
                    "fullurl": "https://en.wikipedia.org/wiki/Thomas_Eakins",
                }
            ]
        }
    }
    session = FakeSession([FakeResponse(200, payload)])
    pages = fetch_extracts_batch(session, ["Thomas Eakins"], sleep=lambda d: None)
    assert pages["Thomas Eakins"]["extract"] == "A painter."


def test_build_extract_query_joins_titles():
    params = build_extract_query(["A B", "C D"])
    assert params["titles"] == "A B|C D"
    assert params["action"] == "query"
    assert params["explaintext"] == 1
    # `exlimit` phải <= 20 (giới hạn user ẩn danh) nếu không sẽ thiếu extract
    assert params["exlimit"] == 20


# ---------------------------------------------------------------- cma bios
def test_load_artists_dedups_by_artist_id(tmp_path):
    """NaN artist_id phải bị bỏ qua; trùng artist_id chỉ giữ 1 dòng."""
    import pandas as pd

    from artrag.data.enrich_wiki import load_artists

    kb = tmp_path / "kb"
    kb.mkdir(parents=True)
    frame = pd.DataFrame(
        [
            {
                "artist_id": "cma-artist-1",
                "artist_name": "A",
                "artist_birth": 1800.0,
                "artist_death": 1850.0,
                "artist_nationality": "X",
            },
            {
                "artist_id": "cma-artist-1",
                "artist_name": "A",
                "artist_birth": 1800.0,
                "artist_death": 1850.0,
                "artist_nationality": "X",
            },
            {
                "artist_id": None,
                "artist_name": None,
                "artist_birth": None,
                "artist_death": None,
                "artist_nationality": None,
            },
        ]
    )
    frame.to_parquet(kb / "artworks.parquet", engine="pyarrow", index=False)

    class _Cfg:
        def get(self, key, default=None):
            if key == "paths.data_dir":
                return str(tmp_path)
            return default

    artists = load_artists(_Cfg())
    assert [a["artist_id"] for a in artists] == ["cma-artist-1"]
    assert artists[0]["artist_birth"] == 1800


def test_load_cma_bios_reads_creators_biography(tmp_path):
    cma_dir = tmp_path / "raw" / "cma"
    cma_dir.mkdir(parents=True)
    payload = {
        "data": [
            {
                "creators": [
                    {"id": 4037, "role": "artist", "biography": "<p>Eakins bio</p>"},
                    {"id": 4038, "role": "artist"},
                ]
            }
        ]
    }
    (cma_dir / "page_00000.json").write_text(json.dumps(payload), encoding="utf-8")

    bios = load_cma_bios(tmp_path / "raw")
    assert bios == {"cma-artist-4037": "Eakins bio"}
