"""T1.4: lam giau tieu su hoa si tu Wikidata + Wikipedia (SYSTEM_DESIGN 2.1, 2.3).

Moi hoa si duy nhat trong KB (dedup theo ``artist_id``) duoc tra cuu:

1. ``wbsearchentities`` (Wikidata) -> danh sach candidate.
2. ``wbgetentities`` -> xac nhan ``P31 == Q5`` (la nguoi) va doc
   ``sitelinks.enwiki``; dung ``P569`` de phan biet cung ten.
3. REST ``page/summary`` cua en.wikipedia.org -> ``extract`` (tom tat) + URL.

Ket qua duoc cache tai ``raw/wiki/{artist_id}.json``; ca ``not_found`` va
``error`` cung duoc cache nen chay lai khong goi lai mang. Wiki rat de
tra HTTP 429, vi du rat limiter + retry/backoff la bat buoc.

Cac ham phan tich (``pick_candidate``, ``label_score``, ``entity_years`` ...)
la ham thuan, test duoc ma khong can mang.

Lenh:
    uv run python -m artrag.data.enrich_wiki --limit 30
    uv run python -m artrag.data.enrich_wiki
"""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, unquote

import pandas as pd
import requests
from rapidfuzz import fuzz

from artrag.config import Config, load_config, paths
from artrag.utils.io import write_json
from artrag.utils.text import strip_html

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKIPEDIA_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"

STATUS_FOUND = "found"
STATUS_NOT_FOUND = "not_found"
STATUS_ERROR = "error"

# nguoi tren Wikidata
Q_HUMAN = "Q5"
# claim: ngay sinh
P_BIRTH = "P569"
# claim: instance of
P_INSTANCE_OF = "P31"

# nguong nhieu nhat cho nhan khi khop nam sinh (khop ten chat hon van duoc dung)
BIRTH_MIN_LABEL_SCORE = 70.0

# nghe nghiep nghe thuat tren Wikidata: chi nhan entity la hoa si moi dung.
# Q1028181=painter, Q483501=artist, Q1281618=sculptor, Q11569986=printmaker,
# Q329439=engraver, Q644687=illustrator, Q33231=photographer,
# Q3303330=calligrapher, Q984276=court painter, Q1437754=xylographer,
# Q33082999=ukiyō-e artist, Q998628=illuminator, Q1114448=cartoonist,
# Q16947657=lithographer, Q21550468=chromolithographer, Q21925567=serigrapher,
# Q18510097=glass engraver, Q42973=architect, Q5322166=designer,
# Q780596=exhibition curator, Q1792450=art historian, Q2500638=creator.
#
# Lưu ý: Q36180 (writer) và Q49757 (poet) **không** được đưa vào đây — chúng là
# *subclass* của Q483501 (artist) trên Wikidata, nên nếu đưa vào thì mọi nhà
# văn/poet sẽ bị coi là hoạ sĩ (false positive).
ARTIST_OCCUPATIONS = [
    "Q1028181",  # painter
    "Q483501",   # artist
    "Q1281618",  # sculptor
    "Q11569986",  # printmaker
    "Q329439",   # engraver
    "Q644687",   # illustrator
    "Q33231",    # photographer
    "Q2500638",  # creator
    "Q3303330",  # calligrapher
    "Q984276",   # court painter
    "Q1437754",  # xylographer
    "Q33082999",  # ukiyo-e artist
    "Q998628",   # illuminator
    "Q1114448",  # cartoonist
    "Q16947657",  # lithographer
    "Q21550468",  # chromolithographer
    "Q21925567",  # serigrapher
    "Q18510097",  # glass engraver
    "Q42973",    # architect
    "Q5322166",  # designer
    "Q780596",   # exhibition curator
    "Q1792450",  # art historian
    "Q1294787",  # artisan
]

# tu khoa trong mo ta Wikidata cho biet entity co phai hoa si khong
ARTIST_DESC_KEYWORDS = (
    "painter",
    "artist",
    "sculptor",
    "printmaker",
    "engraver",
    "illustrator",
    "photographer",
    "calligrapher",
    "ceramist",
    "printmaker",
    "murikari",
    "designer",
    # cac nghe phai them vi chi ban SPARQL loc theo P106 nen mo ta van can
    # de chan cac hoa si KHONG co enwiki (thi `p:P106` rong -> phai dung mo ta).
    "draughtsman",
    "draftsman",
    "miniaturist",
    "potter",
    "woodblock",
    "woodblock print",
    "ukiyō-e",
    "ukiyo-e",
    "lithographer",
    "wood engraver",
    "copper engraver",
    "illuminator",
    "watercolorist",
    "court painter",
    "imperial painter",
    "muralist",
    "figure painter",
    "landscape painter",
    "portrait painter",
    "genre painter",
    "history painter",
    "cartoonist",
    "graphic artist",
    "visual artist",
    "exhibition curator",
    "art historian",
    "architect",
    "silk painter",
    "paper-cut artist",
)


class HttpNotFound(Exception):
    """Server tra 404 (trang/doi tuong khong ton tai)."""


# ---------------------------------------------------------------- helpers
def _cfg_get(cfg: Config, key: str, default: Any) -> Any:
    value = cfg.get(f"data.{key}", None)
    return default if value is None else value


def normalize_name(name: Any) -> str:
    """Chuan hoa ten de so sanh: bo dau, lowercase, gop khoang trang.

    Cac dau nhay phai dua ve mot dang de so sanh duoc giua ten trong KB va
    ten bai enwiki: ``Kano Tan’yū`` (U+2019) vs. ``Kanō Tan'yū`` (U+0027).
    """
    if name is None:
        return ""
    text = strip_html(str(name)).lower()
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("đ", "d")
    return " ".join(text.split())


def _tokens(name: Any) -> list[str]:
    """Chia ten thanh token (chuan hoa da bo dau)."""
    normalized = normalize_name(name)
    return normalized.split() if normalized else []


def _merge_initials(left: list[str], right: list[str]) -> tuple[list[str], list[str]]:
    """Gop chu cai dau voi token dai hon o phia doi dien.

    ``Joseph Mallord William Turner`` vs ``J. M. W. Turner``: cac chu cai
    dau ``J/M/W`` duoc gop vao ``Joseph/Mallord/William`` de hai ben co cung
    tap token.
    """
    left_merged = list(left)
    right_merged = list(right)

    def _apply(shorter: list[str], longer: list[str]) -> None:
        for index, token in enumerate(list(shorter)):
            # token "j." la chu cai dau co dau cham cuoi
            if len(token.rstrip(".")) != 1:
                continue
            initial = token.rstrip(".")[0]
            for other_index, other in enumerate(longer):
                if other.startswith(initial):
                    shorter[index] = other
                    longer.pop(other_index)
                    break

    _apply(left_merged, right_merged)
    _apply(right_merged, left_merged)
    return left_merged, right_merged


def _has_prefix_conflict(left: list[str], right: list[str]) -> bool:
    """Co token nao cua hai ben la tien to cua token kia khong?

    ``Wang Yuan`` va ``Wang Yuanqi`` la hai hoa si khac nhau nhung lai la
    tien to cua nhau theo chuoi -> ``fuzz.ratio`` cho 90 diem (do do dai
    giong nhau) du chung la hai nguoi khac.
    """
    for a in left:
        for b in right:
            if a == b:
                continue
            if a.startswith(b) or b.startswith(a):
                return True
    return False


def label_score(name: Any, label: Any) -> float:
    """Do giong ten (0..100) giua ten trong KB va label cua Wikidata.

    ``rapidfuzz.token_set_ratio`` **khong** phu hop o day: no so sanh tap
    token con lai nen hai ten khong lien quan van duoc diem cao. Vi du
    ``Zhou Chen`` (Ming, sinh 1460) vs ``Shen Zhou`` (sinh 1427) deu chung
    token ``zhou`` -> ``token_set_ratio`` = 88.9, du de qua nguong 88 va
    khong loai duoc hoa si sai.

    Vì vay dung **kiem tra chua** truoc: hai ten chi duoc coi la cung mot
    nguoi khi mot tap token chua tap token con lai (sau khi gop chu cai dau).
    ``Agnolo Bronzino`` chua ``Bronzino`` -> hop le; ``Joseph Mallord William
    Turner`` chua ``J. M. W. Turner`` (sau khi gop) -> hop le; ``Zhou Chen``
    khong chua ``Shen Zhou`` -> khong phai cung nguoi, diem thap.
    """
    left = _tokens(name)
    right = _tokens(label)
    if not left or not right:
        return 0.0
    left_merged, right_merged = _merge_initials(left, right)
    left_set = set(left_merged)
    right_set = set(right_merged)
    # phai la chua **nguyen token**, khong phai chua tien to: `Wang Yuan` khong
    # phai la `Wang Yuanqi` (hoa si khac).
    contained = left_set.issubset(right_set) or right_set.issubset(left_set)
    if contained:
        return float(
            fuzz.token_set_ratio(" ".join(left_merged), " ".join(right_merged))
        )
    if _has_prefix_conflict(left_merged, right_merged):
        # tien to cua nhau -> hai nguoi khac, du chuoi rat giong nhau.
        return 0.0
    # hai tap token khong chua nhau -> khac nguoi -> dung do giong ky tu.
    return float(fuzz.ratio(" ".join(left), " ".join(right)))


def year_from_time(value: Any) -> int | None:
    """Chuoi thoi gian Wikidata/SPARQL -> nam.

    Hai dinh dang xuat hien:
    - Wikidata entity API: ``+1844-07-25T00:00:00Z`` (co dau ``+``)
    - SPARQL endpoint:    ``1844-07-25T00:00:00Z``  (khong co dau ``+``)
    """
    if not isinstance(value, str) or len(value) < 4:
        return None
    text = value[1:] if value[0] in "+-" else value
    try:
        year = int(text[:4])
    except ValueError:
        return None
    if year < 1000 or year > 2100:
        return None
    return year


def _claim_values(entity: Any, prop: str) -> list[Any]:
    """Gia tri cua mot claim (bo qua snak khong co datavalue)."""
    if not isinstance(entity, dict):
        return []
    claims = entity.get("claims")
    if not isinstance(claims, dict):
        return []
    entries = claims.get(prop)
    if not isinstance(entries, list):
        return []
    values: list[Any] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        snak = entry.get("mainsnak")
        if not isinstance(snak, dict):
            continue
        if snak.get("snaktype") not in (None, "value"):
            continue
        datavalue = snak.get("datavalue")
        if not isinstance(datavalue, dict):
            continue
        if "value" in datavalue:
            values.append(datavalue["value"])
    return values


def entity_years(entity: Any) -> list[int]:
    """Cac nam sinh (``P569``) cua entity."""
    years: list[int] = []
    for value in _claim_values(entity, P_BIRTH):
        if isinstance(value, dict):
            year = year_from_time(value.get("time"))
            if year is not None:
                years.append(year)
    return years


def is_human(entity: Any) -> bool:
    """Entity co ``P31`` chua ``Q5`` (con nguoi) hay khong."""
    for value in _claim_values(entity, P_INSTANCE_OF):
        if isinstance(value, dict) and value.get("id") == Q_HUMAN:
            return True
    return False


def entity_label(entity: Any) -> str | None:
    """Nhan tieng Anh cua entity (``labels.en.value``)."""
    if not isinstance(entity, dict):
        return None
    labels = entity.get("labels")
    if not isinstance(labels, dict):
        return None
    en = labels.get("en")
    if not isinstance(en, dict):
        return None
    value = en.get("value")
    return value if isinstance(value, str) and value else None


def entity_sitelink(entity: Any, site: str = "enwiki") -> str | None:
    """Ten bai wiki tuong ung (``sitelinks.enwiki.title``)."""
    if not isinstance(entity, dict):
        return None
    sitelinks = entity.get("sitelinks")
    if not isinstance(sitelinks, dict):
        return None
    link = sitelinks.get(site)
    if not isinstance(link, dict):
        return None
    title = link.get("title")
    return title if isinstance(title, str) and title else None


# ---------------------------------------------------------------- matching
def candidate_years(candidate: Any) -> list[int]:
    """Cac nam nam trong ``description`` cua search result, vi du ``"(c. 1775-1851)"``."""
    if not isinstance(candidate, dict):
        return []
    description = candidate.get("description")
    if not isinstance(description, str):
        return []
    years: list[int] = []
    for token in re.findall(r"\b(\d{3,4})\b", description):
        year = int(token)
        if 1000 <= year <= 2100:
            years.append(year)
    return years


def candidate_name_variants(candidate: Any, entity: Any = None) -> list[str]:
    """Cac bien the ten cua candidate: label, ``match.text``, aliases.

    Wikidata luu nhieu hoa si voi nhan canonical rut gon (vi du ``J. M. W.
    Turner`` trong khi CMA goi ``Joseph Mallord William Turner``). ``wbsearchentities``
    tra ve ``match.text`` va ``aliases`` la cac bien the do, nen phai cham diem
    moi bien the chu khong chi label.
    """
    variants: list[str] = []
    if isinstance(candidate, dict):
        label = candidate.get("label")
        if isinstance(label, str) and label:
            variants.append(label)
        match = candidate.get("match")
        if isinstance(match, dict):
            text = match.get("text")
            if isinstance(text, str) and text:
                variants.append(text)
        aliases = candidate.get("aliases")
        if isinstance(aliases, list):
            variants.extend(a for a in aliases if isinstance(a, str) and a)
    if isinstance(entity, dict):
        label = entity_label(entity)
        if label:
            variants.append(label)
        entity_aliases = entity.get("aliases")
        if isinstance(entity_aliases, dict):
            en = entity_aliases.get("en")
            if isinstance(en, list):
                for item in en:
                    if isinstance(item, dict) and isinstance(item.get("value"), str):
                        variants.append(item["value"])
    # bo trung, giu thu tu
    seen: set[str] = set()
    unique: list[str] = []
    for value in variants:
        key = normalize_name(value)
        if key and key not in seen:
            seen.add(key)
            unique.append(value)
    return unique


def pick_candidate(
    candidates: list[dict[str, Any]],
    entities: dict[str, Any],
    name: str,
    birth_year: int | None = None,
    label_threshold: float = 88.0,
    birth_tolerance: int = 3,
) -> dict[str, Any] | None:
    """Chon candidate khop nhat cho mot hoa si.

    Uu tien: (1) co ``P31=Q5``, (2) co sitelink ``enwiki``, (3) khop ten,
    (4) khop nam sinh. Tra ``None`` neu khong co candidate nao dat yeu cau.
    """
    if not candidates:
        return None

    scored: list[tuple[tuple[int, float], dict[str, Any]]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        qid = candidate.get("id")
        if not isinstance(qid, str) or not qid:
            continue
        entity = entities.get(qid)
        if not isinstance(entity, dict):
            continue
        if not is_human(entity):
            continue
        title = entity_sitelink(entity)
        if not title:
            continue

        variants = candidate_name_variants(candidate, entity)
        best_variant = variants[0] if variants else ""
        best_score = 0.0
        for variant in variants:
            score = label_score(name, variant)
            if score > best_score:
                best_score = score
                best_variant = variant

        years = entity_years(entity) or candidate_years(candidate)
        birth_matched = bool(
            birth_year is not None and any(abs(y - birth_year) <= birth_tolerance for y in years)
        )

        qualifies = best_score >= label_threshold
        if not qualifies and birth_matched and best_score >= BIRTH_MIN_LABEL_SCORE:
            qualifies = True
        if not qualifies:
            continue

        key = (1 if birth_matched else 0, best_score)
        scored.append(
            (
                key,
                {
                    "qid": qid,
                    "wiki_title": title,
                    "matched_label": best_variant,
                    "match_score": round(best_score, 2),
                    "birth_matched": birth_matched,
                    "entity_years": years,
                },
            )
        )

    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


# ---------------------------------------------------------------- batch mode
# Chi ti gia: de do lai bang lenh nay (2026-11-25):
#   26 request wbsearchentities lien tiep -> 16 loi 429, phai doi 45-52 giay.
#   => che do tung artist khong kha thi cho 762 artist (~2.300 request).
#   Che do batch: 25 ten/req SPARQL ~5s, 30 title/req Wikipedia ~5s
#   => ~57 request cho toan bo 762 artist.
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"


def sparql_escape(value: str) -> str:
    """Escape chuoi de nhung vao literal SPARQL."""
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", " ")
        .replace("\r", " ")
    )


def build_sparql_query(terms: list[str] | list[tuple[str, str]]) -> str:
    """Cau SPARQL tra QID + nam sinh + URL bai enwiki cho mot nhom ten.

    ``terms`` la danh sach ten, hoac danh sach ``(ten, ngon_ngu)``. Khi co
    ngon ngu thi literal duoc gan nhan tuong ung — **bat buoc**, vi nhan
    Wikidata cua hoa si Nhat/Trung thuong khong co o ``en`` (vi du
    ``Tsukioka Settei`` chi co nhan ``ja``/``mul``, ``Lin Liang`` chi co
    ``zh``/``fr``/``ru``). Neu ep ``@en`` thi hoa si do bi bo qua hoan toan.

    Loc them nghe nghiep nghe thuat (``wdt:P106`` trong
    ``ARTIST_OCCUPATIONS``) de loai bo nguoi cung ten nhung khac nghe
    (vi du ``Wang Yuan`` hoa si vs. `Wang Yuan (mathematician)`).

    ``?canon`` la **nhan tieng Anh chuan** (``rdfs:label`` voi ``LANG="en"``).
    Can phai lay no rieng biet vi ``?name`` chi la nhan/alias khớp duoc: hoa si
    ``Zhou Chen`` (Ming) co alias ``Zhou Chen`` tren entity ``Shen Zhou`` — mot
    hoa si khac — nen diem so sanh phai dua tren nhan chuan de loai nham.

    ``p:P569/ps:P569`` lay **tat ca** gia tri nam sinh (khong chi rank
    ``preferred`` nhu ``wdt:P569``), vi nhieu hoa si co nhieu nam sinh ghi
    o cac rank khac nhau (vi du ``Tsukioka Settei``: 1710 va 1726).
    """
    values = " ".join(
        f'"{sparql_escape(name)}"@{lang}'
        for name, lang in (terms if _is_term_pairs(terms) else [(t, "en") for t in terms])
    )
    occupations = " ".join(f"wd:{qid}" for qid in ARTIST_OCCUPATIONS)
    return (
        "SELECT ?name ?item ?canon ?birth ?article ?desc WHERE { "
        f"VALUES ?name {{ {values} }} "
        "?item rdfs:label|skos:altLabel ?name ; wdt:P31 wd:Q5 ; wdt:P106 ?occ . "
        f"VALUES ?occ {{ {occupations} }} "
        "OPTIONAL { ?item rdfs:label ?canon FILTER(LANG(?canon)=\"en\") } "
        "OPTIONAL { ?item p:P569/ps:P569 ?birth } "
        "OPTIONAL { ?item schema:description ?desc FILTER(LANG(?desc)=\"en\") } "
        "OPTIONAL { ?article schema:about ?item ; "
        "schema:isPartOf <https://en.wikipedia.org/> . } "
        "}"
    )


def _is_term_pairs(terms: Any) -> bool:
    """Danh sach co phai la cac cap ``(ten, ngon_ngu)`` khong?"""
    if not terms:
        return False
    first = terms[0]
    return isinstance(first, (tuple, list)) and len(first) == 2


def parse_sparql_rows(payload: Any) -> list[dict[str, Any]]:
    """Chuyen JSON SPARQL thanh list dict thuoc tinh don gian."""
    if not isinstance(payload, dict):
        return []
    results = payload.get("results")
    if not isinstance(results, dict):
        return []
    bindings = results.get("bindings")
    if not isinstance(bindings, list):
        return []
    rows: list[dict[str, Any]] = []
    for binding in bindings:
        if not isinstance(binding, dict):
            continue
        row: dict[str, Any] = {}
        for key, value in binding.items():
            if isinstance(value, dict) and isinstance(value.get("value"), str):
                row[key] = value["value"]
        if row:
            rows.append(row)
    return rows


def qid_from_uri(uri: Any) -> str | None:
    """``http://www.wikidata.org/entity/Q214905`` -> ``Q214905``."""
    if not isinstance(uri, str):
        return None
    if "/entity/" not in uri:
        return None
    tail = uri.rsplit("/entity/", 1)[1]
    return tail if tail.startswith("Q") else None


def title_from_article_url(url: Any) -> str | None:
    """``https://en.wikipedia.org/wiki/Thomas_Eakins`` -> ``Thomas Eakins``.

    Phai giai ma phan trong (URL percent-encode): ``Ogata K%C5%8Drin`` ->
    ``Ogata Kōrin``, neu khong se khong khop duoc voi tieu de bai that.
    """
    if not isinstance(url, str):
        return None
    marker = "/wiki/"
    if marker not in url:
        return None
    tail = url.rsplit(marker, 1)[1]
    if "/" in tail:
        tail = tail.split("/", 1)[0]
    return unquote(tail.replace("_", " "))


def build_extract_query(titles: list[str]) -> dict[str, Any]:
    """Tham so cho API Wikipedia lay tom tat nhieu bai mot luc.

    ``pageprops`` cho ``wikibase_item`` (QID cua bai) — duoc dung lam duong
    giai khi ten tren Wikidata khac ten tren enwiki (vi du hoa si Trung
    Quoc: nhan Wikidata la tieng Trong `趙孟頫`, con bai enwiki la
    `Zhao Mengfu`).
    """
    return {
        "action": "query",
        "prop": "extracts|info|pageprops",
        "exintro": 1,
        "explaintext": 1,
        # `exlimit` toi da 20 voi user an danh; neu vuot, API chi tra extract
        # cho 20 trang dau va cac trang sau bi thieu summary.
        "exlimit": 20,
        "redirects": 1,
        "inprop": "url",
        # `disambiguation` cho biet chinh xac trang nao la trang "dinh nghia"
        # — tin hieu chinh thuc cua Wikipedia, thay cho su doan van ban.
        "ppprop": "wikibase_item|disambiguation",
        "titles": "|".join(titles),
        "format": "json",
        "formatversion": 2,
    }


def parse_extract_pages(payload: Any) -> dict[str, dict[str, Any]]:
    """Chuyen JSON Wikipedia thanh map ``title -> {extract, fullurl, qid}``."""
    result: dict[str, dict[str, Any]] = {}
    if not isinstance(payload, dict):
        return result
    query = payload.get("query")
    if not isinstance(query, dict):
        return result
    pages = query.get("pages")
    if not isinstance(pages, list):
        return result
    # ban gia ten -> ten chuan (vi du "Katsushika Hokusai" -> "Hokusai")
    aliases: dict[str, str] = {}
    for key in ("redirects", "normalized"):
        entries = query.get(key)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            source = entry.get("from")
            target = entry.get("to")
            if isinstance(source, str) and isinstance(target, str):
                aliases[source] = target
    for page in pages:
        if not isinstance(page, dict):
            continue
        title = page.get("title")
        if not isinstance(title, str) or not title:
            continue
        extract = page.get("extract")
        pageprops = page.get("pageprops")
        qid = None
        is_disambiguation = False
        if isinstance(pageprops, dict):
            raw_qid = pageprops.get("wikibase_item")
            if isinstance(raw_qid, str) and raw_qid.startswith("Q"):
                qid = raw_qid
            # Wikipedia gan `disambiguation` (gia tri rong) cho trang dinh nghia.
            is_disambiguation = "disambiguation" in pageprops
        entry = {
            "extract": strip_html(extract) if isinstance(extract, str) and extract else None,
            "fullurl": page.get("fullurl"),
            "qid": qid,
            "is_disambiguation": is_disambiguation,
        }
        result[title] = entry
        # gan them key cho ten da bi chuyen huong
        for source, target in aliases.items():
            if target == title:
                result[source] = entry
    return result


def build_search_query(term: str) -> dict[str, Any]:
    """Tham so cho API Wikipedia tim bai theo ten (khi khong co sitelink)."""
    return {
        "action": "query",
        "list": "search",
        "srsearch": term,
        "srnamespace": 0,
        "srlimit": 1,
        "format": "json",
        "formatversion": 2,
    }


def fetch_enwiki_search_titles(
    session: Any,
    terms: list[str],
    timeout_s: float = 60.0,
    max_retries: int = 4,
    backoff_s: float = 3.0,
    limiter: RateLimiter | None = None,
    sleep: Callable[[float], None] = time.sleep,
    user_agent: str | None = None,
) -> dict[str, str]:
    """Tra ``{ten_goc: ten_bai_enwiki}`` cho cac ten co bai tren enwiki.

    Dung cho hoa si **khong co enwiki sitelink tren Wikidata** (vi du hoa si
    Nhat chi duoc ghi o bai Nhat): ten bai enwiki van ton tai va tim duoc
    bang full-text search, nhung SPARQL ``schema:isPartOf`` thi khong tra ve.
    """
    result: dict[str, str] = {}
    if not terms:
        return result
    headers = {"User-Agent": user_agent} if user_agent else None
    for term in terms:
        try:
            payload = http_get_json(
                session,
                WIKIPEDIA_API,
                params=build_search_query(term),
                headers=headers,
                timeout_s=timeout_s,
                max_retries=max_retries,
                backoff_s=backoff_s,
                limiter=limiter,
                sleep=sleep,
            )
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(payload, dict):
            continue
        query = payload.get("query")
        if not isinstance(query, dict):
            continue
        hits = query.get("search")
        if not isinstance(hits, list) or not hits:
            continue
        first = hits[0]
        title = first.get("title") if isinstance(first, dict) else None
        if isinstance(title, str) and title:
            result[term] = title
    return result


def fetch_enwiki_by_titles(
    session: Any,
    titles: list[str],
    timeout_s: float = 60.0,
    max_retries: int = 4,
    backoff_s: float = 3.0,
    limiter: RateLimiter | None = None,
    sleep: Callable[[float], None] = time.sleep,
    user_agent: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Tra cuu bai enwiki theo **ten bai** (khong phai theo nhan Wikidata).

    Dung khi SPARQL theo nhan khong ra ket qua: nhieu hoa si Trung Quoc/Nhat
    co nhan Wikidata o chu Han, nhung van co bai tieng Anh tren enwiki.
    """
    if not titles:
        return {}
    headers = {"User-Agent": user_agent} if user_agent else None
    payload = http_get_json(
        session,
        WIKIPEDIA_API,
        params=build_extract_query(titles),
        headers=headers,
        timeout_s=timeout_s,
        max_retries=max_retries,
        backoff_s=backoff_s,
        limiter=limiter,
        sleep=sleep,
    )
    return parse_extract_pages(payload)


def resolve_names_batch(
    session: Any,
    names: list[str] | list[tuple[str, str]],
    timeout_s: float = 60.0,
    max_retries: int = 4,
    backoff_s: float = 3.0,
    limiter: RateLimiter | None = None,
    sleep: Callable[[float], None] = time.sleep,
    user_agent: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Giai nhieu ten hoa si bang 1 request SPARQL.

    ``names`` la danh sach ten, hoac danh sach ``(ten, ngon_ngu)``.
    Tra ``{ten: [candidate, ...]}``; moi candidate co ``qid``, ``wiki_title``,
    ``wiki_url``, ``birth_year``, ``label``, ``canonical_label``.
    """
    if not names:
        return {}
    query = build_sparql_query(names)
    headers = {"Accept": "application/sparql-results+json"}
    if user_agent:
        headers["User-Agent"] = user_agent
    payload = http_get_json(
        session,
        WIKIDATA_SPARQL,
        params={"query": query, "format": "json"},
        headers=headers,
        timeout_s=timeout_s,
        max_retries=max_retries,
        backoff_s=backoff_s,
        limiter=limiter,
        sleep=sleep,
    )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in parse_sparql_rows(payload):
        name = row.get("name")
        qid = qid_from_uri(row.get("item"))
        if not name or not qid:
            continue
        url = row.get("article")
        title = title_from_article_url(url)
        if not title:
            continue
        year = year_from_time(row.get("birth"))
        desc = row.get("desc")
        canon = row.get("canon")
        grouped.setdefault(name, []).append(
            {
                "qid": qid,
                "wiki_title": title,
                "wiki_url": url,
                "birth_year": year,
                # `canon` = nhan tieng Anh chuan cua entity. KHONG dung `name`
                # (gia tri da khop) lam `label`, vi khi do moi candidate deu
                # goc diem 100 va khong phan biet duoc hoa si nao dung nhat.
                "label": canon or name,
                "canonical_label": canon,
                "matched_name": name,
                "description": desc,
            }
        )
    return grouped


def fetch_extracts_batch(
    session: Any,
    titles: list[str],
    timeout_s: float = 60.0,
    max_retries: int = 4,
    backoff_s: float = 3.0,
    limiter: RateLimiter | None = None,
    sleep: Callable[[float], None] = time.sleep,
    user_agent: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Lay tom tat nhieu bai wikipedia bang 1 request (``action=query``)."""
    if not titles:
        return {}
    headers = {"User-Agent": user_agent} if user_agent else None
    payload = http_get_json(
        session,
        WIKIPEDIA_API,
        params=build_extract_query(titles),
        headers=headers,
        timeout_s=timeout_s,
        max_retries=max_retries,
        backoff_s=backoff_s,
        limiter=limiter,
        sleep=sleep,
    )
    return parse_extract_pages(payload)


def build_occupation_query(qids: list[str]) -> str:
    """Cau SPARQL lay mo ta + nghe + nam sinh cua nhieu QID.

    Dung de kiem tra duong dan du phong (xem ``is_artist_qid``).

    ``p:P569/ps:P569`` lay **tat ca** cac gia tri nam sinh (ke ca rank
    ``normal``/``deprecated``), con ``wdt:P569`` chi lay gia tri ``preferred``.
    Vi du ``Tsukioka Settei`` co nam sinh 1710 (normal) va 1726 (preferred);
    chi lay ``wdt:P569`` se bo qua 1710 va loai oai hoa si khi KB ghi 1710.
    """
    values = " ".join(f"wd:{sparql_escape(qid)}" for qid in qids)
    return (
        "SELECT ?item ?desc ?occ ?birth WHERE { "
        f"VALUES ?item {{ {values} }} "
        "OPTIONAL { ?item schema:description ?desc FILTER(LANG(?desc)=\"en\") } "
        "OPTIONAL { ?item wdt:P106 ?occ . } "
        "OPTIONAL { ?item p:P569/ps:P569 ?birth . } "
        "}"
    )


def fetch_occupations_batch(
    session: Any,
    qids: list[str],
    timeout_s: float = 60.0,
    max_retries: int = 4,
    backoff_s: float = 3.0,
    limiter: RateLimiter | None = None,
    sleep: Callable[[float], None] = time.sleep,
    user_agent: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Tra ``{qid: {"descriptions": [...], "occupations": [Q-id, ...]}}``."""
    if not qids:
        return {}
    headers = {"Accept": "application/sparql-results+json"}
    if user_agent:
        headers["User-Agent"] = user_agent
    payload = http_get_json(
        session,
        WIKIDATA_SPARQL,
        params={"query": build_occupation_query(qids), "format": "json"},
        headers=headers,
        timeout_s=timeout_s,
        max_retries=max_retries,
        backoff_s=backoff_s,
        limiter=limiter,
        sleep=sleep,
    )
    result: dict[str, dict[str, Any]] = {}
    for row in parse_sparql_rows(payload):
        qid = qid_from_uri(row.get("item"))
        if not qid:
            continue
        entry = result.setdefault(
            qid, {"descriptions": [], "occupations": [], "years": []}
        )
        desc = row.get("desc")
        if isinstance(desc, str) and desc not in entry["descriptions"]:
            entry["descriptions"].append(desc)
        occ = row.get("occ")
        if isinstance(occ, str):
            occ_id = occ.rsplit("/", 1)[-1]
            if occ_id.startswith("Q") and occ_id not in entry["occupations"]:
                entry["occupations"].append(occ_id)
        year = year_from_time(row.get("birth"))
        if year is not None and year not in entry["years"]:
            entry["years"].append(year)
    return result


def first_birth_year(qid_info: dict[str, Any] | None) -> int | None:
    """Nam sinh dau tien cua entity (gia tri ``None`` neu khong biet)."""
    if not qid_info:
        return None
    years = qid_info.get("years") or []
    return years[0] if years else None


def is_artist_qid(qid_info: dict[str, Any] | None) -> bool:
    """Entity co phai hoa si khong, theo nghe (P106) va mo ta.

    ``None`` (khong co du lieu) -> khong loai, vi day chi la lop bao ve thu hai.
    """
    if not qid_info:
        return True
    occupations = qid_info.get("occupations") or []
    if occupations:
        return any(occ in ARTIST_OCCUPATIONS for occ in occupations)
    descriptions = qid_info.get("descriptions") or []
    if descriptions:
        return any(
            is_artist_description(desc) for desc in descriptions
        )
    return True


def birth_year_agrees(
    qid_info: dict[str, Any] | None,
    birth_year: int | None,
    tolerance: int = 3,
) -> bool:
    """Nam sinh tren Wikidata co khop nam sinh trong KB khong?

    Neu KB co nam sinh va Wikidata cung co, thi sai so phai <= ``tolerance``.
    Neu mot trong hai khong co -> khong co bang chung phu dinh -> chap nhan.
    """
    if not qid_info or birth_year is None:
        return True
    years = qid_info.get("years") or []
    if not years:
        return True
    return any(abs(int(year) - birth_year) <= tolerance for year in years)


def looks_like_disambiguation(extract: Any, is_disambiguation: bool = False) -> bool:
    """Trang wikipedia co phai trang "dinh nghia" (nhieu nguoi cung ten) khong.

    Vi du ``Hashim``, ``Tao Hong``, ``Le Nain``: la trang liet ke nhieu nguoi
    khac nhau, khong phai bai viet ve mot hoa si -> khong dung lam nguon.

    ``is_disambiguation`` la co ``pageprops.disambiguation`` cua Wikipedia —
    tin hieu chinh thuc, dung hon su doan van ban. Khi khong co tin hieu nay
    thi moi quay ve heuristic.

    Heuristic chi kiem **cau dau tien** cua bai: bai ve hoa si ``Bronzino``
    co cau "His sobriquet, Bronzino, **may refer to** his relatively dark
    skin" — neu tim chuoi bat ky ai thi bai hop le bi loai nham. Con trang
    dinh nghia thi luon bat dau bang cau giai thich ("X may refer to:", ...).
    """
    if is_disambiguation:
        return True
    if not isinstance(extract, str) or not extract:
        return True
    head = _first_sentence(extract[:400]).lower()
    markers = (
        "may refer to",
        "is a common",
        "is a disambiguation",
        "is a given name",
        "is a surname",
        "is a family name",
        "surname and given name",
        "given name and surname",
        "refers to",
        "disambiguation",
    )
    return any(marker in head for marker in markers)


# dau cau ket thuc cua mot cau (bo qua dau cham trong "J. M. W. Turner").
_SENTENCE_END_RE = re.compile(r"[.!?]\s+(?=[A-Z\"'(\u201c])")


def _first_sentence(text: str) -> str:
    """Phan dau tien cua van ban (den het cau dau)."""
    match = _SENTENCE_END_RE.search(text)
    return text[: match.start()] if match else text


def is_artist_description(description: Any) -> bool:
    """Mo ta Wikidata co phai noi ve hoa si khong.

    Dung de loai bo false positive khi trung ten: ``Wang Yuan`` hoa si
    (Q2889034, "Chinese painter") vs. `Wang Yuan (mathematician)`.
    """
    if not isinstance(description, str) or not description:
        return True  # khong co mo ta -> khong loai
    lowered = description.lower()
    return any(keyword in lowered for keyword in ARTIST_DESC_KEYWORDS)


def pick_batch_candidate(
    candidates: list[dict[str, Any]],
    name: str,
    birth_year: int | None = None,
    label_threshold: float = 88.0,
    birth_tolerance: int = 3,
) -> dict[str, Any] | None:
    """Chon candidate tot nhat trong ket qua SPARQL (da co QID + bai wiki).

    Uu tien: (1) entity la hoa si (theo nghe hoac mo ta), (2) khop nam sinh,
    (3) khop ten.

    Diem so sanh dung **nhan chuan** (``canonical_label``) cua entity, khong
    dung gia tri ``name`` da khop. Neu dung ``name`` thi moi candidate deu
    goc diem 100, nen khong phan biet duoc hoa si nao that su dung: vi du
    ``Zhou Chen`` (Ming, sinh 1460) bi match nham sang ``Shen Zhou`` (sinh
    1427) vi ``Shen Zhou`` co alias ``Zhou Chen``.

    Ngoai ra, candidate phai **khac nghe** voi entity da bi loai truoc do: neu
    KB quy cho ``Zhou Chen`` nam sinh 1445 ma entity la 1427 thi do lech
    18 nam -> dung nguoi khac, phai loai.
    """
    best: dict[str, Any] | None = None
    best_key: tuple[int, int, float] = (0, 0, 0.0)
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if not is_artist_description(candidate.get("description")):
            continue
        score = label_score(
            name,
            candidate.get("canonical_label")
            or candidate.get("label")
            or candidate.get("wiki_title"),
        )
        entity_year = candidate.get("birth_year")
        birth_matched = bool(
            birth_year is not None
            and entity_year is not None
            and abs(int(entity_year) - birth_year) <= birth_tolerance
        )
        qualifies = score >= label_threshold
        if not qualifies and birth_matched and score >= BIRTH_MIN_LABEL_SCORE:
            qualifies = True
        if not qualifies:
            continue
        # hoa si + khop nam sinh + khop ten
        key = (1 if birth_matched else 0, 1, score)
        if best is None or key > best_key:
            best_key = key
            best = {
                "qid": candidate.get("qid"),
                "wiki_title": candidate.get("wiki_title"),
                "wiki_url": candidate.get("wiki_url"),
                "matched_label": (
                    candidate.get("canonical_label")
                    or candidate.get("label")
                    or candidate.get("wiki_title")
                ),
                "match_score": round(score, 2),
                "birth_matched": birth_matched,
            }
    return best


# ---------------------------------------------------------------- cache
def wiki_cache_path(raw_dir: str | Path, artist_id: str) -> Path:
    """Duong dan file cache cua mot hoa si (``raw/wiki/{artist_id}.json``)."""
    return Path(raw_dir) / "wiki" / f"{artist_id}.json"


def is_cache_valid(payload: Any, artist_id: str) -> bool:
    """Cache hop le khi ``artist_id`` khop va ``status`` nam trong tap hop hop le."""
    if not isinstance(payload, dict):
        return False
    if payload.get("artist_id") != artist_id:
        return False
    return payload.get("status") in (STATUS_FOUND, STATUS_NOT_FOUND, STATUS_ERROR)


def read_cache(path: str | Path) -> Any:
    """Doc file cache; tra None neu khong ton tai hoac hong."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def build_record(
    artist: dict[str, Any],
    status: str,
    qid: str | None = None,
    wiki_title: str | None = None,
    wiki_url: str | None = None,
    wiki_summary: str | None = None,
    matched_label: str | None = None,
    match_score: float | None = None,
    birth_matched: bool = False,
    cma_bio: str | None = None,
) -> dict[str, Any]:
    """Tao payload cache cho mot hoa si."""
    return {
        "artist_id": artist.get("artist_id"),
        "name": artist.get("artist_name"),
        "birth_year": artist.get("artist_birth"),
        "status": status,
        "wikidata_qid": qid,
        "wiki_title": wiki_title,
        "wiki_url": wiki_url,
        "wiki_summary": wiki_summary,
        "matched_label": matched_label,
        "match_score": match_score,
        "birth_matched": birth_matched,
        "cma_bio": cma_bio,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------- inputs
def load_artists(cfg: Config) -> list[dict[str, Any]]:
    """Doc KB, lay cac hoa si duy nhat (dedup theo ``artist_id``)."""
    p = paths(cfg)
    kb_path = p.kb / "artworks.parquet"
    if not kb_path.exists():
        return []
    frame = pd.read_parquet(kb_path, columns=[
        "artist_id",
        "artist_name",
        "artist_birth",
        "artist_death",
        "artist_nationality",
    ])
    artists: dict[str, dict[str, Any]] = {}
    for row in frame.itertuples(index=False):
        artist_id = row.artist_id
        if artist_id is None or pd.isna(artist_id):
            continue
        artist_id = str(artist_id)
        if artist_id in artists:
            continue
        birth_value = row.artist_birth
        death_value = row.artist_death
        artists[artist_id] = {
            "artist_id": artist_id,
            "artist_name": row.artist_name,
            "artist_birth": None if pd.isna(birth_value) else int(birth_value),
            "artist_death": None if pd.isna(death_value) else int(death_value),
            "artist_nationality": row.artist_nationality,
        }
    return list(artists.values())


def load_cma_bios(raw_dir: str | Path) -> dict[str, str]:
    """Map ``artist_id -> creators[].biography`` (da strip HTML) tu raw CMA."""
    cma_dir = Path(raw_dir) / "cma"
    if not cma_dir.exists():
        return {}
    bios: dict[str, str] = {}
    for path in sorted(cma_dir.glob("page_*.json")):
        payload = read_cache(path)
        if not isinstance(payload, dict):
            continue
        records = payload.get("data")
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            creators = record.get("creators")
            if not isinstance(creators, list):
                continue
            for creator in creators:
                if not isinstance(creator, dict):
                    continue
                creator_id = creator.get("id")
                biography = creator.get("biography")
                if creator_id is None or not biography:
                    continue
                text = strip_html(str(biography))
                if text:
                    bios.setdefault(f"cma-artist-{creator_id}", text)
    return bios


# ngon ngu dua tren he chu duoc dung khi tra cuu Wikidata.
_LANG_BY_SCRIPT = (
    ("\u3040", "\u30ff", "ja"),   # hiragana + katakana -> Nhat
    ("\u4e00", "\u9fff", "zh"),   # han tu -> Trung
    ("\uac00", "\ud7af", "ko"),   # hangul -> Han Quoc
)


def guess_script_language(name: str) -> str:
    """Doan ngon ngu cua mot ten theo he chu.

    Ten ``Kano Tan’yū`` (latin, dau) -> ``en``; ``月岡雪鼎`` -> ``ja``;
    ``林良`` -> ``zh``. Dung cho ``name_in_original_language`` cua CMA.
    """
    if not name:
        return "en"
    for start, end, lang in _LANG_BY_SCRIPT:
        if any(start <= ch <= end for ch in name):
            return lang
    return "en"


def load_cma_original_names(raw_dir: str | Path) -> dict[str, list[str]]:
    """Map ``artist_id -> [ten goc]`` tu ``creators[].name_in_original_language``.

    Nhieu hoa si Trung Quoc/Nhat tren CMA co ten Latin trong KB (vi du
    ``Lin Liang``) nhung nhan Wikidata la chu Han (``林良``). Dung ten goc de
    tra cuu them voi nhan dung ngon ngu -> nhieu hoa si tim thay hon.
    """
    cma_dir = Path(raw_dir) / "cma"
    if not cma_dir.exists():
        return {}
    names: dict[str, list[str]] = {}
    for path in sorted(cma_dir.glob("page_*.json")):
        payload = read_cache(path)
        if not isinstance(payload, dict):
            continue
        records = payload.get("data")
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            creators = record.get("creators")
            if not isinstance(creators, list):
                continue
            for creator in creators:
                if not isinstance(creator, dict):
                    continue
                creator_id = creator.get("id")
                original = creator.get("name_in_original_language")
                if creator_id is None or not original:
                    continue
                text = str(original).strip()
                if not text:
                    continue
                bucket = names.setdefault(f"cma-artist-{creator_id}", [])
                if text not in bucket:
                    bucket.append(text)
    return names


# ---------------------------------------------------------------- network
class RateLimiter:
    """Dam bao giua hai request lien nhau co it nhat ``min_interval_s`` giay."""

    def __init__(
        self,
        min_interval_s: float = 0.35,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.min_interval_s = float(min_interval_s)
        self._sleep = sleep
        self._monotonic = monotonic
        self._last: float | None = None

    def wait(self) -> None:
        if self.min_interval_s <= 0:
            return
        now = self._monotonic()
        if self._last is not None:
            delta = self.min_interval_s - (now - self._last)
            if delta > 0:
                self._sleep(delta)
                now = self._monotonic()
        self._last = now


def _retry_after(response: Any) -> float | None:
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    try:
        raw = headers.get("Retry-After")
    except AttributeError:
        return None
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def http_get_json(
    session: Any,
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout_s: float = 30.0,
    max_retries: int = 4,
    backoff_s: float = 3.0,
    limiter: RateLimiter | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """GET JSON co retry. Nem ``HttpNotFound`` khi 404, ``RuntimeError`` khi loi."""
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        if limiter is not None:
            limiter.wait()
        try:
            response = session.get(url, params=params, headers=headers, timeout=timeout_s)
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            sleep(backoff_s * (2**attempt))
            continue

        status = getattr(response, "status_code", 200)
        if status == 200:
            return response.json()
        if status == 404:
            raise HttpNotFound(f"404 {url}")
        if status == 429 or (isinstance(status, int) and status >= 500):
            last_error = RuntimeError(f"HTTP {status} cho {url}")
            if attempt >= max_retries:
                break
            delay = backoff_s * (2**attempt)
            hinted = _retry_after(response)
            if hinted is not None:
                delay = max(delay, hinted)
            sleep(delay)
            continue
        raise RuntimeError(f"HTTP {status} cho {url}")

    raise RuntimeError(f"Khong goi duoc {url}: {last_error}")


def search_entities(
    session: Any,
    name: str,
    search_limit: int = 10,
    timeout_s: float = 30.0,
    max_retries: int = 4,
    backoff_s: float = 3.0,
    limiter: RateLimiter | None = None,
    sleep: Callable[[float], None] = time.sleep,
    user_agent: str | None = None,
) -> list[dict[str, Any]]:
    """Tim candidate tren Wikidata bang ten hoa si."""
    params = {
        "action": "wbsearchentities",
        "search": name,
        "language": "en",
        "uselang": "en",
        "format": "json",
        "type": "item",
        "limit": int(search_limit),
    }
    payload = http_get_json(
        session,
        WIKIDATA_API,
        params=params,
        headers={"User-Agent": user_agent} if user_agent else None,
        timeout_s=timeout_s,
        max_retries=max_retries,
        backoff_s=backoff_s,
        limiter=limiter,
        sleep=sleep,
    )
    if not isinstance(payload, dict):
        return []
    results = payload.get("search")
    if not isinstance(results, list):
        return []
    return [item for item in results if isinstance(item, dict)]


def fetch_entities(
    session: Any,
    qids: list[str],
    timeout_s: float = 30.0,
    max_retries: int = 4,
    backoff_s: float = 3.0,
    limiter: RateLimiter | None = None,
    sleep: Callable[[float], None] = time.sleep,
    user_agent: str | None = None,
) -> dict[str, Any]:
    """Lay thuoc tinh cua nhieu QID trong mot request."""
    if not qids:
        return {}
    params = {
        "action": "wbgetentities",
        "ids": "|".join(qids),
        "props": "claims|sitelinks|labels|descriptions",
        "languages": "en",
        "format": "json",
    }
    payload = http_get_json(
        session,
        WIKIDATA_API,
        params=params,
        headers={"User-Agent": user_agent} if user_agent else None,
        timeout_s=timeout_s,
        max_retries=max_retries,
        backoff_s=backoff_s,
        limiter=limiter,
        sleep=sleep,
    )
    if not isinstance(payload, dict):
        return {}
    entities = payload.get("entities")
    if not isinstance(entities, dict):
        return {}
    return entities


def fetch_summary(
    session: Any,
    title: str,
    timeout_s: float = 30.0,
    max_retries: int = 4,
    backoff_s: float = 3.0,
    limiter: RateLimiter | None = None,
    sleep: Callable[[float], None] = time.sleep,
    user_agent: str | None = None,
) -> dict[str, Any]:
    """Lay tom tat bai wikipedia qua REST API."""
    payload = http_get_json(
        session,
        WIKIPEDIA_SUMMARY + str(title).replace(" ", "_"),
        headers={"User-Agent": user_agent} if user_agent else None,
        timeout_s=timeout_s,
        max_retries=max_retries,
        backoff_s=backoff_s,
        limiter=limiter,
        sleep=sleep,
    )
    return payload if isinstance(payload, dict) else {}


def _summary_url(payload: dict[str, Any]) -> str | None:
    urls = payload.get("content_urls")
    if isinstance(urls, dict):
        desktop = urls.get("desktop")
        if isinstance(desktop, dict):
            page = desktop.get("page")
            if isinstance(page, str) and page:
                return page
    return None


# ---------------------------------------------------------------- pipeline
def enrich_artist(
    artist: dict[str, Any],
    session: Any,
    limiter: RateLimiter | None = None,
    search_limit: int = 10,
    label_threshold: float = 88.0,
    birth_tolerance: int = 3,
    timeout_s: float = 30.0,
    max_retries: int = 4,
    backoff_s: float = 3.0,
    sleep: Callable[[float], None] = time.sleep,
    user_agent: str | None = None,
    cma_bio: str | None = None,
) -> dict[str, Any]:
    """Tra cuu mot hoa si; KHONG nem loi (loi -> ``status="error"``)."""
    artist_id = artist.get("artist_id")
    name = artist.get("artist_name")
    birth_year = artist.get("artist_birth")

    if not name:
        return build_record(artist, STATUS_NOT_FOUND, cma_bio=cma_bio)

    try:
        candidates = search_entities(
            session,
            str(name),
            search_limit=search_limit,
            timeout_s=timeout_s,
            max_retries=max_retries,
            backoff_s=backoff_s,
            limiter=limiter,
            sleep=sleep,
            user_agent=user_agent,
        )
        if not candidates:
            return build_record(artist, STATUS_NOT_FOUND, cma_bio=cma_bio)

        qids = [
            item["id"]
            for item in candidates
            if isinstance(item.get("id"), str) and item.get("id")
        ]
        entities = fetch_entities(
            session,
            qids,
            timeout_s=timeout_s,
            max_retries=max_retries,
            backoff_s=backoff_s,
            limiter=limiter,
            sleep=sleep,
            user_agent=user_agent,
        )
        best = pick_candidate(
            candidates,
            entities,
            str(name),
            birth_year=birth_year,
            label_threshold=label_threshold,
            birth_tolerance=birth_tolerance,
        )
        if best is None:
            return build_record(artist, STATUS_NOT_FOUND, cma_bio=cma_bio)

        summary_payload = fetch_summary(
            session,
            best["wiki_title"],
            timeout_s=timeout_s,
            max_retries=max_retries,
            backoff_s=backoff_s,
            limiter=limiter,
            sleep=sleep,
            user_agent=user_agent,
        )
        summary = summary_payload.get("extract")
        summary_text = strip_html(str(summary)) if summary else None
        url = _summary_url(summary_payload)
        if not url:
            url = f"https://en.wikipedia.org/wiki/{str(best['wiki_title']).replace(' ', '_')}"

        return build_record(
            artist,
            STATUS_FOUND,
            qid=best["qid"],
            wiki_title=best["wiki_title"],
            wiki_url=url,
            wiki_summary=summary_text,
            matched_label=best["matched_label"],
            match_score=best["match_score"],
            birth_matched=best["birth_matched"],
            cma_bio=cma_bio,
        )
    except HttpNotFound:
        return build_record(artist, STATUS_NOT_FOUND, cma_bio=cma_bio)
    except Exception:  # noqa: BLE001 - khong bao gio nem loi ra ngoai
        return build_record(artist, STATUS_ERROR, cma_bio=cma_bio)


def _resolve_with_split(
    session: Any,
    names: list[str] | list[tuple[str, str]],
    timeout_s: float,
    max_retries: int,
    backoff_s: float,
    limiter: RateLimiter | None,
    user_agent: str | None,
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Giai ten theo lo; neu lo that bai thi tach nho hon va thu lai.

    Query SPARQL co the bi timeout khi nhieu ten kho; tach nho giup "dau
    van" thay vi mat ca lo. Tra list ``(ten, candidates)``.
    """
    if not names:
        return []
    try:
        grouped = resolve_names_batch(
            session,
            names,
            timeout_s=timeout_s,
            max_retries=max_retries,
            backoff_s=backoff_s,
            limiter=limiter,
            user_agent=user_agent,
        )
        return list(grouped.items())
    except Exception as exc:  # noqa: BLE001
        if len(names) == 1:
            print(f"  ! SPARQL that bai cho 1 ten ({names[0]}): {exc}", flush=True)
            return []
        middle = len(names) // 2
        print(
            f"  ! SPARQL that bai ({len(names)} ten), tach nho va thu lai: {exc}",
            flush=True,
        )
        out = _resolve_with_split(
            session, names[:middle], timeout_s, max_retries, backoff_s, limiter, user_agent
        )
        out.extend(
            _resolve_with_split(
                session, names[middle:], timeout_s, max_retries, backoff_s, limiter, user_agent
            )
        )
        return out


def enrich_all(
    cfg: Config,
    limit: int | None = None,
    force: bool = False,
    session: Any | None = None,
) -> dict[str, Any]:
    """Lam giau tat ca hoa si trong KB; tra ve thong ke.

    Mac dinh dung **che do batch** (SPARQL + Wikipedia ``action=query``) vi
    Wikimedia chan anonymous API o ~10 request/burst; che do ``single`` (goi
    tung artist) chi de so sanh va de debug.
    """
    p = paths(cfg, ensure=True)
    artists = load_artists(cfg)
    if limit is not None:
        artists = artists[:limit]
    if not artists:
        return {
            "total": 0,
            "cached": 0,
            "fetched": 0,
            "found": 0,
            "not_found": 0,
            "error": 0,
            "hit_rate": 0.0,
            "examples": [],
        }

    bios = load_cma_bios(p.raw)

    search_limit = int(_cfg_get(cfg, "wiki_search_limit", 10))
    label_threshold = float(_cfg_get(cfg, "wiki_label_threshold", 88.0))
    birth_tolerance = int(_cfg_get(cfg, "wiki_birth_tolerance", 3))
    min_interval = float(_cfg_get(cfg, "wiki_min_interval_s", 0.35))
    timeout_s = float(_cfg_get(cfg, "wiki_timeout_s", 30.0))
    max_retries = int(_cfg_get(cfg, "wiki_max_retries", 4))
    backoff_s = float(_cfg_get(cfg, "wiki_backoff_s", 3.0))
    user_agent = str(_cfg_get(cfg, "wiki_user_agent", "artrag/0.1"))
    batch_size = int(_cfg_get(cfg, "wiki_batch_size", 25))
    extract_batch_size = int(_cfg_get(cfg, "wiki_extract_batch_size", 30))
    progress_every = int(_cfg_get(cfg, "wiki_progress_every", 25))
    mode = str(_cfg_get(cfg, "wiki_mode", "batch"))

    limiter = RateLimiter(min_interval_s=min_interval)
    owns_session = session is None
    if session is None:
        session = requests.Session()

    cached = 0
    fetched = 0
    statuses: dict[str, int] = {STATUS_FOUND: 0, STATUS_NOT_FOUND: 0, STATUS_ERROR: 0}
    examples: list[dict[str, Any]] = []

    def _record_status(record: dict[str, Any]) -> None:
        nonlocal fetched
        statuses[record["status"]] = statuses.get(record["status"], 0) + 1
        if record["status"] == STATUS_FOUND and len(examples) < 3:
            examples.append(record)

    try:
        if mode == "single":
            for artist in artists:
                artist_id = artist["artist_id"]
                path = wiki_cache_path(p.raw, artist_id)
                payload = None if force else read_cache(path)
                if is_cache_valid(payload, artist_id):
                    cached += 1
                    _record_status(payload)
                    continue
                record = enrich_artist(
                    artist,
                    session,
                    limiter=limiter,
                    search_limit=search_limit,
                    label_threshold=label_threshold,
                    birth_tolerance=birth_tolerance,
                    timeout_s=timeout_s,
                    max_retries=max_retries,
                    backoff_s=backoff_s,
                    user_agent=user_agent,
                    cma_bio=bios.get(artist_id),
                )
                write_json(path, record)
                fetched += 1
                _record_status(record)
        else:
            # 1) bo qua artist da co cache hop le
            pending: list[dict[str, Any]] = []
            for artist in artists:
                artist_id = artist["artist_id"]
                path = wiki_cache_path(p.raw, artist_id)
                payload = None if force else read_cache(path)
                if is_cache_valid(payload, artist_id):
                    cached += 1
                    _record_status(payload)
                    continue
                pending.append(artist)

            # 2) giai ten theo lo (SPARQL), gom ca artist trung ten.
            #    Moi artist duoc tra cuu bang nhieu bien the ten:
            #    ten KB (`@en` + `@mul`) va ten goc cua CMA (`@ja`/`@zh`/...).
            #    Can phai vay vi nhan Wikidata cua hoa si Nhat/Trung thuong khong
            #    co o `en` — chi kiem `en` se bo qua ~120 hoa si.
            original_names = load_cma_original_names(p.raw)
            artist_terms: dict[str, list[tuple[str, str]]] = {}
            for artist in pending:
                name = str(artist.get("artist_name") or "")
                if not name:
                    continue
                terms: list[tuple[str, str]] = [(name, "en"), (name, "mul")]
                for original in original_names.get(artist["artist_id"], []):
                    lang = guess_script_language(original)
                    pair = (original, lang)
                    if pair not in terms:
                        terms.append(pair)
                artist_terms[artist["artist_id"]] = terms

            resolved: dict[str, list[dict[str, Any]]] = {}
            for start in range(0, len(pending), batch_size):
                chunk = pending[start : start + batch_size]
                terms: list[tuple[str, str]] = []
                seen_terms: set[tuple[str, str]] = set()
                for artist in chunk:
                    for pair in artist_terms.get(artist["artist_id"], []):
                        if pair not in seen_terms:
                            seen_terms.add(pair)
                            terms.append(pair)
                for resolved_names, grouped in _resolve_with_split(
                    session, terms, timeout_s, max_retries, backoff_s, limiter, user_agent
                ):
                    resolved.setdefault(resolved_names, []).extend(grouped)
                if progress_every > 0 and (start // batch_size + 1) % 5 == 0:
                    print(
                        f"  ... SPARQL {start + len(chunk)}/{len(pending)} "
                        f"ten da giai: {len(resolved)}",
                        flush=True,
                    )

            # 3) lay tom tat wikipedia theo lo
            titles = sorted(
                {
                    str(c["wiki_title"])
                    for candidates in resolved.values()
                    for c in candidates
                    if c.get("wiki_title")
                }
            )
            extracts: dict[str, dict[str, Any]] = {}
            for start in range(0, len(titles), extract_batch_size):
                chunk = titles[start : start + extract_batch_size]
                try:
                    extracts.update(
                        fetch_extracts_batch(
                            session,
                            chunk,
                            timeout_s=timeout_s,
                            max_retries=max_retries,
                            backoff_s=backoff_s,
                            limiter=limiter,
                            user_agent=user_agent,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"  ! Wikipedia that bai ({start // extract_batch_size + 1}): {exc}",
                        flush=True,
                    )
                    chunk = []
                if progress_every > 0 and (start // extract_batch_size + 1) % 5 == 0:
                    print(
                        f"  ... Wikipedia {start + len(chunk)}/{len(titles)} bai",
                        flush=True,
                    )

            # 3b) bai chua co extract (API co gioi han `exlimit`) -> fetch lai tung bai
            missing = [t for t in titles if not (extracts.get(t) or {}).get("extract")]
            if missing:
                print(f"  ... fetch lai {len(missing)} bai thieu extract", flush=True)
                for title in missing:
                    try:
                        found = fetch_enwiki_by_titles(
                            session,
                            [title],
                            timeout_s=timeout_s,
                            max_retries=1,
                            backoff_s=backoff_s,
                            limiter=limiter,
                            user_agent=user_agent,
                        )
                        if found.get(title):
                            extracts[title] = found[title]
                        else:
                            # ten da bi chuyen huong -> API tra ten chuan
                            for page in found.values():
                                if page.get("extract"):
                                    extracts[title] = page
                                    break
                    except Exception:  # noqa: BLE001
                        continue

            # 4) duong dan du phong: tra cuu enwiki theo TEN BAI khi SPARQL that bai
            #    (hoa si Trung Quoc: nhan Wikidata la chu Han, bai enwiki la tieng Anh)
            def _candidates_for(artist: dict[str, Any]) -> list[dict[str, Any]]:
                """Gop candidate cua tat ca bien the ten cua mot artist."""
                out: list[dict[str, Any]] = []
                seen: set[str] = set()
                for term in artist_terms.get(artist["artist_id"], []):
                    for candidate in resolved.get(term[0], []):
                        qid = str(candidate.get("qid"))
                        if qid in seen:
                            continue
                        seen.add(qid)
                        out.append(candidate)
                return out

            unresolved = [
                a
                for a in pending
                if not pick_batch_candidate(
                    _candidates_for(a),
                    str(a.get("artist_name") or ""),
                    birth_year=a.get("artist_birth"),
                    label_threshold=label_threshold,
                    birth_tolerance=birth_tolerance,
                )
            ]
            if unresolved:
                fallback_titles = sorted(
                    {
                        str(a["artist_name"])
                        for a in unresolved
                        if a.get("artist_name")
                    }
                )
                fallback_pages: list[tuple[str, dict[str, Any]]] = []
                for start in range(0, len(fallback_titles), extract_batch_size):
                    chunk = fallback_titles[start : start + extract_batch_size]
                    try:
                        found = fetch_enwiki_by_titles(
                            session,
                            chunk,
                            timeout_s=timeout_s,
                            max_retries=max_retries,
                            backoff_s=backoff_s,
                            limiter=limiter,
                            user_agent=user_agent,
                        )
                    except Exception as exc:  # noqa: BLE001
                        print(
                            f"  ! enwiki fallback that bai "
                            f"({start // extract_batch_size + 1}): {exc}",
                            flush=True,
                        )
                        continue
                    for title, page in found.items():
                        if not page.get("extract"):
                            continue
                        # chi dung key co thuc la ten da hoi (bo qua ten chuan sau redirect)
                        if title not in fallback_titles:
                            continue
                        if looks_like_disambiguation(
                            page["extract"], page.get("is_disambiguation", False)
                        ):
                            continue
                        extracts[title] = page
                        fallback_pages.append((title, page))
                    print(
                        f"  ... enwiki fallback: {len(unresolved)} artist chua khop, "
                        f"{len(fallback_titles)} ten da thu, "
                        f"{len(fallback_pages)} bai co extract",
                        flush=True,
                    )

                # 4b) kiem tra nghe cua cac bai fallback: chi nhan hoa si that su.
                #     Day la lop bao ve chong false positive (vi du "Muhammad Ali"
                #     la võ sinh, khong phai hoa si Mughal).
                if fallback_pages:
                    qids = sorted(
                        {
                            str(page.get("qid"))
                            for _, page in fallback_pages
                            if page.get("qid")
                        }
                    )
                    occ_info: dict[str, dict[str, Any]] = {}
                    for start in range(0, len(qids), 40):
                        chunk = qids[start : start + 40]
                        try:
                            occ_info.update(
                                fetch_occupations_batch(
                                    session,
                                    chunk,
                                    timeout_s=timeout_s,
                                    max_retries=max_retries,
                                    backoff_s=backoff_s,
                                    limiter=limiter,
                                    user_agent=user_agent,
                                )
                            )
                        except Exception as exc:  # noqa: BLE001
                            print(f"  ! kiem tra nghe that bai: {exc}", flush=True)
                    rejected = 0
                    birth_map = {
                        str(a["artist_name"]): a.get("artist_birth")
                        for a in unresolved
                        if a.get("artist_name")
                    }
                    for title, page in fallback_pages:
                        qid = page.get("qid")
                        if not qid:
                            continue
                        info = occ_info.get(str(qid))
                        if not is_artist_qid(info):
                            rejected += 1
                            resolved.get(title, []).clear()
                            continue
                        if not birth_year_agrees(
                            info,
                            birth_map.get(title),
                            birth_tolerance,
                        ):
                            rejected += 1
                            resolved.get(title, []).clear()
                            continue
                        resolved.setdefault(title, []).append(
                            {
                                "qid": qid,
                                "wiki_title": title,
                                "wiki_url": page.get("fullurl"),
                                "birth_year": first_birth_year(info),
                                "label": title,
                                "from_enwiki": True,
                            }
                        )
                    print(
                        f"  ... fallback occupation check: {rejected} bai bi loai",
                        flush=True,
                    )

            # 5) ghi cache tung artist
            for artist in pending:
                artist_id = artist["artist_id"]
                name = artist.get("artist_name")
                best = pick_batch_candidate(
                    _candidates_for(artist),
                    str(name or ""),
                    birth_year=artist.get("artist_birth"),
                    label_threshold=label_threshold,
                    birth_tolerance=birth_tolerance,
                )
                if best is None:
                    record = build_record(
                        artist, STATUS_NOT_FOUND, cma_bio=bios.get(artist_id)
                    )
                else:
                    page = extracts.get(str(best["wiki_title"]), {})
                    if not page.get("extract"):
                        page = extracts.get(str(best.get("matched_label") or ""), {})
                    summary = page.get("extract")
                    if summary and looks_like_disambiguation(
                        summary, page.get("is_disambiguation", False)
                    ):
                        # trang dinh nghia -> khong dung lam tieu su hoa si
                        summary = None
                        record = build_record(
                            artist,
                            STATUS_NOT_FOUND,
                            cma_bio=bios.get(artist_id),
                        )
                        write_json(wiki_cache_path(p.raw, artist_id), record)
                        fetched += 1
                        _record_status(record)
                        continue
                    url = page.get("fullurl") or best.get("wiki_url")
                    if not url:
                        url = (
                            "https://en.wikipedia.org/wiki/"
                            f"{quote(str(best['wiki_title']).replace(' ', '_'))}"
                        )
                    record = build_record(
                        artist,
                        STATUS_FOUND,
                        qid=best["qid"],
                        wiki_title=best["wiki_title"],
                        wiki_url=url,
                        wiki_summary=summary,
                        matched_label=best["matched_label"],
                        match_score=best["match_score"],
                        birth_matched=best["birth_matched"],
                        cma_bio=bios.get(artist_id),
                    )
                write_json(wiki_cache_path(p.raw, artist_id), record)
                fetched += 1
                _record_status(record)

            if progress_every > 0 and pending:
                print(
                    f"  ... resolved {len(pending)} artist "
                    f"(found {statuses[STATUS_FOUND]} | "
                    f"not_found {statuses[STATUS_NOT_FOUND]} | "
                    f"error {statuses[STATUS_ERROR]})",
                    flush=True,
                )
    finally:
        if owns_session:
            session.close()

    total = len(artists)
    found = statuses.get(STATUS_FOUND, 0)
    return {
        "total": total,
        "cached": cached,
        "fetched": fetched,
        "found": found,
        "not_found": statuses.get(STATUS_NOT_FOUND, 0),
        "error": statuses.get(STATUS_ERROR, 0),
        "hit_rate": (found / total) if total else 0.0,
        "examples": examples,
    }


# ---------------------------------------------------------------- CLI
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m artrag.data.enrich_wiki",
        description="Lam giau tieu su hoa si tu Wikidata/Wikipedia (cache raw/wiki).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="chi xu ly N hoa si dau (dung de thu nhanh)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="truy xuat lai ke ca da co cache",
    )
    parser.add_argument("--profile", default=None, help="profile config (mac dinh: local)")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="ghi de config, vi du --set data.wiki_min_interval_s=0.5",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(profile=args.profile, overrides=args.overrides)
    stats = enrich_all(cfg, limit=args.limit, force=args.force)

    print(
        f"enrich_wiki: {stats['found']}/{stats['total']} found "
        f"({stats['hit_rate']:.1%}) | cached {stats['cached']} | "
        f"fetched {stats['fetched']} | not_found {stats['not_found']} | "
        f"error {stats['error']}"
    )

    examples = stats.get("examples") or []
    if examples:
        print("\n-- vi du --")
        for item in examples:
            print(f"{item.get('name')} -> {item.get('wiki_url')}")

    if stats["total"] and stats["hit_rate"] < 0.6 and not args.limit:
        print("CANH BAO: ty le tim thay duoi 60%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
