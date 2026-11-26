"""T1.2: parse bản ghi CMA thành schema ``artworks`` của KB (SYSTEM_DESIGN 2.3).

Các quy tắc parse chi tiết (prev_owner, provenance_text, exhibitions_text,
overview_text, cách tách tên họa sĩ) được mô tả trong SYSTEM_DESIGN 2.3
"Quy tắc parse cụ thể". Hàm ở đây đều thuần, test được không cần mạng.
"""

from __future__ import annotations

import re
from typing import Any

from artrag.utils.text import strip_html

# dấu phân cách năm: "1873-1916", "c. 1629", "1815"
_YEAR_RE = re.compile(r"(?<![\d.])(\d{3,4})(?![\d.])")
_BORN_RE = re.compile(r"^(born|b\.)\b", re.IGNORECASE)
_APPROX_RE = re.compile(r"\b(?:c\.?|circa|ca\.)\b", re.IGNORECASE)
_CMA_MARKER = "cleveland museum of art"


# ---------------------------------------------------------------- helpers
def clean_text(value: Any) -> str:
    """Chuẩn hóa về chuỗi: bỏ HTML, cắt khoảng trắng thừa, null -> ""."""
    if value is None:
        return ""
    return strip_html(str(value))


def _parse_year(token: str) -> int | None:
    """Token năm -> int; bỏ qua 'c.'/'circa' và năm sai định dạng."""
    match = _YEAR_RE.search(token or "")
    if not match:
        return None
    year = int(match.group(1))
    if year < 1000 or year > 2100:
        return None
    return year


def parse_artist_description(description: str) -> dict[str, Any]:
    """Tách ``"<Tên> (<Quốc tịch>, <năm sinh>–<năm tử>)"``.

    Ví dụ:
    - ``"Thomas Eakins (American, 1844–1916)"`` -> name/nationality/birth/death
    - ``"Willem Kalf (Dutch, c. 1629–1684)"`` -> birth = None (có "c.")
    - ``"Thomas Cole (American, born England,1801–1848)"`` -> nationality "American"
    """
    result: dict[str, Any] = {
        "name": None,
        "nationality": None,
        "birth": None,
        "death": None,
    }
    text = clean_text(description)
    if not text:
        return result

    if "(" in text and ")" in text:
        head, _, tail = text.partition("(")
        tail = tail.rsplit(")", 1)[0]
        result["name"] = head.strip() or None
        parts = [p.strip() for p in tail.split(",") if p.strip()]
    else:
        result["name"] = text.strip() or None
        parts = []

    if parts:
        # phần đầu là quốc tịch; cắt "born ..." / "b. ..." ở đuôi
        nationality = parts[0]
        born_match = _BORN_RE.match(nationality)
        if born_match:
            nationality = ""
        nationality = nationality.strip()
        result["nationality"] = nationality or None
        # phần còn là "năm sinh–năm tử" (có thể có "born England" chen giữa)
        year_parts = [p for p in parts[1:] if not _BORN_RE.match(p) or _YEAR_RE.search(p)]
        years: list[tuple[int, bool]] = []  # (năm, không chắc chắn)
        for part in year_parts:
            matches = list(_YEAR_RE.finditer(part))
            approx_positions = [m.start() for m in _APPROX_RE.finditer(part)]
            prev_end = 0
            for match in matches:
                # "c." / "circa" chỉ tác động cho năm ngay sau nó
                uncertain = any(prev_end <= pos < match.start() for pos in approx_positions)
                years.append((int(match.group(1)), uncertain))
                prev_end = match.end()
        if years:
            result["birth"] = None if years[0][1] else years[0][0]
        if len(years) > 1:
            result["death"] = None if years[1][1] else years[1][0]

    return result


def pick_primary_artist(record: dict[str, Any]) -> dict[str, Any] | None:
    """Chọn creator có ``role == "artist"`` đầu tiên; None nếu không có."""
    creators = record.get("creators")
    if not isinstance(creators, list):
        return None
    fallback: dict[str, Any] | None = None
    for creator in creators:
        if not isinstance(creator, dict):
            continue
        if creator.get("role") == "artist":
            return creator
        if fallback is None and creator.get("description"):
            fallback = creator
    return fallback


def artist_info(record: dict[str, Any]) -> dict[str, Any]:
    """Thông tin họa sĩ chính của bản ghi (null-safe)."""
    creator = pick_primary_artist(record)
    if creator is None:
        return {
            "artist_id": None,
            "artist_name": None,
            "artist_role": None,
            "artist_nationality": None,
            "artist_birth": None,
            "artist_death": None,
        }

    parsed = parse_artist_description(str(creator.get("description") or ""))
    creator_id = creator.get("id")
    birth = creator.get("birth_year")
    death = creator.get("death_year")

    return {
        "artist_id": f"cma-artist-{creator_id}" if creator_id is not None else None,
        "artist_name": parsed["name"],
        "artist_role": creator.get("role"),
        "artist_nationality": parsed["nationality"],
        # ưu tiên năm số từ API, fallback về năm tách từ description
        "artist_birth": _parse_year(str(birth)) if birth else parsed["birth"],
        "artist_death": _parse_year(str(death)) if death else parsed["death"],
    }


def provenance_text(entries: Any) -> str:
    """Ghép các entry provenance thành ``"date: description"`` nối bằng "; "."""
    if not isinstance(entries, list):
        return ""
    parts: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        description = clean_text(entry.get("description"))
        date = clean_text(entry.get("date"))
        if not description:
            continue
        parts.append(f"{date}: {description}" if date else description)
    return "; ".join(parts)


def prev_owner(entries: Any) -> str | None:
    """Chủ sở hữu ngay trước CMA: entry ngay trước entry CMA cuối cùng.

    Lấy phần tên trước dấu ``[`` hoặc dấu ``,`` đầu tiên. ``None`` nếu entry
    CMA là entry đầu tiên.
    """
    if not isinstance(entries, list):
        return None
    cma_index: int | None = None
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        description = clean_text(entry.get("description"))
        if _CMA_MARKER in description.lower():
            cma_index = index
    if cma_index is None or cma_index == 0:
        return None

    previous = entries[cma_index - 1]
    if not isinstance(previous, dict):
        return None
    description = clean_text(previous.get("description"))
    if not description:
        return None

    # cắt phần trong ngoặc vuông trước (chứa năm sinh) hoặc trước dấu phẩy
    name = description.split("[", 1)[0]
    name = name.split(",", 1)[0]
    name = name.strip(" .;:-")
    name = name.lstrip("(").strip()
    return name or None


def exhibition_items(exhibitions: Any, max_items: int = 10) -> list[dict[str, Any]]:
    """Danh sách triển lãm hiện tại, mới nhất trước, tối đa ``max_items`` mục."""
    if not isinstance(exhibitions, dict):
        return []
    current = exhibitions.get("current")
    if not isinstance(current, list):
        return []

    items: list[dict[str, Any]] = []
    for entry in current:
        if not isinstance(entry, dict):
            continue
        title = clean_text(entry.get("title"))
        description = clean_text(entry.get("description"))
        if title and description:
            text = f"{title} ({description})"
        else:
            text = title or description
        if not text:
            continue
        items.append(
            {
                "text": text,
                "title": title or None,
                "opening_date": str(entry.get("opening_date") or ""),
            }
        )

    items.sort(key=lambda item: item["opening_date"], reverse=True)
    return items[:max_items]


def exhibitions_text(exhibitions: Any, max_items: int = 10) -> str:
    """Chuỗi triển lãm, nối bằng "; " (xem ``exhibition_items`` cho dạng dữ liệu)."""
    return "; ".join(item["text"] for item in exhibition_items(exhibitions, max_items))


def overview_text(row: dict[str, Any]) -> str:
    """Chuỗi tổng quan theo template §2.3 (bỏ mục rỗng, thêm "(on view)")."""
    artist = row.get("artist_name")
    if artist:
        nationality = row.get("artist_nationality")
        birth = row.get("artist_birth")
        death = row.get("artist_death")
        if nationality and birth and death:
            artist_label = f"{artist} ({nationality}, {birth}–{death})"
        elif nationality and birth:
            artist_label = f"{artist} ({nationality}, {birth})"
        elif nationality:
            artist_label = f"{artist} ({nationality})"
        else:
            artist_label = str(artist)
    else:
        artist_label = ""

    location = row.get("current_location")
    location_label = f"{location} (on view)" if location else ""

    fields = [
        ("Title", row.get("title")),
        ("Artist", artist_label),
        ("Date", row.get("creation_date")),
        ("Medium", row.get("technique")),
        ("Culture", row.get("culture")),
        ("Department", row.get("department")),
        ("Location", location_label),
        ("Credit line", row.get("credit_line")),
    ]
    parts = [f"{label}: {value}" for label, value in fields if value]
    return " | ".join(parts)


def _culture_text(culture: Any) -> str:
    if isinstance(culture, list):
        return "; ".join(clean_text(c) for c in culture if clean_text(c))
    return clean_text(culture)


def parse_artwork(record: dict[str, Any]) -> dict[str, Any]:
    """Bản ghi CMA -> dict dòng của ``artworks.parquet`` (đúng cột §2.3)."""
    cma_id = record.get("id")
    accession = record.get("accession_number")
    images = record.get("images")
    image_url = None
    if isinstance(images, dict):
        web = images.get("web")
        if isinstance(web, dict):
            image_url = web.get("url")

    source_url = record.get("url") or (
        f"https://www.clevelandart.org/art/{accession}" if accession else None
    )

    row: dict[str, Any] = {
        "artwork_id": f"cma-{cma_id}" if cma_id is not None else None,
        "cma_id": cma_id,
        "accession_number": str(accession) if accession else None,
        "title": clean_text(record.get("title")) or None,
        "creation_date": clean_text(record.get("date_text")) or None,
        "date_begin": record.get("creation_date_earliest"),
        "date_end": record.get("creation_date_latest"),
        "culture": _culture_text(record.get("culture")) or None,
        "department": clean_text(record.get("department")) or None,
        "technique": clean_text(record.get("technique")) or None,
        "type": clean_text(record.get("type")) or None,
        "current_location": clean_text(record.get("current_location")) or None,
        "description": clean_text(record.get("description")) or None,
        "did_you_know": clean_text(record.get("did_you_know")) or None,
        "provenance_text": provenance_text(record.get("provenance")) or None,
        "prev_owner": prev_owner(record.get("provenance")),
        "exhibitions_text": exhibitions_text(record.get("exhibitions")) or None,
        "credit_line": clean_text(record.get("creditline")) or None,
        "source_url": source_url,
        "image_url": image_url,
        "image_path": None,
    }
    row.update(artist_info(record))
    return row


ARTWORK_COLUMNS = [
    "artwork_id",
    "cma_id",
    "accession_number",
    "title",
    "creation_date",
    "date_begin",
    "date_end",
    "artist_id",
    "artist_name",
    "artist_role",
    "artist_birth",
    "artist_death",
    "artist_nationality",
    "culture",
    "department",
    "technique",
    "type",
    "current_location",
    "description",
    "did_you_know",
    "provenance_text",
    "prev_owner",
    "exhibitions_text",
    "credit_line",
    "source_url",
    "image_url",
    "image_path",
]
