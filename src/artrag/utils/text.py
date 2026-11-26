"""Tiện ích xử lý văn bản (HTML, chuẩn hóa, tách câu)."""

from __future__ import annotations

import html
import re
import unicodedata

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_DIACRITIC_RE = re.compile(r"[\u0300-\u036f]")
# "đ" (U+0111) không decompose dưới NFD, phải thay thế tay.
_STROKE_D = {"đ": "d", "Đ": "D"}

# Các ký tự có thể kết thúc câu.
_END_CHARS = ".!?…"

# Dấu đóng ngoặc / dấu trích dán được hấp thụ vào câu hiện tại.
_CLOSE_CHARS = "\"'”’)]}"

# Chữ viết tắt có nhiều hơn một chữ cái: dấu chấm sau chúng KHÔNG kết thúc câu.
_ABBREV_TITLES = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "rev", "st", "jr", "sr", "gen", "col",
        "capt", "lt", "sgt", "maj", "hon", "fr", "mt", "ft", "no", "vol",
        "dept", "univ", "assn", "bros", "phd", "esq",
        # doanh nghiệp / xuất bản
        "co", "inc", "ltd", "corp",
        # trích dẫn học thuật
        "cat", "ibid", "ca", "cf", "al", "ed", "eds", "ser", "viz", "pp",
        "fl", "bapt", "esp", "pt", "approx",
    }
)


def strip_html(text: str) -> str:
    """Bỏ thẻ HTML, giải thực thể, gộp khoảng trắng, cắt đầu/cuối."""
    if not text:
        return ""
    out = _TAG_RE.sub(" ", text)
    out = html.unescape(out)
    out = out.replace("\xa0", " ")
    out = _WS_RE.sub(" ", out)
    return out.strip()


def normalize_text(text: str) -> str:
    """Chuẩn hóa: bỏ HTML, lowercase, bỏ dấu tiếng Việt, gộp khoảng trắng."""
    if not text:
        return ""
    out = strip_html(text).lower()
    for src, dst in _STROKE_D.items():
        out = out.replace(src.lower(), dst)
    out = unicodedata.normalize("NFD", out)
    out = _DIACRITIC_RE.sub("", out)
    out = _WS_RE.sub(" ", out)
    return out.strip(" .,;:!?")


def _token_bounds(text: str, index: int) -> tuple[int, int]:
    """Biên của "từ" chứa ký tự tại ``index`` (bao gồm cả dấu chấm bên trong).

    Ví dụ với ``"copied (q. v.), the"``:
      - index trỏ vào dấu chấm sau ``q`` -> ``("q.", "v.")``;
      - index trỏ vào dấu chấm sau ``v`` -> ``("q.", "v.")``.
    """
    start = index
    while start > 0 and not text[start - 1].isspace() and text[start - 1] not in "(\"[":
        start -= 1
    end = index
    while end < len(text) and (text[end].isalnum() or text[end] == "."):
        end += 1
    return start, end

def _dotted_abbr_kind(core: str) -> str | None:
    """Phân loại token nhiều thành phần nối bằng dấu chấm.

    - ``"acronym"``: ``U.S.A`` (chữ cái đơn HOA nối bằng dấu chấm).
    - ``"latin"``: ``q.v``, ``i.e``, ``e.g`` (chữ cái đơn thường nối bằng dấu chấm).
    - ``None``: còn lại.
    """
    parts = [part for part in core.split(".") if part]
    if len(parts) < 2:
        return None
    if not all(len(part) == 1 and part.isalpha() for part in parts):
        return None
    return "acronym" if all(part.isupper() for part in parts) else "latin"


def _is_version_like(core: str) -> bool:
    """Token như ``v.15.1`` / ``no.12`` -> True (số hiệu, dấu chấm không hết câu).

    Điều kiện: có ít nhất một phần chứa chữ số, và mọi phần chỉ gồm chữ số
    hoặc một từ viết tắt ngắn (tối đa 2 chữ cái, ví dụ ``v``, ``no``, ``pt``).
    Nhờ vậy ``few.2`` (một từ bình thường) KHÔNG bị coi là số hiệu.
    """
    parts = [part for part in core.split(".") if part]
    if len(parts) < 2:
        return False
    if not any(any(ch.isdigit() for ch in part) for part in parts):
        return False
    for part in parts:
        if part.isalpha() and len(part) > 2:
            return False
    return True


def _is_single_initial(text: str, start: int, end: int) -> bool:
    """Token là một chữ cái hoa đơn có chấm (``J.`` trong ``J. M. W.``)?"""
    return end - start == 2 and text[end - 1] == "." and text[start].isupper()


def _prev_is_upper_initial(text: str, start: int) -> bool:
    """Token trước có phải một chữ cái hoa đơn đã gạch chấm không (``J. M. W.``)?"""
    index = start - 1
    while index >= 0 and text[index].isspace():
        index -= 1
    if index >= 0 and text[index] == ".":
        prev = index - 1
        if prev >= 0 and text[prev].isupper() and text[prev].isalpha():
            return True
    return False


def _next_is_upper_initial(text: str, end: int) -> bool:
    """Token kế tiếp có phải một chữ cái hoa đơn đã gạch chấm không (``J. M. W.``)?"""
    index = end
    while index < len(text) and text[index].isspace():
        index += 1
    if index >= len(text) or not text[index].isupper() or not text[index].isalpha():
        return False
    # token kế tiếp phải chỉ gồm một chữ cái + dấu chấm
    if index + 1 < len(text) and text[index + 1] == ".":
        after = index + 2
        while after < len(text) and text[after].isspace():
            after += 1
        if after < len(text) and text[after].isalpha():
            return True
    return False


def _next_is_capitalized_word(text: str, end: int) -> bool:
    """Token kế tiếp là một từ viết hoa nhiều chữ cái (``Paul`` trong ``J. Paul``)?

    Dùng để nhận ra tên riêng có chữ cái đầu là chữ cái đơn: ``J. Paul Getty
    Museum``, ``A. Turner``. Trường hợp này dấu chấm KHÔNG phải hết câu.
    """
    index = end
    while index < len(text) and text[index].isspace():
        index += 1
    if index >= len(text) or not text[index].isupper() or not text[index].isalpha():
        return False
    after = index
    while after < len(text) and text[after].isalpha():
        after += 1
    return after - index >= 2


def _abbrev_followed_by_new_sentence(
    text: str, pos: int, parts: int, kind: str
) -> bool:
    """Sau chữ viết tắt có bắt đầu câu mới không?

    - Chữ hoa hoặc chữ số (footnote marker) -> có.
    - Hết văn bản -> có.
    - Chữ thường -> chỉ khi acronym có đúng 2 thành phần (``U.S.``), vì
      ``U.S.A.`` / ``i.e.`` / ``q.v.`` còn nối tiếp câu.
    """
    index = pos
    while index < len(text) and text[index] in _CLOSE_CHARS:
        index += 1
    while index < len(text) and text[index].isspace():
        index += 1
    if index >= len(text):
        return True
    char = text[index]
    if char.isupper() or char.isdigit():
        return True
    return kind == "acronym" and parts == 2 and char.islower()


def _next_is_all_caps_word(text: str, end: int) -> bool:
    """Token kế tiếp là một từ VIẾT HOA toàn bộ (``II``, ``MD``, ``XVIII``)?

    Đây là chữ viết tắt / số La Mã, không phải bắt đầu câu mới. Ví dụ
    ``his brother Jan Jansz. II, and son David Coecke.`` -> dấu chấm sau
    ``Jansz.`` thuộc về từ, không phải hết câu.
    """
    index = end
    while index < len(text) and text[index].isspace():
        index += 1
    if index >= len(text) or not text[index].isalpha():
        return False
    after = index
    while after < len(text) and text[after].isalpha():
        after += 1
    word = text[index:after]
    return len(word) >= 2 and word.isupper()


def _is_sentence_end(text: str, index: int) -> bool:
    """Dấu tại ``index`` có phải kết thúc câu không?"""
    if text[index] != ".":
        return True

    start, end = _token_bounds(text, index)
    token = text[start:end]
    core = token.strip(".")
    if not core:
        return True

    # Token chỉ gồm chữ số -> đánh số mục (``2.`` trong danh sách chú thích),
    # KHÔNG phải hết câu.
    if core.isdigit():
        return False

    # Dấu chấm bám sát dấu phẩy / chấm phẩy (``Jansz.,``, ``etc.,``) -> dấu
    # chấm thuộc về từ, không phải hết câu.
    nxt = index + 1
    while nxt < len(text) and text[nxt] in _CLOSE_CHARS:
        nxt += 1
    if nxt < len(text) and text[nxt] in ",;:":
        return False

    # Từ viết hoa đầu thường đứng trước một từ VIẾT HOA (``Jansz. II,``) ->
    # dấu chấm thuộc về từ, không phải hết câu.
    if len(core) > 1 and core[0].isupper() and not core.isupper():
        if _next_is_all_caps_word(text, end):
            return False

    # Chữ viết tắt nối bằng dấu chấm: "q.v.", "i.e.", "U.S.A." -> chỉ dấu
    # chấm CUỐI CÙNG của token mới có thể kết thúc câu, và còn phải kiểm tra
    # phía sau.
    kind = _dotted_abbr_kind(core)
    if kind is not None:
        last_dot = start + token.rfind(".")
        if index != last_dot:
            return False
        parts = len([part for part in core.split(".") if part])
        return _abbrev_followed_by_new_sentence(text, end, parts, kind)

    # Số phiên bản "v.15.1" -> dấu chấm không hết câu.
    if _is_version_like(core):
        return False

    # Chuỗi chữ cái đầu tách bằng khoảng trắng: "J. M. W." -> dấu chấm cuối
    # là của chữ viết tắt, không phải hết câu.
    if len(core) == 1 and core.isalpha():
        if core.islower():
            # "q.", "v." -> chữ viết tắt Latin, không hết câu.
            return False
        # "J. M. W." -> dấu chấm sau một chữ cái hoa đơn; nếu token trước hoặc
        # token sau cũng là chữ cái hoa đơn gạch chấm -> vẫn là chữ viết tắt.
        if _is_single_initial(text, start, end):
            if (
                _prev_is_upper_initial(text, start)
                or _next_is_upper_initial(text, end)
                or _next_is_capitalized_word(text, end)
                or _next_is_all_caps_word(text, end)
            ):
                return False
        return True

    if core.lower() in _ABBREV_TITLES:
        return False
    return True


def _absorb_footnote_marker(text: str, end: int) -> int:
    """Hấp thụ số đánh dấu trích dẫn chân bài đứng sát sau dấu kết thúc.

    Ví dụ ``works."1 He`` -> phần ``."1`` thuộc về câu trước, nên ranh giới câu
    phải nằm sau số ``1``.
    """
    index = end + 1
    last_digit = end
    while index < len(text) and text[index].isdigit():
        last_digit = index
        index += 1
    return last_digit


def _bracket_depths(text: str) -> list[int]:
    """Độ sâu ngoặc tại mỗi vị trí (trước khi xử lý ký tự tại đó).

    Dấu kết thúc câu nằm trong ngoặc (``(1798-?)``, ``[klod lɔ.ʁɛ̃]``) KHÔNG
    được coi là hết câu, vì vậy cần biết độ sâu ngoặc tại mỗi vị trí.
    """
    depths = [0] * len(text)
    current = 0
    for index, char in enumerate(text):
        depths[index] = current
        if char in "([{":
            current += 1
        elif char in ")]}":
            current = max(0, current - 1)
    return depths


def _span_end_chars(text: str, index: int) -> int:
    """Chỉ số cuối của chuỗi dấu kết thúc tại ``index``.

    Gộp các dấu liên tiếp (``?!``, ``...``) và cả dấu chấm rời nhau bằng
    đúng một khoảng trắng (``. . .`` - dấu chấm lửng trong văn bản CMA).
    """
    length = len(text)
    end = index
    pos = index
    dots = 0
    while pos < length and dots < 3:
        if text[pos] not in _END_CHARS:
            break
        end = pos
        pos += 1
        dots += 1
        if pos < length and text[pos] == ".":
            continue
        # cho phép đúng một khoảng trắng giữa các dấu chấm rời nhau
        after = pos
        while after < length and text[after] == " ":
            after += 1
        if after - pos == 1 and after < length and text[after] == ".":
            pos = after
            continue
        break
    return end


def split_sentences(text: str) -> list[str]:
    """Tách câu, giữ dấu câu ở cuối, bỏ câu rỗng.

    Dấu chấm sau chữ viết tắt KHÔNG kết thúc câu: ``(q.v.)``, ``J. M. W.``,
    ``U.S.A.``, ``Dr.``. Trước đây mọi dấu chấm đều bị coi là hết câu, khiến
    tiểu sử họa sĩ bị cắt vỡ (ví dụ "Millet (q.v." / "v.), ...").

    Quy tắc bổ sung:
    - Dấu ngoặc đóng / dấu trích dán ngay sau dấu kết thúc được hấp thụ vào
      câu hiện tại.
    - Số đánh dấu trích dẫn chân bài (``."1``) thuộc về câu TRƯỚC.
    - Dấu kết thúc nằm trong ngoặc (``(1798-?)``, ``[klod lɔ.ʁɛ̃]``) không phải
      hết câu.
    """
    if not text:
        return []

    sentences: list[str] = []
    depths = _bracket_depths(text)
    start = 0
    index = 0
    length = len(text)

    while index < length:
        char = text[index]
        if char not in _END_CHARS:
            index += 1
            continue

        # dấu kết thúc trong ngoặc không phải hết câu
        if depths[index] > 0:
            index += 1
            continue

        span_end = _span_end_chars(text, index)

        # dấu chấm rời nhau bằng khoảng trắng (``. . .``) -> dấu lửng, KHÔNG
        # phải hết câu.
        if (
            char == "."
            and index + 2 < length
            and text[index + 1] == " "
            and text[index + 2] == "."
        ):
            index = span_end + 1
            continue

        # chỉ kiểm tra chữ viết tắt khi là dấu chấm đơn lẻ
        if char == "." and span_end == index and not _is_sentence_end(text, index):
            index = span_end + 1
            continue
        # hấp thụ dấu ngoặc đóng / dấu trích dán ngay sau dấu kết thúc, rồi
        # nếu sau đó lại là dấu kết thúc thì gộp tiếp (mẫu ``." .``)
        end = span_end
        while True:
            while end + 1 < length and text[end + 1] in _CLOSE_CHARS:
                end += 1
            if (
                end + 1 < length
                and text[end + 1] in _END_CHARS
                and depths[end + 1] == 0
            ):
                end = _span_end_chars(text, end + 1)
                continue
            break

        # hấp thụ số đánh dấu trích dẫn chân bài đứng sát sau dấu kết thúc
        end = _absorb_footnote_marker(text, end)

        piece = text[start : end + 1].strip()
        if piece and any(ch.isalnum() for ch in piece):
            sentences.append(piece)
        start = end + 1
        index = end + 1

    tail = text[start:].strip()
    if tail and any(ch.isalnum() for ch in tail):
        sentences.append(tail)
    return sentences
