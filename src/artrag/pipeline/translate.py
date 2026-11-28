"""Dịch thuật và bảo vệ citation khi dịch (T2.5, SYSTEM_DESIGN 6.3/6.4).

Module này gồm ba phần:

1. **`Translator`** — bọc hai model dịch của registry (`vi2en`, `en2vi`) qua
   `artrag.models.inference.translate` (tái sử dụng, không viết lại pipeline).
   Model được nạp **lazy**; kết quả được cache trên đĩa tại `data/cache/translate`
   với khoá `sha256(model|văn bản)`.

2. **Hàm thuần chia đoạn** (`split_segments`, `join_segments`,
   `translate_segments`) — cơ chế **bảo vệ citation**:

   Chuỗi văn bản được tách theo nhãn citation `[IMG-xx]`. Mỗi đoạn thuần
   (không chứa citation) mới được đưa qua máy dịch; các đoạn chứa citation
   được **giữ nguyên**. Nhờ vậy nhãn citation không bao giờ bị máy dịch làm
   hỏng — không phụ thuộc việc model giữ được ký tự đặc biệt.

   Hàm thuần ở đây **không cần GPU hay model**, nên test được nhanh.

3. **Bảo vệ danh tính** (`protect_names`, `restore_names`) — cơ chế
   **bảo vệ tên riêng** khi dịch EN <-> VI:

   Model dịch `en2vi` (vinai MBart) sai tên riêng khi tên đứng một mình,
   ví dụ `"Thomas Eakins"` -> `"Thich Nhat Hanh"` (đúng là một nhà sư người
   Việt khác). Khi tên nằm trong câu có ngữ cảnh thì model giữ được.

   Vì vậy, trước khi dịch, các cụm danh tính được thay bằng placeholder
   dạng `§0§`, `§1§`…; sau khi dịch thì khôi phục lại tên gốc.

   **Chỉ an toàn với `en2vi`**: đo thực tế cho thấy `en2vi` giữ nguyên
   placeholder ở mọi vị trí, còn `vi2en` **xoá** placeholder đứng ở **đầu
   đoạn** (kéo theo mất nội dung dính liền). Vì vậy cơ chế này **chỉ** được
   bật cho hướng EN -> VI; xem `Translator._translate_core`.

   Ví dụ:

   "Thomas Eakins" -> "§0§" -> "Thomas Eakins" (thay vì "Thich Nhat Hanh")

Ví dụ:

    "The painting shows two men rowing [IMG-01] and a crowd."
    -> đoạn 1: "The painting shows two men rowing "   (dịch)
    -> đoạn 2: "[IMG-01]"                             (giữ nguyên)
    -> đoạn 3: " and a crowd."                        (dịch)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from artrag.errors import BackendError
from artrag.utils.hashing import sha256_text

#: Nhãn citation cục bộ, ví dụ `[IMG-01]`, `[IMG-1]`, `[IMG-U]`.
CITATION_RE = re.compile(r"\[IMG-[A-Za-z0-9][A-Za-z0-9\-_]*\]")

#: Tên model dịch trong registry (khoá của `configs/models.yaml`).
DEFAULT_VI2EN_MODEL = "vi2en"
DEFAULT_EN2VI_MODEL = "en2vi"

#: Tên thư mục cache dịch (dưới `paths.data_dir/cache/`).
DEFAULT_CACHE_DIR = "translate"

#: Ký tự tiếng Việt có dấu (để phát hiện ngôn ngữ khi không cần model).
_VIETNAMESE_CHARS = re.compile(r"[À-ỹĐđ]")

#: Cụm danh tính (proper noun): các từ viết hoa liền nhau, có thể có từ nối
#: viết thường (`van`, `de`, `la`…) để giữ trọn tên như "Vincent van Gogh".
#:
#: Chữ số **không** nằm trong cụm danh tính: "Gallery 207" là một danh từ
#: chứ không phải tên riêng.
#:
#: Dấu gạch nối **có khoảng trắng** (" - ") **không** nối hai từ trong cụm:
#: định dạng evidence là `{title} - {artist}, {year}` nên "Biglin Brothers -
#: Thomas Eakins" phải tách thành hai cụm, không gộp làm một tên.
_NAME_WORD = r"[A-Z][A-Za-z'’]*(?:-[A-Z][A-Za-z'’]*)*"
_NAME_LINK = r"(?:van|von|de|del|della|di|da|do|le|la|du|dos|das)"
_NAME_RE = re.compile(
    rf"\b{_NAME_WORD}(?: (?:{_NAME_WORD}|{_NAME_LINK}))*\b"
)

#: Cụm danh tính **hoặc** nhãn citation — dùng để quét một lần, biết chắc
#: phần nào là citation để giữ nguyên.
_NAME_OR_CITATION_RE = re.compile(
    rf"(?P<citation>{CITATION_RE.pattern})|(?P<name>{_NAME_RE.pattern})"
)

#: Sentinel dùng làm placeholder khi bảo vệ danh tính.
#:
#: Dạng `[[0]]` dễ va chạm với văn bản đã chứa sẵn placeholder (làm hỏng
#: bất biến `restore_names(*protect_names(t)) == t`). Vì vậy dùng sentinel
#: có xác suất va chạm thấp hơn: chữ `Đ` (viết hoa có dấu) hiếm gặp trong
#: văn bản tiếng Anh, và model dịch vẫn giữ nguyên chuỗi này.
_NAME_SENTINEL = "§"

#: Từ chức năng hay đứng đầu câu -> không phải tên riêng.
_NAME_STOP_STARTERS = frozenset(
    {
        "A", "An", "The", "This", "That", "These", "Those", "It", "Its",
        "In", "On", "At", "Of", "For", "From", "By", "With", "As", "And",
        "But", "Or", "So", "There", "Here", "He", "She", "They", "We", "I",
        "His", "Her", "Their", "Our", "My", "Your", "You", "If", "When",
        "While", "After", "Before", "During", "Both", "Each", "Many",
        "Most", "Some", "Two", "Three", "Four", "Five", "Six", "Seven",
        "Eight", "Nine", "Ten", "One", "No", "Not", "Yes", "Do", "Does",
        "Is", "Are", "Was", "Were", "Has", "Have", "Had", "Can", "Could",
        "Will", "Would", "Should", "May", "Might", "Must", "Also", "Only",
        "More", "Much", "Such", "Same", "Other", "Others", "Another",
        "First", "Second", "Third", "Last", "Next", "Now", "Then", "Even",
        "Still", "Yet", "Thus", "Like", "Until",
    }
)

#: Từ tiếng Anh thường gặp — nếu một từ đơn viết hoa mà nằm trong danh sách
#: này thì đó là danh từ/động từ viết hoa ở đầu câu, **không** phải tên riêng.
#:
#: Danh sách rút gọn, đủ cho câu trả lời ngắn kiểu caption/VQA mà hệ thống
#: sinh ra. Cụm nhiều từ ("Claude Monet") không cần danh sách này.
_COMMON_WORDS = frozenset(
    {
        "a", "about", "above", "across", "after", "again", "against", "all",
        "along", "also", "although", "always", "an", "and", "another", "any",
        "are", "around", "as", "at", "back", "be", "because", "been", "before",
        "being", "below", "beside", "besides", "between", "beyond", "both",
        "but", "by", "can", "cannot", "could", "did", "do", "does", "doing",
        "done", "down", "during", "each", "either", "else", "even", "ever",
        "every", "few", "for", "from", "further", "had", "has", "have",
        "having", "he", "her", "here", "hers", "him", "his", "how", "however",
        "i", "if", "in", "into", "is", "it", "its", "itself", "just", "like",
        "made", "make", "makes", "many", "may", "might", "more", "most",
        "much", "must", "my", "neither", "never", "no", "nor", "not", "now",
        "of", "off", "on", "once", "one", "only", "or", "other", "others",
        "our", "ours", "out", "over", "own", "perhaps", "quite", "rather",
        "same", "shall", "she", "should", "since", "so", "some", "such",
        "than", "that", "the", "their", "theirs", "them", "then", "there",
        "these", "they", "this", "those", "though", "through", "thus", "to",
        "too", "under", "until", "up", "upon", "us", "very", "was", "we",
        "were", "what", "when", "where", "whether", "which", "while", "who",
        "whom", "whose", "why", "will", "with", "within", "without", "would",
        "yet", "you", "your", "yours",
        # Danh từ/động từ thường gặp trong mô tả tranh.
        "answer", "artist", "background", "barge", "boat", "canvas",
        "collection", "depicts", "depicting", "drawn", "drawing", "evidence",
        "exhibition", "figure", "figures", "foreground", "gallery", "group",
        "history", "image", "images", "landscape", "life", "man", "men",
        "museum", "oil", "on", "painter", "painting", "paintings", "person",
        "picture", "pictures", "portrait", "shown", "shows", "source",
        "summary", "title", "view", "woman", "women", "work", "works",
        "year", "years", "created", "made", "painted", "features",
        "contains", "including", "known", "large", "left", "right",
        "behind", "front", "center", "centre", "middle",
        "upper", "lower", "part", "side", "edge", "top", "bottom",
        # Số từ -> không phải tên riêng.
        "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "first", "second", "third",
        "fourth", "fifth", "last", "next", "other", "another",
        # Từ viết hoa ở đầu câu trong câu trả lời kiểu caption/VQA.
        "hello", "gallery", "source", "image", "picture", "painting",
        "museum", "collection", "artist", "painter", "title", "wikimedia",
        "wikipedia", "answer", "question", "evidence", "summary",
        "both", "two", "several", "various", "unknown", "untitled",
        "detail", "details", "close", "close-up", "view", "scenes",
        "design", "element", "elements", "the", "this", "that",
        # Ký hiệu nội bộ -> không phải tên riêng.
        "img", "image", "id", "url", "http", "https", "www",
    }
)


@dataclass(frozen=True)
class Segment:
    """Một đoạn của văn bản sau khi tách theo citation.

    `has_citation=True` nghĩa là đoạn chứa ít nhất một nhãn `[IMG-xx]`
    -> **không được** đưa qua máy dịch.
    """

    text: str
    has_citation: bool


# ---------------------------------------------------------------------------
# Hàm thuần: tách / ghép đoạn
# ---------------------------------------------------------------------------


def split_segments(text: str) -> list[Segment]:
    """Tách văn bản thành các đoạn, ranh giới là nhãn citation.

    Mỗi đoạn hoặc là **thuần** (chỉ văn bản) hoặc là **chứa citation**.
    Khoảng trắng giữa các đoạn được giữ nguyên trong đoạn tương ứng, nên
    `join_segments(split_segments(t)) == t` với mọi `t`.
    """
    if not text:
        return []

    segments: list[Segment] = []
    last = 0
    for match in CITATION_RE.finditer(text):
        before = text[last : match.start()]
        if before:
            segments.append(Segment(text=before, has_citation=False))
        segments.append(Segment(text=match.group(0), has_citation=True))
        last = match.end()

    tail = text[last:]
    if tail:
        segments.append(Segment(text=tail, has_citation=False))
    return segments


def join_segments(segments: list[Segment]) -> str:
    """Ghép các đoạn về lại thành chuỗi gốc."""
    return "".join(segment.text for segment in segments)


def protect_citations(text: str) -> list[str]:
    """Tách chuỗi thành các đoạn; đoạn chứa citation được giữ nguyên.

    Hàm thuần — không cần GPU. Kết quả là danh sách đoạn theo đúng thứ tự
    trong văn bản gốc.
    """
    return [segment.text for segment in split_segments(text)]


def translate_segments(
    text: str,
    translate_fn: Callable[[str], str],
) -> str:
    """Dịch các đoạn thuần, giữ nguyên các đoạn chứa citation.

    `translate_fn` nhận nội dung **đã cắt khoảng trắng đầu/cuối** và trả về
    bản dịch. Khoảng trắng đầu/cuối của đoạn được gắn lại sau khi dịch để
    giữ nguyên khoảng cách giữa các đoạn.

    Nếu `translate_fn` là hàm nhận dạng (identity) thì kết quả **đúng bằng**
    văn bản gốc — đây là bất biến được kiểm tra trong test.
    """
    if not text:
        return ""

    out: list[str] = []
    for segment in split_segments(text):
        if segment.has_citation:
            # Đoạn chứa citation -> giữ nguyên, tuyệt đối không dịch.
            out.append(segment.text)
            continue

        core = segment.text.strip()
        if not core:
            out.append(segment.text)
            continue

        lead = segment.text[: len(segment.text) - len(segment.text.lstrip())]
        trail = segment.text[len(segment.text.rstrip()) :]
        out.append(lead + translate_fn(core) + trail)

    return "".join(out)


def looks_vietnamese(text: str) -> bool:
    """Heuristic: văn bản có phải tiếng Việt không?

    - Có ký tự có dấu (À-ỹ, Đ/đ) -> tiếng Việt.
    - Không dấu -> so với một danh sách từ tiếng Việt thường gặp.

    Heuristic này chỉ dùng để **quyết định có cần dịch hay không**, không quyết
    định nội dung; nếu sai thì câu hỏi vẫn được gửi nguyên văn cho model.
    """
    if not text:
        return False
    if _VIETNAMESE_CHARS.search(text):
        return True

    tokens = set(re.findall(r"[a-z]+", text.lower()))
    if not tokens:
        return False
    return bool(tokens & _VI_COMMON_WORDS)


#: Từ tiếng Việt không dấu xuất hiện thường xuyên (dùng cho heuristic).
_VI_COMMON_WORDS = frozenset(
    {
        "anh", "ban", "chao", "co", "chung", "gi", "hoi", "hoa",
        "khong", "la", "mo", "nho", "nhay", "nhung", "nay", "oi", "toi",
        "tranh", "va", "ve", "xin",
    }
)


# ---------------------------------------------------------------------------
# Hàm thuần: bảo vệ danh tính (tên riêng)
# ---------------------------------------------------------------------------


def protect_names(text: str) -> tuple[str, list[str]]:
    """Thay cụm danh tính bằng placeholder ``§0§``, ``§1§``…

    Model dịch `en2vi` (vinai MBart) sai tên riêng khi tên đứng một mình:
    `"Thomas Eakins"` -> `"Thich Nhat Hanh"`. Giữ tên nguyên vẹn là an toàn
    hơn là cố dịch.

    Hàm thuần — không cần GPU. Trả về `(văn bản đã bảo vệ, danh sách tên)`.
    """
    if not text:
        return "", []

    names: list[str] = []
    parts: list[str] = []
    last = 0
    for match in _NAME_OR_CITATION_RE.finditer(text):
        phrase = match.group(0)
        if match.group("citation"):
            # Nhãn citation có cơ chế bảo vệ riêng (`split_segments`), nên
            # được giữ nguyên — không đụng tới ở đây.
            continue
        words = phrase.split()
        if words[0] in _NAME_STOP_STARTERS:
            continue
        if not any(ch.isalpha() for ch in phrase):
            # Cụm chỉ gồm số -> không phải tên riêng.
            continue
        if len(words) == 1 and words[0].lower() in _COMMON_WORDS:
            # Một từ đơn nằm trong danh sách từ thường gặp -> danh từ/động từ
            # viết hoa ở đầu câu, không phải tên riêng.
            continue
        names.append(phrase)
        parts.append(text[last : match.start()])
        parts.append(f"{_NAME_SENTINEL}{len(names) - 1}{_NAME_SENTINEL}")
        last = match.end()
    parts.append(text[last:])
    return "".join(parts), names


def restore_names(text: str, names: list[str]) -> str:
    """Khôi phục các cụm danh tính đã được bảo vệ bằng `protect_names`."""
    for index, name in enumerate(names):
        text = text.replace(
            f"{_NAME_SENTINEL}{index}{_NAME_SENTINEL}",
            name,
        )
    return text


# ---------------------------------------------------------------------------
# Translator
# ---------------------------------------------------------------------------


class Translator:
    """Bộ dịch hai chiều VI <-> EN dùng model `seq2seq` trong registry.

    - Model được nạp **lazy** (chỉ khi cần dịch).
    - Kết quả được cache trên đĩa (`data/cache/translate`).
    - Câu có citation được bảo vệ theo cơ chế chia đoạn (xem
      `translate_segments`).
    """

    def __init__(
        self,
        registry: Any = None,
        vi2en_model: str = DEFAULT_VI2EN_MODEL,
        en2vi_model: str = DEFAULT_EN2VI_MODEL,
        cache_dir: str | Path | None = None,
        use_cache: bool = True,
    ) -> None:
        self.registry = registry
        self.vi2en_model = vi2en_model
        self.en2vi_model = en2vi_model
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.use_cache = bool(use_cache)
        self._cache: Any = None
        #: Số lần gọi model thật (dùng để test cache).
        self.model_calls = 0

    # -- cache -----------------------------------------------------------
    @property
    def cache(self) -> Any:
        """Cache đĩa; None nếu tắt hoặc chưa có thư mục."""
        if not self.use_cache or self.cache_dir is None:
            return None
        if self._cache is None:
            import diskcache  # noqa: PLC0415

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache = diskcache.Cache(str(self.cache_dir))
        return self._cache

    # -- nội bộ ----------------------------------------------------------
    def _model_translate(self, model_name: str, text: str) -> str:
        """Gọi model dịch trong registry (không qua cache)."""
        if self.registry is None:
            raise BackendError(
                "Translator chưa có registry nên không nạp được model dịch.",
                hint="dựng Translator qua backend hoặc truyền registry.",
            )
        from artrag.models import inference  # noqa: PLC0415

        handle = self.registry.get(model_name)
        self.model_calls += 1
        return inference.translate(handle, text)

    def _translate(self, model_name: str, text: str) -> str:
        """Dịch một đoạn văn bản thuần, có cache đĩa."""
        cache = self.cache
        if cache is None:
            return self._model_translate(model_name, text)

        key = sha256_text(f"{model_name}|{text}")
        hit = cache.get(key)
        if hit is not None:
            return str(hit)

        translated = self._model_translate(model_name, text)
        cache.set(key, translated)
        return translated

    # -- interface --------------------------------------------------------
    def _translate_core(self, model_name: str, core: str, direction: str) -> str:
        """Dịch một đoạn thuần theo **hướng** dịch đã cho.

        `direction` là `"vi2en"` hoặc `"en2vi"` — phân nhánh theo **hướng**,
        không theo tên model, để cấu hình đặt nhầm tên model không làm đổi hành vi.

        Hướng EN -> VI: danh tính được thay bằng placeholder **trước** khi dịch
        và khôi phục **sau** khi dịch, vì model `en2vi` dịch sai tên riêng đứng
        một mình (đo thực tế: `"Thomas Eakins"` -> `"Thich Nhat Hanh"`).

        Hướng VI -> EN: **không** dùng placeholder, vì model `vi2en` **xoá**
        placeholder đứng ở đầu đoạn (kéo theo mất nội dung dính liền — đo thực
        tế: đoạn evidence `" §0§, 1873. Oil on canvas…"` thành
        `"Oil on canvas…"`, mất luôn title + artist + năm). Với hướng này, văn
        bản **tiếng Anh** (evidence) được giữ nguyên: model `vi2en` không cần
        dịch nội dung đã là tiếng Anh, và dịch nó chỉ làm mất thông tin.
        """
        if direction == "en2vi":
            if looks_vietnamese(core):
                # Nội dung tiếng Việt: không bảo vệ danh tính (từ đầu câu viết
                # hoa sẽ bị chặn khỏi việc dịch).
                return self._translate(model_name, core)
            protected, names = protect_names(core)
            if not names:
                return self._translate(model_name, core)
            translated = self._translate(model_name, protected)
            return restore_names(translated, names)

        # Hướng VI -> EN: chỉ dịch nội dung tiếng Việt.
        if looks_vietnamese(core):
            return self._translate(model_name, core)
        return core

    def vi_to_en(self, text: str) -> str:
        """Dịch tiếng Việt sang tiếng Anh, giữ nguyên mọi citation."""
        if not text:
            return ""
        return translate_segments(
            text, lambda core: self._translate_core(self.vi2en_model, core, "vi2en")
        )

    def en_to_vi(self, text: str) -> str:
        """Dịch tiếng Anh sang tiếng Việt, giữ nguyên mọi citation."""
        if not text:
            return ""
        return translate_segments(
            text, lambda core: self._translate_core(self.en2vi_model, core, "en2vi")
        )

    def close(self) -> None:
        """Đóng cache đĩa."""
        if self._cache is not None:
            self._cache.close()
            self._cache = None


__all__ = [
    "CITATION_RE",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_EN2VI_MODEL",
    "DEFAULT_VI2EN_MODEL",
    "Segment",
    "Translator",
    "join_segments",
    "looks_vietnamese",
    "protect_citations",
    "protect_names",
    "restore_names",
    "split_segments",
    "translate_segments",
]
