"""Dịch thuật và bảo vệ citation khi dịch (T2.5, SYSTEM_DESIGN 6.3/6.4).

Module này gồm hai phần:

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
    def vi_to_en(self, text: str) -> str:
        """Dịch tiếng Việt sang tiếng Anh, giữ nguyên mọi citation."""
        if not text:
            return ""
        return translate_segments(
            text, lambda core: self._translate(self.vi2en_model, core)
        )

    def en_to_vi(self, text: str) -> str:
        """Dịch tiếng Anh sang tiếng Việt, giữ nguyên mọi citation."""
        if not text:
            return ""
        return translate_segments(
            text, lambda core: self._translate(self.en2vi_model, core)
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
    "split_segments",
    "translate_segments",
]
