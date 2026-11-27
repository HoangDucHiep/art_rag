"""Prompt template cho CLIP (SYSTEM_DESIGN 5.2 bước 4).

Ba template là biến thể của Project 1:
- `T0`: câu truy vấn nguyên văn.
- `T1`: mặc định — thêm "a painting of".
- `T2`: thêm "a museum artwork depicting".

Hàm ở đây là **hàm thuần** (không cần torch) nên test được ngay.
"""

from __future__ import annotations

CLIP_TEMPLATES: dict[str, str] = {
    "T0": "{q}",
    "T1": "a painting of {q}",
    "T2": "a museum artwork depicting {q}",
}

DEFAULT_CLIP_TEMPLATE = "T1"


def render_clip_text(text: str, template: str = DEFAULT_CLIP_TEMPLATE) -> str:
    """Áp template CLIP lên truy vấn.

    Ném `ValueError` nếu tên template không có trong `CLIP_TEMPLATES`.
    """
    key = (template or "").strip()
    if key not in CLIP_TEMPLATES:
        raise ValueError(
            f"Template CLIP không hợp lệ: {template!r}. "
            f"Chọn một trong: {', '.join(sorted(CLIP_TEMPLATES))}"
        )
    return CLIP_TEMPLATES[key].format(q=(text or "").strip())
