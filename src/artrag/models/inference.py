"""Interface suy luận thống nhất cho các model (T2.2).

Mỗi hàm nhận một `ModelHandle` (từ `artrag.registry`) và **dispatch theo
`handle.task`**. Nhờ vậy pipeline ở T3–T5 không cần biết model cụ thể nào
đang được nạp, và cùng một hàm phục vụ được cả `clip` lẫn `sbert` khi tác vụ
trùng nhau (embed text).

Các hàm ở đây là wrapper mỏng quanh model — không chứa logic nghiệp vụ.
"""

from __future__ import annotations

from typing import Any

from artrag.errors import ModelLoadError
from artrag.registry import ModelHandle


def _unsupported(handle: ModelHandle, func: str) -> ModelLoadError:
    return ModelLoadError(
        f"{func}() không hỗ trợ task {handle.task!r} (model {handle.name!r}).",
        hint="kiểm tra task của model trong configs/models.yaml",
    )


def _require(handle: ModelHandle, func: str, *tasks: str) -> None:
    if handle.task not in tasks:
        raise _unsupported(handle, func)


def _as_numpy(array: Any) -> Any:
    """Chuyển tensor sang numpy (float32) để FAISS dùng được."""
    if hasattr(array, "detach"):
        return array.detach().cpu().float().numpy()
    return array


def _l2_normalize(vector: Any) -> Any:
    """Chuẩn hoá L2 một vector numpy."""
    norm = float((vector**2).sum() ** 0.5) or 1.0
    return vector / norm


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def embed_image(handle: ModelHandle, image: Any) -> Any:
    """Vector ảnh (CLIP). Kết quả đã L2-normalize."""
    _require(handle, "embed_image", "clip")
    import torch  # noqa: PLC0415

    with torch.no_grad():
        inputs = handle.processor(images=image, return_tensors="pt").to(handle.device)
        features = handle.model.get_image_features(**inputs)
    return _l2_normalize(_as_numpy(features)[0])


def embed_text(handle: ModelHandle, text: str) -> Any:
    """Vector văn bản. CLIP dùng template (§5.2), sbert dùng tokenizer gốc."""
    _require(handle, "embed_text", "clip", "sbert")
    import torch  # noqa: PLC0415

    with torch.no_grad():
        if handle.task == "clip":
            inputs = handle.processor(
                text=[text], return_tensors="pt", padding=True, truncation=True
            ).to(handle.device)
            features = handle.model.get_text_features(**inputs)
            return _l2_normalize(_as_numpy(features)[0])

        # sbert: SentenceTransformer.encode đã trả về numpy float32.
        vector = handle.model.encode(
            [text], convert_to_numpy=True, normalize_embeddings=True
        )
        return _as_numpy(vector)[0]


# ---------------------------------------------------------------------------
# BLIP
# ---------------------------------------------------------------------------


def itm_score(handle: ModelHandle, image: Any, text: str) -> float:
    """Điểm khớp nội dung ảnh–text của BLIP-ITM.

    `BlipForImageTextRetrieval` với `use_itm_head=True` trả về `itm_score` là
    logits 2 lớp (không phải không phải) -> lấy softmax rồi chọn lớp 1.
    """
    _require(handle, "itm_score", "blip_itm")
    import torch  # noqa: PLC0415

    with torch.no_grad():
        inputs = handle.processor(
            images=image, text=text, return_tensors="pt", truncation=True
        ).to(handle.device)
        outputs = handle.model(**inputs, use_itm_head=True)
        scores = outputs.itm_score
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
        probs = torch.softmax(scores.float(), dim=-1)
    return float(probs[0, 1])


def caption(handle: ModelHandle, image: Any, **gen: Any) -> str:
    """Sinh caption cho ảnh bằng BLIP."""
    _require(handle, "caption", "blip_cap")
    import torch  # noqa: PLC0415

    defaults: dict[str, Any] = {"max_new_tokens": 30, "num_beams": 3}
    defaults.update(gen)
    with torch.no_grad():
        inputs = handle.processor(images=image, return_tensors="pt").to(handle.device)
        output = handle.model.generate(**inputs, **defaults)
    return handle.processor.decode(output[0], skip_special_tokens=True).strip()


def vqa(handle: ModelHandle, image: Any, question: str) -> str:
    """Trả lời câu hỏi về ảnh bằng BLIP-VQA.

    `BlipForQuestionAnswering.generate` bắt buộc có `input_ids` (câu hỏi đã
    tokenize) nên ta gọi `forward` trước để lấy `input_ids` rồi mới generate.
    """
    _require(handle, "vqa", "blip_vqa")
    import torch  # noqa: PLC0415

    with torch.no_grad():
        inputs = handle.processor(
            images=image, text=question, return_tensors="pt", truncation=True
        ).to(handle.device)
        input_ids = inputs["input_ids"]
        output = handle.model.generate(input_ids=input_ids, pixel_values=inputs["pixel_values"])
    return handle.processor.decode(output[0], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Seq2seq và NLI
# ---------------------------------------------------------------------------


def translate(handle: ModelHandle, text: str, **gen: Any) -> str:
    """Dịch hoặc sinh câu trả lời bằng model seq2seq."""
    _require(handle, "translate", "seq2seq")
    import torch  # noqa: PLC0415

    defaults: dict[str, Any] = {"max_new_tokens": 128}
    defaults.update(gen)
    with torch.no_grad():
        inputs = handle.processor(
            [text], return_tensors="pt", truncation=True
        ).to(handle.device)
        output = handle.model.generate(**inputs, **defaults)
    return handle.processor.batch_decode(output, skip_special_tokens=True)[0].strip()


def nli_label(handle: ModelHandle, premise: str, hypothesis: str) -> str:
    """Nhãn NLI: 'entailment' | 'contradiction' | 'neutral'."""
    _require(handle, "nli_label", "nli")
    import torch  # noqa: PLC0415

    with torch.no_grad():
        inputs = handle.processor(
            [premise], [hypothesis], return_tensors="pt", truncation=True
        ).to(handle.device)
        logits = handle.model(**inputs).logits
    index = int(logits[0].argmax())
    id2label = getattr(handle.model.config, "id2label", {}) or {}
    label = id2label.get(index, index)
    return str(label).lower()
