# Trợ lý Bảo tàng Thông minh

Multimodal RAG có dẫn chứng trên khoảng 2.200 tranh CC0 của Cleveland Museum of Art
(kèm tiểu sử họa sĩ từ Wikipedia).

Người dùng hỏi bằng tiếng Việt hoặc upload ảnh. Hệ thống truy xuất lai
(CLIP/FAISS ảnh + BGE-M3 chunk) → rerank BLIP-ITM → evidence (caption/VQA BLIP + snippet)
→ LLM trả lời có citation `[IMG-xx]` hoặc từ chối.

## Cài đặt

```bash
uv sync
uv run pytest -q
```

Yêu cầu Python 3.11 (quản lý bằng `uv python install 3.11`).

## Cấu trúc

- `src/artrag/` — mã nguồn package
- `configs/` — cấu hình default, profile, experiment
- `docs/` — thiết kế hệ thống và kế hoạch triển khai
- `tests/` — unit test (hàm thuần chạy được không cần GPU/model)

## Tài liệu

- [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md) — thiết kế chi tiết
- [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) — kế hoạch triển khai theo task

## Tái tạo artefact (xem T8.4)

```
fetch CMA → build_kb → tải ảnh → build index (CLIP, BGE-M3) → caption → pack
```

## Lệnh thường dùng

```bash
uv run python -m artrag.pipeline.rag "câu hỏi" --json
uv run python -m artrag.eval.run_eval --exp base --limit 10
uv run python -m artrag.app.gradio_app
```
