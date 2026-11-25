# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Dự án
"Trợ lý Bảo tàng Thông minh": Multimodal RAG có dẫn chứng trên khoảng 2.200 tranh CC0 của Cleveland Museum of Art (kèm tiểu sử họa sĩ từ Wikipedia). Người dùng hỏi bằng tiếng Việt hoặc upload ảnh. Hệ thống truy xuất lai (CLIP/FAISS ảnh + BGE-M3 chunk) → rerank BLIP-ITM → evidence (caption/VQA BLIP + snippet) → LLM trả lời có citation `[IMG-xx]` hoặc từ chối. Đây là bài tập lớn, một người làm, cần đo đạc đầy đủ (baseline, ablation, latency, bộ nhớ).

## Tài liệu nguồn (đọc khi cần, không đọc toàn bộ mỗi lần)
- [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md): thiết kế chi tiết; schema dữ liệu §2.3, pipeline online §5, registry/cache/trace §6, đánh giá §8.
- [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md): danh sách task T0.1…T9.x, mỗi task có Files, lệnh Verify, "Done khi" và trạng thái checkbox.

## Cách làm việc (bắt buộc)
- Làm **từng task một** theo IMPLEMENTATION_PLAN. Chỉ đọc § thiết kế được dẫn và các file liệt kê trong task. Không quét lại toàn repo.
- Mỗi task kết thúc bằng việc chạy lệnh Verify của task đó, tick checkbox, rồi commit `T<id>: …`.
- Thiết kế cần đổi thì sửa SYSTEM_DESIGN.md trước (ghi ngắn lý do), sau đó mới code.
- Lệnh gọi mạng hoặc model thật khi verify luôn dùng `--limit` nhỏ.
- Notebook (`notebooks/`) chỉ là wrapper mỏng gọi `python -m artrag.…`; không viết logic trong notebook. Notebook nộp bài `final_end_to_end.ipynb` chỉ làm ở bước cuối (T8.5).

## Lệnh
- Môi trường: `uv sync` (Python 3.11; Python 3.15 của hệ thống không dùng).
- Test: `uv run pytest -q`; chạy một test: `uv run pytest tests/test_fuse.py::test_name -q`.
- Module chạy bằng `uv run python -m artrag.<module> …`, ví dụ `artrag.data.build_kb --check`, `artrag.pipeline.rag "câu hỏi" --json`, `artrag.eval.run_eval --exp base --limit 10`, `artrag.app.gradio_app`.
- Profile: `--profile local|colab|kaggle|api` hoặc biến `ARTRAG_PROFILE`; ghi đè config bằng `--set a.b=c`; thư mục dữ liệu là `ARTRAG_DATA_DIR` (mặc định `./data`).

## Quy ước kiến trúc
- **Config-driven**: ngưỡng, trọng số, model, backend nằm trong `configs/` (default ← profile ← experiment ← `--set`). Một thí nghiệm là một file `configs/experiments/*.yaml`; không hardcode tham số.
- **Mọi model lấy qua `ModelRegistry`** (`src/artrag/registry.py`): lazy load, ngân sách VRAM/LRU, fallback CPU. Không gọi `from_pretrained` rải rác. GPU local là GTX 1650 4GB (không có bf16), nên chạy fp16 trên cuda, fp32 trên CPU.
- **LLM qua interface `LLMBackend`** (`hf_local` Qwen2.5, `openai_compat` cho Groq/NVIDIA NIM, `flan_t5`), luôn bọc cache đĩa và retry. API key chỉ đọc từ biến môi trường hoặc `.env`.
- **Một pipeline cho UI, CLI và eval**: `RAGPipeline.answer(RAGRequest) -> RAGResponse` (gồm evidence table và trace).
- `[IMG-xx]` là nhãn cục bộ trong từng câu trả lời, ánh xạ tới `artwork_id` (`cma-{id}`) qua evidence table. Hậu kiểm sẽ loại citation không hợp lệ.
- Hàm thuần (fuse, abstain rules, citations, context, metrics) phải test được mà không cần GPU hay model.
- `data/` (raw, kb, images, index, cache) không commit; phân phối qua `artrag.tools.artifacts pack/unpack`.
