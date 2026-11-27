# Kế hoạch triển khai: chia task nhỏ, verify được

> Thiết kế chi tiết nằm ở [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md) (mỗi task trỏ về § tương ứng).
> Trạng thái: `[ ]` chưa làm · `[~]` đang làm · `[x]` xong (ghi ngày). Mỗi task xong thì cập nhật checkbox và commit.

## Quy trình cho mỗi task (để tiết kiệm token)
1. Mở đúng **một** task. Chỉ đọc § thiết kế được dẫn và các file liệt kê trong "Files". Không quét lại toàn repo.
2. Code phần tối thiểu để đạt "Done khi". Không làm trước phần của task sau.
3. Chạy lệnh **Verify**. Pass thì tick `[x]`, commit `T<id>: <mô tả>`. Fail thì sửa trong phạm vi task.
4. Nếu phát hiện thiết kế cần đổi: sửa SYSTEM_DESIGN.md (ghi ngắn lý do) **trước** khi code tiếp.
5. Lệnh chạy model thật hoặc mạng chỉ dùng `--limit` nhỏ khi verify. Chạy toàn bộ là bước riêng, ghi rõ trong task.
6. Task đánh dấu ☁️ chạy trên Colab/Kaggle qua notebook mỏng. Local chỉ verify với `--limit` trên CPU.

Quy ước lệnh: mọi lệnh chạy từ gốc repo bằng `uv run …`; Python 3.11; module chạy bằng `python -m artrag.<…>`.

---

## M0. Khung dự án (tuần 1)

### [x] T0.1 Khởi tạo repo (2026-11-25)
- **Files**: `pyproject.toml`, `.gitignore`, `.env.example`, `src/artrag/__init__.py`, `tests/test_smoke.py`, `README.md` (khung).
- **Nội dung**:
  - `git init`; `uv init --package --python 3.11`;
  - deps lõi: `pyyaml pydantic pandas pyarrow requests tqdm python-dotenv rapidfuzz diskcache`; dev: `pytest`;
  - `.gitignore` gồm `data/ .env results/**/predictions.jsonl .venv/`.
- **Verify**: `uv run pytest -q` → 1 passed; `uv run python -c "import artrag; print(artrag.__version__)"`.
- **Done khi**: hai lệnh trên chạy được, commit đầu tiên.

### [x] T0.2 Config loader (§6.1, §9.1) (2026-11-25)
- **Files**: `src/artrag/config.py`, `configs/default.yaml`, `configs/profiles/{local,colab,kaggle,api}.yaml`, `tests/test_config.py`.
- **Nội dung**:
  - `load_config(profile=None, experiment=None, overrides=list[str]) -> Config` (dict truy cập kiểu thuộc tính, hoặc pydantic);
  - deep-merge theo thứ tự §6.1, xử lý `${env:VAR,default}`, `--set a.b=c`;
  - `config_hash()`; `paths()` trả về các thư mục con của data.
- **Verify**: `uv run pytest tests/test_config.py -q` (merge, override, env, hash ổn định).

### [x] T0.3 Trace + errors + utils (§6.5, §6.6) (2026-11-25)
- **Files**: `src/artrag/trace.py`, `errors.py`, `utils/{io.py,text.py,hashing.py}`, `tests/test_trace.py`.
- **Nội dung**:
  - `Trace.step(name, required)` là context manager: đo thời gian, bắt lỗi (step tùy chọn → `degraded`, step bắt buộc → ném lại); `to_dict()`;
  - utils: đọc/ghi jsonl, `strip_html`, `normalize_text`, `split_sentences`, `sha256_json`.
- **Verify**: `uv run pytest tests/test_trace.py -q`.

---

## M1. Dữ liệu (CPU, local, tuần 1)

### [x] T1.1 Fetch CMA (§2.1, §4 bước 1) (2026-11-25)
- **Files**: `src/artrag/data/fetch_cma.py`.
- **Nội dung**:
  - phân trang `limit=1000&skip=…` với `type=Painting&cc0=1&has_image=1`;
  - lưu `raw/cma/page_*.json`; bỏ qua trang đã có; retry/backoff; `--limit` (chỉ lấy N bản ghi đầu, dùng để thử).
- **Verify**:
  - `uv run python -m artrag.data.fetch_cma --limit 50` → tạo 1 file, in "50 records";
  - sau đó chạy đầy đủ `uv run python -m artrag.data.fetch_cma` → in tổng 3.964 (±, dữ liệu có thể đổi).
- **Done khi**: chạy lại lần 2 không gọi mạng (in "cached").

### [x] T1.2 Parse bản ghi CMA (§2.3) (2026-11-25)
- **Files**: `src/artrag/data/cma_parse.py`, `tests/fixtures/cma_sample.json` (3 bản ghi thật cắt từ raw: 1 bản đủ trường, 1 bản thiếu provenance, 1 bản tranh châu Á), `tests/test_cma_parse.py`.
- **Nội dung**:
  - `parse_artwork(rec) -> dict` theo schema `artworks`;
  - `provenance_text`, `prev_owner` (entry ngay trước entry "Cleveland Museum of Art"; lấy phần tên trước dấu `[`, `,`);
  - `exhibitions_text`; `overview_text(row)`.
- **Verify**: `uv run pytest tests/test_cma_parse.py -q` (Eakins: `prev_owner` chứa "Susan MacDowell Eakins"; HTML đã bị strip).

### [x] T1.3 Build KB: artworks + chunks (§2.3, §4 bước 2) (2026-11-25)
- **Files**: `src/artrag/data/build_kb.py`, `src/artrag/data/chunking.py`, `tests/test_chunking.py`.
- **Nội dung**:
  - đọc raw → lọc có description → `artworks.parquet`;
  - `chunks.parquet` (overview, description, did_you_know, provenance, exhibitions) theo quy tắc chunking;
  - `--check` in số tác phẩm, độ phủ từng trường, phân bố department, số chunk theo field.
- **Verify**:
  - `uv run pytest tests/test_chunking.py -q`;
  - `uv run python -m artrag.data.build_kb --check` → ≥2.000 tác phẩm, ≥10 department, không có chunk rỗng hoặc vượt 250 từ.

### [x] T1.4 Enrich Wikipedia (§2.1, §4 bước 3)
- **Files**: `src/artrag/data/enrich_wiki.py`.
- **Nội dung**:
  - với mỗi artist duy nhất: Wikidata `wbsearchentities` (tên + năm sinh để chọn đúng: kiểm tra `P31=Q5` và `P569` nếu có) → enwiki title → REST summary;
  - cache `raw/wiki/{artist_id}.json` (kể cả not_found); giới hạn tốc độ; `--limit`.
- **Verify**: `uv run python -m artrag.data.enrich_wiki --limit 30` → in tỉ lệ tìm thấy (mục tiêu ≥60%), in 3 ví dụ (tên → wiki_url).
- **Chạy đầy đủ**: không giới hạn (khoảng 1.000 artist, khoảng 10 phút).

### [x] T1.5 Artists + chunk tiểu sử
- **Files**: `build_kb.py` (thêm `--with-bio`), `src/artrag/data/artists.py`, `scripts/audit_t15.py`, `tests/test_artists.py`.
- **Nội dung**: `artists.parquet` (ưu tiên `cma_bio`, sau đó `wiki_summary`); thêm chunk `artist_bio` (artwork_id null, có artist_id).
- **Verify**: `uv run python -m artrag.data.build_kb --with-bio --check` → in số artist có bio theo nguồn (`cma`/`wikipedia`/`none`).
- **Ghi chú 2026-11-26 (giai đoạn sửa lỗi)**: sửa `split_sentences` trong `src/artrag/utils/text.py` — dữ liệu thật viết `(q.v.)` (không khoảng trắng) và `."1` (footnote marker) khiến 124 chunk tiểu sử bị cắt vỡ. Bổ sung bộ quy tắc: chữ viết tắt, hấp thụ dấu ngoặc kép, footnote marker, độ sâu ngoặc, số hiệu mục chú thích. Bất biến bắt buộc: `" ".join(split_sentences(text)) == text`. Kết quả: 0/562 bio lệch nguyên văn, 854 chunk `artist_bio` (thay vì 846), `scripts/audit_t15.py` PASS.

### [x] T1b Chunking strategies + ablation so sánh
- **Files**: `src/artrag/data/chunk_strategies.py` (MỚI), `src/artrag/data/chunking.py` (thêm cột `strategy`), `src/artrag/data/artists.py`, `src/artrag/data/build_kb.py` (thêm `--with-variants`), `src/artrag/eval/chunk_eval.py` (MỚI), `configs/experiments/chunk_{sentence,item,semantic,both}.yaml` (MỚI), `tests/test_chunk_strategies.py` (MỚI).
- **Nội dung**: ba chiến lược — C1 `sentence` (baseline: theo câu, gộp 180 từ, overlap 1 câu), C2 `item` (theo mục tự nhiên `"; "`, áp cho `provenance`/`exhibitions`), C3 `semantic` (theo đoạn văn 120 từ, không overlap, áp cho `artist_bio`/`description`). Chiến lược theo field nằm ở config `data.chunking.strategy_<field>`. `chunks.parquet` giữ nguyên baseline; `chunks_variants.parquet` (17.352 chunk) chứa đầy đủ biến thể với hậu tố `#<strategy>`.
- **Verify**: `uv run python -m artrag.eval.chunk_eval` → bảng Markdown so sánh 3 chiến lược (proxy metrics, không cần GPU). `uv run pytest -q` → xanh.
- **Kết quả 2026-11-26**: C2 `item` cắt giữa mục **0,00%** (C1: 2,24%) — loại triệt để lỗi cắt 83/1697 tác phẩm `exhibitions` và 1/2054 `provenance`. C3 `semantic` cắt giữa câu **0,88%** (C1: 37,59%) và redundancy **1,29%** (C1: 13,19%) nhờ không overlap. Tầng 2 (Recall@10 / MRR) **deferred** tới M6/M8.
- **Ghi chú sửa lỗi sau verify**: `chunk_items` ban đầu mất dấu `;` ở ranh giới giữa 2 chunk, vi phạm bất biến "ghép lại chunk phải khớp nguyên văn". Đã sửa bằng `_split_keep_separator` (tách regex giữ separator) + tách token `\S+\s*` khi cắt cứng mục quá dài. Sau sửa: 0/6.523 bản ghi lệch nguyên văn, `verbatim_rate` của C2 tăng 99,97% → 100,00%.

### [x] T1.6 Tải ảnh (§4 bước 5) (2026-11-26)
- **Files**: `src/artrag/data/images.py`, `tests/test_images.py`.
- **Nội dung**: tải song song (8 thread) → RGB → resize cạnh dài 512px → JPEG q90; bỏ qua file đã có; ghi `images_failed.json`; điền `image_path` vào artworks.
- **Verify**: `uv run python -m artrag.data.images --limit 20` → 20 file, kích thước max 512px; sau đó chạy đầy đủ và `--check` in số ảnh = số tác phẩm (trừ ảnh lỗi) và tổng dung lượng.
- **Lưu ý (chốt 2026-11-26)**: tác phẩm tải ảnh lỗi **vẫn giữ trong KB** với `image_path = null` — KHÔNG loại khỏi KB và KHÔNG rebuild với `--require-image`. Lý do: cột `image_path` nullable theo §2.3; nhánh BGE-M3 vẫn phục vụ câu hỏi metadata; nhánh CLIP chỉ trả tác phẩm có ảnh và §5.3 quy định "vắng ở nhánh nào thì nhận 0"; `abstain.py` R3 MISSING_FIELD từ chối câu hỏi VISUAL thiếu ảnh. Thêm `--retry-failed` để tải lại nhóm lỗi mà không tải lại toàn bộ. `build_kb` phải giữ `image_path` cũ nếu file ảnh còn tồn tại.
- **Kết quả 2026-11-26 (verify độc lập bằng subagent)**: tải đủ **2.210/2.210 ảnh**, **0 lỗi**, tổng **110,6 MB**. Mọi file đều là JPEG/RGB với cạnh dài ≤ 512px; `artworks.parquet` có cột `image_path` khớp 1-1 với file trên đĩa. `uv run pytest -q` → **243 passed**.

### [x] T1.7 Đóng gói artefact (2026-11-26)
- **Files**: `src/artrag/tools/artifacts.py`, `src/artrag/tools/__init__.py`, `tests/test_artifacts.py`.
- **Nội dung**: `pack` (zip các thư mục kb, images, index, cache/captions*) và `unpack <zip>`; in kích thước; kèm `manifest.json` trong zip (parts/files/bytes/kb_hash) để `unpack` đối chiếu.
- **Verify**: `uv run python -m artrag.tools.artifacts pack --only kb,images` → tạo zip; `unpack` vào thư mục tạm thì số file khớp.
- **Sau task**: upload zip lên Google Drive và tạo Kaggle Dataset (thủ công, người dùng làm).
- **Kết quả 2026-11-26 (verify độc lập bằng subagent)**: `pack --only kb,images` → **2.214 file** (kb 4 · images 2.210), dung lượng gốc 117,8 MB → zip 116,7 MB; `unpack` đối chiếu manifest → **khớp hoàn toàn**. `cache/` chỉ gom `cache/captions*` (đã kiểm chứng `cache/llm`, `cache/vqa` bị loại). `unpack --strict` trả exit code 1 với thông báo lỗi rõ ràng khi zip lệch manifest. `uv run pytest -q` → **263 passed**.

---

## M2. Registry và backend LLM (tuần 1–2)

### [x] T2.1 Registry lõi, không có model thật (§6.2) (2026-11-26)
- **Files**: `src/artrag/registry.py`, `configs/models.yaml`, `tests/test_registry.py`.
- **Nội dung**: `ModelSpec`, `ModelHandle`; `@register_loader(task)`; `DeviceManager` (interface có `available()`, `mem_allocated()`, `empty_cache()`, có bản giả để test); lazy get, khóa theo tên, LRU với ngân sách và pinned, OOM → evict → thử lại → CPU; `stats()`.
- **Verify**: `uv run pytest tests/test_registry.py -q`. Các test dùng loader giả và GPU giả: (a) chỉ nạp một lần, (b) vượt ngân sách thì evict đúng model LRU, không evict model pinned, (c) OOM thì fallback sang cpu, (d) `stats()` đúng.
- **Kết quả 2026-11-26 (verify độc lập bằng subagent)**: `tests/test_registry.py` → **26 passed**; toàn bộ suite → **289 passed** (263 cũ + 26 mới). `src/artrag/registry.py` **không import torch ở thời điểm import** (torch chỉ được import lazy trong `TorchDeviceManager`) nên module import được khi chưa cài torch — torch sẽ thêm ở T2.2. `configs/models.yaml` khớp **100%** với bảng model trong §6.2 (11 model, cùng `task`/`hf_id`/`est_vram_mb`).
- **Ghi chú thiết kế**: thêm khối `models:` (chỉ có `evict_to: cpu`) vào `configs/default.yaml`; `configs/models.yaml` nằm **ngoài** chuỗi gộp §6.1 và được nạp riêng bởi `load_model_specs(cfg)`, thứ tự `configs/models.yaml` < `cfg["models"]` (profile/experiment/--set) — `configs/profiles/local.yaml` đã có sẵn khối `models:` (ghim `clip`, đẩy `text_embed`/`vi2en`/`en2vi` về CPU) và nay được dùng thật.

### [x] T2.2 Loader thật cho model thị giác và embedding (2026-11-27)
- **Files**: `src/artrag/models/loaders.py` (đăng ký clip, sbert, blip_itm, blip_cap, blip_vqa, seq2seq, nli), `src/artrag/registry.py` (`--smoke`).
- **Deps**: thêm `torch` (index cu121 cho Windows), `transformers`, `sentence-transformers`, `faiss-cpu`, `pillow`.
- **Verify** (local): `uv run python -m artrag.registry --smoke clip blip_itm text_embed` → mỗi model in device/dtype, VRAM thực, thời gian nạp và một lần infer thử (ảnh fixture + câu "a painting").
- **Done khi**: chạy được trên GTX 1650. Nếu model nào không vừa thì ghi lại vào profile `local` (devices: cpu).
- **Kết quả 2026-11-27 (verify độc lập bằng subagent)**: `--smoke` nạp + suy luận thử **cả 7 loader** thành công trên GTX 1650: `clip` (fp16/cuda, 296 MB, dim 512, cos 0.287), `text_embed` (BGE-M3 fp32/cpu, dim 1024), `blip_itm` (fp16/cuda, itm_score 0.684), `blip_cap` (fp16/cuda, caption "a painting of two men rowing on a lake" — đúng với tranh Eakins trong KB), `blip_vqa` (fp16/cuda, trả lời "4"), `vi2en` (fp32/cpu, "Xin chào" → "Good afternoon, sir."), `nli` (fp16/cuda, entailment). `en2vi` cũng đã kiểm riêng ("Hello" → "Xin chào"). `tests/test_loaders.py` → **25 passed**; toàn bộ suite → **314 passed**. Cross-check `configs/models.yaml` với bảng §6.2 của design: **11/11 khớp tuyệt đối**.
- **Ghi chú phụ thuộc (quan trọng cho máy khác)**: `transformers` phải ghim `>=4.46,<4.47` — từ 4.47 trở đi, transformers chặn `torch.load` khi torch < 2.6 (CVE-2025-32434), mà index cu121 chỉ có torch tới 2.5.1, nên các checkpoint chỉ có `.bin` (CLIP, BGE-M3, BLIP-ITM/CAP, vinai) sẽ không nạp được. Thêm `accelerate` (bắt buộc cho `low_cpu_mem_usage=True`), `protobuf` + `sentencepiece` (tokenizer MBart của vinai).
- **Ghi chú thiết kế**: thêm `src_lang`/`tgt_lang` vào `configs/models.yaml` cho model dịch vinai — MBart **bắt buộc** set `src_lang` cho tokenizer và `decoder_start_token_id` = token ngôn ngữ đích khi generate, nếu không sinh ra chuỗi lặp vô nghĩa ("Xin Xin Xin ..."). Số đo thực tế ghi ở §6.2.1 của SYSTEM_DESIGN.
- **Profile `local`**: `text_embed` (BGE-M3) và `vi2en`/`en2vi` giữ ở CPU; `nli` chuyển sang fp16 (539 MB thay vì 1 065 MB, nhãn dự đoán không đổi); `est_vram_mb` của `blip_vqa` nâng 750 → 800 theo số đo.

### [x] T2.3 Interface LLM + cache + retry + openai_compat (§6.3) (2026-11-27)
- **Files**: `src/artrag/llm/{base.py,cache.py,openai_compat.py,__init__.py}` (`get_backend(cfg)`), `tests/test_llm_cache.py`.
- **Deps**: `httpx` (hoặc `openai`).
- **Nội dung**: `LLMResult`; `CachedBackend`, `RetryBackend` (backoff, rpm limiter); client openai-compat chung cho Groq/NIM.
- **Verify**:
  - `uv run pytest tests/test_llm_cache.py -q` (mock HTTP: gọi lần 2 thì `cached=True`; 429 thì retry);
  - lời gọi thật (cần key trong `.env`): `uv run python -m artrag.llm --backend groq --prompt "Xin chào"` → có câu trả lời, in latency; chạy lại thì in `cached`.
- **Kết quả 2026-11-27 (verify độc lập bằng subagent)**: `tests/test_llm_cache.py` → **23 passed**; toàn bộ suite → **337 passed** (314 cũ + 23 mới). Gọi thật: Groq (`qwen/qwen3.8-27b`) trả lời tiếng Việt, lần 2 in `cached=True`; NVIDIA NIM (`nvidia/nemotron-3.5-bolt-e4b`) trả lời sau ~2,3 s. Chuỗi fallback `groq -> nim` hoạt động: khi ép `base_url` của groq thành URL sai, nim tiếp quản.
- **Ghi chú thiết kế**: dùng `requests` (đã có dependency) thay vì `httpx`; thêm `retry.py` (RateLimiter + RetryBackend), `fallback.py` (FallbackBackend) và `__main__.py` (CLI). `api_key_env` nay nhận danh sách qua `api_key_envs` → client xoay vòng key khi provider trả 401/429. `BackendError` mang `status_code`; lỗi 429/5xx → `RetryableBackendError` (retry), 4xx khác → `BackendError` (không retry).
- **Ghi chú tên model**: tên model cũ trong design đã hết hạn — Groq không còn `llama-3.3-70b-versatile`, NIM trả HTTP 410 cho `meta/llama-3.3-70b-instruct`. Đã cập nhật `configs/default.yaml` và `configs/profiles/api.yaml` theo danh sách model hiện hành (đọc 2026-11-27).

### [x] T2.4 Backend hf_local (Qwen) (2026-11-27)
- **Files**: `src/artrag/llm/hf_local.py`, `src/artrag/models/loaders.py` (loader `causal_lm`), `src/artrag/llm/__init__.py`, `src/artrag/llm/__main__.py`, `src/artrag/models/{inference,smoke}.py`, `tests/test_llm_hf_local.py`.
- **Deps**: thêm `bitsandbytes>=0.45`.
- **Verify** (local): `uv run python -m artrag.llm --backend hf_local --model qwen3b --prompt "Thủ đô của Việt Nam là gì?"` → câu trả lời tiếng Việt, in VRAM và tokens/giây.
- **Nếu lỗi bitsandbytes**: đổi sang `qwen1_5b` fp16, ghi vào `profiles/local.yaml` và vào §10 của design.
- **Kết quả 2026-11-27 (verify độc lập bằng subagent)**:
  - `uv run pytest -q` → **361 passed** (337 cũ + 24 mới trong `tests/test_llm_hf_local.py`).
  - Lệnh verify → `Thủ đô của Việt Nam là Hà Nội.` · `device=cuda | dtype=fp16 | quant=4bit | vram_mb=1960.0 | tokens_per_s=1.81–3.57`.
  - `uv run python -m artrag.registry --smoke qwen3b` → OK, sinh "The capital of Vietnam is Hanoi."
  - **bitsandbytes 0.50.2 chạy tốt trên Windows + GTX 1650** → **không cần** đường lùi `qwen1_5b` fp16. Đường lùi vẫn được giữ trong `configs/profiles/local.yaml` và đã kiểm chạy thành công (cuda/fp16, 2 945 MB VRAM, cùng câu trả lời đúng).
- **Ghi chú thiết kế**:
  - Backend local **không** bọc `RetryBackend` (chỉ `CachedBackend`) — lỗi cục bộ không hết khi thử lại mà mỗi lần thử lại tốn hàng chục giây nạp model. Xem `LOCAL_BACKENDS` trong `src/artrag/llm/__init__.py`.
  - Lượng tử hoá 4-bit (NF4) dùng `device_map={"": device}` (không `device_map="auto"` — registry là nguồn sự thật duy nhất về thiết bị) và **bắt buộc phải trên GPU**; loader ném `ModelLoadError` nếu registry chọn CPU.
  - Greedy (`temperature = 0`) phải xoá `temperature`/`top_p`/`top_k` khỏi tham số sinh, nếu không transformers cảnh báo xung đột với `do_sample=False`.
  - Thứ tự ưu tiên tên model: `llm.model` → `backends.hf_local.model` → `qwen3b`. `llm.model` ưu tiên vì profile ghi đè khoá này.
  - Thêm `dtype: fp16` cho `qwen3b` trong `configs/models.yaml` (thiếu thì registry ép `fp32` và báo sai dtype).
  - `qwen3b` 4-bit tiết kiệm hơn `qwen1_5b` fp16 gần 1 GB VRAM (1 960 vs 2 945 MB) → giữ `qwen3b` làm mặc định profile `local`.

### [x] T2.5 Backend flan_t5 + bảo vệ citation khi dịch (2026-11-27)
- **Files**: `src/artrag/llm/flan_t5.py`, `src/artrag/pipeline/translate.py` (vi2en/en2vi, cache, placeholder), `tests/test_translate_placeholder.py`.
- **Verify**: `uv run pytest tests/test_translate_placeholder.py -q` (dùng dịch giả: `[IMG-01]` được giữ nguyên); smoke `uv run python -m artrag.llm --backend flan_t5 --prompt "..."`.
- **Kết quả 2026-11-27 (verify độc lập bằng subagent)**: `tests/test_translate_placeholder.py` → **36 passed**; toàn bộ suite → **395 passed**. Smoke: `flan_t5` nạp cuda/fp32, VRAM đo **947 MB**, sinh ~1,7–1,9 token/giây. Câu trả lời tiếng Việt **không rỗng**, nhãn `[IMG-01]` sống sót qua sinh → chọn nhãn → dịch (`labels=["[IMG-01]"]`). Cache đĩa hoạt động: `cached=True` (latency 0 ms), `data/cache/translate` có 10 entry.
- **Ghi chú thiết kế (2 điểm quan trọng)**:
  - **Cắt output của model encoder-decoder**: FLAN-T5 là encoder-decoder nên output có dạng `[decoder_start_token] + token_sinh_moi`, **không** phải `prompt + token_sinh_moi`. Phải cắt từ vị trí 1 — cắt theo độ dài prompt làm mất token sinh đầu tiên (prompt ngắn -> chuỗi rỗng).
  - **Cơ chế hai lượt**: FLAN-T5-base **không tự phát lại** nhãn `[IMG-xx]` (đo thực tế 0/11 mẫu prompt trả lời có nhãn). Vì vậy lượt 1 sinh câu trả lời, lượt 2 hỏi riêng "câu trả lời này đến từ nhãn bằng chứng nào" rồi chèn nhãn vào (chỉ giữ nhãn có thật trong context, tránh bịa citation).
  - **Chỉ dẫn lượt 1 KHÔNG được nhắc trích nhãn**: câu "cite the evidence labels exactly as written" làm model chỉ phát nhãn (`[IMG-01]. [IMG-01]`) — 8/15 mẫu rỗng nội dung. Chỉ dẫn dùng *"Answer the question in a short sentence, using only the evidence."* -> 15/15 câu hỏi có nội dung đúng. Prompt phải kết thúc bằng `Answer:`.
  - **Bảo vệ citation khi dịch**: văn bản được chia đoạn theo nhãn `[IMG-xx]`; chỉ đoạn thuần mới đưa qua máy dịch, đoạn chứa nhãn giữ nguyên -> nhãn không bao giờ bị máy dịch làm hỏng. Bất biến `join_segments(split_segments(t)) == t`.
  - **Hạn chế của FLAN-T5-base**: không từ chối được, với bằng chứng thiếu nó **bịa** (vd. đấu giá "$500.000"). Logic abstain **không** được đặt trong prompt mà phải ở tầng quy tắc của pipeline (§5.9).

---

## M3. Index offline ☁️ (tuần 1–2)

### [ ] T3.1 VectorIndex + CLIP index (§4 bước 6)
- **Files**: `src/artrag/index/{vector_index.py,clip_index.py}`, `tests/test_vector_index.py`.
- **Nội dung**:
  - `VectorIndex` (build từ npy, lưu/đọc, `search(q,k)`, `meta.json` có `kb_hash`, `StaleIndexError`);
  - `clip_index build|check|search "text"`.
- **Verify**:
  - `uv run pytest tests/test_vector_index.py -q` (vector ngẫu nhiên, top-1 của chính nó);
  - local: `uv run python -m artrag.index.clip_index build --limit 100` rồi `search "two men rowing a boat"` → in top-5 kèm điểm.
- **Chạy đầy đủ** ☁️: notebook (T3.4).

### [ ] T3.2 Text index BGE-M3 (§4 bước 7)
- **Files**: `src/artrag/index/text_index.py`.
- **Verify**: `build --limit 500` (CPU) rồi `search "ai vẽ tranh hai anh em chèo thuyền"` → top chunk thuộc `cma-151904` (nếu nằm trong 500 chunk đầu; không thì chạy với `--ids cma-151904`).

### [ ] T3.3 Caption toàn bộ (§4 bước 8)
- **Files**: `src/artrag/index/captions.py`.
- **Verify**: `build --limit 10` → parquet 10 dòng, in 3 caption và ms/ảnh. Chạy lại thì 0 ảnh mới.

### [ ] T3.4 Notebook offline mỏng ☁️
- **Files**: `notebooks/colab_offline.ipynb` (dùng được cho cả Kaggle).
- **Nội dung** (chỉ các cell gọi lệnh): clone repo → `pip install -e .` → mount Drive hoặc đọc input Kaggle → `unpack` KB và ảnh → `clip_index build` → `text_index build` → `captions build` → `artifacts pack` → lưu lên Drive.
- **Verify**: chạy hết trên Colab T4 và ghi thời gian từng bước vào `results/offline_build.md`. Ở local, `unpack` rồi chạy `clip_index check` và `text_index check` → OK, `kb_hash` khớp.

---

## M4. Truy xuất và rerank (tuần 2)

### [ ] T4.1 Query understanding (§5.2)
- **Files**: `src/artrag/pipeline/query.py`, `configs/qtype_rules.yaml`, `tests/test_query.py`.
- **Verify**: `uv run pytest tests/test_query.py -q`, gồm ≥20 câu VI/EN và QType mong đợi; phát hiện ngôn ngữ; strip cụm nghi vấn; trích tiêu đề trong ngoặc kép. Dùng dịch giả nên không cần model.

### [ ] T4.2 Fuse (hàm thuần) (§5.3)
- **Files**: `src/artrag/pipeline/fuse.py`, `tests/test_fuse.py`.
- **Verify**: `uv run pytest tests/test_fuse.py -q` (minmax với 1 phần tử hoặc tất cả bằng nhau, tác phẩm vắng mặt ở một nhánh, RRF, trọng số theo QType, lan truyền bio, giữ điểm raw).

### [ ] T4.3 Retriever lai (§5.3)
- **Files**: `src/artrag/pipeline/retrieve.py`.
- **Verify**: `uv run python -m artrag.pipeline.retrieve "Ai vẽ bức tranh hai anh em chèo thuyền đua?" --top 10` → in bảng (rank, artwork_id, title, s_clip, s_text, s_fused); `cma-151904` nằm trong top-3. Thêm `--image tests/fixtures/eakins.jpg` → top-1 là `cma-151904`.

### [ ] T4.4 BLIP-ITM rerank (§5.4)
- **Files**: `src/artrag/pipeline/rerank.py`, `tests/test_rerank_combine.py` (phần kết hợp điểm là hàm thuần).
- **Verify**: pytest; CLI `retrieve … --rerank --w 0.5 0.5` in thêm cột s_itm và s_final, kèm latency rerank.

---

## M5. Từ evidence đến câu trả lời (tuần 2)

### [ ] T5.1 Evidence builder + VQA (§5.6)
- **Files**: `src/artrag/pipeline/evidence.py`, `configs/vqa_questions.yaml`.
- **Verify**: `uv run python -m artrag.pipeline.evidence "Trong tranh hai anh em chèo thuyền có bao nhiêu người?"` → in evidence table (IMG-01..03, caption, VQA kèm conf, snippet). Chạy lại thì VQA lấy từ cache.

### [ ] T5.2 Context builder (§5.7)
- **Files**: `src/artrag/pipeline/context.py`, `tests/test_context.py`.
- **Verify**: pytest (không vượt ngân sách; thứ tự ưu tiên đúng; overview của mọi IMG luôn có mặt; cắt snippet).

### [ ] T5.3 Prompts + generate (§5.8)
- **Files**: `configs/prompts/{grounded_vi_v1,grounded_vi_v2_fewshot,grounded_vi_v3_structured,no_rag_vi,grounded_en_flan}.txt`, `src/artrag/pipeline/prompts.py`.
- **Verify**: `uv run python -m artrag.pipeline.prompts --name grounded_vi_v1 --demo` in ra messages hoàn chỉnh với evidence giả.

### [ ] T5.4 Citations + abstention (§5.5, §5.9)
- **Files**: `src/artrag/pipeline/{citations.py,abstain.py}`, `tests/test_citations.py`, `tests/test_abstain.py`.
- **Verify**: pytest. Bao phủ: các biến thể `[IMG-1]`, citation không hợp lệ bị xóa, nhận diện câu từ chối, `force_on_no_citation`, R1/R2/R3, ngoại lệ LOCATION, ảnh upload VISUAL.

### [ ] T5.5 Orchestrator `RAGPipeline` + CLI (§5.1)
- **Files**: `src/artrag/pipeline/rag.py`.
- **Verify** (với backend đang chạy được, ưu tiên hf_local, không thì groq):
  - `uv run python -m artrag.pipeline.rag "Ai đã vẽ bức tranh hai anh em chèo thuyền đua?"` → câu trả lời có `[IMG-01]` trỏ tới `cma-151904`, in trace tóm tắt;
  - `uv run python -m artrag.pipeline.rag "Bức Mona Lisa đang trưng bày ở phòng nào?"` → từ chối, `reason=LOW_SCORE` hoặc `LLM_DECLINED`;
  - `--json` in toàn bộ `RAGResponse`.
- **Done khi**: cả hai câu cho kết quả đúng; tổng latency local được ghi vào ghi chú.

### [ ] T5.6 Hai baseline: no-RAG và retrieval-only
- **Files**: `src/artrag/pipeline/baselines.py`.
- **Verify**: `uv run python -m artrag.pipeline.rag "…" --mode no_rag` và `--mode retrieval_only` cho ra output đúng định dạng `RAGResponse`.

---

## M6. Đánh giá (tuần 3)

### [ ] T6.1 Câu hỏi có đáp án (§8.1 B)
- **Files**: `src/artrag/eval/build_questions.py`, `tests/test_build_questions.py`.
- **Verify**: `uv run python -m artrag.eval.build_questions --only answerable --seed 42` → `eval/qa.jsonl`, in số lượng theo subtype. Pytest: gold khớp KB, có chia dev/test.

### [ ] T6.2 Câu hỏi không có đáp án và câu gây nhiễu
- **Files**: `build_questions.py` (thêm), `src/artrag/eval/famous_outside.yaml`.
- **Verify**: chạy đầy đủ → ≥150 câu; in 2 ví dụ mỗi subtype; danh sách ngoài kho đã được fuzzy check, không có tiêu đề nào khớp ≥90.

### [ ] T6.3 Truy vấn thị giác (LLM rewrite, có cache) (§8.1 A)
- **Files**: `src/artrag/eval/build_visual_queries.py`.
- **Verify**: `--limit 10` → in 10 cặp (description rút gọn → query VI/EN); kiểm tra không có tên riêng (regex các chữ viết hoa liên tiếp); sau đó chạy đầy đủ khoảng 250 câu.

### [ ] T6.4 Metric (§8.2)
- **Files**: `src/artrag/eval/metrics.py`, `tests/test_metrics.py`.
- **Verify**: pytest với các ví dụ tính tay cho Recall@K, MRR, P@5, contains, gold_token_recall, year_match, citation precision/hit, abstention P/R/F1, xử lý correct_premise, p50/p95.

### [ ] T6.5 Groundedness bằng NLI
- **Files**: `src/artrag/eval/groundedness.py`.
- **Verify**: `uv run python -m artrag.eval.groundedness --demo` → 3 cặp (entail / contradict / neutral) cho xác suất hợp lý.

### [ ] T6.6 Chạy eval
- **Files**: `src/artrag/eval/run_eval.py`, `configs/experiments/` (`base.yaml` và một file mỗi ô của ma trận §8.3).
- **Nội dung**: `--exp <name> --set split=dev --limit N --resume`; ghi `predictions.jsonl` từng dòng (chạy tiếp được nếu bị ngắt); chạy riêng `--task retrieval` hoặc `--task qa`.
- **Verify**: `uv run python -m artrag.eval.run_eval --exp base --task qa --limit 10` → `results/base/metrics.json` có đủ các khóa metric; chạy lại với `--resume` thì không chạy lại 10 câu cũ.

### [ ] T6.7 Tổng hợp bảng và phân tích lỗi (§8.5)
- **Files**: `src/artrag/eval/{aggregate.py,error_analysis.py}`.
- **Verify**: sau 2 thí nghiệm nhỏ, `aggregate` sinh `results/tables/T2_end2end.md`; `error_analysis --exp base` sinh csv có cột nhãn lỗi.

---

## M7. Giao diện Gradio (tuần 3)

### [ ] T7.1 Tab Hỏi đáp (§7.1)
- **Files**: `src/artrag/app/gradio_app.py`, `src/artrag/app/render.py` (render citation thành markdown/HTML, hàm thuần, có test).
- **Verify**: `uv run python -m artrag.app.gradio_app` rồi kiểm tra tay theo checklist:
  - [ ] câu trả lời có citation tô màu;
  - [ ] gallery khớp nhãn;
  - [ ] câu từ chối hiển thị lý do;
  - [ ] trace và thời gian theo bước;
  - [ ] slider K và preset trọng số có tác dụng;
  - [ ] đổi backend;
  - [ ] upload ảnh;
  - [ ] lịch sử phiên.

### [ ] T7.2 Tab Tìm kiếm ảnh và tab Caption/VQA (§7.2, §7.3)
- **Verify**: checklist:
  - [ ] lưới CLIP và lưới +BLIP đặt cạnh nhau;
  - [ ] đổi template;
  - [ ] caption với greedy/beam có hiện thời gian;
  - [ ] hỏi nhiều câu VQA trên một ảnh, có lịch sử.

### [ ] T7.3 Ổn định
- **Nội dung**: queue, thông báo lỗi thân thiện (thử: tắt mạng khi dùng backend groq, ảnh hỏng, câu hỏi rỗng), tab Hệ thống, `--share`, `--profile`.
- **Verify**: checklist lỗi ở trên không làm sập app; latency của câu hỏi thứ hai thấp hơn câu đầu (nhờ cache và model đã nạp).

---

## M8. Thí nghiệm và báo cáo (tuần 4–5)

### [ ] T8.1 ☁️ Chạy ma trận thí nghiệm T1–T5 (§8.3) bằng `notebooks/kaggle_eval.ipynb` (mỏng). Verify: đủ file `metrics.json` cho mọi exp; chạy `aggregate`.
### [ ] T8.2 Hiệu chỉnh τ, δ trên dev → cập nhật `default.yaml` → chạy test. Verify: bảng T4 và hình đường cong.
### [ ] T8.3 Phân tích lỗi: chọn ≥10 thành công và ≥10 thất bại, chấm tay 30 mẫu groundedness theo rubric. Verify: `docs/report/error_cases.md`.
### [ ] T8.4 README: hướng dẫn tái tạo (cài đặt → tải artefact hoặc build lại từ đầu → chạy app → chạy eval).
### [ ] T8.5 Notebook nộp bài `notebooks/final_end_to_end.ipynb`: chỉ gọi package (unpack artefact → demo pipeline 5 câu → tải bảng kết quả → mở Gradio). Verify: Restart & Run All chạy hết trên máy local.
### [ ] T8.6 Báo cáo (dàn ý: use case, kiến trúc, dữ liệu, phương pháp, thí nghiệm, phân tích lỗi, hạn chế, hướng phát triển).

---

## M9. Điểm cộng (chỉ khi M0–M7 xong) (§10.2)
- [ ] T9.1 CLIP multilingual text encoder (dùng lại index ảnh) + ablation so với dịch máy.
- [ ] T9.2 Kiểm tra mâu thuẫn bằng NLI (luật R6).
- [ ] T9.3 Cross-encoder `bge-reranker-v2-m3` cho chunk.
- [ ] T9.4 Giọng nói (PhoWhisper + MMS-TTS).
- [ ] T9.5 Thí nghiệm decoding caption (T6 của §8.3).

---

## Mốc thời gian
| Tuần | Task | Mốc kiểm tra |
|---|---|---|
| 1 | T0.1–T1.7, T2.1 | KB ≥2.000 tác phẩm, có ảnh, đã đóng gói |
| 2 | T2.2–T2.5, T3.x ☁️, T4.x, T5.1–T5.3 | index xong; CLI retrieve đúng câu Eakins |
| 3 | T5.4–T5.6, T6.x, T7.x | CLI RAG trả lời/từ chối đúng; app chạy; eval chạy được 10 câu |
| 4 | T8.1–T8.3 | đủ bảng kết quả và phân tích lỗi |
| 5 | T8.4–T8.6, M9 nếu còn thời gian | nộp bài |
