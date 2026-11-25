# HANDOFF

Trang thai: M0 xong, M1 T1.1-T1.3 xong. Tiep theo: M1 (T1.4 Enrich Wikipedia).

## Repo

- Path: `D:\University\RAG\art_rag`
- Git: branch `main`, 8 commit, working tree sach.
- Python: 3.11.15 do `uv` quan ly (`uv python install 3.11` da chay). Khong dung Python 3.15 cua he thong.
- Env: `uv sync` da xong, 29 package. Chay moi thu bang `uv run ...`.

## Da hoan thanh

| Task | Commit | File chinh |
|---|---|---|
| T0.1 | d237efc | `pyproject.toml`, `.gitignore`, `.env.example`, `src/artrag/__init__.py`, `tests/test_smoke.py`, `README.md` |
| T0.2 | 0c0d070 | `src/artrag/config.py`, `configs/default.yaml`, `configs/profiles/{local,colab,kaggle,api}.yaml`, `src/artrag/errors.py`, `tests/test_config.py` |
| T0.3 | 63fed8f | `src/artrag/trace.py`, `src/artrag/utils/{io,text,hashing}.py`, `tests/test_trace.py` |
| docs | 721232c | tick checkbox T0.1-T0.3 trong `docs/IMPLEMENTATION_PLAN.md` |
| docs | ec5cd0f | quy tac parse/chunk §2.3 trong `SYSTEM_DESIGN.md`; khoa `data.*` + `data.chunking.*` trong `configs/default.yaml` |
| T1.1 | 5d3e590 | `src/artrag/data/{__init__,fetch_cma}.py`, `tests/test_fetch_cma.py` |
| T1.2 | 73bb903 | `src/artrag/data/cma_parse.py`, `tests/fixtures/cma_sample.json`, `tests/test_cma_parse.py` |
| T1.3 | a76aff6 | `src/artrag/data/{chunking,build_kb}.py`, `tests/test_chunking.py` |

Test: `uv run pytest -q` -> 94 passed.

## Du lieu da tai (data/, khong commit)

- `data/raw/cma/page_{00000,01000,02000,03000}.json`: 3.964 ban ghi CMA (Painting, CC0, co anh). Moi trang co khoa `_meta` = {skip, limit, count, total, fetched_at}.
- `data/kb/artworks.parquet`: 2.210 tac pham x 27 cot (loc `description` >= 50 ky tu).
- `data/kb/chunks.parquet`: 9.721 chunk x 7 cot; max 250 tu/chunk, khong chunk rong.
- Thong ke thuc te: description 2210, did_you_know 1372, provenance 2054, exhibitions 1697, current_location 438, prev_owner 1772; 11 department; 1.833 tac pham co `artist_name`.

## API quan trong da co san

### `artrag.config`

```python
from artrag.config import load_config, config_hash, paths, parse_overrides

cfg = load_config(profile=None, experiment=None, overrides=["a.b=c"], extra=None)
cfg["retrieval"]["top_n"]   # 20
cfg.retrieval.top_n         # 20
cfg.get("x.y", default)
cfg.raw                     # dict thuan
config_hash(cfg)            # sha256 16 hex, on dinh
p = paths(cfg)              # Paths(data_dir, results_dir, raw, kb, images, index, cache)
paths(cfg, ensure=True)     # tao thu muc con
```

Thu tu gop (§6.1): `default.yaml` < `profiles/{profile}.yaml` < `experiments/{exp}.yaml` < `--set` < `extra`.
Profile lay tu tham so, roi `ARTRAG_PROFILE`, mac dinh `local`. `${env:VAR,default}` duoc resolve; thieu bien va khong co default -> giu nguyen placeholder.

### `artrag.trace`

```python
from artrag.trace import Trace

trace = Trace(config_hash=..., profile="local")
with trace.step("retrieve_clip") as s:
    s.set_info(top1_raw=0.29)
with trace.step("rerank_itm", required=False) as s:   # optional: loi -> degraded, khong nem
    ...
with trace.step("generate") as s:                    # required: loi -> error, nem lai
    ...
trace.to_dict()   # JSON-safe, shape §6.6
trace.summary()   # bang "name | ms | status"
```

### `artrag.errors`

`ArtragError` base (co `message`, `hint`), cac loi con: `DataNotReadyError`, `StaleIndexError`, `ModelLoadError`, `BackendError`, `InvalidInputError`, `ConfigError`.

### `artrag.utils`

- `io`: `ensure_dir`, `read_json`, `write_json`, `read_jsonl`, `write_jsonl` (ghi atomic).
- `text`: `strip_html`, `normalize_text` (bo dau, lowercase), `split_sentences`.
- `hashing`: `sha256_json`, `sha256_text`, `sha256_file`.

## API moi trong M1

### `artrag.data.fetch_cma`

```python
from artrag.data.fetch_cma import page_path, build_params, is_cache_valid, fetch_page, fetch_all

page_path(raw_dir, skip)            # -> raw_dir/cma/page_{skip:05d}.json
build_params(cma_type, skip, limit)  # -> {"type":..., "limit":..., "skip":..., "cc0":1, "has_image":1}
is_cache_valid(payload, skip, limit) # _meta.skip == skip va _meta.limit >= limit
fetch_all(cfg, force=False, limit=None, page_size=None) -> dict
#   -> {"fetched_pages", "cached_pages", "records", "total", "raw_dir"}
```

CLI: `python -m artrag.data.fetch_cma [--limit N] [--page-size N] [--force] [--profile P] [--set a.b=c]`.

### `artrag.data.cma_parse`

```python
from artrag.data.cma_parse import (
    parse_artwork, parse_artist_description, artist_info,
    provenance_text, prev_owner, exhibitions_text, exhibition_items,
    overview_text, ARTWORK_COLUMNS,
)

parse_artwork(record) -> dict       # 27 cot dung §2.3
parse_artist_description("Thomas Eakins (American, 1844-1916)")
#   -> {"name", "nationality", "birth", "death"}; "c."/"active"/"before" -> birth None
prev_owner(provenance_entries) -> str | None
```

### `artrag.data.chunking`

```python
from artrag.data.chunking import count_words, chunk_sentences, chunk_single, make_chunk_rows, CHUNK_COLUMNS

count_words(text) -> int
chunk_sentences(text, max_words, overlap_sentences=1) -> list[str]
chunk_single(text, max_words) -> list[str]
make_chunk_rows(row, sentence_max_words=180, field_max_words=250,
                overlap_sentences=1, max_words_per_chunk=250) -> list[dict]
```

### `artrag.data.build_kb`

```python
from artrag.data.build_kb import load_records, build_rows, make_chunk_rows_for_all, write_parquet, print_stats
```

CLI: `python -m artrag.data.build_kb [--check] [--limit N] [--profile P] [--set a.b=c]`.
`--check` in so tac pham, do phu truong, phan bo department, chunk theo field, va canh bao chunk rong / vuot 250 tu.

## Luu y quan trong

- `normalize_text` bo dau bang NFD + map tay `đ -> d` (U+0111 khong decompose). Can dung cho fuzzy match ten.
- `Config` la dict wrapper tu viet (khong phai pydantic): them section moi khong can sua model.
- `paths()` mac dinh KHONG tao thu muc; dung `ensure=True` neu can.
- `configs/experiments/` chua tao; `load_config(experiment=...)` nem `ConfigError` neu thieu file.
- Chua nap model.
- `parse_artist_description` tra `birth`/`death` = None khi nam co "c."/"circa"/"active"/"before"/"after" (khong doan).
- `prev_owner` lay entry ngay truoc entry CMA **cuoi cung**; lay phan ten truoc `[` hoac `,`.
- `exhibitions_text` toi da 10 muc, moi nhat truoc; mo ta trien lam co dau phay nen KHONG dem bang `text.split("; ")`.
- File raw co khoa `_meta`; xoa `page_04000.json` (trang rong) neu bi tao nho.
- `.gitignore` dung `/data/` (co dau `/` o dau) de khong ignore `src/artrag/data/`.

## Task tiep theo: M1 (con lai)

Theo `docs/IMPLEMENTATION_PLAN.md`:
- **T1.4 Enrich Wikipedia** -> `src/artrag/data/enrich_wiki.py`
  - Wikidata `wbsearchentities` (ten + nam sinh de chon dung: kiem tra `P31=Q5` va `P569`) -> enwiki title -> REST summary
  - cache `raw/wiki/{artist_id}.json` (ke ca not_found), gioi han toc do, `--limit`
  - Verify: `uv run python -m artrag.data.enrich_wiki --limit 30` -> in ti le tim thay (muc tieu >=60%), 3 vi du
- **T1.5 Artists + chunk bio** -> them `--with-bio` vao `build_kb.py` (`artists.parquet`, chunk `artist_bio`)
- **T1.6 Tai anh** -> `src/artrag/data/images.py` (8 thread, RGB, canh dai 512px, JPEG q90, `images_failed.json`)
- **T1.7 Dong goi artefact** -> `src/artrag/tools/artifacts.py` (`pack`/`unpack`)

## Quy tac lam viec (tu CLAUDE.md, bat buoc)

1. Lam tung task mot theo `docs/IMPLEMENTATION_PLAN.md`. Chi doc § thiet ke duoc dan va file liet ke trong task. Khong quet lai toan repo.
2. Ket thuc moi task: chay lenh **Verify** cua task do, tick `[x]`, commit `T<id>: <mo ta>`.
3. Thiet ke can doi -> sua `docs/SYSTEM_DESIGN.md` truoc (ghi ngan ly do), sau do moi code.
4. Lenh goi mang/model that luc verify dung `--limit` nho.
5. Notebook (`notebooks/`) chi la wrapper mong goi `python -m artrag.<...>`; khong viet logic trong notebook.
6. Config-driven: khong hardcode tham so. Moi thu nghiem la 1 file `configs/experiments/*.yaml`.
7. Moi model lay qua `ModelRegistry`; GPU local la GTX 1650 4GB (khong bf16) -> fp16 tren cuda, fp32 tren CPU.
8. LLM qua interface `LLMBackend`, luon boc cache dia + retry. API key chi tu bien moi truong hoac `.env`.
9. `[IMG-xx]` la nhan cuc bo trong tung cau tra loi, map toi `artwork_id` (`cma-{id}`) qua evidence table.
10. Ham thuan (fuse, abstain, citations, context, metrics) phai test duoc khong can GPU/model.
11. `data/` khong commit; phan phoi qua `artrag.tools.artifacts pack/unpack`.
