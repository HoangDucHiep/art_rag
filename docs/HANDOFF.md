# HANDOFF

Trang thai: M0 xong. Tiep theo: M1 (T1.1 Fetch CMA).

## Repo

- Path: `D:\University\RAG\art_rag`
- Git: branch `main`, 4 commit, working tree sach.
- Python: 3.11.15 do `uv` quan ly (`uv python install 3.11` da chay). Khong dung Python 3.15 cua he thong.
- Env: `uv sync` da xong, 29 package. Chay moi thu bang `uv run ...`.

## Da hoan thanh (M0)

| Task | Commit | File chinh |
|---|---|---|
| T0.1 | d237efc | `pyproject.toml`, `.gitignore`, `.env.example`, `src/artrag/__init__.py`, `tests/test_smoke.py`, `README.md` |
| T0.2 | 0c0d070 | `src/artrag/config.py`, `configs/default.yaml`, `configs/profiles/{local,colab,kaggle,api}.yaml`, `src/artrag/errors.py`, `tests/test_config.py` |
| T0.3 | 63fed8f | `src/artrag/trace.py`, `src/artrag/utils/{io,text,hashing}.py`, `tests/test_trace.py` |
| docs | 721232c | tick checkbox T0.1-T0.3 trong `docs/IMPLEMENTATION_PLAN.md` |

Test: `uv run pytest -q` -> 26 passed.

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

## Luu y quan trong

- `normalize_text` bo dau bang NFD + map tay `đ -> d` (U+0111 khong decompose). Can dung cho fuzzy match ten.
- `Config` la dict wrapper tu viet (khong phai pydantic): them section moi khong can sua model.
- `paths()` mac dinh KHONG tao thu muc; dung `ensure=True` neu can.
- `configs/experiments/` chua tao; `load_config(experiment=...)` nem `ConfigError` neu thieu file.
- Chua tao `data/`, chua goi mang, chua nap model.

## Task tiep theo: M1

Theo `docs/IMPLEMENTATION_PLAN.md`:

- **T1.1 Fetch CMA (§2.1, §4 buoc 1)** -> `src/artrag/data/fetch_cma.py`
  - phan trang `limit=1000&skip=...`, `type=Painting&cc0=1&has_image=1`
  - luu `raw/cma/page_*.json`, bo qua trang da co, retry/backoff, `--limit`
  - Verify: `uv run python -m artrag.data.fetch_cma --limit 50` -> 1 file, in "50 records"; chay lai -> in "cached"
  - Can tao `src/artrag/data/__init__.py`
- **T1.2 Parse CMA (§2.3)** -> `src/artrag/data/cma_parse.py`, fixture `tests/fixtures/cma_sample.json`, `tests/test_cma_parse.py`
- **T1.3 Build KB** -> `src/artrag/data/build_kb.py`, `chunking.py`, `tests/test_chunking.py`
- **T1.4 Enrich Wikipedia** -> `src/artrag/data/enrich_wiki.py`
- **T1.5 Artists + chunk bio** -> them `--with-bio` vao `build_kb.py`
- **T1.6 Tai anh** -> `src/artrag/data/images.py`
- **T1.7 Dong goi artefact** -> `src/artrag/tools/artifacts.py`

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
