"""Kiểm tra độc lập kết quả T1.5 (viết để chạy bởi subagent verify).

Dùng:
    uv run python scripts/audit_t15.py
Exit code 0 = PASS, 1 = FAIL.

Kiểm tra ``data/kb/artists.parquet`` và các chunk ``artist_bio`` trong
``data/kb/chunks.parquet``:
- đối chiếu 1:1 với ``artist_id`` trong ``artworks.parquet``;
- ``bio_source`` hợp lệ và nhất quán với nội dung bio;
- mọi chunk ``artist_bio`` có ``artwork_id`` null, ``artist_id`` khác null,
  text không rỗng, không vượt 250 từ; ``chunk_id`` không trùng;
- **NGUYÊN VĂN**: tách lại ``bio`` nguồn bằng ``split_sentences`` phải cho ra
  đúng danh sách câu mà các chunk đã lưu, và ghép lại các chunk phải khớp
  nguyên văn văn bản gốc (sau khi chuẩn hóa khoảng trắng).
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from artrag.utils.text import split_sentences

ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / "data" / "kb"
ARTWORKS = KB / "artworks.parquet"
ARTISTS = KB / "artists.parquet"
CHUNKS = KB / "chunks.parquet"

MAX_WORDS = 250
VALID_SOURCES = {"cma", "wikipedia", "none"}

problems: list[str] = []
notes: list[str] = []


def fail(msg: str) -> None:
    problems.append(msg)


def ok(msg: str) -> None:
    notes.append(msg)


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return str(value).strip()


def main() -> int:
    for path in (ARTWORKS, ARTISTS, CHUNKS):
        if not path.exists():
            fail(f"thiếu file: {path}")
    if problems:
        print("FAIL: thiếu file cần kiểm tra")
        for p in problems:
            print("  [X]", p)
        return 1

    artworks = pd.read_parquet(ARTWORKS)
    artists = pd.read_parquet(ARTISTS)
    chunks = pd.read_parquet(CHUNKS)

    # 1. Schema artists.parquet
    expected_columns = [
        "artist_id",
        "name",
        "birth",
        "death",
        "nationality",
        "cma_bio",
        "wikidata_qid",
        "wiki_title",
        "wiki_url",
        "wiki_summary",
        "bio",
        "bio_source",
    ]
    if list(artists.columns) != expected_columns:
        fail(f"cột của artists.parquet không đúng: {list(artists.columns)}")
    else:
        ok("PASS schema artists.parquet đúng 12 cột")

    # 2. Đối chiếu 1:1 với artist_id trong artworks
    kb_artists = set(artworks["artist_id"].dropna().astype(str).unique())
    art_ids = artists["artist_id"].astype(str)
    counts = Counter(art_ids)
    duplicates = [aid for aid, n in counts.items() if n > 1]
    if duplicates:
        fail(f"{len(duplicates)} artist_id bị trùng trong artists.parquet")
    else:
        ok("PASS mỗi artist_id có đúng 1 dòng")

    missing = kb_artists - set(art_ids)
    extra = set(art_ids) - kb_artists
    ok(f"KB artists = {len(kb_artists)} | artists.parquet = {len(artists)}")
    if missing:
        fail(f"{len(missing)} artist trong artworks.parquet thiếu ở artists.parquet")
    if extra:
        fail(f"{len(extra)} artist trong artists.parquet không có trong artworks.parquet")
    if not missing and not extra:
        ok("PASS đối chiếu artist_id KB <-> artists.parquet: khớp 100%")

    # 3. bio_source hợp lệ
    sources = Counter(artists["bio_source"].astype(str))
    bad_sources = set(sources) - VALID_SOURCES
    if bad_sources:
        fail(f"bio_source không hợp lệ: {bad_sources}")
    else:
        ok(f"PASS bio_source hợp lệ: {dict(sources)}")

    # 4. Nhất quán nội dung bio
    for row in artists.itertuples(index=False):
        source = str(row.bio_source)
        bio = _text(row.bio)
        if source == "none":
            if bio:
                fail(f"{row.artist_id}: bio_source=none nhưng bio không rỗng")
            continue
        if not bio:
            fail(f"{row.artist_id}: bio_source={source} nhưng bio rỗng")
            continue
        if source == "cma":
            if not _text(row.cma_bio):
                fail(f"{row.artist_id}: bio_source=cma nhưng cma_bio rỗng")
            elif bio != _text(row.cma_bio):
                fail(f"{row.artist_id}: bio_source=cma nhưng bio khác cma_bio")
        if source == "wikipedia":
            if not _text(row.wiki_url):
                fail(f"{row.artist_id}: bio_source=wikipedia nhưng wiki_url rỗng")
            if not _text(row.wiki_summary):
                fail(f"{row.artist_id}: bio_source=wikipedia nhưng wiki_summary rỗng")

    # 5. Chunk artist_bio
    bio_chunks = chunks[chunks["field"] == "artist_bio"]
    ok(f"chunk artist_bio = {len(bio_chunks)} / tổng chunk = {len(chunks)}")

    if bio_chunks.empty:
        fail("không có chunk artist_bio nào trong chunks.parquet")
    else:
        if bio_chunks["artwork_id"].notna().any():
            bad = bio_chunks[bio_chunks["artwork_id"].notna()]["chunk_id"].tolist()
            fail(f"{len(bad)} chunk artist_bio có artwork_id khác null: {bad[:5]}")
        else:
            ok("PASS mọi chunk artist_bio có artwork_id null")

        if bio_chunks["artist_id"].isna().any():
            fail("có chunk artist_bio thiếu artist_id")
        else:
            ok("PASS mọi chunk artist_bio có artist_id")

        if not set(bio_chunks["artist_id"].astype(str)) <= kb_artists:
            fail("có chunk artist_bio trỏ tới artist_id không có trong KB")
        else:
            ok("PASS artist_id của chunk artist_bio đều có trong KB")

        empty = [c for c in bio_chunks["text"].tolist() if not _text(c)]
        if empty:
            fail(f"{len(empty)} chunk artist_bio có text rỗng")
        else:
            ok("PASS không có chunk artist_bio rỗng")

        over = bio_chunks[bio_chunks["n_words"] > MAX_WORDS]
        if len(over):
            fail(f"{len(over)} chunk artist_bio vượt {MAX_WORDS} từ")
        else:
            ok(f"PASS không chunk artist_bio nào vượt {MAX_WORDS} từ")

        dup = [cid for cid, n in Counter(bio_chunks["chunk_id"]).items() if n > 1]
        if dup:
            fail(f"{len(dup)} chunk_id artist_bio bị trùng: {dup[:5]}")
        else:
            ok("PASS chunk_id artist_bio không trùng")

        bad_id = [
            str(row.chunk_id)
            for row in bio_chunks.itertuples(index=False)
            if not str(row.chunk_id).startswith(f"{row.artist_id}#artist_bio#")
        ]
        if bad_id:
            fail(f"{len(bad_id)} chunk_id artist_bio sai định dạng: {bad_id[:5]}")
        else:
            ok("PASS định dạng chunk_id = {artist_id}#artist_bio#{i}")

        # chunk_id của các field khác không được trùng với artist_bio
        other_ids = set(chunks[chunks["field"] != "artist_bio"]["chunk_id"].astype(str))
        overlap = other_ids & set(bio_chunks["chunk_id"].astype(str))
        if overlap:
            fail(f"{len(overlap)} chunk_id bị trùng giữa artist_bio và field khác")
        else:
            ok("PASS không trùng chunk_id với các field khác")

    # 6. Chunk không rỗng / không vượt 250 từ (toàn bảng)
    empty_all = sum(1 for t in chunks["text"].tolist() if not _text(t))
    over_all = int((chunks["n_words"] > MAX_WORDS).sum())
    if empty_all or over_all:
        fail(f"chunks.parquet có {empty_all} chunk rỗng và {over_all} chunk vượt {MAX_WORDS} từ")
    else:
        ok(f"PASS toàn bộ {len(chunks)} chunk không rỗng và không vượt {MAX_WORDS} từ")

    # 6b. NGUYÊN VĂN: mỗi chunk phải là một đoạn con CÓ THẬT của bio nguồn,
    #     và ghép lại các chunk (đã bỏ phần overlap) phải tái tạo đúng văn bản
    #     gốc. Đây là kiểm tra bắt được lỗi "v. )" / "Millet (q.v." do chữ viết
    #     tắt bị cắt vỡ.
    bio_by_artist = {
        str(row.artist_id): _text(row.bio)
        for row in artists.itertuples(index=False)
        if _text(row.bio)
    }
    by_artist: dict[str, list[str]] = {}
    for row in bio_chunks.itertuples(index=False):
        by_artist.setdefault(str(row.artist_id), []).append(_text(row.text))

    def _norm(text: str) -> str:
        return " ".join(text.split())

    # 6b-1. Mỗi chunk phải xuất hiện nguyên văn trong bio nguồn.
    not_verbatim = []
    for row in bio_chunks.itertuples(index=False):
        source = bio_by_artist.get(str(row.artist_id))
        if not source:
            continue
        if _norm(_text(row.text)) not in _norm(source):
            not_verbatim.append(str(row.chunk_id))
    if not_verbatim:
        fail(
            f"{len(not_verbatim)} chunk artist_bio không phải đoạn con nguyên văn "
            f"của bio nguồn: {not_verbatim[:5]}"
        )
    else:
        ok("PASS mọi chunk artist_bio là đoạn con nguyên văn của bio nguồn")

    # 6b-2. Ghép lại các chunk (bỏ phần overlap 1 câu) phải tái tạo đúng bio.
    bad_rebuild = 0
    rebuild_examples: list[str] = []
    for artist_id, pieces in by_artist.items():
        source = bio_by_artist.get(artist_id)
        if not source:
            continue
        expected = split_sentences(source)
        stream: list[str] = []
        previous: list[str] = []
        for piece in pieces:
            current = split_sentences(piece)
            overlap = 0
            for k in range(min(len(previous), len(current)), 0, -1):
                if previous[-k:] == current[:k]:
                    overlap = k
                    break
            stream.extend(current[overlap:])
            previous = current
        if [_norm(s) for s in stream] != [_norm(s) for s in expected]:
            bad_rebuild += 1
            if len(rebuild_examples) < 3:
                rebuild_examples.append(
                    f"{artist_id}: {len(stream)} câu khớp {len(expected)} câu nguồn"
                )
    if bad_rebuild:
        fail(
            f"{bad_rebuild} artist khi ghép lại chunk không khớp NGUYÊN VĂN bio nguồn: "
            f"{rebuild_examples}"
        )
    else:
        ok(
            f"PASS ghép lại chunk của {len(by_artist)} artist khớp NGUYÊN VĂN bio nguồn"
        )

    # 6c. Tách lại bio nguồn bằng split_sentences phải bảo toàn nguyên văn.
    bad_split = 0
    split_examples: list[str] = []
    for artist_id, source in bio_by_artist.items():
        if _norm(" ".join(split_sentences(source))) != _norm(source):
            bad_split += 1
            if len(split_examples) < 3:
                split_examples.append(f"{artist_id}: split_sentences không bảo toàn văn bản")
    if bad_split:
        fail(
            f"{bad_split} artist mà split_sentences không bảo toàn nguyên văn bio: "
            f"{split_examples}"
        )
    else:
        ok("PASS split_sentences bảo toàn nguyên văn bio của mọi artist")

    # 6d. Không có mảnh vỡ do chữ viết tắt bị cắt sai: không được có mảnh rác
    #     kiểu "v. )" / "(q. v.", và mỗi CÂU trong chunk phải cân bằng ngoặc
    #     giống hệt như trong bio nguồn (nguồn CMA có vài bio vốn đã lệch
    #     ngoặc - không tính là lỗi chunking).
    broken_patterns = ("q. v. )", "v. )", "(q. v.", "(q.v. )")
    broken = []
    for row in bio_chunks.itertuples(index=False):
        text = _text(row.text)
        if any(pat in text for pat in broken_patterns):
            broken.append(str(row.chunk_id))
            continue
        source = bio_by_artist.get(str(row.artist_id), "")
        for sentence in split_sentences(text):
            if sentence.count("(") != sentence.count(")"):
                # chỉ coi là lỗi khi câu này KHÔNG xuất hiện nguyên văn trong
                # bio nguồn (tức lỗi do tách câu, không phải do nguồn)
                if _norm(sentence) not in _norm(source):
                    broken.append(str(row.chunk_id))
                    break
    if broken:
        fail(f"{len(broken)} chunk artist_bio bị vỡ do chữ viết tắt: {broken[:5]}")
    else:
        ok("PASS không có chunk artist_bio bị vỡ kiểu '(q.v.' / 'v. )' / ngoặc lệch")

    # 7. Số artist có bio
    with_bio = sources.get("cma", 0) + sources.get("wikipedia", 0)
    rate = with_bio / len(artists) if len(artists) else 0.0
    ok(f"artist có bio: {with_bio}/{len(artists)} ({rate:.1%})")
    if rate < 0.6:
        fail(f"tỷ lệ artist có bio {rate:.1%} < 60%")

    print("=" * 70)
    for n in notes:
        print("  ", n)
    print("=" * 70)
    if problems:
        print(f"FAIL ({len(problems)} vấn đề)")
        for p in problems:
            print("  [X]", p)
        return 1
    print("PASS: không có vấn đề")
    return 0


if __name__ == "__main__":
    sys.exit(main())
