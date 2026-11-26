"""Kiểm tra độc lập kết quả T1.4 (viết để chạy bởi subagent verify).

Dùng:
    uv run python scripts/audit_t14.py
Exit code 0 = PASS, 1 = FAIL.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WIKI = ROOT / "data" / "raw" / "wiki"
KB = ROOT / "data" / "kb" / "artworks.parquet"

problems: list[str] = []
notes: list[str] = []


def fail(msg: str) -> None:
    problems.append(msg)


def ok(msg: str) -> None:
    notes.append(msg)


def main() -> int:
    if not WIKI.is_dir():
        fail(f"thư mục cache không tồn tại: {WIKI}")
        return 1

    files = sorted(WIKI.glob("*.json"))
    if not files:
        fail("không có file cache nào trong data/raw/wiki")
        return 1

    records = []
    for path in files:
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:  # noqa: BLE001
            fail(f"{path.name}: JSON không đọc được ({exc})")

    total = len(records)
    status = Counter(r.get("status") for r in records)
    found = status.get("found", 0)
    not_found = status.get("not_found", 0)
    error = status.get("error", 0)

    ok(f"tổng cache files = {total}")
    ok(f"status: found={found} not_found={not_found} error={error}")
    ok(f"hit rate = {found / total * 100:.1f}%")

    # 1. Hit rate >= 60%
    if found / total < 0.60:
        fail(f"hit rate {found / total * 100:.1f}% < 60%")
    else:
        ok("PASS hit rate >= 60%")

    # 2. Không có record error
    if error:
        fail(f"có {error} record status=error")
    else:
        ok("PASS không có record error")

    # 3. Mỗi record found đều phải có qid, url, summary
    for r in records:
        if r.get("status") != "found":
            continue
        aid = r.get("artist_id")
        if not r.get("wikidata_qid"):
            fail(f"{aid}: found nhưng thiếu wikidata_qid")
        if not r.get("wiki_url"):
            fail(f"{aid}: found nhưng thiếu wiki_url")
        if not (r.get("wiki_summary") or "").strip():
            fail(f"{aid}: found nhưng wiki_summary rỗng")
        if not r.get("matched_label"):
            fail(f"{aid}: found nhưng thiếu matched_label")
        if r.get("match_score") is None:
            fail(f"{aid}: found nhưng thiếu match_score")

    # 4. not_found record phải có matched_label rỗng (không match giả)
    for r in records:
        if r.get("status") == "not_found" and r.get("matched_label"):
            fail(f"{r.get('artist_id')}: not_found nhưng vẫn có matched_label")

    # 5. Kiểm tra riêng các trường hợp đã bị false positive trước đây
    by_name = {r.get("name"): r for r in records}
    checks = [
        ("Zhou Chen", "Q3575430", "không được match Shen Zhou Q558863"),
        ("Agnolo Bronzino", None, "phải tìm thấy Bronzino"),
        ("Thomas Eakins", None, "phải tìm thấy"),
        ("Katsushika Ōi", None, "phải tìm thấy"),
    ]
    for name, expected_qid, msg in checks:
        rec = by_name.get(name)
        if rec is None:
            fail(f"không tìm thấy record cho artist '{name}'")
            continue
        if rec.get("status") != "found":
            fail(f"'{name}': status={rec.get('status')} ({msg})")
            continue
        if expected_qid and rec.get("wikidata_qid") != expected_qid:
            fail(f"'{name}': qid={rec.get('wikidata_qid')} != {expected_qid}")
        else:
            ok(f"PASS '{name}' -> {rec.get('wikidata_qid')} ({rec.get('wiki_title')})")

    # 6. Zhou Chen mã số có thể bị nhiều: phải tìm đúng
    zhou = [r for r in records if r.get("name") == "Zhou Chen"]
    for rec in zhou:
        if rec.get("wikidata_qid") == "Q558863":
            fail("Zhou Chen vẫn bị match Shen Zhou Q558863")

    # 7. Không có QID rỗng / không phải QID trong record found
    for r in records:
        if r.get("status") == "found":
            qid = str(r.get("wikidata_qid") or "")
            if not qid.startswith("Q") or not qid[1:].isdigit():
                fail(f"{r.get('artist_id')}: wikidata_qid không hợp lệ: {qid!r}")

    # 8. So sánh số artist trong KB
    try:
        import pandas as pd

        df = pd.read_parquet(KB)
        artists = set(df["artist_id"].dropna().unique()) if "artist_id" in df.columns else set()
        cached = set(r.get("artist_id") for r in records)
        missing = artists - cached
        extra = cached - artists
        ok(f"KB artists = {len(artists)}, cached = {len(cached)}")
        if missing:
            fail(f"{len(missing)} artist trong KB chưa có cache wiki")
        if extra:
            fail(f"{len(extra)} cache không thuộc KB")
        if not missing and not extra:
            ok("PASS đối chiếu artist_id KB <-> cache: khớp 100%")
    except Exception as exc:  # noqa: BLE001
        notes.append(f"không đối chiếu được với KB parquet: {exc}")

    # 9. Phân bố match_score
    scores = [r["match_score"] for r in records if r.get("status") == "found" and isinstance(r.get("match_score"), (int, float))]
    if scores:
        ok(f"match_score: min={min(scores):.1f} median={sorted(scores)[len(scores) // 2]:.1f} max={max(scores):.1f}")
        low = [r for r in records if r.get("status") == "found" and isinstance(r.get("match_score"), (int, float)) and r["match_score"] < 88]
        if low:
            notes.append(f"{len(low)} record found có match_score < 88 (do đường dẫn fallback):")
            for r in low[:15]:
                notes.append(f"    {r.get('name')} -> {r.get('wikidata_qid')} score={r.get('match_score')}")

    # 10. Tồn tại record có wiki_summary rất ngắn (cảnh báo)
    short = [r for r in records if r.get("status") == "found" and len((r.get("wiki_summary") or "").strip()) < 40]
    if short:
        notes.append(f"{len(short)} record found có summary < 40 ký tự (cảnh báo)")
        for r in short[:10]:
            notes.append(f"    {r.get('name')}: {(r.get('wiki_summary') or '')[:80]!r}")

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
