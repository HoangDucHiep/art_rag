"""Kiem tra doc lap ket qua T1.4 (viet de chay boi subagent verify).

Dung:
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
        fail(f"thu muc cache khong ton tai: {WIKI}")
        return 1

    files = sorted(WIKI.glob("*.json"))
    if not files:
        fail("khong co file cache nao trong data/raw/wiki")
        return 1

    records = []
    for path in files:
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:  # noqa: BLE001
            fail(f"{path.name}: JSON khong doc duoc ({exc})")

    total = len(records)
    status = Counter(r.get("status") for r in records)
    found = status.get("found", 0)
    not_found = status.get("not_found", 0)
    error = status.get("error", 0)

    ok(f"tong cache files = {total}")
    ok(f"status: found={found} not_found={not_found} error={error}")
    ok(f"hit rate = {found / total * 100:.1f}%")

    # 1. Hit rate >= 60%
    if found / total < 0.60:
        fail(f"hit rate {found / total * 100:.1f}% < 60%")
    else:
        ok("PASS hit rate >= 60%")

    # 2. Khong co record error
    if error:
        fail(f"co {error} record status=error")
    else:
        ok("PASS khong co record error")

    # 3. Moi record found deu phai co qid, url, summary
    for r in records:
        if r.get("status") != "found":
            continue
        aid = r.get("artist_id")
        if not r.get("wikidata_qid"):
            fail(f"{aid}: found nhung thieu wikidata_qid")
        if not r.get("wiki_url"):
            fail(f"{aid}: found nhung thieu wiki_url")
        if not (r.get("wiki_summary") or "").strip():
            fail(f"{aid}: found nhung wiki_summary rong")
        if not r.get("matched_label"):
            fail(f"{aid}: found nhung thieu matched_label")
        if r.get("match_score") is None:
            fail(f"{aid}: found nhung thieu match_score")

    # 4. not_found record phai co matched_label rong (khong match gia)
    for r in records:
        if r.get("status") == "not_found" and r.get("matched_label"):
            fail(f"{r.get('artist_id')}: not_found nhung van co matched_label")

    # 5. Kiem tra rieng cac truong hop da bi false positive truoc day
    by_name = {r.get("name"): r for r in records}
    checks = [
        ("Zhou Chen", "Q3575430", "khong duoc match Shen Zhou Q558863"),
        ("Agnolo Bronzino", None, "phai tim thay Bronzino"),
        ("Thomas Eakins", None, "phai tim thay"),
        ("Katsushika Ōi", None, "phai tim thay"),
    ]
    for name, expected_qid, msg in checks:
        rec = by_name.get(name)
        if rec is None:
            fail(f"khong tim thay record cho artist '{name}'")
            continue
        if rec.get("status") != "found":
            fail(f"'{name}': status={rec.get('status')} ({msg})")
            continue
        if expected_qid and rec.get("wikidata_qid") != expected_qid:
            fail(f"'{name}': qid={rec.get('wikidata_qid')} != {expected_qid}")
        else:
            ok(f"PASS '{name}' -> {rec.get('wikidata_qid')} ({rec.get('wiki_title')})")

    # 6. Zhou Chen ma so co the bi nhieu: phai tim dung
    zhou = [r for r in records if r.get("name") == "Zhou Chen"]
    for rec in zhou:
        if rec.get("wikidata_qid") == "Q558863":
            fail("Zhou Chen van bi match Shen Zhou Q558863")

    # 7. Khong co QID rong / khong phai QID trong record found
    for r in records:
        if r.get("status") == "found":
            qid = str(r.get("wikidata_qid") or "")
            if not qid.startswith("Q") or not qid[1:].isdigit():
                fail(f"{r.get('artist_id')}: wikidata_qid khong hop le: {qid!r}")

    # 8. So sanh so artist trong KB
    try:
        import pandas as pd

        df = pd.read_parquet(KB)
        artists = set(df["artist_id"].dropna().unique()) if "artist_id" in df.columns else set()
        cached = set(r.get("artist_id") for r in records)
        missing = artists - cached
        extra = cached - artists
        ok(f"KB artists = {len(artists)}, cached = {len(cached)}")
        if missing:
            fail(f"{len(missing)} artist trong KB chua co cache wiki")
        if extra:
            fail(f"{len(extra)} cache khong thuoc KB")
        if not missing and not extra:
            ok("PASS doi chieu artist_id KB <-> cache: khop 100%")
    except Exception as exc:  # noqa: BLE001
        notes.append(f"khong doi chieu duoc voi KB parquet: {exc}")

    # 9. Phan bo match_score
    scores = [r["match_score"] for r in records if r.get("status") == "found" and isinstance(r.get("match_score"), (int, float))]
    if scores:
        ok(f"match_score: min={min(scores):.1f} median={sorted(scores)[len(scores) // 2]:.1f} max={max(scores):.1f}")
        low = [r for r in records if r.get("status") == "found" and isinstance(r.get("match_score"), (int, float)) and r["match_score"] < 88]
        if low:
            notes.append(f"{len(low)} record found co match_score < 88 (de duong dan fallback):")
            for r in low[:15]:
                notes.append(f"    {r.get('name')} -> {r.get('wikidata_qid')} score={r.get('match_score')}")

    # 10. Ton tai record co wiki_summary rat ngan (canh bao)
    short = [r for r in records if r.get("status") == "found" and len((r.get("wiki_summary") or "").strip()) < 40]
    if short:
        notes.append(f"{len(short)} record found co summary < 40 ky tu (canh bao)")
        for r in short[:10]:
            notes.append(f"    {r.get('name')}: {(r.get('wiki_summary') or '')[:80]!r}")

    print("=" * 70)
    for n in notes:
        print("  ", n)
    print("=" * 70)
    if problems:
        print(f"FAIL ({len(problems)} van de)")
        for p in problems:
            print("  [X]", p)
        return 1
    print("PASS: khong co van de")
    return 0


if __name__ == "__main__":
    sys.exit(main())
