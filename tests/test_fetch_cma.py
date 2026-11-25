"""Test T1.1: fetch_cma phan trang, cache trang, retry, gioi han --limit."""

from __future__ import annotations

import json

import pytest
import requests

from artrag.config import load_config, paths
from artrag.data import fetch_cma


def _page(ids: list[int], total: int = 3964) -> dict:
    return {
        "info": {"total": total},
        "data": [{"id": i, "title": f"t{i}"} for i in ids],
    }


class FakeResponse:
    def __init__(self, payload: dict | None = None, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)

    def json(self) -> dict:
        return self._payload or {}


class FakeSession:
    """Session gia: ghi lai cac lan goi, phai len trang theo ``skip``."""

    def __init__(self, pages: dict[int, dict], fail_first: int = 0):
        self.pages = pages
        self.calls: list[dict] = []
        self.fail_first = fail_first
        self._failures_left = fail_first

    def get(self, url, params=None, timeout=None, headers=None):
        self.calls.append({"url": url, "params": dict(params or {}), "timeout": timeout})
        skip = int((params or {}).get("skip", 0))
        if self._failures_left > 0:
            self._failures_left -= 1
            raise requests.ConnectionError("boom")
        if skip not in self.pages:
            return FakeResponse(_page([], total=0), status_code=200)
        return FakeResponse(self.pages[skip])

    def close(self) -> None:
        pass


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTRAG_DATA_DIR", str(tmp_path / "data"))
    return load_config(overrides=["data.cma_backoff_s=0"])


# ---------------------------------------------------------------- helpers
def test_page_path_is_zero_padded():
    assert fetch_cma.page_path("raw", 0).name == "page_00000.json"
    assert fetch_cma.page_path("raw", 2000).name == "page_02000.json"
    assert fetch_cma.page_path("raw", 0).parent.name == "cma"


def test_build_params_filters_cc0_and_images():
    params = fetch_cma.build_params("Painting", skip=1000, limit=500)
    assert params == {
        "type": "Painting",
        "limit": 500,
        "skip": 1000,
        "cc0": 1,
        "has_image": 1,
    }


def test_cache_valid_requires_matching_skip_and_enough_limit():
    payload = {"data": [], "_meta": {"skip": 0, "limit": 1000, "count": 0}}
    assert fetch_cma.is_cache_valid(payload, 0, 1000)
    assert fetch_cma.is_cache_valid(payload, 0, 50)  # trang day du van du cho 50
    assert not fetch_cma.is_cache_valid(payload, 1000, 1000)  # sai skip
    assert not fetch_cma.is_cache_valid(payload, 0, 2000)  # trang cua toi hon yeu cau
    assert not fetch_cma.is_cache_valid({"data": []}, 0, 10)  # thieu _meta
    assert not fetch_cma.is_cache_valid(None, 0, 10)


def test_with_meta_records_count_and_total():
    payload = fetch_cma.with_meta({"data": [{"id": 1}]}, skip=0, limit=50, total=3964)
    meta = payload["_meta"]
    assert meta["skip"] == 0
    assert meta["limit"] == 50
    assert meta["count"] == 1
    assert meta["total"] == 3964
    assert "fetched_at" in meta


# ---------------------------------------------------------------- fetching
def test_fetch_all_writes_pages_and_records(cfg, monkeypatch):
    pages = {0: _page(list(range(1000)))}
    session = FakeSession(pages)
    monkeypatch.setattr(fetch_cma.requests, "Session", lambda: session)

    stats = fetch_cma.fetch_all(cfg, limit=50)

    assert stats["records"] == 50
    assert stats["fetched_pages"] == 1
    out = paths(cfg).raw / "cma" / "page_00000.json"
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert len(payload["data"]) == 50
    assert payload["_meta"]["skip"] == 0
    assert payload["_meta"]["limit"] == 50
    assert payload["info"]["total"] == 3964


def test_second_run_reuses_cache_without_network(cfg, monkeypatch):
    pages = {0: _page(list(range(1000)))}
    session = FakeSession(pages)
    monkeypatch.setattr(fetch_cma.requests, "Session", lambda: session)

    fetch_cma.fetch_all(cfg, limit=50)
    calls_after_first = len(session.calls)

    stats = fetch_cma.fetch_all(cfg, limit=50)

    assert stats["cached_pages"] == 1
    assert stats["fetched_pages"] == 0
    assert stats["records"] == 50
    assert len(session.calls) == calls_after_first  # khong goi mang them


def test_force_refetches_even_when_cached(cfg, monkeypatch):
    session = FakeSession({0: _page(list(range(1000)))})
    monkeypatch.setattr(fetch_cma.requests, "Session", lambda: session)

    fetch_cma.fetch_all(cfg, limit=50)
    stats = fetch_cma.fetch_all(cfg, limit=50, force=True)

    assert stats["fetched_pages"] == 1
    assert stats["cached_pages"] == 0


def test_limit_stops_after_enough_records(cfg, monkeypatch):
    pages = {0: _page(list(range(1000))), 1000: _page(list(range(1000, 2000)))}
    session = FakeSession(pages)
    monkeypatch.setattr(fetch_cma.requests, "Session", lambda: session)

    stats = fetch_cma.fetch_all(cfg, limit=1500)

    assert stats["records"] == 1500
    assert stats["fetched_pages"] == 2
    assert len(session.calls) == 2
    assert session.calls[1]["params"]["skip"] == 1000


def test_full_run_paginates_until_total(cfg, monkeypatch):
    pages = {
        0: _page(list(range(1000))),
        1000: _page(list(range(1000, 2000))),
        2000: _page(list(range(2000, 3000))),
        3000: _page(list(range(3000, 3964))),
    }
    session = FakeSession(pages)
    monkeypatch.setattr(fetch_cma.requests, "Session", lambda: session)

    stats = fetch_cma.fetch_all(cfg)

    assert stats["records"] == 3964
    assert stats["total"] == 3964
    assert stats["fetched_pages"] == 4
    assert [c["params"]["skip"] for c in session.calls] == [0, 1000, 2000, 3000]


def test_last_partial_page_is_cached_not_refetched(cfg, monkeypatch):
    """Trang cuoi it hon page_size (964 < 1000) phai duoc xem la cache hop le."""
    pages = {
        0: _page(list(range(1000))),
        1000: _page(list(range(1000, 2000))),
        2000: _page(list(range(2000, 3000))),
        3000: _page(list(range(3000, 3964))),
    }
    session = FakeSession(pages)
    monkeypatch.setattr(fetch_cma.requests, "Session", lambda: session)

    first = fetch_cma.fetch_all(cfg)
    assert first["records"] == 3964
    assert first["fetched_pages"] == 4
    calls_after_first = len(session.calls)

    second = fetch_cma.fetch_all(cfg)
    assert second["fetched_pages"] == 0
    assert second["cached_pages"] == 4
    assert second["records"] == 3964
    assert len(session.calls) == calls_after_first  # khong goi mang them


def test_no_extra_page_created_beyond_total(cfg, monkeypatch):
    pages = {0: _page(list(range(1000))), 1000: _page(list(range(1000, 3964)))}
    session = FakeSession(pages)
    monkeypatch.setattr(fetch_cma.requests, "Session", lambda: session)

    stats = fetch_cma.fetch_all(cfg)

    assert stats["records"] == 3964
    assert [c["params"]["skip"] for c in session.calls] == [0, 1000]
    assert not (paths(cfg).raw / "cma" / "page_02000.json").exists()


def test_retry_on_network_error_then_success(cfg, monkeypatch):
    session = FakeSession({0: _page(list(range(1000)))}, fail_first=2)
    monkeypatch.setattr(fetch_cma.requests, "Session", lambda: session)

    stats = fetch_cma.fetch_all(cfg, limit=10)

    assert stats["records"] == 10
    assert len(session.calls) == 3  # 2 lan fail + 1 lan thanh cong


def test_retry_gives_up_after_max_retries(cfg, monkeypatch):
    session = FakeSession({0: _page(list(range(1000)))}, fail_first=99)
    monkeypatch.setattr(fetch_cma.requests, "Session", lambda: session)

    with pytest.raises(RuntimeError):
        fetch_cma.fetch_all(cfg, limit=10)


def test_fetch_page_retries_on_http_500():
    calls = {"n": 0}

    class FlakySession:
        def get(self, url, params=None, timeout=None, headers=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return FakeResponse({}, status_code=500)
            return FakeResponse(_page([1]))

    session = FlakySession()
    payload = fetch_cma.fetch_page(
        session, "http://x", {"skip": 0}, backoff_s=0, sleep=lambda _: None
    )
    assert calls["n"] == 2
    assert payload["data"][0]["id"] == 1


def test_cli_prints_records_and_cached(cfg, monkeypatch, capsys):
    session = FakeSession({0: _page(list(range(1000)))})
    monkeypatch.setattr(fetch_cma.requests, "Session", lambda: session)

    assert fetch_cma.main(["--limit", "50"]) == 0
    out = capsys.readouterr().out
    assert "50 records" in out

    assert fetch_cma.main(["--limit", "50"]) == 0
    out = capsys.readouterr().out
    assert "cached" in out
