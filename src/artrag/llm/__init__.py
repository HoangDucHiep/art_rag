"""Interface LLM chung + factory `get_backend` (T2.3, SYSTEM_DESIGN 6.3).

Một pipeline duy nhất cho CLI, UI và eval: tất cả đều gọi qua `LLMBackend`.
Backend được bọc theo chuỗi: `CachedBackend` -> `RetryBackend` -> backend thật.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from artrag.config import Config, load_config, paths
from artrag.errors import BackendError
from artrag.llm.base import LLMBackend, LLMResult, count_message_tokens, estimate_tokens
from artrag.llm.cache import CachedBackend, cache_key
from artrag.llm.fallback import FallbackBackend
from artrag.llm.openai_compat import (
    OpenAICompatBackend,
    build_openai_compat,
    known_backend_names,
    resolve_api_keys,
)
from artrag.llm.retry import RateLimiter, RetryBackend

#: Các backend chưa được hỗ trợ ở T2.3 (sẽ bổ sung ở T2.4/T2.5).
PENDING_BACKENDS = ("hf_local", "flan_t5")


def load_env() -> None:
    """Nạp biến môi trường từ file `.env` (không ghi đè biến đã có)."""
    try:
        from dotenv import load_dotenv  # noqa: PLC0415
    except ImportError:  # pragma: no cover - python-dotenv là dependency
        return
    root = Path(__file__).resolve().parents[3]
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


def _as_dict(value: Any) -> dict[str, Any] | None:
    """Chuyển `Config`/dict về dict thuần; None nếu không phải mapping."""
    if isinstance(value, Config):
        return value.raw
    if isinstance(value, dict):
        return value
    return None


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Đọc một khoá của config (hỗ trợ khoá lồng `a.b`)."""
    if cfg is None:
        return default
    getter = getattr(cfg, "get", None)
    if getter is None:
        return default
    try:
        value = getter(key, default)
    except (KeyError, TypeError):
        return default
    return default if value is None else value


def _backend_node(cfg: Any, name: str) -> dict[str, Any] | None:
    """Node config của backend `name` trong khối `backends:`."""
    backends = _as_dict(_cfg_get(cfg, "backends", {}))
    if not backends:
        return None
    node = backends.get(name)
    if node is None:
        return None
    return _as_dict(node)


def _build_single(name: str, cfg: Any) -> LLMBackend:
    """Dựng một backend (chưa bọc cache/retry) theo tên."""
    node = _backend_node(cfg, name)
    backend_type = str(node.get("type") or "openai_compat") if node else None

    if backend_type == "openai_compat":
        if node is None:
            raise BackendError(
                f"Không có cấu hình cho backend {name!r}.",
                hint=(
                    "các backend có sẵn: "
                    + (", ".join(known_backend_names(cfg)) or "(chưa có)")
                    + " — khai báo trong khối `backends:` của configs/default.yaml."
                ),
            )
        timeout_s = float(_cfg_get(cfg, "llm.timeout_s", 60.0))
        default_gen = _as_dict(_cfg_get(cfg, "llm.gen", {})) or {}
        return build_openai_compat(
            name, node, timeout_s=timeout_s, default_gen=default_gen
        )

    if backend_type in PENDING_BACKENDS:
        raise BackendError(
            f"Backend {backend_type!r} chưa được hỗ trợ ở T2.3.",
            hint="hf_local sẽ có ở T2.4, flan_t5 ở T2.5.",
        )

    raise BackendError(
        f"Không biết loại backend {backend_type!r}.",
        hint="kiểm tra trường `type` trong khối `backends:` của config.",
    )


def _wrap_chain(backend: LLMBackend, cfg: Any, cache_dir: Path | None) -> LLMBackend:
    """Bọc backend bằng cache + retry (cache bên trong, retry bên ngoài)."""
    chain: LLMBackend = backend
    if cache_dir is not None:
        chain = CachedBackend(chain, cache_dir, enabled=True)
    max_retries = int(_cfg_get(cfg, "llm.max_retries", 5))
    backoff_s = float(_cfg_get(cfg, "llm.backoff_s", 1.0))
    limiter = None
    rpm = getattr(backend, "rpm_limit", None)
    if rpm:
        limiter = RateLimiter.from_rpm(rpm)
    chain = RetryBackend(
        chain, max_retries=max_retries, backoff_s=backoff_s, rate_limiter=limiter
    )
    return chain


def get_backend(
    cfg: Config | Any = None,
    backend: str | None = None,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
) -> LLMBackend:
    """Dựng backend LLM theo config.

    Thứ tự ưu tiên: tham số `backend` -> `cfg.llm.backend` -> `hf_local`.
    """
    load_env()
    if cfg is None:
        cfg = load_config()
    elif isinstance(cfg, dict):
        cfg = Config(cfg)

    name = backend or str(_cfg_get(cfg, "llm.backend", "hf_local"))
    primary = _build_single(name, cfg)

    if cache_dir is None:
        cache_name = str(_cfg_get(cfg, "llm.cache_dir", "llm"))
        cache_dir = paths(cfg).cache / cache_name
    resolved_cache = Path(cache_dir) if use_cache else None

    chains: list[LLMBackend] = [_wrap_chain(primary, cfg, resolved_cache)]

    fallback_name = _cfg_get(cfg, "llm.fallback", None)
    if fallback_name:
        fallback = _build_single(str(fallback_name), cfg)
        chains.append(_wrap_chain(fallback, cfg, resolved_cache))

    if len(chains) == 1:
        return chains[0]
    return FallbackBackend(chains)


__all__ = [
    "CachedBackend",
    "FallbackBackend",
    "LLMBackend",
    "LLMResult",
    "OpenAICompatBackend",
    "PENDING_BACKENDS",
    "RateLimiter",
    "RetryBackend",
    "build_openai_compat",
    "cache_key",
    "count_message_tokens",
    "estimate_tokens",
    "get_backend",
    "known_backend_names",
    "load_env",
    "resolve_api_keys",
]
