"""Client OpenAI-compat dùng chung cho Groq và NVIDIA NIM (T2.3, SYSTEM_DESIGN 6.3).

Một client duy nhất phục vụ cả hai provider: chỉ khác `base_url`, `model` và
tên biến môi trường chứa API key. Key được xoay vòng khi provider trả 429/401
(người dùng có thể khai báo nhiều key cho cùng provider).
"""

from __future__ import annotations

import os
import time
from typing import Any, Sequence

from artrag.errors import BackendError, RetryableBackendError
from artrag.llm.base import LLMBackend, LLMResult, estimate_tokens

#: Các mã lỗi HTTP được coi là tạm thời (nên retry).
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

#: Độ dài tối đa của body lỗi (tránh in cả lỗi rất lớn ra console).
_MAX_ERROR_BODY = 500


def _parse_retry_after(headers: Any) -> float | None:
    """Đọc header `Retry-After` (số giây) nếu có."""
    if not headers:
        return None
    try:
        raw = headers.get("Retry-After")
    except AttributeError:
        return None
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


class OpenAICompatBackend:
    """Backend gọi API kiểu OpenAI (`/chat/completions`)."""

    def __init__(
        self,
        name: str,
        base_url: str,
        model: str,
        api_keys: Sequence[str] | None = None,
        rpm_limit: int | float | None = None,
        timeout_s: float = 60.0,
        session: Any | None = None,
        extra_headers: dict[str, str] | None = None,
        default_gen: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.base_url = str(base_url).rstrip("/")
        self.model = model
        self.api_keys: list[str] = [k for k in (api_keys or []) if k]
        self.rpm_limit = rpm_limit
        self.timeout_s = float(timeout_s)
        self.extra_headers = dict(extra_headers or {})
        self.default_gen = dict(default_gen or {})
        self._session = session
        self._key_index = 0
        #: danh sách key đã thử và lý do lỗi (để báo cáo)
        self.key_failures: list[str] = []

    # -- session ---------------------------------------------------------
    @property
    def session(self) -> Any:
        if self._session is None:
            import requests  # noqa: PLC0415

            self._session = requests.Session()
        return self._session

    # -- key -------------------------------------------------------------
    @property
    def current_key(self) -> str | None:
        if not self.api_keys:
            return None
        return self.api_keys[self._key_index % len(self.api_keys)]

    def rotate_key(self) -> str | None:
        """Chuyển sang key tiếp theo (vòng tròn)."""
        if not self.api_keys:
            return None
        self._key_index = (self._key_index + 1) % len(self.api_keys)
        return self.current_key

    # -- interface LLMBackend -------------------------------------------
    def count_tokens(self, text: str) -> int:
        return estimate_tokens(text)

    def generate(self, messages: list[dict], **gen: Any) -> LLMResult:
        if not self.api_keys:
            raise BackendError(
                f"Backend {self.name!r} chưa có API key.",
                hint="đặt biến môi trường API key trong .env (xem .env.example).",
            )

        params = dict(self.default_gen)
        params.update({k: v for k, v in gen.items() if v is not None})
        params.setdefault("max_tokens", 256)
        if "max_new_tokens" in params:
            params["max_tokens"] = params.pop("max_new_tokens")

        body: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages or []),
            **params,
        }

        url = f"{self.base_url}/chat/completions"
        started = time.perf_counter()
        tried_keys = 0
        last_error: Exception | None = None

        while tried_keys < max(1, len(self.api_keys)):
            tried_keys += 1
            key = self.current_key
            try:
                payload = self._post(url, key, body)
            except RetryableBackendError as exc:
                last_error = exc
                # 429/401 -> thử key tiếp theo trước khi ném lỗi
                if self._should_rotate(exc) and len(self.api_keys) > 1:
                    self.key_failures.append(f"{self.name}: {exc.message}")
                    self.rotate_key()
                    continue
                raise
            except BackendError as exc:
                last_error = exc
                # 401 (key sai/hết hạn) -> thử key tiếp theo
                if self._should_rotate(exc) and len(self.api_keys) > 1:
                    self.key_failures.append(f"{self.name}: {exc.message}")
                    self.rotate_key()
                    continue
                raise
            latency_ms = int((time.perf_counter() - started) * 1000)
            return self._to_result(payload, latency_ms)

        raise BackendError(
            f"Backend {self.name!r} thất bại: {last_error}",
            hint="kiểm tra hạn mức rate limit của provider hoặc đổi API key.",
        ) from last_error

    # -- nội bộ ----------------------------------------------------------
    def _should_rotate(self, exc: Exception) -> bool:
        """Có xoay sang key khác không (401 hoặc 429)?"""
        status = getattr(exc, "status_code", None)
        if status is None:
            return False
        return status in (401, 429)

    def _post(self, url: str, key: str | None, body: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }
        try:
            response = self.session.post(
                url, json=body, headers=headers, timeout=self.timeout_s
            )
        except Exception as exc:  # noqa: BLE001 - lỗi mạng -> tạm thời
            raise RetryableBackendError(
                f"Không gọi được {self.name} ({url}): {exc}",
                hint="kiểm tra kết nối mạng.",
            ) from exc

        status = int(getattr(response, "status_code", 0))
        if status >= 400:
            detail = self._error_detail(response)
            message = f"{self.name} trả HTTP {status}: {detail}"
            if status in RETRYABLE_STATUS:
                raise RetryableBackendError(
                    message,
                    hint="provider đang bận hoặc hết hạn mức, sẽ thử lại.",
                    retry_after=_parse_retry_after(getattr(response, "headers", None)),
                    status_code=status,
                )
            raise BackendError(
                message,
                hint="kiểm tra lại payload và tên model.",
                status_code=status,
            )

        try:
            payload = response.json()
        except Exception as exc:  # noqa: BLE001
            raise RetryableBackendError(
                f"{self.name} trả về body không phải JSON: {exc}",
                hint="thử lại; nếu vẫn lỗi thì kiểm tra base_url.",
            ) from exc
        if not isinstance(payload, dict):
            raise RetryableBackendError(
                f"{self.name} trả về JSON không phải object.",
                hint="kiểm tra base_url trong config.",
            )
        return payload

    @staticmethod
    def _error_detail(response: Any) -> str:
        try:
            payload = response.json()
        except Exception:  # noqa: BLE001
            text = getattr(response, "text", "") or ""
            return text[:_MAX_ERROR_BODY]
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                return str(error.get("message") or error)[:_MAX_ERROR_BODY]
            return str(error or payload)[:_MAX_ERROR_BODY]
        return str(payload)[:_MAX_ERROR_BODY]

    def _to_result(self, payload: dict[str, Any], latency_ms: int) -> LLMResult:
        choices = payload.get("choices") or []
        text = ""
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message") or {}
            text = str(message.get("content") or "")
        usage = payload.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        model = str(payload.get("model") or self.model)
        return LLMResult(
            text=text,
            model=model,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached=False,
            backend=self.name,
            raw={
                "finish_reason": (
                    choices[0].get("finish_reason") if choices else None
                ),
                "id": payload.get("id"),
            },
        )


# ---------------------------------------------------------------------------
# Dùng từ config
# ---------------------------------------------------------------------------


def resolve_api_keys(node: Any, env: dict[str, str] | None = None) -> list[str]:
    """Lấy danh sách API key từ config (`api_key_envs` hoặc `api_key_env`).

    Key rỗng (biến môi trường không tồn tại) bị bỏ qua.
    """
    environ = os.environ if env is None else env
    if node is None:
        return []

    def _get(name: Any) -> str | None:
        if not isinstance(name, str) or not name:
            return None
        value = environ.get(name)
        return value.strip() if value else None

    names: list[str] = []
    envs = node.get("api_key_envs") if hasattr(node, "get") else None
    if isinstance(envs, (list, tuple)):
        names.extend(str(n) for n in envs if n)
    single = node.get("api_key_env") if hasattr(node, "get") else None
    if isinstance(single, str) and single:
        names.append(single)
    elif isinstance(single, (list, tuple)):
        names.extend(str(n) for n in single if n)

    keys: list[str] = []
    for name in names:
        value = _get(name)
        if value and value not in keys:
            keys.append(value)
    return keys


def build_openai_compat(
    name: str,
    node: Any,
    timeout_s: float = 60.0,
    default_gen: dict[str, Any] | None = None,
    session: Any | None = None,
) -> OpenAICompatBackend:
    """Dựng backend openai-compat từ một nút config trong `backends:`."""
    if node is None:
        raise BackendError(
            f"Không có cấu hình cho backend {name!r}.",
            hint="thêm mục tương ứng trong khối `backends:` của configs/default.yaml.",
        )

    base_url = str(node.get("base_url") or "").rstrip("/")
    if not base_url:
        raise BackendError(
            f"Backend {name!r} thiếu `base_url`.",
            hint="kiểm tra khối `backends:` trong configs/default.yaml.",
        )
    model = str(node.get("model") or "")
    if not model:
        raise BackendError(
            f"Backend {name!r} thiếu `model`.",
            hint="kiểm tra khối `backends:` trong configs/default.yaml.",
        )

    keys = resolve_api_keys(node)
    if not keys:
        envs = node.get("api_key_envs") or node.get("api_key_env") or "?"
        raise BackendError(
            f"Backend {name!r} chưa có API key nào trong biến môi trường ({envs}).",
            hint="tạo file .env từ .env.example và điền key, hoặc dùng profile khác.",
        )

    return OpenAICompatBackend(
        name=name,
        base_url=base_url,
        model=model,
        api_keys=keys,
        rpm_limit=node.get("rpm_limit"),
        timeout_s=timeout_s,
        session=session,
        default_gen=default_gen,
    )


def known_backend_names(cfg: Any) -> list[str]:
    """Danh sách tên backend khai báo trong khối `backends:` của config."""
    node = cfg.get("backends") if hasattr(cfg, "get") else None
    raw = getattr(node, "raw", node)
    if not isinstance(raw, dict):
        return []
    return sorted(raw.keys())


__all__ = [
    "OpenAICompatBackend",
    "RETRYABLE_STATUS",
    "build_openai_compat",
    "known_backend_names",
    "resolve_api_keys",
]
