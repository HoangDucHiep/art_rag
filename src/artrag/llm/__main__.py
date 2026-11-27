"""CLI cho backend LLM (T2.3).

Chạy: `uv run python -m artrag.llm --backend groq --prompt "Xin chào"`
"""

from __future__ import annotations

import argparse
import json
import sys

from artrag.config import load_config
from artrag.errors import ArtragError
from artrag.llm import get_backend


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m artrag.llm",
        description="Gọi backend LLM (groq / nim / hf_local) để kiểm tra nhanh.",
    )
    parser.add_argument(
        "--backend",
        default=None,
        help="tên backend trong khối `backends:` (groq, nim) hoặc hf_local/flan_t5.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="ghi đè tên model (mặc định lấy trong config).",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="nội dung người dùng.",
    )
    parser.add_argument(
        "--system",
        default=None,
        help="nội dung system prompt (tuỳ chọn).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="số token tối đa được sinh (mặc định lấy trong config `llm.gen`).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="nhiệt độ sinh (mặc định lấy trong config `llm.gen`).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="bỏ qua cache đĩa (luôn gọi API thật).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="in kết quả ở dạng JSON.",
    )
    parser.add_argument("--profile", default=None, help="profile config (mặc định: local).")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="a.b=c",
        help="ghi đè config, ví dụ --set llm.max_retries=3",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(profile=args.profile, overrides=args.overrides)

    messages: list[dict] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})

    gen: dict[str, object] = {}
    if args.max_new_tokens is not None:
        gen["max_new_tokens"] = int(args.max_new_tokens)
    if args.temperature is not None:
        gen["temperature"] = float(args.temperature)

    try:
        backend = get_backend(cfg, backend=args.backend, use_cache=not args.no_cache)
    except ArtragError as exc:
        print(f"LỖI: {exc}", file=sys.stderr)
        return 1

    if args.model:
        for layer in _layers(backend):
            if hasattr(layer, "model"):
                layer.model = args.model

    try:
        result = backend.generate(messages, **gen)
    except ArtragError as exc:
        print(f"LỖI: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - lỗi lạ không làm sập CLI
        print(f"LỖI không lường trước: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0

    print(
        f"{result.backend} | model={result.model} | latency_ms={result.latency_ms} "
        f"| cached={result.cached} | prompt_tokens={result.prompt_tokens} "
        f"| completion_tokens={result.completion_tokens}"
    )
    print("---")
    print(result.text)
    return 0


def _layers(backend: object) -> list[object]:
    """Chuỗi các lớp bọc của backend (để đổi `model`)."""
    layers: list[object] = []
    current: object | None = backend
    while current is not None and len(layers) < 20:
        layers.append(current)
        nxt = getattr(current, "inner", None)
        if nxt is None:
            for sub in getattr(current, "backends", []) or []:
                layers.extend(_layers(sub))
            break
        current = nxt
    return layers


if __name__ == "__main__":
    raise SystemExit(main())
