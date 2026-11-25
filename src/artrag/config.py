"""Config loader: gộp default <-> profile <-> experiment <-> overrides (SYSTEM_DESIGN 6.1)."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from artrag.errors import ConfigError

ENV_PATTERN = re.compile(r"\$\{env:([A-Za-z_][A-Za-z0-9_]*)(?:,([^}]*))?\}")
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
PROFILES_DIR = CONFIGS_DIR / "profiles"
EXPERIMENTS_DIR = CONFIGS_DIR / "experiments"


class Config:
    """Wrapper quanh dict config, truy cập kiểu thuộc tính hoặc khoá 'a.b.c'."""

    __slots__ = ("_data",)

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self._data = data if data is not None else {}

    # -- truy cập -------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            value = self._data[name]
        except KeyError as exc:
            raise AttributeError(f"config has no key {name!r}") from exc
        return Config(value) if isinstance(value, dict) else value

    def __getitem__(self, key: str) -> Any:
        if key in self._data:
            value = self._data[key]
            return Config(value) if isinstance(value, dict) else value
        node: Any = self._data
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                raise KeyError(key)
            node = node[part]
        return Config(node) if isinstance(node, dict) else node

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        try:
            self[key]
        except KeyError:
            return False
        return True

    def __iter__(self):
        return iter(self._data)

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._data, ensure_ascii=False))

    @property
    def raw(self) -> dict[str, Any]:
        return self._data

    def __repr__(self) -> str:
        return f"Config({self._data!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Config):
            return self._data == other._data
        return NotImplemented


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Không tìm thấy file config: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"File config phải là mapping: {path}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Gộp sâu; list trong override thay thế hoàn toàn list của base."""
    out = dict(base)
    for key, value in override.items():
        current = out.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            out[key] = deep_merge(current, value)
        else:
            out[key] = value
    return out


def _resolve_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    if not isinstance(value, str):
        return value

    def _sub(match: re.Match[str]) -> str:
        var, default = match.group(1), match.group(2)
        env_value = os.environ.get(var)
        if env_value is not None:
            return env_value
        if default is not None:
            return default
        return match.group(0)

    return ENV_PATTERN.sub(_sub, value)


def _parse_value(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def _set_dotted(data: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = data
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value


def parse_overrides(overrides: list[str] | None) -> dict[str, Any]:
    """Parse danh sách 'a.b.c=value' thành dict lồng."""
    result: dict[str, Any] = {}
    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"Override không hợp lệ (thiếu '='): {item!r}")
        key, _, raw_value = item.partition("=")
        key = key.strip()
        if not key:
            raise ConfigError(f"Override không hợp lệ (thiếu khoá): {item!r}")
        _set_dotted(result, key, _parse_value(raw_value.strip()))
    return result


def _profile_name(profile: str | None) -> str:
    return profile or os.environ.get("ARTRAG_PROFILE") or "local"


def load_config(
    profile: str | None = None,
    experiment: str | None = None,
    overrides: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> Config:
    """Nạp config theo thứ tự §6.1.

    default.yaml < profiles/{profile}.yaml < experiments/{exp}.yaml < --set < extra
    """
    profile_name = _profile_name(profile)
    merged = _read_yaml(CONFIGS_DIR / "default.yaml")

    profile_path = PROFILES_DIR / f"{profile_name}.yaml"
    if profile_path.exists():
        merged = deep_merge(merged, _read_yaml(profile_path))
    elif profile_name != "local":
        raise ConfigError(f"Không tìm thấy profile {profile_name!r} tại {profile_path}")

    merged["profile"] = profile_name

    if experiment:
        exp_path = EXPERIMENTS_DIR / f"{experiment}.yaml"
        if not exp_path.exists():
            raise ConfigError(
                f"Không tìm thấy experiment {experiment!r} tại {exp_path}"
            )
        merged = deep_merge(merged, _read_yaml(exp_path))
    merged["experiment"] = experiment

    merged = deep_merge(merged, parse_overrides(overrides))
    if extra:
        merged = deep_merge(merged, extra)

    merged = _resolve_env(merged)
    return Config(merged)


def config_hash(cfg: Config | dict[str, Any]) -> str:
    """Hash ổn định của config đã gộp (không phụ thuộc thứ tự khoá)."""
    import hashlib

    data = cfg.raw if isinstance(cfg, Config) else cfg
    payload = json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Paths:
    """Các thư mục con của data_dir (§2.4)."""

    data_dir: Path
    results_dir: Path

    @property
    def raw(self) -> Path:
        return self.data_dir / "raw"

    @property
    def kb(self) -> Path:
        return self.data_dir / "kb"

    @property
    def images(self) -> Path:
        return self.data_dir / "images"

    @property
    def index(self) -> Path:
        return self.data_dir / "index"

    @property
    def cache(self) -> Path:
        return self.data_dir / "cache"

    def all_dirs(self) -> list[Path]:
        return [
            self.data_dir,
            self.raw,
            self.kb,
            self.images,
            self.index,
            self.cache,
            self.results_dir,
        ]

    def ensure(self) -> "Paths":
        for path in self.all_dirs():
            path.mkdir(parents=True, exist_ok=True)
        return self


def paths(cfg: Config, ensure: bool = False) -> Paths:
    """Trả về các thư mục con của data dir (không tạo mặc định)."""
    data_dir = Path(str(cfg.get("paths.data_dir", "./data"))).expanduser()
    results_dir = Path(str(cfg.get("paths.results_dir", "./results"))).expanduser()
    if not data_dir.is_absolute():
        data_dir = (REPO_ROOT / data_dir).resolve()
    if not results_dir.is_absolute():
        results_dir = (REPO_ROOT / results_dir).resolve()
    result = Paths(data_dir=data_dir, results_dir=results_dir)
    if ensure:
        result.ensure()
    return result
