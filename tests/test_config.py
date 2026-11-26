import json

import pytest

from artrag.config import Config, config_hash, load_config, paths, parse_overrides
from artrag.errors import ConfigError


def test_default_config_has_core_sections():
    cfg = load_config()
    assert cfg["retrieval"]["top_n"] == 20
    assert cfg["abstain"]["tau_text"] == 0.45
    assert cfg["llm"]["backend"] == "hf_local"
    assert cfg["profile"] == "local"


def test_deep_merge_profile_overrides_default_and_keeps_others():
    cfg = load_config(profile="api")
    # profile ghi đè default
    assert cfg["llm"]["backend"] == "groq"
    assert cfg["query"]["translator"] == "llm"
    # key không bị ghi đè vẫn giữ nguyên từ default
    assert cfg["retrieval"]["top_n"] == 20
    assert cfg["abstain"]["enabled"] is True
    # nested dict không bị mất
    assert cfg["backends"]["groq"]["rpm_limit"] == 30


def test_override_dotted_key_and_type_coercion():
    cfg = load_config(overrides=["retrieval.top_n=12", "abstain.enabled=false", "llm.gen.max_new_tokens=64"])
    assert cfg["retrieval"]["top_n"] == 12
    assert cfg["abstain"]["enabled"] is False
    assert cfg["llm"]["gen"]["max_new_tokens"] == 64
    # phần còn lại không bị đè
    assert cfg["retrieval"]["n_clip"] == 50


def test_override_replaces_lists():
    cfg = load_config(overrides=["retrieval.branch_weights.GENERAL=[0.9,0.1]"])
    assert cfg["retrieval"]["branch_weights"]["GENERAL"] == [0.9, 0.1]


def test_parse_overrides_invalid():
    with pytest.raises(ConfigError):
        parse_overrides(["no_equals_sign"])
    with pytest.raises(ConfigError):
        parse_overrides(["=value"])


def test_env_interpolation_resolves(monkeypatch, tmp_path):
    monkeypatch.setenv("ARTRAG_DATA_DIR", str(tmp_path / "mydata"))
    cfg = load_config()
    assert cfg["paths"]["data_dir"] == str(tmp_path / "mydata")
    p = paths(cfg)
    assert p.data_dir == (tmp_path / "mydata")


def test_env_interpolation_uses_default(monkeypatch):
    monkeypatch.delenv("ARTRAG_PROFILE", raising=False)
    cfg = load_config()
    # default.yaml dùng ${env:ARTRAG_DATA_DIR,./data}; nếu biến không tồn tại -> "./data"
    assert cfg["paths"]["data_dir"] == "./data"


def test_env_interpolation_leaves_placeholder_when_missing(monkeypatch):
    monkeypatch.delenv("ARTRAG_TEST_MISSING", raising=False)
    cfg = load_config(extra={"some": {"key": "${env:ARTRAG_TEST_MISSING}"}})
    assert cfg["some"]["key"] == "${env:ARTRAG_TEST_MISSING}"


def test_config_hash_is_stable_and_sensitive():
    a = load_config()
    b = load_config()
    assert config_hash(a) == config_hash(b)
    c = load_config(overrides=["retrieval.top_n=99"])
    assert config_hash(c) != config_hash(a)
    # hash chỉ phụ thuộc nội dung, không phụ thuộc thứ tự khóa
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})


def test_config_hash_changes_with_env_value(monkeypatch):
    a = load_config()
    monkeypatch.setenv("ARTRAG_DATA_DIR", "./other")
    b = load_config()
    assert config_hash(a) != config_hash(b)


def test_missing_experiment_raises():
    with pytest.raises(ConfigError):
        load_config(experiment="khong_ton_tai")


def test_missing_profile_raises():
    with pytest.raises(ConfigError):
        load_config(profile="khong_ton_tai")


def test_config_attribute_and_item_access():
    cfg = load_config()
    assert isinstance(cfg, Config)
    assert cfg.retrieval.top_n == 20
    assert cfg["retrieval"]["top_n"] == 20
    assert cfg.get("khong.co.key", "mac_dinh") == "mac_dinh"
    with pytest.raises(AttributeError):
        _ = cfg.khong_ton_tai


def test_paths_are_relative_to_repo_root():
    cfg = load_config()
    p = paths(cfg)
    assert p.data_dir.name == "data"
    assert p.kb.parent == p.data_dir
    assert p.images.parent == p.data_dir
    assert p.index.parent == p.data_dir
    assert p.cache.parent == p.data_dir
    assert p.raw.parent == p.data_dir


def test_config_is_json_serializable():
    cfg = load_config()
    json.dumps(cfg.to_dict())
