import json

import pytest

from taskq.config import DEFAULTS, ConfigError, load_config


def test_defaults_when_nothing_given():
    assert load_config() == DEFAULTS
    assert load_config() is not DEFAULTS


def test_override_single_setting_section():
    config = load_config({"storage": {"path": "/var/tmp/tasks.json"}})
    assert config["storage"]["path"] == "/var/tmp/tasks.json"
    assert config["retry"] == DEFAULTS["retry"]


def test_full_section_override():
    retry = {"max_attempts": 2, "base_delay": 0.1, "factor": 3.0, "max_delay": 5.0}
    assert load_config({"retry": retry})["retry"] == retry


def test_toml_file(tmp_path):
    path = tmp_path / "taskq.toml"
    path.write_text('[queue]\nmax_size = 50\n', encoding="utf-8")
    assert load_config(path)["queue"]["max_size"] == 50


def test_json_file(tmp_path):
    path = tmp_path / "taskq.json"
    path.write_text(json.dumps({"queue": {"max_size": 7}}), encoding="utf-8")
    assert load_config(str(path))["queue"]["max_size"] == 7


def test_unknown_section():
    with pytest.raises(ConfigError, match="unknown"):
        load_config({"retries": {}})


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope.toml")


def test_section_must_be_table():
    with pytest.raises(ConfigError):
        load_config({"queue": 5})
