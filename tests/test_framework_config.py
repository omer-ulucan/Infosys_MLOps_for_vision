import json
from pathlib import Path

import pytest
import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "framework_config"


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.json")), ids=lambda p: p.name)
def test_json_config_is_valid(path):
    with open(path) as f:
        json.load(f)


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_yaml_config_is_valid(path):
    with open(path) as f:
        yaml.safe_load(f)


def test_primary_config_has_required_keys():
    with open(CONFIG_DIR / "vision_sdk_primary_config.json") as f:
        config = json.load(f)
    for key in ("raw_data_source", "pipeline_type", "pipeline_platform", "custom_pipeline_yolo", "gt_report"):
        assert key in config


def test_primary_config_platform_config_exists():
    with open(CONFIG_DIR / "vision_sdk_primary_config.json") as f:
        platform_config = json.load(f).get("platform_config")
    if platform_config:
        assert (CONFIG_DIR.parent / platform_config).is_file()
