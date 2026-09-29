import json

import pytest

from utility.json_ops import create_json, get_dict_from_json, yolo_to_pascal_voc


@pytest.mark.parametrize(
    "yolo_box, img_size, expected",
    [
        # Box centered in a 100x100 image
        ((0.5, 0.5, 0.2, 0.4), (100, 100), [40, 30, 60, 70]),
        # Non-square image
        ((0.25, 0.5, 0.1, 0.2), (200, 100), [40, 40, 60, 60]),
        # Full-image box
        ((0.5, 0.5, 1.0, 1.0), (640, 480), [0, 0, 640, 480]),
        # String inputs, as read from YOLO label files
        (("0.5", "0.5", "0.2", "0.4"), (100, 100), [40, 30, 60, 70]),
    ],
)
def test_yolo_to_pascal_voc(yolo_box, img_size, expected):
    assert yolo_to_pascal_voc(*yolo_box, *img_size) == expected


def test_yolo_to_pascal_voc_clips_negative_coordinates():
    # Box extends past the top-left corner
    assert yolo_to_pascal_voc(0.05, 0.05, 0.2, 0.2, 100, 100) == [0, 0, 15, 15]


def test_create_json_wraps_data_in_user_key(tmp_path):
    create_json(str(tmp_path), "image1", {"distance": "near"})
    with open(tmp_path / "image1.json") as f:
        assert json.load(f) == {"user": {"distance": "near"}}


def test_get_dict_from_json_reads_path_relative_to_cwd(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({"key": "value"}))
    monkeypatch.chdir(tmp_path)
    assert get_dict_from_json("config.json") == {"key": "value"}


def test_get_dict_from_json_returns_none_for_missing_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert get_dict_from_json("missing.json") is None
