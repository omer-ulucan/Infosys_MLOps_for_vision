import pytest
from PIL import Image

import utility.constants as const
from utility.image_processor import crop_object_from_image, yolo_to_xywh


def test_yolo_to_xywh_converts_to_top_left_pixels():
    x, y, width, height = yolo_to_xywh("0 0.5 0.5 0.2 0.4", 200, 100)
    assert (x, y, width, height) == pytest.approx((80.0, 30.0, 40.0, 40.0))


def test_yolo_to_xywh_box_at_origin():
    assert yolo_to_xywh("3 0.1 0.1 0.2 0.2", 50, 50) == pytest.approx((0.0, 0.0, 10.0, 10.0))


def test_crop_object_from_image_returns_crop_of_box_size(tmp_path):
    img = Image.new("RGB", (200, 100))
    cropped = crop_object_from_image(str(tmp_path / "x.jpg"), img, (10, 20, 50, 40),
                                     str(tmp_path), "class_a", False, False)
    assert cropped.size == (50, 40)


def test_crop_object_from_image_saves_crop_when_requested(tmp_path):
    img = Image.new("RGB", (200, 100))
    crop_object_from_image(str(tmp_path / "x.jpg"), img, (10, 20, 50, 40),
                           str(tmp_path), "class_a", True, False)
    saved = tmp_path / const.CROPPED_FOLDER_PREFIX / "class_a" / f"{const.CROPPED_IMAGE_PREFIX}_x.jpg"
    assert saved.exists()
