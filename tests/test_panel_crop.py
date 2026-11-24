"""
Unit tests for per-panel cropping and per-panel prediction.
"""

from PIL import Image

from src.inference.panels import crop_panel, predict_panels

T1 = "| x | y |\n|---|---|\n| 1 | 10 |"
T2 = "| e | i |\n|---|---|\n| 1.5 | 400 |"


def _image():
    return Image.new("RGB", (200, 100))


def test_crop_uses_the_bbox():
    crop = crop_panel(_image(), {"x": 10, "y": 20, "width": 50, "height": 30})
    assert crop.size == (50, 30)


def test_crop_is_clamped_to_the_image():
    crop = crop_panel(_image(), {"x": 150, "y": 80, "width": 100, "height": 100})
    assert crop.size == (50, 20)


def test_margin_grows_the_crop_inside_the_image():
    bbox = {"x": 50, "y": 20, "width": 100, "height": 50}
    assert crop_panel(_image(), bbox, margin=0.1).size == (120, 60)
    assert crop_panel(_image(), {"x": 0, "y": 0, "width": 200, "height": 100}, margin=0.1).size == (200, 100)


def test_predict_panels_keeps_first_table_per_panel():
    seen = []

    def fake_predict(crop):
        seen.append(crop.size)
        return f"{T1}\n\n{T2}" if crop.size == (100, 100) else "no table here"

    bboxes = [
        {"panel_id": "a", "x": 0, "y": 0, "width": 100, "height": 100},
        {"panel_id": "b", "x": 100, "y": 0, "width": 100, "height": 100},
        {"panel_id": "c", "x": 0, "y": 0, "width": 60, "height": 40},
    ]
    out = predict_panels(fake_predict, _image(), bboxes)

    assert seen == [(100, 100), (100, 100), (60, 40)]
    assert out["a"] == {"table": T1, "raw_output": f"{T1}\n\n{T2}", "extra_tables": 1}
    assert out["c"] == {"table": None, "raw_output": "no table here", "extra_tables": 0}
