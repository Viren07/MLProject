import numpy as np
from PIL import Image

from src.grid import apply_condition, conditions_from_config, run_condition, slug


def img():
    return Image.fromarray(np.random.default_rng(1).integers(0, 255, (96, 128, 3), dtype=np.uint8))


def test_conditions_and_slug():
    cfg = {"corruptions": [{"name": "gaussian_noise", "severities": [50, 100]}, {"name": "grayscale", "severities": [100]}]}
    assert conditions_from_config(cfg) == ["clean", "gaussian_noise:50", "gaussian_noise:100", "grayscale:100"]
    assert ":" not in slug("gaussian_noise:50")  # ':' is not allowed in Windows filenames


def test_clean_is_untouched_and_corruptions_change_the_image():
    im = img()
    assert (np.array(apply_condition(im, "clean", "x")) == np.array(im)).all()
    for cond in ("gaussian_noise:50", "pixelate:100", "defocus_blur:75", "blur_gray:100", "grayscale:100", "fisheye:100"):
        assert not (np.array(apply_condition(im, cond, "x")) == np.array(im)).all(), cond


class FakeScorer:
    def score(self, image, options):
        return {"probs": [1.0, 0, 0, 0, 0], "letter_mass": 1.0}


def test_run_condition_end_to_end(tmp_path):
    items = []
    for i in range(3):
        name = f"p{i}.jpg"
        img().save(tmp_path / name)
        items.append({"id": f"i{i}", "image": name, "options": list("abcde"), "true_class": "alpha"})
    rows = run_condition(items, tmp_path, FakeScorer(), "gaussian_noise:50", workers=2)
    assert [r["id"] for r in rows] == ["i0", "i1", "i2"]
