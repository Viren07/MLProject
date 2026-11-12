from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from PIL import Image

import src.partners as partners_mod
from src.corruptions import STD, corrupt_image
from src.grid import render_item

LEVELS = (25, 50, 75, 100)


def rand_img(seed=1, size=(160, 160)):
    return Image.fromarray(np.random.default_rng(seed).integers(40, 220, (size[1], size[0], 3), dtype=np.uint8))


BASE, OTHER = rand_img(1), rand_img(2)
px = lambda im: np.array(im).astype(float)
c = lambda name, lv, key="k": corrupt_image(BASE, name, lv, key, partner=OTHER)


@pytest.mark.parametrize("name", [n for n in STD if n != "grayscale"])
def test_levels_run_are_deterministic_and_strictly_harsher(name):
    dev = []
    for lv in LEVELS:
        a, b = np.array(c(name, lv)), np.array(c(name, lv))
        assert a.shape == (160, 160, 3) and (a == b).all()
        dev.append(np.abs(a.astype(float) - px(BASE)).mean())
    assert all(x < y for x, y in zip(dev, dev[1:])), (name, dev)


def test_noise_differs_across_keys_and_matches_sigma():
    assert not (np.array(c("gaussian_noise", 50, "a")) == np.array(c("gaussian_noise", 50, "b"))).all()
    flat = Image.new("RGB", (200, 200), (128, 128, 128))
    ref = np.random.default_rng(0)
    for lv, sigma in zip(LEVELS, (0.20, 0.40, 0.60, 0.80)):
        out = np.array(corrupt_image(flat, "gaussian_noise", lv, "k")).astype(float) / 255
        expected = np.clip(128 / 255 + ref.normal(0, sigma, 200 * 200 * 3), 0, 1).std()  # clipping shrinks the spread
        assert abs(out.std() - expected) < 0.01, (lv, out.std(), expected)


def test_identical_under_threads():
    with ThreadPoolExecutor(8) as ex:
        outs = list(ex.map(lambda _: np.array(c("gaussian_noise", 75)), range(16)))
    assert all((o == outs[0]).all() for o in outs)


def test_pixelate_100_has_coarse_blocks():
    out = np.array(corrupt_image(rand_img(3, (160, 160)), "pixelate", 100, "k"))
    assert len(np.unique(out.reshape(-1, 3), axis=0)) <= 10 * 10 + 1  # 160 * 0.0625 = 10 blocks per side


def test_cutmix_area_matches_table_and_content_comes_from_partner():
    for lv, area in zip(LEVELS, (0.075, 0.15, 0.225, 0.30)):
        out = np.array(c("cutmix", lv))
        diff = (out != np.array(BASE)).any(axis=2)
        assert area * 0.9 < diff.mean() <= area * 1.05, (lv, diff.mean())
        assert (out[diff] == np.array(OTHER)[diff]).all()


def test_mixup_is_a_convex_blend():
    out = px(c("mixup", 100))
    assert np.abs(out - (0.6 * px(BASE) + 0.4 * px(OTHER))).max() <= 1.0


def test_fisheye_keeps_center_and_grows():
    flat = Image.fromarray(np.tile(np.linspace(0, 255, 121, dtype=np.uint8), (121, 1))[..., None].repeat(3, 2))
    f50, f100 = (px(corrupt_image(flat, "fisheye", s, "k")) for s in (50, 100))
    assert (f50[60, 60] == px(flat)[60, 60]).all()
    assert np.abs(f100 - px(flat)).mean() > np.abs(f50 - px(flat)).mean()


def test_gray_conditions_have_equal_channels_and_blur_gray_is_blur_then_gray():
    for cond in (("grayscale", 100), ("blur_gray", 50), ("blur_gray", 100)):
        a = np.array(c(*cond)); assert (a[..., 0] == a[..., 1]).all() and (a[..., 1] == a[..., 2]).all()
    assert not (np.array(c("blur_gray", 100)) == np.array(c("grayscale", 100))).all()


def test_unknown_names_and_levels_are_rejected():
    for name, lv in (("grayscale", 50), ("pixelate", 60), ("gaussian_noise", 3), ("defocus_blur", 5), ("vignette", 100)):
        with pytest.raises(ValueError):
            c(name, lv)


def test_partner_required_for_cutmix_and_mixup():
    for n in ("cutmix", "mixup"):
        with pytest.raises(ValueError):
            corrupt_image(BASE, n, 50, "k")


def test_partner_pool_excludes_used_images_and_same_class(tmp_path, monkeypatch):
    classes = ("alpha", "beta", "gamma")
    for cls in classes:
        (tmp_path / cls).mkdir()
        for i in range(4):
            rand_img(abs(hash((cls, i))) % 1000).save(tmp_path / cls / f"{i}.jpg")
    listing = {cls: sorted(f"{cls}/{p.name}" for p in (tmp_path / cls).glob("*.jpg")) for cls in classes}
    monkeypatch.setattr(partners_mod, "load_pool", lambda root: listing)
    used = {"alpha/0.jpg", "beta/0.jpg", "gamma/0.jpg", "alpha/1.jpg"}
    pool = partners_mod.PartnerPool(tmp_path, used)
    assert all(p not in used for v in pool.pool.values() for p in v)
    seen = {tuple(np.array(pool.get("alpha", f"k{i}")).flatten()[:6]) for i in range(20)}
    assert len(seen) > 1  # varies with the key
    assert (np.array(pool.get("alpha", "same")) == np.array(pool.get("alpha", "same"))).all()  # deterministic per key


def test_render_item_fetches_a_partner_only_when_needed():
    class Stub:
        def get(self, cls, key):
            assert cls == "alpha"
            return OTHER
    item = {"id": "i1", "true_class": "alpha"}
    assert not (np.array(render_item(BASE, "cutmix:50", item, 0, Stub())) == np.array(BASE)).all()
    render_item(BASE, "pixelate:50", item, 0, None)  # no partner needed
    with pytest.raises(ValueError):
        render_item(BASE, "cutmix:50", item, 0, None)
