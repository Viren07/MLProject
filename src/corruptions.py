"""Image corruptions on one standard severity scale.

Severity is a percent of each family's maximum usable strength: 25, 50, 75 or 100, written
as "family:severity", e.g. "pixelate:75". 100 is the harshest level judged still recognizable
as a room. Percentages are comparable within a family, not across families.

  gaussian_noise  sigma 0.20 / 0.40 / 0.60 / 0.80
  pixelate        resolution kept 50% / 25% / 12.5% / 6.25%
  defocus_blur    disk radius 3.5 / 7 / 10.5 / 14 px
  blur_gray       the same blur, then grayscale
  cutmix          share of the frame taken from another room 7.5% / 15% / 22.5% / 30%
  mixup           weight of the other room 0.10 / 0.20 / 0.30 / 0.40
  fisheye         barrel distortion k 0.5 / 1 / 2 / 4
  grayscale       100 only

Everything is deterministic: random draws come from a generator seeded by (family, severity,
key), where key is stable per image, so the same inputs always give identical pixels. Corrupted
images are therefore regenerated on demand and never stored (the dataset is research-use only).
cutmix and mixup need a second image, taken from images that are in NO split (see partners.py);
the label stays the base room.
"""
from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image

NEEDS_PARTNER = {"cutmix", "mixup"}
STD = {
    "gaussian_noise": {25: 0.20, 50: 0.40, 75: 0.60, 100: 0.80},      # sigma, on a 0-1 scale
    "pixelate": {25: 0.5, 50: 0.25, 75: 0.125, 100: 0.0625},          # fraction of resolution kept
    "defocus_blur": {25: 3.5, 50: 7.0, 75: 10.5, 100: 14.0},          # disk radius in pixels
    "blur_gray": {25: 3.5, 50: 7.0, 75: 10.5, 100: 14.0},
    "cutmix": {25: 0.075, 50: 0.15, 75: 0.225, 100: 0.30},            # share of the frame
    "mixup": {25: 0.10, 50: 0.20, 75: 0.30, 100: 0.40},               # weight of the other image
    "fisheye": {25: 0.5, 50: 1.0, 75: 2.0, 100: 4.0},
    "grayscale": {100: 1.0},
}


def seed_for(name: str, severity: int, key: str) -> int:
    return int(hashlib.sha1(f"{name}:{severity}:{key}".encode()).hexdigest()[:8], 16)


def _rng(name: str, severity: int, key: str) -> np.random.Generator:
    return np.random.default_rng(seed_for(name, severity, key))


def grayscale(img: Image.Image) -> Image.Image:
    return img.convert("L").convert("RGB")  # luminance, kept as 3 channels


def _disk_kernel(radius: float, alias: float = 0.5) -> np.ndarray:
    import cv2
    half = max(8, int(np.ceil(radius)))
    ax = np.arange(-half, half + 1)
    X, Y = np.meshgrid(ax, ax)
    k = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=np.float32)
    k /= k.sum()
    return cv2.GaussianBlur(k, ksize=(3, 3) if half <= 8 else (5, 5), sigmaX=alias)


def _blur(img: Image.Image, radius: float) -> Image.Image:
    import cv2
    kernel = _disk_kernel(radius)
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    out = np.stack([cv2.filter2D(arr[:, :, d], -1, kernel) for d in range(3)], axis=-1)
    return Image.fromarray(np.uint8(np.clip(out, 0, 1) * 255))


def _box(rng: np.random.Generator, w: int, h: int, area: float) -> tuple[int, int, int, int]:
    """Random rectangle with the image's aspect ratio covering `area` of the frame."""
    bw, bh = max(1, round(w * area ** 0.5)), max(1, round(h * area ** 0.5))
    x0, y0 = int(rng.integers(0, w - bw + 1)), int(rng.integers(0, h - bh + 1))
    return x0, y0, x0 + bw, y0 + bh


def mix(img: Image.Image, partner: Image.Image, lam: float) -> Image.Image:
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    b = np.asarray(partner.convert("RGB").resize(img.size), dtype=np.float32)
    return Image.fromarray(np.uint8(np.clip((1 - lam) * a + lam * b + 0.5, 0, 255)))


def fisheye_k(img: Image.Image, k: float) -> Image.Image:
    """Barrel distortion: center magnified, periphery squeezed, black corners."""
    from scipy.ndimage import map_coordinates
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    cy, cx = (h - 1) / 2, (w - 1) / 2
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ny, nx = (yy - cy) / max(cy, 1), (xx - cx) / max(cx, 1)  # edge midpoints at radius 1
    scale = (1 + k * (nx ** 2 + ny ** 2)) / (1 + k)  # source radius / output radius
    sy, sx = ny * scale * max(cy, 1) + cy, nx * scale * max(cx, 1) + cx
    out = np.stack([map_coordinates(arr[..., c], [sy, sx], order=1, mode="constant", cval=0.0)
                    for c in range(3)], axis=-1)
    return Image.fromarray(np.uint8(np.clip(out + 0.5, 0, 255)))


def corrupt_image(img: Image.Image, name: str, severity: int, key: str,
                  partner: Image.Image | None = None) -> Image.Image:
    if name not in STD or severity not in STD[name]:
        raise ValueError(f"{name}:{severity} is not on the standard scale "
                         f"({ {n: sorted(v) for n, v in STD.items()} })")
    if name in NEEDS_PARTNER and partner is None:
        raise ValueError(f"{name} needs a partner image")
    img, v = img.convert("RGB"), STD[name][severity]
    if name == "gaussian_noise":
        arr = np.asarray(img, dtype=np.float32) / 255.0
        noise = _rng(name, severity, key).normal(size=arr.shape, scale=v)
        return Image.fromarray(np.uint8(np.clip(arr + noise, 0, 1) * 255))
    if name == "pixelate":
        w, h = img.size
        small = img.resize((max(1, int(w * v)), max(1, int(h * v))), Image.BOX)
        return small.resize((w, h), Image.NEAREST)
    if name == "defocus_blur":
        return _blur(img, v)
    if name == "blur_gray":
        return grayscale(_blur(img, v))
    if name == "grayscale":
        return grayscale(img)
    if name == "mixup":
        return mix(img, partner, v)
    if name == "fisheye":
        return fisheye_k(img, v)
    base = np.array(img)  # cutmix
    h, w = base.shape[:2]
    other = np.array(partner.convert("RGB").resize((w, h)))
    x0, y0, x1, y1 = _box(_rng(name, severity, key), w, h, v)
    base[y0:y1, x0:x1] = other[y0:y1, x0:x1]
    return Image.fromarray(base)
