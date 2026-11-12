"""REFERENCE ONLY. Not imported by the pipeline.

Implementations of the corruptions that were screened on the 100 pilot images and dropped because
GPT-4o stayed within about 4 points of clean even at their harshest level (results/pilot_screening.md).
Kept so that screening can be reproduced. Severity numbers here are the old 1-7 levels, not the
25/50/75/100 standard scale. `contrast`, `fog` and the early `pixelate` levels came from the
`imagecorruptions` package (pip install imagecorruptions); they are not reproduced here.
"""
import numpy as np
from PIL import Image

VIGNETTE_ONSET = {1: 0.70, 2: 0.60, 3: 0.50, 4: 0.40, 5: 0.30, 6: 0.15, 7: 0.0}
VIGNETTE_POWER = {1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 1, 7: 1}
PATCH_AREA = {1: 0.05, 2: 0.10, 3: 0.20, 4: 0.30, 5: 0.45}
SHUFFLE_GRID = {1: 2, 2: 3, 3: 4, 4: 6, 5: 8, 6: 12}
HF_AMPLITUDE = {1: 8.0, 2: 16.0, 3: 32.0, 4: 48.0, 5: 64.0}
PALETTE = [(255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 0, 0), (255, 255, 255), (255, 0, 0)]


def vignette(img, severity, gray=False):
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - (w - 1) / 2) / max((w - 1) / 2, 1)) ** 2
                + ((yy - (h - 1) / 2) / max((h - 1) / 2, 1)) ** 2) / np.sqrt(2)
    t = np.clip((r - VIGNETTE_ONSET[severity]) / (1 - VIGNETTE_ONSET[severity]), 0, 1)
    out = Image.fromarray(np.uint8(np.clip(arr * (1 - t ** VIGNETTE_POWER[severity])[..., None], 0, 255)))
    return out.convert("L").convert("RGB") if gray else out


def patch(img, severity, rng):
    """High-contrast checkerboard sticker. A distractor patch, not optimized against any model."""
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]
    side = min(round((PATCH_AREA[severity] * w * h) ** 0.5), h, w)
    x0, y0 = int(rng.integers(0, w - side + 1)), int(rng.integers(0, h - side + 1))
    c1, c2 = (PALETTE[i] for i in rng.choice(len(PALETTE), size=2, replace=False))
    cell = max(side // 6, 1)
    yy, xx = np.mgrid[0:side, 0:side]
    mask = ((yy // cell + xx // cell) % 2).astype(bool)
    arr[y0:y0 + side, x0:x0 + side] = np.where(mask[..., None], np.array(c1, np.uint8), np.array(c2, np.uint8))
    return Image.fromarray(arr)


def shuffle(img, severity, rng):
    """Grid scramble: cut into g x g tiles and permute them."""
    g = SHUFFLE_GRID[severity]
    arr = np.array(img.convert("RGB"))
    h, w = (arr.shape[0] // g) * g, (arr.shape[1] // g) * g
    arr = arr[:h, :w]
    th, tw = h // g, w // g
    tiles = [arr[i * th:(i + 1) * th, j * tw:(j + 1) * tw] for i in range(g) for j in range(g)]
    out = np.empty_like(arr)
    for n, t in enumerate(rng.permutation(len(tiles))):
        i, j = divmod(n, g)
        out[i * th:(i + 1) * th, j * tw:(j + 1) * tw] = tiles[t]
    return Image.fromarray(out)


def hfgrid(img, severity, rng):
    """Faint high-frequency sinusoidal grating."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    theta, phase = float(rng.uniform(0, np.pi)), float(rng.uniform(0, 2 * np.pi))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    wave = np.sin(2 * np.pi * (xx * np.cos(theta) + yy * np.sin(theta)) / 4.0 + phase)
    return Image.fromarray(np.uint8(np.clip(arr + HF_AMPLITUDE[severity] * wave[..., None] + 0.5, 0, 255)))
