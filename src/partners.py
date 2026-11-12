"""Partner images for cutmix/mixup: from images in NO split, always a different room class."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .corruptions import seed_for
from .dataset import load_pool


class PartnerPool:
    def __init__(self, images_root, used_paths):
        self.root = Path(images_root)
        used = set(used_paths)
        self.pool = {c: [p for p in paths if p not in used] for c, paths in load_pool(self.root).items()}
        self.classes = sorted(c for c, v in self.pool.items() if v)

    def get(self, true_class: str, key: str) -> Image.Image:
        rng = np.random.default_rng(seed_for("partner", 0, key))
        others = [c for c in self.classes if c != true_class]
        c = others[int(rng.integers(len(others)))]
        path = self.pool[c][int(rng.integers(len(self.pool[c])))]
        with Image.open(self.root / path) as im:
            return im.convert("RGB")
