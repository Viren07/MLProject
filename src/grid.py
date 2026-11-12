"""Run one condition (clean or a corruption at a severity) over a list of items and save probs."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from .corruptions import NEEDS_PARTNER, corrupt_image


def conditions_from_config(cfg: dict) -> list[str]:
    return ["clean"] + [f"{c['name']}:{s}" for c in cfg["corruptions"] for s in c["severities"]]


def slug(cond: str) -> str:
    return cond.replace(":", "-")  # ':' is not allowed in Windows filenames


def apply_condition(img: Image.Image, cond: str, item_id: str, seed: int = 0,
                    partner: Image.Image | None = None) -> Image.Image:
    if cond == "clean":
        return img.convert("RGB")
    name, sev = cond.split(":")
    return corrupt_image(img, name, int(sev), key=f"{seed}:{item_id}", partner=partner)


def render_item(img: Image.Image, cond: str, item: dict, seed: int = 0, partners=None) -> Image.Image:
    """Apply `cond` to an item's image, fetching a partner image when the condition needs one."""
    partner = None
    if cond != "clean" and cond.split(":")[0] in NEEDS_PARTNER:
        if partners is None:
            raise ValueError(f"{cond} needs a PartnerPool")
        partner = partners.get(item["true_class"], f"{seed}:{item['id']}")
    return apply_condition(img, cond, item["id"], seed, partner)


def run_condition(items, root, scorer, cond: str, seed: int = 0, workers: int = 6, desc: str = "",
                  partners=None) -> list[dict]:
    def run(it):
        with Image.open(Path(root) / it["image"]) as im:
            img = render_item(im.convert("RGB"), cond, it, seed, partners)
        r = scorer.score(img, it["options"])
        return {"id": it["id"], "probs": r["probs"], "letter_mass": r["letter_mass"]}

    with ThreadPoolExecutor(workers) as ex:
        results = ex.map(run, items)
        try:
            from tqdm import tqdm
            results = tqdm(results, total=len(items), desc=desc or cond)
        except ImportError:
            pass
        return list(results)


def probs_path(out_dir, model: str, split: str, cond: str) -> Path:
    return Path(out_dir) / model / f"{split}__{slug(cond)}.json"


def save_probs(out_dir, model, split, cond, rows) -> Path:
    p = probs_path(out_dir, model, split, cond)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rows))
    return p


def load_probs(out_dir, model, split, cond):
    p = probs_path(out_dir, model, split, cond)
    return json.loads(p.read_text()) if p.exists() else None
