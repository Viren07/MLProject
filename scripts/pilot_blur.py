"""Quick look: does defocus blur hurt GPT-4o on the 100 PILOT images?

    python scripts/pilot_blur.py

Pilot split only. Prints accuracy, confidence, and how many wrong answers were
confidently wrong (top probability >= 0.9). Saves a few before/after example
images to results/raw/blur_examples/ so you can eyeball the corruption.
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv
from imagecorruptions import corrupt
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import LETTERS, find_images_root, load_manifest, verify_manifest  # noqa: E402
from src.scoring import GPTScorer  # noqa: E402


def blur(img: Image.Image, severity: int) -> Image.Image:
    arr = corrupt(np.array(img.convert("RGB")), corruption_name="defocus_blur", severity=severity)
    return Image.fromarray(np.uint8(arr))


def main() -> int:
    load_dotenv()
    cfg = yaml.safe_load(open("configs/experiment.yaml", encoding="utf-8"))
    mcfg = next(m for m in cfg["models"] if m["provider"] == "openai")
    man = load_manifest(cfg["task"]["manifest"]); verify_manifest(man)
    items = man["splits"]["pilot"]
    root = find_images_root(cfg["task"]["data_dir"])
    scorer = GPTScorer(model=mcfg["name"], cache_dir=cfg["cache_dir"],
                       detail=mcfg.get("image_detail", "high"), jpeg_quality=mcfg.get("jpeg_quality", 95))
    levels = [0] + [s for c in cfg["corruptions"] if c["name"] == "defocus_blur" for s in c["severities"]]

    ex_dir = Path(cfg["cache_dir"]).parent / "blur_examples"; ex_dir.mkdir(parents=True, exist_ok=True)
    for k, it in enumerate(items[:4]):
        with Image.open(root / it["image"]) as im:
            im = im.convert("RGB"); h = 360; w = int(im.width * h / im.height)
            tiles = [im if lv == 0 else blur(im, lv) for lv in levels]
            strip = Image.new("RGB", (w * len(tiles), h))
            for j, t in enumerate(tiles):
                strip.paste(t.resize((w, h)), (j * w, 0))
            strip.save(ex_dir / f"example{k}_sev{'_'.join(map(str, levels))}.jpg", quality=92)

    gold = [LETTERS.index(it["answer"]) for it in items]
    print(f"\n{'severity':>9} {'acc':>6} {'top p':>7} {'wrong':>6} {'conf. wrong':>12}")
    for lv in levels:
        def run(it):
            with Image.open(root / it["image"]) as im:
                return scorer.score(im if lv == 0 else blur(im, lv), it["options"])
        with ThreadPoolExecutor(6) as ex:
            res = list(tqdm(ex.map(run, items), total=len(items), desc=f"sev {lv}", leave=False))
        pred = [max(range(5), key=r["probs"].__getitem__) for r in res]
        top = [max(r["probs"]) for r in res]
        wrong = [i for i in range(len(items)) if pred[i] != gold[i]]
        cw = sum(top[i] >= 0.9 for i in wrong)
        print(f"{'clean' if lv == 0 else lv:>9} {1 - len(wrong) / len(items):6.1%} {np.mean(top):7.3f} {len(wrong):6d} {cw:12d}")
    print(f"\nExample images (clean | blurred): {ex_dir}")
    print("Open them and check the blur looks like a believable out-of-focus photo, not destroyed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
