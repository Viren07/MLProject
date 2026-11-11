"""Quick look: does a corruption hurt GPT-4o on the 100 PILOT images?

    python scripts/pilot_blur.py                                  # defocus_blur from the config
    python scripts/pilot_blur.py gaussian_noise:3,5 contrast:3,5  # any imagecorruptions names

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


def blur(img: Image.Image, severity: int, name: str = "defocus_blur") -> Image.Image:
    np.random.seed(0)  # noise-type corruptions use numpy's global RNG; fixed here for repeatability
    arr = corrupt(np.array(img.convert("RGB")), corruption_name=name, severity=severity)
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
    specs = sys.argv[1:] or [f"{c['name']}:{','.join(map(str, c['severities']))}" for c in cfg["corruptions"]]
    for spec in specs:
        name, sev = spec.split(":")
        run_one(name, [int(x) for x in sev.split(",")], cfg, items, root, scorer)
    return 0


def run_one(name, sevs, cfg, items, root, scorer) -> None:
    levels = [0] + sevs
    print(f"\n=== {name} ===")

    ex_dir = Path(cfg["cache_dir"]).parent / "blur_examples"; ex_dir.mkdir(parents=True, exist_ok=True)
    for k, it in enumerate(items[:4]):
        with Image.open(root / it["image"]) as im:
            im = im.convert("RGB"); h = 360; w = int(im.width * h / im.height)
            tiles = [im if lv == 0 else blur(im, lv, name) for lv in levels]
            strip = Image.new("RGB", (w * len(tiles), h))
            for j, t in enumerate(tiles):
                strip.paste(t.resize((w, h)), (j * w, 0))
            strip.save(ex_dir / f"example{k}_{name}.jpg", quality=92)

    gold = [LETTERS.index(it["answer"]) for it in items]
    print(f"\n{'severity':>9} {'acc':>6} {'top p':>7} {'wrong':>6} {'conf. wrong':>12}")
    for lv in levels:
        def run(it):
            with Image.open(root / it["image"]) as im:
                return scorer.score(im if lv == 0 else blur(im, lv, name), it["options"])
        with ThreadPoolExecutor(6) as ex:
            res = list(tqdm(ex.map(run, items), total=len(items), desc=f"sev {lv}", leave=False))
        pred = [max(range(5), key=r["probs"].__getitem__) for r in res]
        top = [max(r["probs"]) for r in res]
        wrong = [i for i in range(len(items)) if pred[i] != gold[i]]
        cw = sum(top[i] >= 0.9 for i in wrong)
        print(f"{'clean' if lv == 0 else lv:>9} {1 - len(wrong) / len(items):6.1%} {np.mean(top):7.3f} {len(wrong):6d} {cw:12d}")
    print(f"\nExample images (clean | blurred): {ex_dir}")
    print("Open them and check the corruption looks believable, not destroyed.")


if __name__ == "__main__":
    sys.exit(main())
