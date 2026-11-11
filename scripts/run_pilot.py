"""Pilot: run GPT-4o on the pilot split (clean images only) and check difficulty.

    python scripts/run_pilot.py --n 20     # smoke test first
    python scripts/run_pilot.py            # all 100 pilot images

Uses ONLY the pilot split. Never touches calibration or test.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import LETTERS, find_images_root, load_manifest, verify_manifest  # noqa: E402
from src.scoring import GPTScorer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--manifest", default=None, help="override manifest (for exploration)")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    load_dotenv()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    mcfg = next(m for m in cfg["models"] if m["provider"] == "openai")

    man = load_manifest(args.manifest or cfg["task"]["manifest"])
    verify_manifest(man)
    items = man["splits"]["pilot"][: args.n]
    root = find_images_root(cfg["task"]["data_dir"])
    scorer = GPTScorer(model=mcfg["name"], cache_dir=cfg.get("cache_dir", "results/raw/cache"),
                       detail=mcfg.get("image_detail", "high"), jpeg_quality=mcfg.get("jpeg_quality", 95))

    def run(it):
        with Image.open(root / it["image"]) as im:
            return scorer.score(im, it["options"])

    with ThreadPoolExecutor(args.workers) as ex:
        res = list(tqdm(ex.map(run, items), total=len(items), desc="GPT-4o pilot"))

    n = len(items)
    pred = [max(range(len(r["probs"])), key=r["probs"].__getitem__) for r in res]
    gold = [LETTERS.index(it["answer"]) for it in items]
    acc = sum(p == g for p, g in zip(pred, gold)) / n
    maxp = [max(r["probs"]) for r in res]
    mass = [r["letter_mass"] for r in res]
    sing = sum(m >= 0.9 for m in maxp) / n
    print(f"\nImages {n} | accuracy {acc:.1%} | mean top prob {sum(maxp)/n:.3f} | top prob >= 0.9: {sing:.0%}")
    print(f"Letter mass: min {min(mass):.3f}, mean {sum(mass)/n:.3f}  (should be near 1.0)")
    print("Predicted letter counts:", {LETTERS[i]: pred.count(i) for i in range(5)})
    print("Accuracy by correct letter:", {LETTERS[i]: f"{sum(p == g for p, g in zip(pred, gold) if g == i)}/{gold.count(i)}" for i in range(5)})
    print("Served model:", {r["served_model"] for r in res}, "| first tokens:", {r["first_token"] for r in res})
    wrong = [(it["true_class"], it["options"][p]) for it, p, g in zip(items, pred, gold) if p != g]
    print("Mistakes (true -> predicted):", wrong[:10])
    print("\nReading the result:")
    print("  accuracy > 97%  : ceiling effect, too easy. Switch distractor_mode to 'confusable'.")
    print("  accuracy 70-95% : good range, errors leave room for corruption to matter.")
    print("  accuracy < 60%  : too hard or prompt problem. Inspect mistakes before going on.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
