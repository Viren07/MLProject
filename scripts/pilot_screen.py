"""Pilot screen: how much does each corruption level hurt GPT-4o on the 100 PILOT images?

    python scripts/pilot_screen.py                                      # the grid in configs/experiment.yaml
    python scripts/pilot_screen.py pixelate:25,50,75,100 fisheye:100    # any family:levels

Pilot split only (never calibration or test). Prints accuracy, confidence, and how many wrong
answers were confidently wrong (top probability >= 0.9), then one combined table, and writes it
to results/pilot_standard_levels.md. Saves before/after example strips to results/raw/screen_examples/
so you can check each level still looks like a recognizable room.
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import LETTERS, find_images_root, load_manifest, verify_manifest  # noqa: E402
from src.grid import render_item  # noqa: E402
from src.partners import PartnerPool  # noqa: E402
from src.scoring import GPTScorer  # noqa: E402


def main() -> int:
    load_dotenv()
    cfg = yaml.safe_load(open("configs/experiment.yaml", encoding="utf-8"))
    mcfg = next(m for m in cfg["models"] if m["provider"] == "openai")
    man = load_manifest(cfg["task"]["manifest"]); verify_manifest(man)
    items = man["splits"]["pilot"]
    root = find_images_root(cfg["task"]["data_dir"])
    scorer = GPTScorer(model=mcfg["name"], cache_dir=cfg["cache_dir"],
                       detail=mcfg.get("image_detail", "high"), jpeg_quality=mcfg.get("jpeg_quality", 95))
    partners = PartnerPool(root, {it["image"] for sp in man["splits"].values() for it in sp})
    specs = sys.argv[1:] or [f"{c['name']}:{','.join(map(str, c['severities']))}" for c in cfg["corruptions"]]
    allrows = []
    for spec in specs:
        name, sev = spec.split(":")
        allrows += run_one(name, [int(x) for x in sev.split(",")], cfg, items, root, scorer, partners)
    summarize(allrows)
    return 0


def summarize(rows) -> None:
    """One combined table (accuracy by family and level) and a markdown copy for the paper's appendix."""
    levels = sorted({r["level"] for r in rows if r["level"] != 0})
    fams = list(dict.fromkeys(r["name"] for r in rows))
    clean = next(r for r in rows if r["level"] == 0)
    cell = lambda f, lv: next((f"{r['acc']:.0%}" for r in rows if r["name"] == f and r["level"] == lv), "-")
    lines = [f"| family | " + " | ".join(f"L{lv}" for lv in levels) + " |", "|---|" + "---|" * len(levels)]
    lines += [f"| {f} | " + " | ".join(cell(f, lv) for lv in levels) + " |" for f in fams]
    table = "\n".join(lines)
    print(f"\nAccuracy by level (clean = {clean['acc']:.0%}); L25 = 25% of the family's max strength, L100 = max:\n")
    print(table)
    detail = ["| family | level | accuracy | mean top prob | errors | confident errors |", "|---|---|---|---|---|---|"]
    detail += [f"| {r['name']} | {r['level']} | {r['acc']:.0%} | {r['top']:.3f} | {r['wrong']} | {r['cw']} |"
               for r in rows if r["level"] != 0]
    Path("results").mkdir(exist_ok=True)
    Path("results/pilot_standard_levels.md").write_text(
        "# Pilot, standard severity scale (GPT-4o, 100 pilot images)\n\n"
        f"Clean accuracy {clean['acc']:.0%}. Level = percent of each family's maximum usable strength.\n\n"
        + table + "\n\n" + "\n".join(detail) + "\n")
    print("\nwrote results/pilot_standard_levels.md")


def run_one(name, sevs, cfg, items, root, scorer, partners) -> list:
    levels = [0] + sevs
    print(f"\n=== {name} ===")

    ex_dir = Path(cfg["cache_dir"]).parent / "screen_examples"; ex_dir.mkdir(parents=True, exist_ok=True)
    for k, it in enumerate(items[:4]):
        with Image.open(root / it["image"]) as im:
            im = im.convert("RGB"); h = 360; w = int(im.width * h / im.height)
            tiles = [im if lv == 0 else render_item(im, f"{name}:{lv}", it, cfg["seed"], partners) for lv in levels]
            strip = Image.new("RGB", (w * len(tiles), h))
            for j, t in enumerate(tiles):
                strip.paste(t.resize((w, h)), (j * w, 0))
            strip.save(ex_dir / f"example{k}_{name}.jpg", quality=92)

    gold = [LETTERS.index(it["answer"]) for it in items]
    print(f"\n{'severity':>9} {'acc':>6} {'top p':>7} {'wrong':>6} {'conf. wrong':>12}")
    rows = []
    for lv in levels:
        def run(it):
            with Image.open(root / it["image"]) as im:
                return scorer.score(im if lv == 0 else render_item(im, f"{name}:{lv}", it, cfg["seed"], partners), it["options"])
        with ThreadPoolExecutor(6) as ex:
            res = list(tqdm(ex.map(run, items), total=len(items), desc=f"sev {lv}", leave=False))
        pred = [max(range(5), key=r["probs"].__getitem__) for r in res]
        top = [max(r["probs"]) for r in res]
        wrong = [i for i in range(len(items)) if pred[i] != gold[i]]
        cw = sum(top[i] >= 0.9 for i in wrong)
        print(f"{'clean' if lv == 0 else lv:>9} {1 - len(wrong) / len(items):6.1%} {np.mean(top):7.3f} {len(wrong):6d} {cw:12d}")
        rows.append({"name": name, "level": lv, "acc": 1 - len(wrong) / len(items), "top": float(np.mean(top)),
                     "wrong": len(wrong), "cw": cw})
    print(f"\nExample images (clean | corrupted): {ex_dir}")
    print("Open them and check the corruption looks believable, not destroyed.")
    return rows


if __name__ == "__main__":
    sys.exit(main())
