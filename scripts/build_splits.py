"""Build the pilot/calibration/test manifest. Run ONCE, then commit the output.

    python scripts/build_splits.py

Reads settings from configs/experiment.yaml. Unreadable images are dropped from
the pool BEFORE sampling, so the draw stays unbiased and deterministic.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import build_items, find_images_root, load_pool, save_manifest  # noqa: E402


def readable_only(root: Path, pool: dict) -> tuple[dict, list[str]]:
    bad, clean = [], {}
    for c, paths in tqdm(pool.items(), desc="checking images"):
        keep = []
        for p in paths:
            try:
                with Image.open(root / p) as im:
                    im.verify()
                with Image.open(root / p) as im:
                    im.convert("RGB").load()
                keep.append(p)
            except Exception:  # noqa: BLE001
                bad.append(p)
        clean[c] = keep
    return clean, bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--data-dir", default="data/indoor67")
    ap.add_argument("--out", default=None)
    ap.add_argument("--distractor-mode", default=None, help="override config (for exploration)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing manifest")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    t = cfg["task"]
    out = Path(args.out or f"splits/indoor67_seed{cfg['seed']}.json")
    if out.exists() and not args.force:
        print(f"{out} already exists. It is the source of truth; use --force only to audit/rebuild.")
        return 1

    root = find_images_root(args.data_dir)
    raw = load_pool(root)
    print(f"Images root: {root}")
    print(f"Found {len(raw)} classes, {sum(map(len, raw.values()))} images (Indoor67 should have 67 and 15620)")
    pool, bad = readable_only(root, raw)
    print(f"Dropped {len(bad)} unreadable image(s): {bad[:5]}{' ...' if len(bad) > 5 else ''}")

    m = build_items(pool, seed=cfg["seed"], n_pilot=t["n_pilot"], n_calibration=t["n_calibration"],
                    n_test=t["n_test"], n_options=t["n_options"], distractor_mode=args.distractor_mode or t["distractor_mode"])
    m["meta"]["dropped_unreadable"] = sorted(bad)
    save_manifest(m, out)

    print(f"\nWrote {out}")
    for name, items in m["splits"].items():
        letters = Counter(it["answer"] for it in items)
        sizes = Counter()
        for it in items[:50]:
            with Image.open(root / it["image"]) as im:
                sizes[im.size] += 1
        print(f"  {name:12s} n={len(items):4d}  answer letters {dict(sorted(letters.items()))}")
    ex = m["splits"]["test"][0]
    print(f"\nExample: {ex['image']}\n  options {ex['options']}  answer {ex['answer']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
