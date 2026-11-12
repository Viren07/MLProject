"""Score images with the VLM under each condition and save the probabilities.

    python scripts/run_grid.py                     # the Week 1 gate: calibration + test, CLEAN only
    python scripts/run_grid.py --all               # clean + every corruption/severity, on test
    python scripts/run_grid.py --split pilot --all # (never needed; pilot is for screening)

Calibration is always scored clean only. Corrupted conditions are scored on the test split.
Every API call is cached on disk, so reruns and interrupted runs cost nothing extra.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import find_images_root, load_manifest, verify_manifest  # noqa: E402
from src.grid import conditions_from_config, run_condition, save_probs  # noqa: E402
from src.partners import PartnerPool  # noqa: E402
from src.scoring import GPTScorer, OutOfCredits  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--all", action="store_true", help="also score every corruption on the test split")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    load_dotenv()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    mcfg = next(m for m in cfg["models"] if m["provider"] == "openai")
    man = load_manifest(cfg["task"]["manifest"]); verify_manifest(man)
    root = find_images_root(cfg["task"]["data_dir"])
    scorer = GPTScorer(model=mcfg["name"], cache_dir=cfg["cache_dir"],
                       detail=mcfg.get("image_detail", "high"), jpeg_quality=mcfg.get("jpeg_quality", 95))
    out = cfg.get("probs_dir", "results/probs")
    partners = PartnerPool(root, {it["image"] for sp in man["splits"].values() for it in sp})

    jobs = [("calibration", "clean"), ("test", "clean")]
    if args.all:
        jobs += [("test", c) for c in conditions_from_config(cfg) if c != "clean"]
    for split, cond in jobs:
        try:
            rows = run_condition(man["splits"][split], root, scorer, cond, seed=cfg["seed"],
                                 workers=args.workers, desc=f"{split} {cond}", partners=partners)
        except OutOfCredits as e:
            print(f"\nSTOPPED at {split} {cond}: {e}")
            return 2
        low = sum(r["letter_mass"] < 0.95 for r in rows)
        path = save_probs(out, mcfg["name"], split, cond, rows)
        print(f"saved {path}  ({len(rows)} rows, {low} with letter mass < 0.95)")
    print("\nNext: python scripts/analyze.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
