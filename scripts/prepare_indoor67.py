"""Turn the Hugging Face (FiftyOne) copy of Indoor67 into data/indoor67/Images/<class>/*.jpg.

    hf download Voxel51/IndoorSceneRecognition --repo-type dataset --local-dir data/indoor67_raw
    python scripts/prepare_indoor67.py

Labels come from samples.json (ground_truth.label). Files are hard-linked when
possible (no extra disk), otherwise copied. Safe to rerun. Images stay out of git.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import ALL_CLASSES  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/indoor67_raw")
    ap.add_argument("--out", default="data/indoor67/Images")
    args = ap.parse_args()
    raw, out = Path(args.raw), Path(args.out)

    samples = json.load(open(raw / "samples.json", encoding="utf-8"))["samples"]
    labels = {s["ground_truth"]["label"] for s in samples if s.get("ground_truth")}
    if labels != set(ALL_CLASSES):
        print("Class mismatch.\n  missing:", sorted(set(ALL_CLASSES) - labels),
              "\n  unexpected:", sorted(labels - set(ALL_CLASSES)))
        return 1

    counts, seen = Counter(), set()
    for s in tqdm(samples, desc="organizing"):
        label, src = s["ground_truth"]["label"], raw / s["filepath"]
        name = src.name
        if (label, name) in seen:
            raise RuntimeError(f"Duplicate name in class {label}: {name}")
        seen.add((label, name))
        dst = out / label / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        counts[label] += 1

    print(f"\nWrote {sum(counts.values())} images in {len(counts)} classes to {out}")
    print(f"Per class: min {min(counts.values())}, max {max(counts.values())}")
    return 0 if len(counts) == 67 else 1


if __name__ == "__main__":
    sys.exit(main())
