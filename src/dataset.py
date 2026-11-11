"""MIT Indoor67 as a 5-option multiple-choice scene-recognition task.

The committed manifest in splits/ is the source of truth. Both authors load it
and nobody regenerates it, so results are comparable. build_items() is kept so
the manifest can be audited and rebuilt on purpose.

Exchangeability note: pilot, calibration and test are carved out of ONE random
pool by position after a seeded shuffle. That is what makes them exchangeable,
which conformal prediction needs on clean data. Do not split by class or by
the official Train/Test lists.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

LETTERS = "ABCDEFGH"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# Every one of the 67 classes appears in exactly one group. "Confusable" means a
# plausible wrong answer for a robot: rooms from the same family.
CONFUSION_GROUPS: dict[str, list[str]] = {
    "bedroom_like": ["bedroom", "children_room", "nursery", "closet"],
    "home_rooms": ["livingroom", "dining_room", "gameroom", "bathroom"],
    "work_study": ["office", "meeting_room", "classroom", "kindergarden",
                   "computerroom", "laboratorywet"],
    "kitchens": ["kitchen", "pantry", "restaurant_kitchen", "winecellar"],
    "eateries": ["restaurant", "bar", "buffet", "fastfood_restaurant", "deli"],
    "food_retail": ["bakery", "grocerystore", "florist"],
    "shops": ["clothingstore", "shoeshop", "jewelleryshop", "toystore",
              "bookstore", "videostore", "mall", "hairsalon"],
    "medical": ["hospitalroom", "dentaloffice", "operating_room"],
    "culture": ["museum", "library", "artstudio", "church_inside", "cloister"],
    "performance": ["auditorium", "concert_hall", "movietheater", "studiomusic",
                    "tv_studio"],
    "transit": ["airport_inside", "trainstation", "subway", "inside_subway",
                "inside_bus"],
    "circulation": ["corridor", "stairscase", "elevator", "lobby", "waitingroom"],
    "sports_leisure": ["gym", "locker_room", "poolinside", "bowling", "casino"],
    "storage_utility": ["garage", "warehouse", "greenhouse", "laundromat",
                        "prisoncell"],
}
CLASS_TO_GROUP = {c: g for g, cs in CONFUSION_GROUPS.items() for c in cs}
ALL_CLASSES = sorted(CLASS_TO_GROUP)
DISTRACTOR_MODES = ("random", "mixed", "confusable")


# ----------------------------------------------------------------- loading
def find_images_root(data_dir) -> Path:
    """Locate the folder that holds the 67 class subfolders."""
    data_dir = Path(data_dir)
    for p in [data_dir, *sorted(data_dir.rglob("Images"))]:
        if p.is_dir():
            n = sum(1 for d in p.iterdir() if d.is_dir())
            if n == len(ALL_CLASSES):
                return p
    raise FileNotFoundError(
        f"No folder with {len(ALL_CLASSES)} class subfolders under {data_dir}. "
        "Run scripts/download_indoor67.py first."
    )


def load_pool(images_root) -> dict[str, list[str]]:
    """class -> sorted list of 'class/file.jpg' paths (relative to images_root)."""
    root = Path(images_root)
    pool: dict[str, list[str]] = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        names = sorted(f.name for f in d.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        pool[d.name] = [f"{d.name}/{n}" for n in names]
    missing = sorted(set(ALL_CLASSES) - set(pool))
    extra = sorted(set(pool) - set(ALL_CLASSES))
    if missing or extra:
        raise ValueError(
            f"Class names do not match CONFUSION_GROUPS. "
            f"Missing from disk: {missing}. Unknown on disk: {extra}. "
            "Fix the spelling in CONFUSION_GROUPS."
        )
    return pool


# ---------------------------------------------------------------- building
def _pick(rng: np.random.Generator, pool: list[str], k: int) -> list[str]:
    if k <= 0:
        return []
    idx = rng.choice(len(pool), size=k, replace=False)
    return [pool[int(i)] for i in idx]


def pick_distractors(true_class: str, rng: np.random.Generator, mode: str,
                     n_distractors: int) -> list[str]:
    if mode not in DISTRACTOR_MODES:
        raise ValueError(f"distractor_mode must be one of {DISTRACTOR_MODES}")
    same = [c for c in CONFUSION_GROUPS[CLASS_TO_GROUP[true_class]] if c != true_class]
    if mode == "random":
        return _pick(rng, [c for c in ALL_CLASSES if c != true_class], n_distractors)
    n_hard = n_distractors // 2 if mode == "mixed" else n_distractors
    hard = _pick(rng, same, min(n_hard, len(same)))
    outside = [c for c in ALL_CLASSES if c != true_class and c not in same]
    return hard + _pick(rng, outside, n_distractors - len(hard))


def build_items(pool: dict[str, list[str]], seed: int = 0, n_pilot: int = 100,
                n_calibration: int = 500, n_test: int = 500, n_options: int = 5,
                distractor_mode: str = "mixed") -> dict:
    rng = np.random.default_rng(seed)
    classes = sorted(pool)
    total = n_pilot + n_calibration + n_test
    base, extra = divmod(total, len(classes))
    bonus = set(int(i) for i in rng.choice(len(classes), size=extra, replace=False))

    chosen: list[tuple[str, str]] = []
    for ci, c in enumerate(classes):  # class-balanced pool, +1 for `extra` classes
        k = base + (1 if ci in bonus else 0)
        for i in sorted(int(j) for j in rng.choice(len(pool[c]), size=k, replace=False)):
            chosen.append((c, pool[c][i]))
    chosen = [chosen[int(i)] for i in rng.permutation(len(chosen))]

    # Correct-answer position is balanced exactly within each split (VLMs have letter bias,
    # so a lucky skew in the draw would leak into coverage). Slots are seeded-shuffled.
    sizes = [n_pilot, n_calibration, n_test]
    slots: list[int] = []
    for sz in sizes:
        block = np.resize(np.arange(n_options), sz)
        slots += [int(i) for i in rng.permutation(block)]

    items = []
    for n, (c, path) in enumerate(chosen):
        others = pick_distractors(c, rng, distractor_mode, n_options - 1)
        others = [others[int(i)] for i in rng.permutation(len(others))]
        options = others[:slots[n]] + [c] + others[slots[n]:]
        items.append({"id": f"i67-{n:05d}", "image": path, "true_class": c,
                      "options": options, "answer": LETTERS[slots[n]]})

    cut1, cut2 = n_pilot, n_pilot + n_calibration
    manifest = {
        "meta": {
            "dataset": "MIT Indoor67 (Quattoni & Torralba, CVPR 2009)",
            "seed": seed, "n_options": n_options, "distractor_mode": distractor_mode,
            "n_pilot": n_pilot, "n_calibration": n_calibration, "n_test": n_test,
            "classes_sha1": hashlib.sha1("\n".join(ALL_CLASSES).encode()).hexdigest(),
            "confusion_groups": CONFUSION_GROUPS,
        },
        "splits": {"pilot": items[:cut1], "calibration": items[cut1:cut2],
                   "test": items[cut2:cut2 + n_test]},
    }
    verify_manifest(manifest)
    return manifest


# -------------------------------------------------------------- validation
def verify_manifest(m: dict) -> None:
    n_opt = m["meta"]["n_options"]
    seen_images, seen_ids = set(), set()
    for split, items in m["splits"].items():
        for it in items:
            if it["image"] in seen_images:
                raise ValueError(f"image appears twice across splits: {it['image']}")
            if it["id"] in seen_ids:
                raise ValueError(f"duplicate id: {it['id']}")
            seen_images.add(it["image"]); seen_ids.add(it["id"])
            opts = it["options"]
            if len(opts) != n_opt or len(set(opts)) != n_opt:
                raise ValueError(f"{it['id']}: need {n_opt} unique options")
            if opts[LETTERS.index(it["answer"])] != it["true_class"]:
                raise ValueError(f"{it['id']}: answer letter does not match true class")
            if it["image"].split("/")[0] != it["true_class"]:
                raise ValueError(f"{it['id']}: image folder != true class")


def save_manifest(m: dict, path) -> None:
    verify_manifest(m)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(m, f, indent=1)
        f.write("\n")


def load_manifest(path) -> dict:
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    verify_manifest(m)
    return m
