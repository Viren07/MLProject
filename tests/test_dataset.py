import copy
from collections import Counter

import pytest

from src.dataset import (ALL_CLASSES, CLASS_TO_GROUP, CONFUSION_GROUPS, LETTERS,
                         build_items, verify_manifest)


@pytest.fixture(scope="module")
def pool():
    return {c: [f"{c}/img_{i:04d}.jpg" for i in range(120)] for c in ALL_CLASSES}


@pytest.fixture(scope="module")
def manifest(pool):
    return build_items(pool, seed=0)


def test_groups_partition_exactly_67_classes():
    flat = [c for cs in CONFUSION_GROUPS.values() for c in cs]
    assert len(flat) == 67 and len(set(flat)) == 67
    assert all(len(cs) >= 3 for cs in CONFUSION_GROUPS.values())  # >= 2 confusables each


def test_split_sizes(manifest):
    s = manifest["splits"]
    assert (len(s["pilot"]), len(s["calibration"]), len(s["test"])) == (100, 500, 500)


def test_deterministic_and_seed_sensitive(pool, manifest):
    assert build_items(pool, seed=0) == manifest
    assert build_items(pool, seed=1) != manifest


def test_splits_are_disjoint(manifest):
    imgs = [it["image"] for items in manifest["splits"].values() for it in items]
    assert len(imgs) == len(set(imgs)) == 1100


def test_class_balance(manifest):
    counts = Counter(it["true_class"] for items in manifest["splits"].values() for it in items)
    assert set(counts) == set(ALL_CLASSES)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_every_item_is_well_formed(manifest):
    for items in manifest["splits"].values():
        for it in items:
            assert len(set(it["options"])) == 5
            assert it["options"]["ABCDE".index(it["answer"])] == it["true_class"]


def test_mixed_mode_has_two_confusable_and_two_random(manifest):
    for items in manifest["splits"].values():
        for it in items:
            g = CLASS_TO_GROUP[it["true_class"]]
            wrong = [o for o in it["options"] if o != it["true_class"]]
            assert sum(CLASS_TO_GROUP[o] == g for o in wrong) == 2


def test_modes(pool):
    rnd = build_items(pool, seed=0, distractor_mode="random")
    conf = build_items(pool, seed=0, distractor_mode="confusable")
    for it in conf["splits"]["test"]:
        g = CLASS_TO_GROUP[it["true_class"]]
        wrong = [o for o in it["options"] if o != it["true_class"]]
        in_group = sum(CLASS_TO_GROUP[o] == g for o in wrong)
        assert in_group == min(4, len(CONFUSION_GROUPS[g]) - 1)
    assert rnd["splits"]["test"] != conf["splits"]["test"]
    with pytest.raises(ValueError):
        build_items(pool, distractor_mode="nope")


def test_answer_letters_are_exactly_balanced_per_split(manifest):
    for name, items in manifest["splits"].items():
        letters = Counter(it["answer"] for it in items)
        assert set(letters) == set(LETTERS[:5])
        assert all(n == len(items) // 5 for n in letters.values()), name


def test_verify_catches_tampering(manifest):
    bad = copy.deepcopy(manifest)
    bad["splits"]["test"][0]["answer"] = "A" if bad["splits"]["test"][0]["answer"] != "A" else "B"
    with pytest.raises(ValueError):
        verify_manifest(bad)
    leak = copy.deepcopy(manifest)
    leak["splits"]["test"][0]["image"] = leak["splits"]["calibration"][0]["image"]
    with pytest.raises(ValueError):
        verify_manifest(leak)
