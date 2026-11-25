"""Shared helpers for the reporting scripts (analyze, sweep_alpha, robustness)."""
from __future__ import annotations

import numpy as np

from .dataset import LETTERS


def arrays(rows, items):
    """Saved prob rows + manifest items -> (probabilities (n, K), gold index (n,))."""
    gold = {it["id"]: LETTERS.index(it["answer"]) for it in items}
    return (np.array([r["probs"] for r in rows], dtype=float),
            np.array([gold[r["id"]] for r in rows]))


def resolve_model(cfg: dict, name: str | None = None) -> dict:
    """Pick a model block from the config by name, defaulting to the first one listed."""
    models = cfg["models"]
    if name is None:
        return models[0]
    for m in models:
        if m["name"] == name:
            return m
    have = ", ".join(m["name"] for m in models)
    raise SystemExit(f"model {name!r} is not in the config; it has: {have}")


def headline_conditions(conds) -> list[str]:
    """Clean plus every family at full strength: the set worth putting on one chart."""
    return ["clean"] + [c for c in conds if c.endswith(":100")]
