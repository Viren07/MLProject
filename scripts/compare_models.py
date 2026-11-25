"""Compare models on equal footing, and measure how many of their own errors they flag.

    python scripts/compare_models.py                 # matched targets (default)
    python scripts/compare_models.py --fixed         # every model at the config's target instead
    python scripts/compare_models.py --offset 5      # matched target = clean accuracy minus this

Why matched targets. Split conformal reads its cut-off off the calibration scores at the
requested quantile. Ask for less reliability than the model's accuracy and that quantile sits
among images the model got right, where scores are tiny, so shortlists come out singleton or
empty. Ask for more and the quantile is forced into the region of images it got wrong, where
scores are near one, so shortlists widen. A single fixed target therefore lands on opposite
sides of that boundary for models of different accuracy, and the resulting difference in
behaviour says nothing about robustness. Giving each model a target a fixed margin below its
own clean calibration accuracy puts every model on the same side of its own boundary.

Because targets then differ, coverage is no longer comparable across models: 70% is a disaster
against a 90% promise and fine against a 76% one. The comparable quantity is the shortfall,
the model's own target minus the coverage it delivered, which is near zero on clean images by
construction and so isolates the damage done by corruption.

Nothing here touches the test labels to pick a target: accuracy comes from the calibration
split, which calibration is already allowed to see.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.conformal import calibrate, prediction_sets, true_scores  # noqa: E402
from src.dataset import load_manifest  # noqa: E402
from src.grid import conditions_from_config, load_probs  # noqa: E402
from src.report import arrays  # noqa: E402


def measure(P, g, qhat: float) -> dict:
    """Coverage, how the misses break down, and how many of the model's errors got flagged."""
    n = len(g)
    sets = prediction_sets(P, qhat)
    size = sets.sum(axis=1)
    covered = sets[np.arange(n), g]
    wrong = P.argmax(axis=1) != g            # the model's own mistakes, before conformal
    silent = (size == 1) & ~covered          # committed to one answer, and it was wrong
    return {
        "coverage": float(covered.mean()),
        "accuracy": float(1 - wrong.mean()),
        "empty": float((size == 0).mean()),
        "multi_miss": float(((size >= 2) & ~covered).mean()),
        "multi": float((size >= 2).mean()),
        "silent": float(silent.mean()),
        "commit": float((size == 1).mean()),
        "wrong_given_commit": float((~covered[size == 1]).mean()) if (size == 1).any() else float("nan"),
        # of everything the model got wrong, the share conformal did NOT wave through as a
        # single confident answer. 1.0 would mean every mistake came with a visible flag.
        "errors_flagged": float(1 - silent.sum() / wrong.sum()) if wrong.any() else float("nan"),
    }


def model_setup(cfg, model: str, man, offset: float | None):
    """Return (target, qhat, clean calibration accuracy) for one model."""
    out = cfg.get("probs_dir", "results/probs")
    cal = load_probs(out, model, "calibration", "clean")
    if cal is None:
        return None
    Pc, gc = arrays(cal, man["splits"]["calibration"])
    acc = float((Pc.argmax(axis=1) == gc).mean())
    target = acc - offset if offset is not None else 1 - cfg["conformal"]["alpha"]
    return target, calibrate(true_scores(Pc, gc), round(1 - target, 6)), acc


def figures(rows, models, out_dir: Path, tag: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = [c for c in dict.fromkeys(r["condition"] for r in rows) if c != "clean"]
    key = models[0]
    conds.sort(key=lambda c: -next(r["shortfall_points"] for r in rows
                                   if r["condition"] == c and r["model"] == key))
    y = np.arange(len(conds))
    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(conds) + 2))
    h = 0.8 / len(models)
    for i, m in enumerate(models):
        vals = [next(r["shortfall_points"] for r in rows if r["condition"] == c and r["model"] == m)
                for c in conds]
        ax.barh(y + i * h - 0.4 + h / 2, vals, height=h * 0.9, label=m, color=colors[i])
    ax.set_yticks(y)
    ax.set_yticklabels([c.replace("_", " ") for c in conds], fontsize=8)
    ax.set_xlabel("shortfall: the model's own target minus delivered coverage (points)")
    ax.set_title(f"How far each model falls below its promise ({tag} targets)")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"compare_shortfall_{tag}.png", dpi=150)
    print(f"wrote {out_dir / f'compare_shortfall_{tag}.png'}")

    fig, axes = plt.subplots(1, len(models), figsize=(5.6 * len(models), 0.42 * len(conds) + 2.4),
                             sharey=True, sharex=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models):
        get = lambda c, k: next(r[k] for r in rows if r["condition"] == c and r["model"] == m)  # noqa: E731
        left = np.zeros(len(conds))
        for k, lab, col in (("empty", "empty shortlist", "#4c78a8"),
                            ("multi_miss", "shortlist missed", "#f58518"),
                            ("silent", "confident and wrong", "#e45756")):
            vals = np.array([get(c, k) for c in conds])
            ax.barh(y, vals, left=left, label=lab, color=col)
            left += vals
        ax.set_yticks(y)
        ax.set_yticklabels([c.replace("_", " ") for c in conds], fontsize=8)
        ax.set_xlabel("share of test images")
        ax.set_title(m, fontsize=10)
        ax.grid(axis="x", alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12),
                   ncol=3, frameon=False)
    fig.suptitle(f"Where the missing coverage goes ({tag} targets)", y=0.995)
    fig.tight_layout()
    fig.savefig(out_dir / f"compare_failure_modes_{tag}.png", dpi=150)
    print(f"wrote {out_dir / f'compare_failure_modes_{tag}.png'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--offset", type=float, default=5.0,
                    help="matched target = clean calibration accuracy minus this many points")
    ap.add_argument("--fixed", action="store_true",
                    help="use the config's single target for every model instead of matching")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    man = load_manifest(cfg["task"]["manifest"])
    offset = None if args.fixed else args.offset / 100.0

    setups, models = {}, []
    for m in (mm["name"] for mm in cfg["models"]):
        s = model_setup(cfg, m, man, offset)
        if s is not None:
            setups[m] = s
            models.append(m)
    if not models:
        print("No scored models found. Run scripts/run_grid.py first.")
        return 1

    mode = "the config's single target" if args.fixed else f"each model's accuracy minus {args.offset:g} points"
    print(f"Target rule: {mode}\n")
    for m in models:
        target, q, acc = setups[m]
        print(f"  {m:<24} clean calibration accuracy {acc:>6.1%} -> target {target:>6.1%}, cut-off {q:.4f}")

    rows_out = []
    for m in models:
        target, q, _ = setups[m]
        for cond in conditions_from_config(cfg):
            r = load_probs(cfg.get("probs_dir", "results/probs"), m, "test", cond)
            if r is None:
                continue
            P, g = arrays(r, man["splits"]["test"])
            vals = measure(P, g, q)
            rows_out.append({"model": m, "condition": cond, "target": target, "qhat": q,
                             "shortfall_points": 100 * (target - vals["coverage"]), **vals})

    for m in models:
        target = setups[m][0]
        print(f"\n{m}  (target {target:.1%})")
        hdr = (f"  {'condition':<20}{'cover':>8}{'short':>8}{'empty':>8}{'multi':>8}"
               f"{'silent':>8}{'wrong|commit':>14}{'errors flagged':>16}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for r in (x for x in rows_out if x["model"] == m):
            print(f"  {r['condition']:<20}{r['coverage']:>8.1%}{r['shortfall_points']:>7.1f}p"
                  f"{r['empty']:>8.1%}{r['multi']:>8.1%}{r['silent']:>8.1%}"
                  f"{r['wrong_given_commit']:>14.1%}{r['errors_flagged']:>16.1%}")

    print("\nshort = the model's own target minus the coverage it delivered.")
    print("wrong|commit = of the images given a single answer, the share where it was wrong.")
    print("errors flagged = of the model's own mistakes, the share conformal did not wave through")
    print("                 as a single confident answer. Higher is safer.")

    p = Path("results") / ("compare_models_fixed.csv" if args.fixed else "compare_models_matched.csv")
    p.parent.mkdir(exist_ok=True)
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nwrote {p}")

    if not args.no_figure and len(models) > 1:
        d = Path("results/figures")
        d.mkdir(parents=True, exist_ok=True)
        figures(rows_out, models, d, "fixed" if args.fixed else "matched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
