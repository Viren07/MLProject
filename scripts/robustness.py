"""How much of the result depends on which photos happened to be drawn?

    python scripts/robustness.py [--model gpt-4o] [--reps 2000]

Two checks, both from saved probabilities, so neither costs anything:

  1. Calibration re-draw. Resample the calibration photos, recompute the cut-off, and
     reapply it to the fixed test conditions. Shows how far coverage moves when the
     calibration set is the only thing that changes.
  2. Clean re-split. Pool the clean calibration and test photos, cut them at random into
     two halves of the original sizes, and recalibrate. Mean coverage should land on the
     target; this is the check that the whole procedure is wired up correctly.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.conformal import calibrate, true_scores  # noqa: E402
from src.dataset import load_manifest  # noqa: E402
from src.grid import conditions_from_config, load_probs  # noqa: E402
from src.report import arrays, resolve_model  # noqa: E402


def bootstrap_qhats(cal_scores: np.ndarray, alpha: float, reps: int, rng) -> np.ndarray:
    n = len(cal_scores)
    return np.array([calibrate(cal_scores[rng.integers(0, n, n)], alpha) for _ in range(reps)])


def coverage_at(test_scores: np.ndarray, qhats: np.ndarray) -> np.ndarray:
    """Coverage of a fixed test condition at each cut-off, via one sort and a search."""
    s = np.sort(test_scores)
    return np.searchsorted(s, qhats, side="right") / len(s)


def resplit(pooled_scores: np.ndarray, n_cal: int, alpha: float, reps: int, rng) -> np.ndarray:
    out = np.empty(reps)
    for b in range(reps):
        perm = rng.permutation(len(pooled_scores))
        q = calibrate(pooled_scores[perm[:n_cal]], alpha)
        out[b] = float((pooled_scores[perm[n_cal:]] <= q).mean())
    return out


def figure(rows, model: str, target: float, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda r: r["coverage"])
    y = np.arange(len(rows))
    point = [r["coverage"] for r in rows]
    lo = [r["coverage"] - r["cov_p05"] for r in rows]
    hi = [r["cov_p95"] - r["coverage"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(rows) + 1.8))
    ax.errorbar(point, y, xerr=[lo, hi], fmt="o", ms=5, capsize=3, lw=1.2, color="#1f4e79")
    ax.axvline(target, ls="--", lw=1, color="0.5")
    ax.set_ylim(-1.3, len(rows) - 0.4)
    ax.text(target, -1.2, f"promised {target:.0%}", fontsize=8, color="0.45", ha="center", va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels([r["condition"].replace("_", " ") for r in rows], fontsize=8)
    ax.set_xlabel("coverage (bars: 5th to 95th percentile over calibration re-draws)")
    ax.set_title(f"{model}: sensitivity to the calibration draw")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    model = resolve_model(cfg, args.model)["name"]
    out, alpha = cfg.get("probs_dir", "results/probs"), cfg["conformal"]["alpha"]
    man = load_manifest(cfg["task"]["manifest"])
    rng = np.random.default_rng(args.seed)

    cal = load_probs(out, model, "calibration", "clean")
    if cal is None:
        print(f"No clean calibration probs for {model}. Run scripts/run_grid.py first.")
        return 1
    Pc, gc = arrays(cal, man["splits"]["calibration"])
    cal_scores = true_scores(Pc, gc)
    real_q = calibrate(cal_scores, alpha)

    qs = bootstrap_qhats(cal_scores, alpha, args.reps, rng)
    print(f"model {model} | target {1 - alpha:.0%} | {args.reps} re-draws of "
          f"{len(cal_scores)} calibration photos\n")
    print(f"cut-off on the real calibration set: {real_q:.4f}")
    print(f"cut-off across re-draws: median {np.median(qs):.4f}, "
          f"5th {np.percentile(qs, 5):.4f}, 95th {np.percentile(qs, 95):.4f}\n")

    rows_out = []
    hdr = f"{'condition':<20}{'coverage':>10}{'5th':>8}{'95th':>8}{'spread':>9}"
    print(hdr)
    print("-" * len(hdr))
    for cond in conditions_from_config(cfg):
        rows = load_probs(out, model, "test", cond)
        if rows is None:
            continue
        P, g = arrays(rows, man["splits"]["test"])
        ts = true_scores(P, g)
        point = float((ts <= real_q).mean())
        cov = coverage_at(ts, qs)
        p05, p95 = float(np.percentile(cov, 5)), float(np.percentile(cov, 95))
        rows_out.append({"model": model, "condition": cond, "coverage": point,
                         "cov_p05": p05, "cov_p95": p95, "cov_median": float(np.median(cov)),
                         "spread_points": 100 * (p95 - p05)})
        print(f"{cond:<20}{point:>10.1%}{p05:>8.1%}{p95:>8.1%}{100 * (p95 - p05):>8.1f}p")
    print("\nspread = width of the 5th-to-95th band, in percentage points. A wide band means the "
          "number would have come out differently with another 500 calibration photos.")

    test_clean = load_probs(out, model, "test", "clean")
    if test_clean is not None:
        Pt, gt = arrays(test_clean, man["splits"]["test"])
        pooled = np.concatenate([cal_scores, true_scores(Pt, gt)])
        r = resplit(pooled, len(cal_scores), alpha, min(args.reps, 1000), rng)
        print(f"\nClean re-split check ({min(args.reps, 1000)} random halves of "
              f"{len(pooled)} clean photos):")
        print(f"  mean coverage {r.mean():.1%}, sd {r.std(ddof=1):.1%}, "
              f"5th {np.percentile(r, 5):.1%}, 95th {np.percentile(r, 95):.1%}")
        print(f"  target is {1 - alpha:.0%}; the mean sitting on it is the correctness check.")

    p = Path("results") / f"robustness_{model}.csv"
    p.parent.mkdir(exist_ok=True)
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nwrote {p}")

    if not args.no_figure:
        figure(rows_out, model, 1 - alpha, Path("results/figures") / f"robustness_{model}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
