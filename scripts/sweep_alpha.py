"""Does the finding depend on the reliability target we asked for? Recompute it across targets.

    python scripts/sweep_alpha.py [--model gpt-4o]

Reads only saved probabilities, so it costs nothing and needs no API key. Writes a
long-format CSV and a two-panel figure: coverage against the requested target, and how
long the shortlists have to get to deliver it.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.conformal import calibrate, evaluate, prediction_sets, true_scores  # noqa: E402
from src.dataset import load_manifest  # noqa: E402
from src.grid import conditions_from_config, load_probs  # noqa: E402
from src.report import arrays, headline_conditions, resolve_model  # noqa: E402

DEFAULT_ALPHAS = [0.20, 0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02]


def figure(rows, model: str, conds: list[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    targets = sorted({r["target"] for r in rows})
    ax1.plot(targets, targets, ls="--", lw=1, color="0.6", label="what was promised")
    for cond in conds:
        pts = sorted([r for r in rows if r["condition"] == cond], key=lambda r: r["target"])
        if not pts:
            continue
        label = cond.replace(":100", "").replace("_", " ")
        style = {"lw": 2.2, "color": "black"} if cond == "clean" else {"lw": 1.4}
        ax1.plot([p["target"] for p in pts], [p["coverage"] for p in pts], marker="o",
                 ms=3, label=label, **style)
        ax2.plot([p["target"] for p in pts], [p["mean_size"] for p in pts], marker="o",
                 ms=3, label=label, **style)
    for ax, ylab in ((ax1, "coverage actually delivered"), (ax2, "mean shortlist length")):
        ax.set_xlabel("reliability requested")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
    ax1.set_title(f"{model}: what you ask for vs what you get")
    ax2.set_title("cost of asking for more")
    ax2.axhline(1, ls=":", lw=1, color="0.6")
    ax1.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--alphas", type=float, nargs="*", default=DEFAULT_ALPHAS)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    model = resolve_model(cfg, args.model)["name"]
    out = cfg.get("probs_dir", "results/probs")
    man = load_manifest(cfg["task"]["manifest"])

    cal = load_probs(out, model, "calibration", "clean")
    if cal is None:
        print(f"No clean calibration probs for {model}. Run scripts/run_grid.py first.")
        return 1
    Pc, gc = arrays(cal, man["splits"]["calibration"])
    cal_scores = true_scores(Pc, gc)

    test = {}
    for cond in conditions_from_config(cfg):
        rows = load_probs(out, model, "test", cond)
        if rows is not None:
            test[cond] = arrays(rows, man["splits"]["test"])
    if not test:
        print(f"No test probs for {model}.")
        return 1

    rows_out = []
    for a in sorted(args.alphas, reverse=True):
        q = calibrate(cal_scores, a)
        for cond, (P, g) in test.items():
            m = evaluate(P, g, q)
            rows_out.append({"model": model, "alpha": round(a, 4), "target": round(1 - a, 4),
                             "qhat": q, "min_prob_to_enter": max(0.0, 1.0 - q),
                             "condition": cond, "coverage": m["coverage"],
                             "empty_rate": m["empty_rate"], "multi_rate": m["multi_rate"],
                             "mean_size": m["mean_size"],
                             "max_size": int(prediction_sets(P, q).sum(axis=1).max()),
                             "silent_failure": m["silent_failure"]})

    hdr = (f"{'target':>8}{'cut-off':>10}{'min prob':>10}{'clean':>8}{'worst cond':>12}"
           f"{'longest':>9}{'2+ sets':>9}")
    print(f"model {model} | {len(test)} conditions | {len(man['splits']['calibration'])} "
          f"calibration photos\n")
    print(hdr)
    print("-" * len(hdr))
    for a in sorted(args.alphas, reverse=True):
        sub = [r for r in rows_out if r["alpha"] == round(a, 4)]
        clean = next(r for r in sub if r["condition"] == "clean")
        worst = min(sub, key=lambda r: r["coverage"])
        multi = sum(r["multi_rate"] * 500 for r in sub)
        print(f"{1 - a:>8.0%}{clean['qhat']:>10.4f}{clean['min_prob_to_enter']:>10.4f}"
              f"{clean['coverage']:>8.1%}{worst['coverage']:>12.1%}"
              f"{max(r['max_size'] for r in sub):>9}{int(multi):>9}")
    print("\ncut-off = the largest score an option may have and still make the shortlist.")
    print("min prob = how confident the model must be in an option for it to make the shortlist.")
    print("worst cond = the condition with the lowest coverage at that target.")
    print("2+ sets = shortlists holding two or more options, counted over all conditions.")

    p = Path("results") / f"alpha_sweep_{model}.csv"
    p.parent.mkdir(exist_ok=True)
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    print(f"wrote {p}")

    if not args.no_figure:
        figure(rows_out, model, headline_conditions(test), Path("results/figures") / f"alpha_sweep_{model}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
