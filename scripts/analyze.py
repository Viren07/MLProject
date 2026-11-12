"""Calibrate on clean calibration probs, then report coverage etc. for every saved test condition.

    python scripts/analyze.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.conformal import calibrate, evaluate, true_scores  # noqa: E402
from src.dataset import LETTERS, load_manifest  # noqa: E402
from src.grid import conditions_from_config, load_probs  # noqa: E402


def arrays(rows, items):
    gold = {it["id"]: LETTERS.index(it["answer"]) for it in items}
    return np.array([r["probs"] for r in rows]), np.array([gold[r["id"]] for r in rows])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    model = next(m for m in cfg["models"] if m["provider"] == "openai")["name"]
    out, alpha = cfg.get("probs_dir", "results/probs"), cfg["conformal"]["alpha"]
    man = load_manifest(cfg["task"]["manifest"])

    cal = load_probs(out, model, "calibration", "clean")
    if cal is None:
        print("No calibration probs. Run: python scripts/run_grid.py"); return 1
    Pc, gc = arrays(cal, man["splits"]["calibration"])
    qhat = calibrate(true_scores(Pc, gc), alpha)
    print(f"model {model} | alpha {alpha} (target {1 - alpha:.0%}) | calibration n={len(gc)} "
          f"acc={float((Pc.argmax(1) == gc).mean()):.1%} | qhat={qhat:.3e}\n")

    hdr = f"{'condition':<18}{'n':>5}{'coverage':>10}{'95% CI':>16}{'size':>7}{'single':>8}{'empty':>7}{'acc':>7}{'1-set wrong':>13}"
    print(hdr); print("-" * len(hdr))
    rows_out = []
    for cond in conditions_from_config(cfg):
        rows = load_probs(out, model, "test", cond)
        if rows is None:
            continue
        P, g = arrays(rows, man["splits"]["test"])
        m = evaluate(P, g, qhat)
        rows_out.append({"model": model, "condition": cond, **m})
        ci = "[%.1f%%, %.1f%%]" % (100 * m["cov_lo"], 100 * m["cov_hi"])
        print(f"{cond:<18}{m['n']:>5}{m['coverage']:>10.1%}{ci:>16}"
              f"{m['mean_size']:>7.2f}{m['singleton_rate']:>8.1%}{m['empty_rate']:>7.1%}"
              f"{m['accuracy']:>7.1%}{m['singleton_wrong']:>13.1%}")

    clean = next((r for r in rows_out if r["condition"] == "clean"), None)
    if clean:
        ok = 0.86 <= clean["coverage"] <= 0.94
        print(f"\nWeek 1 gate (clean coverage within 86-94%): {'PASS' if ok else 'FAIL'} ({clean['coverage']:.1%})")
    if rows_out:
        p = Path("results") / f"summary_{model}.csv"
        p.parent.mkdir(exist_ok=True)
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0])); w.writeheader(); w.writerows(rows_out)
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
