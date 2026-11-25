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
from src.dataset import load_manifest  # noqa: E402
from src.grid import conditions_from_config, load_probs  # noqa: E402
from src.report import arrays, resolve_model  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--model", default=None, help="model name from the config; default is the first")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    model = resolve_model(cfg, args.model)["name"]
    out, alpha = cfg.get("probs_dir", "results/probs"), cfg["conformal"]["alpha"]
    man = load_manifest(cfg["task"]["manifest"])

    cal = load_probs(out, model, "calibration", "clean")
    if cal is None:
        print("No calibration probs. Run: python scripts/run_grid.py"); return 1
    Pc, gc = arrays(cal, man["splits"]["calibration"])
    qhat = calibrate(true_scores(Pc, gc), alpha)
    print(f"model {model} | alpha {alpha} (target {1 - alpha:.0%}) | calibration n={len(gc)} "
          f"acc={float((Pc.argmax(1) == gc).mean()):.1%} | qhat={qhat:.3e}\n")

    hdr = (f"{'condition':<18}{'n':>5}{'coverage':>10}{'95% CI':>16}{'acc':>7}{'flagged':>9}"
           f"{'empty':>7}{'multi':>7}{'silent fail':>12}")
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
        print(f"{cond:<18}{m['n']:>5}{m['coverage']:>10.1%}{ci:>16}{m['accuracy']:>7.1%}"
              f"{m['flag_rate']:>9.1%}{m['empty_rate']:>7.1%}{m['multi_rate']:>7.1%}{m['silent_failure']:>12.1%}")
    print("\nflagged = empty or multi-option set (the model does not commit). "
          "silent fail = committed to ONE answer that was wrong.")
    print("coverage = 1 - empty - multi misses - silent fail.")

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
