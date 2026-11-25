"""Score images with the VLM under each condition and save the probabilities.

    python scripts/run_grid.py                     # the Week 1 gate: calibration + test, CLEAN only
    python scripts/run_grid.py --all               # clean + every corruption/severity, on test
    python scripts/run_grid.py --all --model clip-vit-base-patch32   # same grid, another model

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
from src.report import resolve_model  # noqa: E402
from src.scoring import GPTScorer, OutOfCredits  # noqa: E402


def build_scorer(mcfg: dict, cfg: dict):
    """One scorer per provider. Every scorer exposes .score(image, options) -> {probs, letter_mass}."""
    provider = mcfg["provider"]
    if provider == "openai":
        return GPTScorer(model=mcfg["name"], cache_dir=cfg["cache_dir"],
                         detail=mcfg.get("image_detail", "high"),
                         jpeg_quality=mcfg.get("jpeg_quality", 95))
    if provider == "clip":
        from src.clip_scorer import CLIPScorer  # imports torch, so only on this path
        return CLIPScorer(model=mcfg["name"], hf_id=mcfg["hf_id"], cache_dir=cfg["cache_dir"],
                          jpeg_quality=mcfg.get("jpeg_quality", 95),
                          prompt_template=mcfg.get("prompt_template", "a photo of a {}"))
    if provider == "gemini":
        from src.gemini_scorer import GeminiScorer
        return GeminiScorer(model=mcfg["name"], cache_dir=cfg["cache_dir"],
                            jpeg_quality=mcfg.get("jpeg_quality", 95),
                            top_logprobs=mcfg.get("top_logprobs", 20))
    if provider == "local":
        from src.local_vlm import LocalVLMScorer  # imports torch, so only on this path
        return LocalVLMScorer(model=mcfg["name"], hf_id=mcfg["hf_id"], cache_dir=cfg["cache_dir"],
                              jpeg_quality=mcfg.get("jpeg_quality", 95),
                              load_in_4bit=mcfg.get("load_in_4bit", False),
                              dtype=mcfg.get("dtype", "bfloat16"),
                              n_options=cfg["task"]["n_options"],
                              max_image_side=mcfg.get("max_image_side"),
                              processor_kwargs=mcfg.get("processor_kwargs"))
    raise SystemExit(f"unknown provider {provider!r} for model {mcfg['name']!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--all", action="store_true", help="also score every corruption on the test split")
    ap.add_argument("--model", default=None, help="model name from the config; default is the first")
    ap.add_argument("--workers", type=int, default=None,
                    help="default: 6 for API models, 1 for local ones")
    args = ap.parse_args()
    load_dotenv()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    mcfg = resolve_model(cfg, args.model)
    man = load_manifest(cfg["task"]["manifest"]); verify_manifest(man)
    root = find_images_root(cfg["task"]["data_dir"])
    scorer = build_scorer(mcfg, cfg)
    workers = args.workers if args.workers is not None else (6 if mcfg["provider"] in {"openai", "gemini"} else 1)
    where = getattr(scorer, "device", None)
    extra = f", on {where}" + (" [4-bit]" if getattr(scorer, "four_bit", False) else "") if where else ""
    print(f"model {mcfg['name']} ({mcfg['provider']}), {workers} worker(s){extra}")
    out = cfg.get("probs_dir", "results/probs")
    partners = PartnerPool(root, {it["image"] for sp in man["splits"].values() for it in sp})

    jobs = [("calibration", "clean"), ("test", "clean")]
    if args.all:
        jobs += [("test", c) for c in conditions_from_config(cfg) if c != "clean"]
    for split, cond in jobs:
        try:
            rows = run_condition(man["splits"][split], root, scorer, cond, seed=cfg["seed"],
                                 workers=workers, desc=f"{split} {cond}", partners=partners)
        except OutOfCredits as e:
            print(f"\nSTOPPED at {split} {cond}: {e}")
            return 2
        low = sum(r["letter_mass"] < 0.95 for r in rows)
        path = save_probs(out, mcfg["name"], split, cond, rows)
        print(f"saved {path}  ({len(rows)} rows, {low} with letter mass < 0.95)")
    tail = "" if mcfg is cfg["models"][0] else f" --model {mcfg['name']}"
    print(f"\nNext: python scripts/analyze.py{tail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
