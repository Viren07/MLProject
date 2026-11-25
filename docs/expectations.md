# Expectations, written 2026-11-11 BEFORE any corrupted test-split result exists

Basis: the 100-image pilot (results/pilot_standard_levels.md) and the clean gate (clean test coverage
92.8%, accuracy 96.0%, 5.4% empty sets, 1.8% silent failures; re-split mean coverage 90.0%).
GPT-4o only. The condition list is frozen in configs/experiment.yaml.

1. Clean coverage is near 90% (met: 92.8%, within the expected sampling spread of about 2 points).
2. Within each family, coverage falls as the level rises.
3. Ordering of damage at level 100 (largest coverage loss first): gaussian_noise, pixelate, blur_gray,
   mixup, defocus_blur, cutmix, fisheye, grayscale. Grayscale alone stays within 5 points of clean.
4. Coverage at level 100 is below 80% for gaussian_noise, pixelate and blur_gray.
5. No prediction set has two or more options (the calibrated threshold is about 0.01, so only one option
   can reach it). The model hedges through empty sets.
6. Noise and pixelation lose coverage mostly through EMPTY sets (the model abstains); blur, blur_gray
   and cutmix lose it more through SILENT failures (confident wrong single answers).
7. The 50% levels are within about 4 points of clean coverage for every family.

Anything that contradicts these is reported as such.

---

# Outcomes, GPT-4o full grid (added 2026-11-11 after the run; the predictions above are unchanged)

Source: results/summary_gpt-4o.csv. Test n=500 per condition, qhat = 9.763e-03.

1. HELD. Clean coverage 92.8% (95% CI 90.2-94.8).
2. HELD. Noise 76.8 / 45.0 / 23.2 at 50 / 75 / 100; pixelate 82.8 / 55.0; defocus 82.4 / 76.8.
3. HELD as to order, but see the correction below. Coverage at 100: noise 23.2, pixelate 55.0,
   blur_gray 58.6, mixup 69.6, defocus 76.8, cutmix 81.0, fisheye 83.8, grayscale 92.4.
   Grayscale stays within 0.4 points of clean.
   CORRECTION (added after scripts/robustness.py, 2026-11-23): the Wilson intervals reported here
   cover only test-set sampling. Re-drawing the calibration set moves coverage by a further 5 to 12
   points for the corrupted conditions (results/robustness_gpt-4o.csv). Once that is included,
   cutmix:100 (71.0-82.8), defocus_blur:100 (70.0-79.0), fisheye:100 (80.0-86.0), pixelate:75 and
   defocus_blur:75 all overlap. Only the grouping survives: grayscale is indistinguishable from clean,
   gaussian_noise is clearly the worst at every level, and pixelate:100 and blur_gray:100 sit clearly
   below the middle group. The exact ordering within the middle group is not resolvable with n=500
   and a single calibration draw, and the paper must not claim it.
4. HELD. Noise 23.2%, pixelate 55.0%, blur_gray 58.6% at level 100.
5. HELD. Multi-option sets are 0.0% in every condition.
6. HALF WRONG. Noise and pixelate lose coverage mostly through empty sets, as predicted. But blur, blur_gray
   and cutmix do too (blur_gray 35.6% empty vs 5.8% silent; defocus 18.0 vs 5.2; cutmix 16.8 vs 2.2).
   Silent failures stay between 1.8% and 7.2% in every condition. The predicted blur/cutmix silent-failure
   pattern did not appear.
7. FAILED. Only noise has a 50% level in the grid: 76.8%, 16 points below clean. The 75% levels of
   pixelate and defocus are about 10 points below clean.

Derived (not pre-registered): error rate among answered items (singleton sets) rises from 1.9% on clean to
10.8% (noise 100), 11.6% (pixelate 100), 9.0% (blur_gray 100). Rows whose answer-letter probability mass is
below 0.95 (120 for noise 100) were already empty sets; forcing them to empty changes no coverage number.

---

# Status of each finding: pre-registered or exploratory (added 2026-11-23)

The seven predictions above were committed before any corrupted test image was scored, and they
all concern how corruption affects coverage. Nothing else in this project was pre-registered.
Everything below was found by looking at results that already existed. None of it is less true
for that, but the paper must label it, because a reader cannot otherwise tell which claims
survived a test and which were discovered.

## Pre-registered (tested against predictions 1 to 7)
- Clean coverage lands near the target.
- Coverage falls as corruption severity rises, within every family.
- Coverage at full strength falls below 80% for gaussian_noise, pixelate and blur_gray.
- No prediction set holds two or more options (true for GPT-4o; see below).
- The damage ordering at full strength. Predicted correctly, then WITHDRAWN as unresolvable once
  the calibration re-draw bands were computed. See the correction under prediction 3.
- Prediction 6 (blur and cutmix would fail through confident wrong answers) FAILED.
- Prediction 7 (mild levels within four points of clean) FAILED.

## Exploratory, found after seeing results
These were not predicted in advance and must be reported as exploratory:

1. The crossover rule. Whether a model abstains or widens its shortlist is set by whether the
   requested target sits below or above that model's own clean calibration accuracy. The
   transition sits at the accuracy in both models where it has been located (GPT-4o 94.8%,
   CLIP ViT-B/32 86.4%). Found while answering why the target was 90%, after the GPT-4o grid was
   complete. Mechanical rather than empirical in origin, but the confirmation is post hoc.
2. Error flagging as a model property. The share of a model's own mistakes that conformal does
   not wave through as a single confident answer differs sharply between models and is NOT
   predicted by accuracy: CLIP ViT-B/32 and ViT-L/14 are two accuracy points apart and differ by
   roughly 30 points in flagging, at every common target tested from 75% upward. Found after the
   CLIP runs.
3. The corruption ranking is model-specific. Fisheye costs GPT-4o 6 points of shortfall and CLIP
   ViT-B/32 40; heavy noise is the reverse. Both CLIP models are hurt most by pixelation while
   GPT-4o is hurt most by noise. Found when the second model landed.
4. Grayscale is free on every model tested so far. Consistent with prediction 3, which expected
   grayscale to be mildest, but the cross-model replication itself was not predicted.
5. Multi-option sets do appear, in models less accurate than the target. Prediction 5 said they
   never would, and that held for GPT-4o but is false in general. It is a special case of the
   crossover rule rather than an independent fact.

## Not a finding
CLIP ViT-L/14 was run to test whether a more accurate model from the same family would flip from
widening to abstaining at a 90% target. Its accuracy came in at 88.4%, still below 90%, so it did
not cross the boundary and the test was inconclusive. Recorded here so the attempt is not quietly
dropped.
