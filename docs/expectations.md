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
3. HELD (exact order). Coverage at 100: noise 23.2, pixelate 55.0, blur_gray 58.6, mixup 69.6,
   defocus 76.8, cutmix 81.0, fisheye 83.8, grayscale 92.4. Neighbouring conditions have overlapping CIs
   (pixelate/blur_gray; defocus/cutmix/fisheye), so only the grouping is solid, not every adjacent pair.
   Grayscale stays within 0.4 points of clean.
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
