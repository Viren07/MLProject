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
