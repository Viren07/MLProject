# Pilot screening of corruptions (GPT-4o, 100 pilot images, clean accuracy 96%)

Historical record. Severity numbers here (1 to 7) are the early screening levels. The pipeline now
uses the standard 25/50/75/100 scale (results/pilot_standard_levels.md). The dropped conditions
(patch, hfgrid, vignette, shuffle) are kept for reference in docs/dropped_corruptions.py.

Pilot split only. Used to choose which conditions enter the frozen grid. With 100 images a
difference of 3 to 4 points is within noise, and the largest pilot drops are likely inflated
by selection, so expect milder drops on the 500-image test split.

| condition | severity | accuracy | mean top prob | errors | confident errors (top p >= 0.9) |
|---|---|---|---|---|---|
| clean | 0 | 96% | 0.995 | 4 | 2 |
| gaussian_noise | 3 / 5 | 94% / 87% | 0.976 / 0.953 | 6 / 13 | 4 / 6 |
| defocus_blur | 2 / 5 | 94% / 92% | 0.989 / 0.967 | 6 / 8 | 3 / 5 |
| contrast | 3 / 5 | 96% / 94% | 0.989 / 0.978 | 4 / 6 | 3 / 4 |
| grayscale | 1 | 97% | 0.988 | 3 | 2 |
| blur_gray | 5 | 85% | 0.948 | 15 | 9 |
| vignette | 3 / 5 | 96% / 96% | 0.987 / 0.995 | 4 / 4 | 2 / 4 |
| vignette_gray | 3 | 96% | 0.985 | 4 | 1 |
| patch | 2 / 3 | 94% / 97% | 0.984 / 0.983 | 6 / 3 | 3 / 3 |
| cutmix | 2 / 3 | 94% / 88% | 0.988 / 0.975 | 6 / 12 | 4 / 6 |
| mixup | 2 / 3 | 93% / 90% | 0.986 / 0.970 | 7 / 10 | 3 / 4 |
| fisheye | 2 / 3 | 95% / 91% | 0.985 / 0.972 | 5 / 9 | 2 / 2 |
| shuffle | 2 / 4 | 97% / 93% | 0.978 / 0.970 | 3 / 7 | 2 / 2 |
| hfgrid | 3 | 95% | 0.989 | 5 | 3 |
| pixelate | 3 / 5 | 95% / 95% | 0.991 / 0.986 | 5 / 5 | 3 / 2 |

## Harsher levels (second screen, same 100 pilot images)

Levels 6 and 7 of noise, blur and pixelate are our own implementations of the imagecorruptions
formulas with harsher parameters. Levels up to 5 are the library's and are unchanged.

| condition | severity | accuracy | mean top prob | errors | confident errors (top p >= 0.9) |
|---|---|---|---|---|---|
| gaussian_noise | 6 / 7 | 75% / 49% | 0.888 / 0.801 | 25 / 51 | 9 / 13 |
| defocus_blur | 6 | 87% | 0.952 | 13 | 8 |
| pixelate | 6 / 7 | 86% / 65% | 0.971 / 0.875 | 14 / 35 | 8 / 13 |
| blur_gray | 6 | 76% | 0.917 | 24 | 11 |
| cutmix | 4 | 86% | 0.961 | 14 | 7 |
| mixup | 4 | 82% | 0.929 | 18 | 5 |
| fisheye | 4 | 91% | 0.978 | 9 | 5 |
| shuffle | 5 | 92% | 0.964 | 8 | 5 |
| patch | 4 | 93% | 0.985 | 7 | 5 |
| hfgrid | 5 | 95% | 0.976 | 5 | 0 |
| vignette | 7 | 96% | 0.992 | 4 | 3 |

Not run (judged unrecognizable by eye): blur_gray 7, defocus_blur 7, shuffle 6, patch 5, cutmix 5+, fisheye 5.

Dropped: fog (imagecorruptions fails on NumPy 2). Distractor mode: mixed 97% vs confusable 96% on the pilot.
