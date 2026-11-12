# Pilot, standard severity scale (GPT-4o, 100 pilot images)

Clean accuracy 96%. Level = percent of each family's maximum usable strength.

| family | L25 | L50 | L75 | L100 |
|---|---|---|---|---|
| gaussian_noise | 95% | 87% | 72% | 50% |
| pixelate | 96% | 95% | 86% | 65% |
| defocus_blur | 96% | 95% | 92% | 87% |
| blur_gray | 95% | 91% | 81% | 76% |
| cutmix | 94% | 93% | 91% | 88% |
| mixup | 96% | 93% | 90% | 82% |
| fisheye | 95% | 95% | 91% | 91% |
| grayscale | - | - | - | 97% |

| family | level | accuracy | mean top prob | errors | confident errors |
|---|---|---|---|---|---|
| gaussian_noise | 25 | 95% | 0.984 | 5 | 3 |
| gaussian_noise | 50 | 87% | 0.962 | 13 | 7 |
| gaussian_noise | 75 | 72% | 0.869 | 28 | 5 |
| gaussian_noise | 100 | 50% | 0.775 | 50 | 10 |
| pixelate | 25 | 96% | 0.983 | 4 | 2 |
| pixelate | 50 | 95% | 0.988 | 5 | 2 |
| pixelate | 75 | 86% | 0.971 | 14 | 8 |
| pixelate | 100 | 65% | 0.875 | 35 | 13 |
| defocus_blur | 25 | 96% | 0.995 | 4 | 3 |
| defocus_blur | 50 | 95% | 0.981 | 5 | 4 |
| defocus_blur | 75 | 92% | 0.969 | 8 | 6 |
| defocus_blur | 100 | 87% | 0.952 | 13 | 8 |
| blur_gray | 25 | 95% | 0.985 | 5 | 3 |
| blur_gray | 50 | 91% | 0.973 | 9 | 5 |
| blur_gray | 75 | 81% | 0.941 | 19 | 8 |
| blur_gray | 100 | 76% | 0.917 | 24 | 11 |
| cutmix | 25 | 94% | 0.993 | 6 | 4 |
| cutmix | 50 | 93% | 0.981 | 7 | 3 |
| cutmix | 75 | 91% | 0.977 | 9 | 5 |
| cutmix | 100 | 88% | 0.962 | 12 | 5 |
| mixup | 25 | 96% | 0.988 | 4 | 2 |
| mixup | 50 | 93% | 0.986 | 7 | 3 |
| mixup | 75 | 90% | 0.970 | 10 | 4 |
| mixup | 100 | 82% | 0.929 | 18 | 5 |
| fisheye | 25 | 95% | 0.984 | 5 | 4 |
| fisheye | 50 | 95% | 0.985 | 5 | 2 |
| fisheye | 75 | 91% | 0.972 | 9 | 2 |
| fisheye | 100 | 91% | 0.978 | 9 | 5 |
| grayscale | 100 | 97% | 0.988 | 3 | 2 |
