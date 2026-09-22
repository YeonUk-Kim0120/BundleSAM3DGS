### 3.2a YCB: existing labels (each run's own last keyframes.yml) vs common labels (every 5th GT frame)

| seq | keyframes used before (BSDF / ours) | seen frac before | seen frac common | BSDF unseen cov % before → common | ours unseen cov % before → common | BSDF seen cm before → common | ours seen cm before → common |
|---|---|---|---|---|---|---|---|
| bleach0 | 33 / 35 | 0.844 / 0.844 | 0.847 | 43 → 43 | 63 → 64 | 0.907 → 0.898 | 0.416 → 0.418 |
| bleach_hard_00_03_chaitanya | 36 / 38 | 0.754 / 0.733 | 0.713 | 10 → 11 | 55 → 53 | 0.585 → 0.457 | 0.321 → 0.317 |
| cracker_box_reorient | 20 / 20 | 0.548 / 0.542 | 0.542 | 27 → 28 | 48 → 48 | 0.271 → 0.271 | 0.509 → 0.509 |
| cracker_box_yalehand0 | 38 / 38 | 0.885 / 0.885 | 0.887 | 45 → 45 | 14 → 13 | 0.876 → 0.876 | 0.368 → 0.368 |
| mustard0 | 37 / 38 | 0.657 / 0.657 | 0.663 | 34 → 33 | 96 → 96 | 0.281 → 0.282 | 0.197 → 0.197 |
| mustard_easy_00_02 | 32 / 29 | 0.635 / 0.614 | 0.637 | 7 → 7 | 99 → 99 | 0.372 → 0.368 | 0.149 → 0.150 |
| sugar_box1 | 43 / 43 | 0.998 / 0.995 | 0.998 | 28 → 27 | 89 → 73 | 0.986 → 0.986 | 0.228 → 0.228 |
| sugar_box_yalehand0 | 43 / 48 | 0.708 / 0.707 | 0.708 | 34 → 33 | 16 → 16 | 0.314 → 0.314 | 0.266 → 0.266 |
| tomato_soup_can_yalehand0 | 216 / 63 | 0.823 / 0.821 | 0.847 | 57 → 59 | 58 → 53 | 1.072 → 1.060 | 0.438 → 0.433 |

### 3.2b Seen-region accuracy (GT→pred over the SEEN region, cm): original BundleSDF (SAM2) vs ours (fulleval_20260912); HO3D from the existing JSONs, YCB from the common-label rescoring

| seq | BundleSDF seen | ours seen | ours − BSDF | BundleSDF seen ≤5 mm % | ours seen ≤5 mm % |
|---|---|---|---|---|---|
| AP10 | 0.361 | 0.372 | +0.011 | 84 | 80 |
| AP11 | 0.294 | 0.359 | +0.065 | 90 | 79 |
| AP12 | 0.276 | 0.246 | -0.031 | 87 | 91 |
| AP13 | 0.245 | 0.230 | -0.015 | 92 | 95 |
| AP14 | 0.345 | 0.219 | -0.126 | 82 | 94 |
| MPM10 | 0.384 | 0.190 | -0.195 | 89 | 98 |
| MPM11 | 0.391 | 0.190 | -0.202 | 78 | 95 |
| MPM12 | 0.327 | 0.146 | -0.181 | 90 | 99 |
| MPM13 | 0.421 | 0.288 | -0.133 | 65 | 85 |
| MPM14 | 0.340 | 0.131 | -0.210 | 84 | 100 |
| SB11 | 0.303 | 0.297 | -0.006 | 84 | 80 |
| SB13 | 0.315 | 0.269 | -0.047 | 84 | 90 |
| SM1 | 0.322 | 0.311 | -0.011 | 94 | 80 |
| **HO3D mean** | **0.333** | **0.250** | **-0.083** (ours better on 11/13) | | |
| bleach0 | 0.898 | 0.418 | -0.480 | 63 | 64 |
| bleach_hard_00_03_chaitanya | 0.457 | 0.317 | -0.140 | 80 | 82 |
| cracker_box_reorient | 0.271 | 0.509 | +0.239 | 92 | 52 |
| cracker_box_yalehand0 | 0.876 | 0.368 | -0.509 | 45 | 73 |
| mustard0 | 0.282 | 0.197 | -0.085 | 88 | 96 |
| mustard_easy_00_02 | 0.368 | 0.150 | -0.218 | 89 | 100 |
| sugar_box1 | 0.986 | 0.228 | -0.758 | 60 | 94 |
| sugar_box_yalehand0 | 0.314 | 0.266 | -0.048 | 86 | 82 |
| tomato_soup_can_yalehand0 | 1.060 | 0.433 | -0.627 | 45 | 68 |
| **YCB mean** | **0.612** | **0.321** | **-0.292** (ours better on 8/9) | | |
