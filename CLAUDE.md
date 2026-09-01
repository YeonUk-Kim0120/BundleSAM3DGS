# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Non-negotiable working rules

These come from `AGENTS.md` and `PROJECT_HANDOFF.md` (the authoritative handoff document, in Korean) and the user enforces them strictly:

- **Do not edit any code without first asking the user which direction to take and getting explicit approval.** Before editing, explain the purpose, target files, implementation direction, and verification plan. Even one-line fixes and new tests require approval. Read-only investigation is always allowed. If scope changes after approval, stop and re-ask.
- **Change only the code that was approved.** Do not touch anything that is not essential to the approved change — no drive-by edits to nearby code, however small.
- **Never delete or overwrite experiment result directories, checkpoints, or logs** without explicit approval. Failed-run artifacts (e.g. the split `+inf` failure directories) are evidence and must be preserved. Output scripts intentionally use `exist_ok=False` — always pass a new unique output directory.
- Do not commit, amend, tag, or push. At meaningful milestones, recommend a commit and suggest a message.
- Keep changes minimal and focused; no drive-by refactoring or reformatting.

When documents conflict, trust in this order: current code + experiment JSON manifests/checkpoints → `PROJECT_HANDOFF.md` → git history (esp. `0113d63`) → `AGENTS.md` → `FIXES_SUMMARY.md`. **`FIXES_SUMMARY.md` is stale**: the rematch-regeneration and CUDACache intrinsics changes it describes were deliberately reverted to baseline in `0113d63` for tracking reproducibility. `BundleSAM3D` (extensionless file) is a personal scratch note, not a source of truth.

## Commands

Everything GPU-related runs inside a Docker container (`nvcr.io/nvidian/bundlesdf` base; GS work uses a container built from `docker/dockerfile.gsplat` with gsplat 1.5.3, PyTorch 2.6.0+cu124):

```bash
cd docker && docker build --network host -t nvcr.io/nvidian/bundlesdf .   # once
cd docker && bash run_container.sh                                        # start container
bash build.sh   # inside container, repo root: rebuilds mycuda + BundleTrack/build
```

`build.sh` wipes and rebuilds the native extensions — run it only after native C++/CUDA changes or for new-environment validation, not as a routine first step. Container default workdir is not the repo; `cd /home/kist/Desktop/BundleSAM3DGS` inside `docker exec`.

### Pipeline entry points

```bash
# Online tracking + reconstruction (global refinement runs automatically afterward)
python run_custom.py --mode run_video --video_dir datasets/YCBInEOAT/mustard0 \
  --out_folder outputs/mustard0 --mask_dir masks_sam2 --use_segmenter 0 --use_gui 0 --debug_level 2

# Re-run only global refinement from saved tracking results
python run_custom.py --mode global_refine --video_dir <data> --out_folder <out>

# HO3D + benchmark
python run_ho3d.py --video_dirs datasets/HO3D_v3/evaluation/SM1 --out_dir outputs/ho3d --mask_dir masks_SAM2
python benchmark_ho3d.py --video_dirs datasets/HO3D_v3/evaluation/SM1 --out_dir outputs/ho3d
```

### Tests

```bash
# CPU/lightweight Gaussian tests
/usr/bin/python3 -m unittest tests.test_gaussian_geometry tests.test_gaussian_budget_strategy

# Single test
/usr/bin/python3 -m unittest tests.test_gaussian_geometry.TestClassName.test_method

# Optional GPU integration (skips silently without gsplat+CUDA — verify tests actually ran)
RUN_GSPLAT_GPU_INTEGRATION=1 GAUSSIAN_TEST_DEVICE=cuda:0 \
  /usr/bin/python3 -m unittest tests.test_gaussian_gpu_integration
```

**Warning:** `tests/test_matching_flow.py` and `tests/test_matching_gpu_integration.py` encode pre-revert expectations (raw-match cache regenerating `_matches`) and may fail against the current deliberate baseline behavior. Do not "fix" the code to make them pass — ask the user whether baseline reproducibility or cache semantics takes priority.

## Architecture

This is BundleSDF (CVPR 2023: 6-DoF tracking + neural reconstruction of unknown objects from RGB-D) being extended with a 3D Gaussian Splatting backend ("BundleSAM3DGS").

### Original BundleSDF pipeline (working, preserved)

- `bundlesdf.py` — orchestrator. Runs tracking and reconstruction in separate processes communicating via a shared frame pool. Currently instantiates **only** `NerfRunner` (`nerf_runner.py`); there is no GS backend branch here yet.
- `BundleTrack/` — native C++17/CUDA tracking (frame registration, LoFTR correspondence, RANSAC, pose-graph optimization), built with CMake into `BundleTrack/build/`, exposed to Python via pybind11 bindings in `BundleTrack/pybind_interface/`. `BundleTrack/LoFTR/` is a bundled feature-matching subproject (weights at `BundleTrack/LoFTR/weights/outdoor_ds.ckpt`).
- `mycuda/` — PyTorch CUDA extensions (pip-installed editable by `build.sh`).
- Segmentation is **not** run online: readers (`YcbineoatReader`, `Ho3dReader` in `datasets/`) load precomputed SAM2 mask PNGs from `masks_sam2/` (custom) or `masks_SAM2/` (HO3D).
- Flow: reader → `BundleSdf`/native tracking → keyframes+poses → `NerfRunner` reconstruction → after `run_video`, `run_one_video_global_nerf()` runs global SDF refinement automatically.
- Config: `config.yml` (Python side), `BundleTrack/config_*.yml` (native side).

### Gaussian Splatting side (standalone, intentionally NOT wired into bundlesdf.py)

GS currently replays **fixed poses from completed tracking logs** offline; results validate the GS representation/data path, not online integration. Do not wire it into `bundlesdf.py` without first agreeing on interface design with the user.

- `gaussian_runner.py` — incremental Gaussian state: coordinates/normalization, RGB-D seeding, per-keyframe novelty append, train/render, checkpoints, PLY export.
- `gaussian_budget_strategy.py` — threshold-free exact-rank percentage topology (duplicate/split/prune, count-neutral relative to post-append N).
- `run_gaussian_incremental.py` — replays a saved tracking log (first 5 KF initialize, then append/update). Inputs: saved `color/`, `depth_filtered/`, `mask/`, `cam_K.txt`, `keyframes.yml`.
- `run_gaussian_{budget_ab,operator_ablation,long_followup,repeated_topology}.py` — controlled topology experiments.
- Configs: `config_gs.yml` (defaults, **still 1 cm voxel**); mustard research must explicitly use `config_gs_1mm*.yml` (1 mm voxel/novelty was adopted).

Key GS design decisions (details in `PROJECT_HANDOFF.md` §6):

- **Coordinate contract:** tracker/saved poses are metric OpenCV camera-to-world (`c2w_cv`). gsplat `viewmats` receive the inverse (`w2c_cv`) of the normalized CV c2w. No OpenGL axis flip before the GS renderer (unlike the SDF interface).
- **Append, not re-init:** each keyframe appends only novel points (KD-tree distance vs `observed_points_metric` history > novelty distance) to existing Gaussians. Known limitation: pruned Gaussians stay in observed history, blocking re-append (stale memory).
- Appending rebuilds the `ParameterDict` and resets all Adam/strategy state — a known, deliberately-unfixed limitation.
- Loss is masked RGB L1 + DSSIM only (no depth/silhouette loss) — experiments showed this can improve RGB while degrading geometry.

### Top open issue

Split topology fails with opacity `+inf`: gsplat `split(revised_opacity=True)` saturates `sigmoid(logit≈16.7)` to 1.0 in float32, and `logit(1.0) = +inf` trips the finite guard. A narrow logit-domain fix was discussed but is **not approved or implemented** — re-present the exact formula/tests to the user before touching it (`PROJECT_HANDOFF.md` §11).

## Data layout

Datasets, logs, weights, and checkpoints are not in git — they must be copied/mounted separately (paths and SHA256s in `PROJECT_HANDOFF.md` §9/§13). Custom data format: `rgb/` (PNG), `depth/` (mm uint16 PNG, same basenames), `masks_sam2/` (0=background), `cam_K.txt` (3×3). Experiment metrics from different suites/GPUs must not be compared directly — compare only against the concurrent control branch within the same experiment.
