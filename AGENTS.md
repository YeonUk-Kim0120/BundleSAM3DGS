# Repository Guidelines

## Project Structure & Module Organization

The root Python modules implement the BundleSDF pipeline: `bundlesdf.py` coordinates tracking and reconstruction, while `nerf_runner.py`, `run_custom.py`, `run_ho3d.py`, and `benchmark_ho3d.py` provide core logic and entry points. Native C++17/CUDA tracking code lives in `BundleTrack/src/`, with pybind11 bindings in `BundleTrack/pybind_interface/`. PyTorch CUDA extensions are built from `mycuda/`. `BundleTrack/LoFTR/` is a bundled feature-matching subproject with its own configs and scripts. Runtime configuration is in `config.yml` and `BundleTrack/config_*.yml`; README media belongs in `media/`.

## Build, Test, and Development Commands

- `(cd docker && docker build --network host -t nvcr.io/nvidian/bundlesdf -f dockerfile .)` builds the GPU development image.
- `(cd docker && bash run_container.sh)` starts the NVIDIA-enabled development container.
- `bash build.sh` runs inside that container from the repository root; it installs `mycuda` editable and rebuilds `BundleTrack/build/` with CMake and Make.
- `python run_custom.py --mode run_video --video_dir <data> --out_folder <output>` runs a custom RGB-D sequence. Follow with `--mode global_refine` for mesh refinement.
- `python run_ho3d.py --video_dirs <sequence> --out_dir <output>` runs HO3D; validate results with the matching `python benchmark_ho3d.py ...` command.

## Coding Style & Naming Conventions

Use four spaces for new Python code and `snake_case` for modules, functions, and variables; use `PascalCase` for classes. When editing legacy two-space Python blocks, match the surrounding code and avoid drive-by reformatting. C++/CUDA uses two-space indentation, `PascalCase` types, `camelCase` methods, and underscore-prefixed fields. Preserve existing lowercase/snake-case YAML keys. No repository-wide formatter or linter is configured.

## Testing Guidelines

There is no top-level unit-test suite, CI workflow, or coverage threshold. For native or CUDA changes, require a clean `bash build.sh`. Smoke-test pipeline changes on a small RGB-D sequence (use `--stride` when useful), then run `benchmark_ho3d.py` for evaluation-sensitive work. New lightweight tests should use `tests/test_<feature>.py` and document required GPU, weights, or datasets.

## Commit & Pull Request Guidelines

History favors short, focused subjects such as `fix minor bug` and `Update CMakeLists.txt`; use an imperative summary and append issue references when applicable. PRs should explain behavior or configuration changes, list verification commands, link related issues, and include screenshots, sample renders, or benchmark deltas for visual or accuracy changes.

## Data and Configuration

Do not commit datasets, generated outputs, compiled extensions, or pretrained weights. Keep machine-specific paths out of source changes; pass them through CLI arguments or local configuration. Expected weight locations are documented in `readme.md`.

## Collaboration and Change Control

- Before modifying code, explain the intended change, the files likely to be affected, the implementation approach, and the verification plan. Wait for explicit user approval before making any code edit. Read-only investigation is allowed before approval.
- If the required scope changes materially after approval, stop and request approval again before continuing.
- Keep changes focused and minimal. Do not modify, reformat, refactor, or clean up unrelated code.
- After implementation, inspect the diff and run checks or tests appropriate to the change. Clearly report anything that could not be verified.
- Do not create commits, amend commits, tag releases, or push branches. When the project reaches a meaningful research or implementation milestone, recommend that the user create a commit and provide a concise summary and suggested commit message.

## Experiment Results and Artifact Management

- Experiments may produce large result directories. Preserve results that represent meaningful progress, such as state-of-the-art performance, substantial metric changes, important qualitative differences, or research milestones.
- For results that are not meaningful enough to preserve in full, first record the essential reproducibility information: purpose, command, configuration, dataset or sequence, relevant code revision, key metrics, and conclusion.
- Never delete an experiment directory or result artifact without explicit user approval.
- Before requesting deletion approval, identify the exact directories or files, summarize the recorded results, and explain why the artifacts appear safe to remove.
