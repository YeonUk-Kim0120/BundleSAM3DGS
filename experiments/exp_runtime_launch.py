"""Run the official entry (run_custom.py / run_ho3d.py) with the MAIN code, overriding two config values at the moment the
pipeline writes its configs (2026-09-18 runtime measurement, main code untouched):
  RT_SYNC_MAX_DELAY=<int>  -> cfg_nerf['sync_max_delay'] (0 = strict sync, the shipped default; large = backend asynchronous
                              as described in the BundleSDF paper, appendix B)
  RT_SPDLOG=<int>          -> cfg_bundletrack['SPDLOG'] (1 = timestamped logs and poses only, no per-frame image dumps;
                              run_ho3d.py hard-codes 2)
usage: RT_SYNC_MAX_DELAY=1000000 RT_SPDLOG=1 python3 experiments/exp_runtime_launch.py run_custom.py <args ...>
"""
import os
import runpy
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)
if len(sys.argv) < 2:
    raise SystemExit(__doc__)
entry = os.path.join(REPO, sys.argv[1])
sys.argv = [entry] + sys.argv[2:]

import Utils  # noqa: E402  (the pipeline's shared ruamel YAML instance)

_sync = os.environ.get("RT_SYNC_MAX_DELAY")
_spdlog = os.environ.get("RT_SPDLOG")
_orig_dump = Utils.yaml.dump


def _dump(data, stream=None, **kw):
    if isinstance(data, dict):
        if _sync is not None and "sync_max_delay" in data:
            data["sync_max_delay"] = int(_sync); print(f"[runtime_launch] sync_max_delay -> {int(_sync)}", flush=True)
        if _spdlog is not None and "SPDLOG" in data:
            data["SPDLOG"] = int(_spdlog); print(f"[runtime_launch] SPDLOG -> {int(_spdlog)}", flush=True)
    return _orig_dump(data, stream, **kw)


Utils.yaml.dump = _dump
print(f"[runtime_launch] {os.path.basename(entry)}; RT_SYNC_MAX_DELAY={_sync!r} RT_SPDLOG={_spdlog!r}", flush=True)
runpy.run_path(entry, run_name="__main__")
