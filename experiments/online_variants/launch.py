"""Run the official pipeline entry (run_custom.py / run_ho3d.py) with the EXPERIMENT COPIES of gaussian_runner and
prior_lifecycle (this directory) injected ahead of the repo modules.  The main code is not modified; the GS child
process (multiprocessing 'spawn') inherits sys.path and the environment, so it imports the copies too.

usage: python experiments/online_variants/launch.py run_custom.py <run_custom args ...>
       GS_ONLINE_VARIANTS=fusion python experiments/online_variants/launch.py run_ho3d.py <run_ho3d args ...>
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
if REPO not in sys.path:
    sys.path.insert(1, REPO)
os.chdir(REPO)
if len(sys.argv) < 2:
    raise SystemExit(__doc__)
entry = os.path.join(REPO, sys.argv[1])
sys.argv = [entry] + sys.argv[2:]
import gaussian_runner  # noqa: E402  (must resolve to the copy)

assert os.path.dirname(os.path.abspath(gaussian_runner.__file__)) == HERE, gaussian_runner.__file__
print(f"[online_variants] using {gaussian_runner.__file__}; GS_ONLINE_VARIANTS={os.environ.get('GS_ONLINE_VARIANTS', '')!r}", flush=True)
runpy.run_path(entry, run_name="__main__")
