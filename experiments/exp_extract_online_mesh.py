"""Mesh (a) of EXP_BATCH_20260921: extract the meshes from the ONLINE final map without any further training
(gaussian_global.run_global_refine with steps = 0) into <run>/final/gs_online0.  Run through
experiments/online_variants/launch.py when the checkpoint was written by the experiment copy."""
import argparse, sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(1, str(REPO))
from gaussian_global import run_global_refine  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--run-dir", type=Path, required=True); ap.add_argument("--device", default="cuda:0")
ap.add_argument("--out-name", default="gs_online0"); a = ap.parse_args()
m = run_global_refine(a.run_dir, out_dir=a.run_dir / "final" / a.out_name, device=a.device, config={"steps": 0})
print(f"online0 extraction: gaussians {m['gaussians_after']}, {m['seconds_total']:.0f}s -> {a.run_dir / 'final' / a.out_name}")
