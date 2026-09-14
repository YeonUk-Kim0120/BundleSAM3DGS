"""Diagnostic (2026-09-14): which extraction step drops the prior-completed back face at fine Poisson resolution?

Takes one global checkpoint, builds the surfel set exactly as ``extract_meshes`` does (opacity >= 0.1, radius <= 10 mm,
normalized distance <= 1.25, SUSPECT included), then runs Poisson at the given depths with / without the 5 % density
cut and with / without the largest-component selection, exports every variant and evaluates P1/P2/P3 with
``eval_mesh_cd.py``.  No training.  usage: see argparse.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from gaussian_global import DEFAULT_GLOBAL_CONFIG, gaussian_surfels, largest_component, load_online_checkpoint, poisson_mesh  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("ycb", "ho3d"), required=True)
    ap.add_argument("--video-dir", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--depths", type=int, nargs="+", default=[9, 8])
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    out = args.out_dir; out.mkdir(parents=True, exist_ok=True)
    ckpt = args.checkpoint or args.run_dir / "final" / "gs" / "checkpoint_global.pt"
    r = load_online_checkpoint(ckpt, device=args.device)
    cfg = DEFAULT_GLOBAL_CONFIG
    means, normals, colors = gaussian_surfels(r, cfg["opacity_min"], cfg["max_scale_mm"], cfg["include_suspect"], cfg["max_radius_norm"])
    ext = means.max(0) - means.min(0)
    print(f"surfels {len(means)}, extent {np.round(ext * 100, 1)} cm", flush=True)
    results = {"surfels": int(len(means)), "extent_cm": (ext * 100).tolist(), "variants": {}}
    for depth in args.depths:
        for q in (0.0, cfg["poisson_density_quantile"]):
            t0 = time.time(); m = poisson_mesh(means, normals, colors, depth, q); t1 = time.time() - t0
            comps = m.split(only_watertight=False)
            for comp in (False, True):
                name = f"d{depth}_q{q:g}_{'largest' if comp else 'all'}"
                mm = largest_component(m) if comp else m
                path = out / f"mesh_{name}.obj"; mm.export(path)
                cmd = ["/usr/bin/python3", str(REPO / "experiments" / "eval_mesh_cd.py"), "--dataset", args.dataset, "--video-dir", str(args.video_dir),
                       "--run-dir", str(args.run_dir), "--mesh", str(path), "--out-json", str(out / f"cd_{name}.json")]
                p = subprocess.run(cmd, capture_output=True, text=True)
                line = [l for l in p.stdout.splitlines() if "P1" in l]
                c = json.load(open(out / f"cd_{name}.json")) if (out / f"cd_{name}.json").exists() else {}
                p3 = c.get("P3_regions", {})
                results["variants"][name] = {"vertices": int(len(mm.vertices)), "components": int(len(comps)) if not comp else 1,
                                             "largest_component_share": float(max(len(cc.vertices) for cc in comps) / len(m.vertices)) if len(comps) else None,
                                             "poisson_seconds": t1, "P1": c.get("P1_original", {}).get("chamfer_cm"), "P2": c.get("P2_full_model", {}).get("chamfer_cm"),
                                             "unseen_cm": p3.get("gt_to_pred_unseen_cm"), "unseen_within5mm": p3.get("unseen_within_5mm_frac"), "seen_cm": p3.get("gt_to_pred_seen_cm")}
                print(name, results["variants"][name], line[-1] if line else p.stderr[-300:], flush=True)
    json.dump(results, open(out / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
