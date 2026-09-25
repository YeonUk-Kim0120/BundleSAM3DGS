"""Collect every cell manifest into logs/exp_vggt_probe1/manifests/index.json (+ per-model checkpoint hashes)."""
import glob, json, os
from pathlib import Path
ROOT = Path("/home/kist/Desktop/BundleSAM3DGS/outputs/exp_vggt_probe1"); OUT = Path("/home/kist/Desktop/BundleSAM3DGS/logs/exp_vggt_probe1/manifests"); OUT.mkdir(exist_ok=True)
idx = []
for m in sorted(glob.glob(str(ROOT / "M-*/I*/*/*/*/manifest.json"))):
    j = json.load(open(m)); j["cell_dir"] = os.path.dirname(m); j.pop("kf_ids", None); idx.append(j)
json.dump(idx, open(OUT / "index.json", "w"), indent=1)
ck = {}
for j in idx: ck.setdefault(j["model"], set()).add((j["ckpt"], j["ckpt_sha256_16"]))
json.dump({k: sorted(v) for k, v in ck.items()}, open(OUT / "checkpoints.json", "w"), indent=1)
print(len(idx), "cells;", {k: sorted(v) for k, v in ck.items()})
