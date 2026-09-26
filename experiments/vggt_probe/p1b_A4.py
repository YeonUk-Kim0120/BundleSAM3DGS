"""A4: Spearman correlations of M-V/I1 B-ref pair errors with swing, roll, min mask area, mask area ratio, mean conf, batch T3 —
pooled and within each swing bin."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from p1b_common import *

df = pd.read_csv(P1B / "A" / "all_pairs.csv"); d = df[(df.model == "M-V") & (df.input == "I1") & (df.batch_mode != "chain")].copy()
d["min_mask"] = np.minimum(d.mask_i, d.mask_j); d["mask_ratio"] = np.minimum(d.mask_i, d.mask_j) / np.maximum(d.mask_i, d.mask_j); d["conf_mean"] = (d.conf_i + d.conf_j) / 2
d["sb"] = [bin_index(x, SWING_BINS) for x in d.swing]
VARS = ["swing", "roll", "min_mask", "mask_ratio", "conf_mean", "T3"]


def rho(x, y):
    ok = np.isfinite(x) & np.isfinite(y); return spearmanr(x[ok], y[ok]).correlation if ok.sum() > 10 else np.nan


doc = ["# A4 상관 — M-V/I1 B-ref 쌍 회전 오차 vs 변수 (Spearman ρ)", ""]
for ds in ["all", "ho3d", "ycb"]:
    dd = d if ds == "all" else d[d.ds == ds]
    doc += [f"## {ds} (n={len(dd)})", "", "| 변수 | 전체 ρ | " + " | ".join(f"ρ in swing {l}" for l in SWING_LABELS) + " |", "|---|---|" + "---|" * len(SWING_LABELS)]
    for v in VARS:
        cells = [f"{rho(dd[v].values, dd.vggt_rot.values):.2f}"]
        for sb in range(len(SWING_LABELS)):
            g = dd[dd.sb == sb]; r = rho(g[v].values, g.vggt_rot.values) if len(g) > 10 else np.nan
            cells.append(f"{r:.2f} (n={len(g)})" if np.isfinite(r) else "–")
        doc.append(f"| {v} | " + " | ".join(cells) + " |")
    doc.append("")
open(LOG / "table_A4.md", "w").write("\n".join(doc)); print("\n".join(doc))
