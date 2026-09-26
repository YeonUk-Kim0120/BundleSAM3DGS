"""Probe 2 diagnostic (not a brief stage; recorded in the results doc): is the render-vs-real failure a domain gap or a code path issue?
For keyframes with small C1 error, VGGT 2-view relative rotation error for (a) [render@GT, real], (b) [real_prev_KF, real], (c) [render@GT_prev, render@GT],
(d) [render@GT, real] with the render's background and the real frame's appearance swapped: real RGB replaced by the render RGB inside the real mask
(isolates appearance from silhouette/occlusion).  Model MD-gt; sequences AP12, MPM12, SM1; 15 keyframes each (seed 0)."""
import numpy as np, pandas as pd
from p2_lib import *
from models import load_model as lv

vggt = lv("M-V"); rows = []
for sq in ["AP12", "MPM12", "SM1"]:
    seq = SeqData("ho3d", sq); m = MeshModel(seq); ids = [k for k in seq.kf_ids if seq.GT.get(k) is not None]
    rng = np.random.default_rng(0); pick = sorted(rng.choice(np.arange(1, len(ids)), 15, replace=False))
    for i in pick:
        k, kp = ids[i], ids[i - 1]; G, Gp = seq.GT[k], seq.GT[kp]; real = seq.frame(k); realp = seq.frame(kp); rG = m.render(G); rGp = m.render(Gp)
        swapped = (np.where(real[2][..., None], rG[0], real[0]).astype(np.uint8), real[1], real[2])   # render colours inside the real (occluded) mask
        def relerr(a, b, Pa, Pb):
            bb = vggt_batch(vggt, seq.K, [a, b], 518); re_ = np.linalg.inv(bb["c2w_b"][0]) @ bb["c2w_b"][1]; rg = np.linalg.inv(Pa) @ Pb
            return L.rot_deg(re_[:3, :3], rg[:3, :3])
        rows.append(dict(seq=sq, kf=k, swing_prev=L.rot_deg((np.linalg.inv(Gp) @ G)[:3, :3], np.eye(3)),
                         render_real=relerr(rG[:3], real, G, G), real_real=relerr(realp, real, Gp, G), render_render=relerr(rGp[:3], rG[:3], Gp, G),
                         render_realmaskRenderRGB=relerr(rG[:3], swapped, G, G), occ=float(real[2].sum() / max(rG[2].sum(), 1))))
        print(rows[-1], flush=True)
d = pd.DataFrame(rows); d.to_csv(LOG / "diag_domain.csv", index=False)
txt = d.groupby("seq")[["render_real", "real_real", "render_render", "render_realmaskRenderRGB", "swing_prev", "occ"]].median().round(2).to_string()
print(txt); open(LOG / "diag_domain.txt", "w").write("median relative-rotation error (deg) per sequence, 15 KFs each\n" + txt + "\n\noverall medians\n" + d[["render_real", "real_real", "render_render", "render_realmaskRenderRGB"]].median().round(2).to_string() + "\n")
