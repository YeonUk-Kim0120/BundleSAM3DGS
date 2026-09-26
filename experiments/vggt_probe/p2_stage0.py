"""Probe 2 stage 0 (AP12, SM1).  a: MD-gt rendered at GT vs sensor depth / SAM2 mask (criteria: depth diff median < 10 mm, IoU median > 0.6).
b: same numbers for MD-prior and MD-map (reference).  c: synthetic recovery — the 'real' frame is render@GT (MD-gt), init = GT rotated 10° about
a random axis through the model centre, 20 KFs per sequence; E-V5x2 median must be < 2°."""
from __future__ import annotations
import argparse, json, sys, time
import numpy as np, cv2, pandas as pd
from p2_lib import *

SEQS = ["AP12", "SM1"]


def render_check(kind, seq, model, step=1):
    rows = []
    for k in seq.kf_ids[::step]:
        G = seq.GT.get(k)
        if G is None: continue
        rgb, depth, mask = seq.frame(k); r_rgb, r_d, r_m = model.render(G)
        v = mask & r_m & (depth > 0.1) & (r_d > 0)
        diff = (r_d[v] - depth[v]) * 1000
        rows.append(dict(model=kind, seq=seq.sq, kf=k, iou=iou(mask, r_m), absdiff_med_mm=float(np.median(np.abs(diff))) if v.sum() else np.nan,
                         signed_med_mm=float(np.median(diff)) if v.sum() else np.nan, n_px=int(v.sum()), occ_ratio=float(mask.sum() / max(r_m.sum(), 1))))
    return rows


def dump_overlay(seq, model, kind, k):
    G = seq.GT[k]; rgb, depth, mask = seq.frame(k); r_rgb, r_d, r_m = model.render(G)
    ov = rgb.copy(); ov[r_m] = (0.5 * ov[r_m] + 0.5 * r_rgb[r_m]).astype(np.uint8); edge = cv2.Canny(mask.astype(np.uint8) * 255, 50, 150) > 0; ov[edge] = (255, 0, 0)
    cv2.imwrite(str(LOG / f"stage0_overlay_{kind}_{seq.sq}_{k}.png"), cv2.cvtColor(np.concatenate([ov, r_rgb], 1), cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--part", required=True); a = ap.parse_args()
    if a.part in ("a", "b"):
        kinds = ["MD-gt"] if a.part == "a" else ["MD-prior", "MD-map"]; rows = []
        for sq in SEQS:
            seq = SeqData("ho3d", sq)
            for kind in kinds:
                model = load_model(kind, seq); rows += render_check(kind, seq, model); dump_overlay(seq, model, kind, seq.kf_ids[len(seq.kf_ids) // 2])
                del model; torch.cuda.empty_cache()
        df = pd.DataFrame(rows); df.to_csv(LOG / f"stage0{a.part}.csv", index=False)
        lines = []
        for (kind, sq), g in df.groupby(["model", "seq"]):
            lines.append(f"0-{a.part} {kind} {sq}: KF {len(g)}, |render - sensor| depth median of per-KF medians {g.absdiff_med_mm.median():.2f} mm (signed {g.signed_med_mm.median():+.2f} mm), "
                         f"IoU(silhouette, SAM2 mask) median {g.iou.median():.3f} (p10 {g.iou.quantile(.1):.3f}), mask/silhouette area median {g.occ_ratio.median():.3f}")
        if a.part == "a":
            ok = all(g.absdiff_med_mm.median() < 10 and g.iou.median() > 0.6 for _, g in df.groupby("seq")); lines.append("0-a " + ("PASS" if ok else "FAIL"))
        print("\n".join(lines)); open(LOG / f"stage0{a.part}.txt", "w").write("\n".join(lines) + "\n")
    else:
        from models import load_model as load_vggt
        vggt = load_vggt("M-V"); lo = Loftr(); rows = []
        for sq in SEQS:
            seq = SeqData("ho3d", sq); model = load_model("MD-gt", seq); rng = np.random.default_rng(0)
            ids = [k for k in seq.kf_ids if seq.GT.get(k) is not None]; pick = sorted(rng.choice(len(ids), 20, replace=False))
            for i in pick:
                k = ids[i]; G = seq.GT[k]; real = model.render(G); P0 = perturb(G, model.center, 10.0, rng)
                rr, dt = run_keyframe(vggt, lo, model, real, seq.K, P0, G, np.random.default_rng(1000 + i))
                for r in rr: r.update(seq=sq, kf=k); rows.append(r)
                print(sq, k, " ".join(f"{r['estimator']}={r['rot']:.2f}" for r in rr), f"{dt:.1f}s", flush=True)
        df = pd.DataFrame(rows); df.to_csv(LOG / "stage0c.csv", index=False); lines = []
        for est, g in df.groupby("estimator"):
            lines.append(f"0-c {est}: n {len(g)}, init rot median {g.rot0.median():.2f}°, est rot median {g.rot.median():.2f}° (p90 {g.rot.quantile(.9):.2f}), trans median {g.trans.median():.1f} mm (init {g.trans0.median():.1f}), success {100*g.success.mean():.0f} %")
        v = df[df.estimator == "E-V5x2"].rot.median(); ok = v < 2.0; lines.append(f"0-c criterion E-V5x2 median {v:.2f}° < 2° -> " + ("PASS" if ok else "FAIL"))
        print("\n".join(lines)); open(LOG / "stage0c.txt", "w").write("\n".join(lines) + "\n")
