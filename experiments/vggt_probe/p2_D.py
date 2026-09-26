"""Probe 2 stage D (Q4): can VGGT depth fill sensor holes?  Probe 1 chain batches (M-V / I1: KF0-4, then [previous last] + next 5) are re-run
and per-frame VGGT depth/conf saved.  GT depth = MD-gt rendered at the GT pose in the original camera, warped with the same I1 map (nearest).
Inside the warped SAM2 mask (and where the GT render covers the pixel): sensor-valid (depth > 0.1 m) vs hole.  VGGT depth is converted to
original-camera z at the source pixel (ray-distance preserving), calibrated per frame on sensor-valid pixels by (i) scale (median ratio) and
(ii) affine (least squares), and compared with GT (mm), all pixels and conf top-50 % (within the frame's mask)."""
from __future__ import annotations
import argparse, json, time
import numpy as np, pandas as pd, torch
from p2_lib import *

SEQS = [("ho3d", s) for s in ["AP12", "MPM10", "MPM12", "SB11", "SM1"]] + [("ycb", "mustard0"), ("ycb", "tomato_soup_can_yalehand0")]
N_SAMPLE = 3000   # random pixels kept per frame and category for pooled statistics (seed fixed)


def chain_batches(ids):
    b = [ids[0:5]]; i = 5
    while i < len(ids): b.append([ids[i - 1]] + ids[i:i + 5]); i += 5
    return b


def run(ds, sq, vggt, S=518):
    out = OUT / "D" / sq; (out / "vggt").mkdir(parents=True, exist_ok=False); seq = SeqData(ds, sq); model = MeshModel(seq); K = seq.K
    rng = np.random.default_rng(0); done = set(); pools = {}; frows = []; t0 = time.time()
    for batch in chain_batches(seq.kf_ids):
        frames = [seq.frame(f) for f in batch]
        Rs = [L.centre_rotation(K, L.mask_centroid(m)) for _, _, m in frames]; fv = L.virtual_focal(K, [m for _, _, m in frames], Rs, S)
        Kv = np.array([[fv, 0, S / 2], [0, fv, S / 2], [0, 0, 1.0]]); w = [L.warp_I1(rgb, d, m, K, R, Kv, S) for (rgb, d, m), R in zip(frames, Rs)]
        r = vggt.infer(torch.from_numpy(np.stack([x[0] for x in w])).permute(0, 3, 1, 2).contiguous())
        rn_v = L.ray_norm(Kv, (S, S))
        for i, f in enumerate(batch):
            if f in done or seq.GT.get(f) is None: continue
            done.add(f); img, dsen, mv, (px, py) = w[i]
            np.savez_compressed(out / "vggt" / f"{f}.npz", depth=r["depth"][i].astype(np.float16), conf=r["conf"][i].astype(np.float16), mask_v=mv, Kv=Kv, R=Rs[i])
            _, dgt, _ = model.render(seq.GT[f]); dgt_v = cv2.remap(dgt, px, py, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            rn_o = np.linalg.norm(np.stack([px + 0.5, py + 0.5, np.ones_like(px)], -1).astype(np.float64) @ np.linalg.inv(K).T, axis=-1)
            zv = r["depth"][i].astype(np.float64) * rn_v / rn_o               # VGGT depth as original-camera z at the source pixel
            conf = r["conf"][i]; cov = mv & (dgt_v > 0); valid = cov & (dsen > 0.1); hole = cov & ~(dsen > 0.1)
            if valid.sum() < 200: continue
            thr = np.percentile(conf[mv], 50); top = conf >= thr
            s_i = float(np.median(dsen[valid] / zv[valid])); A = np.c_[zv[valid], np.ones(valid.sum())]; a_, b_ = np.linalg.lstsq(A, dsen[valid], rcond=None)[0]
            est = {"scale": s_i * zv, "affine": a_ * zv + b_}
            frows.append(dict(seq=sq, kf=f, mask_px=int(cov.sum()), hole_px=int(hole.sum()), hole_ratio=float(hole.sum() / max(cov.sum(), 1)), s_i=s_i, a=float(a_), b=float(b_),
                              sensor_err_med_mm=float(np.median(np.abs(dsen[valid] - dgt_v[valid])) * 1000)))
            def put(key, arr):
                if len(arr) == 0: return
                arr = arr if len(arr) <= N_SAMPLE else rng.choice(arr, N_SAMPLE, replace=False); pools.setdefault(key, []).append(arr)
            put(("sensor", "valid", "all"), np.abs(dsen[valid] - dgt_v[valid]) * 1000); put(("sensor", "valid", "top"), np.abs(dsen[valid & top] - dgt_v[valid & top]) * 1000)
            for m_ in ("scale", "affine"):
                for sel, sm in (("all", np.ones_like(top)), ("top", top)):
                    put((m_, "valid", sel), np.abs(est[m_][valid & sm] - dgt_v[valid & sm]) * 1000); put((m_, "hole", sel), np.abs(est[m_][hole & sm] - dgt_v[hole & sm]) * 1000)
    fr = pd.DataFrame(frows); fr.to_csv(out / "frames.csv", index=False)
    stats = []
    for key, lst in pools.items():
        a = np.concatenate(lst); stats.append(dict(seq=sq, source=key[0], pixels=key[1], subset=key[2], n=len(a), med_mm=float(np.median(a)), p90_mm=float(np.percentile(a, 90))))
    st = pd.DataFrame(stats); st.to_csv(out / "stats.csv", index=False)
    json.dump(dict(seq=sq, dataset=ds, frames=len(fr), hole_ratio_pooled=float(fr.hole_px.sum() / max(fr.mask_px.sum(), 1)), hole_ratio_median=float(fr.hole_ratio.median()) if len(fr) else None,
                   n_sample=N_SAMPLE, vggt_ckpt_sha256_16=vggt.ckpt_sha, seconds=time.time() - t0, finished=time.strftime("%F %T")), open(out / "manifest.json", "w"), indent=1)
    print(f"D {sq}: frames {len(fr)}, hole ratio {fr.hole_px.sum()/max(fr.mask_px.sum(),1):.3f}, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    from models import load_model as lv
    vggt = lv("M-V")
    for ds, sq in SEQS:
        if (OUT / "D" / sq / "manifest.json").exists(): print("skip", sq); continue
        for attempt in range(2):
            try: run(ds, sq, vggt); break
            except Exception as ex:
                import traceback, shutil; print(f"FAIL D {sq} attempt {attempt}: {ex}\n{traceback.format_exc()}", flush=True)
                d = OUT / "D" / sq
                if d.exists(): shutil.move(str(d), str(d) + f"_failed{attempt}")
