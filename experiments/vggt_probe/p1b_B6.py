"""B6 pair benchmark: LoFTR+depth (BundleTrack-style preprocessing + 3D-3D RANSAC/Procrustes) vs VGGT 2-view (I1) vs VGGT 2-view + roll
on keyframe pairs sampled per GT swing bin (<= 60 per bin per sequence, seed 0).  Writes outputs/exp_vggt_probe1b/B6/pairs_bench.csv and table L."""
from __future__ import annotations
import csv, json, random, sys, time, os
from pathlib import Path
import numpy as np, cv2, torch, yaml
sys.path.insert(0, "/home/kist/Desktop/BundleSAM3DGS")
import probe_lib as L
from models import load_model
from p1b_common import *
from p1b_B_lib import roll_psi, Rz, warp_I1_bg

CFG = yaml.safe_load(open("/home/kist/Desktop/BundleSAM3DGS/BundleTrack/config_ho3d.yml"))
OUT_SIDE = int(CFG["feature_corres"]["resize"]); MIN_MATCH = int(CFG["feature_corres"]["min_match_with_ref"]); RANSAC = CFG["ransac"]
SEQS = ["AP12", "MPM10", "MPM12", "SB11", "SM1"]; PER_BIN = 60; SEED = 0


# ------------------------------------------------------------- LoFTR preprocessing (FeatureManager.cpp processImagePair)
def roi_of(mask):
    ys, xs = np.nonzero(mask); return np.array([xs.min(), xs.max(), ys.min(), ys.max()])   # umin, umax, vmin, vmax


def rotate_tf(H, W, rot):
    tf = np.eye(3); tf[0, 2] -= W / 2; tf[1, 2] -= H / 2
    R = np.eye(3); R[:2, :2] = [[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]]; tf = R @ tf
    corners = np.array([[0, 0, 1], [W, 0, 1], [0, H, 1], [W, H, 1]], float) @ tf.T
    T = np.eye(3); T[0, 2] = -corners[:, 0].min(); T[1, 2] = -corners[:, 1].min(); return T @ tf


def process_pair(grayA, grayB, roiA, roiB, c2wA, c2wB, out_size=OUT_SIDE, margin=10):
    """A = later frame (query), B = earlier.  Rotate B into A by the camera-z component of R_A R_B^-1 (w2c rotations), crop each ROI with a
    10 px margin, scale both by the common max_dim, resize to out_size square (gray).  Returns imgA, imgB, tfA, tfB (3x3 forward)."""
    H, W = grayA.shape; RA = c2wA[:3, :3].T; RB = c2wB[:3, :3].T; R_BA = RA @ np.linalg.inv(RB)
    ang = np.arccos(np.clip((np.trace(R_BA) - 1) / 2, -1, 1)); ax = np.array([R_BA[2, 1] - R_BA[1, 2], R_BA[0, 2] - R_BA[2, 0], R_BA[1, 0] - R_BA[0, 1]])
    ax = ax / np.linalg.norm(ax) if np.linalg.norm(ax) > 1e-9 else np.array([0, 0, 1.0]); rot = float(ang * ax[2])
    tfA = np.eye(3); tfB = rotate_tf(H, W, rot)
    corners = np.array([[roiB[0], roiB[2], 1], [roiB[0], roiB[3], 1], [roiB[1], roiB[2], 1], [roiB[1], roiB[3], 1]], float) @ tfB.T
    uvB_min = corners[:, :2].min(0); uvB_max = corners[:, :2].max(0)
    t = np.eye(3); t[0, 2] = -roiA[0] + margin; t[1, 2] = -roiA[2] + margin; tfA = t @ tfA
    t = np.eye(3); t[0, 2] = -uvB_min[0] + margin; t[1, 2] = -uvB_min[1] + margin; tfB = t @ tfB
    WA = roiA[1] - roiA[0] + 2 * margin; HA = roiA[3] - roiA[2] + 2 * margin; WB = uvB_max[0] - uvB_min[0] + 2 * margin; HB = uvB_max[1] - uvB_min[1] + 2 * margin
    max_dim = max(WA, HA, WB, HB)
    t = np.eye(3); t[:2, :2] *= max_dim / max(WA, HA); tfA = t @ tfA
    t = np.eye(3); t[:2, :2] *= max_dim / max(WB, HB); tfB = t @ tfB
    t = np.eye(3); t[:2, :2] *= out_size / max_dim; tfA = t @ tfA; tfB = t @ tfB
    imgA = cv2.warpPerspective(grayA, tfA, (out_size, out_size)); imgB = cv2.warpPerspective(grayB, tfB, (out_size, out_size))
    return imgA, imgB, tfA, tfB


def backproject(uv, depth, K):
    z = depth[uv[:, 1], uv[:, 0]]; x = (uv[:, 0] + 0.0 - K[0, 2]) / K[0, 0] * z; y = (uv[:, 1] - K[1, 2]) / K[1, 1] * z; return np.stack([x, y, z], -1)


def kabsch(P, Q):
    """R, t with Q ≈ R P + t (P: source, Q: target), batched over leading dims."""
    mp, mq = P.mean(-2, keepdims=True), Q.mean(-2, keepdims=True); H = np.swapaxes(P - mp, -1, -2) @ (Q - mq)
    U, _, Vt = np.linalg.svd(H); d = np.sign(np.linalg.det(np.swapaxes(Vt, -1, -2) @ np.swapaxes(U, -1, -2)))
    D = np.tile(np.eye(3), H.shape[:-2] + (1, 1)); D[..., 2, 2] = d; R = np.swapaxes(Vt, -1, -2) @ D @ np.swapaxes(U, -1, -2)
    t = mq.squeeze(-2) - (R @ mp.squeeze(-2)[..., None]).squeeze(-1); return R, t


def ransac_procrustes(PA, PB, rng, inlier_dist=RANSAC["inlier_dist"], max_iter=RANSAC["max_iter"], n_sample=RANSAC["num_sample"]):
    """3-point Procrustes RANSAC (points of B mapped into A: PA ≈ R PB + t).  Returns (R, t, inlier mask)."""
    n = len(PA)
    if n < n_sample: return None, None, np.zeros(n, bool)
    idx = np.array([rng.choice(n, n_sample, replace=False) for _ in range(max_iter)]); R, t = kabsch(PB[idx], PA[idx])
    pred = np.einsum("kij,nj->kni", R, PB) + t[:, None, :]; d = np.linalg.norm(pred - PA[None], axis=-1); inl = d < inlier_dist; cnt = inl.sum(1); best = int(np.argmax(cnt))
    m = inl[best]
    if m.sum() >= n_sample:
        Rf, tf = kabsch(PB[m][None], PA[m][None]); Rf, tf = Rf[0], tf[0]; m = np.linalg.norm((Rf @ PB.T).T + tf - PA, axis=1) < inlier_dist
        if m.sum() >= n_sample: Rf, tf = kabsch(PB[m][None], PA[m][None]); Rf, tf = Rf[0], tf[0]
        return Rf, tf, m
    return R[best], t[best], m


def main():
    from loftr_wrapper import LoftrRunner
    out = P1B / "B6"; out.mkdir(exist_ok=True); rng = np.random.default_rng(SEED); random.seed(SEED)
    loftr = LoftrRunner(); vggt = load_model("M-V"); SZ = vggt.size
    rows = []; t0 = time.time()
    for sq in SEQS:
        ds = "ho3d"; S = load_seq(ds, sq); seq = L.Sequence(ds, sq); K = S.K; ids = [k for k in S.kf_ids if S.GT.get(k) is not None]
        # sample pairs per swing bin
        cand = {b: [] for b in range(len(SWING_LABELS))}
        for a in range(len(ids)):
            for b_ in range(a + 1, len(ids)):
                sw, tw = swing_twist(rel(S.GT[ids[a]], S.GT[ids[b_]])[:3, :3]); cand[bin_index(sw, SWING_BINS)].append((ids[a], ids[b_], sw, tw))
        pairs = []
        for b_, lst in cand.items(): pairs += random.sample(lst, min(PER_BIN, len(lst)))
        print(sq, "pairs", len(pairs), {SWING_LABELS[b_]: len(v) for b_, v in cand.items()}, flush=True)
        for (fi, fj, sw, tw) in pairs:   # fi earlier (B), fj later (A = query)
            rgb_i, d_i, m_i = seq.frame(fi); rgb_j, d_j, m_j = seq.frame(fj); snap = S.snapshot_at(fj); c2w_i_on = snap.get(fi, S.C1[fi]); c2w_j_on = S.C1[fj]
            rg = rel(S.GT[fi], S.GT[fj]); r2 = rel(S.C2[fi], S.C2[fj]); c2_rot = rot_deg(r2[:3, :3], rg[:3, :3])
            # ---- 1. LoFTR + depth (A = j later, B = i earlier; poses = online: i from snapshot at j, j = C1)
            gA = cv2.cvtColor(rgb_j, cv2.COLOR_RGB2GRAY); gB = cv2.cvtColor(rgb_i, cv2.COLOR_RGB2GRAY)
            imgA, imgB, tfA, tfB = process_pair(gA, gB, roi_of(m_j), roi_of(m_i), c2w_j_on, c2w_i_on)
            corres = loftr.predict(rgbAs=imgA[None, ..., None], rgbBs=imgB[None, ..., None])[0]
            n_raw = len(corres); n_inl = 0; success = 0; lo_rot = np.nan; n_valid = 0
            if n_raw > 0:
                pa = corres[:, :2].astype(np.float64); pb = corres[:, 2:4].astype(np.float64)
                pa = (np.c_[pa, np.ones(len(pa))] @ np.linalg.inv(tfA).T); pa = pa[:, :2] / pa[:, 2:3]; pb = (np.c_[pb, np.ones(len(pb))] @ np.linalg.inv(tfB).T); pb = pb[:, :2] / pb[:, 2:3]
                ua = np.round(pa).astype(int); ub = np.round(pb).astype(int); H, W = m_i.shape
                ok = (ua[:, 0] >= 0) & (ua[:, 0] < W) & (ua[:, 1] >= 0) & (ua[:, 1] < H) & (ub[:, 0] >= 0) & (ub[:, 0] < W) & (ub[:, 1] >= 0) & (ub[:, 1] < H)
                ua, ub = ua[ok], ub[ok]; ok = m_j[ua[:, 1], ua[:, 0]] & m_i[ub[:, 1], ub[:, 0]] & (d_j[ua[:, 1], ua[:, 0]] > 0.1) & (d_i[ub[:, 1], ub[:, 0]] > 0.1); ua, ub = ua[ok], ub[ok]; n_valid = len(ua)
                if n_valid >= RANSAC["num_sample"]:
                    PA = backproject(ua, d_j, K); PB = backproject(ub, d_i, K); R, t, inl = ransac_procrustes(PA, PB, rng); n_inl = int(inl.sum())
                    if R is not None and n_inl >= MIN_MATCH:
                        success = 1; T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t        # p_j = T p_i  ->  T = inv(c2w_j) c2w_i = rel(j, i)
                        lo_rot = rot_deg(np.linalg.inv(T)[:3, :3], rg[:3, :3])
            # ---- 2. VGGT 2-view (I1) and 3. + roll
            res = {}
            for tag in ["v2", "v2roll"]:
                frames = [(rgb_i, d_i, m_i), (rgb_j, d_j, m_j)]; Rc = [L.centre_rotation(K, L.mask_centroid(m)) for _, _, m in frames]
                if tag == "v2roll":
                    R_first = c2w_i_on[:3, :3]; Rs = []
                    for idx, Ri in enumerate([c2w_i_on[:3, :3], c2w_j_on[:3, :3]]): psi, proj, flag = roll_psi(R_first, Ri, Rc[idx]); Rs.append(Rz(psi) @ Rc[idx])
                else: Rs = Rc
                fv = L.virtual_focal(K, [m for _, _, m in frames], Rs, SZ); Kv = np.array([[fv, 0, SZ / 2], [0, fv, SZ / 2], [0, 0, 1.0]])
                w = [warp_I1_bg(rgb, d, m, K, R, Kv, SZ) for (rgb, d, m), R in zip(frames, Rs)]
                r = vggt.infer(torch.from_numpy(np.stack([x[0] for x in w])).permute(0, 3, 1, 2).contiguous())
                c2w = [np.linalg.inv(r["w2c"][a]) @ blockdiag(Rs[a]) for a in range(2)]; re_ = rel(c2w[0], c2w[1]); res[tag] = (rot_deg(re_[:3, :3], rg[:3, :3]), tdir_deg(re_[:3, 3], rg[:3, 3]))
            rows.append(dict(seq=sq, kf_i=fi, kf_j=fj, swing=sw, roll=tw, swing_bin=SWING_LABELS[bin_index(sw, SWING_BINS)], loftr_raw=n_raw, loftr_valid=n_valid, loftr_inliers=n_inl, loftr_success=success, loftr_success30=int(n_inl >= 30), loftr_rot=lo_rot,
                             vggt2_rot=res["v2"][0], vggt2_tdir=res["v2"][1], vggt2roll_rot=res["v2roll"][0], vggt2roll_tdir=res["v2roll"][1], c2_rot=c2_rot, c1_rot=rot_deg(rel(S.C1[fi], S.C1[fj])[:3, :3], rg[:3, :3])))
        print(sq, "done", len(rows), f"{time.time()-t0:.0f}s", flush=True)
    with open(out / "pairs_bench.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    json.dump(dict(seqs=SEQS, per_bin=PER_BIN, seed=SEED, n_pairs=len(rows), loftr="BundleTrack/LoFTR outdoor_ds.ckpt, match_coarse.thr 0.2 (loftr_wrapper)", preprocess=f"processImagePair replica: rotate B into A by camera-z axis-angle component of R_A R_B^-1 (online poses), ROI=mask bbox +10 px margin, common max_dim scale, {OUT_SIDE}x{OUT_SIDE} gray warpPerspective",
                   filter="matches inside both masks and depth > 0.1 m (no normal check)", ransac=dict(num_sample=RANSAC["num_sample"], max_iter=RANSAC["max_iter"], inlier_dist=RANSAC["inlier_dist"], min_match=MIN_MATCH), vggt_ckpt=vggt.ckpt_sha, finished=time.strftime("%F %T")), open(out / "manifest.json", "w"), indent=1)
    print("B6 done", len(rows))


if __name__ == "__main__":
    main()
