"""Data, warping, batching, scale/alignment and metrics for EXP_VGGT_PROBE1 (brief sections 3, 5, 6, 7, 8)."""
from __future__ import annotations
import glob, os, sys
from pathlib import Path
import numpy as np, torch, yaml, cv2
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments")); sys.path.append(str(REPO / "BundleTrack/scripts"))
from exp_feedback_gradient_probe import GroundTruth, rot_deg  # noqa: E402
from exp_hybrid_realistic_replay import load_snapshots  # noqa: E402

BASE = Path("/home/kist/Desktop/BundleSDF_baseline_outputs")


# ------------------------------------------------------------------ data
class Sequence:
    def __init__(self, dataset, seq):
        self.dataset, self.seq = dataset, seq
        self.run = BASE / f"full_eval_sam2_{dataset}" / dataset / seq
        self.video = REPO / ("datasets/HO3D_v3/evaluation" if dataset == "ho3d" else "datasets/YCBInEOAT") / seq   # dataset in {ho3d, ycb}
        self.K = np.loadtxt(self.run / "cam_K.txt").reshape(3, 3).astype(np.float64)
        self.snaps = load_snapshots(self.run)                       # [(frame, {kf: c2w})] in frame order
        self.frame_index = {f: i for i, (f, _) in enumerate(self.snaps)}
        self.kf_ids = list(self.snaps[-1][1].keys())                # creation order (yaml order of the last snapshot)
        self.C2 = dict(self.snaps[-1][1]); self.C1 = {}
        for f, kf in self.snaps:
            for k, P in kf.items():
                self.C1.setdefault(k, P)
        assert all(k in self.C1 for k in self.kf_ids)
        self.gt = GroundTruth(dataset, self.video, self.run)
        self.has_gt = {k: self.gt.gt_c2w(k) is not None for k in self.kf_ids}
        self._cache = {}

    def snapshot_at(self, kf):                                      # R-online: BSDF snapshot at keyframe kf's own frame
        return self.snaps[self.frame_index[kf]][1]

    def frame(self, fid):
        if fid not in self._cache:
            rgb = cv2.cvtColor(cv2.imread(str(self.run / "color" / f"{fid}.png")), cv2.COLOR_BGR2RGB)
            depth = cv2.imread(str(self.run / "depth_filtered" / f"{fid}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
            mask = cv2.imread(str(self.run / "mask" / f"{fid}.png"), cv2.IMREAD_GRAYSCALE) > 0
            self._cache[fid] = (rgb, depth, mask)
        return self._cache[fid]

    def gt_c2w(self, kf): return self.gt.gt_c2w(kf)


def pose_err(c2w, gt):
    return rot_deg(c2w[:3, :3], gt[:3, :3]), float(np.linalg.norm(c2w[:3, 3] - gt[:3, 3]) * 1000)


# ------------------------------------------------------------------ warps (section 5)
def rodrigues(axis, ang):
    axis = np.asarray(axis, dtype=np.float64); n = np.linalg.norm(axis)
    if n < 1e-12 or abs(ang) < 1e-12: return np.eye(3)
    a = axis / n; K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def centre_rotation(K, c):
    """R such that R r = z_hat, r = normalize(K^-1 [c;1]) (minimal rotation)."""
    r = np.linalg.inv(K) @ np.array([c[0], c[1], 1.0]); r /= np.linalg.norm(r); z = np.array([0, 0, 1.0])
    ax = np.cross(r, z); ang = float(np.arccos(np.clip(r @ z, -1, 1)))
    R = rodrigues(ax, ang); assert np.allclose(R @ r, z, atol=1e-9); return R


def mask_centroid(mask):
    ys, xs = np.nonzero(mask); return np.array([xs.mean(), ys.mean()])


def virtual_focal(seqK, masks, Rs, S, margin=0.4):
    """Common f_v: after rotation every mask pixel must project within margin*S of the image centre."""
    Kinv = np.linalg.inv(seqK); worst = 0.0
    for m, R in zip(masks, Rs):
        ys, xs = np.nonzero(m); p = np.stack([xs, ys, np.ones_like(xs)], 0).astype(np.float64)
        d = R @ (Kinv @ p); d = d / d[2:3]; worst = max(worst, float(np.abs(d[:2]).max()))
    return margin * S / max(worst, 1e-9)


def warp_I1(rgb, depth, mask, K, R, Kv, S):
    """Virtual camera view: inverse map p = K R^T Kv^-1 p_v.  RGB bilinear, mask/depth nearest, outside/white = 1.0.
    Returns img[S,S,3] float, depth_v[S,S] (sensor depth resampled, z of the ORIGINAL camera), mask_v[S,S], map (px,py)."""
    u, v = np.meshgrid(np.arange(S, dtype=np.float64) + 0.5, np.arange(S, dtype=np.float64) + 0.5)   # pixel centres
    pv = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(Kv).T
    p = pv @ (K @ R.T).T; p = p[..., :2] / p[..., 2:3]; px = (p[..., 0] - 0.5).astype(np.float32); py = (p[..., 1] - 0.5).astype(np.float32)
    H, W = mask.shape
    rgbf = rgb.astype(np.float32) / 255.0; rgbf[~mask] = 1.0
    img = cv2.remap(rgbf, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(1.0, 1.0, 1.0))
    mask_v = cv2.remap(mask.astype(np.uint8), px, py, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
    depth_v = cv2.remap(depth, px, py, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    img[~mask_v] = 1.0
    return img, depth_v, mask_v, (px, py)


def unwarp_mask(mask_v, K, R, Kv, S, H, W):
    """Forward map original pixels -> virtual pixels (nearest) to check IoU of the round trip."""
    u, v = np.meshgrid(np.arange(W, dtype=np.float64) + 0.5, np.arange(H, dtype=np.float64) + 0.5)
    p = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(K).T
    q = p @ (Kv @ R).T; q = q[..., :2] / q[..., 2:3]; qx = (q[..., 0] - 0.5).astype(np.float32); qy = (q[..., 1] - 0.5).astype(np.float32)
    return cv2.remap(mask_v.astype(np.uint8), qx, qy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0


def warp_I0(rgb, depth, mask, K, S):
    """Baseline: masked (white outside), long side -> S, short side centre-padded white; K updated."""
    H, W = mask.shape; scale = S / max(H, W); nh, nw = int(round(H * scale)), int(round(W * scale)); top, left = (S - nh) // 2, (S - nw) // 2
    rgbf = rgb.astype(np.float32) / 255.0; rgbf[~mask] = 1.0
    img = np.ones((S, S, 3), np.float32); img[top:top + nh, left:left + nw] = cv2.resize(rgbf, (nw, nh), interpolation=cv2.INTER_LINEAR)
    m = np.zeros((S, S), bool); m[top:top + nh, left:left + nw] = cv2.resize(mask.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST) > 0
    d = np.zeros((S, S), np.float32); d[top:top + nh, left:left + nw] = cv2.resize(depth, (nw, nh), interpolation=cv2.INTER_NEAREST)
    img[~m] = 1.0
    Kn = K.copy(); Kn[0, :] *= scale; Kn[1, :] *= scale; Kn[0, 2] += left; Kn[1, 2] += top
    u, v = np.meshgrid(np.arange(S, dtype=np.float32), np.arange(S, dtype=np.float32)); px = (u - left) / scale; py = (v - top) / scale
    return img, d, m, Kn, (px, py)


# ------------------------------------------------------------------ batches (section 6)
def optical_axis(c2w): return c2w[:3, 2] / np.linalg.norm(c2w[:3, 2])


def ang(a, b): return float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))


def select_reference(new_ids, cand_ids, ref_pose, new_pose, k):
    """Section 6 rule: first = candidate with the smallest mean optical-axis angle to the new KFs; then half of the
    remaining nearest to the new KFs, the other half by farthest-point sampling among the chosen."""
    if not cand_ids: return []
    new_axes = [optical_axis(new_pose[n]) for n in new_ids]
    mean_ang = {c: np.mean([ang(optical_axis(ref_pose[c]), a) for a in new_axes]) for c in cand_ids}
    order = sorted(cand_ids, key=lambda c: mean_ang[c]); chosen = [order[0]]; rest = order[1:]
    n_near = (k - 1) // 2 if k - 1 > 0 else 0
    chosen += rest[:n_near]; rest = rest[n_near:]
    while len(chosen) < k and rest:
        far = max(rest, key=lambda c: min(ang(optical_axis(ref_pose[c]), optical_axis(ref_pose[x])) for x in chosen))
        chosen.append(far); rest.remove(far)
    return chosen[:k]


def make_batches(seq: Sequence, mode: str, M=5, k=8):
    """Returns list of dict(ref=[kf...], new=[kf...]) in batch order; batch 0 = KF0 reference + KF1..4 new."""
    ids = seq.kf_ids; batches = [{"ref": [ids[0]], "new": ids[1:5]}]
    if mode == "chain":
        i = 5
        while i < len(ids):
            batches.append({"ref": [ids[i - 1]], "new": ids[i:i + 5]}); i += 5
    else:
        i = 5
        while i < len(ids):
            new = ids[i:i + M]; cand = ids[:i]; snap = seq.snapshot_at(new[-1])
            ref_pose = {c: snap.get(c, seq.C1[c]) for c in cand}   # R-online: snapshot at the newest new KF's frame
            batches.append({"ref": select_reference(new, cand, ref_pose, seq.C1, k), "new": new}); i += M
    return batches


# ------------------------------------------------------------------ scale + alignment (section 7)
def proj_so3(M):
    u, _, vt = np.linalg.svd(M); R = u @ vt
    if np.linalg.det(R) < 0: u[:, -1] *= -1; R = u @ vt
    return R


def ray_norm(K, S_or_HW):
    """||K^-1 [p;1]|| at every pixel centre of an SxS (or HxW) grid."""
    H, W = (S_or_HW, S_or_HW) if np.isscalar(S_or_HW) else S_or_HW
    u, v = np.meshgrid(np.arange(W, dtype=np.float64) + 0.5, np.arange(H, dtype=np.float64) + 0.5)
    p = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(K).T; return np.linalg.norm(p, axis=-1)


def frame_scale(depth_sensor_v, mask_orig, pmap, K_orig, z_vggt, conf, Kv, min_px=200):
    """Valid pixels: inside the 3-px-eroded original mask, sensor depth > 0.1 m, conf in the top 50 % of the frame's mask.
    Ray distances: rho_s = D_s(p) ||K^-1 [p;1]||, rho_v = z_v(p_v) ||Kv^-1 [p_v;1]||.  Returns (s_i, n_valid, rho_s, rho_v);
    s_i = nan and rho arrays None when fewer than min_px valid pixels."""
    er = cv2.erode(mask_orig.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0     # 3 px erosion
    px, py = pmap; H, W = mask_orig.shape
    xi = np.clip(np.round(px).astype(int), 0, W - 1); yi = np.clip(np.round(py).astype(int), 0, H - 1)
    inside = (px >= 0) & (px < W) & (py >= 0) & (py < H) & er[yi, xi]
    valid = inside & (depth_sensor_v > 0.1)
    if valid.sum() == 0: return np.nan, 0, None, None
    thr = np.percentile(conf[valid], 50); valid &= conf >= thr
    rn_s = np.linalg.norm(np.stack([xi + 0.5, yi + 0.5, np.ones_like(xi, dtype=np.float64)], -1) @ np.linalg.inv(K_orig).T, axis=-1)
    rho_s = depth_sensor_v[valid] * rn_s[valid]; rho_v = z_vggt[valid] * ray_norm(Kv, z_vggt.shape)[valid]
    ok = rho_v > 1e-6; rho_s, rho_v = rho_s[ok], rho_v[ok]
    if ok.sum() < min_px: return np.nan, int(ok.sum()), None, None
    return float(np.median(rho_s / rho_v)), int(ok.sum()), rho_s, rho_v


def align(ref_known, ref_est):
    """T in SE(3), batch -> object frame: R_T = proj_SO3(sum R_r R_hat_r^T), t_T = median_r(t_r - R_T t_hat_r)."""
    M = sum(a[:3, :3] @ b[:3, :3].T for a, b in zip(ref_known, ref_est)); R = proj_so3(M)
    t = np.median(np.stack([a[:3, 3] - R @ b[:3, 3] for a, b in zip(ref_known, ref_est)]), 0)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t; return T


def relative(c2w_i, c2w_j): return np.linalg.inv(c2w_i) @ c2w_j


def pairwise_E2(est, gt):
    """Gauge-free: over all pairs, relative rotation error (deg) and relative translation direction error (deg)."""
    rots, dirs = [], []
    n = len(est)
    for i in range(n):
        for j in range(i + 1, n):
            if est[i] is None or est[j] is None or gt[i] is None or gt[j] is None: continue
            re, rg = relative(est[i], est[j]), relative(gt[i], gt[j])
            rots.append(rot_deg(re[:3, :3], rg[:3, :3]))
            te, tg = re[:3, 3], rg[:3, 3]
            if np.linalg.norm(te) > 1e-6 and np.linalg.norm(tg) > 1e-6: dirs.append(ang(te / np.linalg.norm(te), tg / np.linalg.norm(tg)))
    return (float(np.median(rots)) if rots else np.nan, float(np.median(dirs)) if dirs else np.nan)


def scale_gt(est_batch, gt, R_T):
    """s_gt = argmin_s sum ||s R_T t_hat_ij - t_gt_ij||^2 over pairs (translations expressed in the object frame)."""
    num = den = 0.0
    n = len(est_batch)
    for i in range(n):
        for j in range(i + 1, n):
            if gt[i] is None or gt[j] is None: continue
            a = R_T @ (est_batch[j][:3, 3] - est_batch[i][:3, 3]); b = gt[j][:3, 3] - gt[i][:3, 3]
            num += a @ b; den += a @ a
    return num / den if den > 0 else np.nan


def warp_I2(rgb, depth, mask, K, S, margin=0.4):
    """Diagnostic crop (section 5, I2): square crop centred on the mask bbox centre, enlarged so the mask stays within
    margin*S of the image centre, resized to S x S, NO principal-point correction: the downstream code assumes the
    principal point at the image centre (K_assumed) and applies no rotation.  Returns like warp_I0 (img, depth, mask, K_assumed, map)."""
    H, W = mask.shape; ys, xs = np.nonzero(mask); cx, cy = (xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0
    side = max(xs.max() - xs.min(), ys.max() - ys.min()) / (2 * margin); scale = S / side   # bbox half-extent -> margin*S from the centre (same limit as I1)
    u, v = np.meshgrid(np.arange(S, dtype=np.float32) + 0.5, np.arange(S, dtype=np.float32) + 0.5)
    px = (u - S / 2) / scale + cx - 0.5; py = (v - S / 2) / scale + cy - 0.5
    rgbf = rgb.astype(np.float32) / 255.0; rgbf[~mask] = 1.0
    img = cv2.remap(rgbf, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(1.0, 1.0, 1.0))
    m = cv2.remap(mask.astype(np.uint8), px, py, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
    d = cv2.remap(depth, px, py, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0); img[~m] = 1.0
    Ka = np.array([[K[0, 0] * scale, 0, S / 2], [0, K[1, 1] * scale, S / 2], [0, 0, 1.0]])   # assumed: principal point at the crop centre
    return img, d, m, Ka, (px, py)
