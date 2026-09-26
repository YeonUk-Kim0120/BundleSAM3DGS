"""Probe 1b stage-B building blocks (new file; Probe 1 code imported unchanged): reference selection variants (B1 nearest, B2 bands),
roll-normalised virtual camera (B3), warps with background colour (B4) and no-mask padding (B5), and pair rows."""
from __future__ import annotations
import numpy as np, cv2
import probe_lib as L
from p1b_common import swing_twist, rel, rot_deg, tdir_deg, pair_type, align_variant


# ---------------------------------------------------------------- reference selection (online info only: refs R-online, new C1)
def ang(a, b): return L.ang(a, b)


def select_nearest(new_ids, cand_ids, ref_pose, new_pose, k):
    """B1: k candidates with the smallest mean optical-axis angle to the new KFs (no farthest-point)."""
    if not cand_ids: return []
    new_axes = [L.optical_axis(new_pose[n]) for n in new_ids]
    score = {c: np.mean([ang(L.optical_axis(ref_pose[c]), a) for a in new_axes]) for c in cand_ids}
    return sorted(cand_ids, key=lambda c: score[c])[:k]


BANDS = [(15, 30), (30, 60), (60, 90)]


def select_bands(new_ids, cand_ids, ref_pose, new_pose, per_band=2):
    """B2: first = smallest axis angle; then 2 per band [15,30), [30,60), [60,90) nearest to the band centre; empty slots filled by
    the nearest remaining.  Returns (refs, composition dict band -> [ids])."""
    if not cand_ids: return [], {}
    new_axes = [L.optical_axis(new_pose[n]) for n in new_ids]
    score = {c: float(np.mean([ang(L.optical_axis(ref_pose[c]), a) for a in new_axes])) for c in cand_ids}
    order = sorted(cand_ids, key=lambda c: score[c]); first = order[0]; chosen = [first]; comp = {"first": [first]}
    for lo, hi in BANDS:
        inband = [c for c in order if lo <= score[c] < hi and c not in chosen]
        pick = sorted(inband, key=lambda c: abs(score[c] - (lo + hi) / 2))[:per_band]; chosen += pick; comp[f"[{lo},{hi})"] = pick
    need = 1 + per_band * len(BANDS) - len(chosen); fill = [c for c in order if c not in chosen][:max(need, 0)]; chosen += fill; comp["fill"] = fill
    return chosen, comp


# ---------------------------------------------------------------- roll normalisation (B3)
def Rz(psi): c, s = np.cos(psi), np.sin(psi); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def roll_psi(R_first_c2w, R_i_c2w, R_c):
    """psi_i so that the first reference's camera-up direction, seen in frame i's virtual camera, points to image up (-y).
    u = -R_first[:,1] (object frame); u_ci = R_i^T u; u_vi = R_c u_ci; psi from (x, y); psi = 0 (flagged) if projection < 0.2."""
    u = -R_first_c2w[:, 1]; u_ci = R_i_c2w.T @ u; u_vi = R_c @ u_ci; x, y = u_vi[0], u_vi[1]; proj = float(np.hypot(x, y))
    if proj < 0.2: return 0.0, proj, True
    # rotate (x, y) onto (0, -1): angle of (x,y) is atan2(y, x); target angle -pi/2
    psi = -np.pi / 2 - np.arctan2(y, x)
    return float(psi), proj, False


# ---------------------------------------------------------------- warps
def warp_I1_bg(rgb, depth, mask, K, R, Kv, S, bg=1.0):
    """Same as probe_lib.warp_I1 but with a configurable background value (white 1.0 / black 0.0 / gray 0.5)."""
    u, v = np.meshgrid(np.arange(S, dtype=np.float64) + 0.5, np.arange(S, dtype=np.float64) + 0.5)
    pv = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(Kv).T
    p = pv @ (K @ R.T).T; p = p[..., :2] / p[..., 2:3]; px = (p[..., 0] - 0.5).astype(np.float32); py = (p[..., 1] - 0.5).astype(np.float32)
    rgbf = rgb.astype(np.float32) / 255.0; rgbf[~mask] = bg
    img = cv2.remap(rgbf, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(bg, bg, bg))
    mask_v = cv2.remap(mask.astype(np.uint8), px, py, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
    depth_v = cv2.remap(depth, px, py, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    img[~mask_v] = bg
    return img, depth_v, mask_v, (px, py)


def warp_I0_nomask(rgb, depth, mask, K, S):
    """B5 negative control: original image WITHOUT masking, long side -> S, short side centre-padded (black, as VGGT's own loader)."""
    H, W = mask.shape; scale = S / max(H, W); nh, nw = int(round(H * scale)), int(round(W * scale)); top, left = (S - nh) // 2, (S - nw) // 2
    img = np.zeros((S, S, 3), np.float32); img[top:top + nh, left:left + nw] = cv2.resize(rgb.astype(np.float32) / 255.0, (nw, nh), interpolation=cv2.INTER_LINEAR)
    m = np.zeros((S, S), bool); m[top:top + nh, left:left + nw] = cv2.resize(mask.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST) > 0
    d = np.zeros((S, S), np.float32); d[top:top + nh, left:left + nw] = cv2.resize(depth, (nw, nh), interpolation=cv2.INTER_NEAREST)
    Kn = K.copy(); Kn[0, :] *= scale; Kn[1, :] *= scale; Kn[0, 2] += left; Kn[1, 2] += top
    u, v = np.meshgrid(np.arange(S, dtype=np.float32), np.arange(S, dtype=np.float32)); px = (u - left) / scale; py = (v - top) / scale
    return img, d, m, Kn, (px, py)


# ---------------------------------------------------------------- pair rows (same columns as A1)
def pair_rows(cell, b_id, ids, roles, c2w_est, S, mask_area, conf, T3):
    rows = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            fi, fj = ids[i], ids[j]; gi, gj = S.GT.get(fi), S.GT.get(fj)
            if gi is None or gj is None: continue
            rg = rel(gi, gj); re_ = rel(c2w_est[i], c2w_est[j]); sw, tw = swing_twist(rg[:3, :3]); r2 = rel(S.C2[fi], S.C2[fj]); r1 = rel(S.C1[fi], S.C1[fj])
            rows.append([cell["model"], cell["input"], cell["batch"], cell["ds"], cell["seq"], b_id, fi, fj, roles[fi], roles[fj], pair_type(roles[fi], roles[fj]), sw, tw,
                         rot_deg(re_[:3, :3], rg[:3, :3]), tdir_deg(re_[:3, 3], rg[:3, 3]), rot_deg(r2[:3, :3], rg[:3, :3]), rot_deg(r1[:3, :3], rg[:3, :3]),
                         mask_area[fi], mask_area[fj], conf[fi], conf[fj], T3])
    return rows
