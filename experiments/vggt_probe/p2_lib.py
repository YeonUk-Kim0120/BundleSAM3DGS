"""Probe 2 library: model-render anchors for VGGT (EXP_VGGT_PROBE2_BRIEF.md).  New file; Probe 1/1b code is imported unchanged.
Models (MeshModel = MD-gt via offscreen_renderer.ModelRendererOffscreen; GSModel = MD-prior / MD-map via GaussianRunner._rasterize),
estimators E-V2, E-V5 (b)/(d), E-V5x2, E-L, E-I, E-V->I, and per-keyframe row assembly.  All poses are metric OpenCV c2w in the BSDF
run's object frame (same gauge as Probe 1)."""
from __future__ import annotations
import os, pickle, sys, time
from pathlib import Path
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np, cv2, torch
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[1]
for p in (str(HERE), str(REPO), str(REPO / "experiments"), str(REPO / "BundleTrack/scripts")):
    if p not in sys.path: sys.path.insert(0, p)
import probe_lib as L                                   # noqa: E402
from p1b_common import align_variant, blockdiag         # noqa: E402
from p1b_B_lib import warp_I1_bg                        # noqa: E402
import p1b_B6 as B6                                      # noqa: E402  (process_pair, roi_of, backproject, ransac_procrustes, MIN_MATCH)

P1B_CACHE = REPO / "outputs/exp_vggt_probe1b/cache"
OUT = REPO / "outputs/exp_vggt_probe2"; LOG = REPO / "logs/exp_vggt_probe2"
HO3D = ["AP10", "AP11", "AP12", "AP13", "AP14", "MPM10", "MPM11", "MPM12", "MPM13", "MPM14", "SB11", "SB13", "SM1"]
H, W = 480, 640

# ------------------------------------------------------------------ fixed hyper-parameters (set before stage A; recorded in manifests)
HP = dict(S=518, background=1.0, orbit_deg=10.0, min_render_px=200, align_d_threshold_deg=10.0,
          icp_voxel_m=0.002, icp_normal_radius_m=0.01, icp_normal_max_nn=30, icp_stages_m=[0.02, 0.01, 0.005], icp_max_iter=30,
          loftr=dict(resize=B6.OUT_SIDE, min_match=B6.MIN_MATCH, ransac=B6.RANSAC), success=(5.0, 20.0), strict=(2.0, 10.0), worsen_margin_deg=2.0)


# ------------------------------------------------------------------ data
class SeqData:
    """Poses from the Probe 1b cache (C1, C2, GT, snapshots, K, kf_ids) + frame loader reading the same BSDF-run files as Probe 1."""
    def __init__(self, ds, sq):
        d = pickle.load(open(P1B_CACHE / f"{ds}_{sq}.pkl", "rb")); self.__dict__.update(d); self.ds, self.sq = ds, sq; self.run = Path(d["run"])
        self.video = REPO / ("datasets/HO3D_v3/evaluation" if ds == "ho3d" else "datasets/YCBInEOAT") / sq

    def frame(self, fid):
        rgb = cv2.cvtColor(cv2.imread(str(self.run / "color" / f"{fid}.png")), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(str(self.run / "depth_filtered" / f"{fid}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        mask = cv2.imread(str(self.run / "mask" / f"{fid}.png"), cv2.IMREAD_GRAYSCALE) > 0
        return rgb, depth, mask

    def align(self):
        from exp_feedback_gradient_probe import GroundTruth
        return GroundTruth(self.ds, self.video, self.run).align


# ------------------------------------------------------------------ models
_KEEP_RENDERERS = []   # pyrender: a garbage-collected OffscreenRenderer calls eglTerminate on the shared display and kills newer contexts


class MeshModel:
    """MD-gt: GT textured mesh moved into the run object frame with GroundTruth.align; rendered with the repository pyrender renderer
    (constructor defaults: ambient light 1.0, black background, znear 0.1, zfar 2)."""
    name = "MD-gt"

    def __init__(self, seq: SeqData):
        from offscreen_renderer import ModelRendererOffscreen
        from eval_mesh_cd import load_gt_mesh
        mesh = load_gt_mesh(seq.ds, seq.sq); self.align = seq.align(); mesh.apply_transform(self.align)
        self.center = np.asarray(mesh.vertices, dtype=np.float64).mean(0); self.K = seq.K
        self.r = ModelRendererOffscreen([], seq.K, H, W); self.r.add_mesh(mesh); self.source = "eval_mesh_cd.load_gt_mesh + GroundTruth.align"
        _KEEP_RENDERERS.append(self.r)

    def render(self, c2w):
        color, depth = self.r.render([np.linalg.inv(c2w)])
        depth = np.asarray(depth, dtype=np.float32); mask = depth > 0
        return np.asarray(color[..., :3], dtype=np.uint8), depth, mask


class GSModel:
    """MD-prior / MD-map: a GaussianRunner checkpoint rendered at arbitrary metric c2w with the runner's own rasterizer (RGB+ED, alpha > 0.5)."""

    def __init__(self, ckpt, name, K, device="cuda:0"):
        from gaussian_runner import GaussianRunner
        self.name = name; self.source = str(ckpt); self.r = GaussianRunner.load_checkpoint(ckpt, device=device); self.dev = torch.device(device)
        means = self.r.normalization.metric_points(self.r.splats["means"].detach().cpu().numpy()); self.center = means.astype(np.float64).mean(0)
        self.K = torch.tensor(K, dtype=torch.float32, device=self.dev)[None]; self.n = int(self.r.num_gaussians)
        cfg = self.r.config; self.sh = min(self.r.total_steps // int(cfg["sh_degree_interval"]), int(cfg["sh_degree"]))

    @torch.no_grad()
    def render(self, c2w):
        c2w_n = torch.from_numpy(self.r.normalization.normalize_c2w(np.asarray(c2w, dtype=np.float64))).to(self.dev)[None]
        res = self.r._rasterize(K=self.K, c2w=c2w_n, width=W, height=H, sh_degree=self.sh, render_mode="RGB+ED", absgrad=False)
        col = res.colors[0]; alpha = res.alpha[0, ..., 0]; rgb = col[..., :3].clamp(0, 1); depth = col[..., 3] / self.r.normalization.scale
        mask = alpha > 0.5; depth = torch.where(mask, depth, torch.zeros_like(depth))
        return (rgb * 255).round().byte().cpu().numpy(), depth.float().cpu().numpy(), mask.cpu().numpy()


def prior_ckpt(sq): return OUT / "models" / "prior" / sq / "checkpoint_prior.pt"
def map_ckpt(sq): return REPO / "outputs/exp_batch_20260922/reanchor_v2/ho3d" / sq / "final/gs_online0/checkpoint_global.pt"


def load_model(kind, seq: SeqData, device="cuda:0"):
    if kind == "MD-gt": return MeshModel(seq)
    if kind == "MD-prior": return GSModel(prior_ckpt(seq.sq), "MD-prior", seq.K, device)
    if kind == "MD-map": return GSModel(map_ckpt(seq.sq), "MD-map", seq.K, device)
    raise ValueError(kind)


# ------------------------------------------------------------------ poses
def orbit(P, center, deg, axis):
    """Camera P orbited about the model centre around P's own x or y axis (object frame) by deg; roll about the optical axis unchanged."""
    a = P[:3, 0] if axis == "x" else P[:3, 1]; R = L.rodrigues(a, np.radians(deg)); Q = np.eye(4)
    Q[:3, :3] = R @ P[:3, :3]; Q[:3, 3] = center + R @ (P[:3, 3] - center); return Q


def anchor_poses(P, center, deg):
    return [P, orbit(P, center, deg, "x"), orbit(P, center, -deg, "x"), orbit(P, center, deg, "y"), orbit(P, center, -deg, "y")]


def perturb(G, center, deg, rng, trans_sigma_m=0.0):
    """GT camera rotated about the model centre around a random axis by deg (+ N(0, sigma) translation noise per axis)."""
    ax = rng.normal(size=3); R = L.rodrigues(ax / np.linalg.norm(ax), np.radians(deg)); Q = np.eye(4)
    Q[:3, :3] = R @ G[:3, :3]; Q[:3, 3] = center + R @ (G[:3, 3] - center) + (rng.normal(0, trans_sigma_m, 3) if trans_sigma_m > 0 else 0)
    return Q


def pose_err(P, G): return L.pose_err(P, G)          # (rot deg, camera-centre distance mm)


def iou(a, b):
    u = (a | b).sum(); return float((a & b).sum() / u) if u else 0.0


# ------------------------------------------------------------------ VGGT with render anchors
class Renders:
    """Render cache for one estimator call: list of (rgb, depth, mask, c2w)."""
    def __init__(self, model, poses):
        t = time.time(); self.items = [(*model.render(P), P) for P in poses]; self.time = time.time() - t


def vggt_batch(vggt, K, images, S):
    """images: list of (rgb uint8, depth, mask).  Common-f_v I1 warp (white background) -> VGGT -> pooled scale s -> c2w_b (virtual rotation
    undone, translation × s).  Returns dict(c2w_b, s, s_i, ok)."""
    t = time.time()
    Rs = [L.centre_rotation(K, L.mask_centroid(m)) for (_, _, m) in images]
    fv = L.virtual_focal(K, [m for (_, _, m) in images], Rs, S); Kv = np.array([[fv, 0, S / 2], [0, fv, S / 2], [0, 0, 1.0]])
    w = [warp_I1_bg(rgb, d, m, K, R, Kv, S, HP["background"]) for (rgb, d, m), R in zip(images, Rs)]
    t_warp = time.time() - t
    out = vggt.infer(torch.from_numpy(np.stack([x[0] for x in w])).permute(0, 3, 1, 2).contiguous())
    s_i, all_s, all_v = [], [], []
    for i in range(len(images)):
        s, n, rs, rv = L.frame_scale(w[i][1], images[i][2], w[i][3], K, out["depth"][i], out["conf"][i], Kv); s_i.append(s)
        if rs is not None: all_s.append(rs); all_v.append(rv)
    s_b = float(np.median(np.concatenate(all_s) / np.concatenate(all_v))) if all_s else np.nan
    c2w_b = []
    for i in range(len(images)):
        P = np.linalg.inv(out["w2c"][i]); P[:3, 3] *= s_b if np.isfinite(s_b) else 1.0; c2w_b.append(P @ blockdiag(Rs[i]))
    return dict(c2w_b=c2w_b, s=s_b, s_i=s_i, ok=np.isfinite(s_b), t_vggt=time.time() - t, t_warp=t_warp, t_infer=out["time_s"], depth=out["depth"], conf=out["conf"], warped=w, Kv=Kv, Rs=Rs)


def v_align(known, est, mode):
    return L.align(known[:1], est[:1]) if mode == "b" else align_variant("d", known, est, ["first"] + ["near"] * (len(known) - 1))


def anchor_residual(T, known, est):
    e = [L.pose_err(T @ b, a) for a, b in zip(known, est)]; return float(np.median([x[0] for x in e])), float(np.median([x[1] for x in e]))


def run_V(vggt, model, real, K, P0, n_anchor, deg, S):
    """E-V2 (n_anchor 1) / E-V5 (n_anchor 5).  Returns dict with est_b, est_d (V5), signals, times; None estimates if a render is empty."""
    poses = [P0] if n_anchor == 1 else anchor_poses(P0, model.center, deg)
    R = Renders(model, poses); items = R.items
    if any(m.sum() < HP["min_render_px"] for (_, _, m, _) in items) or real[2].sum() < HP["min_render_px"]:
        return dict(est_b=None, est_d=None, fail="empty_render", t_render=R.time, t_vggt=0.0, render0=items[0])
    b = vggt_batch(vggt, K, [(rgb, d, m) for (rgb, d, m, _) in items] + [real], S)
    known = [P for (*_, P) in items]; est_anchor = b["c2w_b"][:n_anchor]; real_b = b["c2w_b"][-1]
    Tb = v_align(known, est_anchor, "b"); out = dict(est_b=Tb @ real_b, s=b["s"], t_render=R.time, t_vggt=b["t_vggt"], render0=items[0], fail="")
    if n_anchor > 1:
        Td = v_align(known, est_anchor, "d"); out["est_d"] = Td @ real_b
        out["T1_rot"], out["T1_trans"] = anchor_residual(Td, known, est_anchor)
        sa = np.array([x for x in b["s_i"][:n_anchor] if np.isfinite(x)]); out["s_spread"] = float(sa.std() / sa.mean()) if len(sa) >= 2 else np.nan
    return out


# ------------------------------------------------------------------ LoFTR (render vs real) and model ICP
class Loftr:
    def __init__(self):
        from loftr_wrapper import LoftrRunner
        self.r = LoftrRunner()

    def estimate(self, real, render, K, P0, rng):
        """A = real (sensor depth), B = render@P0 (render depth).  BundleTrack-style pair preprocessing (both poses = P0 -> no in-plane
        rotation), matches inside both masks with depth > 0.1 m, 3-point RANSAC + Procrustes (config_ho3d.yml), success = inliers >= 5."""
        t = time.time(); rgb_a, d_a, m_a = real; rgb_b, d_b, m_b = render[:3]
        gA = cv2.cvtColor(rgb_a, cv2.COLOR_RGB2GRAY); gB = cv2.cvtColor(rgb_b, cv2.COLOR_RGB2GRAY)
        imgA, imgB, tfA, tfB = B6.process_pair(gA, gB, B6.roi_of(m_a), B6.roi_of(m_b), P0, P0)
        corres = self.r.predict(rgbAs=imgA[None, ..., None], rgbBs=imgB[None, ..., None])[0]
        n_inl = 0; est = None
        if len(corres):
            pa = corres[:, :2].astype(np.float64); pb = corres[:, 2:4].astype(np.float64)
            pa = np.c_[pa, np.ones(len(pa))] @ np.linalg.inv(tfA).T; pa = pa[:, :2] / pa[:, 2:3]; pb = np.c_[pb, np.ones(len(pb))] @ np.linalg.inv(tfB).T; pb = pb[:, :2] / pb[:, 2:3]
            ua = np.round(pa).astype(int); ub = np.round(pb).astype(int)
            ok = (ua[:, 0] >= 0) & (ua[:, 0] < W) & (ua[:, 1] >= 0) & (ua[:, 1] < H) & (ub[:, 0] >= 0) & (ub[:, 0] < W) & (ub[:, 1] >= 0) & (ub[:, 1] < H)
            ua, ub = ua[ok], ub[ok]; ok = m_a[ua[:, 1], ua[:, 0]] & m_b[ub[:, 1], ub[:, 0]] & (d_a[ua[:, 1], ua[:, 0]] > 0.1) & (d_b[ub[:, 1], ub[:, 0]] > 0.1); ua, ub = ua[ok], ub[ok]
            if len(ua) >= B6.RANSAC["num_sample"]:
                PA = B6.backproject(ua, d_a, K); PB = B6.backproject(ub, d_b, K); Rr, tt, inl = B6.ransac_procrustes(PA, PB, rng); n_inl = int(inl.sum())
                if Rr is not None and n_inl >= B6.MIN_MATCH:
                    T = np.eye(4); T[:3, :3] = Rr; T[:3, 3] = tt; est = P0 @ np.linalg.inv(T)        # p_real_cam = T p_render_cam
        return dict(est=est if est is not None else P0.copy(), success=int(est is not None), inliers=n_inl, n_raw=len(corres), t=time.time() - t)


def backproject_mask(depth, mask, K):
    v, u = np.nonzero(mask & (depth > 0.1)); z = depth[v, u]
    return np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], -1)


def icp(real, render, K, P_init):
    """Point-to-plane ICP: source = real masked sensor depth (camera frame), target = render@P_init depth in the object frame (normals
    estimated); init = P_init (camera -> object = c2w); 20 -> 10 -> 5 mm, 30 iterations each; 2 mm voxels."""
    import open3d as o3d
    t = time.time(); src_pts = backproject_mask(real[1], real[2], K); tgt_cam = backproject_mask(render[1], render[2], K)
    if len(src_pts) < 50 or len(tgt_cam) < 50: return dict(est=P_init.copy(), fitness=0.0, rmse=np.nan, t=time.time() - t, fail="few_points")
    tgt_pts = tgt_cam @ P_init[:3, :3].T + P_init[:3, 3]
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src_pts)).voxel_down_sample(HP["icp_voxel_m"])
    tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tgt_pts)).voxel_down_sample(HP["icp_voxel_m"])
    tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=HP["icp_normal_radius_m"], max_nn=HP["icp_normal_max_nn"]))
    T = P_init.copy(); res = None
    for dist in HP["icp_stages_m"]:
        res = o3d.pipelines.registration.registration_icp(src, tgt, dist, T, o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                                                         o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=HP["icp_max_iter"]))
        T = np.asarray(res.transformation).copy()
    return dict(est=T, fitness=float(res.fitness), rmse=float(res.inlier_rmse), t=time.time() - t, fail="")


# ------------------------------------------------------------------ one keyframe, all estimators
def run_keyframe(vggt, loftr, model, real, K, P0, G, rng, S=518, orbit_list=(10.0,)):
    """Returns (rows, extra).  rows: estimator -> dict(est, signals, times).  orbit_list: render orbit angles for E-V5 / x2 / V->I."""
    res = {}; t0 = time.time()
    v2 = run_V(vggt, model, real, K, P0, 1, 0.0, S); render0 = v2["render0"]
    res["E-V2"] = dict(est=v2["est_b"], t_render=v2["t_render"], t_vggt=v2["t_vggt"], fail=v2["fail"])
    # online signals at P0
    sil0 = render0[2]; sig = dict(iou_P0=iou(real[2], sil0), occ_ratio=float(real[2].sum() / max(sil0.sum(), 1)), render_px=int(sil0.sum()))
    lo = loftr.estimate(real, render0, K, P0, rng); res["E-L"] = dict(est=lo["est"], success=lo["success"], inliers=lo["inliers"], t_loftr=lo["t"])
    ic = icp(real, render0, K, P0); res["E-I"] = dict(est=ic["est"], fitness=ic["fitness"], rmse=ic["rmse"], t_icp=ic["t"], fail=ic["fail"]); sig["icp_fitness"] = ic["fitness"]; sig["icp_rmse"] = ic["rmse"]
    for deg in orbit_list:
        tag = "" if deg == 10.0 else f"_r{int(deg)}"
        v5 = run_V(vggt, model, real, K, P0, 5, deg, S)
        res[f"E-V5b{tag}"] = dict(est=v5["est_b"], t_render=v5["t_render"], t_vggt=v5["t_vggt"], fail=v5["fail"])
        res[f"E-V5d{tag}"] = dict(est=v5["est_d"], t_render=v5["t_render"], t_vggt=v5["t_vggt"], fail=v5["fail"], T1_rot=v5.get("T1_rot", np.nan), T1_trans=v5.get("T1_trans", np.nan), s_spread=v5.get("s_spread", np.nan))
        if v5["est_d"] is not None:
            P1 = v5["est_d"]; v52 = run_V(vggt, model, real, K, P1, 5, deg, S)
            P2 = v52["est_d"] if v52["est_d"] is not None else P1
            res[f"E-V5x2{tag}"] = dict(est=P2, t_render=v5["t_render"] + v52["t_render"], t_vggt=v5["t_vggt"] + v52["t_vggt"], fail=v52["fail"])
            rP2 = Renders(model, [P2]).items[0]; ic2 = icp(real, rP2, K, P2)
            res[f"E-VI{tag}"] = dict(est=ic2["est"], fitness=ic2["fitness"], rmse=ic2["rmse"], t_icp=ic2["t"], fail=ic2["fail"])
        else:
            res[f"E-V5x2{tag}"] = dict(est=None, fail=v5["fail"]); res[f"E-VI{tag}"] = dict(est=None, fail=v5["fail"])
        if tag == "":
            sig.update(T1_rot=res["E-V5d"]["T1_rot"], T1_trans=res["E-V5d"]["T1_trans"], s_spread=res["E-V5d"]["s_spread"])
    rows = []
    e0 = pose_err(P0, G)
    for name, r in res.items():
        est = r["est"] if r.get("est") is not None else P0            # failed estimators keep P0 (flagged)
        e = pose_err(est, G)
        row = dict(estimator=name, rot0=e0[0], trans0=e0[1], rot=e[0], trans=e[1], success=int(e[0] < HP["success"][0] and e[1] < HP["success"][1]),
                   strict=int(e[0] < HP["strict"][0] and e[1] < HP["strict"][1]), improved=int(e[0] < e0[0]), worsened=int(e[0] > e0[0] + HP["worsen_margin_deg"]),
                   fail=r.get("fail", "") or ("" if r.get("est") is not None else "none"), loftr_success=r.get("success", np.nan), loftr_inliers=r.get("inliers", np.nan),
                   fitness=r.get("fitness", np.nan), rmse=r.get("rmse", np.nan), t_render=r.get("t_render", 0.0), t_vggt=r.get("t_vggt", 0.0), t_loftr=r.get("t_loftr", 0.0), t_icp=r.get("t_icp", 0.0), **sig)
        row.update({f"init_{i}": v for i, v in enumerate(P0.reshape(-1))}); row.update({f"est_{i}": v for i, v in enumerate(est.reshape(-1))})
        rows.append(row)
    return rows, time.time() - t0
