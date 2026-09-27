"""EXP_BATCH_20260927 3.3 unit tests for experiments/hygiene_runner.py on mustard0 (5 initial + 10 update keyframes, GT poses, 500 / 500).
Frames are read (read-only) from the BundleSDF SAM2 baseline run (color / depth_filtered / mask / cam_K / ob_in_cam); keyframes = the first 15
of its final keyframes.yml.  (1) A1 with GT poses: promotion rate of candidates born in updates 1..6 (>= 4 later keyframes) must be >= 70 %.
(2) keyframe 5 (the first update keyframe) perturbed by 3 deg (object-centre rotation): its candidates' discard rate >= 70 %, others ~ (1).
(3) switch off (HYG_MODE None) vs the main-code GaussianRunner: Gaussian counts and losses per update (base run twice for the GPU noise level)."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, torch, yaml
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments")); sys.path.append(str(REPO / "BundleTrack/scripts"))
import exp_feedback_gradient_probe_gtmap as G  # noqa: E402
from exp_feedback_gradient_probe_gtmap import GroundTruth, init_runner_like_online, perturbed_pose, SAM3D_ROOT  # noqa: E402
from gaussian_runner import GaussianFrame, GaussianRunner  # noqa: E402
from run_gaussian_incremental import load_frame  # noqa: E402
from hygiene_runner import make_hygiene_runner  # noqa: E402

RUN = Path("/home/kist/Desktop/BundleSDF_baseline_outputs/full_eval_sam2_ycb/ycb/mustard0"); VIDEO = REPO / "datasets/YCBInEOAT/mustard0"
PP = {k: SAM3D_ROOT / "output_YCBInEOAT_sam2mask_mesh" / f"mustard0_{v}" for k, v in (("mesh_npz", "mesh_depth.npz"), ("pose_json", "mesh_depth.json"), ("gaussian_ply", "splat_depth.ply"))}
CFG = REPO / "config_gs_2dgs_1mm_lifecycle.yml"; OUT = REPO / "logs/exp_batch_20260927/unit_tests"


def kf_ids():
    import glob, os
    last = sorted(glob.glob(str(RUN / "*/keyframes.yml")), key=lambda p: os.path.basename(os.path.dirname(p)))[-1]
    return [k.replace("keyframe_", "") for k in yaml.safe_load(open(last)).keys()][:15]


def replay(cls, perturb_kf=None, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    gt = GroundTruth("ycb", VIDEO, RUN); K = np.loadtxt(RUN / "cam_K.txt").reshape(3, 3).astype(np.float32); ids = kf_ids()
    fr = lambda fid, P: (lambda f: GaussianFrame(fid, f.rgb, f.depth, f.mask, f.K, f.c2w_cv).validated())(load_frame(RUN, fid, K, P.astype(np.float32)))
    saved = G.GaussianRunner; G.GaussianRunner = cls
    try: r = init_runner_like_online(CFG, "cuda:0", PP, [fr(f, gt.gt_c2w(f)) for f in ids[:5]], gt.gt_c2w(ids[0]).astype(np.float32), 500)
    finally: G.GaussianRunner = saved
    log = []
    for i, f in enumerate(ids[5:], start=5):
        P = gt.gt_c2w(f)
        if perturb_kf is not None and i == perturb_kf: P = perturbed_pose(P, r.normalization, 3.0, 0.0, 12345)
        st = r.update([fr(f, P)], train_steps=500); log.append(dict(kf_index=i, gaussians=st.gaussians_after_train, first_loss=st.first_loss, final_loss=st.final_loss))
    return r, log


def test3_from_checkpoint():
    """The prior Sim(3) alignment is not bit-reproducible across runs, so both classes start from ONE saved initial state (checkpoint incl.
    generator / RNG states) and run the same 10 GT-pose updates; main vs main from the same checkpoint gives the GPU noise level."""
    gt = GroundTruth("ycb", VIDEO, RUN); K = np.loadtxt(RUN / "cam_K.txt").reshape(3, 3).astype(np.float32); ids = kf_ids()
    fr = lambda fid, P: (lambda f: GaussianFrame(fid, f.rgb, f.depth, f.mask, f.K, f.c2w_cv).validated())(load_frame(RUN, fid, K, P.astype(np.float32)))
    r0 = init_runner_like_online(CFG, "cuda:0", PP, [fr(f, gt.gt_c2w(f)) for f in ids[:5]], gt.gt_c2w(ids[0]).astype(np.float32), 500)
    ck = OUT / "_test3_init_checkpoint.pt"; r0.save_checkpoint(ck); del r0; torch.cuda.empty_cache(); logs = []
    for cls in (GaussianRunner, GaussianRunner, make_hygiene_runner(None)):
        r = cls.load_checkpoint(ck, device="cuda:0"); log = []
        for i, f in enumerate(ids[5:], start=5):
            st = r.update([fr(f, gt.gt_c2w(f))], train_steps=500); log.append(dict(kf_index=i, gaussians=st.gaussians_after_train, first_loss=st.first_loss, final_loss=st.final_loss))
        logs.append(log); del r; torch.cuda.empty_cache()
    ck.unlink()
    return logs


def promo(r, births):
    return {b: (r.view_promoted.get(b, 0), r.view_born.get(b, 0)) for b in births}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True); lines = []; t0 = time.time()
    if len(sys.argv) > 1 and sys.argv[1] == "--diag2":
        # diagnostic for (2): how far does the 3 deg perturbation move keyframe 5's candidate points, and does a larger error get discarded?
        out = []
        for deg in (3.0, 10.0):
            gt = GroundTruth("ycb", VIDEO, RUN); K = np.loadtxt(RUN / "cam_K.txt").reshape(3, 3).astype(np.float32); ids = kf_ids()
            fr = lambda fid, P: (lambda f: GaussianFrame(fid, f.rgb, f.depth, f.mask, f.K, f.c2w_cv).validated())(load_frame(RUN, fid, K, P.astype(np.float32)))
            saved = G.GaussianRunner; G.GaussianRunner = make_hygiene_runner("A1")
            try: r = init_runner_like_online(CFG, "cuda:0", PP, [fr(f, gt.gt_c2w(f)) for f in ids[:5]], gt.gt_c2w(ids[0]).astype(np.float32), 500)
            finally: G.GaussianRunner = saved
            disp = None
            for i, f in enumerate(ids[5:], start=5):
                P = gt.gt_c2w(f)
                if i == 5:
                    Pp = perturbed_pose(P, r.normalization, deg, 0.0, 12345); r.update([fr(f, Pp)], train_steps=500)
                    x = r.cand["pos"][r.cand["birth"] == 5]; A = Pp @ np.linalg.inv(P); xt = (x - A[:3, 3]) @ np.linalg.inv(A[:3, :3]).T
                    disp = np.linalg.norm(x - xt, axis=1) * 1000
                else:
                    r.update([fr(f, P)], train_steps=500)
            pr, bo = r.view_promoted.get(5, 0), r.view_born.get(5, 0)
            out.append(f"KF5 perturbed {deg:.0f} deg: its candidates displaced by median {np.median(disp):.2f} mm (p90 {np.percentile(disp, 90):.2f}) from their true position; "
                       f"promoted/born {pr}/{bo} -> discard rate {1 - pr / max(bo, 1):.3f}")
            del r; torch.cuda.empty_cache()
        text = "\n".join(["diagnostic for unit test (2) (A1 thresholds as specified: depth tol 2 cm, spread 5 mm, >= 4 obs, streak >= 3, angle >= 5 deg)"] + out)
        print(text); (OUT / "hygiene_unit_test2_diag.txt").write_text(text + "\n"); sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "--only3":
        lb1, lb2, lh = test3_from_checkpoint()
        dg = max(abs(a["gaussians"] - b["gaussians"]) for a, b in zip(lb1, lh)); dl = max(abs(a["final_loss"] - b["final_loss"]) for a, b in zip(lb1, lh))
        dgb = max(abs(a["gaussians"] - b["gaussians"]) for a, b in zip(lb1, lb2)); dlb = max(abs(a["final_loss"] - b["final_loss"]) for a, b in zip(lb1, lb2))
        text = (f"(3) switch off vs main code, both from one initial checkpoint, 10 GT-pose updates x 500 steps: max |dGaussians| {dg}, max |d final loss| {dl:.3e} "
                f"(main vs main from the same checkpoint: {dgb}, {dlb:.3e}) -> {'PASS' if dg <= dgb and dl <= max(dlb * 1.5, 1e-7) else 'CHECK'}\n"
                f"    counts main {[x['gaussians'] for x in lb1]}\n    counts off  {[x['gaussians'] for x in lh]}\n    final loss main {[round(x['final_loss'], 6) for x in lb1]}\n    final loss off  {[round(x['final_loss'], 6) for x in lh]}")
        print(text); (OUT / "hygiene_unit_test3.txt").write_text(text + "\n"); sys.exit(0)
    A1 = make_hygiene_runner("A1")
    r1, _ = replay(A1); p1 = promo(r1, range(5, 11)); tot1 = sum(v[0] for v in p1.values()) / max(sum(v[1] for v in p1.values()), 1)
    lines.append(f"(1) A1, GT poses: promoted/born per birth keyframe {p1}; pooled promotion rate {tot1:.3f} -> {'PASS' if tot1 >= 0.70 else 'FAIL'} (>= 0.70)")
    lines.append(f"    per-cycle log: " + json.dumps(r1.hyg_log[-3:], default=str)[:1500])
    r2, _ = replay(A1, perturb_kf=5); p2 = promo(r2, range(5, 11))
    disc5 = 1 - p2[5][0] / max(p2[5][1], 1); oth = sum(v[0] for k, v in p2.items() if k != 5) / max(sum(v[1] for k, v in p2.items() if k != 5), 1)
    oth1 = sum(v[0] for k, v in p1.items() if k != 5) / max(sum(v[1] for k, v in p1.items() if k != 5), 1)
    lines.append(f"(2) KF5 perturbed 3 deg: KF5 candidates promoted/born {p2[5]} -> discard rate {disc5:.3f} {'PASS' if disc5 >= 0.70 else 'FAIL'} (>= 0.70); "
                 f"other keyframes pooled promotion {oth:.3f} vs {oth1:.3f} in (1)")
    lb1, lb2, lh = test3_from_checkpoint()
    dg = max(abs(a["gaussians"] - b["gaussians"]) for a, b in zip(lb1, lh)); dl = max(abs(a["final_loss"] - b["final_loss"]) for a, b in zip(lb1, lh))
    dgb = max(abs(a["gaussians"] - b["gaussians"]) for a, b in zip(lb1, lb2)); dlb = max(abs(a["final_loss"] - b["final_loss"]) for a, b in zip(lb1, lb2))
    lines.append(f"(3) switch off vs main code: max |dGaussians| {dg}, max |d final loss| {dl:.2e} (main vs main: {dgb}, {dlb:.2e}) -> "
                 f"{'PASS' if dg == 0 and dl <= max(dlb, 1e-6) * 1.0 + 1e-6 else 'CHECK'}; counts per update main {[x['gaussians'] for x in lb1]} vs off {[x['gaussians'] for x in lh]}")
    lines.append(f"({time.time() - t0:.0f}s)"); text = "\n".join(lines); print(text); (OUT / "hygiene_unit_tests.txt").write_text(text + "\n")
