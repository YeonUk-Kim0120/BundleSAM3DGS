"""MD-prior: the aligned SAM3D prior as the 9/22 Q2 harness (exp_hybrid_realistic_replay) holds it right after initialisation, i.e.
before the first keyframe append.  Mirrors exp_spring_anchor.build_runner line by line (same config, same 5 frames with the snapshot-of-KF4
poses, same Sim(3) alignment on KF0, 500 initial steps) but also returns the refined Sim(3) so that it can be recorded.
Writes outputs/exp_vggt_probe2/models/prior/<seq>/{checkpoint_prior.pt, manifest.json}."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from p2_lib import REPO, OUT, SeqData, HO3D
import exp_spring_anchor as E
from gaussian_runner import SceneNormalization, load_gaussian_config
from run_gaussian_incremental import load_frame

CFG = REPO / "config_gs_2dgs_1mm_lifecycle.yml"   # exp_hybrid_realistic_replay default (9/22 commands did not override it)


def build(sq, device="cuda:0"):
    from run_sam3d_alignment import DEFAULT_CONFIG as ALIGN_DEFAULTS, align_prior_sim3, prepare_target
    from sam3d_prior import load_mesh_prior, load_sam3d_gaussian_ply, load_sam3d_pose_or_refined, sample_surfels, transfer_gaussian_colors, transform_surfels_canonical_to_cv_camera
    seq = SeqData("ho3d", sq); K = np.asarray(seq.K, dtype=np.float32); kf_ids = seq.kf_ids
    snap5 = seq.snaps[seq.frame_index[kf_ids[4]]][1]
    frames0 = [load_frame(seq.run, i, K, snap5[i].astype(np.float32)) for i in kf_ids[:5]]; cam_in_obs0 = snap5[kf_ids[0]].astype(np.float32)
    sub = "output_HO3D_sam2mask_mesh"
    pp = {k: E.SAM3D_ROOT / sub / f"{sq}_{v}" for k, v in (("mesh_npz", "mesh_depth.npz"), ("pose_json", "mesh_depth.json"), ("gaussian_ply", "splat_depth.ply"))}
    # ---- exp_spring_anchor.build_runner (verbatim logic)
    dev = torch.device(device); torch.cuda.set_device(dev)
    rc = load_gaussian_config(CFG); rc["device"] = device
    prior = load_mesh_prior(pp["mesh_npz"]); init_pose = load_sam3d_pose_or_refined(pp["pose_json"])
    surfels = sample_surfels(prior, 20000, seed=0, radius_multiplier=0.75)
    surfels, _ = transfer_gaussian_colors(surfels, load_sam3d_gaussian_ply(pp["gaussian_ply"]))
    ac = dict(ALIGN_DEFAULTS); ac["use_ssim"] = True; ac["use_ms_ssim"] = False
    f0 = frames0[0]
    target = prepare_target({"frame_id": f0.frame_id, "rgb": f0.rgb, "depth": f0.depth, "mask": f0.mask, "K": f0.K}, ac, dev)
    refined, _, _, status = align_prior_sim3(surfels, init_pose, target, ac, dev, verbose=False)
    scv = transform_surfels_canonical_to_cv_camera(surfels, refined)
    c0 = np.asarray(cam_in_obs0, dtype=np.float64)
    m = scv.means.numpy() @ c0[:3, :3].T + c0[:3, 3][None]
    centre = m.mean(0); radius = float(np.linalg.norm(m - centre, axis=1).max())
    norm = SceneNormalization(scale=1.0 / max(radius * 1.2, 1e-6), translation=-centre)
    r = E.BirthRunner(rc, norm, device=device)
    r.initialize_from_prior(scv, cam_in_obs0, frames0, train_steps=500)
    # ----
    out = OUT / "models" / "prior" / sq; out.mkdir(parents=True, exist_ok=False)
    r.save_checkpoint(out / "checkpoint_prior.pt")
    sim3 = {k: (v.detach().cpu().numpy().tolist() if torch.is_tensor(v) else v) for k, v in vars(refined).items()} if hasattr(refined, "__dict__") else str(refined)
    json.dump(dict(seq=sq, prior_paths={k: str(v) for k, v in pp.items()}, runner_config=str(CFG), init_frames=kf_ids[:5], init_pose_source="snapshot of KF4 (as exp_hybrid_realistic_replay)",
                   cam_in_obs0=c0.tolist(), refined_sim3_canonical_to_cv_camera_of_KF0=sim3, align_status=str(status), gaussians=int(r.num_gaussians), initial_steps=500,
                   checkpoint=str(out / "checkpoint_prior.pt"), finished=time.strftime("%F %T")), open(out / "manifest.json", "w"), indent=1)
    print(f"prior {sq}: {r.num_gaussians} Gaussians, status {status}", flush=True)
    del r; torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seqs", default=",".join(HO3D)); a = ap.parse_args()
    for sq in a.seqs.split(","):
        if (OUT / "models" / "prior" / sq / "checkpoint_prior.pt").exists(): print("skip", sq); continue
        for attempt in range(2):
            try: build(sq); break
            except Exception as ex:
                import traceback; print(f"FAIL {sq} attempt {attempt}: {ex}\n{traceback.format_exc()}", flush=True)
                import shutil; d = OUT / "models" / "prior" / sq
                if d.exists(): shutil.move(str(d), str(d) + f"_failed{attempt}")
