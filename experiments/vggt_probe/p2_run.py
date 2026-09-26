"""Probe 2 stage A (initial pose = C1, all HO3D keyframes with GT) and stage B (initial pose = GT rotated by delta about a random axis through the
GT-mesh centre + N(0, 1 cm) translation noise; 40 fixed-seed keyframes of AP12 / MPM12 / SM1; delta in {5,10,20,30,45,60}; E-V5 orbits 10 and 20 deg).
Writes outputs/exp_vggt_probe2/<stage>/<model>/<seq>/{kf.csv, manifest.json}; skips sequences whose output exists."""
from __future__ import annotations
import argparse, json, time, shutil
import numpy as np, pandas as pd
from p2_lib import *

B_SEQS = ["AP12", "MPM12", "SM1"]; DELTAS = [5, 10, 20, 30, 45, 60]; N_B = 40


def gt_mesh_center(seq):
    from eval_mesh_cd import load_gt_mesh
    m = load_gt_mesh(seq.ds, seq.sq); A = seq.align(); V = np.asarray(m.vertices, dtype=np.float64); return (V @ A[:3, :3].T + A[:3, 3]).mean(0)


def run_seq(stage, kind, sq, vggt, lo, every, log):
    out = OUT / stage / kind / sq
    if (out / "manifest.json").exists(): log(f"SKIP {stage} {kind} {sq}"); return
    if out.exists(): shutil.move(str(out), str(out) + f"_partial_{int(time.time())}")
    out.mkdir(parents=True); seq = SeqData("ho3d", sq); model = load_model(kind, seq); t0 = time.time(); rows = []
    ids = [k for k in seq.kf_ids if seq.GT.get(k) is not None]
    if stage == "A":
        jobs = [(k, None, seq.C1[k]) for k in ids[::every]]; orbits = (10.0,)
    else:
        c_gt = gt_mesh_center(seq); pick = sorted(np.random.default_rng(12345).choice(len(ids), min(N_B, len(ids)), replace=False)); jobs = []
        for d in DELTAS:
            for j, i in enumerate(pick):
                rng = np.random.default_rng(100000 * d + j); jobs.append((ids[i], d, perturb(seq.GT[ids[i]], c_gt, float(d), rng, 0.01)))
        orbits = (10.0, 20.0)
    for n, (k, d, P0) in enumerate(jobs):
        real = seq.frame(k)
        rr, dt = run_keyframe(vggt, lo, model, real, seq.K, P0, seq.GT[k], np.random.default_rng(7 + n), S=HP["S"], orbit_list=orbits)
        for r in rr: r.update(stage=stage, model=kind, seq=sq, kf=k, delta=d if d is not None else np.nan, t_kf=dt); rows.append(r)
        if n % 25 == 0:
            g = pd.DataFrame(rows[-len(rr) * min(25, n + 1):]); log(f"  {stage} {kind} {sq} {n+1}/{len(jobs)} {dt:.1f}s/kf | " + " ".join(f"{e}={v:.2f}" for e, v in g.groupby('estimator').rot.median().items()) + f" | rot0 {g.rot0.median():.2f}")
        if n % 200 == 0: pd.DataFrame(rows).to_csv(out / "kf.partial.csv", index=False)
    df = pd.DataFrame(rows); df.to_csv(out / "kf.csv", index=False); (out / "kf.partial.csv").unlink(missing_ok=True)
    summ = {e: dict(rot_med=float(g.rot.median()), success=float(g.success.mean()), worsened=float(g.worsened.mean())) for e, g in df.groupby("estimator")}
    json.dump(dict(stage=stage, model=kind, model_source=model.source, model_center=model.center.tolist(), seq=sq, n_jobs=len(jobs), every=every, hp=HP,
                   vggt_ckpt=vggt.ckpt, vggt_ckpt_sha256_16=vggt.ckpt_sha, seconds=time.time() - t0, rot0_median=float(df.rot0.median()), summary=summ, finished=time.strftime("%F %T")),
              open(out / "manifest.json", "w"), indent=1, default=str)
    log(f"DONE {stage} {kind} {sq} jobs {len(jobs)} {time.time()-t0:.0f}s | " + " ".join(f"{e}={v['rot_med']:.2f}/{100*v['success']:.0f}%" for e, v in summ.items()) + f" | rot0 {df.rot0.median():.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True); ap.add_argument("--models", required=True); ap.add_argument("--seqs", default=None); ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--tag", default=""); a = ap.parse_args()
    from models import load_model as load_vggt
    vggt = load_vggt("M-V"); lo = Loftr(); prog = open(LOG / f"progress_{a.stage}{a.tag}.txt", "a")
    def log(m): print(m, flush=True); prog.write(m + "\n"); prog.flush()
    seqs = a.seqs.split(",") if a.seqs else (HO3D if a.stage == "A" else B_SEQS)
    log(f"START {a.stage} models={a.models} seqs={seqs} every={a.every} {time.strftime('%F %T')}")
    for kind in a.models.split(","):
        for sq in seqs:
            for attempt in range(2):
                try: run_seq(a.stage, kind, sq, vggt, lo, a.every, log); break
                except Exception as ex:
                    import traceback; log(f"FAIL {a.stage} {kind} {sq} attempt {attempt}: {ex}\n{traceback.format_exc()}")
    log(f"END {time.strftime('%F %T')}")
