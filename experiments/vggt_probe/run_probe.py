"""EXP_VGGT_PROBE1 cell runner (brief sections 6-8, 10).  VGGT runs once per batch; alignment is done twice, with
R-online (BSDF snapshot at the batch's newest new KF, chain: previous estimate) and R-oracle (GT) reference poses.
Writes <out>/kf.csv, batches.jsonl, snapshots/<frame>/keyframes.yml (online), run.log.  Main code untouched."""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np, torch, yaml
sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_lib as L
from models import load_model

KF_COLS = (["kf", "batch", "role", "has_gt", "rot_online", "trans_online", "rot_oracle", "trans_oracle", "rot_c1", "trans_c1", "rot_c2", "trans_c2",
            "s_i", "n_valid_px", "T1_rot", "T1_trans", "T1o_rot", "T1o_trans", "T2_s_cv", "T3_fov_diff", "T4_depth_agree", "T5_conf", "T6_mask_area", "T7_rot", "T7_trans",
            "used_ref_ids", "warp", "batch_time_s", "peak_mem_gb"] + [f"c2w_online_{i}" for i in range(16)] + [f"c2w_oracle_{i}" for i in range(16)])


def fov_deg(f, S): return float(2 * np.degrees(np.arctan(S / (2 * f))))


def run_cell(model, seq: L.Sequence, input_mode: str, batch_mode: str, M: int, k: int, out: Path, S: int, log):
    out.mkdir(parents=True, exist_ok=False); (out / "snapshots").mkdir()
    batches = L.make_batches(seq, batch_mode, M=M, k=k)
    est_on, est_or = {}, {}                              # kf -> c2w metric in the object frame (online / oracle alignment)
    rows = []; bj = open(out / "batches.jsonl", "w"); K = seq.K
    for bi, b in enumerate(batches):
        ids = b["ref"] + b["new"]; n_ref = len(b["ref"]); frames = [seq.frame(f) for f in ids]
        if input_mode == "I1":
            Rs = [L.centre_rotation(K, L.mask_centroid(m)) for _, _, m in frames]
            fv = L.virtual_focal(K, [m for _, _, m in frames], Rs, S); Kv = np.array([[fv, 0, S / 2], [0, fv, S / 2], [0, 0, 1.0]])
            warped = [L.warp_I1(rgb, d, m, K, R, Kv, S) for (rgb, d, m), R in zip(frames, Rs)]
            imgs = np.stack([w[0] for w in warped]); depths = [w[1] for w in warped]; masks_v = [w[2] for w in warped]; pmaps = [w[3] for w in warped]; Ks_v = [Kv] * len(ids)
        else:
            wf = L.warp_I2 if input_mode == "I2" else L.warp_I0
            warped = [wf(rgb, d, m, K, S) for (rgb, d, m) in frames]
            imgs = np.stack([w[0] for w in warped]); depths = [w[1] for w in warped]; masks_v = [w[2] for w in warped]; pmaps = [w[4] for w in warped]; Ks_v = [w[3] for w in warped]; Rs = [np.eye(3)] * len(ids)
        r = model.infer(torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous())
        # ---- scale (section 7): s_i per frame, batch s = median over all valid pixels pooled
        s_i, nvalid, t4, all_s, all_v = [], [], [], [], []
        for i in range(len(ids)):
            s, n, rho_s, rho_v = L.frame_scale(depths[i], frames[i][2], pmaps[i], K, r["depth"][i], r["conf"][i], Ks_v[i])
            s_i.append(s); nvalid.append(n)
            if rho_s is not None: all_s.append(rho_s); all_v.append(rho_v)
        if all_s:
            rs, rv = np.concatenate(all_s), np.concatenate(all_v); s_batch = float(np.median(rs / rv)); t4_batch = float(np.median(np.abs(s_batch * rv - rs) / rs))
        else:
            s_batch, t4_batch = np.nan, np.nan
        for i in range(len(ids)):
            _, _, rho_s, rho_v = L.frame_scale(depths[i], frames[i][2], pmaps[i], K, r["depth"][i], r["conf"][i], Ks_v[i]) if np.isfinite(s_i[i]) else (0, 0, None, None)
            t4.append(float(np.median(np.abs(s_batch * rho_v - rho_s) / rho_s)) if rho_s is not None and np.isfinite(s_batch) else np.nan)
        sv = np.array([x for x in s_i if np.isfinite(x)]); t2 = float(sv.std() / sv.mean()) if len(sv) >= 2 else np.nan
        t3 = float(np.mean([abs(fov_deg(r["K"][i][0, 0], S) - fov_deg(Ks_v[i][0, 0], S)) for i in range(len(ids))]))
        t5 = [float(r["conf"][i][masks_v[i]].mean()) if masks_v[i].any() else np.nan for i in range(len(ids))]
        # ---- batch-frame poses: undo virtual rotation, apply scale
        c2w_raw = [np.linalg.inv(w) for w in r["w2c"]]
        def scaled(sc):
            outp = []
            for i in range(len(ids)):
                P = c2w_raw[i].copy(); P[:3, 3] *= sc; B = np.eye(4); B[:3, :3] = Rs[i]; outp.append(P @ B)
            return outp
        c2w_b = scaled(s_batch if np.isfinite(s_batch) else 1.0)
        # ---- reference poses: online = chain estimate (B-chain) / BSDF snapshot at the newest new KF (B-ref); oracle = GT
        snap = seq.snapshot_at(b["new"][-1])
        if batch_mode == "chain":
            ref_on = [est_on.get(rr, seq.C1[rr]) for rr in b["ref"]]
        else:
            ref_on = [snap.get(rr, seq.C1[rr]) for rr in b["ref"]]
        ref_or = [seq.gt_c2w(rr) for rr in b["ref"]]
        T_on = L.align(ref_on, c2w_b[:n_ref]); c2w_on = [T_on @ P for P in c2w_b]
        ok_or = [i for i in range(n_ref) if ref_or[i] is not None]
        if ok_or:
            T_or = L.align([ref_or[i] for i in ok_or], [c2w_b[i] for i in ok_or]); c2w_or = [T_or @ P for P in c2w_b]
        else:
            T_or = None; c2w_or = [None] * len(ids)
        t1 = [L.pose_err(c2w_on[i], ref_on[i]) for i in range(n_ref)]; t1o = [L.pose_err(c2w_or[i], ref_or[i]) for i in ok_or]
        T1 = (float(np.median([x[0] for x in t1])), float(np.median([x[1] for x in t1]))); T1o = (float(np.median([x[0] for x in t1o])), float(np.median([x[1] for x in t1o]))) if t1o else (np.nan, np.nan)
        gt_all = [seq.gt_c2w(f) for f in ids]
        s_gt = L.scale_gt(c2w_raw, gt_all, T_or[:3, :3]) if T_or is not None else np.nan
        e2_on = L.pairwise_E2(c2w_on, gt_all); e2_or = L.pairwise_E2(c2w_or, gt_all) if T_or is not None else (np.nan, np.nan)
        for i, f in enumerate(ids):
            role = "ref" if i < n_ref else "new"; g = gt_all[i]
            if role == "new": est_on[f] = c2w_on[i]; est_or[f] = c2w_or[i]
            e = lambda P: L.pose_err(P, g) if (g is not None and P is not None) else (np.nan, np.nan)
            eo, er_, e1, e2 = e(c2w_on[i]), e(c2w_or[i]), e(seq.C1[f]), e(seq.C2[f]); t7 = L.pose_err(c2w_on[i], seq.C1[f])
            row = dict(kf=f, batch=bi, role=role, has_gt=int(g is not None), rot_online=eo[0], trans_online=eo[1], rot_oracle=er_[0], trans_oracle=er_[1], rot_c1=e1[0], trans_c1=e1[1], rot_c2=e2[0], trans_c2=e2[1],
                       s_i=s_i[i], n_valid_px=nvalid[i], T1_rot=T1[0], T1_trans=T1[1], T1o_rot=T1o[0], T1o_trans=T1o[1], T2_s_cv=t2, T3_fov_diff=t3, T4_depth_agree=t4[i], T5_conf=t5[i], T6_mask_area=int(frames[i][2].sum()),
                       T7_rot=t7[0], T7_trans=t7[1], used_ref_ids=" ".join(b["ref"]), warp=input_mode, batch_time_s=r["time_s"], peak_mem_gb=r["peak_mem_gb"])
            for j, v in enumerate(c2w_on[i].reshape(-1)): row[f"c2w_online_{j}"] = v
            for j in range(16): row[f"c2w_oracle_{j}"] = c2w_or[i].reshape(-1)[j] if c2w_or[i] is not None else np.nan
            rows.append(row)
        bj.write(json.dumps(dict(batch=bi, ref=b["ref"], new=b["new"], s=s_batch, s_i=s_i, n_valid_px=nvalid, s_gt=s_gt, E3_err=abs(s_batch / s_gt - 1) if np.isfinite(s_gt) and s_gt else np.nan,
                                 E2_online=e2_on, E2_oracle=e2_or, T1=T1, T1_oracle=T1o, T2=t2, T3=t3, T4=t4_batch, T5=float(np.nanmean(t5)), time_s=r["time_s"], peak_mem_gb=r["peak_mem_gb"],
                                 T_online=T_on.tolist(), T_oracle=T_or.tolist() if T_or is not None else None, K_vggt=[k_.tolist() for k_ in r["K"]], Kv=Ks_v[0].tolist(), w2c_raw=[w.tolist() for w in r["w2c"]], R_virtual=[R.tolist() for R in Rs], ids=ids)) + "\n"); bj.flush()
        snapd = {f"keyframe_{seq.kf_ids[0]}": {"cam_in_ob": seq.C1[seq.kf_ids[0]].reshape(-1).tolist()}}
        snapd.update({f"keyframe_{f}": {"cam_in_ob": est_on[f].reshape(-1).tolist()} for f in est_on})
        d = out / "snapshots" / b["new"][-1]; d.mkdir(); yaml.safe_dump(snapd, open(d / "keyframes.yml", "w"))
        nn = rows[-len(b["new"]):]
        log(f"  batch {bi}: ref={b['ref']} new={b['new']} s={s_batch:.3f} s_gt={s_gt:.3f} T1={T1[0]:.2f}deg/{T1[1]:.1f}mm T3={t3:.2f} | new E1 online rot={np.nanmedian([x['rot_online'] for x in nn]):.2f} trans={np.nanmedian([x['trans_online'] for x in nn]):.1f}"
            f" oracle rot={np.nanmedian([x['rot_oracle'] for x in nn]):.2f} trans={np.nanmedian([x['trans_oracle'] for x in nn]):.1f} | C1 rot={np.nanmedian([x['rot_c1'] for x in nn]):.2f} t={r['time_s']:.2f}s")
    bj.close()
    with open(out / "kf.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=KF_COLS); w.writeheader(); [w.writerow(rw) for rw in rows]
    return rows


def summarize(rows):
    new = [r for r in rows if r["role"] == "new" and r["has_gt"]]
    f = lambda key: float(np.nanmedian([r[key] for r in new])) if new else float("nan")
    return dict(n_new=len(new), rot_online=f("rot_online"), trans_online=f("trans_online"), rot_oracle=f("rot_oracle"), trans_oracle=f("trans_oracle"), rot_c1=f("rot_c1"), trans_c1=f("trans_c1"), rot_c2=f("rot_c2"), trans_c2=f("trans_c2"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="M-V"); ap.add_argument("--input", default="I1"); ap.add_argument("--batch", default="ref"); ap.add_argument("--M", type=int, default=5); ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--dataset", required=True); ap.add_argument("--seq", required=True); ap.add_argument("--out", required=True); ap.add_argument("--max_batches", type=int, default=0)
    a = ap.parse_args(); out = Path(a.out); model = load_model(a.model); seq = L.Sequence(a.dataset, a.seq)
    if a.max_batches: seq.kf_ids = seq.kf_ids[:5 + (a.max_batches - 1) * (a.M if a.batch == "ref" else 5)]
    t0 = time.time(); rows = run_cell(model, seq, a.input, a.batch, a.M, a.k, out, model.size, lambda m: print(m, flush=True))
    summ = summarize(rows); summ["n_kf"] = len(seq.kf_ids); summ["wall_s"] = time.time() - t0
    json.dump(dict(args=vars(a), model=model.name, ckpt=model.ckpt, ckpt_sha256_16=model.ckpt_sha, size=model.size, patch=model.patch, torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                   bsdf_run=str(seq.run), kf_ids=seq.kf_ids, summary=summ, started=time.strftime("%F %T", time.localtime(t0))), open(out / "manifest.json", "w"), indent=1)
    print(f"DONE {a.model} {a.input} {a.batch}({a.M},{a.k}) {a.dataset}/{a.seq}: new {summ['n_new']} rot online/oracle/C1/C2 = {summ['rot_online']:.2f}/{summ['rot_oracle']:.2f}/{summ['rot_c1']:.2f}/{summ['rot_c2']:.2f} "
          f"trans = {summ['trans_online']:.1f}/{summ['trans_oracle']:.1f}/{summ['trans_c1']:.1f}/{summ['trans_c2']:.1f} mm {summ['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
