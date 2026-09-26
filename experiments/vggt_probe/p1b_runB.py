"""Probe 1b stage B runner (M-V only).  Conditions: B1-k8, B1-k4 (nearest references), B2 (baseline bands), B1-roll (B1-k8 + roll
normalisation), B4-black, B4-gray (background), B5 (no-mask chain, negative control).  Outputs Probe-1-style kf.csv, batches.jsonl,
pairs.csv, manifest.json under outputs/exp_vggt_probe1b/B/<cond>/<seq>/.  Alignment (a) and (d) × online/oracle."""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import numpy as np, torch
import probe_lib as L
from models import load_model
from p1b_common import *
from p1b_B_lib import select_nearest, select_bands, roll_psi, Rz, warp_I1_bg, warp_I0_nomask
from p1b_A1 import COLS as PAIR_COLS

KF_COLS = ["kf", "batch", "role", "has_gt", "rot_a_online", "trans_a_online", "rot_d_online", "trans_d_online", "rot_a_oracle", "trans_a_oracle", "rot_d_oracle", "trans_d_oracle",
           "rot_c1", "trans_c1", "rot_c2", "trans_c2", "s_i", "n_valid_px", "T5_conf", "T6_mask_area", "psi_deg", "psi_flag"]


def ptype(ri, rj):
    if ri == "new" and rj == "new": return "new-new"
    if ri == "new": return "new-" + rj
    if rj == "new": return "new-" + ri
    return "ref-ref"


def run(cond, ds, sq, out, S_img, log):
    S = load_seq(ds, sq); seq = S_img; K = S.K; ids_all = S.kf_ids; model = load_model("M-V"); SZ = model.size
    out.mkdir(parents=True, exist_ok=False); bj = open(out / "batches.jsonl", "w"); rows = []; prow = []
    k = 4 if cond == "B1-k4" else 8; bg = {"B4-black": 0.0, "B4-gray": 0.5}.get(cond, 1.0); use_roll = cond == "B1-roll"
    # ---- batch list
    batches = []
    if cond == "B5":
        i = 5; batches.append(dict(ref=[ids_all[0]], new=ids_all[1:5], roles=None, comp=None))
        while i < len(ids_all): batches.append(dict(ref=[ids_all[i - 1]], new=ids_all[i:i + 5], roles=None, comp=None)); i += 5
    else:
        batches.append(dict(ref=[ids_all[0]], new=ids_all[1:5], roles={ids_all[0]: "first", **{n: "new" for n in ids_all[1:5]}}, comp=None))
        for i in range(5, len(ids_all)):
            new = [ids_all[i]]; cand = ids_all[:i]; snap = S.snapshot_at(new[-1]); ref_pose = {c: snap.get(c, S.C1[c]) for c in cand}
            if cond == "B2":
                refs, comp = select_bands(new, cand, ref_pose, S.C1); roles = {}
                for band, lst in comp.items():
                    for r in lst: roles[r] = "first" if band == "first" else ("fill" if band == "fill" else "band" + band)
            else:
                refs = select_nearest(new, cand, ref_pose, S.C1, k); comp = None; roles = {refs[0]: "first", **{r: "near" for r in refs[1:]}}
            roles[new[0]] = "new"; batches.append(dict(ref=refs, new=new, roles=roles, comp=comp))
    # ---- run
    for bi, b in enumerate(batches):
        ids = b["ref"] + b["new"]; n_ref = len(b["ref"]); frames = [seq.frame(f) for f in ids]; snap = S.snapshot_at(b["new"][-1])
        ref_on = [snap.get(r, S.C1[r]) for r in b["ref"]]
        if cond == "B5":
            w = [warp_I0_nomask(rgb, d, m, K, SZ) for (rgb, d, m) in frames]; imgs = np.stack([x[0] for x in w]); depths = [x[1] for x in w]; masks_v = [x[2] for x in w]; Ks_v = [x[3] for x in w]; pmaps = [x[4] for x in w]; Rs = [np.eye(3)] * len(ids); psis = [(0.0, 1.0, False)] * len(ids)
        else:
            Rc = [L.centre_rotation(K, L.mask_centroid(m)) for _, _, m in frames]; psis = [(0.0, 1.0, False)] * len(ids)
            if use_roll:
                R_first = ref_on[0][:3, :3]; Rs = []; psis = []
                for i, f in enumerate(ids):
                    Ri = (ref_on[i] if i < n_ref else S.C1[f])[:3, :3]; psi, proj, flag = roll_psi(R_first, Ri, Rc[i]); Rs.append(Rz(psi) @ Rc[i]); psis.append((float(np.degrees(psi)), proj, flag))
            else:
                Rs = Rc
            fv = L.virtual_focal(K, [m for _, _, m in frames], Rs, SZ); Kv = np.array([[fv, 0, SZ / 2], [0, fv, SZ / 2], [0, 0, 1.0]])
            w = [warp_I1_bg(rgb, d, m, K, R, Kv, SZ, bg) for (rgb, d, m), R in zip(frames, Rs)]; imgs = np.stack([x[0] for x in w]); depths = [x[1] for x in w]; masks_v = [x[2] for x in w]; pmaps = [x[3] for x in w]; Ks_v = [Kv] * len(ids)
        r = model.infer(torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous())
        s_i, nvalid, all_s, all_v = [], [], [], []
        for i in range(len(ids)):
            s, n, rho_s, rho_v = L.frame_scale(depths[i], frames[i][2], pmaps[i], K, r["depth"][i], r["conf"][i], Ks_v[i]); s_i.append(s); nvalid.append(n)
            if rho_s is not None: all_s.append(rho_s); all_v.append(rho_v)
        s_batch = float(np.median(np.concatenate(all_s) / np.concatenate(all_v))) if all_s else np.nan
        c2w_raw = [np.linalg.inv(w_) for w_ in r["w2c"]]; c2w_b = []
        for i in range(len(ids)):
            P = c2w_raw[i].copy(); P[:3, 3] *= s_batch if np.isfinite(s_batch) else 1.0; c2w_b.append(P @ blockdiag(Rs[i]))
        roles = b["roles"] or {**{b["ref"][0]: "first"}, **{n: "new" for n in b["new"]}}; roles_ref = [roles[x] for x in b["ref"]]
        ok_or = [i for i, x in enumerate(b["ref"]) if S.GT.get(x) is not None]; ref_or = [S.GT[b["ref"][i]] for i in ok_or]
        Ts = {}
        for src in ["online", "oracle"]:
            for mode in ["a", "d"]:
                if src == "online": Ts[(src, mode)] = align_variant(mode, ref_on, c2w_b[:n_ref], roles_ref)
                else: Ts[(src, mode)] = align_variant(mode, ref_or, [c2w_b[i] for i in ok_or], [roles_ref[i] for i in ok_or]) if ok_or else None
        gt_all = [S.GT.get(f) for f in ids]; s_gt = L.scale_gt(c2w_raw, gt_all, Ts[("oracle", "a")][:3, :3]) if Ts[("oracle", "a")] is not None else np.nan
        t5 = {f: float(r["conf"][i][masks_v[i]].mean()) if masks_v[i].any() else np.nan for i, f in enumerate(ids)}; t6 = {f: int(frames[i][2].sum()) for i, f in enumerate(ids)}
        for i, f in enumerate(ids):
            g = gt_all[i]; role = roles[f]
            e = lambda T: L.pose_err(T @ c2w_b[i], g) if (g is not None and T is not None) else (np.nan, np.nan)
            ea, ed, eao, edo = e(Ts[("online", "a")]), e(Ts[("online", "d")]), e(Ts[("oracle", "a")]), e(Ts[("oracle", "d")]); e1 = L.pose_err(S.C1[f], g) if g is not None else (np.nan, np.nan); e2 = L.pose_err(S.C2[f], g) if g is not None else (np.nan, np.nan)
            rows.append(dict(kf=f, batch=bi, role=role, has_gt=int(g is not None), rot_a_online=ea[0], trans_a_online=ea[1], rot_d_online=ed[0], trans_d_online=ed[1], rot_a_oracle=eao[0], trans_a_oracle=eao[1], rot_d_oracle=edo[0], trans_d_oracle=edo[1],
                             rot_c1=e1[0], trans_c1=e1[1], rot_c2=e2[0], trans_c2=e2[1], s_i=s_i[i], n_valid_px=nvalid[i], T5_conf=t5[f], T6_mask_area=t6[f], psi_deg=psis[i][0], psi_flag=int(psis[i][2])))
        # pairs (gauge-free, from c2w_b)
        cell = dict(model="M-V", input=("I1" if cond != "B5" else "I0nomask"), batch=cond, ds=ds, seq=sq)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                fi, fj = ids[i], ids[j]; gi, gj = S.GT.get(fi), S.GT.get(fj)
                if gi is None or gj is None: continue
                rg = rel(gi, gj); re_ = rel(c2w_b[i], c2w_b[j]); sw, tw = swing_twist(rg[:3, :3]); r2 = rel(S.C2[fi], S.C2[fj]); r1 = rel(S.C1[fi], S.C1[fj])
                prow.append([cell["model"], cell["input"], cond, ds, sq, bi, fi, fj, roles[fi], roles[fj], ptype(roles[fi], roles[fj]), sw, tw, rot_deg(re_[:3, :3], rg[:3, :3]), tdir_deg(re_[:3, 3], rg[:3, 3]),
                             rot_deg(r2[:3, :3], rg[:3, :3]), rot_deg(r1[:3, :3], rg[:3, :3]), t6[fi], t6[fj], t5[fi], t5[fj], np.nan])
        bj.write(json.dumps(dict(batch=bi, ref=b["ref"], new=b["new"], ids=ids, roles=[roles[f] for f in ids], composition=b["comp"], s=s_batch, s_i=s_i, n_valid_px=nvalid, s_gt=s_gt, E3_err=abs(s_batch / s_gt - 1) if np.isfinite(s_gt) and s_gt else np.nan,
                                 psi=[p[0] for p in psis], psi_proj=[p[1] for p in psis], psi_flag=[int(p[2]) for p in psis], T_online_a=Ts[("online", "a")].tolist(), T_online_d=Ts[("online", "d")].tolist(),
                                 T_oracle_a=Ts[("oracle", "a")].tolist() if Ts[("oracle", "a")] is not None else None, T_oracle_d=Ts[("oracle", "d")].tolist() if Ts[("oracle", "d")] is not None else None,
                                 w2c_raw=[w_.tolist() for w_ in r["w2c"]], R_virtual=[R.tolist() for R in Rs], K_vggt=[k_.tolist() for k_ in r["K"]], time_s=r["time_s"], peak_mem_gb=r["peak_mem_gb"])) + "\n"); bj.flush()
        nn = rows[-len(b["new"]):]
        log(f"  {cond} {sq} batch {bi}: refs={b['ref']} new={b['new']} s={s_batch:.3f} | new rot a/d online {np.nanmedian([x['rot_a_online'] for x in nn]):.2f}/{np.nanmedian([x['rot_d_online'] for x in nn]):.2f} oracle {np.nanmedian([x['rot_a_oracle'] for x in nn]):.2f}/{np.nanmedian([x['rot_d_oracle'] for x in nn]):.2f} C1 {np.nanmedian([x['rot_c1'] for x in nn]):.2f}")
    bj.close()
    with open(out / "kf.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=KF_COLS); w.writeheader(); [w.writerow(x) for x in rows]
    with open(out / "pairs.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(PAIR_COLS); w.writerows(prow)
    new = [x for x in rows if x["role"] == "new" and x["has_gt"]]
    summ = {key: float(np.nanmedian([x[key] for x in new])) for key in ["rot_a_online", "rot_d_online", "rot_a_oracle", "rot_d_oracle", "rot_c1", "rot_c2"]}; summ["n_new"] = len(new); summ["n_pairs"] = len(prow)
    json.dump(dict(cond=cond, model="M-V", ckpt=model.ckpt, ckpt_sha256_16=model.ckpt_sha, size=SZ, ds=ds, seq=sq, bsdf_run=S.run, k=k, bg=bg, roll=use_roll, torch=torch.__version__, gpu=torch.cuda.get_device_name(0), summary=summ, finished=time.strftime("%F %T")), open(out / "manifest.json", "w"), indent=1)
    return summ


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--conds", required=True); ap.add_argument("--seqs", required=True); ap.add_argument("--stage", default="B"); a = ap.parse_args()
    prog = open(LOG / f"progress_{a.stage}.txt", "a")
    def log(m): print(m, flush=True); prog.write(m + "\n"); prog.flush()
    seqs = a.seqs.split(","); conds = a.conds.split(",")
    for sq in seqs:
        ds = "ho3d" if sq in HO3D else "ycb"; seq_img = L.Sequence(ds, sq)
        for cond in conds:
            out = P1B / "B" / cond / sq
            if out.exists(): log(f"SKIP {cond} {sq}"); continue
            t0 = time.time()
            for attempt in range(2):
                try:
                    summ = run(cond, ds, sq, out, seq_img, log)
                    log(f"DONE {cond} {sq} new={summ['n_new']} rot a/d online={summ['rot_a_online']:.2f}/{summ['rot_d_online']:.2f} oracle={summ['rot_a_oracle']:.2f}/{summ['rot_d_oracle']:.2f} C1={summ['rot_c1']:.2f} C2={summ['rot_c2']:.2f} {time.time()-t0:.0f}s"); break
                except Exception as ex:
                    import traceback, shutil; log(f"FAIL {cond} {sq} attempt {attempt}: {ex}\n{traceback.format_exc()}")
                    if out.exists(): shutil.move(str(out), str(out) + f"_failed{attempt}")
    log("END")
