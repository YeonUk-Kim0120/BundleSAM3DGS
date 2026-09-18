"""Per-write-back analysis of the v1 pose feedback against the tracker's motion and error jumps (2026-09-17, user request).

Every write-back of the adopted runs (outputs/exp_fusion_20260915/adopt) is one event.  The tracker waits for the backend,
so a write-back lands in the same frame as the keyframe it consumed (verified per run: 'lag' column, expected 0).  Per event:
  - motion at the keyframe, as the mean displacement of the object's model points between the previous frame and the
    keyframe (mm/frame; rotation and translation combined) and the rotation (deg/frame): GT motion, the on-run tracker's
    own motion (its keyframe pose is saved AFTER the write-back, so this contains the newest-view delta), the pure tracker
    motion averaged over the 5 frames before the keyframe, and the tracker-alone run's motion at the same frame;
  - rise_on = ADD_on(f-1) - ADD_on(f-6): the on-run tracker's error jump over the 5 frames before the write-back (pre-feedback);
    rise_off = the same for the tracker-alone run over the same frames;
  - the write-back delta of the newest view and the max over views (gs_online/feedback_log.json), RANSAC inliers vs the
    previous frame (run log);
  - immediate effect = [ADD_on(f) - ADD_on(f-1)] - [ADD_off(f) - ADD_off(f-1)]  (f = keyframe frame);
  - cycle effect     = [ADD_on(f'-1) - ADD_on(f-1)] - [ADD_off(f'-1) - ADD_off(f-1)]  (f' = next write-back frame): the
    on-off gap change over exactly the frames this write-back influences; negative = the feedback helped.
No thresholds: groups are helped (cycle effect < 0) vs hurt (> 0), deciles of motion / rise, and top-10 rankings.
ADD curves are taken from logs/exp_fusion_20260915/add_over_frames/per_frame_add.json (first-frame aligned, as the evaluator).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments")); sys.path.append(str(REPO / "BundleTrack/scripts"))

HO = ["AP10", "AP11", "AP12", "AP13", "AP14", "MPM10", "MPM11", "MPM12", "MPM13", "MPM14", "SB11", "SB13", "SM1"]
YC = ["bleach0", "bleach_hard_00_03_chaitanya", "cracker_box_reorient", "cracker_box_yalehand0", "mustard0",
      "mustard_easy_00_02", "sugar_box1", "sugar_box_yalehand0", "tomato_soup_can_yalehand0"]


def load_poses(run_dir: Path):
    files = sorted(glob.glob(str(run_dir / "ob_in_cam" / "*.txt")))
    ids = [os.path.splitext(os.path.basename(f))[0] for f in files]
    return ids, np.array([np.loadtxt(f).reshape(4, 4) for f in files])


def gt_and_model(ds: str, video_dir: Path):
    if ds == "ho3d":
        from data_reader import Ho3dReader  # noqa: E402
        r = Ho3dReader(str(video_dir))
        G = [None if r.get_gt_pose(i) is None else np.asarray(r.get_gt_pose(i), dtype=np.float64) for i in range(len(r.color_files))]
        pts = np.asarray(r.get_gt_mesh().vertices, dtype=np.float64)
    else:
        import trimesh
        from eval_add_ycbineoat import VIDEO_TO_OBJECT  # noqa: E402
        files = sorted(glob.glob(str(video_dir / "annotated_poses" / "*")))
        G = [np.loadtxt(f).reshape(4, 4) for f in files]
        pts = np.asarray(trimesh.load(str(REPO / "datasets/YCB_Video_Models/models" / VIDEO_TO_OBJECT[video_dir.name] / "textured_simple.obj")).vertices, dtype=np.float64)
    rng = np.random.RandomState(0)
    if len(pts) > 2000:
        pts = pts[rng.choice(len(pts), 2000, replace=False)]
    return G, pts


def motion(P_list, pts):
    """Mean displacement of the model points (mm) and rotation (deg) between consecutive frames; nan where unavailable."""
    n = len(P_list); dt = np.full(n, np.nan); dr = np.full(n, np.nan)
    hom = np.c_[pts, np.ones(len(pts))]
    prev_pts = None; prev_R = None
    for i in range(n):
        P = P_list[i]
        if P is None:
            prev_pts = None; continue
        cur = (P @ hom.T).T[:, :3]
        if prev_pts is not None:
            dt[i] = np.linalg.norm(cur - prev_pts, axis=1).mean() * 1000.0
            R = P[:3, :3] @ prev_R.T
            dr[i] = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        prev_pts, prev_R = cur, P[:3, :3]
    return dt, dr


def parse_log(path: Path):
    cur = None; events = []; inl = {}
    for line in open(path, errors="ignore"):
        m = re.search(r"processNewFrame done (\S+)", line)
        if m:
            cur = m.group(1); continue
        m = re.search(r"synced pose from nerf, latest nerf frame (\S+)", line)
        if m:
            events.append((m.group(1), cur)); continue
        m = re.search(r"ransac makes match betwee frame (\S+) (\S+) #inliers=(\d+)", line)
        if m:
            inl.setdefault(m.group(1), {})[m.group(2)] = int(m.group(3))
    return events, inl


def analyze_sequence(ds, seq, run_root, off_root, log_dir, add_curves):
    video_dir = Path("datasets/HO3D_v3/evaluation") / seq if ds == "ho3d" else Path("datasets/YCBInEOAT") / seq
    run_dir = run_root / ds / seq
    offs = sorted(glob.glob(str(off_root / f"fbabl_{ds}_{seq}_off_*"))) or (sorted(glob.glob(str(off_root / "ablation_fb_off_r1_*"))) if seq == "mustard0" else [])
    log = log_dir / f"run_adopt_{ds}_{seq}.log"
    if not offs or not log.exists() or not (run_dir / "ob_in_cam").exists():
        print("skip", ds, seq, "missing inputs"); return None
    ids, P_on = load_poses(run_dir); ids_off, P_off = load_poses(Path(offs[-1]))
    assert ids == ids_off, f"{seq}: on/off frame sets differ"
    idx_of = {s: i for i, s in enumerate(ids)}; idx_of.update({s.lstrip("0") or "0": i for i, s in enumerate(ids)})
    G, pts = gt_and_model(ds, video_dir)
    n = len(ids); assert len(G) == n, f"{seq}: GT count {len(G)} vs frames {n}"
    i0 = next(i for i in range(n) if G[i] is not None)
    A_on = [P_on[i] @ np.linalg.inv(P_on[i0]) @ G[i0] for i in range(n)]
    A_off = [P_off[i] @ np.linalg.inv(P_off[i0]) @ G[i0] for i in range(n)]
    mv_gt, rot_gt = motion(G, pts); mv_on, rot_on = motion(A_on, pts); mv_off, rot_off = motion(A_off, pts)
    c = add_curves[seq]; add_on = np.full(n, np.nan); add_off = np.full(n, np.nan)
    for i, a, b in zip(c["ids"], c["add_on"], c["add_off"]):
        add_on[i] = a; add_off[i] = b
    events, inl = parse_log(log)
    fb = json.load(open(run_dir / "gs_online" / "feedback_log.json"))["records"]
    if len(fb) != len(events):
        print(f"WARN {seq}: feedback records {len(fb)} vs synced events {len(events)}")
    m = min(len(fb), len(events)); rows = []
    for j in range(m):
        kf, landing = events[j]; f = idx_of[kf]; lag = idx_of[landing] - f if landing in idx_of else np.nan
        f_next = idx_of[events[j + 1][0]] if j + 1 < m else None
        rec = fb[j]
        prev_id = ids[f - 1] if f >= 1 else None
        inliers = inl.get(kf, inl.get(kf.lstrip("0") or "0", {})).get(prev_id, None) if prev_id else None
        if inliers is None and prev_id:
            inliers = inl.get(kf, {}).get(prev_id.lstrip("0") or "0", None)
        def at(arr, i):
            return float(arr[i]) if (i is not None and 0 <= i < n) else np.nan
        rise_on = at(add_on, f - 1) - at(add_on, f - 6); rise_off = at(add_off, f - 1) - at(add_off, f - 6)
        imm = (at(add_on, f) - at(add_on, f - 1)) - (at(add_off, f) - at(add_off, f - 1))
        cyc = ((at(add_on, f_next - 1) - at(add_on, f - 1)) - (at(add_off, f_next - 1) - at(add_off, f - 1))) if f_next is not None else np.nan
        rows.append({"dataset": ds, "seq": seq, "j": j, "event": rec["event"], "kf": kf, "frame": f, "lag": lag, "next_frame": f_next if f_next is not None else -1,
                     "cycle_len": (f_next - f) if f_next is not None else np.nan,
                     "gt_move_mm": at(mv_gt, f), "gt_rot_deg": at(rot_gt, f), "gt_move_pre5_mm": float(np.nanmean(mv_gt[max(0, f - 5):f])) if f > 0 else np.nan,
                     "on_move_mm": at(mv_on, f), "on_rot_deg": at(rot_on, f), "on_move_pre5_mm": float(np.nanmean(mv_on[max(0, f - 5):f])) if f > 0 else np.nan,
                     "off_move_mm": at(mv_off, f),
                     "rise_on_cm": rise_on, "rise_off_cm": rise_off,
                     "delta_new_mm": float(rec["per_view_trans_mm"][-1]), "delta_new_deg": float(rec["per_view_rot_deg"][-1]),
                     "delta_max_mm": float(rec["max_trans_mm"]), "inliers_prev": inliers if inliers is not None else np.nan,
                     "add_on_pre": at(add_on, f - 1), "add_off_pre": at(add_off, f - 1), "gap_pre": at(add_on, f - 1) - at(add_off, f - 1),
                     "imm_effect_cm": imm, "cycle_effect_cm": cyc})
    first, last = rows[0], rows[-1]
    sanity = {"sum_cycle_effects": float(np.nansum([r["cycle_effect_cm"] for r in rows])),
              "gap_change_first_to_last": float((last["gap_pre"] - first["gap_pre"])), "max_lag": float(np.nanmax([r["lag"] for r in rows])), "events": m,
              "ADD_on": float(np.nanmean(add_on)), "ADD_off": float(np.nanmean(add_off))}
    return rows, sanity, (add_on, add_off)


def q(a, p):
    a = np.asarray(a, dtype=np.float64); a = a[~np.isnan(a)]
    return float(np.percentile(a, p)) if len(a) else np.nan


def fmt_group(rows, label):
    cyc = np.array([r["cycle_effect_cm"] for r in rows]); ok = ~np.isnan(cyc); rows = [r for r, k in zip(rows, ok) if k]; cyc = cyc[ok]
    def med(key):
        return f"{q([r[key] for r in rows], 50):.2f} [{q([r[key] for r in rows], 25):.2f}, {q([r[key] for r in rows], 75):.2f}]"
    return (f"| {label} | {len(rows)} | {cyc.mean():+.3f} | {cyc.sum():+.2f} | {med('gt_move_mm')} | {med('gt_rot_deg')} | {med('on_move_pre5_mm')} | {med('on_move_mm')} | "
            f"{med('rise_on_cm')} | {med('delta_new_mm')} | {med('inliers_prev')} | {med('gap_pre')} |")


GROUP_HEADER = ("| group | n | mean cycle effect (cm) | sum (cm) | GT motion mm/frame median [q25,q75] | GT rotation deg/frame | tracker motion, 5 frames before kf | "
                "tracker motion at kf (contains delta) | rise_on cm (5 frames before) | delta newest view mm | inliers vs prev | gap before (on−off, cm) |\n"
                "|---|---|---|---|---|---|---|---|---|---|---|---|")


def decile_table(rows, key, label, unit):
    vals = np.array([r[key] for r in rows]); cyc = np.array([r["cycle_effect_cm"] for r in rows]); imm = np.array([r["imm_effect_cm"] for r in rows])
    ok = ~np.isnan(vals) & ~np.isnan(cyc); vals, cyc, imm = vals[ok], cyc[ok], imm[ok]
    order = np.argsort(vals); out = [f"| decile of {label} | range ({unit}) | n | mean cycle effect (cm) | helped (cycle < 0) % | mean immediate effect (cm) |", "|---|---|---|---|---|---|"]
    for d in range(10):
        sel = order[int(d * len(order) / 10):int((d + 1) * len(order) / 10)]
        if len(sel) == 0: continue
        v, cc, ii = vals[sel], cyc[sel], imm[sel]
        out.append(f"| {d + 1} | {v.min():.2f} – {v.max():.2f} | {len(sel)} | {cc.mean():+.3f} | {100 * (cc < 0).mean():.0f} | {np.nanmean(ii):+.3f} |")
    return "\n".join(out)


def top_table(rows, key, k=10, reverse=True):
    rr = [r for r in rows if not np.isnan(r[key]) and not np.isnan(r["cycle_effect_cm"])]
    rr = sorted(rr, key=lambda r: r[key], reverse=reverse)[:k]
    out = ["| seq | kf frame | GT motion mm/f | GT rot deg/f | tracker motion 5f before | rise_on cm | rise_off cm | delta new mm | inliers | ADD on before | ADD off before | immediate (cm) | cycle effect (cm) | cycle frames |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rr:
        out.append(f"| {r['seq']} | {r['kf']} | {r['gt_move_mm']:.1f} | {r['gt_rot_deg']:.1f} | {r['on_move_pre5_mm']:.1f} | {r['rise_on_cm']:+.2f} | {r['rise_off_cm']:+.2f} | {r['delta_new_mm']:.1f} | "
                   f"{r['inliers_prev']:.0f} | {r['add_on_pre']:.2f} | {r['add_off_pre']:.2f} | {r['imm_effect_cm']:+.2f} | {r['cycle_effect_cm']:+.2f} | {r['cycle_len']:.0f} |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, default=Path("outputs/exp_fusion_20260915/adopt"))
    ap.add_argument("--off-root", type=Path, default=Path("outputs/_archive_records_20260912"))
    ap.add_argument("--log-dir", type=Path, default=Path("logs/exp_fusion_20260915"))
    ap.add_argument("--add-json", type=Path, default=Path("logs/exp_fusion_20260915/add_over_frames/per_frame_add.json"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seqs", nargs="*", default=None)
    args = ap.parse_args()
    out = args.out_dir; out.mkdir(parents=True, exist_ok=True)
    add_curves = json.load(open(args.add_json))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    all_rows = []; sanity = {}; md = ["# Feedback write-backs vs tracker motion / error jumps (adopted runs, 22 sequences)\n",
                                      "cycle effect < 0 = the feedback helped over the frames this write-back influenced; > 0 = hurt.  Motion = mean displacement of the model points between consecutive frames.\n"]
    for ds, seqs in (("ho3d", HO), ("ycb", YC)):
        for s in seqs:
            if args.seqs and s not in args.seqs: continue
            res = analyze_sequence(ds, s, args.run_root, args.off_root, args.log_dir, add_curves)
            if res is None: continue
            rows, san, (add_on, add_off) = res; sanity[s] = san; all_rows += rows
            with open(out / f"cycles_{s}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
            fig, ax = plt.subplots(figsize=(14, 4.5)); x = np.arange(len(add_on))
            ax.plot(x, add_on, lw=0.9, label=f"feedback on ({np.nanmean(add_on):.2f})"); ax.plot(x, add_off, lw=0.9, alpha=0.8, label=f"tracker alone ({np.nanmean(add_off):.2f})")
            ax2 = ax.twinx()
            fr = np.array([r["frame"] for r in rows]); cy = np.array([r["cycle_effect_cm"] for r in rows]); ok = ~np.isnan(cy)
            ax2.bar(fr[ok], cy[ok], width=np.maximum(1, 0.8 * np.array([r["cycle_len"] for r in rows])[ok]), color=np.where(cy[ok] < 0, "green", "red"), alpha=0.35, align="edge")
            ax2.axhline(0, color="k", lw=0.5); ax2.set_ylabel("cycle effect (cm): green helped, red hurt")
            ax.set_xlabel("frame"); ax.set_ylabel("ADD (cm)"); ax.legend(loc="upper left"); ax.set_title(f"{ds}/{s}: ADD and the effect of every write-back")
            fig.tight_layout(); fig.savefig(out / f"cycles_{s}.png", dpi=90); plt.close(fig)
            print(f"{ds}/{s}: events {san['events']}, max lag {san['max_lag']:.0f}, sum cycle effects {san['sum_cycle_effects']:+.2f} vs gap change {san['gap_change_first_to_last']:+.2f}")
    with open(out / "cycles_all.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys())); w.writeheader(); w.writerows(all_rows)
    json.dump(sanity, open(out / "sanity.json", "w"), indent=1)
    md.append("## Sanity: write-back lag (frames between keyframe and landing) and sum of cycle effects vs gap change\n")
    md.append("| seq | events | max lag | sum of cycle effects (cm) | gap change first→last write-back (cm) | ADD on | ADD off |\n|---|---|---|---|---|---|---|")
    for s, v in sanity.items():
        md.append(f"| {s} | {v['events']} | {v['max_lag']:.0f} | {v['sum_cycle_effects']:+.2f} | {v['gap_change_first_to_last']:+.2f} | {v['ADD_on']:.3f} | {v['ADD_off']:.3f} |")
    for ds in ("ho3d", "ycb"):
        rows = [r for r in all_rows if r["dataset"] == ds and r["event"] != "train_000"]
        if not rows: continue
        md.append(f"\n## {ds.upper()}: write-backs grouped by their effect (update cycles only; the 5-view initial write-back is listed separately)\n")
        md.append(GROUP_HEADER)
        md.append(fmt_group(rows, "all update write-backs"))
        md.append(fmt_group([r for r in rows if r["cycle_effect_cm"] < 0], "helped (cycle effect < 0)"))
        md.append(fmt_group([r for r in rows if r["cycle_effect_cm"] > 0], "hurt (cycle effect > 0)"))
        md.append(fmt_group([r for r in rows if r["imm_effect_cm"] < 0], "immediate effect < 0"))
        md.append(fmt_group([r for r in rows if r["imm_effect_cm"] > 0], "immediate effect > 0"))
        init = [r for r in all_rows if r["dataset"] == ds and r["event"] == "train_000"]
        if init: md.append(fmt_group(init, "initial 5-view write-back"))
        md.append(f"\n### {ds.upper()}: deciles of GT motion at the keyframe\n"); md.append(decile_table(rows, "gt_move_mm", "GT motion at kf", "mm/frame"))
        md.append(f"\n### {ds.upper()}: deciles of the tracker's own motion over the 5 frames before the keyframe\n"); md.append(decile_table(rows, "on_move_pre5_mm", "tracker motion (5 frames before)", "mm/frame"))
        md.append(f"\n### {ds.upper()}: deciles of the tracker error jump before the write-back (rise_on = ADD_on(f-1) − ADD_on(f-6))\n"); md.append(decile_table(rows, "rise_on_cm", "rise_on", "cm"))
        md.append(f"\n### {ds.upper()}: deciles of the write-back delta of the newest view\n"); md.append(decile_table(rows, "delta_new_mm", "delta", "mm"))
        md.append(f"\n### {ds.upper()}: 10 write-backs with the sharpest tracker error rise before them\n"); md.append(top_table(rows, "rise_on_cm"))
        md.append(f"\n### {ds.upper()}: 10 write-backs at the fastest GT motion\n"); md.append(top_table(rows, "gt_move_mm"))
        md.append(f"\n### {ds.upper()}: 10 write-backs that helped most / hurt most (cycle effect)\n"); md.append(top_table(rows, "cycle_effect_cm", reverse=False)); md.append(""); md.append(top_table(rows, "cycle_effect_cm", reverse=True))
        def sp(a, b):
            a = np.array([r[a] for r in rows], dtype=np.float64); b = np.array([r[b] for r in rows], dtype=np.float64); k = ~np.isnan(a) & ~np.isnan(b)
            return f"{spearmanr(a[k], b[k]).correlation:+.2f} (n={k.sum()})"
        md.append(f"\n### {ds.upper()}: Spearman correlations with the cycle effect\n")
        md.append("GT motion " + sp("gt_move_mm", "cycle_effect_cm") + ", GT rotation " + sp("gt_rot_deg", "cycle_effect_cm") + ", tracker motion (5 before) " + sp("on_move_pre5_mm", "cycle_effect_cm")
                  + ", rise_on " + sp("rise_on_cm", "cycle_effect_cm") + ", delta " + sp("delta_new_mm", "cycle_effect_cm") + ", inliers " + sp("inliers_prev", "cycle_effect_cm")
                  + ", gap before " + sp("gap_pre", "cycle_effect_cm") + "; with the immediate effect: rise_on " + sp("rise_on_cm", "imm_effect_cm") + ", delta " + sp("delta_new_mm", "imm_effect_cm")
                  + ", GT motion " + sp("gt_move_mm", "imm_effect_cm"))
        # per-sequence helped/hurt summary
        md.append(f"\n### {ds.upper()}: per sequence\n")
        md.append("| seq | write-backs | helped n | helped sum (cm) | hurt n | hurt sum (cm) | GT motion median, helped | GT motion median, hurt | rise_on median, helped | rise_on median, hurt |\n|---|---|---|---|---|---|---|---|---|---|")
        for s in sorted(set(r["seq"] for r in rows), key=lambda x: (HO + YC).index(x)):
            rr = [r for r in rows if r["seq"] == s and not np.isnan(r["cycle_effect_cm"])]; h = [r for r in rr if r["cycle_effect_cm"] < 0]; u = [r for r in rr if r["cycle_effect_cm"] > 0]
            md.append(f"| {s} | {len(rr)} | {len(h)} | {sum(r['cycle_effect_cm'] for r in h):+.2f} | {len(u)} | {sum(r['cycle_effect_cm'] for r in u):+.2f} | {q([r['gt_move_mm'] for r in h], 50):.2f} | {q([r['gt_move_mm'] for r in u], 50):.2f} | {q([r['rise_on_cm'] for r in h], 50):+.2f} | {q([r['rise_on_cm'] for r in u], 50):+.2f} |")
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        for ax, (kx, lab) in zip(axes.ravel(), (("gt_move_mm", "GT motion at kf (mm/frame)"), ("on_move_pre5_mm", "tracker motion, 5 frames before kf (mm/frame)"), ("rise_on_cm", "tracker error rise before write-back (cm)"), ("delta_new_mm", "write-back delta of newest view (mm)"))):
            xs = np.array([r[kx] for r in rows]); ys = np.array([r["cycle_effect_cm"] for r in rows]); ax.scatter(xs, ys, s=8, alpha=0.5); ax.axhline(0, color="k", lw=0.6)
            ax.set_xlabel(lab); ax.set_ylabel("cycle effect (cm), < 0 helped"); ax.grid(alpha=0.3)
        fig.suptitle(f"{ds}: every write-back of the adopted runs"); fig.tight_layout(); fig.savefig(out / f"scatter_{ds}.png", dpi=90); plt.close(fig)
    open(out / "summary.md", "w").write("\n".join(md)); print("wrote", out / "summary.md")


if __name__ == "__main__":
    main()
