"""EXP_BATCH_20260927 stage 0 (Q0): split the map contamination of a noop (tracker-only) run into a jagged part (keyframe-to-keyframe
inconsistency) and a consistent drift.  For keyframe k the consumption pose c2w_k is its pose in the first poses_before_gs.txt that
contains k (= what the GS backend used to build the map from k).  A_k = c2w_k @ inv(c2w_gt,k) (GroundTruth first-frame gauge).
  d_abs(k) = mean_p |A_k p - p|,   d_loc(k) = median_{j=1..4} mean_p |A_{k+j} p - A_k p|   (p = GT mesh samples, tracker frame).
Reports per sequence (median, p90, ratio, fractions > 3 / 5 mm), a plot, and the brief's decision rule.  Read-only on run dirs.

  python3 experiments/exp_contamination_split.py --runs ho3d:AP12:<run_dir> ycb:mustard0:<run_dir> ... --out-dir logs/exp_batch_20260927/stage0
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments")); sys.path.append(str(REPO / "BundleTrack/scripts"))
from exp_feedback_gradient_probe_gtmap import GroundTruth, gt_mesh_in_tracker_frame, load_cycles  # noqa: E402

N_MESH = 20000; WINDOW = 4


def consumption_poses(cycles):
    first = {}
    for cyc in cycles:
        for fid, P in zip(cyc["ids"], cyc["before"]):
            if fid not in first: first[fid] = np.asarray(P, dtype=np.float64)
    return first


def disp(A, p):
    return float(np.linalg.norm(p @ A[:3, :3].T + A[:3, 3] - p, axis=1).mean() * 1000.0)


def analyse(ds, seq, run_dir: Path):
    video = REPO / ("datasets/HO3D_v3/evaluation" if ds == "ho3d" else "datasets/YCBInEOAT") / seq
    gt = GroundTruth(ds, video, run_dir); cycles = load_cycles(run_dir)
    cons = consumption_poses(cycles); order = [f for f in cycles[-1]["ids"] if f in cons]
    p = gt_mesh_in_tracker_frame(ds, seq, gt, n_samples=N_MESH)
    A = {}
    for f in order:
        g = gt.gt_c2w(f)
        if g is not None: A[f] = cons[f] @ np.linalg.inv(g)
    ids = [f for f in order if f in A]; rows = []
    for i, f in enumerate(ids):
        nxt = ids[i + 1:i + 1 + WINDOW]
        d_abs = disp(A[f], p)
        d_loc = float(np.median([disp(A[g] @ np.linalg.inv(A[f]), p @ A[f][:3, :3].T + A[f][:3, 3]) for g in nxt])) if nxt else np.nan
        rows.append(dict(kf=f, index=i, d_abs_mm=d_abs, d_loc_mm=d_loc, n_window=len(nxt)))
    return rows, len(cycles)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--out-dir", type=Path, required=True); a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True); summary = {}; allrows = {}
    for spec in a.runs:
        ds, seq, rd = spec.split(":", 2); rows, ncyc = analyse(ds, seq, Path(rd)); allrows[seq] = rows
        da = np.array([r["d_abs_mm"] for r in rows]); dl = np.array([r["d_loc_mm"] for r in rows if np.isfinite(r["d_loc_mm"])])
        ratio = np.array([r["d_loc_mm"] / r["d_abs_mm"] for r in rows if np.isfinite(r["d_loc_mm"]) and r["d_abs_mm"] > 0])
        summary[seq] = dict(dataset=ds, run_dir=rd, cycles=ncyc, keyframes=len(rows), d_abs_median=float(np.median(da)), d_abs_p90=float(np.percentile(da, 90)),
                            d_loc_median=float(np.median(dl)), d_loc_p90=float(np.percentile(dl, 90)), ratio_of_medians=float(np.median(dl) / np.median(da)),
                            ratio_median=float(np.median(ratio)), frac_dloc_gt3=float((dl > 3).mean()), frac_dloc_gt5=float((dl > 5).mean()))
        print(seq, json.dumps(summary[seq]), flush=True)
    n_noisy = sum(1 for s in summary.values() if s["frac_dloc_gt3"] >= 0.10)
    decision = "A1-A3 all (jagged contamination in >= 2 of 4 sequences)" if n_noisy >= 2 else "A1 only on AP12, MPM12; map contamination is mainly bias"
    summary["_decision"] = dict(sequences_with_frac_dloc_gt3_ge_10pct=n_noisy, rule="d_loc > 3 mm in >= 10 % of keyframes in >= 2 of 4 sequences -> A1..A3", decision=decision,
                                window=WINDOW, mesh_samples=N_MESH)
    json.dump(dict(summary=summary, rows=allrows), open(a.out_dir / "contamination_split.json", "w"), indent=1)
    print("DECISION:", decision, flush=True)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axs = plt.subplots(len(allrows), 1, figsize=(10, 2.6 * len(allrows)), squeeze=False)
        for ax, (seq, rows) in zip(axs[:, 0], allrows.items()):
            x = [r["index"] for r in rows]; ax.plot(x, [r["d_abs_mm"] for r in rows], label="d_abs (absolute)"); ax.plot(x, [r["d_loc_mm"] for r in rows], label="d_loc (next 4 KF)")
            ax.axhline(3, color="gray", ls="--", lw=0.8); ax.set_title(seq); ax.set_ylabel("mm"); ax.legend(loc="upper left", fontsize=8)
        axs[-1, 0].set_xlabel("keyframe index"); fig.tight_layout(); fig.savefig(a.out_dir / "contamination_split.png", dpi=110)
    except Exception as e:
        print("plot failed", e)
