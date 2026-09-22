"""Runtime measurement in the BundleSDF paper's terms (appendix B: tracking thread per-frame time and its components,
backend training round time) plus the end-to-end throughput (2026-09-18).

Per run directory: per-frame periods from the ob_in_cam file modification times (the tracker writes one file per frame at
the end of its per-frame loop); with the pipeline log, the tracker's components from the BundleTrack timestamps
(LoFTR+overhead = last 'into pairs' -> 'start multi pair ransac'; RANSAC = ... -> last 'ransac makes match';
pose graph = OptimizerGPU begin -> finish; save = saveNewframeResult welcome -> done; other = remainder), keyframe frames
(GS: '[GS backend] cycle' lines; SDF: 'synced pose from nerf' lines), backend round times (GS: cycle seconds printed by
the backend; SDF: interval between consecutive write-backs while the backend is busy = an upper bound).
usage: measure_runtime.py --run-dir <dir> [--log <pipeline log>] [--label ...] [--add-json ...]  (repeatable via --runs file)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np

TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3})\]")


def parse_ts(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").timestamp()


def from_mtimes(run_dir: Path):
    fs = sorted(glob.glob(str(run_dir / "ob_in_cam" / "*.txt")))
    m = np.array([os.stat(f).st_mtime for f in fs]); dt = np.diff(m)
    return dict(frames=len(fs), total_s=float(m[-1] - m[0]) if len(m) > 1 else np.nan, periods=dt)


def from_log(log: Path):
    frames = []; cur = None; backend_cycles = []; synced_ts = []
    for ln in open(log, errors="ignore"):
        if "processNewFrame start" in ln:
            cur = {"id": ln.split()[-1], "ts": [], "pairs": None, "kf": False, "ev": {}}; frames.append(cur); continue
        if "[GS backend] cycle" in ln:
            m = re.search(r": ([\d.]+)s", ln); backend_cycles.append(float(m.group(1)) if m else np.nan)
            if cur: cur["kf"] = True
        if "synced pose from nerf" in ln and cur:
            cur["kf"] = True; synced_ts.append(cur["ts"][-1] if cur["ts"] else np.nan)
        if cur is None: continue
        if "frame_pairs:" in ln: cur["pairs"] = int(ln.split()[-1])
        m = TS.match(ln)
        if m:
            ts = parse_ts(m.group(1)); cur["ts"].append(ts); e = cur["ev"]
            if "into pairs" in ln: e["pairs_last"] = ts
            elif "start multi pair ransac" in ln: e["ransac_start"] = ts
            elif "ransac makes match" in ln: e["ransac_last"] = ts
            elif "OptimizerGPU begin" in ln: e["opt_begin"] = ts
            elif "OptimizerGPU finish" in ln: e["opt_end"] = ts
            elif "Welcome saveNewframeResult" in ln: e["save_start"] = ts
            elif "saveNewframeResult done" in ln: e["save_end"] = ts
    rows = []
    for i in range(1, len(frames) - 1):
        f, g = frames[i], frames[i + 1]
        if not f["ts"] or not g["ts"]: continue
        e = f["ev"]; period = g["ts"][0] - f["ts"][0]
        lo = e.get("ransac_start", np.nan) - e.get("pairs_last", np.nan); ra = e.get("ransac_last", np.nan) - e.get("ransac_start", np.nan)
        op = e.get("opt_end", np.nan) - e.get("opt_begin", np.nan); sv = e.get("save_end", np.nan) - e.get("save_start", np.nan)
        rows.append(dict(period=period, kf=f["kf"], pairs=f["pairs"], loftr=lo, ransac=ra, opt=op, save=sv))
    sdf_rounds = np.diff([t for t in synced_ts if np.isfinite(t)]) if len(synced_ts) > 2 else np.array([])
    return rows, np.array(backend_cycles), sdf_rounds


def summarize(label, run_dir: Path, log: Path | None, add_json: Path | None):
    out = {"label": label}; mt = from_mtimes(run_dir)
    p = mt["periods"]; ok = p[(p > 0) & (p < 300)]
    out.update(frames=mt["frames"], total_s=mt["total_s"], eff_hz=mt["frames"] / mt["total_s"] if mt["total_s"] else np.nan,
               period_median_ms=1000 * np.median(ok), period_mean_ms=1000 * ok.mean(), period_p90_ms=1000 * np.percentile(ok, 90))
    if log and log.exists():
        rows, gs_cycles, sdf_rounds = from_log(log)
        kf = [r for r in rows if r["kf"]]; nk = [r for r in rows if not r["kf"]]
        def mean_ms(key, rr): 
            v = np.array([r[key] for r in rr], dtype=float); v = v[np.isfinite(v)]; return 1000 * v.mean() if len(v) else np.nan
        out.update(n_keyframe_frames=len(kf), n_other_frames=len(nk),
                   track_period_nonkf_ms=mean_ms("period", nk), track_period_kf_ms=mean_ms("period", kf),
                   loftr_ms=mean_ms("loftr", rows), ransac_ms=mean_ms("ransac", rows), posegraph_ms=mean_ms("opt", rows), save_ms=mean_ms("save", rows),
                   pairs_mean=float(np.nanmean([r["pairs"] for r in rows if r["pairs"] is not None])) if rows else np.nan)
        comp = np.nansum([out["loftr_ms"], out["ransac_ms"], out["posegraph_ms"]]); out["component_sum_ms"] = comp; out["component_hz"] = 1000 / comp if comp else np.nan
        out["other_ms"] = out["track_period_nonkf_ms"] - comp - (out["save_ms"] if np.isfinite(out["save_ms"]) else 0)
        if len(gs_cycles): out.update(backend="GS", backend_rounds=int(len(gs_cycles)), backend_round_s=float(np.nanmean(gs_cycles)))
        elif len(sdf_rounds): out.update(backend="SDF", backend_rounds=int(len(sdf_rounds) + 1), backend_round_s=float(np.median(sdf_rounds)))
    if add_json and add_json.exists():
        d = json.load(open(add_json)); r = d[0] if isinstance(d, list) else d; out["ADD_cm"] = float(r["ADD_err_cm"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True, help="JSON list of {label, run_dir, log, add_json}")
    ap.add_argument("--out-md", type=Path, required=True)
    args = ap.parse_args()
    res = [summarize(r["label"], Path(r["run_dir"]), Path(r["log"]) if r.get("log") else None, Path(r["add_json"]) if r.get("add_json") else None) for r in json.load(open(args.runs))]
    def f(v, nd=0):
        return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:.{nd}f}")
    L = ["| run | frames | wall s | end-to-end Hz | per-frame median ms (all) | non-keyframe frame ms | keyframe frame ms | LoFTR ms (pairs) | RANSAC ms | pose graph ms | save ms | other ms | component sum ms → Hz | backend rounds × s | ADD cm |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for o in res:
        L.append(f"| {o['label']} | {o['frames']} | {f(o['total_s'])} | {f(o['eff_hz'], 2)} | {f(o['period_median_ms'])} | {f(o.get('track_period_nonkf_ms'))} | {f(o.get('track_period_kf_ms'))} | "
                 f"{f(o.get('loftr_ms'))} ({f(o.get('pairs_mean'), 1)}) | {f(o.get('ransac_ms'), 1)} | {f(o.get('posegraph_ms'))} | {f(o.get('save_ms'))} | {f(o.get('other_ms'))} | "
                 f"{f(o.get('component_sum_ms'))} → {f(o.get('component_hz'), 1)} | {o.get('backend_rounds', '–')} × {f(o.get('backend_round_s'), 1)} | {f(o.get('ADD_cm'), 3)} |")
    open(args.out_md, "w").write("\n".join(L) + "\n"); print("\n".join(L))
    json.dump(res, open(args.out_md.with_suffix(".json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
