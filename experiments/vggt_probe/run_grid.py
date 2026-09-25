"""Sequential driver for a list of probe cells (one process, model loaded once).  Progress -> <log>/progress_<stage>.txt"""
from __future__ import annotations
import argparse, itertools, json, sys, time, traceback
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_lib as L
from models import load_model
from run_probe import run_cell, summarize

HO3D = ["AP10", "AP11", "AP12", "AP13", "AP14", "MPM10", "MPM11", "MPM12", "MPM13", "MPM14", "SB11", "SB13", "SM1"]
YCB = ["bleach0", "bleach_hard_00_03_chaitanya", "cracker_box_reorient", "cracker_box_yalehand0", "mustard0", "mustard_easy_00_02", "sugar_box1", "sugar_box_yalehand0", "tomato_soup_can_yalehand0"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True); ap.add_argument("--model", default="M-V"); ap.add_argument("--inputs", default="I1"); ap.add_argument("--batches", default="ref:5:8")
    ap.add_argument("--datasets", default="ho3d"); ap.add_argument("--seqs", default=""); ap.add_argument("--out_root", default="/home/kist/Desktop/BundleSAM3DGS/outputs/exp_vggt_probe1")
    ap.add_argument("--log_dir", default="/home/kist/Desktop/BundleSAM3DGS/logs/exp_vggt_probe1")
    a = ap.parse_args()
    model = load_model(a.model); prog = open(Path(a.log_dir) / f"progress_{a.stage}.txt", "a")
    def log(m):
        print(m, flush=True); prog.write(m + "\n"); prog.flush()
    cells = []
    for ds in a.datasets.split(","):
        seqs = a.seqs.split(",") if a.seqs else (HO3D if ds == "ho3d" else YCB)
        for sq, inp, bspec in itertools.product(seqs, a.inputs.split(","), a.batches.split(",")):
            cells.append((ds, sq, inp, bspec))
    log(f"START stage={a.stage} model={a.model} ckpt={model.ckpt_sha} cells={len(cells)} {time.strftime('%F %T')}")
    for ds, sq, inp, bspec in cells:
        parts = bspec.split(":"); bmode = parts[0]; M = int(parts[1]) if len(parts) > 1 else 5; k = int(parts[2]) if len(parts) > 2 else 8
        btag = f"ref{M}-{k}" if bmode == "ref" else "chain"; tag = f"{a.model}/{inp}/{btag}"
        out = Path(a.out_root) / a.model / inp / btag / ds / sq
        if out.exists():
            log(f"SKIP exists {out}"); continue
        t0 = time.time()
        try:
            seq = L.Sequence(ds, sq)
            rows = run_cell(model, seq, inp, bmode, M, k, out, model.size, lambda m: None)
            summ = summarize(rows); summ["n_kf"] = len(seq.kf_ids); summ["wall_s"] = time.time() - t0
            json.dump(dict(stage=a.stage, model=model.name, ckpt=model.ckpt, ckpt_sha256_16=model.ckpt_sha, size=model.size, patch=model.patch, input=inp, batch=bmode, M=M, k=k, dataset=ds, seq=sq,
                           torch=torch.__version__, gpu=torch.cuda.get_device_name(0), bsdf_run=str(seq.run), kf_ids=seq.kf_ids, summary=summ, started=time.strftime("%F %T", time.localtime(t0))), open(out / "manifest.json", "w"), indent=1)
            log(f"DONE {tag} {ds}/{sq} kf={summ['n_kf']} new={summ['n_new']} rot online/oracle/C1/C2={summ['rot_online']:.2f}/{summ['rot_oracle']:.2f}/{summ['rot_c1']:.2f}/{summ['rot_c2']:.2f} trans={summ['trans_online']:.1f}/{summ['trans_oracle']:.1f}/{summ['trans_c1']:.1f}/{summ['trans_c2']:.1f} {summ['wall_s']:.0f}s")
        except Exception:
            log(f"FAIL {tag} {ds}/{sq}\n" + traceback.format_exc())
    log(f"END stage={a.stage} {time.strftime('%F %T')}")


if __name__ == "__main__":
    main()
