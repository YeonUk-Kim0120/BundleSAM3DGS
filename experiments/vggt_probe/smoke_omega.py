"""Stage 0-c smoke for M-Omega (brief section 9): M-O x I1 x KF0-KF4 (+1 batch) on AP12 and mustard0.  Prints E1, s, s_i, T3;
sanity: s within +-50 % of the KF0 mask mean ray distance.  Writes logs/exp_vggt_probe1/omega_smoke.txt and SMOKE_PASS marker."""
import sys, time, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_lib as L
from models import load_model
from run_probe import run_cell, summarize

LOG = Path("/home/kist/Desktop/BundleSAM3DGS/logs/exp_vggt_probe1"); OUT = Path("/home/kist/Desktop/BundleSAM3DGS/outputs/exp_vggt_probe1/stage0c_omega")
lines = []; ok_all = True; model = load_model("M-O")
lines.append(f"M-O smoke {time.strftime('%F %T')} ckpt sha256[:16] {model.ckpt_sha}")
for ds, sq in [("ho3d", "AP12"), ("ycb", "mustard0")]:
    seq = L.Sequence(ds, sq); seq.kf_ids = seq.kf_ids[:10]
    rgb, d, m = seq.frame(seq.kf_ids[0]); rn = L.ray_norm(seq.K, m.shape); v = m & (d > 0.1); ray = float((d * rn)[v].mean())
    rows = run_cell(model, seq, "I1", "ref", 5, 8, OUT / f"{sq}_I1_ref", model.size, lambda s: lines.append(s))
    bl = [json.loads(l) for l in open(OUT / f"{sq}_I1_ref" / "batches.jsonl")]
    s0 = bl[0]["s"]; ok = abs(s0 / ray - 1) <= 0.5; ok_all &= ok
    summ = summarize(rows)
    lines.append(f"{sq}: s(batch0)={s0:.3f} vs KF0 mean ray dist {ray:.3f} m -> {'OK' if ok else 'FAIL'}; s_i={np.round(bl[0]['s_i'],3).tolist()}; T3(FoV diff)={bl[0]['T3']:.2f} deg; "
                 f"new-KF E1 rot online/oracle/C1/C2 = {summ['rot_online']:.2f}/{summ['rot_oracle']:.2f}/{summ['rot_c1']:.2f}/{summ['rot_c2']:.2f} deg, trans {summ['trans_online']:.1f}/{summ['trans_oracle']:.1f}/{summ['trans_c1']:.1f}/{summ['trans_c2']:.1f} mm; time/batch {bl[-1]['time_s']:.2f}s peak {bl[-1]['peak_mem_gb']:.1f} GB")
lines.append("SMOKE " + ("PASS" if ok_all else "FAIL"))
open(LOG / "omega_smoke.txt", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
if ok_all: (LOG / "SMOKE_PASS_omega").touch()
