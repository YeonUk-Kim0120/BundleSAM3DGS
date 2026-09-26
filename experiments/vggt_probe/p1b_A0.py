"""A0: reproduce new-KF online/oracle c2w from batches.jsonl alone (T_online/T_oracle × section-2 pose reconstruction) and
compare with kf.csv c2w_online_*/c2w_oracle_* on 3 random cells (seed 0).  Must agree within 1e-6."""
import csv, random, sys
import numpy as np
from p1b_common import *

cells = list(iter_cells()); random.seed(0); pick = random.sample(cells, 3)
lines = []; ok_all = True
for c in pick:
    bl = read_batches(c["dir"]); kf = read_kf(c["dir"]); worst_on = worst_or = 0.0; n = 0
    for b in bl:
        c2w = c2w_batch(b); T_on = np.array(b["T_online"]); T_or = np.array(b["T_oracle"]) if b["T_oracle"] is not None else None
        for i, f in enumerate(b["ids"]):
            if f not in b["new"]: continue
            row = [r for r in kf[f] if int(float(r["batch"])) == b["batch"] and r["role"] == "new"][0]
            stored_on = np.array([float(row[f"c2w_online_{j}"]) for j in range(16)]).reshape(4, 4)
            worst_on = max(worst_on, np.abs(T_on @ c2w[i] - stored_on).max())
            if T_or is not None:
                stored_or = np.array([float(row[f"c2w_oracle_{j}"]) for j in range(16)]).reshape(4, 4)
                worst_or = max(worst_or, np.abs(T_or @ c2w[i] - stored_or).max())
            n += 1
    ok = worst_on < 1e-6 and worst_or < 1e-6; ok_all &= ok
    lines.append(f"A0 {c['model']}/{c['input']}/{c['batch']} {c['ds']}/{c['seq']}: new KF {n}, max |Δ| online {worst_on:.2e}, oracle {worst_or:.2e} -> {'PASS' if ok else 'FAIL'}")
lines.append("A0 " + ("PASS" if ok_all else "FAIL")); print("\n".join(lines)); open(LOG / "A0.txt", "w").write("\n".join(lines) + "\n")
sys.exit(0 if ok_all else 1)
