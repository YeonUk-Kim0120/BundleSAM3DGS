"""Score the unit tests (brief 2): (1) GT input -> d median after < 1 mm; (2) keyframe 10 GT+3 deg -> its d decreases, others ~ (1)."""
import json, sys
import numpy as np
sys.path.insert(0, "/home/kist/Desktop/BundleSAM3DGS/experiments")
from joint_refit_eval import Evaluator, summary
U = "/home/kist/Desktop/BundleSAM3DGS/outputs/exp_batch_20260928/unit"; ev = Evaluator("/home/kist/Desktop/BundleSAM3DGS/logs/exp_batch_20260928/inputs/unit_mustard0_gt.json")
lines = []
for arm in sys.argv[1:]:
    try:
        a = json.load(open(f"{U}/{arm}_gt/result.json"))["rounds"][0]; b = json.load(open(f"{U}/{arm}_gt_pert10/result.json"))["rounds"][0]
    except FileNotFoundError as e:
        lines.append(f"{arm}: missing {e.filename}"); continue
    ra = ev.compare(a["kf_ids"], a["poses_in"], a["poses_out"]); rb = ev.compare(b["kf_ids"], b["poses_in"], b["poses_out"]); s = summary(ra)
    oth = [i for i in range(len(rb)) if i != 10]; ok1 = s["d1_med"] < 1.0; ok2 = rb[10]["d1"] < rb[10]["d0"]
    lines.append(f"{arm}: (1) GT input -> d median {s['d1_med']:.2f} mm (p90 {s['d1_p90']:.2f}) {'PASS' if ok1 else 'FAIL'} (< 1 mm) | (2) KF10 d {rb[10]['d0']:.2f} -> {rb[10]['d1']:.2f} mm, "
                 f"rot {rb[10]['rot0']:.2f} -> {rb[10]['rot1']:.2f} deg {'PASS' if ok2 else 'FAIL'} (decrease); others median {np.median([rb[i]['d1'] for i in oth]):.2f} vs (1) {np.median([ra[i]['d1'] for i in oth]):.2f} mm; "
                 f"time {a['seconds']:.0f}s peak {a['peak_mem_gb']:.1f} GB")
text = "\n".join(lines); print(text)
open("/home/kist/Desktop/BundleSAM3DGS/logs/exp_batch_20260928/unit_tests/unit_results.txt", "a").write(text + "\n")
