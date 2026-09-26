"""Debug record (Probe 2, 9/26): AP10 first keyframe, E-V2 input pair and relative rotation real vs synthetic."""
import sys
from p2_lib import *
from models import load_model as lv
v = lv("M-V"); seq = SeqData("ho3d", "AP10"); m = MeshModel(seq); k = seq.kf_ids[0]; G = seq.GT[k]; P0 = seq.C1[k]
real = seq.frame(k); r = m.render(P0)
print("rot0", pose_err(P0, G), "mask px real", real[2].sum(), "render px", r[2].sum(), "iou", iou(real[2], r[2]))
b = vggt_batch(v, seq.K, [r[:3], real], 518)
rel_est = np.linalg.inv(b["c2w_b"][0]) @ b["c2w_b"][1]; print("VGGT rel rot deg (real)", L.rot_deg(rel_est[:3, :3], np.eye(3)))
grid = np.concatenate([b["warped"][0][0], b["warped"][1][0]], 1)
cv2.imwrite(str(LOG / "debug_AP10_kf0_V2_input.png"), cv2.cvtColor((cv2.resize(grid, (0, 0), fx=0.5, fy=0.5) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
b2 = vggt_batch(v, seq.K, [r[:3], m.render(G)], 518); re2 = np.linalg.inv(b2["c2w_b"][0]) @ b2["c2w_b"][1]; print("VGGT rel rot deg (synthetic)", L.rot_deg(re2[:3, :3], np.eye(3)))
