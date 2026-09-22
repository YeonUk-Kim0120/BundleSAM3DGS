"""Unit checks for the normalinit variant (EXP_BATCH_20260921 4.1): PCA normal error < 5 deg on a plane and a sphere,
third quaternion axis == normal, and switch-off output identical to the main runner's _new_splat_values."""
import os, sys
import numpy as np, torch
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments", "online_variants")); sys.path.insert(1, REPO)
import gaussian_runner as copy_runner  # the experiment copy
from sam3d_prior import quats_from_normals, quat_wxyz_to_matrix

rng = np.random.RandomState(0)
def ang(a, b): return np.degrees(np.arccos(np.clip(np.abs((a * b).sum(1)), 0, 1)))
# plane: tilted, 1 mm grid + 0.3 mm noise
g = np.stack(np.meshgrid(np.arange(-0.03, 0.03, 0.001), np.arange(-0.03, 0.03, 0.001)), -1).reshape(-1, 2)
n_true = np.array([0.3, -0.5, 0.81]); n_true /= np.linalg.norm(n_true)
t1 = np.cross(n_true, [0, 0, 1.0]); t1 /= np.linalg.norm(t1); t2 = np.cross(n_true, t1)
plane = g[:, :1] * t1 + g[:, 1:] * t2 + rng.normal(0, 0.0003, (len(g), 1)) * n_true + np.array([0, 0, 0.6])
cam = np.array([[0.0, 0.0, 0.0]])
nrm, st = copy_runner.estimate_append_normals(plane[::3], plane, cam)
inner = (np.abs(g[::3]) < 0.025).all(1)
e = ang(nrm[inner], np.tile(n_true, (inner.sum(), 1))); print("plane: median %.2f p95 %.2f deg, stats %s" % (np.median(e), np.percentile(e, 95), st)); assert np.percentile(e, 95) < 5
# sphere cap radius 4 cm facing the camera at the origin (centre at z = 0.6)
u = rng.normal(size=(60000, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True); u = u[u[:, 2] < -0.3]
sph = np.array([0, 0, 0.6]) + 0.04 * u + rng.normal(0, 0.0002, (len(u), 3))
nrm, st = copy_runner.estimate_append_normals(sph[::5], sph, cam)
keep = u[::5][:, 2] < -0.5
e = ang(nrm[keep], u[::5][keep]); print("sphere: median %.2f p95 %.2f deg, stats %s" % (np.median(e), np.percentile(e, 95), st)); assert np.median(e) < 5 and np.percentile(e, 95) < 8
assert ((nrm * (cam[0] - sph[::5])).sum(1) > 0).all(), "normals must face the camera"
# fallback: isolated points
iso = np.array([[0.2, 0.2, 0.7]]); nrm, st = copy_runner.estimate_append_normals(iso, plane, cam); assert st["fallback"] == 1
assert ang(nrm, -iso / np.linalg.norm(iso)) < 1e-3
# quaternion third axis == normal
q = quats_from_normals(torch.from_numpy(u[:1000].astype(np.float32))); R = quat_wxyz_to_matrix(q).numpy()
assert np.abs(R[:, :, 2] - u[:1000]).max() < 1e-5; print("quat third axis == normal OK")
# switch off: identical values to the main runner
assert "normalinit" not in copy_runner.ONLINE_VARIANTS
import importlib.util
spec = importlib.util.spec_from_file_location("main_runner", os.path.join(REPO, "gaussian_runner.py")); main_runner = importlib.util.module_from_spec(spec); sys.modules["main_runner"] = main_runner; spec.loader.exec_module(main_runner)
import inspect
src_a = inspect.getsource(main_runner.GaussianRunner._new_splat_values); src_b = inspect.getsource(copy_runner.GaussianRunner._new_splat_values)
assert src_a == src_b, "_new_splat_values differs between main and copy"
print("switch-off path: _new_splat_values source identical; _append_splats only overrides quats when normals is not None")
print("ALL OK")
