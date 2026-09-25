"""Model wrappers for EXP_VGGT_PROBE1.  infer(images float[N,3,S,S] in [0,1]) -> dict(w2c[N,4,4] OpenCV world->camera with
frame 0 = identity, K[N,3,3], depth[N,S,S] z-depth, conf[N,S,S], time_s, peak_mem_gb).  Pose decoding uses the repos' own
functions; normalisation = ToTensor only ([0,1], no mean/std), as in each repo's load_fn."""
from __future__ import annotations
import hashlib, os, time
import numpy as np, torch


def sha256(path, n=16):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()[:n]


class VGGTWrapper:
    name = "M-V"; size = 518; patch = 14

    def __init__(self, device="cuda"):
        from huggingface_hub import hf_hub_download
        from vggt.models.vggt import VGGT
        self.ckpt = hf_hub_download("facebook/VGGT-1B", "model.pt"); self.ckpt_sha = sha256(self.ckpt)
        self.model = VGGT(); self.model.load_state_dict(torch.load(self.ckpt, map_location="cpu")); self.model = self.model.to(device).eval(); self.device = device

    @torch.no_grad()
    def infer(self, images: torch.Tensor) -> dict:
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        assert images.ndim == 4 and images.shape[1] == 3 and images.shape[2] == self.size == images.shape[3]
        x = images.to(self.device).float(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t = time.time()
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            out = self.model(x)
        torch.cuda.synchronize(); dt = time.time() - t
        extri, intri = pose_encoding_to_extri_intri(out["pose_enc"], image_size_hw=(self.size, self.size))
        N = x.shape[0]; w2c = np.tile(np.eye(4), (N, 1, 1)); w2c[:, :3, :4] = extri[0].float().cpu().numpy()
        return {"w2c": w2c, "K": intri[0].float().cpu().numpy(), "depth": out["depth"][0, ..., 0].float().cpu().numpy(), "conf": out["depth_conf"][0].float().cpu().numpy(),
                "time_s": dt, "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9}


class OmegaWrapper:
    """VGGT-Omega (vggt_omega_1b_512, patch 16, S = 512) loaded from a local checkpoint file (no download)."""
    name = "M-O"; size = 512; patch = 16
    CKPT = os.environ.get("VGGT_OMEGA_CKPT", "/home/kist/checkpoints/vggt_omega/vggt_omega_1b_512.pt")

    def __init__(self, device="cuda"):
        from vggt_omega.models.vggt_omega import VGGTOmega
        self.ckpt = self.CKPT; self.ckpt_sha = sha256(self.ckpt)
        sd = torch.load(self.ckpt, map_location="cpu", weights_only=False); sd = sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd
        self.model = VGGTOmega(patch_size=16, autocast=(device != "cpu")); self.model.load_state_dict(sd, strict=True); self.model = self.model.to(device).eval(); self.device = device

    @torch.no_grad()
    def infer(self, images: torch.Tensor) -> dict:
        from vggt_omega.utils.pose_enc import encoding_to_camera
        assert images.ndim == 4 and images.shape[1] == 3 and images.shape[2] == self.size == images.shape[3]
        x = images.to(self.device).float(); cuda = self.device != "cpu"
        if cuda: torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        t = time.time(); out = self.model(x)   # the model applies its own bf16 autocast on cuda
        if cuda: torch.cuda.synchronize()
        dt = time.time() - t
        pe = out["pose_enc"]; pe = pe[-1] if isinstance(pe, (list, tuple)) else pe   # iterative camera head: take the last iteration
        extri, intri = encoding_to_camera(pe, image_size_hw=(self.size, self.size))
        N = x.shape[0]; w2c = np.tile(np.eye(4), (N, 1, 1)); w2c[:, :3, :4] = extri[0].float().cpu().numpy()
        d = out["depth"][0]; d = d[..., 0] if d.ndim == 4 else d; c = out["depth_conf"][0]; c = c[..., 0] if c.ndim == 4 else c
        return {"w2c": w2c, "K": intri[0].float().cpu().numpy(), "depth": d.float().cpu().numpy(), "conf": c.float().cpu().numpy(), "time_s": dt,
                "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if cuda else 0.0}


def load_model(name, device="cuda"):
    return VGGTWrapper(device) if name == "M-V" else OmegaWrapper(device)
