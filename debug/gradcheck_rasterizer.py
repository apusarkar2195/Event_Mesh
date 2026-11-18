"""
Finite-difference check of the triangle rasteriser's analytical gradients.

Builds a small synthetic scene (triangulated grid facing the camera), renders
it through the full mesh_splatting render() path, and compares analytical
gradients from backward() against central finite differences for:

  - camera pose (the 6-vector tau, ordered [trans, rot], applied as
    new_w2c = SE3_exp(tau) @ w2c, matching utils.pose_utils.update_pose)
  - vertex positions (sampled)
  - vertex opacities (rasteriser input, sampled)
  - SH DC features (sampled)

The pose check is run with optimize_pose_only=True (the tracking path) and
the geometry checks with optimize_pose_only=False (the mapping path).

The forward pass never reads cam_rot_delta / cam_trans_delta; they exist only
as autograd handles for grad_tau. The numerical pose gradient therefore
perturbs the view matrix directly with SE3_exp.

Run:  python debug/gradcheck_rasterizer.py
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mesh_splatting.triangle_renderer import render
from mesh_splatting.utils.graphics_utils import getProjectionMatrix2
from utils.pose_utils import SE3_exp

DEVICE = "cuda"
torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Minimal mocks for the render() interface
# ---------------------------------------------------------------------------

class MockPipe:
    debug = False
    convert_SHs_python = False


class MockTriangles:
    """Pass-through model: get_vertex_weight returns the stored opacity tensor
    directly, so finite differences on `opacity` probe the rasteriser input."""

    def __init__(self, vertices, tri_idx, opacity, features_dc, sigma):
        self.vertices = vertices                    # (V, 3) leaf, requires_grad
        self._triangle_indices = tri_idx            # (T, 3) int32
        self.opacity = opacity                      # (V, 1) leaf, requires_grad
        self._features_dc = features_dc             # (V, 1, 3) leaf, requires_grad
        self._features_rest = torch.zeros(
            (vertices.shape[0], 0, 3), device=DEVICE)
        self.sigma = sigma
        self.active_sh_degree = 0
        self.max_sh_degree = 0
        self.scaling = 1                            # no upsampling
        self.config = None

    @property
    def get_vertices(self):
        return self.vertices

    @property
    def get_triangle_indices(self):
        return self._triangle_indices

    @property
    def get_triangles_points(self):
        return self.vertices[self._triangle_indices.long()]

    @property
    def get_vertex_weight(self):
        return self.opacity

    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def get_sigma(self):
        return self.sigma


class MockCamera:
    def __init__(self, w2c, fx, fy, cx, cy, W, H):
        self.image_width = W
        self.image_height = H
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.FoVx = 2.0 * math.atan(W / (2.0 * fx))
        self.FoVy = 2.0 * math.atan(H / (2.0 * fy))
        self.set_pose(w2c)
        self.cam_rot_delta = torch.zeros(3, device=DEVICE, requires_grad=True)
        self.cam_trans_delta = torch.zeros(3, device=DEVICE, requires_grad=True)
        self.exposure_a = torch.zeros(1, device=DEVICE)
        self.exposure_b = torch.zeros(1, device=DEVICE)

    def set_pose(self, w2c):
        self.w2c = w2c
        self.world_view_transform = w2c.T.contiguous()
        proj = getProjectionMatrix2(
            znear=0.01, zfar=100.0,
            fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy,
            W=self.image_width, H=self.image_height,
        ).to(DEVICE)
        self.projection_matrix = proj.T.contiguous()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0)
            .bmm(self.projection_matrix.unsqueeze(0))
            .squeeze(0)
        )
        self.camera_center = self.world_view_transform.inverse()[3, :3]


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------

def build_scene(grid=8, extent=1.5, depth=2.0, jitter=0.15):
    """Triangulated grid roughly on the plane z=depth, camera at origin
    looking down +z (OpenCV convention)."""
    lin = torch.linspace(-extent, extent, grid, device=DEVICE)
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")
    zz = depth + jitter * torch.randn_like(xx)
    verts = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

    faces = []
    for r in range(grid - 1):
        for c in range(grid - 1):
            i = r * grid + c
            faces.append([i, i + 1, i + grid])
            faces.append([i + 1, i + grid + 1, i + grid])
    tri_idx = torch.tensor(faces, dtype=torch.int32, device=DEVICE)

    V = verts.shape[0]
    opacity = (0.4 + 0.5 * torch.rand(V, 1, device=DEVICE))
    features_dc = 0.8 * torch.randn(V, 1, 3, device=DEVICE)
    return verts, tri_idx, opacity, features_dc


def make_loss_weights(H, W):
    """Fixed random weights so the loss is linear in the rendered image."""
    g = torch.Generator(device="cpu").manual_seed(42)
    wc = torch.randn(3, H, W, generator=g).to(DEVICE)
    wd = torch.randn(1, H, W, generator=g).to(DEVICE)
    return wc, wd


def compute_loss(pkg, wc, wd, mode):
    if mode == "color":
        return (pkg["render"] * wc).sum()
    if mode == "depth":
        return (pkg["expected_depth"] * wd).sum()
    return (pkg["render"] * wc).sum() + (pkg["expected_depth"] * wd).sum()


# ---------------------------------------------------------------------------
# Gradient checks
# ---------------------------------------------------------------------------

def analytic_grads(pc, cam, pipe, bg, wc, wd, mode, optimize_pose_only):
    for t in (pc.vertices, pc.opacity, pc._features_dc,
              cam.cam_rot_delta, cam.cam_trans_delta):
        if t.grad is not None:
            t.grad = None
    pkg = render(cam, pc, pipe, bg, optimize_pose_only=optimize_pose_only)
    loss = compute_loss(pkg, wc, wd, mode)
    loss.backward()
    return {
        "loss": loss.item(),
        "tau": torch.cat([
            cam.cam_trans_delta.grad.detach().clone().reshape(-1)
            if cam.cam_trans_delta.grad is not None else torch.zeros(3, device=DEVICE),
            cam.cam_rot_delta.grad.detach().clone().reshape(-1)
            if cam.cam_rot_delta.grad is not None else torch.zeros(3, device=DEVICE),
        ]),
        "vertices": pc.vertices.grad.detach().clone()
        if pc.vertices.grad is not None else None,
        "opacity": pc.opacity.grad.detach().clone()
        if pc.opacity.grad is not None else None,
        "features_dc": pc._features_dc.grad.detach().clone()
        if pc._features_dc.grad is not None else None,
    }


@torch.no_grad()
def fd_loss(pc, cam, pipe, bg, wc, wd, mode):
    pkg = render(cam, pc, pipe, bg, optimize_pose_only=False)
    return compute_loss(pkg, wc, wd, mode).item()


def fd_pose(pc, cam, pipe, bg, wc, wd, mode, eps):
    """Central differences of the loss wrt tau, applied as
    new_w2c = SE3_exp(tau) @ w2c (the update_pose convention)."""
    base_w2c = cam.w2c.clone()
    grad = torch.zeros(6)
    for i in range(6):
        vals = []
        for sgn in (+1.0, -1.0):
            tau = torch.zeros(6, device=DEVICE)
            tau[i] = sgn * eps
            cam.set_pose(SE3_exp(tau) @ base_w2c)
            vals.append(fd_loss(pc, cam, pipe, bg, wc, wd, mode))
        grad[i] = (vals[0] - vals[1]) / (2 * eps)
    cam.set_pose(base_w2c)
    return grad.to(DEVICE)


def fd_tensor(pc, cam, pipe, bg, wc, wd, mode, tensor, indices, eps):
    """Central differences for a sample of flat indices into `tensor`."""
    flat = tensor.detach().reshape(-1)
    grad = torch.zeros(len(indices))
    for k, idx in enumerate(indices):
        orig = flat[idx].item()
        flat[idx] = orig + eps
        lp = fd_loss(pc, cam, pipe, bg, wc, wd, mode)
        flat[idx] = orig - eps
        lm = fd_loss(pc, cam, pipe, bg, wc, wd, mode)
        flat[idx] = orig
        grad[k] = (lp - lm) / (2 * eps)
    return grad.to(DEVICE)


def report(name, analytic, numeric, tol=0.05):
    analytic = analytic.reshape(-1).float()
    numeric = numeric.reshape(-1).float()
    denom = torch.maximum(analytic.abs(), numeric.abs()).clamp_min(1e-6)
    rel = (analytic - numeric).abs() / denom
    cos = torch.nn.functional.cosine_similarity(
        analytic.unsqueeze(0), numeric.unsqueeze(0)).item()
    print(f"\n[{name}]  cosine={cos:+.5f}  median_rel_err={rel.median():.4f}  "
          f"max_rel_err={rel.max():.4f}")
    for i in range(len(analytic)):
        flag = "  <-- MISMATCH" if rel[i] > tol and \
            max(abs(analytic[i]), abs(numeric[i])) > 1e-3 else ""
        print(f"   [{i:3d}] analytic={analytic[i]:+.6e}  "
              f"numeric={numeric[i]:+.6e}  rel={rel[i]:.4f}{flag}")
    return cos


def main():
    sigma = 0.5  # matches Training.set_sigma in the replica config
    W = H = 64
    fx = fy = 60.0
    cx, cy = W / 2.0, H / 2.0

    verts, tri_idx, opacity, features_dc = build_scene()
    verts = verts.requires_grad_(True)
    opacity = opacity.requires_grad_(True)
    features_dc = features_dc.requires_grad_(True)

    pc = MockTriangles(verts, tri_idx, opacity, features_dc, sigma)
    w2c = torch.eye(4, device=DEVICE)
    cam = MockCamera(w2c, fx, fy, cx, cy, W, H)
    pipe = MockPipe()
    bg = torch.zeros(3, device=DEVICE)
    wc, wd = make_loss_weights(H, W)

    g = torch.Generator().manual_seed(7)
    n_sample = 12
    v_idx = torch.randperm(verts.numel(), generator=g)[:n_sample].tolist()
    o_idx = torch.randperm(opacity.numel(), generator=g)[:n_sample].tolist()
    f_idx = torch.randperm(features_dc.numel(), generator=g)[:n_sample].tolist()

    for mode in ("color", "depth"):
        print("=" * 70)
        print(f"LOSS MODE: {mode}")
        print("=" * 70)

        # --- pose (tracking path) ---
        ag = analytic_grads(pc, cam, pipe, bg, wc, wd, mode,
                            optimize_pose_only=True)
        print(f"loss = {ag['loss']:.6f}")
        for eps in (1e-4, 3e-4):
            ng = fd_pose(pc, cam, pipe, bg, wc, wd, mode, eps)
            report(f"pose tau [trans(3), rot(3)] optimize_pose_only=True "
                   f"eps={eps:g}", ag["tau"], ng)

        # --- pose gradients on the mapping path, for reference ---
        ag_m = analytic_grads(pc, cam, pipe, bg, wc, wd, mode,
                              optimize_pose_only=False)
        ng = fd_pose(pc, cam, pipe, bg, wc, wd, mode, 3e-4)
        report("pose tau optimize_pose_only=False eps=3e-4", ag_m["tau"], ng)

        # --- geometry (mapping path) ---
        if ag_m["vertices"] is not None:
            ng = fd_tensor(pc, cam, pipe, bg, wc, wd, mode,
                           pc.vertices, v_idx, 2e-3)
            report("vertices (sampled) eps=2e-3",
                   ag_m["vertices"].reshape(-1)[v_idx], ng)
        if mode == "color":
            if ag_m["opacity"] is not None:
                ng = fd_tensor(pc, cam, pipe, bg, wc, wd, mode,
                               pc.opacity, o_idx, 1e-3)
                report("opacity (sampled) eps=1e-3",
                       ag_m["opacity"].reshape(-1)[o_idx], ng)
            if ag_m["features_dc"] is not None:
                ng = fd_tensor(pc, cam, pipe, bg, wc, wd, mode,
                               pc._features_dc, f_idx, 1e-2)
                report("features_dc (sampled) eps=1e-2",
                       ag_m["features_dc"].reshape(-1)[f_idx], ng)


if __name__ == "__main__":
    main()
