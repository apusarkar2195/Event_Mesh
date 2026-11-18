"""
Isolate which gradient path breaks the tau-vs-vertex identity on the paper
rasteriser.

Scenes (colour loss, camera at identity):
  A. two disjoint triangles, far apart on screen (no overlap, no sharing)
  B. two triangles sharing an edge (adjacent, minimal overlap)
  C. two overlapping triangles, disjoint vertex sets (one behind the other)

Each scene runs with (i) varied vertex colours and (ii) uniform colours.
Uniform colours null the barycentric screen-gradient path (dL_db0 = dL_db1 =
dL_db2 cancels because barycentrics sum to one), so:

  identity holds with uniform but breaks with varied colours
      -> barycentric path pair (renderCUDA -> propagateScreenGradsToPose vs
         computeVertexColorsCUDA) is inconsistent
  identity breaks only for overlapping scenes
      -> transmittance-chain (multi-triangle blending) path
  identity breaks only with shared vertices
      -> per-vertex accumulation / indexing

Run:  python debug/gradcheck_isolate.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from debug.gradcheck_rasterizer import (
    MockCamera, MockPipe, MockTriangles, make_loss_weights, analytic_grads,
)
from debug.gradcheck_vertices_full import identity_tau

DEVICE = "cuda"
torch.manual_seed(0)


def scene_disjoint():
    v = torch.tensor([
        [-1.3, -0.9, 2.0], [-0.4, -0.8, 2.1], [-0.9, -0.1, 1.9],   # left tri
        [0.4, 0.3, 2.05], [1.3, 0.4, 1.95], [0.8, 1.0, 2.1],       # right tri
    ], device=DEVICE)
    t = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32, device=DEVICE)
    return v, t


def scene_shared_edge():
    v = torch.tensor([
        [-0.9, -0.7, 2.0], [0.8, -0.6, 2.1], [0.7, 0.8, 1.9], [-0.8, 0.7, 2.05],
    ], device=DEVICE)
    t = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32, device=DEVICE)
    return v, t


def scene_overlap():
    v = torch.tensor([
        [-0.8, -0.6, 2.0], [0.9, -0.5, 2.0], [0.0, 0.9, 2.0],      # front
        [-0.7, -0.4, 2.6], [1.0, -0.3, 2.6], [0.1, 1.0, 2.6],      # behind
    ], device=DEVICE)
    t = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32, device=DEVICE)
    return v, t


def run(name, verts, tris, uniform):
    V = verts.shape[0]
    opacity = (0.5 + 0.3 * torch.rand(V, 1, device=DEVICE))
    if uniform:
        features = torch.full((V, 1, 3), 0.6, device=DEVICE)
    else:
        features = 0.8 * torch.randn(V, 1, 3, device=DEVICE)
    pc = MockTriangles(verts.clone().requires_grad_(True), tris,
                       opacity.requires_grad_(True),
                       features.requires_grad_(True), 0.5)
    W = H = 64
    cam = MockCamera(torch.eye(4, device=DEVICE), 60.0, 60.0, W / 2, H / 2, W, H)
    pipe = MockPipe()
    bg = torch.zeros(3, device=DEVICE)
    wc, wd = make_loss_weights(H, W)

    ag = analytic_grads(pc, cam, pipe, bg, wc, wd, "color",
                        optimize_pose_only=False)
    tau_v = identity_tau(pc.vertices.detach(), ag["vertices"])
    diff = (ag["tau"] - tau_v)
    scale = torch.maximum(ag["tau"].abs(), tau_v.abs()).clamp_min(1e-3)
    worst = (diff.abs() / scale).max().item()
    status = "OK   " if worst < 0.01 else "BREAK"
    print(f"{status} {name:38s} colours={'uniform' if uniform else 'varied '}"
          f"  worst_rel={worst:8.4f}")
    if worst >= 0.01:
        print("        tau analytic :", [f"{x:+9.3f}" for x in ag["tau"].tolist()])
        print("        tau from vtx :", [f"{x:+9.3f}" for x in tau_v.tolist()])


def main():
    for name, fn in (("A disjoint far-apart", scene_disjoint),
                     ("B shared edge", scene_shared_edge),
                     ("C overlapping disjoint verts", scene_overlap)):
        for uniform in (False, True):
            v, t = fn()
            run(name, v, t, uniform)


if __name__ == "__main__":
    main()
