"""
Locate the source of the colour-vs-pose discontinuity.

1. Forward determinism: same pose rendered repeatedly must give the same loss.
2. Fine loss profile along a fixed tau direction: a staircase means the
   forward render jumps under infinitesimal pose changes.
3. Single-triangle scene: removes triangle sort-order and overlap effects;
   if the profile is smooth there, the jumps come from multi-triangle
   interactions (sorting / aggregation), not the per-triangle maths.

Run:  python debug/gradcheck_profile.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from debug.gradcheck_rasterizer import (
    MockCamera, MockPipe, MockTriangles, build_scene,
    make_loss_weights, analytic_grads, fd_loss,
)
from utils.pose_utils import SE3_exp

DEVICE = "cuda"
torch.manual_seed(0)


def profile(pc, cam, pipe, bg, wc, wd, mode, u, span, n=41):
    base_w2c = cam.w2c.clone()
    ts = torch.linspace(-span, span, n)
    losses = []
    for t in ts:
        cam.set_pose(SE3_exp(t.item() * u) @ base_w2c)
        losses.append(fd_loss(pc, cam, pipe, bg, wc, wd, mode))
    cam.set_pose(base_w2c)
    print(f"  t in [{-span:g}, {span:g}], {n} samples; "
          f"loss range=[{min(losses):.6f}, {max(losses):.6f}]")
    prev = losses[0]
    for t, L in zip(ts.tolist(), losses):
        jump = L - prev
        prev = L
        print(f"    t={t:+10.2e}  loss={L:.6f}  step={jump:+.6f}")


def main():
    sigma = 0.5
    W = H = 64
    fx = fy = 60.0
    pipe = MockPipe()
    bg = torch.zeros(3, device=DEVICE)
    wc, wd = make_loss_weights(H, W)

    # --- full grid scene ---
    verts, tri_idx, opacity, features_dc = build_scene()
    pc = MockTriangles(verts.requires_grad_(True), tri_idx,
                       opacity.requires_grad_(True),
                       features_dc.requires_grad_(True), sigma)
    cam = MockCamera(torch.eye(4, device=DEVICE), fx, fy, W / 2, H / 2, W, H)

    print("=== forward determinism (grid scene, color) ===")
    vals = [fd_loss(pc, cam, pipe, bg, wc, wd, "color") for _ in range(5)]
    print("  ", vals)

    g = torch.Generator().manual_seed(3)
    u = torch.randn(6, generator=g).to(DEVICE)
    u = u / u.norm()

    print("\n=== loss profile along u (grid scene, color) ===")
    profile(pc, cam, pipe, bg, wc, wd, "color", u, span=1e-4)

    print("\n=== loss profile along u (grid scene, depth) ===")
    profile(pc, cam, pipe, bg, wc, wd, "depth", u, span=1e-4)

    # --- single triangle scene ---
    print("\n=== single triangle scene ===")
    sv = torch.tensor([[-0.8, -0.5, 2.0],
                       [0.9, -0.4, 2.1],
                       [0.05, 0.8, 1.95]], device=DEVICE)
    st = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=DEVICE)
    so = torch.tensor([[0.7], [0.8], [0.6]], device=DEVICE)
    sf = 0.8 * torch.randn(3, 1, 3, device=DEVICE)
    pc1 = MockTriangles(sv.requires_grad_(True), st,
                        so.requires_grad_(True),
                        sf.requires_grad_(True), sigma)
    cam1 = MockCamera(torch.eye(4, device=DEVICE), fx, fy, W / 2, H / 2, W, H)

    print("\n--- profile along u (single tri, color) ---")
    profile(pc1, cam1, pipe, bg, wc, wd, "color", u, span=1e-4)

    print("\n--- FD vs analytic tau (single tri, color) ---")
    ag = analytic_grads(pc1, cam1, pipe, bg, wc, wd, "color", True)
    print("analytic tau:", [f"{x:+.4f}" for x in ag["tau"].tolist()])
    base_w2c = cam1.w2c.clone()
    for eps in (1e-5, 3e-5, 1e-4, 3e-4, 1e-3):
        fd6 = []
        for i in range(6):
            vals = []
            for sgn in (+1.0, -1.0):
                tau = torch.zeros(6, device=DEVICE)
                tau[i] = sgn * eps
                cam1.set_pose(SE3_exp(tau) @ base_w2c)
                vals.append(fd_loss(pc1, cam1, pipe, bg, wc, wd, "color"))
            fd6.append((vals[0] - vals[1]) / (2 * eps))
        cam1.set_pose(base_w2c)
        print(f"eps={eps:8.0e} fd tau:", [f"{x:+.4f}" for x in fd6])


if __name__ == "__main__":
    main()
