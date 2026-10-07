"""
Directional-derivative convergence sweep for the pose gradient, plus a
backward determinism check.

For a fixed random direction u in tau-space, computes the central
finite-difference directional derivative (loss(exp(+eps*u) w2c) -
loss(exp(-eps*u) w2c)) / (2 eps) over a range of eps, and compares with the
analytic grad_tau . u. If FD converges as eps shrinks but to a different
value, the analytic gradient is wrong; if FD never converges, the forward is
not locally smooth and pointwise FD comparisons are unreliable.

Run:  python debug/gradcheck_sweep.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from debug.gradcheck_rasterizer import (
    MockCamera, MockPipe, MockTriangles, build_scene, compute_loss,
    make_loss_weights, analytic_grads, fd_loss,
)
from utils.pose_utils import SE3_exp

DEVICE = "cuda"
torch.manual_seed(0)


def main():
    sigma = 0.5
    W = H = 64
    fx = fy = 60.0

    verts, tri_idx, opacity, features_dc = build_scene()
    verts = verts.requires_grad_(True)
    opacity = opacity.requires_grad_(True)
    features_dc = features_dc.requires_grad_(True)

    pc = MockTriangles(verts, tri_idx, opacity, features_dc, sigma)
    cam = MockCamera(torch.eye(4, device=DEVICE), fx, fy, W / 2, H / 2, W, H)
    pipe = MockPipe()
    bg = torch.zeros(3, device=DEVICE)
    wc, wd = make_loss_weights(H, W)

    # --- determinism of the analytic backward ---
    print("=== analytic backward determinism (5 repeats, color loss) ===")
    taus = []
    for _ in range(5):
        ag = analytic_grads(pc, cam, pipe, bg, wc, wd, "color",
                            optimize_pose_only=True)
        taus.append(ag["tau"])
    taus = torch.stack(taus)
    print("tau mean :", taus.mean(0).tolist())
    print("tau std  :", taus.std(0).tolist())

    print("\n=== pose_only=True vs False (analytic tau, color loss) ===")
    ag_t = analytic_grads(pc, cam, pipe, bg, wc, wd, "color", True)
    ag_f = analytic_grads(pc, cam, pipe, bg, wc, wd, "color", False)
    print("pose_only=True :", ag_t["tau"].tolist())
    print("pose_only=False:", ag_f["tau"].tolist())

    # --- directional derivative sweep ---
    g = torch.Generator().manual_seed(3)
    for mode in ("color", "depth"):
        ag = analytic_grads(pc, cam, pipe, bg, wc, wd, mode, True)
        for trial in range(2):
            u = torch.randn(6, generator=g).to(DEVICE)
            u = u / u.norm()
            analytic_dd = (ag["tau"] * u).sum().item()
            print(f"\n=== sweep mode={mode} trial={trial} "
                  f"u={[f'{x:+.3f}' for x in u.tolist()]} ===")
            print(f"analytic directional derivative = {analytic_dd:+.4f}")
            base_w2c = cam.w2c.clone()
            for eps in (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3):
                vals = []
                for sgn in (+1.0, -1.0):
                    cam.set_pose(SE3_exp(sgn * eps * u) @ base_w2c)
                    vals.append(fd_loss(pc, cam, pipe, bg, wc, wd, mode))
                fd = (vals[0] - vals[1]) / (2 * eps)
                cam.set_pose(base_w2c)
                print(f"  eps={eps:8.0e}  fd={fd:+12.4f}  "
                      f"ratio_fd_over_analytic={fd / analytic_dd:+.4f}")

    # --- same sweep for the two worst vertex coordinates from the first run ---
    print("\n=== vertex coordinate sweeps (color loss) ===")
    ag = analytic_grads(pc, cam, pipe, bg, wc, wd, "color", False)
    flat_grad = ag["vertices"].reshape(-1)
    flat = pc.vertices.detach().reshape(-1)
    g2 = torch.Generator().manual_seed(7)
    v_idx = torch.randperm(pc.vertices.numel(), generator=g2)[:12].tolist()
    for k in (0, 7):  # indices that mismatched in gradcheck_rasterizer
        idx = v_idx[k]
        print(f"\nflat vertex index {idx}: analytic={flat_grad[idx]:+.4f}")
        orig = flat[idx].item()
        for eps in (1e-4, 3e-4, 1e-3, 2e-3, 4e-3):
            flat[idx] = orig + eps
            lp = fd_loss(pc, cam, pipe, bg, wc, wd, "color")
            flat[idx] = orig - eps
            lm = fd_loss(pc, cam, pipe, bg, wc, wd, "color")
            flat[idx] = orig
            fd = (lp - lm) / (2 * eps)
            print(f"  eps={eps:8.0e}  fd={fd:+12.4f}")


if __name__ == "__main__":
    main()
