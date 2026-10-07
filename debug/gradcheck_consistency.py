"""
Internal consistency check of grad_tau against the (FD-verified) vertex
gradients.

With the camera at identity (R=I, t=0), a left-multiplied pose perturbation
new_w2c = SE3_exp(tau) @ w2c moves every camera-space point by
rho + theta x p, identically to moving the world vertices. Hence, exactly:

    dL/dtau[0:3] = sum_v dL/dp_v
    dL/dtau[3:6] = sum_v p_v x dL/dp_v

for any loss that depends on the scene only through the vertices (colour
loss with SH degree 0: no separate campos dependence). The vertex gradients
were verified against finite differences, so a mismatch here isolates the
bug to the pose accumulation paths.

Run:  python debug/gradcheck_consistency.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from debug.gradcheck_rasterizer import (
    MockCamera, MockPipe, MockTriangles, build_scene,
    make_loss_weights, analytic_grads,
)

DEVICE = "cuda"
torch.manual_seed(0)


def expected_tau_from_vertex_grads(verts, vgrad):
    trans = vgrad.sum(dim=0)
    rot = torch.cross(verts, vgrad, dim=1).sum(dim=0)
    return torch.cat([trans, rot])


def run(name, pc, cam, pipe, bg, wc, wd, mode):
    for pose_only in (True, False):
        ag = analytic_grads(pc, cam, pipe, bg, wc, wd, mode,
                            optimize_pose_only=pose_only)
        if ag["vertices"] is None:
            print(f"[{name} mode={mode} pose_only={pose_only}] "
                  "no vertex grads (skipped)")
            tau_v = None
        else:
            tau_v = expected_tau_from_vertex_grads(
                pc.vertices.detach(), ag["vertices"])
        print(f"\n[{name} mode={mode} pose_only={pose_only}]")
        print("  analytic tau        :",
              [f"{x:+10.4f}" for x in ag["tau"].tolist()])
        if tau_v is not None:
            print("  tau from vertexgrads:",
                  [f"{x:+10.4f}" for x in tau_v.tolist()])
            ratio = ag["tau"] / tau_v
            print("  ratio               :",
                  [f"{x:+10.4f}" for x in ratio.tolist()])


def main():
    sigma = 0.5
    W = H = 64
    fx = fy = 60.0
    pipe = MockPipe()
    bg = torch.zeros(3, device=DEVICE)
    wc, wd = make_loss_weights(H, W)

    # single triangle
    sv = torch.tensor([[-0.8, -0.5, 2.0],
                       [0.9, -0.4, 2.1],
                       [0.05, 0.8, 1.95]], device=DEVICE)
    st = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=DEVICE)
    so = torch.tensor([[0.7], [0.8], [0.6]], device=DEVICE)
    sf = 0.8 * torch.randn(3, 1, 3, device=DEVICE)
    pc1 = MockTriangles(sv.requires_grad_(True), st,
                        so.requires_grad_(True), sf.requires_grad_(True),
                        sigma)
    cam1 = MockCamera(torch.eye(4, device=DEVICE), fx, fy, W / 2, H / 2, W, H)
    print("FD reference tau (from gradcheck_profile, single tri, color):")
    print("   [+1.96, +72.74, +8.15, -150.44, +4.61, +12.16] (eps=1e-5)")
    run("single-tri", pc1, cam1, pipe, bg, wc, wd, "color")
    run("single-tri", pc1, cam1, pipe, bg, wc, wd, "depth")

    # grid scene
    verts, tri_idx, opacity, features_dc = build_scene()
    pc = MockTriangles(verts.requires_grad_(True), tri_idx,
                       opacity.requires_grad_(True),
                       features_dc.requires_grad_(True), sigma)
    cam = MockCamera(torch.eye(4, device=DEVICE), fx, fy, W / 2, H / 2, W, H)
    run("grid", pc, cam, pipe, bg, wc, wd, "color")
    run("grid", pc, cam, pipe, bg, wc, wd, "depth")


if __name__ == "__main__":
    main()
