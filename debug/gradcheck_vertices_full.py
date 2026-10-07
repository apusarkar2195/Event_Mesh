"""
Exhaustive finite-difference check of vertex gradients, plus the
tau-vs-vertex identity evaluated against FD rather than analytic grads.

Scenes: (a) two triangles sharing an edge (minimal multi-triangle case),
(b) the full grid. For each, FD every vertex coordinate at two eps values
(coords whose FD estimate is unstable between eps values are flagged and
excluded from aggregates), then:

  - per-vertex comparison analytic vs FD, grouped to expose patterns
    (which vertices are wrong, how many triangles share them)
  - "true" tau from the identity using FD vertex grads:
        tau[0:3] = sum_v g_v,   tau[3:6] = sum_v p_v x g_v   (identity pose)
    compared against the analytic tau, and against the identity applied to
    the analytic vertex grads.

Run:  python debug/gradcheck_vertices_full.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from debug.gradcheck_rasterizer import (
    MockCamera, MockPipe, MockTriangles, build_scene,
    make_loss_weights, analytic_grads, fd_loss,
)

DEVICE = "cuda"
torch.manual_seed(0)


def fd_all_vertices(pc, cam, pipe, bg, wc, wd, mode, eps):
    flat = pc.vertices.detach().reshape(-1)
    grad = torch.zeros(flat.numel())
    for idx in range(flat.numel()):
        orig = flat[idx].item()
        flat[idx] = orig + eps
        lp = fd_loss(pc, cam, pipe, bg, wc, wd, mode)
        flat[idx] = orig - eps
        lm = fd_loss(pc, cam, pipe, bg, wc, wd, mode)
        flat[idx] = orig
        grad[idx] = (lp - lm) / (2 * eps)
    return grad.to(DEVICE)


def identity_tau(verts, grads):
    return torch.cat([
        grads.sum(dim=0),
        torch.cross(verts, grads, dim=1).sum(dim=0),
    ])


def analyse(name, pc, cam, pipe, bg, wc, wd, mode="color"):
    print(f"\n{'=' * 70}\nSCENE: {name}  (loss={mode})\n{'=' * 70}")
    ag = analytic_grads(pc, cam, pipe, bg, wc, wd, mode,
                        optimize_pose_only=False)
    an = ag["vertices"].reshape(-1)

    fd1 = fd_all_vertices(pc, cam, pipe, bg, wc, wd, mode, 1e-4)
    fd2 = fd_all_vertices(pc, cam, pipe, bg, wc, wd, mode, 3e-4)

    # Stability: the two eps estimates must agree for the FD to be trusted.
    scale = torch.maximum(fd1.abs(), fd2.abs()).clamp_min(1e-3)
    stable = ((fd1 - fd2).abs() / scale) < 0.05
    fd = 0.5 * (fd1 + fd2)

    denom = torch.maximum(an.abs(), fd.abs()).clamp_min(1e-3)
    rel = (an - fd).abs() / denom
    bad = stable & (rel > 0.10)

    V = pc.vertices.shape[0]
    tri = pc._triangle_indices.long()
    incident = torch.zeros(V, dtype=torch.long)
    for k in range(3):
        incident.scatter_add_(0, tri[:, k].cpu(),
                              torch.ones(tri.shape[0], dtype=torch.long))

    print(f"coords: {an.numel()}  stable-FD: {int(stable.sum())}  "
          f"bad (stable & rel>0.10): {int(bad.sum())}")
    print(f"median rel err over stable coords: "
          f"{rel[stable].median().item():.4f}")

    bad_v = sorted(set((torch.nonzero(bad).reshape(-1) // 3).tolist()))
    print(f"vertices with at least one bad coord: {len(bad_v)} of {V}")
    for v in bad_v[:15]:
        comps = []
        for c in range(3):
            i = 3 * v + c
            mark = "BAD" if bad[i] else ("unst" if not stable[i] else "ok")
            comps.append(f"{'xyz'[c]}: an={an[i]:+9.3f} fd={fd[i]:+9.3f} {mark}")
        print(f"  v{v:3d} (tris={incident[v]:2d})  " + "   ".join(comps))
    if len(bad_v) > 15:
        print(f"  ... and {len(bad_v) - 15} more")

    verts = pc.vertices.detach()
    tau_an = ag["tau"]
    tau_from_an = identity_tau(verts, ag["vertices"])
    tau_from_fd = identity_tau(verts, fd.reshape(-1, 3))
    print("\ntau analytic          :", [f"{x:+10.3f}" for x in tau_an.tolist()])
    print("tau from analytic vgrad:", [f"{x:+10.3f}" for x in tau_from_an.tolist()])
    print("tau from FD vgrad      :", [f"{x:+10.3f}" for x in tau_from_fd.tolist()])


def main():
    sigma = 0.5
    W = H = 64
    fx = fy = 60.0
    pipe = MockPipe()
    bg = torch.zeros(3, device=DEVICE)
    wc, wd = make_loss_weights(H, W)

    # --- two triangles sharing an edge ---
    qv = torch.tensor([[-0.9, -0.7, 2.0],
                       [0.8, -0.6, 2.1],
                       [0.7, 0.8, 1.9],
                       [-0.8, 0.7, 2.05]], device=DEVICE)
    qt = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32, device=DEVICE)
    qo = torch.tensor([[0.7], [0.8], [0.6], [0.75]], device=DEVICE)
    qf = 0.8 * torch.randn(4, 1, 3, device=DEVICE)
    pc2 = MockTriangles(qv.requires_grad_(True), qt,
                        qo.requires_grad_(True), qf.requires_grad_(True),
                        sigma)
    cam2 = MockCamera(torch.eye(4, device=DEVICE), fx, fy, W / 2, H / 2, W, H)
    analyse("two triangles sharing an edge", pc2, cam2, pipe, bg, wc, wd)

    # --- full grid ---
    verts, tri_idx, opacity, features_dc = build_scene()
    pc = MockTriangles(verts.requires_grad_(True), tri_idx,
                       opacity.requires_grad_(True),
                       features_dc.requires_grad_(True), sigma)
    cam = MockCamera(torch.eye(4, device=DEVICE), fx, fy, W / 2, H / 2, W, H)
    analyse("8x8 grid (98 triangles)", pc, cam, pipe, bg, wc, wd)


if __name__ == "__main__":
    main()
