"""
Capture the raw (P+V, 6) grad_tau rows of the paper rasteriser on the
minimal failing scene (two triangles sharing an edge, uniform colours) and
compare against the identity from vertex gradients and finite differences.

Rows [0:P]  = per-triangle contributions from the preprocess backward
              (edge normal/offset path + triangle-depth path).
Rows [P:]   = per-vertex contributions from propagateScreenGradsToPose
              (barycentric + vertex-depth paths; ~0 with uniform colours).

Run:  python debug/gradcheck_taurows.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import diff_triangle_rasterization as dtr

CAPTURE = {}
_orig = dtr._C.rasterize_triangles_backward


def _wrapped(*args):
    out = _orig(*args)
    CAPTURE["vertex_grads"] = out[0].clone()
    CAPTURE["tau_rows"] = out[4].clone()
    return out


dtr._C.rasterize_triangles_backward = _wrapped

from debug.gradcheck_rasterizer import (          # noqa: E402
    MockCamera, MockPipe, MockTriangles, make_loss_weights, analytic_grads,
    fd_loss,
)
from debug.gradcheck_vertices_full import identity_tau  # noqa: E402

DEVICE = "cuda"
torch.manual_seed(0)


def main():
    verts = torch.tensor([
        [-0.9, -0.7, 2.0], [0.8, -0.6, 2.1], [0.7, 0.8, 1.9], [-0.8, 0.7, 2.05],
    ], device=DEVICE)
    tris = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32, device=DEVICE)
    V = verts.shape[0]
    P = tris.shape[0]
    opacity = (0.5 + 0.3 * torch.rand(V, 1, device=DEVICE))
    features = torch.full((V, 1, 3), 0.6, device=DEVICE)

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
    rows = CAPTURE["tau_rows"]
    print(f"tau_rows shape: {tuple(rows.shape)}  (P={P}, V={V})")
    for i in range(rows.shape[0]):
        label = f"tri{i}" if i < P else f"vtx{i - P}"
        print(f"  {label}: " + " ".join(f"{x:+10.4f}" for x in rows[i].tolist()))

    tau_pre = rows[:P].sum(dim=0)
    tau_bary = rows[P:].sum(dim=0)
    tau_v = identity_tau(pc.vertices.detach(), ag["vertices"])
    print("\nsum rows[:P]  (preprocess) :",
          [f"{x:+9.3f}" for x in tau_pre.tolist()])
    print("sum rows[P:]  (barycentric):",
          [f"{x:+9.3f}" for x in tau_bary.tolist()])
    print("tau total analytic         :",
          [f"{x:+9.3f}" for x in ag["tau"].tolist()])
    print("identity from vertex grads :",
          [f"{x:+9.3f}" for x in tau_v.tolist()])

    print("\nanalytic vertex grads:")
    for v in range(V):
        print(f"  v{v}: " + " ".join(
            f"{x:+10.4f}" for x in ag["vertices"][v].tolist()))

    # FD vertex grads for ground truth (uniform colours -> only edge path)
    print("\nFD vertex grads (eps=1e-4):")
    flat = pc.vertices.detach().reshape(-1)
    fd = torch.zeros(V * 3)
    for i in range(V * 3):
        orig = flat[i].item()
        flat[i] = orig + 1e-4
        lp = fd_loss(pc, cam, pipe, bg, wc, wd, "color")
        flat[i] = orig - 1e-4
        lm = fd_loss(pc, cam, pipe, bg, wc, wd, "color")
        flat[i] = orig
        fd[i] = (lp - lm) / 2e-4
    fd = fd.reshape(V, 3)
    for v in range(V):
        print(f"  v{v}: " + " ".join(f"{x:+10.4f}" for x in fd[v].tolist()))
    print("\nidentity from FD vgrads    :",
          [f"{x:+9.3f}" for x in identity_tau(
              pc.vertices.detach().cpu(), fd).tolist()])


if __name__ == "__main__":
    main()
