"""
Save depth renders from a saved TriangleModel checkpoint.

Usage:
    .venv/bin/python scripts/saver_triangle.py \
        --result_dir results/replica_office0/2026-03-03-12-52-51 \
        --iteration 1800

Output: <result_dir>/depth_renders/
    <uid:05d>_rgb.png    - uint8 RGB render
    <uid:05d>_depth.npy  - float32 depth map (metres, shape H x W)
    cameras.npz          - intrinsics + world_view_transform + projection_matrix
"""

import argparse
import math
import os

import numpy as np
import torch
import yaml
from PIL import Image as PILImage
from tqdm import tqdm

from mesh_splatting.scene.triangle_model import TriangleModel
from mesh_splatting.triangle_renderer import render
from mesh_splatting.utils.graphics_utils import getProjectionMatrix2
from utils.cameras import Camera


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def make_camera(uid, R, T, fx, fy, cx, cy, W, H, device="cuda"):
    znear, zfar = 0.01, 100.0
    projection_matrix = getProjectionMatrix2(
        znear=znear, zfar=zfar, fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H
    ).transpose(0, 1)

    fovx = focal2fov(fx, W)
    fovy = focal2fov(fy, H)

    gt_T = torch.eye(4, device=device)  # not needed for rendering
    cam = Camera(
        uid=uid,
        color=None,
        depth=None,
        gt_T=gt_T,
        projection_matrix=projection_matrix,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        fovx=fovx,
        fovy=fovy,
        image_height=H,
        image_width=W,
        device=device,
    )
    # Set optimized pose (world_view_transform uses self.R and self.T)
    cam.R = R.to(device=device)
    cam.T = T.to(device=device)
    return cam


class PipelineParams:
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_dir",
        required=True,
        help="e.g. results/replica_office0/2026-03-03-12-52-51",
    )
    parser.add_argument("--iteration", type=int, default=1800)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    ckpt_dir = os.path.join(args.result_dir, "mesh", f"iteration_{args.iteration}")
    out_dir = os.path.join(args.result_dir, "depth_renders")
    os.makedirs(out_dir, exist_ok=True)

    # ── Load config ──────────────────────────────────────────────────────────
    cfg_path = os.path.join(args.result_dir, "config.yml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cal = cfg["Dataset"]["Calibration"]
    fx, fy = float(cal["fx"]), float(cal["fy"])
    cx, cy = float(cal["cx"]), float(cal["cy"])
    W, H = int(cal["width"]), int(cal["height"])
    print(f"Intrinsics: fx={fx} fy={fy} cx={cx} cy={cy} W={W} H={H}")

    # ── Load triangle model ───────────────────────────────────────────────────
    print(f"Loading model from {ckpt_dir}")
    model = TriangleModel(sh_degree=0)
    model.load_parameters_for_slam_resume(ckpt_dir, device=args.device)
    print(f"Loaded {model._triangle_indices.shape[0]} triangles")

    # ── Load keyframe poses ───────────────────────────────────────────────────
    kf_path = os.path.join(ckpt_dir, "keyframes_state.pt")
    kf_state = torch.load(kf_path, map_location="cpu")
    keyframes = kf_state["keyframes"]
    print(f"Loaded {len(keyframes)} keyframes")

    pipe = PipelineParams()
    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=args.device)

    uid_list = []
    fx_list, fy_list, cx_list, cy_list = [], [], [], []
    widths, heights = [], []
    wvts, proj_mats = [], []

    with torch.no_grad():
        for kf in tqdm(keyframes, desc="Rendering keyframes"):
            uid = kf["uid"]
            R = kf["R"]  # (3, 3)
            T = kf["T"]  # (3,)

            cam = make_camera(uid, R, T, fx, fy, cx, cy, W, H, device=args.device)
            render_pkg = render(cam, model, pipe, background)

            rgb = render_pkg["render"]               # (3, H, W) downsampled
            depth = render_pkg["render_depth_median"] # (1, H, W)

            # Use full-res if available and same size
            full_rgb = render_pkg.get("full_image")
            if full_rgb is not None and full_rgb.shape[-2:] == (H, W):
                rgb = full_rgb

            rgb_np = np.clip(rgb.permute(1, 2, 0).cpu().numpy(), 0.0, 1.0)
            PILImage.fromarray((rgb_np * 255).astype(np.uint8)).save(
                os.path.join(out_dir, f"{uid:05d}_rgb.png")
            )

            depth_np = depth.squeeze(0).cpu().numpy().astype(np.float32)
            np.save(os.path.join(out_dir, f"{uid:05d}_depth.npy"), depth_np)

            uid_list.append(uid)
            fx_list.append(fx)
            fy_list.append(fy)
            cx_list.append(cx)
            cy_list.append(cy)
            widths.append(W)
            heights.append(H)
            wvts.append(cam.world_view_transform.cpu().numpy())
            proj_mats.append(cam.projection_matrix.cpu().numpy())

    np.savez(
        os.path.join(out_dir, "cameras.npz"),
        frame_ids=np.array(uid_list, dtype=np.int32),
        fx=np.array(fx_list, dtype=np.float32),
        fy=np.array(fy_list, dtype=np.float32),
        cx=np.array(cx_list, dtype=np.float32),
        cy=np.array(cy_list, dtype=np.float32),
        widths=np.array(widths, dtype=np.int32),
        heights=np.array(heights, dtype=np.int32),
        world_view_transforms=np.stack(wvts).astype(np.float32),
        projection_matrices=np.stack(proj_mats).astype(np.float32),
    )
    print(f"Saved {len(uid_list)} depth renders to {out_dir}")


if __name__ == "__main__":
    main()
