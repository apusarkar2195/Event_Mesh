#!/usr/bin/env python3
"""
Evaluate Replica SLAM results: render from GT poses, compute PSNR/SSIM/LPIPS.

For each scene in Replica (office0-4, room0-2):
  1. Find latest SLAM run in results/replica_<scene>/
  2. Load config → dataset path + calibration
  3. Load Delaunay-refined triangle model from results/replica_meshes/<scene>/
  4. Load keyframes (UIDs + estimated poses) from the latest checkpoint
  5. For each keyframe (and every-5-frames), set GT pose, render:
       a) meshsplatting normal
       b) meshsplatting opaque  (opacity=0.99, sigma=0.0001)
       c) Open3D mesh renderer  (mesh_delaunay.ply)
  6. Compute PSNR / SSIM / LPIPS and record FPS
  7. Save rendered images (keyframes only) and JSON results

Usage (from repo root):
  .venv/bin/python scripts/eval_replica.py [--results-dir results] [--out-base results/replica_meshes]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

# ── repo root on the path ────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import open3d as o3d
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from mesh_splatting.render_utils import get_psnr
from mesh_splatting.scene.triangle_model import TriangleModel
from mesh_splatting.triangle_renderer import render
from mesh_splatting.utils.graphics_utils import getProjectionMatrix2
from mesh_splatting.utils.loss_utils import ssim
from utils.cameras import Camera
from utils.config_utils import load_config
from utils.dataset import load_dataset

# ─────────────────────────────────────────────────────────────────────────────
REPLICA_SCENES = [
    "office0", "office1", "office2", "office3", "office4",
    "room0", "room1", "room2",
]

# ─────────────────────────────────────────────────────────────────────────────
# Directory helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_latest_run_dir(results_base: Path, folder: str) -> Path | None:
    run_dir = results_base / folder
    if not run_dir.is_dir():
        return None
    subdirs = [p for p in run_dir.iterdir() if p.is_dir()]
    return max(subdirs, key=lambda p: p.stat().st_mtime) if subdirs else None


def find_latest_iteration(mesh_dir: Path) -> Path | None:
    if not mesh_dir.is_dir():
        return None
    pattern = re.compile(r"iteration_(\d+)$")
    candidates = [(int(m.group(1)), p) for p in mesh_dir.iterdir()
                  if p.is_dir() and (m := pattern.match(p.name))]
    return sorted(candidates)[-1][1] if candidates else None


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline params shim (avoids argparse overhead)
# ─────────────────────────────────────────────────────────────────────────────

def make_pipe() -> SimpleNamespace:
    pipe = SimpleNamespace()
    pipe.convert_SHs_python = False
    pipe.compute_cov3D_python = False
    pipe.depth_ratio = 1.0
    pipe.debug = False
    return pipe


# ─────────────────────────────────────────────────────────────────────────────
# Opaque render override
# ─────────────────────────────────────────────────────────────────────────────

def render_opaque(cam, triangles, pipe, background, opacity: float = 0.99, sigma: float = 0.0001):
    """Temporarily override opacity and sigma, render, then restore."""
    orig_sigma = triangles._sigma
    orig_weight = triangles.vertex_weight.data.clone()
    orig_floor = triangles.opacity_floor

    # Set opaque parameters
    triangles._sigma = math.log(sigma)          # inverse_exponential_activation
    triangles.opacity_floor = 0.0
    inv_sig_val = math.log(opacity / (1.0 - opacity))  # logit(opacity)
    with torch.no_grad():
        triangles.vertex_weight.data.fill_(inv_sig_val)

    try:
        with torch.no_grad():
            result = render(cam, triangles, pipe, background)
    finally:
        triangles._sigma = orig_sigma
        triangles.vertex_weight.data.copy_(orig_weight)
        triangles.opacity_floor = orig_floor

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Open3D offscreen mesh renderer
# ─────────────────────────────────────────────────────────────────────────────

class Open3DRenderer:
    """
    Offscreen renderer for a triangle mesh using Open3D.

    Coordinate convention note
    --------------------------
    The SLAM world frame inherits the Replica traj.txt convention (OpenGL-like:
    Y-up, camera looks along -Z in camera space).  Open3D's OffscreenRenderer
    expects OpenCV convention (Y-down, camera looks along +Z).  We apply the
    standard Y/Z flip to the extrinsic to convert between them:

        extrinsic_opencv = diag([1,-1,-1,1]) @ extrinsic_slam
    """

    # Flip matrix to convert OpenGL w2c → OpenCV w2c
    _FLIP = np.diag([1.0, -1.0, -1.0, 1.0])

    def __init__(self, mesh_path: Path, W: int, H: int,
                 fx: float, fy: float, cx: float, cy: float):
        self.W = W
        self.H = H
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        self.renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)
        self.renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])

        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        # Use unlit shader so baked vertex colours are evaluated without
        # lighting artefacts (the mesh has colours from the SH model).
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"

        self.renderer.scene.add_geometry("mesh", mesh, mat)

    def render(self, R_w2c: np.ndarray, T_w2c: np.ndarray):
        """
        Render from the given world-to-camera pose.

        R_w2c: (3,3) rotation  (world → camera, SLAM convention)
        T_w2c: (3,)  translation (world → camera)

        Returns (rgb_uint8 H×W×3, depth_f32 H×W  in metres).
        """
        # Build 4×4 SLAM extrinsic
        extrinsic_slam = np.eye(4, dtype=np.float64)
        extrinsic_slam[:3, :3] = R_w2c
        extrinsic_slam[:3, 3] = T_w2c

        # Convert to OpenCV convention
        extrinsic = self._FLIP @ extrinsic_slam

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            self.W, self.H, self.fx, self.fy, self.cx, self.cy
        )
        self.renderer.setup_camera(intrinsic, extrinsic)

        rgb_img = self.renderer.render_to_image()
        depth_img = self.renderer.render_to_depth_image(z_in_view_space=True)

        rgb = np.asarray(rgb_img).copy()   # uint8 H×W×3
        depth = np.asarray(depth_img).copy()  # float32 H×W

        depth = np.where(np.isinf(depth) | (depth <= 0), 0.0, depth).astype(np.float32)

        return rgb, depth


# ─────────────────────────────────────────────────────────────────────────────
# Image save helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_rgb(img_chw: torch.Tensor, path: Path):
    """Save a (3,H,W) float [0,1] tensor as BGR PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (img_chw.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


def save_rgb_numpy(img_hwc: np.ndarray, path: Path):
    """Save a (H,W,3) uint8 numpy array (RGB) as BGR PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img_hwc, cv2.COLOR_RGB2BGR))


def save_depth_png(depth_hw: np.ndarray, path: Path, scale: float = 1000.0):
    """Save depth (metres) as 16-bit PNG (millimetres by default)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    d16 = (depth_hw * scale).clip(0, 65535).astype(np.uint16)
    cv2.imwrite(str(path), d16)


def depth_tensor_to_numpy(depth_chw: torch.Tensor) -> np.ndarray:
    """(1,H,W) float tensor → (H,W) numpy float32."""
    return depth_chw.squeeze(0).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(rendered: torch.Tensor, gt: torch.Tensor,
                    cal_lpips) -> tuple[float, float, float]:
    """All tensors (3,H,W) float [0,1] on CUDA."""
    mask = gt > 0
    psnr_score = get_psnr(rendered[mask].unsqueeze(0), gt[mask].unsqueeze(0)).item()
    ssim_score = ssim(rendered.unsqueeze(0), gt.unsqueeze(0)).item()
    with torch.no_grad():
        lpips_score = cal_lpips(rendered.unsqueeze(0), gt.unsqueeze(0)).item()
    return psnr_score, ssim_score, lpips_score


# ─────────────────────────────────────────────────────────────────────────────
# Camera from GT pose
# ─────────────────────────────────────────────────────────────────────────────

def make_gt_camera(dataset, idx: int, proj_matrix: torch.Tensor) -> Camera:
    """Create a Camera for frame *idx* and immediately set its pose to GT.

    GT poses come from numpy (float64); the projection matrix and CUDA
    rasterizer expect float32, so we cast here.
    """
    cam = Camera.init_from_dataset(dataset, idx, proj_matrix)
    cam.update_RT(cam.R_gt.float(), cam.T_gt.float())
    return cam


# ─────────────────────────────────────────────────────────────────────────────
# Per-scene evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_scene(
    scene_name: str,
    run_dir: Path,
    state_path: Path,   # iteration_XXXX directory (for keyframes_state.pt)
    mesh_out_dir: Path,  # results/replica_meshes/<scene>
    repo_root: Path,
    save_images: bool = True,
):
    print(f"\n{'='*60}")
    print(f"  Scene: {scene_name}")
    print(f"  Run:   {run_dir}")
    print(f"  State: {state_path}")
    print(f"  Mesh:  {mesh_out_dir}")
    print(f"{'='*60}")

    # ── config & dataset ────────────────────────────────────────────────────
    config_path = run_dir / "config.yml"
    config = load_config(str(config_path))

    dataset_path_raw = config["Dataset"]["dataset_path"]
    dataset_path = (
        dataset_path_raw if os.path.isabs(dataset_path_raw)
        else str(repo_root / dataset_path_raw)
    )
    print(f"  Dataset: {dataset_path}")

    args = SimpleNamespace(distributed=False)
    dataset = load_dataset(args, dataset_path, config)

    # ── projection matrix ───────────────────────────────────────────────────
    proj_matrix = getProjectionMatrix2(
        znear=0.01, zfar=100.0,
        fx=dataset.fx, fy=dataset.fy,
        cx=dataset.cx, cy=dataset.cy,
        W=dataset.width, H=dataset.height,
    ).transpose(0, 1)   # matches slam.py convention

    # ── triangle model (Delaunay refined) ───────────────────────────────────
    model_pt = mesh_out_dir / "point_cloud_state_dict.pt"
    if not model_pt.exists():
        print(f"  [skip] model not found: {model_pt}")
        return None

    triangles = TriangleModel(sh_degree=0, config=config)
    triangles.load_parameters(str(mesh_out_dir), device="cuda")
    print(f"  Loaded model: {triangles.vertices.shape[0]} vertices, "
          f"{triangles._triangle_indices.shape[0]} triangles")

    # ── pipe & background ───────────────────────────────────────────────────
    pipe = make_pipe()
    background = torch.zeros(3, dtype=torch.float32, device="cuda")

    # ── LPIPS (shared across all frames) ────────────────────────────────────
    cal_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to("cuda")

    # ── Open3D renderer ─────────────────────────────────────────────────────
    mesh_ply = mesh_out_dir / "mesh_delaunay.ply"
    if not mesh_ply.exists():
        mesh_ply = mesh_out_dir / "mesh_delaunay_postprocess.ply"
    if not mesh_ply.exists():
        print(f"  [skip] mesh PLY not found in {mesh_out_dir}")
        return None

    print(f"  Loading Open3D renderer (EGL init, may take ~10s)...")
    o3d_renderer = Open3DRenderer(
        mesh_ply,
        W=dataset.width, H=dataset.height,
        fx=dataset.fx, fy=dataset.fy,
        cx=dataset.cx, cy=dataset.cy,
    )

    # ── load keyframe UIDs ───────────────────────────────────────────────────
    kf_path = state_path / "keyframes_state.pt"
    if not kf_path.exists():
        print(f"  [warn] keyframes_state.pt not found, using every frame")
        kf_uids = list(range(0, dataset.num_imgs, 5))
    else:
        payload = torch.load(str(kf_path), map_location="cpu")
        kf_states = payload.get("keyframes", [])
        kf_uids = [int(s["uid"]) for s in kf_states
                   if 0 <= int(s["uid"]) < dataset.num_imgs]
    print(f"  Keyframes: {len(kf_uids)}")

    # ── output dirs ─────────────────────────────────────────────────────────
    render_dir = mesh_out_dir / "renders"
    dirs = {
        "rgb":         render_dir / "rgb",
        "depth":       render_dir / "depth",
        "rgb_opaque":  render_dir / "rgb_opaque",
        "depth_opaque":render_dir / "depth_opaque",
        "rgb_o3d":     render_dir / "rgb_open3d",
        "depth_o3d":   render_dir / "depth_open3d",
    }
    if save_images:
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Helper: render one frame in all three modes
    # ─────────────────────────────────────────────────────────────────────────

    def render_frame(uid: int, do_save: bool):
        """
        Returns dict with per-mode (psnr, ssim, lpips, time_s) or None on error.
        Processes one camera at a time to keep memory low.
        """
        try:
            cam = make_gt_camera(dataset, uid, proj_matrix)
        except Exception as e:
            print(f"    [warn] frame {uid}: {e}")
            return None

        gt = cam.original_image.cuda()   # (3,H,W) float [0,1]

        # R_gt, T_gt are stored on cam.device already
        R_np = cam.R_gt.cpu().numpy()
        T_np = cam.T_gt.cpu().numpy()

        results = {}

        # ── splatting (normal) ──────────────────────────────────────────────
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            pkg = render(cam, triangles, pipe, background)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        rendered = torch.clamp(pkg["render"], 0, 1)
        depth_np = depth_tensor_to_numpy(pkg["surf_depth"])

        if do_save:
            save_rgb(rendered, dirs["rgb"] / f"{uid:06d}.png")
            save_depth_png(depth_np, dirs["depth"] / f"{uid:06d}.png")

        p, s, l = compute_metrics(rendered, gt, cal_lpips)
        results["splatting"] = dict(psnr=p, ssim=s, lpips=l, time_s=t1 - t0)
        del pkg, rendered, depth_np

        # ── splatting (opaque) ──────────────────────────────────────────────
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        pkg_o = render_opaque(cam, triangles, pipe, background)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        rendered_o = torch.clamp(pkg_o["render"], 0, 1)
        depth_np_o = depth_tensor_to_numpy(pkg_o["surf_depth"])

        if do_save:
            save_rgb(rendered_o, dirs["rgb_opaque"] / f"{uid:06d}.png")
            save_depth_png(depth_np_o, dirs["depth_opaque"] / f"{uid:06d}.png")

        p, s, l = compute_metrics(rendered_o, gt, cal_lpips)
        results["opaque"] = dict(psnr=p, ssim=s, lpips=l, time_s=t1 - t0)
        del pkg_o, rendered_o, depth_np_o

        # ── Open3D ──────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        rgb_o3d, depth_o3d = o3d_renderer.render(R_np, T_np)
        t1 = time.perf_counter()

        # Convert Open3D RGB (uint8 H×W×3) → torch tensor (3,H,W) float [0,1]
        rgb_o3d_t = (
            torch.from_numpy(rgb_o3d.copy()).float() / 255.0
        ).permute(2, 0, 1).cuda()

        if do_save:
            save_rgb_numpy(rgb_o3d, dirs["rgb_o3d"] / f"{uid:06d}.png")
            save_depth_png(depth_o3d, dirs["depth_o3d"] / f"{uid:06d}.png")

        p, s, l = compute_metrics(rgb_o3d_t, gt, cal_lpips)
        results["open3d"] = dict(psnr=p, ssim=s, lpips=l, time_s=t1 - t0)
        del rgb_o3d_t, rgb_o3d, depth_o3d

        del cam, gt
        torch.cuda.empty_cache()

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Keyframe evaluation (save images)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n  -- Keyframe evaluation --")
    kf_results: dict[str, list] = {
        "splatting": [], "opaque": [], "open3d": []
    }

    for i, uid in enumerate(kf_uids):
        frame_res = render_frame(uid, do_save=save_images)
        if frame_res is None:
            continue
        for mode in ("splatting", "opaque", "open3d"):
            kf_results[mode].append(frame_res[mode])
        if (i + 1) % 20 == 0 or i == 0:
            print(f"    [{i+1}/{len(kf_uids)}] frame {uid}  "
                  f"PSNR splatting={frame_res['splatting']['psnr']:.2f}  "
                  f"o3d={frame_res['open3d']['psnr']:.2f}")

    # ─────────────────────────────────────────────────────────────────────────
    # Every-5-frames evaluation (no image saving)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n  -- Every-5-frames evaluation --")
    all_uids = list(range(0, dataset.num_imgs, 5))
    ev5_results: dict[str, list] = {
        "splatting": [], "opaque": [], "open3d": []
    }

    for i, uid in enumerate(all_uids):
        frame_res = render_frame(uid, do_save=False)
        if frame_res is None:
            continue
        for mode in ("splatting", "opaque", "open3d"):
            ev5_results[mode].append(frame_res[mode])
        if (i + 1) % 50 == 0 or i == 0:
            print(f"    [{i+1}/{len(all_uids)}] frame {uid}  "
                  f"PSNR={frame_res['splatting']['psnr']:.2f}")

    # ─────────────────────────────────────────────────────────────────────────
    # Aggregate metrics
    # ─────────────────────────────────────────────────────────────────────────
    def aggregate(results_list: list[dict]) -> dict:
        if not results_list:
            return {}
        keys = results_list[0].keys()
        out = {}
        for k in keys:
            vals = [r[k] for r in results_list]
            out[f"mean_{k}"] = float(np.mean(vals))
            out[f"std_{k}"] = float(np.std(vals))
        fps_times = [r["time_s"] for r in results_list if r["time_s"] > 0]
        out["fps"] = float(1.0 / np.mean(fps_times)) if fps_times else 0.0
        return out

    summary = {
        "scene": scene_name,
        "keyframes": {
            mode: aggregate(kf_results[mode])
            for mode in ("splatting", "opaque", "open3d")
        },
        "every5": {
            mode: aggregate(ev5_results[mode])
            for mode in ("splatting", "opaque", "open3d")
        },
    }

    # ── pretty print ────────────────────────────────────────────────────────
    print(f"\n  {'─'*50}")
    for split_name, split_data in [("keyframes", summary["keyframes"]),
                                    ("every5",    summary["every5"])]:
        print(f"  [{split_name}]")
        for mode, metrics in split_data.items():
            if not metrics:
                continue
            print(f"    {mode:12s}  "
                  f"PSNR={metrics.get('mean_psnr', 0):.3f}  "
                  f"SSIM={metrics.get('mean_ssim', 0):.4f}  "
                  f"LPIPS={metrics.get('mean_lpips', 0):.4f}  "
                  f"FPS={metrics.get('fps', 0):.1f}")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate Replica SLAM results")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out-base", type=Path, default=Path("results/replica_meshes"))
    parser.add_argument("--scenes", nargs="+", default=REPLICA_SCENES,
                        help="Which scenes to evaluate (default: all)")
    parser.add_argument("--no-save-images", action="store_true",
                        help="Skip saving rendered images (only compute metrics)")
    args = parser.parse_args()

    repo_root = _ROOT
    results_base = (repo_root / args.results_dir).resolve()
    out_base = (repo_root / args.out_base).resolve()

    all_summaries = {}

    for scene in args.scenes:
        folder = f"replica_{scene}"
        run_dir = find_latest_run_dir(results_base, folder)
        if run_dir is None:
            print(f"[skip] {folder}: no run directory found")
            continue

        mesh_dir = run_dir / "mesh"
        iter_dir = find_latest_iteration(mesh_dir)
        if iter_dir is None:
            print(f"[skip] {folder}: no iteration_* checkpoint found")
            continue

        mesh_out_dir = out_base / scene
        if not mesh_out_dir.is_dir():
            print(f"[skip] {scene}: mesh output dir not found: {mesh_out_dir}")
            continue

        try:
            summary = evaluate_scene(
                scene_name=scene,
                run_dir=run_dir,
                state_path=iter_dir,
                mesh_out_dir=mesh_out_dir,
                repo_root=repo_root,
                save_images=not args.no_save_images,
            )
        except Exception as e:
            import traceback
            print(f"[ERROR] {scene}: {e}")
            traceback.print_exc()
            continue

        if summary is not None:
            all_summaries[scene] = summary

    # ── save combined JSON ───────────────────────────────────────────────────
    out_json = out_base / "eval_results.json"
    with open(out_json, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nResults saved to {out_json}")

    # ── cross-scene averages ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  CROSS-SCENE AVERAGES")
    print("="*60)
    for split in ("keyframes", "every5"):
        print(f"\n  [{split}]")
        for mode in ("splatting", "opaque", "open3d"):
            psnrs, ssims, lpipss, fpss = [], [], [], []
            for scene, s in all_summaries.items():
                m = s.get(split, {}).get(mode, {})
                if not m:
                    continue
                psnrs.append(m.get("mean_psnr", 0))
                ssims.append(m.get("mean_ssim", 0))
                lpipss.append(m.get("mean_lpips", 0))
                fpss.append(m.get("fps", 0))
            if not psnrs:
                continue
            print(f"    {mode:12s}  "
                  f"PSNR={np.mean(psnrs):.3f}  "
                  f"SSIM={np.mean(ssims):.4f}  "
                  f"LPIPS={np.mean(lpipss):.4f}  "
                  f"FPS={np.mean(fpss):.1f}")


if __name__ == "__main__":
    main()
