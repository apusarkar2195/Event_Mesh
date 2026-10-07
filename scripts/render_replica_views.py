#!/usr/bin/env python3
"""
Render RGB and depth from mesh-splatting and from Open3D mesh for Replica runs.

For each (state_path, run_dir, out_dir):
  - Load config from run_dir/config.yml (or config.yaml)
  - Load GT poses from dataset (config Dataset.dataset_path)
  - Render with mesh-splatting -> saves to out_dir/renders/rgb, renders/depth
  - Render with sigma=0.0001 and opacity~0.99 -> saves to renders/rgb_opaque, renders/depth_opaque
  - Render mesh_delaunay.ply with Open3D at same poses -> saves to out_dir/open3d/
  - Repeat using only keyframe poses from keyframes_state.pt -> out_dir/keyframes/renders/, keyframes/open3d/

Usage:
  python scripts/render_replica_views.py \\
    --state_path results/replica_office0/2026-03-01-23-29-09/mesh/iteration_2000 \\
    --out_dir results/replica_meshes/office0

  Or run for multiple scenes (e.g. from run_delaunay_replica_latest output dirs).
  If config has no dataset_path, pass --dataset_base /path/to/replica/sequences
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import open3d as o3d
import open3d.visualization.rendering as rendering

# Repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from argparse import Namespace
from mesh_splatting.scene.triangle_model import TriangleModel
from mesh_splatting.triangle_renderer import render as triangle_render
from mesh_splatting.arguments import PipelineParams
from mesh_splatting.utils.graphics_utils import getProjectionMatrix2
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.cameras import Camera
from utils.render_utils import save_img_u8, save_img_depth_png

# Optional metrics
try:
    import lpips
    _LPIPS_FN = None

    def _get_lpips():
        global _LPIPS_FN
        if _LPIPS_FN is None:
            _LPIPS_FN = lpips.LPIPS(net="alex").cuda().eval()
        return _LPIPS_FN
except Exception:
    _get_lpips = None

try:
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    from skimage.metrics import structural_similarity as ssim_fn
except Exception:
    psnr_fn = None
    ssim_fn = None


def _compute_pose_alignment(keyframes: list, dataset, n_gt: int):
    """
    Compute SE(3) alignment from keyframe poses to GT poses (Procrustes on camera centers).
    Returns T_align (4x4) so that c2w_gt_frame ≈ T_align @ c2w_kf; and list of (uid, c2w_kf, c2w_gt) for valid keyframes.
    """
    src_centers = []
    tgt_centers = []
    valid = []
    for kf in keyframes:
        uid = int(kf["uid"])
        if uid < 0 or uid >= n_gt:
            continue
        _, _, pose_gt = dataset[uid]
        pose_gt = pose_gt.cpu().numpy() if torch.is_tensor(pose_gt) else np.asarray(pose_gt)
        c2w_gt = np.asarray(pose_gt, dtype=np.float64).reshape(4, 4)
        R_kf = kf["R"].cpu().numpy() if torch.is_tensor(kf["R"]) else np.asarray(kf["R"])
        T_kf = kf["T"].cpu().numpy() if torch.is_tensor(kf["T"]) else np.asarray(kf["T"])
        c2w_kf = np.eye(4, dtype=np.float64)
        c2w_kf[:3, :3] = R_kf
        c2w_kf[:3, 3] = T_kf.ravel()
        src_centers.append(c2w_kf[:3, 3])
        tgt_centers.append(c2w_gt[:3, 3])
        valid.append((uid, c2w_kf, c2w_gt))
    if len(valid) < 3:
        return np.eye(4), valid
    src = np.array(src_centers).T  # 3 x N
    tgt = np.array(tgt_centers).T  # 3 x N
    mean_src = src.mean(axis=1, keepdims=True)
    mean_tgt = tgt.mean(axis=1, keepdims=True)
    H = (src - mean_src) @ (tgt - mean_tgt).T
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = mean_tgt - R @ mean_src
    T_align = np.eye(4, dtype=np.float64)
    T_align[:3, :3] = R
    T_align[:3, 3] = t.ravel()
    return T_align, valid


def _apply_alignment(c2w_kf: np.ndarray, T_align: np.ndarray):
    """Return R (3,3) and T (3,) for camera from aligned c2w = T_align @ c2w_kf."""
    c2w = T_align @ c2w_kf
    return c2w[:3, :3].copy(), c2w[:3, 3].copy()


def _compute_metrics(render: torch.Tensor, gt: torch.Tensor, lpips_fn=None):
    """render, gt: (C,H,W) in [0,1]. Returns dict with PSNR, SSIM, LPIPS (if available)."""
    out = {}
    render = render.detach().float().clamp(0, 1)
    gt = gt.float().clamp(0, 1)
    if render.shape != gt.shape:
        return out
    # PSNR
    if psnr_fn is not None:
        r = render.permute(1, 2, 0).cpu().numpy()
        g = gt.permute(1, 2, 0).cpu().numpy()
        out["PSNR"] = float(psnr_fn(g, r, data_range=1.0))
    # SSIM
    if ssim_fn is not None:
        r = render.permute(1, 2, 0).cpu().numpy()
        g = gt.permute(1, 2, 0).cpu().numpy()
        out["SSIM"] = float(ssim_fn(g, r, channel_axis=2, data_range=1.0))
    # LPIPS (expects [-1,1] and batch dimension)
    if lpips_fn is not None and render.device.type == "cuda":
        r = (render.unsqueeze(0) * 2 - 1).to(render.device)
        g = (gt.unsqueeze(0) * 2 - 1).to(render.device)
        out["LPIPS"] = float(lpips_fn(r, g).item())
    return out


def _load_image_to_tensor(path: Path):
    """Load PNG as tensor (3,H,W) in [0,1]."""
    try:
        import imageio
        img = imageio.v3.imread(str(path))
    except Exception:
        import cv2
        img = cv2.imread(str(path))
        if img is not None:
            img = img[:, :, ::-1]
    if img is None:
        return None
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    t = torch.from_numpy(arr).float() / 255.0
    if t.dim() == 2:
        t = t.unsqueeze(0).expand(3, -1, -1)
    else:
        t = t.permute(2, 0, 1)
    return t


def _compute_metrics_o3d(dataset, frame_indices, o3d_rgb_dir: Path, lpips_fn=None):
    """Load saved Open3D RGB and GT per frame; return list of metric dicts."""
    results = []
    o3d_rgb_dir = Path(o3d_rgb_dir)
    for idx, i in enumerate(frame_indices):
        path = o3d_rgb_dir / f"{idx:05d}.png"
        if not path.is_file():
            continue
        render_t = _load_image_to_tensor(path)
        if render_t is None:
            continue
        gt_img = dataset[i][0]
        if torch.is_tensor(gt_img):
            gt_img = gt_img.cpu().float().clamp(0, 1)
            if gt_img.dim() == 4:
                gt_img = gt_img[0]
        else:
            gt_img = torch.from_numpy(np.asarray(gt_img)).float().clamp(0, 1)
            if gt_img.dim() == 3:
                gt_img = gt_img.permute(2, 0, 1)
        if render_t.shape != gt_img.shape:
            continue
        m = _compute_metrics(render_t, gt_img, lpips_fn)
        if m:
            results.append(m)
    return results


def _mean_metrics(metrics_list: list) -> dict:
    """Average PSNR, SSIM, LPIPS over list of metric dicts."""
    if not metrics_list:
        return {}
    out = {}
    for k in ("PSNR", "SSIM", "LPIPS"):
        vals = [m[k] for m in metrics_list if k in m]
        if vals:
            out[k] = float(np.mean(vals))
    return out


def _write_metrics(
    out_dir: Path,
    tag: str,
    n_frames: int,
    metrics_rgb: list,
    metrics_opaque: list,
    metrics_o3d: list,
    elapsed_total: float,
    t_rgb: float,
    t_opaque: float,
    t_o3d: float,
):
    """Write metrics.json and print PSNR/SSIM/LPIPS and speed for rgb, rgb_opaque, open3d."""
    out_dir = Path(out_dir)
    mean_rgb = _mean_metrics(metrics_rgb)
    mean_opaque = _mean_metrics(metrics_opaque)
    mean_o3d = _mean_metrics(metrics_o3d)
    sec_rgb = t_rgb / max(1, n_frames)
    sec_opaque = t_opaque / max(1, n_frames)
    sec_o3d = t_o3d / max(1, n_frames)
    data = {
        tag: {
            "rgb": {**mean_rgb, "sec_per_frame": sec_rgb, "fps": 1.0 / sec_rgb if sec_rgb > 0 else 0},
            "rgb_opaque": {**mean_opaque, "sec_per_frame": sec_opaque, "fps": 1.0 / sec_opaque if sec_opaque > 0 else 0},
            "open3d": {**mean_o3d, "sec_per_frame": sec_o3d, "fps": 1.0 / sec_o3d if sec_o3d > 0 else 0},
        }
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "metrics.json"
    import json
    existing = {}
    if json_path.is_file():
        try:
            with open(json_path) as f:
                existing = json.load(f)
        except Exception:
            pass
    existing.update(data)
    with open(json_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"  [{tag}] rgb:      PSNR={mean_rgb.get('PSNR', 0):.2f} SSIM={mean_rgb.get('SSIM', 0):.3f} LPIPS={mean_rgb.get('LPIPS', 0):.3f}  {1/sec_rgb:.2f} fps", flush=True)
    print(f"  [{tag}] rgb_opaq: PSNR={mean_opaque.get('PSNR', 0):.2f} SSIM={mean_opaque.get('SSIM', 0):.3f} LPIPS={mean_opaque.get('LPIPS', 0):.3f}  {1/sec_opaque:.2f} fps", flush=True)
    print(f"  [{tag}] open3d:   PSNR={mean_o3d.get('PSNR', 0):.2f} SSIM={mean_o3d.get('SSIM', 0):.3f} LPIPS={mean_o3d.get('LPIPS', 0):.3f}  {1/sec_o3d:.2f} fps", flush=True)


def _get_dataset_path(config: dict, run_dir: Path, dataset_base: Path | None, scene_short: str) -> Path:
    ds = config.get("Dataset", {})
    path = ds.get("dataset_path") or config.get("dataset_path")
    if path:
        p = Path(path)
        return p if p.is_absolute() else (ROOT / p)
    if dataset_base is not None:
        return Path(dataset_base) / scene_short
    raise ValueError(
        f"No dataset_path in config and no --dataset_base. "
        f"Set Dataset.dataset_path in {run_dir}/config.yml or pass --dataset_base"
    )


def _load_pipe(config: dict):
    from argparse import ArgumentParser
    parser = ArgumentParser()
    pipe_template = PipelineParams(parser)
    args = parser.parse_args([])
    return pipe_template.extract(args)


def _get_dataset_and_proj(config: dict, dataset_path: Path):
    """Load dataset and projection matrix; callers create one camera at a time to avoid OOM."""
    ds_args = Namespace()
    dataset = load_dataset(ds_args, str(dataset_path), config)
    proj = getProjectionMatrix2(
        znear=0.01, zfar=100.0,
        fx=dataset.fx, fy=dataset.fy, cx=dataset.cx, cy=dataset.cy,
        W=dataset.width, H=dataset.height,
    ).transpose(0, 1)
    return dataset, proj, dataset.width, dataset.height


def _load_keyframes(state_dir: Path):
    """Load keyframe list from keyframes_state.pt. Returns list of dicts with uid, R, T."""
    kf_path = Path(state_dir) / "keyframes_state.pt"
    if not kf_path.is_file():
        return []
    payload = torch.load(kf_path, map_location="cpu")
    return payload.get("keyframes", [])


def _render_mesh_splatting(cam, triangles, pipe, background, sigma_override=None, opacity_override=None):
    if sigma_override is not None:
        triangles.set_sigma(sigma_override)
    if opacity_override is not None:
        w_prev = triangles.vertex_weight.data.clone()
        # Use model's inverse so we respect opacity_floor
        lo = triangles.opacity_floor + 1e-6
        hi = 1.0 - 1e-6
        target = max(lo, min(hi, opacity_override))
        t = torch.tensor(target, device=triangles.vertex_weight.device, dtype=triangles.vertex_weight.dtype)
        logit = triangles.inverse_opacity_activation(t)
        triangles.vertex_weight.data.fill_(logit.item())
    with torch.no_grad():
        pkg = triangle_render(cam, triangles, pipe, background, optimize_pose_only=False)
    if sigma_override is not None:
        triangles.set_sigma(0.001)
    if opacity_override is not None:
        triangles.vertex_weight.data.copy_(w_prev)
    return pkg


def _save_render_outputs(pkg, out_dir: Path, prefix: str):
    out_dir = Path(out_dir)
    rgb = (pkg["render"].clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    depth = pkg["expected_depth"][0].detach().cpu().numpy()
    (out_dir / "rgb").mkdir(parents=True, exist_ok=True)
    (out_dir / "depth").mkdir(parents=True, exist_ok=True)
    save_img_u8(rgb, str(out_dir / "rgb" / f"{prefix}.png"))
    save_img_depth_png(depth, str(out_dir / "depth" / f"{prefix}.png"))


def _render_o3d_mesh_from_views(
    mesh_path: Path,
    view_matrices: list,
    w: int,
    h: int,
    fov_deg: float,
    out_dir: Path,
):
    """Render Open3D mesh from each world-to-camera 4x4 view matrix; save RGB and depth."""
    if not mesh_path.is_file():
        print(f"[skip] Open3D render: {mesh_path} not found")
        return
    if not view_matrices:
        return
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not mesh.has_triangles():
        print(f"[skip] Open3D render: {mesh_path} has no triangles")
        return
    mesh.compute_vertex_normals()
    material = rendering.MaterialRecord()
    material.shader = "defaultLit"
    render = rendering.OffscreenRenderer(w, h)
    render.scene.add_geometry("mesh", mesh, material)
    render.scene.set_background([0, 0, 0, 1])
    (out_dir / "rgb").mkdir(parents=True, exist_ok=True)
    (out_dir / "depth").mkdir(parents=True, exist_ok=True)
    n_views = len(view_matrices)
    for i, W2C in enumerate(view_matrices):
        if (i + 1) % 50 == 0 or i == 0 or i == n_views - 1:
            print(f"      Open3D view {i + 1}/{n_views}", flush=True)
        W2C = np.asarray(W2C, dtype=np.float64)
        C2W = np.linalg.inv(W2C)
        eye = C2W[:3, 3]
        center = eye + C2W[:3, :3] @ np.array([0, 0, 1.0])
        up = -C2W[:3, 1]
        render.setup_camera(fov_deg, center, eye, up, 0.01, 100.0)
        rgb = render.render_to_image()
        depth_img = render.render_to_depth_image(z_in_view_space=True)
        prefix = f"{i:05d}"
        o3d.io.write_image(str(out_dir / "rgb" / f"{prefix}.png"), rgb, 9)
        d = np.asarray(depth_img)
        if d.dtype != np.float32:
            d = d.astype(np.float32)
        save_img_depth_png(d, str(out_dir / "depth" / f"{prefix}.png"))
    render.destroy()


def run_one(
    state_path: Path,
    run_dir: Path,
    out_dir: Path,
    dataset_base: Path | None,
    scene_short: str,
):
    state_path = Path(state_path)
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    config_path = run_dir / "config.yml"
    if not config_path.is_file():
        config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        print(f"[skip] No config at {run_dir}")
        return
    config = load_config(str(config_path))
    dataset_path = _get_dataset_path(config, run_dir, dataset_base, scene_short)
    if not dataset_path.exists():
        print(f"[skip] Dataset not found: {dataset_path}")
        return
    state_dir = state_path if state_path.is_dir() else state_path.parent
    if not (state_dir / "point_cloud_state_dict.pt").is_file():
        print(f"[skip] No state at {state_dir}")
        return
    pipe = _load_pipe(config)
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    triangles = TriangleModel(sh_degree=0, config=config)
    triangles.load_parameters(str(state_dir), device="cuda")
    triangles.set_sigma(0.001)

    # ---- GT poses: one camera at a time to avoid OOM ----
    dataset, proj, w, h = _get_dataset_and_proj(config, dataset_path)
    n_gt = len(dataset)
    if n_gt == 0:
        print(f"[skip] No GT frames for {scene_short}")
        return
    fov_deg = math.degrees(getattr(dataset, "fovy", math.radians(60)))
    out_renders = out_dir / "renders"
    (out_renders / "rgb_opaque").mkdir(parents=True, exist_ok=True)
    (out_renders / "depth_opaque").mkdir(parents=True, exist_ok=True)
    view_matrices_gt = []
    metrics_gt_rgb = []
    metrics_gt_opaque = []
    lpips_fn = _get_lpips() if _get_lpips else None
    print(f"  GT mesh-splatting: {n_gt} frames (rgb=blend 0.5, rgb_opaque=0.99) ...", flush=True)
    t0 = time.perf_counter()
    t_rgb, t_opaque = 0.0, 0.0
    for i in range(n_gt):
        if i == 0 or (i + 1) % 10 == 0 or i == n_gt - 1:
            print(f"    frame {i + 1}/{n_gt}", flush=True)
        cam = Camera.init_from_dataset(dataset, i, proj)
        cam.update_RT(cam.gt_T[:3, :3], cam.gt_T[:3, 3])
        prefix = f"{i:05d}"
        t_a = time.perf_counter()
        pkg = _render_mesh_splatting(
            cam, triangles, pipe, background,
            opacity_override=0.5,
        )
        t_rgb += time.perf_counter() - t_a
        _save_render_outputs(pkg, out_renders, prefix)
        t_b = time.perf_counter()
        pkg_opaque = _render_mesh_splatting(
            cam, triangles, pipe, background,
            sigma_override=0.0001, opacity_override=0.99,
        )
        save_img_u8(
            (pkg_opaque["render"].clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy(),
            str(out_renders / "rgb_opaque" / f"{prefix}.png"),
        )
        save_img_depth_png(
            pkg_opaque["expected_depth"][0].detach().cpu().numpy(),
            str(out_renders / "depth_opaque" / f"{prefix}.png"),
        )
        t_opaque += time.perf_counter() - t_b
        view_matrices_gt.append(cam.world_view_transform.T.detach().cpu().numpy())
        if cam.original_image is not None:
            gt_img = cam.original_image.detach().float().clamp(0, 1)
            if gt_img.dim() == 4:
                gt_img = gt_img[0]
            m_rgb = _compute_metrics(pkg["render"], gt_img, lpips_fn)
            m_opaque = _compute_metrics(pkg_opaque["render"], gt_img, lpips_fn)
            if m_rgb:
                metrics_gt_rgb.append(m_rgb)
            if m_opaque:
                metrics_gt_opaque.append(m_opaque)
        cam.clean()
        del cam
        del pkg
        del pkg_opaque
        if (i + 1) % 50 == 0:
            torch.cuda.empty_cache()
    elapsed = time.perf_counter() - t0
    mesh_ply = out_dir / "mesh_delaunay.ply"
    print(f"  GT Open3D mesh: {n_gt} views ...", flush=True)
    t_o3d_gt = time.perf_counter()
    _render_o3d_mesh_from_views(mesh_ply, view_matrices_gt, w, h, fov_deg, out_dir / "open3d")
    t_o3d_gt = time.perf_counter() - t_o3d_gt
    del view_matrices_gt
    torch.cuda.empty_cache()
    metrics_gt_o3d = _compute_metrics_o3d(
        dataset, list(range(n_gt)), out_dir / "open3d" / "rgb", lpips_fn
    )
    print(f"  GT mesh-splatting done in {elapsed:.1f}s ({elapsed / max(1, n_gt):.2f}s/frame)", flush=True)
    _write_metrics(
        out_dir,
        "gt",
        n_gt,
        metrics_gt_rgb,
        metrics_gt_opaque,
        metrics_gt_o3d,
        elapsed,
        t_rgb,
        t_opaque,
        t_o3d_gt,
    )
    print(f"[done] GT: {out_dir} ({n_gt} views)")

    # ---- Keyframe poses: align to GT then render one camera at a time ----
    keyframes = _load_keyframes(state_dir)
    if not keyframes:
        print(f"[skip] No keyframe cameras for {scene_short}")
        return
    T_align, valid = _compute_pose_alignment(keyframes, dataset, n_gt)
    if not valid:
        print(f"[skip] No valid keyframe poses for alignment")
        return
    print(f"  Keyframe pose alignment: {len(valid)} poses aligned to GT (Procrustes)", flush=True)
    out_kf = out_dir / "keyframes"
    (out_kf / "renders" / "rgb_opaque").mkdir(parents=True, exist_ok=True)
    (out_kf / "renders" / "depth_opaque").mkdir(parents=True, exist_ok=True)
    view_matrices_kf = []
    metrics_kf_rgb = []
    metrics_kf_opaque = []
    device = next(triangles.parameters()).device
    n_kf = len(valid)
    t_rgb_kf, t_opaque_kf = 0.0, 0.0
    print(f"  Keyframe mesh-splatting: {n_kf} frames (rgb=blend 0.5, rgb_opaque=0.99) ...", flush=True)
    t0_kf = time.perf_counter()
    for i, (uid, c2w_kf, _) in enumerate(valid):
        if (i + 1) % 10 == 0 or i == 0 or i == n_kf - 1:
            print(f"    kf {i + 1}/{n_kf}", flush=True)
        R_aligned, T_aligned = _apply_alignment(c2w_kf, T_align)
        R_t = torch.from_numpy(R_aligned).float().to(device)
        T_t = torch.from_numpy(T_aligned).float().to(device)
        cam = Camera.init_from_dataset(dataset, uid, proj)
        cam.update_RT(R_t, T_t)
        prefix = f"{i:05d}"
        t_a = time.perf_counter()
        pkg = _render_mesh_splatting(
            cam, triangles, pipe, background,
            opacity_override=0.5,
        )
        t_rgb_kf += time.perf_counter() - t_a
        _save_render_outputs(pkg, out_kf / "renders", prefix)
        t_b = time.perf_counter()
        pkg_opaque = _render_mesh_splatting(
            cam, triangles, pipe, background,
            sigma_override=0.0001, opacity_override=0.99,
        )
        t_opaque_kf += time.perf_counter() - t_b
        save_img_u8(
            (pkg_opaque["render"].clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy(),
            str(out_kf / "renders" / "rgb_opaque" / f"{prefix}.png"),
        )
        save_img_depth_png(
            pkg_opaque["expected_depth"][0].detach().cpu().numpy(),
            str(out_kf / "renders" / "depth_opaque" / f"{prefix}.png"),
        )
        view_matrices_kf.append(cam.world_view_transform.T.detach().cpu().numpy())
        if cam.original_image is not None:
            gt_img = cam.original_image.detach().float().clamp(0, 1)
            if gt_img.dim() == 4:
                gt_img = gt_img[0]
            m_rgb = _compute_metrics(pkg["render"], gt_img, lpips_fn)
            m_opaque = _compute_metrics(pkg_opaque["render"], gt_img, lpips_fn)
            if m_rgb:
                metrics_kf_rgb.append(m_rgb)
            if m_opaque:
                metrics_kf_opaque.append(m_opaque)
        cam.clean()
        del cam
        del pkg
        del pkg_opaque
        if (i + 1) % 50 == 0:
            torch.cuda.empty_cache()
    elapsed_kf = time.perf_counter() - t0_kf
    print(f"  Keyframe Open3D mesh: {n_kf} views ...", flush=True)
    t_o3d_kf = time.perf_counter()
    _render_o3d_mesh_from_views(mesh_ply, view_matrices_kf, w, h, fov_deg, out_kf / "open3d")
    t_o3d_kf = time.perf_counter() - t_o3d_kf
    kf_uids = [uid for (uid, _, _) in valid]
    metrics_kf_o3d = _compute_metrics_o3d(
        dataset, kf_uids, out_kf / "open3d" / "rgb", lpips_fn
    )
    print(f"  Keyframe mesh-splatting done in {elapsed_kf:.1f}s", flush=True)
    _write_metrics(
        out_dir,
        "keyframes",
        n_kf,
        metrics_kf_rgb,
        metrics_kf_opaque,
        metrics_kf_o3d,
        elapsed_kf,
        t_rgb_kf,
        t_opaque_kf,
        t_o3d_kf,
    )
    print(f"[done] Keyframes: {out_kf} ({n_kf} views)")


REPLICA_FOLDERS = [
    "replica_office0", "replica_office1", "replica_office2", "replica_office3", "replica_office4",
    "replica_room0", "replica_room1", "replica_room2",
]


def _find_latest_iteration(mesh_dir: Path) -> Path | None:
    import re
    if not mesh_dir.is_dir():
        return None
    pattern = re.compile(r"iteration_(\d+)$")
    best = None
    best_num = -1
    for p in mesh_dir.iterdir():
        if not p.is_dir():
            continue
        m = pattern.match(p.name)
        if m:
            n = int(m.group(1))
            if n > best_num:
                best_num = n
                best = p
    return best


def _find_latest_run(results_base: Path, folder: str) -> Path | None:
    run_dir = results_base / folder
    if not run_dir.is_dir():
        return None
    subdirs = [p for p in run_dir.iterdir() if p.is_dir()]
    if not subdirs:
        return None
    return max(subdirs, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description="Render Replica views (mesh-splat + Open3D mesh)")
    parser.add_argument("--state_path", type=str, help="Path to mesh state dir (e.g. .../mesh/iteration_2000)")
    parser.add_argument("--out_dir", type=str, help="Output base (e.g. results/replica_meshes/office0)")
    parser.add_argument("--dataset_base", type=str, default=None, help="If config has no dataset_path, use dataset_base/SCENE")
    parser.add_argument("--run_dir", type=str, default=None, help="Run dir containing config.yml (default: state_path/../..)")
    parser.add_argument("--scene_short", type=str, default=None, help="Scene name for dataset_base (e.g. office0). Default: from out_dir basename.")
    parser.add_argument("--all", action="store_true", help="Run for all Replica scenes: discover latest run/iteration under --results_dir, output to --replica_meshes_dir/NAME")
    parser.add_argument("--results_dir", type=str, default="results", help="Base results dir for --all (default: results)")
    parser.add_argument("--replica_meshes_dir", type=str, default="results/replica_meshes", help="Base output for --all (default: results/replica_meshes)")
    args = parser.parse_args()

    if args.all:
        results_base = (ROOT / args.results_dir).resolve()
        out_base = (ROOT / args.replica_meshes_dir).resolve()
        dataset_base = Path(args.dataset_base).resolve() if args.dataset_base else None
        for folder in REPLICA_FOLDERS:
            run_dir = _find_latest_run(results_base, folder)
            if run_dir is None:
                print(f"[skip] {folder}: no run dir")
                continue
            iter_dir = _find_latest_iteration(run_dir / "mesh")
            if iter_dir is None:
                print(f"[skip] {folder}: no iteration_* in mesh/")
                continue
            short = folder.replace("replica_", "")
            out_dir = out_base / short
            if not (out_dir / "mesh_delaunay.ply").is_file():
                print(f"[skip] {short}: no mesh_delaunay.ply in {out_dir}")
                continue
            print(f"[run] {short} <- {iter_dir}")
            run_one(iter_dir, run_dir, out_dir, dataset_base, short)
        return

    if not args.state_path or not args.out_dir:
        parser.error("--state_path and --out_dir required (or use --all)")
    state_path = Path(args.state_path).resolve()
    out_dir = Path(args.out_dir).resolve()
    run_dir = Path(args.run_dir).resolve() if args.run_dir else (state_path.parent.parent)
    scene_short = args.scene_short or out_dir.name
    dataset_base = Path(args.dataset_base).resolve() if args.dataset_base else None
    run_one(state_path, run_dir, out_dir, dataset_base, scene_short)


if __name__ == "__main__":
    main()
