#!/usr/bin/env python3
"""
Offline mesh finetuning/export pipeline from a saved point_cloud_state_dict.pt.

Stages:
1) Geometry+appearance finetune with sigma/opacity annealing and ORIGINAL-style densification.
2) Restricted Delaunay + frozen-point appearance polish (order configurable).
3) Save updated state_dict and export colored PLY.
"""
import os
import sys
from pathlib import Path
from argparse import ArgumentParser, Namespace
import random

import numpy as np
import torch
import torch.nn.functional as F
import rerun as rr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mesh_splatting.scene.triangle_model import TriangleModel
from mesh_splatting.triangle_renderer import render
from mesh_splatting.utils.graphics_utils import getProjectionMatrix2
from mesh_splatting.arguments import OptimizationParams, PipelineParams
from meshslam.create_ply import export_ply
from utils.cameras import Camera
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.loss_utils import l1_loss, ssim
from utils.gui_utils import compute_mesh_colors, rerun_show_cam


def make_opt_from_config(config: dict):
    parser = ArgumentParser()
    opt_template = OptimizationParams(parser)
    pipe_template = PipelineParams(parser)
    args = parser.parse_args([])
    opt = opt_template.extract(args)
    pipe = pipe_template.extract(args)
    for key, value in config.get("opt_params", {}).items():
        if hasattr(opt, key):
            setattr(opt, key, value)
    return opt, pipe


def build_cameras(config: dict, dataset_path: str, num_frames: int | None, frame_stride: int):
    ds_args = Namespace()
    dataset = load_dataset(ds_args, dataset_path, config)
    proj = getProjectionMatrix2(
        znear=0.01,
        zfar=100.0,
        fx=dataset.fx,
        fy=dataset.fy,
        cx=dataset.cx,
        cy=dataset.cy,
        W=dataset.width,
        H=dataset.height,
    ).transpose(0, 1)
    n = len(dataset.poses)
    if num_frames is not None:
        n = min(n, num_frames)
    cams = [Camera.init_from_dataset(dataset, i, proj) for i in range(n)]
    cams = cams[::max(1, frame_stride)]
    return dataset, cams


def load_slam_keyframes(state_dir: str):
    kf_path = os.path.join(state_dir, "keyframes_state.pt")
    if not os.path.isfile(kf_path):
        return None
    payload = torch.load(kf_path, map_location="cpu")
    keyframes = payload.get("keyframes", [])
    if not keyframes:
        return None
    return keyframes


def select_slam_keyframe_cameras(cams: list, keyframes: list):
    cams_by_uid = {int(cam.uid): cam for cam in cams}
    selected = []
    missing = 0
    for kf in keyframes:
        uid = int(kf["uid"])
        cam = cams_by_uid.get(uid)
        if cam is None:
            missing += 1
            continue
        cam.update_RT(kf["R"].to(device=cam.device), kf["T"].to(device=cam.device))
        selected.append(cam)
    if missing > 0:
        print(f"Warning: {missing} saved keyframes were outside selected camera range/stride.")
    return selected


def anneal_sigma(triangles: TriangleModel, opt, iteration: int):
    initial_sigma = float(opt.set_sigma)
    final_sigma = 0.001
    sigma_start = int(opt.sigma_start)
    sigma_until = max(int(opt.sigma_until), sigma_start + 1)
    if iteration < sigma_start:
        current_sigma = initial_sigma
    else:
        progress = min(1.0, (iteration - sigma_start) / (sigma_until - sigma_start))
        current_sigma = initial_sigma - (initial_sigma - final_sigma) * progress
    triangles.set_sigma(current_sigma)


def anneal_opacity_floor(triangles: TriangleModel, opt, iteration: int):
    init_opacity = 0.1
    final_opacity = 0.9999
    start_iter = int(opt.start_opacity_floor)
    end_iter = max(int(opt.final_opacity_iter), start_iter + 1)
    if iteration <= start_iter:
        return
    a = min(1.0, max(0.0, (iteration - start_iter) / (end_iter - start_iter)))
    current_opacity = init_opacity + (final_opacity - init_opacity) * a
    current_opacity = min(current_opacity, final_opacity)
    triangles.update_min_weight(current_opacity)


def compute_mapping_like_loss(image, gt_image, triangles: TriangleModel, opt):
    pixel_loss = l1_loss(image, gt_image)
    ssim_value = ssim(image, gt_image)
    loss_image = (1.0 - opt.lambda_dssim) * pixel_loss + opt.lambda_dssim * (1.0 - ssim_value)
    lambda_weight = float(getattr(opt, "lambda_weight", 0.0))
    loss_opacity = torch.abs(triangles.get_vertex_weight).mean() * lambda_weight if lambda_weight > 0 else 0.0
    loss = loss_image + loss_opacity
    return loss


def update_importance_buffers(triangles: TriangleModel, render_pkg: dict):
    image_size = render_pkg["scaling"].detach()
    mask = image_size > triangles.image_size
    triangles.image_size[mask] = image_size[mask]
    importance_score = render_pkg["max_blending"].detach()
    mask = importance_score > triangles.importance_score
    triangles.importance_score[mask] = importance_score[mask]
    pixel_count = render_pkg["triangle_was_rendered"].detach()
    mask = pixel_count > triangles.pixel_count
    triangles.pixel_count[mask] = pixel_count[mask]


def log_mesh_rerun_simple(triangles: TriangleModel, viewpoint_cam, static: bool = False):
    """
    Rerun mesh logging without the "new mesh" highlight path.
    """
    verts = triangles.vertices.detach().cpu().numpy()
    tris = triangles._triangle_indices.detach().cpu().numpy()
    if verts.size == 0 or tris.size == 0:
        return
    colors_precomp = compute_mesh_colors(triangles, viewpoint_cam)
    rr.log(
        "/mesh_ft",
        rr.Mesh3D(
            vertex_positions=triangles.get_vertices.detach().cpu().numpy(),
            triangle_indices=triangles.get_triangle_indices.detach().cpu().numpy(),
            vertex_colors=(colors_precomp * 255).to(torch.uint8).detach().cpu().numpy(),
        ),
        static=static,
    )


def log_finetune_camera_set_rerun(cams: list):
    if not cams:
        return
    centers = []
    uids = []
    for cam in cams:
        centers.append(cam.camera_center.detach().cpu().numpy())
        uids.append(int(cam.uid))
    centers = np.asarray(centers, dtype=np.float32)
    colors = np.tile(np.array([[255, 200, 0]], dtype=np.uint8), (centers.shape[0], 1))
    rr.log(
        "/finetune/cameras/points",
        rr.Points3D(positions=centers, colors=colors, radii=0.015),
        static=True,
    )
    # Camera trajectory ordering by uid for readability.
    order = np.argsort(np.asarray(uids))
    rr.log(
        "/finetune/cameras/trajectory",
        rr.LineStrips3D(
            strips=[centers[order]],
            colors=[255, 200, 0],
            radii=0.003,
        ),
        static=True,
    )


def log_current_camera_rerun(cam, render_pkg: dict):
    image = render_pkg["render"]
    rgb = (
        (torch.clamp(image, min=0, max=1.0) * 255)
        .byte()
        .permute(1, 2, 0)
        .contiguous()
        .cpu()
        .numpy()
    )
    rerun_show_cam("/finetune/current_cam", rgb, cam, static=False, use_rerun=True)


def densify_original_compat(triangles: TriangleModel, opt, iteration: int):
    """
    Uses original densify logic from add_new_gs_original with a compatibility fix
    for _update_params_fast signature in this repo.
    """
    current_num_points = triangles.vertices.shape[0]
    target_num = min(opt.max_points, int(triangles.add_percentage * current_num_points))
    num_gs = max(0, target_num - current_num_points)
    if num_gs <= 0:
        return 0

    probs = triangles.importance_score.squeeze()
    areas = triangles.triangle_areas().squeeze()
    probs = torch.where(areas < triangles.size_probs_zero, torch.zeros_like(probs), probs)
    probs = torch.where(triangles.image_size < triangles.size_probs_zero_image_space, torch.zeros_like(probs), probs)

    if probs.numel() == 0 or float(probs.sum()) <= 0.0:
        return 0

    rand_idx = triangles._sample_alives(probs=probs, num=min(num_gs, probs.numel()))
    k = min(int(opt.splitt_large_triangles), areas.numel())
    _, top_idx = torch.topk(areas, k, largest=True, sorted=False)
    add_idx = torch.unique(torch.cat([rand_idx, top_idx.to(rand_idx.device)]), sorted=False)
    if add_idx.numel() == 0:
        return 0

    # Compatibility: current _update_params_fast takes only selected_indices.
    new_vertices, new_vertex_weight, new_features_dc, new_features_rest, new_triangles = triangles._update_params_fast(add_idx)
    triangles.densification_postfix(new_vertices, new_vertex_weight, new_features_dc, new_features_rest, new_triangles)

    mask = torch.ones(triangles._triangle_indices.shape[0], dtype=torch.bool, device=triangles._triangle_indices.device)
    mask[add_idx] = False
    # triangles.prune_triangles(mask)
    return int(add_idx.numel())


def build_mesh_optimizer(triangles: TriangleModel, opt):
    param_groups = [
        {"params": [triangles._features_dc], "lr": float(opt.feature_lr), "name": "f_dc"},
        {"params": [triangles.vertex_weight], "lr": float(opt.weight_lr), "name": "vertex_weight"},
        {"params": [triangles.vertices], "lr": float(opt.lr_triangles_points_init), "name": "vertices"},
    ]
    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


def set_point_requires_grad(triangles: TriangleModel, requires_grad: bool):
    triangles.vertices.requires_grad_(requires_grad)


def run_train_loop(
    triangles: TriangleModel,
    optimizer: torch.optim.Optimizer,
    cams: list,
    pipe,
    background: torch.Tensor,
    opt,
    iters: int,
    densify: bool,
    densify_interval: int,
    start_iter: int = 1,
    use_rerun: bool = False,
    rerun_log_interval: int = 100,
):
    for i in range(start_iter, start_iter + iters):
        triangles.update_learning_rate(i)
        anneal_sigma(triangles, opt, i)
        anneal_opacity_floor(triangles, opt, i)

        cam = cams[random.randint(0, len(cams) - 1)]
        render_pkg = render(cam, triangles, pipe, background)
        update_importance_buffers(triangles, render_pkg)

        gt = cam.original_image.cuda()
        image = render_pkg["render"]
        loss = compute_mapping_like_loss(image, gt, triangles, opt)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if densify and i >= int(opt.densify_from_iter) and i <= int(opt.densify_until_iter) and i % max(1, densify_interval) == 0:
            n_added = densify_original_compat(triangles, opt, i)
            if n_added > 0:
                print(f"[iter {i}] densified triangles from {n_added} selected seeds")

        if i % 100 == 0:
            print(f"[iter {i}] loss={loss.item():.6f} tris={triangles._triangle_indices.shape[0]} verts={triangles.vertices.shape[0]}")
        if use_rerun and (i % max(1, rerun_log_interval) == 0):
            log_current_camera_rerun(cam, render_pkg)
            log_mesh_rerun_simple(triangles, cam, static=False)

            normals = render_pkg["rend_normal"]
            surf_normals = render_pkg["surf_normal"]

            normals_np = np.transpose(normals.detach().cpu().numpy(), (1, 2, 0))  # (H, W, 3)
            surf_normals_np = np.transpose(surf_normals.detach().cpu().numpy(), (1, 2, 0))

            # Normalize to 0-255 (normals are typically in [-1, 1] or [0, 1])
            normals_np = ((normals_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            surf_normals_np = ((surf_normals_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)

            rr.log("/normals_tracked", rr.Image(normals_np), static=False)
            rr.log("/surf_normals_tracked", rr.Image(surf_normals_np), static=False)


def export_state_and_ply(
    triangles: TriangleModel,
    out_dir: str,
    ply_name: str,
    zup: bool,
    clamp_features_dc: float | None = 4.0,
    smooth: str | None = None,
    smooth_iters: int = 2,
):
    """Export state and PLY. clamp_features_dc: clamp SH DC to [-x,x]. smooth: mean/median/huber."""
    from utils.mesh_color_utils import smooth_vertex_colors

    os.makedirs(out_dir, exist_ok=True)
    triangles.save_parameters(out_dir)
    SH_C0 = 0.28209479177387814
    dc = triangles._features_dc
    if clamp_features_dc is not None and clamp_features_dc > 0:
        dc = torch.clamp(dc, -clamp_features_dc, clamp_features_dc)
    colors = torch.clamp(SH_C0 * dc + 0.5, 0.0, 1.0)
    colors = colors.detach().cpu().numpy().squeeze()
    if colors.ndim == 3:
        colors = colors.squeeze(0)
    if smooth and smooth_iters > 0:
        tris = triangles._triangle_indices.detach().cpu().numpy()
        colors = smooth_vertex_colors(colors, tris, method=smooth, iterations=smooth_iters)
    colors_u8 = (colors * 255.0).round().astype(np.uint8)
    verts_np = triangles.vertices.detach().cpu().numpy()
    faces_np = triangles._triangle_indices.detach().cpu().numpy()
    ply_path = os.path.join(out_dir, ply_name)
    export_ply(ply_path, verts_np, faces_np, colors_u8, zup_to_yup=zup)
    print(f"Saved state_dict to: {os.path.join(out_dir, 'point_cloud_state_dict.pt')}")
    print(f"Saved PLY to: {ply_path}")


def final_importance_prune(triangles: TriangleModel, cams: list, pipe, background: torch.Tensor, threshold: float):
    """
    Recompute per-triangle importance over all cameras and prune low-importance triangles.
    """
    if threshold is None:
        return 0
    triangles.importance_score = torch.zeros(
        (triangles._triangle_indices.shape[0]), dtype=torch.float, device="cuda"
    )
    for cam in cams:
        render_pkg = render(cam, triangles, pipe, background)
        importance_score = render_pkg["max_blending"].detach()
        mask = importance_score > triangles.importance_score
        triangles.importance_score[mask] = importance_score[mask]

    keep_mask = (triangles.importance_score > float(threshold)).squeeze()
    n_before = int(triangles._triangle_indices.shape[0])
    # triangles.prune_triangles(keep_mask)

    device = triangles.vertices.device
    used_vertex_mask = torch.zeros(triangles.vertices.shape[0], dtype=torch.bool, device=device)
    if triangles._triangle_indices.numel() > 0:
        used_vertex_mask[triangles._triangle_indices.flatten()] = True
    triangles._prune_vertices(used_vertex_mask)

    n_after = int(triangles._triangle_indices.shape[0])
    print(f"Final importance prune @ threshold={threshold:.4f}: {n_before} -> {n_after} triangles")
    return n_before - n_after


def main():
    parser = ArgumentParser("Offline mesh finetune + export")
    parser.add_argument("--config", required=True, help="YAML config used for SLAM/camera setup")
    parser.add_argument("--dataset_path", required=True, help="Dataset path used to build training cameras")
    parser.add_argument("--state_path", required=True, help="Path to point_cloud_state_dict.pt OR its directory")
    parser.add_argument("--out_dir", required=True, help="Output directory for updated state and ply")
    parser.add_argument("--num_frames", type=int, default=None, help="Optional cap on number of cameras")
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--use_slam_keyframes", action="store_true", help="Use saved SLAM keyframe poses/views from keyframes_state.pt")
    parser.add_argument("--densify_iters", type=int, default=2000, help="Stage-1 iterations with original densify")
    parser.add_argument("--densify_interval", type=int, default=200)
    parser.add_argument("--stage1_vertex_lr", type=float, default=None, help="Override stage-1 vertex LR (triangles points)")
    parser.add_argument("--stage1_vertex_lr_scale", type=float, default=1.0, help="Multiply config vertex LR by this factor when stage1_vertex_lr is not set")
    parser.add_argument("--polish_iters", type=int, default=1500, help="Frozen-point color/opacity finetune iterations")
    parser.add_argument(
        "--rd_order",
        choices=["before", "after"],
        default="before",
        help="before: RD first then appearance-only finetune. after: pretrain (unfrozen points) then RD then appearance-only finetune.",
    )
    parser.add_argument("--post_rd_polish_iters", type=int, default=300, help="Extra frozen-point polish after RD when rd_order=after")
    parser.add_argument("--final_importance_threshold", type=float, default=None, help="Optional final prune threshold on max_blending importance (e.g. 0.2)")
    parser.add_argument("--use_rerun", action="store_true", help="Visualize mesh during finetuning in Rerun")
    parser.add_argument("--rerun_log_interval", type=int, default=100, help="Rerun mesh log interval in iterations")
    parser.add_argument("--sigma_until_override", type=int, default=None, help="Optional override for sigma anneal end iteration")
    parser.add_argument("--sigma_start_override", type=int, default=None, help="Optional override for sigma anneal start iteration")
    parser.add_argument("--zup", action="store_true", help="Convert Z-up to Y-up in PLY export")
    parser.add_argument("--clamp_features", type=float, default=4.0, help="Clamp SH DC to [-x,x] before PLY export (default: 4.0, 0=disable)")
    parser.add_argument("--smooth", choices=["mean", "median", "huber"], default=None, help="Smooth vertex colors: mean, median, or huber (trimmed mean)")
    parser.add_argument("--smooth_iters", type=int, default=2, help="Smoothing iterations (default: 2)")
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    config = load_config(args.config)
    opt, pipe = make_opt_from_config(config)
    # By default, anneal sigma across the actual offline run horizon.
    # If rd_order=before, we skip pretrain with unfrozen points.
    pretrain_iters = int(args.densify_iters) if args.rd_order == "after" else 0
    offline_total_iters = int(
        pretrain_iters
        + int(args.polish_iters)
        + (int(args.post_rd_polish_iters) if args.rd_order == "after" else 0)
    )
    if args.sigma_start_override is not None:
        opt.sigma_start = int(args.sigma_start_override)
    if args.sigma_until_override is not None:
        opt.sigma_until = int(args.sigma_until_override)
    else:
        opt.sigma_until = max(1, offline_total_iters)
    print(f"Sigma anneal schedule: start={opt.sigma_start}, until={opt.sigma_until}, target_final=0.0001")
    dataset, all_cams = build_cameras(config, args.dataset_path, args.num_frames, args.frame_stride)
    if len(all_cams) == 0:
        raise RuntimeError("No cameras available for finetuning.")

    print("num cameras: ", len(all_cams))
    state_dir = args.state_path
    if state_dir.endswith(".pt"):
        state_dir = str(Path(state_dir).parent)


    cams = all_cams
    if args.use_slam_keyframes:
        kf_state = load_slam_keyframes(state_dir)
        if kf_state is None:
            print("Warning: --use_slam_keyframes requested but keyframes_state.pt was not found. Falling back to dataset cameras.")
        else:
            cams = select_slam_keyframe_cameras(all_cams, kf_state)
            if len(cams) == 0:
                print("Warning: no saved keyframes matched current camera set. Falling back to dataset cameras.")
                cams = all_cams
            else:
                print(f"Using {len(cams)} SLAM keyframe cameras for offline finetuning.")

    triangles = TriangleModel(sh_degree=0, config=config)
    triangles.load_parameters(state_dir, device="cuda")
    if args.stage1_vertex_lr is not None:
        opt.lr_triangles_points_init = float(args.stage1_vertex_lr)
    else:
        opt.lr_triangles_points_init = float(opt.lr_triangles_points_init) * float(args.stage1_vertex_lr_scale)
    triangles.training_setup(opt, opt.feature_lr, opt.weight_lr, opt.lr_triangles_points_init)
    triangles.add_percentage = opt.add_percentage
    triangles.size_probs_zero = opt.size_probs_zero
    triangles.size_probs_zero_image_space = opt.size_probs_zero_image_space
    triangles.scaling = opt.upscaling_factor

    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    if args.use_rerun:
        rr.init("triangle_mesh_extract_finetune", spawn=True)
        log_finetune_camera_set_rerun(cams)

    def run_rd_and_freeze():
        print("Running restricted delaunay...")
        triangles.run_restricted_delaunay()
        print("Freezing point positions (vertices)")
        set_point_requires_grad(triangles, False)

    if args.rd_order == "before":
        if args.densify_iters > 0:
            print("Note: --densify_iters is ignored when rd_order=before (RD is run first, points stay frozen).")
        run_rd_and_freeze()
        print("Stage: frozen-point color/opacity polish (anneal sigma/opacity)")
        mesh_opt = build_mesh_optimizer(triangles, opt)
        run_train_loop(
            triangles=triangles,
            optimizer=mesh_opt,
            cams=cams,
            pipe=pipe,
            background=background,
            opt=opt,
            iters=args.densify_iters,
            densify=True, 
            densify_interval=args.densify_interval,
            start_iter=1,
            use_rerun=args.use_rerun,
            rerun_log_interval=args.rerun_log_interval,
        )
    else:
        # Pretrain first with unfrozen points and original-style densification.
        if args.densify_iters > 0:
            print("Stage 1: geometry+appearance pretrain with unfrozen points (ORIGINAL densify)")
            run_train_loop(
                triangles=triangles,
                optimizer=triangles.optimizer,
                cams=cams,
                pipe=pipe,
                background=background,
                opt=opt,
                iters=args.densify_iters,
                densify=True,
                densify_interval=args.densify_interval,
                start_iter=1,
                use_rerun=args.use_rerun,
                rerun_log_interval=args.rerun_log_interval,
            )

        run_rd_and_freeze()
        stage2_start = int(args.densify_iters) + 1
        print("Stage 2: frozen-point color/opacity polish (anneal sigma/opacity)")
        mesh_opt = build_mesh_optimizer(triangles, opt)
        run_train_loop(
            triangles=triangles,
            optimizer=mesh_opt,
            cams=cams,
            pipe=pipe,
            background=background,
            opt=opt,
            iters=args.polish_iters,
            densify=True,
            densify_interval=args.densify_interval,
            start_iter=stage2_start,
            use_rerun=args.use_rerun,
            rerun_log_interval=args.rerun_log_interval,
        )
        if args.post_rd_polish_iters > 0:
            print("Stage 2b: post-RD frozen-point polish")
            mesh_opt = build_mesh_optimizer(triangles, opt)
            run_train_loop(
                triangles=triangles,
                optimizer=mesh_opt,
                cams=cams,
                pipe=pipe,
                background=background,
                opt=opt,
                iters=args.post_rd_polish_iters,
                densify=True,
                densify_interval=args.densify_interval,
                start_iter=stage2_start + args.polish_iters,
                use_rerun=args.use_rerun,
                rerun_log_interval=args.rerun_log_interval,
            )

    # final_importance_prune(triangles, cams, pipe, background, args.final_importance_threshold)
    clamp_dc = args.clamp_features if args.clamp_features > 0 else None
    smooth = args.smooth
    smooth_iters = args.smooth_iters if smooth else 0
    export_state_and_ply(
        triangles, args.out_dir, "mesh_finetuned.ply", args.zup,
        clamp_features_dc=clamp_dc, smooth=smooth, smooth_iters=smooth_iters,
    )
    print("Done.")


if __name__ == "__main__":
    main()
