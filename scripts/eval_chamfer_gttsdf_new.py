#!/usr/bin/env python3
"""
Evaluate a reconstruction against a GT TSDF mesh built from dataset depths/poses.

GT mesh : TSDF-fused from GT RGB + depth + poses loaded via load_gt_data()
          (same approach as eval_tsdf.py, NOT the raw _mesh.ply file)
Est mesh: either a pre-saved PLY (--tsdf_ply) or TSDF-fused inline from depth_renders/

Metrics (identical to eval_tsdf.py):
    chamfer_cm, gt→est_mean_cm, est→gt_mean_cm,
    gt→est_median_cm, est→gt_median_cm,
    completeness / accuracy at 1, 2, 3, 5, 10 cm

Alignment: SE(3) Umeyama, no scale (same as eval_chamfer.py --gt_align)

Usage:
    # Pre-saved estimated PLY, SE(3)-align to GT:
    python eval_chamfer_gttsdf.py <render_dir> --tsdf_ply path/to/est.ply --gt_align

    # Inline TSDF fusion + alignment:
    python eval_chamfer_gttsdf.py <render_dir> --gt_align \\
        --voxel_length 0.04 --depth_trunc 5.0

    # Save both meshes for inspection:
    python eval_chamfer_gttsdf.py <render_dir> --gt_align \\
        --save_gt_ply gt_tsdf.ply --save_ply est_tsdf.ply
"""
import time
import argparse
import json
import os
import sys

import numpy as np
import open3d as o3d
import rerun as rr
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image as PILImage

from view_reconstruction import (
    load_camera_data,
    load_gt_data,
    compute_se3_alignment,
    apply_alignment_to_mesh,
    c2w_from_wvt,
)


# ─────────────────────────────────────────────────────────────────────────────
# Depth L1 evaluation
# ─────────────────────────────────────────────────────────────────────────────

def compute_depth_l1(render_dir, dataset, data, arr_indices):
    """Per-image mean absolute depth error, then averaged over images.

    For each frame: load est depth from render_dir, load GT depth from dataset,
    mask pixels where both are valid (> 0), compute mean |est - gt| per image,
    then return the mean and std of those per-image means.

    Returns dict with keys: depth_l1_mean_m, depth_l1_std_m, n_frames.
    """
    frame_ids = data["frame_ids"]
    per_image_l1 = []
    n_skip = 0

    for ai in tqdm(arr_indices, desc="  Depth L1"):
        fid = int(frame_ids[ai])
        depth_path = os.path.join(render_dir, f"{fid:05d}_depth.npy")
        if not os.path.exists(depth_path):
            n_skip += 1
            continue
        try:
            _, depth_gt, _ = dataset[fid]
            if depth_gt is None:
                n_skip += 1
                continue
            depth_est = np.load(depth_path).astype(np.float32)
            depth_gt  = np.asarray(depth_gt, dtype=np.float32)

            valid = (depth_gt > 0) & (depth_est > 0)
            if valid.sum() == 0:
                n_skip += 1
                continue

            l1_depth_err = np.abs(depth_est[valid] - depth_gt[valid]).mean()
            per_image_l1.append(l1_depth_err)
            del depth_gt, depth_est, valid
        except Exception as exc:
            n_skip += 1
            print(f"  [skip depth l1] frame {fid}: {exc}")

    if n_skip:
        print(f"  [depth l1] Skipped {n_skip} frames")
    if not per_image_l1:
        raise RuntimeError("No frames usable for depth L1 — check render_dir and dataset.")

    arr = np.array(per_image_l1, dtype=np.float64)
    print(np.sort(arr))
    return {
        "depth_l1_mean_m": float(arr.mean()),
        "depth_l1_std_m":  float(arr.std()),
        "depth_l1_n_frames": len(arr),
    }


# ─────────────────────────────────────────────────────────────────────────────
# GT TSDF construction
# ─────────────────────────────────────────────────────────────────────────────

def build_gt_tsdf(dataset, config, data, arr_indices, voxel_length, depth_trunc):
    """TSDF-fuse GT RGB + depth + poses for the frames in arr_indices.

    Accepts an already-loaded (dataset, config) to avoid a redundant load.
    Frames are streamed one-by-one; the TSDF volume is freed immediately
    after mesh extraction.

    Returns (mesh, gt_centers) where gt_centers is (N, 3) float32.
    """
    cal = config["Dataset"]["Calibration"]
    fx, fy = float(cal["fx"]), float(cal["fy"])
    cx, cy = float(cal["cx"]), float(cal["cy"])
    W,  H  = int(cal["width"]), int(cal["height"])
    intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=voxel_length * 5,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    frame_ids  = data["frame_ids"]
    gt_centers = []
    n_ok = n_skip = 0

    print(f"  [gt tsdf] Fusing {len(arr_indices)} frames  "
          f"(voxel={voxel_length:.4f} m, trunc={voxel_length * 5:.4f} m, "
          f"depth_trunc={depth_trunc} m)")

    for ai in tqdm(arr_indices, desc="  GT TSDF integrate"):
        fid = int(frame_ids[ai])
        try:
            img_t, depth_np, pose_t = dataset[fid]
            if depth_np is None:
                n_skip += 1
                continue

            color = (img_t.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            depth = depth_np.astype(np.float32)
            T_w2c = (pose_t.cpu().numpy()
                     if hasattr(pose_t, "cpu") else np.asarray(pose_t)).astype(np.float64)

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.ascontiguousarray(color)),
                o3d.geometry.Image(np.ascontiguousarray(depth)),
                depth_scale=1.0,
                depth_trunc=depth_trunc,
                convert_rgb_to_intensity=False,
            )
            volume.integrate(rgbd, intrinsic, T_w2c)
            gt_centers.append(np.linalg.inv(T_w2c)[:3, 3])
            n_ok += 1

            del img_t, depth_np, color, depth, rgbd

        except Exception as exc:
            n_skip += 1
            print(f"  [skip gt] frame {fid}: {exc}")

    if n_skip:
        print(f"  [gt tsdf] Skipped {n_skip} frames (no GT depth or error)")
    if n_ok == 0:
        raise RuntimeError("No GT frames integrated — check dataset_path and that the dataset has depth.")

    mesh = volume.extract_triangle_mesh()
    del volume  # free multi-GB TSDF volume before returning the mesh
    mesh.compute_vertex_normals()
    return mesh, np.array(gt_centers, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Est TSDF construction (streaming from saved renders — no GaussianExtractor)
# ─────────────────────────────────────────────────────────────────────────────

def build_est_tsdf(render_dir, data, arr_indices, voxel_length, depth_trunc):
    """TSDF-fuse saved rendered RGB + depth + poses for the frames in arr_indices.

    Streams frames one-by-one from disk to avoid loading the full sequence
    into RAM (which GaussianExtractor.load_renders() does).
    """
    frame_ids = data["frame_ids"]
    wvts      = data["world_view_transforms"]
    fxs, fys  = data["fx"], data["fy"]
    cxs, cys  = data["cx"], data["cy"]
    widths    = data["widths"]
    heights   = data["heights"]

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=voxel_length * 5,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    n_ok = n_skip = 0
    print(f"  [est tsdf] Fusing {len(arr_indices)} frames  "
          f"(voxel={voxel_length:.4f} m, trunc={voxel_length * 5:.4f} m, "
          f"depth_trunc={depth_trunc} m)")

    for ai in tqdm(arr_indices, desc="  Est TSDF integrate"):
        fid = int(frame_ids[ai])
        rgb_path   = os.path.join(render_dir, f"{fid:05d}_rgb.png")
        depth_path = os.path.join(render_dir, f"{fid:05d}_depth.npy")
        if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
            n_skip += 1
            continue

        try:
            color  = np.array(PILImage.open(rgb_path), dtype=np.uint8)
            depth  = np.load(depth_path).astype(np.float32)
            depth[depth <= 0] = 0  # Open3D treats 0 as no measurement
            T_w2c  = wvts[ai].T.astype(np.float64)  # cameras.npz stores world_view_transform; W2C = wvt.T

            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                int(widths[ai]), int(heights[ai]),
                float(fxs[ai]), float(fys[ai]),
                float(cxs[ai]), float(cys[ai]),
            )
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.ascontiguousarray(color)),
                o3d.geometry.Image(np.ascontiguousarray(depth)),
                depth_scale=1.0,
                depth_trunc=depth_trunc,
                convert_rgb_to_intensity=False,
            )
            volume.integrate(rgbd, intrinsic, T_w2c)
            n_ok += 1

            del color, depth, rgbd

        except Exception as exc:
            n_skip += 1
            print(f"  [skip est] frame {fid}: {exc}")

    if n_skip:
        print(f"  [est tsdf] Skipped {n_skip} frames")
    if n_ok == 0:
        raise RuntimeError("No est frames integrated — check render_dir.")

    mesh = volume.extract_triangle_mesh()
    del volume
    mesh.compute_vertex_normals()
    return mesh


# ─────────────────────────────────────────────────────────────────────────────
# ICP refinement
# ─────────────────────────────────────────────────────────────────────────────

def icp_refine(mesh_gt, mesh_est, n_samples, max_iter, threshold):
    """Point-to-plane ICP using normals from the mesh. Returns (mesh_est_refined, T_icp).

    Normals come directly from mesh sampling (no re-estimation needed since
    both meshes already have compute_vertex_normals() applied).
    """
    pcd_gt  = mesh_gt.sample_points_uniformly(n_samples)
    pcd_est = mesh_est.sample_points_uniformly(n_samples)

    result = o3d.pipelines.registration.registration_icp(
        pcd_est, pcd_gt,
        threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
    )
    del pcd_gt, pcd_est

    T = result.transformation
    print(f"  [icp] fitness={result.fitness:.4f}  inlier_rmse={result.inlier_rmse*100:.3f} cm")
    print(f"  [icp] transform:\n{T}")

    mesh_est = mesh_est.transform(T)
    mesh_est.compute_vertex_normals()
    return mesh_est, T


# ─────────────────────────────────────────────────────────────────────────────
# Chamfer distance
# ─────────────────────────────────────────────────────────────────────────────

def chamfer_distance(mesh_a, mesh_b, n_samples=200_000):
    """Symmetric Chamfer distance via O3D point sampling."""
    pcd_a = mesh_a.sample_points_uniformly(n_samples)
    pcd_b = mesh_b.sample_points_uniformly(n_samples)
    d_ab  = np.asarray(pcd_a.compute_point_cloud_distance(pcd_b))
    d_ba  = np.asarray(pcd_b.compute_point_cloud_distance(pcd_a))
    del pcd_a, pcd_b

    # calculate rmse too
    d_ab_rmse = np.sqrt((d_ab ** 2).mean())
    d_ba_rmse = np.sqrt((d_ba ** 2).mean())

    

    metrics = {
        "chamfer_cm":       (d_ab.mean() + d_ba.mean()) / 2 * 100,
        "gt→est_mean_cm":   d_ab.mean() * 100,
        "est→gt_mean_cm":   d_ba.mean() * 100,
        "gt→est_median_cm": np.median(d_ab) * 100,
        "est→gt_median_cm": np.median(d_ba) * 100,


        "gt→est_rmse_cm":   d_ab_rmse * 100,
        "est→gt_rmse_cm":   d_ba_rmse * 100,
        'chamfer_rmse_cm': (d_ab_rmse + d_ba_rmse) / 2 * 100,
    }
    for t_cm, t_m in [(1, 0.01), (2, 0.02), (3, 0.03), (5, 0.05), (10, 0.10)]:
        metrics[f"completeness_<{t_cm}cm_%"] = (d_ab < t_m).mean() * 100
        metrics[f"accuracy_<{t_cm}cm_%"]     = (d_ba < t_m).mean() * 100
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("render_dir", help="Path to the depth_renders/ directory")

    ap.add_argument("--tsdf_ply", type=str, default=None, metavar="PATH",
                    help="Pre-saved estimated TSDF PLY; skips inline fusion")
    ap.add_argument("--dataset_path", type=str, default=None, metavar="PATH",
                    help="Override dataset path from config.yml")
    ap.add_argument("--gt_align", action="store_true",
                    help="SE(3) Umeyama-align estimated poses to GT (no scale)")
    ap.add_argument("--align_stride", type=int, default=1, metavar="N",
                    help="Frame stride for SE(3) alignment (default: 1)")
    ap.add_argument("--n_samples", type=int, default=1_000_000, metavar="N",
                    help="Points sampled per mesh for Chamfer (default: 1 000 000)")
    ap.add_argument("--no_rerun", action="store_true",
                    help="Skip Rerun visualisation")
    ap.add_argument("--rerun_max_tris", type=int, default=500_000, metavar="N",
                    help="Decimate meshes to this many triangles before Rerun upload (default: 500 000)")
    ap.add_argument("--max_error", type=float, default=0.05, metavar="F",
                    help="Max error in metres for error colormap (default: 0.05 = 5 cm)")
    ap.add_argument("--save_gt_ply", type=str, default=None, metavar="PATH",
                    help="Save GT TSDF mesh to this PLY path")

    g_icp = ap.add_argument_group("ICP refinement  (applied after gt_align, before Chamfer)")
    g_icp.add_argument("--icp", action="store_true",
                       help="Run point-to-plane ICP on GT vs est PCDs before Chamfer")
    g_icp.add_argument("--icp_samples", type=int, default=200_000, metavar="N",
                       help="Points sampled from each mesh for ICP (default: 200 000)")
    g_icp.add_argument("--icp_max_iter", type=int, default=50, metavar="N",
                       help="ICP iteration limit (default: 50)")
    g_icp.add_argument("--icp_threshold", type=float, default=0.1, metavar="F",
                       help="ICP correspondence threshold in metres (default: 0.1)")

    g = ap.add_argument_group("TSDF params  (shared by GT and est inline fusion)")
    g.add_argument("--voxel_length", type=float, default=0.004, metavar="F",
                   help="TSDF voxel size in metres (default: 0.004)")
    g.add_argument("--depth_trunc",  type=float, default=10.0,  metavar="F",
                   help="Max integration depth in metres (default: 10.0)")

    g2 = ap.add_argument_group("Est inline fusion  (ignored when --tsdf_ply is set)")
    g2.add_argument("--stride",     type=int,  default=1,    metavar="N",
                    help="Frame stride (default: 1)")
    g2.add_argument("--max_frames", type=int,  default=None, metavar="N",
                    help="Cap total frames (default: all)")
    g2.add_argument("--save_ply",   type=str,  default=None, metavar="PATH",
                    help="Save estimated TSDF mesh to this PLY path")

    g3 = ap.add_argument_group("Delaunay mesh comparison")
    g3.add_argument("--delaunay_ply", type=str, default=None, metavar="PATH",
                    help="Delaunay mesh PLY to evaluate alongside TSDF")
    g3.add_argument("--delaunay_visible_ply", type=str, default=None, metavar="PATH",
                    help="Visibility-filtered Delaunay mesh PLY")

    args = ap.parse_args()
    timing = {}

    data        = load_camera_data(args.render_dir)
    frame_ids   = data["frame_ids"]
    arr_indices = np.arange(len(frame_ids))[::args.stride]
    if args.max_frames is not None:
        arr_indices = arr_indices[:args.max_frames]

    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(args.render_dir)), "tsdf_chamfer_gt"
    )
    os.makedirs(out_dir, exist_ok=True)

    # Load dataset once — reused for GT TSDF and SE(3) alignment
    print("\n[Dataset]")
    dataset, config = load_gt_data(args.render_dir, args.dataset_path)

    # ── Depth L1 ──────────────────────────────────────────────────────────────
    print("\n[Depth L1]")
    depth_metrics = compute_depth_l1(args.render_dir, dataset, data, arr_indices)
    print(f"  mean  : {depth_metrics['depth_l1_mean_m']*100:.3f} cm  "
          f"(std {depth_metrics['depth_l1_std_m']*100:.3f} cm)  "
          f"over {depth_metrics['depth_l1_n_frames']} frames")

    # ── GT TSDF ───────────────────────────────────────────────────────────────
    print("\n[GT TSDF]")
    t0 = time.perf_counter()
    mesh_gt, gt_centers = build_gt_tsdf(
        dataset, config, data, arr_indices,
        voxel_length=args.voxel_length,
        depth_trunc=args.depth_trunc,
    )
    timing["gt_tsdf_s"] = time.perf_counter() - t0
    gt_n_verts = len(mesh_gt.vertices)
    gt_n_faces = len(mesh_gt.triangles)
    print(f"  {gt_n_verts:,} verts  {gt_n_faces:,} faces")
    gt_ply = args.save_gt_ply or os.path.join(out_dir, "gt_tsdf.ply")
    o3d.io.write_triangle_mesh(gt_ply, mesh_gt)
    print(f"  Saved → {gt_ply}")
    del mesh_gt  # free before est fusion — will reload from gt_ply for Chamfer

    # ── Est mesh ──────────────────────────────────────────────────────────────
    print("\n[Est mesh]")
    t0 = time.perf_counter()
    if args.tsdf_ply:
        print(f"  Loading pre-saved PLY: {args.tsdf_ply}")
        mesh_est = o3d.io.read_triangle_mesh(args.tsdf_ply)
        mesh_est.compute_vertex_normals()
    else:
        mesh_est = build_est_tsdf(
            args.render_dir, data, arr_indices,
            voxel_length=args.voxel_length,
            depth_trunc=args.depth_trunc,
        )
        if args.save_ply:
            o3d.io.write_triangle_mesh(args.save_ply, mesh_est)
            print(f"  Saved → {args.save_ply}")
    timing["est_tsdf_s"] = time.perf_counter() - t0
    est_n_verts = len(mesh_est.vertices)
    est_n_faces = len(mesh_est.triangles)
    print(f"  {est_n_verts:,} verts  {est_n_faces:,} faces")

    # ── SE(3) alignment (no scale) ────────────────────────────────────────────
    A                   = None
    est_centers_raw     = None
    est_centers_aligned = None
    if args.gt_align:
        print("\n[SE(3) alignment]")
        t0 = time.perf_counter()
        try:
            A = compute_se3_alignment(data, dataset, stride=args.align_stride)
            mesh_est = apply_alignment_to_mesh(mesh_est, A)
            wvts = data["world_view_transforms"]
            est_centers_raw     = np.array([c2w_from_wvt(wvts[ai])[:3, 3]
                                             for ai in arr_indices], dtype=np.float32)
            est_centers_aligned = (A[:3, :3] @ est_centers_raw.T + A[:3, 3:]).T.astype(np.float32)
        except (FileNotFoundError, ValueError) as exc:
            print(f"  [warn] Alignment failed: {exc}")
        timing["align_s"] = time.perf_counter() - t0

    # dataset no longer needed
    del dataset

    # ── Load Delaunay meshes and apply SE(3) alignment ────────────────────────
    mesh_del = mesh_del_vis = None
    if args.delaunay_ply and os.path.exists(args.delaunay_ply):
        print(f"\n[Delaunay] Loading {args.delaunay_ply}")
        mesh_del = o3d.io.read_triangle_mesh(args.delaunay_ply)
        mesh_del.compute_vertex_normals()
        if A is not None:
            mesh_del = apply_alignment_to_mesh(mesh_del, A)
        print(f"  {len(mesh_del.vertices):,} verts  {len(mesh_del.triangles):,} faces")
    if args.delaunay_visible_ply and os.path.exists(args.delaunay_visible_ply):
        print(f"\n[Delaunay visible] Loading {args.delaunay_visible_ply}")
        mesh_del_vis = o3d.io.read_triangle_mesh(args.delaunay_visible_ply)
        mesh_del_vis.compute_vertex_normals()
        if A is not None:
            mesh_del_vis = apply_alignment_to_mesh(mesh_del_vis, A)
        print(f"  {len(mesh_del_vis.vertices):,} verts  {len(mesh_del_vis.triangles):,} faces")

    # Reload GT mesh (was freed before est fusion to save RAM)
    print(f"\n[GT mesh] Reloading from {gt_ply}")
    mesh_gt = o3d.io.read_triangle_mesh(gt_ply)
    mesh_gt.compute_vertex_normals()

    # ── ICP refinement ────────────────────────────────────────────────────────
    if args.icp:
        print(f"\n[ICP]  threshold={args.icp_threshold} m  max_iter={args.icp_max_iter}")
        t0 = time.perf_counter()
        mesh_est, T_icp = icp_refine(
            mesh_gt, mesh_est,
            n_samples=args.icp_samples,
            max_iter=args.icp_max_iter,
            threshold=args.icp_threshold,
        )
        timing["icp_s"] = time.perf_counter() - t0
        if mesh_del is not None:
            mesh_del = mesh_del.transform(T_icp)
            mesh_del.compute_vertex_normals()
        if mesh_del_vis is not None:
            mesh_del_vis = mesh_del_vis.transform(T_icp)
            mesh_del_vis.compute_vertex_normals()

    # ── Chamfer ───────────────────────────────────────────────────────────────
    print(f"\n[Chamfer]  n_samples={args.n_samples:,}")

    t0 = time.perf_counter()
    metrics_tsdf = chamfer_distance(mesh_gt, mesh_est, n_samples=args.n_samples)
    timing["chamfer_tsdf_s"] = time.perf_counter() - t0
    print(f"\n  [TSDF vs GT]  chamfer={metrics_tsdf['chamfer_cm']:.3f} cm")

    metrics_del = metrics_del_vis = None
    if mesh_del is not None:
        t0 = time.perf_counter()
        metrics_del = chamfer_distance(mesh_gt, mesh_del, n_samples=args.n_samples)
        timing["chamfer_delaunay_s"] = time.perf_counter() - t0
        print(f"  [Delaunay vs GT]  chamfer={metrics_del['chamfer_cm']:.3f} cm")
    if mesh_del_vis is not None:
        t0 = time.perf_counter()
        metrics_del_vis = chamfer_distance(mesh_gt, mesh_del_vis, n_samples=args.n_samples)
        timing["chamfer_delaunay_visible_s"] = time.perf_counter() - t0
        print(f"  [Delaunay(vis) vs GT]  chamfer={metrics_del_vis['chamfer_cm']:.3f} cm")

    print(f"\n{'='*52}")
    for k, v in metrics_tsdf.items():
        print(f"  [tsdf] {k:<32s}: {v:.3f}")
    print(f"{'='*52}\n")

    all_metrics = {
        "tsdf": {**metrics_tsdf, **depth_metrics},
        "timing_s": timing,
        "mesh_info": {
            "gt_verts":  gt_n_verts,
            "gt_faces":  gt_n_faces,
            "est_verts": est_n_verts,
            "est_faces": est_n_faces,
            "voxel_length": args.voxel_length,
        },
    }
    if metrics_del is not None:
        all_metrics["delaunay"] = metrics_del
    if metrics_del_vis is not None:
        all_metrics["delaunay_visible"] = metrics_del_vis

    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Metrics saved → {metrics_path}")

    # ── Rerun ─────────────────────────────────────────────────────────────────
    if not args.no_rerun:
        import matplotlib.cm as cm

        _VIZ_PTS = 300_000

        def _error_colors(distances, max_error):
            """Map distances to turbo colormap, clamped at max_error."""
            normed = np.clip(distances / max_error, 0.0, 1.0)
            rgba = cm.turbo(normed)  # (N, 4) float in [0, 1]
            return (rgba[:, :3] * 255).astype(np.uint8)

        def _log_pcd(entity, mesh, color):
            pcd = mesh.sample_points_uniformly(number_of_points=_VIZ_PTS)
            pts = np.asarray(pcd.points, dtype=np.float32)
            rr.log(entity, rr.Points3D(pts, radii=0.003, colors=color), static=True)

        def _log_pcd_error(entity, mesh_src, mesh_tgt, max_error):
            """Sample points from mesh_src, color by distance to mesh_tgt."""
            pcd_src = mesh_src.sample_points_uniformly(number_of_points=_VIZ_PTS)
            pcd_tgt = mesh_tgt.sample_points_uniformly(number_of_points=_VIZ_PTS)
            pts = np.asarray(pcd_src.points, dtype=np.float32)
            dists = np.asarray(pcd_src.compute_point_cloud_distance(pcd_tgt))
            colors = _error_colors(dists, max_error)
            rr.log(entity, rr.Points3D(pts, radii=0.003, colors=colors), static=True)

        print("\n[Rerun]")
        rrd_path = os.path.join(out_dir, "chamfer_eval.rrd")
        rr.init("chamfer_gttsdf_eval")
        rr.save(rrd_path)

        # Flat-colored point clouds
        _log_pcd("/mesh/gt",  mesh_gt,  [180, 180, 180])
        _log_pcd("/mesh/est", mesh_est, [100, 180, 255])

        # Error-colored point clouds (distance to nearest point on other mesh)
        print(f"  Error colormap: 0 → {args.max_error*100:.1f} cm (turbo)")
        _log_pcd_error("/mesh/est_error", mesh_est, mesh_gt, args.max_error)
        _log_pcd_error("/mesh/gt_error",  mesh_gt, mesh_est, args.max_error)

        del mesh_gt, mesh_est

        if gt_centers is not None and len(gt_centers) > 1:
            rr.log("/traj/gt/pts",  rr.Points3D(gt_centers, radii=0.025,
                                                 colors=[50, 200, 50]), static=True)
            rr.log("/traj/gt/line", rr.LineStrips3D([gt_centers.tolist()],
                                                    colors=[[50, 200, 50]]), static=True)
        if est_centers_raw is not None and len(est_centers_raw) > 1:
            rr.log("/traj/est_raw/pts",  rr.Points3D(est_centers_raw, radii=0.025,
                                                       colors=[220, 120, 50]), static=True)
            rr.log("/traj/est_raw/line", rr.LineStrips3D([est_centers_raw.tolist()],
                                                          colors=[[220, 120, 50]]), static=True)
        if est_centers_aligned is not None and len(est_centers_aligned) > 1:
            rr.log("/traj/est_aligned/pts",  rr.Points3D(est_centers_aligned, radii=0.025,
                                                          colors=[50, 120, 220]), static=True)
            rr.log("/traj/est_aligned/line", rr.LineStrips3D([est_centers_aligned.tolist()],
                                                               colors=[[50, 120, 220]]), static=True)

        lines = ["── metrics ──"] + [f"  {k}: {v:.3f}" for k, v in metrics_tsdf.items()]
        rr.log("/metrics", rr.TextDocument("\n".join(lines)), static=True)

        print(f"  Rerun recording saved → {rrd_path}")
        print("  Open with: rerun {rrd_path}")
        print("  green  = GT trajectory")
        print("  orange = est trajectory (raw)")
        print("  blue   = est trajectory (SE(3)-aligned)")
    else:
        del mesh_gt, mesh_est


if __name__ == "__main__":
    main()
