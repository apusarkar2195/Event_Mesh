"""
Evaluate triangle SLAM reconstruction against ground-truth TSDF mesh.

Pipeline:
  1. Load GT keyframes (RGB + depth) from the original dataset
  2. Load rendered keyframes from <result_dir>/depth_renders/
  3. Align estimated trajectory to GT via Umeyama similarity transform
  4. Build GT TSDF mesh  (GT depths + GT poses)
  5. Build Est TSDF mesh (rendered depths + aligned estimated poses)
  6. Compute Chamfer distance between the two meshes
  7. Visualize both meshes overlaid in Rerun

Usage:
    .venv/bin/python scripts/eval_tsdf.py \\
        --result_dir results/replica_office0/2026-03-03-12-52-51 \\
        --iteration 1800
"""

import argparse
import os

import numpy as np
import open3d as o3d
import rerun as rr
import torch
import yaml
from PIL import Image as PILImage
from tqdm import tqdm

from mesh_splatting.utils.graphics_utils import getWorld2View2
from utils.dataset import load_dataset


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def umeyama_alignment(src: np.ndarray, dst: np.ndarray, allow_scale: bool = True):
    """Estimate similarity transform mapping src → dst (N×3 arrays).
    Returns (scale, R 3×3, t 3) such that dst ≈ scale * R @ src + t.
    """
    n = len(src)
    mu_src = src.mean(0);  mu_dst = dst.mean(0)
    src_c  = src - mu_src; dst_c  = dst - mu_dst
    var_src = (src_c ** 2).sum() / n
    H = (dst_c.T @ src_c) / n
    U, S, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))])
    R = U @ D @ Vt
    s = float((S * D.diagonal()).sum() / var_src) if allow_scale else 1.0
    t = mu_dst - s * (R @ mu_src)
    return s, R, t


def similarity_matrix(s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3]  = t
    return T


def camera_center(T_w2c: np.ndarray) -> np.ndarray:
    """Camera position in world from a 4×4 world-to-camera matrix."""
    return np.linalg.inv(T_w2c)[:3, 3]


# ─────────────────────────────────────────────────────────────────────────────
# TSDF
# ─────────────────────────────────────────────────────────────────────────────

def build_tsdf(
    frames,           # list of (color uint8 H×W×3, depth float32 H×W, T_w2c 4×4)
    intrinsic_o3d,
    voxel_length: float = 0.04,
    depth_trunc: float = 5.0,
):
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=voxel_length * 3,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    for color_np, depth_np, T_w2c in tqdm(frames, desc="  TSDF integrate", leave=False):
        color_o3d = o3d.geometry.Image(np.ascontiguousarray(color_np))
        depth_o3d = o3d.geometry.Image(np.ascontiguousarray(depth_np))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1.0,           # already in metres
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )
        volume.integrate(rgbd, intrinsic_o3d, T_w2c.astype(np.float64))
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh


# ─────────────────────────────────────────────────────────────────────────────
# Chamfer distance
# ─────────────────────────────────────────────────────────────────────────────

def chamfer_distance(mesh_a, mesh_b, n_samples: int = 200_000) -> dict:
    pcd_a = mesh_a.sample_points_uniformly(n_samples)
    pcd_b = mesh_b.sample_points_uniformly(n_samples)
    d_ab = np.asarray(pcd_a.compute_point_cloud_distance(pcd_b))
    d_ba = np.asarray(pcd_b.compute_point_cloud_distance(pcd_a))
    return {
        "chamfer_cm":          (d_ab.mean() + d_ba.mean()) / 2 * 100,
        "gt→est_mean_cm":      d_ab.mean() * 100,
        "est→gt_mean_cm":      d_ba.mean() * 100,
        "gt→est_median_cm":    np.median(d_ab) * 100,
        "est→gt_median_cm":    np.median(d_ba) * 100,
        "completeness_<1cm_%": (d_ab < 0.01).mean() * 100,
        "accuracy_<1cm_%":     (d_ba < 0.01).mean() * 100,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Rerun helpers
# ─────────────────────────────────────────────────────────────────────────────

def rr_mesh(entity: str, mesh: o3d.geometry.TriangleMesh, fallback_rgb=(180, 180, 180)):
    verts  = np.asarray(mesh.vertices,  dtype=np.float32)
    faces  = np.asarray(mesh.triangles, dtype=np.uint32)
    if mesh.has_vertex_colors():
        vc = (np.asarray(mesh.vertex_colors) * 255).clip(0, 255).astype(np.uint8)
    else:
        vc = np.tile(np.array(fallback_rgb, dtype=np.uint8), (len(verts), 1))
    rr.log(entity, rr.Mesh3D(vertex_positions=verts, triangle_indices=faces,
                              vertex_colors=vc), static=True)


def rr_traj(entity: str, centers: np.ndarray, color):
    rr.log(entity + "/pts",  rr.Points3D(centers, colors=color, radii=0.025), static=True)
    rr.log(entity + "/line", rr.LineStrips3D([centers.tolist()], colors=[color]), static=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--iteration",    type=int,   default=1800)
    ap.add_argument("--voxel_length", type=float, default=0.04,
                    help="TSDF voxel size in metres (default 4 cm)")
    ap.add_argument("--depth_trunc",  type=float, default=5.0,
                    help="Max integration depth in metres")
    ap.add_argument("--n_samples",    type=int,   default=200_000,
                    help="Points sampled per mesh for Chamfer")
    ap.add_argument("--no_rerun",      action="store_true")
    ap.add_argument("--delaunay_mesh", default=None,
                    help="Path to a Delaunay PLY to compare against GT TSDF "
                         "(e.g. results/replica_meshes/office0/mesh_delaunay_postprocess.ply)")
    ap.add_argument("--delaunay_visible_mesh", default=None,
                    help="Path to visibility-filtered Delaunay PLY "
                         "(e.g. results/replica_meshes/office0/mesh_delaunay_visible.ply)")
    args = ap.parse_args()

    ckpt_dir    = os.path.join(args.result_dir, "mesh", f"iteration_{args.iteration}")
    renders_dir = os.path.join(args.result_dir, "depth_renders")
    out_dir     = os.path.join(args.result_dir, "tsdf_meshes")
    os.makedirs(out_dir, exist_ok=True)

    # ── Config ────────────────────────────────────────────────────────────────
    with open(os.path.join(args.result_dir, "config.yml")) as f:
        config = yaml.safe_load(f)

    cal = config["Dataset"]["Calibration"]
    fx, fy = float(cal["fx"]), float(cal["fy"])
    cx, cy = float(cal["cx"]), float(cal["cy"])
    W,  H  = int(cal["width"]), int(cal["height"])
    intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

    # ── Keyframe poses (estimated) ────────────────────────────────────────────
    kf_path   = os.path.join(ckpt_dir, "keyframes_state.pt")
    kf_state  = torch.load(kf_path, map_location="cpu")
    keyframes = kf_state["keyframes"]
    uid_list  = [kf["uid"] for kf in keyframes]
    print(f"Loaded {len(keyframes)} keyframes")

    est_T_w2c = {}
    for kf in keyframes:
        uid   = kf["uid"]
        R, T  = kf["R"].cuda(), kf["T"].cuda()
        est_T_w2c[uid] = getWorld2View2(R, T).cpu().numpy().astype(np.float64)

    # ── GT data from dataset ──────────────────────────────────────────────────
    dataset_path = config["Dataset"]["dataset_path"]
    if not os.path.isabs(dataset_path):
        project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
        dataset_path = os.path.normpath(os.path.join(project_root, dataset_path))
    print(f"Dataset: {dataset_path}")
    dataset = load_dataset(None, dataset_path, config)

    gt_T_w2c, gt_color, gt_depth = {}, {}, {}
    for uid in tqdm(uid_list, desc="Loading GT frames"):
        img_t, depth_np, pose_t = dataset[uid]
        # pose_t is T_w2c (ReplicaParser inverts traj.txt c2w → w2c)
        gt_T_w2c[uid] = pose_t.cpu().numpy().astype(np.float64)
        gt_color[uid] = (img_t.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        gt_depth[uid] = depth_np.astype(np.float32) if depth_np is not None else None

    # ── Umeyama: align est trajectory → GT frame ─────────────────────────────
    centers_est = np.stack([camera_center(est_T_w2c[u]) for u in uid_list])
    centers_gt  = np.stack([camera_center(gt_T_w2c[u])  for u in uid_list])

    s, R_al, t_al = umeyama_alignment(centers_est, centers_gt, allow_scale=True)
    T_align     = similarity_matrix(s, R_al, t_al)
    T_align_inv = np.linalg.inv(T_align)
    print(f"Umeyama: scale={s:.4f}  |t|={np.linalg.norm(t_al):.3f} m")

    # T_w2c_aligned = T_w2c_est @ inv(T_align)
    # (transforms world_gt coords → camera via estimated extrinsics)
    aligned_T_w2c = {u: est_T_w2c[u] @ T_align_inv for u in uid_list}
    centers_aligned = np.stack([camera_center(aligned_T_w2c[u]) for u in uid_list])
    ate = np.linalg.norm(centers_aligned - centers_gt, axis=1).mean()
    print(f"ATE after alignment: {ate * 100:.2f} cm")

    # ── Rendered depth + colour ───────────────────────────────────────────────
    est_color, est_depth = {}, {}
    for uid in uid_list:
        rgb_p   = os.path.join(renders_dir, f"{uid:05d}_rgb.png")
        depth_p = os.path.join(renders_dir, f"{uid:05d}_depth.npy")
        if not (os.path.exists(rgb_p) and os.path.exists(depth_p)):
            continue
        est_color[uid] = np.array(PILImage.open(rgb_p).convert("RGB"), dtype=np.uint8)
        d = np.load(depth_p).astype(np.float32)
        d[d <= 0] = 0.0   # zero out background
        # scale to GT metric units
        d *= s
        est_depth[uid] = d

    valid_uids = [u for u in uid_list
                  if u in est_depth and gt_depth.get(u) is not None]
    print(f"Valid frames for TSDF: {len(valid_uids)}")

    # ── GT TSDF ───────────────────────────────────────────────────────────────
    print("\n[GT TSDF]")
    gt_frames = [(gt_color[u], gt_depth[u], gt_T_w2c[u]) for u in valid_uids]
    mesh_gt = build_tsdf(gt_frames, intrinsic_o3d,
                         voxel_length=args.voxel_length,
                         depth_trunc=args.depth_trunc)
    print(f"  {len(mesh_gt.vertices):,} verts  {len(mesh_gt.triangles):,} faces")
    o3d.io.write_triangle_mesh(os.path.join(out_dir, "gt_tsdf.ply"), mesh_gt)

    # ── Est TSDF ──────────────────────────────────────────────────────────────
    print("\n[Est TSDF]")
    est_frames = [(est_color[u], est_depth[u], aligned_T_w2c[u]) for u in valid_uids]
    mesh_est = build_tsdf(est_frames, intrinsic_o3d,
                          voxel_length=args.voxel_length,
                          depth_trunc=args.depth_trunc)
    print(f"  {len(mesh_est.vertices):,} verts  {len(mesh_est.triangles):,} faces")
    o3d.io.write_triangle_mesh(os.path.join(out_dir, "est_tsdf.ply"), mesh_est)
    print(f"Saved meshes → {out_dir}")

    # ── Delaunay meshes (optional) ────────────────────────────────────────────
    def _load_delaunay(path: str | None, label: str):
        if not path:
            return None
        if not os.path.exists(path):
            print(f"  Warning: {label} not found at {path}")
            return None
        print(f"\n[{label}]  {path}")
        m = o3d.io.read_triangle_mesh(path)
        m.compute_vertex_normals()
        print(f"  {len(m.vertices):,} verts  {len(m.triangles):,} faces")
        m.transform(T_align)   # estimated world → GT world
        return m

    mesh_delaunay         = _load_delaunay(args.delaunay_mesh,         "Delaunay postprocess")
    mesh_delaunay_visible = _load_delaunay(args.delaunay_visible_mesh, "Delaunay visible")

    # ── Chamfer ───────────────────────────────────────────────────────────────
    import json
    all_metrics = {}

    print("\n[Chamfer: GT TSDF vs Est TSDF]")
    metrics_tsdf = chamfer_distance(mesh_gt, mesh_est, n_samples=args.n_samples)
    for k, v in metrics_tsdf.items():
        print(f"  {k:<30s}: {v:.3f}")
    all_metrics["tsdf"] = metrics_tsdf

    if mesh_delaunay is not None:
        print("\n[Chamfer: GT TSDF vs Delaunay postprocess]")
        metrics_del = chamfer_distance(mesh_gt, mesh_delaunay, n_samples=args.n_samples)
        for k, v in metrics_del.items():
            print(f"  {k:<30s}: {v:.3f}")
        all_metrics["delaunay"] = metrics_del

    if mesh_delaunay_visible is not None:
        print("\n[Chamfer: GT TSDF vs Delaunay visible]")
        metrics_vis = chamfer_distance(mesh_gt, mesh_delaunay_visible, n_samples=args.n_samples)
        for k, v in metrics_vis.items():
            print(f"  {k:<30s}: {v:.3f}")
        all_metrics["delaunay_visible"] = metrics_vis

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    # ── Rerun ─────────────────────────────────────────────────────────────────
    if not args.no_rerun:
        print("\n[Rerun]")
        rr.init("tsdf_eval", spawn=True)

        # GT mesh: green  |  Est TSDF: blue  |  Delaunay: orange
        rr_mesh("/mesh/gt",  mesh_gt,  fallback_rgb=(120, 200, 120))
        rr_mesh("/mesh/est", mesh_est, fallback_rgb=(120, 160, 220))
        if mesh_delaunay is not None:
            rr_mesh("/mesh/delaunay",         mesh_delaunay,         fallback_rgb=(220, 160,  80))
        if mesh_delaunay_visible is not None:
            rr_mesh("/mesh/delaunay_visible",  mesh_delaunay_visible, fallback_rgb=(220, 200, 100))

        # Trajectories
        rr_traj("/traj/gt",          centers_gt,      color=[50, 200, 50])
        rr_traj("/traj/est_raw",     centers_est,     color=[220, 120, 50])
        rr_traj("/traj/est_aligned", centers_aligned, color=[50, 120, 220])

        lines = []
        for tag, m in all_metrics.items():
            lines.append(f"── {tag} ──")
            lines += [f"  {k}: {v:.3f}" for k, v in m.items()]
        rr.log("/metrics", rr.TextDocument("\n".join(lines)), static=True)

        print("Rerun viewer spawned.")


if __name__ == "__main__":
    main()
