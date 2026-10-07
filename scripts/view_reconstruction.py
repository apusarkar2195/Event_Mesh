#!/usr/bin/env python3
"""
Visualize a MonoGS reconstruction from saved depth renders.

Two modes:

  depth  (default)
      Unproject depth maps into a coloured point cloud and draw camera
      frustums along the trajectory.  Fast; no TSDF integration needed.

  mesh
      Run TSDF volumetric fusion over every saved frame and display the
      resulting coloured triangle mesh.  Recommended for final inspection.

Usage
-----
  python view_reconstruction.py <depth_renders_dir> [options]

Examples
--------
  # Quick look at depth point cloud, every 5th frame
  python view_reconstruction.py results/.../depth_renders --stride 5

  # TSDF mesh fusion with defaults
  python view_reconstruction.py results/.../depth_renders --mode mesh

  # Fine-grained mesh, larger depth range, save to disk
  python view_reconstruction.py results/.../depth_renders --mode mesh \\
      --voxel_size 0.003 --sdf_trunc 0.015 --depth_trunc 4.0 \\
      --save_ply output_mesh.ply

  # Depth cloud capped at 3 m, voxel-downsampled
  python view_reconstruction.py results/.../depth_renders \\
      --depth_trunc 3.0 --voxel 0.005 --stride 2
"""

import argparse
import os
import sys
import yaml
import trimesh
import numpy as np
import open3d as o3d
import rerun as rr
import torch
from PIL import Image
from evo.core.geometry import umeyama_alignment

# Allow imports from the MonoGS project root regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def load_camera_data(render_dir: str):
    path = os.path.join(render_dir, "cameras.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"cameras.npz not found in {render_dir}")
    return np.load(path)


def w2c_from_wvt(wvt: np.ndarray) -> np.ndarray:
    """world_view_transform.T == W2C extrinsic (MonoGS convention)."""
    return wvt.T


def c2w_from_wvt(wvt: np.ndarray) -> np.ndarray:
    return np.linalg.inv(w2c_from_wvt(wvt))


def unproject_depth(depth, rgb, fx, fy, cx, cy, c2w, depth_trunc):
    """Return (N,3) world-space points and (N,3) colours for one frame."""
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32))
    valid = (depth > 1e-3) & (depth < depth_trunc)
    z = depth[valid]
    x = (u[valid] - cx) / fx * z
    y = (v[valid] - cy) / fy * z
    pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=1)   # (N,4)
    pts_world = (c2w @ pts_cam.T).T[:, :3]                    # (N,3)
    return pts_world.astype(np.float32), rgb[valid].astype(np.float32)



def apply_error_colormap(error, max_error):
    """Map (H,W) absolute error array to (H,W,3) RGB via the plasma colormap."""
    try:
        import matplotlib.cm as cm
        rgba = cm.plasma(np.clip(error / max_error, 0.0, 1.0))
        return rgba[:, :, :3].astype(np.float32)
    except ImportError:
        # Fallback: linear red channel only
        t = np.clip(error / max_error, 0.0, 1.0)[..., None]
        return np.concatenate([t, 1.0 - t, np.zeros_like(t)], axis=-1).astype(np.float32)


def load_gt_data(render_dir, dataset_path_override=None):
    """Load the dataset referenced by config.yml sitting next to depth_renders/.

    Returns (dataset, config) or raises FileNotFoundError / RuntimeError.
    The dataset's __getitem__(idx) returns (image, depth_metres, pose).
    depth is None when the dataset is monocular (has_depth=False).
    """
    import yaml
    from types import SimpleNamespace
    from utils.dataset import load_dataset

    save_dir = os.path.dirname(os.path.abspath(render_dir))
    config_path = os.path.join(save_dir, "config.yml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"config.yml not found at {config_path}\n"
            "Make sure render_dir is <save_dir>/depth_renders/ "
            "or pass --dataset_path explicitly."
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if dataset_path_override:
        config["Dataset"]["dataset_path"] = dataset_path_override

    dataset_path = config["Dataset"].get("dataset_path", "")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at '{dataset_path}'.\n"
            "Use --dataset_path to point to its current location."
        )

    dataset = load_dataset(SimpleNamespace(), dataset_path, config)
    return dataset, config


def load_gt_mesh(render_dir, dataset_path_override=None):
    """Load the GT scene mesh derived from the dataset path.

    Convention (Replica etc.): if the dataset is at  .../blabla/office0
    the GT mesh lives at                              .../blabla/office0.ply

    render_dir           : path to the depth_renders/ directory (used to find config.yml)
    dataset_path_override: if given, use this instead of the config's dataset_path
    """


    if dataset_path_override:
        dataset_path = dataset_path_override
    else:
        save_dir    = os.path.dirname(os.path.abspath(render_dir))
        config_path = os.path.join(save_dir, "config.yml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        dataset_path = config["Dataset"].get("dataset_path", "")

    mesh_path = dataset_path.rstrip("/").rstrip("\\") + "_mesh.ply"
    #mesh_path = '/media/makezur/Extreme SSD/Replica/office1_mesh.ply'
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"GT mesh not found at '{mesh_path}'")

    tm = trimesh.load(mesh_path, force="mesh", process=False)
    # Convert to Open3D TriangleMesh (trimesh already triangulates polygons)
    gt_mesh = o3d.geometry.TriangleMesh()
    gt_mesh.vertices  = o3d.utility.Vector3dVector(np.asarray(tm.vertices, dtype=np.float64))
    gt_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(tm.faces,    dtype=np.int32))
    if tm.visual is not None and hasattr(tm.visual, "vertex_colors") and tm.visual.vertex_colors is not None:
        cols = np.asarray(tm.visual.vertex_colors, dtype=np.float64)[:, :3] / 255.0
        gt_mesh.vertex_colors = o3d.utility.Vector3dVector(cols)
    gt_mesh.compute_vertex_normals()
    print(f"[gt mesh] {mesh_path}"
          f"  ({len(gt_mesh.vertices):,} verts, {len(gt_mesh.triangles):,} tris)")
    return gt_mesh


def replace_depthmaps_with_gt(extractor, arr_indices, data, dataset):
    """Replace rendered depth maps in the extractor with GT depths from the dataset.

    extractor   : GaussianExtractor already loaded with from_saved_renders()
    arr_indices : positional indices into cameras.npz (same ones used when loading)
    data        : cameras.npz NpzFile
    dataset     : MonocularDataset (or similar) with has_depth=True
    """
    frame_ids = data["frame_ids"]
    replaced = 0
    for i, arr_idx in enumerate(arr_indices):
        fid = int(frame_ids[arr_idx])
        try:
            _, gt_dep, _ = dataset[fid]
            if gt_dep is not None:
                gt_dep = np.asarray(gt_dep, dtype=np.float32)
                extractor.depthmaps[i] = torch.from_numpy(gt_dep).unsqueeze(0)  # (1,H,W)
                replaced += 1
        except Exception as exc:
            print(f"  [skip gt depth] frame {fid}: {exc}")
    print(f"[gt] Replaced {replaced}/{len(arr_indices)} depth maps with GT")
    return replaced


def compute_se3_alignment(data, dataset, stride=1):
    """Compute a SE(3) Umeyama alignment from the estimated to the GT camera trajectory.

    Uses every stride-th frame in cameras.npz for which a GT pose is available.
    No scale correction is applied.

    data    : cameras.npz NpzFile
    dataset : MonocularDataset (or similar); pose returned by dataset[idx] is W2C.
    stride  : use every N-th frame to speed up loading (default: 1 = all frames)

    Returns a (4, 4) float64 transform A such that  A @ C2W_est ≈ C2W_gt.
    """
    frame_ids = data["frame_ids"]
    indices   = np.arange(len(frame_ids))[::stride]
    est_centers, gt_centers = [], []

    for i in indices:
        fid = int(frame_ids[i])
        try:
            _, _, pose_tensor = dataset[fid]
            # dataset stores W2C; invert to get C2W
            w2c_gt = pose_tensor.cpu().numpy() if hasattr(pose_tensor, "cpu") else np.asarray(pose_tensor)
            c2w_gt  = np.linalg.inv(w2c_gt)

            wvt     = data["world_view_transforms"][i]
            c2w_est = c2w_from_wvt(wvt)

            est_centers.append(c2w_est[:3, 3])
            gt_centers.append(c2w_gt[:3, 3])
        except Exception as exc:
            print(f"  [skip align] frame {fid}: {exc}")

    n = len(est_centers)
    if n < 3:
        raise ValueError(f"Need ≥ 3 frames for SE(3) alignment, got {n}")

    x = np.stack(est_centers, axis=1).astype(np.float64)  # (3, n) – estimated
    y = np.stack(gt_centers,  axis=1).astype(np.float64)  # (3, n) – GT target

    R_align, t_align, _ = umeyama_alignment(x, y, with_scale=False)

    A = np.eye(4, dtype=np.float64)
    A[:3, :3] = R_align
    A[:3, 3]  = t_align

    rms = float(np.sqrt(np.mean(np.sum((R_align @ x + t_align[:, None] - y) ** 2, axis=0))))
    print(f"[align] SE(3) Umeyama over {n} frames (stride={stride}) — RMS = {rms:.4f} m")
    return A


def apply_alignment_to_extractor(extractor, A):
    """Apply a 4x4 SE(3) transform A to every camera pose in the extractor.

    A maps from the estimated world frame to the GT world frame (A @ C2W_est ≈ C2W_gt).
    Recomputes the bounding sphere after updating all poses.
    """
    for cam in extractor.viewpoint_stack:
        wvt     = cam.world_view_transform.cpu().numpy().astype(np.float64)
        c2w_new = A @ np.linalg.inv(wvt.T)        # A @ C2W_est
        wvt_new = np.linalg.inv(c2w_new).T        # back to world_view_transform convention
        cam.world_view_transform = torch.from_numpy(wvt_new.astype(np.float32)).cuda()
    extractor.estimate_bounding_sphere()


def apply_alignment_to_mesh(mesh, A):
    """Apply a 4×4 SE(3) transform A to all vertices of an Open3D TriangleMesh.

    A maps from the estimated world frame to the GT world frame (A @ C2W_est ≈ C2W_gt).
    Works on any mesh regardless of how it was produced (TSDF fusion or external PLY).
    Vertex normals are recomputed after the transform.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)           # (N, 3)
    verts_h = np.hstack([verts, np.ones((len(verts), 1))])        # (N, 4)
    mesh.vertices = o3d.utility.Vector3dVector((A @ verts_h.T).T[:, :3])
    mesh.compute_vertex_normals()
    return mesh


def _log_mesh_to_rerun(entity, o3d_mesh):
    """Log an Open3D TriangleMesh to Rerun under the given entity path."""
    verts = np.asarray(o3d_mesh.vertices,      dtype=np.float32)
    tris  = np.asarray(o3d_mesh.triangles,     dtype=np.uint32)
    norms = (np.asarray(o3d_mesh.vertex_normals, dtype=np.float32)
             if o3d_mesh.has_vertex_normals() else None)
    cols  = ((np.asarray(o3d_mesh.vertex_colors) * 255).astype(np.uint8)
             if o3d_mesh.has_vertex_colors() else None)
    rr.log(entity, rr.Mesh3D(
        vertex_positions=verts,
        triangle_indices=tris,
        vertex_normals=norms,
        vertex_colors=cols,
    ))
    print(f"  → rerun:{entity}  ({len(verts):,} verts, {len(tris):,} tris)")


# ---------------------------------------------------------------------------
# Depth mode
# ---------------------------------------------------------------------------

def run_depth_mode(render_dir, data, arr_indices, args):
    n = len(arr_indices)
    frame_ids = data["frame_ids"]
    print(f"[depth] Unprojecting {n} frames …")

    # Optionally load GT depths / compute alignment
    dataset  = None
    A_align  = None   # SE(3) transform: estimated world → GT world

    if args.gt_depth or args.gt_align:
        try:
            dataset, _ = load_gt_data(render_dir, args.dataset_path)
            if args.gt_depth and not getattr(dataset, "has_depth", False):
                print("[warn] Dataset has no GT depth (monocular). --gt_depth ignored.")
                dataset = None
            if dataset and args.gt_depth:
                print("[gt] GT depths loaded — error cloud coloured by plasma colormap")
                print(f"      error scale: 0 – {args.gt_error_scale} m  (blue → yellow)")
        except FileNotFoundError as e:
            print(f"[warn] Could not load GT data: {e}")

    if args.gt_align:
        if dataset is None:
            try:
                dataset, _ = load_gt_data(render_dir, args.dataset_path)
            except FileNotFoundError as e:
                print(f"[warn] Could not load GT data for alignment: {e}")
        if dataset is not None:
            try:
                A_align = compute_se3_alignment(data, dataset, stride=args.align_stride)
            except (ValueError, ImportError) as e:
                print(f"[warn] SE(3) alignment failed: {e}")

    all_pts,     all_cols     = [], []   # rendered cloud
    all_gt_pts,  all_gt_cols  = [], []   # GT error cloud
    cam_log = []                          # (c2w, fx, fy, cx, cy, W, H) per frame

    for _, arr_idx in enumerate(arr_indices):
        fid = int(frame_ids[arr_idx])

        rgb_path   = os.path.join(render_dir, f"{fid:05d}_rgb.png")
        depth_path = os.path.join(render_dir, f"{fid:05d}_depth.npy")
        if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
            print(f"  [skip] frame {fid}: files missing")
            continue

        rgb          = np.array(Image.open(rgb_path)).astype(np.float32) / 255.0
        rendered_dep = np.load(depth_path)

        wvt = data["world_view_transforms"][arr_idx]
        c2w = c2w_from_wvt(wvt)
        if A_align is not None:
            c2w = A_align @ c2w
        fx  = float(data["fx"][arr_idx])
        fy  = float(data["fy"][arr_idx])
        cx  = float(data["cx"][arr_idx])
        cy  = float(data["cy"][arr_idx])
        W   = int(data["widths"][arr_idx])
        H   = int(data["heights"][arr_idx])

        pts, cols = unproject_depth(rendered_dep, rgb, fx, fy, cx, cy, c2w,
                                    args.depth_trunc)
        all_pts.append(pts)
        all_cols.append(cols)

        # GT depth cloud coloured by per-pixel absolute error
        if dataset is not None:
            try:
                _, gt_dep, _ = dataset[fid]
                if gt_dep is not None:
                    gt_dep = np.asarray(gt_dep, dtype=np.float32)
                    error  = np.abs(rendered_dep - gt_dep)
                    error_rgb = apply_error_colormap(error, args.gt_error_scale)
                    gt_pts, gt_cols = unproject_depth(
                        gt_dep, error_rgb, fx, fy, cx, cy, c2w, args.depth_trunc
                    )
                    all_gt_pts.append(gt_pts)
                    all_gt_cols.append(gt_cols)
            except Exception as exc:
                print(f"  [skip gt] frame {fid}: {exc}")

        cam_log.append((c2w.copy(), fx, fy, cx, cy, W, H))

    if not all_pts:
        print("No frames loaded — nothing to display.")
        return

    rr.init("MonoGS reconstruction", spawn=True)

    # Rendered point cloud
    points = np.concatenate(all_pts, axis=0)
    colors = np.concatenate(all_cols, axis=0)
    print(f"Rendered points: {len(points):,}")
    if args.voxel > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd     = pcd.voxel_down_sample(args.voxel)
        points  = np.asarray(pcd.points, dtype=np.float32)
        colors  = np.asarray(pcd.colors, dtype=np.float32)
        print(f"After voxel downsampling ({args.voxel} m): {len(points):,} points")
    rr.log("depth/rendered", rr.Points3D(points, colors=(colors * 255).astype(np.uint8)))

    # GT error cloud
    if all_gt_pts:
        gt_points = np.concatenate(all_gt_pts, axis=0)
        gt_colors = np.concatenate(all_gt_cols, axis=0)
        print(f"GT error points: {len(gt_points):,}  (plasma: blue=low, yellow=high error)")
        if args.voxel > 0:
            gt_pcd = o3d.geometry.PointCloud()
            gt_pcd.points = o3d.utility.Vector3dVector(gt_points)
            gt_pcd.colors = o3d.utility.Vector3dVector(gt_colors)
            gt_pcd    = gt_pcd.voxel_down_sample(args.voxel)
            gt_points = np.asarray(gt_pcd.points, dtype=np.float32)
            gt_colors = np.asarray(gt_pcd.colors, dtype=np.float32)
        rr.log("depth/gt_error",
               rr.Points3D(gt_points, colors=(gt_colors * 255).astype(np.uint8)))

    # GT mesh
    if args.gt_mesh:
        try:
            gt_mesh = load_gt_mesh(render_dir, args.dataset_path)
            _log_mesh_to_rerun("mesh/gt", gt_mesh)
        except FileNotFoundError as e:
            print(f"[warn] {e}")

    # Camera trajectory and positions
    centers = np.array([c[:3, 3] for c, *_ in cam_log], dtype=np.float32)
    rr.log("cameras/trajectory", rr.LineStrips3D([centers], colors=[[200, 200, 200]]))
    rr.log("cameras/centers",    rr.Points3D(centers, radii=0.01, colors=[[180, 180, 50]]))
    rr.log("cameras/first",      rr.Points3D([centers[0]],  radii=0.03, colors=[[0, 220, 0]]))
    rr.log("cameras/last",       rr.Points3D([centers[-1]], radii=0.03, colors=[[0, 80, 255]]))


# ---------------------------------------------------------------------------
# Mesh mode
# ---------------------------------------------------------------------------

def build_tsdf_mesh(
    render_dir,
    data,
    arr_indices,
    *,
    voxel_size=0.02,
    sdf_trunc=0.08,
    depth_trunc=3.0,
    gt_depth=False,
    dataset_path=None,
    post_process=False,
    post_process_clusters=1,
):
    """Load saved renders, run TSDF fusion, and return the resulting mesh.

    SE(3) alignment is intentionally NOT done here — call apply_alignment_to_mesh()
    afterwards so the same step works for externally-provided PLY meshes too.
    """
    from mesh_utils import GaussianExtractor, post_process_mesh

    print(f"[tsdf] Fusing {len(arr_indices)} frames …")
    print(f"       voxel_size={voxel_size}  sdf_trunc={sdf_trunc}  depth_trunc={depth_trunc}")

    extractor = GaussianExtractor.from_saved_renders(render_dir, arr_indices=arr_indices)

    if gt_depth:
        try:
            dataset, _ = load_gt_data(render_dir, dataset_path)
            if not getattr(dataset, "has_depth", False):
                print("[warn] Dataset has no GT depth (monocular). --gt_depth ignored.")
            else:
                replace_depthmaps_with_gt(extractor, arr_indices, data, dataset)
        except FileNotFoundError as e:
            print(f"[warn] Could not load GT data: {e}")

    print("[tsdf] Running extract_mesh_bounded …")
    mesh = extractor.extract_mesh_bounded(
        voxel_size=voxel_size,
        sdf_trunc=sdf_trunc,
        depth_trunc=depth_trunc,
    )

    if post_process:
        mesh = post_process_mesh(mesh, cluster_to_keep=post_process_clusters)

    mesh.compute_vertex_normals()
    return mesh


def run_mesh_mode(render_dir, data, arr_indices, args):
    mesh = build_tsdf_mesh(
        render_dir, data, arr_indices,
        voxel_size=args.voxel_size,
        sdf_trunc=args.sdf_trunc,
        depth_trunc=args.depth_trunc,
        gt_depth=args.gt_depth,
        dataset_path=args.dataset_path,
        post_process=args.post_process,
        post_process_clusters=args.post_process_clusters,
    )

    if args.gt_align:
        try:
            dataset, _ = load_gt_data(render_dir, args.dataset_path)
            A = compute_se3_alignment(data, dataset, stride=args.align_stride)
            mesh = apply_alignment_to_mesh(mesh, A)
        except (FileNotFoundError, ValueError, ImportError) as e:
            print(f"[warn] SE(3) alignment failed: {e}")

    if args.save_ply:
        o3d.io.write_triangle_mesh(args.save_ply, mesh)
        print(f"Mesh saved to {args.save_ply}")

    traj_centers = np.array([
        c2w_from_wvt(data["world_view_transforms"][ai])[:3, 3]
        for ai in arr_indices
    ], dtype=np.float32)

    rr.init("MonoGS reconstruction", spawn=True)

    _log_mesh_to_rerun("mesh/tsdf", mesh)

    if args.gt_mesh:
        try:
            gt_mesh = load_gt_mesh(render_dir, args.dataset_path)
            if not gt_mesh.has_vertex_colors():
                gt_mesh.paint_uniform_color([0.65, 0.65, 0.65])
            gt_mesh.compute_vertex_normals()
            _log_mesh_to_rerun("mesh/gt", gt_mesh)
        except FileNotFoundError as e:
            print(f"[warn] {e}")

    rr.log("cameras/trajectory", rr.LineStrips3D([traj_centers], colors=[[0, 220, 0]]))
    rr.log("cameras/first",      rr.Points3D([traj_centers[0]],  radii=0.03, colors=[[0, 220, 0]]))
    rr.log("cameras/last",       rr.Points3D([traj_centers[-1]], radii=0.03, colors=[[0, 80, 255]]))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("render_dir", help="Path to the depth_renders/ directory")
    parser.add_argument("--mode", choices=["depth", "mesh"], default="depth",
                        help="Visualisation mode (default: depth)")

    # Shared
    parser.add_argument("--depth_trunc", type=float, default=3.0, metavar="F",
                        help="Max depth in metres (default: 3.0)")
    parser.add_argument("--gt_depth", action="store_true",
                        help="Load GT depths from the dataset referenced by config.yml. "
                             "depth mode: adds an error-coloured GT cloud. "
                             "mesh mode: replaces rendered depths with GT for TSDF fusion.")
    parser.add_argument("--dataset_path", type=str, default=None, metavar="PATH",
                        help="Override the dataset path from config.yml "
                             "(useful if data has moved since the run)")
    parser.add_argument("--gt_error_scale", type=float, default=0.5, metavar="F",
                        help="Depth error mapped to maximum colour in the plasma map, "
                             "in metres (default: 0.5)")
    parser.add_argument("--gt_align", action="store_true",
                        help="SE(3) Umeyama-align the estimated camera trajectory to GT "
                             "before fusion/unprojection (no scale correction). "
                             "Requires GT poses in the dataset.")
    parser.add_argument("--align_stride", type=int, default=10, metavar="N",
                        help="Use every N-th frame when computing SE(3) alignment "
                             "(default: 10 — fast; use 1 for maximum accuracy)")
    parser.add_argument("--gt_mesh", action="store_true",
                        help="Overlay the GT scene mesh derived from the dataset path. "
                             "Follows the convention: dataset at .../office0 → "
                             "GT mesh at .../office0.ply  (Replica, ScanNet, etc.). "
                             "Use --gt_align to bring both into the same frame.")

    # Depth-mode options
    g = parser.add_argument_group("depth mode")
    g.add_argument("--stride",       type=int,   default=1,   metavar="N",
                   help="Use every N-th frame (default: 1)")
    g.add_argument("--max_frames",   type=int,   default=None, metavar="N",
                   help="Cap total frames (default: all)")
    g.add_argument("--voxel",        type=float, default=0.0,  metavar="F",
                   help="Point-cloud voxel downsampling size, 0=off (default: 0)")
    g.add_argument("--frustum_scale", type=float, default=0.05, metavar="F",
                   help="Camera frustum size (default: 0.05)")

    # Mesh-mode options
    g2 = parser.add_argument_group("mesh mode")
    g2.add_argument("--voxel_size",  type=float, default=0.004, metavar="F",
                    help="TSDF voxel size (default: 0.004)")
    g2.add_argument("--sdf_trunc",   type=float, default=0.02,  metavar="F",
                    help="TSDF truncation distance (default: 0.02)")
    g2.add_argument("--post_process", action="store_true",
                    help="Remove floater clusters after fusion")
    g2.add_argument("--post_process_clusters", type=int, default=1000, metavar="N",
                    help="Clusters to keep in post-processing (default: 1000)")
    g2.add_argument("--save_ply",    type=str,   default=None,  metavar="PATH",
                    help="Save extracted mesh to this .ply file")

    args = parser.parse_args()

    render_dir = args.render_dir
    data       = load_camera_data(render_dir)

    # Build array-position indices (not frame IDs) for selected frames
    total = len(data["frame_ids"])
    arr_indices = np.arange(total)[::args.stride]
    if args.max_frames is not None:
        arr_indices = arr_indices[:args.max_frames]

    if args.mode == "depth":
        run_depth_mode(render_dir, data, arr_indices, args)
    else:
        run_mesh_mode(render_dir, data, arr_indices, args)


if __name__ == "__main__":
    main()
