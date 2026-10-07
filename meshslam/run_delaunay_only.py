#!/usr/bin/env python3
"""
Load a triangle mesh from a saved state, set sigma small, run Restricted Delaunay only,
and display before/after in Rerun. No cameras, no finetuning.
"""
import os
import sys
from pathlib import Path
from argparse import Namespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import rerun as rr
import open3d as o3d

from mesh_splatting.scene.triangle_model import TriangleModel
from mesh_splatting.triangle_renderer import render
from mesh_splatting.utils.graphics_utils import getProjectionMatrix2
from mesh_splatting.arguments import OptimizationParams, PipelineParams
from utils.config_utils import load_config
from utils.mesh_color_utils import smooth_vertex_colors
from utils.mesh_utils import post_process_mesh
from utils.dataset import load_dataset
from utils.cameras import Camera

SH_C0 = 0.28209479177387814


def make_pipe_from_config(config: dict):
    from argparse import ArgumentParser
    parser = ArgumentParser()
    pipe_template = PipelineParams(parser)
    args = parser.parse_args([])
    return pipe_template.extract(args)


def build_cameras_and_keyframes(config, dataset_path, state_dir, num_render: int):
    """Load dataset, build cameras, load SLAM keyframes, select N evenly spaced."""
    ds_args = Namespace()
    dataset = load_dataset(ds_args, dataset_path, config)
    proj = getProjectionMatrix2(
        znear=0.01, zfar=100.0,
        fx=dataset.fx, fy=dataset.fy, cx=dataset.cx, cy=dataset.cy,
        W=dataset.width, H=dataset.height,
    ).transpose(0, 1)
    all_cams = [Camera.init_from_dataset(dataset, i, proj) for i in range(len(dataset))]
    kf_path = os.path.join(state_dir, "keyframes_state.pt")
    if not os.path.isfile(kf_path):
        raise FileNotFoundError(f"keyframes_state.pt not found in {state_dir}. Required for --render_keyframes.")
    payload = torch.load(kf_path, map_location="cpu")
    keyframes = payload.get("keyframes", [])
    if not keyframes:
        raise RuntimeError("No keyframes in keyframes_state.pt")
    cams_by_uid = {int(c.uid): c for c in all_cams}
    kf_cams = []
    for kf in keyframes:
        uid = int(kf["uid"])
        c = cams_by_uid.get(uid)
        if c is not None:
            c = cams_by_uid[uid]
            c.update_RT(kf["R"].to(device=c.device), kf["T"].to(device=c.device))
            kf_cams.append(c)
    if len(kf_cams) < num_render:
        selected = kf_cams
    else:
        indices = np.linspace(0, len(kf_cams) - 1, num_render, dtype=int)
        selected = [kf_cams[i] for i in indices]
    return selected


def save_image(tensor: torch.Tensor, path: str):
    """Save (3,H,W) float [0,1] to PNG."""
    arr = (tensor.detach().cpu().clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    import cv2
    cv2.imwrite(path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


def vertex_colors_from_features(
    triangles: TriangleModel,
    clamp_dc: float | None = None,
    smooth: str | None = None,
    smooth_iters: int = 1,
) -> np.ndarray:
    """Compute vertex colors from SH DC (no camera needed)."""
    dc = triangles._features_dc
    if clamp_dc is not None and clamp_dc > 0:
        dc = torch.clamp(dc, -clamp_dc, clamp_dc)
    colors = torch.clamp(SH_C0 * dc + 0.5, 0.0, 1.0)
    colors = colors.detach().cpu().numpy().squeeze()
    if colors.ndim == 3:
        colors = colors.squeeze(0)
    if smooth and smooth_iters > 0:
        tris = triangles._triangle_indices.detach().cpu().numpy()
        colors = smooth_vertex_colors(colors, tris, method=smooth, iterations=smooth_iters)
    return (colors * 255).round().astype(np.uint8)


def filter_triangles_by_opacity(triangles: TriangleModel, threshold: float) -> int:
    """Remove triangles with min vertex opacity < threshold, then prune unused vertices. Returns number removed."""
    if threshold <= 0:
        return 0
    triangle_vertex_weights = triangles.opacity_activation(
        triangles.vertex_weight[triangles._triangle_indices]
    )
    min_weights = triangle_vertex_weights.min(dim=1).values
    keep_mask = (min_weights >= threshold).squeeze()
    n_before = triangles._triangle_indices.shape[0]
    triangles.prune_triangles(keep_mask)
    n_after = triangles._triangle_indices.shape[0]
    # Prune orphan vertices
    device = triangles.vertices.device
    used_vertex_mask = torch.zeros(triangles.vertices.shape[0], dtype=torch.bool, device=device)
    if triangles._triangle_indices.numel() > 0:
        used_vertex_mask[triangles._triangle_indices.flatten()] = True
    triangles._prune_vertices(used_vertex_mask)
    return int(n_before - n_after)


def triangles_to_o3d_mesh(
    triangles: TriangleModel,
    clamp_dc: float | None = None,
    smooth: str | None = None,
    smooth_iters: int = 1,
) -> o3d.geometry.TriangleMesh:
    """Convert TriangleModel to Open3D mesh with vertex colors."""
    verts = triangles.vertices.detach().cpu().numpy().astype(np.float64)
    tris = triangles._triangle_indices.detach().cpu().numpy().astype(np.int32)
    colors = vertex_colors_from_features(
        triangles, clamp_dc=clamp_dc, smooth=smooth, smooth_iters=smooth_iters
    )
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts),
        o3d.utility.Vector3iVector(tris),
    )
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    return mesh


def log_mesh_rerun(
    triangles: TriangleModel,
    path: str,
    static: bool = True,
    clamp_dc: float | None = None,
    smooth: str | None = None,
    smooth_iters: int = 1,
):
    """Log triangle mesh to Rerun at the given path."""
    verts = triangles.vertices.detach().cpu().numpy()
    tris = triangles._triangle_indices.detach().cpu().numpy()
    if verts.size == 0 or tris.size == 0:
        return
    colors = vertex_colors_from_features(
        triangles, clamp_dc=clamp_dc, smooth=smooth, smooth_iters=smooth_iters
    )
    rr.log(
        path,
        rr.Mesh3D(
            vertex_positions=triangles.get_vertices.detach().cpu().numpy(),
            triangle_indices=triangles.get_triangle_indices.detach().cpu().numpy(),
            vertex_colors=colors,
        ),
        static=static,
    )


def log_o3d_mesh_rerun(mesh: o3d.geometry.TriangleMesh, path: str, static: bool = True):
    """Log Open3D mesh to Rerun at the given path."""
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    if verts.size == 0 or tris.size == 0:
        return
    colors = np.ones((len(verts), 3), dtype=np.uint8) * 128
    if mesh.has_vertex_colors():
        colors = (np.asarray(mesh.vertex_colors) * 255).round().astype(np.uint8)
    rr.log(path, rr.Mesh3D(vertex_positions=verts, triangle_indices=tris, vertex_colors=colors), static=static)


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser("Run Restricted Delaunay on mesh, display before/after")
    parser.add_argument("--config", required=True, help="YAML config (for TriangleModel)")
    parser.add_argument("--state_path", required=True, help="Path to state dir or point_cloud_state_dict.pt")
    parser.add_argument("--sigma", type=float, default=0.001, help="Sigma value to set (default: 0.001)")
    parser.add_argument("--opacity_threshold", type=float, default=0.99, help="Filter out triangles with opacity < this. Set to 0 to disable.")
    parser.add_argument("--clamp_features", type=float, default=4.0, help="Clamp SH DC to [-x,x] before color conversion (default: 4.0). Set 0 to disable.")
    parser.add_argument("--smooth", choices=["mean", "median", "huber"], default=None, help="Smooth vertex colors to reduce speckling: mean (Laplacian), median (robust), huber (trimmed mean)")
    parser.add_argument("--smooth_iters", type=int, default=2, help="Smoothing iterations (default: 2)")
    parser.add_argument("--cluster_to_keep", type=int, default=1000, help="Post-process: keep top N clusters by size (like TSDF pipeline). Set 0 to disable.")
    parser.add_argument("--out_dir", default=None, help="Optional: save state and PLY after Delaunay")
    parser.add_argument("--zup", action="store_true", help="Z-up to Y-up in PLY export")
    parser.add_argument("--render_keyframes", type=int, default=0, help="Render from N keyframes and save comparison images. Requires --dataset_path.")
    parser.add_argument("--dataset_path", default=None, help="Dataset path (required when --render_keyframes > 0). Overrides config.")
    parser.add_argument("--time-only", action="store_true", help="Just time run_restricted_delaunay() and exit, no Rerun/rendering/saving.")
    args = parser.parse_args()

    state_dir = args.state_path
    if state_dir.endswith(".pt"):
        state_dir = str(Path(state_dir).parent)
    if not os.path.isfile(os.path.join(state_dir, "point_cloud_state_dict.pt")):
        raise FileNotFoundError(f"Expected point_cloud_state_dict.pt in {state_dir}")

    config = load_config(args.config)
    triangles = TriangleModel(sh_degree=0, config=config)
    triangles.load_parameters(state_dir, device="cuda")

    mem_mb = torch.cuda.memory_allocated() / (1024 * 1024)
    print(f"Loaded: {triangles.vertices.shape[0]} vertices, {triangles._triangle_indices.shape[0]} triangles")
    print(f"Model GPU memory: {mem_mb:.1f} MB")

    triangles.set_sigma(args.sigma)

    if args.opacity_threshold > 0:
        n_removed = filter_triangles_by_opacity(triangles, args.opacity_threshold)
        print(f"Filtered {n_removed} triangles with opacity < {args.opacity_threshold}")
        print(f"After filter: {triangles.vertices.shape[0]} vertices, {triangles._triangle_indices.shape[0]} triangles")

    clamp_dc = args.clamp_features if args.clamp_features > 0 else None
    smooth = args.smooth
    smooth_iters = args.smooth_iters if smooth else 0

    if args.time_only:
        import time
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        triangles.run_restricted_delaunay()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        v = triangles.vertices.shape[0]
        t = triangles._triangle_indices.shape[0]
        print(f"Delaunay: {elapsed:.3f}s  ({v} verts, {t} tris after)")
        return

    rr.init("delaunay_only", spawn=True)

    render_cams = []
    pipe = None
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    if args.render_keyframes > 0:
        ds_path = args.dataset_path or config.get("Dataset", {}).get("dataset_path")
        if not ds_path:
            raise ValueError("--render_keyframes requires --dataset_path or Dataset.dataset_path in config")
        render_cams = build_cameras_and_keyframes(config, ds_path, state_dir, args.render_keyframes)
        pipe = make_pipe_from_config(config)
        render_out = (Path(args.out_dir) / "renders") if args.out_dir else Path("/tmp/delaunay_renders")
        render_out.mkdir(parents=True, exist_ok=True)
        print(f"Rendering from {len(render_cams)} keyframes to {render_out}")

    def do_render(cam, triangles, sigma_override=None):
        if sigma_override is not None:
            triangles.set_sigma(sigma_override)
        with torch.no_grad():
            pkg = render(cam, triangles, pipe, background, optimize_pose_only=False)
        if sigma_override is not None:
            triangles.set_sigma(args.sigma)
        return pkg["render"]

    # Before Delaunay: save GT and renders
    if render_cams:
        for i, cam in enumerate(render_cams):
            prefix = f"{i:03d}"
            save_image(cam.original_image.cuda(), str(render_out / f"{prefix}_gt.png"))
            save_image(do_render(cam, triangles), str(render_out / f"{prefix}_tri_before.png"))
            save_image(do_render(cam, triangles, sigma_override=0.0001), str(render_out / f"{prefix}_tri_before_hard.png"))
        print(f"Saved before-Delaunay renders for {len(render_cams)} keyframes")

    # Before Delaunay
    log_mesh_rerun(triangles, "/mesh/before", static=True, clamp_dc=clamp_dc, smooth=smooth, smooth_iters=smooth_iters)
    print("Logged mesh BEFORE Delaunay at /mesh/before")

    # Run Restricted Delaunay
    triangles.run_restricted_delaunay()

    # After Delaunay: save renders (default and smoothed)
    if render_cams:
        for i, cam in enumerate(render_cams):
            prefix = f"{i:03d}"
            save_image(do_render(cam, triangles), str(render_out / f"{prefix}_mesh_after_delaunay.png"))
        if smooth and smooth_iters > 0:
            dc_backup = triangles._features_dc.clone()
            colors_smooth = vertex_colors_from_features(triangles, clamp_dc=clamp_dc, smooth=smooth, smooth_iters=smooth_iters)
            dc_new = (torch.from_numpy(colors_smooth.astype(np.float32) / 255.0) - 0.5) / SH_C0
            dc_new = dc_new.reshape(-1, 1, 3).to(device=triangles._features_dc.device, dtype=triangles._features_dc.dtype)
            triangles._features_dc = dc_new
            for i, cam in enumerate(render_cams):
                prefix = f"{i:03d}"
                save_image(do_render(cam, triangles), str(render_out / f"{prefix}_mesh_after_delaunay_smooth.png"))
            triangles._features_dc = dc_backup
        print(f"Saved after-Delaunay renders for {len(render_cams)} keyframes")

    # After Delaunay
    log_mesh_rerun(triangles, "/mesh/after", static=True, clamp_dc=clamp_dc, smooth=smooth, smooth_iters=smooth_iters)
    print("Logged mesh AFTER Delaunay at /mesh/after")

    mesh_postprocess = None
    if args.cluster_to_keep > 0:
        mesh_o3d = triangles_to_o3d_mesh(
            triangles, clamp_dc=clamp_dc, smooth=smooth, smooth_iters=smooth_iters
        )
        mesh_postprocess = post_process_mesh(mesh_o3d, cluster_to_keep=args.cluster_to_keep)
        log_o3d_mesh_rerun(mesh_postprocess, "/mesh/after_postprocess", static=True)
        print("Logged mesh AFTER post-process (cluster filter) at /mesh/after_postprocess")

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        triangles.save_parameters(args.out_dir)
        colors_u8 = vertex_colors_from_features(triangles, clamp_dc=clamp_dc, smooth=smooth, smooth_iters=smooth_iters)
        verts_np = triangles.vertices.detach().cpu().numpy()
        faces_np = triangles._triangle_indices.detach().cpu().numpy()
        from meshslam.create_ply import export_ply

        ply_path = os.path.join(args.out_dir, "mesh_delaunay.ply")
        export_ply(ply_path, verts_np, faces_np, colors_u8, zup_to_yup=args.zup)
        print(f"Saved state to {args.out_dir}")
        print(f"Saved PLY to {ply_path}")

        if mesh_postprocess is not None:
            ply_post_path = os.path.join(args.out_dir, "mesh_delaunay_postprocess.ply")
            o3d.io.write_triangle_mesh(ply_post_path, mesh_postprocess)
            print(f"Saved post-processed PLY to {ply_post_path}")

    print("Done. View in Rerun: /mesh/before, /mesh/after" + (", /mesh/after_postprocess" if mesh_postprocess else ""))


if __name__ == "__main__":
    main()
