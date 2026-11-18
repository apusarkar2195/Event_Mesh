import rerun as rr
import numpy as np

from scipy.spatial.transform import Rotation as R

import torch
from mesh_splatting.utils.sh_utils import eval_sh

def compute_mesh_colors(triangles, viewpoint_cam):
    features = triangles.get_features
    shs_view = features.transpose(1, 2).view(
        -1, 3, (triangles.max_sh_degree + 1) ** 2
    )
    shs_dc = shs_view[:, :, 0:1]
    dir_pp = triangles.get_vertices - viewpoint_cam.camera_center
    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(0, shs_dc, dir_pp_normalized)
    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    return torch.clamp(colors_precomp, 0.0, 1.0)

def log_trajectories_rerun(keyframes, num_frames_so_far=None, use_rerun=True, static=True):
    """
    Log GT and tracked camera trajectories to rerun for comparison.

    - viewpoint_stack: list of Camera; each has gt_T and (R, T) for tracked pose.
    - num_frames_so_far: number of frames to include (default: len(viewpoint_stack)).
    - use_rerun: if False, no logging (visualization disabled).
    """
    if not use_rerun:
        return

    if not keyframes:
        return

    gt_centers = []
    tracked_centers = []
    for kf in keyframes:
        gt_centers.append(kf.camera_center_gt.detach().cpu().numpy())
        tracked_centers.append(kf.camera_center.detach().cpu().numpy())

    gt_centers = np.array(gt_centers, dtype=np.float32)
    tracked_centers = np.array(tracked_centers, dtype=np.float32)

        # Align GT to tracked frame for display (same as ATE alignment: tracked -> gt gives R, t)
    if len(gt_centers) >= 3 and len(tracked_centers) >= 3:
        print("Aligning GT to tracked frame")
        from utils.eval_utils import umeyama_se3
        R_align, t_align = umeyama_se3(tracked_centers, gt_centers)
        # Inverse: put GT into tracked frame: p_tracked_frame = (p_gt - t) @ R
        gt_aligned = (gt_centers - t_align) @ R_align
    else:
        gt_aligned = gt_centers
        print("Not enough frames to align GT to tracked frame")
        return

    gt_aligned = np.asarray(gt_aligned, dtype=np.float32)
    tracked_centers = np.asarray(tracked_centers, dtype=np.float32)


    # One line strip per trajectory (green = GT, red = tracked)
    rr.log(
        "/trajectory/gt",
        rr.LineStrips3D(
            strips=[gt_aligned],
            colors=[0, 0, 255],
            radii=0.003,
        ),
        static=static,
    )
    rr.log(
        "/trajectory/tracked",
        rr.LineStrips3D(
            strips=[tracked_centers],
            colors=[255, 0, 0],
            radii=0.003,
        ),
        static=static,
    )

    # Points for clearer visibility at each frame
    rr.log(
        "/trajectory/gt/points",
        rr.Points3D(
            positions=gt_aligned,
            colors=np.tile([0, 255, 0], (len(gt_aligned), 1)).astype(np.uint8),
            radii=0.01,
        ),
        static=static,
    )
    rr.log(
        "/trajectory/tracked/points",
        rr.Points3D(
            positions=tracked_centers,
            colors=np.tile([255, 0, 255], (len(tracked_centers), 1)).astype(np.uint8),
            radii=0.01,
        ),
        static=static,
    )


def rerun_show_cam(cam_name, image, camera, static=True, use_rerun=True):
    if not use_rerun:
        return
    fx, fy, cx, cy = camera.fx, camera.fy, camera.cx, camera.cy
    h, w = image.shape[:2]


    q = R.from_matrix(camera.world_view_transform[:3, :3].detach().cpu().numpy()).as_quat()
    t = camera.camera_center.detach().cpu().numpy()

    rr.log(cam_name,
            rr.Transform3D(
                translation=t,
                rotation=rr.Quaternion(xyzw=q),
            ),
            rr.Pinhole(
                        focal_length=(fx, fy),
                        width=w,
                        height=h,
                        image_plane_distance=0.1,
                        aspect_ratio=w / h),
            
            rr.Image(image),
            static=static,
        )
    # rr.log(cam_name, rr.ViewCoordinates.RUB)


def log_mesh_rerun(triangles, viewpoint_cam, use_rerun=True, static=True):
    """
    Log the triangle mesh to Rerun. If the model has recently added triangles
    (densification), log them a second time under /mesh/new with green color
    so they are highlighted. Clears the 'new' range after logging.
    - triangles: scene.triangle_model (has .vertices, ._triangle_indices, ._new_triangle_range)
    """
    if not use_rerun:
        return
    verts = triangles.vertices.detach().cpu().numpy()
    tris = triangles._triangle_indices.detach().cpu().numpy()
    if verts.size == 0 or tris.size == 0:
        return
    # Default: gray vertex colors for full mesh
    num_verts = verts.shape[0]
    vertex_colors_gray = np.tile([180, 180, 180], (num_verts, 1)).astype(np.uint8)
    colors_precomp = compute_mesh_colors(triangles, viewpoint_cam)
    triangles_points = triangles.get_vertices
    triangle_indices = triangles.get_triangle_indices
    rr.log(
        "/mesh_dt",
        rr.Mesh3D(
            vertex_positions=triangles_points.cpu().detach().flatten(0, 1),
            triangle_indices=triangle_indices.cpu().detach(),
            vertex_colors=((colors_precomp * 255).to(torch.uint8).detach().cpu().numpy()),
        ),
        static=static,
    )
    # Highlight new triangles if we just densified
    new_range = getattr(triangles, "_new_triangle_range", None)
    if new_range is not None:
        start, end = new_range
        if end > start:
            new_tris = tris[start:end]
            # vertex_colors_green = np.tile([0, 255, 0], (num_verts, 1)).astype(np.uint8)
            vertex_colours = ((colors_precomp * 255).to(torch.uint8).detach().cpu().numpy())
            rr.log(
                "/new_mesh",
                rr.Mesh3D(
                    vertex_positions=verts,
                    triangle_indices=new_tris,
                    vertex_colors=vertex_colours,
                ),
                static=static,
            )
        triangles._new_triangle_range = None
