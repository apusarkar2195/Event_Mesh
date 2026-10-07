#
# The original code is under the following copyright:
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE_GS.md file.
#
# For inquiries contact george.drettakis@inria.fr
#
# The modifications of the code are under the following copyright:
# Copyright (C) 2025, University of Liege
# TELIM research group, http://www.telecom.ulg.ac.be/
# All rights reserved.
# The modifications are under the LICENSE.md file.
#
# For inquiries contact jan.held@uliege.be
#

import torch
import math
import numpy as np
from diff_triangle_rasterization import TriangleRasterizationSettings, TriangleRasterizer
from mesh_splatting.scene.triangle_model import TriangleModel
from mesh_splatting.utils.sh_utils import eval_sh
from mesh_splatting.utils.point_utils import depth_to_normal
import torch.nn.functional as F

def normals_world_to_view(view, normal_world):
    # normal_world: [H,W,3]
    # c2w: camera -> world; so R_cw = c2w[:3,:3].T maps world -> camera
    c2w = (view.world_view_transform.T).inverse()
    R_cw = c2w[:3,:3].T  # world->camera rotation

    H, W, _ = normal_world.shape
    n_cam = (R_cw @ normal_world.reshape(-1,3).T).T.reshape(H,W,3)
    n_cam = F.normalize(n_cam, dim=-1)

    # Optional: ensure normals face the camera (negative z if your camera looks along -Z)
    # Flip so z <= 0 (adjust sign according to your convention)
    flip = (n_cam[...,2] > 0).unsqueeze(-1)
    n_cam = torch.where(flip, -n_cam, n_cam)

    return n_cam  # view-space normals [H,W,3]


def transform_point_4x3(points, matrix):
    """
    Transform 3D points using a 4x4 matrix (matches the CUDA implementation exactly)
    points: (N, 3) tensor of 3D points
    matrix: (4, 4) transformation matrix (COLUMN-MAJOR like OpenGL/CUDA)
    returns: (N, 3) transformed points
    """
    # The CUDA code uses column-major layout
    # We need to manually implement the exact same transformation
    
    # Flatten the matrix to access elements like CUDA does
    m = matrix.flatten()
    
    # Pre-allocate output (all elements overwritten below)
    transformed = torch.empty_like(points)
    
    # Apply the EXACT same transformation as CUDA's transformPoint4x3
    transformed[:, 0] = m[0] * points[:, 0] + m[4] * points[:, 1] + m[8] * points[:, 2] + m[12]
    transformed[:, 1] = m[1] * points[:, 0] + m[5] * points[:, 1] + m[9] * points[:, 2] + m[13]
    transformed[:, 2] = m[2] * points[:, 0] + m[6] * points[:, 1] + m[10] * points[:, 2] + m[14]
    
    return transformed


def compute_image_2d_pytorch_exact(vertices, projmatrix, W, H):
    """
    EXACT match to the CUDA kernel implementation - ROW-MAJOR matrix layout
    """
    # Flatten the matrix to access elements like CUDA does
    m = projmatrix.flatten()
    
    # Transform each vertex exactly like the CUDA kernel
    x = vertices[:, 0]
    y = vertices[:, 1] 
    z = vertices[:, 2]
    
    # EXACT same calculation as transformPoint4x4 in CUDA (ROW-MAJOR)
    p_clip_x = m[0] * x + m[4] * y + m[8] * z + m[12]   # row0: m[0], m[4], m[8], m[12]
    p_clip_y = m[1] * x + m[5] * y + m[9] * z + m[13]   # row1: m[1], m[5], m[9], m[13]
    p_clip_z = m[2] * x + m[6] * y + m[10] * z + m[14]  # row2: m[2], m[6], m[10], m[14]
    p_clip_w = m[3] * x + m[7] * y + m[11] * z + m[15]  # row3: m[3], m[7], m[11], m[15]
    
    # Perspective division (same as CUDA)
    invw = 1.0 / (p_clip_w + 1e-8)
    ndc_x = p_clip_x * invw
    ndc_y = p_clip_y * invw
    
    # Convert to pixel coordinates (same ndc2Pix as CUDA) - FIXED: return [x, y] not [y, x]
    image_2D_pytorch = torch.stack([
        ((ndc_x + 1.0) * W - 1.0) * 0.5,  # x coordinate (width)
        ((ndc_y + 1.0) * H - 1.0) * 0.5   # y coordinate (height)
    ], dim=1)
    
    return image_2D_pytorch


def render(viewpoint_camera, pc : TriangleModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, optimize_pose_only = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    
    triangles_indices = pc.get_triangle_indices  # the idx of the 3 vertices of each triangl
    vertices = pc.get_vertices # contains all the vertices of the triangles
    vertex_weights = pc.get_vertex_weight  # contains the weights of the vertices for each vertex in the triangles
    scaling = torch.zeros_like(triangles_indices[:, 0], dtype=pc.get_triangles_points.dtype, requires_grad=True, device="cuda").detach()

    vertex_index = pc._triangle_indices.shape[0]

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    H_init = int(viewpoint_camera.image_height)
    W_init = int(viewpoint_camera.image_width)

    upsample = pc.scaling

    H = upsample * H_init
    W = upsample * W_init

    proj_matrix_raw = viewpoint_camera.projection_matrix


    min_triangle_area_pixels = 0.5
    opacity_aggregation = "mean"
    if getattr(pc, "config", None) is not None:
        min_triangle_area_pixels = pc.config.get("Training", {}).get("min_triangle_area_pixels", 0.5)
        opacity_aggregation = pc.config.get("Training", {}).get("opacity_aggregation", "mean")

    raster_settings = TriangleRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        projmatrix_raw=proj_matrix_raw,
        cx=viewpoint_camera.cx if viewpoint_camera.cx is not None else 0,
        cy=viewpoint_camera.cy if viewpoint_camera.cy is not None else 0,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        min_triangle_area_pixels=min_triangle_area_pixels,
        optimize_pose_only=optimize_pose_only,
        opacity_min=(opacity_aggregation == "min"),
    )

    rasterizer = TriangleRasterizer(raster_settings=raster_settings)

    sigma = pc.get_sigma

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features

    else:
        colors_precomp = override_color

    # Rasterize visible triangles to image, obtain their radii (on screen). 
    rendered_image, radii, scaling, allmap, max_blending, was_rendered  = rasterizer(
        vertices=vertices,
        triangles_indices=triangles_indices,
        vertex_weights=vertex_weights.squeeze(),
        sigma=sigma,
        shs = shs,
        colors_precomp = colors_precomp,
        scaling = scaling,
        w=viewpoint_camera.cam_rot_delta,
        trans=viewpoint_camera.cam_trans_delta,
       )

    radii = radii[:vertex_index]
    scaling = scaling[:vertex_index]
    max_blending = max_blending[:vertex_index]

    # Batch all 5 downsampling ops into one F.interpolate call (5 kernel launches -> 1).
    # Layout: rendered_image(0:3), allmap[0:1]=expected_depth(3:4),
    #         allmap[1:2]=alpha(4:5), allmap[2:5]=normal(5:8), allmap[5:6]=depth_median(8:9)
    to_ds = torch.cat([rendered_image, allmap[0:6]], dim=0).unsqueeze(0)  # (1, 9, H, W)
    ds = F.interpolate(to_ds, size=(H_init, W_init), mode="area").squeeze(0)  # (9, H0, W0)
    rendered_image_small = ds[0:3]
    render_expected_depth = torch.nan_to_num(ds[3:4], 0, 0)
    render_alpha          = ds[4:5]
    render_normal         = ds[5:8]
    render_depth_median   = torch.nan_to_num(ds[8:9], 0, 0)

    # transform normal from view space to world space
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)

    # gets the id per pixel of the triangle influencing it (not downsampled)
    render_id = allmap[6:7]

    surf_depth = render_depth_median
    # assume the depth points form the 'surface' and generate pseudo surface normal for regularizations.
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * render_alpha.detach()

    V = vertices.shape[0]
    idx = triangles_indices[was_rendered > 0].reshape(-1).long()
    vertex_rendered = torch.zeros(V, device=triangles_indices.device, dtype=was_rendered.dtype)
    vertex_rendered.index_fill_(0, idx, 1)

    vertices_cam = transform_point_4x3(vertices, viewpoint_camera.world_view_transform)  # (V, 3)
    vertex_depths_pytorch = vertices_cam[:, 2]  # (V,)
    image_2D_pytorch = compute_image_2d_pytorch_exact(vertices, viewpoint_camera.full_proj_transform, W_init, H_init)
    if image_2D_pytorch.requires_grad:
        image_2D_pytorch.retain_grad()

    return {
        "render": rendered_image_small,
        "visibility_filter": radii > 0,
        "radii": radii,
        "scaling": scaling,
        "max_blending": max_blending,
        "vertex_depth_out": vertex_depths_pytorch,
        "image_2D": image_2D_pytorch,
        "vertex_rendered": vertex_rendered,
        "full_image": rendered_image,
        "render_normal_full": allmap[2:5],
        "triangle_was_rendered": was_rendered,
        "curr_visibility": (was_rendered > 0).long(),
        "rend_alpha": render_alpha,
        "rend_normal": render_normal,
        "rend_ids": render_id,
        "surf_depth": surf_depth,
        "surf_normal": surf_normal,
        "expected_depth": render_expected_depth,
        "depth_full": allmap[5:6],
        "render_depth_median": render_depth_median,
    }





 