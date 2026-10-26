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
import numpy as np
from mesh_splatting.utils.general_utils import inverse_sigmoid, get_expon_lr_func
from torch import nn
import os
from mesh_splatting.utils.system_utils import mkdir_p
from mesh_splatting.utils.sh_utils import RGB2SH
from mesh_splatting.utils.graphics_utils import BasicPointCloud
import math
from simple_knn._C import distCUDA2
import math
import rdel
import open3d as o3d

import rerun as rr
import scipy.ndimage


def random_rotation_matrices(num_matrices, device='cpu'):
    """
    Returns a tensor of shape (num_matrices, 3, 3) containing 
    random 3D rotation matrices.
    """
    axis = torch.randn(num_matrices, 3, device=device)
    axis = axis / axis.norm(dim=-1, keepdim=True)
    
    angles = 2.0 * math.pi * torch.rand(num_matrices, device=device)
    sin_t = torch.sin(angles)
    cos_t = torch.cos(angles)
    
    K = torch.zeros(num_matrices, 3, 3, device=device)
    ux, uy, uz = axis[:, 0], axis[:, 1], axis[:, 2]
    K[:, 0, 1] = -uz
    K[:, 0, 2] =  uy
    K[:, 1, 0] =  uz
    K[:, 1, 2] = -ux
    K[:, 2, 0] = -uy
    K[:, 2, 1] =  ux
    
    K2 = K.bmm(K)
    
    I = torch.eye(3, device=device).unsqueeze(0).expand(num_matrices, -1, -1)
    
    sin_term = sin_t.view(-1, 1, 1) * K
    cos_term = (1.0 - cos_t).view(-1, 1, 1) * K2
    
    return I + sin_term + cos_term


def normal_aligned_rotation_matrices(normals, device='cpu'):
    """
    Given unit normals [N, 3], compute rotation matrices [N, 3, 3]
    that map the local z-axis to each normal. The resulting x and y
    axes span the tangent plane, so an equilateral triangle placed
    in the XY plane will have its face normal aligned with the input normal.
    """
    N = normals.shape[0]
    n = torch.nn.functional.normalize(normals, dim=-1, eps=1e-8)

    ref = torch.zeros(N, 3, device=device)
    ref[:, 0] = 1.0
    close_to_x = (torch.abs(n[:, 0]) > 0.9)
    ref[close_to_x, 0] = 0.0
    ref[close_to_x, 1] = 1.0

    dot = (ref * n).sum(dim=-1, keepdim=True)
    u1 = ref - dot * n
    u1 = torch.nn.functional.normalize(u1, dim=-1, eps=1e-8)
    u2 = torch.linalg.cross(n, u1)
    u2 = torch.nn.functional.normalize(u2, dim=-1, eps=1e-8)

    R = torch.stack([u1, u2, n], dim=2)  # [N, 3, 3]
    return R


def fibonacci_directions(nb_points, device='cpu'):
    """
    Generate nb_points points on the unit sphere using a Fibonacci approach.
    Returns a tensor of shape (nb_points, 3).
    """
    directions = []
    for i in range(nb_points):
        z_coord = 1.0 - (2.0 * i / (nb_points - 1))
        z_coord = torch.tensor(z_coord, device=device)
        radius_xy = torch.sqrt(1.0 - z_coord * z_coord)
        theta = math.pi * (3.0 - math.sqrt(5.0)) * i
        
        x_unit = radius_xy * torch.cos(torch.tensor(theta, device=device))
        y_unit = radius_xy * torch.sin(torch.tensor(theta, device=device))
        
        directions.append(torch.stack([x_unit, y_unit, z_coord]))
    return torch.stack(directions, dim=0)


def generate_triangles_in_chunks(x, y, z, radii, nb_points=3, chunk_size=2000, normals=None):
    """
    Generate equilateral triangles around each center point.
    If normals [N, 3] are provided, triangles are oriented so their face
    normal aligns with the given surface normal (tangent-plane construction).
    Otherwise falls back to random rotations.
    """
    device = x.device
    num_centers = x.shape[0]

    if normals is not None:
        angles = [2 * math.pi * i / nb_points for i in range(nb_points)]
        base_dirs = torch.stack([
            torch.tensor([math.cos(a), math.sin(a), 0.0], device=device)
            for a in angles
        ], dim=0)
    else:
        base_dirs = fibonacci_directions(nb_points, device=device)

    out_points = torch.zeros(num_centers, nb_points, 3, device=device)

    for start_idx in range(0, num_centers, chunk_size):
        end_idx = min(start_idx + chunk_size, num_centers)

        x_chunk = x[start_idx:end_idx]
        y_chunk = y[start_idx:end_idx]
        z_chunk = z[start_idx:end_idx]
        r_chunk = radii[start_idx:end_idx]

        chunk_size_actual = x_chunk.shape[0]

        if normals is not None:
            R_chunk = normal_aligned_rotation_matrices(normals[start_idx:end_idx], device=device)
        else:
            R_chunk = random_rotation_matrices(chunk_size_actual, device=device)

        for i in range(nb_points):
            dir_i = base_dirs[i]

            dir_i_expanded = dir_i.view(1, 3, 1).expand(chunk_size_actual, -1, -1)

            rotated = R_chunk.bmm(dir_i_expanded)
            rotated = rotated.squeeze(-1)

            scaled = rotated * r_chunk.view(-1, 1)

            centers = torch.stack([x_chunk, y_chunk, z_chunk], dim=1)

            result_pts = centers + scaled

            out_points[start_idx:end_idx, i, :] = result_pts

    return out_points


class TriangleModel:

    def setup_functions(self):
        self.eps = 1e-6
        self.opacity_floor = 0.0
        self.opacity_activation = lambda x: self.opacity_floor + (1.0 - self.opacity_floor) * torch.sigmoid(x)
        # Matching inverse for any y in [m, 1): logit( (y - m)/(1 - m) )
        self.inverse_opacity_activation = lambda y: inverse_sigmoid(
            ((y.clamp(self.opacity_floor + self.eps, 1.0 - self.eps) - self.opacity_floor) /
            (1.0 - self.opacity_floor + self.eps))
        )

        self.exponential_activation = lambda x:math.exp(x)
        self.inverse_exponential_activation = lambda y: math.log(y)

    def __init__(self, sh_degree : int, use_sparse_adam : bool = False, config : dict = None):

        self._triangles = torch.empty(0) # can be deleted eventually

        self.size_probs_zero = 0.0
        self.size_probs_zero_image_space = 0.0
        self.vertices = torch.empty(0)
        self._triangle_indices = torch.empty(0)
        self.vertex_weight = torch.empty(0)

        self._sigma = 0
        self.active_sh_degree = 0
        self.max_sh_degree = max(0, int(sh_degree))
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self.optimizer = None
        self.image_size = 0
        self.pixel_count = 0
        self.importance_score = 0
        self.gradient_accum = 0  # Per-triangle accumulated gradient norm
        self.denom = 0  # Per-triangle view count (denominator for averaging)
        self.n_obs = 0  # Per-triangle co-visibility count (keyframes observing this triangle)
        self.unique_kfIDs = 0  # Per-triangle ID of the keyframe that created it
        self.add_percentage = 1.3
        # Range of triangle indices added in last densification (start, end), for Rerun highlight
        self._new_triangle_range = None

        self.scaling = 1

        self.laplacian_update_freq = 50  # Update every 50 iterations
        self.iteration_count = 0

        self.use_sparse_adam = use_sparse_adam

        self.config = config


        self.setup_functions()

    def create_pcd_from_image(self, cam_info, init=False, scale=2.0, depthmap=None, under_reconstructed_mask=None, rendered_depth=None):
        cam = cam_info
        image_ab =  (cam.original_image - cam.exposure_b) / torch.exp(cam.exposure_a) 
        image_ab = torch.clamp(image_ab, 0.0, 1.0)
        rgb_raw = (image_ab * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()

        if depthmap is not None:
            rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
            depth = o3d.geometry.Image(depthmap.astype(np.float32))
        else:
            depth_raw = cam.depth
            if depth_raw is None:
                depth_raw = np.empty((cam.image_height, cam.image_width))

            if self.config["Dataset"]["sensor_type"] == "monocular":
                depth_raw = (
                    np.ones_like(depth_raw)
                    + (np.random.randn(depth_raw.shape[0], depth_raw.shape[1]) - 0.5)
                    * 0.05
                ) * scale

            rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
            depth = o3d.geometry.Image(depth_raw.astype(np.float32))

        return self.create_pcd_from_image_and_depth(cam, rgb, depth, under_reconstructed_mask, init, rendered_depth=rendered_depth)


    def create_pcd_from_image_and_depth(self, cam, rgb, depth, under_reconstructed_mask=None, init=False, rendered_depth=None):
        if init:
            downsample_factor = self.config["Dataset"]["pcd_downsample_init"]
        else:
            downsample_factor = self.config["Dataset"]["pcd_downsample"]
        point_size = self.config["Dataset"]["point_size"]
        if "adaptive_pointsize" in self.config["Dataset"]:
            if self.config["Dataset"]["adaptive_pointsize"]:
                point_size = min(0.05, point_size * np.median(depth))

        if under_reconstructed_mask is not None and rendered_depth is not None:
            # Hybrid depth infill for under-reconstructed regions (only when enabled):
            # 1) Use rendered depth from current mesh where available
            # 2) Fall back to median depth + noise where rendered depth is also missing

            depth_np = np.asarray(depth).copy()

            if hasattr(under_reconstructed_mask, "cpu"):
                mask_np = under_reconstructed_mask.cpu().numpy().astype(bool)
            else:
                mask_np = np.asarray(under_reconstructed_mask).astype(bool)
            if mask_np.ndim == 3: mask_np = mask_np.squeeze()

            # Pixels that need filling: under-reconstructed AND no hardware depth
            pixels_to_fill = mask_np & (depth_np <= 0)

            if pixels_to_fill.any():
                # --- Stage 1: fill from rendered depth ---
                if rendered_depth is not None:
                    if hasattr(rendered_depth, "cpu"):
                        rend_np = rendered_depth.detach().squeeze().cpu().numpy()
                    else:
                        rend_np = np.asarray(rendered_depth).squeeze()
                    # Use rendered depth where it's valid (> 0)
                    fill_from_render = pixels_to_fill & (rend_np > 0)
                    if fill_from_render.any():
                        depth_np[fill_from_render] = rend_np[fill_from_render]
                        pixels_to_fill = pixels_to_fill & ~fill_from_render

                # --- Stage 2: remaining holes get median depth + noise ---
                if pixels_to_fill.any():
                    valid_depths = depth_np[depth_np > 0]
                    if valid_depths.size > 0:
                        median_d = np.median(valid_depths)
                    else:
                        median_d = 2.0  # last-resort default
                    n_fill = pixels_to_fill.sum()
                    depth_np[pixels_to_fill] = median_d + np.random.randn(n_fill) * 0.05 * median_d

                depth = o3d.geometry.Image(depth_np.astype(np.float32))


        
        max_depth = self.config["Training"].get("max_depth", 0)
        depth_trunc = max_depth if max_depth > 0 else 100.0
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb,
            depth,
            depth_scale=1.0,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )

        # random_raw_depth = (
        #     np.ones_like(depth.depth_np)
        #     + (np.random.randn(depth.depth_np.shape[0], depth.depth_np.shape[1]) - 0.5)
        #     * 0.05
        # ) * 2.0
        # random_depth = o3d.geometry.Image(random_raw_depth.astype(np.float32))


        W2C = cam.world_view_transform.T.detach().cpu().numpy().astype(np.float64)
        pcd_tmp = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(
                cam.image_width,
                cam.image_height,
                cam.fx,
                cam.fy,
                cam.cx,
                cam.cy,
            ),
            extrinsic=W2C,
            project_valid_depth_only=True,
        )

        if hasattr(cam, 'normal_raw') and cam.normal_raw is not None and cam.depth is not None:
            valid_normal = cam.normal_raw[cam.depth > 0].copy()
            valid_normal[np.abs(valid_normal) < 1e-9] = 1e-9
            R_c2w = W2C[0:3, 0:3].T
            normal_global = (R_c2w @ valid_normal.T).T
            pcd_tmp.normals = o3d.utility.Vector3dVector(
                normal_global.astype(np.float32)
            )

        pcd_tmp = pcd_tmp.random_down_sample(1.0 / downsample_factor)
        new_xyz = np.asarray(pcd_tmp.points)
        new_rgb = np.asarray(pcd_tmp.colors)
        new_normals = (
            np.asarray(pcd_tmp.normals).copy()
            if pcd_tmp.has_normals() and len(pcd_tmp.normals) > 0
            else np.zeros((new_xyz.shape[0], 3))
        )
        pcd = BasicPointCloud(
            points=new_xyz, colors=new_rgb, normals=new_normals
        )
        self.ply_input = pcd

        fused_point_cloud = torch.from_numpy(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.from_numpy(np.asarray(pcd.colors)).float().cuda())
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

   


   
        total_points = np.asarray(pcd.points)  # naming for clarity
        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(total_points)).float().cuda()),
            1e-7
        )  # [N]
        init_size = 2.23 * 0.7
        init_pt = self.config.get("opt_params", {}).get("init_point_size", 1.0)
        init_size = init_size * init_pt

        radii = init_size * torch.sqrt(dist2).unsqueeze(1)  # [N,1]

        # Clamp radii to avoid massive triangles from numerical instability / sparse neighbors.
        radii_flat = radii.squeeze()
        radii_flat = torch.nan_to_num(radii_flat, nan=0.0, posinf=1e6, neginf=1e-6)
        radii = radii_flat.unsqueeze(1)

        pct = self.config.get("Dataset", {}).get("radii_percentile_clip", 0.99)
        if pct < 1.0 and radii_flat.numel() > 0:
            cap = torch.quantile(radii_flat.float(), pct).clamp(min=1e-6)
            radii = torch.clamp(radii, max=cap)

        radii_max = self.config.get("Dataset", {}).get("radii_max", None)
        if radii_max is not None and radii_max > 0:
            radii = torch.clamp(radii, max=float(radii_max))
        

        # dist2 = (
        #     torch.clamp_min(
        #         distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
        #         0.0000001,
        #     )
        #     * point_size
        # )
        # scales = torch.log(torch.sqrt(dist2))[..., None]
        # if not self.isotropic:
        #     scales = scales.repeat(1, 3)

        # rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        # rots[:, 0] = 1
        opacities = (
            0.5
            * 
            torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )


        vert_weight = inverse_sigmoid(
            opacities * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float32, device="cuda")
        )
   

        # REMINDER: PCD here is not a torch tensor, should change back for mapping
        return pcd, features, vert_weight, radii

    # TODO : Create pcd from the random points.
    def create_pcd_from_random_points(self, scene_scale, init_num_pts=100000):
        points = scene_scale * (torch.rand((init_num_pts, 3)) * 2 -1) # N,3
        rgb = torch.rand((init_num_pts, 3)) # N,3
        
        fused_point_cloud = torch.from_numpy(np.asarray(points)).float().cuda()
        fused_color = RGB2SH(torch.from_numpy(np.asarray(rgb)).float().cuda())
        features = ( torch.zeros(fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2).float().cuda() ) # N,3,K
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        pcd = BasicPointCloud(points=np.asarray(points), colors=np.asarray(rgb), normals=None)

        total_points = np.asarray(points)
        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(total_points)).float().cuda()), 1e-7     # N,
        )
        init_size = 2.23 * 0.7
        raddi = init_size * torch.sqrt(dist2).unsqueeze(1) # N,1
        raddi_flat = raddi.squeeze()
        raddi_flat = torch.nan_to_num(raddi_flat, nan=0.0, posinf=1e6, neginf=1e-6)
        raddi = raddi_flat.unsqueeze(1) # N,1

        opacities = (
            0.5 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")  # N,1
        )

        vert_weight = inverse_sigmoid(
            opacities * torch.ones( (fused_point_cloud.shape[0], 1), dtype=torch.float32, device="cuda" )
        )  

        return pcd, features, vert_weight, raddi


    def extend_from_pcd(self, cam, pcd, radii, features, weights, under_reconstructed_mask=None, kf_id=0):
        # densification_postfix

        # For now generate a triangle per point in the point cloud - an equilateral triangle
        # of radius 0.01. These triangles can be optimised freely and then a restricted delaunay
        # can be run on top to create a mesh with the photometrically consistent free triangles.

        pcd_points = np.asarray(pcd.points)
        fused_point_cloud = torch.tensor(pcd_points, dtype=torch.float32, device="cuda")  # [N,3]
        pcd_normals = np.asarray(pcd.normals)
        normals_tensor = None
        if pcd_normals is not None and np.abs(pcd_normals).sum() > 0:
            normals_tensor = torch.tensor(pcd_normals, dtype=torch.float32, device="cuda")

        coverage_aware_spawn = self.config.get("Training", {}).get("coverage_aware_spawn", True)
        if under_reconstructed_mask is not None and coverage_aware_spawn:
            device = fused_point_cloud.device
            W2C = cam.world_view_transform.T.detach().to(device)  # (4, 4)
            fx, fy = cam.fx, cam.fy
            cx, cy = cam.cx, cam.cy
            H, W = int(cam.image_height), int(cam.image_width)

            if hasattr(under_reconstructed_mask, "cpu"):
                mask = under_reconstructed_mask.to(device).float()
            else:
                mask = torch.from_numpy(np.asarray(under_reconstructed_mask)).float().to(device)
            if mask.dim() == 3:
                mask = mask.squeeze(0)
            if mask.shape[0] != H or mask.shape[1] != W:
                mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(0).unsqueeze(0), size=(H, W), mode="nearest"
                ).squeeze(0).squeeze(0)

            ones = torch.ones(fused_point_cloud.shape[0], 1, device=device, dtype=fused_point_cloud.dtype)
            p_w = torch.cat([fused_point_cloud, ones], dim=1)  # (N, 4)
            p_c = (p_w @ W2C.T)[:, :3]  # (N, 3)

            z = p_c[:, 2]
            u = (fx * p_c[:, 0] / z + cx)
            v = (fy * p_c[:, 1] / z + cy)
            u_int = u.round().long().clamp(0, W - 1)
            v_int = v.round().long().clamp(0, H - 1)

            valid = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            keep = valid & (mask[v_int, u_int].bool())

            n_before = fused_point_cloud.shape[0]
            if not keep.any():
                print(f"[coverage-spawn] filtered all {n_before} points — skipping spawn")
                return
            fused_point_cloud = fused_point_cloud[keep]
            features = features[keep]
            weights = weights[keep]
            radii = radii[keep]
            if normals_tensor is not None:
                normals_tensor = normals_tensor[keep]
            print(f"[coverage-spawn] kept {keep.sum().item()}/{n_before} points in under-reconstructed regions")







        nb_points = 3
        x, y, z = fused_point_cloud[:, 0], fused_point_cloud[:, 1], fused_point_cloud[:, 2]

        points_per_triangle = generate_triangles_in_chunks(x, y, z, radii, nb_points=nb_points, normals=normals_tensor)  # [N,3,3]
        N = fused_point_cloud.shape[0] 
        n_vertices_new = N * nb_points

        _points = points_per_triangle.reshape(n_vertices_new, 3).contiguous()  # [3N,3]
        # New triangle indices are 0..3N-1 locally; offset by current vertex count for combined array
        vertex_base = self.vertices.shape[0]
        faces = torch.arange(n_vertices_new, device=_points.device, dtype=torch.int64).view(N, nb_points)
        faces = (faces + vertex_base).to(torch.int32)  # indices into combined vertex array

        # Features and weights are per-point (N); we have 3 vertices per point -> repeat to [3N, ...]
        feat_dc = features[:, :, 0:1].transpose(1, 2).contiguous()   # [N, 1, 3]
        feat_rest = features[:, :, 1:].transpose(1, 2).contiguous()  # [N, 15, 3]
        feat_dc = feat_dc.repeat_interleave(nb_points, dim=0)        # [3N, 1, 3]
        feat_rest = feat_rest.repeat_interleave(nb_points, dim=0)   # [3N, 15, 3]
        new_features_dc = nn.Parameter(feat_dc.requires_grad_(True))
        new_features_rest = nn.Parameter(feat_rest.requires_grad_(False))

        new_weights = nn.Parameter(weights.repeat_interleave(nb_points, dim=0).requires_grad_(True))  # [3N, 1]
        new_vertices = nn.Parameter(_points.requires_grad_(True))
        
        self.densification_postfix(new_vertices, new_weights, new_features_dc, new_features_rest, faces, kf_id=kf_id)


    def save_parameters(self, path):

        mkdir_p(path)

        point_cloud_state_dict = {}

        point_cloud_state_dict["triangles_points"] = self.vertices
        point_cloud_state_dict["_triangle_indices"] = self._triangle_indices
        point_cloud_state_dict["vertex_weight"] = self.vertex_weight
        point_cloud_state_dict["sigma"] = self._sigma
        point_cloud_state_dict["active_sh_degree"] = 0
        point_cloud_state_dict["features_dc"] = self._features_dc
        V = self._features_dc.shape[0]
        if self.max_sh_degree == 0 or self.active_sh_degree == 0:
            point_cloud_state_dict["features_rest"] = torch.zeros((V, 0, 3), dtype=torch.float32, device=self._features_dc.device)
        else:
            point_cloud_state_dict["features_rest"] = self._features_rest
        point_cloud_state_dict["importance_score"] = self.importance_score
        point_cloud_state_dict["image_size"] = self.image_size
        point_cloud_state_dict["pixel_count"] = self.pixel_count

        torch.save(point_cloud_state_dict, os.path.join(path, 'point_cloud_state_dict.pt'))


    def load_ply_file(self, path, device="cuda", active_sh_degree=3, assume_yup_to_zup=False, training_args=None):
        import trimesh
        """
        Load vertices, faces, and SH features from a PLY file into the current object.
        Fields not derivable from the PLY, like vertex_weight, sigma, importance_score, are ignored.
        """
        SH_C0 = 0.28209479177387814

        def _to_float01(colors_np):
            if colors_np is None:
                return None
            if colors_np.ndim != 2 or colors_np.shape[1] < 3:
                return None
            rgb = colors_np[:, :3]
            if rgb.dtype == np.uint8:
                return rgb.astype(np.float32) / 255.0
            return rgb.astype(np.float32)

        ply_path = path if path.lower().endswith(".ply") else os.path.join(path, "mesh.ply")
        if not os.path.isfile(ply_path):
            raise FileNotFoundError(f"PLY not found at '{ply_path}'")

        mesh = trimesh.load(ply_path, process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            # merge scene geometry into one mesh if needed
            try:
                mesh = trimesh.util.concatenate([g for g in mesh.dump()])
            except Exception as e:
                raise ValueError("Loaded PLY is not a Trimesh and could not be merged") from e

        verts_np = mesh.vertices.astype(np.float32).copy()
        if assume_yup_to_zup:
            # If your PLY is Y-up and you want Z-up internally, apply inverse of (x, y, z)->(x, z, -y)
            y = verts_np[:, 1].copy()
            z = verts_np[:, 2].copy()
            verts_np[:, 1] = -z
            verts_np[:, 2] = y

        faces_np = mesh.faces.astype(np.int32).copy() if mesh.faces is not None else np.empty((0, 3), np.int32)

        colors01 = None
        if getattr(mesh, "visual", None) is not None and hasattr(mesh.visual, "vertex_colors"):
            colors01 = _to_float01(mesh.visual.vertex_colors)

        V = int(verts_np.shape[0])
        verts = torch.from_numpy(verts_np).to(device=device, dtype=torch.float32).detach().clone().requires_grad_(True)
        faces = torch.from_numpy(faces_np).to(device=device, dtype=torch.int32)

        # features_dc: infer from colors if available, otherwise default to gray which maps to f_dc=0
        if colors01 is not None and colors01.shape[0] == V:
            f_dc_rgb = ((colors01 - 0.5) / SH_C0).clip(-4.0, 4.0).astype(np.float32)
        else:
            f_dc_rgb = np.zeros((V, 3), dtype=np.float32)
        features_dc = torch.from_numpy(f_dc_rgb).to(device=device, dtype=torch.float32).unsqueeze(1).detach().clone().requires_grad_(True)

        # features_rest: zeros with shape [V, (deg+1)^2 - 1, 3]
        deg = int(active_sh_degree)
        num_coeff_total = (deg + 1) ** 2
        num_rest = max(0, num_coeff_total - 1)
        features_rest = torch.zeros((V, num_rest, 3), device=device, dtype=torch.float32, requires_grad=False)

        # Assign to object
        self.vertices = verts.requires_grad_(True)
        self._triangle_indices = faces
        self.active_sh_degree = deg
        self._features_dc = features_dc.requires_grad_(True)
        self._features_rest = features_rest.requires_grad_(False)

        opacity_weight = 1.0
        self.opacity_floor = 0.9999
        vert_weight = inverse_sigmoid(opacity_weight * torch.ones((self.vertices.shape[0], 1), dtype=torch.float, device="cuda")) 
        self.vertex_weight = nn.Parameter(vert_weight.requires_grad_(True))
        self._sigma = self.inverse_exponential_activation(0.0001)

        # Optional, quick report
        print(f"Loaded PLY: {ply_path}")
        print(f"Vertices: {V}, Faces: {faces.shape[0]}, SH degree: {deg}, features_rest per color: {num_rest}")

        self.image_size = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device="cuda")
        self.importance_score = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device="cuda")
        self.pixel_count = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.int, device="cuda")
        self.gradient_accum = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device="cuda")
        self.denom = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device="cuda")
        self.n_obs = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.int, device="cuda")
        self.unique_kfIDs = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.int, device="cuda")

        if training_args != None:
            self.optimizer = None
            self.triangle_scheduler_args = None
            param_groups = [
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                {'params': [self.vertices], 'lr': training_args.lr_triangles_points_init, "name": "vertices"},
                {'params': [self.vertex_weight], 'lr': training_args.weight_lr, "name": "vertex_weight"}
            ]
            self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15) # torch.optim.SGD(param_groups, lr=0.0, momentum=0.0)

            self.triangle_scheduler_args = get_expon_lr_func(lr_init=training_args.lr_triangles_points_init,
                                                            lr_final=training_args.lr_triangles_points_init/100,
                                                            lr_delay_mult=training_args.position_lr_delay_mult,
                                                            max_steps=training_args.position_lr_max_steps)



    def load_parameters_for_slam_resume(self, path, device="cuda"):
        """Load checkpoint for SLAM resume. Restores geometry + buffers, skips optimizer (caller uses training_setup)."""
        state = torch.load(os.path.join(path, "point_cloud_state_dict.pt"), map_location=device)
        n_tri = state["_triangle_indices"].shape[0]
        self.vertices = state["triangles_points"].to(device).to(torch.float32).detach().clone().requires_grad_(True)
        self._triangle_indices = state["_triangle_indices"].to(device).to(torch.int32)
        self.vertex_weight = state["vertex_weight"].to(device).to(torch.float32).detach().clone().requires_grad_(True)
        sigma_val = state["sigma"]
        self._sigma = sigma_val.to(device) if hasattr(sigma_val, "to") else sigma_val
        self.active_sh_degree = min(state.get("active_sh_degree", 0), self.max_sh_degree)
        self._features_dc = state["features_dc"].to(device).to(torch.float32).detach().clone().requires_grad_(True)
        V = self._features_dc.shape[0]
        if self.max_sh_degree == 0 or self.active_sh_degree == 0:
            self._features_rest = torch.zeros((V, 0, 3), dtype=torch.float32, device=device).requires_grad_(False)
        else:
            fr = state.get("features_rest")
            self._features_rest = fr.to(device).to(torch.float32).detach().clone().requires_grad_(False) if fr is not None else torch.zeros((V, 0, 3), dtype=torch.float32, device=device).requires_grad_(False)
        self.importance_score = state["importance_score"].to(device).to(torch.float32).detach().clone()
        self.image_size = state["image_size"].to(device).to(torch.float32).clone()
        self.pixel_count = state["pixel_count"].to(device).to(torch.int32).clone()
        self.opacity_floor = 0.999
        self._triangle_indices = self._triangle_indices.to(torch.int32)
        self.gradient_accum = torch.zeros((n_tri,), dtype=torch.float32, device=device)
        self.denom = torch.zeros((n_tri,), dtype=torch.float32, device=device)
        self.n_obs = torch.zeros((n_tri,), dtype=torch.int32, device=device)
        self.unique_kfIDs = torch.zeros((n_tri,), dtype=torch.int32, device=device)
        print("SLAM resume: loaded triangles:", self._triangle_indices.shape, "vertices:", self.vertices.shape)

    def load_parameters(self, path, device="cuda", segment=False, ratio_threshold = 0.75):
        # 1. Load the dict you saved
        state = torch.load(os.path.join(path, "point_cloud_state_dict.pt"), map_location=device)

        # 2. Restore everything you put in there (one line each)
        self.vertices            = state["triangles_points"].to(device).to(torch.float32).detach().clone().requires_grad_(True)
        self._triangle_indices   = state["_triangle_indices"].to(device).to(torch.int32)
        self.vertex_weight       = state["vertex_weight"].to(device).to(torch.float32).detach().clone().requires_grad_(True)
        self._sigma              = state["sigma"]
        self.active_sh_degree    = min(state.get("active_sh_degree", 0), self.max_sh_degree)
        self._features_dc        = state["features_dc"].to(device).to(torch.float32).detach().clone().requires_grad_(True)
        V = self._features_dc.shape[0]
        if self.max_sh_degree == 0 or self.active_sh_degree == 0:
            self._features_rest = torch.zeros((V, 0, 3), dtype=torch.float32, device=device).requires_grad_(False)
        else:
            fr = state.get("features_rest")
            self._features_rest = fr.to(device).to(torch.float32).detach().clone().requires_grad_(False) if fr is not None else torch.zeros((V, 0, 3), dtype=torch.float32, device=device).requires_grad_(False)
        self.importance_score = state["importance_score"].to(device).to(torch.float32).detach().clone()

        print("triangles: ", self._triangle_indices.shape)
        print("vertices: ", self.vertices.shape)

        # For object extraction
        if segment:
            base = os.path.dirname(os.path.dirname(path))
            triangle_hits = torch.load(os.path.join(base, 'segmentation/triangle_hits_mask.pt'))
            triangle_hits_total = torch.load(os.path.join(base, 'segmentation/triangle_hits_total.pt'))

            min_hits = 1

            # Handle division by zero - triangles with no renders get ratio 0
            triangle_ratio = torch.zeros_like(triangle_hits, dtype=torch.float32)
            valid_mask = triangle_hits_total > 0
            triangle_ratio[valid_mask] = triangle_hits[valid_mask].float() / triangle_hits_total[valid_mask].float()

            # Create the keep mask: triangles must meet both ratio and minimum hits criteria
            keep_mask = (triangle_ratio >= ratio_threshold) & (triangle_hits >= min_hits)
            #keep_mask = ~keep_mask

            with torch.no_grad():
                self._triangle_indices = self._triangle_indices[keep_mask]

        ################################################################

        self.opacity_floor = 0.999
        self._triangle_indices = self._triangle_indices.to(torch.int32)

        param_groups = [
            {'params': [self._features_dc], 'lr': 0.0016, "name": "f_dc"},
            {'params': [self.vertices], 'lr': 0.0001, "name": "vertices"},
            {'params': [self.vertex_weight], 'lr': 0.0, "name": "vertex_weight"}
        ]
        self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15) # torch.optim.SGD(param_groups, lr=0.0, momentum=0.0)

        self.image_size = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device="cuda")
        self.importance_score = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device="cuda")
        self.pixel_count = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.int, device="cuda")
        self.gradient_accum = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device="cuda")
        self.denom = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device="cuda")
        self.n_obs = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.int, device="cuda")
        self.unique_kfIDs = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.int, device="cuda")


    def capture(self):
        return (
            self.active_sh_degree,
            self._features_dc,
            self._features_rest,
            self.optimizer.state_dict(),
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._features_dc, 
        self._features_rest,
        opt_dict) = model_args
        self.training_setup(training_args)
        self.optimizer.load_state_dict(opt_dict)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    @property 
    def get_triangles_points(self): 
        return self._triangles

    @property
    def get_triangle_indices(self):
        return self._triangle_indices

    @property
    def get_vertices(self):
        return self.vertices

    @property
    def get_sigma(self):
        return self.exponential_activation(self._sigma)

    @property
    def get_features(self):
        # main features
        features_dc   = self._features_dc
        features_rest = self._features_rest
        feats_main = torch.cat((features_dc, features_rest), dim=1)  # [Vmain, F, 3]
        return feats_main
      
       
    @property
    def get_vertex_weight(self):
        main_w = self.opacity_activation(self.vertex_weight)
        return main_w

    def oneupSHdegree(self):
            if self.active_sh_degree < self.max_sh_degree:
                self.active_sh_degree += 1


    def create_from_pcd(self, pcd : BasicPointCloud, radii, opacity : float, set_sigma : float):

        init_size = 2.23
        nb_points = 3  # 3 verts per triangle

        # --- Load PCD ---
        pcd_points = np.asarray(pcd.points)            # [N,3] (CPU, np)
        pcd_colors = np.asarray(pcd.colors)            # [N,3] (CPU, np)

        if radii == None:
            dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd_points)).float().cuda()), 1e-7) # N,
            radii = init_size * torch.sqrt(dist2).unsqueeze(1) # N,1
            radii_flat = radii.squeeze()
            radii_flat = torch.nan_to_num(radii_flat, nan=0.0, posinf=1e6, neginf=1e-6)
            radii = radii_flat.unsqueeze(1) # N,1  

        fused_point_cloud = torch.tensor(pcd_points, dtype=torch.float32, device="cuda")  # [N,3]
        fused_color_rgb   = torch.tensor(pcd_colors, dtype=torch.float32, device="cuda")  # [N,3]
        fused_color_sh    = RGB2SH(fused_color_rgb)                                       # [N,3]

        # SH features per *vertex* will be built after expansion to 3N
        # but we keep your original features layout
        base_feat_dim = (self.max_sh_degree + 1) ** 2

        # --- Scene size (same logic) ---
        x, y, z = fused_point_cloud[:, 0], fused_point_cloud[:, 1], fused_point_cloud[:, 2]
        width  = x.max() - x.min()
        height = y.max() - y.min()
        depth  = z.max() - z.min()
        scene_size = torch.max(torch.stack([width, height, depth]))
        if scene_size.item() > 300:
            print("Scene is large, we increase the threshold")
            self.large = True

        

        # --- Per-point radii using your GPU NN distance (distCUDA2 returns squared NN dist) ---
        # total_points = pcd_points  # naming for clarity
        # dist2 = torch.clamp_min(
        #     distCUDA2(torch.from_numpy(np.asarray(total_points)).float().cuda()),
        #     1e-7
        # )  # [N]

        # --- Create 1 independent triangle per point (returns [N,3,3]) ---
        pcd_normals = np.asarray(pcd.normals)
        normals_tensor = None
        #================================================================================================
        # TODO : commented code as the pcd.normals are None and not applicable 
        # if pcd_normals is not None and np.abs(pcd_normals).sum() > 0:
        #     normals_tensor = torch.tensor(pcd_normals, dtype=torch.float32, device="cuda")
        #================================================================================================
        points_per_triangle = generate_triangles_in_chunks(x, y, z, radii, nb_points=nb_points, normals=normals_tensor)  # [N,3,3]

        # --- Flatten to vertex buffer and build triangle indices ---
        N = fused_point_cloud.shape[0]   # number of triangles
        _points = points_per_triangle.reshape(N * nb_points, 3).contiguous()  # [3N,3]
        faces = torch.arange(N * nb_points, device=_points.device, dtype=torch.int64).view(N, nb_points)  # [N,3]
        faces = faces.to(torch.int32)  # match your Delaunay path dtype

        # --- Per-vertex SH features (color from source point, repeated to its 3 verts) ---
        per_vertex_color_sh = fused_color_sh.repeat_interleave(nb_points, dim=0)  # [3N,3]
        features = torch.zeros((per_vertex_color_sh.shape[0], 3, base_feat_dim),
                            dtype=torch.float32, device="cuda")                 # [3N,3,F]
        features[:, :3, 0] = per_vertex_color_sh
        # features[:, 3:, 1:] stays zero (no higher SH bands initialized)

        # --- Parameters aligned with your Delaunay initializer ---
        self.vertices = nn.Parameter(_points.requires_grad_(True))                 # [3N,3]
        self._triangle_indices = faces                                            # [N,3] int32

        if opacity == None:
            opacity = (
            0.5 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")  # N,1
            )  

        vert_weight = inverse_sigmoid(
            opacity * torch.ones((self.vertices.shape[0], 1), dtype=torch.float32, device="cuda")
        )
        self.vertex_weight = nn.Parameter(vert_weight.requires_grad_(True))        # [3N,1]

        # solid triangles
        self._sigma = self.inverse_exponential_activation(set_sigma)

        # SH feature tensors: DC (RGB) only optimized; rest kept at zero (no higher-order SH)
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(False))

        # Per-triangle buffers (match Delaunay sizing by triangles count)
        self.image_size = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float32, device="cuda")
        self.importance_score = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float32, device="cuda")
        self.pixel_count = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.int, device="cuda")
        self.n_obs = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.int, device="cuda")
        self.unique_kfIDs = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.int, device="cuda")

    def training_setup(self, training_args, lr_features, weight_lr, lr_triangles_init):
      
        l = [
            {'params': [self._features_dc], 'lr': lr_features, "name": "f_dc"},
            {'params': [self.vertices], 'lr': lr_triangles_init, "name": "vertices"},
            {'params': [self.vertex_weight], 'lr': weight_lr, "name": "vertex_weight"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.triangle_scheduler_args = get_expon_lr_func(lr_init=lr_triangles_init,
                                                        lr_final=lr_triangles_init/100,
                                                        lr_delay_mult=training_args.position_lr_delay_mult,
                                                        max_steps=training_args.position_lr_max_steps)

    def set_sigma(self, sigma):
        self._sigma = self.inverse_exponential_activation(sigma)

    def run_prune(
        self,
        visibility_in_frustum,
        prune_triangles_threshold: float,
        prune_size: float,
        frustum_union=None,
        use_importance_mask: bool = False,
        prune_importance_threshold: float = 0.01,
        vertex_weight_threshold: float = 0.8,
        prune_coviz: int = 0,
        num_keyframes: int = 0,
        kf_window_ids: list = None,
    ):
        """Build triangle keep mask from opacity/importance/size/co-visibility, prune triangles, then prune unused vertices."""
        triangle_vertex_weights = self.opacity_activation(
            self.vertex_weight[self._triangle_indices]
        )
        mean_weights = triangle_vertex_weights.mean(dim=1)

        mask_opacity = (mean_weights <= prune_triangles_threshold).squeeze()
        mask_importance = (self.importance_score <= prune_importance_threshold).squeeze()
        mask_size = (self.image_size > prune_size).squeeze()
        mask_visibility = (visibility_in_frustum > 0).squeeze()

        delete_mask = mask_opacity | mask_size
        if use_importance_mask:
            delete_mask = delete_mask | mask_importance


        # if kf_window_ids is not None and len(kf_window_ids) > 0:
        #     sorted_window = sorted(kf_window_ids, reverse=True)
        #     # Protect triangles spawned in the newest 2 keyframes (index 0 and 1)
        #     grace_cutoff = sorted_window[min(1, len(sorted_window) - 1)] 
        #     is_in_grace_period = (self.unique_kfIDs >= grace_cutoff).squeeze()
            
        #     # Remove grace period triangles from the execution block
        #     delete_mask = delete_mask & ~is_in_grace_period
        # else:
        #     is_in_grace_period = torch.zeros_like(delete_mask, dtype=torch.bool)

        # if (prune_coviz > 0
        #     and num_keyframes >= prune_coviz
        #     and isinstance(self.n_obs, torch.Tensor) and self.n_obs.numel() > 0
        #     and isinstance(self.unique_kfIDs, torch.Tensor) and self.unique_kfIDs.numel() > 0
        # ):
        #     mask_low_coviz = (self.n_obs < prune_coviz).squeeze()
        #     if kf_window_ids is not None and len(kf_window_ids) >= prune_coviz:
        #         sorted_window = sorted(kf_window_ids, reverse=True)
        #         recency_cutoff = sorted_window[min(prune_coviz - 1, len(sorted_window) - 1)]
        #         mask_young = (self.unique_kfIDs >= recency_cutoff).squeeze()
        #         mask_coviz = mask_low_coviz & mask_young
        #     else:
        #         mask_coviz = mask_low_coviz
        #     delete_mask = delete_mask | mask_coviz

        # Triangles whose min vertex opacity is below the CUDA stopping_influence
        # threshold (0.01) are permanently invisible: preprocessCUDA returns early for
        # them so they never get radii, are never rasterised, and can never accumulate
        # gradient or importance score. Prune them unconditionally — no visibility gate.
        CUDA_STOPPING_INFLUENCE = 0.01
        mask_opacity_dead = mean_weights.squeeze() < CUDA_STOPPING_INFLUENCE

        # All other bad triangles (opacity/importance/size) are gated on visibility:
        # only prune if they were actually rendered (had a chance to optimise).
        delete_mask = (delete_mask & mask_visibility) | mask_opacity_dead
        keep_mask = ~delete_mask

        n_pruned = delete_mask.sum().item()
        n_dead = mask_opacity_dead.sum().item()
        n_bad_visible = (delete_mask & mask_visibility & ~mask_opacity_dead).sum().item()
        n_low_importance = (mask_importance & mask_visibility).sum().item() if use_importance_mask else 0
        print(f"[prune] opacity_dead={n_dead}  bad_visible={n_bad_visible}  low_importance={n_low_importance}  "
              f"total_pruned={n_pruned}  kept={keep_mask.sum().item()}/{delete_mask.numel()}")
        self.prune_triangles(keep_mask)

        device = self.vertices.device
        used_vertex_mask = torch.zeros(
            self.vertices.shape[0], dtype=torch.bool, device=device
        )
        if self._triangle_indices.numel() > 0:
            flat_indices = self._triangle_indices.flatten()
            used_vertex_mask[flat_indices] = True
        weight_mask = (self.get_vertex_weight.squeeze() >= vertex_weight_threshold)
        mask_out = self.vertices.shape[0]
        vertex_mask = weight_mask[:mask_out] | used_vertex_mask
        self._prune_vertices(vertex_mask)

        return n_pruned, keep_mask

    def prune_edge_on_triangles(self, keyframe_cameras, cos_threshold=0.1, min_keyframes=3):
        """Prune triangles that are consistently edge-on (nearly invisible) across keyframes.

        A triangle is edge-on when |dot(normal, view_dir)| < cos_threshold.
        If edge-on in all keyframes where it's geometrically visible (and seen by >= min_keyframes),
        it gets pruned.

        Returns number of triangles pruned.
        """
        if self._triangle_indices.numel() == 0 or len(keyframe_cameras) < min_keyframes:
            return 0, None

        device = self.vertices.device
        tri_idx = self._triangle_indices.long()
        v0 = self.vertices[tri_idx[:, 0]]
        v1 = self.vertices[tri_idx[:, 1]]
        v2 = self.vertices[tri_idx[:, 2]]

        # Triangle normals (not necessarily unit length)
        normals = torch.cross(v1 - v0, v2 - v0, dim=-1)  # (T, 3)
        normal_len = normals.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        normals = normals / normal_len  # unit normals

        # Triangle centroids
        centroids = (v0 + v1 + v2) / 3.0  # (T, 3)

        n_tri = tri_idx.shape[0]
        edge_on_count = torch.zeros(n_tri, device=device, dtype=torch.int32)
        visible_count = torch.zeros(n_tri, device=device, dtype=torch.int32)

        for cam in keyframe_cameras:
            # Camera center in world coords
            R = cam.R if hasattr(cam, 'R') else None
            T = cam.T if hasattr(cam, 'T') else None
            if R is None or T is None:
                continue
            R_t = R.to(device).float()
            T_t = T.to(device).float()
            # Camera center: C = -R^T @ T
            cam_center = -R_t.T @ T_t  # (3,)

            # View direction from camera to each triangle centroid
            view_dirs = centroids - cam_center.unsqueeze(0)  # (T, 3)
            view_len = view_dirs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            view_dirs = view_dirs / view_len

            # Check if triangles are in front of camera (z > 0 in camera frame)
            # p_cam = R @ p_world + T
            p_cam_z = (R_t[2, :].unsqueeze(0) * centroids).sum(-1) + T_t[2]
            in_front = p_cam_z > 0.1

            cos_angle = (normals * view_dirs).sum(dim=-1).abs()  # (T,)
            is_edge_on = cos_angle < cos_threshold

            visible_count += in_front.int()
            edge_on_count += (in_front & is_edge_on).int()

        # Triangle is problematic if: seen by enough keyframes AND edge-on in all of them
        seen_enough = visible_count >= min_keyframes
        always_edge_on = (edge_on_count == visible_count) & seen_enough

        if not always_edge_on.any():
            return 0, None

        keep_mask = ~always_edge_on
        n_pruned = always_edge_on.sum().item()
        print(f"[prune-edge-on] pruned {n_pruned} triangles (cos<{cos_threshold} in all {min_keyframes}+ views)")
        self.prune_triangles(keep_mask)

        # Don't prune unused vertices here — vertex count must stay consistent
        # with per-vertex gradient accumulators (avg_grad, pix, etc.) used by
        # add_new_gs. Orphaned vertices get cleaned up in the next run_prune.

        return n_pruned, keep_mask

    def reset_vertex_weight_nonvisible(self, visibility_mask, reset_value=0.4):
        """Reset vertex weights for non-visible vertices to a low value.

        Like MonoGS reset_opacity_nonvisible: vertices that aren't seen by any
        recent keyframe get their weight reduced.  If the optimizer can't push
        them back up (because no view benefits from them), they'll be pruned.
        """
        # visibility_mask is per-triangle; map to per-vertex
        visible_verts = torch.zeros(self.vertices.shape[0], dtype=torch.bool,
                                    device=self.vertices.device)
        if visibility_mask.any():
            visible_tri_indices = self._triangle_indices[visibility_mask]
            visible_verts[visible_tri_indices.flatten()] = True

        nonvisible = ~visible_verts
        if not nonvisible.any():
            return 0

        # Convert reset_value to raw parameter space (inverse of opacity_activation)
        raw_reset = self.inverse_opacity_activation(
            torch.tensor(reset_value, device=self.vertices.device)
        )
        with torch.no_grad():
            # Only reduce weights — never increase them via reset
            should_reset = nonvisible & (self.vertex_weight.squeeze() > raw_reset)
            n_reset = should_reset.sum().item()
            if n_reset > 0:
                self.vertex_weight.data[should_reset] = raw_reset
        return n_reset

    def update_learning_rate_delaunay(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "vertices":
                    lr = self.triangle_scheduler_args(iteration)
                    param_group['lr'] = lr
                    return lr
    
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "vertices":
                    if iteration < 1000:
                        lr = self.triangle_scheduler_args(iteration)
                    else:
                        lr = self.triangle_scheduler_args(iteration)
                    param_group['lr'] = lr
                    return lr

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] not in tensors_dict:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
    

        return optimizable_tensors
    

    def densification_postfix(self, new_vertices, new_vertex_weight, new_features_dc, new_features_rest, new_triangles, kf_id=0, inherit_kfIDs=None):
        # Create dictionary of new tensors to append
        d = {
            "vertices": new_vertices,
            "vertex_weight": new_vertex_weight,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
        }
        
        # Append new tensors to optimizer
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        
        # Update model parameters
        self.vertices = optimizable_tensors["vertices"]
        self.vertex_weight = optimizable_tensors["vertex_weight"]
        self._features_dc = optimizable_tensors["f_dc"]
        if "f_rest" in optimizable_tensors:
            self._features_rest = optimizable_tensors["f_rest"]
        else:
            self._features_rest = torch.cat([self._features_rest.detach(), new_features_rest.detach()], dim=0)
        
        # Update triangle indices
        old_tri_count = self._triangle_indices.shape[0]
        n_new = new_triangles.shape[0]
        self._triangle_indices = torch.cat([
            self._triangle_indices, 
            new_triangles
        ], dim=0)
        # Track new triangles for Rerun highlight (cleared after logging)
        self._new_triangle_range = (old_tri_count, self._triangle_indices.shape[0])

        self.image_size = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device="cuda")
        self.importance_score = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device="cuda")
        self.pixel_count = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.int, device="cuda")
        self.gradient_accum = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device="cuda")
        self.denom = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device="cuda")
        self.n_obs = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.int, device="cuda")

        old_kfIDs = self.unique_kfIDs if isinstance(self.unique_kfIDs, torch.Tensor) and self.unique_kfIDs.numel() > 0 else torch.empty(0, dtype=torch.int, device="cuda")
        if inherit_kfIDs is not None:
            new_kfIDs = inherit_kfIDs.to(dtype=torch.int, device="cuda")
        else:
            new_kfIDs = torch.full((n_new,), kf_id, dtype=torch.int, device="cuda")
        self.unique_kfIDs = torch.cat([old_kfIDs, new_kfIDs], dim=0)


    def _update_params_fast(self, selected_indices):
        selected_indices = torch.unique(selected_indices)
        selected_triangles_indices = self._triangle_indices[selected_indices]  # [S, 3]
        S = selected_triangles_indices.shape[0]
        
        edges = torch.cat([
            selected_triangles_indices[:, [0, 1]],
            selected_triangles_indices[:, [0, 2]],
            selected_triangles_indices[:, [1, 2]]
        ], dim=0) 
        edges_sorted, _ = torch.sort(edges, dim=1)
        
        unique_edges_tensor, unique_indices = torch.unique(
            edges_sorted, return_inverse=True, dim=0
        )  
        M = unique_edges_tensor.shape[0]
        
        v0 = self.vertices[unique_edges_tensor[:, 0]]
        v1 = self.vertices[unique_edges_tensor[:, 1]]
        new_vertices = (v0 + v1) / 2.0
        
        new_vertex_base = self.vertices.shape[0]
        
        unique_edges_cpu = unique_edges_tensor.cpu()
        edge_to_midpoint = {}
        for i in range(M):
            edge_tuple = (unique_edges_cpu[i, 0].item(), unique_edges_cpu[i, 1].item())
            edge_to_midpoint[edge_tuple] = new_vertex_base + i

        new_triangles_list = []
        selected_triangles_cpu = selected_triangles_indices.cpu()
        
        for i in range(S):
            tri = selected_triangles_cpu[i]
            a, b, c = tri[0].item(), tri[1].item(), tri[2].item()
            
            ab = (min(a, b), max(a, b))
            ac = (min(a, c), max(a, c))
            bc = (min(b, c), max(b, c))
            
            m_ab = edge_to_midpoint[ab]
            m_ac = edge_to_midpoint[ac]
            m_bc = edge_to_midpoint[bc]

            new_triangles_list.append([a, m_ab, m_ac])
            new_triangles_list.append([b, m_ab, m_bc])
            new_triangles_list.append([c, m_ac, m_bc])
            new_triangles_list.append([m_ab, m_bc, m_ac])
        
        subdivided_triangles = torch.tensor(
            new_triangles_list, 
            dtype=torch.int32, 
            device=self._triangle_indices.device
        )

        u, v = unique_edges_tensor[:, 0], unique_edges_tensor[:, 1]
        new_features_dc = (self._features_dc[u] + self._features_dc[v]) / 2.0
        new_features_rest = (self._features_rest[u] + self._features_rest[v]) / 2.0
        
        opacity_u = self.opacity_activation(self.vertex_weight[u])
        opacity_v = self.opacity_activation(self.vertex_weight[v])
        avg_opacity = (opacity_u + opacity_v) / 2.0
        avg_opacity = torch.clamp(avg_opacity, self.opacity_floor + self.eps, 1 - self.eps)
        new_vertex_weight = self.inverse_opacity_activation(avg_opacity)

        new_triangles = subdivided_triangles
        
        return (
            new_vertices,
            new_vertex_weight,
            new_features_dc,
            new_features_rest,
            new_triangles
        )


    def _prune_vertex_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ["vertices", "vertex_weight", "f_dc", "f_rest"]:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    # Prune optimizer state
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                    
                    del self.optimizer.state[group['params'][0]]
                    # Update parameter
                    group['params'][0] = nn.Parameter(group['params'][0][mask].requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state
                    optimizable_tensors[group["name"]] = group['params'][0]
                else:
                    group['params'][0] = nn.Parameter(group['params'][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group['params'][0]
        
        # Update model parameters
        for name, tensor in optimizable_tensors.items():
            if name == "vertices":
                self.vertices = tensor
            elif name == "vertex_weight":
                self.vertex_weight = tensor
            elif name == "f_dc":
                self._features_dc = tensor
            elif name == "f_rest":
                self._features_rest = tensor

        if "f_rest" not in optimizable_tensors and isinstance(self._features_rest, torch.Tensor) and self._features_rest.shape[0] == mask.shape[0]:
            self._features_rest = self._features_rest[mask].detach()

    def _prune_vertices(self, vertex_mask: torch.Tensor):
        device = vertex_mask.device
        oldV = vertex_mask.numel()

        # Create mapping from old vertex IDs to new IDs (-1 for removed vertices)
        new_id = torch.full((oldV,), -1, dtype=torch.long, device=device)
        kept = torch.nonzero(vertex_mask, as_tuple=True)[0]
        new_id[kept] = torch.arange(kept.numel(), device=device, dtype=torch.long)

        # Remap triangle indices and drop triangles with removed vertices
        if self._triangle_indices.numel() > 0:
            remapped = new_id[self._triangle_indices.long()]
            valid_tris = (remapped >= 0).all(dim=1)
            remapped = remapped[valid_tris]
            self._triangle_indices = remapped.to(torch.int32).contiguous()

            if isinstance(self.image_size, torch.Tensor) and self.image_size.numel() > 0:
                self.image_size = self.image_size[valid_tris]
            if isinstance(self.importance_score, torch.Tensor) and self.importance_score.numel() > 0:
                self.importance_score = self.importance_score[valid_tris]
            if isinstance(self.pixel_count, torch.Tensor) and self.pixel_count.numel() > 0:
                self.pixel_count = self.pixel_count[valid_tris]
            if isinstance(self.gradient_accum, torch.Tensor) and self.gradient_accum.numel() > 0:
                self.gradient_accum = self.gradient_accum[valid_tris]
            if isinstance(self.n_obs, torch.Tensor) and self.n_obs.numel() > 0:
                self.n_obs = self.n_obs[valid_tris]
            if isinstance(self.unique_kfIDs, torch.Tensor) and self.unique_kfIDs.numel() > 0:
                self.unique_kfIDs = self.unique_kfIDs[valid_tris]
            if isinstance(self.denom, torch.Tensor) and self.denom.numel() > 0:
                self.denom = self.denom[valid_tris]
            

        # Prune vertex-related parameters using the initial mask
        self._prune_vertex_optimizer(vertex_mask)

        # After initial pruning, check for unreferenced vertices
        current_vertex_count = self.vertices.shape[0]
        if current_vertex_count > 0:
            # Identify vertices still referenced by triangles
            if self._triangle_indices.numel() > 0:
                referenced_vertices = torch.unique(self._triangle_indices)
                mask_referenced = torch.zeros(current_vertex_count, dtype=torch.bool, device=device)
                mask_referenced[referenced_vertices] = True
            else:
                mask_referenced = torch.zeros(current_vertex_count, dtype=torch.bool, device=device)

            # Remove unreferenced vertices
            if not mask_referenced.all():
                # Prune vertex parameters
                self._prune_vertex_optimizer(mask_referenced)

                # Remap triangle indices if triangles exist
                if self._triangle_indices.numel() > 0:
                    new_id2 = torch.full((current_vertex_count,), -1, dtype=torch.long, device=device)
                    kept2 = torch.nonzero(mask_referenced, as_tuple=True)[0]
                    new_id2[kept2] = torch.arange(kept2.numel(), device=device, dtype=torch.long)
                    self._triangle_indices = new_id2[self._triangle_indices.long()].to(torch.int32).contiguous()

    def prune_triangles(self, mask):
        self._triangle_indices = self._triangle_indices[mask]
        self._triangle_indices = self._triangle_indices.to(torch.int32)
        self.image_size = self.image_size[mask]
        self.importance_score = self.importance_score[mask]
        self.pixel_count = self.pixel_count[mask]
        if isinstance(self.gradient_accum, torch.Tensor) and self.gradient_accum.numel() > 0:
            self.gradient_accum = self.gradient_accum[mask]
        if isinstance(self.denom, torch.Tensor) and self.denom.numel() > 0:
            self.denom = self.denom[mask]
        if isinstance(self.n_obs, torch.Tensor) and self.n_obs.numel() > 0:
            self.n_obs = self.n_obs[mask]
        if isinstance(self.unique_kfIDs, torch.Tensor) and self.unique_kfIDs.numel() > 0:
            self.unique_kfIDs = self.unique_kfIDs[mask]

    def _sample_alives(self, probs, num, alive_indices=None):
        torch.manual_seed(1)  # always same "random" indices
        probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
        sampled_idxs = torch.multinomial(probs, num, replacement=False)
        if alive_indices is not None:
            sampled_idxs = alive_indices[sampled_idxs]
        return sampled_idxs

    def add_densification_stats_triangles(self, visibility_filter, image_2D=None):
        """
        Accumulate per-triangle gradient importance for densification.

        When image_2D is provided (and has .grad), uses the MonoGS-style 2D
        viewspace position gradient: ‖∂L/∂(screen_x, screen_y)‖ averaged over
        a triangle's 3 vertices.  Falls back to 3D vertex gradient norms.
        """
        num_triangles = self._triangle_indices.shape[0]
        device = self._triangle_indices.device

        if not isinstance(self.gradient_accum, torch.Tensor) or self.gradient_accum.shape[0] != num_triangles:
            self.gradient_accum = torch.zeros(num_triangles, dtype=torch.float, device=device)
        if not isinstance(self.denom, torch.Tensor) or self.denom.shape[0] != num_triangles:
            self.denom = torch.zeros(num_triangles, dtype=torch.float, device=device)

        if image_2D is not None and image_2D.grad is not None:
            grad_2d = image_2D.grad  # (V, 2)
            grad_norm = grad_2d.norm(dim=-1)  # (V,)
            tri_grads = grad_norm[self._triangle_indices]  # (T, 3)
            grad_tri = tri_grads.mean(dim=1)  # (T,)
        elif self.vertices.grad is not None:
            tri_vertex_grads = self.vertices.grad[self._triangle_indices]  # (T, 3, 3)
            grad_tri = tri_vertex_grads.norm(dim=-1).mean(dim=1)  # (T,)
        else:
            return

        if visibility_filter is not None:
            update_mask = visibility_filter
        else:
            update_mask = torch.ones(num_triangles, dtype=torch.bool, device=device)

        self.gradient_accum[update_mask] += grad_tri[update_mask]
        self.denom[update_mask] += 1.0

    def densify_clone_and_split(self, cap_max, visibility=None,
                                 triangle_pixel_counts=None):
        """
        MonoGS-style clone-and-split densification using 2D viewspace gradients.

        Clone: high-grad, small triangles → duplicate in place (same vertices).
        Split: high-grad, large triangles → subdivide 1→4 (Loop subdivision).

        Size criterion: pixel coverage relative to image area (like MonoGS uses
        Gaussian scale relative to scene extent).
        """
        num_tri = self._triangle_indices.shape[0]
        device = self._triangle_indices.device
        training_cfg = self.config.get("Training", {}) if self.config else {}

        grad_threshold = training_cfg.get("densify_grad_threshold_2d", 0.0002)
        blur_fraction = training_cfg.get("densify_blur_fraction", 0.0002)
        ds_cfg = self.config.get("Dataset", {}) if self.config else {}
        H = ds_cfg.get("height", 480)
        W = ds_cfg.get("width", 640)
        size_threshold_px = blur_fraction * H * W

        has_stats = (
            isinstance(self.gradient_accum, torch.Tensor)
            and self.gradient_accum.shape[0] == num_tri
            and isinstance(self.denom, torch.Tensor)
            and self.denom.shape[0] == num_tri
        )
        if has_stats:
            avg_grad = self.gradient_accum / self.denom.float().clamp(min=1)
            avg_grad[torch.isnan(avg_grad)] = 0.0
            seen = self.denom > 0
        else:
            return torch.ones(num_tri, dtype=torch.bool, device=device)

        vis = visibility
        if vis is None:
            vis = torch.ones(num_tri, dtype=torch.bool, device=device)

        if triangle_pixel_counts is not None:
            pix = triangle_pixel_counts.squeeze().float()
        else:
            pix = torch.zeros(num_tri, dtype=torch.float32, device=device)

        high_grad = (avg_grad >= grad_threshold) & seen & vis
        is_large = pix >= size_threshold_px
        is_small = ~is_large

        split_mask = high_grad & is_large
        clone_mask = high_grad & is_small

        budget = training_cfg.get("densify_budget", 500)
        clone_enabled = training_cfg.get("densify_clone", True)

        # --- Clone: duplicate small high-grad triangles ---
        n_cloned = 0
        if not clone_enabled:
            clone_idx = torch.empty(0, dtype=torch.long, device=device)
        else:
            clone_idx = clone_mask.nonzero(as_tuple=False).squeeze(1)
            if clone_idx.numel() > budget:
                _, top_k = torch.topk(avg_grad[clone_idx], budget)
                clone_idx = clone_idx[top_k]

        if clone_idx.numel() > 0:
            sel_tris = self._triangle_indices[clone_idx]  # (C, 3)
            flat = sel_tris.reshape(-1)
            new_verts = self.vertices[flat].detach().clone().reshape(-1, 3)
            new_weights = self.vertex_weight[flat].detach().clone().reshape(-1, 1)
            new_fdc = self._features_dc[flat].detach().clone()
            new_frest = self._features_rest[flat].detach().clone()
            v_base = self.vertices.shape[0]
            new_tri_indices = torch.arange(
                3 * clone_idx.numel(), device=device, dtype=torch.int32
            ).reshape(-1, 3) + v_base

            clone_kfIDs = None
            if isinstance(self.unique_kfIDs, torch.Tensor) and self.unique_kfIDs.numel() > 0:
                clone_kfIDs = self.unique_kfIDs[clone_idx]

            self.densification_postfix(
                new_verts, new_weights, new_fdc, new_frest,
                new_tri_indices, inherit_kfIDs=clone_kfIDs,
            )
            n_cloned = clone_idx.numel()

        # --- Split: subdivide large high-grad triangles ---
        split_idx = split_mask.nonzero(as_tuple=False).squeeze(1)
        if split_idx.numel() > budget:
            _, top_k = torch.topk(avg_grad[split_idx], budget)
            split_idx = split_idx[top_k]

        n_split = 0
        if split_idx.numel() > 0:
            new_vertices, new_weight, new_fdc, new_frest, new_tris = \
                self._update_params_separate(split_idx)

            split_kfIDs = None
            if isinstance(self.unique_kfIDs, torch.Tensor) and self.unique_kfIDs.numel() > 0:
                split_kfIDs = self.unique_kfIDs[split_idx].repeat_interleave(4)

            self.densification_postfix(
                new_vertices, new_weight, new_fdc, new_frest,
                new_tris, inherit_kfIDs=split_kfIDs,
            )
            n_split = split_idx.numel()

        # Build keep mask (size = current total after all appends)
        mask = torch.ones(self._triangle_indices.shape[0], dtype=torch.bool, device=device)
        if n_split > 0:
            mask[split_idx] = False
        self.prune_triangles(mask)

        # Reset stats
        if has_stats:
            self.gradient_accum.zero_()
            self.denom.zero_()

        # Cap total triangles
        if cap_max > 0 and self._triangle_indices.shape[0] > cap_max:
            excess = self._triangle_indices.shape[0] - cap_max
            cap_mask = torch.ones(self._triangle_indices.shape[0], dtype=torch.bool, device=device)
            cap_mask[-excess:] = False
            self.prune_triangles(cap_mask)

        print(
            f"densify_cs: cloned={n_cloned}  split={n_split}  "
            f"grad_thresh={grad_threshold}  size_thresh={size_threshold_px:.0f}px  "
            f"total_tris={self._triangle_indices.shape[0]}"
        )

        return mask

    def _update_params_separate(self, selected_indices):
        """
        Subdivide selected triangles into 4 smaller independent triangles.
        Each sub-triangle gets its own 3 unique vertices (no sharing).
        """
        selected_indices = torch.unique(selected_indices)
        selected_triangles_indices = self._triangle_indices[selected_indices]  # [S, 3]
        S = selected_triangles_indices.shape[0]

        # Early exit if no triangles selected
        if S == 0:
            return (
                torch.empty((0, 3), device=self.vertices.device),
                torch.empty((0, 1), device=self.vertices.device),
                torch.empty((0, *self._features_dc.shape[1:]), device=self.vertices.device),
                torch.empty((0, *self._features_rest.shape[1:]), device=self.vertices.device),
                torch.empty((0, 3), dtype=torch.int32, device=self.vertices.device)
            )

        # 1. Get vertex positions [S, 3]
        A = self.vertices[selected_triangles_indices[:, 0]]
        B = self.vertices[selected_triangles_indices[:, 1]]
        C = self.vertices[selected_triangles_indices[:, 2]]

        # 2. Calculate midpoints
        M_ab = (A + B) / 2.0
        M_bc = (B + C) / 2.0
        M_ca = (C + A) / 2.0

        # 3. Create 12 vertices per original triangle (4 sub-triangles * 3 vertices)
        # We use a explicit reshape to [12*S, 3]
        new_vertices = torch.stack([
            A, M_ab, M_ca,    # Tri 1
            B, M_bc, M_ab,    # Tri 2
            C, M_ca, M_bc,    # Tri 3
            M_ab, M_bc, M_ca  # Tri 4
        ], dim=1).reshape(12 * S, 3)

        # 4. Helper for features (interpolates and reshapes safely)
        def get_feat(feat_tensor):
            fA = feat_tensor[selected_triangles_indices[:, 0]]
            fB = feat_tensor[selected_triangles_indices[:, 1]]
            fC = feat_tensor[selected_triangles_indices[:, 2]]
            fM_ab = (fA + fB) / 2.0
            fM_bc = (fB + fC) / 2.0
            fM_ca = (fC + fA) / 2.0
            
            stacked = torch.stack([
                fA, fM_ab, fM_ca,
                fB, fM_bc, fM_ab,
                fC, fM_ca, fM_bc,
                fM_ab, fM_bc, fM_ca
            ], dim=1)
            
            # CRITICAL FIX: Use 12*S explicitly instead of -1
            new_shape = [12 * S] + list(feat_tensor.shape[1:])
            return stacked.reshape(*new_shape)

        new_features_dc = get_feat(self._features_dc)
        new_features_rest = get_feat(self._features_rest)

        # 5. Handle Opacity/Weights
        wA = self.vertex_weight[selected_triangles_indices[:, 0]]
        wB = self.vertex_weight[selected_triangles_indices[:, 1]]
        wC = self.vertex_weight[selected_triangles_indices[:, 2]]
        
        # We average in the parameter space (inverse opacity)
        wM_ab = (wA + wB) / 2.0
        wM_bc = (wB + wC) / 2.0
        wM_ca = (wC + wA) / 2.0

        new_vertex_weight = torch.stack([
            wA, wM_ab, wM_ca,
            wB, wM_bc, wM_ab,
            wC, wM_ca, wM_bc,
            wM_ab, wM_bc, wM_ca
        ], dim=1).reshape(12 * S, 1)

        # 6. Generate Triangle Indices (Vectorized)
        v_base = self.vertices.shape[0]
        # Every 3 vertices in the new block form 1 triangle
        new_triangles = torch.arange(12 * S, device=self.vertices.device, dtype=torch.int32).reshape(4 * S, 3) + v_base

        return (
            new_vertices,
            new_vertex_weight,
            new_features_dc,
            new_features_rest,
            new_triangles
        )


    def add_new_gs_original_with_visibility(self, cap_max, splitt_large_triangles, visibility=None):
        """
        Original-style densification:
        - sample by importance/size filters
        - optionally restrict to currently visible triangles
        - split selected + top-area triangles
        - prune original selected triangles

        Args:
            cap_max: max number of vertices allowed
            splitt_large_triangles: number of largest triangles to force-split
            visibility: optional tensor [T] (bool/int), where >0 means visible
        Returns:
            int: number of selected source triangles that were split
        """
        current_num_points = int(visibility.sum().item())
        target_num = min(cap_max, int(self.add_percentage * current_num_points))
        num_gs = max(0, target_num - current_num_points)

        if num_gs <= 0:
            return 0

        # Base probabilities from importance
        probs = self.importance_score.squeeze()  # [T]
        areas = self.triangle_areas().squeeze()  # [T]

        # Original geometric filters
        probs = torch.where(areas < self.size_probs_zero, torch.zeros_like(probs), probs)
        probs = torch.where(
            self.image_size < self.size_probs_zero_image_space,
            torch.zeros_like(probs),
            probs,
        )

        # Optional visibility gating
        if visibility is not None:
            vis = visibility
            if vis.dtype != torch.bool:
                vis = vis > 0
            vis = vis.to(probs.device).flatten()

            # Keep sizes aligned defensively
            n = min(vis.numel(), probs.numel())
            gated = torch.zeros_like(probs, dtype=probs.dtype)
            gated[:n] = torch.where(vis[:n], probs[:n], torch.zeros_like(probs[:n]))
            probs = gated

        # If nothing left to sample, exit
        if probs.numel() == 0 or float(probs.sum()) <= 0.0:
            return 0

        # Random samples from alive probabilities
        num_sample = min(num_gs, probs.numel())
        rand_idx = self._sample_alives(probs=probs, num=num_sample)

        # # Force-split largest triangles (optionally visibility-restricted)
        # if visibility is not None:
        #     valid_area = torch.where((visibility.to(areas.device).flatten() > 0), areas, torch.full_like(areas, -1e9))
        #     k = min(splitt_large_triangles, int((valid_area > -1e8).sum().item()))
        #     top_idx = torch.topk(valid_area, k, largest=True, sorted=False).indices if k > 0 else torch.empty(0, dtype=torch.long, device=areas.device)
        # else:
        #     k = min(splitt_large_triangles, areas.numel())
        #     top_idx = torch.topk(areas, k, largest=True, sorted=False).indices

        # Combine + dedup
        # add_idx = torch.unique(torch.cat([rand_idx, top_idx.to(rand_idx.device)]), sorted=False)
        if rand_idx.numel() == 0:
            return 0

        # NOTE: if your _update_params_fast signature is (selected_indices) in this repo, use:
        new_vertices, new_vertex_weight, new_features_dc, new_features_rest, new_triangles = self._update_params_fast(rand_idx)

        if isinstance(self.unique_kfIDs, torch.Tensor) and self.unique_kfIDs.numel() > 0:
            tris_per_source = new_triangles.shape[0] // rand_idx.shape[0]
            parent_kfIDs = self.unique_kfIDs[rand_idx].repeat_interleave(tris_per_source)
        else:
            parent_kfIDs = None
        self.densification_postfix(
            new_vertices, new_vertex_weight, new_features_dc, new_features_rest, new_triangles, inherit_kfIDs=parent_kfIDs
        )

        # Remove original triangles that were split (same behavior as original)
        mask = torch.ones(self._triangle_indices.shape[0], dtype=torch.bool, device=self._triangle_indices.device)
        mask[rand_idx] = False
        self.prune_triangles(mask)

        return mask

    def add_new_gs_original(self, cap_max, splitt_large_triangles):

        current_num_points = self.vertices.shape[0]
        target_num = min(cap_max, int(self.add_percentage * current_num_points))
        num_gs = max(0, target_num - current_num_points)

        if num_gs <= 0:
            return 0

        # Find indexes based on proba
        triangle_transp = self.importance_score
        probs = triangle_transp.squeeze()

        areas = self.triangle_areas().squeeze()
        probs = torch.where(areas < self.size_probs_zero, torch.zeros_like(probs), probs)
        probs = torch.where(self.image_size < self.size_probs_zero_image_space, torch.zeros_like(probs), probs) # dont splitt if smaller than 10

        rand_idx = self._sample_alives(probs=probs, num=num_gs)

        # Split the largest triangles
        split_large = splitt_large_triangles
        k = min(split_large, areas.numel())  
        _, top_idx = torch.topk(areas, k, largest=True, sorted=False)

        # 3) combine and deduplicate
        add_idx = torch.unique(torch.cat([rand_idx, top_idx.to(rand_idx.device)]), sorted=False)

        (new_vertices, new_vertex_weight, new_features_dc, new_features_rest, new_triangles) = self._update_params_fast(add_idx)

        if isinstance(self.unique_kfIDs, torch.Tensor) and self.unique_kfIDs.numel() > 0:
            tris_per_source = new_triangles.shape[0] // add_idx.shape[0]
            parent_kfIDs = self.unique_kfIDs[add_idx].repeat_interleave(tris_per_source)
        else:
            parent_kfIDs = None
        self.densification_postfix(new_vertices, new_vertex_weight, new_features_dc, new_features_rest, new_triangles, inherit_kfIDs=parent_kfIDs)

        mask = torch.ones(self._triangle_indices.shape[0], dtype=torch.bool)
        mask[add_idx] = False
        self.prune_triangles(mask)

    def add_new_gs(self, cap_max, splitt_large_triangles,
                   visibility=None, visibility_in_frustum=None,
                   triangle_pixel_counts=None):
        """
        Densification: select triangles to subdivide (1 → 4 sub-triangles) using a
        combined gradient + size criterion, analogous to MonoGS densify_and_split/clone.

        Criterion A — gradient-based (primary):
            avg_grad = gradient_accum / denom   (per-triangle mean vertex-grad norm)
            Threshold: `densify_grad_percentile`-th percentile of avg_grad over seen triangles.
            Motivation: high avg_grad means the render loss is pushing vertices hard in that
            region but geometry is not fine enough to resolve it → needs subdivision.

        Criterion B — size-based (secondary / fallback):
            Triangles whose pixel count is in the top-`splitt_large_triangles` set, subject to
            a minimum-pixel-count gate (`densify_min_pixel_count`).
            Motivation: keeps the existing blurry-split behaviour for large textureless regions
            where gradients may be small but triangles are simply too coarse.

        Priority when over budget:
            avg_grad × pixel_count (gradient-weighted coverage).
        """
        # Accept either keyword (mapping.py passes visibility_in_frustum=...)
        vis = visibility_in_frustum if visibility_in_frustum is not None else visibility
        assert vis is not None and vis.numel() == self._triangle_indices.shape[0]

        num_tri = self._triangle_indices.shape[0]
        device  = self._triangle_indices.device

        training_cfg = self.config.get("Training", {}) if self.config else {}
        grad_threshold_abs = training_cfg.get("densify_grad_threshold", 0.0)
        grad_percentile    = training_cfg.get("densify_grad_percentile", 0.95)
        min_pixel_count    = training_cfg.get("densify_min_pixel_count", 4)

        # ── build a per-triangle pixel-count vector (zeros if not provided) ──
        if triangle_pixel_counts is not None:
            pix = triangle_pixel_counts.squeeze().float()
        else:
            pix = torch.zeros(num_tri, dtype=torch.float32, device=device)

        # ── Criterion A: gradient-based ──────────────────────────────────────
        has_stats = (
            isinstance(self.gradient_accum, torch.Tensor)
            and self.gradient_accum.shape[0] == num_tri
            and isinstance(self.denom, torch.Tensor)
            and self.denom.shape[0] == num_tri
        )
        if has_stats:
            avg_grad = self.gradient_accum / self.denom.float().clamp(min=1)
            avg_grad[torch.isnan(avg_grad)] = 0.0
            seen = self.denom > 0  # triangles observed at least once
        else:
            avg_grad = torch.zeros(num_tri, device=device)
            seen     = torch.zeros(num_tri, dtype=torch.bool, device=device)

        grad_mask = torch.zeros(num_tri, dtype=torch.bool, device=device)
        grad_threshold = 0.0
        if seen.any():
            if grad_threshold_abs > 0:
                grad_threshold = grad_threshold_abs
            else:
                grad_threshold = torch.quantile(avg_grad[seen], grad_percentile)
            grad_mask = (avg_grad >= grad_threshold) & seen & (pix >= min_pixel_count)

        # ── Criterion B: size-based (blur split) ─────────────────────────────
        size_mask = torch.zeros(num_tri, dtype=torch.bool, device=device)
        blur_fraction = training_cfg.get("densify_blur_fraction", 0.0)
        if blur_fraction > 0 and pix.any():
            ds_cfg = self.config.get("Dataset", {}) if self.config else {}
            H = ds_cfg.get("height", 480)
            W = ds_cfg.get("width", 640)
            blur_pixel_threshold = blur_fraction * H * W
            size_mask = pix >= blur_pixel_threshold
        elif splitt_large_triangles > 0 and pix.any():
            k = min(splitt_large_triangles, num_tri)
            _, top_size = torch.topk(pix, k, largest=True, sorted=False)
            size_mask[top_size] = True
            size_mask &= pix >= min_pixel_count

        # ── Combine ──────────────────────────────────────────────────────────
        combined = grad_mask | size_mask
        add_idx  = combined.nonzero(as_tuple=False).squeeze(1)

        budget = splitt_large_triangles if splitt_large_triangles > 0 else 500
        if add_idx.numel() > budget:
            priority = (avg_grad[add_idx] + 1e-8) * (pix[add_idx] + 1e-8)
            _, top_k = torch.topk(priority, budget)
            add_idx  = add_idx[top_k]

        _grad_str = f"abs={grad_threshold_abs}" if grad_threshold_abs > 0 else f"pct={grad_percentile}"
        _blur_str = f"  blur_frac={blur_fraction}" if blur_fraction > 0 else ""
        print(
            f"densify: grad_mask={grad_mask.sum().item()} ({_grad_str}, "
            f"thresh={float(grad_threshold):.6f})  "
            f"size_mask={size_mask.sum().item()}{_blur_str}  "
            f"→ splitting {add_idx.numel()} triangles"
        )

        # ── Reset accumulated stats (stale after topology change) ────────────
        if has_stats:
            self.gradient_accum.zero_()
            self.denom.zero_()

        # ── Subdivide selected triangles ─────────────────────────────────────
        new_vertices, new_vertex_weight, new_features_dc, new_features_rest, new_triangles = \
            self._update_params_separate(add_idx)

        print(f"  new triangles: {new_triangles.shape[0]}")

        if isinstance(self.unique_kfIDs, torch.Tensor) and self.unique_kfIDs.numel() > 0 and add_idx.numel() > 0:
            parent_kfIDs = self.unique_kfIDs[add_idx].repeat_interleave(4)
        else:
            parent_kfIDs = None
        self.densification_postfix(new_vertices, new_vertex_weight, new_features_dc,
                                   new_features_rest, new_triangles,
                                   inherit_kfIDs=parent_kfIDs)

        # Remove the originals (now replaced by 4 children each).
        # Mask is built AFTER postfix so its length matches the enlarged tensor.
        mask = torch.ones(self._triangle_indices.shape[0], dtype=torch.bool)
        if add_idx.numel() > 0:
            mask[add_idx] = False
        self.prune_triangles(mask)

        num_new = new_triangles.shape[0]
        self._new_triangle_range = (self._triangle_indices.shape[0] - num_new,
                                    self._triangle_indices.shape[0])
        return mask
        

    def update_min_weight(self, new_min_weight: float, preserve_outputs: bool = True):
        new_m = float(max(0.0, min(new_min_weight, 1.0 - 1e-4)))

        # 1) grab the current realized opacities y (under the old floor)
        with torch.no_grad():
            mask = self.vertices.shape[0]
            y = self.get_vertex_weight[:mask].detach()
            y = y.clamp(new_m + self.eps, 1.0 - self.eps)   # clamp to the *new* floor
        self.opacity_floor = new_m
        new_logits = self.inverse_opacity_activation(y)
        with torch.no_grad():
            self.vertex_weight.data.copy_(new_logits)


    def triangle_areas(self):
        tri = self.vertices[self._triangle_indices]                    # [T, 3, 3]
        AB  = tri[:, 1] - tri[:, 0]                                    # [T, 3]
        AC  = tri[:, 2] - tri[:, 0]                                    # [T, 3]
        cross_prod = torch.cross(AB, AC, dim=1)                        # [T, 3]
        areas = 0.5 * torch.linalg.norm(cross_prod, dim=1)             # [T]
        areas = torch.nan_to_num(areas, nan=0.0, posinf=0.0, neginf=0.0)
        return areas

    def triangle_aspect_ratios(self):
        """Per-triangle max_edge / min_edge. 1.0 = equilateral, large = sliver."""
        v = self.vertices[self._triangle_indices]          # (T, 3, 3)
        a = (v[:, 1] - v[:, 0]).norm(dim=-1)
        b = (v[:, 2] - v[:, 1]).norm(dim=-1)
        c = (v[:, 0] - v[:, 2]).norm(dim=-1)
        edges = torch.stack([a, b, c], dim=-1)             # (T, 3)
        return edges.max(dim=-1).values / (edges.min(dim=-1).values + 1e-8)

    def mean_aspect_ratio(self):
        """Scalar mean aspect ratio across all triangles (diagnostic)."""
        return self.triangle_aspect_ratios().mean().item()


    
    def run_restricted_delaunay(self, visibility_mask=None):
        """
        Run restricted Delaunay on the mesh.
        If visibility_mask (T,) bool is provided and has any True, only retriangulate
        the visible triangles; keep the rest unchanged. Reduces holes in less-observed regions.
        """
        device = self.vertices.device
        if visibility_mask is not None:
            vis = visibility_mask.bool().to(device)
            if vis.ndim > 1:
                vis = vis.flatten()
            n_vis = vis.count_nonzero().item()
            if n_vis == 0:
                return
            if n_vis == self._triangle_indices.shape[0]:
                visibility_mask = None  # fall through to full rdel
            else:
                self._run_restricted_delaunay_visible(vis)
                return

        print("Running restricted delaunay... for ", self.vertices.shape[0], " vertices.")
        self._triangle_indices = self._triangle_indices.detach().cpu().numpy()
        faces_ = rdel.run(
            self.vertices.detach().cpu().numpy(),
            self._triangle_indices,
            verbose=False,
            orient=False,
        )
        self._triangle_indices = torch.as_tensor(np.asarray(faces_, dtype=np.int64), device=device).contiguous()
        self._triangle_indices = self._triangle_indices.to(torch.int32)
        print("We have after re-delaunay ", self._triangle_indices.shape[0], " triangles.")
        self._reset_triangle_aux_tensors()

    def _reset_triangle_aux_tensors(self):
        """Reset per-triangle auxiliary tensors to zeros."""
        T = self._triangle_indices.shape[0]
        dev = self._triangle_indices.device
        self.image_size = torch.zeros((T,), dtype=torch.float, device=dev)
        self.importance_score = torch.zeros((T,), dtype=torch.float, device=dev)
        self.pixel_count = torch.zeros((T,), dtype=torch.int, device=dev)
        self.gradient_accum = torch.zeros((T,), dtype=torch.float, device=dev)
        self.denom = torch.zeros((T,), dtype=torch.float, device=dev)
        self.n_obs = torch.zeros((T,), dtype=torch.int, device=dev)
        self.unique_kfIDs = torch.zeros((T,), dtype=torch.int, device=dev)

    def _run_restricted_delaunay_visible(self, visibility_mask):
        """Retriangulate only triangles with visibility_mask True; keep the rest."""
        device = self.vertices.device
        n_tri = self._triangle_indices.shape[0]
        keep_mask = ~visibility_mask
        n_keep = keep_mask.count_nonzero().item()
        keep_tris = self._triangle_indices[keep_mask].clone()
        vis_tris = self._triangle_indices[visibility_mask]

        used_vertex_ids = torch.unique(vis_tris.flatten())
        v_old2new = torch.full((self.vertices.shape[0],), -1, dtype=torch.long, device=device)
        v_old2new[used_vertex_ids] = torch.arange(used_vertex_ids.shape[0], device=device)

        faces_local = v_old2new[vis_tris].cpu().numpy()
        verts_local = self.vertices[used_vertex_ids].detach().cpu().numpy()

        if faces_local.shape[0] < 4 or verts_local.shape[0] < 4:
            return

        faces_new_local = rdel.run(verts_local, faces_local, verbose=False, orient=False)
        faces_new_global = used_vertex_ids.cpu().numpy()[np.asarray(faces_new_local, dtype=np.int64)]

        new_tris = torch.as_tensor(faces_new_global, dtype=torch.int32, device=device)
        self._triangle_indices = torch.cat([keep_tris, new_tris], dim=0)

        n_new = new_tris.shape[0]
        dev = device

        def _pad_aux(name, default_dtype, default_val=0):
            t = getattr(self, name, None)
            if isinstance(t, torch.Tensor) and t.shape[0] == n_tri:
                kept = t[keep_mask]
            else:
                kept = torch.full((n_keep,), default_val, dtype=default_dtype, device=dev)
            new_zeros = torch.zeros(n_new, dtype=default_dtype, device=dev)
            setattr(self, name, torch.cat([kept, new_zeros], dim=0))

        _pad_aux("image_size", torch.float)
        _pad_aux("importance_score", torch.float)
        _pad_aux("pixel_count", torch.int)
        _pad_aux("gradient_accum", torch.float)
        _pad_aux("denom", torch.float)
        _pad_aux("n_obs", torch.int)
        _pad_aux("unique_kfIDs", torch.int)

        print("Rdel visible: ", visibility_mask.count_nonzero().item(), " tris -> ", n_new, " tris (kept ", n_keep, " unchanged)")

    def run_restricted_delaunay_from_pcd(self, points, triangle_indices, features, opacities):

        print("Running restricted delaunay... for ", points.shape[0], " vertices.")

        pcd = points.detach().cpu().numpy()
       
        faces_ = rdel.run(
            pcd,
            triangle_indices,
            verbose=False,  # print timings and extra logs if True
            orient=False    # try to consistently orient face normals if True
        )

        faces_ = torch.as_tensor(np.asarray(faces_, dtype=np.int64), device='cuda').contiguous()
        faces_ = faces_.to(torch.int32)
        
        delaunay_result_dict = {
            "faces": faces_,
            "pcd": points,
            "features": features,
            "opacities": opacities
        }
        return delaunay_result_dict