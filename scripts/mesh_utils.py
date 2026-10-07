#
# Copyright (C) 2024, ShanghaiTech
# SVIP research group, https://github.com/svip-lab
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  huangbb@shanghaitech.edu.cn
#

import torch
import numpy as np
import os
import math
from tqdm import tqdm


from functools import partial
import open3d as o3d
import trimesh
from PIL import Image as PILImage


def focus_point_fn(poses: np.ndarray) -> np.ndarray:
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt

def save_img_u8(img, pth):
  """Save an image (probably RGB) in [0, 1] to disk as a uint8 PNG."""
  with open(pth, 'wb') as f:
    Image.fromarray(
        (np.clip(np.nan_to_num(img), 0., 1.) * 255.).astype(np.uint8)).save(
            f, 'PNG')


def save_img_f32(depthmap, pth):
  """Save an image (probably a depthmap) to disk as a float32 TIFF."""
  with open(pth, 'wb') as f:
    Image.fromarray(np.nan_to_num(depthmap).astype(np.float32)).save(f, 'TIFF')

def post_process_mesh(mesh, cluster_to_keep=1000):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0

def to_cam_open3d(viewpoint_stack):
    camera_traj = []
    for i, viewpoint_cam in enumerate(viewpoint_stack):
        W = viewpoint_cam.image_width
        H = viewpoint_cam.image_height
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, 0, 1]]).float().cuda().T
        intrins =  (viewpoint_cam.projection_matrix @ ndc2pix)[:3,:3].T
        intrinsic=o3d.camera.PinholeCameraIntrinsic(
            width=viewpoint_cam.image_width,
            height=viewpoint_cam.image_height,
            cx = intrins[0,2].item(),
            cy = intrins[1,2].item(), 
            fx = intrins[0,0].item(), 
            fy = intrins[1,1].item()
        )

        extrinsic=np.asarray((viewpoint_cam.world_view_transform.T).cpu().numpy())
        camera = o3d.camera.PinholeCameraParameters()
        camera.extrinsic = extrinsic
        camera.intrinsic = intrinsic
        camera_traj.append(camera)

    return camera_traj


def save_depth_renders(cameras_dict, gaussians, pipe, background, save_dir, kf_ids=None):
    """Render frames and save RGB, depth, and camera parameters to disk.

    Output directory: <save_dir>/depth_renders/
      <fid:05d>_rgb.png   - uint8 RGB render
      <fid:05d>_depth.npy - float32 depth map (metres, shape H x W)
      cameras.npz         - intrinsics + world_view_transform + projection_matrix

    The saved data is compatible with GaussianExtractor.from_saved_renders() for
    later mesh extraction without needing the trained Gaussians in memory.

    cameras_dict : dict[int -> Camera]  (frontend.cameras)
    gaussians    : GaussianModel
    pipe         : pipeline params
    background   : background color tensor (cuda)
    save_dir     : root result directory
    kf_ids       : if given, only save these frame IDs (e.g. frontend.kf_indices)
    """
    from gaussian_splatting.gaussian_renderer import render as gs_render

    out_dir = os.path.join(save_dir, "depth_renders")
    os.makedirs(out_dir, exist_ok=True)

    all_ids = sorted(cameras_dict.keys())
    frame_ids = sorted(kf_ids) if kf_ids is not None else all_ids
    print(f"[save_depth_renders] saving {len(frame_ids)} frames"
          + (" (keyframes only)" if kf_ids is not None else ""))

    fx_list, fy_list, cx_list, cy_list = [], [], [], []
    widths, heights = [], []
    wvts, proj_mats = [], []

    with torch.no_grad():
        for fid in tqdm(frame_ids, desc="Saving depth renders"):
            cam = cameras_dict[fid]
            render_pkg = gs_render(cam, gaussians, pipe, background)

            rgb   = render_pkg["render"]  # (3, H, W)
            depth = render_pkg["depth"]   # (1, H, W)

            # RGB → uint8 PNG
            rgb_np = np.clip(rgb.permute(1, 2, 0).cpu().numpy(), 0.0, 1.0)
            PILImage.fromarray((rgb_np * 255).astype(np.uint8)).save(
                os.path.join(out_dir, f"{fid:05d}_rgb.png")
            )

            # Depth → float32 npy (H, W)
            depth_np = depth.squeeze(0).cpu().numpy().astype(np.float32)
            np.save(os.path.join(out_dir, f"{fid:05d}_depth.npy"), depth_np)

            fx_list.append(cam.fx)
            fy_list.append(cam.fy)
            cx_list.append(cam.cx)
            cy_list.append(cam.cy)
            widths.append(cam.image_width)
            heights.append(cam.image_height)
            wvts.append(cam.world_view_transform.cpu().numpy())
            proj_mats.append(cam.projection_matrix.cpu().numpy())

    np.savez(
        os.path.join(out_dir, "cameras.npz"),
        frame_ids=np.array(frame_ids, dtype=np.int32),
        fx=np.array(fx_list, dtype=np.float32),
        fy=np.array(fy_list, dtype=np.float32),
        cx=np.array(cx_list, dtype=np.float32),
        cy=np.array(cy_list, dtype=np.float32),
        widths=np.array(widths, dtype=np.int32),
        heights=np.array(heights, dtype=np.int32),
        world_view_transforms=np.stack(wvts).astype(np.float32),
        projection_matrices=np.stack(proj_mats).astype(np.float32),
    )
    print(f"Saved {len(frame_ids)} depth renders to {out_dir}")


class _CameraProxy:
    """Minimal camera stand-in built from saved numpy arrays.

    Satisfies the interface expected by to_cam_open3d() and extract_mesh_bounded():
      .world_view_transform  – cuda float tensor (4x4)
      .projection_matrix     – cuda float tensor (4x4)
      .image_width / .image_height
      .gt_alpha_mask         – None (no mask available from saved renders)
    """

    def __init__(self, world_view_transform, projection_matrix, width, height):
        self.world_view_transform = (
            torch.from_numpy(world_view_transform).float().cuda()
        )
        self.projection_matrix = (
            torch.from_numpy(projection_matrix).float().cuda()
        )
        self.image_width  = int(width)
        self.image_height = int(height)
        self.gt_alpha_mask = None


class GaussianExtractor(object):
    def __init__(self, gaussians, render, pipe, bg_color=None):
        """
        a class that extracts attributes a scene presented by 2DGS

        Usage example:
        >>> gaussExtrator = GaussianExtractor(gaussians, render, pipe)
        >>> gaussExtrator.reconstruction(view_points)
        >>> mesh = gaussExtractor.export_mesh_bounded(...)
        """
        if bg_color is None:
            bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.gaussians = gaussians
        self.render = partial(render, pipe=pipe, bg_color=background)
        self.clean()

    @classmethod
    def from_saved_renders(cls, render_dir, arr_indices=None):
        """Create a GaussianExtractor from pre-saved depth renders (no Gaussians needed).

        Loads the data written by save_depth_renders() and returns an extractor
        ready to call extract_mesh_bounded() on.

        render_dir  : path to the depth_renders/ directory
        arr_indices : optional array of positional indices into cameras.npz to
                      load a subset of frames (e.g. for stride / max_frames).
                      None loads every frame.
        """
        instance = cls.__new__(cls)
        instance.clean()
        instance.load_renders(render_dir, arr_indices=arr_indices)
        return instance

    def load_renders(self, render_dir, arr_indices=None):
        """Populate rgbmaps, depthmaps, and viewpoint_stack from a saved render directory.

        After this call, extract_mesh_bounded() / extract_mesh_unbounded() work without
        needing the original Gaussian model.

        arr_indices : optional 1-D array of positional indices to load only a
                      subset of the saved frames.  None loads every frame.
        """
        data = np.load(os.path.join(render_dir, "cameras.npz"))
        frame_ids = data["frame_ids"]

        if arr_indices is None:
            arr_indices = np.arange(len(frame_ids))

        self.clean()
        for arr_idx in tqdm(arr_indices, desc="Loading renders"):
            fid = int(frame_ids[arr_idx])

            rgb_np = (
                np.array(PILImage.open(os.path.join(render_dir, f"{fid:05d}_rgb.png")))
                .astype(np.float32) / 255.0
            )
            depth_np = np.load(os.path.join(render_dir, f"{fid:05d}_depth.npy"))

            self.rgbmaps.append(torch.from_numpy(rgb_np).permute(2, 0, 1))   # (3,H,W)
            self.depthmaps.append(torch.from_numpy(depth_np).unsqueeze(0))    # (1,H,W)
            self.viewpoint_stack.append(_CameraProxy(
                world_view_transform=data["world_view_transforms"][arr_idx],
                projection_matrix=data["projection_matrices"][arr_idx],
                width=int(data["widths"][arr_idx]),
                height=int(data["heights"][arr_idx]),
            ))

        self.estimate_bounding_sphere()
        print(f"Loaded {len(frame_ids)} frames from {render_dir}")

    @torch.no_grad()
    def clean(self):
        self.depthmaps = []
        # self.alphamaps = []
        self.rgbmaps = []
        # self.normals = []
        # self.depth_normals = []
        self.viewpoint_stack = []

    @torch.no_grad()
    def reconstruction(self, viewpoint_stack):
        """
        reconstruct radiance field given cameras
        """
        self.clean()
        self.viewpoint_stack = viewpoint_stack
        for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="reconstruct radiance fields"):
            render_pkg = self.render(viewpoint_cam, self.gaussians)
            rgb = render_pkg['render']
            alpha = render_pkg['rend_alpha']
            normal = torch.nn.functional.normalize(render_pkg['rend_normal'], dim=0)
            depth = render_pkg['surf_depth']
            depth_normal = render_pkg['surf_normal']
            self.rgbmaps.append(rgb.cpu())
            self.depthmaps.append(depth.cpu())
            # self.alphamaps.append(alpha.cpu())
            # self.normals.append(normal.cpu())
            # self.depth_normals.append(depth_normal.cpu())
        
        # self.rgbmaps = torch.stack(self.rgbmaps, dim=0)
        # self.depthmaps = torch.stack(self.depthmaps, dim=0)
        # self.alphamaps = torch.stack(self.alphamaps, dim=0)
        # self.depth_normals = torch.stack(self.depth_normals, dim=0)
        self.estimate_bounding_sphere()

    def estimate_bounding_sphere(self):
        """
        Estimate the bounding sphere given camera pose
        """
        # from utils.render_utils import transform_poses_pca, focus_point_fn
        torch.cuda.empty_cache()
        c2ws = np.array([np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in self.viewpoint_stack])
        poses = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])
        center = (focus_point_fn(poses))
        self.radius = np.linalg.norm(c2ws[:,:3,3] - center, axis=-1).min()
        self.center = torch.from_numpy(center).float().cuda()
        print(f"The estimated bounding radius is {self.radius:.2f}")
        print(f"Use at least {2.0 * self.radius:.2f} for depth_trunc")

    @torch.no_grad()
    def extract_mesh_bounded(self, voxel_size=0.004, sdf_trunc=0.02, depth_trunc=3, mask_backgrond=True):
        """
        Perform TSDF fusion given a fixed depth range, used in the paper.
        
        voxel_size: the voxel size of the volume
        sdf_trunc: truncation value
        depth_trunc: maximum depth range, should depended on the scene's scales
        mask_backgrond: whether to mask backgroud, only works when the dataset have masks

        return o3d.mesh
        """
        print("Running tsdf volume integration ...")
        print(f'voxel_size: {voxel_size}')
        print(f'sdf_trunc: {sdf_trunc}')
        print(f'depth_truc: {depth_trunc}')

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length= voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        for i, cam_o3d in tqdm(enumerate(to_cam_open3d(self.viewpoint_stack)), desc="TSDF integration progress"):
            rgb = self.rgbmaps[i]
            depth = self.depthmaps[i]
            
            # if we have mask provided, use it
            if mask_backgrond and (self.viewpoint_stack[i].gt_alpha_mask is not None):
                depth[(self.viewpoint_stack[i].gt_alpha_mask < 0.5)] = 0

            # make open3d rgbd
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.asarray(np.clip(rgb.permute(1,2,0).cpu().numpy(), 0.0, 1.0) * 255, order="C", dtype=np.uint8)),
                o3d.geometry.Image(np.asarray(depth.permute(1,2,0).cpu().numpy(), order="C")),
                depth_trunc = depth_trunc, convert_rgb_to_intensity=False,
                depth_scale = 1.0
            )

            volume.integrate(rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)

        mesh = volume.extract_triangle_mesh()
        return mesh

    @torch.no_grad()
    def extract_mesh_unbounded(self, resolution=1024):
        """
        Experimental features, extracting meshes from unbounded scenes, not fully test across datasets. 
        return o3d.mesh
        """
        def contract(x):
            mag = torch.linalg.norm(x, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag))
        
        def uncontract(y):
            mag = torch.linalg.norm(y, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, y, (1 / (2-mag) * (y/mag)))

        def compute_sdf_perframe(i, points, depthmap, rgbmap, viewpoint_cam):
            """
                compute per frame sdf
            """
            new_points = torch.cat([points, torch.ones_like(points[...,:1])], dim=-1) @ viewpoint_cam.full_proj_transform
            z = new_points[..., -1:]
            pix_coords = (new_points[..., :2] / new_points[..., -1:])
            mask_proj = ((pix_coords > -1. ) & (pix_coords < 1.) & (z > 0)).all(dim=-1)
            sampled_depth = torch.nn.functional.grid_sample(depthmap.cuda()[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(-1, 1)
            sampled_rgb = torch.nn.functional.grid_sample(rgbmap.cuda()[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(3,-1).T
            sdf = (sampled_depth-z)
            return sdf, sampled_rgb, mask_proj

        def compute_unbounded_tsdf(samples, inv_contraction, voxel_size, return_rgb=False):
            """
                Fusion all frames, perform adaptive sdf_funcation on the contract spaces.
            """
            if inv_contraction is not None:
                mask = torch.linalg.norm(samples, dim=-1) > 1
                # adaptive sdf_truncation
                sdf_trunc = 5 * voxel_size * torch.ones_like(samples[:, 0])
                sdf_trunc[mask] *= 1/(2-torch.linalg.norm(samples, dim=-1)[mask].clamp(max=1.9))
                samples = inv_contraction(samples)
            else:
                sdf_trunc = 5 * voxel_size

            tsdfs = torch.ones_like(samples[:,0]) * (-1)
            rgbs = torch.zeros((samples.shape[0], 3)).cuda()

            weights = torch.ones_like(samples[:,0])
            for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="TSDF integration progress"):
                sdf, rgb, mask_proj = compute_sdf_perframe(i, samples,
                    depthmap = self.depthmaps[i],
                    rgbmap = self.rgbmaps[i],
                    viewpoint_cam=self.viewpoint_stack[i],
                )

                # volume integration
                sdf = sdf.flatten()
                mask_proj = mask_proj & (sdf > -sdf_trunc)
                sdf = torch.clamp(sdf / sdf_trunc, min=-1.0, max=1.0)[mask_proj]
                w = weights[mask_proj]
                wp = w + 1
                tsdfs[mask_proj] = (tsdfs[mask_proj] * w + sdf) / wp
                rgbs[mask_proj] = (rgbs[mask_proj] * w[:,None] + rgb[mask_proj]) / wp[:,None]
                # update weight
                weights[mask_proj] = wp
            
            if return_rgb:
                return tsdfs, rgbs

            return tsdfs

        normalize = lambda x: (x - self.center) / self.radius
        unnormalize = lambda x: (x * self.radius) + self.center
        inv_contraction = lambda x: unnormalize(uncontract(x))

        N = resolution
        voxel_size = (self.radius * 2 / N)
        print(f"Computing sdf gird resolution {N} x {N} x {N}")
        print(f"Define the voxel_size as {voxel_size}")
        sdf_function = lambda x: compute_unbounded_tsdf(x, inv_contraction, voxel_size)
        from utils.mcube_utils import marching_cubes_with_contraction
        R = contract(normalize(self.gaussians.get_xyz)).norm(dim=-1).cpu().numpy()
        R = np.quantile(R, q=0.95)
        R = min(R+0.01, 1.9)

        mesh = marching_cubes_with_contraction(
            sdf=sdf_function,
            bounding_box_min=(-R, -R, -R),
            bounding_box_max=(R, R, R),
            level=0,
            resolution=N,
            inv_contraction=inv_contraction,
        )
        
        # coloring the mesh
        torch.cuda.empty_cache()
        mesh = mesh.as_open3d
        print("texturing mesh ... ")
        _, rgbs = compute_unbounded_tsdf(torch.tensor(np.asarray(mesh.vertices)).float().cuda(), inv_contraction=None, voxel_size=voxel_size, return_rgb=True)
        mesh.vertex_colors = o3d.utility.Vector3dVector(rgbs.cpu().numpy())
        return mesh

    @torch.no_grad()
    def export_image(self, path):
        render_path = os.path.join(path, "renders")
        gts_path = os.path.join(path, "gt")
        vis_path = os.path.join(path, "vis")
        os.makedirs(render_path, exist_ok=True)
        os.makedirs(vis_path, exist_ok=True)
        os.makedirs(gts_path, exist_ok=True)
        for idx, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="export images"):
            gt = viewpoint_cam.original_image[0:3, :, :]
            save_img_u8(gt.permute(1,2,0).cpu().numpy(), os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            save_img_u8(self.rgbmaps[idx].permute(1,2,0).cpu().numpy(), os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            save_img_f32(self.depthmaps[idx][0].cpu().numpy(), os.path.join(vis_path, 'depth_{0:05d}'.format(idx) + ".tiff"))
            # save_img_u8(self.normals[idx].permute(1,2,0).cpu().numpy() * 0.5 + 0.5, os.path.join(vis_path, 'normal_{0:05d}'.format(idx) + ".png"))
            # save_img_u8(self.depth_normals[idx].permute(1,2,0).cpu().numpy() * 0.5 + 0.5, os.path.join(vis_path, 'depth_normal_{0:05d}'.format(idx) + ".png"))
