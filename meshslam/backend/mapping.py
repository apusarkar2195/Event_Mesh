"""
Mapping: keyframe insertion (map growth + prune) and map optimisation
with current view + keyframe replay.
Designed to run in the same process as the frontend for now; later can run
in a separate process with MapState (or a serialized snapshot) as the interface.
"""
import time
from typing import Optional, Callable

import torch
import random

from mesh_splatting.triangle_renderer import render
from utils.slam_utils import get_loss_mapping

from .map_state import MapState
from mesh_splatting.utils.loss_utils import l1_loss, ssim, vertex_depth_loss_hr, equilateral_regularizer, d2n_tblr, point_to_normal
import numpy as np
import rerun as rr

from utils.slam_utils import get_depth_loss
import open3d as o3d
from utils.gui_utils import log_mesh_rerun

from utils.pose_utils import update_pose


try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
    print("Using fused SSIM for faster training.")
except:
    FUSED_SSIM_AVAILABLE = False
try:
    from diff_triangle_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
    print("Sparse Adam optimizer available")
except:
    SPARSE_ADAM_AVAILABLE = False


class MappingEngine:
    """Backend: map growth (add keyframe), keyframe buffer, and mapping optimization."""

    def __init__(self, map_state: MapState, opt, pipe, config: dict):
        self.map_state = map_state
        self.opt = opt
        self.pipe = pipe
        self.config = config
        self.triangles = map_state.triangles
        self.background = map_state.background
        self.keyframe_buffer = map_state.keyframe_buffer
        self.initial_ba_done = False

    @staticmethod
    def _loss_plateaued(losses, window=10, min_rel_improvement=0.005):
        """True if the mapping loss has not improved by min_rel_improvement (relative) over the last `window` iters."""
        if len(losses) < window:
            return False
        recent = losses[-window:]
        rel_improvement = (recent[0] - recent[-1]) / (recent[0] + 1e-9)
        return rel_improvement < min_rel_improvement

    def add_keyframe(self, viewpoint_cam, under_reconstructed_mask=None, rendered_depth=None) -> None:
        """Extend the map from this viewpoint, prune, and push to keyframe buffer."""

             # Map growth
        pcd, features, opacities, radii = self.triangles.create_pcd_from_image(
            viewpoint_cam, init=False, depthmap=viewpoint_cam.depth,
            under_reconstructed_mask=under_reconstructed_mask,
            rendered_depth=rendered_depth,
        )

        self.triangles.extend_from_pcd(viewpoint_cam, pcd, radii, features, opacities, under_reconstructed_mask, kf_id=viewpoint_cam.uid)


    def prune(self, visibility_in_frustum, frustum_union=None, use_importance_mask: bool = False) -> None:
        prune_coviz = self.config.get("Training", {}).get("prune_coviz", 0)
        prune_importance_threshold = self.config.get("opt_params", {}).get("prune_importance_threshold", 0.01)
        kf_window_ids = [kf.uid for kf in self.keyframe_buffer]
        n_pruned, keep_mask = self.triangles.run_prune(
            visibility_in_frustum=visibility_in_frustum,
            frustum_union=frustum_union,
            prune_triangles_threshold=self.opt.prune_triangles_threshold,
            prune_size=self.opt.prune_size,
            use_importance_mask=use_importance_mask,
            prune_importance_threshold=prune_importance_threshold,
            prune_coviz=prune_coviz,
            num_keyframes=len(self.keyframe_buffer),
            kf_window_ids=kf_window_ids,
        )

        return keep_mask

    def densify_and_prune(self, last_keyframe_visibility) -> None: 
        print("densifying...")

        visibility_in_frustum = last_keyframe_visibility > 0

        self.triangles.add_new_gs(cap_max=self.opt.max_points, 
        splitt_large_triangles=self.opt.splitt_large_triangles,
        visibility_in_frustum=visibility_in_frustum)

        # self.prune(use_importance_mask=False, 
        # visibility_in_frustum=visibility_in_frustum)

    def _fd_check_vertex_grad(
        self,
        viewpoint,
        num_samples: int = 20,
        eps: float = 1e-5,
    ) -> list:
        """
        Finite-difference check of vertex gradients. Call when triangles.vertices
        has requires_grad=True (e.g. during mapping).

        Returns list of (v_idx, coord, num_grad, ana_grad, rel_err).
        """
        vertices = self.triangles.vertices
        if not vertices.requires_grad:
            raise RuntimeError("vertices must have requires_grad=True for FD check")

        # Get analytical gradient: forward + backward
        render_pkg = render(viewpoint, self.triangles, self.pipe, self.background)
        loss = get_loss_mapping(
            self.config,
            render_pkg["render"],
            render_pkg["surf_depth"],
            viewpoint,
            render_pkg["rend_alpha"],
        )
        loss.backward()

        g_ana = vertices.grad.clone()
        vertices.grad = None

        def loss_fn():
            rp = render(viewpoint, self.triangles, self.pipe, self.background)
            return get_loss_mapping(
                self.config,
                rp["render"],
                rp["surf_depth"],
                viewpoint,
                rp["rend_alpha"],
            )

        vertex_rendered = render_pkg["vertex_rendered"]
        rendered_idx = torch.where(vertex_rendered > 0)[0]
        if rendered_idx.numel() > 0:
            n = min(num_samples, rendered_idx.numel())
            perm = torch.randperm(rendered_idx.numel(), device=rendered_idx.device)[:n]
            sample_idx = rendered_idx[perm]
        else:
            V = vertices.shape[0]
            n = min(num_samples, V)
            sample_idx = torch.randperm(V, device=vertices.device)[:n]

        errors = []
        flat = vertices.data.view(-1)
        for vi in sample_idx:
            vi = vi.item()
            for coord in range(3):
                idx = vi * 3 + coord
                orig = flat[idx].item()
                flat[idx] = orig + eps
                L_plus = loss_fn().item()
                flat[idx] = orig - eps
                L_minus = loss_fn().item()
                flat[idx] = orig
                num_grad = (L_plus - L_minus) / (2.0 * eps)
                ana_grad = g_ana[vi, coord].item()
                denom = abs(ana_grad) + 1e-9
                rel_err = abs(num_grad - ana_grad) / denom
                errors.append((vi, coord, num_grad, ana_grad, rel_err))

        return errors

    def _prune_vertices(self) -> None:
        """Remove vertices that are no longer used by any triangle."""
        device = self.triangles.vertices.device
        used_vertex_mask = torch.zeros(
            self.triangles.vertices.shape[0], dtype=torch.bool, device=device
        )
        if self.triangles._triangle_indices.numel() > 0:
            flat_indices = self.triangles._triangle_indices.flatten()
            used_vertex_mask[flat_indices] = True
        weight_mask = (self.triangles.get_vertex_weight.squeeze() >= 0.8)
        mask_out = self.triangles.vertices.shape[0]
        vertex_mask = weight_mask[:mask_out] | used_vertex_mask
        self.triangles._prune_vertices(vertex_mask)



    def _build_mapping_optimizer(
        self,
        keyframes,
        pose_window: Optional[int] = None,
        pose_lr_scale: Optional[float] = None,
    ):
        """Build optimizer for keyframe pose deltas used during mapping."""
        lr_cam = self.config["Training"]["lr"]
        default_pose_lr_scale = self.config["Training"].get("mapping_pose_lr_scale", 0.5)
        if pose_lr_scale is None:
            pose_lr_scale = default_pose_lr_scale

        # Keep keyframe 0 fixed as anchor; optimize most recent keyframes first.
        optimizable_keyframes = [
            kf for kf in reversed(keyframes) if getattr(kf, "uid", None) != 0
        ]
        if pose_window is not None:
            optimizable_keyframes = optimizable_keyframes[: max(0, int(pose_window))]

        opt_params = []
        for kf in optimizable_keyframes:
            if not hasattr(kf, "cam_rot_delta") or kf.cam_rot_delta is None:
                continue
            opt_params.append(
                {
                    "params": [kf.cam_rot_delta],
                    "lr": lr_cam["cam_rot_delta"] * pose_lr_scale,
                    "name": f"kf_rot_{kf.uid}",
                }
            )
            opt_params.append(
                {
                    "params": [kf.cam_trans_delta],
                    "lr": lr_cam["cam_trans_delta"] * pose_lr_scale,
                    "name": f"kf_trans_{kf.uid}",
                }
            )

            opt_params.append(
                {
                    "params": [kf.exposure_a],
                    "lr": 0.01,
                    "name": "exposure_a_{}".format(kf.uid),
                }
            )
            opt_params.append(
                {
                    "params": [kf.exposure_b],
                    "lr": 0.01,
                    "name": "exposure_b_{}".format(kf.uid),
                }
            )

        if len(opt_params) == 0:
            return None, []
        return torch.optim.Adam(opt_params), optimizable_keyframes

    @torch.no_grad()
    def evaluate_photometric_objective(
        self,
        pose_window: Optional[int] = None,
        keyframes: Optional[list] = None,
    ) -> dict:
        """
        Evaluate mapping photometric objective on recent keyframes.
        Returns mean L1, mean DSSIM term, and their weighted objective.
        """
        eval_source = list(self.keyframe_buffer) if keyframes is None else list(keyframes)
        if len(eval_source) == 0:
            return {
                "count": 0,
                "mean_l1": 0.0,
                "mean_dssim_term": 0.0,
                "mean_photo_obj": 0.0,
            }

        eval_keyframes = [kf for kf in reversed(eval_source) if getattr(kf, "uid", None) != 0]
        if pose_window is not None:
            eval_keyframes = eval_keyframes[: max(0, int(pose_window))]
        if len(eval_keyframes) == 0:
            eval_keyframes = [eval_source[-1]]

        l1_vals = []
        dssim_terms = []
        photo_objs = []
        lambda_dssim = self.opt.lambda_dssim

        for kf in eval_keyframes:
            render_pkg = render(kf, self.triangles, self.pipe, self.background)
            image = render_pkg["render"]
            gt_image = kf.original_image.cuda()
            l1 = l1_loss(image, gt_image)
            dssim_term = 1.0 - ssim(image, gt_image)
            photo_obj = (1.0 - lambda_dssim) * l1 + lambda_dssim * dssim_term

            l1_vals.append(float(l1.item()))
            dssim_terms.append(float(dssim_term.item()))
            photo_objs.append(float(photo_obj.item()))

        return {
            "count": len(eval_keyframes),
            "mean_l1": float(np.mean(l1_vals)),
            "mean_dssim_term": float(np.mean(dssim_terms)),
            "mean_photo_obj": float(np.mean(photo_objs)),
        }

    def depth_losses(self, render_pkg, viewpoint_cam, iteration):
          # Vertex depth regularization
        # Lvertex_depth_pure = 0.0
        lambda_vertex = self.opt.lambda_vertex #if iteration > self.opt.start_vertex_opt else 0
        if lambda_vertex > 0:
            depth_down = render_pkg["surf_depth"]
            vertex_depth_out = render_pkg["vertex_depth_out"]
            image_2D = render_pkg["image_2D"]
            vertex_rendered = render_pkg["vertex_rendered"]
            Lvertex_depth_pure = vertex_depth_loss_hr(
                vertex_depth_out,
                image_2D,
                vertex_rendered,
                depth_down,
                max_diff_threshold=self.opt.max_diff_threshold,
            )
            Lvertex_depth = lambda_vertex * Lvertex_depth_pure
            # loss += Lvertex_depth
        else:
            Lvertex_depth = torch.tensor(0.0, device=render_pkg["render"].device)

        return Lvertex_depth


    def run_mapping_iterations(
        self,
        num_iters: int,
        viewpoint_cam,
        replay_size: int = 0,
        pose_window: Optional[int] = None,
        pose_lr_scale: Optional[float] = None,
        loss_history: Optional[list] = None, 
        log_every: int = 1,
        view_iteration: int = 0,
        total_views: int = 0,
        viz_callback: Optional[Callable] = None,
    ) -> list:
        """
        Run num_iters mapping steps: current view + optional keyframe replay,
        backward to triangles, one optimizer step per iteration.
        """
        if loss_history is None:
            loss_history = []
        replay_size = min(replay_size, len(self.keyframe_buffer))  # if use_replay else 0

        did_rdel = False
        if self.config["Training"]["rdel"]:
            training_cfg = self.config.get("Training", {})
            rdel_global = training_cfg.get("rdel_global", False)
            rdel_visible_only = training_cfg.get("rdel_visible_only", False)
            rdel_importance_pct = training_cfg.get("rdel_importance_percentile", 1.0)

            visibility = None
            if rdel_global:
                pass  # visibility stays None → run on all triangles
            elif rdel_visible_only:
                vis = self.keyframe_buffer.get_last_keyframe_visibility()
                if vis is not None:
                    n_tri = self.triangles._triangle_indices.shape[0]
                    if vis.numel() < n_tri:
                        pad = torch.zeros(n_tri - vis.numel(), device=vis.device, dtype=vis.dtype)
                        vis = torch.cat([vis, pad], dim=0)
                    else:
                        vis = vis[:n_tri].contiguous()
                    visibility = vis.bool()

            if not rdel_global and rdel_importance_pct < 1.0:
                imp = self.triangles.importance_score.squeeze()
                n_tri = self.triangles._triangle_indices.shape[0]
                if imp.numel() >= n_tri:
                    imp = imp[:n_tri]
                    if visibility is not None:
                        imp_candidates = imp[visibility]
                    else:
                        imp_candidates = imp
                    if imp_candidates.numel() > 0:
                        q = torch.quantile(imp_candidates.float(), 1.0 - rdel_importance_pct)
                        imp_mask = (imp >= q)
                        visibility = imp_mask if visibility is None else (visibility & imp_mask)

            self.triangles.run_restricted_delaunay(visibility_mask=visibility)
            self.keyframe_buffer.pad_visibilities(self.triangles._triangle_indices.shape[0])
            did_rdel = True

        assert self.keyframe_buffer.get_last_keyframe().uid == viewpoint_cam.uid



    
        # at this point the last keyframe is the current view
        current_visibility = self.keyframe_buffer.get_last_keyframe_visibility()
        replay_keyframes = self.keyframe_buffer.get_replay_sample_with_visibility_and_distance(replay_size, current_visibility, viewpoint_cam.camera_center)
        # add current view to the replay
        replay_keyframes = [viewpoint_cam] + replay_keyframes

        # Add random replay keyframes that are not already in the replay window.
        # If that unique pool is too small, fall back to sampling from the replay window.
        all_keyframes = list(self.keyframe_buffer)
        existing_uids = {getattr(kf, "uid", None) for kf in replay_keyframes}
        unique_pool = [
            kf for kf in all_keyframes if getattr(kf, "uid", None) not in existing_uids
        ]

        random_keyframes = []
        if len(unique_pool) > 0:
            n_unique = min(4, len(unique_pool))
            random_keyframes.extend(random.sample(unique_pool, n_unique))

        n_missing = 4 - len(random_keyframes)
        if n_missing > 0 and len(replay_keyframes) > 0:
            # Fallback explicitly from the selected replay window.
            selected_uids = {getattr(kf, "uid", None) for kf in random_keyframes}
            fallback_pool = [
                kf for kf in replay_keyframes if getattr(kf, "uid", None) not in selected_uids
            ]
            if len(fallback_pool) > 0:
                n_fallback = min(n_missing, len(fallback_pool))
                random_keyframes.extend(random.sample(fallback_pool, n_fallback))

        replay_keyframes = replay_keyframes + random_keyframes

        # Ensure each replay keyframe appears at most once (stable order by first appearance).
        replay_unique = []
        seen_uids = set()
        for kf in replay_keyframes:
            uid = getattr(kf, "uid", None)
            if uid in seen_uids:
                continue
            seen_uids.add(uid)
            replay_unique.append(kf)
        replay_keyframes = replay_unique

        training_cfg = self.config.get("Training", {})
        initial_ba_cfg = training_cfg.get("initial_ba", {})
        buffer_size = len(self.keyframe_buffer)
        active_window_size = len(replay_keyframes)

        effective_num_iters = num_iters
        effective_pose_window = pose_window
        effective_pose_lr_scale = pose_lr_scale

        print("buffer_size, effective_pose_window: ", buffer_size, pose_window)
        print("active_window_size: ", active_window_size)

        should_initial_ba = (
            initial_ba_cfg.get("enabled", True)
            and (not self.initial_ba_done)
            and active_window_size > 0
            and buffer_size > effective_pose_window
        )
        pre_ba_metrics = None
        if should_initial_ba:
            effective_num_iters = int(
                initial_ba_cfg.get("mapping_iters", max(6 * num_iters, 120))
            )
            effective_pose_window = int(
                initial_ba_cfg.get("pose_window", active_window_size)
            )
            if effective_pose_lr_scale is None:
                effective_pose_lr_scale = float(
                    initial_ba_cfg.get(
                        "pose_lr_scale",
                        2.0 * training_cfg.get("mapping_pose_lr_scale", 0.5),
                    )
                )

            print(
                f"[initial BA] trigger=buffer_eq_replay "
                f"buffer={buffer_size} replay={active_window_size} "
                f"iters={effective_num_iters} "
                f"pose_window={effective_pose_window} "
                f"pose_lr_scale={effective_pose_lr_scale:.3f}"
            )
            pre_ba_metrics = self.evaluate_photometric_objective(
                pose_window=effective_pose_window,
                keyframes=replay_keyframes,
            )
            print(
                f"[initial BA][before] kfs={pre_ba_metrics['count']} "
                f"photo={pre_ba_metrics['mean_photo_obj']:.6f} "
                f"l1={pre_ba_metrics['mean_l1']:.6f} "
                f"dssim={pre_ba_metrics['mean_dssim_term']:.6f}"
            )


        print("learning rate: ", effective_pose_lr_scale)
        keyframe_optimizer, optimized_keyframes = self._build_mapping_optimizer(
            replay_keyframes,
            pose_window=effective_pose_window,
            pose_lr_scale=effective_pose_lr_scale,
        )

        if training_cfg.get("check_vertex_grad_fd", False):
            print("[FD] Checking vertex gradients...")
            fd_errors = self._fd_check_vertex_grad(viewpoint_cam, num_samples=10)
            for v_idx, coord, num_grad, ana_grad, rel_err in fd_errors:
                print(f"  v={v_idx} c={coord}: num={num_grad:.6f} ana={ana_grad:.6f} rel_err={rel_err:.4f}")
            print(f"[FD] Checked {len(fd_errors)} vertex coords")

        mapping_steps = range(total_views)
        # pixel_losses = [0.0] * total_views

        # Replay past keyframes
        strips = []

        num_triangles = self.triangles._triangle_indices.shape[0]
        _dev = self.triangles.vertices.device
        vis_union = torch.zeros(num_triangles, device=_dev, dtype=torch.bool)
        frustum_union = torch.zeros(num_triangles, device=_dev, dtype=torch.bool)

        exposure_init_itr = 20
        initialization = True if view_iteration < exposure_init_itr else False


        init_mapping_iters = training_cfg.get("init_mapping_iters", 150)
        if viewpoint_cam.uid <= 1:
            effective_num_iters = init_mapping_iters

        # When retriangulation ran this keyframe, optionally use a dedicated
        # mapping-iteration count for the extra refinement of the map.
        rdel_mapping_iters = training_cfg.get("rdel_mapping_iters", None)
        if (
            did_rdel
            and rdel_mapping_iters is not None
            and not should_initial_ba
            and viewpoint_cam.uid > 1
        ):
            effective_num_iters = int(rdel_mapping_iters)


        self.triangles.n_obs.fill_(0)

        logging_loss_total = 0
        mapping_iter_losses = []

        triangle_pixel_counts = torch.zeros(num_triangles, device=self.triangles.vertices.device, dtype=torch.long)

        # Pre-load GT tensors to GPU once - avoids repeated CPU->GPU transfers inside the hot loop.
        _device = self.triangles.vertices.device
        for kf in replay_keyframes:
            kf._gt_image_gpu = kf.original_image.to(_device, non_blocking=True)
            kf._rgb_mask_gpu = kf.rgb_pixel_mask_mapping.to(_device, non_blocking=True)
            kf._gt_depth_gpu = kf.gt_depth.to(_device, non_blocking=True)

        for render_iter in range(effective_num_iters):
            loss_total = 0
            iter_loss_total = 0.0
            for i, kf in enumerate(replay_keyframes):
                render_pkg_kf = render(kf, self.triangles, self.pipe, self.background)

                pixel_count = render_pkg_kf["triangle_was_rendered"].detach()

                triangle_pixel_counts += pixel_count

                loss_kf = get_loss_mapping(self.config, render_pkg_kf["render"], render_pkg_kf["surf_depth"], kf, render_pkg_kf["rend_alpha"], initialization=initialization)
                equilateral_loss = equilateral_regularizer(self.triangles._triangle_indices, self.triangles.vertices)
                loss_equilateral = self.opt.lambda_equilateral * equilateral_loss

                normals = -render_pkg_kf["rend_normal"]
                surf_normals = -render_pkg_kf["surf_normal"]
                normal_error = (1 - (normals * surf_normals).sum(dim=0))[None]
                normal_loss =  normal_error.mean()
                loss_normal = self.opt.lambda_normals * normal_loss

                loss_normal_pseudo = torch.tensor(0.0, device=kf.original_image.device)

                if hasattr(kf, 'normal') and kf.normal is not None and kf.depth is not None:
                    bae_normals = kf.normal.to(device=normals.device)  # (3, H, W) in camera space
                    if bae_normals.dim() == 4:
                        bae_normals = bae_normals.squeeze(0)  # (3, H, W)

                    W2C = kf.world_view_transform # (4, 4) world-to-camera
                    R_w2c = W2C[:3, :3].T
                    H, W = bae_normals.shape[1], bae_normals.shape[2]
                    render_normals_cam = (R_w2c @ normals.view(3, -1)).view(3, H, W)
                    

                    gt_depth = torch.as_tensor(kf.depth, dtype=torch.float32, device=normals.device)
                    depth_mask = (gt_depth > 0.01).view(1, H, W).float()
                    normal_mask = (bae_normals.abs().sum(dim=0, keepdim=True) > 0.01).float()

                    normal_error_super = (1 - (render_normals_cam * bae_normals * depth_mask * normal_mask).sum(dim=0))[None]
                    loss_normal_pseudo = self.opt.lambda_normals_super * normal_error_super.mean()


                # Opacity regularizer: pushes vertex opacities toward 0 or 1
                lambda_opacity = self.config.get("opt_params", {}).get("lambda_opacity", 0.0)
                if lambda_opacity > 0:
                    loss_opacity = lambda_opacity * torch.abs(self.triangles.get_vertex_weight).mean()
                else:
                    loss_opacity = 0.0

                current_kf_loss = loss_kf + loss_normal_pseudo + loss_normal + loss_equilateral + loss_opacity
                current_kf_loss.backward()
         
                image_size = render_pkg_kf["scaling"].detach()
                mask = image_size > self.triangles.image_size
                self.triangles.image_size[mask] = image_size[mask]

                importance_score = render_pkg_kf["max_blending"].detach()
                mask = importance_score > self.triangles.importance_score
                self.triangles.importance_score[mask] = importance_score[mask]

                self.keyframe_buffer.update_visibility(kf.uid, render_pkg_kf["curr_visibility"])
                vis_union = torch.logical_or(vis_union, render_pkg_kf["curr_visibility"])
                if render_iter == 0:
                    self.triangles.n_obs += render_pkg_kf["curr_visibility"].long()
                    # Frustum membership: centroid z > 0.2 in camera space (matches CUDA in_frustum_triangle threshold).
                    with torch.no_grad():
                        tri_verts = self.triangles.vertices[self.triangles._triangle_indices]  # (T, 3, 3)
                        centroids = tri_verts.mean(dim=1)  # (T, 3)
                        ones = torch.ones(centroids.shape[0], 1, device=centroids.device)
                        c_hom = torch.cat([centroids, ones], dim=1)  # (T, 4)
                        # world_view_transform = T_w2c.T; c_hom @ wvt gives camera-space coords
                        c_cam = c_hom @ kf.world_view_transform  # (T, 4)
                        frustum_union = frustum_union | (c_cam[:, 2] > 0.2)


                if self.config["Results"]["use_rerun"]:
                    p_cur = viewpoint_cam.camera_center.detach().cpu().numpy().reshape(3)
                    p_kf  = kf.camera_center.detach().cpu().numpy().reshape(3)
                    strips.append(np.array([p_cur, p_kf], dtype=np.float32))
                    rr.log(
                        "/strips",
                        rr.LineStrips3D(strips=strips, colors=[0, 255, 0], radii=0.005),
                        static=True,
                    )

                    if i == 0:
                           # log image
                        image = render_pkg_kf["render"]
                        image_ab = (torch.exp(viewpoint_cam.exposure_a)) * image + viewpoint_cam.exposure_b
                        rgb_tracked = (
                            (torch.clamp(image_ab, min=0, max=1.0) * 255)
                            .byte()
                            .permute(1, 2, 0)
                            .contiguous()
                            .cpu()
                            .numpy()
                        )
                        rr.log("mapping/image", rr.Image(rgb_tracked), static=False)

                    logging_loss_total += current_kf_loss.item()
                    iter_loss_total += current_kf_loss.item()


                 

            mapping_iter_losses.append(iter_loss_total / max(1, len(replay_keyframes)))
            last_image_2D = render_pkg_kf.get("image_2D") if render_pkg_kf else None
            self.triangles.add_densification_stats_triangles(vis_union, image_2D=last_image_2D)
            self.triangles.optimizer.step()
            self.triangles.optimizer.zero_grad(set_to_none=True)
            # Clamp features to reduce bright/saturated artifacts in less-observed regions
            clamp = self.config.get("opt_params", {}).get("feature_dc_clamp", 0)
            if clamp > 0:
                with torch.no_grad():
                    self.triangles._features_dc.clamp_(-clamp, clamp)
            if keyframe_optimizer is not None:
                keyframe_optimizer.step()
                keyframe_optimizer.zero_grad(set_to_none=True)

            for kf in optimized_keyframes:
                if not kf.uid == 0:
                    update_pose(kf)

        # Release pre-loaded GPU tensors.
        for kf in replay_keyframes:
            for _attr in ("_gt_image_gpu", "_rgb_mask_gpu", "_gt_depth_gpu"):
                if hasattr(kf, _attr):
                    delattr(kf, _attr)

        if should_initial_ba:
            post_ba_metrics = self.evaluate_photometric_objective(
                pose_window=effective_pose_window,
                keyframes=replay_keyframes,
            )
            pre_photo = max(pre_ba_metrics["mean_photo_obj"], 1e-9)
            rel_gain = (
                (pre_ba_metrics["mean_photo_obj"] - post_ba_metrics["mean_photo_obj"])
                / pre_photo
                * 100.0
            )
            print(
                f"[initial BA][after]  kfs={post_ba_metrics['count']} "
                f"photo={post_ba_metrics['mean_photo_obj']:.6f} "
                f"l1={post_ba_metrics['mean_l1']:.6f} "
                f"dssim={post_ba_metrics['mean_dssim_term']:.6f} "
                f"gain={rel_gain:+.2f}%"
            )
            self.initial_ba_done = True


        
        if self.config["Results"]["use_rerun"] and view_iteration % log_every == 0:
            log_mesh_rerun(self.triangles, viewpoint_cam, use_rerun=self.config["Results"]["use_rerun"], static=False)
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # print("pixel_losses: ", pixel_losses)

            pixel_losses = {}
            with torch.no_grad():
                for kf in self.keyframe_buffer:
                    render_pkg_kf = render(kf, self.triangles, self.pipe, self.background)
                    loss_kf = get_loss_mapping(self.config, render_pkg_kf["render"], render_pkg_kf["expected_depth"], kf, render_pkg_kf["rend_alpha"], initialization=False)
                    pixel_losses[kf.uid] = pixel_losses.get(kf.uid, 0.0) + loss_kf.item()


            average_mapping_loss = logging_loss_total / (effective_num_iters * len(replay_keyframes))

            fig, ax = plt.subplots(figsize=(6, 4))
            sorted_uids = sorted(pixel_losses.keys())
            sorted_losses = [pixel_losses[uid] for uid in sorted_uids]
            ax.plot(sorted_uids, sorted_losses, label="Per-keyframe loss")
            # Plot average mapping loss for the replay window as a horizontal red line
            ax.axhline(
                average_mapping_loss, color="red", linestyle="--", linewidth=2, label="Avg. mapping loss (replay window)"
            )
            ax.set_xlabel("Camera UID")
            ax.set_ylabel("Photometric loss")
            ax.set_title("Mapping photometric loss")
            ax.set_ylim(0, .2)
            ax.set_xlim(0, total_views)
            ax.legend()
            fig.tight_layout()
            fig.canvas.draw()
            img = np.asarray(fig.canvas.buffer_rgba())
            img = img[:, :, :3].copy()  # RGBA -> RGB
            plt.close(fig)
            rr.log("mapping/loss_curve", rr.Image(img), static=False)




        return vis_union, render_pkg_kf, triangle_pixel_counts, mapping_iter_losses, frustum_union
