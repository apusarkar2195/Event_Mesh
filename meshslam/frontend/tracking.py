"""
Tracking: initialise pose from previous frame, optimise pose to convergence.
Uses the current map (triangles) for rendering; designed so it can later receive
map updates from a separate backend process.
"""
from typing import Optional, Callable, List

import torch

from mesh_splatting.triangle_renderer import render
from mesh_splatting.utils.graphics_utils import getWorld2View2
from utils.slam_utils import get_loss_tracking, get_median_depth, get_loss_tracking_test
from utils.pose_utils import update_pose

from ..backend.map_state import MapState
import numpy as np

from contextlib import contextmanager

import lietorch

@contextmanager
def detach_triangles(triangles):
    
    """Context manager to temporarily detach triangle parameters."""
    # Store original requires_grad states
    original_states = {
        'vertices': triangles.vertices.requires_grad,
        'vertex_weight': triangles.vertex_weight.requires_grad,
        'features_dc': triangles._features_dc.requires_grad,
        'features_rest': triangles._features_rest.requires_grad,
    }
    
    # Detach
    triangles.vertices = triangles.vertices.detach()
    triangles.vertex_weight = triangles.vertex_weight.detach()
    triangles._features_dc = triangles._features_dc.detach()
    triangles._features_rest = triangles._features_rest.detach()
    
    try:
        yield
    finally:
        # Restore
        triangles.vertices.requires_grad_(original_states['vertices'])
        triangles.vertex_weight.requires_grad_(original_states['vertex_weight'])
        triangles._features_dc.requires_grad_(original_states['features_dc'])
        triangles._features_rest.requires_grad_(original_states['features_rest'])


class TrackingEngine:
    """Frontend: pose initialisation and tracking (optimise pose w.r.t. current frame)."""

    def __init__(self, map_state: MapState, pipe, config: dict):
        self.map_state = map_state
        self.pipe = pipe
        self.config = config
        self.triangles = map_state.triangles
        self.background = map_state.background
        self.pose_optimizer = None  # Will hold the optimizer
        self.pose_optimizer_state = None  # Will hold state_dict when not keyframing

    def init_pose_from_previous(
        self, current_viewpoint, previous_viewpoint
    ) -> None:
        """Set current frame pose from the previous frame's optimised pose."""
        with torch.no_grad():
            current_viewpoint.update_RT(
                previous_viewpoint.R.clone(), previous_viewpoint.T.clone()
            )

            current_viewpoint.cam_rot_delta.data.copy_(previous_viewpoint.cam_rot_delta.data)
            current_viewpoint.cam_trans_delta.data.copy_(previous_viewpoint.cam_trans_delta.data)
            current_viewpoint.exposure_a.data.copy_(previous_viewpoint.exposure_a.data)
            current_viewpoint.exposure_b.data.copy_(previous_viewpoint.exposure_b.data)
            # if hasattr(current_viewpoint, "cam_rot_delta") and current_viewpoint.cam_rot_delta is not None:
            #     current_viewpoint.cam_rot_delta.data.zero_()
            # if hasattr(current_viewpoint, "cam_trans_delta") and current_viewpoint.cam_trans_delta is not None:
            #     current_viewpoint.cam_trans_delta.data.zero_()

    def run_tracking(
        self,
        viewpoint,
        cur_frame_idx: int,
        tracking_itr_num: Optional[int] = None,
        use_fd_grad: bool = False,
        viz_callback: Optional[Callable] = None,
    ) -> dict:
        """
        Optimise camera pose for the current frame: render, tracking loss, backward, update_pose.
        Returns the last render_pkg (e.g. for median depth).
        """
        if tracking_itr_num is None:
            tracking_itr_num = self.config["Training"]["tracking_itr_num"]

        # lr_cam = self.config["Training"]["lr"]
        # opt_params = [
        #     {"params": [viewpoint.cam_rot_delta], "lr": lr_cam["cam_rot_delta"], "name": f"rot_{viewpoint.uid}"},
        #     {"params": [viewpoint.cam_trans_delta], "lr": lr_cam["cam_trans_delta"], "name": f"trans_{viewpoint.uid}"},
        #     {"params": [viewpoint.exposure_a], "lr": 0.01, "name": f"exposure_a_{viewpoint.uid}"},
        #     {"params": [viewpoint.exposure_b], "lr": 0.01, "name": f"exposure_b_{viewpoint.uid}"},
        # ]

        lr_cam = self.config["Training"]["lr"]
        opt_params = [
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": lr_cam["cam_rot_delta"],
                "name": f"rot_{viewpoint.uid}",
            },
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": lr_cam["cam_trans_delta"],
                "name": f"trans_{viewpoint.uid}",
            },
            {"params": [viewpoint.exposure_a], "lr": 0.01, "name": f"exposure_a_{viewpoint.uid}"},
            {"params": [viewpoint.exposure_b], "lr": 0.01, "name": f"exposure_b_{viewpoint.uid}"},
        ]
        pose_optimizer = torch.optim.Adam(opt_params)

    
        # if not is_previous_kf and self.pose_optimizer_state is not None:
        #     pose_optimizer.load_state_dict(self.pose_optimizer_state)
    

   
        tracking_losses: List[float] = []
        tracking_early_stop = self.config.get("Training", {}).get("tracking_early_stop", True)
        # with detach_triangles(self.triangles):
        for tracking_itr in range(tracking_itr_num):
            render_pkg = render(
                viewpoint, self.triangles, self.pipe, self.background,
                optimize_pose_only=True, # TODO: measure with and without
            )

            image = render_pkg["render"]
            depth = render_pkg["expected_depth"]
            opacity = render_pkg["rend_alpha"]
            median_depth = render_pkg["render_depth_median"]
            normals = render_pkg["rend_normal"]
            surf_normals = render_pkg["surf_normal"]
            # normal_error = (1 - (normals * surf_normals).sum(dim=0))[None]
            # normal_loss = self.opt.lambda_normals * normal_error.mean()

            pose_optimizer.zero_grad()
            loss_tracking, l1 = get_loss_tracking(
                self.config, image, depth, opacity, viewpoint
            )
            loss_tracking.backward()

            if use_fd_grad:

                fd_errors = self._fd_check_pose_grad(viewpoint, eps=1e-4)
                for i, num_grad, ana_grad, rel_err in fd_errors:
                    if i < 3:
                        viewpoint.cam_trans_delta.data.zero_()
                        viewpoint.cam_trans_delta.grad[i] = num_grad
                    else:
                        viewpoint.cam_rot_delta.data.zero_()
                        viewpoint.cam_rot_delta.grad[i - 3] = num_grad

                    flag = " [FD unreliable: |grad|<0.01]" if abs(ana_grad) < 0.01 else ""
                    print(f"FD error: {i}, num: {num_grad}, ana: {ana_grad}, rel: {rel_err}{flag}")

                print("vertices grad: ", self.triangles.vertices.grad)
                fd_vertex_errors = self._fd_check_vertex_grad(viewpoint, eps=1e-3)
                for v_idx, coord, num_grad, ana_grad, rel_err in fd_vertex_errors:
                    if self.triangles.vertices.grad is not None:
                        self.triangles.vertices.grad[v_idx, coord] = num_grad
                    print(f"FD vertex v={v_idx} c={coord}: num={num_grad}, ana={ana_grad}, rel={rel_err}")

            tracking_losses.append(loss_tracking.item())
   
            with torch.no_grad():
                pose_optimizer.step()
                converged = update_pose(viewpoint)
            if converged and tracking_early_stop:
                break

            if viz_callback:
                viz_callback(render_pkg, viewpoint, l1)

        self.median_depth = get_median_depth(median_depth, opacity)

        return render_pkg, l1, tracking_losses

    def _fd_check_pose_grad(self, viewpoint, eps: float = 1e-5) -> list:
        """Finite-difference check of pose gradients. Returns list of (i, num_grad, ana_grad, rel_err)."""
        R_orig = viewpoint.R.clone()
        T_orig = viewpoint.T.clone()

        def loss_fn(vp, gt_image=None):
            from utils.pose_utils import SE3_exp
            tau = torch.cat([vp.cam_trans_delta, vp.cam_rot_delta], axis=0)
            T_w2c_base = getWorld2View2(R_orig, T_orig)
            T_w2c_eff = lietorch.SE3.exp(tau).matrix() @ T_w2c_base
            vp.update_RT(T_w2c_eff[:3, :3], T_w2c_eff[:3, 3])
            render_pkg = render(vp, self.triangles, self.pipe, self.background)
            image = render_pkg["render"] if gt_image is None else gt_image
            depth = render_pkg["expected_depth"]
            opacity = render_pkg["rend_alpha"]
            loss, _ = get_loss_tracking(self.config, image, depth, opacity, vp)
      
            return loss

        g_trans = viewpoint.cam_trans_delta.grad.clone().flatten()
        g_rot = viewpoint.cam_rot_delta.grad.clone().flatten()
        g_ana = torch.cat([g_trans, g_rot])
        trans = viewpoint.cam_trans_delta.data
        rot = viewpoint.cam_rot_delta.data
        errors = []

        for i in range(6):
            param, j = (trans, i) if i < 3 else (rot, i - 3)
            orig_val = param.flatten()[j].item()
            param.flatten()[j] = orig_val + eps
            L_plus = loss_fn(viewpoint).item()
            param.flatten()[j] = orig_val - eps
            L_minus = loss_fn(viewpoint).item()
            param.flatten()[j] = orig_val
            num_grad = (L_plus - L_minus) / (2.0 * eps)
            ana_grad = g_ana[i].item()
            rel = abs(num_grad - ana_grad) / (abs(ana_grad) + 1e-9)
            errors.append((i, num_grad, ana_grad, rel))

        viewpoint.update_RT(R_orig, T_orig)
        return errors

    def _fd_check_vertex_grad(
        self,
        viewpoint,
        num_samples: int = 20,
        eps: float = 1e-5,
    ) -> list:
        """
        Finite-difference check of vertex gradients. Call when triangles.vertices
        has requires_grad=True (e.g. from mapping, or tracking without detach_triangles).

        Returns list of (v_idx, coord, num_grad, ana_grad, rel_err).
        """
        # self.triangles.vertices.data = self.triangles.vertices.data.double()
        vertices = self.triangles.vertices
        if not vertices.requires_grad:
            raise RuntimeError("vertices must have requires_grad=True for FD check")

        # Get analytical gradient: forward + backward
        render_pkg = render(viewpoint, self.triangles, self.pipe, self.background)
        loss, _ = get_loss_tracking(
            self.config,
            render_pkg["render"],
            render_pkg["expected_depth"],
            render_pkg["rend_alpha"],
            viewpoint,
        )
        loss.backward()

        g_ana = vertices.grad.clone()
        # Keep vertices.grad so caller can overwrite sampled entries with num_grad

        def loss_fn():
            rp = render(viewpoint, self.triangles, self.pipe, self.background)
            l, _ = get_loss_tracking(
                self.config,
                rp["render"],
                rp["expected_depth"],
                rp["rend_alpha"],
                viewpoint,
            )
            return l

        # Prefer vertices that were rendered (more likely to have non-zero grad)
        vertex_rendered = render_pkg["vertex_rendered"]
        rendered_idx = torch.where(vertex_rendered > 0)[0]
        if rendered_idx.numel() > 0:
            # Sample from rendered vertices
            n = min(num_samples, rendered_idx.numel())
            perm = torch.randperm(rendered_idx.numel(), device=rendered_idx.device)[:n]
            sample_idx = rendered_idx[perm]
        else:
            # Fallback: random sample
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

    def is_keyframe(self, current_viewpoint, current_render_pkg) -> bool:
        kf_translation = self.config["Training"]["kf_translation"]
        kf_min_translation = self.config["Training"]["kf_min_translation"]
        kf_overlap = self.config["Training"]["kf_overlap"]

        last_keyframe = self.map_state.keyframe_buffer.get_last_keyframe()
        # add first keyframe
        if last_keyframe is None:
            print("No keyframe found, adding current frame as keyframe")
            return True
        if current_render_pkg is None:
            print("No render package found, not adding current frame as keyframe")
            return False
        if current_render_pkg["curr_visibility"] is None:
            print("No visibility found for current frame, not adding current frame as keyframe")
            return False
        last_kf_visibility = self.map_state.keyframe_buffer.get_visibility(last_keyframe.uid)
        if last_kf_visibility is None:
            print("No visibility found for last keyframe, adding current frame as keyframe")
            return True


        # dist = torch.norm((current_pose_C2W @ last_pose_C2W)[0:3, 3])
        dist = torch.norm(current_viewpoint.camera_center - last_keyframe.camera_center)
        dist_check = dist > kf_translation * self.median_depth
        dist_check2 = dist > kf_min_translation * self.median_depth
        
        union = torch.logical_or(
            current_render_pkg["curr_visibility"], last_kf_visibility
        ).count_nonzero()
        intersection = torch.logical_and(
            current_render_pkg["curr_visibility"], last_kf_visibility
        ).count_nonzero()
        point_ratio_2 = intersection / union
        print(f"point_ratio_2: {point_ratio_2}, dist_check2: {dist_check2}, dist_check: {dist_check}")

        # check nan
        if torch.isnan(point_ratio_2):
            print("point_ratio_2 is nan, not adding current frame as keyframe")
            return True
        
        return (point_ratio_2 < kf_overlap and dist_check2) or dist_check
