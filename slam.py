"""
Triangle Splatting SLAM
Single-threaded for now; structured so backend and frontend can later run in
separate processes (map state / pose messages as interface).
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from utils import keyframe_buffer

from utils.normal_utils import Depth2normal

import contextlib
import copy
import time
from argparse import ArgumentParser

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import torch

from mesh_splatting.scene.triangle_model import TriangleModel
from mesh_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from mesh_splatting.utils.sh_utils import eval_sh
from mesh_splatting.triangle_renderer import render

from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.gui_utils import rerun_show_cam, log_trajectories_rerun, log_mesh_rerun
from utils.cameras import Camera
from utils.pose_utils import SE3_exp
from utils.keyframe_buffer import KeyframeBuffer
from utils.eval_utils import compute_ate_se3_from_cameras, eval_ate, Log, eval_rendering

from utils.profiler import SLAMProfiler
from mesh_splatting.arguments import ModelParams, PipelineParams, OptimizationParams

from meshslam.backend import MapState, MappingEngine
from meshslam.frontend import TrackingEngine

import json
import os
import yaml
from datetime import datetime
import wandb
from pathlib import Path
from mesh_splatting.utils.system_utils import mkdir_p

import multiprocessing as mp
from meshslam.gui.gui_utils import TrianglePacket, ParamsGUI, FakeQueue



from utils.normal_utils import normal_to_rgb

# from fused_ssim import fused_ssim



SLAMProfiler_ = SLAMProfiler()


class LazyCameraProvider:
    """Dict camera store that lazily creates cameras on access,
    while exposing __len__ and __getitem__ so eval functions work
    as if it were a list."""

    def __init__(self, cameras_dict, total_frames, get_camera_fn):
        self._cameras = cameras_dict
        self._total = total_frames
        self._get = get_camera_fn

    def __len__(self):
        return self._total

    def __getitem__(self, idx):
        return self._get(idx)

    def __contains__(self, idx):
        return 0 <= idx < self._total

    def __iter__(self):
        for i in range(self._total):
            yield self._get(i)


class MeshSLAMTrainer:
    """
    owns dataset, cameras, config; creates and runs
    backend (mapping + keyframes) and frontend (tracking) in sequence.
    """

    def __init__(self, args, config, dataset_path, config_path, opt, pipe, num_frames=10,
                 resume_from=None, resume_iteration=None):
        self.args = args
        self.config = config
        self.dataset_path = dataset_path
        self.config_path = config_path
        self.opt = opt
        self.pipe = pipe
        self.resume_from = resume_from
        self.resume_iteration = resume_iteration or 0

        self.dataset = load_dataset(args, dataset_path, config)

        print("Dataset length:", len(self.dataset))

        self.projection_matrix = self._setup_projection_matrix()

        self.monocular = self.config["Dataset"]["sensor_type"] == "monocular"
        self.total_frames = min(num_frames or len(self.dataset.poses), len(self.dataset.poses))
        self.cameras = {}

        # Background and map state (backend owns triangles + keyframes)
        if opt.random_background:
            self.background = torch.rand(3, device="cuda")
        elif config.get("model_params", {}).get("white_background", False):
            self.background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        else:
            self.background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        self.timings = {}
        self.loss_history = []
        self._tracking_final_losses = []  # (frame_idx, final_loss) per tracked frame
        self._mapping_final_losses = []   # (frame_idx, final_loss) per keyframe
        self._loss_log_path = os.path.join(config["Results"]["save_dir"], "loss_log.jsonl")

        # GUI process (Open3D viewer, alternative to rerun)
        # Must use 'spawn' - CUDA is already initialised in the main process,
        # and Linux's default 'fork' cannot re-initialise CUDA in the child.
        self._use_gui = self.config.get("Results", {}).get("use_gui", False)
        if self._use_gui:
            _mp_ctx = mp.get_context("spawn")
            self._q_main2vis = _mp_ctx.Queue()
            self._q_vis2main = _mp_ctx.Queue()
            _params_gui = ParamsGUI(
                pipe=pipe,
                background=self.background.cpu(),
                q_main2vis=self._q_main2vis,
                q_vis2main=self._q_vis2main,
            )
            from meshslam.gui import slam_gui
            self._gui_process = _mp_ctx.Process(
                target=slam_gui.run, args=(_params_gui,), daemon=True
            )
            self._gui_process.start()
            time.sleep(3)  # wait for Open3D window to initialise
        else:
            self._q_main2vis = FakeQueue()
            self._q_vis2main = FakeQueue()
        self._freeze_vertices = False
        self._gui_process = None

        # Build map: triangles + keyframe buffer
        if self.resume_from:
            self._initialize_from_checkpoint()
        else:
            self._initialize_model()
        self.depth2normal = Depth2normal(d_min=0.0, d_max=10.0, k=4, d=1, min_nghbr=4, gamma=0.05, gamma_exception=False)
        maxlen = self.config.get("Training", {}).get("window_size", None)

        keyframe_buffer = KeyframeBuffer(maxlen=maxlen, device="cuda")
        self.map_state = MapState(
            triangles=self.triangles,
            background=self.background,
            keyframe_buffer=keyframe_buffer,
            device=torch.device("cuda"),
            timings=self.timings,
            loss_history=self.loss_history,
            model_path=self.config["Results"]["save_dir"],
        )

        # Backend and frontend (same process for now)
        self.backend = MappingEngine(self.map_state, opt, pipe, config)
        self.frontend = TrackingEngine(self.map_state, pipe, config)

        # Convenience alias for code that still references self.triangles
        self.triangles = self.map_state.triangles

        if self.resume_from:
            self._load_keyframes_from_checkpoint()

    def _initialize_from_checkpoint(self):
        """Load triangles from checkpoint for SLAM resume."""
        sh_degree = 0
        self.triangles = TriangleModel(sh_degree=sh_degree, config=self.config)
        self.triangles.load_parameters_for_slam_resume(self.resume_from, device="cuda")
        self.triangles.training_setup(
            self.opt,
            self.opt.feature_lr,
            self.opt.weight_lr,
            self.opt.lr_triangles_points_init,
        )
        self.triangles.add_percentage = self.opt.add_percentage
        self.triangles.size_probs_zero = self.opt.size_probs_zero
        self.triangles.size_probs_zero_image_space = self.opt.size_probs_zero_image_space
        self.triangles.scaling = self.opt.upscaling_factor
        Log("Initialized from checkpoint:", self.resume_from)

    def _load_keyframes_from_checkpoint(self):
        """Reconstruct keyframe buffer from saved keyframes_state.pt and update visibilities."""
        kf_path = os.path.join(self.resume_from, "keyframes_state.pt")
        if not os.path.isfile(kf_path):
            Log("Warning: keyframes_state.pt not found, keyframe buffer will be empty")
            return
        payload = torch.load(kf_path, map_location="cpu")
        keyframe_state = payload.get("keyframes", [])
        if not keyframe_state:
            return
        for kf_dict in keyframe_state:
            uid = int(kf_dict["uid"])
            if uid < 0 or uid >= self.total_frames:
                continue
            cam = self._get_camera(uid)
            R = kf_dict["R"].to(device=cam.device)
            T = kf_dict["T"].to(device=cam.device)
            cam.update_RT(R, T)
            render_pkg = render(cam, self.triangles, self.pipe, self.background)
            self.backend.keyframe_buffer.add_with_visibility(cam, render_pkg)
        self.backend.keyframe_buffer.pad_visibilities(self.triangles._triangle_indices.shape[0])
        Log("Loaded", len(keyframe_state), "keyframes from checkpoint")

    def _setup_projection_matrix(self):
        return getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.dataset.fx,
            fy=self.dataset.fy,
            cx=self.dataset.cx,
            cy=self.dataset.cy,
            W=self.dataset.width,
            H=self.dataset.height,
        ).transpose(0, 1)

    def _get_camera(self, idx):
        """Lazily create and cache a Camera for frame idx."""
        if idx in self.cameras:
            return self.cameras[idx]
        camera = Camera.init_from_dataset(self.dataset, idx, self.projection_matrix)
        camera.compute_grad_mask(self.config)
        if self.monocular:
            camera.depth = None
        self.cameras[idx] = camera
        return camera

    def _initialize_model(self):
        t0 = time.perf_counter()
        sh_degree = 0
        self.triangles = TriangleModel(sh_degree=sh_degree, config=self.config)

        start_frame = getattr(self.args, "start_frame", 0) or 0
        cam0 = self._get_camera(start_frame)
        pcd, features, opacities, radii = self.triangles.create_pcd_from_image(
            cam_info=cam0, init=True, depthmap=cam0.depth
        )
        
        print(f"Point cloud shape: {pcd.points.shape}")
        self.triangles.create_from_pcd(pcd, radii, self.opt.set_weight, self.opt.set_sigma)
   
        self.triangles.training_setup(
            self.opt,
            self.opt.feature_lr,
            self.opt.weight_lr,
            self.opt.lr_triangles_points_init,
        )

        self.triangles.add_percentage = self.opt.add_percentage
        self.triangles.size_probs_zero = self.opt.size_probs_zero
        self.triangles.size_probs_zero_image_space = self.opt.size_probs_zero_image_space
        self.triangles.scaling = self.opt.upscaling_factor
            

    # -------------------------------------------------------------------------
    # Visualization and reporting 
    # -------------------------------------------------------------------------



    def _log_visualizations(self, render_pkg, viewpoint_cam, name="", static=True):
        if not self.config.get("Results", {}).get("use_rerun", True):
            return
        image = render_pkg["render"]
        image_ab = (torch.exp(viewpoint_cam.exposure_a)) * image + viewpoint_cam.exposure_b
        gt_image = viewpoint_cam.original_image
        gt_image = (
            (torch.clamp(gt_image, min=0, max=1.0) * 255)
            .byte()
            .permute(1, 2, 0)
            .contiguous()
            .cpu()
            .numpy()
        )
        rgb_render = (
            (torch.clamp(image_ab, min=0, max=1.0) * 255)
            .byte()
            .permute(1, 2, 0)
            .contiguous()
            .cpu()
            .numpy()
        )
        rerun_show_cam(f"/{name}/cam_{viewpoint_cam.uid}", rgb_render, viewpoint_cam, static=static)
        rr.log(f"/{name}/gt_image", rr.Image(gt_image), static=static)
     
        # rr.log(
        #     "/points",
        #     rr.Points3D(
        #         positions=triangles_points.cpu().detach().flatten(0, 1),
        #         colors=((colors_precomp * 255).to(torch.uint8)).cpu().detach(),
        #     ),
        # )

    def _log_tracked_cam(self, render_pkg, viewpoint_cam, l1, static=True):
        if not self.config["Results"]["use_rerun"]:
            return
        image = render_pkg["render"]
        image_ab = (torch.exp(viewpoint_cam.exposure_a)) * image + viewpoint_cam.exposure_b
        rgb_tracked = (
            (torch.clamp(image_ab, min=0, max=1.0) * 255)
            .byte()
            .permute(1, 2, 0)
            .contiguous()
            .cpu()
            .numpy()
        )
        rerun_show_cam("/cam_tracked", rgb_tracked, viewpoint_cam, static=static)
        gt_image = viewpoint_cam.original_image
        gt_image = (
            (torch.clamp(gt_image, min=0, max=1.0) * 255)
            .byte()
            .permute(1, 2, 0)
            .contiguous()
            .cpu()
            .numpy()
        )
        rr.log("/gt_image_tracked", rr.Image(gt_image), static=static)
        # print("l1: ", l1.detach().cpu().numpy())

        l1_t = l1.detach().cpu()
        while l1_t.dim() > 3:
            l1_t = l1_t.squeeze(0)
        l1_np = l1_t.permute(1, 2, 0).numpy()
        l1_range = l1_np.max() - l1_np.min()
        l1_np = (l1_np - l1_np.min()) / (l1_range + 1e-8) * 255
        rr.log("/l1_tracked", rr.Image(l1_np.astype(np.uint8)), static=True)



    def _print_timings(self):
        print("\n" + "=" * 50)
        print("Timing summary")
        print("=" * 50)
        for name in [
            "init_create_pcd",
            "init_create_from_pcd",
            "init_run_restricted_delaunay",
            "init_training_setup",
            "init_total",
        ]:
            if name in self.timings:
                print(f"  {name}: {self.timings[name]:.3f}s")
        for name in ["train_iter_avg_s", "train_view_avg_s", "train_total_s"]:
            if name in self.timings:
                print(f"  {name}: {self.timings[name]:.3f}s")
        print("=" * 50 + "\n")

    def _plot_loss_curve(self, save_path="loss_curve.png"):
        if not self.loss_history:
            return
        steps = range(1, len(self.loss_history) + 1)
        total = [h["total_loss"] for h in self.loss_history]
        pixel = [h["pixel_loss"] for h in self.loss_history]
        opacity = [h["opacity_loss"] for h in self.loss_history]
        normal = [h["normal_loss"] for h in self.loss_history]
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes[0, 0].plot(steps, total, "b-", linewidth=0.5)
        axes[0, 0].set_title("Total loss")
        axes[0, 0].set_xlabel("Iteration")
        axes[0, 1].plot(steps, pixel, "g-", linewidth=0.5)
        axes[0, 1].set_title("Pixel (L1) loss")
        axes[0, 1].set_xlabel("Iteration")
        axes[1, 0].plot(steps, opacity, "r-", linewidth=0.5)
        axes[1, 0].set_title("Opacity loss")
        axes[1, 0].set_xlabel("Iteration")
        axes[1, 1].plot(steps, normal, "m-", linewidth=0.5)
        axes[1, 1].set_title("Normal loss")
        axes[1, 1].set_xlabel("Iteration")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Loss curve saved to {save_path}")

    def _plot_convergence_curve(self, losses, title, xlabel="Iteration", ylabel="Loss"):
        """Render a convergence curve to a numpy RGB image for rerun."""
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(losses, linewidth=1)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        return img

    def _plot_loss_trend(self, tracking_data, mapping_data):
        """Render per-frame tracking and per-keyframe mapping final-loss trends."""
        fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        if tracking_data:
            frames, vals = zip(*tracking_data)
            axes[0].plot(frames, vals, linewidth=1)
        axes[0].set_title("Tracking final loss / frame", fontsize=9)
        axes[0].set_xlabel("Frame idx", fontsize=8)
        axes[0].set_ylabel("Loss", fontsize=8)
        axes[0].grid(True, alpha=0.3)
        if mapping_data:
            frames, vals = zip(*mapping_data)
            axes[1].plot(frames, vals, linewidth=1, color="orange")
        axes[1].set_title("Mapping final loss / keyframe", fontsize=9)
        axes[1].set_xlabel("Frame idx", fontsize=8)
        axes[1].set_ylabel("Loss", fontsize=8)
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        return img

    def _record_loss(self, loss_type, frame_idx, view_iter, losses):
        """Append one loss record to the JSONL log file."""
        record = {
            "type": loss_type,
            "frame_idx": frame_idx,
            "view_iter": view_iter,
            "num_iters": len(losses),
            "final_loss": losses[-1] if losses else None,
            "losses": losses,
        }
        mkdir_p(self.config["Results"]["save_dir"])
        with open(self._loss_log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _perturb_pose(self, viewpoint_cam, noise_scale=0.05):
        """Perturb current pose for tracking test (e.g. from GT)."""
        with torch.no_grad():
            T_gt = torch.eye(4, device=viewpoint_cam.device)
            T_gt[:3, :3] = viewpoint_cam.gt_T[:3, :3]
            T_gt[:3, 3] = viewpoint_cam.gt_T[:3, 3]
            noise_trans = torch.randn(3, device=viewpoint_cam.device) * noise_scale
            noise_rot = torch.randn(3, device=viewpoint_cam.device) * noise_scale
            noise_tau = torch.cat([noise_trans, noise_rot], dim=0)
            T_noisy = SE3_exp(noise_tau) @ T_gt
            viewpoint_cam.update_RT(T_noisy[:3, :3], T_noisy[:3, 3])

    # -------------------------------------------------------------------------
    # GUI signal handling (shared between pause loop and normal run)
    # -------------------------------------------------------------------------

    def _handle_gui_signal(self, vis_pkt, render_for, replay_size, viz_cb, log_every, view_iteration, total_views):
        """Process a single GUI packet: freeze, retriangulate, mapping."""
        # On pause, send the exact current mesh as an editing base so the GUI's
        # vertex order/count line up 1:1 for live (index-based) deform edits.
        if getattr(vis_pkt, "flag_pause", False):
            try:
                last_kf = self.backend.keyframe_buffer.get_last_keyframe()
                base = TrianglePacket(
                    triangles=self.triangles,
                    current_frame=last_kf,
                    keyframes=list(self.map_state.keyframe_buffer),
                )
                base.edit_base = True
                self._q_main2vis.put_nowait(base)
            except Exception:
                pass
        # Freeze / unfreeze vertex gradients (None = no change from this packet)
        freeze = getattr(vis_pkt, "flag_freeze_vertices", None)
        if freeze is not None and freeze != self._freeze_vertices:
            self._freeze_vertices = freeze
            for pg in self.triangles.optimizer.param_groups:
                if pg["name"] == "vertices":
                    if self._freeze_vertices:
                        pg["_saved_lr"] = pg["lr"]
                        pg["lr"] = 0.0
                        Log("Vertex gradients frozen", tag="GUI")
                    else:
                        pg["lr"] = pg.get("_saved_lr", pg["lr"])
                        Log("Vertex gradients unfrozen", tag="GUI")

        # Live deform: apply GUI vertex edits to the tracked mesh in place.
        # SLAM is paused during a live edit, so the count/order match the base.
        edited = getattr(vis_pkt, "edited_vertices", None)
        if edited is not None:
            try:
                v = torch.as_tensor(edited, dtype=self.triangles.vertices.dtype,
                                    device=self.triangles.vertices.device)
                if v.shape == self.triangles.vertices.shape:
                    with torch.no_grad():
                        self.triangles.vertices.data.copy_(v)
                    Log("Applied live vertex edit from GUI", tag="GUI")
                else:
                    Log(f"Dropped stale vertex edit ({tuple(v.shape)} vs "
                        f"{tuple(self.triangles.vertices.shape)}) - mesh changed",
                        tag="GUI")
            except Exception as e:
                Log(f"Vertex edit failed: {e}", tag="GUI")

        # Commit thrown physics objects to the tracked map (added as triangles,
        # then optimised by mapping like any other geometry).
        commit = getattr(vis_pkt, "commit_objects", None)
        if commit is not None:
            try:
                self._commit_objects_to_map(commit)
            except Exception as e:
                Log(f"Commit to map failed: {e}", tag="GUI")

        # Retriangulate
        if getattr(vis_pkt, "flag_retriangulate", False):
            fix_verts = getattr(vis_pkt, "retriangulate_fix_vertices", False)
            Log(f"Retriangulating (GUI request, fix_vertices={fix_verts})…", tag="GUI")
            self.triangles.run_restricted_delaunay()
            self.backend.keyframe_buffer.pad_visibilities(
                self.triangles._triangle_indices.shape[0]
            )
            Log("Retriangulation done.", tag="GUI")
            last_kf = self.backend.keyframe_buffer.get_last_keyframe()
            if last_kf is not None:
                saved_lr = None
                if fix_verts:
                    for pg in self.triangles.optimizer.param_groups:
                        if pg["name"] == "vertices":
                            saved_lr = pg["lr"]
                            pg["lr"] = 0.0
                            break
                training_cfg = self.config.get("Training", {})
                self.backend.run_mapping_iterations(
                    viewpoint_cam=last_kf,
                    num_iters=50, # init mapping iters
                    replay_size=replay_size,
                    pose_window=training_cfg["initial_ba"]["pose_window"],
                    pose_lr_scale=training_cfg.get("mapping_pose_lr_scale", 0.5),
                    loss_history=self.loss_history,
                    viz_callback=viz_cb,
                    log_every=log_every,
                    view_iteration=view_iteration,
                    total_views=total_views,
                )
                if saved_lr is not None:
                    for pg in self.triangles.optimizer.param_groups:
                        if pg["name"] == "vertices":
                            pg["lr"] = saved_lr
                            break
            # Send updated mesh to GUI as a new editing base (count changed).
            try:
                echo = TrianglePacket(
                    triangles=self.triangles,
                    current_frame=last_kf,
                    keyframes=list(self.map_state.keyframe_buffer),
                )
                echo.edit_base = True
                self._q_main2vis.put_nowait(echo)
            except Exception:
                pass

        # TSDF fusion from keyframe depth maps
        if getattr(vis_pkt, "flag_tsdf_fusion", False):
            voxel_size = getattr(vis_pkt, "tsdf_voxel_size", 0.02)
            Log(f"Running TSDF fusion (voxel={voxel_size:.3f}m)…", tag="GUI")
            tsdf_mesh = self._build_tsdf_mesh(voxel_length=voxel_size)
            if tsdf_mesh is not None:
                last_kf = self.backend.keyframe_buffer.get_last_keyframe()
                pkt = TrianglePacket(
                    triangles=self.triangles,
                    current_frame=last_kf,
                    keyframes=list(self.map_state.keyframe_buffer),
                )
                # Serialize as numpy - Open3D meshes can't be pickled across processes
                pkt.tsdf_mesh = {
                    "vertices": np.asarray(tsdf_mesh.vertices).copy(),
                    "triangles": np.asarray(tsdf_mesh.triangles).copy(),
                    "vertex_normals": np.asarray(tsdf_mesh.vertex_normals).copy(),
                    "vertex_colors": np.asarray(tsdf_mesh.vertex_colors).copy()
                        if tsdf_mesh.has_vertex_colors() else None,
                }
                try:
                    self._q_main2vis.put_nowait(pkt)
                except Exception:
                    pass
                Log("TSDF fusion done.", tag="GUI")

    def _commit_objects_to_map(self, commit):
        """Add GUI-thrown physics objects to the tracked mesh as triangles,
        so subsequent mapping optimises them. Uses the densification add path
        (keeps optimiser state and per-triangle accumulators consistent)."""
        from mesh_splatting.utils.sh_utils import RGB2SH

        device = self.triangles.vertices.device
        verts = torch.as_tensor(commit["verts"], dtype=torch.float32, device=device)
        faces = torch.as_tensor(commit["faces"], dtype=torch.int32, device=device)
        colors = torch.as_tensor(commit["colors"], dtype=torch.float32, device=device)
        n_v = verts.shape[0]
        if n_v == 0 or faces.shape[0] == 0:
            return

        # Offset face indices to point into the extended vertex array.
        v0 = self.triangles.vertices.shape[0]
        new_triangles = faces.to(torch.int64) + v0
        new_triangles = new_triangles.to(self.triangles._triangle_indices.dtype)

        # SH DC colour, opaque vertex weight (logit), zero higher-order SH.
        new_features_dc = RGB2SH(colors).unsqueeze(1)              # (N,1,3)
        f_rest_dim = self.triangles._features_rest.shape[1]
        new_features_rest = torch.zeros(n_v, f_rest_dim, 3, device=device)
        new_vertex_weight = torch.full((n_v, 1), 5.0, device=device)  # sigmoid(5)≈0.99

        last_kf = self.backend.keyframe_buffer.get_last_keyframe()
        kf_id = int(getattr(last_kf, "uid", 0)) if last_kf is not None else 0

        self.triangles.densification_postfix(
            verts, new_vertex_weight, new_features_dc, new_features_rest,
            new_triangles, kf_id=kf_id,
        )
        # Keep per-triangle keyframe visibility tensors aligned with the new count.
        self.backend.keyframe_buffer.pad_visibilities(
            self.triangles._triangle_indices.shape[0]
        )
        Log(f"Committed {n_v} verts / {faces.shape[0]} triangles to map "
            f"(total {self.triangles._triangle_indices.shape[0]})", tag="GUI")

        # Push the extended mesh back to the GUI as a new editing base
        # (vertex count changed, so the GUI must re-sync).
        try:
            echo = TrianglePacket(
                triangles=self.triangles,
                current_frame=last_kf,
                keyframes=list(self.map_state.keyframe_buffer),
            )
            echo.edit_base = True
            self._q_main2vis.put_nowait(echo)
        except Exception:
            pass

    def _build_tsdf_mesh(self, voxel_length=0.02, depth_trunc=5.0):
        """Build an Open3D mesh via TSDF fusion of all keyframe depth maps."""
        import open3d as o3d

        keyframes = list(self.backend.keyframe_buffer)
        if not keyframes:
            Log("No keyframes for TSDF fusion", tag="GUI")
            return None

        kf0 = keyframes[0]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            int(kf0.image_width), int(kf0.image_height),
            float(kf0.fx), float(kf0.fy), float(kf0.cx), float(kf0.cy),
        )

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_length,
            sdf_trunc=voxel_length * 3,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )
        for kf in keyframes:
            depth_np = kf.depth
            if depth_np is None:
                gt_d = getattr(kf, "gt_depth", None)
                if gt_d is not None:
                    depth_np = gt_d.squeeze().cpu().numpy()
            if depth_np is None:
                continue
            depth_np = np.ascontiguousarray(depth_np.astype(np.float32))

            # Render the learned colors from the current mesh for this keyframe
            with torch.no_grad():
                render_pkg = render(kf, self.triangles, self.pipe, self.background)
            color_t = render_pkg["render"]  # (3, H, W) in [0, 1]
            color_np = (color_t.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
            color_np = np.ascontiguousarray(color_np)

            T_w2c = kf.world_view_transform.T.detach().cpu().numpy().astype(np.float64)

            color_o3d = o3d.geometry.Image(color_np)
            depth_o3d = o3d.geometry.Image(depth_np)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_o3d,
                depth_scale=1.0,
                depth_trunc=depth_trunc,
                convert_rgb_to_intensity=False,
            )
            volume.integrate(rgbd, intrinsic, T_w2c)

        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        Log(f"TSDF mesh: {len(mesh.vertices)} verts, {len(mesh.triangles)} faces "
            f"(voxel={voxel_length:.3f}m)", tag="GUI")
        return mesh

    # -------------------------------------------------------------------------
    # Main train loop: backend (mapping) then frontend (tracking) per frame
    # -------------------------------------------------------------------------

    def train(self):
        iter_times = []
        view_times = []


        frame_stride = self.config["Training"]["frame_stride"]
        start_frame = getattr(self.args, "start_frame", 0) or 0
        frame_indices = list(range(start_frame, self.total_frames, frame_stride))

        # CAn use to filter out frames with dynamic objects etc.
        # Config format: list of [start, end] ranges (inclusive) to skip
        skip_ranges = self.config["Training"].get("skip_frames", [])
        if skip_ranges:
            skip_set = set()
            for r in skip_ranges:
                skip_set.update(range(r[0], r[1] + 1))
            before = len(frame_indices)
            frame_indices = [f for f in frame_indices if f not in skip_set]
            print(f"Skipping {before - len(frame_indices)} frames from {len(skip_ranges)} ranges")

        total_views = len(frame_indices)
        print(
            f"\nTraining on {total_views} camera frames"
            + (f" (start={start_frame}, stride={frame_stride}, from {self.total_frames} total)..." if start_frame > 0 or frame_stride > 1 else "...")
        )

        render_for = self.config["Training"]["iterations_per_view"]
        log_every = self.config["Training"]["log_interval"]
        print("log_every: ", log_every)
        t_view_start = time.perf_counter()

        use_replay = self.config["Training"]["use_mapping_replay"]
        replay_size = self.config["Training"]["mapping_replay_size"]
        use_rerun = self.config["Results"]["use_rerun"]

        profile_steps = getattr(self.args, "profile_steps", 5)
        start_view = 1
        prev_cam = None
        if self.resume_iteration > 0:
            start_view = self.resume_iteration + 1
            prev_frame_idx = frame_indices[self.resume_iteration - 1]
            prev_cam = self._get_camera(prev_frame_idx)
            kf_payload = torch.load(
                os.path.join(self.resume_from, "keyframes_state.pt"), map_location="cpu"
            )
            kf_list = kf_payload.get("keyframes", [])
            if kf_list:
                last_kf = kf_list[-1]
                prev_cam.update_RT(
                    last_kf["R"].to(device=prev_cam.device),
                    last_kf["T"].to(device=prev_cam.device),
                )
            Log("Resume: starting from view_iteration", start_view, "prev_cam uid:", prev_cam.uid)

        if getattr(self.args, "profile", False):
            # Define where TensorBoard logs will be saved
            tb_log_dir = os.path.join(self.config["Results"]["tensorboard_dir"], "tb_profile_logs")
            os.makedirs(tb_log_dir, exist_ok=True)

            prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                profile_memory=True,
                record_shapes=True,
                with_stack=True,
                schedule=torch.profiler.schedule(
                    wait=1, 
                    warmup=1, 
                    active=profile_steps, 
                    repeat=1
                ),
                # This handles exporting the trace to TensorBoard format
                # on_trace_ready=torch.profiler.tensorboard_trace_handler(tb_log_dir)
            )
            prof.__enter__()

            torch.cuda.memory._record_memory_history(max_entries=100000)

        with torch.autograd.profiler.emit_nvtx():
            for view_iteration in range(start_view, total_views + 1):
                frame_idx = frame_indices[(view_iteration - 1) % total_views]

                viewpoint_cam = self._get_camera(frame_idx)

                # --- Check for GUI pause at top of loop ---
                if self._use_gui:
                    try:
                        while True:
                            vis_pkt = self._q_vis2main.get_nowait()
                            self._handle_gui_signal(vis_pkt, render_for, replay_size, None, log_every, view_iteration, total_views)
                            if vis_pkt.flag_pause:
                                if getattr(vis_pkt, "flag_snapshot", False):
                                    pause_tag = f"pause_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                                    self.map_state.save_mesh(pause_tag)
                                    Log("Paused: saved state to", os.path.join(self.config["Results"]["save_dir"], "mesh", f"iteration_{pause_tag}"), tag="GUI")
                                else:
                                    Log("Paused (editing live state)", tag="GUI")
                                # Block until explicit resume
                                while True:
                                    time.sleep(0.1)
                                    try:
                                        vis_pkt = self._q_vis2main.get_nowait()
                                        self._handle_gui_signal(vis_pkt, render_for, replay_size, None, log_every, view_iteration, total_views)
                                        if getattr(vis_pkt, "flag_resume", False):
                                            break
                                    except Exception:
                                        pass
                    except Exception:
                        pass

                SLAMProfiler_.start("iteration_total")

                if view_iteration == 1:
                    self.backend.keyframe_buffer.add(viewpoint_cam)
                    is_kf = True
                    print("added first keyframe")
                else:
                    prev = prev_cam
                    self.frontend.init_pose_from_previous(viewpoint_cam, prev)
                    # --- Frontend: track pose to convergence ---
                    track_viz = (
                        (lambda pkg, cam, l1: self._log_tracked_cam(pkg, cam, l1))
                        if (view_iteration % log_every == 0 and use_rerun)
                        else None
                    )
                    print("running tracking")
                    SLAMProfiler_.start("frontend_total")
                    render_pkg, loss_tracking, tracking_losses = self.frontend.run_tracking(
                        viewpoint_cam,
                        cur_frame_idx=view_iteration,
                        use_fd_grad=self.config["Training"]["use_fd_pose_grad"],
                        viz_callback=track_viz,
                    )
                    curr_visibility = render_pkg.get("curr_visibility")
                    print("curr_visibility: ", curr_visibility.count_nonzero().item())
                    is_kf = self.frontend.is_keyframe(viewpoint_cam, render_pkg)

                    # Tracking quality gate: reject keyframe if tracking barely converged
                    min_reduction = self.config["Training"].get("kf_min_tracking_reduction", 0.0)
                    if is_kf and min_reduction > 0 and len(tracking_losses) >= 2:
                        reduction = 1.0 - tracking_losses[-1] / (tracking_losses[0] + 1e-8)
                        if reduction < min_reduction:
                            print(f"Tracking quality gate: rejecting keyframe (reduction={reduction:.2%} < {min_reduction:.0%})")
                            is_kf = False

                    # --- Loss logging: tracking ---
                    self._record_loss("tracking", frame_idx, view_iteration, tracking_losses)
                    if tracking_losses:
                        self._tracking_final_losses.append((frame_idx, tracking_losses[-1]))
                    if use_rerun and tracking_losses:
                        img = self._plot_convergence_curve(
                            tracking_losses,
                            f"Tracking convergence (frame {frame_idx}, {len(tracking_losses)} iters)",
                        )
                        rr.log("tracking/convergence_curve", rr.Image(img))


                    if view_iteration % log_every == 0 and use_rerun:
                        under_reconstructed_mask = render_pkg["rend_alpha"] < self.config["Training"]["under_reconstructed_mask_threshold"]
                        rr.log("mapping/under_reconstructed_mask", rr.Image(under_reconstructed_mask.cpu().numpy().astype(np.uint8) * 255))
                        
                        l1_t = loss_tracking.detach().cpu()
                        while l1_t.dim() > 3:
                            l1_t = l1_t.squeeze(0)
                        l1_np = l1_t.permute(1, 2, 0).numpy()
                        l1_range = l1_np.max() - l1_np.min()
                        l1_np = (l1_np - l1_np.min()) / (l1_range + 1e-8) * 255
                        rr.log("tracking/loss_tracking", rr.Image(l1_np.astype(np.uint8)))

                    print("is_kf: ", is_kf)
                    if is_kf:
                        print("KEYFRAMING")
                        # if len(self.backend.keyframe_buffer) > 2:
                            # self.backend.densify_and_prune(self.backend.keyframe_buffer.get_penultimate_keyframe_visibility())
                        under_reconstructed_mask = render_pkg["rend_alpha"] < self.config["Training"]["under_reconstructed_mask_threshold"]
                        infill_depth = self.config["Training"].get("infill_underreconstructed_depth", False)
                        self.backend.add_keyframe(
                            viewpoint_cam,
                            under_reconstructed_mask=under_reconstructed_mask,
                            rendered_depth=render_pkg["render_depth_median"] if infill_depth else None,
                        )
                        self.triangles.set_sigma(self.opt.set_sigma)
                        self.backend.keyframe_buffer.add_with_visibility(viewpoint_cam, render_pkg)
                        self.backend.keyframe_buffer.pad_visibilities(self.triangles._triangle_indices.shape[0])

                        if view_iteration % log_every == 0 and use_rerun:
                            log_trajectories_rerun(list(self.map_state.keyframe_buffer), use_rerun=use_rerun, static=False)
                            log_mesh_rerun(self.triangles, viewpoint_cam, use_rerun=use_rerun, static=False)
                            # log mask
                            rr.log("mapping/under_reconstructed_mask", rr.Image(under_reconstructed_mask.cpu().numpy().astype(np.uint8) * 255))

                            # visualize the image

                            image = render_pkg["render"]
                            image_ab = (torch.exp(viewpoint_cam.exposure_a)) * image + viewpoint_cam.exposure_b
                            rgb_tracked = (
                                (torch.clamp(image_ab, min=0, max=1.0) * 255)
                                .byte()
                                .permute(1, 2, 0)
                                .contiguous()
                                .cpu()
                                .numpy()
                            )

                            # image_ab = torch.clamp(image_ab, 0.0, 1.0)
                            # rgb_raw = (image_ab * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
                            rr.log("mapping/image", rr.Image(rgb_tracked))

                            gt_image = viewpoint_cam.original_image
                            gt_image = (
                                (torch.clamp(gt_image, min=0, max=1.0) * 255)
                                .byte()
                                .permute(1, 2, 0)
                                .contiguous()
                                .cpu()
                                .numpy()
                            )
                            rr.log("mapping/gt_image", rr.Image(gt_image))



                            # l1_t = loss_tracking.detach().cpu()
                            # while l1_t.dim() > 3:
                            #     l1_t = l1_t.squeeze(0)
                            # l1_np = l1_t.permute(1, 2, 0).numpy()
                            # l1_range = l1_np.max() - l1_np.min()
                            # l1_np = (l1_np - l1_np.min()) / (l1_range + 1e-8) * 255
                            # rr.log("/l1_tracked", rr.Image(l1_np.astype(np.uint8)), static=False)





                            # log keyframe camera
                            self._log_visualizations(render_pkg, viewpoint_cam, name="keyframe", static=False)


                            normals = -render_pkg["rend_normal"]
                            surf_normals = -render_pkg["surf_normal"]

                            bae = viewpoint_cam.normal  # (3, H, W) in camera space
                            if bae.dim() == 4:
                                bae = bae.squeeze(0)  # (3, H, W)

                            W2C = viewpoint_cam.world_view_transform # (4, 4) world-to-camera
                            R_w2c = W2C[:3, :3].T # (3, 3) camera-to-world rotation
                            H, W = bae.shape[1], bae.shape[2]
                            # pseudo_normals_world =  bae.view(3, H, W) 
                            normals_cam = (R_w2c @ normals.view(3, -1)).view(3, H, W)

                            surf_normals_cam = (R_w2c @ surf_normals.view(3, -1)).view(3, H, W)
                            


                            np_normals_cam = normal_to_rgb(normals_cam)
                            rr.log("/normals_cam_tracked", rr.Image(np_normals_cam), static=False)

                            np_surf_normals_cam = normal_to_rgb(surf_normals_cam)
                            rr.log("/surf_normals_cam_tracked", rr.Image(np_surf_normals_cam), static=False)

                            # These are in the wrong coordinate system!
                            # np_normals = normal_to_rgb(normals)
                            # rr.log("/normals_tracked", rr.Image(np_normals), static=False)
                            # np_surf_normals = normal_to_rgb(surf_normals)
                            # rr.log("/surf_normals_tracked", rr.Image(np_surf_normals), static=False)

                            gt_normals = viewpoint_cam.normal
                            gt_np_normals = normal_to_rgb(gt_normals)
                            rr.log("/gt_normals_tracked", rr.Image(gt_np_normals), static=False)


        
                SLAMProfiler_.stop("frontend_total")
                SLAMProfiler_.start("backend_total")
                # --- Backend: add keyframe (map growth + prune), then mapping iterations ---
                t_iter_start = time.perf_counter()
                viz_cb = (
                    (lambda pkg, cam: self._log_visualizations(pkg, cam, False))
                    if (view_iteration % log_every == 0 and use_rerun)
                    else None
                )

                # densify every 20 keyframes

                if view_iteration == 490:
                    print("Dumping memory snapshot...")
                    torch.cuda.memory._dump_snapshot("slam_memory_snapshot.pickle")
                    torch.cuda.memory._record_memory_history(enabled=False)

            
                with torch.profiler.record_function("mapping_iteration"):
                    if is_kf: # and view_iteration == 1:


                        print(f"running mapping iterations: {is_kf}")
                        training_cfg = self.config.get("Training", {})

                        currently_visible_in_window, render_pkg_kf, triangle_pixel_counts, mapping_iter_losses, frustum_union = self.backend.run_mapping_iterations(
                            viewpoint_cam=viewpoint_cam,
                            num_iters=render_for,
                            replay_size=replay_size,
                            pose_window=training_cfg["initial_ba"]["pose_window"],
                            pose_lr_scale=training_cfg.get("mapping_pose_lr_scale", 0.5),
                            loss_history=self.loss_history,
                            viz_callback=viz_cb,
                            log_every=log_every,
                            view_iteration=view_iteration,
                            total_views=total_views,
                        )

                        # --- Loss logging: mapping ---
                        self._record_loss("mapping", frame_idx, view_iteration, mapping_iter_losses)
                        if mapping_iter_losses:
                            self._mapping_final_losses.append((frame_idx, mapping_iter_losses[-1]))
                        if use_rerun and mapping_iter_losses:
                            img = self._plot_convergence_curve(
                                mapping_iter_losses,
                                f"Mapping convergence (frame {frame_idx}, {len(mapping_iter_losses)} iters)",
                                ylabel="Mean loss / kf",
                            )
                            rr.log("mapping/convergence_curve", rr.Image(img))

                        # Trend chart (tracking + mapping final loss over all frames so far)
                        if use_rerun and (self._tracking_final_losses or self._mapping_final_losses):
                            trend_img = self._plot_loss_trend(
                                self._tracking_final_losses, self._mapping_final_losses
                            )
                            rr.log("losses/trend", rr.Image(trend_img))


                           # --- Send GUI update packet ---
                    if self._use_gui:
                        try:
                            self._q_main2vis.put_nowait(TrianglePacket(
                                triangles=self.triangles,
                                current_frame=viewpoint_cam,
                                keyframes=list(self.map_state.keyframe_buffer),
                            ))
                        except Exception:
                            pass

                    n_tri = self.backend.triangles._triangle_indices.shape[0]
                    if is_kf and viewpoint_cam.uid > 1 and currently_visible_in_window.numel() == n_tri:

                        keep_mask = self.backend.prune(currently_visible_in_window, frustum_union=frustum_union, use_importance_mask=True)
                        self.backend.keyframe_buffer.apply_prune_mask(keep_mask)
                        # Pruning changes triangle count/order; remap the union visibility
                        # so add_new_gs receives a mask aligned with current triangles.
                        currently_visible_in_window = currently_visible_in_window[: keep_mask.numel()]
                        currently_visible_in_window = currently_visible_in_window[keep_mask]

                        # Prune triangles that are consistently edge-on across keyframes
                        edge_on_cfg = self.config.get("Training", {})
                        edge_on_cos = edge_on_cfg.get("prune_edge_on_cos", 0.0)
                        if edge_on_cos > 0:
                            kf_cams = list(self.map_state.keyframe_buffer)
                            n_eo, eo_keep = self.triangles.prune_edge_on_triangles(
                                kf_cams,
                                cos_threshold=edge_on_cos,
                                min_keyframes=edge_on_cfg.get("prune_edge_on_min_kf", 3),
                            )
                            if n_eo > 0 and eo_keep is not None:
                                self.backend.keyframe_buffer.apply_prune_mask(eo_keep)
                                currently_visible_in_window = currently_visible_in_window[:eo_keep.numel()]
                                currently_visible_in_window = currently_visible_in_window[eo_keep]
                                # Update keep_mask so triangle_pixel_counts[keep_mask]
                                # stays aligned with the current triangle set
                                expanded = keep_mask.clone()
                                kept_indices = expanded.nonzero(as_tuple=True)[0]
                                expanded[kept_indices[~eo_keep]] = False
                                keep_mask = expanded

                        # Periodically reset vertex weights for non-visible vertices
                        # (like MonoGS reset_opacity_nonvisible) so floaters fade and get pruned.
                        reset_cfg = self.config.get("Training", {})
                        weight_reset_interval = reset_cfg.get("weight_reset_interval", 10)
                        if weight_reset_interval > 0 and view_iteration % weight_reset_interval == 0:
                            vis_mask = currently_visible_in_window > 0
                            n_reset = self.triangles.reset_vertex_weight_nonvisible(
                                vis_mask,
                                reset_value=reset_cfg.get("weight_reset_value", 0.4),
                            )
                            if n_reset > 0:
                                Log(f"Reset {n_reset} non-visible vertex weights", tag="DENSIFY")

                        # --- Densification stats (before) ---
                        tri_before = self.backend.triangles._triangle_indices.shape[0]
                        aspect_before = self.backend.triangles.mean_aspect_ratio()

                        # Gate densification on convergence plateau
                        densify_cfg = self.config["Training"]
                        on_convergence = densify_cfg.get("densify_on_convergence", False)
                        plateaued = self.backend._loss_plateaued(
                            mapping_iter_losses,
                            window=densify_cfg.get("densify_plateau_window", 10),
                            min_rel_improvement=densify_cfg.get("densify_plateau_min_improvement", 0.005),
                        )
                        should_densify = (not on_convergence) or plateaued

                        if should_densify:
                            densify_mode = densify_cfg.get("densify_mode", "split_only")
                            if densify_mode == "clone_and_split":
                                split_keep_mask = self.backend.triangles.densify_clone_and_split(
                                    cap_max=self.opt.max_points,
                                    visibility=currently_visible_in_window,
                                    triangle_pixel_counts=triangle_pixel_counts[keep_mask],
                                )
                            else:
                                split_keep_mask = self.backend.triangles.add_new_gs(
                                    cap_max=self.opt.max_points,
                                    splitt_large_triangles=self.opt.splitt_large_triangles,
                                    visibility=currently_visible_in_window,
                                    triangle_pixel_counts=triangle_pixel_counts[keep_mask],
                                )
                            self.backend.keyframe_buffer.pad_visibilities(split_keep_mask.numel())
                            self.backend.keyframe_buffer.apply_prune_mask(split_keep_mask)

                            # --- Densification stats (after) ---
                            tri_after = self.backend.triangles._triangle_indices.shape[0]
                            aspect_after = self.backend.triangles.mean_aspect_ratio()

                            densify_record = {
                                "type": "densify",
                                "frame_idx": frame_idx,
                                "view_iter": view_iteration,
                                "plateaued": plateaued,
                                "tri_before": tri_before,
                                "tri_after": tri_after,
                                "aspect_before": round(aspect_before, 4),
                                "aspect_after": round(aspect_after, 4),
                            }
                            mkdir_p(self.config["Results"]["save_dir"])
                            with open(self._loss_log_path, "a") as f:
                                f.write(json.dumps(densify_record) + "\n")
                            print(
                                f"[densify] plateaued={plateaued}  "
                                f"tris {tri_before}→{tri_after} (+{tri_after - tri_before})  "
                                f"aspect {aspect_before:.2f}→{aspect_after:.2f}"
                            )

                        # log_mesh_rerun(self.triangles, viewpoint_cam, use_rerun=use_rerun, static=False)

                log_mesh_rerun(self.triangles, viewpoint_cam, use_rerun=use_rerun, static=False)

             

                if prev_cam is not None and prev_cam.uid != viewpoint_cam.uid:
                    if self.map_state.keyframe_buffer.get_by_uid(prev_cam.uid) is not None:
                        prev_cam.offload_to_cpu()
                    else:
                        self.cameras.pop(prev_cam.uid, None)
            

                prev_cam = viewpoint_cam

                SLAMProfiler_.stop("iteration_total")
                SLAMProfiler_.stop("backend_total")

                if self.config["Results"]["save_results"]:
                    torch.cuda.synchronize()
                    total_elapsed_s = SLAMProfiler_.total_seconds("iteration_total")
                    current_fps = view_iteration / total_elapsed_s if total_elapsed_s > 0 else 0.0

                    frontend_avg_ms = SLAMProfiler_.total_seconds("frontend_total") / view_iteration * 1000
                    backend_avg_ms = SLAMProfiler_.total_seconds("backend_total") / view_iteration * 1000
                    Log("Frontend avg (ms)", frontend_avg_ms, tag="Eval")
                    Log("Backend avg (ms)", backend_avg_ms, tag="Eval")
                    Log("Total time (s)", total_elapsed_s, tag="Eval")
                    Log("FPS", current_fps, tag="Eval")
                    Log("Number of triangles", self.triangles._triangle_indices.shape[0], tag="Eval")

                    if view_iteration > 3 and view_iteration % self.config["Results"]["save_trj_kf_intv"] == 0:
                        for kf in self.map_state.keyframe_buffer:
                            if kf.uid in self.cameras:
                                self.cameras[kf.uid].update_RT(kf.R, kf.T)

                        pose_window = self.config["Training"]["initial_ba"]["pose_window"]
                        slice_start = pose_window if len(self.map_state.keyframe_buffer) > pose_window else 0

                        cam_provider = LazyCameraProvider(self.cameras, self.total_frames, self._get_camera)
                        ATE = eval_ate(
                            cam_provider,
                            self.map_state.keyframe_buffer.get_keyframes_ids(),
                            self.config["Results"]["save_dir"],
                            view_iteration,
                            monocular=self.monocular,
                            plot_axes=self.config.get("Results", {}).get("plot_axes", "xy"),
                        )
                        self.map_state.save_mesh(view_iteration)

                if getattr(self.args, "profile", False):
                    torch.cuda.synchronize()
                    prof.step()
                    # Account for wait (1) + warmup (1) + active (profile_steps)
                    total_profile_iters = 1 + 1 + profile_steps 
                    if getattr(self.args, "profile_exit", False) and view_iteration >= total_profile_iters:
                        break


        # if prof is not None:
        #     prof.__exit__(None, None, None)
        #     print(f"\n[Profile] TensorBoard logs saved to {tb_log_dir}")
        #     print("Run `tensorboard --logdir=path/to/tb_profile_logs` to view them.")

        if getattr(self.args, "profile", False):
            prof.__exit__(None, None, None)
            trace_dir = self.config["Results"].get("save_dir", ".")
            os.makedirs(trace_dir, exist_ok=True)
            trace_path = os.path.join(trace_dir, "slam_profile_trace.json")
            prof.export_chrome_trace(trace_path)
            print("\n" + "=" * 60 + "\n[Profile] By CUDA time (top 30):")
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
            print("\n[Profile] By CUDA memory (top 30):")
            print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=30))
            print(f"\n[Profile] Trace saved to {trace_path} (open in chrome://tracing)")
            print("=" * 60 + "\n")

        # --- After training: total time, FPS, then rendering eval ---
        total_train_time_s = SLAMProfiler_.total_seconds("iteration_total")
        num_frames_processed = len(SLAMProfiler_.timings["iteration_total"])
        train_fps = num_frames_processed / total_train_time_s if total_train_time_s > 0 else 0.0
        Log("Total train time (s)", total_train_time_s, tag="Eval")
        Log("Train FPS", train_fps, tag="Eval")

        if self.config["Results"]["save_results"]:


            # export triangle map 

            # cleaning of triangles that we do not need
            keyframes = list(self.map_state.keyframe_buffer)
            # self.triangles.importance_score = torch.zeros((self.triangles._triangle_indices.shape[0]), dtype=torch.float, device="cuda")
            # while keyframes:
            #     kf = keyframes.pop(0)
            #     render_pkg = render(kf, self.triangles, self.pipe, self.background)

            #     importance_score = render_pkg["max_blending"].detach()
            #     mask = importance_score > self.triangles.importance_score
            #     self.triangles.importance_score[mask] = importance_score[mask]
            # mask_importance  = (self.triangles.importance_score <= 0.2).squeeze() 
            # self.triangles.prune_triangles(~mask_importance) # delete all the remaining triangles that do not have an influence
            # self.map_state.keyframe_buffer.apply_prune_mask(~mask_importance)
            # device = self.triangles.vertices.device
            # used_vertex_mask = torch.zeros(self.triangles.vertices.shape[0], 
            #                             dtype=torch.bool, 
            #                             device=device)
            # if self.triangles._triangle_indices.numel() > 0:
            #     # Flatten indices and mark used vertices
            #     flat_indices = self.triangles._triangle_indices.flatten()
            #     used_vertex_mask[flat_indices] = True
            
            # vertex_mask = used_vertex_mask
            # self.triangles._prune_vertices(vertex_mask)

            self.map_state.save_mesh(num_frames_processed)
            print("Training is done")

            pose_window = self.config["Training"]["initial_ba"]["pose_window"]

            cam_provider = LazyCameraProvider(self.cameras, self.total_frames, self._get_camera)
            self._eval_ate = eval_ate(
                cam_provider,
                self.map_state.keyframe_buffer.get_keyframes_ids(),
                self.config["Results"]["save_dir"],
                num_frames_processed,
                monocular=self.monocular,
                plot_axes=self.config.get("Results", {}).get("plot_axes", "xy"),
            )
            self._eval_rendering = eval_rendering(
                cam_provider,
                self.triangles,
                self.dataset,
                self.config["Results"]["save_dir"],
                self.pipe,
                self.background,
                kf_indices=self.map_state.keyframe_buffer.get_keyframes_ids(),
                iteration=num_frames_processed,
            )
            columns = ["tag", "psnr", "ssim", "lpips", "RMSE ATE", "FPS"]
            metrics_table = wandb.Table(columns=columns)
            metrics_table.add_data(
                "Before",
                self._eval_rendering["mean_psnr"],
                self._eval_rendering["mean_ssim"],
                self._eval_rendering["mean_lpips"],
                self._eval_ate,
                train_fps,
            )

        # Signal GUI that training is done
        if self._use_gui:
            self._q_main2vis.put(TrianglePacket(finish=True))
            if self._gui_process is not None:
                self._gui_process.join(timeout=10)

        self.timings["train_iter_avg_s"] = (
            sum(iter_times) / len(iter_times) if iter_times else 0
        )
        self.timings["train_view_avg_s"] = (
            sum(view_times) / len(view_times) if view_times else 0
        )
        self.timings["train_total_s"] = total_train_time_s
        self._print_timings()
        self._plot_loss_curve()
        SLAMProfiler_.summary()
        peak_gpu_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak GPU memory: {peak_gpu_gb:.2f} GB")

        save_dir = self.config["Results"].get("save_dir", "")
        if save_dir:
            timing_data = {
                "num_frames": num_frames_processed,
                "train_total_s": total_train_time_s,
                "train_fps": train_fps,
                "peak_gpu_gb": round(peak_gpu_gb, 3),
                "train_iter_avg_s": self.timings["train_iter_avg_s"],
                "train_view_avg_s": self.timings["train_view_avg_s"],
                "profiler": {},
            }
            for name, values in SLAMProfiler_.timings.items():
                timing_data["profiler"][name] = {
                    "avg_ms": round((sum(values) / len(values)) * 1000, 2) if values else 0,
                    "total_s": round(sum(values), 3),
                    "count": len(values),
                }
            if hasattr(self, "_eval_ate"):
                timing_data["ate_rmse_m"] = self._eval_ate
            if hasattr(self, "_eval_rendering"):
                timing_data["psnr"] = self._eval_rendering.get("mean_psnr")
                timing_data["ssim"] = self._eval_rendering.get("mean_ssim")
                timing_data["lpips"] = self._eval_rendering.get("mean_lpips")
            timing_data["num_triangles"] = self.backend.triangles._triangle_indices.shape[0]
            timing_path = os.path.join(save_dir, "timing.json")
            mkdir_p(save_dir)
            with open(timing_path, "w") as f:
                json.dump(timing_data, f, indent=2)
            print(f"Timing saved to {timing_path}")

def main():
    parser = ArgumentParser(description="Mesh SLAM (triangle splatting)")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to config YAML (e.g. configs/mono/tum/fr3_office.yaml)",
    )
    parser.add_argument("--eval", action="store_true", help="Evaluation mode")
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--view_only", action="store_true", default=False)
    parser.add_argument("--add_keyframes", action="store_true", default=False)
    parser.add_argument("--iterations_per_view", type=int, default=20)
    parser.add_argument("--frame_stride", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=2)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--profile", action="store_true", help="Profile with torch.profiler (time + memory)")
    parser.add_argument("--profile_steps", type=int, default=5, help="Number of view iterations to profile")
    parser.add_argument("--profile_exit", action="store_true", help="Exit after profiling (for quick profile runs)")
    parser.add_argument("--start_frame", type=int, default=0, help="Skip first N frames of the dataset")
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint folder (e.g. results/replica_room1/.../mesh/iteration_1250) to resume SLAM from",
    )
    parser.add_argument(
        "-o", "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config value (e.g. -o Results.use_rerun=True -o Training.frame_stride=2). Dot notation for nested keys.",
    )

    args = parser.parse_args(sys.argv[1:])
    config = load_config(args.config)

    resume_from = getattr(args, "resume_from", None)
    resume_iteration = None
    if resume_from:
        resume_from = os.path.abspath(resume_from)
        if not os.path.isdir(resume_from):
            raise SystemExit(f"--resume_from must be a directory: {resume_from}")
        # Parse iteration from path (e.g. .../iteration_1250 -> 1250)
        basename = os.path.basename(resume_from.rstrip("/"))
        if basename.startswith("iteration_"):
            try:
                resume_iteration = int(basename.split("_")[1])
            except (IndexError, ValueError):
                pass
        if resume_iteration is None:
            raise SystemExit(f"Could not parse iteration number from --resume_from path: {resume_from}")
        # Use config and save_dir from the checkpoint run
        run_dir = os.path.dirname(os.path.dirname(resume_from))
        config_path_in_run = os.path.join(run_dir, "config.yml")
        if os.path.isfile(config_path_in_run):
            with open(config_path_in_run, "r") as f:
                config = yaml.safe_load(f)
            Log("Resume: loaded config from", config_path_in_run)
        config["Results"]["save_dir"] = run_dir
        config["Results"]["save_trj_kf_intv"] = 1
        if args.dataset_path:
            config.setdefault("Dataset", {})["dataset_path"] = args.dataset_path
        Log("Resume: continuing run from iteration", resume_iteration, "save_dir:", run_dir)

    # Apply command-line overrides (after resume so overrides win)
    for override in args.override:
        if "=" not in override:
            raise SystemExit(f"Override must be KEY=VALUE, got: {override}")
        key_path, _, value_str = override.partition("=")
        keys = key_path.strip().split(".")
        value = yaml.safe_load(value_str)
        d = config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    if args.eval:
        config.setdefault("Results", {})["save_results"] = True
        config.setdefault("Results", {})["use_gui"] = False
        config.setdefault("Results", {})["eval_rendering"] = True

    dataset_path = (
        args.dataset_path
        or config.get("Dataset", {}).get("dataset_path")
        or config.get("dataset_path")
    )
    if not dataset_path:
        raise SystemExit(
            "Dataset path required: set Dataset.dataset_path in config or pass --dataset_path"
        )

    num_frames = args.num_frames or config.get("Training", {}).get("num_frames", None)
 
    iterations_per_view = (
        args.iterations_per_view
        or config["Training"]["iterations_per_view"]
    )
    frame_stride = (
        args.frame_stride
        if args.frame_stride is not None
        else config["Training"]["frame_stride"]
    )
    log_interval = args.log_interval or config["Training"]["log_interval"]

    lp = ModelParams(ArgumentParser())
    pipe = PipelineParams(ArgumentParser())
    opt = OptimizationParams(ArgumentParser())
    # Apply opt_params from config (overrides defaults)
    for key, value in config.get("opt_params", {}).items():
        if hasattr(opt, key):
            setattr(opt, key, value)



    
    if config["Results"]["save_results"]:
        mkdir_p(config["Results"]["save_dir"])
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if resume_iteration is None:
            path = (dataset_path or "").rstrip("/").split("/")
            save_dir = os.path.join(
                config["Results"]["save_dir"],
                (path[-2] + "_" + path[-1]) if len(path) >= 2 else "run",
                current_datetime,
            )
            config["Results"]["save_dir"] = save_dir
            mkdir_p(save_dir)
            with open(os.path.join(save_dir, "config.yml"), "w") as f:
                yaml.dump(config, f)
            Log("Saving results to " + save_dir)
        else:
            Log("Resume: reusing save_dir", config["Results"]["save_dir"])

        use_wandb = config.get("Results", {}).get("use_wandb", False)
        wandb.init(
            project="triangle-slam",
            name=f"{Path(args.config).stem}_{current_datetime}" + (f"_resume{resume_iteration}" if resume_iteration else ""),
            config=config,
            mode=None if use_wandb else "disabled",
        )
        wandb.define_metric("frame_idx")
        wandb.define_metric("ate*", step_metric="frame_idx")


    trainer = MeshSLAMTrainer(
        args, config, dataset_path, args.config, opt, pipe, num_frames=num_frames,
        resume_from=resume_from, resume_iteration=resume_iteration,
    )
    # if use rerun
    if config["Results"]["use_rerun"]:
        rr.init("triangle_splatting", spawn=True)
    if not args.view_only:
        trainer.train()
    print("Done")


if __name__ == "__main__":
    main()
