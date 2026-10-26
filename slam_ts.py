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
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
# import rerun as rr
import torch

#==================================================
# Imported Libraries
#==================================================
import shutil
from event_datasets.dataset import get_dataset
from event_datasets.utils import get_camera_rays
from event_median_pool import MedianPool2d
from tikhonov_regularizor import tikhonov_regularization
from event_loss_utils import compute_white_balance_loss, compute_ssim_loss
from event_tools.eval_ate import pose_evaluation, save_pose_as_kitti_evo, align_and_est_scale
from event_general_utils import render_ev_accumulation, BasicPointCloud, SH2RGB
from torch.utils.data import DataLoader
from tqdm import tqdm
import pypose as pp
import imageio
from event_colormaps import apply_colormap
from event_spline.spline import SE3_to_se3, se3_to_SE3, se3_to_SE3_m44
#===================================================

from mesh_splatting.scene.triangle_model import TriangleModel

from mesh_splatting.utils.graphics_utils import focal2fov, getProjectionMatrix2, getWorld2View2
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
# import wandb
from pathlib import Path
from mesh_splatting.utils.system_utils import mkdir_p

import multiprocessing as mp
from meshslam.gui.gui_utils import TrianglePacket, ParamsGUI, FakeQueue



from utils.normal_utils import normal_to_rgb

from fused_ssim import fused_ssim

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"



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

to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

def lin_log(color, linlog_thres=1):
    """
    Input: 
    :color torch.Tensor of (N_rand_events, 1 or 3). 1 if use_luma, else 3 (rgb).
           We pass rgb here, if we want to treat r,g,b separately in the loss (each pixel must obey event constraint).
    """
    # Compute the required slope for linear region (below luma_thres)
    # we need natural log (v2e writes ln and "it comes from exponential relation")
    lin_slope = np.log(linlog_thres) / linlog_thres

    # Peform linear-map for smaller thres, and log-mapping for above thresh
    lin_log_rgb = torch.where(color < linlog_thres, lin_slope * color, torch.log(color))
    return lin_log_rgb


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

        self.init_num_pts = 100000

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dataset = get_dataset(config)

        self.dataset.fovx = focal2fov(self.dataset.fx, self.dataset.W) 
        self.dataset.fovy = focal2fov(self.dataset.fy, self.dataset.H)
        
        self.create_pose_data()
        
        # self.gs_dataset_cfg = gs_dataset_cfg
        # self.gs_opt_cfg = gs_opt_cfg
        # self.gs_pipe_cfg = gs_pipe_cfg

        self.control_knot_poses = None
        self.control_knot_ts = None
        self.control_knot_delta_t = 0.15 # create control knot every 5 frames 

        self.events_for_tracking = None
        self.events_for_BA = [] 
        
        self.scene_scale = self.config["mapping"]["bounding_size"]
        print("Scene scale(or size of bounding box):", self.scene_scale)

        # self.dataset = load_dataset(args, dataset_path, config)
        # print("Dataset length:", len(self.dataset))

        self.projection_matrix = self._setup_projection_matrix().to(self.device) # projection matrix is row major order
        # print("From MeshSlamTrianer Init")
        # print(f"The projection matrix shape : {self.projection_matrix.shape}")
        # print(f"The projection matrix : {self.projection_matrix}")

        self.median_filter = MedianPool2d(kernel_size=5, same=True)
        self.median_filter_dvs = MedianPool2d(kernel_size=5, same=True)
        
        height = 480
        width = 640

        mask = np.ones((height, width), dtype=np.uint8)
        self.vector_mask = torch.from_numpy(mask).float().cuda()
        
        # self.scene_extent = None
        self.color_mask = np.zeros((self.dataset.H, self.dataset.W, 3))

        if self.config["mapping"]["color_channels"]==3:
            self.color_mask[0::2, 0::2, 0] = 1  # r

            self.color_mask[0::2, 1::2, 1] = 1  # g
            self.color_mask[1::2, 0::2, 1] = 1  # g

            self.color_mask[1::2, 1::2, 2] = 1  # b
        else:
            self.color_mask[...] = 1

        # self.color_mask = self.color_mask.reshape((-1, 3))
        self.color_mask = torch.from_numpy(self.color_mask).float().cuda()

        #===============================================================================================
        # TODO : Commented Code
        # self.monocular = self.config["Dataset"]["sensor_type"] == "monocular"
        # self.total_frames = min(num_frames or len(self.dataset.poses), len(self.dataset.poses))
        # self.cameras = {}
        

        # Background and map state (backend owns triangles + keyframes)
        # if opt.random_background:
        #     self.background = torch.rand(3, device="cuda")
        # elif config.get("model_params", {}).get("white_background", False):
        #     self.background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        # else:
        #     self.background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")   
        #===============================================================================================

        self.timings = {}
        self.loss_history = []
        self._tracking_final_losses = []  # (frame_idx, final_loss) per tracked frame
        self._mapping_final_losses = []   # (frame_idx, final_loss) per keyframe
        self._loss_log_path = os.path.join(config["Results"]["save_dir"], "loss_log.jsonl")

        # set the background color
        # 0-white, 1-black, 2-grey
        if self.config["mapping"]["background"]==0:
            self.background=torch.ones(3, device=self.device) # white
        elif self.config["mapping"]["background"]==1:
            self.background=torch.zeros(3, device=self.device) # black
        elif self.config["mapping"]["background"]==2:
            self.background=(159./255.)*torch.ones(3, device=self.device) # grey
        else:
            raise ValueError("Not implemented background color!")

        #========================================================================================
        # TODO : Commented Code. changed the gui process is commented 
        # GUI process (Open3D viewer, alternative to rerun)
        # Must use 'spawn' - CUDA is already initialised in the main process,
        # and Linux's default 'fork' cannot re-initialise CUDA in the child.
        # self._use_gui = self.config.get("Results", {}).get("use_gui", False)
        # if self._use_gui:
        #     _mp_ctx = mp.get_context("spawn")
        #     self._q_main2vis = _mp_ctx.Queue()
        #     self._q_vis2main = _mp_ctx.Queue()
        #     _params_gui = ParamsGUI(
        #         pipe=pipe,
        #         background=self.background.cpu(),
        #         q_main2vis=self._q_main2vis,
        #         q_vis2main=self._q_vis2main,
        #     )
        #     from meshslam.gui import slam_gui
        #     self._gui_process = _mp_ctx.Process(
        #         target=slam_gui.run, args=(_params_gui,), daemon=True
        #     )
        #     self._gui_process.start()
        #     time.sleep(3)  # wait for Open3D window to initialise
        # else:
        #     self._q_main2vis = FakeQueue()
        #     self._q_vis2main = FakeQueue()
        # self._freeze_vertices = False
        # self._gui_process = None
        #========================================================================================

        # Build map: triangles + keyframe buffer
        # if self.resume_from:
        #     self._initialize_from_checkpoint()
        # else:
        #     # self._initialize_model()
        #     self._initialize_model_from_random_points()

        # print(f"Model inittialize randomly, Total triangles : {self.triangles.vertices.shape[0]}")    
        
        #=================================================================================
        # TODO : commented code : No keyframing
        # self.depth2normal = Depth2normal(d_min=0.0, d_max=10.0, k=4, d=1, min_nghbr=4, gamma=0.05, gamma_exception=False)
        # maxlen = self.config.get("Training", {}).get("window_size", None)

        # keyframe_buffer = KeyframeBuffer(maxlen=maxlen, device="cuda")
        #=================================================================================

        # self.map_state = MapState(
        #     triangles=self.triangles,
        #     background=self.background,
        #     # keyframe_buffer=keyframe_buffer,
        #     device=torch.device("cuda"),
        #     timings=self.timings,
        #     loss_history=self.loss_history,
        #     model_path=self.config["Results"]["save_dir"],
        # )

        # # Backend and frontend (same process for now)
        # self.backend = MappingEngine(self.map_state, opt, pipe, config)
        # self.frontend = TrackingEngine(self.map_state, pipe, config)

        # # Convenience alias for code that still references self.triangles
        # self.triangles = self.map_state.triangles

        # if self.resume_from:
        #     self._load_keyframes_from_checkpoint()

    #==============================================================
    # TODO : Added function
    def create_pose_data(self):
        '''
        Create the pose data
        '''
        self.est_c2w_data = {}
        self.est_c2w_ts = {}
        self.ctrl_knot_se3_all = {}
        self.ctrl_knot_ts_all = {}
        self.load_gt_pose()        

    def load_gt_pose(self):
        '''
        Load the ground truth pose
        '''
        self.pose_gt = {}
        for i, pose in enumerate(self.dataset.poses):
            self.pose_gt[self.dataset.frame_ids[i]] = pose 
    #==============================================================           

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
            znear=0.01,   # self.config["cam"]["near"]  changed 0.01
            zfar=100.0, #self.config["cam"]["far"], # changed 100.0
            fx=self.dataset.fx,
            fy=self.dataset.fy,
            cx=self.dataset.cx,
            cy=self.dataset.cy,
            W=self.dataset.W, # changed width -> W
            H=self.dataset.H, # changed height -> H
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

    # TODO : Function for the random initialization
    def _initialize_model_from_random_points(self):
        t0 = time.perf_counter()
        sh_degree = 0
        self.triangles = TriangleModel(sh_degree=sh_degree, config=self.config)

        pcd, features, opacities, radii = self.triangles.create_pcd_from_random_points(
            scene_scale=self.scene_scale, init_num_pts=self.init_num_pts
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

        print("================= Triangle Initialized With Randomly Sampled Points ==================== ")
        print(f"*************** Number of vertices ; {self.triangles.vertices.shape[0]}")
        print("********** Number of Vertices :", len(self.triangles.vertices))
        print("********** Number of Trianges :", len(self.triangles._triangles))
    

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

    def create_densify_points(self, _cur_frame_id, events_stream, T_se3_start, T_se3_end, num_new_pts=1000):
        c2w_start = T_se3_start.Exp().matrix().unsqueeze(0).cuda()  #[1, 4, 4]
        render_ev_start = self.render_triangles(c2w_start)
        c2w_end =  T_se3_end.Exp().matrix().unsqueeze(0).cuda()  #[1, 4, 4]
        render_ev_end = self.render_triangles(c2w_end)
        
        eps = 1e-5
        syn_event_acc = torch.log(render_ev_end["render"]**2.2+eps)-torch.log(render_ev_start["render"]**2.2+eps)
        
        gt_event_map = self.post_process_event_image(events_stream)

        # error_map = (gt_event_map - syn_event_acc).abs()
        # pixel_uv_with_event = torch.where(error_map > 0.2)
        # sampled_event_pixel_idx = torch.randperm(pixel_uv_with_event[0].shape[0])[:num_new_pts]
        # indice_h = pixel_uv_with_event[0][sampled_event_pixel_idx]
        # indice_w = pixel_uv_with_event[1][sampled_event_pixel_idx]
        
        create_densify_points_num = self.config["mapping"]["create_densify_points_num"]
        densify_alpha_threshold = self.config["mapping"]["densify_alpha_threshold"]
        # non-visibility mask
        alpha_start = render_ev_end["rend_alpha"].squeeze().detach()
        # alpha_start = alpha_start[0][:,:,0]  # [H, W]
        non_vis_mask = torch.where(alpha_start < densify_alpha_threshold)
        sampled_event_pixel_idx = torch.randperm(non_vis_mask[0].shape[0])[:create_densify_points_num]
        indice_h = non_vis_mask[0][sampled_event_pixel_idx].to(self.device)
        indice_w = non_vis_mask[1][sampled_event_pixel_idx].to(self.device)

        #
        fx = self.dataset.fx
        fy = self.dataset.fy
        cx = self.dataset.cx
        cy = self.dataset.cy
        H = self.dataset.H
        W = self.dataset.W

        sampled_rays = get_camera_rays(H, W, fx, fy, cx, cy, type='OpenCV').to(self.device) 
        sampled_rays = sampled_rays[indice_h, indice_w, :].to(T_se3_start.device)
        depth_gs = render_ev_end['expected_depth'].squeeze()[indice_h, indice_w].unsqueeze(-1)
        depth_gs = depth_gs.detach()
        depth_img_gs = render_ev_end['expected_depth'].squeeze()
        img_gs = render_ev_end['render'].detach().cpu().numpy()
        
        save_dir_path  = os.path.join(self.config["data"]["output"], self.config["data"]["exp_name"], "depth_for_new_gs")
        depth_tmp = depth_img_gs.detach()
        depth_tmp = depth_tmp.view(depth_tmp.shape[0], depth_tmp.shape[1], -1)
        depth_img = apply_colormap(depth_tmp)
        depth_img = depth_img.cpu().numpy()
        if self.config["visualize_inter_img"]:
            imageio.imwrite(os.path.join(save_dir_path, f"frame{_cur_frame_id}_gs_depth.png"), to8b(depth_img)) 
        
        if self.config["add_new_gs_from_estDepth"]:
            print("&&&&&&&&&&&&&& creat new Gaussian Points from estimated depth map &&&&&&&&&&&&&&&&&&")
            # estimate depth from the rendered image, using diffuser
            from PIL import Image
            from diffusers import DiffusionPipeline
            from diffusers.utils import load_image
            depthEst_pretrained_model_path = self.config["depthEst_pretrained_model_path"]
            pipe = DiffusionPipeline.from_pretrained(
                        depthEst_pretrained_model_path,
                        custom_pipeline="marigold_depth_estimation",
                        # torch_dtype=torch.float16,                # (optional) Run with half-precision (16-bit float).
                        local_files_only=True
                    )
            pipe.to("cuda")
            # image: Image.Image = load_image(img_path)
            image = Image.fromarray(to8b(img_gs))
            pipeline_output = pipe(
                                    image,                  # Input image.
                                    # denoising_steps=30,     # (optional) Number of denoising steps of each inference pass. Default: 10.
                                    # ensemble_size=10,       # (optional) Number of inference passes in the ensemble. Default: 10.
                                    # processing_res=768,     # (optional) Maximum resolution of processing. If set to 0: will not resize at all. Defaults to 768.
                                    # match_input_res=True,   # (optional) Resize depth prediction to match input resolution.
                                    # batch_size=10,           # (optional) Inference batch size, no bigger than `num_ensemble`. If set to 0, the script will automatically decide the proper batch size. Defaults to 0.
                                    # color_map="Spectral",   # (optional) Colormap used to colorize the depth map. Defaults to "Spectral". Set to `None` to skip colormap generation.
                                    # show_progress_bar=True, # (optional) If true, will show progress bars of the inference progress.
                                )
            depth_img_est: np.ndarray = pipeline_output.depth_np  
            # save the estimated images
            depth_colored: Image.Image = pipeline_output.depth_colored      # Colorized prediction
            imageio.imwrite(os.path.join(save_dir_path, f"frame{_cur_frame_id}_img.png"), to8b(img_gs)) 
            depth_uint16 = (depth_img_est * 65535.0).astype(np.uint16)
            Image.fromarray(depth_uint16).save(os.path.join(save_dir_path, f"frame{_cur_frame_id}_depth_map.png"), mode="I;16")
            depth_colored.save(os.path.join(save_dir_path, f"frame{_cur_frame_id}_depth_colored.png"))
            # 
            
            depth_est = torch.from_numpy(depth_img_est).cuda()[indice_h, indice_w].unsqueeze(-1)
            depth_aligned = self.depth_alignment(depth_est, depth_gs)
        else:
            depth_aligned = depth_gs

        # initialize points on a frontal parallel plane at infinity
        # depth = 100
        sampled_rays = (sampled_rays * depth_aligned).transpose(1,0).to(T_se3_start.device)

        #
        T_cam_to_wld = se3_to_SE3(T_se3_start)
        sampled_rays = torch.matmul(T_cam_to_wld[:3, :3], sampled_rays) + T_cam_to_wld[:3, 3].unsqueeze(-1) 
        points = sampled_rays.transpose(1, 0)

        # colors
        # colors = render_ev_start['image'][indice_h, indice_w].unsqueeze(-1).repeat(1, 3)
        shs = torch.rand((points.shape[0], 3)) / 255.0

        # normals
        normals = torch.zeros_like(points)
        normals[:, 2] = 1.

        # create pcd
        # pcd = BasicPointCloud(points=points, colors=colors, normals=normals)
        pcd = BasicPointCloud(points=points.detach().cpu(), colors=SH2RGB(shs), normals=normals.detach().cpu())

        return pcd    

    def render_triangles(self, cam_2_world):
        world_2_cam = torch.linalg.inv(cam_2_world).squeeze().transpose(0,1) # 4,4 row major order
        full_proj_trans = (world_2_cam.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze() # 4,4 row major order
        camera_center = cam_2_world.squeeze()[:3, 3] # last column

        # print(f"w2c device : {world_2_cam.device}")
        # print(f"full_proj device : {full_proj_trans.device}")
        # print(f"cam center device : {camera_center.device}")

        # TODO : Need to define the passing parameters
        rets = render(  self.dataset.H, self.dataset.W,
                        self.dataset.fovx, self.dataset.fovy, world_2_cam,
                        full_proj_trans, self.projection_matrix,
                        camera_center, self.triangles, self.pipe, self.background)    
             

        if self.config["mapping"]["color_channels"] == 3:
            rets["render"] = rets["render"].permute(1, 2, 0) # H,W,3
        else:
            rets["render"] = rets["render"][0, :, :] #[H, W]

        return rets   

    def predict_pose_se3_lie(self, pose_prev_se3_lie, pose_prevprev_se3_lie):
        pose_delta_SE3_lie = pose_prevprev_se3_lie.Inv().Exp() * pose_prev_se3_lie.Exp()
        predicted_SE3_lie = pose_prev_se3_lie.Exp() * pose_delta_SE3_lie
        return predicted_SE3_lie.Log()    

    def accumulate_event_to_img(self, height, width, events_stream, polarity_offset=0.0):
        if self.config["dataset"] == 'eventnerf_real':
            events_map = np.zeros((self.dataset.H, self.dataset.W))
            xs = events_stream[:, 0].detach().cpu().numpy()
            ys = events_stream[:, 1].detach().cpu().numpy()
            ps = events_stream[:, 3].detach().cpu().numpy()
            ts = events_stream[:, 2].detach().cpu().numpy()
            polarity_offset = self.config["polarity_offset"]
            resolution_level = 1
            xs = xs.astype(np.int16)
            ys = ys.astype(np.int16)
            accumulate_events(xs, ys, ts, ps, events_map, resolution_level, polarity_offset)
            events_map = torch.from_numpy(events_map).float().cuda()
        else:
            x_window = events_stream[:, 0]
            y_window = events_stream[:, 1]
            pol_window = events_stream[:, 3]  
            polarity_offset = self.config["polarity_offset"]
            pol_window = pol_window + polarity_offset  # offset for polarity
            assert pol_window.shape[0] > 0
        
            # create indices for sparse tensor
            events_num = x_window.shape[0]
            indices= torch.cat([y_window.unsqueeze(0), x_window.unsqueeze(0), torch.arange(events_num).unsqueeze(0).to(events_stream.device)], dim=0)
            events_map_sparse = torch.sparse_coo_tensor(indices=indices, values=pol_window, size=(height, width, events_num))
            events_map_sum = torch.sparse.sum(events_map_sparse, dim=-1)

            events_map = events_map_sum.to_dense()
            
        threshold = self.config["event"]["threshold"]
        events_map =  events_map * threshold
        if self.config["event"]["clip"]:
            events_map = torch.clip(events_map, self.config["event"]["clip_min"], self.config["event"]["clip_max"])
        
        # #debug
        # pos_event_mask = (events_map>0).float()
        # neg_event_mask = (events_map<0).float()
        # pos_event_avg = (events_map*pos_event_mask).sum()/pos_event_mask.sum()
        # neg_event_avg = (events_map*neg_event_mask).sum()/neg_event_mask.sum()
        # print(f"Events_map: min={events_map.min():.3f}, max={events_map.max():.3f}, mean={events_map.mean():.3f}, pos_event_avg={pos_event_avg:.3f}, neg_event_avg={neg_event_avg:.3f}")

        return events_map         

    def post_process_event_image(self, events_stream, _sigma=0.001):
        #
        if self.config["dataset"] == "tum_vie" and self.config["data"]["downsample_factor"]>1:
            events_map = self.accumulate_event_to_img(self.dataset.H_old, self.dataset.W_old, events_stream) 
        else:
            events_map = self.accumulate_event_to_img(self.dataset.H, self.dataset.W, events_stream) 

        if self.config["dataset"] == "tum_vie":
            if self.config["mapping"]["use_median_filter"]:
                events_map = self.median_filter(events_map.unsqueeze(0).unsqueeze(0)).squeeze()
            if self.config["data"]["downsample_factor"]>1:
                events_map = events_map.cpu().numpy()     
                # events_map = cv2.fisheye.undistortImage(events_map, self.dataset.K_old, self.dataset.dist_coeffs, Knew=self.dataset.K_old, new_size=(self.dataset.W_old, self.dataset.H_old))
                events_map = cv2.fisheye.undistortImage(events_map, self.dataset.K, self.dataset.dist_coeffs, Knew=self.dataset.K_new, new_size=(self.dataset.W, self.dataset.H))
                events_map = cv2.resize(events_map, (self.dataset.W, self.dataset.H), interpolation=cv2.INTER_LINEAR) #cv2.INTER_AREA     
                events_map = torch.from_numpy(events_map).cuda()
            else:
                events_map = events_map.cpu().numpy()
                # events_map = cv2.fisheye.undistortImage(events_map, self.dataset.K, self.dataset.dist_coeffs, Knew=self.dataset.K, new_size=(self.dataset.W, self.dataset.H))
                events_map = cv2.fisheye.undistortImage(events_map, self.dataset.K, self.dataset.dist_coeffs, Knew=self.dataset.K_new, new_size=(self.dataset.W, self.dataset.H))
                events_map = torch.from_numpy(events_map).cuda()
        # elif self.config["dataset"] == "vector":
        #     events_map = events_map.cpu().numpy()
            
        #     # undisted_img = cv2.undistort(events_map, self.dataset.K_old, self.dataset.dist_coeffs, newCameraMatrix=self.dataset.K_new)
        #     undisted_img = cv2.undistort(events_map, self.dataset.K_old, self.dataset.dist_coeffs)
        #     events_map = torch.from_numpy(undisted_img).cuda()
        elif self.config["dataset"] == "dev_real" or self.config["dataset"] == "rpg_evo_stereo":
            events_map = events_map.cpu().numpy()
            # undisted_img = cv2.undistort(events_map, self.dataset.K_old, self.dataset.dist_coeffs, newCameraMatrix=self.dataset.K_new)
            undisted_img = cv2.undistort(events_map, self.dataset.K, self.dataset.dist_coeffs)
            events_map = torch.from_numpy(undisted_img).cuda()
            if self.config["mapping"]["use_median_filter"]:
                events_map = self.median_filter_dvs(events_map.unsqueeze(0).unsqueeze(0)).squeeze()
        
        if self.config["blur_event"]:
            # blur
            blurrer = v2.GaussianBlur(kernel_size=(5,5), sigma=_sigma)
            events_map = blurrer(events_map.unsqueeze(0).unsqueeze(0)).squeeze()
        
        return events_map    

    def get_poses_lie(self, control_knot_poses, control_knot_ts, query_t, mode = 'cspline'):
        if mode == 'linear':
            if np.abs(query_t - control_knot_ts[0]) < 1e-6:
                query_t = query_t + 1e-6
            if np.abs(query_t - control_knot_ts[-1]) < 1e-6:
                query_t = query_t - 1e-6

            assert query_t >= control_knot_ts[0]   # numpy
            assert query_t <= control_knot_ts[-1]
            
            idx_end = np.searchsorted(control_knot_ts, query_t)
            idx_start = idx_end - 1
            assert idx_start>=0
            assert control_knot_ts[idx_start]<=query_t
            assert control_knot_ts[idx_end]>=query_t
            
            ctrl_knot_t_start = control_knot_ts[idx_start]
            ctrl_knot_delta_t = control_knot_ts[idx_end] - control_knot_ts[idx_start]
            
            elapsed_t = query_t - ctrl_knot_t_start
            tau = elapsed_t/ctrl_knot_delta_t # normalize to (0,1)
            
            # tau = torch.tensor(tau).reshape([1,1])
            # segment = torch.stack([control_knot_poses[idx_start], control_knot_poses[idx_end]], dim=0).unsqueeze(0)
            # pose_interp = linear_interpolation(segment, tau)[0][0]  # SE3, [G]
            # pose_interp = pose_interp.Log() # se3, [6]
            
            pose_interp_se3_lie =  control_knot_poses[idx_start]*(1.-tau) +  control_knot_poses[idx_end]*tau
            pose_interp = pose_interp_se3_lie.Exp()
            
        elif mode == 'cspline':
            if np.abs(query_t - control_knot_ts[1]) < 1e-6:
                query_t = query_t + 1e-6
            if np.abs(query_t - control_knot_ts[-2]) < 1e-6:
                query_t = query_t - 1e-6

            assert query_t >= control_knot_ts[1]
            assert query_t <= control_knot_ts[-2]
            
            idx_end = np.searchsorted(control_knot_ts, query_t)
            idx_start = idx_end - 1
            assert idx_start>=0
            assert control_knot_ts[idx_start]<=query_t
            assert control_knot_ts[idx_end]>=query_t
            
            ctrl_knot_t_start = control_knot_ts[idx_start]
            ctrl_knot_delta_t = control_knot_ts[idx_end] - control_knot_ts[idx_start]
            
            elapsed_t = query_t - ctrl_knot_t_start
            tau = elapsed_t/ctrl_knot_delta_t # normalize to (0,1)
            
            segment = torch.stack([control_knot_poses[idx_start-1],control_knot_poses[idx_start], 
                                control_knot_poses[idx_end], control_knot_poses[idx_end]], dim=0).unsqueeze(0)
            tau = torch.tensor(tau).reshape([1,1])
            
            pose_interp = cubic_bspline_interpolation(segment, tau)[0][0]  # SE3, [7]
        return pose_interp 

    def initialize_gaussian_scene(self, events_stream, num_pixels_to_sample, T_cam_to_wld, threshold = 0.1, depth_map=None):
        if depth_map is not None:
            assert(depth_map.dim()==2)

        threshold = self.config["event"]["threshold"]
        if depth_map is None:
            # depth = 100
            num_pts = 100_000
            print(f"Generating random point cloud ({num_pts})...")
            
            # We create random points inside the bounds of the synthetic Blender scenes
            # xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
            # xyz = np.random.random((num_pts, 3)) * 7.0 - 3.5
            bounding_size = self.config["bounding_size"]
            xyz = np.random.random((num_pts, 3))*bounding_size - 0.5*bounding_size
            
            shs = np.random.random((num_pts, 3)) / 255.0
            xyz = torch.from_numpy(xyz).cuda().float()
            shs = torch.from_numpy(shs).cuda().float()
            pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=torch.zeros_like(xyz).cuda().float())
        
        else:
            print('Use provided depth map for scene initialization...')
            events_map = self.post_process_event_image(events_stream)
    
            # sample pixels with events
            fx = self.dataset.fx
            fy = self.dataset.fy
            cx = self.dataset.cx
            cy = self.dataset.cy
            H = self.dataset.H
            W = self.dataset.W
            
            min_depth = self.config["mapping"]["min_depth"]
            max_depth = self.config["mapping"]["max_depth"]
            if self.config["initialization"]["gaussian_init_sfm_mask"]==1: # using event map mask; 
                pixel_uv_with_event = torch.where(events_map.abs() > threshold)
                sampled_event_pixel_idx = torch.randperm(pixel_uv_with_event[0].shape[0])[:num_pixels_to_sample]
                indice_h = pixel_uv_with_event[0][sampled_event_pixel_idx].to(self.device)
                indice_w = pixel_uv_with_event[1][sampled_event_pixel_idx].to(self.device)
            elif self.config["initialization"]["gaussian_init_sfm_mask"]==0: # using depth map mask
                pixel_uv_with_event = torch.where((depth_map.abs() > min_depth) & (depth_map.abs() < max_depth))
                sampled_event_pixel_idx = torch.randperm(pixel_uv_with_event[0].shape[0])[:num_pixels_to_sample]
                indice_h = pixel_uv_with_event[0][sampled_event_pixel_idx].to(self.device)
                indice_w = pixel_uv_with_event[1][sampled_event_pixel_idx].to(self.device)
            elif self.config["initialization"]["gaussian_init_sfm_mask"]==2: # using both event and depth map mask
                pixel_uv_with_event = torch.where((depth_map.abs() > min_depth) & (depth_map.abs() < max_depth))
                sampled_event_pixel_idx = torch.randperm(pixel_uv_with_event[0].shape[0])[:int(num_pixels_to_sample*0.5)]
                indice_h1 = pixel_uv_with_event[0][sampled_event_pixel_idx].to(self.device)
                indice_w1 = pixel_uv_with_event[1][sampled_event_pixel_idx].to(self.device)
                
                pixel_uv_with_event = torch.where(events_map.abs() > threshold)
                sampled_event_pixel_idx = torch.randperm(pixel_uv_with_event[0].shape[0])[:int(num_pixels_to_sample*0.5)]
                indice_h2 = pixel_uv_with_event[0][sampled_event_pixel_idx].to(self.device)
                indice_w2 = pixel_uv_with_event[1][sampled_event_pixel_idx].to(self.device)
                
                indice_h = torch.cat([indice_h1, indice_h2])
                indice_w = torch.cat([indice_w1, indice_w2])
            else:
                raise ValueError("wrong option for gaussian_init_sfm_mask")
            
            
            #
            sampled_rays = get_camera_rays(H, W, fx, fy, cx, cy, type='OpenCV').to(self.device)
            sampled_rays = sampled_rays[indice_h, indice_w, :]
        
            depth = depth_map[indice_h, indice_w].unsqueeze(-1).float()
    
            sampled_rays = (sampled_rays * depth).transpose(1,0).to(self.device)
            #
            sampled_rays = torch.matmul(T_cam_to_wld[:3, :3], sampled_rays) + T_cam_to_wld[:3, 3].unsqueeze(-1) 
            points = sampled_rays.transpose(1, 0)
            # colors
            min = events_map.min()
            max = events_map.max()
            events_map = (events_map - min) / (max - min)
            colors = events_map[indice_h, indice_w].unsqueeze(-1).repeat(1, 3)
            # normals
            normals = torch.zeros_like(colors)
            normals[:, 2] = 1.
            # create pcd
            pcd = BasicPointCloud(points=points, colors=colors, normals=normals)

        return pcd    

    def initialization(self, init_batch, niters, traj_mode='cspline', depth_map=None):
        traj_mode = 'linear'
        
        num_batch = len(init_batch)
        frame_ts_all = []
        frame_id_all = []
        events_all = []
        preSum_event_batch = [0]
        for i in range(num_batch):
            frame_id = init_batch[i]["frame_id"].item()
            frame_ts = init_batch[i]["pose_ts"].item()
            frame_id_all.append(frame_id)
            frame_ts_all.append(frame_ts)
            cur_event = init_batch[i]['events'].squeeze().to(self.device)
            preSum_event_batch.append(cur_event.shape[0]+preSum_event_batch[-1])
            events_all.append(cur_event)
        num_events_to_skip = self.config["num_events_to_skip"]
        preSum_event_batch[0] = preSum_event_batch[0]+num_events_to_skip
        preSum_event_batch[-1] = preSum_event_batch[-1]-num_events_to_skip
        
        # Using only random initialization only
        # TODO : Try with the depth map based initialization
        if depth_map is None:
            print('#########################   random initialization   #########################')
            self._initialize_model_from_random_points()
            # TODO : Need to initialize the triangle if not done earlier
        else:
            # initialize Gaussians with depth_map
            print('#########################   initialization with depth_map   #########################')
            # initialize Gaussians
            if self.config["initialization"]["retain_pose"]:
                assert(len(self.ctrl_knot_se3_all)>0)
                idx_ = frame_id_all[-1]
                T_cam2wld_Rt = se3_to_SE3_m44(self.ctrl_knot_se3_all[idx_]).cuda()
            else:
                T_cam2wld_Rt = torch.eye(4).to(self.device)
            
            gaussian_num_sfm = self.config["initialization"]["gaussian_num_sfm"]
            pcd = self.initialize_gaussian_scene(events_all[0], gaussian_num_sfm, T_cam2wld_Rt, depth_map=depth_map)
            # self.gs_model.create_from_tensor_pcd(pcd, spatial_lr_scale=8.23)
            # self.gs_model.training_setup(self.gs_opt_cfg)
            
            # feature_dim = 32 if self.gs_cfg.app_opt else None
            if self.config["retain_old_gs"]:
                print(f"============================= retain old gs ===============================")
                # TODO : Change the code
                new_triangles = TriangleModel(sh_degree = 0, config=self.config)
                new_triangles.create_from_pcd(pcd, radii=None, opacity=self.opt.set_weight, set_sigma=self.opt.set_sigma)
                
                new_triangles.training_setup(
                    self.opt,
                    self.opt.feature_lr,
                    self.opt.weight_lr,
                    self.opt.lr_triangles_points_init,
                )

                self.triangles.densification_postfix(new_triangles.vertices, new_triangles.vertex_weight, new_triangles._features_dc, new_triangles._features_rest, new_triangles._triangle_indices)
                
                
                # new_gs = pcd_2_gs(
                #         points= pcd.points.detach(),
                #         init_opacity=self.gs_cfg.init_opa,
                #         init_scale=self.gs_cfg.init_scale,
                #         scene_scale=self.scene_scale,
                #         sh_degree=self.gs_cfg.sh_degree,
                #         feature_dim=feature_dim,
                #         device=self.device,
                #         )
                # add_new_gs(self.splats, self.optimizers, new_gs)
            # else:
            #     feature_dim = 32 if self.gs_cfg.app_opt else None
            #     self.splats, self.optimizers = create_splats_with_optimizers(
            #         "sfm",
            #         init_num_pts=self.gs_cfg.init_num_pts,
            #         init_extent=self.gs_cfg.init_extent,
            #         init_opacity=self.gs_cfg.init_opa,
            #         init_scale=self.gs_cfg.init_scale,
            #         scene_scale=self.scene_scale,
            #         sh_degree=self.gs_cfg.sh_degree,
            #         sparse_grad=self.gs_cfg.sparse_grad,
            #         batch_size=self.gs_cfg.batch_size,
            #         feature_dim=feature_dim,
            #         device=self.device,
            #         points=pcd.points
            #     )

        #==========================================
        # TODO : Triangle densification strategy required
        #==========================================        
        
        frame_ts_all_withzero = frame_ts_all.copy()
        frame_ts_all_withzero.insert(0, 0.0)
        
        events_all = torch.cat(events_all, dim=0)
        ts_all = events_all[:,2].detach().cpu().numpy()
        t_data_min =  ts_all.min()
        t_data_max = ts_all.max()
        event_total_num = ts_all.shape[0]
        
        incre_sampling_seg_num_expected = self.config["initialization"]["incre_sampling_seg_num_expected"]
        min_n_winsize = self.config["initialization"]["min_n_winsize"]
        max_n_winsize = self.config["initialization"]["max_n_winsize"]
        
        incre_sampling_segs_end = np.linspace(10, event_total_num-10, incre_sampling_seg_num_expected).astype(int)
        start_tmp_ = np.searchsorted(incre_sampling_segs_end, max_n_winsize)
        if(incre_sampling_segs_end[start_tmp_]<=max_n_winsize):
            start_tmp_ = start_tmp_+1
        incre_sampling_segs_end = incre_sampling_segs_end[start_tmp_:]
        assert incre_sampling_segs_end.shape[0]>1
        assert incre_sampling_segs_end[0]>max_n_winsize
        events_per_seg = incre_sampling_segs_end[1]-incre_sampling_segs_end[0]
        incre_sampling_seg_num = incre_sampling_segs_end.shape[0]
        print(f"*******************************  events_num: {event_total_num}  *********************************")
        print(f"incremental sampling number: {incre_sampling_seg_num}")
        print(f"****events_per_seg={events_per_seg}, min_n_winsize={min_n_winsize},max_n_winsize={max_n_winsize}, ")
        
        ctrl_knot_ts = frame_ts_all.copy()
        ctrl_knot_ts.insert(0, t_data_min)
        if ctrl_knot_ts[-1]<frame_ts_all[-1]:
            ctrl_knot_ts.append(frame_ts_all[-1])
        ctrl_knot_ts[0] = ctrl_knot_ts[0]-0.0001
        ctrl_knot_ts[-1] = ctrl_knot_ts[-1]+0.0001
        ctrl_knot_idx = frame_id_all.copy()
        ctrl_knot_idx.insert(0, frame_id_all[0]-1)
        
        print(f"****t_data_min={t_data_min}, t_data_max={t_data_max}")
        print(f"control knot ts: {ctrl_knot_ts}")
        
        
        if len(self.ctrl_knot_ts_all)==0 or (not self.config["initialization"]["retain_pose"]):
            print("============== initialize control knots with random noise ==============")
            # set poses of control knots
            if self.config["use_gt_pose_to_opt"]:
                print("Setting trajectory")
                ctrl_knot_se3 = pp.randn_se3(len(ctrl_knot_ts), sigma=0.001) #LieTensor
                ctrl_knot_se3[0] = SE3_to_se3(self.pose_gt[frame_id_all[0]-1])
                for i in range(1, len(ctrl_knot_ts)):
                    ctrl_knot_se3[i] = SE3_to_se3(self.pose_gt[frame_id_all[i-1]])
            else:
                ctrl_knot_se3 = pp.randn_se3(len(ctrl_knot_ts), sigma=0.001) #LieTensor
        else:
            print("============== initialize control knots with pose of last stage ==============")
            assert(len(self.ctrl_knot_se3_all) == len(ctrl_knot_ts))
            ctrl_knot_se3 = pp.randn_se3(len(ctrl_knot_ts), sigma=0.001) #LieTensor
            for i in range(len(ctrl_knot_ts)):
                ctrl_knot_se3[i] = self.ctrl_knot_se3_all[i]
            
        print("finish control knots initialization")
        
        ctrl_knot_se3 = torch.nn.Parameter(ctrl_knot_se3, requires_grad=True)
        pose_optimizer = torch.optim.Adam([{"params": ctrl_knot_se3, "lr": self.config['pose_lr']}])
        
        color_channels=self.config["mapping"]["color_channels"]
        
        training_batch_size = self.config["initialization"]["training_batch_size"]
        visualize_every_iter = self.config["initialization"]["visualize_every_iter"]
        blur_sigma = self.config["blur_sigma"]

        #==========================================
        # Triangle specific parameters
        need_delaunay = False
        run_restricted_delaunay = self.opt.densify_until_iter + 1000


        # define the scheduler for sigma and opacity
        initial_sigma = self.opt.set_sigma
        final_sigma = 0.0001
        sigma_start = self.opt.sigma_start
        total_iters = self.opt.sigma_until

        init_opacity = 0.1
        final_opacity = .9999
        total_iters_opacity = self.opt.final_opacity_iter

        lambda_weight = self.opt.lambda_weight
        prune_triangles = self.opt.prune_triangles_threshold
        prune_size = self.opt.prune_size
        start_upsampling = self.opt.start_upsampling
        splitt_large_triangles = self.opt.splitt_large_triangles
        #==========================================

        for iter in range(niters):
            # if i%100 == 0:
            #     blur_sigma = blur_sigma/2

            #===============================================================
            if need_delaunay:
                with torch.no_grad():
                    self.triangles.run_restricted_delaunay()
                need_delaunay = False

            # Supersampling
            if iter == start_upsampling:
                self.triangles.scaling = self.opt.upscaling_factor
            if iter == start_upsampling + 5000:
                self.triangles.scaling = 4

            self.triangles.update_learning_rate(iter)

            # Sigma schedule
            if iter < sigma_start:
                current_sigma = initial_sigma
            else:
                progress = (iter - sigma_start) / (total_iters - sigma_start)
                progress = min(progress, 1.0)
                current_sigma = initial_sigma - (initial_sigma - final_sigma) * progress
            self.triangles.set_sigma(current_sigma)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iter % 1000 == 0:
                self.triangles.oneupSHdegree()

            #===================================================================
            
            loss_event = torch.zeros([1], dtype=torch.float64, device=self.device, requires_grad=True)
            loss_no_event = torch.zeros([1], dtype=torch.float64, device=self.device, requires_grad=True)
            loss_ssim = torch.zeros([1], dtype=torch.float64, device=self.device, requires_grad=True)
            loss_white_balance = torch.zeros([1], dtype=torch.float64, device=self.device, requires_grad=True)
            loss_tr = torch.zeros([1], dtype=torch.float64, device=self.device, requires_grad=True)
            
            # incremental random sampling
            random_start_end_idx = []
            for ii in range(incre_sampling_seg_num):
                end_idx_ = incre_sampling_segs_end[ii]
                winsize_ = np.random.randint(min_n_winsize, max_n_winsize)
                start_idx_ = end_idx_-winsize_
                random_start_end_idx.append([start_idx_, end_idx_])
            indices = np.random.permutation(len(random_start_end_idx)).tolist()[:training_batch_size]
            
            list_img_ev_start = []
            list_img_ev_end = []
            list_gt_events_acc = []
            list_syn_event_acc = []
            
            linlog_thres = self.config["event"]["linlog_thres"]
            # intra-chunk sampling
            for j in indices:
                idx_ev_start = random_start_end_idx[j][0]
                idx_ev_end = random_start_end_idx[j][1]
                selected_event_stream = events_all[idx_ev_start:idx_ev_end]
                t_ev_start = ts_all[idx_ev_start]
                t_ev_end = ts_all[idx_ev_end]

                T_SE3_ev_start = self.get_poses_lie(ctrl_knot_se3, ctrl_knot_ts, t_ev_start, mode=traj_mode)
                T_SE3_ev_end = self.get_poses_lie(ctrl_knot_se3, ctrl_knot_ts, t_ev_end, mode=traj_mode)
                
                # forward
                # TODO : Need to change for the triangle setting
                c2w_start = T_SE3_ev_start.matrix().unsqueeze(0).cuda()  #[1, 4, 4]
                render_pkg_start = self.render_triangles(c2w_start)
                c2w_end = T_SE3_ev_end.matrix().unsqueeze(0).cuda()  #[1, 4, 4]
                render_pkg_end = self.render_triangles(c2w_end)

                img_ev_start = render_pkg_start["render"]
                img_ev_end = render_pkg_end["render"]

                image = render_pkg_start["render"]
                depth = render_pkg_start["expected_depth"]
                opacity = render_pkg_start["rend_alpha"]
                median_depth = render_pkg_start["render_depth_median"]
                normals = render_pkg_start["rend_normal"]
                surf_normals = render_pkg_start["surf_normal"]

                # print(f"from initialization........................")
                # print(f" Image shape ; {image.shape}")
                # print(f"Depth shape : {depth.shape}")
                # print(f"opacity shape : {opacity.shape}")
                # print(f"median_depth shape : {median_depth.shape}")
                # print(f"normals shape : {normals.shape}")
                # print(f"surf_normal shape : {surf_normals.shape}")

                if self.config["use_linLog"]:
                    pred_linlog_start = lin_log(img_ev_start*255, linlog_thres=linlog_thres) # (B, Nevs, 1)
                    pred_linlog_end = lin_log(img_ev_end*255, linlog_thres=linlog_thres) # (B, Nevs, 1)
                    syn_event_acc = pred_linlog_end - pred_linlog_start
                else:
                    # syn_event_acc = (log(img_ev_end) - log(img_ev_start))
                    eps = 0.1
                    syn_event_acc = torch.log(img_ev_end**2.2+eps)-torch.log(img_ev_start**2.2+eps)
                
                # compute event loss
                gt_events_acc = self.post_process_event_image(selected_event_stream, _sigma=blur_sigma)    
                
                # if self.config["use_mask_event_loss"]:
                event_mask = (gt_events_acc.abs() != 0).float() 
                no_event_mask = (gt_events_acc.abs() == 0).float()
                
                # no-event loss
                no_event_gaussian_cov = self.config["mapping"]["no_event_gaussian_cov"]
                gt_no_events = self.config["event"]["threshold"] * no_event_gaussian_cov * torch.randn_like(gt_events_acc).cuda()
                gt_no_events = gt_no_events*no_event_mask
                
                if self.config["mapping"]["color_channels"]==3:
                    gt_events_acc = gt_events_acc.unsqueeze(-1).repeat(1, 1, 3)
                    # gt_events_acc = np.tile(gt_events_acc[..., None], (1, 1, 3))
                    gt_events_acc = gt_events_acc*self.color_mask
                    syn_event_acc = syn_event_acc*self.color_mask
                
                if self.config["seprate_event_noevent_loss"]:
                    loss_event = loss_event + (event_mask*(gt_events_acc - syn_event_acc)**2).sum() / event_mask.sum()
                    loss_no_event = loss_no_event + (no_event_mask*(gt_no_events - syn_event_acc)**2).sum() / no_event_mask.sum()
                else:
                    # gt_events_acc = gt_events_acc+gt_no_events # TODO: xiugai !!!
                    gt_events_acc = gt_events_acc
                    loss_event = loss_event + ((gt_events_acc - syn_event_acc)**2).mean()
                
                
                if self.config["mapping"]["color_channels"]==3:
                    loss_ssim = loss_ssim + compute_ssim_loss(gt_events_acc, syn_event_acc, channel=3)
                else:
                    loss_ssim = loss_ssim + compute_ssim_loss(gt_events_acc, syn_event_acc, channel=1)
                    
                # # visualize vector mask on event image
                # vector_vis_event = gt_events_acc*self.vector_mask
                # vector_events_acc = render_ev_accumulation(vector_vis_event.cpu().numpy(), self.dataset.H, self.dataset.W)
                # save_dir = os.path.join(self.config["data"]["output"], self.config["data"]["exp_name"])
                # img_path = os.path.join(save_dir, f"vector_mask_event.jpg")  
                # imageio.imwrite(img_path, vector_events_acc)
                # os.system.exit(0)
                
                list_img_ev_start.append(img_ev_start.detach())
                list_img_ev_end.append(img_ev_end.detach())
                list_gt_events_acc.append(gt_events_acc)
                list_syn_event_acc.append(syn_event_acc.detach())
            
            # rgb loss
            if self.config["mapping"]["use_rgb_loss"]:
                rgb_idx_ = np.random.randint(num_batch)
                rgb_c2w_ = init_batch[rgb_idx_]["c2w"].cuda()
                rgb_gt_ = init_batch[rgb_idx_]["rgb"][0][:, :,0].cuda()
                # TODO : Need to change for the triangle models
                render_pkg_rgb = self.render_triangles(rgb_c2w_)
                rgb_est = render_pkg_rgb["render"]
                rgb_w = self.config["mapping"]["loss_rgb_weight"]
                loss_rgb = rgb_w*((rgb_est-rgb_gt_)**2).mean()
            else:
                loss_rgb = torch.zeros([1], dtype=torch.float64, device=self.device, requires_grad=True)
            
            # tikhonov regularization loss
            tr_w = self.config["mapping"]["tr_loss_weight"]
            # print(f"TR LOSS : the depth shape : {render_pkg_start['expected_depth'].squeeze().unsqueeze(-1).shape}")
            loss_tr = tr_w*tikhonov_regularization(render_pkg_start["expected_depth"].squeeze().unsqueeze(-1))
            
            if self.config["mapping"]["use_white_balance_loss"]:
                # white balance loss 
                white_balance_weight = self.config["mapping"]["white_balance_weight"]
                loss_white_balance = white_balance_weight*compute_white_balance_loss(img_ev_end.mean())
            
            # TODO : need to apply triangle adaptive control stretegy
            # self.strategy.step_pre_backward(
            #     params=self.splats,
            #     optimizers=self.optimizers,
            #     state=self.strategy_state,
            #     step=iter,
            #     info=render_pkg_end["info"],
            # )
            
            loss_event = loss_event/len(indices)
            loss_ssim = loss_ssim/len(indices)
            loss_no_event = loss_no_event/len(indices)
            # summary the loss
            factor_ = self.config["mapping"]["ssim_loss_factor_"]
            loss_event = (1-factor_)*loss_event
            loss_ssim = factor_*loss_ssim
            
            # # isotropic loss
            # scaling = self.splats["scales"]  # [N, 3]
            # isotropic_loss_all = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            # iso_w = self.config["mapping"]["loss_isotropic_weight"]
            # loss_isotropic = iso_w * isotropic_loss_all.mean()
            
            # loss_total = loss_event + loss_ssim + loss_isotropic + loss_rgb 
            loss_total = loss_event + loss_ssim + loss_no_event + loss_white_balance + loss_tr

            #=============================================================================================
            # Pose gradient check
            # g = torch.autograd.grad(loss_total, c2w_start, retain_graph=True, allow_unused=True)
            # print("d loss / d c2w:", g)
            #=============================================================================================
            
            # optimize 
            pose_optimizer.zero_grad()
            loss_total.backward()
            torch.cuda.synchronize()

            #=============================================================================================
            # Pose gradient check
            # print("grad:", ctrl_knot_se3.grad is None,
            #     None if ctrl_knot_se3.grad is None else ctrl_knot_se3.grad.abs().max().item())
            # print("c2w requires_grad:", c2w_start.requires_grad)
            #=============================================================================================

            #======================================================
            # Mesh Splatting Densification
            image_size = render_pkg_start["scaling"].detach()
            mask = image_size > self.triangles.image_size
            self.triangles.image_size[mask] = image_size[mask]

            importance_score = render_pkg_start["max_blending"].detach()
            mask = importance_score > self.triangles.importance_score
            self.triangles.importance_score[mask] = importance_score[mask]

            pixel_count = render_pkg_start["triangle_was_rendered"].detach() # Not used but could be useful. Gives per triangle, the number of pixels it covered in the current render
            mask = pixel_count > self.triangles.pixel_count
            self.triangles.pixel_count[mask] = pixel_count[mask]

            with torch.no_grad():
                # Handle pruning operations
                if iter % 500 == 0 and iter < run_restricted_delaunay:
                    
                    # Building masks to delete triangles
                    triangle_vertex_weights = self.triangles.opacity_activation(
                        self.triangles.vertex_weight[self.triangles._triangle_indices]
                    ) 
                    min_weights = triangle_vertex_weights.min(dim=1).values

                    mask_opacity     = (min_weights <= prune_triangles).squeeze()              # delete if too low
                    mask_importance  = (self.triangles.importance_score <= prune_triangles).squeeze()  # delete if too low
                    mask_size        = (self.triangles.image_size > prune_size).squeeze()                 # delete if too big

                    delete_mask = mask_opacity | mask_size

                    # if number_of_training_views < 500: # only delete if the number of views are below 500. Otherwise, we might delete too much
                    delete_mask = delete_mask | mask_importance

                    keep_mask   = ~delete_mask 

                    if iter > self.opt.start_pruning:
                        self.triangles.prune_triangles(keep_mask)
                
                    # We prune vertices that are no longer used
                    device = self.triangles.vertices.device
                    used_vertex_mask = torch.zeros(self.triangles.vertices.shape[0], 
                                                dtype=torch.bool, 
                                                device=device)
                    if self.triangles._triangle_indices.numel() > 0:
                        flat_indices = self.triangles._triangle_indices.flatten()
                        used_vertex_mask[flat_indices] = True
                    
                    weight_mask = (self.triangles.get_vertex_weight.squeeze() >= prune_triangles)
                    mask_out = self.triangles.vertices.shape[0]
                    vertex_mask = weight_mask[:mask_out] | used_vertex_mask

                    self.triangles._prune_vertices(vertex_mask)


                    triangle_vertex_weights = self.triangles.opacity_activation(
                        self.triangles.vertex_weight[self.triangles._triangle_indices]
                    )  # [T,3]

                    needs_densification = (iter < self.opt.densify_until_iter and 
                                        iter % self.opt.densification_interval == 0 and 
                                        iter > self.opt.densify_from_iter)
                    
                    if needs_densification:
                        self.triangles.add_new_gs_original(cap_max=self.opt.max_points, splitt_large_triangles=splitt_large_triangles)
    

                    if iter > self.opt.start_opacity_floor:
                        start_iter = self.opt.start_opacity_floor
                        end_iter = total_iters_opacity  # the iteration where you want to reach final_opacity
                        a = min(1.0, max(0.0, (iter - start_iter) / max(1, end_iter - start_iter)))
                        current_opacity = init_opacity + (final_opacity - init_opacity) * a
                        current_opacity = min(current_opacity, final_opacity)
                        self.triangles.update_min_weight(current_opacity)

                        prune_triangles += 0.01 
                        mask_out = self.triangles.vertices.shape[0]
                        triangle_vertex_weights = self.triangles.get_vertex_weight[:mask_out][self.triangles._triangle_indices]
                elif iter == run_restricted_delaunay:
                    need_delaunay = True
                elif iter % 500 == 0 and iter > run_restricted_delaunay + 1000:

                    if iter > self.opt.start_opacity_floor:
                        start_iter = self.opt.start_opacity_floor
                        end_iter = total_iters_opacity  # the iteration where you want to reach final_opacity
                        a = min(1.0, max(0.0, (iter - start_iter) / max(1, end_iter - start_iter)))
                        current_opacity = init_opacity + (final_opacity - init_opacity) * a
                        current_opacity = min(current_opacity, final_opacity)
                        self.triangles.update_min_weight(current_opacity)

                        prune_triangles += 0.01 
                        mask_out = self.triangles.vertices.shape[0]
                        triangle_vertex_weights = self.triangles.get_vertex_weight[:mask_out][self.triangles._triangle_indices]
                

                if iter < self.opt.iterations:
                    self.triangles.optimizer.step()
                    self.triangles.optimizer.zero_grad(set_to_none = True)

            #======================================================
            
            # densification
            # TODO : Need to change the densification strategy for the triangles
            # self.strategy.step_post_backward(
            #     params=self.splats,
            #     optimizers=self.optimizers,
            #     state=self.strategy_state,
            #     step=iter,
            #     info=render_pkg_end["info"],
            # )
            
            # TODO : Need to change the optimizer also
            # for optimizer in self.optimizers.values():
            #     optimizer.step()
            #     optimizer.zero_grad(set_to_none=True)
            
            # if self.config["opt_pose"] and i>1000:


            pose_optimizer.step()

            #============================================================================================
            # Pose update check
            # before = ctrl_knot_se3.detach().clone()
            # pose_optimizer.step()
            # print("max pose delta:", (ctrl_knot_se3.detach() - before).abs().max().item())
            #=============================================================================================

            # if self.config["opt_pose"]:
            #     if (depth_map is None) or (depth_map is not None and iter>200):
            #         pose_optimizer.step()

            # visualization
            save_path = os.path.join(self.config["data"]["output"], self.config["data"]["exp_name"], "initialization")
            if iter % visualize_every_iter == 0 and self.config["visualize_inter_img"]:
                print_str = f"iter {iter}, loss={loss_total.item()}, event={loss_event.item()}, ssim={loss_ssim.item()}, white={loss_white_balance.item()}, noevent={loss_no_event.item()}, tr={loss_tr.item()}"
                print(print_str)
                with torch.no_grad():
                    # plot pose estimation
                    accum_est_poses_Rt = []
                    accum_gt_poses_Rt = []
                    tag = f"pose"
                    
                    # # save ply file
                    # ply_path = os.path.join(save_path, f"pointCloud_it{iter}.ply")
                    # self.save_ply(ply_path)
                    
                    for n in range(num_batch):
                        if traj_mode == "cspline":
                            if n==0 or n==(num_batch-1):
                                continue
                        k = frame_id_all[n]
                        ts = frame_ts_all[n]                        
                        accum_est_poses_Rt.append(self.get_poses_lie(ctrl_knot_se3, ctrl_knot_ts, ts, mode=traj_mode).matrix()) 
                        accum_gt_poses_Rt.append(init_batch[n]['c2w'].squeeze())                        
                    
                    if self.config["dataset"] == "rpg_evo":
                        accum_gt_poses_Rt = accum_est_poses_Rt
                    # plot 
                    pose_evaluation(accum_gt_poses_Rt, accum_est_poses_Rt, 1, save_path, tag, f"init_f{iter:03}")
                    save_pose_as_kitti_evo(accum_gt_poses_Rt, accum_est_poses_Rt, save_path, f"init_f{iter:03}")
                    
                    if self.config["initialization"]["visualize_intermediate_img"]:
                        # save intermediate images 
                        images_combined = []
                        for m in range(len(list_img_ev_start)):
                            diff = (list_gt_events_acc[m] - list_syn_event_acc[m]) **2
                            
                            if color_channels == 1:
                                gt_events_acc = render_ev_accumulation(list_gt_events_acc[m].cpu().numpy(), self.dataset.H, self.dataset.W)
                                gt_events_grey = np.repeat(list_gt_events_acc[m].cpu().numpy()[:, :, np.newaxis], repeats=3, axis=2) 
                                
                                syn_event_acc = list_syn_event_acc[m].detach().cpu().numpy()
                                syn_event_acc1 = np.where(syn_event_acc > self.config["event"]["threshold"], syn_event_acc, 0)
                                syn_event_acc2 = np.where(syn_event_acc < -self.config["event"]["threshold"], syn_event_acc, 0)
                                syn_event_acc = syn_event_acc1 + syn_event_acc2
                                syn_event_acc = render_ev_accumulation(syn_event_acc, self.dataset.H, self.dataset.W)
                                
                                syn_event_grey = np.repeat(list_syn_event_acc[m].detach().abs().cpu().numpy()[:, :, np.newaxis], repeats=3, axis=2) 
                                diff = np.repeat(diff.detach().abs().cpu().numpy()[:, :, np.newaxis], repeats=3, axis=2)  
                                
                                # render_ev_accumulation(syn_event_acc.cpu().numpy(), self.dataset.H, self.dataset.W)
                                img_ev_start = np.repeat(list_img_ev_start[m].detach().cpu().numpy()[:, :, np.newaxis], repeats=3, axis=2)
                                img_ev_end = np.repeat(list_img_ev_end[m].detach().cpu().numpy()[:, :, np.newaxis], repeats=3, axis=2)
                            else:
                                gt_events_acc = render_ev_accumulation(list_gt_events_acc[m].sum(-1).cpu().numpy(), self.dataset.H, self.dataset.W)
                                syn_event_acc = list_syn_event_acc[m].detach().abs().cpu().numpy()
                                diff = diff.detach().abs().cpu().numpy()
                                
                                # render_ev_accumulation(syn_event_acc.cpu().numpy(), self.dataset.H, self.dataset.W)
                                img_ev_start = list_img_ev_start[m].detach().cpu().numpy()
                                img_ev_end = list_img_ev_end[m].detach().cpu().numpy()
                                
                                syn_event_grey = list_gt_events_acc[m].cpu().numpy()
                                gt_events_grey = list_gt_events_acc[m].cpu().numpy()

                            image_combined = np.concatenate([img_ev_start,img_ev_end, gt_events_acc, gt_events_grey,syn_event_acc, syn_event_grey, diff], axis=1)
                            images_combined.append(image_combined)
                        
                        images_combined = np.concatenate(images_combined, axis=0)
                        images_combined =  to8b(images_combined)
                        tag_img = "init_"
                        img_path = os.path.join(save_path, tag_img + f"f{iter:03}_img.jpg")  
                        imageio.imwrite(img_path, images_combined)
                    
                    # render ALL images with depth
                    if self.config["render_eventview_img"]:
                        images_combined = []
                        for n in range(num_batch):
                            if n%30!=0:
                                continue
                            idx_ = frame_id_all[n]
                            ts = frame_ts_all[n]
                            T_SE3 = self.get_poses_lie(ctrl_knot_se3, ctrl_knot_ts, ts, mode=traj_mode)
                            
                            # TODO : Need to change for the triangles splatting
                            c2w_ = T_SE3.matrix().unsqueeze(0).cuda()  #[1, 4, 4]
                            render_pkg_start = self.render_triangles(c2w_)
                            
                            # TODO : Need to find the correct depth
                            depth_img_gs = render_pkg_start["expected_depth"].squeeze().unsqueeze(-1)  # [H, W, 1]
                            depth_tmp = depth_img_gs.detach()
                            depth_tmp = depth_tmp.view(depth_tmp.shape[0], depth_tmp.shape[1], -1)
                            depth_img = apply_colormap(depth_tmp)
                            depth_img = depth_img.cpu().numpy()
                            depth_img = to8b(depth_img)
                            
                            # rendered_img = render_pkg_start[0][..., 0:3][0]  # [H, W, 3]
                            rendered_img = render_pkg_start["render"]  # [H, W] or [H, W, 3]
                            if rendered_img.shape[-1]!=3:
                                rendered_img = torch.tile(rendered_img[:,:,None], (1,1,3))
                            img = rendered_img.detach().cpu().numpy()
                            img = to8b(img)
                            
                            image_combined = np.concatenate([img, depth_img], axis=1)
                            images_combined.append(image_combined)
                        images_combined = np.concatenate(images_combined, axis=0)
                        im_name = f"iter_{iter}_vis.jpg"
                        imageio.imwrite(os.path.join(save_path, im_name), images_combined)
        # TODO : Need to change for triangle vertices
        print("********** Number of Vertices :", len(self.triangles.vertices))
        print("********** Number of Trianges :", len(self.triangles._triangles))
        
        # update est_poses
        for n in range(num_batch):
            k = frame_id_all[n]
            ts = frame_ts_all[n]                      
            self.est_c2w_data[k] = self.get_poses_lie(ctrl_knot_se3, ctrl_knot_ts, ts, mode=traj_mode).matrix().cuda()
            self.est_c2w_ts[k] = ts
        
        # render ALL images 
        if self.config["render_eventview_img"]:
            with torch.no_grad():
                for n in range(num_batch):
                    idx_ = frame_id_all[n]
                    ts = frame_ts_all[n]
                    T_SE3 = self.get_poses_lie(ctrl_knot_se3, ctrl_knot_ts, ts, mode=traj_mode)
                    
                    # TODO : Need to changes for the triangles
                    c2w_ = T_SE3.matrix().unsqueeze(0).cuda()  #[1, 4, 4]
                    rendered_img = self.render_triangles(c2w_)['render']
                    
                    rendered_img = rendered_img.detach().cpu().numpy()
                    rendered_img =  to8b(rendered_img)
                    im_name = f"frame{idx_}_init.jpg"
                    imageio.imwrite(os.path.join(save_path, im_name), rendered_img)
                    if self.config["evaluate_init_image"]:
                        imageio.imwrite(os.path.join(self.dataset.event_save_path, im_name), rendered_img)

                    #===================================================================================================================
                    # if self.config["render_tumvie_rgbCam_img"] and self.config["dataset"]=="tum_vie":
                    #     # render rbg camera image
                    #     # TODO : Need to change
                    #     img_ev_start_rgb = self.render_with_gsplat(self.gs_model, T_se3, render_tumvie_rgb=True)['image']
                    #     img_ev_start_rgb = np.repeat(img_ev_start_rgb.detach().cpu().numpy()[:, :, np.newaxis], repeats=3, axis=2)
                    #     img_path = os.path.join(save_path, tag_img + f"frame{idx_}_init_rgbCam.jpg")  
                    #     imageio.imwrite(img_path, to8b(img_ev_start_rgb))
                    #===================================================================================================================    
        
        # save control knot pose and ts
        for ii in range(len(ctrl_knot_idx)):
            idx_ = ctrl_knot_idx[ii]
            self.ctrl_knot_ts_all[idx_] = ctrl_knot_ts[ii]
            self.ctrl_knot_se3_all[idx_] = ctrl_knot_se3[ii].detach().clone()  # [], cuda
        
        print("======================= finished initialization  =========================")   


    def tracking(self, cur_frame_id, tracking_batch, niter=1000):
        num_batch = len(tracking_batch)
        
        print(f"Tracking frame{cur_frame_id-num_batch+1} to frame{cur_frame_id}")
        
        frame_ts_all = []
        frame_id_all = []
        events_all = []
        for i in range(num_batch):
            fid_ = tracking_batch[i]["frame_id"].item()
            frame_ts = tracking_batch[i]["pose_ts"].item()
            
            frame_id_all.append(fid_)
            frame_ts_all.append(frame_ts)
            
            cur_event = tracking_batch[i]['events'].squeeze()
            events_all.append(cur_event.to(self.device))
        events_all = torch.cat(events_all, dim=0)
        
        t_data_start = events_all[0][2].item()
        t_data_end = events_all[-1][2].item()
        
        # control knot management        
        active_ctrl_knot_idx = []
        for ii in range(len(frame_id_all)):
            print("add a new control knot")
            idx_ = frame_id_all[ii]
            self.ctrl_knot_ts_all[idx_] =  frame_ts_all[ii]
            active_ctrl_knot_idx.append(idx_)
            if not idx_ in self.ctrl_knot_se3_all:
                if self.config["use_gt_pose_to_opt"]:
                    self.ctrl_knot_se3_all[idx_] = SE3_to_se3(self.pose_gt[idx_])
                else:
                    # predict next control pose
                    self.ctrl_knot_se3_all[idx_] = self.predict_pose_se3_lie(self.ctrl_knot_se3_all[idx_-1], self.ctrl_knot_se3_all[idx_-2])
        
        if frame_ts_all[-1]>t_data_end:
            print(f"&&&&&&&&&&&        WARNNING: t_data_end-frame_ts_all[-1]={t_data_end-frame_ts_all[-1]}    &&&&&&&&&&&")
            print("&&&&&&&&&&&        Hard assignment:  frame_ts_all[-1] = t_data_end-1e-4    &&&&&&&&&&&")
            frame_ts_all[-1] = t_data_end-1e-4
        
        active_ctrl_knot_idx.insert(0, active_ctrl_knot_idx[0]-1)
        ctrl_knot_ts = []
        for idx_ in active_ctrl_knot_idx:
            ctrl_knot_ts.append(self.ctrl_knot_ts_all[idx_])
        # ctrl_knot_len = len(active_ctrl_knot_idx)
        
        # change to 2-knot batch tracking mode
        ctrl_knot_len = 2
        active_ctrl_knot_idx_src = active_ctrl_knot_idx.copy()
        ctrl_knot_ts_src = ctrl_knot_ts.copy()
        active_ctrl_knot_idx = [active_ctrl_knot_idx[0], active_ctrl_knot_idx[-1]]
        ctrl_knot_ts = [ctrl_knot_ts[0], ctrl_knot_ts[-1]]
        
        t_event_min = events_all[:, 2].min().cpu().numpy()
        t_event_max = events_all[:, 2].max().cpu().numpy()
        ctrl_knot_ts[0] = t_event_min-1e-6
        ctrl_knot_ts[-1] = t_event_max+1e-6
        
        if self.config["use_relative_pose_to_opt"]==1:
            ctrl_knot_se3_rel_opt = pp.identity_se3(ctrl_knot_len) #LieTensor
            ctrl_knot_se3_base = pp.identity_se3(ctrl_knot_len)
            for ii in range(ctrl_knot_len):
                idx_ = active_ctrl_knot_idx[ii]
                ctrl_knot_se3_base[ii] = self.ctrl_knot_se3_all[idx_]
            ctrl_knot_se3_rel_opt = torch.nn.Parameter(ctrl_knot_se3_rel_opt, requires_grad=True)
            pose_optimizer = torch.optim.Adam([{"params": ctrl_knot_se3_rel_opt, "lr": self.config['pose_lr']}])
        elif self.config["use_relative_pose_to_opt"]==0:
            ctrl_knot_se3 = pp.identity_se3(ctrl_knot_len) #LieTensor
            for ii in range(ctrl_knot_len):
                idx_ = active_ctrl_knot_idx[ii]
                ctrl_knot_se3[ii] = self.ctrl_knot_se3_all[idx_]
            ctrl_knot_se3 = torch.nn.Parameter(ctrl_knot_se3, requires_grad=True)        
            pose_optimizer = torch.optim.Adam([{"params": ctrl_knot_se3, "lr": self.config['pose_lr']}])
        elif self.config["use_relative_pose_to_opt"]==2:  # fix the first pose
            ctrl_knot_se3_base = self.ctrl_knot_se3_all[active_ctrl_knot_idx[0]-1] 
            ctrl_knot_se3_rel_opt = pp.identity_se3(ctrl_knot_len)
            for ii in range(ctrl_knot_len):
                ctrl_knot_se3_rel_opt[ii] = (self.ctrl_knot_se3_all[active_ctrl_knot_idx[ii]].Exp()*(ctrl_knot_se3_base.Inv().Exp())).Log()
            ctrl_knot_se3_rel_opt = torch.nn.Parameter(ctrl_knot_se3_rel_opt, requires_grad=True)
            pose_optimizer = torch.optim.Adam([{"params": ctrl_knot_se3_rel_opt, "lr": self.config['pose_lr']}])
        else:
            raise ValueError("wrong option for pose optimization")

        # optimize the pose of new added control knot
        num_totel_events = events_all.shape[0]
        _num_events_for_optim = self.config["num_events_window_for_tracking"]
        num_events_to_skip = self.config["num_events_to_skip"]
        
        # num_events_for_optim = np.min([num_totel_events-1, _num_events_for_optim])
        num_events_for_optim = np.min([_num_events_for_optim, num_totel_events-2*num_events_to_skip])-1
        print(f"*******************************  events_num: {num_totel_events}  *********************************")
        print(f"************************  N_window for init: {num_events_for_optim}  ***************")
        prev_loss = 1e6 
        
        mask_boundary_size = self.config["mask_boundary_size"]
        blur_sigma = self.config["blur_sigma"]
        
        linlog_thres = self.config["event"]["linlog_thres"]
        for i in range(niter):
            if i%100 == 0:
                blur_sigma = blur_sigma/2
            if self.config["use_relative_pose_to_opt"]==1:
                ctrl_knot_se3 = (ctrl_knot_se3_base.Exp()*ctrl_knot_se3_rel_opt.Exp()).Log()
            elif self.config["use_relative_pose_to_opt"]==2:
                ctrl_knot_se3 = (ctrl_knot_se3_rel_opt.Exp()*ctrl_knot_se3_base.Exp()).Log()
            
            # sample random chunks of events with num_events_for_optim events in total
            
            idx_ev_start = np.random.randint(num_events_to_skip, num_totel_events-num_events_for_optim-num_events_to_skip)
            t_ev_start = events_all[idx_ev_start][2].item()
            t_ev_end = events_all[idx_ev_start + num_events_for_optim][2].item()

            # interpolate event start pose + end pose
            T_SE3_ev_start = self.get_poses_lie(ctrl_knot_se3, ctrl_knot_ts, t_ev_start, mode='linear')
            T_SE3_ev_end = self.get_poses_lie(ctrl_knot_se3, ctrl_knot_ts, t_ev_end, mode='linear')
            
            # forward
            c2w_start = T_SE3_ev_start.matrix().unsqueeze(0).cuda()  #[1, 4, 4]
            render_pkg_start = self.render_triangles(c2w_start)
            c2w_end = T_SE3_ev_end.matrix().unsqueeze(0).cuda()  #[1, 4, 4]
            render_pkg_end = self.render_triangles(c2w_end)
            
            img_ev_start = render_pkg_start["render"]
            img_ev_end = render_pkg_end["render"]
            
            if self.config["use_linLog"]:
                pred_linlog_start = lin_log(img_ev_start*255, linlog_thres=linlog_thres) # (B, Nevs, 1)
                pred_linlog_end = lin_log(img_ev_end*255, linlog_thres=linlog_thres) # (B, Nevs, 1)
                syn_event_acc = pred_linlog_end - pred_linlog_start
            else:
                syn_event_acc = (log(img_ev_end) - log(img_ev_start))
            
            # compute event loss
            with torch.no_grad():
                gt_events_acc = self.post_process_event_image(events_all[idx_ev_start:idx_ev_start+num_events_for_optim, :], _sigma=blur_sigma)
                if self.config["only_track_event_area"]:
                    mask = (gt_events_acc.abs() > 0.1).double().detach()
                else:
                    mask = torch.ones_like(gt_events_acc)
                
                mask_boundary = torch.ones_like(mask)
                mask_boundary[:mask_boundary_size,:] = 0
                mask_boundary[-mask_boundary_size:,:] = 0
                mask_boundary[:,:mask_boundary_size] = 0
                mask_boundary[:,-mask_boundary_size:] = 0
                mask = mask * mask_boundary
                
                # visibility mask
                # TODO : Change the code
                alpha_start = render_pkg_start["rend_alpha"].squeeze()
                # alpha_start = alpha_start[0][:,:,0]  # [H, W]
                alpha_end = render_pkg_end["rend_alpha"].squeeze()
                # alpha_end = alpha_end[0][:,:,0]  # [H, W]
                tracking_mask_alpha_threshold = self.config["mapping"]["tracking_mask_alpha_threshold"]
                visibility_mask = ((alpha_start > tracking_mask_alpha_threshold)*(alpha_end > tracking_mask_alpha_threshold)).double().detach()
                
                mask = mask * visibility_mask
                
                # if self.config["dataset"] == "vector":
                #     mask = mask * self.vector_mask

                if self.config["use_uncertainty_in_Tracking"]:
                    # check uncertainty 
                    img_uncertainty = 1./ (ret_ev_start['uncertainty'] * ret_ev_end['uncertainty'] + 1.0)
                    mask = mask * img_uncertainty
            
            if self.config["mapping"]["color_channels"]==3:
                gt_events_acc = gt_events_acc.unsqueeze(-1).repeat(1, 1, 3)
                gt_events_acc = gt_events_acc*self.color_mask
                syn_event_acc = syn_event_acc*self.color_mask
                mask = mask.unsqueeze(-1).repeat(1, 1, 3)

            loss_event = (mask * (gt_events_acc - syn_event_acc)**2).sum() / mask.sum()
            if self.config["mapping"]["color_channels"]==3:
                loss_ssim = compute_ssim_loss(gt_events_acc, syn_event_acc, channel=3)
            else:
                loss_ssim = compute_ssim_loss(mask*gt_events_acc, mask*syn_event_acc, channel=1)
            
            factor_ = self.config["mapping"]["ssim_loss_factor_"]
            loss_event = (1-factor_)*loss_event
            loss_ssim = factor_*loss_ssim

            loss_total = loss_event + loss_ssim

            # # TODO:check early break criteria
            # if loss_event > prev_loss * 1.5:
            #     print('[tracker]: early break at iteration:', i, 'prev_loss:', prev_loss, 'cur_loss:', loss_event.item())
            #     break
            # prev_loss = loss_event.item()
            
            # backward
            pose_optimizer.zero_grad()
            loss_total.backward()
            torch.cuda.synchronize()
            if self.config["opt_pose"]:
                pose_optimizer.step()
            
            # visualization
            if i % 100 == 0 and self.config["visualize_inter_img"]:
                tracking_info_ = f"iter {i}, loss={loss_total.item()}, event={loss_event.item()}, ssim={loss_ssim.item()}"
                print(tracking_info_)
                path_to_save = os.path.join(self.config["data"]["output"], self.config["data"]["exp_name"], 'tracking')
                with torch.no_grad():
                    if self.config["visualize_inter_img_in_Tracking_BA"]:
                        # mask
                        gt_events_acc =  gt_events_acc*mask
                        syn_event_acc = syn_event_acc*mask
                        img_ev_start = img_ev_start*mask
                        img_ev_end = img_ev_end*mask
                        
                        # visualize event map using red-blue image
                        diff = (gt_events_acc - syn_event_acc) * 2
                        
                        if self.config["mapping"]["color_channels"] == 1:
                            diff = np.repeat(diff.detach().abs().cpu().numpy()[:, :, np.newaxis], repeats=3, axis=2)  
                            # render_ev_accumulation(diff.cpu().numpy(), self.dataset.H, self.dataset.W)
                            gt_events_acc = render_ev_accumulation(gt_events_acc.cpu().numpy(), self.dataset.H, self.dataset.W)
                            syn_event_acc = np.repeat(syn_event_acc.detach().abs().cpu().numpy()[:, :, np.newaxis], repeats=3, axis=2) 
                            # render_ev_accumulation(syn_event_acc.cpu().numpy(), self.dataset.H, self.dataset.W)
                            img_ev_start = np.repeat(img_ev_start.detach().cpu().numpy()[:, :, np.newaxis], repeats=3, axis=2)
                            img_ev_end = np.repeat(img_ev_end.detach().cpu().numpy()[:, :, np.newaxis], repeats=3, axis=2)
                            mask = torch.tile(mask[:,:,None], (1,1,3))
                        else:
                            gt_events_acc = render_ev_accumulation(gt_events_acc.sum(-1).cpu().numpy(), self.dataset.H, self.dataset.W)
                            img_ev_start = img_ev_start.detach().cpu().numpy()
                            img_ev_end = img_ev_end.detach().cpu().numpy()
                            syn_event_acc = syn_event_acc.detach().cpu().numpy()
                            diff = diff.detach().cpu().numpy()
                        
                        image_combined = np.concatenate([img_ev_start,img_ev_end, gt_events_acc, syn_event_acc, diff], axis=1)
                        image_combined =  to8b(image_combined)
                        frame_id = cur_frame_id
                        im_name = f"tracking_f{frame_id:03}_{i:04}_img.jpg"
                        imageio.imwrite(os.path.join(path_to_save, im_name), image_combined)
                        
                        # visualize mask
                        alpha_end = torch.tile(alpha_end[:,:,None], (1,1,3))
                        alpha_img_end = alpha_end.detach().cpu().numpy()
                        alpha_start = torch.tile(alpha_start[:,:,None], (1,1,3))
                        alpha_img_start = alpha_start.detach().cpu().numpy()
                        
                        alpha_start_mask = (alpha_start > 0.8).double().detach()
                        alpha_start_mask = alpha_start_mask.detach().cpu().numpy()
                        
                        alpha_end_mask = (alpha_end > 0.8).double().detach()
                        alpha_end_mask = alpha_end_mask.detach().cpu().numpy()
                        
                        visibility_mask = torch.tile(visibility_mask[:,:,None], (1,1,3))
                        vis_mask_img = visibility_mask.detach().cpu().numpy()
                        
                        mask_img = mask.detach().cpu().numpy()
                        mask_img1 = np.concatenate([alpha_img_start, alpha_img_end, mask_img], axis=1)
                        mask_img2 = np.concatenate([alpha_start_mask, alpha_end_mask, vis_mask_img], axis=1)
                        tracking_mask_img = np.concatenate([mask_img1, mask_img2], axis=0)
                        tracking_mask_img =  to8b(tracking_mask_img)
                        im_name = f"tracking_f{frame_id:03}_{i:04}_mask_img.jpg"
                        imageio.imwrite(os.path.join(path_to_save, im_name), tracking_mask_img)
                    
                    # self.est_c2w_data[cur_frame_id] = se3_to_SE3_m44(self.get_poses(mutable_control_knot_poses, ctrl_knot_ts, frame_ts, mode='linear'))
                    for ii in range(len(frame_ts_all)):
                        ts = frame_ts_all[ii]
                        idx = frame_id_all[ii]
                        self.est_c2w_data[idx] = self.get_poses_lie(ctrl_knot_se3, ctrl_knot_ts, ts, mode='linear').matrix().cuda()
                    if self.config["dataset"] == "rpg_evo":
                        self.pose_gt = self.est_c2w_data
                    pose_evaluation(self.pose_gt, self.est_c2w_data, 1, path_to_save,"pose", f"tracking_f{frame_id:03}_{i:04}")
                    save_pose_as_kitti_evo(self.pose_gt, self.est_c2w_data, path_to_save, f"tracking_f{frame_id:03}_{i:04}")
        
        if self.config["use_relative_pose_to_opt"]==1:
            ctrl_knot_se3 = (ctrl_knot_se3_base.Exp()*ctrl_knot_se3_rel_opt.Exp()).Log()
        elif self.config["use_relative_pose_to_opt"]==2:
            ctrl_knot_se3 = (ctrl_knot_se3_rel_opt.Exp()*ctrl_knot_se3_base.Exp()).Log()
        
        # update est_poses
        for ii in range(len(frame_ts_all)):
            ts = frame_ts_all[ii]
            idx = frame_id_all[ii]
            self.est_c2w_data[idx] = self.get_poses_lie(ctrl_knot_se3, ctrl_knot_ts, ts, mode='linear').matrix()
            self.est_c2w_ts[idx] = ts
        
        # save control knot pose
        # self.ctrl_knot_se3_all[cur_frame_id-1] = mutable_control_knot_poses[0].detach().clone()  # [], cuda # TODO: 保存这个control knot pose后，轨迹反而不好，待研究
        for ii in range(len(active_ctrl_knot_idx)):
            # if self.config["use_relative_pose_to_opt"]!=2:
            if ii==0:
                continue
            idx = active_ctrl_knot_idx[ii]
            self.ctrl_knot_se3_all[idx] = ctrl_knot_se3[ii].detach().clone()  # [],lieTensor, cuda
        # interpolate the control knot pose
        for ii in range(len(active_ctrl_knot_idx_src)-2):
            ii = ii+1
            ts = ctrl_knot_ts_src[ii]
            idx = active_ctrl_knot_idx_src[ii]
            self.ctrl_knot_se3_all[idx] = self.get_poses_lie(ctrl_knot_se3, ctrl_knot_ts, ts, mode='linear').Log().detach().clone()  # se3,lieTensor, cuda
        
        print("======================= finished tracking  =========================")    

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        
    def bundle_adjustment(self, BA_batch,  _traj_mode, _num_iters=100001, _seg_num=50, _opt_pose=True, _global_BA=False):
        # Bundle adjustment 
        
        num_batch = len(BA_batch)
        frame_ts_all = []
        frame_id_all = []
        events_all = []
        preSum_event_batch = [0]
        for i in range(num_batch):
            frame_id = BA_batch[i]["frame_id"].item()
            frame_ts = self.est_c2w_ts[frame_id]
            frame_id_all.append(frame_id)
            frame_ts_all.append(frame_ts)
            cur_event = BA_batch[i]['events'].squeeze().to(self.device)
            preSum_event_batch.append(cur_event.shape[0]+preSum_event_batch[-1])
            events_all.append(cur_event)
        num_events_to_skip = self.config["num_events_to_skip"]
        preSum_event_batch[0] = preSum_event_batch[0]+num_events_to_skip
        preSum_event_batch[-1] = preSum_event_batch[-1]-num_events_to_skip
        
        events_all = torch.cat(events_all, dim=0)
        ts_all = events_all[:,2].detach().cpu().numpy()
        t_data_min =  ts_all.min()
        t_data_max = ts_all.max()
        
        # "incremental" random sampling
        event_total_num = ts_all.shape[0]
        incre_sampling_seg_num_expected = _seg_num
        min_n_winsize = self.config["BA"]["min_n_winsize"]
        max_n_winsize = self.config["BA"]["max_n_winsize"]
        
        incre_sampling_segs_end = np.linspace(10, event_total_num-10, incre_sampling_seg_num_expected).astype(int)
        start_tmp_ = np.searchsorted(incre_sampling_segs_end, max_n_winsize)
        if(incre_sampling_segs_end[start_tmp_]<=max_n_winsize):
            start_tmp_ = start_tmp_+1
        incre_sampling_segs_end = incre_sampling_segs_end[start_tmp_:]
        assert incre_sampling_segs_end.shape[0]>1
        assert incre_sampling_segs_end[0]>max_n_winsize
        events_per_seg = incre_sampling_segs_end[1]-incre_sampling_segs_end[0]
        incre_sampling_seg_num = incre_sampling_segs_end.shape[0]
        print(f"*******************************  BA events_num: {event_total_num}  *********************************")
        print(f"incremental sampling number: {incre_sampling_seg_num}")
        print(f"****events_per_seg={events_per_seg}, min_n_winsize={min_n_winsize},max_n_winsize={max_n_winsize}, ")
        
        
        if frame_id_all[-1] >= self.config["start_cspline_idx"] and self.config["traj_mode_BA"]=="cspline":
            traj_mode = "cspline"
        else:
            traj_mode = "linear"
        
        # initialize trajectories
        active_ctrl_knot_ts = []
        active_ctrl_knot_idx = frame_id_all.copy()
        active_ctrl_knot_idx.insert(0, frame_id_all[0]-1)
        
        if traj_mode == "cspline":
            active_ctrl_knot_se3 = pp.identity_se3(len(active_ctrl_knot_idx)+1) #LieTensor
            for ii in range(len(active_ctrl_knot_idx)):
                idx = active_ctrl_knot_idx[ii]
                active_ctrl_knot_ts.append(self.ctrl_knot_ts_all[idx])
                active_ctrl_knot_se3[ii] = self.ctrl_knot_se3_all[idx]
            # add another control knot
            print("============================== use cspline interplation ==================================")
            active_ctrl_knot_idx.append(active_ctrl_knot_idx[-1]+1)
            active_ctrl_knot_ts.append(active_ctrl_knot_ts[-1] + active_ctrl_knot_ts[-1]-active_ctrl_knot_ts[-2])
            # predict pose
            pred_vel = self.predict_velocity(se3_to_SE3(active_ctrl_knot_se3[-2]), se3_to_SE3(active_ctrl_knot_se3[-3]))
            active_ctrl_knot_se3[-1] = self.predict_pose_se3_lie(active_ctrl_knot_se3[-2], active_ctrl_knot_se3[-3])
            # don't sample the first chunk
            active_frame_id_all = frame_id_all.copy()
            active_frame_ts_all = frame_ts_all.copy()
            del active_frame_id_all[0]
            del active_frame_ts_all[0]
            del preSum_event_batch[0]
            # _num_iters += 500
        else:
            active_frame_id_all = frame_id_all.copy()
            active_frame_ts_all = frame_ts_all.copy()
            active_ctrl_knot_se3 = pp.identity_se3(len(active_ctrl_knot_idx)) #LieTensor
            for ii in range(len(active_ctrl_knot_idx)):
                idx = active_ctrl_knot_idx[ii]
                # TODO: ALL the idx is from 1
                active_ctrl_knot_ts.append(self.ctrl_knot_ts_all[idx])
                active_ctrl_knot_se3[ii] = self.ctrl_knot_se3_all[idx]
        
        active_ctrl_knot_ts[0] = t_data_min-0.0001
        active_ctrl_knot_ts[-1] = t_data_max+0.0001
        
        print(f"****t_data_min={t_data_min}, t_data_max={t_data_max}")
        print(f"control knot ts: {active_ctrl_knot_ts}")
        
        if self.config["use_relative_pose_to_opt"]==1:
            active_ctrl_knot_se3_rel_opt = pp.identity_se3(active_ctrl_knot_se3.shape[0]) #LieTensor
            active_ctrl_knot_se3_base = active_ctrl_knot_se3.clone()
            active_ctrl_knot_se3_rel_opt = torch.nn.Parameter(active_ctrl_knot_se3_rel_opt, requires_grad=True)        
            pose_optimizer = torch.optim.Adam([{"params": active_ctrl_knot_se3_rel_opt, "lr": self.config['pose_lr']}])
        elif self.config["use_relative_pose_to_opt"]==0:
            active_ctrl_knot_se3 = torch.nn.Parameter(active_ctrl_knot_se3, requires_grad=True)
            pose_optimizer = torch.optim.Adam([{"params": active_ctrl_knot_se3, "lr": self.config['pose_lr']}])
        elif self.config["use_relative_pose_to_opt"]==2:  # fix the first pose
            base_idx = active_ctrl_knot_idx[0]-1
            if base_idx<0:
                base_idx = 0
                print("====================== Warning: Don't fix the first pose ======================")
            ctrl_knot_len = len(active_ctrl_knot_idx)
            ctrl_knot_se3_base = self.ctrl_knot_se3_all[base_idx] 
            ctrl_knot_se3_rel_opt = pp.identity_se3(ctrl_knot_len)
            for ii in range(ctrl_knot_len):
                ctrl_knot_se3_rel_opt[ii] = (self.ctrl_knot_se3_all[active_ctrl_knot_idx[ii]].Exp()*(ctrl_knot_se3_base.Inv().Exp())).Log()
            ctrl_knot_se3_rel_opt = torch.nn.Parameter(ctrl_knot_se3_rel_opt, requires_grad=True)
            pose_optimizer = torch.optim.Adam([{"params": ctrl_knot_se3_rel_opt, "lr": self.config['pose_lr']}])
        else:
            raise ValueError("wrong option for pose optimization")
        
        active_num_batch = len(active_frame_id_all)
        
        blur_sigma = self.config["blur_sigma"]
        batch_cnt = 0
        loss_total = None
        pose_optimizer.zero_grad()
        
        # densification setting
        # self.gs_cfg.refine_start_iter = self.config["mapping"]["refine_start_iter"]
        # self.gs_cfg.refine_stop_iter= self.config["mapping"]["refine_stop_iter"]
        # self.gs_cfg.refine_every = self.config["mapping"]["refine_every"]
        
        #=====================================Need To change======================================
        # Densification Strategy
        # strategy_BA = DefaultStrategy(
        #     verbose=True,
        #     scene_scale=self.scene_scale,
        #     prune_opa=self.gs_cfg.prune_opa,
        #     grow_grad2d=self.gs_cfg.grow_grad2d,
        #     grow_scale3d=self.gs_cfg.grow_scale3d,
        #     prune_scale3d=self.gs_cfg.prune_scale3d,
        #     # refine_scale2d_stop_iter=4000, # splatfacto behavior
        #     refine_start_iter=self.gs_cfg.refine_start_iter,
        #     refine_stop_iter=self.gs_cfg.refine_stop_iter,
        #     refine_every=self.gs_cfg.refine_every,
            
        #     reset_every=self.gs_cfg.reset_every,
        #     absgrad=self.gs_cfg.absgrad,
        #     revised_opacity=self.gs_cfg.revised_opacity,
        # )
        # strategy_BA.check_sanity(self.splats, self.optimizers)
        # strategy_state_BA = strategy_BA.initialize_state()
        #================================================================================================
        
        training_batch_size = self.config["BA"]["training_batch_size"]
        visualize_every_iter = self.config["BA"]["visualize_every_iter"]
        if _global_BA:
            save_path = os.path.join(self.config["data"]["output"], self.config["data"]["exp_name"], "global_BA")
        else:
            save_path = os.path.join(self.config["data"]["output"], self.config["data"]["exp_name"], "BA")
        

        #==========================================
        # Triangle specific parameters
        need_delaunay = False
        run_restricted_delaunay = self.opt.densify_until_iter + 1000


        # define the scheduler for sigma and opacity
        initial_sigma = self.opt.set_sigma
        final_sigma = 0.0001
        sigma_start = self.opt.sigma_start
        total_iters = self.opt.sigma_until

        init_opacity = 0.1
        final_opacity = .9999
        total_iters_opacity = self.opt.final_opacity_iter

        lambda_weight = self.opt.lambda_weight
        prune_triangles = self.opt.prune_triangles_threshold
        prune_size = self.opt.prune_size
        start_upsampling = self.opt.start_upsampling
        splitt_large_triangles = self.opt.splitt_large_triangles
        #==========================================


        for i in range(_num_iters):
            if i%100 == 0:
                blur_sigma = blur_sigma/2

            #===============================================================
            if need_delaunay:
                with torch.no_grad():
                    self.triangles.run_restricted_delaunay()
                need_delaunay = False

            # Supersampling
            if iter == start_upsampling:
                self.triangles.scaling = self.opt.upscaling_factor
            if iter == start_upsampling + 5000:
                self.triangles.scaling = 4

            self.triangles.update_learning_rate(_num_iters)

            # Sigma schedule
            if _num_iters < sigma_start:
                current_sigma = initial_sigma
            else:
                progress = (_num_iters - sigma_start) / (total_iters - sigma_start)
                progress = min(progress, 1.0)
                current_sigma = initial_sigma - (initial_sigma - final_sigma) * progress
            self.triangles.set_sigma(current_sigma)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if _num_iters % 1000 == 0:
                self.triangles.oneupSHdegree()

            #===================================================================
                
            
            batch_cnt += 1
            if self.config["use_relative_pose_to_opt"]==1:
                active_ctrl_knot_se3 = (active_ctrl_knot_se3_base.Exp()*active_ctrl_knot_se3_rel_opt.Exp()).Log()
            elif self.config["use_relative_pose_to_opt"]==2:
                active_ctrl_knot_se3 = (ctrl_knot_se3_rel_opt.Exp()*ctrl_knot_se3_base.Exp()).Log() 
            
            loss_event = torch.zeros([1], dtype=torch.float64, device=self.device, requires_grad=True)
            loss_no_event = torch.zeros([1], dtype=torch.float64, device=self.device, requires_grad=True)
            loss_ssim = torch.zeros([1], dtype=torch.float64, device=self.device, requires_grad=True)
            loss_white_balance = torch.zeros([1], dtype=torch.float64, device=self.device, requires_grad=True)
            loss_tr = torch.zeros([1], dtype=torch.float64, device=self.device, requires_grad=True)
            
            # incremental random sampling
            random_start_end_idx = []
            for ii in range(incre_sampling_seg_num):
                end_idx_ = incre_sampling_segs_end[ii]
                winsize_ = np.random.randint(min_n_winsize, max_n_winsize)
                start_idx_ = end_idx_-winsize_
                random_start_end_idx.append([start_idx_, end_idx_])
            indices = np.random.permutation(len(random_start_end_idx)).tolist()[:training_batch_size]
            
            list_img_ev_start = []
            list_img_ev_end = []
            list_gt_events_acc = []
            list_syn_event_acc = []
            
            linlog_thres = self.config["event"]["linlog_thres"]
            # incremental random sampling
            for j in indices:
                idx_ev_start = random_start_end_idx[j][0]
                idx_ev_end = random_start_end_idx[j][1]
                selected_event_stream = events_all[idx_ev_start:idx_ev_end]
                t_ev_start = ts_all[idx_ev_start]
                t_ev_end = ts_all[idx_ev_end]
                
                T_SE3_ev_start = self.get_poses_lie(active_ctrl_knot_se3, active_ctrl_knot_ts, t_ev_start, mode=traj_mode)
                T_SE3_ev_end = self.get_poses_lie(active_ctrl_knot_se3, active_ctrl_knot_ts, t_ev_end, mode=traj_mode)
                
                c2w_start = T_SE3_ev_start.matrix().unsqueeze(0).cuda()  #[1, 4, 4]
                render_pkg_start = self.render_triangles(c2w_start)
                c2w_end = T_SE3_ev_end.matrix().unsqueeze(0).cuda()  #[1, 4, 4]
                render_pkg_end = self.render_triangles(c2w_end)
                
                img_ev_start = render_pkg_start["render"]
                img_ev_end = render_pkg_end["render"]
                
                #=================================Need To Change=========================================
                # strategy_BA.step_pre_backward(
                #     params=self.splats,
                #     optimizers=self.optimizers,
                #     state=strategy_state_BA,
                #     step=i,
                #     info=render_pkg_end["info"],
                # )
                #================================================================================================
                
                if self.config["use_linLog"]:
                    pred_linlog_start = lin_log(img_ev_start*255, linlog_thres=linlog_thres) # (B, Nevs, 1)
                    pred_linlog_end = lin_log(img_ev_end*255, linlog_thres=linlog_thres) # (B, Nevs, 1)
                    syn_event_acc = pred_linlog_end - pred_linlog_start
                else:
                    # syn_event_acc = (log(img_ev_end) - log(img_ev_start))
                    eps = 0.1
                    syn_event_acc = torch.log(img_ev_end**2.2+eps)-torch.log(img_ev_start**2.2+eps)
                
                # compute event loss
                gt_events_acc = self.post_process_event_image(selected_event_stream, _sigma=blur_sigma)
                # if self.config["use_mask_event_loss"]:
                event_mask = (gt_events_acc.abs() != 0).float() 
                no_event_mask = (gt_events_acc.abs() == 0).float()
                
                # if self.config["dataset"] == "vector":
                #     loss_event = loss_event + (self.vector_mask * (gt_events_acc - syn_event_acc)**2).sum() / self.vector_mask.sum()
                # else:
                
                # no-event loss
                no_event_gaussian_cov = self.config["mapping"]["no_event_gaussian_cov"]
                gt_no_events = self.config["event"]["threshold"] * no_event_gaussian_cov * torch.randn_like(gt_events_acc).cuda()
                gt_no_events = gt_no_events*no_event_mask
                
                if self.config["mapping"]["color_channels"]==3:
                    gt_events_acc = gt_events_acc.unsqueeze(-1).repeat(1, 1, 3)
                    gt_no_events = gt_no_events.unsqueeze(-1).repeat(1, 1, 3)
                    # gt_events_acc = np.tile(gt_events_acc[..., None], (1, 1, 3))
                    gt_events_acc = gt_events_acc*self.color_mask
                    syn_event_acc = syn_event_acc*self.color_mask
                
                if self.config["seprate_event_noevent_loss"]:
                    loss_event = loss_event + (event_mask*(gt_events_acc - syn_event_acc)**2).sum() / event_mask.sum()
                    loss_no_event = loss_no_event + (no_event_mask*(gt_no_events - syn_event_acc)**2).sum() / no_event_mask.sum()
                else:
                    gt_events_acc = gt_events_acc+gt_no_events
                    loss_event = loss_event + ((gt_events_acc - syn_event_acc)**2).mean()
                
                if self.config["mapping"]["color_channels"]==3:
                    loss_ssim = loss_ssim + compute_ssim_loss(gt_events_acc, syn_event_acc, channel=3)
                else:
                    loss_ssim = loss_ssim + compute_ssim_loss(gt_events_acc, syn_event_acc, channel=1)
                
                list_img_ev_start.append(img_ev_start.detach())
                list_img_ev_end.append(img_ev_end.detach())
                list_gt_events_acc.append(gt_events_acc)
                list_syn_event_acc.append(syn_event_acc.detach())
            
            if self.config["mapping"]["use_white_balance_loss"]:
                # white balance loss 
                white_balance_weight = self.config["mapping"]["white_balance_weight"]
                # loss_white_balance = white_balance_weight * (img_ev_end.mean() - 0.5) ** 2
                loss_white_balance = white_balance_weight*torch.mean((img_ev_end - 0.5) ** 2)
            
            #========== TODO : Change code=================
            # tikhonov regularization loss
            tr_w = self.config["mapping"]["tr_loss_weight"]
            loss_tr = tr_w*tikhonov_regularization(render_pkg_start["expected_depth"].squeeze().unsqueeze(-1))
            
            loss_event = loss_event/len(indices)
            loss_ssim = loss_ssim/len(indices)
            loss_no_event = loss_no_event/len(indices)
            # summary the loss
            factor_ = self.config["mapping"]["ssim_loss_factor_"]
            loss_event = (1-factor_)*loss_event
            loss_ssim = factor_*loss_ssim
            
            # # isotropic loss
            # scaling = self.splats["scales"]  # [N, 3]
            # isotropic_loss_all = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            # iso_w = self.config["mapping"]["loss_isotropic_weight"]
            # loss_isotropic = iso_w * isotropic_loss_all.mean()
            
            # loss_total = loss_event + loss_ssim + loss_isotropic+loss_white_balance
            loss_total = loss_event + loss_ssim + loss_white_balance + loss_no_event + loss_tr
            
            loss_total.backward()

            #======================================================
            # Mesh Splatting Densification
            image_size = render_pkg_start["scaling"].detach()
            mask = image_size > self.triangles.image_size
            self.triangles.image_size[mask] = image_size[mask]

            importance_score = render_pkg_start["max_blending"].detach()
            mask = importance_score > self.triangles.importance_score
            self.triangles.importance_score[mask] = importance_score[mask]

            pixel_count = render_pkg_start["triangle_was_rendered"].detach() # Not used but could be useful. Gives per triangle, the number of pixels it covered in the current render
            mask = pixel_count > self.triangles.pixel_count
            self.triangles.pixel_count[mask] = pixel_count[mask]

            with torch.no_grad():
                # Handle pruning operations
                if _num_iters % 500 == 0 and _num_iters < run_restricted_delaunay:
                    
                    # Building masks to delete triangles
                    triangle_vertex_weights = self.triangles.opacity_activation(
                        self.triangles.vertex_weight[self.triangles._triangle_indices]
                    ) 
                    min_weights = triangle_vertex_weights.min(dim=1).values

                    mask_opacity     = (min_weights <= prune_triangles).squeeze()              # delete if too low
                    mask_importance  = (self.triangles.importance_score <= prune_triangles).squeeze()  # delete if too low
                    mask_size        = (self.triangles.image_size > prune_size).squeeze()                 # delete if too big

                    delete_mask = mask_opacity | mask_size

                    # if number_of_training_views < 500: # only delete if the number of views are below 500. Otherwise, we might delete too much
                    delete_mask = delete_mask | mask_importance

                    keep_mask   = ~delete_mask 

                    if _num_iters > self.opt.start_pruning:
                        self.triangles.prune_triangles(keep_mask)
                
                    # We prune vertices that are no longer used
                    device = self.triangles.vertices.device
                    used_vertex_mask = torch.zeros(self.triangles.vertices.shape[0], 
                                                dtype=torch.bool, 
                                                device=device)
                    if self.triangles._triangle_indices.numel() > 0:
                        flat_indices = self.triangles._triangle_indices.flatten()
                        used_vertex_mask[flat_indices] = True
                    
                    weight_mask = (self.triangles.get_vertex_weight.squeeze() >= prune_triangles)
                    mask_out = self.triangles.vertices.shape[0]
                    vertex_mask = weight_mask[:mask_out] | used_vertex_mask

                    self.triangles._prune_vertices(vertex_mask)


                    triangle_vertex_weights = self.triangles.opacity_activation(
                        self.triangles.vertex_weight[self.triangles._triangle_indices]
                    )  # [T,3]

                    needs_densification = (_num_iters < self.opt.densify_until_iter and 
                                        _num_iters % self.opt.densification_interval == 0 and 
                                        _num_iters > self.opt.densify_from_iter)
                    
                    if needs_densification:
                        self.triangles.add_new_gs_original(cap_max=self.opt.max_points, splitt_large_triangles=splitt_large_triangles)
    

                    if _num_iters > self.opt.start_opacity_floor:
                        start_iter = self.opt.start_opacity_floor
                        end_iter = total_iters_opacity  # the iteration where you want to reach final_opacity
                        a = min(1.0, max(0.0, (_num_iters - start_iter) / max(1, end_iter - start_iter)))
                        current_opacity = init_opacity + (final_opacity - init_opacity) * a
                        current_opacity = min(current_opacity, final_opacity)
                        self.triangles.update_min_weight(current_opacity)

                        prune_triangles += 0.01 
                        mask_out = self.triangles.vertices.shape[0]
                        triangle_vertex_weights = self.triangles.get_vertex_weight[:mask_out][self.triangles._triangle_indices]
                elif _num_iters == run_restricted_delaunay:
                    need_delaunay = True
                elif _num_iters % 500 == 0 and _num_iters > run_restricted_delaunay + 1000:

                    if _num_iters > self.opt.start_opacity_floor:
                        start_iter = self.opt.start_opacity_floor
                        end_iter = total_iters_opacity  # the iteration where you want to reach final_opacity
                        a = min(1.0, max(0.0, (_num_iters - start_iter) / max(1, end_iter - start_iter)))
                        current_opacity = init_opacity + (final_opacity - init_opacity) * a
                        current_opacity = min(current_opacity, final_opacity)
                        self.triangles.update_min_weight(current_opacity)

                        prune_triangles += 0.01 
                        mask_out = self.triangles.vertices.shape[0]
                        triangle_vertex_weights = self.triangles.get_vertex_weight[:mask_out][self.triangles._triangle_indices]
                

                if _num_iters < self.opt.iterations:
                    self.triangles.optimizer.step()
                    self.triangles.optimizer.zero_grad(set_to_none = True)

            #======================================================
            
            if _opt_pose:
                pose_optimizer.step()
                pose_optimizer.zero_grad()
                

            # loss_total = None
            #========== TODO : Need To Change the Code =============================
            # for optimizer in self.optimizers.values():
            #     optimizer.step()
            #     optimizer.zero_grad(set_to_none=True)
            
            # densification
            # strategy_BA.step_post_backward(
            #     params=self.splats,
            #     optimizers=self.optimizers,
            #     state=strategy_state_BA,
            #     step=i,
            #     info=render_pkg_end["info"],
            # )
            #===============================================================================

            # visualization
            frame_id_vis = frame_id_all[-1]
            if i % visualize_every_iter == 0 and self.config["visualize_inter_img"]:
                print_str = f"iter {i}, loss={loss_total.item()}, event={loss_event.item()}, ssim={loss_ssim.item()}, white={loss_white_balance.item()}, noevent={loss_no_event.item()}, tr={loss_tr.item()}"
                print(print_str)
                
                with torch.no_grad():
                    # plot pose estimation
                    accum_est_poses_Rt = []
                    accum_gt_poses_Rt = []
                    
                    tag = f"chunk_BA_{num_batch:03}_{i:04}"
                    
                    # save ply file
                    # ply_path = os.path.join(save_path, f"pointCloud_f{frame_id_vis}_it{i}.ply")
                    # self.save_ply(ply_path)
                    
                    for n in range(active_num_batch):
                        k = active_frame_id_all[n]
                        ts = active_frame_ts_all[n]                        
                        accum_est_poses_Rt.append(self.get_poses_lie(active_ctrl_knot_se3, active_ctrl_knot_ts, ts, mode=traj_mode).matrix())
                        if traj_mode == "linear":
                            accum_gt_poses_Rt.append(BA_batch[n]['c2w'].squeeze())
                        else:
                            accum_gt_poses_Rt.append(BA_batch[n+1]['c2w'].squeeze())
                    
                    # plot 
                    if self.config["dataset"] == "rpg_evo":
                        accum_gt_poses_Rt = accum_est_poses_Rt
                    pose_evaluation(accum_gt_poses_Rt, accum_est_poses_Rt, 1, save_path, "pose", f"BA_f{frame_id_vis:03}_{i:04}")
                    save_pose_as_kitti_evo(accum_gt_poses_Rt, accum_est_poses_Rt, save_path, f"BA_f{frame_id_vis:03}_{i:04}")
                    
                    if self.config["visualize_inter_img"]:
                        # save intermediate images 
                        images_combined = []
                        for m in range(len(list_img_ev_start)):
                            diff = (list_gt_events_acc[m] - list_syn_event_acc[m]) **2
                            
                            if self.config["mapping"]["color_channels"] == 1:
                                gt_events_acc = render_ev_accumulation(list_gt_events_acc[m].cpu().numpy(), self.dataset.H, self.dataset.W)
                                diff = np.repeat(diff.detach().abs().cpu().numpy()[:, :, np.newaxis], repeats=3, axis=2)  
                                syn_event_acc = np.repeat(list_syn_event_acc[m].detach().abs().cpu().numpy()[:, :, np.newaxis], repeats=3, axis=2) 
                                img_ev_start = np.repeat(list_img_ev_start[m].detach().cpu().numpy()[:, :, np.newaxis], repeats=3, axis=2)
                                img_ev_end = np.repeat(list_img_ev_end[m].detach().cpu().numpy()[:, :, np.newaxis], repeats=3, axis=2)
                            else:
                                gt_events_acc = render_ev_accumulation(list_gt_events_acc[m].sum(-1).cpu().numpy(), self.dataset.H, self.dataset.W)
                                img_ev_start = list_img_ev_start[m].detach().cpu().numpy()
                                img_ev_end = list_img_ev_end[m].detach().cpu().numpy()
                                syn_event_acc = list_syn_event_acc[m].detach().abs().cpu().numpy()
                                diff = diff.detach().cpu().numpy()
                            
                            image_combined = np.concatenate([img_ev_start,img_ev_end, gt_events_acc, syn_event_acc, diff], axis=1)
                            images_combined.append(image_combined)
                        
                        images_combined = np.concatenate(images_combined, axis=0)
                        images_combined =  to8b(images_combined)
                        img_path = os.path.join(save_path, f"BA_f{frame_id_vis:03}_{i:04}" + "_img.jpg")  
                        imageio.imwrite(img_path, images_combined)
        
        if self.config["use_relative_pose_to_opt"]==1:
            active_ctrl_knot_se3 = (active_ctrl_knot_se3_base.Exp()*active_ctrl_knot_se3_rel_opt.Exp()).Log()
        elif self.config["use_relative_pose_to_opt"]==2:
            active_ctrl_knot_se3 = (ctrl_knot_se3_rel_opt.Exp()*ctrl_knot_se3_base.Exp()).Log()
        
        # update est_poses
        for n in range(active_num_batch):
            k = active_frame_id_all[n]
            ts = active_frame_ts_all[n]                      
            self.est_c2w_data[k] = self.get_poses_lie(active_ctrl_knot_se3, active_ctrl_knot_ts, ts, mode=traj_mode).matrix().cuda()
            self.est_c2w_ts[k] = ts
        
        if self.config["visualize_inter_img"] or _global_BA:
            if self.config["render_eventview_img"]:
                # render ALL images 
                with torch.no_grad():
                    for n in range(active_num_batch):
                        k = active_frame_id_all[n]
                        ts = active_frame_ts_all[n]                      
                        T_SE3 = self.get_poses_lie(active_ctrl_knot_se3, active_ctrl_knot_ts, ts, mode=traj_mode)
                        
                        c2w_ = T_SE3.matrix().unsqueeze(0).cuda()  #[1, 4, 4]
                        rendered_img = self.render_triangles(c2w_)["render"]
                                
                                
                        event_img_ = rendered_img.detach().cpu().numpy()
                        event_img_ =  to8b(event_img_)
                        im_name = f"BA_f{frame_id_vis:03}_f{k}" + "_img.jpg"
                        imageio.imwrite(os.path.join(save_path, im_name), event_img_)
                        
                        # ======= TODO : Need To Change ===============================
                        if self.config["render_tumvie_rgbCam_img"] and self.config["dataset"]=="tum_vie":
                            # render rbg camera image
                            img_rgbCam = self.render_with_gsplat(self.gs_model, T_se3, render_tumvie_rgb=True)['image']
                            img_rgbCam = np.repeat(img_rgbCam.detach().cpu().numpy()[:, :, np.newaxis], repeats=3, axis=2)
                            img_path = os.path.join(save_path, f"BA_f{frame_id_vis:03}_f{k}" + "_img_rgbCam.jpg")  
                            imageio.imwrite(img_path, to8b(img_rgbCam))
                        # ======================================================================    
            # plot the whole trajectory
            if self.config["dataset"] == "rpg_evo":
                self.pose_gt = self.est_c2w_data
            pose_evaluation(self.pose_gt, self.est_c2w_data, 1, save_path, "pose", f"BA_f{frame_id_vis:03}_whole")
            save_pose_as_kitti_evo(self.pose_gt, self.est_c2w_data, save_path, f"BA_f{frame_id_vis:03}_whole")
        
        
        # save control knot pose
        for ii in range(len(active_ctrl_knot_idx)):
            idx = active_ctrl_knot_idx[ii]
            self.ctrl_knot_se3_all[idx] = active_ctrl_knot_se3[ii].detach().clone()  # [], cuda
        
        print("======================= finished BA  =========================")
            
           

    # -------------------------------------------------------------------------
    # Main train loop: backend (mapping) then frontend (tracking) per frame
    # -------------------------------------------------------------------------

    def train(self):

        init_batch = []
        BA_batch = []
        tracking_batch = []
        BA_count_cnt = 0
        creat_new_gs_cnt = 0
        
        num_optim_steps_tracking = self.config["num_opti_steps_for_tracking"]
        num_optim_steps_BA = self.config["num_opti_steps_for_BA"]
        num_opti_steps_for_global_BA = self.config["num_opti_steps_for_global_BA"]
        num_opti_steps_for_final_global_BA = self.config["num_opti_steps_for_final_global_BA"]
        refine_img_rendering_ = self.config["refine_img_rendering_"]
        
        # variable for initialization
        bInitIsDone = False

        # start optimization 
        data_loader = DataLoader(self.dataset, num_workers=self.config['data']['num_workers'])

        for i, batch in tqdm(enumerate(data_loader)):
            cur_frame_id = batch["frame_id"].item()
            
            BA_batch.append(batch)
            if(len(BA_batch) > 100):
                BA_batch.pop(0)
            
            if bInitIsDone==False:
                init_batch.append(batch)
                
                if len(init_batch) < self.config["initialization"]['num_frames_for_init']:
                    continue
                
                # estimate the depth_map
                depth_estimation_init_flag = self.config["depth_estimation_init"]
                if depth_estimation_init_flag==0:
                    print("************************ Don't use depth initialization ************************")
                    depth_map = None
                elif depth_estimation_init_flag==1:
                    print("************************ Estimate depth during initialization ************************")
                    
                    # depth estimation
                    from PIL import Image
                    from diffusers import DiffusionPipeline
                    from diffusers.utils import load_image
                    
                    self.initialization(init_batch, niters=self.config['num_opti_steps_for_depthEst_init'], traj_mode='linear', depth_map=None)
                    
                    depthEst_pretrained_model_path = self.config["depthEst_pretrained_model_path"]
                    pipe = DiffusionPipeline.from_pretrained(
                                depthEst_pretrained_model_path,
                                custom_pipeline="marigold_depth_estimation",
                                # torch_dtype=torch.float16,                # (optional) Run with half-precision (16-bit float).
                                local_files_only=True
                            )
                    pipe.to("cuda")
                    
                    bDepthIsEst = True
                    
                    init_path = os.path.join(self.config["data"]["output"], self.config["data"]["exp_name"], 'initialization')
                    save_dir_path = os.path.join(self.config["data"]["output"], self.config["data"]["exp_name"])
                    img_path = os.path.join(init_path, f"frame{cur_frame_id}_init.jpg")
                    image: Image.Image = load_image(img_path)
                    pipeline_output = pipe(
                                            image,                  # Input image.
                                            # denoising_steps=30,     # (optional) Number of denoising steps of each inference pass. Default: 10.
                                            # ensemble_size=10,       # (optional) Number of inference passes in the ensemble. Default: 10.
                                            # processing_res=768,     # (optional) Maximum resolution of processing. If set to 0: will not resize at all. Defaults to 768.
                                            # match_input_res=True,   # (optional) Resize depth prediction to match input resolution.
                                            # batch_size=10,           # (optional) Inference batch size, no bigger than `num_ensemble`. If set to 0, the script will automatically decide the proper batch size. Defaults to 0.
                                            # color_map="Spectral",   # (optional) Colormap used to colorize the depth map. Defaults to "Spectral". Set to `None` to skip colormap generation.
                                            # show_progress_bar=True, # (optional) If true, will show progress bars of the inference progress.
                                        )
                    depth: np.ndarray = pipeline_output.depth_np                    # Predicted depth map
                    depth_colored: Image.Image = pipeline_output.depth_colored      # Colorized prediction
                    shutil.copy(img_path, os.path.join(save_dir_path, f"init_{cur_frame_id}_rendered_img.jpg"))
                    # Save as uint16 PNG
                    depth_uint16 = (depth * 65535.0).astype(np.uint16)
                    Image.fromarray(depth_uint16).save(os.path.join(save_dir_path, "./depth_map.png"), mode="I;16")
                    # save numpy data
                    np.save(os.path.join(save_dir_path, "./depth_data.npy"), depth)
                    # Save colorized depth map
                    depth_colored.save(os.path.join(save_dir_path, "./depth_colored.png"))
                    
                    # release the GPU memory
                    del pipe
                    torch.cuda.empty_cache()
                    
                    depth_path = os.path.join(save_dir_path, "depth_data.npy")
                    print('[initialization]: load depth map from', depth_path)
                    depth_map = torch.from_numpy(np.load(depth_path)).to(self.device)
                    
                    
                elif depth_estimation_init_flag==2:
                    print("************************ Estimate depth during initialization ************************")
                    depth_path = self.config["initialization"]["depth_path"]
                    print('[initialization]: load depth map from', depth_path)
                    depth_map = torch.from_numpy(np.load(depth_path)).to(self.device)
                
                # initialize...
                # TODO : Check for the without depth initialization
                self.initialization(init_batch, niters=self.config['num_opti_steps_for_init'], traj_mode='linear', depth_map=depth_map)
                
                bInitIsDone = True
            else:
                if self.config["initOnly"]:
                    # exit(0)
                    break
                print("Process frame", i+1)
                
                tracking_batch.append(batch)
                
                if len(tracking_batch) >= self.config["tracking_batch_size"]:
                    BA_count_cnt = BA_count_cnt+1
                    # tracking 
                    self.tracking(cur_frame_id, tracking_batch, num_optim_steps_tracking)
                    tracking_batch = []
                
                if BA_count_cnt >= self.config["BA_every_track"]:
                    BA_count_cnt = 0
                    creat_new_gs_cnt = creat_new_gs_cnt+1
                    if creat_new_gs_cnt >= self.config["add_new_gs_step"] and self.config["gassian_incre_growing"]:
                        events_stream = BA_batch[-1]['events'].squeeze().to(self.device)
                        new_pcd = self.create_densify_points(cur_frame_id,
                                                            events_stream,
                                                            SE3_to_se3(self.est_c2w_data[cur_frame_id-1][:3,:4]).cuda(), 
                                                            SE3_to_se3(self.est_c2w_data[cur_frame_id][:3,:4]).cuda())
                        # self.gs_model.add_new_points(new_pcd, 8.25)
                        # feature_dim = 32 if self.gs_cfg.app_opt else None
                        feature_dim = None
                        print(f"============================= point num:{new_pcd.points.shape[0]} ===============================")
                        if(new_pcd.points.shape[0]>50):
                            # TODO : Change the code
                            new_triangles = TriangleModel(sh_degree = 0, config=self.config)
                            new_triangles.create_from_pcd(new_pcd, radii=None, opacity=self.opt.set_weight, set_sigma=self.opt.set_sigma)
                            
                            new_triangles.training_setup(
                                self.opt,
                                self.opt.feature_lr,
                                self.opt.weight_lr,
                                self.opt.lr_triangles_points_init,
                            )

                            self.triangles.densification_postfix(new_triangles.vertices, new_triangles.vertex_weight, new_triangles._features_dc, new_triangles._features_rest, new_triangles._triangle_indices)
                            
                            # new_gs = pcd_2_gs(
                            #     points= new_pcd.points.detach(),
                            #     init_opacity=self.gs_cfg.init_opa,
                            #     init_scale=self.gs_cfg.init_scale,
                            #     scene_scale=self.scene_scale,
                            #     sh_degree=self.gs_cfg.sh_degree,
                            #     feature_dim=feature_dim,
                            #     device=self.device,
                            #     )
                            # add_new_gs(self.splats, self.optimizers, new_gs)
                            # print("Number of New GS:", len(new_gs["means"]))
                            # print("Number of GS:", len(self.splats["means"]))
                            print("********** Number of Vertices :", len(self.triangles.vertices))
                            print("********** Number of Trianges :", len(self.triangles._triangles))
                        else:
                            print("Not enough points to create a new GS, skipped.")
                    
                    sliding_window_sz = self.config["sliding_window_sz"]
                    traj_mode = 'linear'
                    seg_num_ = self.config["BA"]["incre_sampling_seg_num_expected"]
                    self.bundle_adjustment(BA_batch[-sliding_window_sz:], traj_mode, num_optim_steps_BA, seg_num_)
                
                if (i+1)>=20 and (i+1)%self.config["BA"]["global_BA_step"]==0:
                    print("******************************** global BA ********************************")
                    traj_mode = 'linear'
                    global_BA_seg = self.config["BA"]["global_BA_seg"]
                    seg_num_ = len(BA_batch)*global_BA_seg
                    if seg_num_<50:
                        seg_num_=50
                    elif seg_num_>1000:
                        seg_num_=1000
                    print(f"\n\n ------------------ num_opti_steps_for_global_BA = {num_opti_steps_for_global_BA}---------------------")
                    self.bundle_adjustment(BA_batch, traj_mode, num_opti_steps_for_global_BA, seg_num_, _global_BA=True)
                    if refine_img_rendering_:
                        self.bundle_adjustment(BA_batch, traj_mode, num_opti_steps_for_global_BA, seg_num_, _opt_pose=False,_global_BA=True)
                
        # global BA
        if self.config["global_BA"] and (not self.config["initOnly"]):
            print("******************************** global BA ********************************")
            traj_mode = 'linear'
            seg_num_ = len(BA_batch)*5
            if seg_num_<50:
                seg_num_=50
            elif seg_num_>1000:
                seg_num_=1000
            self.bundle_adjustment(BA_batch, traj_mode, num_opti_steps_for_final_global_BA, seg_num_, _global_BA=True)
            if refine_img_rendering_:
                self.bundle_adjustment(BA_batch, traj_mode, num_opti_steps_for_final_global_BA, seg_num_, _opt_pose=False, _global_BA=True)
        
        if self.config["evaluate_img"]:
            # render ALL images 
            with torch.no_grad():
                if self.config["use_gt_pose_to_opt"]:
                    print("******* use gt pose to render images *******")
                    for tmp_idx_ in self.dataset.val_img_idx:
                        c2w_ = self.dataset.original_gt_poses[tmp_idx_].reshape(-1, 4, 4).cuda()
                        render_pkg_ = self.rasterize_splats(camtoworlds=c2w_, render_mode="RGB")
                        event_img_ = render_pkg_["image"].detach().cpu().numpy()
                        
                        event_img_ =  to8b(event_img_)
                        im_name = f"f{tmp_idx_}_{float(self.dataset.original_gt_poses_ts[tmp_idx_]):07.3f}s.jpg"

                        save_path = os.path.join(self.config["data"]["output"], self.config["data"]["exp_name"], "img_eval/est")
                        imageio.imwrite(os.path.join(save_path, im_name), event_img_)
                else:
                    print("******* use interpolated pose to render images *******")
                    total_frame_num = len(self.dataset.val_img_idx)
                    
                    cntl_knot_sorted_keys = sorted(list(self.ctrl_knot_ts_all.keys()))
                    cntl_knot_num = len(cntl_knot_sorted_keys)
                    active_ctrl_knot_ts = []
                    active_ctrl_knot_se3 = pp.identity_se3(cntl_knot_num) #LieTensor
                    for ii in range(cntl_knot_num):
                        active_ctrl_knot_ts.append(self.ctrl_knot_ts_all[cntl_knot_sorted_keys[ii]])
                        active_ctrl_knot_se3[ii] = self.ctrl_knot_se3_all[cntl_knot_sorted_keys[ii]]
                    
                    for ii in range(total_frame_num):
                        idx_ = self.dataset.val_img_idx[ii]
                        ts = self.dataset.val_img_ts[ii]
                        T_SE3 = self.get_poses_lie(active_ctrl_knot_se3, active_ctrl_knot_ts, ts, mode="linear")
                        
                        c2w_ = T_SE3.matrix().unsqueeze(0).cuda()  #[1, 4, 4]
                        render_pkg_ = self.rasterize_splats(camtoworlds=c2w_, render_mode="RGB")
                        event_img_ = render_pkg_["image"].detach().cpu().numpy()
                        
                        event_img_ =  to8b(event_img_)
                        im_name = f"f{idx_}_{ts:07.3f}s.jpg"
                        save_path = os.path.join(self.config["data"]["output"], self.config["data"]["exp_name"], "img_eval/est")
                        imageio.imwrite(os.path.join(save_path, im_name), event_img_)
                        
                        if self.config["render_tumvie_rgbCam_img"] and self.config["dataset"]=="tum_vie":
                            # render rbg camera image
                            img_rgbCam = self.render_with_gsplat(self.gs_model, T_se3, render_tumvie_rgb=True)['image']
                            img_rgbCam = np.repeat(img_rgbCam.detach().cpu().numpy()[:, :, np.newaxis], repeats=3, axis=2)
                            img_path = os.path.join(save_path, f"frame{idx_}_img_rgbCam_final.jpg")  
                            imageio.imwrite(img_path, to8b(img_rgbCam))
        
        print("================ finished ================")





        #=========================================================================================================

                                    # Triangle Splatting SLAM training loop

        #=========================================================================================================

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

# def main():
#     parser = ArgumentParser(description="Mesh SLAM (triangle splatting)")
#     parser.add_argument(
#         "--config", type=str, required=True,
#         help="Path to config YAML (e.g. configs/mono/tum/fr3_office.yaml)",
#     )
#     parser.add_argument("--eval", action="store_true", help="Evaluation mode")
#     parser.add_argument("--num_frames", type=int, default=None)
#     parser.add_argument("--view_only", action="store_true", default=False)
#     parser.add_argument("--add_keyframes", action="store_true", default=False)
#     parser.add_argument("--iterations_per_view", type=int, default=20)
#     parser.add_argument("--frame_stride", type=int, default=None)
#     parser.add_argument("--log_interval", type=int, default=2)
#     parser.add_argument("--dataset_path", type=str, default=None)
#     parser.add_argument("--profile", action="store_true", help="Profile with torch.profiler (time + memory)")
#     parser.add_argument("--profile_steps", type=int, default=5, help="Number of view iterations to profile")
#     parser.add_argument("--profile_exit", action="store_true", help="Exit after profiling (for quick profile runs)")
#     parser.add_argument("--start_frame", type=int, default=0, help="Skip first N frames of the dataset")
#     parser.add_argument(
#         "--resume_from",
#         type=str,
#         default=None,
#         help="Path to checkpoint folder (e.g. results/replica_room1/.../mesh/iteration_1250) to resume SLAM from",
#     )
#     parser.add_argument(
#         "-o", "--override",
#         action="append",
#         default=[],
#         metavar="KEY=VALUE",
#         help="Override config value (e.g. -o Results.use_rerun=True -o Training.frame_stride=2). Dot notation for nested keys.",
#     )

#     args = parser.parse_args(sys.argv[1:])
#     # config = load_config(args.config)
#     # TODO : Change the config loading file
#     cfg = config.load_config(args.config)

#     resume_from = getattr(args, "resume_from", None)
#     resume_iteration = None
#     if resume_from:
#         resume_from = os.path.abspath(resume_from)
#         if not os.path.isdir(resume_from):
#             raise SystemExit(f"--resume_from must be a directory: {resume_from}")
#         # Parse iteration from path (e.g. .../iteration_1250 -> 1250)
#         basename = os.path.basename(resume_from.rstrip("/"))
#         if basename.startswith("iteration_"):
#             try:
#                 resume_iteration = int(basename.split("_")[1])
#             except (IndexError, ValueError):
#                 pass
#         if resume_iteration is None:
#             raise SystemExit(f"Could not parse iteration number from --resume_from path: {resume_from}")
#         # Use config and save_dir from the checkpoint run
#         run_dir = os.path.dirname(os.path.dirname(resume_from))
#         config_path_in_run = os.path.join(run_dir, "config.yml")
#         if os.path.isfile(config_path_in_run):
#             with open(config_path_in_run, "r") as f:
#                 config = yaml.safe_load(f)
#             Log("Resume: loaded config from", config_path_in_run)
#         config["Results"]["save_dir"] = run_dir
#         config["Results"]["save_trj_kf_intv"] = 1
#         if args.dataset_path:
#             config.setdefault("Dataset", {})["dataset_path"] = args.dataset_path
#         Log("Resume: continuing run from iteration", resume_iteration, "save_dir:", run_dir)

#     # Apply command-line overrides (after resume so overrides win)
#     for override in args.override:
#         if "=" not in override:
#             raise SystemExit(f"Override must be KEY=VALUE, got: {override}")
#         key_path, _, value_str = override.partition("=")
#         keys = key_path.strip().split(".")
#         value = yaml.safe_load(value_str)
#         d = config
#         for k in keys[:-1]:
#             d = d.setdefault(k, {})
#         d[keys[-1]] = value

#     if args.eval:
#         config.setdefault("Results", {})["save_results"] = True
#         config.setdefault("Results", {})["use_gui"] = False
#         config.setdefault("Results", {})["eval_rendering"] = True

#     dataset_path = (
#         args.dataset_path
#         or config.get("Dataset", {}).get("dataset_path")
#         or config.get("dataset_path")
#     )
#     if not dataset_path:
#         raise SystemExit(
#             "Dataset path required: set Dataset.dataset_path in config or pass --dataset_path"
#         )

#     num_frames = args.num_frames or config.get("Training", {}).get("num_frames", None)
 
#     iterations_per_view = (
#         args.iterations_per_view
#         or config["Training"]["iterations_per_view"]
#     )
#     frame_stride = (
#         args.frame_stride
#         if args.frame_stride is not None
#         else config["Training"]["frame_stride"]
#     )
#     log_interval = args.log_interval or config["Training"]["log_interval"]

#     lp = ModelParams(ArgumentParser())
#     pipe = PipelineParams(ArgumentParser())
#     opt = OptimizationParams(ArgumentParser())
#     # Apply opt_params from config (overrides defaults)
#     for key, value in config.get("opt_params", {}).items():
#         if hasattr(opt, key):
#             setattr(opt, key, value)



    
#     if config["Results"]["save_results"]:
#         mkdir_p(config["Results"]["save_dir"])
#         current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
#         if resume_iteration is None:
#             path = (dataset_path or "").rstrip("/").split("/")
#             save_dir = os.path.join(
#                 config["Results"]["save_dir"],
#                 (path[-2] + "_" + path[-1]) if len(path) >= 2 else "run",
#                 current_datetime,
#             )
#             config["Results"]["save_dir"] = save_dir
#             mkdir_p(save_dir)
#             with open(os.path.join(save_dir, "config.yml"), "w") as f:
#                 yaml.dump(config, f)
#             Log("Saving results to " + save_dir)
#         else:
#             Log("Resume: reusing save_dir", config["Results"]["save_dir"])

#         use_wandb = config.get("Results", {}).get("use_wandb", False)
#         wandb.init(
#             project="triangle-slam",
#             name=f"{Path(args.config).stem}_{current_datetime}" + (f"_resume{resume_iteration}" if resume_iteration else ""),
#             config=config,
#             mode=None if use_wandb else "disabled",
#         )
#         wandb.define_metric("frame_idx")
#         wandb.define_metric("ate*", step_metric="frame_idx")


#     trainer = MeshSLAMTrainer(
#         args, config, dataset_path, args.config, opt, pipe, num_frames=num_frames,
#         resume_from=resume_from, resume_iteration=resume_iteration,
#     )
#     # if use rerun
#     if config["Results"]["use_rerun"]:
#         rr.init("triangle_splatting", spawn=True)
#     if not args.view_only:
#         trainer.train()
#     print("Done")

if __name__ == '__main__':            
    print('Start running...')
    parser = argparse.ArgumentParser(description='Arguments for running the NICE-SLAM/iMAP*.')
    parser.add_argument('--config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str, help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str, help='output folder, this have higher priority, can overwrite the one in config file')
    
    # lp = ModelParams(parser)
    # op = OptimizationParams(parser)
    # pparams_ = PipelineParams(parser)

    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.output is not None:
        cfg['data']['output'] = args.output
    
    print(f"\n ***************************************************")
    print(f" experiment name: {cfg['data']['exp_name']}")
    print(f" *************************************************** \n")

    print("Saving config and script...")
    save_path = os.path.join(cfg["data"]["output"], cfg['data']['exp_name'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(os.path.join(save_path, 'initialization')):
        os.makedirs(os.path.join(save_path, 'initialization'))
    if not os.path.exists(os.path.join(save_path, "tracking")):
        os.makedirs(os.path.join(save_path, "tracking"))
    if not os.path.exists(os.path.join(save_path, "BA")):
        os.makedirs(os.path.join(save_path, "BA"))
    if not os.path.exists(os.path.join(save_path, "global_BA")):
        os.makedirs(os.path.join(save_path, "global_BA"))
    if not os.path.exists(os.path.join(save_path, "depth_for_new_gs")):
        os.makedirs(os.path.join(save_path, "depth_for_new_gs"))
    if not os.path.exists(os.path.join(save_path, "img_eval")):
        os.makedirs(os.path.join(save_path, "img_eval"))
    if not os.path.exists(os.path.join(save_path, "img_eval/gt")):
        os.makedirs(os.path.join(save_path, "img_eval/gt"))
    if not os.path.exists(os.path.join(save_path, "img_eval/est")):
        os.makedirs(os.path.join(save_path, "img_eval/est"))
    # shutil.copy("coslam_blur.py", os.path.join(save_path, 'coslam_blur.py'))


    # TODO : Need to change
    lp = ModelParams(ArgumentParser())
    pipe = PipelineParams(ArgumentParser())
    opt = OptimizationParams(ArgumentParser())
    # Apply opt_params from config (overrides defaults)
    # for key, value in config.get("opt_params", {}).items():
    #     if hasattr(opt, key):
    #         setattr(opt, key, value)

    with open(os.path.join(save_path, 'config.json'),"w", encoding='utf-8') as f:
        f.write(json.dumps(cfg, indent=4))
        
    # copy the config file to output directory!!!
    config_file_name = args.config.split("/")[-1]
    shutil.copy(args.config, os.path.join(save_path, config_file_name))

    ###############
    # slam = SLAM(cfg)

    # slam.run()

    # TODO : Changed the code
    trainer = MeshSLAMTrainer(args, cfg, None, args.config, opt, pipe, 0, None, None)
    print("Event dataset loading is completed")
    trainer.train()



if __name__ == "__main__":
    main()
