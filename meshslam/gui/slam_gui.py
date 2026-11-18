"""
Triangle SLAM interactive viewer.

Runs in a separate process. Receives TrianglePacket updates via a queue and
renders the current mesh from any novel viewpoint the user chooses in the
Open3D 3D viewport.  No rerun required.

Launch via:
    python -m meshslam.gui.slam_gui          # standalone (empty scene)
or automatically from slam.py when Results.use_gui: True
"""
import math
import threading
import time
import sys
from pathlib import Path

import numpy as np
import torch
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

_root = Path(__file__).resolve().parents[3]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
_mesh_splatting = _root / "mesh_splatting"
if str(_mesh_splatting) not in sys.path:
    sys.path.insert(0, str(_mesh_splatting))

from meshslam.gui.gui_utils import (
    GUICamera, GUITriangles, TrianglePacket, PreviewCamera,
    Packet_vis2main, get_latest_queue, cv_gl,
)

triangle_render = None  # lazy-loaded after Open3D window is created

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


def _create_frustum_lineset(color=(0.0, 1.0, 0.0), size=0.02):
    """Return a canonical camera-frustum LineSet (pointing along +Z)."""
    pts = np.array([
        [0, 0, 0],
        [ 1, -0.5, 2],
        [-1, -0.5, 2],
        [ 1,  0.5, 2],
        [-1,  0.5, 2],
    ], dtype=np.float64) * size
    lines  = [[0,1],[0,2],[0,3],[0,4],[1,2],[1,3],[2,4],[3,4]]
    colors = [list(color)] * len(lines)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def _cam_c2w(cam_dict):
    """Return 4x4 C2W numpy matrix from a packed camera dict."""
    from mesh_splatting.utils.graphics_utils import getWorld2View2
    R = torch.from_numpy(cam_dict["R"]).float()
    T = torch.from_numpy(cam_dict["T"]).float()
    T_w2c = getWorld2View2(R, T).numpy()   # (4,4)
    return np.linalg.inv(T_w2c)            # C2W


class SLAM_GUI:
    """Interactive viewer for Triangle SLAM."""

    RENDER_FPS = 20.0

    def __init__(self, params_gui=None):
        self.step = 0
        self.process_finished = False

        self.q_main2vis = None
        self.q_vis2main = None
        self.pipe        = None
        self.background  = None

        self.triangles_gui: GUITriangles | None = None
        self.paused = False
        self.follow_cam = True

        self.frustum_dict = {}   # uid → name in scene
        self._trajectory_points = []  # list of (x,y,z) camera centers in order
        self._trajectory_len_drawn = 0  # length last committed to the scene
        self._pending_update = False  # prevents queueing multiple _scene_updates
        self._gt_frustum_names = []   # names of GT frustum objects in scene
        self._gt_trajectory_name = None
        self._gt_poses_visible = False
        self._est_poses = {}  # uid → C2W (4x4) estimated
        self._gt_poses = {}   # uid → C2W (4x4) ground truth
        self._first_cam_c2w = None  # 4x4 C2W of first frame (gravity prior)
        self.ts_prev = 0.0

        self._deform_mode = False
        self._drag_state = None
        self._original_verts = None
        self._elastic_anim = None  # dict with animation state when active
        self._recording = False
        self._record_frames = []  # list of numpy uint8 HxWx3 arrays
        self._sigma_slider_initialized = False
        self._saved_features_dc = None

        # Image preview panels (current render / GT / difference)
        self._current_cam_dict = None   # packed current-frame dict
        self._current_gt_img = None     # GT RGB uint8 HWC for current frame
        self._preview_aspect = 0.75     # h / w, updated when GT arrives
        self._preview_width = 320       # render width for the preview panels
        self._preview_ts_prev = 0.0     # throttle timer for preview updates
        self._preview_fps = 10.0

        self._init_widget()

        if params_gui is not None:
            self.q_main2vis  = params_gui.q_main2vis
            self.q_vis2main  = params_gui.q_vis2main
            self.pipe        = params_gui.pipe
            self.background  = params_gui.background.to("cuda")

        threading.Thread(target=self._update_thread, daemon=True).start()

    # ------------------------------------------------------------------
    # Widget setup
    # ------------------------------------------------------------------

    def _init_widget(self):
        self.window_w, self.window_h = 1600, 900

        self.window = gui.Application.instance.create_window(
            "Triangle SLAM Viewer", self.window_w, self.window_h
        )
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)

        # 3-D scene
        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        cg = rendering.ColorGrading(
            rendering.ColorGrading.Quality.ULTRA,
            rendering.ColorGrading.ToneMapping.LINEAR,
        )
        self.widget3d.scene.view.set_color_grading(cg)
        self.window.add_child(self.widget3d)

        # Side panel
        em = self.window.theme.font_size
        self.panel = gui.Vert(0.5 * em, gui.Margins(0.5 * em))

        # Pause button
        self.pause_btn = gui.Button("Pause")
        self.pause_btn.set_on_clicked(self._on_pause)
        self.panel.add_child(self.pause_btn)

        # Save Snapshot button (pause + save state to disk)
        self.snapshot_btn = gui.Button("Save Snapshot")
        self.snapshot_btn.set_on_clicked(self._on_snapshot_pause)
        self.panel.add_child(self.snapshot_btn)

        # Follow-camera checkbox
        self.follow_chbox = gui.Checkbox("Follow camera")
        self.follow_chbox.checked = True
        self.follow_chbox.set_on_checked(lambda v: setattr(self, "follow_cam", v))
        self.panel.add_child(self.follow_chbox)

        self.panel.add_child(gui.Label("Follow distance"))
        self._follow_dist = gui.Slider(gui.Slider.DOUBLE)
        self._follow_dist.set_limits(0.0, 2.0)
        self._follow_dist.double_value = 0.3
        self.panel.add_child(self._follow_dist)

        # Show trajectory lines between consecutive frustums
        self._trajectory_chbox = gui.Checkbox("Show trajectory")
        self._trajectory_chbox.checked = False
        self.panel.add_child(self._trajectory_chbox)

        # Show camera frustums
        self.cameras_chbox = gui.Checkbox("Show keyframe frustums")
        self.cameras_chbox.checked = True
        self.cameras_chbox.set_on_checked(self._on_toggle_frustums)
        self.panel.add_child(self.cameras_chbox)

        # Show GT poses (Umeyama-aligned)
        self._gt_chbox = gui.Checkbox("Show GT poses")
        self._gt_chbox.checked = False
        self._gt_chbox.set_on_checked(self._on_toggle_gt_poses)
        self.panel.add_child(self._gt_chbox)

        # Render mode
        self.panel.add_child(gui.Label("Render mode"))
        self.mode_rgb      = gui.Button("RGB")
        self.mode_depth    = gui.Button("Depth")
        self.mode_opacity  = gui.Button("Opacity")
        self.mode_normal   = gui.Button("Normal")
        self.mode_surf_nrm = gui.Button("Surf.Nrm")
        self._render_mode = "rgb"
        self.mode_rgb.set_on_clicked(lambda: self._set_mode("rgb"))
        self.mode_depth.set_on_clicked(lambda: self._set_mode("depth"))
        self.mode_opacity.set_on_clicked(lambda: self._set_mode("opacity"))
        self.mode_normal.set_on_clicked(lambda: self._set_mode("normal"))
        self.mode_surf_nrm.set_on_clicked(lambda: self._set_mode("surf_normal"))
        btn_row = gui.Horiz(0.25 * em)
        btn_row.add_child(self.mode_rgb)
        btn_row.add_child(self.mode_depth)
        btn_row.add_child(self.mode_opacity)
        self.panel.add_child(btn_row)
        btn_row2 = gui.Horiz(0.25 * em)
        btn_row2.add_child(self.mode_normal)
        btn_row2.add_child(self.mode_surf_nrm)
        self.panel.add_child(btn_row2)

        # Toggle the current/GT/difference preview panels
        self._previews_chbox = gui.Checkbox("Show image previews")
        self._previews_chbox.checked = True
        self._previews_chbox.set_on_checked(self._on_toggle_previews)
        self.panel.add_child(self._previews_chbox)

        # Triangle count label (always visible)
        self.tri_count_label = gui.Label("Triangles: -")
        self.panel.add_child(self.tri_count_label)

        # ---- Triangle Parameters (collapsible, hidden by default) ----
        tri_params = gui.CollapsableVert(
            "Triangle Parameters", 0.25 * em, gui.Margins(0.5 * em, 0, 0, 0)
        )
        tri_params.set_is_open(False)
        self._tri_params_section = tri_params

        # Sigma slider
        tri_params.add_child(gui.Label("Sigma (log scale)"))
        self._sigma_slider = gui.Slider(gui.Slider.DOUBLE)
        self._sigma_slider.set_limits(-7.0, 2.0)
        self._sigma_slider.double_value = 0.0
        self._sigma_slider.set_on_value_changed(self._on_sigma_changed)
        tri_params.add_child(self._sigma_slider)
        self._sigma_label = gui.Label("σ = -")
        tri_params.add_child(self._sigma_label)

        # Screenshot button
        self.screenshot_btn = gui.Button("Screenshot")
        self.screenshot_btn.set_on_clicked(self._on_screenshot)
        tri_params.add_child(self.screenshot_btn)

        self._record_btn = gui.Button("Record Video")
        self._record_btn.set_on_clicked(self._on_toggle_record)
        tri_params.add_child(self._record_btn)

        # TSDF fusion button + voxel size slider + view toggle
        tsdf_row = gui.Horiz(0.25 * em)
        self._tsdf_btn = gui.Button("TSDF Fusion")
        self._tsdf_btn.set_on_clicked(self._on_tsdf_fusion)
        tsdf_row.add_child(self._tsdf_btn)
        self._tsdf_view_chbox = gui.Checkbox("View TSDF")
        self._tsdf_view_chbox.checked = False
        self._tsdf_view_chbox.set_on_checked(self._on_tsdf_view_toggle)
        tsdf_row.add_child(self._tsdf_view_chbox)
        tri_params.add_child(tsdf_row)

        self._tsdf_voxel_label = gui.Label("Voxel: 0.020 m")
        tri_params.add_child(self._tsdf_voxel_label)
        self._tsdf_voxel_slider = gui.Slider(gui.Slider.DOUBLE)
        self._tsdf_voxel_slider.set_limits(0.001, 0.10)
        self._tsdf_voxel_slider.double_value = 0.02
        self._tsdf_voxel_slider.set_on_value_changed(self._on_tsdf_voxel_changed)
        tri_params.add_child(self._tsdf_voxel_slider)

        self._tsdf_mesh = None       # stored Open3D mesh
        self._tsdf_view_active = False  # True = show TSDF instead of rasterizer

        # View Delaunay mesh as Open3D geometry
        self._mesh_view_chbox = gui.Checkbox("View Mesh (O3D)")
        self._mesh_view_chbox.checked = False
        self._mesh_view_chbox.set_on_checked(self._on_mesh_view_toggle)
        tri_params.add_child(self._mesh_view_chbox)
        self._mesh_view_active = False

        self._freeze_verts_chbox = gui.Checkbox("Freeze vertices")
        self._freeze_verts_chbox.checked = False
        self._freeze_verts_chbox.set_on_checked(self._on_freeze_verts_toggle)
        tri_params.add_child(self._freeze_verts_chbox)

        # Random / restore colours
        color_row = gui.Horiz(0.25 * em)
        self._rand_color_btn = gui.Button("Random Colours")
        self._rand_color_btn.set_on_clicked(self._on_random_colors)
        self._restore_color_btn = gui.Button("Restore Colours")
        self._restore_color_btn.set_on_clicked(self._on_restore_colors)
        color_row.add_child(self._rand_color_btn)
        color_row.add_child(self._restore_color_btn)
        tri_params.add_child(color_row)

        self.panel.add_child(tri_params)

        # Retriangulate button (visible, outside any dropdown)
        self._retri_fix_verts = gui.Checkbox("Fix vertices")
        self._retri_fix_verts.checked = False
        self.retriangulate_btn = gui.Button("Retriangulate")
        self.retriangulate_btn.set_on_clicked(self._on_retriangulate)
        retri_row = gui.Horiz(0.25 * em)
        retri_row.add_child(self.retriangulate_btn)
        retri_row.add_child(self._retri_fix_verts)
        self.panel.add_child(retri_row)

        # Edit the live tracked mesh (deformations + committed physics objects go
        # back to the SLAM map) vs. a throwaway local copy. Default: live.
        self._edit_live_map = True
        self._edit_live_chbox = gui.Checkbox("Edit live map (SLAM)")
        self._edit_live_chbox.checked = True
        self._edit_live_chbox.set_on_checked(self._on_toggle_edit_live)
        self.panel.add_child(self._edit_live_chbox)

        # ---- Mesh Editing (collapsible); Deform Mode stays visible beside header ----
        self._deform_chbox = gui.Checkbox("Deform Mode")
        self._deform_chbox.checked = False
        self._deform_chbox.set_on_checked(self._on_deform_toggle)

        mesh_edit = gui.CollapsableVert(
            "Mesh Editing", 0.25 * em, gui.Margins(0.5 * em, 0, 0, 0)
        )
        mesh_edit.set_is_open(False)
        self._mesh_edit_section = mesh_edit

        mesh_hdr_row = gui.Horiz(0.5 * em)
        mesh_hdr_row.add_child(mesh_edit)
        mesh_hdr_row.add_child(self._deform_chbox)
        self.panel.add_child(mesh_hdr_row)

        mesh_edit.add_child(gui.Label("Brush radius"))
        self._brush_slider = gui.Slider(gui.Slider.DOUBLE)
        self._brush_slider.set_limits(0.005, 1.0)
        self._brush_slider.double_value = 1.0
        mesh_edit.add_child(self._brush_slider)

        self._connectivity_chbox = gui.Checkbox("Connectivity-aware")
        self._connectivity_chbox.checked = True
        mesh_edit.add_child(self._connectivity_chbox)

        mesh_edit.add_child(gui.Label("Diffusion iters"))
        self._diffusion_iters = gui.Slider(gui.Slider.INT)
        self._diffusion_iters.set_limits(1, 50)
        self._diffusion_iters.int_value = 15
        mesh_edit.add_child(self._diffusion_iters)

        smooth_row = gui.Horiz(0.25 * em)
        self._smooth_btn = gui.Button("Smooth")
        self._smooth_btn.set_on_clicked(self._on_smooth)
        self._smooth_iters = gui.Slider(gui.Slider.INT)
        self._smooth_iters.set_limits(1, 20)
        self._smooth_iters.int_value = 1
        smooth_row.add_child(self._smooth_btn)
        smooth_row.add_child(self._smooth_iters)
        mesh_edit.add_child(smooth_row)

        self._elastic_chbox = gui.Checkbox("Elastic")
        self._elastic_chbox.checked = False
        mesh_edit.add_child(self._elastic_chbox)

        mesh_edit.add_child(gui.Label("Frequency"))
        self._elastic_freq = gui.Slider(gui.Slider.DOUBLE)
        self._elastic_freq.set_limits(0.5, 20.0)
        self._elastic_freq.double_value = 1.0
        mesh_edit.add_child(self._elastic_freq)

        mesh_edit.add_child(gui.Label("Decay"))
        self._elastic_decay = gui.Slider(gui.Slider.DOUBLE)
        self._elastic_decay.set_limits(0.5, 10.0)
        self._elastic_decay.double_value = 1.0
        mesh_edit.add_child(self._elastic_decay)

        self._move_component_chbox = gui.Checkbox("Move component")
        self._move_component_chbox.checked = False
        mesh_edit.add_child(self._move_component_chbox)

        self._reset_verts_btn = gui.Button("Reset Vertices")
        self._reset_verts_btn.set_on_clicked(self._on_reset_verts)
        mesh_edit.add_child(self._reset_verts_btn)

        self._deform_status = gui.Label("")
        mesh_edit.add_child(self._deform_status)

        # Physics simulation. Objects are injected into the triangle map as
        # extra triangles and drawn by the CUDA rasteriser (see _extend_with_physics).
        self.panel.add_child(gui.Label("Physics"))
        self._physics = None  # PhysicsEngine instance, created on first use
        self._physics_active = False
        self._physics_geom_names = []  # tracked Open3D geometry names (legacy)
        self._physics_spawn_mode = None  # "ball" when waiting for a click
        self._physics_templates_ready = False
        self._saved_phys = None  # saved tensors during physics-extended render
        # pybullet keeps its own static collision mesh; mark it dirty whenever the
        # rendered mesh changes so we re-sync it (lazily, before the next spawn)
        # and balls always collide with exactly what is shown.
        self._physics_mesh_dirty = False

        phys_row1 = gui.Horiz(0.25 * em)
        self._phys_ball_btn = gui.Button("Drop Ball")
        self._phys_ball_btn.set_on_clicked(self._on_spawn_ball)
        phys_row1.add_child(self._phys_ball_btn)
        self._phys_run_chbox = gui.Checkbox("Run Physics")
        self._phys_run_chbox.checked = False
        self._phys_run_chbox.set_on_checked(self._on_physics_toggle)
        phys_row1.add_child(self._phys_run_chbox)
        self._phys_clear_btn = gui.Button("Clear")
        self._phys_clear_btn.set_on_clicked(self._on_physics_clear)
        phys_row1.add_child(self._phys_clear_btn)
        self.panel.add_child(phys_row1)

        # Ball size slider
        self._ball_radius = 0.02
        ball_size_row = gui.Horiz(0.25 * em)
        ball_size_row.add_child(gui.Label("Ball size"))
        self._ball_size_slider = gui.Slider(gui.Slider.DOUBLE)
        self._ball_size_slider.set_limits(0.005, 0.1)
        self._ball_size_slider.double_value = self._ball_radius
        self._ball_size_slider.set_on_value_changed(self._on_ball_size_changed)
        ball_size_row.add_child(self._ball_size_slider)
        self.panel.add_child(ball_size_row)

        # Ball speed slider
        self._ball_speed = 2.0
        ball_speed_row = gui.Horiz(0.25 * em)
        ball_speed_row.add_child(gui.Label("Ball speed"))
        self._ball_speed_slider = gui.Slider(gui.Slider.DOUBLE)
        self._ball_speed_slider.set_limits(0.5, 10.0)
        self._ball_speed_slider.double_value = self._ball_speed
        self._ball_speed_slider.set_on_value_changed(lambda v: setattr(self, "_ball_speed", v))
        ball_speed_row.add_child(self._ball_speed_slider)
        self.panel.add_child(ball_speed_row)

        # Commit the current physics objects to the tracked SLAM map so mapping
        # optimises them (requires "Edit live map"). The objects stop simulating.
        self._phys_commit_btn = gui.Button("Commit to map")
        self._phys_commit_btn.set_on_clicked(self._on_commit_objects)
        self.panel.add_child(self._phys_commit_btn)

        self._phys_status = gui.Label("")
        self.panel.add_child(self._phys_status)

        self.window.add_child(self.panel)

        # Image preview panels: current render / GT / difference. These float
        # over the bottom-left of the 3D view and are positioned in _on_layout.
        placeholder = o3d.geometry.Image(np.zeros((3, 4, 3), dtype=np.uint8))
        self._preview_titles = ["Current", "Ground truth", "Difference"]
        self._preview_labels = []
        self._preview_widgets = []
        for title in self._preview_titles:
            lbl = gui.Label(title)
            img_w = gui.ImageWidget(placeholder)
            self.window.add_child(lbl)
            self.window.add_child(img_w)
            self._preview_labels.append(lbl)
            self._preview_widgets.append(img_w)

        # Default material for frustum lines
        self.lit = rendering.MaterialRecord()
        self.lit.shader = "unlitLine"
        self.lit.line_width = 1.0

        # Set a comfortable default camera pose
        bounds = o3d.geometry.AxisAlignedBoundingBox([-3, -3, -3], [3, 3, 3])
        self.widget3d.setup_camera(60.0, bounds, [0, 0, 0])

    def _on_layout(self, ctx):
        r = self.window.content_rect
        panel_w = 340
        self.widget3d_width = r.width - panel_w
        self.widget3d.frame = gui.Rect(r.x, r.y, self.widget3d_width, r.height)
        self.panel.frame    = gui.Rect(r.x + self.widget3d_width, r.y, panel_w, r.height)
        self._layout_previews(r)

    def _layout_previews(self, r):
        """Position the three preview thumbnails along the bottom-left of the 3D view."""
        if not getattr(self, "_preview_widgets", None):
            return
        show = self._previews_chbox.checked
        em = self.window.theme.font_size
        label_h = int(em + 4)
        margin = 8
        gap = 8
        thumb_w = 240
        thumb_h = max(1, int(thumb_w * self._preview_aspect))
        y_img = r.y + r.height - thumb_h - margin
        y_lbl = y_img - label_h
        for i, (lbl, img_w) in enumerate(zip(self._preview_labels, self._preview_widgets)):
            x = r.x + margin + i * (thumb_w + gap)
            lbl.visible = show
            img_w.visible = show
            if show:
                lbl.frame = gui.Rect(x, y_lbl, thumb_w, label_h)
                img_w.frame = gui.Rect(x, y_img, thumb_w, thumb_h)

    def _on_toggle_previews(self, checked):
        self._layout_previews(self.window.content_rect)
        self.window.set_needs_layout()

    def _on_close(self):
        # Auto-save recorded frames if recording was in progress
        if self._recording and self._record_frames:
            self._recording = False
            self._auto_save_video()
        self.process_finished = True
        return True

    def _auto_save_video(self):
        """Save recorded frames to the working directory without a file dialog."""
        from datetime import datetime
        try:
            import cv2
            frames = self._record_frames
            if not frames:
                return
            path = f"slam_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(path, fourcc, self.RENDER_FPS, (w, h))
            for f in frames:
                writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            writer.release()
            print(f"[GUI] Auto-saved {len(frames)} frames to {path}")
        except Exception as e:
            print(f"[GUI] Auto-save failed: {e}")
        self._record_frames = []

    def _on_pause(self):
        self.paused = not self.paused
        self.pause_btn.text = "Resume" if self.paused else "Pause"
        if self.q_vis2main is not None:
            pkt = Packet_vis2main()
            pkt.flag_pause = self.paused
            pkt.flag_resume = not self.paused
            try:
                self.q_vis2main.put_nowait(pkt)
            except Exception:
                pass

    def _on_snapshot_pause(self):
        """Pause and save a snapshot to disk (old default behaviour)."""
        if not self.paused:
            self.paused = True
            self.pause_btn.text = "Resume"
        if self.q_vis2main is not None:
            pkt = Packet_vis2main()
            pkt.flag_pause = True
            pkt.flag_snapshot = True
            try:
                self.q_vis2main.put_nowait(pkt)
            except Exception:
                pass

    def _on_toggle_frustums(self, show):
        for name in self.frustum_dict:
            self.widget3d.scene.show_geometry(name, show)

    def _on_toggle_gt_poses(self, show):
        self._gt_poses_visible = show
        if show:
            self._rebuild_gt_overlay()
        else:
            self._remove_gt_overlay()

    def _remove_gt_overlay(self):
        for name in self._gt_frustum_names:
            if self.widget3d.scene.has_geometry(name):
                self.widget3d.scene.remove_geometry(name)
        self._gt_frustum_names.clear()
        if self._gt_trajectory_name and self.widget3d.scene.has_geometry(self._gt_trajectory_name):
            self.widget3d.scene.remove_geometry(self._gt_trajectory_name)
            self._gt_trajectory_name = None

    def _rebuild_gt_overlay(self):
        """Align GT poses to estimated poses via Umeyama and display."""
        self._remove_gt_overlay()

        if len(self._gt_poses) < 3 or len(self._est_poses) < 3:
            return

        # Match by uid
        common_uids = sorted(set(self._est_poses.keys()) & set(self._gt_poses.keys()))
        if len(common_uids) < 3:
            return

        est_centers = np.array([self._est_poses[u][:3, 3] for u in common_uids])
        gt_centers = np.array([self._gt_poses[u][:3, 3] for u in common_uids])

        # Umeyama: find R, t such that aligned_gt = gt @ R.T + t
        from utils.eval_utils import umeyama_se3
        R_align, t_align = umeyama_se3(gt_centers, est_centers)

        # Build alignment transform (4x4)
        T_align = np.eye(4)
        T_align[:3, :3] = R_align
        T_align[:3, 3] = t_align

        # Add aligned GT frustums (red)
        mat = rendering.MaterialRecord()
        mat.shader = "unlitLine"
        mat.line_width = 2.0

        aligned_gt_centers = []
        for uid in common_uids:
            gt_c2w = self._gt_poses[uid]
            # Align: apply T_align to GT pose
            aligned_c2w = T_align @ gt_c2w
            aligned_gt_centers.append(aligned_c2w[:3, 3].copy())

            frustum = _create_frustum_lineset(color=(1.0, 0.0, 0.0), size=0.02)
            frustum.transform(aligned_c2w)
            name = f"gt_cam_{uid}"
            self.widget3d.scene.add_geometry(name, frustum, mat)
            self._gt_frustum_names.append(name)

        # GT trajectory line (red)
        if len(aligned_gt_centers) >= 2:
            pts = np.array(aligned_gt_centers)
            lines = [[i, i + 1] for i in range(len(pts) - 1)]
            colors = [[1.0, 0.0, 0.0]] * len(lines)
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(pts)
            ls.lines = o3d.utility.Vector2iVector(lines)
            ls.colors = o3d.utility.Vector3dVector(colors)
            self._gt_trajectory_name = "gt_trajectory"
            self.widget3d.scene.add_geometry(self._gt_trajectory_name, ls, mat)

    def _on_screenshot(self):
        """Save current render, optional view-with-frustums, and normals to PNGs."""
        out = self._render()
        if out is None:
            return
        result, cam = out
        from datetime import datetime
        base_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = f"slam_screenshot_{base_ts}.png"

        # Build all images from one render: current mode, normal, surf_normal
        img_main = self._result_to_o3d(result, cam)
        img_normal = self._result_to_o3d_normal(result, cam, "normal")
        img_surf_normal = self._result_to_o3d_normal(result, cam, "surf_normal")
        # Optional: 3D view (with frustums) - capture from scene
        view_img = self._capture_scene_view()

        def _on_ok(path):
            try:
                import os
                base, _ = os.path.splitext(path)
                o3d.io.write_image(path, img_main)
                if img_normal is not None:
                    o3d.io.write_image(f"{base}_normal.png", img_normal)
                if img_surf_normal is not None:
                    o3d.io.write_image(f"{base}_surf_normal.png", img_surf_normal)
                if view_img is not None:
                    o3d.io.write_image(f"{base}_with_frustums.png", view_img)
            except Exception:
                pass
            self.window.close_dialog()

        def _on_cancel():
            self.window.close_dialog()

        fd = gui.FileDialog(gui.FileDialog.SAVE, "Save screenshot", self.window.theme)
        fd.add_filter(".png", "PNG image")
        fd.set_path(default_path)
        fd.set_on_done(_on_ok)
        fd.set_on_cancel(_on_cancel)
        self.window.show_dialog(fd)

    def _on_toggle_record(self):
        if not self._recording:
            self._recording = True
            self._record_frames = []
            self._record_btn.text = "Stop Recording"
            self._deform_status.text = "Recording…"
        else:
            self._recording = False
            self._record_btn.text = "Record Video"
            self._deform_status.text = ""
            if self._record_frames:
                self._save_video()

    def _save_video(self):
        from datetime import datetime
        default_path = f"slam_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"

        def _on_ok(path):
            try:
                import cv2
                frames = self._record_frames
                h, w = frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(path, fourcc, self.RENDER_FPS, (w, h))
                for f in frames:
                    writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                writer.release()
                self._deform_status.text = f"Saved {len(frames)} frames"
            except Exception as e:
                self._deform_status.text = f"Save failed: {e}"
            self._record_frames = []
            self.window.close_dialog()

        def _on_cancel():
            self._record_frames = []
            self.window.close_dialog()

        fd = gui.FileDialog(gui.FileDialog.SAVE, "Save video", self.window.theme)
        fd.add_filter(".mp4", "MP4 video")
        fd.set_path(default_path)
        fd.set_on_done(_on_ok)
        fd.set_on_cancel(_on_cancel)
        self.window.show_dialog(fd)

    def _on_retriangulate(self):
        # Retriangulation yields a connected mesh; drop sigma so the triangles
        # render crisply rather than over-blurred. (Slider is log-scale.)
        self._sigma_slider.double_value = -1.5
        self._on_sigma_changed(-1.5)  # setting double_value doesn't fire the callback
        if self.q_vis2main is not None:
            pkt = Packet_vis2main()
            pkt.flag_retriangulate = True
            pkt.retriangulate_fix_vertices = self._retri_fix_verts.checked
            try:
                self.q_vis2main.put_nowait(pkt)
            except Exception:
                pass

    def _on_freeze_verts_toggle(self, checked):
        if self.q_vis2main is not None:
            pkt = Packet_vis2main()
            pkt.flag_freeze_vertices = checked
            try:
                self.q_vis2main.put_nowait(pkt)
            except Exception:
                pass

    def _on_tsdf_voxel_changed(self, val):
        self._tsdf_voxel_label.text = f"Voxel: {val:.3f} m"

    def _on_tsdf_fusion(self):
        if self.q_vis2main is not None:
            pkt = Packet_vis2main()
            pkt.flag_tsdf_fusion = True
            pkt.tsdf_voxel_size = self._tsdf_voxel_slider.double_value
            try:
                self.q_vis2main.put_nowait(pkt)
            except Exception:
                pass

    def _on_tsdf_view_toggle(self, checked):
        self._tsdf_view_active = checked
        if self._tsdf_mesh is not None:
            self.widget3d.scene.show_geometry("tsdf_mesh", checked)
        # Disable mesh view if enabling TSDF view
        if checked and self._mesh_view_active:
            self._mesh_view_chbox.checked = False
            self._on_mesh_view_toggle(False)

    def _on_mesh_view_toggle(self, checked):
        self._mesh_view_active = checked
        if checked:
            self._update_o3d_mesh()
            # Disable TSDF view
            if self._tsdf_view_active:
                self._tsdf_view_chbox.checked = False
                self._on_tsdf_view_toggle(False)
        else:
            if self.widget3d.scene.has_geometry("delaunay_mesh"):
                self.widget3d.scene.show_geometry("delaunay_mesh", False)

    def _update_o3d_mesh(self):
        """Build and display the current Delaunay mesh as Open3D geometry."""
        if self.triangles_gui is None:
            return
        verts = self.triangles_gui.vertices.detach().cpu().numpy().astype(np.float64)
        faces = self.triangles_gui._triangle_indices.cpu().numpy().astype(np.int32)

        # Get vertex colors from SH DC coefficients (degree 0 → constant color)
        # _features_dc is (V, 1, 3), SH DC = color * SH_C0 where SH_C0 ≈ 0.2821
        import math
        SH_C0 = 0.5 / math.sqrt(math.pi)  # ≈ 0.2821
        features_dc = self.triangles_gui._features_dc.detach().cpu().numpy()  # (V, 1, 3)
        colors = features_dc[:, 0, :] * SH_C0 + 0.5  # SH → linear RGB
        colors = np.clip(colors, 0, 1)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        mesh.compute_vertex_normals()

        name = "delaunay_mesh"
        if self.widget3d.scene.has_geometry(name):
            self.widget3d.scene.remove_geometry(name)
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        self.widget3d.scene.add_geometry(name, mesh, mat)
        self.widget3d.scene.show_geometry(name, True)

    def _update_tsdf_mesh(self, mesh_dict):
        """Store a new TSDF mesh from serialized numpy arrays.

        mesh_dict: {"vertices": (V,3), "triangles": (F,3), "vertex_normals": (V,3),
                     "vertex_colors": (V,3) or None}.
        """
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(mesh_dict["vertices"])
        mesh.triangles = o3d.utility.Vector3iVector(mesh_dict["triangles"])
        if mesh_dict.get("vertex_normals") is not None:
            mesh.vertex_normals = o3d.utility.Vector3dVector(mesh_dict["vertex_normals"])
        if mesh_dict.get("vertex_colors") is not None:
            mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_dict["vertex_colors"])

        self._tsdf_mesh = mesh
        name = "tsdf_mesh"
        if self.widget3d.scene.has_geometry(name):
            self.widget3d.scene.remove_geometry(name)
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"  # show vertex colors directly, no lighting
        self.widget3d.scene.add_geometry(name, mesh, mat)
        # Only show if the TSDF view toggle is active
        self.widget3d.scene.show_geometry(name, self._tsdf_view_active)

    # ------------------------------------------------------------------
    # Physics simulation
    # ------------------------------------------------------------------

    def _compute_dominant_normal(self):
        """Robust gravity direction via normal histogram with camera-down prior.

        Bins face normals (area-weighted) on the unit sphere using a coarse
        icosphere grid, then picks the bin closest to the first camera's down
        direction - this corresponds to horizontal planes (ground/table) in a
        Manhattan-world scene.
        """
        verts = self.triangles_gui.vertices.detach().cpu().numpy().astype(np.float64)
        faces = self.triangles_gui._triangle_indices.cpu().numpy().astype(np.int32)
        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)  # (F, 3), magnitude = 2*area
        areas = np.linalg.norm(cross, axis=1, keepdims=True)
        # Normalise to unit normals; skip degenerate faces
        valid = (areas.squeeze() > 1e-12)
        normals = np.zeros_like(cross)
        normals[valid] = cross[valid] / areas[valid]
        areas = areas.squeeze()

        # Camera-down prior: OpenCV +Y of first frame's C2W
        if self._first_cam_c2w is not None:
            cam_down = self._first_cam_c2w[:3, 1].copy()  # OpenCV y-axis = down
            cam_down /= np.linalg.norm(cam_down) + 1e-12
        else:
            cam_down = np.array([0.0, 0.0, -1.0])  # fallback

        # Coarse bin centres on the unit sphere (icosphere subdivisions=2 -> 162 verts)
        import trimesh
        ico = trimesh.creation.icosphere(subdivisions=2)
        bin_dirs = np.array(ico.vertices, dtype=np.float64)  # (N_bins, 3)

        # Assign each valid face normal to its nearest bin (cosine similarity)
        dots = normals[valid] @ bin_dirs.T  # (F_valid, N_bins)
        assignments = dots.argmax(axis=1)   # (F_valid,)
        valid_areas = areas[valid]

        # Accumulate area-weighted votes per bin
        n_bins = len(bin_dirs)
        bin_weights = np.zeros(n_bins, dtype=np.float64)
        np.add.at(bin_weights, assignments, valid_areas)

        # Only consider bins within 60 deg of camera-down prior
        cos_thresh = np.cos(np.radians(60.0))
        cam_dots = bin_dirs @ cam_down
        candidate_mask = cam_dots >= cos_thresh

        if not candidate_mask.any():
            cos_thresh = np.cos(np.radians(90.0))
            candidate_mask = cam_dots >= cos_thresh

        if not candidate_mask.any():
            return cam_down  # ultimate fallback: just use camera down

        masked_weights = np.where(candidate_mask, bin_weights, -1.0)
        best_bin = masked_weights.argmax()

        # Refine: area-weighted average of normals assigned to this bin
        in_bin = (assignments == best_bin)
        if in_bin.any():
            w = valid_areas[in_bin, np.newaxis]
            refined = (normals[valid][in_bin] * w).sum(axis=0)
            n = np.linalg.norm(refined)
            if n > 1e-12:
                return refined / n

        return bin_dirs[best_bin]

    def _ensure_physics(self, surface_normal=None):
        """Lazily create the PhysicsEngine and load the current mesh.

        Gravity is set along the dominant mesh normal (area-weighted, with a
        camera-down prior), so it points into the ground/table plane.
        """
        if self._physics is not None:
            return True
        if self.triangles_gui is None:
            return False
        try:
            from meshslam.gui.physics import PhysicsEngine
        except ImportError:
            self._phys_status.text = "Physics needs pybullet: pip install pybullet"
            print("Physics disabled: pybullet is not installed (pip install pybullet)")
            return False
        self._physics = PhysicsEngine()
        verts = self.triangles_gui.vertices.detach().cpu().numpy().astype(np.float64)
        faces = self.triangles_gui._triangle_indices.cpu().numpy().astype(np.int32)
        self._physics.set_mesh(verts, faces)
        # Dominant normal points along camera-down (into the ground), so use it
        # directly as the gravity direction (no negation needed).
        self._physics_gravity_dir = self._compute_dominant_normal()
        self._physics.set_gravity((self._physics_gravity_dir * 9.81).tolist())
        return True

    def _refresh_physics_mesh(self):
        """Re-sync pybullet's static collision mesh to the currently rendered
        mesh, if it has changed. Cheap no-op when nothing changed."""
        if self._physics is None or self.triangles_gui is None:
            return
        if not self._physics_mesh_dirty:
            return
        verts = self.triangles_gui.vertices.detach().cpu().numpy().astype(np.float64)
        faces = self.triangles_gui._triangle_indices.cpu().numpy().astype(np.int32)
        self._physics.set_mesh(verts, faces)
        self._physics_mesh_dirty = False

    def _on_ball_size_changed(self, value):
        self._ball_radius = value

    def _on_spawn_ball(self):
        self._physics_spawn_mode = "ball"
        self._phys_status.text = "Click to shoot ball"
        self.widget3d.set_on_mouse(self._on_mouse_physics)

    def _reset_physics_mouse(self):
        """Leave spawn mode and restore the default mouse handler."""
        self._physics_spawn_mode = None
        if self._physics is not None:
            self._phys_status.text = ""
        if self._deform_mode:
            self.widget3d.set_on_mouse(self._on_mouse_deform)
        else:
            self.widget3d.set_on_mouse(
                lambda _: gui.Widget.EventCallbackResult.IGNORED
            )

    def _on_mouse_physics(self, event):
        if self._physics_spawn_mode is None:
            return gui.Widget.EventCallbackResult.IGNORED

        if (event.type == gui.MouseEvent.Type.BUTTON_DOWN
                and event.buttons & int(gui.MouseButton.LEFT)):
            if self.triangles_gui is None:
                return gui.Widget.EventCallbackResult.IGNORED

            origin, direction = self._pixel_to_ray(event.x, event.y)
            if self._ensure_physics():
                # Collide against exactly what is rendered now.
                self._refresh_physics_mesh()
                # Shoot ball from near the image plane towards the cursor ray
                spawn_pos = origin + direction * 0.3  # ~30cm in front of camera
                velocity = (direction * self._ball_speed).tolist()
                self._physics.spawn_ball(spawn_pos, velocity=velocity,
                                         radius=self._ball_radius)

                if not self._physics_active:
                    self._phys_run_chbox.checked = True
                    self._on_physics_toggle(True)

            # Balls stay armed for rapid fire. If physics is unavailable, exit
            # so the "needs pybullet" status hint stays visible.
            if self._physics is None:
                self._reset_physics_mouse()
            return gui.Widget.EventCallbackResult.CONSUMED

        # Right-click exits ball-shooting mode
        if (event.type == gui.MouseEvent.Type.BUTTON_DOWN
                and event.buttons & int(gui.MouseButton.RIGHT)):
            self._reset_physics_mouse()
            return gui.Widget.EventCallbackResult.CONSUMED

        return gui.Widget.EventCallbackResult.IGNORED

    def _on_physics_toggle(self, checked):
        self._physics_active = checked

    def _on_physics_clear(self):
        if self._physics is not None:
            self._physics.clear()
        # Clean up any legacy Open3D geometries
        for name in self._physics_geom_names:
            if self.widget3d.scene.has_geometry(name):
                self.widget3d.scene.remove_geometry(name)
        self._physics_geom_names.clear()

    def _tick_physics(self):
        """Step the physics sim. Visuals are rendered via the CUDA rasteriser."""
        if not self._physics_active or self._physics is None:
            return
        # Gravity direction is fixed (set once from the dominant mesh normal)
        self._physics.step(n=4)

    # ------------------------------------------------------------------
    # Physics -> CUDA rasteriser integration
    # ------------------------------------------------------------------

    def _init_physics_templates(self):
        """Create template meshes for physics objects (cached on first use)."""
        import trimesh

        # Ball: icosphere with 42 verts, 80 faces
        ball = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
        self._tmpl_ball_v = torch.tensor(ball.vertices, dtype=torch.float32)
        self._tmpl_ball_f = torch.tensor(ball.faces, dtype=torch.int32)

        # Water droplet: simpler sphere, 12 verts, 20 faces
        water = trimesh.creation.icosphere(subdivisions=0, radius=1.0)
        self._tmpl_water_v = torch.tensor(water.vertices, dtype=torch.float32)
        self._tmpl_water_f = torch.tensor(water.faces, dtype=torch.int32)

        # Confetti: flat quad (2 triangles)
        self._tmpl_confetti_v = torch.tensor([
            [-1.0, -0.5, 0.0], [1.0, -0.5, 0.0],
            [1.0, 0.5, 0.0], [-1.0, 0.5, 0.0],
        ], dtype=torch.float32)
        self._tmpl_confetti_f = torch.tensor([
            [0, 1, 2], [0, 2, 3],
        ], dtype=torch.int32)

        self._physics_templates_ready = True

    def _extend_with_physics(self):
        """Temporarily extend triangles_gui with physics object meshes for CUDA rendering.

        Returns True if tensors were extended (and must be restored after render).
        """
        if not self._physics_active or self._physics is None:
            return False
        if self.triangles_gui is None:
            return False
        if not self._physics_templates_ready:
            self._init_physics_templates()

        ball_pos = self._physics.get_ball_positions()
        liq_pos = self._physics.get_liquid_positions()
        liq_types = self._physics.get_liquid_types()

        if len(ball_pos) == 0 and len(liq_pos) == 0:
            return False

        tg = self.triangles_gui
        device = tg.vertices.device
        SH_C0 = 0.28209479177387814

        # Save originals
        self._saved_phys = {
            'vertices': tg.vertices,
            '_triangle_indices': tg._triangle_indices,
            '_features_dc': tg._features_dc,
            '_features_rest': tg._features_rest,
            '_vertex_weight': tg._vertex_weight,
            '_triangles': tg._triangles,
        }

        all_verts = []
        all_faces = []
        all_colors = []
        V_offset = tg.vertices.shape[0]

        confetti_rgbs = [
            [1.0, 0.2, 0.2], [0.2, 1.0, 0.3], [0.2, 0.4, 1.0], [1.0, 0.9, 0.1],
            [1.0, 0.4, 0.8], [0.1, 1.0, 0.9], [1.0, 0.5, 0.0], [0.7, 0.3, 1.0],
        ]

        # Add balls
        for pos in ball_pos:
            v = self._tmpl_ball_v * self._ball_radius + torch.tensor(pos, dtype=torch.float32)
            f = self._tmpl_ball_f + V_offset
            V_offset += v.shape[0]
            rgb = torch.tensor([0.85, 0.15, 0.1], dtype=torch.float32)
            fdc = ((rgb - 0.5) / SH_C0).unsqueeze(0).expand(v.shape[0], -1)
            all_verts.append(v)
            all_faces.append(f)
            all_colors.append(fdc)

        # Add liquid / water / confetti
        for i, pos in enumerate(liq_pos):
            ptype = liq_types[i] if i < len(liq_types) else "liquid"
            if ptype == "water":
                tmpl_v, tmpl_f = self._tmpl_water_v, self._tmpl_water_f
                scale = 0.002
                rgb = torch.tensor([0.15, 0.5, 0.95], dtype=torch.float32)
            elif ptype == "confetti":
                tmpl_v, tmpl_f = self._tmpl_confetti_v, self._tmpl_confetti_f
                scale = 0.006
                rgb = torch.tensor(confetti_rgbs[i % len(confetti_rgbs)], dtype=torch.float32)
            else:
                tmpl_v, tmpl_f = self._tmpl_water_v, self._tmpl_water_f
                scale = 0.005
                rgb = torch.tensor([0.1, 0.4, 0.9], dtype=torch.float32)

            v = tmpl_v * scale + torch.tensor(pos, dtype=torch.float32)
            f = tmpl_f + V_offset
            V_offset += v.shape[0]
            fdc = ((rgb - 0.5) / SH_C0).unsqueeze(0).expand(v.shape[0], -1)
            all_verts.append(v)
            all_faces.append(f)
            all_colors.append(fdc)

        if not all_verts:
            self._saved_phys = None
            return False

        extra_v = torch.cat(all_verts, dim=0).to(device)
        extra_f = torch.cat(all_faces, dim=0).to(device)
        extra_c = torch.cat(all_colors, dim=0).to(device)  # (V_extra, 3)
        V_extra = extra_v.shape[0]

        # Extend all tensors the renderer reads
        tg.vertices = torch.cat([tg.vertices, extra_v], dim=0)
        tg._triangle_indices = torch.cat([tg._triangle_indices, extra_f], dim=0)
        tg._features_dc = torch.cat([tg._features_dc, extra_c.unsqueeze(1)], dim=0)

        F_rest = tg._features_rest.shape[1]
        extra_rest = torch.zeros(V_extra, F_rest, 3, device=device)
        tg._features_rest = torch.cat([tg._features_rest, extra_rest], dim=0)

        # High weight -> sigmoid(5) ~ 0.993 -> fully opaque.
        # Match the shape of the existing weight tensor (may be (V,) or (V,1)).
        w_shape = tg._vertex_weight.shape
        if len(w_shape) == 2:
            extra_weight = torch.full((V_extra, w_shape[1]), 5.0, device=device)
        else:
            extra_weight = torch.full((V_extra,), 5.0, device=device)
        tg._vertex_weight = torch.cat([tg._vertex_weight, extra_weight], dim=0)

        tg._triangles = tg.vertices[tg._triangle_indices]
        return True

    def _restore_physics(self):
        """Restore original tensors after physics-extended rendering."""
        if self._saved_phys is None:
            return
        tg = self.triangles_gui
        for key, val in self._saved_phys.items():
            setattr(tg, key, val)
        self._saved_phys = None

    def _collect_physics_meshes(self):
        """Build world-space (verts Nx3, faces Mx3, colours Nx3) for all current
        physics objects, for baking into the tracked map. Returns None if empty."""
        if self._physics is None:
            return None
        if not self._physics_templates_ready:
            self._init_physics_templates()
        ball_pos = self._physics.get_ball_positions()
        liq_pos = self._physics.get_liquid_positions()
        liq_types = self._physics.get_liquid_types()
        if len(ball_pos) == 0 and len(liq_pos) == 0:
            return None

        confetti_rgbs = [
            [1.0, 0.2, 0.2], [0.2, 1.0, 0.3], [0.2, 0.4, 1.0], [1.0, 0.9, 0.1],
            [1.0, 0.4, 0.8], [0.1, 1.0, 0.9], [1.0, 0.5, 0.0], [0.7, 0.3, 1.0],
        ]
        all_v, all_f, all_c = [], [], []
        v_off = 0

        def add(tmpl_v, tmpl_f, scale, pos, rgb):
            nonlocal v_off
            v = tmpl_v.cpu().numpy() * scale + np.asarray(pos, dtype=np.float32)
            f = tmpl_f.cpu().numpy().astype(np.int64) + v_off
            v_off += v.shape[0]
            all_v.append(v.astype(np.float32))
            all_f.append(f)
            all_c.append(np.tile(np.asarray(rgb, dtype=np.float32), (v.shape[0], 1)))

        for pos in ball_pos:
            add(self._tmpl_ball_v, self._tmpl_ball_f, self._ball_radius, pos, [0.85, 0.15, 0.1])
        for i, pos in enumerate(liq_pos):
            ptype = liq_types[i] if i < len(liq_types) else "liquid"
            if ptype == "water":
                add(self._tmpl_water_v, self._tmpl_water_f, 0.002, pos, [0.15, 0.5, 0.95])
            elif ptype == "confetti":
                add(self._tmpl_confetti_v, self._tmpl_confetti_f, 0.006, pos,
                    confetti_rgbs[i % len(confetti_rgbs)])
            else:
                add(self._tmpl_water_v, self._tmpl_water_f, 0.005, pos, [0.1, 0.4, 0.9])

        return {
            "verts": np.concatenate(all_v, axis=0),
            "faces": np.concatenate(all_f, axis=0).astype(np.int32),
            "colors": np.concatenate(all_c, axis=0),
        }

    def _on_commit_objects(self):
        """Send the current physics objects to the SLAM process to be added to
        the tracked map, then drop them from the simulation."""
        if not self._edit_live_map:
            self._phys_status.text = "Enable 'Edit live map' to commit"
            return
        meshes = self._collect_physics_meshes()
        if meshes is None:
            self._phys_status.text = "Nothing to commit"
            return
        if self.q_vis2main is not None:
            try:
                pkt = Packet_vis2main()
                pkt.commit_objects = meshes
                self.q_vis2main.put_nowait(pkt)
            except Exception:
                pass
        # Objects are now owned by the map; stop simulating them here.
        if self._physics is not None:
            self._physics.clear()
        n = meshes["verts"].shape[0]
        self._phys_status.text = f"Committed {n} verts to map"

    # ------------------------------------------------------------------
    # Mesh editing: deform mode
    # ------------------------------------------------------------------

    def _on_toggle_edit_live(self, checked):
        self._edit_live_map = checked

    def _send_vertices_to_main(self):
        """Push the current GUI vertex positions to the SLAM process so the
        tracked mesh is edited in place (count must match on the other side)."""
        if self.q_vis2main is None or self.triangles_gui is None:
            return
        try:
            pkt = Packet_vis2main()
            pkt.edited_vertices = (
                self.triangles_gui.vertices.detach().cpu().numpy().astype(np.float32)
            )
            self.q_vis2main.put_nowait(pkt)
        except Exception:
            pass

    def _send_pause_to_main(self, paused):
        """Pause/resume the SLAM loop. On pause, main replies with an edit_base
        packet so the GUI edits the exact current tracked mesh (1:1 indices)."""
        if self.q_vis2main is None:
            return
        try:
            pkt = Packet_vis2main()
            if paused:
                pkt.flag_pause = True
            else:
                pkt.flag_resume = True
            self.q_vis2main.put_nowait(pkt)
        except Exception:
            pass

    def _on_deform_toggle(self, enabled):
        self._deform_mode = enabled
        if enabled:
            # Snapshot current vertices so Reset Vertices can restore them
            if self.triangles_gui is not None:
                self._original_verts = self.triangles_gui.vertices.detach().cpu().clone()
            self.widget3d.set_on_mouse(self._on_mouse_deform)
            self._deform_status.text = "Left-click + drag on mesh"
            # Editing the live map: pause SLAM so the mesh can't change under the
            # edit. Main sends an edit_base packet to sync the GUI's vertices 1:1.
            if self._edit_live_map:
                self._send_pause_to_main(True)
        else:
            self.widget3d.set_on_mouse(
                lambda _: gui.Widget.EventCallbackResult.IGNORED
            )
            self._drag_state = None
            self._deform_status.text = ""
            if self._edit_live_map:
                self._send_vertices_to_main()   # commit final positions
                self._send_pause_to_main(False)  # resume SLAM on the edited mesh

    def _on_mouse_deform(self, event):
        if not self._deform_mode or self.triangles_gui is None:
            return gui.Widget.EventCallbackResult.IGNORED

        if (event.type == gui.MouseEvent.Type.BUTTON_DOWN
                and event.buttons & int(gui.MouseButton.LEFT)):
            hit = self._raycast(event.x, event.y)
            if hit is not None:
                # Cancel any running elastic animation
                self._elastic_anim = None
                # Snapshot positions before this drag
                pre_drag_verts = self.triangles_gui.vertices.detach().clone()
                self._drag_state = {
                    "hit_point": hit,
                    "prev_x": event.x,
                    "prev_y": event.y,
                    "precomp": None,
                    "pre_drag_verts": pre_drag_verts,
                    "component_mask": None,
                }
                if self._move_component_chbox.checked:
                    self._drag_state["component_mask"] = self._find_connected_component(hit)
                elif self._connectivity_chbox.checked:
                    self._drag_state["precomp"] = self._precompute_deform_region(hit)
                self._deform_status.text = "Dragging…"
                return gui.Widget.EventCallbackResult.CONSUMED
            return gui.Widget.EventCallbackResult.IGNORED

        if event.type == gui.MouseEvent.Type.DRAG and self._drag_state is not None:
            dx = event.x - self._drag_state["prev_x"]
            dy = event.y - self._drag_state["prev_y"]
            if dx != 0 or dy != 0:
                self._apply_deform(self._drag_state["hit_point"], dx, dy)
                self._drag_state["prev_x"] = event.x
                self._drag_state["prev_y"] = event.y
            return gui.Widget.EventCallbackResult.CONSUMED

        if event.type == gui.MouseEvent.Type.BUTTON_UP:
            was_dragging = self._drag_state is not None
            if was_dragging and self._elastic_chbox.checked:
                # Start elastic wobble animation back to pre-drag position
                pre_drag = self._drag_state["pre_drag_verts"]
                current = self.triangles_gui.vertices.detach().clone()
                displacement = current - pre_drag
                if displacement.abs().max() > 1e-7:
                    self._elastic_anim = {
                        "rest_verts": pre_drag,
                        "displacement": displacement,
                        "t0": time.time(),
                        "frequency": self._elastic_freq.double_value,
                        "decay": self._elastic_decay.double_value,
                    }
            elif was_dragging and self._edit_live_map:
                # Non-elastic edit persists: push it to the tracked SLAM mesh.
                self._send_vertices_to_main()
            if was_dragging:
                self._physics_mesh_dirty = True  # geometry moved
                self._refresh_physics_mesh()  # balls collide with the deformed mesh
            self._drag_state = None
            if was_dragging:
                self._deform_status.text = "Left-click + drag on mesh"
            return (gui.Widget.EventCallbackResult.CONSUMED if was_dragging
                    else gui.Widget.EventCallbackResult.IGNORED)

        return gui.Widget.EventCallbackResult.IGNORED

    def _pixel_to_ray(self, px, py):
        """Return (origin, direction) ray in world space for a window pixel."""
        view_mat = self.widget3d.scene.camera.get_view_matrix()
        T_w2c = cv_gl @ view_mat
        C2W   = np.linalg.inv(T_w2c)
        ray_origin = C2W[:3, 3].astype(np.float32)

        frame = self.widget3d.frame
        lx = px - frame.x
        ly = py - frame.y
        w, h = float(frame.width), float(frame.height)

        vfov = self.widget3d.scene.camera.get_field_of_view()
        fy   = h / (2.0 * np.tan(np.radians(vfov) / 2.0))
        cx, cy = w / 2.0, h / 2.0

        dir_cam = np.array([(lx - cx) / fy, (ly - cy) / fy, 1.0], dtype=np.float32)
        dir_cam /= np.linalg.norm(dir_cam)
        dir_world = (C2W[:3, :3] @ dir_cam).astype(np.float32)
        dir_world /= np.linalg.norm(dir_world)
        return ray_origin, dir_world

    def _raycast(self, px, py, return_normal=False):
        """Cast a ray; return hit world point or None.

        If return_normal=True, returns (hit_point, normal) or (None, None).
        """
        import open3d as _o3d
        verts = self.triangles_gui.vertices.detach().cpu().numpy().astype(np.float64)
        faces = self.triangles_gui._triangle_indices.cpu().numpy().astype(np.int32)

        o3d_mesh = _o3d.geometry.TriangleMesh()
        o3d_mesh.vertices  = _o3d.utility.Vector3dVector(verts)
        o3d_mesh.triangles = _o3d.utility.Vector3iVector(faces)
        mesh_t = _o3d.t.geometry.TriangleMesh.from_legacy(o3d_mesh)
        scene  = _o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh_t)

        origin, direction = self._pixel_to_ray(px, py)
        rays   = _o3d.core.Tensor(np.array([[*origin, *direction]], dtype=np.float32))
        result = scene.cast_rays(rays)
        t_hit  = float(result["t_hit"].numpy()[0])
        if np.isfinite(t_hit):
            hit = origin + t_hit * direction
            if return_normal:
                normal = result["primitive_normals"].numpy()[0].astype(np.float64)
                return hit, normal
            return hit
        if return_normal:
            return None, None
        return None

    def _find_connected_component(self, hit_point):
        """Flood-fill from the nearest vertex to find its connected component."""
        verts = self.triangles_gui.vertices
        faces = self.triangles_gui._triangle_indices.long()
        V = verts.shape[0]
        device = verts.device

        # Build adjacency
        i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]
        src = torch.cat([i0, i1, i1, i2, i2, i0])
        dst = torch.cat([i1, i0, i2, i1, i0, i2])

        # Find seed
        hit_t = torch.tensor(hit_point, dtype=torch.float32, device=device)
        seed_idx = (verts - hit_t).norm(dim=1).argmin().item()

        # BFS via scatter: propagate reachability
        reached = torch.zeros(V, dtype=torch.bool, device=device)
        reached[seed_idx] = True
        for _ in range(200):  # enough hops for any component
            new_reached = reached.clone()
            new_reached[dst[reached[src]]] = True
            if (new_reached == reached).all():
                break
            reached = new_reached

        return reached

    def _precompute_deform_region(self, hit_point):
        """Pre-compute adjacency and vertex roles for connectivity-aware deformation."""
        verts = self.triangles_gui.vertices
        faces = self.triangles_gui._triangle_indices.long()
        V = verts.shape[0]
        device = verts.device
        radius = self._brush_slider.double_value

        # Build undirected edge list (same pattern as _on_smooth)
        i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]
        src = torch.cat([i0, i1, i1, i2, i2, i0])
        dst = torch.cat([i1, i0, i2, i1, i0, i2])
        ones = torch.ones(len(src), 1, device=device)

        # Find nearest vertex to hit point
        hit_t = torch.tensor(hit_point, dtype=torch.float32, device=device)
        seed_idx = (verts - hit_t).norm(dim=1).argmin().item()

        # Bellman-Ford: approximate geodesic distance using edge lengths
        edge_lengths = (verts[src] - verts[dst]).norm(dim=1)
        geo_dist = torch.full((V,), float('inf'), device=device)
        geo_dist[seed_idx] = 0.0

        max_hops = 50
        for _ in range(max_hops):
            candidate = geo_dist[src] + edge_lengths
            new_geo = geo_dist.clone()
            new_geo.scatter_reduce_(0, dst, candidate, reduce='amin')
            if (new_geo == geo_dist).all():
                break
            geo_dist = new_geo

        # Classify vertices: handle (inner) / free (outer) / fixed
        inner_radius = radius * 0.3
        handle_mask = geo_dist <= inner_radius
        free_mask = (geo_dist > inner_radius) & (geo_dist <= radius)

        # Gaussian falloff weight for handle vertices
        handle_weight = torch.zeros(V, device=device)
        handle_weight[handle_mask] = torch.exp(
            -geo_dist[handle_mask] ** 2 / (2.0 * (inner_radius * 0.5) ** 2)
        )

        # Pre-compute neighbor count
        nb_cnt = torch.zeros(V, 1, device=device)
        nb_cnt.scatter_add_(0, dst.unsqueeze(1), ones)

        return {
            "src": src,
            "dst": dst,
            "nb_cnt": nb_cnt,
            "handle_mask": handle_mask,
            "free_mask": free_mask,
            "handle_weight": handle_weight,
        }

    def _apply_deform(self, hit_point, dx_pixels, dy_pixels):
        """Displace vertices using connectivity-aware diffusion or Euclidean falloff."""
        view_mat = self.widget3d.scene.camera.get_view_matrix()
        T_w2c    = cv_gl @ view_mat
        C2W      = np.linalg.inv(T_w2c)

        depth = float(np.linalg.norm(hit_point - C2W[:3, 3]))
        right = C2W[:3, 0]
        down  = C2W[:3, 1]

        vfov = self.widget3d.scene.camera.get_field_of_view()
        fy   = self.window.size.height / (2.0 * np.tan(np.radians(vfov) / 2.0))

        delta = ((dx_pixels / fy) * depth * right
               + (dy_pixels / fy) * depth * down).astype(np.float32)

        verts  = self.triangles_gui.vertices
        device = verts.device
        delta_t = torch.tensor(delta, dtype=torch.float32, device=device)

        # Move entire connected component - uniform translation, no falloff
        comp_mask = self._drag_state.get("component_mask") if self._drag_state else None
        if comp_mask is not None:
            with torch.no_grad():
                verts.data[comp_mask] += delta_t
            self.triangles_gui._triangles = verts[self.triangles_gui._triangle_indices]
            return

        precomp = self._drag_state.get("precomp") if self._drag_state else None

        if precomp is None:
            # Euclidean falloff (original behaviour)
            radius = self._brush_slider.double_value
            hit_t  = torch.tensor(hit_point, dtype=torch.float32, device=device)
            dist = (verts - hit_t).norm(dim=1)
            mask = dist < radius
            if not mask.any():
                return
            weight = torch.exp(-dist[mask] ** 2 / (2.0 * (radius * 0.5) ** 2))
            with torch.no_grad():
                verts.data[mask] += weight.unsqueeze(1) * delta_t
        else:
            # Connectivity-aware displacement diffusion
            V = verts.shape[0]
            src = precomp["src"]
            dst = precomp["dst"]
            nb_cnt = precomp["nb_cnt"]
            handle_mask = precomp["handle_mask"]
            free_mask = precomp["free_mask"]
            handle_weight = precomp["handle_weight"]

            if not handle_mask.any():
                return

            # Initial displacement: handle vertices get weighted delta
            disp = torch.zeros(V, 3, device=device)
            disp[handle_mask] = handle_weight[handle_mask].unsqueeze(1) * delta_t

            # Diffuse displacement through mesh connectivity
            n_iters = self._diffusion_iters.int_value
            for _ in range(n_iters):
                nb_sum = torch.zeros_like(disp)
                nb_sum.scatter_add_(0, dst.unsqueeze(1).expand(-1, 3), disp[src])
                nb_avg = nb_sum / nb_cnt.clamp(min=1)
                # Free vertices blend toward neighbor average; handles stay pinned
                disp[free_mask] = 0.7 * nb_avg[free_mask] + 0.3 * disp[free_mask]

            with torch.no_grad():
                active = handle_mask | free_mask
                verts.data[active] += disp[active]

        self.triangles_gui._triangles = verts[self.triangles_gui._triangle_indices]

    def _on_smooth(self):
        if self.triangles_gui is None:
            return
        iters = self._smooth_iters.int_value
        verts = self.triangles_gui.vertices
        faces = self.triangles_gui._triangle_indices.long()
        V = verts.shape[0]

        i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]
        src = torch.cat([i0, i1, i1, i2, i2, i0])
        dst = torch.cat([i1, i0, i2, i1, i0, i2])
        ones = torch.ones(len(src), 1, device=verts.device)

        with torch.no_grad():
            v = verts.clone()
            for _ in range(iters):
                nb_sum = torch.zeros_like(v)
                nb_cnt = torch.zeros(V, 1, device=v.device)
                nb_sum.scatter_add_(0, dst.unsqueeze(1).expand(-1, 3), v[src])
                nb_cnt.scatter_add_(0, dst.unsqueeze(1), ones)
                laplacian = nb_sum / nb_cnt.clamp(min=1)
                v = v + 0.5 * (laplacian - v)
            verts.data[:] = v
            self.triangles_gui._triangles = verts[self.triangles_gui._triangle_indices]

    def _on_reset_verts(self):
        if self._original_verts is None or self.triangles_gui is None:
            return
        with torch.no_grad():
            self.triangles_gui.vertices.data[:] = (
                self._original_verts.to(self.triangles_gui.vertices.device)
            )
            self.triangles_gui._triangles = (
                self.triangles_gui.vertices[self.triangles_gui._triangle_indices]
            )
        self._physics_mesh_dirty = True
        self._refresh_physics_mesh()  # existing balls re-settle onto the reset mesh
        if self._edit_live_map:
            self._send_vertices_to_main()

    def _on_sigma_changed(self, log_sigma):
        import math
        self._sigma_label.text = f"σ = {math.exp(log_sigma):.4f}"
        if self.triangles_gui is not None:
            self.triangles_gui._sigma_val = log_sigma

    def _on_random_colors(self):
        if self.triangles_gui is None:
            return
        tg = self.triangles_gui
        # Save original so we can restore
        self._saved_features_dc = tg._features_dc.detach().cpu().clone()

        faces  = tg._triangle_indices.long()   # (F, 3)
        F      = faces.shape[0]
        V      = tg.vertices.shape[0]
        device = tg.vertices.device

        # One random RGB per triangle → convert to SH DC coefficient
        # rendered colour = SH_C0 * f_dc + 0.5,  SH_C0 = 0.28209479177387814
        SH_C0 = 0.28209479177387814
        rand_rgb = torch.rand(F, 3, device=device)          # (F, 3) in [0, 1]
        rand_fdc = (rand_rgb - 0.5) / SH_C0                # (F, 3)

        # Scatter to vertices: each vertex gets the colour of its last-written triangle
        new_fdc = torch.zeros(V, 3, device=device)
        face_flat  = faces.reshape(-1)                      # (3F,)
        color_flat = rand_fdc.repeat_interleave(3, dim=0)  # (3F, 3)
        new_fdc[face_flat] = color_flat
        tg._features_dc = new_fdc.unsqueeze(1)             # (V, 1, 3)

    def _on_restore_colors(self):
        if self._saved_features_dc is None or self.triangles_gui is None:
            return
        self.triangles_gui._features_dc = (
            self._saved_features_dc.to(self.triangles_gui.vertices.device)
        )

    def _set_mode(self, mode):
        self._render_mode = mode

    # ------------------------------------------------------------------
    # Data reception
    # ------------------------------------------------------------------

    def _receive_data(self):
        if self.q_main2vis is None:
            return
        pkt: TrianglePacket = get_latest_queue(self.q_main2vis)
        if pkt is None:
            return

        if pkt.finish:
            # Auto-save any in-progress recording before quitting
            if self._recording and self._record_frames:
                self._recording = False
                self._auto_save_video()
            self.process_finished = True
            return

        # Update mesh proxy (skip if deform mode is active to preserve edits)
        # Normally ingest the latest mesh, but skip while deforming so edits are
        # not clobbered - EXCEPT an edit_base packet, which is the exact mesh we
        # must edit against (sent when SLAM pauses for a live edit).
        is_base = getattr(pkt, "edit_base", False)
        if pkt.has_triangles and (not self._deform_mode or is_base):
            if self.triangles_gui is None:
                self.triangles_gui = GUITriangles(pkt, device="cuda")
                if not self._sigma_slider_initialized:
                    import math
                    log_sigma = self.triangles_gui._sigma_val
                    self._sigma_slider.double_value = float(log_sigma)
                    self._sigma_label.text = f"σ = {math.exp(log_sigma):.4f}"
                    self._sigma_slider_initialized = True
            else:
                prev_v = self.triangles_gui.vertices.shape[0]
                self.triangles_gui._update(pkt, device="cuda")
                # Re-apply user's sigma choice after packet overwrites it
                self.triangles_gui._sigma_val = self._sigma_slider.double_value
                # A vertex-count change invalidates any in-flight elastic animation.
                if self.triangles_gui.vertices.shape[0] != prev_v:
                    self._elastic_anim = None
            n = self.triangles_gui._triangle_indices.shape[0]
            self.tri_count_label.text = f"Triangles: {n:,}"
            # Rendered mesh changed -> pybullet collision mesh is now stale.
            self._physics_mesh_dirty = True
            # Adopt this as the editing base so Reset restores to it.
            if is_base:
                self._original_verts = self.triangles_gui.vertices.detach().cpu().clone()

        # Update TSDF comparison mesh if present
        if getattr(pkt, "tsdf_mesh", None) is not None:
            self._update_tsdf_mesh(pkt.tsdf_mesh)

        # Add / update keyframe frustums
        for cam_dict in pkt.keyframes:
            self._add_frustum(cam_dict, color=(0.0, 1.0, 0.0))

        # Current frame (cyan)
        if pkt.current_frame is not None:
            self._add_frustum(pkt.current_frame, color=(0.0, 1.0, 1.0), current=True)

            # Keep the current frame + its GT image for the preview panels.
            self._current_cam_dict = pkt.current_frame
            gt_img = pkt.current_frame.get("gt_image")
            if gt_img is not None:
                self._current_gt_img = gt_img
                aspect = gt_img.shape[0] / max(1, gt_img.shape[1])
                if abs(aspect - self._preview_aspect) > 1e-3:
                    self._preview_aspect = aspect
                    self.window.set_needs_layout()

            # Track trajectory and collect poses for GT alignment
            C2W = _cam_c2w(pkt.current_frame)
            if self._first_cam_c2w is None:
                self._first_cam_c2w = C2W.copy()  # gravity prior (camera-down)
            self._trajectory_points.append(C2W[:3, 3].copy())
            self._update_trajectory()

            uid = pkt.current_frame["uid"]
            self._est_poses[uid] = C2W
            if "R_gt" in pkt.current_frame:
                R_gt = pkt.current_frame["R_gt"]
                T_gt = pkt.current_frame["T_gt"]
                from mesh_splatting.utils.graphics_utils import getWorld2View2
                import torch
                T_w2c_gt = getWorld2View2(torch.from_numpy(R_gt).float(),
                                          torch.from_numpy(T_gt).float()).numpy()
                self._gt_poses[uid] = np.linalg.inv(T_w2c_gt)

            # Refresh GT overlay if visible
            if self._gt_poses_visible:
                self._rebuild_gt_overlay()

            if self.follow_cam:
                self._follow(pkt.current_frame)

    def _add_frustum(self, cam_dict, color=(0, 1, 0), current=False):
        uid  = cam_dict["uid"]
        name = f"cam_{uid}" if not current else "cam_current"

        C2W = _cam_c2w(cam_dict)
        ls  = _create_frustum_lineset(color=color, size=0.02)

        # Transform into world space
        pts   = np.asarray(ls.points)
        pts_h = np.hstack([pts, np.ones((len(pts), 1))])
        pts_w = (C2W @ pts_h.T)[:3].T
        ls.points = o3d.utility.Vector3dVector(pts_w)

        try:
            if name in self.frustum_dict:
                self.widget3d.scene.remove_geometry(name)
            self.widget3d.scene.add_geometry(name, ls, self.lit)
            self.frustum_dict[name] = True
            self.widget3d.scene.show_geometry(name, self.cameras_chbox.checked)
        except Exception:
            pass

    def _follow(self, cam_dict):
        C2W = _cam_c2w(cam_dict)
        forward = C2W[:3, 2]               # camera forward (OpenCV +Z)
        up      = -C2W[:3, 1]              # OpenCV y-down → up is -Y
        cam_pos = C2W[:3, 3]
        dist    = self._follow_dist.double_value
        # Pull the viewing eye back behind the frustum
        eye    = cam_pos - forward * dist
        centre = cam_pos + forward * 0.3   # look ahead of the frustum
        self.widget3d.scene.camera.look_at(centre, eye, up)

    def _update_trajectory(self):
        """Draw/update a line strip connecting consecutive camera centers."""
        name = "_trajectory_line"
        pts = self._trajectory_points
        show = self._trajectory_chbox.checked and len(pts) >= 2
        if not show:
            if name in self.frustum_dict:
                self.widget3d.scene.show_geometry(name, False)
            return
        if len(pts) == self._trajectory_len_drawn:
            return
        self._trajectory_len_drawn = len(pts)
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(np.array(pts, dtype=np.float64))
        lines = [[i, i + 1] for i in range(len(pts) - 1)]
        ls.lines = o3d.utility.Vector2iVector(lines)
        ls.colors = o3d.utility.Vector3dVector(
            np.tile([1.0, 1.0, 0.0], (len(lines), 1))  # yellow
        )
        if name in self.frustum_dict:
            self.widget3d.scene.remove_geometry(name)
        self.widget3d.scene.add_geometry(name, ls, self.lit)
        self.frustum_dict[name] = True

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _get_current_gui_cam(self):
        """Build a GUICamera from the current Open3D viewport."""
        view = self.widget3d.scene.camera.get_view_matrix()  # (4,4) numpy, OpenGL
        vfov = self.widget3d.scene.camera.get_field_of_view()
        h = int(self.window.size.height)
        w = int(self.widget3d_width)
        return GUICamera(view, vfov, h, w, device="cuda")

    def _render(self):
        if self.triangles_gui is None or self.pipe is None:
            return None
        global triangle_render
        if triangle_render is None:
            from mesh_splatting.triangle_renderer import render
            triangle_render = render
        cam = self._get_current_gui_cam()
        with torch.no_grad():
            result = triangle_render(cam, self.triangles_gui, self.pipe, self.background)
        return result, cam

    def _result_to_o3d(self, result, cam):
        import imgviz
        import torch.nn.functional as F

        if self._render_mode == "depth":
            depth = result["expected_depth"][0].detach().cpu().numpy()
            dmax  = float(depth.max()) or 1.0
            rgb   = imgviz.depth2rgb(depth, min_value=0.0, max_value=dmax, colormap="jet")
            return o3d.geometry.Image(rgb.astype(np.uint8))

        if self._render_mode == "opacity":
            alpha = result["rend_alpha"][0].detach().cpu().numpy()
            rgb   = imgviz.depth2rgb(alpha, min_value=0.0, max_value=1.0, colormap="jet")
            return o3d.geometry.Image(rgb.astype(np.uint8))

        if self._render_mode == "normal":
            # Match rerun convention: negate world-space normals, rotate to view space,
            # per-pixel normalise, then map [-1,1] → [0,255].
            # world_view_transform = T_w2c.T, [:3,:3] = R_cw, .T = R_wc (world→cam).
            n = -result["rend_normal"].detach()                     # (3,H,W) world, negated
            R_wc = cam.world_view_transform[:3, :3].T              # (3,3) world→cam
            n_cam = (R_wc @ n.reshape(3, -1)).reshape(3, *n.shape[1:])  # (3,H,W)
            n_cam = F.normalize(n_cam.permute(1, 2, 0), dim=-1).permute(2, 0, 1)
            img = ((n_cam + 1.0) * 0.5).clamp(0, 1)
            img = (img * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
            return o3d.geometry.Image(img)

        if self._render_mode == "surf_normal":
            # Match rerun convention: same negate + rotate + normalise pipeline.
            n = -result["surf_normal"].detach()                     # (3,H,W) negated
            R_wc = cam.world_view_transform[:3, :3].T
            n_cam = (R_wc @ n.reshape(3, -1)).reshape(3, *n.shape[1:])
            n_cam = F.normalize(n_cam.permute(1, 2, 0), dim=-1).permute(2, 0, 1)
            img = ((n_cam + 1.0) * 0.5).clamp(0, 1)
            img = (img * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
            return o3d.geometry.Image(img)

        # Default: RGB
        img = (
            result["render"].clamp(0, 1) * 255
        ).byte().permute(1, 2, 0).contiguous().cpu().numpy()
        return o3d.geometry.Image(img)

    def _result_to_o3d_normal(self, result, cam, mode):
        """Return normal or surf_normal as o3d Image for saving; mode in ('normal', 'surf_normal')."""
        import torch.nn.functional as F
        key = "rend_normal" if mode == "normal" else "surf_normal"
        if key not in result:
            return None
        n = -result[key].detach()
        R_wc = cam.world_view_transform[:3, :3].T
        n_cam = (R_wc @ n.reshape(3, -1)).reshape(3, *n.shape[1:])
        n_cam = F.normalize(n_cam.permute(1, 2, 0), dim=-1).permute(2, 0, 1)
        img = ((n_cam + 1.0) * 0.5).clamp(0, 1)
        img = (img * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
        return o3d.geometry.Image(img)

    def _capture_scene_view(self):
        """Capture the 3D widget view (includes frustums if visible). Returns o3d.Image or None."""
        try:
            w = int(self.widget3d_width)
            h = int(self.window.size.height)
            return gui.Application.instance.render_to_image(self.widget3d.scene, w, h)
        except Exception:
            pass
        return None

    def _render_gui(self):
        ts = time.perf_counter()
        if ts - self.ts_prev < 1.0 / self.RENDER_FPS:
            return
        self.ts_prev = ts

        # When viewing TSDF or O3D mesh, skip rasterizer - Open3D renders directly
        if self._tsdf_view_active and self._tsdf_mesh is not None:
            self.widget3d.scene.set_background([0, 0, 0, 1])
            return
        if self._mesh_view_active:
            self._update_o3d_mesh()
            self.widget3d.scene.set_background([0, 0, 0, 1])
            return

        # Temporarily inject physics objects as triangles so the rasteriser
        # draws them with the scene, then restore the original tensors.
        physics_extended = self._extend_with_physics()
        out = self._render()
        if physics_extended:
            self._restore_physics()
        if out is None:
            return
        result, cam = out
        img = self._result_to_o3d(result, cam)
        self.widget3d.scene.set_background([0, 0, 0, 1], img)

        if self._recording:
            # Capture the full scene view (includes frustums, trajectory, etc.)
            scene_img = self._capture_scene_view()
            if scene_img is not None:
                arr = np.asarray(scene_img)
                if arr.ndim == 3:
                    self._record_frames.append(arr[:, :, :3].copy())

    # ------------------------------------------------------------------
    # Update thread
    # ------------------------------------------------------------------

    def _tick_elastic(self):
        """Advance damped oscillation animation if active."""
        anim = self._elastic_anim
        if anim is None:
            return
        # The mesh may have been replaced (densify / commit / re-sync) since the
        # animation started; if vertex counts no longer match, cancel it.
        if (self.triangles_gui is None
                or anim["rest_verts"].shape[0] != self.triangles_gui.vertices.shape[0]):
            self._elastic_anim = None
            return
        import math
        t = time.time() - anim["t0"]
        envelope = math.exp(-anim["decay"] * t)
        if envelope < 0.005:
            # Animation done - snap to rest position
            with torch.no_grad():
                self.triangles_gui.vertices.data[:] = anim["rest_verts"].to(
                    self.triangles_gui.vertices.device
                )
                self.triangles_gui._triangles = (
                    self.triangles_gui.vertices[self.triangles_gui._triangle_indices]
                )
            self._elastic_anim = None
            self._physics_mesh_dirty = True  # vertices returned to rest
            self._refresh_physics_mesh()
            # Commit the settled (rest) pose to the tracked map.
            if self._edit_live_map and self._deform_mode:
                self._send_vertices_to_main()
            return
        scale = envelope * math.cos(2.0 * math.pi * anim["frequency"] * t)
        with torch.no_grad():
            device = self.triangles_gui.vertices.device
            self.triangles_gui.vertices.data[:] = (
                anim["rest_verts"].to(device) + scale * anim["displacement"].to(device)
            )
            self.triangles_gui._triangles = (
                self.triangles_gui.vertices[self.triangles_gui._triangle_indices]
            )

    def _scene_update(self):
        self._tick_elastic()
        self._tick_physics()
        self._receive_data()
        try:
            self._render_gui()
        except Exception as e:
            print(f"GUI render error (non-fatal): {e}")
        self._update_preview_panels()

    def _update_preview_panels(self):
        """Refresh the current-render / GT / difference thumbnails."""
        if not self._previews_chbox.checked:
            return
        if (self.triangles_gui is None or self.pipe is None
                or self._current_cam_dict is None or self._current_gt_img is None):
            return

        ts = time.perf_counter()
        if ts - self._preview_ts_prev < 1.0 / self._preview_fps:
            return
        self._preview_ts_prev = ts

        global triangle_render
        if triangle_render is None:
            from mesh_splatting.triangle_renderer import render
            triangle_render = render

        try:
            cam = PreviewCamera(self._current_cam_dict, self._preview_width, device="cuda")
            # Include live (un-committed) physics objects so the preview matches
            # the main view; restore the mesh straight after.
            physics_extended = self._extend_with_physics()
            with torch.no_grad():
                result = triangle_render(cam, self.triangles_gui, self.pipe, self.background)
            if physics_extended:
                self._restore_physics()
            # Apply the same learned autoexposure the optimisation applies to the
            # render (exp(a)*render + b) so Current and the raw GT are comparable.
            render_t = result["render"]
            exp_a = self._current_cam_dict.get("exposure_a")
            exp_b = self._current_cam_dict.get("exposure_b")
            if exp_a is not None and exp_b is not None:
                render_t = math.exp(exp_a) * render_t + exp_b
            render_rgb = (
                render_t.clamp(0, 1) * 255
            ).byte().permute(1, 2, 0).contiguous().cpu().numpy()
        except Exception as e:
            print(f"Preview render failed: {e}")
            self._restore_physics()
            return

        gt_rgb = self._current_gt_img
        # Match GT to the rendered size if they differ (e.g. rounding).
        if gt_rgb.shape[:2] != render_rgb.shape[:2]:
            import cv2
            gt_rgb = cv2.resize(gt_rgb, (render_rgb.shape[1], render_rgb.shape[0]),
                                interpolation=cv2.INTER_AREA)

        diff = np.abs(render_rgb.astype(np.int16) - gt_rgb.astype(np.int16))
        diff = diff.clip(0, 255).astype(np.uint8)

        # The GT is already undistorted (same space as the render); undistortion
        # leaves black invalid borders. The SLAM loss ignores these via
        # rgb_pixel_mask, so mask them here too: zero the diff where the GT is
        # near-black, otherwise the invalid border dominates the difference view.
        valid = gt_rgb.sum(axis=2) > 8  # ~ rgb_boundary_threshold on 0-255 sum
        diff[~valid] = 0

        for img_w, arr in zip(self._preview_widgets, (render_rgb, gt_rgb, diff)):
            img_w.update_image(o3d.geometry.Image(np.ascontiguousarray(arr)))

    def _update_thread(self):
        while True:
            time.sleep(0.01)
            self.step += 1
            if self.process_finished:
                gui.Application.instance.quit()
                break

            if self._pending_update:
                continue

            def _update():
                try:
                    if self.step % 3 == 0:
                        self._scene_update()
                    if self.step >= int(1e9):
                        self.step = 0
                finally:
                    self._pending_update = False

            self._pending_update = True
            gui.Application.instance.post_to_main_thread(self.window, _update)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run(params_gui=None):
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    SLAM_GUI(params_gui)
    app.run()


def main():
    run()


if __name__ == "__main__":
    main()
