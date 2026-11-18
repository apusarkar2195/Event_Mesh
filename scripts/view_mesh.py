#!/usr/bin/env python3
"""
Load a saved Triangle SLAM mesh and view it in the interactive GUI.

Usage
-----
  python scripts/view_mesh.py <iteration_dir>

Example
-------
  python scripts/view_mesh.py results/tum_.../mesh/iteration_2515
"""

import argparse
import math
import os
import sys
import threading
from pathlib import Path

import numpy as np
import torch

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "mesh_splatting"))

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import rerun as rr

from meshslam.gui.slam_gui import SLAM_GUI
from meshslam.gui.gui_utils import GUITriangles, cv_gl


def _make_packet(orig_gui, new_faces_np):
    """Build a fake packet for GUITriangles using orig's vertex data and new face indices."""
    class _Pkt:
        pass
    pkt = _Pkt()
    pkt.vertices         = orig_gui.vertices.cpu()
    pkt.triangle_indices = torch.as_tensor(
        np.asarray(new_faces_np, dtype=np.int64)
    ).contiguous().to(torch.int32)
    pkt.vertex_weight    = orig_gui._vertex_weight.cpu()
    pkt.features_dc      = orig_gui._features_dc.cpu()
    pkt.features_rest    = orig_gui._features_rest.cpu()
    pkt._sigma           = orig_gui._sigma_val
    pkt.opacity_floor    = orig_gui.opacity_floor
    pkt.active_sh_degree = orig_gui.active_sh_degree
    pkt.max_sh_degree    = orig_gui.max_sh_degree
    pkt.scaling          = orig_gui.scaling
    return pkt


class RefineCam:
    """Minimal camera for feature refinement — wraps saved R, T with a GT image."""

    def __init__(self, uid, R_np, T_np, image, fx, fy, cx, cy, H, W, device="cuda"):
        import math as _math
        from mesh_splatting.utils.graphics_utils import getWorld2View2, getProjectionMatrix2

        self.uid = uid
        self.original_image   = image.to(device)
        self.image_height, self.image_width = H, W
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.FoVy = 2 * _math.atan(H / (2 * fy))
        self.FoVx = 2 * _math.atan(W / (2 * fx))

        R_t  = torch.from_numpy(R_np)
        T_t  = torch.from_numpy(T_np)
        T_w2c = getWorld2View2(R_t, T_t).numpy()               # (4,4) numpy
        self.world_view_transform = torch.from_numpy(T_w2c.T).float().to(device)

        proj = getProjectionMatrix2(znear=0.01, zfar=100.0,
                                    fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H)
        self.projection_matrix    = proj.T.to(device)
        self.full_proj_transform  = (self.world_view_transform.unsqueeze(0)
                                     .bmm(self.projection_matrix.unsqueeze(0)).squeeze(0))
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        self.R = torch.from_numpy(R_np).float().to(device)
        self.T = torch.from_numpy(T_np).float().to(device)
        self.cam_rot_delta   = torch.zeros(3, device=device)
        self.cam_trans_delta = torch.zeros(3, device=device)
        self.exposure_a      = torch.zeros(1, device=device)
        self.exposure_b      = torch.zeros(1, device=device)


class MeshViewer(SLAM_GUI):
    """SLAM_GUI subclass that preloads a saved mesh instead of receiving live data."""

    def __init__(self, iter_dir: str, frustum_size: float = 0.05):
        self._iter_dir = iter_dir
        self._frustum_size = frustum_size
        self._data_loaded = False
        self._original_triangles_gui = None
        self._delaunay_triangles_gui = None
        self._rdt_running = False
        self._deform_mode = False
        self._drag_state = None
        self._original_verts = None   # CPU clone saved at load for reset
        self._refine_running = False
        self._wireframe_on = False

        super().__init__(params_gui=None)

        from mesh_splatting.arguments import PipelineParams
        from argparse import ArgumentParser as AP
        self.pipe = PipelineParams(AP())
        self.background = torch.zeros(3, device="cuda")

    # ------------------------------------------------------------------
    # Widget overrides
    # ------------------------------------------------------------------

    def _init_widget(self):
        super()._init_widget()
        em = self.window.theme.font_size

        # --- Sigma slider ---
        self.panel.add_child(gui.Label("Sigma (log scale)"))
        self._sigma_slider = gui.Slider(gui.Slider.DOUBLE)
        self._sigma_slider.set_limits(-7.0, 2.0)
        self._sigma_slider.double_value = -0.693   # will be overwritten on load
        self._sigma_slider.set_on_value_changed(self._on_sigma_changed)
        self.panel.add_child(self._sigma_slider)
        self._sigma_label = gui.Label("σ = 0.500")
        self.panel.add_child(self._sigma_label)

        # --- Wireframe ---
        self._wire_chbox = gui.Checkbox("Wireframe")
        self._wire_chbox.checked = False
        self._wire_chbox.set_on_checked(self._on_wireframe_toggle)
        self.panel.add_child(self._wire_chbox)

        self._wire_color = gui.ColorEdit()
        self._wire_color.color_value = gui.Color(1.0, 1.0, 1.0)
        self._wire_color.set_on_value_changed(self._on_wire_color_changed)
        self.panel.add_child(self._wire_color)

        self.panel.add_child(gui.Label("Line width"))
        self._wire_width_slider = gui.Slider(gui.Slider.DOUBLE)
        self._wire_width_slider.set_limits(0.1, 10.0)
        self._wire_width_slider.double_value = 0.5
        self._wire_width_slider.set_on_value_changed(self._on_wire_width_changed)
        self.panel.add_child(self._wire_width_slider)

        # --- Restricted Delaunay ---
        self.panel.add_child(gui.Label("Retriangulation"))

        self._rdt_chbox = gui.Checkbox("Restricted Delaunay")
        self._rdt_chbox.checked = False
        self._rdt_chbox.set_on_checked(self._on_rdt_toggle)
        self.panel.add_child(self._rdt_chbox)

        self._rdt_status = gui.Label("")
        self.panel.add_child(self._rdt_status)

        # --- Mesh editing ---
        self.panel.add_child(gui.Label("Mesh Editing"))

        self._remove_chbox = gui.Checkbox("Remove Mode")
        self._remove_chbox.checked = False
        self._remove_chbox.set_on_checked(self._on_remove_toggle)
        self.panel.add_child(self._remove_chbox)

        self._remove_status = gui.Label("")
        self.panel.add_child(self._remove_status)

        self._deform_chbox = gui.Checkbox("Deform Mode")
        self._deform_chbox.checked = False
        self._deform_chbox.set_on_checked(self._on_deform_toggle)
        self.panel.add_child(self._deform_chbox)

        self.panel.add_child(gui.Label("Brush radius"))
        self._brush_slider = gui.Slider(gui.Slider.DOUBLE)
        self._brush_slider.set_limits(0.005, 1.0)
        self._brush_slider.double_value = 0.1
        self.panel.add_child(self._brush_slider)

        self._connectivity_chbox = gui.Checkbox("Connectivity-aware")
        self._connectivity_chbox.checked = True
        self.panel.add_child(self._connectivity_chbox)

        self.panel.add_child(gui.Label("Diffusion iters"))
        self._diffusion_iters = gui.Slider(gui.Slider.INT)
        self._diffusion_iters.set_limits(1, 50)
        self._diffusion_iters.int_value = 15
        self.panel.add_child(self._diffusion_iters)

        smooth_row = gui.Horiz(0.25 * em)
        self._smooth_btn = gui.Button("Laplacian Smooth")
        self._smooth_btn.set_on_clicked(self._on_smooth)
        self._smooth_iters = gui.Slider(gui.Slider.INT)
        self._smooth_iters.set_limits(1, 20)
        self._smooth_iters.int_value = 1
        smooth_row.add_child(self._smooth_btn)
        smooth_row.add_child(self._smooth_iters)
        self.panel.add_child(smooth_row)

        self._reset_verts_btn = gui.Button("Reset Vertices")
        self._reset_verts_btn.set_on_clicked(self._on_reset_verts)
        self.panel.add_child(self._reset_verts_btn)

        self._deform_status = gui.Label("")
        self.panel.add_child(self._deform_status)

        # --- Feature refinement ---
        self.panel.add_child(gui.Label("Feature Refinement"))

        refine_row = gui.Horiz(0.25 * em)
        self._refine_btn = gui.Button("Refine")
        self._refine_btn.set_on_clicked(self._on_refine)
        self._refine_iters = gui.Slider(gui.Slider.INT)
        self._refine_iters.set_limits(1, 200)
        self._refine_iters.int_value = 30
        refine_row.add_child(self._refine_btn)
        refine_row.add_child(self._refine_iters)
        self.panel.add_child(refine_row)

        self._refine_status = gui.Label("")
        self.panel.add_child(self._refine_status)

    # ------------------------------------------------------------------
    # Scene update (no live queue)
    # ------------------------------------------------------------------

    def _scene_update(self):
        if not self._data_loaded:
            self._load_data()
            self._data_loaded = True
        self._render_gui()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self):
        """Load mesh state and keyframe frustums (called once on main thread)."""
        pc_path = os.path.join(self._iter_dir, "point_cloud_state_dict.pt")
        kf_path = os.path.join(self._iter_dir, "keyframes_state.pt")

        # --- Mesh ---
        print(f"Loading mesh from {pc_path}")
        state = torch.load(pc_path, map_location="cpu")

        class _Pkt:
            pass

        pkt = _Pkt()
        pkt.vertices         = state["triangles_points"]
        pkt.triangle_indices = state["_triangle_indices"]
        pkt.vertex_weight    = state["vertex_weight"]
        pkt.features_dc      = state["features_dc"]
        pkt.features_rest    = state["features_rest"]
        pkt._sigma           = float(state["sigma"])
        pkt.opacity_floor    = 0.999
        pkt.active_sh_degree = int(state["active_sh_degree"])
        pkt.max_sh_degree    = 0
        pkt.scaling          = 1

        self.triangles_gui = GUITriangles(pkt, device="cuda")
        self._original_triangles_gui = self.triangles_gui
        self._original_verts = self.triangles_gui.vertices.detach().cpu().clone()

        n_tri = self.triangles_gui._triangle_indices.shape[0]
        n_vert = self.triangles_gui.vertices.shape[0]
        self.tri_count_label.text = f"Triangles: {n_tri:,}"
        print(f"  {n_vert:,} vertices  {n_tri:,} triangles")

        log_sigma = self.triangles_gui._sigma_val
        self._sigma_slider.double_value = float(log_sigma)
        self._sigma_label.text = f"σ = {math.exp(log_sigma):.4f}"

        # --- Keyframe frustums ---
        if os.path.exists(kf_path):
            kf_state = torch.load(kf_path, map_location="cpu")
            keyframes = kf_state.get("keyframes", [])
            print(f"Adding {len(keyframes)} keyframe frustums …")
            for kf in keyframes:
                cam_dict = {
                    "uid": int(kf["uid"]),
                    "R":   kf["R"].numpy().astype(np.float32),
                    "T":   kf["T"].numpy().astype(np.float32),
                }
                self._add_frustum_sized(cam_dict, color=(0.0, 1.0, 0.0))
            print("  Done.")
        else:
            print(f"  [warn] No keyframes_state.pt found at {kf_path}")

        # Fit the viewport to the mesh bounding box
        verts = self.triangles_gui.vertices.detach().cpu().numpy()
        lo = verts.min(axis=0).tolist()
        hi = verts.max(axis=0).tolist()
        bounds = o3d.geometry.AxisAlignedBoundingBox(lo, hi)
        centre = ((np.array(lo) + np.array(hi)) * 0.5).tolist()
        self.widget3d.setup_camera(60.0, bounds, centre)

        # --- Log to Rerun ---
        self._log_rerun(kf_path if os.path.exists(kf_path) else None)

    # ------------------------------------------------------------------
    # Rerun logging
    # ------------------------------------------------------------------

    def _log_rerun(self, kf_path):
        """Log the mesh and camera frustums to Rerun."""
        import yaml
        from meshslam.gui.slam_gui import _cam_c2w

        rr.init("mesh_viewer", spawn=True)

        # Log mesh
        verts = self.triangles_gui.vertices.detach().cpu().numpy().astype(np.float32)
        faces = self.triangles_gui._triangle_indices.cpu().numpy().astype(np.uint32)
        rr.log("mesh", rr.Mesh3D(
            vertex_positions=verts,
            triangle_indices=faces,
        ))

        # Load intrinsics from config if available
        run_dir  = Path(self._iter_dir).parents[1]
        cfg_path = run_dir / "config.yml"
        fx, fy, cx, cy, W, H = 400.0, 400.0, 320.0, 240.0, 640, 480
        if cfg_path.exists():
            with open(cfg_path) as f:
                config = yaml.safe_load(f)
            calib = (config.get("Dataset") or {}).get("Calibration", {})
            fx = calib.get("fx", fx)
            fy = calib.get("fy", fy)
            cx = calib.get("cx", cx)
            cy = calib.get("cy", cy)
            W  = calib.get("width", W)
            H  = calib.get("height", H)

        # Log camera frustums
        if kf_path is not None:
            kf_state = torch.load(kf_path, map_location="cpu")
            keyframes = kf_state.get("keyframes", [])

            for kf in keyframes:
                uid  = int(kf["uid"])
                R_np = kf["R"].numpy().astype(np.float32)
                T_np = kf["T"].numpy().astype(np.float32)

                cam_dict = {"R": R_np, "T": T_np}
                C2W = _cam_c2w(cam_dict)  # (4, 4) numpy

                entity = f"cameras/cam_{uid}"
                rr.log(entity, rr.Transform3D(
                    translation=C2W[:3, 3],
                    mat3x3=C2W[:3, :3],
                ))
                rr.log(entity, rr.Pinhole(
                    focal_length=[fx, fy],
                    principal_point=[cx, cy],
                    width=W,
                    height=H,
                    image_plane_distance=0.15,
                ))

            print(f"  Logged {len(keyframes)} cameras to Rerun")

    # ------------------------------------------------------------------
    # Frustum helper
    # ------------------------------------------------------------------

    def _add_frustum_sized(self, cam_dict, color=(0, 1, 0)):
        from meshslam.gui.slam_gui import _cam_c2w, _create_frustum_lineset

        uid  = cam_dict["uid"]
        name = f"cam_{uid}"

        C2W = _cam_c2w(cam_dict)
        ls  = _create_frustum_lineset(color=color, size=self._frustum_size)

        pts   = np.asarray(ls.points)
        pts_h = np.hstack([pts, np.ones((len(pts), 1))])
        pts_w = (C2W @ pts_h.T)[:3].T
        ls.points = o3d.utility.Vector3dVector(pts_w)

        if name not in self.frustum_dict:
            self.frustum_dict[name] = True
        self.widget3d.scene.remove_geometry(name)
        self.widget3d.scene.add_geometry(name, ls, self.lit)
        self.widget3d.scene.show_geometry(name, self.cameras_chbox.checked)

    # ------------------------------------------------------------------
    # Restricted Delaunay toggle
    # ------------------------------------------------------------------

    def _on_rdt_toggle(self, is_checked):
        if is_checked:
            if self._delaunay_triangles_gui is not None:
                self._swap_triangles(self._delaunay_triangles_gui)
            elif not self._rdt_running:
                self._rdt_running = True
                self._rdt_chbox.enabled = False
                self._rdt_status.text = "Running RDT…"
                threading.Thread(target=self._run_delaunay_thread, daemon=True).start()
        else:
            self._swap_triangles(self._original_triangles_gui)
            self._rdt_status.text = ""

    def _run_delaunay_thread(self):
        import rdel as rdel_module

        orig = self._original_triangles_gui
        verts = orig.vertices.detach().cpu().numpy()
        faces = orig._triangle_indices.cpu().numpy()

        print(f"Running restricted Delaunay on {len(faces):,} triangles …")
        try:
            new_faces = rdel_module.run(verts, faces, verbose=False, orient=False)
            new_faces = np.asarray(new_faces)
            print(f"  RDT complete: {len(faces):,} → {len(new_faces):,} triangles")
        except Exception as exc:
            print(f"[error] RDT failed: {exc}")
            new_faces = None

        gui.Application.instance.post_to_main_thread(
            self.window, lambda: self._on_rdt_done(new_faces)
        )

    def _on_rdt_done(self, new_faces):
        self._rdt_running = False
        self._rdt_chbox.enabled = True

        if new_faces is None:
            self._rdt_chbox.checked = False
            self._rdt_status.text = "RDT failed"
            return

        pkt = _make_packet(self._original_triangles_gui, new_faces)
        self._delaunay_triangles_gui = GUITriangles(pkt, device="cuda")
        self._swap_triangles(self._delaunay_triangles_gui)

    def _swap_triangles(self, new_gui):
        self.triangles_gui = new_gui
        n = new_gui._triangle_indices.shape[0]
        self.tri_count_label.text = f"Triangles: {n:,}"
        if new_gui is self._delaunay_triangles_gui:
            self._rdt_status.text = f"RDT: {n:,} tris"
        if self._wireframe_on:
            self._update_wireframe()

    # ------------------------------------------------------------------
    # Sigma slider
    # ------------------------------------------------------------------

    def _on_sigma_changed(self, log_sigma):
        self._sigma_label.text = f"σ = {math.exp(log_sigma):.4f}"
        for g in (self._original_triangles_gui, self._delaunay_triangles_gui):
            if g is not None:
                g._sigma_val = log_sigma

    # ------------------------------------------------------------------
    # Wireframe overlay (depth-tested 2-D, drawn on the rendered image)
    # ------------------------------------------------------------------

    def _on_wireframe_toggle(self, enabled):
        self._wireframe_on = enabled

    def _wire_color_rgb(self):
        c = self._wire_color.color_value
        return [c.red, c.green, c.blue]

    def _update_wireframe(self):
        """No-op: wireframe is now composited per-frame inside _result_to_o3d."""

    def _on_wire_color_changed(self, _color):
        pass  # redrawn automatically every frame

    def _on_wire_width_changed(self, _width):
        pass

    def _result_to_o3d(self, result, cam):
        img = super()._result_to_o3d(result, cam)
        if self._wireframe_on and self.triangles_gui is not None:
            img_np = np.asarray(img).copy()
            self._overlay_wireframe(img_np, result, cam)
            img = o3d.geometry.Image(img_np)
        return img

    def _overlay_wireframe(self, img_np, result, cam):
        """Draw only the front-most (depth-tested) wireframe edges onto img_np."""
        import cv2

        # --- depth buffer from the rasteriser ---
        depth_buf = result["expected_depth"]
        if depth_buf.dim() == 3:
            depth_buf = depth_buf[0]
        depth_np = depth_buf.detach().cpu().numpy()   # (H, W) camera-space Z
        H, W = depth_np.shape

        # --- project all vertices to pixel space ---
        verts  = self.triangles_gui.vertices.detach()  # (V, 3)
        V      = verts.shape[0]
        device = verts.device

        # world → camera  (world_view_transform = T_w2c.T, so T_w2c = .T)
        T_w2c  = cam.world_view_transform.T            # (4, 4)
        verts_h = torch.cat([verts, torch.ones(V, 1, device=device)], dim=1)  # (V, 4)
        v_cam  = (T_w2c @ verts_h.T).T                # (V, 4)
        X, Y, Z = v_cam[:, 0], v_cam[:, 1], v_cam[:, 2]

        # pinhole projection → pixel coords
        fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
        Z_np   = Z.cpu().numpy()
        px     = (fx * X / Z + cx).cpu().numpy()      # (V,)
        py     = (fy * Y / Z + cy).cpu().numpy()      # (V,)

        # --- depth test: vertex visible if its depth ≈ the rendered depth buffer ---
        ui = np.round(px).astype(np.int32)
        vi = np.round(py).astype(np.int32)
        in_bounds = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H) & (Z_np > 0)
        ui_c = np.clip(ui, 0, W - 1)
        vi_c = np.clip(vi, 0, H - 1)
        buf_z = depth_np[vi_c, ui_c]

        # relative + absolute tolerance (handles varying scene scales)
        eps     = np.maximum(0.02 * np.abs(buf_z), 0.01)
        visible = in_bounds & (buf_z > 0) & (np.abs(Z_np - buf_z) < eps)

        # --- collect visible edges (both endpoints must be visible) ---
        faces = self.triangles_gui._triangle_indices.cpu().numpy()  # (F, 3)
        a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
        vis_ab = visible[a] & visible[b]
        vis_bc = visible[b] & visible[c]
        vis_ca = visible[c] & visible[a]

        srcs = np.concatenate([a[vis_ab], b[vis_bc], c[vis_ca]])
        dsts = np.concatenate([b[vis_ab], c[vis_bc], a[vis_ca]])
        if len(srcs) == 0:
            return

        # --- draw ---
        color = tuple(int(ch * 255) for ch in self._wire_color_rgb())
        lw    = max(1, int(self._wire_width_slider.double_value))
        x1 = np.round(px[srcs]).astype(np.int32)
        y1 = np.round(py[srcs]).astype(np.int32)
        x2 = np.round(px[dsts]).astype(np.int32)
        y2 = np.round(py[dsts]).astype(np.int32)
        for i in range(len(srcs)):
            cv2.line(img_np, (x1[i], y1[i]), (x2[i], y2[i]), color, lw, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Mesh editing: deform mode
    # ------------------------------------------------------------------

    def _on_deform_toggle(self, enabled):
        self._deform_mode = enabled
        if enabled:
            # Disable remove mode if active
            if self._remove_chbox.checked:
                self._remove_chbox.checked = False
                self._remove_status.text = ""
            self.widget3d.set_on_mouse(self._on_mouse_deform)
            self._deform_status.text = "Left-click + drag on mesh"
        else:
            self.widget3d.set_on_mouse(
                lambda _: gui.Widget.EventCallbackResult.IGNORED
            )
            self._drag_state = None
            self._deform_status.text = ""

    def _on_mouse_deform(self, event):
        if not self._deform_mode or self.triangles_gui is None:
            return gui.Widget.EventCallbackResult.IGNORED

        if (event.type == gui.MouseEvent.Type.BUTTON_DOWN
                and event.buttons & int(gui.MouseButton.LEFT)):
            hit = self._raycast(event.x, event.y)
            if hit is not None:
                self._drag_state = {
                    "hit_point": hit,
                    "prev_x": event.x,
                    "prev_y": event.y,
                    "precomp": None,
                }
                if self._connectivity_chbox.checked:
                    self._drag_state["precomp"] = self._precompute_deform_region(hit)
                self._deform_status.text = "Dragging…"
                return gui.Widget.EventCallbackResult.CONSUMED
            # Missed mesh — let camera handle it
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
            self._drag_state = None
            if was_dragging:
                self._deform_status.text = "Left-click + drag on mesh"
            return (gui.Widget.EventCallbackResult.CONSUMED if was_dragging
                    else gui.Widget.EventCallbackResult.IGNORED)

        return gui.Widget.EventCallbackResult.IGNORED

    def _pixel_to_ray(self, px, py):
        """Return (origin, direction) ray in world space for a window pixel."""
        view_mat = self.widget3d.scene.camera.get_view_matrix()  # OpenGL
        T_w2c = cv_gl @ view_mat                                  # OpenCV
        C2W   = np.linalg.inv(T_w2c)

        ray_origin = C2W[:3, 3].astype(np.float32)

        # Widget-local pixel coords (event coords are in window space)
        frame = self.widget3d.frame
        lx = px - frame.x
        ly = py - frame.y
        w, h = float(frame.width), float(frame.height)

        vfov = self.widget3d.scene.camera.get_field_of_view()
        fy   = h / (2.0 * np.tan(np.radians(vfov) / 2.0))
        cx, cy = w / 2.0, h / 2.0

        # OpenCV convention: +X right, +Y down, +Z forward
        dir_cam = np.array([(lx - cx) / fy, (ly - cy) / fy, 1.0], dtype=np.float32)
        dir_cam /= np.linalg.norm(dir_cam)

        dir_world = (C2W[:3, :3] @ dir_cam).astype(np.float32)
        dir_world /= np.linalg.norm(dir_world)

        return ray_origin, dir_world

    def _raycast(self, px, py):
        """Cast a ray from the camera through pixel (px, py); return hit world point or None."""
        verts = self.triangles_gui.vertices.detach().cpu().numpy().astype(np.float64)
        faces = self.triangles_gui._triangle_indices.cpu().numpy().astype(np.int32)

        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices  = o3d.utility.Vector3dVector(verts)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)

        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(o3d_mesh)
        scene  = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh_t)

        origin, direction = self._pixel_to_ray(px, py)
        rays   = o3d.core.Tensor(np.array([[*origin, *direction]], dtype=np.float32))
        result = scene.cast_rays(rays)
        t_hit  = float(result["t_hit"].numpy()[0])

        if np.isfinite(t_hit):
            return origin + t_hit * direction
        return None

    def _precompute_deform_region(self, hit_point):
        """Pre-compute adjacency and vertex roles for connectivity-aware deformation."""
        verts = self.triangles_gui.vertices
        faces = self.triangles_gui._triangle_indices.long()
        V = verts.shape[0]
        device = verts.device
        radius = self._brush_slider.double_value

        # Build undirected edge list
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

    # ------------------------------------------------------------------
    # Mesh editing: remove mode
    # ------------------------------------------------------------------

    def _on_remove_toggle(self, enabled):
        if enabled:
            # Disable deform mode if active
            if self._deform_mode:
                self._deform_chbox.checked = False
                self._on_deform_toggle(False)
            self.widget3d.set_on_mouse(self._on_mouse_remove)
            self._remove_status.text = "Click on mesh to remove triangles"
        else:
            self.widget3d.set_on_mouse(
                lambda _: gui.Widget.EventCallbackResult.IGNORED
            )
            self._remove_status.text = ""

    def _on_mouse_remove(self, event):
        if self.triangles_gui is None:
            return gui.Widget.EventCallbackResult.IGNORED

        if (event.type == gui.MouseEvent.Type.BUTTON_DOWN
                and event.buttons & int(gui.MouseButton.LEFT)):
            hit_info = self._raycast_triangle(event.x, event.y)
            if hit_info is not None:
                hit_point, _ = hit_info
                self._remove_triangles_near(hit_point)
                return gui.Widget.EventCallbackResult.CONSUMED
            return gui.Widget.EventCallbackResult.IGNORED

        return gui.Widget.EventCallbackResult.IGNORED

    def _raycast_triangle(self, px, py):
        """Cast a ray; return (hit_world_point, triangle_index) or None."""
        verts = self.triangles_gui.vertices.detach().cpu().numpy().astype(np.float64)
        faces = self.triangles_gui._triangle_indices.cpu().numpy().astype(np.int32)

        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices  = o3d.utility.Vector3dVector(verts)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)

        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(o3d_mesh)
        scene  = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh_t)

        origin, direction = self._pixel_to_ray(px, py)
        rays   = o3d.core.Tensor(np.array([[*origin, *direction]], dtype=np.float32))
        result = scene.cast_rays(rays)
        t_hit  = float(result["t_hit"].numpy()[0])

        if np.isfinite(t_hit):
            hit_point = origin + t_hit * direction
            tri_idx   = int(result["primitive_ids"].numpy()[0])
            return hit_point, tri_idx
        return None

    def _remove_triangles_near(self, hit_point):
        """Remove all triangles whose centroid is within the brush radius of hit_point."""
        radius = self._brush_slider.double_value
        tg     = self.triangles_gui
        faces  = tg._triangle_indices  # (F, 3) on GPU
        verts  = tg.vertices           # (V, 3) on GPU

        # Compute triangle centroids
        tri_verts = verts[faces.long()]          # (F, 3, 3)
        centroids = tri_verts.mean(dim=1)        # (F, 3)

        hit_t = torch.tensor(hit_point, dtype=torch.float32, device=verts.device)
        dist  = (centroids - hit_t).norm(dim=1)  # (F,)
        keep  = dist >= radius

        n_removed = int((~keep).sum().item())
        if n_removed == 0:
            self._remove_status.text = "No triangles in range"
            return

        new_faces = faces[keep]
        tg._triangle_indices = new_faces.contiguous()
        tg._triangles = verts[new_faces.long()]

        n = new_faces.shape[0]
        self.tri_count_label.text = f"Triangles: {n:,}"
        self._remove_status.text = f"Removed {n_removed} triangles ({n:,} left)"
        if self._wireframe_on:
            self._update_wireframe()

    # ------------------------------------------------------------------
    # Mesh editing: Laplacian smooth
    # ------------------------------------------------------------------

    def _on_smooth(self):
        if self.triangles_gui is None:
            return

        iters  = self._smooth_iters.int_value
        verts  = self.triangles_gui.vertices
        faces  = self.triangles_gui._triangle_indices.long()
        V      = verts.shape[0]

        # Undirected edge list: both directions for each triangle edge
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
        if self._wireframe_on:
            self._update_wireframe()

    # ------------------------------------------------------------------
    # Mesh editing: reset vertices
    # ------------------------------------------------------------------

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
        if self._wireframe_on:
            self._update_wireframe()

    # ------------------------------------------------------------------
    # Feature refinement
    # ------------------------------------------------------------------

    def _on_refine(self):
        if self._refine_running or self.triangles_gui is None:
            return
        self._refine_running = True
        self._refine_btn.enabled = False
        self._refine_status.text = "Loading dataset…"
        n_iters = self._refine_iters.int_value
        threading.Thread(target=self._refine_thread, args=(n_iters,), daemon=True).start()

    def _refine_thread(self, n_iters):
        import random
        import sys
        import yaml
        from types import SimpleNamespace
        from mesh_splatting.triangle_renderer import render as triangle_render

        # mesh_splatting/utils/ shadows the project-root utils/ package;
        # insert _root at the front so utils.dataset resolves correctly.
        sys.path.insert(0, str(_root))
        from utils.dataset import load_dataset

        def _post(fn):
            gui.Application.instance.post_to_main_thread(self.window, fn)

        # Locate run-level config (iter_dir = .../mesh/iteration_N)
        run_dir  = Path(self._iter_dir).parents[1]
        cfg_path = run_dir / "config.yml"
        if not cfg_path.exists():
            _post(lambda: self._on_refine_done("config.yml not found"))
            return

        with open(cfg_path) as f:
            config = yaml.safe_load(f)

        dataset_path = (config.get("Dataset") or {}).get("dataset_path", "")
        if not dataset_path or not Path(dataset_path).exists():
            _post(lambda: self._on_refine_done(f"Dataset not found: {dataset_path}"))
            return

        try:
            dataset = load_dataset(SimpleNamespace(), dataset_path, config)
        except Exception as exc:
            _post(lambda e=exc: self._on_refine_done(f"Dataset load failed: {e}"))
            return

        calib  = (config.get("Dataset") or {}).get("Calibration", {})
        fx     = getattr(dataset, "fx", None) or calib["fx"]
        fy     = getattr(dataset, "fy", None) or calib["fy"]
        cx     = getattr(dataset, "cx", None) or calib["cx"]
        cy     = getattr(dataset, "cy", None) or calib["cy"]
        H      = getattr(dataset, "height", None) or calib["height"]
        W      = getattr(dataset, "width",  None) or calib["width"]

        # Build cameras from saved keyframe poses + dataset GT images
        kf_path = Path(self._iter_dir) / "keyframes_state.pt"
        kf_data = torch.load(kf_path, map_location="cpu").get("keyframes", [])

        cameras = []
        for kf in kf_data:
            uid = int(kf["uid"])
            try:
                gt_image, _, _ = dataset[uid]   # (3, H, W) float32 on device
            except Exception as exc:
                print(f"[refine] skip kf {uid}: {exc}")
                continue
            R_np = kf["R"].numpy().astype(np.float32)
            T_np = kf["T"].numpy().astype(np.float32)
            cameras.append(RefineCam(uid, R_np, T_np, gt_image, fx, fy, cx, cy, H, W))

        if not cameras:
            _post(lambda: self._on_refine_done("No keyframe images loaded"))
            return

        print(f"[refine] {len(cameras)} cameras, {n_iters} iters")
        _post(lambda: setattr(self._refine_status, "text", f"0/{n_iters}  —"))

        tg = self.triangles_gui
        # Ensure leaf tensors (non-leaf can arise from .cpu()→.to("cuda") chain)
        tg._features_dc    = tg._features_dc.detach().requires_grad_(True)
        tg._vertex_weight  = tg._vertex_weight.detach().requires_grad_(True)

        opt_cfg   = config.get("opt_params", {})
        feature_lr = opt_cfg.get("feature_lr", 0.0025)
        weight_lr  = opt_cfg.get("weight_lr",  0.05)

        optimizer = torch.optim.Adam([
            {"params": [tg._features_dc],   "lr": feature_lr},
            {"params": [tg._vertex_weight], "lr": weight_lr},
        ], lr=0.0, eps=1e-15)

        bg = self.background

        batch = min(8, len(cameras))
        for it in range(n_iters):
            total_loss = 0.0
            for cam in random.sample(cameras, batch):
                with torch.enable_grad():
                    pkg  = triangle_render(cam, tg, self.pipe, bg)
                    loss = (pkg["render"] - cam.original_image).abs().mean()
                    loss.backward()
                total_loss += loss.item()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            avg = total_loss / batch
            it1 = it + 1
            _post(lambda i=it1, l=avg: setattr(
                self._refine_status, "text", f"{i}/{n_iters}  loss={l:.4f}"
            ))

        _post(lambda: self._on_refine_done(None))

    def _on_refine_done(self, err):
        self._refine_running = False
        self._refine_btn.enabled = True
        if err:
            self._refine_status.text = f"Error: {err}"
            print(f"[refine] {err}")
        else:
            self._refine_status.text = "Done"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("iter_dir", nargs="?",
                        default="results/tum_rgbd_dataset_freiburg3_long_office_household"
                                "/2026-03-01-19-47-23/mesh/iteration_2515",
                        help="Path to iteration directory containing "
                             "point_cloud_state_dict.pt and keyframes_state.pt")
    parser.add_argument("--frustum_size", type=float, default=0.05,
                        help="Camera frustum scale in metres (default: 0.05)")
    args = parser.parse_args()

    iter_dir = str(Path(args.iter_dir).resolve())
    if not os.path.isdir(iter_dir):
        print(f"[error] Directory not found: {iter_dir}")
        sys.exit(1)

    app = o3d.visualization.gui.Application.instance
    app.initialize()
    MeshViewer(iter_dir, frustum_size=args.frustum_size)
    app.run()


if __name__ == "__main__":
    main()
