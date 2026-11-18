"""
GUI utilities for Triangle SLAM viewer.
"""
import queue
import torch
import numpy as np


# OpenGL → OpenCV convention flip (y and z axes)
cv_gl = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float64)


def get_latest_queue(q):
    """Drain queue, return only the most recent item."""
    message = None
    while True:
        try:
            msg = q.get_nowait()
            if message is not None:
                del message
            message = msg
        except queue.Empty:
            break
    return message


class TrianglePacket:
    """
    Data container for SLAM → GUI communication.
    All tensors are detached CPU copies so they cross process boundaries safely.
    """
    def __init__(
        self,
        triangles=None,
        current_frame=None,
        keyframes=None,
        finish=False,
    ):
        self.has_triangles = False
        if triangles is not None:
            self.has_triangles = True
            self.vertices       = triangles.vertices.detach().cpu()
            self.triangle_indices = triangles._triangle_indices.cpu()
            self.vertex_weight  = triangles.vertex_weight.detach().cpu()
            self.features_dc    = triangles._features_dc.detach().cpu()
            self.features_rest  = triangles._features_rest.detach().cpu()
            self._sigma         = float(triangles._sigma)
            self.opacity_floor  = float(triangles.opacity_floor)
            self.active_sh_degree = int(triangles.active_sh_degree)
            self.max_sh_degree  = int(triangles.max_sh_degree)
            self.scaling        = int(triangles.scaling)

        # Current frame carries the GT image for the preview panels.
        self.current_frame = _pack_cam(current_frame, include_image=True)
        self.keyframes = [_pack_cam(kf) for kf in keyframes] if keyframes else []
        self.finish = finish
        # Optional TSDF comparison mesh (Open3D TriangleMesh, already on CPU)
        self.tsdf_mesh = None
        # True = GUI should adopt this mesh as the editing base even in deform
        # mode (sent when SLAM pauses for a live edit, or after a count change).
        self.edit_base = False


def _pack_cam(cam, include_image=False, image_max_width=320):
    """Extract the minimal pose + intrinsics from a Camera object.

    When include_image is True, the ground-truth RGB image is attached
    (downsampled to image_max_width) for the GUI preview panels.
    """
    if cam is None:
        return None
    d = {
        "R":    cam.R.detach().cpu().numpy().astype(np.float32),
        "T":    cam.T.detach().cpu().numpy().astype(np.float32),
        "fx":   float(cam.fx),
        "fy":   float(cam.fy),
        "cx":   float(cam.cx),
        "cy":   float(cam.cy),
        "FoVx": float(cam.FoVx),
        "FoVy": float(cam.FoVy),
        "H":    int(cam.image_height),
        "W":    int(cam.image_width),
        "uid":  int(cam.uid),
    }
    if hasattr(cam, "R_gt") and cam.R_gt is not None:
        d["R_gt"] = cam.R_gt.detach().cpu().numpy().astype(np.float32)
        d["T_gt"] = cam.T_gt.detach().cpu().numpy().astype(np.float32)

    # Learned per-frame autoexposure (affine brightness), so the GUI can apply
    # the same correction to the render as the optimisation does.
    if getattr(cam, "exposure_a", None) is not None:
        d["exposure_a"] = float(cam.exposure_a.detach().cpu().reshape(-1)[0])
        d["exposure_b"] = float(cam.exposure_b.detach().cpu().reshape(-1)[0])

    if include_image and getattr(cam, "original_image", None) is not None:
        # Downsampled GT RGB as uint8 HWC for the preview panels.
        img = cam.original_image.detach()              # (C, H, W), float [0, 1]
        c, h, w = img.shape
        scale = min(1.0, float(image_max_width) / float(w))
        if scale < 1.0:
            new_h, new_w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0), size=(new_h, new_w),
                mode="bilinear", align_corners=False,
            ).squeeze(0)
        img = (img.clamp(0, 1) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
        d["gt_image"] = img
    return d


class Packet_vis2main:
    flag_pause = False
    flag_resume = False
    flag_snapshot = False
    flag_retriangulate = False
    retriangulate_fix_vertices = False
    flag_freeze_vertices = None  # None = no change; True/False = set freeze state
    flag_tsdf_fusion = False
    # Live mesh editing (GUI -> main):
    edited_vertices = None  # np (V,3) float32: new vertex positions for the tracked mesh
    commit_objects = None   # dict(verts (N,3), faces (M,3), colors (N,3)): physics objects to add to the map


class ParamsGUI:
    def __init__(self, pipe, background, q_main2vis, q_vis2main):
        self.pipe        = pipe
        self.background  = background
        self.q_main2vis  = q_main2vis
        self.q_vis2main  = q_vis2main


class FakeQueue:
    """Drop-in for mp.Queue when GUI is disabled."""
    def put(self, *a, **kw): pass
    def get_nowait(self): raise queue.Empty
    def empty(self): return True


# ---------------------------------------------------------------------------
# GUITriangles: lightweight proxy for TriangleModel used inside the GUI process
# ---------------------------------------------------------------------------

class GUITriangles:
    """
    Wraps TrianglePacket tensors to expose the same interface that
    mesh_splatting.triangle_renderer.render() expects.
    All tensors live on CUDA.
    """

    def __init__(self, packet: TrianglePacket, device="cuda"):
        self._update(packet, device)

    def _update(self, packet: TrianglePacket, device="cuda"):
        self.vertices          = packet.vertices.to(device)
        self._triangle_indices = packet.triangle_indices.to(device)
        self._vertex_weight    = packet.vertex_weight.to(device)
        self._features_dc      = packet.features_dc.to(device)
        self._features_rest    = packet.features_rest.to(device)
        self._sigma_val        = packet._sigma
        self.opacity_floor     = packet.opacity_floor
        self.active_sh_degree  = packet.active_sh_degree
        self.max_sh_degree     = packet.max_sh_degree
        self.scaling           = packet.scaling
        self.config            = None  # renderer falls back to defaults

        # Derived: _triangles used by get_triangles_points
        v = self.vertices[self._triangle_indices]  # (T, 3, 3)
        self._triangles = v

    # --- Properties mirroring TriangleModel ---

    @property
    def get_vertices(self):
        return self.vertices

    @property
    def get_triangle_indices(self):
        return self._triangle_indices

    @property
    def get_triangles_points(self):
        return self._triangles

    @property
    def get_vertex_weight(self):
        return self.opacity_floor + (1.0 - self.opacity_floor) * torch.sigmoid(self._vertex_weight)

    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def get_sigma(self):
        import math
        return math.exp(self._sigma_val)

    @property
    def vertex_weight(self):
        return self._vertex_weight

    @property
    def _sigma(self):
        return self._sigma_val


# ---------------------------------------------------------------------------
# Minimal camera for GUI rendering (no grad, no nn.Module overhead)
# ---------------------------------------------------------------------------

class GUICamera:
    """
    Lightweight camera built from the Open3D viewport's view matrix.
    Exposes the same interface as utils.cameras.Camera for render().
    """

    def __init__(self, view_matrix_o3d, vfov_deg, widget_h, widget_w, device="cuda"):
        import math

        # Convert OpenGL → OpenCV convention
        T_w2c = torch.from_numpy(
            (cv_gl @ view_matrix_o3d).astype(np.float32)
        ).to(device)  # (4, 4)

        self.image_height = int(widget_h)
        self.image_width  = int(widget_w)
        self.FoVy = math.radians(vfov_deg)
        self.FoVx = 2.0 * math.atan(
            math.tan(self.FoVy * 0.5) * widget_w / widget_h
        )
        self.fx = widget_w / (2.0 * math.tan(self.FoVx * 0.5))
        self.fy = widget_h / (2.0 * math.tan(self.FoVy * 0.5))
        self.cx = widget_w / 2.0
        self.cy = widget_h / 2.0

        # Matches Camera.world_view_transform = getWorld2View2(R, T).T = T_w2c.T
        self.world_view_transform = T_w2c.T  # (4, 4)

        from mesh_splatting.utils.graphics_utils import getProjectionMatrix2
        proj = getProjectionMatrix2(
            znear=0.01, zfar=100.0,
            fx=self.fx, fy=self.fy,
            cx=self.cx, cy=self.cy,
            W=self.image_width, H=self.image_height,
        ).to(device)
        self.projection_matrix = proj.T  # matches Camera convention

        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0)
            .bmm(self.projection_matrix.unsqueeze(0))
            .squeeze(0)
        )

        # camera_center = C2W[3, :3]  (last row of inverse of world_view_transform)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        # Renderer reads these even when optimize_pose_only=False; provide zero defaults
        self.cam_rot_delta   = torch.zeros(3, device=device)
        self.cam_trans_delta = torch.zeros(3, device=device)
        self.exposure_a      = torch.zeros(1, device=device)
        self.exposure_b      = torch.zeros(1, device=device)


class PreviewCamera:
    """Render camera built from a packed current-frame dict (R, T + intrinsics).

    Renders the map from a SLAM frame's own pose, at a reduced resolution, for
    the GUI preview panels. width is the target render width; intrinsics are
    scaled to match so the result lines up with the (downsampled) GT image.
    """

    def __init__(self, cam_dict, width, device="cuda"):
        import math
        from mesh_splatting.utils.graphics_utils import getWorld2View2, getProjectionMatrix2

        W0, H0 = int(cam_dict["W"]), int(cam_dict["H"])
        scale = min(1.0, float(width) / float(W0))
        self.image_width  = max(1, int(round(W0 * scale)))
        self.image_height = max(1, int(round(H0 * scale)))
        self.fx = float(cam_dict["fx"]) * scale
        self.fy = float(cam_dict["fy"]) * scale
        self.cx = float(cam_dict["cx"]) * scale
        self.cy = float(cam_dict["cy"]) * scale
        self.FoVx = float(cam_dict["FoVx"])
        self.FoVy = float(cam_dict["FoVy"])

        R = torch.from_numpy(cam_dict["R"]).float()
        T = torch.from_numpy(cam_dict["T"]).float()
        self.world_view_transform = getWorld2View2(R, T).T.to(device)

        proj = getProjectionMatrix2(
            znear=0.01, zfar=100.0,
            fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy,
            W=self.image_width, H=self.image_height,
        ).to(device)
        self.projection_matrix = proj.T
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0)
            .bmm(self.projection_matrix.unsqueeze(0))
            .squeeze(0)
        )
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        self.cam_rot_delta   = torch.zeros(3, device=device)
        self.cam_trans_delta = torch.zeros(3, device=device)
        self.exposure_a      = torch.zeros(1, device=device)
        self.exposure_b      = torch.zeros(1, device=device)
