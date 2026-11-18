"""
Image save helpers and pose utilities used by mesh_utils and related pipelines.
"""
import numpy as np
import os


def save_img_u8(arr, path):
    """Save a numpy image as PNG. arr: (H, W, C) or (H, W), float [0, 1] or uint8 [0, 255]."""
    arr = np.asarray(arr)
    if arr.dtype in (np.float32, np.float64):
        arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        import imageio
        imageio.imwrite(path, arr)
    except Exception:
        import cv2
        if arr.shape[-1] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, arr)


def save_img_f32(arr, path):
    """Save a float32 image (e.g. depth map). arr: (H, W). Uses TIFF for float."""
    arr = np.asarray(arr, dtype=np.float32)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    import imageio
    imageio.imwrite(path, arr)


def save_img_depth_png(arr, path, scale=1000.0):
    """Save float32 depth as 16-bit PNG. arr: (H, W) in meters; scale (default 1000) gives mm."""
    arr = np.asarray(arr, dtype=np.float32)
    arr = (arr * scale).clip(0, 65535).astype(np.uint16)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        import imageio
        imageio.imwrite(path, arr)
    except Exception:
        import cv2
        cv2.imwrite(path, arr)


def focus_point_fn(poses):
    """
    Return 3D focus point from camera poses. poses: (N, 3, 4) c2w (rotation + translation).
    Returns (3,) world-space point (e.g. mean of camera positions).
    """
    poses = np.asarray(poses)
    if poses.ndim == 3 and poses.shape[1] >= 3 and poses.shape[2] >= 4:
        return poses[:, :3, 3].mean(axis=0)
    return np.zeros(3, dtype=np.float64)


def transform_poses_pca(poses):
    """
    Optional: center or normalize poses (e.g. for PCA alignment).
    poses: (N, 3, 4). Returns poses (default: identity pass-through).
    """
    return np.asarray(poses)
