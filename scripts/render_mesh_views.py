"""
Render the GT TSDF mesh and Delaunay-visible mesh from GT keyframe poses and
compare them side-by-side.

For each replica scene:
  1. Load GT camera poses from the dataset at keyframe UIDs
  2. Load gt_tsdf.ply  (already in GT frame)
  3. Load mesh_delaunay_visible.ply, apply T_align (est → GT frame)
  4. Render both from GT camera poses (unlit, vertex colours)
  5. Save GT TSDF | Delaunay-visible side-by-side to comparison/

Output:
  results/replica_meshes/<scene>/renders/
    gt_tsdf/            <uid:05d>.png
    delaunay_visible/   <uid:05d>.png
    comparison/         <uid:05d>.png   (GT TSDF | Delaunay visible)

Usage:
    .venv/bin/python scripts/render_mesh_views.py
    .venv/bin/python scripts/render_mesh_views.py --scenes office0 room1
    .venv/bin/python scripts/render_mesh_views.py --overwrite
"""

import argparse
import re
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as o3dr
import torch
import yaml
from PIL import Image, ImageDraw
from tqdm import tqdm

from mesh_splatting.utils.graphics_utils import getWorld2View2
from utils.dataset import load_dataset


# ─────────────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────────────

def latest_iteration(run_dir: Path) -> int | None:
    mesh_dir = run_dir / "mesh"
    if not mesh_dir.exists():
        return None
    iters = [
        int(m.group(1))
        for p in mesh_dir.iterdir()
        if (m := re.match(r"^iteration_(\d+)$", p.name))
        and (p / "keyframes_state.pt").exists()
    ]
    return max(iters) if iters else None


def latest_run_dir(scene_dir: Path) -> Path | None:
    ts_re = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$")
    candidates = sorted(
        (p for p in scene_dir.iterdir() if p.is_dir() and ts_re.match(p.name)),
        reverse=True,
    )
    for p in candidates:
        if (p / "config.yml").exists() and latest_iteration(p) is not None:
            return p
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Alignment helpers (same as eval_tsdf.py)
# ─────────────────────────────────────────────────────────────────────────────

def umeyama_alignment(src: np.ndarray, dst: np.ndarray, allow_scale: bool = True):
    n = len(src)
    mu_src = src.mean(0);  mu_dst = dst.mean(0)
    src_c  = src - mu_src; dst_c  = dst - mu_dst
    var_src = (src_c ** 2).sum() / n
    H = (dst_c.T @ src_c) / n
    U, S, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))])
    R = U @ D @ Vt
    s = float((S * D.diagonal()).sum() / var_src) if allow_scale else 1.0
    t = mu_dst - s * (R @ mu_src)
    return s, R, t


def similarity_matrix(s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3]  = t
    return T


def camera_center(T_w2c: np.ndarray) -> np.ndarray:
    return np.linalg.inv(T_w2c)[:3, 3]


# ─────────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────────

LABEL_H = 22

def render_mesh_offscreen(
    mesh: o3d.geometry.TriangleMesh,
    uid_list: list[int],
    T_w2c_list: list[np.ndarray],
    fx: float, fy: float, cx: float, cy: float,
    W: int, H: int,
    out_dir: Path,
) -> dict[int, np.ndarray]:
    """Render mesh (unlit) from each camera pose; save PNGs; return {uid: H×W×3}."""
    out_dir.mkdir(parents=True, exist_ok=True)

    rend = o3dr.OffscreenRenderer(W, H)
    rend.scene.set_background([0.05, 0.05, 0.05, 1.0])

    mat = o3dr.MaterialRecord()
    mat.shader = "defaultUnlit"
    rend.scene.add_geometry("mesh", mesh, mat)

    intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
    imgs: dict[int, np.ndarray] = {}

    for uid, T_w2c in tqdm(zip(uid_list, T_w2c_list), total=len(uid_list),
                            desc="  render", leave=False):
        rend.setup_camera(intrinsic, T_w2c.astype(np.float64))
        img_np = np.asarray(rend.render_to_image())[:, :, :3]  # RGBA → RGB
        Image.fromarray(img_np).save(out_dir / f"{uid:05d}.png")
        imgs[uid] = img_np

    rend.scene.remove_geometry("mesh")
    return imgs


def make_comparison(
    uid_list: list[int],
    imgs_a: dict[int, np.ndarray],
    imgs_b: dict[int, np.ndarray],
    label_a: str,
    label_b: str,
    out_dir: Path,
):
    """Save label_a | label_b side-by-side to out_dir/comparison/."""
    comp_dir = out_dir / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    for uid in uid_list:
        if uid not in imgs_a or uid not in imgs_b:
            continue
        strips = []
        for label, img in [(label_a, imgs_a[uid]), (label_b, imgs_b[uid])]:
            W = img.shape[1]
            bar = Image.new("RGB", (W, LABEL_H), (30, 30, 30))
            ImageDraw.Draw(bar).text((4, 4), label, fill=(210, 210, 210))
            strips.append(np.vstack([np.array(bar), img]))
        Image.fromarray(np.hstack(strips)).save(comp_dir / f"{uid:05d}.png")


# ─────────────────────────────────────────────────────────────────────────────
# Per-scene
# ─────────────────────────────────────────────────────────────────────────────

def process_scene(
    run_dir: Path,
    meshes_dir: Path,
    iteration: int,
    overwrite: bool,
):
    out_dir = meshes_dir / "renders"
    if not overwrite and (out_dir / "comparison").exists():
        print("  Renders already exist — skipping (--overwrite to redo)")
        return

    gt_tsdf_ply = run_dir / "tsdf_meshes" / "gt_tsdf.ply"
    del_vis_ply = meshes_dir / "mesh_delaunay_visible.ply"
    if not gt_tsdf_ply.exists():
        print(f"  WARNING: gt_tsdf.ply not found at {gt_tsdf_ply} — skipping")
        return
    if not del_vis_ply.exists():
        print(f"  WARNING: mesh_delaunay_visible.ply not found — skipping")
        return

    # ── Intrinsics ────────────────────────────────────────────────────────
    with open(run_dir / "config.yml") as f:
        cfg = yaml.safe_load(f)
    cal = cfg["Dataset"]["Calibration"]
    fx, fy = float(cal["fx"]), float(cal["fy"])
    cx, cy = float(cal["cx"]), float(cal["cy"])
    W, H   = int(cal["width"]), int(cal["height"])

    # ── Keyframe UIDs + estimated poses ───────────────────────────────────
    kf_path  = run_dir / "mesh" / f"iteration_{iteration}" / "keyframes_state.pt"
    kf_state = torch.load(kf_path, map_location="cpu")
    keyframes = kf_state["keyframes"]
    uid_list  = [kf["uid"] for kf in keyframes]

    est_T_w2c: dict[int, np.ndarray] = {}
    for kf in keyframes:
        uid = kf["uid"]
        est_T_w2c[uid] = (
            getWorld2View2(kf["R"].cuda(), kf["T"].cuda()).cpu().numpy().astype(np.float64)
        )

    # ── GT poses from dataset ─────────────────────────────────────────────
    dataset_path = cfg["Dataset"]["dataset_path"]
    if not Path(dataset_path).is_absolute():
        dataset_path = str(Path(__file__).parent.parent / dataset_path)
    dataset = load_dataset(None, dataset_path, cfg)

    gt_T_w2c: dict[int, np.ndarray] = {}
    for uid in tqdm(uid_list, desc="  loading GT poses", leave=False):
        _, _, pose_t = dataset[uid]
        gt_T_w2c[uid] = pose_t.cpu().numpy().astype(np.float64)

    # ── Umeyama: est → GT frame (to align Delaunay mesh) ─────────────────
    centers_est = np.stack([camera_center(est_T_w2c[u]) for u in uid_list])
    centers_gt  = np.stack([camera_center(gt_T_w2c[u])  for u in uid_list])
    s, R_al, t_al = umeyama_alignment(centers_est, centers_gt)
    T_align = similarity_matrix(s, R_al, t_al)
    print(f"  Umeyama scale={s:.4f}")

    gt_T_w2c_list = [gt_T_w2c[u] for u in uid_list]

    # ── GT TSDF mesh (already in GT frame) ───────────────────────────────
    print("  Loading gt_tsdf.ply")
    mesh_gt = o3d.io.read_triangle_mesh(str(gt_tsdf_ply))
    mesh_gt.compute_vertex_normals()
    print(f"  {len(mesh_gt.triangles):,} faces")

    print("  Rendering GT TSDF")
    imgs_gt = render_mesh_offscreen(
        mesh_gt, uid_list, gt_T_w2c_list,
        fx, fy, cx, cy, W, H,
        out_dir / "gt_tsdf",
    )

    # ── Delaunay visible (est frame → GT frame via T_align) ───────────────
    print("  Loading mesh_delaunay_visible.ply")
    mesh_del = o3d.io.read_triangle_mesh(str(del_vis_ply))
    mesh_del.compute_vertex_normals()
    mesh_del.transform(T_align)
    print(f"  {len(mesh_del.triangles):,} faces")

    print("  Rendering Delaunay visible")
    imgs_del = render_mesh_offscreen(
        mesh_del, uid_list, gt_T_w2c_list,
        fx, fy, cx, cy, W, H,
        out_dir / "delaunay_visible",
    )

    # ── Side-by-side comparison ───────────────────────────────────────────
    print("  Creating comparison images")
    make_comparison(uid_list, imgs_gt, imgs_del, "GT TSDF", "Delaunay visible", out_dir)
    print(f"  Saved → {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default="results")
    ap.add_argument("--scenes", nargs="*",
                    help="Scene names (e.g. office0 room1). Default: all.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-render even if renders already exist")
    args = ap.parse_args()

    results_root = Path(args.results_root)
    meshes_root  = results_root / "replica_meshes"

    scene_names = sorted(
        p.name for p in meshes_root.iterdir()
        if p.is_dir() and (p / "mesh_delaunay_visible.ply").exists()
    )
    if args.scenes:
        scene_names = [s for s in scene_names if s in args.scenes]

    print(f"Scenes: {scene_names}")

    for scene in scene_names:
        print(f"\n{'='*60}")
        print(f"  {scene}")
        meshes_dir = meshes_root / scene
        scene_dir  = results_root / f"replica_{scene}"
        if not scene_dir.exists():
            print(f"  No SLAM results at {scene_dir}")
            continue

        run_dir = latest_run_dir(scene_dir)
        if run_dir is None:
            print("  No valid SLAM run found")
            continue

        iteration = latest_iteration(run_dir)
        print(f"  Run: {run_dir.name}  iter={iteration}")

        try:
            process_scene(run_dir, meshes_dir, iteration, args.overwrite)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    print("\nDone.")


if __name__ == "__main__":
    main()
