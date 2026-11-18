"""
Compute Depth L1 [cm] for Replica scenes by comparing rendered depth .npy files
against GT depth PNGs from the dataset.

Usage:
    python scripts/eval_depth_l1.py
"""

import numpy as np
from pathlib import Path
from PIL import Image

DEPTH_SCALE = 6553.5  # uint16 -> meters

DATASET_ROOT = Path("datasets/replica")

# Latest run directories per scene (edit as needed)
SCENES = {
    "r0": ("replica_room0",   "2026-08-26-00-45-57"),
    "r1": ("replica_room1",   "2026-08-26-02-34-56"),
    "r2": ("replica_room2",   "2026-08-26-04-33-54"),
    "o0": ("replica_office0", "2026-08-25-15-00-33"),
    "o1": ("replica_office1", "2026-08-25-16-57-03"),
    "o2": ("replica_office2", "2026-08-25-18-38-55"),
    "o3": ("replica_office3", "2026-08-25-20-44-08"),
    "o4": ("replica_office4", "2026-08-25-22-41-06"),
}

RESULTS_ROOT = Path("results")


def compute_depth_l1(scene_key, scene_dir, run_id):
    render_dir = RESULTS_ROOT / scene_dir / run_id / "depth_renders"
    dataset_dir = DATASET_ROOT / scene_dir.replace("replica_", "")

    if not render_dir.exists():
        print(f"  [SKIP] {scene_key}: render dir not found: {render_dir}")
        return None
    if not dataset_dir.exists():
        print(f"  [SKIP] {scene_key}: dataset dir not found: {dataset_dir}")
        return None

    # Load frame IDs from cameras.npz
    cam_file = render_dir / "cameras.npz"
    if cam_file.exists():
        cam = np.load(cam_file)
        frame_ids = cam["frame_ids"].tolist()
    else:
        # Infer from filenames
        frame_ids = sorted(int(p.stem.split("_")[0]) for p in render_dir.glob("*_depth.npy"))

    l1_errors = []
    missing = 0
    for fid in frame_ids:
        render_path = render_dir / f"{fid:05d}_depth.npy"
        gt_path = dataset_dir / "results" / f"depth{fid:06d}.png"

        if not render_path.exists() or not gt_path.exists():
            missing += 1
            continue

        rendered = np.load(render_path)  # float32, meters
        gt_raw = np.array(Image.open(gt_path), dtype=np.float32)
        gt = gt_raw / DEPTH_SCALE  # meters

        # Only evaluate where GT depth is valid (>0)
        mask = gt > 0
        if not mask.any():
            continue

        l1 = np.abs(rendered[mask] - gt[mask]).mean() * 100  # -> cm
        l1_errors.append(l1)

    if not l1_errors:
        print(f"  [WARN] {scene_key}: no valid frames")
        return None

    if missing:
        print(f"  [INFO] {scene_key}: {missing} frames missing (skipped)")

    mean_l1 = np.mean(l1_errors)
    return mean_l1


results = {}
for key, (scene_dir, run_id) in SCENES.items():
    l1 = compute_depth_l1(key, scene_dir, run_id)
    results[key] = l1
    if l1 is not None:
        print(f"  {key}: {l1:.2f} cm")
    else:
        print(f"  {key}: N/A")

keys = ["r0", "r1", "r2", "o0", "o1", "o2", "o3", "o4"]
vals = [results[k] for k in keys]
valid = [v for v in vals if v is not None]
avg = np.mean(valid) if valid else float("nan")

print()
print("=" * 60)
print("Depth L1 [cm]  (ours)")
print("-" * 60)
header = "  ".join(f"{k:>6}" for k in keys) + f"  {'avg':>6}"
row = "  ".join(f"{v:6.2f}" if v is not None else f"{'N/A':>6}" for v in vals) + f"  {avg:6.2f}"
print(header)
print(row)
print()
print("Reference:")
print("  MonoGS:    3.00  3.47  4.66  3.10  6.08  6.15  4.77  4.94   4.52")
print("  MonoGS-2D: 0.45  0.28  0.57  0.37  0.59  0.85  0.62  0.63   0.54")
