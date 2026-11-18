#!/usr/bin/env python3
"""
Debug mesh color / spherical harmonics issues.
Prints statistics on features_dc and identifies potential causes of speckling.
Usage:
  python meshslam/debug_mesh_colors.py --config CONFIG --state_path STATE_DIR
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from mesh_splatting.scene.triangle_model import TriangleModel
from utils.config_utils import load_config

SH_C0 = 0.28209479177387814


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--state_path", required=True)
    args = parser.parse_args()

    state_dir = args.state_path
    if state_dir.endswith(".pt"):
        state_dir = str(Path(state_dir).parent)
    config = load_config(args.config)
    triangles = TriangleModel(sh_degree=0, config=config)
    triangles.load_parameters(state_dir, device="cuda")

    dc = triangles._features_dc.detach().cpu().numpy()
    print("=" * 60)
    print("features_dc (SH DC coefficients) statistics")
    print("=" * 60)
    print(f"  Shape: {dc.shape}")
    print(f"  Min:   {dc.min():.4f}")
    print(f"  Max:   {dc.max():.4f}")
    print(f"  Mean:  {dc.mean():.4f}")
    print(f"  Std:   {dc.std():.4f}")
    n_nan = np.isnan(dc).sum()
    n_inf = np.isinf(dc).sum()
    print(f"  NaN count: {n_nan}")
    print(f"  Inf count: {n_inf}")

    # RGB range for reasonable display: features_dc in [-1.77, 1.77] -> RGB in [0, 1]
    # Values outside ~[-4, 4] will saturate
    rgb = np.clip(SH_C0 * dc + 0.5, 0, 1)
    saturated_low = (rgb == 0).sum()
    saturated_high = (rgb == 1).sum()
    print(f"\n  RGB (SH_C0 * dc + 0.5) clamped to [0,1]:")
    print(f"  Pixels at 0 (saturated low):  {saturated_low}")
    print(f"  Pixels at 1 (saturated high): {saturated_high}")
    if saturated_low + saturated_high > dc.size * 0.01:
        print("  -> Many saturated values can cause speckling (flat bright/dark patches)")

    print("\n  Per-channel (R,G,B) stats:")
    for i, c in enumerate("RGB"):
        ch = dc[..., i].flatten()
        print(f"    {c}: min={ch.min():.4f} max={ch.max():.4f} mean={ch.mean():.4f}")

    print("\n" + "=" * 60)
    print("Likely causes of speckling:")
    print("=" * 60)
    print("1. Extreme features_dc: values outside [-4, 4] saturate -> bright/dark spots")
    print("   Fix: clamp features_dc before export, or re-run appearance polish (extract_mesh)")
    print("2. Delaunay connectivity: new triangles connect vertices from different regions")
    print("   with different colors -> sharp gradients at triangle boundaries")
    print("   Fix: run extract_mesh polish phase to re-optimize colors after Delaunay")
    print("3. NaN/Inf in features_dc -> undefined colors")
    print("=" * 60)


if __name__ == "__main__":
    main()
