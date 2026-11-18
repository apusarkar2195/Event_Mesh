"""
Run saver_triangle + eval_chamfer_gttsdf for every latest replica run.

For each replica_<scene>/ folder:
  1. Find the most recent timestamped run dir (skips *_best, *copy* suffixes)
  2. Find the latest iteration that has keyframes_state.pt
  3. Run saver_triangle.py        if depth_renders are missing
  4. Run eval_chamfer_gttsdf_new.py   (--gt_align --icp --no_rerun)
  5. Print a summary table at the end

Usage:
    .venv/bin/python scripts/run_replica_eval.py [--results_root results]
                                                  [--voxel_length 0.004]
                                                  [--depth_trunc 5.0]
                                                  [--min_iters 100]
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


PYTHON = sys.executable
SCRIPT_DIR = Path(__file__).parent
SAVER  = SCRIPT_DIR / "saver_triangle.py"
EVAL   = SCRIPT_DIR / "eval_chamfer_gttsdf_new.py"


def latest_run_dir(scene_dir: Path) -> Path | None:
    """Return the most-recent YYYY-MM-DD-HH-MM-SS sub-directory that contains
    actual SLAM data (a config.yml and at least one iteration with keyframes).
    Walks newest-first so partial/empty dirs (e.g. created for tsdf output only)
    are skipped automatically."""
    ts_re = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$")
    candidates = sorted(
        (p for p in scene_dir.iterdir()
         if p.is_dir() and ts_re.match(p.name)),
        reverse=True,
    )
    for p in candidates:
        if (p / "config.yml").exists() and latest_iteration(p) is not None:
            return p
    return None


def latest_iteration(run_dir: Path) -> int | None:
    """Return the highest iteration number that has a keyframes_state.pt."""
    mesh_dir = run_dir / "mesh"
    if not mesh_dir.exists():
        return None
    iters = []
    for p in mesh_dir.iterdir():
        m = re.match(r"^iteration_(\d+)$", p.name)
        if m and (p / "keyframes_state.pt").exists():
            iters.append(int(m.group(1)))
    return max(iters) if iters else None


def depth_render_count(run_dir: Path) -> int:
    dr = run_dir / "depth_renders"
    if not dr.exists():
        return 0
    return sum(1 for f in dr.iterdir() if f.suffix == ".npy")


def run(cmd: list[str], label: str) -> subprocess.CompletedProcess:
    print(f"\n  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  [FAILED] {label} returned {result.returncode}")
    return result


def parse_chamfer(run_dir: Path) -> dict | None:
    """Read Chamfer metrics saved by eval_chamfer_gttsdf_new.py."""
    metrics_path = run_dir / "tsdf_chamfer_gt" / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default="results")
    ap.add_argument("--voxel_length", type=float, default=0.004)
    ap.add_argument("--depth_trunc",  type=float, default=5.0)
    ap.add_argument("--n_samples",    type=int,   default=200_000)
    ap.add_argument("--icp_threshold",type=float, default=0.1,
                    help="ICP correspondence threshold in metres (default: 0.1)")
    ap.add_argument("--icp_max_iter", type=int,   default=50)
    ap.add_argument("--min_iters",    type=int,   default=100,
                    help="Skip runs with fewer than this many iterations")
    ap.add_argument("--scenes", nargs="*",
                    help="Only process these scenes (e.g. office0 room1). "
                         "Default: all replica_* folders.")
    args = ap.parse_args()

    results_root = Path(args.results_root)
    scene_dirs = sorted(
        p for p in results_root.iterdir()
        if p.is_dir()
        and p.name.startswith("replica_")
        and p.name != "replica_meshes"
    )
    if args.scenes:
        scene_dirs = [d for d in scene_dirs
                      if any(d.name.endswith(s) for s in args.scenes)]

    print(f"Found {len(scene_dirs)} replica scene(s): "
          + ", ".join(d.name for d in scene_dirs))

    rows = []   # (scene, iter, status, metrics_dict)

    for scene_dir in scene_dirs:
        scene = scene_dir.name.replace("replica_", "")
        print(f"\n{'='*60}")
        print(f"  Scene: {scene}")

        run_dir = latest_run_dir(scene_dir)
        if run_dir is None:
            print("  No timestamped run dir found — skipping")
            rows.append((scene, "-", "no_run", {}))
            continue

        iteration = latest_iteration(run_dir)
        if iteration is None or iteration < args.min_iters:
            print(f"  Latest iteration={iteration} < min_iters={args.min_iters} — skipping")
            rows.append((scene, str(iteration), "too_few_iters", {}))
            continue

        print(f"  Run dir : {run_dir}")
        print(f"  Iteration: {iteration}")

        # ── Step 1: saver_triangle if renders are missing ────────────────────
        n_renders = depth_render_count(run_dir)
        if n_renders == 0:
            print(f"  depth_renders missing → running saver_triangle.py")
            r = run(
                [PYTHON, str(SAVER),
                 "--result_dir", str(run_dir),
                 "--iteration",  str(iteration)],
                "saver_triangle",
            )
            if r.returncode != 0:
                rows.append((scene, str(iteration), "saver_failed", {}))
                continue
        else:
            print(f"  depth_renders already present ({n_renders} frames) — skipping saver")

        # ── Step 2: eval_chamfer_gttsdf ──────────────────────────────────────
        render_dir   = run_dir / "depth_renders"
        meshes_scene = results_root / "replica_meshes" / scene
        delaunay_ply = meshes_scene / "mesh_delaunay_postprocess.ply"
        visible_ply  = meshes_scene / "mesh_delaunay_visible.ply"
        cmd = [
            PYTHON, str(EVAL),
            str(render_dir),
            "--gt_align",
            "--icp",
            "--voxel_length",  str(args.voxel_length),
            "--depth_trunc",   str(args.depth_trunc),
            "--n_samples",     str(args.n_samples),
            "--icp_threshold", str(args.icp_threshold),
            "--icp_max_iter",  str(args.icp_max_iter),
            "--no_rerun",
        ]
        if delaunay_ply.exists():
            cmd += ["--delaunay_ply", str(delaunay_ply)]
            print(f"  Delaunay:         {delaunay_ply.name}")
        else:
            print(f"  No Delaunay mesh at {delaunay_ply}")
        if visible_ply.exists():
            cmd += ["--delaunay_visible_ply", str(visible_ply)]
            print(f"  Delaunay visible: {visible_ply.name}")
        else:
            print(f"  No Delaunay visible mesh at {visible_ply}")

        print(f"  Running eval_chamfer_gttsdf_new.py")
        r = run(cmd, "eval_chamfer_gttsdf_new")
        if r.returncode != 0:
            rows.append((scene, str(iteration), "eval_failed", {}))
            continue

        metrics = parse_chamfer(run_dir)
        rows.append((scene, str(iteration), "ok", metrics or {}))

    # ── Summary table ────────────────────────────────────────────────────────
    W = 100
    nan = float("nan")

    def chamfer(m, tag):
        if not m or tag not in m:
            return f"{'—':>8}"
        return f"{m[tag].get('chamfer_cm', nan):8.3f}"

    def timing(m):
        t = m.get("timing_s", {}).get("est_tsdf_s", nan) if m else nan
        return f"{t:>9.1f}" if t == t else f"{'—':>9}"  # nan check

    print(f"\n{'='*W}")
    print("SUMMARY  (Chamfer cm ↓,  est-TSDF time in seconds)")
    print(f"{'='*W}")
    print(f"{'scene':<12}  {'iter':>6}  {'status':<14}  "
          f"{'TSDF':>8}  {'Delaunay':>8}  {'Del.vis':>8}  {'est-TSDF(s)':>11}")
    print("-" * W)
    for scene, itr, status, m in rows:
        base = f"{scene:<12}  {itr:>6}  {status:<14}"
        if status == "ok" and m:
            print(base
                  + "  " + chamfer(m, "tsdf")
                  + "  " + chamfer(m, "delaunay")
                  + "  " + chamfer(m, "delaunay_visible")
                  + "  " + timing(m))
        else:
            print(base)

    # Save summary JSON
    summary_path = results_root / "replica_eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            [{"scene": s, "iteration": i, "status": st, "metrics": m}
             for s, i, st, m in rows],
            f, indent=2,
        )
    print(f"\nSummary saved → {summary_path}")


if __name__ == "__main__":
    main()
