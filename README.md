# Triangle Splatting SLAM

Official code release for **Triangle Splatting SLAM**.

[Project page](https://nmjfry.github.io/triangle-splatting-slam/) &nbsp;·&nbsp; [Paper (arXiv)](https://arxiv.org/abs/2605.31419)

Nicholas Fry, Eric Dexheimer, Kirill Mazur, Paul H. J. Kelly, Andrew J. Davison.

---



## Prerequisites

- Python 3.11
- An NVIDIA GPU with the CUDA toolkit installed (tested with CUDA 12.4)
- [uv](https://docs.astral.sh/uv/)

## Installation

```bash
# --recursive fetches the CUDA-extension submodules
git clone --recursive https://github.com/nmjfry/ts-slam.git
cd ts-slam

# Create the virtual environment, install all Python dependencies and
# build the CUDA extensions
uv sync
source .venv/bin/activate
```

The first `uv sync` typically takes a while: it downloads the
CUDA build of PyTorch (~2 GB) and compiles five CUDA/C++ extensions from
source, with no output by default. Run `uv sync -v` instead to stream the
download and build logs. Subsequent syncs use cached wheels and finish in
seconds.

If the repository was cloned without `--recursive`, fetch the submodules with
`git submodule update --init --recursive`.

The CUDA extensions under `submodules/` are git submodules: the differentiable
triangle rasteriser
([`diff-triangle-rasterization-w-pose`](https://github.com/nmjfry/diff-triangle-rasterization-w-pose),
also usable standalone), `simple-knn`, and `effrdel` (restricted Delaunay).

Note: after changing a submodule (pulling a new commit or switching its
branch), plain `uv sync` may silently reinstall a cached wheel built from the
old sources. Force a rebuild with

```bash
uv sync --reinstall-package diff-triangle-rasterization
```

## Datasets

```bash
bash scripts/download_tum.sh       # → datasets/tum/
bash scripts/download_replica.sh   # → datasets/replica/
```

## Running SLAM

The entry point is `slam.py`. A config selects the dataset and all
hyperparameters.

```bash
# Replica (RGB-D)
python slam.py --config configs/rgbd/replica/room0.yaml

# TUM RGB-D
python slam.py --config configs/rgbd/tum/fr1_desk.yaml
```

Useful flags:

| Flag | Effect |
|------|--------|
| `--eval` | Evaluation mode: save results, disable GUI, compute rendering metrics |
| `--dataset_path PATH` | Override the dataset path from the config |
| `--num_frames N` | Stop after N frames |
| `--start_frame N` | Skip the first N frames |
| `--resume_from DIR` | Resume from a saved checkpoint (e.g. `results/.../mesh/iteration_1250`) |
| `-o KEY=VALUE` | Override any config value with dot notation, e.g. `-o Results.use_gui=True` |

Run all paper sequences:

```bash
bash scripts/run_replica_sequences.sh
bash scripts/run_tum_sequences.sh
```

### Live GUI

The interactive Open3D GUI is enabled via `Results.use_gui` in the config (the
live configs set it on). To enable it for any run:

```bash
python slam.py --config configs/rgbd/replica/room0.yaml -o Results.use_gui=True
```

### Live camera demo

Run directly from a connected sensor (intrinsics are read from the hardware):

```bash
python slam.py --config configs/live/realsense.yaml      # Intel RealSense
python slam.py --config configs/live/realsense_dt.yaml   # RealSense + on-the-fly Delaunay meshing
python slam.py --config configs/live/kinect.yaml         # Azure Kinect
python slam.py --config configs/live/webcam.yaml         # Monocular webcam
```

RealSense requires `pyrealsense2`, which `uv sync` installs by default (it is
in the `live` dependency group; skip it with `uv sync --no-default-groups`).
The Kinect config requires the Azure Kinect SDK.

## Meshing and the interactive viewer

The GUI provides a "Retriangulate" button that runs Delaunay triangulation on the current map, followed by a photometric optimisation to improve the appearance of the mesh. The vertices may either be frozen during this phase (by ticking "freeze vertices") or left free to move.

For the best results, make the triangles fully opaque once the Delaunay step has finished by reducing the sigma value to its minimum (in the triangle settings).

SLAM saves checkpoints to `results/<run>/mesh/iteration_<N>/`
(`point_cloud_state_dict.pt` + `keyframes_state.pt`).

**Convert a triangle soup to a connected mesh** (restricted Delaunay):

```bash
python meshslam/run_delaunay_only.py \
  --config configs/rgbd/replica/room0.yaml \
  --state_path results/<run>/mesh/iteration_<N> \
  --out_dir meshes/room0
```



**A separate viewer with mesh editing and deformation tools:**

```bash
python scripts/view_mesh.py results/<run>/mesh/iteration_<N>
```

The viewer exposes the sigma slider, wireframe rendering, restricted-Delaunay
meshing, and the mesh editing and deformation tools described in the paper;
most of these are also available in the main GUI.

## Evaluation

```bash
# Tracking (ATE) + rendering (PSNR / SSIM / LPIPS) on Replica
python scripts/eval_replica.py --config configs/rgbd/replica/room0.yaml --state_path results/<run>/mesh/iteration_<N>

# Mesh geometry (Chamfer distance against the ground-truth TSDF)
python scripts/run_replica_eval.py --results_root results

# Depth L1
python scripts/eval_depth_l1.py ...
```

## Repository layout

```
slam.py              entry point
meshslam/            SLAM system
  frontend/          tracking
  backend/           mapping + map state
  gui/               interactive Open3D GUI + mesh editing/physics
  extract_mesh.py    offline mesh refinement + export
  run_delaunay_only.py  triangle-soup → connected mesh
mesh_splatting/      triangle model, differentiable renderer, utilities
utils/               datasets, cameras, config, eval, pose utilities
configs/             rgbd/{tum,replica}, mono/tum, live/{realsense,kinect,webcam}
scripts/             dataset download, batch runs, evaluation, viewers
submodules/          CUDA extensions (all git submodules):
  diff-triangle-mesh-rasterization   triangle rasteriser (diff-triangle-rasterization-w-pose)
  simple-knn                         k-NN (Inria)
  effrdel                            restricted Delaunay (SonSang/effrdel, MIT)
```

## Experimental: DiffSoup backend

The `diffsoup` branch contains an untuned variant of the system in which the
soft triangle-splatting rasteriser is replaced by
[DiffSoup](https://github.com/kenji-tojo/diffsoup) (Tojo et al., CVPR 2026),
a hard depth-tested rasteriser with stochastic binary opacity. See
`DIFFSOUP_PORT.md` on that branch for design notes, limitations and results;
setting `TSSLAM_RENDERER=legacy` restores the original rasteriser at runtime.
In our initial runs the extracted meshes are noticeably cleaner with this
backend.

## Citation

```bibtex
@inproceedings{fry2026triangle,
  title     = {Triangle Splatting {SLAM}},
  author    = {Fry, Nicholas and Dexheimer, Eric and Mazur, Kirill and Kelly, Paul H. J. and Davison, Andrew J.},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

Please also cite Gaussian Splatting SLAM
([MonoGS](https://github.com/muskie82/MonoGS)), on which this system builds:

```bibtex
@inproceedings{matsukimonogs,
  title={{G}aussian {S}platting {SLAM}},
  author={Hidenobu Matsuki and Riku Murai and Paul H. J. Kelly and Andrew J. Davison},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```

## Acknowledgements

This work was supported by the EPSRC On-Sensor Computer Vision grant (EP/Y020499/1). The SLAM system builds on [Gaussian Splatting SLAM](https://github.com/muskie82/MonoGS) (MonoGS): several utilities under `utils/` derive from its code, and the rasteriser adapts the pose-gradient formulation of its [`diff-gaussian-rasterization-w-pose`](https://github.com/rmurai0610/diff-gaussian-rasterization-w-pose). `mesh_splatting/` and the triangle rasteriser build on [Mesh Splatting](https://github.com/meshsplatting/mesh-splatting); `simple-knn` is from [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and `effrdel` (restricted Delaunay) is by [Sanghyun Son](https://github.com/SonSang/effrdel). See the `LICENSE*` files for upstream terms.

## License

Released under a non-commercial licence adapted from the
[MonoGS licence agreement](https://github.com/muskie82/MonoGS/blob/main/LICENSE.md)
(Imperial College London); see [`LICENSE.md`](LICENSE.md) for the full terms and the
component licences.

**Important:** this project also depends on code under the non-commercial
[Gaussian-Splatting license](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md) (`simple-knn` and the 3DGS-derived parts of the triangle rasteriser), so the released system is for **non-commercial research and evaluation use only**. See `mesh_splatting/LICENSE*` and the submodule `LICENSE*` files for upstream terms.
