# Differentiable Triangle Rasterization (with pose optimisation)

Differentiable triangle rasteriser used by
**Triangle Splatting SLAM** ([code](https://github.com/nmjfry/ts-slam),
[project page](https://nmjfry.github.io/triangle-splatting-slam/)).

This is a fork of the [Mesh Splatting](https://github.com/meshsplatting/mesh-splatting)
triangle rasteriser, with added support for SLAM. The loss is back-propagated to
the **camera pose** (6-DoF) and to the **vertex world positions** and
**per-vertex opacity**, so the same rasteriser can be used for both tracking and
mapping. Triangle opacity is the **mean** of its three vertex weights.

The camera-pose gradient follows the analytic SE(3) Jacobian formulation of
[Gaussian Splatting SLAM](https://github.com/muskie82/MonoGS) (MonoGS), and
adapts code from its
[`diff-gaussian-rasterization-w-pose`](https://github.com/rmurai0610/diff-gaussian-rasterization-w-pose)
rasteriser (in particular `cuda_rasterizer/math.h` and the pose backward pass).

If you use this rasteriser in your own research, please cite the works below.

## BibTeX

MeshSplatting, from which this rasteriser is forked:

```bibtex
@article{Held2025MeshSplatting,
title = {MeshSplatting: Differentiable Rendering with Opaque Meshes},
author = {Held, Jan and Son, Sanghyun and Vandeghen, Renaud and Rebain, Daniel and Gadelha, Matheus and Zhou, Yi and Cioppa, Anthony and G Lin, Ming C. and Van Droogenbroeck, Marc and Tagliasacchi, Andrea},
journal = {arXiv},
year = {2025}
}
```

3D Gaussian Splatting, on which the CUDA rasterisation pipeline is based:

```bibtex
@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```

Gaussian Splatting SLAM, from which the camera-pose gradient formulation and
parts of the pose backward pass are adapted:

```bibtex
@inproceedings{Matsuki:Murai:etal:CVPR2024,
  title={{G}aussian {S}platting {SLAM}},
  author={Hidenobu Matsuki and Riku Murai and Paul H. J. Kelly and Andrew J. Davison},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```

Triangle Splatting SLAM, for which the pose optimisation was developed:

```bibtex
@inproceedings{fry2026triangle,
  title     = {Triangle Splatting {SLAM}},
  author    = {Fry, Nicholas and Dexheimer, Eric and Mazur, Kirill and Kelly, Paul H. J. and Davison, Andrew J.},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```