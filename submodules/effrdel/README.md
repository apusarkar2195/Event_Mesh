## Efficient Restricted Delaunay Triangulation for Mesh Recovery

This repository implements **Restricted Delaunay Triangulation (RDT)** to reconstruct a connected, self-intersection–free surface from a potentially defective *triangle soup*.  
Given an input mesh that may have inconsistent connectivity or artifacts, we:

1. Build the **3D Delaunay triangulation** of the input vertices.
2. Select a **restricted subset of Delaunay faces** that best matches the surface topology.
3. Return a clean, connected triangle mesh.

The implementation is optimized with a **BVH** and **CPU/GPU vectorization**.

### Installation

This library depends on [PyTorch](https://pytorch.org/). Install a PyTorch build that matches your CUDA setup, then install this package:

```bash
pip install -e .
```

### Usage

Pass the input vertices and faces; the function returns the recovered faces:

```python
import rdel

faces = rdel.run(
    input_verts,
    input_faces,
    verbose=False,  # print timings and extra logs if True
    orient=False    # try to consistently orient face normals if True
)
```

* **`input_verts`**: NumPy array of shape `(V, 3)` with vertex positions.
* **`input_faces`**: NumPy array of shape `(F, 3)` with triangle indices describing the target topology.
* **`verbose`**: Enables timing and diagnostic prints.
* **`orient`**: Attempts to make face normals consistent.
* **Return — `faces`**: NumPy array of triangle indices into `input_verts`. The output is connected and free of self-intersections, with topology similar to `input_faces`.