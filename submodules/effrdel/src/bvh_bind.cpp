#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "lbvh.hpp"

namespace py = pybind11;

static void check_verts(py::array v){
    if (v.ndim() != 2 || v.shape(1) != 3)
        throw std::runtime_error("verts must be (N,3)");
}
static void check_faces(py::array f){
    if (f.ndim() != 2 || f.shape(1) != 3)
        throw std::runtime_error("faces must be (F,3)");
}

py::dict build_lbvhnumpy(py::array verts_np, py::array faces_np){
    check_verts(verts_np);
    check_faces(faces_np);

    // 1) copy from numpy to std::vector (float32 + int32 권장)
    const int64_t V = verts_np.shape(0);
    const int64_t F = faces_np.shape(0);

    // verts: support float32/float64
    std::vector<lbvh::Vec3f> verts;
    verts.resize((size_t)V);
    if (py::isinstance<py::array_t<float>>(verts_np)) {
        auto v = verts_np.cast<py::array_t<float>>();
        auto buf = v.unchecked<2>();
        for (int64_t i=0;i<V;++i){
            verts[(size_t)i] = lbvh::Vec3f(buf(i,0), buf(i,1), buf(i,2));
        }
    } else if (py::isinstance<py::array_t<double>>(verts_np)) {
        auto v = verts_np.cast<py::array_t<double>>();
        auto buf = v.unchecked<2>();
        for (int64_t i=0;i<V;++i){
            verts[(size_t)i] = lbvh::Vec3f((float)buf(i,0), (float)buf(i,1), (float)buf(i,2));
        }
    } else {
        throw std::runtime_error("verts dtype must be float32 or float64");
    }

    // faces: support int32/int64
    std::vector<std::array<int,3>> faces;
    faces.resize((size_t)F);
    if (py::isinstance<py::array_t<int32_t>>(faces_np)) {
        auto f = faces_np.cast<py::array_t<int32_t>>();
        auto buf = f.unchecked<2>();
        for (int64_t i=0;i<F;++i){
            faces[(size_t)i] = { buf(i,0), buf(i,1), buf(i,2) };
        }
    } else if (py::isinstance<py::array_t<int64_t>>(faces_np)) {
        auto f = faces_np.cast<py::array_t<int64_t>>();
        auto buf = f.unchecked<2>();
        for (int64_t i=0;i<F;++i){
            faces[(size_t)i] = { (int)buf(i,0), (int)buf(i,1), (int)buf(i,2) };
        }
    } else {
        throw std::runtime_error("faces dtype must be int32 or int64");
    }

    // 2) build BVH
    lbvh::BuildInput in{verts, faces};
    lbvh::BVH b = lbvh::buildLBVH(in);

    const int Nnodes = (int)b.node_min.size();
    const int Ntri   = (int)b.tri_idx_sorted.size();

    // 3) make numpy arrays (contiguous)
    auto node_min = py::array_t<float>({Nnodes, 3});
    auto node_max = py::array_t<float>({Nnodes, 3});
    auto left     = py::array_t<int32_t>({Nnodes});
    auto right    = py::array_t<int32_t>({Nnodes});
    auto parent   = py::array_t<int32_t>({Nnodes});
    auto leaf_first = py::array_t<int32_t>({Nnodes});
    auto leaf_count = py::array_t<int32_t>({Nnodes});
    auto tri_idx_sorted = py::array_t<int32_t>({Ntri});

    auto mn = node_min.mutable_unchecked<2>();
    auto mx = node_max.mutable_unchecked<2>();
    auto l  = left.mutable_unchecked<1>();
    auto r  = right.mutable_unchecked<1>();
    auto p  = parent.mutable_unchecked<1>();
    auto lf = leaf_first.mutable_unchecked<1>();
    auto lc = leaf_count.mutable_unchecked<1>();
    auto ti = tri_idx_sorted.mutable_unchecked<1>();

    for (int i=0;i<Nnodes;++i){
        mn(i,0) = b.node_min[i].x; mn(i,1) = b.node_min[i].y; mn(i,2) = b.node_min[i].z;
        mx(i,0) = b.node_max[i].x; mx(i,1) = b.node_max[i].y; mx(i,2) = b.node_max[i].z;
        l(i) = b.left[i];
        r(i) = b.right[i];
        p(i) = b.parent[i];
        lf(i) = b.leaf_first[i];
        lc(i) = b.leaf_count[i];
    }
    for (int i=0;i<Ntri;++i) ti(i) = b.tri_idx_sorted[i];

    // 4) pack dict
    py::dict out;
    out["node_min"] = node_min;
    out["node_max"] = node_max;
    out["left"]     = left;
    out["right"]    = right;
    out["parent"]   = parent;
    out["leaf_first"] = leaf_first;
    out["leaf_count"] = leaf_count;
    out["tri_idx_sorted"] = tri_idx_sorted;
    out["root"] = py::int_(b.root);
    out["n_internal"] = py::int_(b.n_internal);
    out["n_leaves"]   = py::int_(b.n_leaves);
    return out;
}

PYBIND11_MODULE(lbvh_bind, m) {
    m.doc() = "LBVH builder (triangles) returning NumPy arrays";
    m.def("build_lbvh", &build_lbvhnumpy,
          py::arg("verts"), py::arg("faces"),
          R"doc(
Build LBVH over triangles.

Args:
  verts: (V,3) float32/float64
  faces: (F,3) int32/int64

Returns dict of NumPy arrays:
  node_min (Nnodes,3) float32
  node_max (Nnodes,3) float32
  left, right, parent (Nnodes,) int32
  leaf_first, leaf_count (Nnodes,) int32
  tri_idx_sorted (F,) int32
  root, n_internal, n_leaves (ints)
)doc");
}
