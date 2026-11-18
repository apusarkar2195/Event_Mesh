import gstaichi as ti
import numpy as np
import torch as th

# --------------------
# Initialize
# --------------------
def ti_init():
    try:
        ti.init(arch=ti.cuda, default_fp=ti.f32)
    except Exception:
        ti.init(arch=ti.cpu,  default_fp=ti.f32)

# --------------------
# Global fields (BVH / Mesh)
# --------------------
node_min  = None  # ti.Vector.field(3, ti.f32)
node_max  = None
left      = None  # ti.field(ti.i32)
right     = None
leaf_first= None  # ti.field(ti.i32)
leaf_count= None
tri_idx_sorted = None  # ti.field(ti.i32)

verts_f   = None  # ti.Vector.field(3, ti.f32)
faces_i   = None  # ti.Vector.field(3, ti.i32)

root_idx  = 0

# --------------------
# Upload/Assign
# --------------------
def upload_bvh_and_mesh(bvh: dict, verts: np.ndarray, faces: np.ndarray):
    """
    bvh: rdel.lbvh_bind.build_lbvh(...) 반환 dict
    verts: (V,3) float32/float64
    faces: (F,3) int32/int64
    """
    global node_min, node_max, left, right, leaf_first, leaf_count, tri_idx_sorted
    global verts_f, faces_i, root_idx

    Nn = int(bvh["node_min"].shape[0])
    V  = int(verts.shape[0])
    F  = int(faces.shape[0])

    v_np = verts.astype(np.float32, copy=False)
    f_np = faces.astype(np.int32,   copy=False)

    node_min   = ti.Vector.field(3, dtype=ti.f32, shape=Nn)
    node_max   = ti.Vector.field(3, dtype=ti.f32, shape=Nn)
    left       = ti.field(dtype=ti.i32, shape=Nn)
    right      = ti.field(dtype=ti.i32, shape=Nn)
    leaf_first = ti.field(dtype=ti.i32, shape=Nn)
    leaf_count = ti.field(dtype=ti.i32, shape=Nn)
    tri_idx_sorted = ti.field(dtype=ti.i32, shape=int(bvh["tri_idx_sorted"].shape[0]))

    verts_f = ti.Vector.field(3, dtype=ti.f32, shape=V)
    faces_i = ti.Vector.field(3, dtype=ti.i32, shape=F)

    node_min.from_numpy(bvh["node_min"].astype(np.float32, copy=False))
    node_max.from_numpy(bvh["node_max"].astype(np.float32, copy=False))
    left.from_numpy(bvh["left"].astype(np.int32, copy=False))
    right.from_numpy(bvh["right"].astype(np.int32, copy=False))
    leaf_first.from_numpy(bvh["leaf_first"].astype(np.int32, copy=False))
    leaf_count.from_numpy(bvh["leaf_count"].astype(np.int32, copy=False))
    tri_idx_sorted.from_numpy(bvh["tri_idx_sorted"].astype(np.int32, copy=False))

    verts_f.from_numpy(v_np)
    faces_i.from_numpy(f_np)

    root_idx = int(bvh["root"])

# --------------------
# Intersect Utility (@ti.func)
# --------------------
@ti.func
def seg_aabb_hit(p0: ti.types.vector(3, ti.f32),
                 p1: ti.types.vector(3, ti.f32),
                 bmin: ti.types.vector(3, ti.f32),
                 bmax: ti.types.vector(3, ti.f32)) -> ti.i32:
    # Slab Test: Segment (t in [0,1])
    d  = p1 - p0
    tmin = ti.cast(0.0, ti.f32)
    tmax = ti.cast(1.0, ti.f32)
    eps = 1e-9
    hit = 1
    for k in ti.static(range(3)):
        if ti.abs(d[k]) < eps:
            # Direction 0: p0 is out of range
            if (p0[k] < bmin[k]) or (p0[k] > bmax[k]):
                hit = 0
        else:
            invd = 1.0 / d[k]
            t0 = (bmin[k] - p0[k]) * invd
            t1 = (bmax[k] - p0[k]) * invd
            lo = ti.min(t0, t1)
            hi = ti.max(t0, t1)
            tmin = ti.max(tmin, lo)
            tmax = ti.min(tmax, hi)
            if tmin > tmax:
                hit = 0
    return hit

@ti.func
def seg_tri_hit(p0: ti.types.vector(3, ti.f32),
                p1: ti.types.vector(3, ti.f32),
                v0: ti.types.vector(3, ti.f32),
                v1: ti.types.vector(3, ti.f32),
                v2: ti.types.vector(3, ti.f32)) -> ti.i32:
    result = 0
    # Möller–Trumbore (Boundary included)
    eps = 1e-9
    dir = p1 - p0
    e1 = v1 - v0
    e2 = v2 - v0
    pvec = dir.cross(e2)
    det = e1.dot(pvec)
    if ti.abs(det) > eps:
        inv_det = 1.0 / det
        tvec = p0 - v0
        u = tvec.dot(pvec) * inv_det
        qvec = tvec.cross(e1)
        v = dir.dot(qvec) * inv_det
        t = e2.dot(qvec) * inv_det

        # Boundary included: Allow a little negative
        if (u >= -eps) and (v >= -eps) and (u + v <= 1.0 + eps) and (t >= -eps) and (t <= 1.0 + eps):
            # Degenerate filter
            area2 = e1.cross(e2).norm_sqr()
            seg2  = dir.norm_sqr()
            if (area2 > eps*eps) and (seg2 > eps*eps):
                result = 1
    return result

# --------------------
# Kernel: ANY-HIT (Segment Set)
# --------------------
MAX_STACK = 64  # Tree depth enough (increase if needed)

@ti.kernel
def segments_any_hit_bvh(P0: ti.types.ndarray(dtype=ti.f32, ndim=2),
                         P1: ti.types.ndarray(dtype=ti.f32, ndim=2),
                         out_hits: ti.types.ndarray(dtype=ti.i8, ndim=1)):
    E = P0.shape[0]
    for e in range(E):
        p0 = ti.Vector([P0[e, 0], P0[e, 1], P0[e, 2]])
        p1 = ti.Vector([P1[e, 0], P1[e, 1], P1[e, 2]])

        hit = ti.i32(0)

        # Fast AABB Test (Root and Segment)
        if seg_aabb_hit(p0, p1, node_min[root_idx], node_max[root_idx]) == 1:
            # per-thread small stack
            stack = ti.Vector.zero(ti.i32, MAX_STACK)
            sp = 0
            stack[sp] = root_idx
            sp += 1

            while sp > 0 and hit == 0:
                sp -= 1
                n = stack[sp]

                if seg_aabb_hit(p0, p1, node_min[n], node_max[n]) == 0:
                    continue

                if left[n] < 0 and right[n] < 0:
                    # Leaf
                    first = leaf_first[n]
                    count = leaf_count[n]
                    for i in range(count):
                        tri_id = tri_idx_sorted[first + i]
                        f = faces_i[tri_id]
                        v0 = verts_f[f[0]]
                        v1 = verts_f[f[1]]
                        v2 = verts_f[f[2]]
                        if seg_tri_hit(p0, p1, v0, v1, v2) == 1:
                            hit = 1
                            break
                else:
                    # Internal node → child push (order doesn't matter / if you want, push the closer one first)
                    l = left[n]
                    r = right[n]
                    # Stack overflow prevention (normal depth doesn't trigger)
                    if sp + 2 <= MAX_STACK - 1:
                        stack[sp] = l; sp += 1
                        stack[sp] = r; sp += 1
                    else:
                        # Very rarely deep cases: Conservatively push one only
                        stack[sp] = l; sp += 1

        out_hits[e] = ti.i8(1) if hit == 1 else ti.i8(0)

# --------------------
# Convenience function: Batch processing
# --------------------
def query_segments_any_hit(P0: th.Tensor, P1: th.Tensor, batch_edges: int = 2_000_000) -> th.Tensor:
    E = P0.shape[0]
    hits = th.zeros((E,), dtype=th.bool)
    s = 0
    while s < E:
        e = min(E, s + batch_edges)
        P0_sub = P0[s:e].to(dtype=th.float32)
        P1_sub = P1[s:e].to(dtype=th.float32)
        out = th.zeros((e - s,), dtype=th.int8)

        segments_any_hit_bvh(P0_sub, P1_sub, out)  # Call kernel
        hits[s:e] = (out > 0).to(dtype=th.bool)
        s = e
    return hits
