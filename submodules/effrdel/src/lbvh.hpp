#pragma once
#include <vector>
#include <array>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <limits>
#include <cassert>

namespace lbvh {

// ------------------------ Small math helpers ------------------------
struct Vec3f {
    float x, y, z;
    Vec3f() : x(0), y(0), z(0) {}
    Vec3f(float X, float Y, float Z) : x(X), y(Y), z(Z) {}
    Vec3f operator+(const Vec3f& o) const { return {x+o.x, y+o.y, z+o.z}; }
    Vec3f operator-(const Vec3f& o) const { return {x-o.x, y-o.y, z-o.z}; }
    Vec3f operator*(float s) const { return {x*s, y*s, z*s}; }
};

inline Vec3f minv(const Vec3f& a, const Vec3f& b) {
    return { std::min(a.x,b.x), std::min(a.y,b.y), std::min(a.z,b.z) };
}
inline Vec3f maxv(const Vec3f& a, const Vec3f& b) {
    return { std::max(a.x,b.x), std::max(a.y,b.y), std::max(a.z,b.z) };
}

struct AABB {
    Vec3f bmin, bmax;
    AABB() {
        const float inf = std::numeric_limits<float>::infinity();
        bmin = { inf,  inf,  inf};
        bmax = {-inf, -inf, -inf};
    }
    AABB(const Vec3f& mn, const Vec3f& mx) : bmin(mn), bmax(mx) {}
    void expand(const Vec3f& p){
        bmin = minv(bmin, p);
        bmax = maxv(bmax, p);
    }
    void expand(const AABB& a){
        bmin = minv(bmin, a.bmin);
        bmax = maxv(bmax, a.bmax);
    }
    static AABB unite(const AABB& a, const AABB& b){
        AABB r; r.bmin = minv(a.bmin, b.bmin); r.bmax = maxv(a.bmax, b.bmax); return r;
    }
};

// ------------------------ Morton encoding (10 bits/axis = 30-bit key) ------------------------
inline uint32_t expandBits(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}
inline uint32_t morton3D_10bit(float x, float y, float z) {
    auto clamp01 = [](float a){ 
        // [0,1) 로 clamp (1.0f는 nextafter로 살짝 줄임)
        return std::min(std::max(a, 0.0f), std::nextafter(1.0f, 0.0f)); 
    };
    x = clamp01(x); y = clamp01(y); z = clamp01(z);
    const uint32_t xx = (uint32_t)(x * 1024.0f);
    const uint32_t yy = (uint32_t)(y * 1024.0f);
    const uint32_t zz = (uint32_t)(z * 1024.0f);
    return (expandBits(xx) << 2) | (expandBits(yy) << 1) | (expandBits(zz));
}

// count leading zeros for 32-bit
inline int clz32(uint32_t x){
#if defined(__GNUG__) || defined(__clang__)
    return x ? __builtin_clz(x) : 32;
#else
    int n = 0; while ((x & 0x80000000u)==0 && n<32){ x <<= 1; ++n; } return x ? n : 32;
#endif
}

// Longest Common Prefix between (code[i], i) and (code[j], j)
inline int commonPrefix(const std::vector<uint32_t>& code, int i, int j){
    const int n = (int)code.size();
    if (j < 0 || j >= n) return -1;
    uint32_t ci = code[i];
    uint32_t cj = code[j];
    if (ci == cj){
        uint32_t x = (uint32_t)(i ^ j);
        return 32 + clz32(x); // tie-breaker on index
    } else {
        uint32_t x = ci ^ cj;
        return clz32(x);
    }
}

// ------------------------ BVH data ------------------------
struct BVH {
    // Node arrays (size = 2*N-1 for N>1, else 1)
    std::vector<Vec3f> node_min;   // (Nnodes, 3)
    std::vector<Vec3f> node_max;   // (Nnodes, 3)
    std::vector<int>   left;       // (Nnodes)  child index or -1 if leaf
    std::vector<int>   right;      // (Nnodes)
    std::vector<int>   parent;     // (Nnodes)  -1 for root
    std::vector<int>   leaf_first; // (Nnodes)  valid if leaf: index into tri_idx_sorted
    std::vector<int>   leaf_count; // (Nnodes)  valid if leaf: =1 here

    std::vector<int>   tri_idx_sorted; // (Nprims) prim id per leaf in Morton order

    int  root = 0;
    int  n_internal = 0;
    int  n_leaves = 0;

    inline int nodeCount() const { return (int)node_min.size(); }
    inline bool isLeaf(int n) const { return left[n] < 0 && right[n] < 0; }
};

// ------------------------ Build LBVH with 1 triangle per leaf ------------------------
struct BuildInput {
    const std::vector<Vec3f>& verts;               // (V,3)
    const std::vector<std::array<int,3>>& faces;   // (F,3)
};

inline BVH buildLBVH(const BuildInput& in)
{
    const auto& V = in.verts;
    const auto& F = in.faces;
    const int N = (int)F.size();
    assert(N > 0);

    // 1) triangle AABBs, centroids & scene bounds
    std::vector<AABB> triBBox(N);
    std::vector<Vec3f> centroid(N);
    AABB scene;
    for (int i=0;i<N;++i){
        const auto& f = F[i];
        const Vec3f a = V[(size_t)f[0]];
        const Vec3f b = V[(size_t)f[1]];
        const Vec3f c = V[(size_t)f[2]];
        AABB bbi; bbi.expand(a); bbi.expand(b); bbi.expand(c);
        triBBox[i] = bbi;
        Vec3f cent = (a + b + c) * (1.0f/3.0f);
        centroid[i] = cent;
        scene.expand(bbi);
    }

    // 2) Morton codes of centroids (normalize to scene AABB)
    Vec3f smin = scene.bmin, smax = scene.bmax;
    Vec3f ext = smax - smin;
    ext.x = (std::fabs(ext.x) < 1e-20f) ? 1.0f : ext.x;
    ext.y = (std::fabs(ext.y) < 1e-20f) ? 1.0f : ext.y;
    ext.z = (std::fabs(ext.z) < 1e-20f) ? 1.0f : ext.z;

    std::vector<uint32_t> morton(N);
    std::vector<int>      primIdx(N);
    for (int i=0;i<N;++i){
        Vec3f q = {(centroid[i].x - smin.x)/ext.x,
                   (centroid[i].y - smin.y)/ext.y,
                   (centroid[i].z - smin.z)/ext.z};
        morton[i] = morton3D_10bit(q.x, q.y, q.z);
        primIdx[i] = i;
    }

    // 3) sort by (morton, primIdx) for determinism
    std::vector<int> order(N);
    for (int i=0;i<N;++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(),
        [&](int a, int b){
            if (morton[a] < morton[b]) return true;
            if (morton[a] > morton[b]) return false;
            return primIdx[a] < primIdx[b];
        });

    // apply permutation
    std::vector<uint32_t> code(N);
    std::vector<int>      primSorted(N);
    for (int i=0;i<N;++i){
        code[i]       = morton[order[i]];
        primSorted[i] = primIdx[order[i]];
    }

    // 4) allocate nodes
    const int nInternal = (N > 1) ? (N - 1) : 0;
    const int nLeaves   = N;
    const int nNodes    = nInternal + nLeaves;
    const int leafBase  = nInternal;

    BVH bvh;
    bvh.node_min.resize(nNodes);
    bvh.node_max.resize(nNodes);
    bvh.left.resize(nNodes, -1);
    bvh.right.resize(nNodes, -1);
    bvh.parent.resize(nNodes, -1);
    bvh.leaf_first.resize(nNodes, -1);
    bvh.leaf_count.resize(nNodes, 0);
    bvh.tri_idx_sorted = primSorted;
    bvh.root = 0;
    bvh.n_internal = nInternal;
    bvh.n_leaves   = nLeaves;

    // helpers: LCP & range/split (Karras 2012)
    auto lcp = [&](int i, int j)->int { return commonPrefix(code, i, j); };

    auto determineRange = [&](int i, int n, int& first, int& last){
        int lcpL = lcp(i, i-1);
        int lcpR = lcp(i, i+1);
        int d = (lcpR > lcpL) ? +1 : -1;

        int lcpMin = lcp(i, i - d);
        int lmax = 2;
        while ((i + lmax*d) >= 0 && (i + lmax*d) < n && lcp(i, i + lmax*d) > lcpMin){
            lmax *= 2;
        }
        int l = 0;
        int t = lmax;
        do {
            int step = (t + 1) >> 1;
            int idx = i + (l + step)*d;
            if (idx >= 0 && idx < n && lcp(i, idx) > lcpMin){
                l += step;
            }
            t -= step;
        } while (t > 0);
        last  = i + l*d;
        first = std::min(i, last);
        last  = std::max(i, last);
    };

    auto findSplit = [&](int first, int last)->int{
        int lcpFirstLast = lcp(first, last);
        if (lcpFirstLast >= 64) return (first + last) >> 1; // all equal (rare)
        int split = first;
        int step = last - first;
        do {
            step = (step + 1) >> 1;
            int newSplit = split + step;
            if (newSplit < last){
                int lcpFirstNew = lcp(first, newSplit);
                if (lcpFirstNew > lcpFirstLast){
                    split = newSplit;
                }
            }
        } while (step > 1);
        return split;
    };

    // 5) set topology
    if (N == 1){
        // single leaf at node 0
        bvh.leaf_first[0] = 0;
        bvh.leaf_count[0] = 1;
    } else {
        for (int i=0; i<nInternal; ++i){
            int first, last;
            determineRange(i, N, first, last);
            int split = findSplit(first, last);

            int leftChild  = (split == first) ? (leafBase + split) : split;
            int rightChild = (split + 1 == last) ? (leafBase + split + 1) : (split + 1);

            bvh.left[i]  = leftChild;
            bvh.right[i] = rightChild;
            bvh.parent[leftChild]  = i;
            bvh.parent[rightChild] = i;
        }
        for (int i=0;i<nLeaves;++i){
            int leaf = leafBase + i;
            bvh.leaf_first[leaf] = i;  // index into tri_idx_sorted
            bvh.leaf_count[leaf] = 1;
        }
    }

    // 6) compute node AABBs (bottom-up)
    if (N == 1){
        int tri  = bvh.tri_idx_sorted[0];
        bvh.node_min[0] = triBBox[tri].bmin;
        bvh.node_max[0] = triBBox[tri].bmax;
    } else {
        for (int i=0;i<nLeaves;++i){
            int leaf = leafBase + i;
            int tri  = bvh.tri_idx_sorted[i];
            bvh.node_min[leaf] = triBBox[tri].bmin;
            bvh.node_max[leaf] = triBBox[tri].bmax;
        }
        std::vector<int> pending(nInternal, 2);
        std::vector<int> stack; stack.reserve(nInternal);
        for (int i=0;i<nLeaves;++i){
            int p = bvh.parent[leafBase + i];
            if (p >= 0){
                if (--pending[p] == 0) stack.push_back(p);
            }
        }
        while (!stack.empty()){
            int n = stack.back(); stack.pop_back();
            int L = bvh.left[n];
            int R = bvh.right[n];
            AABB a(bvh.node_min[L], bvh.node_max[L]);
            a.expand(AABB(bvh.node_min[R], bvh.node_max[R]));
            bvh.node_min[n] = a.bmin;
            bvh.node_max[n] = a.bmax;

            int p = bvh.parent[n];
            if (p >= 0){
                if (--pending[p] == 0) stack.push_back(p);
            }
        }
    }

    bvh.root = 0;
    return bvh;
}

} // namespace lbvh
