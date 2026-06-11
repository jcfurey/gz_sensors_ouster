// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#include "raycast_scene.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <numeric>
#include <vector>

namespace gz_gpu_ouster_lidar {
namespace rc {

namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr int kLeafTris = 4;

struct TriBound {
    float bmin[3], bmax[3], centroid[3];
};

/// Recursive median-split build. `order` entries and node child indices are
/// GLOBAL (the mesh occupies [first_order, first_order+count) in the order
/// array and appends nodes to the global node vector), so traversal needs
/// no per-mesh offsets.
int buildBvhNode(std::vector<MeshBvhNode> & nodes, std::vector<int> & order,
                 const std::vector<TriBound> & tb, int tri_base,
                 int first_order, int count, int depth)
{
    const int node_idx = static_cast<int>(nodes.size());
    nodes.emplace_back();
    {
        MeshBvhNode & n = nodes.back();
        for (int a = 0; a < 3; ++a) {
            n.bmin[a] = kInf;
            n.bmax[a] = -kInf;
        }
        for (int i = first_order; i < first_order + count; ++i) {
            const TriBound & b =
                tb[static_cast<size_t>(order[i] - tri_base)];
            for (int a = 0; a < 3; ++a) {
                n.bmin[a] = std::min(n.bmin[a], b.bmin[a]);
                n.bmax[a] = std::max(n.bmax[a], b.bmax[a]);
            }
        }
    }

    if (count <= kLeafTris || depth > 48) {
        MeshBvhNode & n = nodes[static_cast<size_t>(node_idx)];
        n.first = first_order;
        n.count = count;
        return node_idx;
    }

    // Median split on the longest axis of the node bounds.
    int axis = 0;
    {
        const MeshBvhNode & n = nodes[static_cast<size_t>(node_idx)];
        const float ext[3] = {n.bmax[0] - n.bmin[0], n.bmax[1] - n.bmin[1],
                              n.bmax[2] - n.bmin[2]};
        if (ext[1] > ext[axis]) axis = 1;
        if (ext[2] > ext[axis]) axis = 2;
    }
    const int mid = first_order + count / 2;
    std::nth_element(
        order.begin() + first_order, order.begin() + mid,
        order.begin() + first_order + count,
        [&tb, tri_base, axis](int a, int b) {
            return tb[static_cast<size_t>(a - tri_base)].centroid[axis] <
                   tb[static_cast<size_t>(b - tri_base)].centroid[axis];
        });

    // Children are built after this node; indices recorded afterwards
    // (the node vector may reallocate during recursion, so don't hold a
    // reference across the recursive calls).
    const int left = buildBvhNode(nodes, order, tb, tri_base,
                                  first_order, count / 2, depth + 1);
    const int right = buildBvhNode(nodes, order, tb, tri_base,
                                   mid, count - count / 2, depth + 1);
    nodes[static_cast<size_t>(node_idx)].left = left;
    nodes[static_cast<size_t>(node_idx)].right = right;
    return node_idx;
}

}  // namespace

int Scene::addMesh(const std::vector<float> & verts,
                   const std::vector<int> & tris)
{
    const int n_tris = static_cast<int>(tris.size()) / 3;
    if (n_tris == 0 || verts.size() < 9) return -1;

    const int vert_base = static_cast<int>(verts_.size()) / 3;
    const int tri_base = static_cast<int>(tris_.size()) / 3;
    const int order_base = static_cast<int>(order_.size());

    verts_.insert(verts_.end(), verts.begin(), verts.end());
    tris_.reserve(tris_.size() + tris.size());
    std::transform(tris.begin(), tris.end(), std::back_inserter(tris_),
                   [vert_base](int vi) { return vi + vert_base; });

    std::vector<TriBound> tb(static_cast<size_t>(n_tris));
    for (int i = 0; i < n_tris; ++i) {
        TriBound & b = tb[static_cast<size_t>(i)];
        for (int a = 0; a < 3; ++a) {
            b.bmin[a] = kInf;
            b.bmax[a] = -kInf;
        }
        for (int k = 0; k < 3; ++k) {
            const int vi = tris[static_cast<size_t>(3 * i + k)];
            for (int a = 0; a < 3; ++a) {
                const float c = verts[static_cast<size_t>(3 * vi + a)];
                b.bmin[a] = std::min(b.bmin[a], c);
                b.bmax[a] = std::max(b.bmax[a], c);
            }
        }
        for (int a = 0; a < 3; ++a) {
            b.centroid[a] = 0.5f * (b.bmin[a] + b.bmax[a]);
        }
    }

    order_.resize(static_cast<size_t>(order_base + n_tris));
    std::iota(order_.begin() + order_base, order_.end(), tri_base);

    const int root = buildBvhNode(nodes_, order_, tb, tri_base,
                                  order_base, n_tris, 0);
    ++mesh_count_;
    return root;
}

int Scene::addInstance(GeomType type, const float size[3], float retro,
                       int root_node, float spec, float transmit)
{
    RcInstance inst;
    inst.type = type;
    inst.size[0] = size[0];
    inst.size[1] = size[1];
    inst.size[2] = size[2];
    inst.retro = retro;
    inst.spec = (spec > 0.0f) ? spec : 0.0f;
    inst.transmit = (transmit > 0.0f)
        ? ((transmit < 1.0f) ? transmit : 1.0f) : 0.0f;
    inst.root_node = (type == GeomType::kMesh) ? root_node : -1;

    LocalBounds lb{};
    switch (type) {
        case GeomType::kPlane: {
            const float hx = (size[0] > 0.0f) ? size[0] : kRcHugeExtent;
            const float hy = (size[1] > 0.0f) ? size[1] : kRcHugeExtent;
            lb.bmin[0] = -hx; lb.bmax[0] = hx;
            lb.bmin[1] = -hy; lb.bmax[1] = hy;
            lb.bmin[2] = -1.0e-3f; lb.bmax[2] = 1.0e-3f;
            break;
        }
        case GeomType::kBox:
            for (int a = 0; a < 3; ++a) {
                lb.bmin[a] = -size[a];
                lb.bmax[a] = size[a];
            }
            break;
        case GeomType::kSphere:
            for (int a = 0; a < 3; ++a) {
                lb.bmin[a] = -size[0];
                lb.bmax[a] = size[0];
            }
            break;
        case GeomType::kCylinder:
            lb.bmin[0] = -size[0]; lb.bmax[0] = size[0];
            lb.bmin[1] = -size[0]; lb.bmax[1] = size[0];
            lb.bmin[2] = -size[1]; lb.bmax[2] = size[1];
            break;
        case GeomType::kMesh:
            if (root_node >= 0 &&
                root_node < static_cast<int>(nodes_.size())) {
                const MeshBvhNode & n =
                    nodes_[static_cast<size_t>(root_node)];
                for (int a = 0; a < 3; ++a) {
                    lb.bmin[a] = n.bmin[a];
                    lb.bmax[a] = n.bmax[a];
                }
            }
            break;
    }

    instances_.push_back(inst);
    bounds_.push_back(lb);
    return static_cast<int>(instances_.size()) - 1;
}

SceneView Scene::view() const
{
    SceneView v;
    v.instances = instances_.data();
    v.n_instances = static_cast<int>(instances_.size());
    v.verts = verts_.data();
    v.n_vert_floats = static_cast<int>(verts_.size());
    v.tris = tris_.data();
    v.n_tri_ints = static_cast<int>(tris_.size());
    v.order = order_.data();
    v.n_order = static_cast<int>(order_.size());
    v.nodes = nodes_.data();
    v.n_nodes = static_cast<int>(nodes_.size());
    return v;
}

void Scene::computeXform(int instance_idx,
                         const float r_lw[9], const float t_lw[3],
                         InstanceXform & out) const
{
    // world→local = transpose(r_lw), t = -r_lw^T · t_lw.
    out.r[0] = r_lw[0]; out.r[1] = r_lw[3]; out.r[2] = r_lw[6];
    out.r[3] = r_lw[1]; out.r[4] = r_lw[4]; out.r[5] = r_lw[7];
    out.r[6] = r_lw[2]; out.r[7] = r_lw[5]; out.r[8] = r_lw[8];
    for (int a = 0; a < 3; ++a) {
        out.t[a] = -(out.r[3 * a] * t_lw[0] + out.r[3 * a + 1] * t_lw[1] +
                     out.r[3 * a + 2] * t_lw[2]);
    }

    // World AABB from the 8 transformed local corners.
    const LocalBounds & lb = bounds_[static_cast<size_t>(instance_idx)];
    for (int a = 0; a < 3; ++a) {
        out.bmin[a] = kInf;
        out.bmax[a] = -kInf;
    }
    for (int corner = 0; corner < 8; ++corner) {
        const RcV3 c{(corner & 1) ? lb.bmax[0] : lb.bmin[0],
                     (corner & 2) ? lb.bmax[1] : lb.bmin[1],
                     (corner & 4) ? lb.bmax[2] : lb.bmin[2]};
        const RcV3 p = rcXformPoint(r_lw, t_lw, c);
        out.bmin[0] = std::min(out.bmin[0], p.x);
        out.bmin[1] = std::min(out.bmin[1], p.y);
        out.bmin[2] = std::min(out.bmin[2], p.z);
        out.bmax[0] = std::max(out.bmax[0], p.x);
        out.bmax[1] = std::max(out.bmax[1], p.y);
        out.bmax[2] = std::max(out.bmax[2], p.z);
    }
}

void castScan(const SceneView & scene,
              const InstanceXform * xforms,
              const float * beam_alt_deg,
              const float * beam_az_deg,
              const float sensor_r[9], const float sensor_t[3],
              const ScanParams & sp,
              float * range_out, float * retro_out,
              const float * col_r, const float * col_t,
              float * nir_out)
{
    const int n = sp.H * sp.W;

    #pragma omp parallel for schedule(static) if(n > 4096)  // NOLINT(whitespace/parens)
    for (int idx = 0; idx < n; ++idx) {
        float range, retro;
        rcCastOneRay(scene.instances, scene.n_instances,
                     scene.verts, scene.tris, scene.order, scene.nodes,
                     xforms, beam_alt_deg, beam_az_deg,
                     sensor_r, sensor_t, sp, idx, kInf, range, retro,
                     col_r, col_t,
                     nir_out ? &nir_out[idx] : nullptr);
        range_out[idx] = range;
        if (retro_out) retro_out[idx] = retro;
    }
}

}  // namespace rc
}  // namespace gz_gpu_ouster_lidar
