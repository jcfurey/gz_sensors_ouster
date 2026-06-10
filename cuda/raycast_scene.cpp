// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#include "raycast_scene.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>

namespace gz_gpu_ouster_lidar {
namespace rc {

namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr float kPi = 3.14159265358979f;
constexpr float kTriEps = 1.0e-8f;
constexpr int kLeafTris = 4;
// "Infinite" plane half-extent: large enough for any practical world while
// keeping AABB corner transforms finite.
constexpr float kHugeExtent = 1.0e6f;

struct V3 {
    float x, y, z;
};

inline V3 sub(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline V3 cross(V3 a, V3 b)
{
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

/// Apply a row-major rotation + translation: r·p + t.
inline V3 xformPoint(const float r[9], const float t[3], V3 p)
{
    return {r[0] * p.x + r[1] * p.y + r[2] * p.z + t[0],
            r[3] * p.x + r[4] * p.y + r[5] * p.z + t[1],
            r[6] * p.x + r[7] * p.y + r[8] * p.z + t[2]};
}

inline V3 rotate(const float r[9], V3 p)
{
    return {r[0] * p.x + r[1] * p.y + r[2] * p.z,
            r[3] * p.x + r[4] * p.y + r[5] * p.z,
            r[6] * p.x + r[7] * p.y + r[8] * p.z};
}

/// Ray/AABB slab test on [tmin, tmax]. Robust to zero direction components
/// (division yields ±inf which the min/max ordering handles).
inline bool hitAabb(V3 o, V3 inv_d, const float bmin[3], const float bmax[3],
                    float tmin, float tmax)
{
    float t0 = (bmin[0] - o.x) * inv_d.x;
    float t1 = (bmax[0] - o.x) * inv_d.x;
    float lo = std::min(t0, t1), hi = std::max(t0, t1);
    t0 = (bmin[1] - o.y) * inv_d.y;
    t1 = (bmax[1] - o.y) * inv_d.y;
    lo = std::max(lo, std::min(t0, t1));
    hi = std::min(hi, std::max(t0, t1));
    t0 = (bmin[2] - o.z) * inv_d.z;
    t1 = (bmax[2] - o.z) * inv_d.z;
    lo = std::max(lo, std::min(t0, t1));
    hi = std::min(hi, std::max(t0, t1));
    return lo <= hi && hi >= tmin && lo <= tmax;
}

/// Möller–Trumbore. Returns hit parameter in (tmin, tmax) or -1.
inline float hitTriangle(V3 o, V3 d, V3 v0, V3 v1, V3 v2,
                         float tmin, float tmax)
{
    const V3 e1 = sub(v1, v0);
    const V3 e2 = sub(v2, v0);
    const V3 p = cross(d, e2);
    const float det = dot(e1, p);
    if (std::fabs(det) < kTriEps) return -1.0f;
    const float inv_det = 1.0f / det;
    const V3 s = sub(o, v0);
    const float u = dot(s, p) * inv_det;
    if (u < 0.0f || u > 1.0f) return -1.0f;
    const V3 q = cross(s, e1);
    const float v = dot(d, q) * inv_det;
    if (v < 0.0f || u + v > 1.0f) return -1.0f;
    const float t = dot(e2, q) * inv_det;
    return (t > tmin && t < tmax) ? t : -1.0f;
}

inline float hitSphere(V3 o, V3 d, float radius, float tmin, float tmax)
{
    const float b = dot(o, d);
    const float c = dot(o, o) - radius * radius;
    const float disc = b * b - c;
    if (disc < 0.0f) return -1.0f;
    const float sq = std::sqrt(disc);
    float t = -b - sq;
    if (t <= tmin) t = -b + sq;
    return (t > tmin && t < tmax) ? t : -1.0f;
}

inline float hitBox(V3 o, V3 d, const float h[3], float tmin, float tmax)
{
    const V3 inv{1.0f / d.x, 1.0f / d.y, 1.0f / d.z};
    const float bmin[3] = {-h[0], -h[1], -h[2]};
    const float bmax[3] = {h[0], h[1], h[2]};
    float t0 = (bmin[0] - o.x) * inv.x;
    float t1 = (bmax[0] - o.x) * inv.x;
    float lo = std::min(t0, t1), hi = std::max(t0, t1);
    t0 = (bmin[1] - o.y) * inv.y;
    t1 = (bmax[1] - o.y) * inv.y;
    lo = std::max(lo, std::min(t0, t1));
    hi = std::min(hi, std::max(t0, t1));
    t0 = (bmin[2] - o.z) * inv.z;
    t1 = (bmax[2] - o.z) * inv.z;
    lo = std::max(lo, std::min(t0, t1));
    hi = std::min(hi, std::max(t0, t1));
    if (lo > hi) return -1.0f;
    // Entry point if outside the box, exit point if inside.
    const float t = (lo > tmin) ? lo : hi;
    return (t > tmin && t < tmax) ? t : -1.0f;
}

inline float hitCylinder(V3 o, V3 d, float radius, float half_len,
                         float tmin, float tmax)
{
    float best = -1.0f;
    // Lateral surface: quadratic in the xy plane.
    const float a = d.x * d.x + d.y * d.y;
    if (a > 1.0e-12f) {
        const float b = (o.x * d.x + o.y * d.y) / a;
        const float c = (o.x * o.x + o.y * o.y - radius * radius) / a;
        const float disc = b * b - c;
        if (disc >= 0.0f) {
            const float sq = std::sqrt(disc);
            for (const float t : {-b - sq, -b + sq}) {
                if (t <= tmin || t >= tmax) continue;
                const float z = o.z + t * d.z;
                if (std::fabs(z) <= half_len &&
                    (best < 0.0f || t < best)) {
                    best = t;
                    break;  // candidates are ordered; first valid is nearest
                }
            }
        }
    }
    // End caps.
    if (std::fabs(d.z) > 1.0e-12f) {
        for (const float zc : {half_len, -half_len}) {
            const float t = (zc - o.z) / d.z;
            if (t <= tmin || t >= tmax) continue;
            const float x = o.x + t * d.x;
            const float y = o.y + t * d.y;
            if (x * x + y * y <= radius * radius &&
                (best < 0.0f || t < best)) {
                best = t;
            }
        }
    }
    return best;
}

inline float hitPlane(V3 o, V3 d, const float size[3], float tmin, float tmax)
{
    if (std::fabs(d.z) < 1.0e-12f) return -1.0f;
    const float t = -o.z / d.z;
    if (t <= tmin || t >= tmax) return -1.0f;
    const float hx = (size[0] > 0.0f) ? size[0] : kHugeExtent;
    const float hy = (size[1] > 0.0f) ? size[1] : kHugeExtent;
    const float x = o.x + t * d.x;
    const float y = o.y + t * d.y;
    return (std::fabs(x) <= hx && std::fabs(y) <= hy) ? t : -1.0f;
}

inline float hitMesh(const MeshData & m, V3 o, V3 d, float tmin, float tmax)
{
    if (m.nodes.empty()) return -1.0f;
    const V3 inv{1.0f / d.x, 1.0f / d.y, 1.0f / d.z};
    float best = -1.0f;
    float limit = tmax;
    int stack[64];
    int sp = 0;
    stack[sp++] = 0;
    while (sp > 0) {
        const MeshBvhNode & n = m.nodes[stack[--sp]];
        if (!hitAabb(o, inv, n.bmin, n.bmax, tmin, limit)) continue;
        if (n.left < 0) {
            for (int k = 0; k < n.count; ++k) {
                const int tri = m.order[n.first + k];
                const int * idx = &m.tris[3 * tri];
                const V3 v0{m.verts[3 * idx[0]], m.verts[3 * idx[0] + 1],
                            m.verts[3 * idx[0] + 2]};
                const V3 v1{m.verts[3 * idx[1]], m.verts[3 * idx[1] + 1],
                            m.verts[3 * idx[1] + 2]};
                const V3 v2{m.verts[3 * idx[2]], m.verts[3 * idx[2] + 1],
                            m.verts[3 * idx[2] + 2]};
                const float t = hitTriangle(o, d, v0, v1, v2, tmin, limit);
                if (t > 0.0f) {
                    best = t;
                    limit = t;
                }
            }
        } else if (sp + 2 <= 64) {
            stack[sp++] = n.left;
            stack[sp++] = n.right;
        }
    }
    return best;
}

inline float hitInstance(const Instance & inst, const Scene & scene,
                         V3 o, V3 d, float tmin, float tmax)
{
    switch (inst.type) {
        case GeomType::kPlane:    return hitPlane(o, d, inst.size, tmin, tmax);
        case GeomType::kBox:      return hitBox(o, d, inst.size, tmin, tmax);
        case GeomType::kSphere:   return hitSphere(o, d, inst.size[0], tmin, tmax);
        case GeomType::kCylinder:
            return hitCylinder(o, d, inst.size[0], inst.size[1], tmin, tmax);
        case GeomType::kMesh:
            if (inst.mesh < 0 ||
                inst.mesh >= static_cast<int>(scene.meshes.size())) {
                return -1.0f;
            }
            return hitMesh(scene.meshes[inst.mesh], o, d, tmin, tmax);
    }
    return -1.0f;
}

// ── BVH construction ─────────────────────────────────────────────────────────

struct TriBound {
    float bmin[3], bmax[3], centroid[3];
};

int buildBvhNode(MeshData & m, std::vector<TriBound> & tb,
                 int first, int count, int depth)
{
    const int node_idx = static_cast<int>(m.nodes.size());
    m.nodes.emplace_back();
    {
        MeshBvhNode & n = m.nodes.back();
        for (int a = 0; a < 3; ++a) {
            n.bmin[a] = kInf;
            n.bmax[a] = -kInf;
        }
        for (int i = first; i < first + count; ++i) {
            const TriBound & b = tb[m.order[i]];
            for (int a = 0; a < 3; ++a) {
                n.bmin[a] = std::min(n.bmin[a], b.bmin[a]);
                n.bmax[a] = std::max(n.bmax[a], b.bmax[a]);
            }
        }
    }

    if (count <= kLeafTris || depth > 48) {
        MeshBvhNode & n = m.nodes[node_idx];
        n.first = first;
        n.count = count;
        return node_idx;
    }

    // Median split on the longest axis of the node bounds.
    int axis = 0;
    {
        const MeshBvhNode & n = m.nodes[node_idx];
        const float ext[3] = {n.bmax[0] - n.bmin[0], n.bmax[1] - n.bmin[1],
                              n.bmax[2] - n.bmin[2]};
        if (ext[1] > ext[axis]) axis = 1;
        if (ext[2] > ext[axis]) axis = 2;
    }
    const int mid = first + count / 2;
    std::nth_element(
        m.order.begin() + first, m.order.begin() + mid,
        m.order.begin() + first + count,
        [&tb, axis](int a, int b) {
            return tb[a].centroid[axis] < tb[b].centroid[axis];
        });

    // Children are built after this node; indices recorded afterwards
    // (m.nodes may reallocate during recursion, so don't hold a reference).
    const int left = buildBvhNode(m, tb, first, count / 2, depth + 1);
    const int right = buildBvhNode(m, tb, mid, count - count / 2, depth + 1);
    m.nodes[node_idx].left = left;
    m.nodes[node_idx].right = right;
    return node_idx;
}

}  // namespace

bool buildMeshBvh(MeshData & mesh)
{
    const int n_tris = static_cast<int>(mesh.tris.size()) / 3;
    if (n_tris == 0 || mesh.verts.size() < 9) return false;

    std::vector<TriBound> tb(static_cast<size_t>(n_tris));
    for (int i = 0; i < n_tris; ++i) {
        TriBound & b = tb[static_cast<size_t>(i)];
        for (int a = 0; a < 3; ++a) {
            b.bmin[a] = kInf;
            b.bmax[a] = -kInf;
        }
        for (int k = 0; k < 3; ++k) {
            const int vi = mesh.tris[3 * i + k];
            for (int a = 0; a < 3; ++a) {
                const float c = mesh.verts[3 * vi + a];
                b.bmin[a] = std::min(b.bmin[a], c);
                b.bmax[a] = std::max(b.bmax[a], c);
            }
        }
        for (int a = 0; a < 3; ++a) {
            b.centroid[a] = 0.5f * (b.bmin[a] + b.bmax[a]);
        }
    }

    mesh.order.resize(static_cast<size_t>(n_tris));
    std::iota(mesh.order.begin(), mesh.order.end(), 0);
    mesh.nodes.clear();
    mesh.nodes.reserve(static_cast<size_t>(2 * n_tris));
    buildBvhNode(mesh, tb, 0, n_tris, 0);

    for (int a = 0; a < 3; ++a) {
        mesh.bmin[a] = mesh.nodes[0].bmin[a];
        mesh.bmax[a] = mesh.nodes[0].bmax[a];
    }
    return true;
}

void finalizeInstance(const Scene & scene, Instance & inst)
{
    switch (inst.type) {
        case GeomType::kPlane: {
            const float hx = (inst.size[0] > 0.0f) ? inst.size[0] : kHugeExtent;
            const float hy = (inst.size[1] > 0.0f) ? inst.size[1] : kHugeExtent;
            inst.local_bmin[0] = -hx; inst.local_bmax[0] = hx;
            inst.local_bmin[1] = -hy; inst.local_bmax[1] = hy;
            inst.local_bmin[2] = -1.0e-3f; inst.local_bmax[2] = 1.0e-3f;
            break;
        }
        case GeomType::kBox:
            for (int a = 0; a < 3; ++a) {
                inst.local_bmin[a] = -inst.size[a];
                inst.local_bmax[a] = inst.size[a];
            }
            break;
        case GeomType::kSphere:
            for (int a = 0; a < 3; ++a) {
                inst.local_bmin[a] = -inst.size[0];
                inst.local_bmax[a] = inst.size[0];
            }
            break;
        case GeomType::kCylinder:
            inst.local_bmin[0] = -inst.size[0]; inst.local_bmax[0] = inst.size[0];
            inst.local_bmin[1] = -inst.size[0]; inst.local_bmax[1] = inst.size[0];
            inst.local_bmin[2] = -inst.size[1]; inst.local_bmax[2] = inst.size[1];
            break;
        case GeomType::kMesh:
            if (inst.mesh >= 0 &&
                inst.mesh < static_cast<int>(scene.meshes.size())) {
                const MeshData & m = scene.meshes[static_cast<size_t>(inst.mesh)];
                for (int a = 0; a < 3; ++a) {
                    inst.local_bmin[a] = m.bmin[a];
                    inst.local_bmax[a] = m.bmax[a];
                }
            }
            break;
    }
}

void computeXform(const Instance & inst,
                  const float r_lw[9], const float t_lw[3],
                  InstanceXform & out)
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
    for (int a = 0; a < 3; ++a) {
        out.bmin[a] = kInf;
        out.bmax[a] = -kInf;
    }
    for (int corner = 0; corner < 8; ++corner) {
        const V3 c{(corner & 1) ? inst.local_bmax[0] : inst.local_bmin[0],
                   (corner & 2) ? inst.local_bmax[1] : inst.local_bmin[1],
                   (corner & 4) ? inst.local_bmax[2] : inst.local_bmin[2]};
        const V3 p = xformPoint(r_lw, t_lw, c);
        out.bmin[0] = std::min(out.bmin[0], p.x);
        out.bmin[1] = std::min(out.bmin[1], p.y);
        out.bmin[2] = std::min(out.bmin[2], p.z);
        out.bmax[0] = std::max(out.bmax[0], p.x);
        out.bmax[1] = std::max(out.bmax[1], p.y);
        out.bmax[2] = std::max(out.bmax[2], p.z);
    }
}

void castScan(const Scene & scene,
              const InstanceXform * xforms,
              const float * beam_alt_deg,
              const float * beam_az_deg,
              const float sensor_r[9], const float sensor_t[3],
              const ScanParams & sp,
              float * range_out, float * retro_out)
{
    const int n = sp.H * sp.W;
    const float deg_per_col = 360.0f / static_cast<float>(sp.W);
    const float n_off = sp.beam_origin_m;
    const int n_inst = static_cast<int>(scene.instances.size());

    #pragma omp parallel for schedule(static) if(n > 4096)  // NOLINT(whitespace/parens)
    for (int idx = 0; idx < n; ++idx) {
        const int beam = idx / sp.W;
        const int m = idx % sp.W;

        // Encoder azimuth (column) and full beam azimuth (with calibration
        // offset). The ray origin sits on the beam-origin circle at the
        // ENCODER azimuth — that is the convention the Ouster XYZ LUT
        // inverts, so reporting r = s + n reconstructs the exact hit point.
        const float az_enc = -static_cast<float>(m) * deg_per_col *
                             kPi / 180.0f;
        const float az = az_enc + beam_az_deg[beam] * kPi / 180.0f;
        const float el = beam_alt_deg[beam] * kPi / 180.0f;

        const float ce = std::cos(el);
        const V3 d_s{ce * std::cos(az), ce * std::sin(az), std::sin(el)};
        const V3 o_s{n_off * std::cos(az_enc), n_off * std::sin(az_enc), 0.0f};

        const V3 o = xformPoint(sensor_r, sensor_t, o_s);
        const V3 d = rotate(sensor_r, d_s);
        const V3 inv_d{1.0f / d.x, 1.0f / d.y, 1.0f / d.z};

        const float tmin = sp.near_clip;
        float best = sp.max_range - n_off;  // current nearest hit (ray param)
        int best_inst = -1;

        for (int i = 0; i < n_inst; ++i) {
            const InstanceXform & x = xforms[i];
            if (!hitAabb(o, inv_d, x.bmin, x.bmax, tmin, best)) continue;
            const V3 o_l = xformPoint(x.r, x.t, o);
            const V3 d_l = rotate(x.r, d);
            const float t = hitInstance(
                scene.instances[static_cast<size_t>(i)], scene,
                o_l, d_l, tmin, best);
            if (t > 0.0f && t < best) {
                best = t;
                best_inst = i;
            }
        }

        if (best_inst >= 0) {
            range_out[idx] = best + n_off;
            if (retro_out) {
                retro_out[idx] =
                    scene.instances[static_cast<size_t>(best_inst)].retro;
            }
        } else {
            range_out[idx] = kInf;
            if (retro_out) retro_out[idx] = 0.0f;
        }
    }
}

}  // namespace rc
}  // namespace gz_gpu_ouster_lidar
