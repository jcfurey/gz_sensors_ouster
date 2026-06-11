// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0
//
// Shared, backend-agnostic raycast math for the full per-beam raycast mode.
//
// Mirrors the ray_processor_math.hpp pattern: every function is GZ_OUSTER_HD
// so the CUDA, HIP and SYCL backends device-compile the exact same
// intersectors and BVH traversal the CPU fallback runs, and the types here
// are flat PODs so a scene uploads to a device as five plain arrays.
//
// Mesh storage is fully rebased at build time (cuda/raycast_scene.cpp):
// BVH child/leaf indices, triangle order entries and vertex indices are all
// global into the concatenated arrays, so traversal needs no per-mesh
// offsets — an instance just carries the root node index of its mesh.

#pragma once

#include "ray_processor_math.hpp"  // GZ_OUSTER_HD + gzm math shim

#include <cstdint>

namespace gz_gpu_ouster_lidar {
namespace rc {

// ── Flat scene PODs ──────────────────────────────────────────────────────────

enum class GeomType : int {
    kPlane = 0,    ///< z = 0 plane, finite half-extents size[0..1] (0 → infinite)
    kBox,          ///< axis-aligned box, half-extents size[0..2]
    kSphere,       ///< radius size[0]
    kCylinder,     ///< z axis, radius size[0], half-length size[1]
    kMesh,         ///< triangle mesh rooted at node `root_node`
};

struct MeshBvhNode {
    float bmin[3];
    float bmax[3];
    int left = -1;    ///< global child node index, -1 for leaf
    int right = -1;
    int first = 0;    ///< leaf: first entry in the global order array
    int count = 0;    ///< leaf: triangle count
};

/// Immutable per-instance geometry + material (device-friendly).
struct RcInstance {
    GeomType type = GeomType::kBox;
    float size[3] = {0, 0, 0};
    float retro = 0.0f;     ///< laser_retro: diffuse reflectance kd (0 = unset)
    float spec = 0.0f;      ///< specular coefficient ks (visual material specular)
    float transmit = 0.0f;  ///< transmittance τ ∈ [0,1] (visual transparency)
    int root_node = -1;     ///< kMesh: global BVH root node index
};

/// Per-scan rigid transform of one instance.
struct InstanceXform {
    float r[9];     ///< world→local rotation (row-major)
    float t[3];     ///< world→local translation: p_l = r·p_w + t
    float bmin[3];  ///< world-space AABB for the broad-phase reject
    float bmax[3];
};

struct ScanParams {
    int H = 0;                 ///< beam count (output rows)
    int W = 0;                 ///< columns per frame (output cols)
    float max_range = 120.0f;  ///< metres; hits beyond this are misses
    float near_clip = 0.3f;    ///< metres; hits closer than this are ignored
    float beam_origin_m = 0.0f;  ///< lidar-origin→beam-origin offset (metres)
};

constexpr float kRcTriEps = 1.0e-8f;
constexpr int kRcBvhStack = 64;
/// "Infinite" plane half-extent: large enough for any practical world while
/// keeping AABB corner transforms finite.
constexpr float kRcHugeExtent = 1.0e6f;
/// Grazing-incidence floor for the cos(incidence) reflectance factor: keeps
/// the effective reflectance positive (0 means "retro unset" downstream) and
/// caps the noise-weighting blow-up at extreme angles, where the Lambertian
/// cosine model is unreliable anyway (validated only up to ~20° incidence;
/// Kaasalainen et al., Remote Sens. 3(10), 2011).
constexpr float kRcMinCosInc = 0.01f;
/// Transmittance below this is treated as opaque (no continuation segment).
constexpr float kRcTransmitMin = 0.05f;
/// Offset past a transparent surface before the continuation cast.
constexpr float kRcSegEps = 1.0e-3f;
/// Segment-local tmin for the continuation cast (the sensor near clip
/// applies to the first segment only — objects just behind glass are valid).
constexpr float kRcSegTmin = 1.0e-4f;

// ── Small vector helpers ─────────────────────────────────────────────────────

struct RcV3 {
    float x, y, z;
};

GZ_OUSTER_HD inline RcV3 rcSub(RcV3 a, RcV3 b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
GZ_OUSTER_HD inline RcV3 rcCross(RcV3 a, RcV3 b)
{
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}
GZ_OUSTER_HD inline float rcDot(RcV3 a, RcV3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

/// Apply a row-major rotation + translation: r·p + t.
GZ_OUSTER_HD inline RcV3 rcXformPoint(const float r[9], const float t[3], RcV3 p)
{
    return {r[0] * p.x + r[1] * p.y + r[2] * p.z + t[0],
            r[3] * p.x + r[4] * p.y + r[5] * p.z + t[1],
            r[6] * p.x + r[7] * p.y + r[8] * p.z + t[2]};
}

GZ_OUSTER_HD inline RcV3 rcRotate(const float r[9], RcV3 p)
{
    return {r[0] * p.x + r[1] * p.y + r[2] * p.z,
            r[3] * p.x + r[4] * p.y + r[5] * p.z,
            r[6] * p.x + r[7] * p.y + r[8] * p.z};
}

// ── Intersectors (instance-local frame; rays have unit direction) ───────────

/// One slab of a robust ray/AABB test. Explicitly handles axis-parallel
/// rays (a zero direction component would otherwise produce 0·inf = NaN
/// when the origin sits exactly on a slab boundary): parallel + outside →
/// miss; parallel + inside → the slab adds no constraint. Updates the
/// running [lo, hi] interval and returns false on a definite miss.
GZ_OUSTER_HD inline bool rcSlabAxis(float o, float d, float bmin, float bmax,
                                    float & lo, float & hi)
{
    if (rpmath::gzm::fabs_(d) < 1.0e-12f) {
        return o >= bmin && o <= bmax;
    }
    const float inv = 1.0f / d;
    float t0 = (bmin - o) * inv;
    float t1 = (bmax - o) * inv;
    if (t0 > t1) {
        const float tmp = t0;
        t0 = t1;
        t1 = tmp;
    }
    lo = rpmath::gzm::fmax_(lo, t0);
    hi = rpmath::gzm::fmin_(hi, t1);
    return true;
}

/// Ray/AABB slab test on [tmin, tmax]; NaN-free for axis-parallel rays.
GZ_OUSTER_HD inline bool rcHitAabb(RcV3 o, RcV3 d,
    const float bmin[3], const float bmax[3], float tmin, float tmax)
{
    float lo = tmin, hi = tmax;
    if (!rcSlabAxis(o.x, d.x, bmin[0], bmax[0], lo, hi)) return false;
    if (!rcSlabAxis(o.y, d.y, bmin[1], bmax[1], lo, hi)) return false;
    if (!rcSlabAxis(o.z, d.z, bmin[2], bmax[2], lo, hi)) return false;
    return lo <= hi;
}

/// Möller–Trumbore. Returns hit parameter in (tmin, tmax) or -1.
GZ_OUSTER_HD inline float rcHitTriangle(RcV3 o, RcV3 d,
    RcV3 v0, RcV3 v1, RcV3 v2, float tmin, float tmax)
{
    const RcV3 e1 = rcSub(v1, v0);
    const RcV3 e2 = rcSub(v2, v0);
    const RcV3 p = rcCross(d, e2);
    const float det = rcDot(e1, p);
    if (rpmath::gzm::fabs_(det) < kRcTriEps) return -1.0f;
    const float inv_det = 1.0f / det;
    const RcV3 s = rcSub(o, v0);
    const float u = rcDot(s, p) * inv_det;
    if (u < 0.0f || u > 1.0f) return -1.0f;
    const RcV3 q = rcCross(s, e1);
    const float v = rcDot(d, q) * inv_det;
    if (v < 0.0f || u + v > 1.0f) return -1.0f;
    const float t = rcDot(e2, q) * inv_det;
    return (t > tmin && t < tmax) ? t : -1.0f;
}

GZ_OUSTER_HD inline float rcHitSphere(RcV3 o, RcV3 d, float radius,
                                      float tmin, float tmax)
{
    const float b = rcDot(o, d);
    const float c = rcDot(o, o) - radius * radius;
    const float disc = b * b - c;
    if (disc < 0.0f) return -1.0f;
    const float sq = rpmath::gzm::sqrt_(disc);
    float t = -b - sq;
    if (t <= tmin) t = -b + sq;
    return (t > tmin && t < tmax) ? t : -1.0f;
}

GZ_OUSTER_HD inline float rcHitBox(RcV3 o, RcV3 d, const float h[3],
                                   float tmin, float tmax)
{
    // Unbounded interval here (not [tmin, tmax]) so an origin inside the
    // box still yields the exit point below; NaN-free via rcSlabAxis.
    constexpr float kBig = 3.0e38f;
    float lo = -kBig, hi = kBig;
    if (!rcSlabAxis(o.x, d.x, -h[0], h[0], lo, hi)) return -1.0f;
    if (!rcSlabAxis(o.y, d.y, -h[1], h[1], lo, hi)) return -1.0f;
    if (!rcSlabAxis(o.z, d.z, -h[2], h[2], lo, hi)) return -1.0f;
    if (lo > hi) return -1.0f;
    // Entry point if outside the box, exit point if inside.
    const float t = (lo > tmin) ? lo : hi;
    return (t > tmin && t < tmax) ? t : -1.0f;
}

GZ_OUSTER_HD inline float rcHitCylinder(RcV3 o, RcV3 d, float radius,
    float half_len, float tmin, float tmax)
{
    float best = -1.0f;
    // Lateral surface: quadratic in the xy plane.
    const float a = d.x * d.x + d.y * d.y;
    if (a > 1.0e-12f) {
        const float b = (o.x * d.x + o.y * d.y) / a;
        const float c = (o.x * o.x + o.y * o.y - radius * radius) / a;
        const float disc = b * b - c;
        if (disc >= 0.0f) {
            const float sq = rpmath::gzm::sqrt_(disc);
            // Candidates are ordered; take the first valid one.
            float t = -b - sq;
            for (int k = 0; k < 2; ++k, t = -b + sq) {
                if (t <= tmin || t >= tmax) continue;
                const float z = o.z + t * d.z;
                if (rpmath::gzm::fabs_(z) <= half_len) {
                    best = t;
                    break;
                }
            }
        }
    }
    // End caps.
    if (rpmath::gzm::fabs_(d.z) > 1.0e-12f) {
        float zc = half_len;
        for (int k = 0; k < 2; ++k, zc = -half_len) {
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

GZ_OUSTER_HD inline float rcHitPlane(RcV3 o, RcV3 d, const float size[3],
                                     float tmin, float tmax)
{
    if (rpmath::gzm::fabs_(d.z) < 1.0e-12f) return -1.0f;
    const float t = -o.z / d.z;
    if (t <= tmin || t >= tmax) return -1.0f;
    const float hx = (size[0] > 0.0f) ? size[0] : kRcHugeExtent;
    const float hy = (size[1] > 0.0f) ? size[1] : kRcHugeExtent;
    const float x = o.x + t * d.x;
    const float y = o.y + t * d.y;
    return (rpmath::gzm::fabs_(x) <= hx && rpmath::gzm::fabs_(y) <= hy)
        ? t : -1.0f;
}

/// BVH traversal over the globally rebased arrays. `root` is the mesh's
/// root node index; `order` holds global triangle indices; `tris` holds
/// global vertex indices. `hit_tri` (optional) receives the global index of
/// the winning triangle, for normal reconstruction.
GZ_OUSTER_HD inline float rcHitMesh(
    const float * verts, const int * tris, const int * order,
    const MeshBvhNode * nodes, int root,
    RcV3 o, RcV3 d, float tmin, float tmax, int * hit_tri = nullptr)
{
    if (root < 0) return -1.0f;
    float best = -1.0f;
    float limit = tmax;
    int stack[kRcBvhStack];
    int sp = 0;
    stack[sp++] = root;
    while (sp > 0) {
        const MeshBvhNode & n = nodes[stack[--sp]];
        if (!rcHitAabb(o, d, n.bmin, n.bmax, tmin, limit)) continue;
        if (n.left < 0) {
            for (int k = 0; k < n.count; ++k) {
                const int tri = order[n.first + k];
                const int * idx = &tris[3 * tri];
                const RcV3 v0{verts[3 * idx[0]], verts[3 * idx[0] + 1],
                              verts[3 * idx[0] + 2]};
                const RcV3 v1{verts[3 * idx[1]], verts[3 * idx[1] + 1],
                              verts[3 * idx[1] + 2]};
                const RcV3 v2{verts[3 * idx[2]], verts[3 * idx[2] + 1],
                              verts[3 * idx[2] + 2]};
                const float t = rcHitTriangle(o, d, v0, v1, v2, tmin, limit);
                if (t > 0.0f) {
                    best = t;
                    limit = t;
                    if (hit_tri) *hit_tri = tri;
                }
            }
        } else if (sp + 2 <= kRcBvhStack) {
            stack[sp++] = n.left;
            stack[sp++] = n.right;
        }
    }
    return best;
}

GZ_OUSTER_HD inline float rcHitInstance(const RcInstance & inst,
    const float * verts, const int * tris, const int * order,
    const MeshBvhNode * nodes,
    RcV3 o, RcV3 d, float tmin, float tmax, int * hit_tri = nullptr)
{
    switch (inst.type) {
        case GeomType::kPlane:
            return rcHitPlane(o, d, inst.size, tmin, tmax);
        case GeomType::kBox:
            return rcHitBox(o, d, inst.size, tmin, tmax);
        case GeomType::kSphere:
            return rcHitSphere(o, d, inst.size[0], tmin, tmax);
        case GeomType::kCylinder:
            return rcHitCylinder(o, d, inst.size[0], inst.size[1],
                                 tmin, tmax);
        case GeomType::kMesh:
            return rcHitMesh(verts, tris, order, nodes, inst.root_node,
                             o, d, tmin, tmax, hit_tri);
    }
    return -1.0f;
}

/// cos of the incidence angle between the (unit) ray direction and the
/// surface normal at a known hit, computed in the instance-local frame.
/// Clamped to [kRcMinCosInc, 1].
///
/// Used to model the incidence-angle dependence of the received return: for
/// an extended Lambertian target the lidar equation gives
/// P_received ∝ ρ · cos(α) / R² — an oblique surface returns less light, so
/// its apparent reflectance is ρ·cos(α). (Kashani et al., "A Review of LIDAR
/// Radiometric Processing", Sensors 15(11), 2015; Kaasalainen et al., Remote
/// Sens. 3(10), 2011; same model as the HELIOS++ simulator, Winiwarter et
/// al., Remote Sens. Environ. 269, 2022.)
GZ_OUSTER_HD inline float rcCosIncidence(const RcInstance & inst,
    const float * verts, const int * tris,
    RcV3 o_l, RcV3 d_l, float t, int hit_tri)
{
    const RcV3 p{o_l.x + t * d_l.x, o_l.y + t * d_l.y, o_l.z + t * d_l.z};
    RcV3 n{0.0f, 0.0f, 1.0f};
    switch (inst.type) {
        case GeomType::kPlane:
            break;  // n = +z by construction
        case GeomType::kBox: {
            // The hit face is the axis where |p_i| reaches its half-extent
            // first (largest normalised coordinate).
            const float rx = rpmath::gzm::fabs_(p.x) /
                             rpmath::gzm::fmax_(inst.size[0], 1.0e-12f);
            const float ry = rpmath::gzm::fabs_(p.y) /
                             rpmath::gzm::fmax_(inst.size[1], 1.0e-12f);
            const float rz = rpmath::gzm::fabs_(p.z) /
                             rpmath::gzm::fmax_(inst.size[2], 1.0e-12f);
            if (rx >= ry && rx >= rz) {
                n = RcV3{1.0f, 0.0f, 0.0f};
            } else if (ry >= rz) {
                n = RcV3{0.0f, 1.0f, 0.0f};
            } else {
                n = RcV3{0.0f, 0.0f, 1.0f};
            }
            break;
        }
        case GeomType::kSphere:
            n = p;  // radial; normalised via |n| below
            break;
        case GeomType::kCylinder: {
            // Cap if the hit sits at an end disc, else the lateral surface.
            const float half_len = inst.size[1];
            if (rpmath::gzm::fabs_(p.z) >= half_len * (1.0f - 1.0e-4f)) {
                n = RcV3{0.0f, 0.0f, 1.0f};
            } else {
                n = RcV3{p.x, p.y, 0.0f};
            }
            break;
        }
        case GeomType::kMesh: {
            if (hit_tri < 0) return 1.0f;
            const int * idx = &tris[3 * hit_tri];
            const RcV3 v0{verts[3 * idx[0]], verts[3 * idx[0] + 1],
                          verts[3 * idx[0] + 2]};
            const RcV3 v1{verts[3 * idx[1]], verts[3 * idx[1] + 1],
                          verts[3 * idx[1] + 2]};
            const RcV3 v2{verts[3 * idx[2]], verts[3 * idx[2] + 1],
                          verts[3 * idx[2] + 2]};
            n = rcCross(rcSub(v1, v0), rcSub(v2, v0));
            break;
        }
    }
    const float nn = rpmath::gzm::sqrt_(rcDot(n, n));
    if (nn < 1.0e-12f) return 1.0f;  // degenerate normal: no attenuation
    const float c = rpmath::gzm::fabs_(rcDot(d_l, n)) / nn;
    return rpmath::gzm::fmin_(rpmath::gzm::fmax_(c, kRcMinCosInc), 1.0f);
}

/// Monostatic apparent reflectance of a hit: Lambertian diffuse term
/// kd·cos(α) plus a specular lobe ks·cos(2α)⁸ (zero past 45°).
///
/// For a monostatic lidar the receiver sits at the emitter, so the Phong
/// lobe around the mirror direction r̂ contributes ∝ (r̂·(−d̂))ⁿ =
/// cos(2α)ⁿ: a smooth/glossy surface (glass, polished or dark metallic
/// paint) returns strongly only when viewed near surface-normal and
/// "drops quickly" off-normal — the empirically observed behaviour of
/// lidar on glass (Velas et al., arXiv:1909.12483 §III) and the cause of
/// the missing-points signature of glossy black vehicles. n = 8 is a
/// fixed lobe width (≈ half-power at ~9° off normal), chosen qualitative:
/// real lobe widths vary per material and are not exposed by SDF.
GZ_OUSTER_HD inline float rcApparentReflectance(const RcInstance & inst,
                                                float cos_inc)
{
    float rho = inst.retro * cos_inc;
    if (inst.spec > 0.0f) {
        const float c2 = 2.0f * cos_inc * cos_inc - 1.0f;  // cos(2α)
        if (c2 > 0.0f) {
            float lobe = c2 * c2;   // cos(2α)²
            lobe *= lobe;           // ⁴
            lobe *= lobe;           // ⁸
            rho += inst.spec * lobe;
        }
    }
    return rho;
}

/// Nearest hit of one ray over all instances. Returns the hit parameter
/// (or -1), the winning instance in `inst_out` (-1 for a miss) and, for
/// mesh hits, the winning global triangle index in `tri_out`.
GZ_OUSTER_HD inline float rcNearestHit(
    const RcInstance * instances, int n_instances,
    const float * verts, const int * tris, const int * order,
    const MeshBvhNode * nodes, const InstanceXform * xforms,
    RcV3 o, RcV3 d, float tmin, float tmax,
    int & inst_out, int & tri_out)
{
    float best = tmax;
    inst_out = -1;
    tri_out = -1;
    for (int i = 0; i < n_instances; ++i) {
        const InstanceXform & x = xforms[i];
        if (!rcHitAabb(o, d, x.bmin, x.bmax, tmin, best)) continue;
        const RcV3 o_l = rcXformPoint(x.r, x.t, o);
        const RcV3 d_l = rcRotate(x.r, d);
        int tri = -1;
        const float t = rcHitInstance(instances[i], verts, tris, order,
                                      nodes, o_l, d_l, tmin, best, &tri);
        if (t > 0.0f && t < best) {
            best = t;
            inst_out = i;
            tri_out = tri;
        }
    }
    return (inst_out >= 0) ? best : -1.0f;
}

/// Apparent reflectance of a known hit (instance-local cos incidence
/// recomputed from the world-frame ray).
GZ_OUSTER_HD inline float rcHitReflectance(
    const RcInstance * instances, const InstanceXform * xforms,
    const float * verts, const int * tris,
    RcV3 o, RcV3 d, float t, int inst, int tri)
{
    const InstanceXform & x = xforms[inst];
    const RcV3 o_l = rcXformPoint(x.r, x.t, o);
    const RcV3 d_l = rcRotate(x.r, d);
    const float cos_inc =
        rcCosIncidence(instances[inst], verts, tris, o_l, d_l, t, tri);
    return rcApparentReflectance(instances[inst], cos_inc);
}

/// Cast one output pixel (beam × measurement id) against the whole scene.
/// Writes the reported Ouster range (metres; `inf_value` for a miss — the
/// value satisfying the XYZ-LUT reconstruction, see castScan docs) and the
/// nearest hit's APPARENT reflectance: laser_retro × cos(incidence)
/// (0 on a miss, or when laser_retro is unset).
GZ_OUSTER_HD inline void rcCastOneRay(
    const RcInstance * instances, int n_instances,
    const float * verts, const int * tris, const int * order,
    const MeshBvhNode * nodes,
    const InstanceXform * xforms,
    const float * beam_alt_deg, const float * beam_az_deg,
    const float * sensor_r, const float * sensor_t,
    const ScanParams & sp, int idx, float inf_value,
    float & range_out, float & retro_out)
{
    const int beam = idx / sp.W;
    const int m = idx % sp.W;
    const float deg_per_col = 360.0f / static_cast<float>(sp.W);
    const float n_off = sp.beam_origin_m;

    // Encoder azimuth (column) and full beam azimuth (with calibration
    // offset; sign convention in rpmath::beamRayAzimuthDeg). The ray origin
    // sits on the beam-origin circle at the ENCODER azimuth — that is the
    // convention the Ouster XYZ LUT inverts, so reporting r = s + n
    // reconstructs the exact hit point.
    const float az_enc = -static_cast<float>(m) * deg_per_col *
                         rpmath::kPi / 180.0f;
    const float az = rpmath::beamRayAzimuthDeg(beam_az_deg[beam], m,
                                               deg_per_col) *
                     rpmath::kPi / 180.0f;
    const float el = beam_alt_deg[beam] * rpmath::kPi / 180.0f;

    const float ce = rpmath::gzm::cos_(el);
    const RcV3 d_s{ce * rpmath::gzm::cos_(az), ce * rpmath::gzm::sin_(az),
                   rpmath::gzm::sin_(el)};
    const RcV3 o_s{n_off * rpmath::gzm::cos_(az_enc),
                   n_off * rpmath::gzm::sin_(az_enc), 0.0f};

    const RcV3 o = rcXformPoint(sensor_r, sensor_t, o_s);
    const RcV3 d = rcRotate(sensor_r, d_s);

    const float t_budget = sp.max_range - n_off;

    int inst0 = -1, tri0 = -1;
    const float t0 = rcNearestHit(instances, n_instances, verts, tris, order,
                                  nodes, xforms, o, d, sp.near_clip, t_budget,
                                  inst0, tri0);
    if (inst0 < 0) {
        range_out = inf_value;
        retro_out = 0.0f;
        return;
    }

    // Apparent reflectance of the front surface: kd·cos(α) + ks·cos(2α)⁸
    // (see rcApparentReflectance) — the extended-Lambertian lidar equation
    // P ∝ ρ_app/R² plus the monostatic specular lobe. Folding the angular
    // terms in here makes the downstream signal, reflectivity byte and
    // noise weighting all respond to oblique/glossy surfaces the way a real
    // return does. laser_retro == 0 with no specular stays 0 and keeps its
    // base_reflectivity fallback downstream.
    const float tau =
        rpmath::gzm::fmin_(rpmath::gzm::fmax_(instances[inst0].transmit, 0.0f),
                           1.0f);
    float rho = rcHitReflectance(instances, xforms, verts, tris,
                                 o, d, t0, inst0, tri0) * (1.0f - tau);
    float range = t0 + n_off;

    // Transparent surface (glass): continue one segment behind it. The
    // surface return keeps (1−τ); the behind-glass return is attenuated by
    // τ² (the pulse crosses the pane twice); single-return mode reports the
    // STRONGEST candidate by received power ρ/R², matching how lidar sees
    // through windows in practice (Velas et al., arXiv:1909.12483 §III:
    // near-normal glass returns the pane, otherwise the object behind, with
    // weakened intensity). One continuation segment only — a second pane
    // behind the first is treated as opaque.
    if (tau >= kRcTransmitMin) {
        const float seg_start = t0 + kRcSegEps;
        const RcV3 p{o.x + d.x * seg_start, o.y + d.y * seg_start,
                     o.z + d.z * seg_start};
        int inst1 = -1, tri1 = -1;
        const float t1 = rcNearestHit(instances, n_instances, verts, tris,
                                      order, nodes, xforms, p, d, kRcSegTmin,
                                      t_budget - seg_start, inst1, tri1);
        if (inst1 >= 0) {
            const float rho1 = rcHitReflectance(instances, xforms, verts,
                                                tris, p, d, t1, inst1, tri1) *
                               tau * tau;
            const float range1 = seg_start + t1 + n_off;
            if (rho1 * range * range > rho * range1 * range1) {
                rho = rho1;
                range = range1;
            }
        }
    }

    range_out = range;
    retro_out = rho;
}

}  // namespace rc
}  // namespace gz_gpu_ouster_lidar
