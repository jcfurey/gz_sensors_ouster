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
    float retro = 0.0f;     ///< laser_retro: diffuse reflectance kd
    float spec = 0.0f;      ///< specular coefficient ks (visual material specular)
    float transmit = 0.0f;  ///< transmittance τ ∈ [0,1] (visual transparency)
    int has_retro = 0;      ///< 1 when laser_retro was explicitly authored
    int root_node = -1;     ///< kMesh: global BVH root node index
    /// Optional packed RGBA8 response map in SceneView::response_texels.
    /// R=865 nm diffuse reflectance, G=passive NIR albedo, B=specular
    /// coefficient, A=opacity. Negative offset means scalar SDF material.
    int response_offset = -1;
    int response_width = 0;
    int response_height = 0;
};

struct RcMaterialSample {
    float diffuse = 0.0f;
    float nir = 0.0f;
    float spec = 0.0f;
    float transmit = 0.0f;
};

/// Per-scan rigid transform of one instance.
struct InstanceXform {
    float r[9];     ///< world→local rotation (row-major)
    float t[3];     ///< world→local translation: p_l = r·p_w + t
    float bmin[3];  ///< world-space AABB for the broad-phase reject
    float bmax[3];
};

/// Shape of an obscurant volume, in the volume's own local frame.
enum class ObscurantType : int {
    kBox = 0,     ///< box, half-extents half[0..2]
    kEllipsoid,   ///< semi-axes half[0..2] (a sphere when all three are equal)
    kCylinder,    ///< z axis, semi-axes half[0..1], half-length half[2]
};

/// One participating-medium volume: smoke, dust, fog or spray that the beam
/// travels THROUGH rather than bounces off.
///
/// Inside the volume the medium is homogeneous with extinction coefficient
/// σ_ext = `sigma` [1/m], so Beer–Lambert gives the one-way transmittance
/// over a path of length L as exp(−σ·L). The volume backscatter coefficient
/// follows from the extinction-to-backscatter ratio (the "lidar ratio")
/// S = σ_ext/β_π [sr] — the standard atmospheric-lidar parameterisation,
/// measured per aerosol type: ≈18–20 sr for fog/water cloud, ≈40–50 sr for
/// dust, ≈50–70 sr for biomass-burning smoke (Müller et al., JGR 112 D16202,
/// 2007; Ackermann, J. Atmos. Ocean. Technol. 15, 1998).
struct RcObscurant {
    float r[9];             ///< world→local rotation (row-major)
    float t[3];             ///< world→local translation: p_l = r·p_w + t
    float half[3];          ///< local half-extents / semi-axes (metres)
    float sigma = 0.0f;     ///< extinction coefficient σ_ext [1/m]
    float lidar_ratio = 50.0f;  ///< S = σ_ext/β_π [sr]
    float albedo = 0.8f;    ///< single-scattering albedo ω (NEAR_IR airlight)
    /// Platt's multiple-scattering factor η ∈ (0, 1]: the fraction of the
    /// extinction that the RECEIVER actually experiences.
    ///
    /// σ_ext removes light in every direction, but smoke, dust and fog are
    /// strongly forward-peaked (asymmetry g ≈ 0.7–0.9), so much of that light
    /// is deflected by only a few milliradians and stays inside a real
    /// receiver's field of view. Treating it as lost — η = 1, pure single
    /// scattering — makes dense media too opaque. Platt's approximation keeps
    /// the single-scattering form and folds the recovery into one path
    /// factor, so the measured backscatter is β_π·exp(−2·η·τ).
    ///
    /// η ≈ 1 for optically thin media or a narrow field of view (the
    /// default, which reproduces the pure single-scattering limit exactly);
    /// η ≈ 0.5–0.8 for dense fog and smoke at typical lidar fields of view.
    /// Real η varies along the path as multiple scattering builds up; a
    /// single path-averaged value is Platt's own working approximation and
    /// is all this model can support without a beam-cone model.
    ///
    /// Platt, J. Atmos. Sci. 30, 1973; J. Appl. Meteorol. 20, 1981.
    float ms_factor = 1.0f;
    ObscurantType type = ObscurantType::kEllipsoid;
};

/// Obscurant volumes ride inside ScanParams, which every backend already
/// passes to its kernel BY VALUE (RcCastArgs in the CUDA/HIP kernels, the
/// captured sp_copy in SYCL) — the same trick ResampleParams uses for its
/// panel array. That keeps the whole feature out of the Backend interface at
/// the cost of a fixed cap.
///
/// 16 is chosen against two budgets, and the static_assert below pins the
/// first so nobody raises it blindly:
///   * kernel argument space — CUDA and HIP allow 4 KB, and real Level Zero
///     devices report ≥ 2 KB, against ~1.2 KB of ScanParams here;
///   * per-thread scratch — the medium sampler keeps two float[N] span arrays
///     plus their indices on the stack, so N also sets the local-memory
///     footprint of every ray.
///
/// A world with more obscurants than this keeps the ones nearest the sensor
/// and logs what it dropped (see gatherObscurants); it never silently loses
/// smoke.
constexpr int kMaxObscurants = 16;
/// Half-extents are clamped to this before any divide: gz particle emitters
/// are routinely authored flat (`<size>10 10 0</size>`).
constexpr float kRcObscurantMinHalf = 1.0e-3f;
/// Optical depth below which a medium is treated as absent — exp(−2τ) differs
/// from 1 by < 0.02% here, so neither the attenuation nor a backscatter
/// return is observable.
constexpr float kRcTauMin = 1.0e-4f;
/// Floor for Platt's multiple-scattering factor. η = 0 would make a medium
/// perfectly transparent to the laser while still returning backscatter,
/// which is not a physical state.
constexpr double kRcMinMultipleScattering = 1.0e-3;

struct ScanParams {
    int H = 0;                 ///< beam count (output rows)
    int W = 0;                 ///< columns per frame (output cols)
    float max_range = 120.0f;  ///< metres; hits beyond this are misses
    float near_clip = 0.3f;    ///< metres; hits closer than this are ignored
    float beam_origin_m = 0.0f;  ///< lidar-origin→beam-origin offset (metres)
    // Ambient-illumination model for the NEAR_IR channel (Ouster NIR counts
    // ambient sunlight reflected off the scene, not laser return):
    // nir = albedo · (sun_ambient + sun_diffuse · max(0, n̂·(−ŝ))), where ŝ
    // is the sun's PROPAGATION direction (gz convention) and n̂ the
    // sensor-facing surface normal. Defaults (no directional light in the
    // world): ambient-only, nir = albedo.
    float sun_dir[3] = {0.0f, 0.0f, -1.0f};  ///< propagation direction (unit)
    float sun_diffuse = 0.0f;  ///< sun term weight (0 = no sun)
    float sun_ambient = 1.0f;  ///< ambient term weight
    /// Physical diffuse reflectance used when a visual omits laser_retro.
    /// Derived once per scan from ProcessParams::base_reflectivity so missing
    /// materials participate correctly in incidence/extinction/arbitration.
    float fallback_retro = rpmath::kDefaultRetro;
    /// Sensor gain used to turn the integrated medium-return profile into an
    /// expected photon count. This is the same base_signal consumed by the
    /// downstream channel model, so changing sensor sensitivity changes both
    /// the chance of detecting aerosol and the reported SIGNAL consistently.
    float base_signal = 800.0f;

    // ── Participating media (smoke / dust / fog) ─────────────────────────
    RcObscurant obscurants[kMaxObscurants];
    int n_obscurants = 0;      ///< valid entries in obscurants[] (0 = off)
    /// Effective range gate of one pulse, ΔR = c·τ_pulse/2 [m]. A hard target
    /// returns all of its energy from one surface; a distributed medium only
    /// returns the energy scattered from the slab the detector integrates
    /// over, so the medium's apparent reflectance carries this factor.
    float pulse_gate_m = 0.6f;
    /// Per-scan salt for the medium-range draw. The pixel index supplies the
    /// spatial key; changing this salt decorrelates successive scans.
    uint32_t rng_salt = 0;
};

// ScanParams is copied into kernel argument space by every backend. OpenCL's
// floor for CL_DEVICE_MAX_PARAMETER_SIZE is 1 KB and real Level Zero / CUDA /
// HIP devices are well above it, but the margin is finite — if this fires,
// shrink RcObscurant or move the array to a device buffer beside `xforms`
// rather than nudging the bound.
static_assert(sizeof(ScanParams) <= 2048,
              "ScanParams must stay inside the smallest backend's kernel "
              "argument budget");

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
/// Specular coefficient at or above which a surface also produces the
/// mirror-bounce ghost path. SDF has no roughness channel, so this is the
/// mirror/gloss discriminator: keep paint-like materials below it (the demo
/// glossy box uses 0.45) and actual mirrors/glass near 1.
constexpr float kRcMirrorMin = 0.5f;
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

/// Rotate by the transpose (= inverse) of a row-major rotation.
GZ_OUSTER_HD inline RcV3 rcRotateT(const float r[9], RcV3 p)
{
    return {r[0] * p.x + r[3] * p.y + r[6] * p.z,
            r[1] * p.x + r[4] * p.y + r[7] * p.z,
            r[2] * p.x + r[5] * p.y + r[8] * p.z};
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

/// Surface normal (instance-local frame, NOT normalised, sign arbitrary)
/// at a known hit point `p` of instance `inst`. Returns {0,0,0} only for a
/// degenerate mesh triangle.
GZ_OUSTER_HD inline RcV3 rcSurfaceNormalLocal(const RcInstance & inst,
    const float * verts, const int * tris, RcV3 p, int hit_tri)
{
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
            if (hit_tri < 0) return RcV3{0.0f, 0.0f, 0.0f};
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
    return n;
}

/// Compute a conventional bottom-left-origin UV coordinate at a known local
/// hit. Primitive mappings are deterministic and repeatable; triangle meshes
/// interpolate their authored vertex UVs using barycentric coordinates.
GZ_OUSTER_HD inline bool rcHitUv(const RcInstance & inst,
    const float * verts, const int * tris, const float * texcoords,
    RcV3 p, int hit_tri, float & u, float & v)
{
    switch (inst.type) {
        case GeomType::kPlane: {
            const float hx = rpmath::gzm::fmax_(inst.size[0], 1.0e-12f);
            const float hy = rpmath::gzm::fmax_(inst.size[1], 1.0e-12f);
            u = 0.5f + p.x / (2.0f * hx);
            v = 0.5f + p.y / (2.0f * hy);
            return true;
        }
        case GeomType::kBox: {
            const float hx = rpmath::gzm::fmax_(inst.size[0], 1.0e-12f);
            const float hy = rpmath::gzm::fmax_(inst.size[1], 1.0e-12f);
            const float hz = rpmath::gzm::fmax_(inst.size[2], 1.0e-12f);
            const float rx = rpmath::gzm::fabs_(p.x) / hx;
            const float ry = rpmath::gzm::fabs_(p.y) / hy;
            const float rz = rpmath::gzm::fabs_(p.z) / hz;
            if (rx >= ry && rx >= rz) {
                const float s = (p.x >= 0.0f) ? -1.0f : 1.0f;
                u = 0.5f + s * p.y / (2.0f * hy);
                v = 0.5f + p.z / (2.0f * hz);
            } else if (ry >= rz) {
                const float s = (p.y >= 0.0f) ? 1.0f : -1.0f;
                u = 0.5f + s * p.x / (2.0f * hx);
                v = 0.5f + p.z / (2.0f * hz);
            } else {
                u = 0.5f + p.x / (2.0f * hx);
                const float s = (p.z >= 0.0f) ? 1.0f : -1.0f;
                v = 0.5f + s * p.y / (2.0f * hy);
            }
            return true;
        }
        case GeomType::kSphere: {
            const float r = rpmath::gzm::sqrt_(rcDot(p, p));
            if (r <= 1.0e-12f) return false;
            const float z = rpmath::gzm::fmin_(
                rpmath::gzm::fmax_(p.z / r, -1.0f), 1.0f);
            u = 0.5f + rpmath::gzm::atan2_(p.y, p.x) /
                       (2.0f * rpmath::kPi);
            v = 0.5f + rpmath::gzm::asin_(z) / rpmath::kPi;
            return true;
        }
        case GeomType::kCylinder: {
            const float radius = rpmath::gzm::fmax_(inst.size[0], 1.0e-12f);
            const float half_len = rpmath::gzm::fmax_(inst.size[1], 1.0e-12f);
            if (rpmath::gzm::fabs_(p.z) >= half_len * (1.0f - 1.0e-4f)) {
                u = 0.5f + p.x / (2.0f * radius);
                const float s = (p.z >= 0.0f) ? 1.0f : -1.0f;
                v = 0.5f + s * p.y / (2.0f * radius);
            } else {
                u = 0.5f + rpmath::gzm::atan2_(p.y, p.x) /
                           (2.0f * rpmath::kPi);
                v = 0.5f + p.z / (2.0f * half_len);
            }
            return true;
        }
        case GeomType::kMesh: {
            if (hit_tri < 0 || verts == nullptr || tris == nullptr ||
                texcoords == nullptr) return false;
            const int * idx = &tris[3 * hit_tri];
            const RcV3 p0{verts[3 * idx[0]], verts[3 * idx[0] + 1],
                          verts[3 * idx[0] + 2]};
            const RcV3 p1{verts[3 * idx[1]], verts[3 * idx[1] + 1],
                          verts[3 * idx[1] + 2]};
            const RcV3 p2{verts[3 * idx[2]], verts[3 * idx[2] + 1],
                          verts[3 * idx[2] + 2]};
            const RcV3 e0 = rcSub(p1, p0);
            const RcV3 e1 = rcSub(p2, p0);
            const RcV3 ep = rcSub(p, p0);
            const float d00 = rcDot(e0, e0);
            const float d01 = rcDot(e0, e1);
            const float d11 = rcDot(e1, e1);
            const float d20 = rcDot(ep, e0);
            const float d21 = rcDot(ep, e1);
            const float denom = d00 * d11 - d01 * d01;
            if (rpmath::gzm::fabs_(denom) <= 1.0e-20f) return false;
            const float b1 = (d11 * d20 - d01 * d21) / denom;
            const float b2 = (d00 * d21 - d01 * d20) / denom;
            const float b0 = 1.0f - b1 - b2;
            u = b0 * texcoords[2 * idx[0]] +
                b1 * texcoords[2 * idx[1]] +
                b2 * texcoords[2 * idx[2]];
            v = b0 * texcoords[2 * idx[0] + 1] +
                b1 * texcoords[2 * idx[1] + 1] +
                b2 * texcoords[2 * idx[2] + 1];
            return true;
        }
    }
    return false;
}

GZ_OUSTER_HD inline int rcWrapIndex(int i, int n)
{
    const int r = i % n;
    return (r < 0) ? r + n : r;
}

GZ_OUSTER_HD inline float rcResponseChannel(const uint8_t * texels,
    const RcInstance & inst, int x, int y, int channel)
{
    x = rcWrapIndex(x, inst.response_width);
    y = rcWrapIndex(y, inst.response_height);
    const int pixel = y * inst.response_width + x;
    return static_cast<float>(
        texels[inst.response_offset + 4 * pixel + channel]) / 255.0f;
}

/// Bilinear, repeating response-map sample. UV is bottom-left-origin while
/// image rows are top-down, hence the vertical flip before addressing texels.
GZ_OUSTER_HD inline RcMaterialSample rcSampleResponse(
    const uint8_t * texels, const RcInstance & inst, float u, float v)
{
    RcMaterialSample out;
    const float uw = u - rpmath::gzm::floor_(u);
    const float vw = v - rpmath::gzm::floor_(v);
    const float x = uw * static_cast<float>(inst.response_width) - 0.5f;
    const float y = (1.0f - vw) *
                    static_cast<float>(inst.response_height) - 0.5f;
    const int x0 = static_cast<int>(rpmath::gzm::floor_(x));
    const int y0 = static_cast<int>(rpmath::gzm::floor_(y));
    const float ax = x - static_cast<float>(x0);
    const float ay = y - static_cast<float>(y0);
    float channels[4];
    for (int c = 0; c < 4; ++c) {
        const float a00 = rcResponseChannel(texels, inst, x0, y0, c);
        const float a10 = rcResponseChannel(texels, inst, x0 + 1, y0, c);
        const float a01 = rcResponseChannel(texels, inst, x0, y0 + 1, c);
        const float a11 = rcResponseChannel(texels, inst, x0 + 1, y0 + 1, c);
        const float top = a00 * (1.0f - ax) + a10 * ax;
        const float bot = a01 * (1.0f - ax) + a11 * ax;
        channels[c] = top * (1.0f - ay) + bot * ay;
    }
    out.diffuse = channels[0];
    out.nir = channels[1];
    out.spec = channels[2];
    out.transmit = 1.0f - channels[3];
    return out;
}

GZ_OUSTER_HD inline RcMaterialSample rcMaterialAtHit(
    const RcInstance & inst, const float * verts, const int * tris,
    const float * texcoords, const uint8_t * response_texels,
    RcV3 p, int hit_tri, float fallback_retro)
{
    RcMaterialSample out;
    out.diffuse = inst.has_retro ? inst.retro : fallback_retro;
    out.nir = out.diffuse;
    out.spec = inst.spec;
    out.transmit = inst.transmit;
    if (response_texels == nullptr || inst.response_offset < 0 ||
        inst.response_width <= 0 || inst.response_height <= 0) {
        return out;
    }
    float u = 0.0f, v = 0.0f;
    if (!rcHitUv(inst, verts, tris, texcoords, p, hit_tri, u, v)) return out;
    return rcSampleResponse(response_texels, inst, u, v);
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
    const RcV3 n = rcSurfaceNormalLocal(inst, verts, tris, p, hit_tri);
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
GZ_OUSTER_HD inline float rcApparentReflectance(float diffuse, float spec,
                                                float cos_inc)
{
    float rho = diffuse * cos_inc;
    if (spec > 0.0f) {
        const float c2 = 2.0f * cos_inc * cos_inc - 1.0f;  // cos(2α)
        if (c2 > 0.0f) {
            float lobe = c2 * c2;   // cos(2α)²
            lobe *= lobe;           // ⁴
            lobe *= lobe;           // ⁸
            rho += spec * lobe;
        }
    }
    return rho;
}

GZ_OUSTER_HD inline float rcApparentReflectance(const RcInstance & inst,
                                                float cos_inc,
                                                float fallback_retro)
{
    const float diffuse = inst.has_retro ? inst.retro : fallback_retro;
    return rcApparentReflectance(diffuse, inst.spec, cos_inc);
}

/// Nearest hit of one ray over all instances. Returns the hit parameter
/// (or -1), the winning instance in `inst_out` (-1 for a miss) and, for
/// mesh hits, the winning global triangle index in `tri_out`.
/// Narrow-phase test of one instance, updating the running best hit.
/// Shared by both broad-phase strategies in rcNearestHit below.
GZ_OUSTER_HD inline void rcTestInstance(
    const RcInstance * instances,
    const float * verts, const int * tris, const int * order,
    const MeshBvhNode * nodes, const InstanceXform * xforms,
    RcV3 o, RcV3 d, float tmin, int i,
    float & best, int & inst_out, int & tri_out)
{
    const InstanceXform & x = xforms[i];
    if (!rcHitAabb(o, d, x.bmin, x.bmax, tmin, best)) return;
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

/// Nearest hit across the whole scene.
///
/// Broad phase: when a top-level BVH (TLAS) over the instances' world AABBs is
/// supplied, traverse it — cost is O(log n_instances) per ray for a spatially
/// separated scene instead of the O(n_instances) linear scan, and the `best`
/// culling prunes whole subtrees. The TLAS is rebuilt per scan from the same
/// world AABBs already computed for `xforms` (see rc::buildTlas), so it costs
/// nothing extra to keep current.
///
/// Passing tlas_nodes == nullptr (or an empty TLAS) falls back to the linear
/// scan, which stays the right choice for a handful of instances and keeps
/// every backend correct even if it does not upload a TLAS.
GZ_OUSTER_HD inline float rcNearestHit(
    const RcInstance * instances, int n_instances,
    const float * verts, const int * tris, const int * order,
    const MeshBvhNode * nodes, const InstanceXform * xforms,
    RcV3 o, RcV3 d, float tmin, float tmax,
    int & inst_out, int & tri_out,
    const MeshBvhNode * tlas_nodes = nullptr,
    const int * tlas_order = nullptr,
    int n_tlas_nodes = 0)
{
    float best = tmax;
    inst_out = -1;
    tri_out = -1;

    if (tlas_nodes == nullptr || tlas_order == nullptr || n_tlas_nodes <= 0) {
        for (int i = 0; i < n_instances; ++i) {
            rcTestInstance(instances, verts, tris, order, nodes, xforms,
                           o, d, tmin, i, best, inst_out, tri_out);
        }
        return (inst_out >= 0) ? best : -1.0f;
    }

    // TLAS traversal. Root is node 0 (rc::buildTlas emits it first). Stack
    // depth is bounded by the tree depth, which the builder caps well below
    // kRcBvhStack; the explicit capacity guard keeps a malformed/deeper tree
    // from overrunning the stack on device.
    int stack[kRcBvhStack];
    int sp = 0;
    stack[sp++] = 0;
    while (sp > 0) {
        const MeshBvhNode & nd = tlas_nodes[stack[--sp]];
        if (!rcHitAabb(o, d, nd.bmin, nd.bmax, tmin, best)) continue;
        if (nd.left < 0) {
            for (int k = 0; k < nd.count; ++k) {
                rcTestInstance(instances, verts, tris, order, nodes, xforms,
                               o, d, tmin, tlas_order[nd.first + k],
                               best, inst_out, tri_out);
            }
        } else if (sp + 2 <= kRcBvhStack) {
            stack[sp++] = nd.left;
            stack[sp++] = nd.right;
        }
    }
    return (inst_out >= 0) ? best : -1.0f;
}

/// Apparent reflectance of a known hit (instance-local cos incidence
/// recomputed from the world-frame ray).
GZ_OUSTER_HD inline float rcHitReflectance(
    const RcInstance * instances, const InstanceXform * xforms,
    const float * verts, const int * tris, const float * texcoords,
    const uint8_t * response_texels, RcV3 o, RcV3 d, float t,
    int inst, int tri, float fallback_retro,
    RcMaterialSample * material_out = nullptr)
{
    const InstanceXform & x = xforms[inst];
    const RcV3 o_l = rcXformPoint(x.r, x.t, o);
    const RcV3 d_l = rcRotate(x.r, d);
    const RcV3 p_l{o_l.x + t * d_l.x, o_l.y + t * d_l.y,
                   o_l.z + t * d_l.z};
    const RcMaterialSample material = rcMaterialAtHit(
        instances[inst], verts, tris, texcoords, response_texels,
        p_l, tri, fallback_retro);
    if (material_out) *material_out = material;
    const float cos_inc =
        rcCosIncidence(instances[inst], verts, tris, o_l, d_l, t, tri);
    return rcApparentReflectance(material.diffuse, material.spec, cos_inc);
}

// ── Participating media: smoke / dust / fog obscuration ─────────────────────
//
// A beam crossing an obscurant volume does two things a clear beam does not:
// it LOSES energy on the way out and back (Beer–Lambert extinction), and it
// gains a competing return SCATTERED BACK BY THE MEDIUM ITSELF, which a
// single-return sensor may report instead of the real target. Both are the
// dominant lidar-in-smoke artifacts, and both are modeled here.
//
// Everything below is a pure function of ScanParams — no RNG state, no extra
// buffers — so the CPU fallback and the CUDA/HIP/SYCL kernels run identical
// code with no change to the Backend interface.

/// Entry/exit ray parameters of one obscurant, clipped to [t_lo, t_hi].
/// False when the ray misses the volume or the clipped span is empty.
///
/// Ellipsoid and elliptic-cylinder cases divide the local ray by the
/// semi-axes: scaling origin and direction by the same per-axis factors
/// leaves the ray parameter t untouched, so the roots come out directly in
/// the caller's parameterisation.
GZ_OUSTER_HD inline bool rcObscurantSpan(const RcObscurant & ob,
    RcV3 o, RcV3 d, float t_lo, float t_hi, float & a, float & b)
{
    const RcV3 o_l = rcXformPoint(ob.r, ob.t, o);
    const RcV3 d_l = rcRotate(ob.r, d);
    const float hx = rpmath::gzm::fmax_(ob.half[0], kRcObscurantMinHalf);
    const float hy = rpmath::gzm::fmax_(ob.half[1], kRcObscurantMinHalf);
    const float hz = rpmath::gzm::fmax_(ob.half[2], kRcObscurantMinHalf);

    float lo = t_lo, hi = t_hi;
    switch (ob.type) {
        case ObscurantType::kBox: {
            if (!rcSlabAxis(o_l.x, d_l.x, -hx, hx, lo, hi)) return false;
            if (!rcSlabAxis(o_l.y, d_l.y, -hy, hy, lo, hi)) return false;
            if (!rcSlabAxis(o_l.z, d_l.z, -hz, hz, lo, hi)) return false;
            break;
        }
        case ObscurantType::kEllipsoid: {
            const RcV3 os{o_l.x / hx, o_l.y / hy, o_l.z / hz};
            const RcV3 ds{d_l.x / hx, d_l.y / hy, d_l.z / hz};
            const float qa = rcDot(ds, ds);
            if (qa < 1.0e-20f) return false;
            const float qb = rcDot(os, ds);
            const float qc = rcDot(os, os) - 1.0f;
            const float disc = qb * qb - qa * qc;
            if (disc < 0.0f) return false;
            const float sq = rpmath::gzm::sqrt_(disc);
            lo = rpmath::gzm::fmax_(lo, (-qb - sq) / qa);
            hi = rpmath::gzm::fmin_(hi, (-qb + sq) / qa);
            break;
        }
        case ObscurantType::kCylinder: {
            const float ox = o_l.x / hx, oy = o_l.y / hy;
            const float dx = d_l.x / hx, dy = d_l.y / hy;
            const float qa = dx * dx + dy * dy;
            const float qc = ox * ox + oy * oy - 1.0f;
            if (qa < 1.0e-20f) {
                // Parallel to the axis: inside the tube for all t, or never.
                if (qc > 0.0f) return false;
            } else {
                const float qb = ox * dx + oy * dy;
                const float disc = qb * qb - qa * qc;
                if (disc < 0.0f) return false;
                const float sq = rpmath::gzm::sqrt_(disc);
                lo = rpmath::gzm::fmax_(lo, (-qb - sq) / qa);
                hi = rpmath::gzm::fmin_(hi, (-qb + sq) / qa);
            }
            if (!rcSlabAxis(o_l.z, d_l.z, -hz, hz, lo, hi)) return false;
            break;
        }
    }
    a = lo;
    b = hi;
    return b > a;
}

/// True when the world-space point lies inside the volume.
GZ_OUSTER_HD inline bool rcObscurantContains(const RcObscurant & ob, RcV3 p_w)
{
    const RcV3 p = rcXformPoint(ob.r, ob.t, p_w);
    const float hx = rpmath::gzm::fmax_(ob.half[0], kRcObscurantMinHalf);
    const float hy = rpmath::gzm::fmax_(ob.half[1], kRcObscurantMinHalf);
    const float hz = rpmath::gzm::fmax_(ob.half[2], kRcObscurantMinHalf);
    switch (ob.type) {
        case ObscurantType::kBox:
            return rpmath::gzm::fabs_(p.x) <= hx &&
                   rpmath::gzm::fabs_(p.y) <= hy &&
                   rpmath::gzm::fabs_(p.z) <= hz;
        case ObscurantType::kEllipsoid: {
            const RcV3 s{p.x / hx, p.y / hy, p.z / hz};
            return rcDot(s, s) <= 1.0f;
        }
        case ObscurantType::kCylinder: {
            const float sx = p.x / hx, sy = p.y / hy;
            return sx * sx + sy * sy <= 1.0f &&
                   rpmath::gzm::fabs_(p.z) <= hz;
        }
    }
    return false;
}

/// One-way optical depth τ = ∫σ_ext ds accumulated over every obscurant on
/// [t_lo, t_hi]. Overlapping volumes are handled EXACTLY without any interval
/// merging, because the integral of a sum is the sum of the integrals:
/// ∫Σσ_i ds = Σ∫σ_i ds.
///
/// The return value is the PHYSICAL optical depth, which is what the passive
/// NEAR_IR airlight transfer uses. `tau_eff_out` (optional) receives the
/// ATTENUATING optical depth ∫η·σ_ext ds that the active lidar round trip
/// sees instead (see RcObscurant::ms_factor); the two are equal when every
/// volume on the path leaves η at 1. `albedo_out` (optional) receives the
/// τ-weighted mean single-scattering albedo over the same path.
GZ_OUSTER_HD inline float rcOpticalDepth(const ScanParams & sp,
    RcV3 o, RcV3 d, float t_lo, float t_hi, float * albedo_out = nullptr,
    float * tau_eff_out = nullptr)
{
    float tau = 0.0f;
    float tau_eff = 0.0f;
    float w_albedo = 0.0f;
    for (int i = 0; i < sp.n_obscurants; ++i) {
        const RcObscurant & ob = sp.obscurants[i];
        if (ob.sigma <= 0.0f) continue;
        float a, b;
        if (!rcObscurantSpan(ob, o, d, t_lo, t_hi, a, b)) continue;
        const float dtau = ob.sigma * (b - a);
        tau += dtau;
        tau_eff += dtau * ob.ms_factor;
        w_albedo += dtau * ob.albedo;
    }
    if (albedo_out != nullptr) {
        *albedo_out = (tau > 0.0f) ? (w_albedo / tau) : 0.0f;
    }
    if (tau_eff_out != nullptr) *tau_eff_out = tau_eff;
    return tau;
}

/// Two-way transmittance exp(−2·∫η·σ_ext ds) along ONE STRAIGHT LEG.
///
/// This belongs in the arbitration between candidate returns, not after it.
/// The surface, behind-glass and mirror-ghost candidates on a single beam sit
/// at different depths and therefore lose different amounts of light, so
/// comparing their UNATTENUATED powers picks the wrong one as soon as the
/// medium is thick enough. For a pane at 5 m in front of a bright object at
/// 12 m the crossover is only σ ≈ 0.23 /m — a visibility of about 17 m, i.e.
/// ordinary smoke, not a corner case.
///
/// Transmittance composes multiplicatively over legs, which is what the
/// mirror ghost needs: it travels out to the mirror and then along a
/// reflected leg, so each leg gets its own call and the two are multiplied.
/// Integrating a single straight line to the ghost's reported range would
/// sample the medium behind the mirror, which the pulse never enters.
///
/// Returns exactly 1 when no media are configured, keeping the whole feature
/// free for worlds that do not use it.
GZ_OUSTER_HD inline float rcMediumTransmit2(const ScanParams & sp,
    RcV3 o, RcV3 d, float t_lo, float t_hi)
{
    if (sp.n_obscurants <= 0 || t_hi <= t_lo) return 1.0f;
    float tau_eff = 0.0f;
    (void)rcOpticalDepth(sp, o, d, t_lo, t_hi, nullptr, &tau_eff);
    if (tau_eff <= kRcTauMin) return 1.0f;
    return rpmath::gzm::exp_(-2.0f * tau_eff);
}

/// Deterministic uniform draw in (0, 1), shared by every backend.
///
/// Native GPU RNGs would make CUDA, HIP and SYCL produce different clouds.
/// This integer hash instead gives each (pixel, scan, stream) tuple a stable
/// draw while the scan salt still makes the plume evolve over time.
GZ_OUSTER_HD inline float rcHashUnit(uint32_t pixel, uint32_t salt,
                                     uint32_t stream = 0)
{
    uint32_t x = pixel * 0x9E3779B9u ^ salt * 0x85EBCA6Bu ^
                 stream * 0xC2B2AE35u;
    x ^= x >> 16;
    x *= 0x7FEB352Du;
    x ^= x >> 15;
    x *= 0x846CA68Bu;
    x ^= x >> 16;
    const float u = static_cast<float>(x) * 2.3283064e-10f;
    return rpmath::gzm::fmin_(rpmath::gzm::fmax_(u, 1.0e-7f), 0.9999999f);
}

/// Integral of exp(-2*k*x) over [0, length], evaluated without catastrophic
/// cancellation for optically thin segments.
GZ_OUSTER_HD inline float rcDecayIntegral(float k, float length)
{
    const float q = 2.0f * k * length;
    if (q < 1.0e-3f) {
        return length * (1.0f - 0.5f * q + q * q / 6.0f);
    }
    return (1.0f - rpmath::gzm::exp_(-q)) / (2.0f * k);
}

/// Mean rejection acceptance under one segment's analytic proposal.
///
/// Three-point Gauss-Legendre quadrature is evaluated in proposal-CDF space,
/// where the integrand is only the bounded acceptance ratio. The tighter-
/// envelope choice in rcSelectMediumSegment keeps that ratio smooth even for
/// optically thick or long spans, making this a cheap, stable estimate of the
/// physical q-profile mass rather than its upper envelope.
GZ_OUSTER_HD inline float rcProposalAcceptanceMean(
    bool use_exp, float k, float length, float r0, float r1)
{
    float mean = 0.0f;
    for (int j = 0; j < 3; ++j) {
        const float u = j == 0 ? 0.1127016654f
                      : j == 1 ? 0.5f : 0.8872983346f;
        const float w = j == 1 ? 0.4444444444f : 0.2777777778f;
        float x = 0.0f;
        float accept = 0.0f;
        if (use_exp) {
            const float q = 2.0f * k * length;
            x = (q < 1.0e-3f)
                ? u * length
                : -rpmath::gzm::log_(
                    1.0f - u * (1.0f - rpmath::gzm::exp_(-q))) /
                    (2.0f * k);
            const float ratio = r0 / (r0 + x);
            accept = ratio * ratio;
        } else {
            const float inv_r = 1.0f / r0 -
                u * (1.0f / r0 - 1.0f / r1);
            x = 1.0f / inv_r - r0;
            accept = rpmath::gzm::exp_(-2.0f * k * x);
        }
        mean += w * accept;
    }
    return mean;
}

/// Walk the constant-property segments made by a set of ray/volume spans.
///
/// The desired range density is the received backscatter-power profile
///
///   q(t) = beta(t) * exp(-2*tau_eff(t)) / (t + n_off)^2.
///
/// Within one segment beta and k=d(tau_eff)/dt are constant. Two simple
/// proposal envelopes are available there: retain the exponential and bound
/// 1/R^2 by its near value, or retain 1/R^2 and bound the exponential by its
/// near value. The tighter envelope is recorded implicitly by `use_exp_out`.
/// A negative `pick` only computes the total proposal mass; otherwise this
/// selects the segment containing that cumulative mass.
GZ_OUSTER_HD inline bool rcSelectMediumSegment(
    const ScanParams & sp, RcV3 o, RcV3 d, float n_off, float t_end,
    const float * sa, const float * sb, const int * oi, int n,
    float pick, float & total_out, float & a_out, float & b_out,
    float & beta_out, float & k_out, float & tau_out, bool & use_exp_out,
    float * profile_mass_out)
{
    const float t_start = rpmath::gzm::fmax_(sp.near_clip, 0.0f);
    if (t_start >= t_end) {
        total_out = 0.0f;
        return false;
    }

    float tau_eff = 0.0f;
    (void)rcOpticalDepth(sp, o, d, 0.0f, t_start, nullptr, &tau_eff);
    float total = 0.0f;
    float profile_mass = 0.0f;
    float t = t_start;
    for (int step = 0; step < 2 * kMaxObscurants + 1; ++step) {
        float next = t_end;
        float beta = 0.0f;
        float k = 0.0f;
        for (int i = 0; i < n; ++i) {
            const RcObscurant & ob = sp.obscurants[oi[i]];
            if (t >= sa[i] && t < sb[i]) {
                k += ob.sigma * ob.ms_factor;
                if (ob.lidar_ratio > 0.0f) beta += ob.sigma / ob.lidar_ratio;
                if (sb[i] < next) next = sb[i];
            } else if (sa[i] > t && sa[i] < next) {
                next = sa[i];
            }
        }
        if (next <= t) break;

        if (beta > 0.0f && k > 0.0f) {
            const float r0 = rpmath::gzm::fmax_(t + n_off, 1.0e-3f);
            const float r1 = rpmath::gzm::fmax_(next + n_off, r0 + 1.0e-6f);
            const float trans = rpmath::gzm::exp_(-2.0f * tau_eff);
            const float w_exp = beta * trans * rcDecayIntegral(k, next - t) /
                                (r0 * r0);
            const float w_inv = beta * trans * (1.0f / r0 - 1.0f / r1);
            const bool use_exp = w_exp <= w_inv;
            const float weight = use_exp ? w_exp : w_inv;
            if (weight > 0.0f) {
                profile_mass += weight * rcProposalAcceptanceMean(
                    use_exp, k, next - t, r0, r1);
                if (pick >= 0.0f && pick < total + weight) {
                    total_out = total + weight;
                    if (profile_mass_out != nullptr) {
                        *profile_mass_out = profile_mass;
                    }
                    a_out = t;
                    b_out = next;
                    beta_out = beta;
                    k_out = k;
                    tau_out = tau_eff;
                    use_exp_out = use_exp;
                    return true;
                }
                total += weight;
            }
        }

        tau_eff += k * (next - t);
        t = next;
        if (t >= t_end) break;
    }
    total_out = total;
    if (profile_mass_out != nullptr) *profile_mass_out = profile_mass;
    return false;
}

/// Draw one detected range from the complete range-resolved backscatter
/// profile. Overlap is exact: each piecewise segment sums beta and attenuating
/// extinction from every active volume, even when their lidar ratios differ.
///
/// First, the integrated profile becomes an expected photon count
/// lambda = pi*base_signal*integral(q dr). A Poisson zero-count gate therefore
/// leaves weak / distant aerosol beams empty instead of turning every volume
/// intersection into a point. Conditional on a detection, rejection sampling
/// uses the tighter of the two analytic envelopes described above, so both
/// two-way transmittance and 1/R^2 spreading are present in the range draw.
/// Six attempts keep device execution bounded; numerical rejection failure
/// simply means this pulse produced no medium candidate.
GZ_OUSTER_HD inline bool rcSampleMediumReturn(
    const ScanParams & sp, RcV3 o, RcV3 d, float t_end, float n_off,
    uint32_t pixel, float & range_out, float & rho_out)
{
    if (sp.pulse_gate_m <= 0.0f) return false;

    float sa[kMaxObscurants], sb[kMaxObscurants];
    int oi[kMaxObscurants];
    int n = 0;
    for (int i = 0; i < sp.n_obscurants; ++i) {
        const RcObscurant & ob = sp.obscurants[i];
        if (ob.sigma <= 0.0f) continue;
        float a, b;
        if (!rcObscurantSpan(ob, o, d, 0.0f, t_end, a, b)) continue;
        sa[n] = a;
        sb[n] = b;
        oi[n] = i;
        ++n;
    }
    if (n == 0) return false;

    float total = 0.0f;
    float profile_mass = 0.0f;
    float a = 0.0f, b = 0.0f, beta = 0.0f, k = 0.0f, tau = 0.0f;
    bool use_exp = false;
    (void)rcSelectMediumSegment(sp, o, d, n_off, t_end, sa, sb, oi, n,
                                -1.0f, total, a, b, beta, k, tau, use_exp,
                                &profile_mass);
    if (total <= 0.0f || profile_mass <= 0.0f) return false;

    // Each range gate carries base_signal*pi*q(r)*dr expected photons; their
    // sum over the column is Poisson with this mean. P(N>0)=1-exp(-lambda).
    const float lambda = rpmath::gzm::fmax_(sp.base_signal, 0.0f) *
                         rpmath::kPi * profile_mass;
    const float detection_probability =
        1.0f - rpmath::gzm::exp_(-lambda);
    if (rcHashUnit(pixel, sp.rng_salt, 0u) > detection_probability) {
        return false;
    }

    constexpr uint32_t kAttempts = 6;
    for (uint32_t attempt = 0; attempt < kAttempts; ++attempt) {
        const uint32_t stream = 1u + 3u * attempt;
        const float pick = rcHashUnit(pixel, sp.rng_salt, stream) * total;
        float walked = 0.0f;
        if (!rcSelectMediumSegment(sp, o, d, n_off, t_end, sa, sb, oi, n,
                                   pick, walked, a, b, beta, k, tau,
                                   use_exp, nullptr)) {
            continue;
        }

        const float u = rcHashUnit(pixel, sp.rng_salt, stream + 1u);
        const float length = b - a;
        const float r0 = rpmath::gzm::fmax_(a + n_off, 1.0e-3f);
        float x = 0.0f;
        float accept = 0.0f;
        if (use_exp) {
            const float q = 2.0f * k * length;
            x = (q < 1.0e-3f)
                ? u * length
                : -rpmath::gzm::log_(
                    1.0f - u * (1.0f - rpmath::gzm::exp_(-q))) /
                    (2.0f * k);
            const float ratio = r0 / (r0 + x);
            accept = ratio * ratio;
        } else {
            const float r1 = rpmath::gzm::fmax_(b + n_off, r0 + 1.0e-6f);
            const float inv_r = 1.0f / r0 -
                u * (1.0f / r0 - 1.0f / r1);
            x = 1.0f / inv_r - r0;
            accept = rpmath::gzm::exp_(-2.0f * k * x);
        }

        if (rcHashUnit(pixel, sp.rng_salt, stream + 2u) > accept) continue;

        const float tau_s = tau + k * x;
        const float rho = rpmath::kPi * beta * sp.pulse_gate_m *
                          rpmath::gzm::exp_(-2.0f * tau_s);
        if (rho <= 0.0f) return false;
        range_out = a + x + n_off;
        rho_out = rho;
        return true;
    }
    return false;
}

/// Apply the participating medium to an already-resolved return.
///
/// On entry `range`/`rho` describe the hard-target candidate (`rho == 0` and
/// a non-finite `range` when the beam missed everything); on exit they
/// describe what the detector actually reports. Three effects, in order:
///
///  1. **Two-way extinction.** The pulse crosses the medium going out and
///     coming back, so the target's apparent reflectance is scaled by
///     exp(−2·η·τ) — Platt's form, where η ≤ 1 credits back the
///     forward-scattered light a real receiver still collects (see
///     RcObscurant::ms_factor). Because the whole downstream pipeline is
///     driven by that one number, this single multiply correctly dims
///     SIGNAL, lowers the
///     calibrated REFLECTIVITY byte, widens the range noise and — through
///     the √ρ detection limit in rpmath::dropoutProbability — makes targets
///     disappear entirely once the smoke is thick enough. No downstream
///     stage needs to know that smoke exists.
///
///  2. **Backscatter from the medium itself.** A slab of medium at range r
///     returns P(r) ∝ β_π·ΔR·exp(−2τ(r))/r², which is exactly the form the
///     rest of the pipeline expects (`base_signal·ρ/r²`), so the medium gets
///     an apparent reflectance ρ_med = π·β_π·ΔR·exp(−2τ_eff(r)). The
///     integrated profile first passes a Poisson photon-count gate, then a
///     range is drawn conditional on detection. Both operations include
///     overlapping media, two-way extinction and 1/r² spreading. This leaves
///     weak intersections empty and represents the pulse-to-pulse variation
///     of a finite aerosol population instead of turning a homogeneous volume
///     into a solid object. The sampled medium return then competes with the
///     attenuated surface by received power, the same rule the glass/mirror
///     paths use.
///
///  3. **NEAR_IR airlight.** Illuminated smoke scatters ambient light into
///     the receiver, so the ambient channel composites as
///     L = L_target·exp(−τ) + ω·illum·(1 − exp(−τ)) — Koschmieder's airlight
///     equation. Dense smoke therefore GLOWS in NEAR_IR while it darkens the
///     laser channels, matching what a real Ouster shows in fog or smoke.
///
/// Extinction of the hard target is NOT done here: each candidate carries its
/// own exp(−2·η·τ) into the arbitration in rcCastOneRay (rcMediumTransmit2),
/// because candidates at different depths lose different amounts of light and
/// the comparison must be made on what the detector would actually receive.
/// The mirror ghost composes the transmittance of its two legs rather than
/// integrating the straight line to its folded reported range.
///
/// References: Rasshofer et al., Adv. Radio Sci. 9, 2011 (lidar in adverse
/// weather); Hahner et al., *Fog Simulation on Real LiDAR Point Clouds*,
/// ICCV 2021 (arXiv:2108.05249) and Kilic et al., *LISA*, arXiv:2107.07004
/// (the same extinction + medium-backscatter decomposition applied as point
/// cloud augmentation); Koschmieder 1924 for the airlight composite.
GZ_OUSTER_HD inline void rcApplyObscurants(
    const ScanParams & sp, RcV3 o, RcV3 d, float n_off, float t_los,
    uint32_t pixel, float & range, float & rho, float * nir_val)
{
    const bool hit = rpmath::gzm::isfinite_(range);
    // `t_los` is the forward extent of the LINE OF SIGHT, supplied by the
    // caller — the geometric depth at which the sight line terminates, or the
    // whole range budget when the beam missed (smoke against open sky still
    // returns). It is deliberately NOT `range - n_off`: a mirror ghost is
    // reported at its folded path length while the sight line stops at the
    // mirror, and the medium beyond the mirror is not on the beam's path.
    if (t_los <= 0.0f) return;

    // The hard-target candidate arrives already attenuated: each candidate
    // multiplies in its own exp(−2·η·τ) before the arbitration in
    // rcCastOneRay, because candidates at different depths lose different
    // amounts of light and the comparison has to be made on what the
    // detector would actually receive.
    float path_albedo = 0.0f;
    const float tau =
        rcOpticalDepth(sp, o, d, 0.0f, t_los, &path_albedo, nullptr);
    if (tau <= kRcTauMin) return;

    // The PASSIVE ambient channel uses the PHYSICAL depth τ, not the
    // η-weighted one the laser round trip sees: its Koschmieder composite
    // below already accounts for light scattered back into a wide field of
    // view — that IS the airlight term. Applying η here too would correct
    // for the same physics twice.
    const float trans_ambient = rpmath::gzm::exp_(-tau);

    // 1. NEAR_IR airlight, over the whole sight line regardless of which
    //    candidate won, because Ouster's ambient channel is a passive
    //    measurement of the column rather than a gated sample at the
    //    reported range.
    if (nir_val != nullptr) {
        const float illum = sp.sun_ambient + sp.sun_diffuse;
        *nir_val = *nir_val * trans_ambient +
                   path_albedo * illum * (1.0f - trans_ambient);
    }

    // 2. Stochastic range draw from the complete received-backscatter profile.
    //    The scan salt evolves the plume; the shared downstream model still
    //    supplies electronic shot/range/dropout noise afterward.
    float range_m = 0.0f, rho_m = 0.0f;
    if (!rcSampleMediumReturn(
            sp, o, d, t_los, n_off, pixel, range_m, rho_m)) return;

    // Strongest-return arbitration by received power ρ/R² (same rule as the
    // glass and mirror candidates). Written as a cross-multiplication so the
    // miss case never has to divide by an infinite range.
    if (!hit || rho_m * range * range > rho * range_m * range_m) {
        range = range_m;
        rho = rho_m;
    }
}

/// Cast one output pixel (beam × measurement id) against the whole scene.
/// Writes the reported Ouster range (metres; `inf_value` for a miss — the
/// value satisfying the XYZ-LUT reconstruction, see castScan docs) and the
/// nearest hit's APPARENT reflectance: diffuse reflectance × cos(incidence)
/// plus the material's specular lobe (0 on a miss; an omitted laser_retro uses
/// ScanParams::fallback_retro).
///
/// `col_r`/`col_t` (optional, both or neither): per-COLUMN sensor→world
/// poses for motion distortion — column m casts from col_r[9m..]/col_t[3m..]
/// instead of the single sensor_r/sensor_t pose, modelling the rolling-
/// shutter sweep of a spinning lidar (a scan's columns are acquired over a
/// full period; ego motion during it skews the cloud by roughly the
/// distance travelled — see docs/MODEL_REFERENCES.md §9).
GZ_OUSTER_HD inline void rcCastOneRay(
    const RcInstance * instances, int n_instances,
    const float * verts, const float * texcoords,
    const int * tris, const int * order, const MeshBvhNode * nodes,
    const uint8_t * response_texels,
    const InstanceXform * xforms,
    const float * beam_alt_deg, const float * beam_az_deg,
    const float * sensor_r, const float * sensor_t,
    const ScanParams & sp, int idx, float inf_value,
    float & range_out, float & retro_out,
    const float * col_r = nullptr, const float * col_t = nullptr,
    float * nir_out = nullptr,
    const MeshBvhNode * tlas_nodes = nullptr,
    const int * tlas_order = nullptr,
    int n_tlas_nodes = 0)
{
    const int beam = idx / sp.W;
    const int m = idx % sp.W;
    const float deg_per_col = 360.0f / static_cast<float>(sp.W);
    const float n_off = sp.beam_origin_m;

    const float * sr = sensor_r;
    const float * st = sensor_t;
    if (col_r != nullptr && col_t != nullptr) {
        sr = col_r + 9 * m;
        st = col_t + 3 * m;
    }

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

    const RcV3 o = rcXformPoint(sr, st, o_s);
    const RcV3 d = rcRotate(sr, d_s);

    const float t_budget = sp.max_range - n_off;

    int inst0 = -1, tri0 = -1;
    const float t0 = rcNearestHit(instances, n_instances, verts, tris, order,
                                  nodes, xforms, o, d, sp.near_clip, t_budget,
                                  inst0, tri0,
                                  tlas_nodes, tlas_order, n_tlas_nodes);
    if (inst0 < 0) {
        // A beam that hits nothing can still return from a participating
        // medium along its path — smoke against open sky is a return, not a
        // miss — so the obscurant stage runs before giving up.
        float miss_range = inf_value;
        float miss_rho = 0.0f;
        float miss_nir = 0.0f;
        if (sp.n_obscurants > 0) {
            rcApplyObscurants(sp, o, d, n_off, t_budget,
                              static_cast<uint32_t>(idx),
                              miss_range, miss_rho,
                              nir_out ? &miss_nir : nullptr);
        }
        range_out = miss_range;
        retro_out = miss_rho;
        if (nir_out) *nir_out = miss_nir;
        return;
    }

    // Winning candidate's hit data, for the NEAR_IR ambient model below
    // (surface candidate adopts first; transmission/ghost override on win).
    int w_inst = inst0, w_tri = tri0;
    RcV3 w_o = o, w_d = d;
    float w_t = t0;

    // Apparent reflectance of the front surface: kd·cos(α) + ks·cos(2α)⁸
    // (see rcApparentReflectance) — the extended-Lambertian lidar equation
    // P ∝ ρ_app/R² plus the monostatic specular lobe. Folding the angular
    // terms in here makes the downstream signal, reflectivity byte and
    // noise weighting all respond to oblique/glossy surfaces the way a real
    // return does. A missing laser_retro is resolved here to fallback_retro;
    // an explicitly authored zero remains zero.
    RcMaterialSample material0;
    float rho = rcHitReflectance(instances, xforms, verts, tris,
                                 texcoords, response_texels,
                                 o, d, t0, inst0, tri0, sp.fallback_retro,
                                 &material0);
    const float tau = rpmath::gzm::fmin_(
        rpmath::gzm::fmax_(material0.transmit, 0.0f), 1.0f);
    rho *= (1.0f - tau);
    float range = t0 + n_off;

    // Candidates sit at DIFFERENT depths, so a medium dims them by different
    // amounts and the arbitration below has to compare what the detector
    // would actually receive rather than the clear-air powers.
    //
    // Every candidate travels the leg [0, t0] on (o, d) — the surface stops
    // there, the behind-glass return continues past it, the mirror ghost
    // turns there. Its transmittance is therefore a common POSITIVE factor,
    // which cannot change the argmax: factor it out, arbitrate on the extra
    // depth each candidate adds, and apply the shared leg once to the
    // winner. Two things fall out of that. In dense smoke the shared factor
    // can underflow to zero, which would collapse every candidate to the
    // same value and make the comparison meaningless — the relative form
    // stays well-conditioned. And the glass candidate then only has to
    // integrate its extra leg rather than the whole path.
    //
    // `rho` holds the surface candidate, whose relative factor is 1.
    //
    // Forward extent of the LINE OF SIGHT, which is not the reported range:
    // a mirror ghost is reported at its total folded path length, but the
    // sight line still stops at the mirror. Drives the ambient airlight and
    // the medium's own return below.
    float t_los = t0;

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
                                      t_budget - seg_start, inst1, tri1,
                                      tlas_nodes, tlas_order, n_tlas_nodes);
        if (inst1 >= 0) {
            const float rho1 = rcHitReflectance(instances, xforms, verts,
                                                tris, texcoords,
                                                response_texels,
                                                p, d, t1, inst1, tri1,
                                                sp.fallback_retro) *
                               tau * tau;
            const float range1 = seg_start + t1 + n_off;
            // Same straight ray, but deeper. Only the EXTRA leg past t0
            // counts here; the shared prefix is applied to the winner below.
            const float t_far = seg_start + t1;
            const float rho1_rel =
                rho1 * rcMediumTransmit2(sp, o, d, t0, t_far);
            if (rho1_rel * range * range > rho * range1 * range1) {
                rho = rho1_rel;
                range = range1;
                t_los = t_far;
                w_inst = inst1; w_tri = tri1;
                w_o = p; w_d = d; w_t = t1;
            }
        }
    }

    // Mirror ghost (Velas et al., arXiv:1909.12483 §III): a strongly
    // specular surface bounces the beam onto a third object, whose diffuse
    // return retraces the path — the sensor, unaware, reports a point
    // BEHIND the mirror along the original beam at the total path length
    // t0 + t2. The pulse interacts with the mirror twice, hence the
    // ((1−τ)·ks)² weight (glass ghosts are weak, true mirrors strong).
    if (material0.spec >= kRcMirrorMin) {
        const InstanceXform & x0 = xforms[inst0];
        const RcV3 o_l = rcXformPoint(x0.r, x0.t, o);
        const RcV3 d_l = rcRotate(x0.r, d);
        const RcV3 p_l{o_l.x + t0 * d_l.x, o_l.y + t0 * d_l.y,
                       o_l.z + t0 * d_l.z};
        const RcV3 n_l = rcSurfaceNormalLocal(instances[inst0], verts, tris,
                                              p_l, tri0);
        const float nn = rpmath::gzm::sqrt_(rcDot(n_l, n_l));
        if (nn > 1.0e-12f) {
            // World-frame unit normal (x0.r is world→local; transpose maps
            // back), then the specular reflection of d about it. The
            // reflection is invariant to the normal's sign.
            RcV3 n_w = rcRotateT(x0.r, n_l);
            n_w = RcV3{n_w.x / nn, n_w.y / nn, n_w.z / nn};
            const float dn = rcDot(d, n_w);
            const RcV3 refl{d.x - 2.0f * dn * n_w.x,
                            d.y - 2.0f * dn * n_w.y,
                            d.z - 2.0f * dn * n_w.z};
            const RcV3 hit{o.x + d.x * t0, o.y + d.y * t0, o.z + d.z * t0};
            const RcV3 g0{hit.x + refl.x * kRcSegEps,
                          hit.y + refl.y * kRcSegEps,
                          hit.z + refl.z * kRcSegEps};
            int inst2 = -1, tri2 = -1;
            const float t2 = rcNearestHit(instances, n_instances, verts,
                                          tris, order, nodes, xforms, g0,
                                          refl, kRcSegTmin,
                                          t_budget - t0 - kRcSegEps,
                                          inst2, tri2,
                                          tlas_nodes, tlas_order,
                                          n_tlas_nodes);
            if (inst2 >= 0) {
                const float mirror_eff = (1.0f - tau) * material0.spec;
                const float rho2 =
                    rcHitReflectance(instances, xforms, verts, tris,
                                     texcoords, response_texels,
                                     g0, refl, t2, inst2, tri2,
                                     sp.fallback_retro) *
                    mirror_eff * mirror_eff;
                const float range2 = t0 + kRcSegEps + t2 + n_off;
                // The ghost's path BENDS: out to the mirror — which is the
                // shared prefix leg, applied below — and then along the
                // reflected leg, which is all that is relative here.
                // Transmittance composes over legs; a single straight
                // integral to range2 would instead pass through the mirror
                // and sample medium the pulse never met.
                const float rho2_rel = rho2 *
                    rcMediumTransmit2(sp, g0, refl, 0.0f, t2);
                if (rho2_rel * range * range > rho * range2 * range2) {
                    rho = rho2_rel;
                    range = range2;
                    // The sight line still terminates at the mirror.
                    t_los = t0;
                    w_inst = inst2; w_tri = tri2;
                    w_o = g0; w_d = refl; w_t = t2;
                }
            }
        }
    }

    // The leg every candidate shared, applied exactly once now that the
    // winner is known (see the factoring note above the glass block).
    rho *= rcMediumTransmit2(sp, o, d, 0.0f, t0);

    // NEAR_IR ambient factor at the winning hit: albedo × Lambert sun term
    // (view-independent — ambient radiance off a Lambertian surface does not
    // depend on the sensor's incidence angle, unlike the laser return; see
    // ScanParams sun fields and docs/MODEL_REFERENCES.md §6).
    float nir_val = 0.0f;
    if (nir_out) {
        const InstanceXform & xw = xforms[w_inst];
        const RcV3 o_l = rcXformPoint(xw.r, xw.t, w_o);
        const RcV3 d_l = rcRotate(xw.r, w_d);
        const RcV3 p_l{o_l.x + w_t * d_l.x, o_l.y + w_t * d_l.y,
                       o_l.z + w_t * d_l.z};
        const RcV3 n_l = rcSurfaceNormalLocal(instances[w_inst], verts, tris,
                                              p_l, w_tri);
        float illum = sp.sun_ambient;
        const float nn = rpmath::gzm::sqrt_(rcDot(n_l, n_l));
        if (sp.sun_diffuse > 0.0f && nn > 1.0e-12f) {
            RcV3 n_w = rcRotateT(xw.r, n_l);
            n_w = RcV3{n_w.x / nn, n_w.y / nn, n_w.z / nn};
            // Orient the normal toward the sensor (the visible side).
            if (rcDot(n_w, w_d) > 0.0f) {
                n_w = RcV3{-n_w.x, -n_w.y, -n_w.z};
            }
            const float lambert = -(n_w.x * sp.sun_dir[0] +
                                    n_w.y * sp.sun_dir[1] +
                                    n_w.z * sp.sun_dir[2]);
            illum += sp.sun_diffuse * rpmath::gzm::fmax_(lambert, 0.0f);
        }
        const RcMaterialSample winning_material = rcMaterialAtHit(
            instances[w_inst], verts, tris, texcoords, response_texels,
            p_l, w_tri, sp.fallback_retro);
        nir_val = winning_material.nir * illum;
    }

    // Smoke / dust / fog: the winning candidate is already attenuated by its
    // own path (above); what remains is the medium's competing backscatter
    // return and the NEAR_IR airlight, both over the sight line.
    if (sp.n_obscurants > 0) {
        rcApplyObscurants(sp, o, d, n_off, t_los,
                          static_cast<uint32_t>(idx), range, rho,
                          nir_out ? &nir_val : nullptr);
    }

    range_out = range;
    retro_out = rho;
    if (nir_out) *nir_out = nir_val;
}

}  // namespace rc
}  // namespace gz_gpu_ouster_lidar
