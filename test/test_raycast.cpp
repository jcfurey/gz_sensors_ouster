// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0
//
// Full per-beam raycast mode: intersector correctness (sphere / box /
// cylinder / plane / mesh), BVH-vs-brute-force equivalence, beam-origin
// parallax, retro of the nearest hit, near-clip behaviour, axis-parallel
// slab robustness, and the castScan / processDepth entries through
// RayProcessor (CPU backend — the GPU kernels run the identical shared
// rcCastOneRay and are compile-checked by the smoke CI jobs).

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include "gz_gpu_ouster_lidar/ray_processor.hpp"
#include "raycast_scene.hpp"

namespace gz_gpu_ouster_lidar {

namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();
const float kIdentityR[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
const float kZeroT[3] = {0, 0, 0};

rc::InstanceXform xformAt(const rc::Scene & scene, int instance_idx,
                          float x, float y, float z,
                          const float * r = kIdentityR)
{
    const float t[3] = {x, y, z};
    rc::InstanceXform out;
    scene.computeXform(instance_idx, r, t, out);
    return out;
}

rc::ScanParams scanParams(int H, int W, float beam_origin = 0.0f)
{
    rc::ScanParams sp;
    sp.H = H;
    sp.W = W;
    sp.max_range = 120.0f;
    sp.near_clip = 0.3f;
    sp.beam_origin_m = beam_origin;
    return sp;
}

/// Cast with an identity sensor pose via the CPU reference entry.
/// Columns: m = 0 → azimuth 0 (+x), m = W/4 → -90° (-y), etc.
std::vector<float> cast(const rc::Scene & scene,
                        const std::vector<rc::InstanceXform> & xf,
                        const std::vector<float> & alt,
                        const std::vector<float> & az,
                        const rc::ScanParams & sp,
                        std::vector<float> * retro_out = nullptr)
{
    std::vector<float> range(static_cast<size_t>(sp.H) * sp.W);
    if (retro_out) retro_out->resize(range.size());
    rc::castScan(scene.view(), xf.data(), alt.data(), az.data(),
                 kIdentityR, kZeroT, sp, range.data(),
                 retro_out ? retro_out->data() : nullptr);
    return range;
}

/// Axis-aligned box mesh (12 triangles) with the given half extents.
void makeBoxMesh(float hx, float hy, float hz,
                 std::vector<float> & verts, std::vector<int> & tris)
{
    const float v[8][3] = {
        {-hx, -hy, -hz}, {hx, -hy, -hz}, {hx, hy, -hz}, {-hx, hy, -hz},
        {-hx, -hy, hz},  {hx, -hy, hz},  {hx, hy, hz},  {-hx, hy, hz}};
    for (auto & p : v) {
        verts.insert(verts.end(), {p[0], p[1], p[2]});
    }
    const int f[12][3] = {
        {0, 2, 1}, {0, 3, 2},  // -z
        {4, 5, 6}, {4, 6, 7},  // +z
        {0, 1, 5}, {0, 5, 4},  // -y
        {2, 3, 7}, {2, 7, 6},  // +y
        {1, 2, 6}, {1, 6, 5},  // +x
        {3, 0, 4}, {3, 4, 7}};  // -x
    for (auto & tri : f) {
        tris.insert(tris.end(), {tri[0], tri[1], tri[2]});
    }
}

/// Reference Möller–Trumbore for the brute-force comparison.
float refTriangle(const float o[3], const float d[3],
                  const float * v0, const float * v1, const float * v2)
{
    const float e1[3] = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
    const float e2[3] = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
    const float p[3] = {d[1] * e2[2] - d[2] * e2[1],
                        d[2] * e2[0] - d[0] * e2[2],
                        d[0] * e2[1] - d[1] * e2[0]};
    const float det = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2];
    if (std::fabs(det) < 1e-8f) return -1.0f;
    const float inv = 1.0f / det;
    const float s[3] = {o[0] - v0[0], o[1] - v0[1], o[2] - v0[2]};
    const float u = (s[0] * p[0] + s[1] * p[1] + s[2] * p[2]) * inv;
    if (u < 0.0f || u > 1.0f) return -1.0f;
    const float q[3] = {s[1] * e1[2] - s[2] * e1[1],
                        s[2] * e1[0] - s[0] * e1[2],
                        s[0] * e1[1] - s[1] * e1[0]};
    const float w = (d[0] * q[0] + d[1] * q[1] + d[2] * q[2]) * inv;
    if (w < 0.0f || u + w > 1.0f) return -1.0f;
    return (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) * inv;
}

const float kSphereSize[3] = {1.0f, 0.0f, 0.0f};

}  // namespace

TEST(Raycast, SphereOnAxis)
{
    rc::Scene scene;
    const int si = scene.addInstance(rc::GeomType::kSphere, kSphereSize, 0.0f);

    std::vector<rc::InstanceXform> xf = {xformAt(scene, si, 5.0f, 0.0f, 0.0f)};
    const std::vector<float> alt = {0.0f}, az = {0.0f};
    const auto range = cast(scene, xf, alt, az, scanParams(1, 4));

    EXPECT_NEAR(range[0], 4.0f, 1e-4f);  // m=0: +x → front of the sphere
    EXPECT_EQ(range[1], kInf);           // m=1: -y → miss
    EXPECT_EQ(range[2], kInf);           // m=2: -x → miss
    EXPECT_EQ(range[3], kInf);           // m=3: +y → miss
}

TEST(Raycast, BeamOriginParallaxMatchesXyzLut)
{
    // Reported range r satisfies the Ouster XYZ-LUT reconstruction
    // xyz = (r−n)·d̂ + n·[cosθ,sinθ,0]. For an on-axis target the hit point
    // is at x = 4 regardless of n, so r must equal 4 for any n.
    rc::Scene scene;
    const int si = scene.addInstance(rc::GeomType::kSphere, kSphereSize, 0.0f);
    std::vector<rc::InstanceXform> xf = {xformAt(scene, si, 5.0f, 0.0f, 0.0f)};
    const std::vector<float> alt = {0.0f}, az = {0.0f};

    for (const float n : {0.0f, 0.015806f, 0.0277f}) {
        const auto range = cast(scene, xf, alt, az, scanParams(1, 4, n));
        EXPECT_NEAR(range[0], 4.0f, 1e-4f) << "beam_origin=" << n;
    }
}

TEST(Raycast, BoxCylinderPlane)
{
    rc::Scene scene;
    const float box_size[3] = {1.0f, 1.0f, 1.0f};
    const float cyl_size[3] = {0.5f, 1.0f, 0.0f};   // radius, half length
    const float plane_size[3] = {10.0f, 10.0f, 0.0f};
    const int bi = scene.addInstance(rc::GeomType::kBox, box_size, 0.0f);
    const int ci = scene.addInstance(rc::GeomType::kCylinder, cyl_size, 0.0f);
    const int pi = scene.addInstance(rc::GeomType::kPlane, plane_size, 0.0f);

    // Box face at y = -4 (hit via the m = W/4 → -y column), cylinder side
    // at x = 2.5, plane z = -2 (hit via a -45° beam).
    std::vector<rc::InstanceXform> xf = {
        xformAt(scene, bi, 0.0f, -5.0f, 0.0f),
        xformAt(scene, ci, 3.0f, 0.0f, 0.0f),
        xformAt(scene, pi, 0.0f, 0.0f, -2.0f)};

    const std::vector<float> alt = {0.0f, -45.0f};
    const std::vector<float> az = {0.0f, 0.0f};
    const auto range = cast(scene, xf, alt, az, scanParams(2, 4));

    EXPECT_NEAR(range[0], 2.5f, 1e-4f);              // beam 0, +x → cylinder
    EXPECT_NEAR(range[1], 4.0f, 1e-4f);              // beam 0, -y → box face
    EXPECT_NEAR(range[4], 2.0f * std::sqrt(2.0f), 1e-4f);  // beam 1, +x −45° → plane
}

TEST(Raycast, CylinderEndCap)
{
    rc::Scene scene;
    const float cyl_size[3] = {0.5f, 1.0f, 0.0f};
    const int ci = scene.addInstance(rc::GeomType::kCylinder, cyl_size, 0.0f);

    // Cylinder below the sensor; a straight-down beam hits the +z cap.
    std::vector<rc::InstanceXform> xf = {xformAt(scene, ci, 0.0f, 0.0f, -4.0f)};
    const std::vector<float> alt = {-90.0f}, az = {0.0f};
    const auto range = cast(scene, xf, alt, az, scanParams(1, 4));
    EXPECT_NEAR(range[0], 3.0f, 1e-4f);  // cap at z = -3
}

TEST(Raycast, FinitePlaneExtentsMiss)
{
    rc::Scene scene;
    const float plane_size[3] = {1.0f, 1.0f, 0.0f};  // small patch
    const int pi = scene.addInstance(rc::GeomType::kPlane, plane_size, 0.0f);

    // A -45° beam crosses z = -2 at x = 2, outside the ±1 patch.
    std::vector<rc::InstanceXform> xf = {xformAt(scene, pi, 0.0f, 0.0f, -2.0f)};
    const std::vector<float> alt = {-45.0f}, az = {0.0f};
    const auto range = cast(scene, xf, alt, az, scanParams(1, 4));
    EXPECT_EQ(range[0], kInf);
}

TEST(Raycast, AxisParallelGrazingRayIsNanFree)
{
    // Regression for the 0·inf = NaN slab-test hazard: a ray travelling
    // exactly parallel to a box face, with its origin exactly at the slab
    // boundary (z == +h). Must hit the face cleanly, not NaN-miss.
    rc::Scene scene;
    const float box_size[3] = {1.0f, 1.0f, 1.0f};
    const int bi = scene.addInstance(rc::GeomType::kBox, box_size, 0.0f);
    std::vector<rc::InstanceXform> xf = {xformAt(scene, bi, 5.0f, 0.0f, -1.0f)};

    // el = 0, az = 0 → d = (1, 0, 0) exactly; sensor z == box top (z = 0).
    const std::vector<float> alt = {0.0f}, az = {0.0f};
    const auto range = cast(scene, xf, alt, az, scanParams(1, 4));
    ASSERT_TRUE(std::isfinite(range[0]));
    EXPECT_NEAR(range[0], 4.0f, 1e-4f);  // front face at x = 4
}

TEST(Raycast, MeshBoxMatchesAnalyticBox)
{
    rc::Scene mesh_scene;
    std::vector<float> verts;
    std::vector<int> tris;
    makeBoxMesh(1.0f, 1.0f, 1.0f, verts, tris);
    const int root = mesh_scene.addMesh(verts, tris);
    ASSERT_GE(root, 0);
    const float msize[3] = {0.0f, 0.0f, 0.0f};
    const int mi = mesh_scene.addInstance(rc::GeomType::kMesh, msize, 0.0f, root);

    rc::Scene box_scene;
    const float box_size[3] = {1.0f, 1.0f, 1.0f};
    const int bi = box_scene.addInstance(rc::GeomType::kBox, box_size, 0.0f);

    constexpr int H = 8, W = 32;
    std::vector<float> alt(H), az(H, 1.3f);
    for (int i = 0; i < H; ++i) alt[i] = -20.0f + 5.0f * i;

    std::vector<rc::InstanceXform> mesh_xf = {
        xformAt(mesh_scene, mi, 6.0f, 0.5f, 0.0f)};
    std::vector<rc::InstanceXform> box_xf = {
        xformAt(box_scene, bi, 6.0f, 0.5f, 0.0f)};
    const auto r_mesh = cast(mesh_scene, mesh_xf, alt, az, scanParams(H, W));
    const auto r_box = cast(box_scene, box_xf, alt, az, scanParams(H, W));

    int hits = 0;
    for (size_t i = 0; i < r_mesh.size(); ++i) {
        if (std::isfinite(r_box[i])) {
            ++hits;
            EXPECT_NEAR(r_mesh[i], r_box[i], 1e-3f) << "ray " << i;
        } else {
            EXPECT_EQ(r_mesh[i], kInf) << "ray " << i;
        }
    }
    EXPECT_GT(hits, 0) << "test scene produced no hits — invalid setup";
}

TEST(Raycast, BvhMatchesBruteForce)
{
    // Random triangle soup; the BVH must report exactly the brute-force
    // nearest hit for every ray.
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> pos(-5.0f, 5.0f);
    std::uniform_real_distribution<float> jit(-0.8f, 0.8f);

    std::vector<float> verts;
    std::vector<int> tris;
    constexpr int kTris = 300;
    for (int i = 0; i < kTris; ++i) {
        const float cx = pos(rng), cy = pos(rng), cz = pos(rng);
        for (int k = 0; k < 3; ++k) {
            verts.insert(verts.end(),
                {cx + jit(rng), cy + jit(rng), cz + jit(rng)});
        }
        tris.insert(tris.end(), {3 * i, 3 * i + 1, 3 * i + 2});
    }

    rc::Scene scene;
    const int root = scene.addMesh(verts, tris);
    ASSERT_GE(root, 0);
    const float msize[3] = {0.0f, 0.0f, 0.0f};
    const int mi = scene.addInstance(rc::GeomType::kMesh, msize, 0.0f, root);

    // The instance sits 12 m up so rays start outside the near clip.
    std::vector<rc::InstanceXform> xf = {xformAt(scene, mi, 12.0f, 0.0f, 0.0f)};

    constexpr int H = 16, W = 64;
    std::vector<float> alt(H), az(H, 0.0f);
    for (int i = 0; i < H; ++i) alt[i] = -30.0f + 4.0f * i;
    const auto sp = scanParams(H, W);
    const auto range = cast(scene, xf, alt, az, sp);

    int hits = 0;
    for (int idx = 0; idx < H * W; ++idx) {
        const int beam = idx / W, m = idx % W;
        const float az_r = (-m * 360.0f / W) * 3.14159265f / 180.0f;
        const float el_r = alt[beam] * 3.14159265f / 180.0f;
        const float d[3] = {std::cos(el_r) * std::cos(az_r),
                            std::cos(el_r) * std::sin(az_r),
                            std::sin(el_r)};
        const float o[3] = {-12.0f, 0.0f, 0.0f};  // instance-local origin

        float best = sp.max_range;
        bool hit = false;
        for (int tri = 0; tri < kTris; ++tri) {
            const float t = refTriangle(
                o, d,
                &verts[static_cast<size_t>(3 * tris[3 * tri])],
                &verts[static_cast<size_t>(3 * tris[3 * tri + 1])],
                &verts[static_cast<size_t>(3 * tris[3 * tri + 2])]);
            if (t > sp.near_clip && t < best) {
                best = t;
                hit = true;
            }
        }
        if (hit) {
            ++hits;
            ASSERT_TRUE(std::isfinite(range[idx])) << "ray " << idx;
            EXPECT_NEAR(range[idx], best, 1e-3f) << "ray " << idx;
        } else {
            EXPECT_EQ(range[idx], kInf) << "ray " << idx;
        }
    }
    EXPECT_GT(hits, 30) << "soup produced too few hits — invalid setup";
}

TEST(Raycast, RetroOfNearestHit)
{
    rc::Scene scene;
    const float ssize[3] = {0.5f, 0.0f, 0.0f};
    const int near_i = scene.addInstance(rc::GeomType::kSphere, ssize, 1.5f);
    const int far_i = scene.addInstance(rc::GeomType::kSphere, ssize, 0.3f);
    std::vector<rc::InstanceXform> xf = {
        xformAt(scene, near_i, 3.0f, 0.0f, 0.0f),
        xformAt(scene, far_i, 6.0f, 0.0f, 0.0f)};

    const std::vector<float> alt = {0.0f}, az = {0.0f};
    std::vector<float> retro;
    const auto range = cast(scene, xf, alt, az, scanParams(1, 4), &retro);

    EXPECT_NEAR(range[0], 2.5f, 1e-4f);
    // Nearest sphere's laser_retro; the beam hits the sphere dead-centre, so
    // the incidence cosine is 1 and the apparent reflectance is unattenuated.
    EXPECT_NEAR(retro[0], 1.5f, 1e-6f);
}

TEST(Raycast, RetroAttenuatedByIncidenceCosine)
{
    // Apparent reflectance = laser_retro × cos(incidence) — the extended-
    // Lambertian lidar equation (see rcCosIncidence).
    //
    // Sphere of radius 0.5 offset 0.3 m sideways from the beam axis: the
    // impact parameter b gives sin(α) = b/r at the first intersection, so
    // cos(α) = sqrt(1 − (0.3/0.5)²) = 0.8, and the hit lands at
    // x = 3 − sqrt(r² − b²) = 2.6 m.
    rc::Scene scene;
    const float ssize[3] = {0.5f, 0.0f, 0.0f};
    const int si = scene.addInstance(rc::GeomType::kSphere, ssize, 1.0f);
    std::vector<rc::InstanceXform> xf = {xformAt(scene, si, 3.0f, 0.3f, 0.0f)};

    const std::vector<float> alt = {0.0f}, az = {0.0f};
    std::vector<float> retro;
    const auto range = cast(scene, xf, alt, az, scanParams(1, 4), &retro);

    EXPECT_NEAR(range[0], 2.6f, 1e-4f);
    EXPECT_NEAR(retro[0], 0.8f, 1e-4f);
}

TEST(Raycast, SpecularLobePeaksAtNormalIncidence)
{
    // Monostatic specular return ks·cos(2α)⁸ (rcApparentReflectance):
    // a glossy surface (kd = 0, ks = 1) returns full strength when viewed
    // dead-on and effectively nothing at cos(α) = 0.8 (cos 2α = 0.28,
    // 0.28⁸ ≈ 4e−5) — the missing-points signature of glossy/black paint.
    rc::Scene scene;
    const float ssize[3] = {0.5f, 0.0f, 0.0f};
    const int si = scene.addInstance(rc::GeomType::kSphere, ssize,
                                     /*retro=*/0.0f, -1, /*spec=*/1.0f);

    // Dead-centre hit: cos α = 1.
    std::vector<rc::InstanceXform> xf = {xformAt(scene, si, 3.0f, 0.0f, 0.0f)};
    const std::vector<float> alt = {0.0f}, az = {0.0f};
    std::vector<float> retro;
    cast(scene, xf, alt, az, scanParams(1, 4), &retro);
    EXPECT_NEAR(retro[0], 1.0f, 1e-4f);

    // Oblique hit (impact parameter 0.3, cos α = 0.8): lobe ≈ 0.
    xf = {xformAt(scene, si, 3.0f, 0.3f, 0.0f)};
    cast(scene, xf, alt, az, scanParams(1, 4), &retro);
    EXPECT_LT(retro[0], 1e-3f);
}

TEST(Raycast, GlassTransmissionReportsStrongestReturn)
{
    // Velas et al. (arXiv:1909.12483 §III) single-return behaviour: a
    // transparent pane returns (1−τ)·ρ_surface; the object behind returns
    // τ²·ρ_object (double pass); the stronger received power ρ/R² wins.
    //
    // Pane (ks 0.5, τ 0.9) face-on at 2 m: ρ_surf = 0.5·0.1 = 0.05,
    // power 0.05/4 = 0.0125. Wall (kd 1.0) behind at 4.5 m:
    // ρ = 0.81, power 0.81/20.25 = 0.04 → the wall wins.
    rc::Scene scene;
    // Plane local +z normal rotated to face the sensor (−x).
    const float r_lw[9] = {0, 0, -1, 0, 1, 0, 1, 0, 0};
    const float psize[3] = {1.0f, 1.0f, 0.0f};
    const int pane = scene.addInstance(rc::GeomType::kPlane, psize,
                                       /*retro=*/0.0f, -1, /*spec=*/0.5f,
                                       /*transmit=*/0.9f);
    const float bsize[3] = {0.5f, 0.5f, 0.5f};
    const int wall = scene.addInstance(rc::GeomType::kBox, bsize, 1.0f);

    std::vector<rc::InstanceXform> xf = {
        xformAt(scene, pane, 2.0f, 0.0f, 0.0f, r_lw),
        xformAt(scene, wall, 5.0f, 0.0f, 0.0f)};
    const std::vector<float> alt = {0.0f}, az = {0.0f};
    std::vector<float> retro;
    auto range = cast(scene, xf, alt, az, scanParams(1, 4), &retro);
    EXPECT_NEAR(range[0], 4.5f, 1e-3f);
    EXPECT_NEAR(retro[0], 0.81f, 1e-3f);

    // Mostly opaque pane (τ 0.2): surface return 0.5·0.8 = 0.4 at 2 m
    // (power 0.1) beats the weakened wall (1·0.04/20.25 ≈ 0.002).
    rc::Scene scene2;
    const int pane2 = scene2.addInstance(rc::GeomType::kPlane, psize,
                                         0.0f, -1, 0.5f, 0.2f);
    const int wall2 = scene2.addInstance(rc::GeomType::kBox, bsize, 1.0f);
    std::vector<rc::InstanceXform> xf2 = {
        xformAt(scene2, pane2, 2.0f, 0.0f, 0.0f, r_lw),
        xformAt(scene2, wall2, 5.0f, 0.0f, 0.0f)};
    range = cast(scene2, xf2, alt, az, scanParams(1, 4), &retro);
    EXPECT_NEAR(range[0], 2.0f, 1e-3f);
    EXPECT_NEAR(retro[0], 0.4f, 1e-3f);
}

TEST(Raycast, MirrorGhostReportedBehindMirror)
{
    // Velas et al. §III: a mirror bounces the beam onto a side object; the
    // return retraces the path and the sensor reports a ghost point along
    // the ORIGINAL beam at the total path length.
    //
    // Mirror plane at (2,0,0), normal (−1,1,0)/√2 (45° about z): the +x
    // beam reflects to +y and hits a box face at (2,2.5,0), t2 = 2.5.
    // Direct surface return: kd = 0 and cos(2·45°) = 0 kills the lobe → 0.
    // Ghost: ρ = (ks·(1−τ))²·ρ_box = 1²·1 = 1 at range 2 + 2.5 = 4.5.
    rc::Scene scene;
    const float r_lw[9] = {0.70710678f, 0.0f, -0.70710678f,
                           0.70710678f, 0.0f,  0.70710678f,
                           0.0f,       -1.0f,  0.0f};
    const float psize[3] = {1.0f, 1.0f, 0.0f};
    const int mirror = scene.addInstance(rc::GeomType::kPlane, psize,
                                         /*retro=*/0.0f, -1, /*spec=*/1.0f);
    const float bsize[3] = {0.5f, 0.5f, 0.5f};
    const int box = scene.addInstance(rc::GeomType::kBox, bsize, 1.0f);

    std::vector<rc::InstanceXform> xf = {
        xformAt(scene, mirror, 2.0f, 0.0f, 0.0f, r_lw),
        xformAt(scene, box, 2.0f, 3.0f, 0.0f)};
    const std::vector<float> alt = {0.0f}, az = {0.0f};
    std::vector<float> retro;
    const auto range = cast(scene, xf, alt, az, scanParams(1, 4), &retro);

    EXPECT_NEAR(range[0], 4.5f, 5e-3f);
    EXPECT_NEAR(retro[0], 1.0f, 1e-3f);
}

TEST(Raycast, PlaneRetroFollowsGrazingAngle)
{
    // Ground plane 1 m below the sensor, beam at −30° elevation:
    // cos(incidence to the +z normal) = |sin(30°)| = 0.5, range = 1/sin30 = 2.
    rc::Scene scene;
    const float psize[3] = {0.0f, 0.0f, 0.0f};  // infinite plane
    const int pi = scene.addInstance(rc::GeomType::kPlane, psize, 1.0f);
    std::vector<rc::InstanceXform> xf = {xformAt(scene, pi, 0.0f, 0.0f, -1.0f)};

    const std::vector<float> alt = {-30.0f}, az = {0.0f};
    std::vector<float> retro;
    const auto range = cast(scene, xf, alt, az, scanParams(1, 4), &retro);

    EXPECT_NEAR(range[0], 2.0f, 1e-3f);
    EXPECT_NEAR(retro[0], 0.5f, 1e-4f);
}

TEST(Raycast, NearClipSeesThroughCloseHit)
{
    rc::Scene scene;
    const float close_size[3] = {0.05f, 0.0f, 0.0f};
    const float far_size[3] = {1.0f, 0.0f, 0.0f};
    const int ci = scene.addInstance(rc::GeomType::kSphere, close_size, 0.0f);
    const int fi = scene.addInstance(rc::GeomType::kSphere, far_size, 0.0f);

    // Close sphere inside the 0.3 m near clip (the sensor housing case);
    // the beam must report the far target instead.
    std::vector<rc::InstanceXform> xf = {
        xformAt(scene, ci, 0.2f, 0.0f, 0.0f),
        xformAt(scene, fi, 10.0f, 0.0f, 0.0f)};
    const std::vector<float> alt = {0.0f}, az = {0.0f};
    const auto range = cast(scene, xf, alt, az, scanParams(1, 4));
    EXPECT_NEAR(range[0], 9.0f, 1e-4f);
}

TEST(Raycast, SunDrivenNearIrFactor)
{
    // NEAR_IR ambient model: nir = albedo·(ambient + diffuse·max(0, n̂·(−ŝ)))
    // with ŝ the sun's propagation direction. Ground plane 1 m below the
    // sensor (normal +z), beam at −30° elevation, albedo 1.
    rc::Scene scene;
    const float psize[3] = {0.0f, 0.0f, 0.0f};
    const int pi = scene.addInstance(rc::GeomType::kPlane, psize, 1.0f);
    std::vector<rc::InstanceXform> xf = {xformAt(scene, pi, 0.0f, 0.0f, -1.0f)};

    const std::vector<float> alt = {-30.0f}, az = {0.0f};
    std::vector<float> range(4), nir(4);

    // Sun straight down (ŝ = −z): Lambert term 1 → nir = 0.3 + 0.7 = 1.
    auto sp = scanParams(1, 4);
    sp.sun_dir[0] = 0.0f; sp.sun_dir[1] = 0.0f; sp.sun_dir[2] = -1.0f;
    sp.sun_diffuse = 0.7f;
    sp.sun_ambient = 0.3f;
    rc::castScan(scene.view(), xf.data(), alt.data(), az.data(),
                 kIdentityR, kZeroT, sp, range.data(), nullptr,
                 nullptr, nullptr, nir.data());
    EXPECT_NEAR(nir[0], 1.0f, 1e-4f);

    // Sun on the horizon (ŝ = +x): Lambert term 0 → ambient only, 0.3.
    sp.sun_dir[0] = 1.0f; sp.sun_dir[1] = 0.0f; sp.sun_dir[2] = 0.0f;
    rc::castScan(scene.view(), xf.data(), alt.data(), az.data(),
                 kIdentityR, kZeroT, sp, range.data(), nullptr,
                 nullptr, nullptr, nir.data());
    EXPECT_NEAR(nir[0], 0.3f, 1e-4f);

    // Default ScanParams (no sun configured): ambient-only, nir = albedo.
    rc::castScan(scene.view(), xf.data(), alt.data(), az.data(),
                 kIdentityR, kZeroT, scanParams(1, 4), range.data(), nullptr,
                 nullptr, nullptr, nir.data());
    EXPECT_NEAR(nir[0], 1.0f, 1e-4f);
}

TEST(Raycast, MotionDistortionPerColumnPoses)
{
    // Rolling-shutter motion distortion: each column casts from its own
    // sensor pose. Sensor translating +x inside a radius-50 shell centred
    // at the origin; column m at x_m = 0.5·m. Analytic ranges
    // (|p + t·d| = 50, columns at az_enc = −90°·m):
    //   m=0  d=(+1,0,0), x=0   → 50
    //   m=1  d=(0,−1,0), x=0.5 → sqrt(2500 − 0.25)  ≈ 49.9975
    //   m=2  d=(−1,0,0), x=1.0 → 51
    //   m=3  d=(0,+1,0), x=1.5 → sqrt(2500 − 2.25)  ≈ 49.97749
    constexpr int W = 4;
    rc::Scene scene;
    const float shell_size[3] = {50.0f, 0.0f, 0.0f};
    const int si = scene.addInstance(rc::GeomType::kSphere, shell_size, 0.0f);
    std::vector<rc::InstanceXform> xf = {xformAt(scene, si, 0.0f, 0.0f, 0.0f)};

    std::vector<float> col_r(9 * W), col_t(3 * W, 0.0f);
    for (int m = 0; m < W; ++m) {
        std::memcpy(&col_r[static_cast<size_t>(9 * m)], kIdentityR,
                    9 * sizeof(float));
        col_t[static_cast<size_t>(3 * m)] = 0.5f * static_cast<float>(m);
    }

    const std::vector<float> alt = {0.0f}, az = {0.0f};
    std::vector<float> range(W);
    rc::castScan(scene.view(), xf.data(), alt.data(), az.data(),
                 kIdentityR, kZeroT, scanParams(1, W), range.data(), nullptr,
                 col_r.data(), col_t.data());

    EXPECT_NEAR(range[0], 50.0f, 1e-3f);
    EXPECT_NEAR(range[1], 49.9975f, 1e-3f);
    EXPECT_NEAR(range[2], 51.0f, 1e-3f);
    EXPECT_NEAR(range[3], 49.97749f, 1e-3f);

    // Static path (null tables) is unchanged: every column reads 50.
    rc::castScan(scene.view(), xf.data(), alt.data(), az.data(),
                 kIdentityR, kZeroT, scanParams(1, W), range.data(), nullptr);
    for (int m = 0; m < W; ++m) {
        EXPECT_NEAR(range[m], 50.0f, 1e-4f) << "column " << m;
    }
}

TEST(Raycast, UniformShellExactRange)
{
    // A large sphere centred on the sensor: with zero beam-origin offset
    // every beam reads the radius exactly — no interpolation error exists
    // anywhere in this pipeline.
    constexpr int H = 16, W = 64;
    constexpr float kR = 50.0f;

    rc::Scene scene;
    const float shell_size[3] = {kR, 0.0f, 0.0f};
    const int si = scene.addInstance(rc::GeomType::kSphere, shell_size, 0.0f);
    std::vector<rc::InstanceXform> xf = {xformAt(scene, si, 0.0f, 0.0f, 0.0f)};

    std::vector<float> alt(H), az(H);
    for (int i = 0; i < H; ++i) {
        alt[i] = -40.0f + 5.0f * i;
        az[i] = (i % 4 - 1.5f) * 2.1f;
    }
    const auto range = cast(scene, xf, alt, az, scanParams(H, W));
    for (int i = 0; i < H * W; ++i) {
        EXPECT_NEAR(range[i], kR, 5e-3f) << "ray " << i;
    }
}

TEST(Raycast, CastScanThroughRayProcessor)
{
    // The backend entry: dispatch → castScan must agree with the CPU
    // reference (on the CPU backend they are the same shared math; on a
    // GPU build this same test exercises the device kernel).
    ::setenv("GZ_OUSTER_BACKEND", "cpu", 1);

    constexpr int H = 8, W = 32;
    rc::Scene scene;
    const float shell_size[3] = {20.0f, 0.0f, 0.0f};
    const int si = scene.addInstance(rc::GeomType::kSphere, shell_size, 0.7f);
    std::vector<rc::InstanceXform> xf = {xformAt(scene, si, 0.0f, 0.0f, 0.0f)};

    std::vector<float> alt(H), az(H, 0.5f);
    for (int i = 0; i < H; ++i) alt[i] = -20.0f + 5.0f * i;
    const auto sp = scanParams(H, W);

    const int n = H * W;
    std::vector<float> range(n), retro(n);
    const float sr[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const float st[3] = {0, 0, 0};

    RayProcessor proc;
    ASSERT_TRUE(proc.usesCpuFallback());
    proc.castScan(scene.view(), 1u, xf.data(), alt.data(), az.data(),
                  sr, st, sp, range.data(), retro.data());

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(range[i], 20.0f, 5e-3f) << "ray " << i;
        EXPECT_NEAR(retro[i], 0.7f, 1e-6f) << "ray " << i;
    }
}

TEST(Raycast, ProcessDepthThroughRayProcessor)
{
    // The noise/channel stage consumes the raycast output directly:
    // exact ranges in, range_mm + retro-driven reflectivity out.
    constexpr int H = 2, W = 8;
    const int n = H * W;
    std::vector<float> depth(n, 7.5f);
    std::vector<float> retro(n, 0.8f);
    depth[3] = kInf;  // one miss

    RayProcessParams pp{};
    pp.H = H;
    pp.W = W;
    pp.base_signal = 800.0f;
    pp.base_reflectivity = 50.0f;
    pp.max_range = 120.0f;  // all noise terms 0 → deterministic

    std::vector<uint32_t> range(n);
    std::vector<uint16_t> signal(n);
    std::vector<uint8_t> refl(n);
    std::vector<uint16_t> nearir(n);

    RayProcessor proc;
    proc.processDepth(depth.data(), retro.data(),
                      range.data(), signal.data(), refl.data(),
                      nearir.data(), pp);

    for (int i = 0; i < n; ++i) {
        if (i == 3) {
            EXPECT_EQ(range[i], 0u);
        } else {
            EXPECT_EQ(range[i], 7500u);
            // retro 0.8 → Lambertian band: 0.8 × 100 = 80.
            EXPECT_EQ(static_cast<int>(refl[i]), 80);
            EXPECT_GT(signal[i], 0u);
        }
    }
}

}  // namespace gz_gpu_ouster_lidar
