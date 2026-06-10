// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "gz_gpu_ouster_lidar/ray_processor.hpp"
#include "panel_layout.hpp"
#include "panel_test_utils.hpp"
#include "ray_processor_math.hpp"

namespace gz_gpu_ouster_lidar {

using testutil::makeAllInfRaw;
using testutil::makeBeams;
using testutil::makeRig;
using testutil::makeUniformRangeRaw;
using testutil::noNoise;

// ── Layout construction ──────────────────────────────────────────────────────

TEST(PanelLayout, CylindricalRigForOs1Geometry)
{
    // OS1-like: ±22.5° beams.
    const auto layout = makeRig(-22.5f, 22.5f, 64, 512);
    ASSERT_EQ(layout.n_panels, 4);
    EXPECT_FALSE(layout.hemispherical);
    EXPECT_EQ(layout.rp.n_panels, 4);
    EXPECT_GT(layout.rp.raw_n, 0);

    // Packed offsets are contiguous and sum to raw_n.
    int expected_offset = 0;
    for (int i = 0; i < layout.n_panels; ++i) {
        const auto & p = layout.rp.panels[i];
        EXPECT_EQ(p.offset, expected_offset) << "panel " << i;
        expected_offset += p.width * p.height;
    }
    EXPECT_EQ(layout.rp.raw_n, expected_offset);
}

TEST(PanelLayout, HemisphericalRigForDomeGeometry)
{
    // OSDome-like: 0..90° beams → 8 side sectors + zenith cap.
    const auto layout = makeRig(0.0f, 90.0f, 128, 1024);
    ASSERT_EQ(layout.n_panels, 9);
    EXPECT_TRUE(layout.hemispherical);
}

TEST(PanelLayout, RejectsUnsupportedGeometry)
{
    // A full-sphere band has no rig.
    const auto layout = buildOusterPanelLayout(-90.0, 90.0, 64, 512, 2.0);
    EXPECT_EQ(layout.n_panels, 0);
}

TEST(PanelLayout, FullBeamCoverageCylindrical)
{
    // OS0-like: the widest cylindrical product (±45° + margin), with
    // non-zero per-beam azimuth offsets.
    constexpr int H = 32, W = 256;
    const auto layout = makeRig(-46.0f, 46.0f, H, W);
    ASSERT_EQ(layout.n_panels, 4);

    auto beam_alt = makeBeams(H, -45.0f, 45.0f);
    std::vector<float> beam_az(H);
    for (int i = 0; i < H; ++i) beam_az[i] = (i % 4 - 1.5f) * 2.1f;

    EXPECT_EQ(0, countUncoveredRays(layout.rp, beam_alt.data(), beam_az.data()));
}

TEST(PanelLayout, FullBeamCoverageHemispherical)
{
    constexpr int H = 64, W = 256;
    const auto layout = makeRig(-1.0f, 91.0f, H, W);
    ASSERT_GT(layout.n_panels, 0);

    auto beam_alt = makeBeams(H, 0.0f, 90.0f);
    std::vector<float> beam_az(H, 1.5f);

    EXPECT_EQ(0, countUncoveredRays(layout.rp, beam_alt.data(), beam_az.data()));
}

TEST(PanelLayout, ForwardDirectionHitsAPanelCentreColumn)
{
    const auto layout = makeRig(-22.5f, 22.5f, 64, 512);
    float dx, dy, dz, u, v, cosp;
    rpmath::beamDirection(0.0f, 0.0f, dx, dy, dz);
    const int pi = rpmath::panelForDirection(layout.rp, dx, dy, dz, u, v, cosp);
    ASSERT_GE(pi, 0);
    const auto & p = layout.rp.panels[pi];
    // Forward (+x) is the yaw-0 panel's optical axis: principal point, cos 1.
    EXPECT_NEAR(u, p.cx, 1e-3f);
    EXPECT_NEAR(v, p.cy, 1e-3f);
    EXPECT_NEAR(cosp, 1.0f, 1e-5f);
}

// ── Resampling through RayProcessor ──────────────────────────────────────────

TEST(Resample, UniformRangeProducesUniformRange)
{
    // Surfaces at a constant Euclidean range all around: every output pixel
    // must read that range regardless of which panel and image position the
    // beam lands on. This exercises the planar-depth → range cosine
    // correction across panel boundaries.
    constexpr int H = 8, W = 64;
    constexpr float kRange = 20.0f;

    const auto layout = makeRig(-10.0f, 10.0f, H, W);
    ASSERT_EQ(layout.n_panels, 4);

    const auto raw = makeUniformRangeRaw(layout.rp, kRange);
    auto beam_alt = makeBeams(H, -10.0f, 10.0f);
    std::vector<float> beam_az(H, 0.0f);
    const auto pp = noNoise(H, W);

    const int n = H * W;
    std::vector<uint32_t> range(n);
    std::vector<uint16_t> signal(n);
    std::vector<uint8_t>  refl(n);
    std::vector<uint16_t> nearir(n);

    RayProcessor proc;
    proc.processRaw(raw.data(), beam_alt.data(), beam_az.data(), layout.rp,
                    range.data(), signal.data(), refl.data(), nearir.data(), pp);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(static_cast<double>(range[i]), 20000.0, 30.0)
            << "pixel " << i << " range mismatch";
    }
}

TEST(Resample, UniformRangeHemisphericalRig)
{
    // Same invariant through the dome rig: pitched side panels + zenith cap.
    constexpr int H = 16, W = 64;
    constexpr float kRange = 12.0f;

    const auto layout = makeRig(0.0f, 90.0f, H, W);
    ASSERT_GT(layout.n_panels, 4);

    const auto raw = makeUniformRangeRaw(layout.rp, kRange);
    auto beam_alt = makeBeams(H, 0.5f, 89.5f);
    std::vector<float> beam_az(H, 0.0f);
    const auto pp = noNoise(H, W);

    const int n = H * W;
    std::vector<uint32_t> range(n);
    std::vector<uint16_t> signal(n);
    std::vector<uint8_t>  refl(n);
    std::vector<uint16_t> nearir(n);

    RayProcessor proc;
    proc.processRaw(raw.data(), beam_alt.data(), beam_az.data(), layout.rp,
                    range.data(), signal.data(), refl.data(), nearir.data(), pp);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(static_cast<double>(range[i]), 12000.0, 30.0)
            << "pixel " << i << " range mismatch";
    }
}

TEST(Resample, AllInfProducesZeroRange)
{
    constexpr int H = 4, W = 32;

    const auto layout = makeRig(-5.0f, 5.0f, H, W);
    const auto raw = makeAllInfRaw(layout.rp);
    auto beam_alt = makeBeams(H, -5.0f, 5.0f);
    std::vector<float> beam_az(H, 0.0f);
    const auto pp = noNoise(H, W);

    const int n = H * W;
    std::vector<uint32_t> range(n);
    std::vector<uint16_t> signal(n);
    std::vector<uint8_t>  refl(n);
    std::vector<uint16_t> nearir(n);

    RayProcessor proc;
    proc.processRaw(raw.data(), beam_alt.data(), beam_az.data(), layout.rp,
                    range.data(), signal.data(), refl.data(), nearir.data(), pp);

    for (int i = 0; i < n; ++i) {
        EXPECT_EQ(range[i], 0u) << "inf depth should produce range=0 at pixel " << i;
    }
}

TEST(Resample, FarClipReadsAsMiss)
{
    // Planar depth at/beyond far_clip is the renderer's "no return" value;
    // it must not leak through as a wall of returns at max range.
    constexpr int H = 2, W = 16;
    constexpr float kFar = 50.0f;

    auto layout = makeRig(-5.0f, 5.0f, H, W, kFar);
    std::vector<float> raw(static_cast<size_t>(layout.rp.raw_n), kFar);

    auto beam_alt = makeBeams(H, -5.0f, 5.0f);
    std::vector<float> beam_az(H, 0.0f);
    const auto pp = noNoise(H, W, kFar);

    const int n = H * W;
    std::vector<uint32_t> range(n);
    std::vector<uint16_t> signal(n);
    std::vector<uint8_t>  refl(n);
    std::vector<uint16_t> nearir(n);

    RayProcessor proc;
    proc.processRaw(raw.data(), beam_alt.data(), beam_az.data(), layout.rp,
                    range.data(), signal.data(), refl.data(), nearir.data(), pp);

    for (int i = 0; i < n; ++i) {
        EXPECT_EQ(range[i], 0u) << "far-clipped depth should be a miss at pixel " << i;
    }
}

TEST(Resample, BeamOriginSubtraction)
{
    constexpr int H = 1, W = 16;
    constexpr float kRange = 10.0f;
    constexpr float kBeamOriginM = 0.05f;  // 50 mm

    auto layout = makeRig(-5.0f, 5.0f, H, W);
    layout.rp.beam_origin_m = kBeamOriginM;

    const auto raw = makeUniformRangeRaw(layout.rp, kRange);
    std::vector<float> beam_alt = {0.0f};  // 0 elevation → cos(0) = 1
    std::vector<float> beam_az = {0.0f};
    const auto pp = noNoise(H, W);

    const int n = H * W;
    std::vector<uint32_t> range(n);
    std::vector<uint16_t> signal(n);
    std::vector<uint8_t>  refl(n);
    std::vector<uint16_t> nearir(n);

    RayProcessor proc;
    proc.processRaw(raw.data(), beam_alt.data(), beam_az.data(), layout.rp,
                    range.data(), signal.data(), refl.data(), nearir.data(), pp);

    // depth - beam_origin * cos(0) = 10 - 0.05 = 9.95 m = 9950 mm
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(static_cast<double>(range[i]), 9950.0, 30.0)
            << "beam origin subtraction at pixel " << i;
    }
}

TEST(Resample, AzimuthOffsetShiftsColumns)
{
    // An azimuth-dependent scene: range varies smoothly with azimuth, so a
    // per-beam azimuth offset must change the sampled values.
    constexpr int H = 1, W = 64;

    const auto layout = makeRig(-5.0f, 5.0f, H, W);

    // Fill each panel pixel by reconstructing its sensor-frame azimuth and
    // setting range = 5 + cos(az) (smooth, 360°-periodic).
    std::vector<float> raw(static_cast<size_t>(layout.rp.raw_n));
    for (int i = 0; i < layout.n_panels; ++i) {
        const auto & p = layout.rp.panels[i];
        for (int v = 0; v < p.height; ++v) {
            for (int u = 0; u < p.width; ++u) {
                const float a = (p.cx - static_cast<float>(u)) / p.fx;
                const float b = (p.cy - static_cast<float>(v)) / p.fy;
                // Panel-frame direction (1, a, b) → sensor frame via R^T.
                const float * r = p.r;
                const float sx = r[0] * 1.0f + r[3] * a + r[6] * b;
                const float sy = r[1] * 1.0f + r[4] * a + r[7] * b;
                const float az = std::atan2(sy, sx);
                const float rng = 5.0f + std::cos(az);
                raw[p.offset + v * p.width + u] =
                    rng / std::sqrt(1.0f + a * a + b * b);
            }
        }
    }

    std::vector<float> beam_alt = {0.0f};
    std::vector<float> beam_az_zero = {0.0f};
    std::vector<float> beam_az_offset = {17.0f};

    const auto pp = noNoise(H, W);
    const int n = H * W;
    std::vector<uint32_t> range_zero(n), range_offset(n);
    std::vector<uint16_t> sig(n);
    std::vector<uint8_t>  refl(n);
    std::vector<uint16_t> nearir(n);

    RayProcessor proc;
    proc.processRaw(raw.data(), beam_alt.data(), beam_az_zero.data(), layout.rp,
                    range_zero.data(), sig.data(), refl.data(), nearir.data(), pp);
    proc.processRaw(raw.data(), beam_alt.data(), beam_az_offset.data(), layout.rp,
                    range_offset.data(), sig.data(), refl.data(), nearir.data(), pp);

    // The offset rotates the sampled azimuth, so columns must differ — and
    // by the expected shift: column m samples az = az_off − m·(360/W), so a
    // +17° offset reproduces column m's zero-offset value at m + 17/(360/W).
    bool any_differ = false;
    for (int i = 0; i < n; ++i) {
        if (range_zero[i] != range_offset[i]) {
            any_differ = true;
            break;
        }
    }
    EXPECT_TRUE(any_differ) << "Azimuth offset should shift output columns";

    // Quantitative check at a few columns (17° ≈ 3.02 columns at W=64;
    // compare against the analytic scene instead of integer-shifted output).
    const float deg_per_col = 360.0f / W;
    for (int m = 0; m < n; m += 7) {
        const float az = (17.0f - m * deg_per_col) * 3.14159265f / 180.0f;
        const float expect_mm = (5.0f + std::cos(az)) * 1000.0f;
        EXPECT_NEAR(static_cast<float>(range_offset[m]), expect_mm, 40.0f)
            << "column " << m;
    }
}

}  // namespace gz_gpu_ouster_lidar
