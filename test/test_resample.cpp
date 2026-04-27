// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "gz_gpu_ouster_lidar/ray_processor.hpp"

namespace gz_gpu_ouster_lidar {

// Helper: create a uniform GpuRays-like buffer (depth, retro, unused channels)
static std::vector<float> makeUniformRaw(int gpu_H, int gpu_W, int gpu_chan,
                                          float depth, float retro = 0.5f)
{
    std::vector<float> raw(gpu_H * gpu_W * gpu_chan, 0.0f);
    for (int r = 0; r < gpu_H; ++r) {
        for (int c = 0; c < gpu_W; ++c) {
            int base = (r * gpu_W + c) * gpu_chan;
            raw[base] = depth;
            if (gpu_chan >= 2) raw[base + 1] = retro;
        }
    }
    return raw;
}

// Helper: create ResampleParams
static ResampleParams makeResampleParams(int H, int W, int gpu_H, int gpu_W,
                                          int gpu_chan, float min_alt, float max_alt,
                                          float beam_origin_m = 0.0f)
{
    ResampleParams rp{};
    rp.H = H;
    rp.W = W;
    rp.gpu_H = gpu_H;
    rp.gpu_W = gpu_W;
    rp.gpu_chan = gpu_chan;
    rp.min_alt = min_alt;
    rp.v_range = max_alt - min_alt;
    rp.deg_per_col = 360.0f / W;
    rp.beam_origin_m = beam_origin_m;
    rp.half_W = W / 2;
    return rp;
}

// Helper: create no-noise RayProcessParams
static RayProcessParams noNoise(int H, int W)
{
    RayProcessParams p{};
    p.H = H;
    p.W = W;
    p.base_signal = 800.0f;
    p.base_reflectivity = 50.0f;
    p.range_noise_min_std = 0.0f;
    p.range_noise_max_std = 0.0f;
    p.max_range = 120.0f;
    p.signal_noise_scale = 0.0f;
    p.nearir_noise_scale = 0.0f;
    p.dropout_rate_close = 0.0f;
    p.dropout_rate_far = 0.0f;
    p.edge_discon_threshold = 0.0f;
    return p;
}

TEST(Resample, UniformDepthProducesUniformRange)
{
    constexpr int H = 8, W = 16;
    constexpr int gpu_H = 32, gpu_W = 64, gpu_chan = 3;
    constexpr float depth = 20.0f;

    auto raw = makeUniformRaw(gpu_H, gpu_W, gpu_chan, depth);

    // Beam angles evenly spaced within the GpuRays VFOV
    std::vector<float> beam_alt(H);
    std::vector<float> beam_az(H, 0.0f);
    float min_alt = -10.0f, max_alt = 10.0f;
    for (int i = 0; i < H; ++i) {
        beam_alt[i] = min_alt + (max_alt - min_alt) * (i + 0.5f) / H;
    }

    auto rp = makeResampleParams(H, W, gpu_H, gpu_W, gpu_chan, min_alt, max_alt);
    auto pp = noNoise(H, W);

    const int n = H * W;
    std::vector<uint32_t> range(n);
    std::vector<uint16_t> signal(n);
    std::vector<uint8_t>  refl(n);
    std::vector<uint16_t> nearir(n);

    RayProcessor proc;
    proc.processRaw(raw.data(), beam_alt.data(), beam_az.data(), rp,
                    range.data(), signal.data(), refl.data(), nearir.data(), pp);

    // All output pixels should have range ~20000 mm
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(static_cast<double>(range[i]), 20000.0, 100.0)
            << "pixel " << i << " range mismatch";
    }
}

TEST(Resample, AllInfProducesZeroRange)
{
    constexpr int H = 4, W = 8;
    constexpr int gpu_H = 16, gpu_W = 32, gpu_chan = 3;

    auto raw = makeUniformRaw(gpu_H, gpu_W, gpu_chan,
                               std::numeric_limits<float>::infinity());

    std::vector<float> beam_alt(H);
    std::vector<float> beam_az(H, 0.0f);
    float min_alt = -5.0f, max_alt = 5.0f;
    for (int i = 0; i < H; ++i) {
        beam_alt[i] = min_alt + (max_alt - min_alt) * (i + 0.5f) / H;
    }

    auto rp = makeResampleParams(H, W, gpu_H, gpu_W, gpu_chan, min_alt, max_alt);
    auto pp = noNoise(H, W);

    const int n = H * W;
    std::vector<uint32_t> range(n);
    std::vector<uint16_t> signal(n);
    std::vector<uint8_t>  refl(n);
    std::vector<uint16_t> nearir(n);

    RayProcessor proc;
    proc.processRaw(raw.data(), beam_alt.data(), beam_az.data(), rp,
                    range.data(), signal.data(), refl.data(), nearir.data(), pp);

    for (int i = 0; i < n; ++i) {
        EXPECT_EQ(range[i], 0u) << "inf depth should produce range=0 at pixel " << i;
    }
}

TEST(Resample, BeamOriginSubtraction)
{
    constexpr int H = 1, W = 4;
    constexpr int gpu_H = 8, gpu_W = 16, gpu_chan = 3;
    constexpr float depth = 10.0f;
    constexpr float beam_origin_m = 0.05f;  // 50mm

    auto raw = makeUniformRaw(gpu_H, gpu_W, gpu_chan, depth);

    std::vector<float> beam_alt = {0.0f};  // 0 elevation → cos(0) = 1
    std::vector<float> beam_az = {0.0f};

    auto rp = makeResampleParams(H, W, gpu_H, gpu_W, gpu_chan, -5.0f, 5.0f, beam_origin_m);
    auto pp = noNoise(H, W);

    const int n = H * W;
    std::vector<uint32_t> range(n);
    std::vector<uint16_t> signal(n);
    std::vector<uint8_t>  refl(n);
    std::vector<uint16_t> nearir(n);

    RayProcessor proc;
    proc.processRaw(raw.data(), beam_alt.data(), beam_az.data(), rp,
                    range.data(), signal.data(), refl.data(), nearir.data(), pp);

    // At 0 elevation: depth - beam_origin * cos(0) = 10 - 0.05 = 9.95m = 9950mm
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(static_cast<double>(range[i]), 9950.0, 50.0)
            << "beam origin subtraction at pixel " << i;
    }
}

TEST(Resample, AzimuthOffsetShiftsColumns)
{
    constexpr int H = 1, W = 32;
    constexpr int gpu_H = 4, gpu_W = 32, gpu_chan = 3;

    // Create a raw buffer with a depth gradient across columns
    std::vector<float> raw(gpu_H * gpu_W * gpu_chan, 0.0f);
    for (int r = 0; r < gpu_H; ++r) {
        for (int c = 0; c < gpu_W; ++c) {
            int base = (r * gpu_W + c) * gpu_chan;
            raw[base] = 5.0f + static_cast<float>(c) * 0.1f;  // 5.0 to 8.1
            raw[base + 1] = 0.5f;
        }
    }

    std::vector<float> beam_alt = {0.0f};
    std::vector<float> beam_az_zero = {0.0f};
    std::vector<float> beam_az_offset = {11.25f};  // 360/32 = 11.25 → shift by 1 column

    auto rp = makeResampleParams(H, W, gpu_H, gpu_W, gpu_chan, -5.0f, 5.0f);
    auto pp = noNoise(H, W);

    const int n = H * W;
    std::vector<uint32_t> range_zero(n), range_offset(n);
    std::vector<uint16_t> sig(n);
    std::vector<uint8_t>  refl(n);
    std::vector<uint16_t> nearir(n);

    RayProcessor proc;
    proc.processRaw(raw.data(), beam_alt.data(), beam_az_zero.data(), rp,
                    range_zero.data(), sig.data(), refl.data(), nearir.data(), pp);
    proc.processRaw(raw.data(), beam_alt.data(), beam_az_offset.data(), rp,
                    range_offset.data(), sig.data(), refl.data(), nearir.data(), pp);

    // With azimuth offset, the column sampling should shift.
    // The exact mapping depends on the mirror-column remapping (half_W),
    // but the key property is that the output differs when az offset changes.
    bool any_differ = false;
    for (int i = 0; i < n; ++i) {
        if (range_zero[i] != range_offset[i]) {
            any_differ = true;
            break;
        }
    }
    EXPECT_TRUE(any_differ) << "Azimuth offset should shift output columns";
}

}  // namespace gz_gpu_ouster_lidar
