// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

#include "gz_gpu_ouster_lidar/cuda_ray_processor.hpp"
#include "ray_processor_cpu_impl.hpp"

namespace gz_gpu_ouster_lidar {

// Helper: create default RayProcessParams with noise disabled
static RayProcessParams noNoiseParams(int H, int W)
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

// ---------------------------------------------------------------------------
// Deterministic (no-noise) tests
// ---------------------------------------------------------------------------

TEST(NoiseModel, ValidDepthProducesNonZeroRange)
{
    constexpr int H = 4, W = 8;
    const int n = H * W;
    std::vector<float> depth(n, 10.0f);
    std::vector<float> retro(n, 0.5f);
    std::vector<uint32_t> range(n);
    std::vector<uint16_t> signal(n);
    std::vector<uint8_t>  refl(n);
    std::vector<uint16_t> nearir(n);

    auto p = noNoiseParams(H, W);
    processCpu(depth.data(), retro.data(),
                 range.data(), signal.data(), refl.data(), nearir.data(), p);

    for (int i = 0; i < n; ++i) {
        EXPECT_EQ(range[i], 10000u) << "index " << i;  // 10 m * 1000 mm/m
        EXPECT_GT(signal[i], 0u) << "index " << i;
    }
}

TEST(NoiseModel, InvalidDepthProducesZeroRange)
{
    constexpr int H = 2, W = 4;
    const int n = H * W;
    std::vector<float> depth(n);
    std::vector<float> retro(n, 0.5f);
    // Mix of invalid values: inf, -inf, NaN, near-zero
    depth[0] = std::numeric_limits<float>::infinity();
    depth[1] = -std::numeric_limits<float>::infinity();
    depth[2] = std::numeric_limits<float>::quiet_NaN();
    depth[3] = 0.0f;   // below 0.001 threshold
    depth[4] = 0.0005f; // below threshold
    depth[5] = -1.0f;   // negative
    depth[6] = std::numeric_limits<float>::infinity();
    depth[7] = std::numeric_limits<float>::quiet_NaN();

    std::vector<uint32_t> range(n);
    std::vector<uint16_t> signal(n);
    std::vector<uint8_t>  refl(n);
    std::vector<uint16_t> nearir(n);

    auto p = noNoiseParams(H, W);
    processCpu(depth.data(), retro.data(),
                 range.data(), signal.data(), refl.data(), nearir.data(), p);

    for (int i = 0; i < n; ++i) {
        EXPECT_EQ(range[i], 0u) << "invalid depth at index " << i << " should produce range=0";
        EXPECT_EQ(signal[i], 0u) << "invalid depth at index " << i << " should produce signal=0";
    }
}

TEST(NoiseModel, RangeIsDepthTimesThousand)
{
    constexpr int H = 1, W = 4;
    const int n = H * W;
    std::vector<float> depth = {1.0f, 5.5f, 0.123f, 99.999f};
    std::vector<float> retro(n, 0.5f);
    std::vector<uint32_t> range(n);
    std::vector<uint16_t> signal(n);
    std::vector<uint8_t>  refl(n);
    std::vector<uint16_t> nearir(n);

    auto p = noNoiseParams(H, W);
    processCpu(depth.data(), retro.data(),
                 range.data(), signal.data(), refl.data(), nearir.data(), p);

    EXPECT_EQ(range[0], 1000u);
    EXPECT_EQ(range[1], 5500u);
    EXPECT_EQ(range[2], 123u);
    EXPECT_EQ(range[3], 99999u);
}

TEST(NoiseModel, SignalFollowsInverseSquareLaw)
{
    constexpr int H = 1, W = 3;
    const int n = H * W;
    std::vector<float> depth = {1.0f, 2.0f, 4.0f};
    std::vector<float> retro(n, 1.0f);
    std::vector<uint32_t> range(n);
    std::vector<uint16_t> signal(n);
    std::vector<uint8_t>  refl(n);
    std::vector<uint16_t> nearir(n);

    auto p = noNoiseParams(H, W);
    p.base_signal = 1600.0f;
    processCpu(depth.data(), retro.data(),
                 range.data(), signal.data(), refl.data(), nearir.data(), p);

    // signal = base_signal * intensity / r^2
    // At 1m: 1600/1 = 1600, at 2m: 1600/4 = 400, at 4m: 1600/16 = 100
    EXPECT_EQ(signal[0], 1600u);
    EXPECT_EQ(signal[1], 400u);
    EXPECT_EQ(signal[2], 100u);
}

TEST(NoiseModel, ReflectivityLambertianScale)
{
    // retro <= 1.0 maps to 0-100 linearly
    constexpr int H = 1, W = 3;
    const int n = H * W;
    std::vector<float> depth(n, 10.0f);
    std::vector<float> retro = {0.0f, 0.5f, 1.0f};
    std::vector<uint32_t> range(n);
    std::vector<uint16_t> signal(n);
    std::vector<uint8_t>  refl(n);
    std::vector<uint16_t> nearir(n);

    auto p = noNoiseParams(H, W);
    processCpu(depth.data(), retro.data(),
                 range.data(), signal.data(), refl.data(), nearir.data(), p);

    // retro=0 is invalid (<=0 check), falls through to base_reflectivity
    EXPECT_EQ(refl[0], static_cast<uint8_t>(p.base_reflectivity));
    EXPECT_EQ(refl[1], 50u);   // 0.5 * 100 = 50
    EXPECT_EQ(refl[2], 100u);  // 1.0 * 100 = 100
}

TEST(NoiseModel, ReflectivityRetroScale)
{
    // retro > 1.0 maps via log2 into 101-255
    constexpr int H = 1, W = 2;
    const int n = H * W;
    std::vector<float> depth(n, 10.0f);
    std::vector<float> retro = {2.0f, 4.0f};
    std::vector<uint32_t> range(n);
    std::vector<uint16_t> signal(n);
    std::vector<uint8_t>  refl(n);
    std::vector<uint16_t> nearir(n);

    auto p = noNoiseParams(H, W);
    processCpu(depth.data(), retro.data(),
                 range.data(), signal.data(), refl.data(), nearir.data(), p);

    // 100 + log2(2) * 22 = 100 + 22 = 122
    EXPECT_EQ(refl[0], 122u);
    // 100 + log2(4) * 22 = 100 + 44 = 144
    EXPECT_EQ(refl[1], 144u);
}

TEST(NoiseModel, NullRetroUsesDefaults)
{
    constexpr int H = 1, W = 2;
    const int n = H * W;
    std::vector<float> depth(n, 10.0f);
    std::vector<uint32_t> range(n);
    std::vector<uint16_t> signal(n);
    std::vector<uint8_t>  refl(n);
    std::vector<uint16_t> nearir(n);

    auto p = noNoiseParams(H, W);
    p.base_reflectivity = 42.0f;
    processCpu(depth.data(), nullptr,
                 range.data(), signal.data(), refl.data(), nearir.data(), p);

    for (int i = 0; i < n; ++i) {
        EXPECT_EQ(refl[i], 42u);
        EXPECT_EQ(nearir[i], 0u);  // nir=0 when retro is null
    }
}

// ---------------------------------------------------------------------------
// Statistical noise tests
// ---------------------------------------------------------------------------

TEST(NoiseModel, RangeNoiseAddsVariance)
{
    constexpr int H = 1, W = 10000;
    const int n = H * W;
    const float depth_val = 50.0f;
    std::vector<float> depth(n, depth_val);
    std::vector<float> retro(n, 0.5f);
    std::vector<uint32_t> range(n);
    std::vector<uint16_t> signal(n);
    std::vector<uint8_t>  refl(n);
    std::vector<uint16_t> nearir(n);

    auto p = noNoiseParams(H, W);
    p.range_noise_min_std = 0.01f;
    p.range_noise_max_std = 0.03f;
    processCpu(depth.data(), retro.data(),
                 range.data(), signal.data(), refl.data(), nearir.data(), p);

    // Compute variance of range values (in mm)
    double sum = 0, sum2 = 0;
    int valid = 0;
    for (int i = 0; i < n; ++i) {
        if (range[i] > 0) {
            double v = static_cast<double>(range[i]);
            sum += v;
            sum2 += v * v;
            ++valid;
        }
    }
    ASSERT_GT(valid, n / 2);  // most points should survive
    double mean = sum / valid;
    double var = sum2 / valid - mean * mean;

    // Mean should be near 50000 mm (50 m)
    EXPECT_NEAR(mean, 50000.0, 500.0);  // within 0.5 m
    // Variance should be non-zero (noise was applied)
    EXPECT_GT(var, 0.0);
}

TEST(NoiseModel, DropoutsReduceValidCount)
{
    constexpr int H = 1, W = 50000;
    const int n = H * W;
    std::vector<float> depth(n, 100.0f);  // far range
    std::vector<float> retro(n, 0.5f);
    std::vector<uint32_t> range(n);
    std::vector<uint16_t> signal(n);
    std::vector<uint8_t>  refl(n);
    std::vector<uint16_t> nearir(n);

    auto p = noNoiseParams(H, W);
    p.dropout_rate_close = 0.0f;
    p.dropout_rate_far = 0.10f;  // 10% at max range
    p.max_range = 120.0f;
    processCpu(depth.data(), retro.data(),
                 range.data(), signal.data(), refl.data(), nearir.data(), p);

    int valid = 0;
    for (int i = 0; i < n; ++i) {
        if (range[i] > 0) ++valid;
    }
    int dropped = n - valid;

    // Should have some dropouts (at 100/120 range, expect ~8% * refl_factor)
    EXPECT_GT(dropped, 0);
    // But not all dropped
    EXPECT_LT(dropped, n);
    // Rough bounds: dropout rate at 100/120 * max_range ~ 8.3%, with refl_factor ~2
    // so effective rate ~16.7%.  Allow wide margin for randomness.
    double drop_rate = static_cast<double>(dropped) / n;
    EXPECT_GT(drop_rate, 0.02);
    EXPECT_LT(drop_rate, 0.50);
}

TEST(NoiseModel, ZeroNoiseProducesDeterministicOutput)
{
    constexpr int H = 2, W = 4;
    const int n = H * W;
    std::vector<float> depth(n, 25.0f);
    std::vector<float> retro(n, 0.8f);
    std::vector<uint32_t> range1(n), range2(n);
    std::vector<uint16_t> signal1(n), signal2(n);
    std::vector<uint8_t>  refl1(n), refl2(n);
    std::vector<uint16_t> nearir1(n), nearir2(n);

    auto p = noNoiseParams(H, W);
    processCpu(depth.data(), retro.data(),
                 range1.data(), signal1.data(), refl1.data(), nearir1.data(), p);
    processCpu(depth.data(), retro.data(),
                 range2.data(), signal2.data(), refl2.data(), nearir2.data(), p);

    EXPECT_EQ(range1, range2);
    EXPECT_EQ(signal1, signal2);
    EXPECT_EQ(refl1, refl2);
    EXPECT_EQ(nearir1, nearir2);
}

TEST(NoiseModel, EdgeDiscontinuityCausesDropouts)
{
    constexpr int H = 3, W = 3;
    const int n = H * W;
    // Center pixel at 10m, neighbors at 20m → large depth jump
    std::vector<float> depth = {
        20.0f, 20.0f, 20.0f,
        20.0f, 10.0f, 20.0f,
        20.0f, 20.0f, 20.0f
    };
    std::vector<float> retro(n, 0.5f);

    auto p = noNoiseParams(H, W);
    p.edge_discon_threshold = 5.0f;  // 5m threshold, 10m jump → suppress

    // Run many times to check that the center pixel drops out at ~50% rate
    int center_dropped = 0;
    const int trials = 1000;
    for (int t = 0; t < trials; ++t) {
        std::vector<uint32_t> range(n);
        std::vector<uint16_t> signal(n);
        std::vector<uint8_t>  refl(n);
        std::vector<uint16_t> nearir(n);
        processCpu(depth.data(), retro.data(),
                     range.data(), signal.data(), refl.data(), nearir.data(), p);
        if (range[4] == 0) ++center_dropped;  // center = index 4
    }

    // Center pixel should be suppressed ~50% of the time (uni(rng) < 0.5)
    double rate = static_cast<double>(center_dropped) / trials;
    EXPECT_GT(rate, 0.30);
    EXPECT_LT(rate, 0.70);
}

}  // namespace gz_gpu_ouster_lidar
