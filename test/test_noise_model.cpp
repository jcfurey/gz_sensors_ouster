// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

#include "gz_gpu_ouster_lidar/ray_processor.hpp"
#include "ray_processor_cpu_impl.hpp"
#include "ray_processor_math.hpp"

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
    p.range_noise_reference_range = 120.0f;
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

// A 100% target carries x = 1e4 at the 2 m knot (and, held, anywhere closer)
// and 2e4 at 10 m; the detector saturates at 5000.
static SignalResponseParams handCheckedResponse()
{
    SignalResponseParams r{};
    r.knots = 2;
    r.range_m[0] = 2.0f;
    r.range_m[1] = 10.0f;
    r.signal_m2[0] = 4.0e4f;
    r.signal_m2[1] = 2.0e6f;
    r.saturation = 5000.0f;
    r.shot_noise = 1.25f;
    r.relative_noise = 0.01f;
    r.mode_gain = 1.0f;
    return r;
}

// The documented saturation c = x / (1 + (x / 5000)^(4/3))^(3/4), and its
// slope, evaluated independently in double with pow.
static double saturated(double x)
{
    return x / std::pow(1.0 + std::pow(x / 5000.0, 4.0 / 3.0), 0.75);
}

static double saturationSlope(double x)
{
    return std::pow(1.0 + std::pow(x / 5000.0, 4.0 / 3.0), -1.75);
}

TEST(NoiseModel, MeasuredResponseSaturatesInsteadOfFollowingInverseSquare)
{
    constexpr int H = 1, W = 6;
    const int n = H * W;
    std::vector<float> depth = {0.7f, 2.0f, 6.0f, 20.0f, 3.0f, 3.0f};
    std::vector<float> retro = {0.5f, 0.5f, 0.3f, 0.25f, 1.0e6f, 0.0f};
    std::vector<uint32_t> range(n);
    std::vector<uint16_t> signal(n);
    std::vector<uint8_t>  refl(n);
    std::vector<uint16_t> nearir(n);

    auto p = noNoiseParams(H, W);
    p.signal_response = handCheckedResponse();
    processCpu(depth.data(), retro.data(),
                 range.data(), signal.data(), refl.data(), nearir.data(), p);

    // x = .5 * 1e4 = 5000, the same closer than the first knot. Float
    // against double may land a count apart.
    EXPECT_NEAR(signal[1], saturated(5000.0), 1.0);
    EXPECT_EQ(signal[0], signal[1]);
    // Midway signal_m2 is linear in range: x = .3 * 1.02e6 / 36 = 8500.
    EXPECT_NEAR(signal[2], saturated(8500.0), 1.0);
    // Beyond the last knot, 1/r^2: x = .25 * 2e6 / 400 = 1250.
    EXPECT_NEAR(signal[3], saturated(1250.0), 1.0);
    // A retroreflector saturates; a present black surface returns nothing.
    EXPECT_GE(signal[4], 4999u);
    EXPECT_LE(signal[4], 5000u);
    EXPECT_EQ(signal[5], 0u);
    // Reflectivity stays the calibrated byte.
    EXPECT_EQ(refl[0], 50u);
}

TEST(NoiseModel, MeasuredResponseScalesWithBaseSignalAndShots)
{
    constexpr int H = 1, W = 1;
    std::vector<float> depth(1, 2.0f);
    std::vector<float> retro(1, 0.25f);
    std::vector<uint32_t> range(1);
    std::vector<uint16_t> signal(1);
    std::vector<uint8_t>  refl(1);
    std::vector<uint16_t> nearir(1);

    auto p = noNoiseParams(H, W);
    p.signal_response = handCheckedResponse();
    // Twice the reference sensitivity doubles x to 5000; half the shots per
    // measurement (2048x10 against a 1024x10 recording) halve the output.
    p.base_signal = 1600.0f;
    p.signal_response.mode_gain = 0.5f;
    processCpu(depth.data(), retro.data(),
                 range.data(), signal.data(), refl.data(), nearir.data(), p);
    EXPECT_NEAR(signal[0], 0.5 * saturated(5000.0), 1.0);
}

TEST(NoiseModel, MeasuredResponseNoiseHasTheModelledSpread)
{
    constexpr int H = 1, W = 40000;
    const int n = H * W;
    std::vector<float> depth(n, 2.0f);
    std::vector<float> retro(n, 0.5f);
    std::vector<uint32_t> range(n);
    std::vector<uint16_t> signal(n);
    std::vector<uint8_t>  refl(n);
    std::vector<uint16_t> nearir(n);

    auto p = noNoiseParams(H, W);
    p.signal_response = handCheckedResponse();
    p.signal_noise_scale = 1.0f;
    processCpu(depth.data(), retro.data(),
                 range.data(), signal.data(), refl.data(), nearir.data(), p, 7u);

    double sum = 0.0, sum_sq = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += signal[i];
        sum_sq += static_cast<double>(signal[i]) * signal[i];
    }
    const double mean = sum / n;
    const double stddev = std::sqrt(sum_sq / n - mean * mean);
    // At x = 5000: shot 1.25 * dc/dx * sqrt(5000) and floor .01 * c in
    // quadrature, 40 counts or 1.3% of the signal (the inverse-square model's
    // Poisson sigma would be sqrt(c) = 55).
    const double sigma = std::hypot(
        1.25 * saturationSlope(5000.0) * std::sqrt(5000.0),
        0.01 * saturated(5000.0));
    EXPECT_NEAR(mean, saturated(5000.0) - 0.5, 1.0);  // truncation to counts
    EXPECT_NEAR(stddev, sigma, 0.03 * sigma);
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

    // An explicitly black material is present and must remain black.
    EXPECT_EQ(refl[0], 0u);
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

    // 15.89 bytes per doubling, (255 - 100) / log2(864):
    // 100 + log2(2) * 15.89 = 115.9
    EXPECT_EQ(refl[0], 115u);
    // 100 + log2(4) * 15.89 = 131.8
    EXPECT_EQ(refl[1], 131u);
}

TEST(NoiseModel, ReflectivityByteInverseCoversBothBands)
{
    // Every byte re-encodes as itself, including those whose bin edge rounds
    // just below the bin in float (53 / 100 = 0.52999997; 59, 214, 240).
    for (int b = 0; b <= 255; ++b) {
        const float retro = rpmath::reflectivityByteToRetro(static_cast<float>(b));
        EXPECT_EQ(static_cast<int>(rpmath::reflectivityToByte(retro)), b)
            << "byte=" << b << " retro=" << retro;
    }
    // Edges that already round trip stay exact: the default 50 is 0.5.
    EXPECT_EQ(rpmath::reflectivityByteToRetro(0.0f), 0.0f);
    EXPECT_EQ(rpmath::reflectivityByteToRetro(50.0f), 0.5f);
    EXPECT_EQ(rpmath::reflectivityByteToRetro(100.0f), 1.0f);
    // A fractional byte keeps its whole part, as the miss path truncates it.
    EXPECT_EQ(rpmath::reflectivityToByte(rpmath::reflectivityByteToRetro(52.9999f)), 52u);
    EXPECT_EQ(rpmath::reflectivityToByte(rpmath::reflectivityByteToRetro(53.5f)), 53u);
    EXPECT_FLOAT_EQ(rpmath::reflectivityByteToRetro(-10.0f), 0.0f);
    EXPECT_EQ(rpmath::reflectivityToByte(
                  rpmath::reflectivityByteToRetro(300.0f)),
              255u);
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

    // Expected per-sample sigma:
    //   t = depth / max_range = 50 / 120 ≈ 0.417
    //   sigma = min_std + t*(max_std - min_std) = 10 + 0.417*20 = 18.3 mm
    //   refl_factor at retro=0.5 = min(1/sqrt(0.5), 2) ≈ 1.414  (σ ∝ 1/√ρ;
    //   see rangeNoiseSigma)
    //   effective sigma ≈ 25.9 mm; expected variance ≈ 671 mm²
    //   SE of mean over N=10000 ≈ 0.26 mm
    // Bound mean to ±5 mm (~19σ_mean) and variance to within 2× of expected.
    EXPECT_NEAR(mean, 50000.0, 5.0);
    EXPECT_GT(var, 350.0);
    EXPECT_LT(var, 1400.0);
}

TEST(NoiseModel, DropoutsReduceValidCount)
{
    constexpr int H = 1, W = 50000;
    const int n = H * W;
    // Detection calibration is disabled in this generic dropout-only test;
    // isolate the user-tunable random miss term at a far range.
    std::vector<float> depth(n, 90.0f);
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

    // Expected drop rate: t=90/120=0.75, p_drop = 0.75*0.10 = 0.075;
    // refl_factor at retro=0.5 = min(1/0.5, 3) = 2.0 → effective rate
    // ≈ 0.15. SE over N=50000 ≈ 0.0016.
    // Bound to ±0.03 (~19σ) so a regression that doubles or zeroes the
    // rate fails immediately.
    double drop_rate = static_cast<double>(dropped) / n;
    EXPECT_GT(drop_rate, 0.120);
    EXPECT_LT(drop_rate, 0.180);
}

TEST(NoiseModel, ProductDetectionCurvePreservesD90Anchors)
{
    // Rev7 OS1 is specified at >90% detection at 90 m on a 10% target and
    // 170 m on an 80% target. At the dark-target D90 anchor, approximately
    // 90% should survive; the bright target at the same distance is well
    // inside its envelope and should be nearly complete.
    constexpr int H = 1, W = 20000;
    const int n = H * W;
    std::vector<float> depth(n, 90.0f);
    std::vector<uint32_t> range(n);
    std::vector<uint16_t> signal(n);
    std::vector<uint8_t>  refl(n);
    std::vector<uint16_t> nearir(n);

    auto p = noNoiseParams(H, W);
    p.max_range = 233.0f;
    p.detection_range_10 = 90.0f;
    p.detection_range_80 = 170.0f;
    p.detection_rolloff = 0.15f;

    std::vector<float> dark(n, 0.1f);
    processCpu(depth.data(), dark.data(),
                 range.data(), signal.data(), refl.data(), nearir.data(), p);
    int dark_valid = 0;
    for (int i = 0; i < n; ++i) {
        if (range[i] > 0) ++dark_valid;
    }
    EXPECT_GT(static_cast<double>(dark_valid) / n, 0.87);
    EXPECT_LT(static_cast<double>(dark_valid) / n, 0.93);

    std::vector<float> bright(n, 0.8f);
    processCpu(depth.data(), bright.data(),
                 range.data(), signal.data(), refl.data(), nearir.data(), p);
    int valid = 0;
    for (int i = 0; i < n; ++i) {
        if (range[i] > 0) ++valid;
    }
    EXPECT_GT(static_cast<double>(valid) / n, 0.99);
}

TEST(NoiseModel, FalseAlarmsInventReturnsOnMisses)
{
    // Solar-background false alarms (Jin et al., IET RSN 14, 2020): a
    // no-return pixel becomes a spurious point with probability
    // false_alarm_rate, uniformly distributed over (0, max_range], at the
    // signal floor.
    constexpr int H = 1, W = 50000;
    const int n = H * W;
    std::vector<float> depth(n, std::numeric_limits<float>::infinity());
    std::vector<uint32_t> range(n);
    std::vector<uint16_t> signal(n);
    std::vector<uint8_t>  refl(n);
    std::vector<uint16_t> nearir(n);

    auto p = noNoiseParams(H, W);
    p.false_alarm_rate = 0.02f;
    processCpu(depth.data(), nullptr,
                 range.data(), signal.data(), refl.data(), nearir.data(), p);

    const uint32_t max_mm =
        static_cast<uint32_t>(p.max_range * 1000.0f) + 1u;
    int invented = 0;
    for (int i = 0; i < n; ++i) {
        if (range[i] > 0u) {
            ++invented;
            EXPECT_LE(range[i], max_mm) << "i=" << i;
            EXPECT_EQ(signal[i], 1u) << "i=" << i;  // noise-floor signal
        }
    }
    // Expected 2% of 50000 = 1000; SE ≈ 31. Bound to ±0.6% (~10σ).
    const double rate = static_cast<double>(invented) / n;
    EXPECT_GT(rate, 0.014);
    EXPECT_LT(rate, 0.026);
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

    // Center pixel should be suppressed ~50% of the time (uni(rng) < 0.5).
    // SE over 1000 trials ≈ 0.0158; bound to ±0.05 (~3σ).
    double rate = static_cast<double>(center_dropped) / trials;
    EXPECT_GT(rate, 0.45);
    EXPECT_LT(rate, 0.55);
}

// ---------------------------------------------------------------------------
// Boundary-condition tests for core math formulas
// ---------------------------------------------------------------------------

TEST(NoiseModel, ReflectivitySaturatesAt255)
{
    // rv=1024: 100 + log2(1024)*15.89 = 258.9 → clamped to 255 (864 and up).
    constexpr int H = 1, W = 1;
    std::vector<float> depth(1, 10.0f);
    std::vector<float> retro(1, 1024.0f);
    std::vector<uint32_t> range(1);
    std::vector<uint16_t> signal(1);
    std::vector<uint8_t>  refl(1);
    std::vector<uint16_t> nearir(1);

    auto p = noNoiseParams(H, W);
    processCpu(depth.data(), retro.data(),
                 range.data(), signal.data(), refl.data(), nearir.data(), p);

    EXPECT_EQ(refl[0], 255u);
}

TEST(NoiseModel, RangeNoiseSigmaCapDoublesVariance)
{
    // kRangeRetroFloor=0.25: weight = min(1/sqrt(0.25), 2.0) = 2.0 (cap).
    // kRangeRetroMax=2.0:    weight = min(1/sqrt(1.0),  2.0) = 1.0 (no cap).
    // Same constant sigma → effective sigma doubles → variance quadruples.
    constexpr int H = 1, W = 80000;
    const int n = H * W;
    std::vector<float> depth(n, 50.0f);
    std::vector<uint32_t> range(n);
    std::vector<uint16_t> signal(n);
    std::vector<uint8_t>  refl(n);
    std::vector<uint16_t> nearir(n);

    auto p = noNoiseParams(H, W);
    p.range_noise_min_std = 0.02f;
    p.range_noise_max_std = 0.02f;  // constant sigma, no range dependency

    auto variance = [&](float rv_val, uint64_t seed) {
        std::vector<float> retro_v(n, rv_val);
        processCpu(depth.data(), retro_v.data(),
                     range.data(), signal.data(), refl.data(), nearir.data(), p, seed);
        double sum = 0, sum2 = 0;
        int cnt = 0;
        for (int i = 0; i < n; ++i) {
            if (range[i] > 0) {
                const double v = static_cast<double>(range[i]);
                sum += v; sum2 += v * v; ++cnt;
            }
        }
        const double mean = sum / cnt;
        return sum2 / cnt - mean * mean;
    };

    const double var_dark   = variance(0.25f, 1u);  // retro at cap floor
    const double var_bright = variance(1.0f,  2u);  // retro with no cap

    // Expected ratio = (2σ)²/σ² = 4; allow 3-5 for statistical headroom.
    const double ratio = var_dark / var_bright;
    EXPECT_GT(ratio, 3.0) << "ratio=" << ratio;
    EXPECT_LT(ratio, 5.0) << "ratio=" << ratio;
}

TEST(NoiseModel, DropoutCapFloorEqualizesVeryDarkTargets)
{
    // kDropoutRetroFloor=0.33: retro=0.05 and retro=0.33 both floor to 0.33,
    // giving weight=min(1/0.33, 3.0)=3.0.  Effective dropout rates must be
    // within 1% of each other.
    constexpr int H = 1, W = 60000;
    const int n = H * W;
    std::vector<float> depth(n, 10.0f);
    std::vector<uint32_t> range(n);
    std::vector<uint16_t> signal(n);
    std::vector<uint8_t>  refl(n);
    std::vector<uint16_t> nearir(n);

    auto p = noNoiseParams(H, W);
    p.dropout_rate_close = 0.2f;
    p.dropout_rate_far   = 0.4f;

    auto dropout_rate = [&](float rv, uint64_t seed) {
        std::vector<float> retro(n, rv);
        processCpu(depth.data(), retro.data(),
                     range.data(), signal.data(), refl.data(), nearir.data(), p, seed);
        int dropped = 0;
        for (int i = 0; i < n; ++i) {
            if (range[i] == 0) ++dropped;
        }
        return static_cast<double>(dropped) / n;
    };

    const double rate_very_dark = dropout_rate(0.05f, 1u);
    const double rate_at_floor  = dropout_rate(0.33f, 2u);

    EXPECT_NEAR(rate_very_dark, rate_at_floor, 0.01)
        << "very_dark=" << rate_very_dark << " at_floor=" << rate_at_floor;
}

TEST(NoiseModel, CalibratedDetectionIsSmoothAndMonotonic)
{
    const float p80 = rpmath::detectionProbability(
        80.0f, 0.1f, 90.0f, 170.0f, 0.0f, 0.0f, 0.15f, 233.0f);
    const float p90 = rpmath::detectionProbability(
        90.0f, 0.1f, 90.0f, 170.0f, 0.0f, 0.0f, 0.15f, 233.0f);
    const float p100 = rpmath::detectionProbability(
        100.0f, 0.1f, 90.0f, 170.0f, 0.0f, 0.0f, 0.15f, 233.0f);
    EXPECT_GT(p80, p90);
    EXPECT_NEAR(p90, 0.9f, 1e-5f);
    EXPECT_GT(p90, p100);
    EXPECT_GT(p100, 0.0f);  // no hard cliff immediately beyond D90
}

TEST(NoiseModel, RetroreflectorsMaintainHighDetectionProbability)
{
    // OS1 profile: max_range 120, d90 anchors 45 m / 100 m
    const float p_retro1 = rpmath::detectionProbability(
        5.0f, 1.0f, 45.0f, 100.0f, 0.0f, 0.0f, 0.15f, 120.0f);
    const float p_retro13 = rpmath::detectionProbability(
        5.0f, 1.3f, 45.0f, 100.0f, 0.0f, 0.0f, 0.15f, 120.0f);
    const float p_retro8 = rpmath::detectionProbability(
        5.0f, 8.0f, 45.0f, 100.0f, 0.0f, 0.0f, 0.15f, 120.0f);

    EXPECT_GE(p_retro1, 0.999f);
    EXPECT_GE(p_retro13, 0.999f);
    EXPECT_GE(p_retro8, 0.999f);
    EXPECT_LE(p_retro1, p_retro13);
    EXPECT_LE(p_retro13, p_retro8);
}

TEST(NoiseModel, MinimumRangeAndResolutionAreProductSpecific)
{
    constexpr int H = 1, W = 3;
    std::vector<float> depth = {0.49f, 0.5014f, 0.5061f};
    std::vector<float> retro(W, 0.8f);
    std::vector<uint32_t> range(W);
    std::vector<uint16_t> signal(W);
    std::vector<uint8_t> refl(W);
    std::vector<uint16_t> nearir(W);

    auto p = noNoiseParams(H, W);
    p.min_range = 0.5f;
    p.range_resolution = 0.008f;  // Ouster low-data profile
    processCpu(depth.data(), retro.data(), range.data(), signal.data(),
               refl.data(), nearir.data(), p, 7u);

    EXPECT_EQ(range[0], 0u);
    EXPECT_EQ(range[1], 504u);
    EXPECT_EQ(range[2], 504u);
}

TEST(NoiseModel, SignalFloorDoesNotCrashAtVeryShortRange)
{
    // At d=0.01 m: r²=kMinDenom exactly, guarding the 1/r² denominator.
    // Signal = base_signal / kMinDenom → very large, clamped to uint16 max.
    // The important thing is no crash, no NaN/Inf, and range > 0.
    constexpr int H = 1, W = 1;
    std::vector<float> depth(1, 0.01f);  // > kValidDepthMin=0.001 → valid
    std::vector<float> retro(1, 1.0f);
    std::vector<uint32_t> range(1);
    std::vector<uint16_t> signal(1);
    std::vector<uint8_t>  refl(1);
    std::vector<uint16_t> nearir(1);

    auto p = noNoiseParams(H, W);
    processCpu(depth.data(), retro.data(),
                 range.data(), signal.data(), refl.data(), nearir.data(), p);

    EXPECT_GT(range[0],  0u);
    EXPECT_GT(signal[0], 0u);
}

}  // namespace gz_gpu_ouster_lidar
