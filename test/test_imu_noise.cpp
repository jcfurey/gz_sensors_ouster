// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <vector>

#include "imu_noise.hpp"

namespace gz_gpu_ouster_lidar {

namespace {

// Zero nominal input; reuse across tests that want to look only at noise.
constexpr Vec3 kZeroVec{0.0, 0.0, 0.0};

// Helper: drive applyImuNoise N times and accumulate per-axis stats on
// (measured.av - nominal_av_static) and (measured.la - nominal_la_static).
struct AxisStats {
    double sum = 0.0;
    double sum2 = 0.0;
    int n = 0;
    void add(double x) { sum += x; sum2 += x * x; ++n; }
    double mean() const { return n > 0 ? sum / n : 0.0; }
    double var()  const {
        if (n < 2) return 0.0;
        const double m = mean();
        return sum2 / n - m * m;
    }
};

}  // namespace

// ── Determinism tests ──────────────────────────────────────────────────────

TEST(ImuNoise, AllZerosLeavesNominalUnchanged)
{
    Vec3 gyro_bias = kZeroVec;
    Vec3 accel_bias = kZeroVec;
    std::mt19937_64 rng{42};

    const Vec3 nom_av{0.1, -0.2, 0.05};
    const Vec3 nom_la{0.0, 0.0, 9.81};

    const auto out = applyImuNoise(
        nom_av, nom_la, gyro_bias, accel_bias,
        0.0, 0.0, 0.0, 0.0, 0.01, rng);

    // No noise → measurement equals nominal exactly.
    EXPECT_DOUBLE_EQ(out.av.x, nom_av.x);
    EXPECT_DOUBLE_EQ(out.av.y, nom_av.y);
    EXPECT_DOUBLE_EQ(out.av.z, nom_av.z);
    EXPECT_DOUBLE_EQ(out.la.x, nom_la.x);
    EXPECT_DOUBLE_EQ(out.la.y, nom_la.y);
    EXPECT_DOUBLE_EQ(out.la.z, nom_la.z);

    // Bias state untouched (no walk, no draws).
    EXPECT_DOUBLE_EQ(gyro_bias.x, 0.0);
    EXPECT_DOUBLE_EQ(gyro_bias.y, 0.0);
    EXPECT_DOUBLE_EQ(gyro_bias.z, 0.0);
    EXPECT_DOUBLE_EQ(accel_bias.x, 0.0);
    EXPECT_DOUBLE_EQ(accel_bias.y, 0.0);
    EXPECT_DOUBLE_EQ(accel_bias.z, 0.0);

    // Reported white-noise stds are also zero.
    EXPECT_DOUBLE_EQ(out.gyro_white_std, 0.0);
    EXPECT_DOUBLE_EQ(out.accel_white_std, 0.0);
}

TEST(ImuNoise, SameSeedReproduces)
{
    auto run = [] {
        Vec3 gb = kZeroVec, ab = kZeroVec;
        std::mt19937_64 rng{12345};
        std::vector<Vec3> samples;
        samples.reserve(100);
        for (int i = 0; i < 100; ++i) {
            const auto out = applyImuNoise(
                kZeroVec, kZeroVec, gb, ab,
                1e-3, 1e-2, 1e-5, 1e-4, 0.01, rng);
            samples.push_back(out.av);
        }
        return samples;
    };
    const auto a = run();
    const auto b = run();
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_DOUBLE_EQ(a[i].x, b[i].x);
        EXPECT_DOUBLE_EQ(a[i].y, b[i].y);
        EXPECT_DOUBLE_EQ(a[i].z, b[i].z);
    }
}

TEST(ImuNoise, DifferentSeedsDecorrelate)
{
    Vec3 gb1 = kZeroVec, ab1 = kZeroVec;
    Vec3 gb2 = kZeroVec, ab2 = kZeroVec;
    std::mt19937_64 rng1{1};
    std::mt19937_64 rng2{2};

    const auto out1 = applyImuNoise(
        kZeroVec, kZeroVec, gb1, ab1,
        1e-3, 1e-2, 0.0, 0.0, 0.01, rng1);
    const auto out2 = applyImuNoise(
        kZeroVec, kZeroVec, gb2, ab2,
        1e-3, 1e-2, 0.0, 0.0, 0.01, rng2);

    // Two seeds, two samples — should not match.
    EXPECT_NE(out1.av.x, out2.av.x);
    EXPECT_NE(out1.la.x, out2.la.x);
}

// ── Statistical tests ──────────────────────────────────────────────────────

TEST(ImuNoise, WhiteNoiseVarianceMatchesDensity)
{
    constexpr int N = 100000;
    constexpr double dt = 0.01;
    constexpr double gyro_density  = 1.75e-4;   // rad/s/√Hz
    constexpr double accel_density = 2.3e-3;    // m/s²/√Hz

    Vec3 gb = kZeroVec, ab = kZeroVec;
    std::mt19937_64 rng{777};
    AxisStats gx, gy, gz, ax, ay, az;

    for (int i = 0; i < N; ++i) {
        const auto out = applyImuNoise(
            kZeroVec, kZeroVec, gb, ab,
            gyro_density, accel_density,
            0.0, 0.0,  // no bias walk so the white-noise signal is clean
            dt, rng);
        gx.add(out.av.x);  gy.add(out.av.y);  gz.add(out.av.z);
        ax.add(out.la.x);  ay.add(out.la.y);  az.add(out.la.z);
    }

    // Per-sample variance should match (density / sqrt(dt))² = density² / dt.
    const double exp_gyro_var  = (gyro_density  * gyro_density)  / dt;
    const double exp_accel_var = (accel_density * accel_density) / dt;

    // Variance of sample variance over N i.i.d. normals ≈ 2σ⁴/N.
    // For N=100000 our 1σ on var/expected_var is ~0.5%; bound to ±10%
    // gives plenty of headroom while still catching order-of-magnitude
    // regressions.
    const double tol_g = exp_gyro_var  * 0.10;
    const double tol_a = exp_accel_var * 0.10;
    EXPECT_NEAR(gx.var(), exp_gyro_var,  tol_g);
    EXPECT_NEAR(gy.var(), exp_gyro_var,  tol_g);
    EXPECT_NEAR(gz.var(), exp_gyro_var,  tol_g);
    EXPECT_NEAR(ax.var(), exp_accel_var, tol_a);
    EXPECT_NEAR(ay.var(), exp_accel_var, tol_a);
    EXPECT_NEAR(az.var(), exp_accel_var, tol_a);

    // Means should be ~0 (SE = sigma/√N). Bound to 10σ_mean.
    const double se_g = std::sqrt(exp_gyro_var  / N) * 10.0;
    const double se_a = std::sqrt(exp_accel_var / N) * 10.0;
    EXPECT_NEAR(gx.mean(), 0.0, se_g);
    EXPECT_NEAR(ax.mean(), 0.0, se_a);
}

TEST(ImuNoise, BiasDriftGrowsLinearly)
{
    // With white noise off and only the bias walk active, the bias state
    // evolves as a random walk. After N steps with drift = walk * sqrt(dt),
    // var(bias) ≈ N * drift² = N * walk² * dt.
    constexpr int N = 50000;
    constexpr double dt = 0.01;
    constexpr double walk = 1e-3;  // larger than realistic so the test is sensitive

    Vec3 gb = kZeroVec, ab = kZeroVec;
    std::mt19937_64 rng{42};
    for (int i = 0; i < N; ++i) {
        applyImuNoise(kZeroVec, kZeroVec, gb, ab,
                      0.0, 0.0,           // no white noise
                      walk, walk, dt, rng);
    }

    // Final bias is one realisation of a random walk; we can't assert the
    // value, but |bias| should be within ~5σ of zero where σ = walk * √(N*dt).
    const double sigma = walk * std::sqrt(static_cast<double>(N) * dt);
    EXPECT_LT(std::abs(gb.x), 5.0 * sigma);
    EXPECT_LT(std::abs(gb.y), 5.0 * sigma);
    EXPECT_LT(std::abs(gb.z), 5.0 * sigma);
    EXPECT_LT(std::abs(ab.x), 5.0 * sigma);

    // And it should NOT be zero (with overwhelming probability).
    EXPECT_NE(gb.x, 0.0);
    EXPECT_NE(ab.z, 0.0);
}

TEST(ImuNoise, NoWhiteNoiseSkipsRngDraws)
{
    // If white-noise stds are 0, applyImuNoise should not consume RNG state
    // for the white-noise step. Verify by comparing two RNGs: one driven
    // through applyImuNoise (with bias walk only), one drawn from manually.
    Vec3 gb = kZeroVec, ab = kZeroVec;
    std::mt19937_64 rng_a{99};
    std::mt19937_64 rng_b{99};

    constexpr double walk = 1e-4;
    constexpr double dt = 0.01;

    applyImuNoise(kZeroVec, kZeroVec, gb, ab,
                  0.0, 0.0, walk, walk, dt, rng_a);

    // Manually draw 6 samples (3 gyro bias + 3 accel bias) from rng_b.
    std::normal_distribution<double> n{0.0, 1.0};
    for (int i = 0; i < 6; ++i) (void)n(rng_b);

    // Now both RNGs should be in identical state — next draw must match.
    EXPECT_DOUBLE_EQ(n(rng_a), n(rng_b));
}

TEST(ImuNoise, NoBiasWalkSkipsRngDraws)
{
    // Symmetric: with both bias_walks=0 and only white noise enabled,
    // applyImuNoise should consume exactly 6 RNG draws (3 gyro white + 3
    // accel white) — not 12.
    Vec3 gb = kZeroVec, ab = kZeroVec;
    std::mt19937_64 rng_a{31337};
    std::mt19937_64 rng_b{31337};

    applyImuNoise(kZeroVec, kZeroVec, gb, ab,
                  1e-3, 1e-3, 0.0, 0.0, 0.01, rng_a);

    std::normal_distribution<double> n{0.0, 1.0};
    for (int i = 0; i < 6; ++i) (void)n(rng_b);
    EXPECT_DOUBLE_EQ(n(rng_a), n(rng_b));
}

TEST(ImuNoise, NominalValuePreservedAsMean)
{
    // With non-zero nominal input and noise enabled, the long-run mean of
    // the measurement should converge to nominal + (drifted) bias, NOT to
    // some arbitrary offset. With bias walk = 0, mean → nominal.
    constexpr int N = 50000;
    constexpr double dt = 0.01;
    const Vec3 nom_la{0.0, 0.0, 9.81};

    Vec3 gb = kZeroVec, ab = kZeroVec;
    std::mt19937_64 rng{2024};
    AxisStats az;
    for (int i = 0; i < N; ++i) {
        const auto out = applyImuNoise(
            kZeroVec, nom_la, gb, ab,
            0.0, 2.3e-3,
            0.0, 0.0,
            dt, rng);
        az.add(out.la.z);
    }

    // mean ≈ 9.81 ± 5σ_mean
    const double white = 2.3e-3 / std::sqrt(dt);
    const double se = white / std::sqrt(static_cast<double>(N));
    EXPECT_NEAR(az.mean(), 9.81, 5.0 * se);
}

}  // namespace gz_gpu_ouster_lidar
