// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>

// Test the parameter validation logic used in Configure().
// This doesn't instantiate the full plugin — it tests the clamping/validation
// rules in isolation.

namespace gz_gpu_ouster_lidar {

// Mirrors the validation logic from GzGpuOusterLidarSystem::Configure()
struct ValidatedParams {
    double range_noise_min_std = 0.003;
    double range_noise_max_std = 0.015;
    double signal_noise_scale  = 1.0;
    double nearir_noise_scale  = 1.0;
    double dropout_rate_close  = 0.0005;
    double dropout_rate_far    = 0.03;
    double edge_discon_threshold = 0.15;
    double base_signal         = 800.0;
    double base_reflectivity   = 50.0;
    double max_range           = 120.0;
    double lidar_hz            = 10.0;

    void validate()
    {
        range_noise_min_std   = std::max(0.0, range_noise_min_std);
        range_noise_max_std   = std::max(0.0, range_noise_max_std);
        signal_noise_scale    = std::max(0.0, signal_noise_scale);
        nearir_noise_scale    = std::max(0.0, nearir_noise_scale);
        dropout_rate_close    = std::clamp(dropout_rate_close, 0.0, 1.0);
        dropout_rate_far      = std::clamp(dropout_rate_far,   0.0, 1.0);
        edge_discon_threshold = std::max(0.0, edge_discon_threshold);
        base_signal           = std::max(0.0, base_signal);
        base_reflectivity     = std::clamp(base_reflectivity, 0.0, 255.0);
        max_range             = std::max(1.0, max_range);
        if (lidar_hz <= 0.0) lidar_hz = 10.0;
    }
};

TEST(ParameterValidation, DefaultsAreValid)
{
    ValidatedParams p;
    p.validate();
    EXPECT_DOUBLE_EQ(p.range_noise_min_std, 0.003);
    EXPECT_DOUBLE_EQ(p.range_noise_max_std, 0.015);
    EXPECT_DOUBLE_EQ(p.signal_noise_scale, 1.0);
    EXPECT_DOUBLE_EQ(p.nearir_noise_scale, 1.0);
    EXPECT_DOUBLE_EQ(p.dropout_rate_close, 0.0005);
    EXPECT_DOUBLE_EQ(p.dropout_rate_far, 0.03);
    EXPECT_DOUBLE_EQ(p.edge_discon_threshold, 0.15);
    EXPECT_DOUBLE_EQ(p.base_signal, 800.0);
    EXPECT_DOUBLE_EQ(p.base_reflectivity, 50.0);
    EXPECT_DOUBLE_EQ(p.max_range, 120.0);
    EXPECT_DOUBLE_EQ(p.lidar_hz, 10.0);
}

TEST(ParameterValidation, NegativeNoiseClampedToZero)
{
    ValidatedParams p;
    p.range_noise_min_std = -1.0;
    p.range_noise_max_std = -0.5;
    p.signal_noise_scale  = -100.0;
    p.nearir_noise_scale  = -0.001;
    p.edge_discon_threshold = -5.0;
    p.base_signal = -10.0;
    p.validate();

    EXPECT_DOUBLE_EQ(p.range_noise_min_std, 0.0);
    EXPECT_DOUBLE_EQ(p.range_noise_max_std, 0.0);
    EXPECT_DOUBLE_EQ(p.signal_noise_scale, 0.0);
    EXPECT_DOUBLE_EQ(p.nearir_noise_scale, 0.0);
    EXPECT_DOUBLE_EQ(p.edge_discon_threshold, 0.0);
    EXPECT_DOUBLE_EQ(p.base_signal, 0.0);
}

TEST(ParameterValidation, DropoutRatesClampedTo01)
{
    ValidatedParams p;
    p.dropout_rate_close = -0.5;
    p.dropout_rate_far = 1.5;
    p.validate();

    EXPECT_DOUBLE_EQ(p.dropout_rate_close, 0.0);
    EXPECT_DOUBLE_EQ(p.dropout_rate_far, 1.0);
}

TEST(ParameterValidation, ReflectivityClampedTo0_255)
{
    ValidatedParams p;

    p.base_reflectivity = -10.0;
    p.validate();
    EXPECT_DOUBLE_EQ(p.base_reflectivity, 0.0);

    p.base_reflectivity = 300.0;
    p.validate();
    EXPECT_DOUBLE_EQ(p.base_reflectivity, 255.0);

    p.base_reflectivity = 128.0;
    p.validate();
    EXPECT_DOUBLE_EQ(p.base_reflectivity, 128.0);
}

TEST(ParameterValidation, MaxRangeMinimumIsOne)
{
    ValidatedParams p;
    p.max_range = 0.0;
    p.validate();
    EXPECT_DOUBLE_EQ(p.max_range, 1.0);

    p.max_range = -50.0;
    p.validate();
    EXPECT_DOUBLE_EQ(p.max_range, 1.0);

    p.max_range = 0.5;
    p.validate();
    EXPECT_DOUBLE_EQ(p.max_range, 1.0);
}

TEST(ParameterValidation, ZeroLidarHzResetsToDefault)
{
    ValidatedParams p;
    p.lidar_hz = 0.0;
    p.validate();
    EXPECT_DOUBLE_EQ(p.lidar_hz, 10.0);

    p.lidar_hz = -5.0;
    p.validate();
    EXPECT_DOUBLE_EQ(p.lidar_hz, 10.0);
}

TEST(ParameterValidation, PositiveLidarHzUnchanged)
{
    ValidatedParams p;
    p.lidar_hz = 20.0;
    p.validate();
    EXPECT_DOUBLE_EQ(p.lidar_hz, 20.0);

    p.lidar_hz = 0.5;
    p.validate();
    EXPECT_DOUBLE_EQ(p.lidar_hz, 0.5);
}

TEST(ParameterValidation, LargeValuesPassThrough)
{
    ValidatedParams p;
    p.range_noise_min_std = 100.0;
    p.range_noise_max_std = 200.0;
    p.signal_noise_scale  = 50.0;
    p.base_signal = 1e6;
    p.max_range = 1000.0;
    p.validate();

    EXPECT_DOUBLE_EQ(p.range_noise_min_std, 100.0);
    EXPECT_DOUBLE_EQ(p.range_noise_max_std, 200.0);
    EXPECT_DOUBLE_EQ(p.signal_noise_scale, 50.0);
    EXPECT_DOUBLE_EQ(p.base_signal, 1e6);
    EXPECT_DOUBLE_EQ(p.max_range, 1000.0);
}

}  // namespace gz_gpu_ouster_lidar
