// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#include "gz_gpu_ouster_lidar/ray_processor.hpp"
#include "conformance_v1.hpp"
#include <gtest/gtest.h>
#include <cstdlib>
#include <limits>
#include <string>

namespace gz_gpu_ouster_lidar {
namespace {
class OpticalConformance : public testing::TestWithParam<const char *> {};

TEST_P(OpticalConformance, ProductionBackendMatchesSharedNoiselessContract)
{
    const char * old = std::getenv("GZ_OUSTER_BACKEND");
    const std::string previous = old ? old : "";
    setenv("GZ_OUSTER_BACKEND", GetParam(), 1);
    RayProcessor processor(42);
    if (old) setenv("GZ_OUSTER_BACKEND", previous.c_str(), 1);
    else unsetenv("GZ_OUSTER_BACKEND");
    if (std::string(processor.backendName()) != GetParam()) {
        GTEST_SKIP() << GetParam() << " is unavailable on this build/device";
    }
    RayProcessParams params{};
    params.H = params.W = 1;
    params.base_signal = 800;
    params.base_reflectivity = 50;
    params.min_range = .1f;
    params.max_range = 120;
    params.range_resolution = .001f;
    const ouster_sim_core::OpticalChannelModel reference(
        ouster_sim_core::OpticalChannelModelConfig{});
    for (const auto & fixture : ouster_sim_core::conformance_v1::opticalCases) {
        SCOPED_TRACE(fixture.name);
        const float depth = fixture.depth > 0 ? static_cast<float>(fixture.depth) :
            std::numeric_limits<float>::infinity();
        const float retro = static_cast<float>(fixture.reflectance.value_or(0));
        const float ambient = static_cast<float>(fixture.ambient.value_or(0));
        uint32_t range;
        uint16_t signal, nearir;
        uint8_t reflectivity;
        processor.processDepth(&depth, fixture.reflectance ? &retro : nullptr,
            &range, &signal, &reflectivity, &nearir, params,
            fixture.ambient ? &ambient : nullptr);
        const auto expected = reference.process(fixture.input(), {});
        EXPECT_EQ(range, expected.range_mm);
        EXPECT_EQ(signal, expected.signal);
        EXPECT_EQ(reflectivity, expected.reflectivity);
        EXPECT_EQ(nearir, expected.near_ir);
    }
    // With product detection enabled, a present black surface is dropped;
    // missing material retains its fallback detection response.
    params.detection_range_10 = 45;
    params.detection_range_80 = 90;
    params.detection_rolloff = .15f;
    const float depth = 10, black = 0;
    uint32_t range;
    uint16_t signal, nearir;
    uint8_t reflectivity;
    processor.processDepth(&depth, &black, &range, &signal, &reflectivity, &nearir, params);
    EXPECT_EQ(range, 0u);
    EXPECT_EQ(signal, 0u);
    EXPECT_EQ(reflectivity, 0u);
    EXPECT_EQ(nearir, 0u);
}

INSTANTIATE_TEST_SUITE_P(Backends, OpticalConformance,
                        testing::Values("cpu", "cuda", "hip", "sycl"));
}  // namespace
}  // namespace gz_gpu_ouster_lidar
