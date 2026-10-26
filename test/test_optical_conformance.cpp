// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#include "gz_gpu_ouster_lidar/ray_processor.hpp"
#include "conformance_v1.hpp"
#include "ouster_lidar_profile.hpp"
#include <gtest/gtest.h>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>

namespace gz_gpu_ouster_lidar {
namespace {
class OpticalConformance : public testing::TestWithParam<const char *> {
protected:
    // The production processor on the requested backend, or nullptr when
    // this build/device lacks it.
    static std::unique_ptr<RayProcessor> makeProcessor(const char * backend)
    {
        const char * old = std::getenv("GZ_OUSTER_BACKEND");
        const std::string previous = old ? old : "";
        setenv("GZ_OUSTER_BACKEND", backend, 1);
        auto processor = std::make_unique<RayProcessor>(42);
        if (old) {
            setenv("GZ_OUSTER_BACKEND", previous.c_str(), 1);
        } else {
            unsetenv("GZ_OUSTER_BACKEND");
        }
        if (std::string(processor->backendName()) != backend) return nullptr;
        return processor;
    }
};

TEST_P(OpticalConformance, ProductionBackendMatchesSharedNoiselessContract)
{
    const auto processor = makeProcessor(GetParam());
    if (!processor) {
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
        processor->processDepth(&depth, fixture.reflectance ? &retro : nullptr,
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
    processor->processDepth(&depth, &black, &range, &signal, &reflectivity, &nearir, params);
    EXPECT_EQ(range, 0u);
    EXPECT_EQ(signal, 0u);
    EXPECT_EQ(reflectivity, 0u);
    EXPECT_EQ(nearir, 0u);
}

TEST_P(OpticalConformance, ProductionBackendMatchesMeasuredSignalResponse)
{
    const auto processor = makeProcessor(GetParam());
    if (!processor) {
        GTEST_SKIP() << GetParam() << " is unavailable on this build/device";
    }
    OusterProfileRequest request;
    request.prod_line = "OS1-64";
    request.hardware_revision = "rev07";
    request.firmware_major = 3;
    request.firmware_minor = 2;
    request.beam_count = 64;
    const auto profile = resolveOusterLidarProfile(request);
    ASSERT_TRUE(profile.signal_response.measured());

    // 2048x10 gathers half the recordings' shots per measurement.
    RayProcessParams params{};
    params.H = params.W = 1;
    params.base_signal = 800;
    params.base_reflectivity = 50;
    params.min_range = .1f;
    params.max_range = 120;
    params.range_resolution = .001f;
    params.signal_response =
        signalResponseParams(profile.signal_response, 2048, 10.0);
    ouster_sim_core::OpticalChannelModelConfig config;
    config.signal_response = profile.signal_response;
    config.signal_mode_gain =
        ouster_sim_core::signalModeGain(profile.signal_response, 2048, 10.0);
    const ouster_sim_core::OpticalChannelModel reference(config);

    // Across the held near field, the knots, the far field and saturation.
    // Reflectances are exact in float, so both sides see the same value.
    for (const float depth : {0.4f, 1.5f, 2.3f, 3.2f, 4.5f, 8.7f, 15.0f, 40.0f, 110.0f}) {
        for (const float retro : {0.03125f, 0.25f, 0.875f, 40.0f}) {
            SCOPED_TRACE(testing::Message() << depth << " m, reflectance " << retro);
            uint32_t range;
            uint16_t signal, nearir;
            uint8_t reflectivity;
            processor->processDepth(&depth, &retro, &range, &signal,
                                    &reflectivity, &nearir, params);
            ouster_sim_core::NormalizedOpticalReturn input;
            input.return_kind = ouster_sim_core::OpticalReturnKind::kSurface;
            input.geometric_hit = true;
            input.path_length_m = input.reported_range_m = depth;
            input.apparent_reflectance = retro;
            const auto expected = reference.process(input, {});
            EXPECT_EQ(range, expected.range_mm);
            // Float kernels against the double reference: one count.
            EXPECT_NEAR(signal, expected.signal, 1.0);
            EXPECT_EQ(reflectivity, expected.reflectivity);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Backends, OpticalConformance,
                        testing::Values("cpu", "cuda", "hip", "sycl"));
}  // namespace
}  // namespace gz_gpu_ouster_lidar
