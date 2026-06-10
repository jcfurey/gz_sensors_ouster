// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0
//
// Backend dispatch / selection coverage. The other suites exercise the CPU
// math directly (processCpu/applyImuNoise) or construct a default RayProcessor
// without asserting which backend was chosen. This suite drives the
// dispatcher's selection logic explicitly: the GZ_OUSTER_BACKEND override, the
// auto fallback to CPU when no GPU backend is present, and the backendName()/
// usesCpuFallback() reporting. It also runs one processRaw() end-to-end through
// the RayProcessor wrapper so the dispatch → backend → output path is covered.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "gz_gpu_ouster_lidar/ray_processor.hpp"
#include "panel_test_utils.hpp"

namespace gz_gpu_ouster_lidar {

namespace {

// RAII guard: set GZ_OUSTER_BACKEND for the duration of a test and restore the
// previous value (or unset) on teardown, so tests don't leak env state.
class BackendEnvGuard {
public:
    explicit BackendEnvGuard(const char * value)
    {
        const char * prev = std::getenv(kVar);
        had_prev_ = prev != nullptr;
        if (had_prev_) prev_ = prev;
        if (value) {
            ::setenv(kVar, value, 1);
        } else {
            ::unsetenv(kVar);
        }
    }
    ~BackendEnvGuard()
    {
        if (had_prev_) {
            ::setenv(kVar, prev_.c_str(), 1);
        } else {
            ::unsetenv(kVar);
        }
    }
    BackendEnvGuard(const BackendEnvGuard &) = delete;
    BackendEnvGuard & operator=(const BackendEnvGuard &) = delete;

private:
    static constexpr const char * kVar = "GZ_OUSTER_BACKEND";
    bool had_prev_ = false;
    std::string prev_;
};

}  // namespace

// Forcing the CPU backend by env var must always succeed (CPU is always
// compiled in) and report itself as the CPU fallback.
TEST(Dispatch, EnvForcesCpuBackend)
{
    BackendEnvGuard guard("cpu");
    RayProcessor proc;
    EXPECT_STREQ(proc.backendName(), "cpu");
    EXPECT_TRUE(proc.usesCpuFallback());
}

// An unrecognised GZ_OUSTER_BACKEND value must not throw — the dispatcher logs
// a warning and falls back to auto selection (CPU in a CPU-only build).
TEST(Dispatch, UnknownEnvFallsBackToAuto)
{
    BackendEnvGuard guard("definitely-not-a-backend");
    RayProcessor proc;
    // CPU is the last resort and always available, so auto must resolve to a
    // non-empty, valid backend name.
    EXPECT_STRNE(proc.backendName(), "none");
}

// With no override, auto selection must yield a usable backend. In the CI
// CPU-only build that is "cpu"; on a GPU build it would be the GPU backend, so
// we only assert a valid (non-"none") selection here.
TEST(Dispatch, AutoSelectionAlwaysResolves)
{
    BackendEnvGuard guard(nullptr);  // ensure env is unset
    RayProcessor proc;
    EXPECT_STRNE(proc.backendName(), "none");
}

// The configured seed is reported back verbatim, and a zero seed (the
// non-deterministic default) is distinct from an explicit non-zero seed.
TEST(Dispatch, SeedIsReported)
{
    RayProcessor a;            // default seed 0
    RayProcessor b{12345u};
    EXPECT_EQ(a.seed(), 0u);
    EXPECT_EQ(b.seed(), 12345u);
}

// End-to-end through the dispatcher: a uniform-range panel rig must resample
// to a uniform range, exercising RayProcessor::processRaw → backend → outputs.
TEST(Dispatch, ProcessRawThroughWrapperProducesUniformRange)
{
    BackendEnvGuard guard("cpu");

    constexpr int H = 8, W = 16;
    constexpr float kRange = 15.0f;

    const auto layout = testutil::makeRig(-10.0f, 10.0f, H, W);
    ASSERT_GT(layout.n_panels, 0);
    const auto raw = testutil::makeUniformRangeRaw(layout.rp, kRange);
    const auto beam_alt = testutil::makeBeams(H, -10.0f, 10.0f);
    std::vector<float> beam_az(H, 0.0f);

    const auto pp = testutil::noNoise(H, W);
    // all noise terms left at 0 → deterministic resample only

    const int n = H * W;
    std::vector<uint32_t> range(n);
    std::vector<uint16_t> signal(n);
    std::vector<uint8_t>  refl(n);
    std::vector<uint16_t> nearir(n);

    RayProcessor proc;
    ASSERT_TRUE(proc.usesCpuFallback());
    proc.processRaw(raw.data(), beam_alt.data(), beam_az.data(), layout.rp,
                    range.data(), signal.data(), refl.data(), nearir.data(), pp);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(static_cast<double>(range[i]), 15000.0, 100.0)
            << "pixel " << i << " range mismatch through dispatcher";
    }
}

}  // namespace gz_gpu_ouster_lidar
