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

// End-to-end through the dispatcher: a uniform depth field must resample to a
// uniform range, exercising RayProcessor::processRaw → backend → outputs.
TEST(Dispatch, ProcessRawThroughWrapperProducesUniformRange)
{
    BackendEnvGuard guard("cpu");

    constexpr int H = 8, W = 16;
    constexpr int gpu_H = 32, gpu_W = 16, gpu_chan = 3;
    constexpr float depth = 15.0f;

    std::vector<float> raw(static_cast<size_t>(gpu_H) * gpu_W * gpu_chan, 0.0f);
    for (int r = 0; r < gpu_H; ++r) {
        for (int c = 0; c < gpu_W; ++c) {
            const int base = (r * gpu_W + c) * gpu_chan;
            raw[base] = depth;
            raw[base + 1] = 0.5f;
        }
    }

    std::vector<float> beam_alt(H), beam_az(H, 0.0f);
    const float min_alt = -10.0f, max_alt = 10.0f;
    for (int i = 0; i < H; ++i) {
        beam_alt[i] = min_alt + (max_alt - min_alt) * (i + 0.5f) / H;
    }

    ResampleParams rp{};
    rp.H = H; rp.W = W;
    rp.gpu_H = gpu_H; rp.gpu_W = gpu_W; rp.gpu_chan = gpu_chan;
    rp.min_alt = min_alt; rp.v_range = max_alt - min_alt;
    rp.deg_per_col = 360.0f / W;
    rp.beam_origin_m = 0.0f;
    rp.half_W = W / 2;

    RayProcessParams pp{};
    pp.H = H; pp.W = W;
    pp.base_signal = 800.0f;
    pp.base_reflectivity = 50.0f;
    pp.max_range = 120.0f;
    // all noise terms left at 0 → deterministic resample only

    const int n = H * W;
    std::vector<uint32_t> range(n);
    std::vector<uint16_t> signal(n);
    std::vector<uint8_t>  refl(n);
    std::vector<uint16_t> nearir(n);

    RayProcessor proc;
    ASSERT_TRUE(proc.usesCpuFallback());
    proc.processRaw(raw.data(), beam_alt.data(), beam_az.data(), rp,
                    range.data(), signal.data(), refl.data(), nearir.data(), pp);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(static_cast<double>(range[i]), 15000.0, 100.0)
            << "pixel " << i << " range mismatch through dispatcher";
    }
}

}  // namespace gz_gpu_ouster_lidar
