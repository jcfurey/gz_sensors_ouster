// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0
//
// Intel SYCL / oneAPI DPC++ backend.
//
// Targets Intel iGPUs (UHD/Iris Xe) and Arc discrete GPUs. Also works on
// AMD/NVIDIA via AdaptiveCpp (formerly hipSYCL), though CUDA/HIP are the
// preferred paths on those vendors and will be chosen first by the
// dispatcher.
//
// Memory model: always uses sycl::malloc_shared (Unified Shared Memory).
// On Intel iGPUs this maps into integrated DRAM with no explicit copy —
// equivalent to the HIP APU managed-memory path. On discrete Arc, the
// runtime page-migrates on first touch; slightly slower than explicit
// copies but keeps this file simple and correct.
//
// RNG: plain SYCL has no standard RNG library. We inline a tiny
// counter-based generator (splitmix64 → Box-Muller) seeded per-thread from
// the user seed. Output distribution is not bit-identical to curand XORWOW
// but is statistically equivalent for noise simulation.

#include "backend.hpp"

#include <sycl/sycl.hpp>
#include "ray_processor_math.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>

namespace gz_gpu_ouster_lidar {

namespace {

// ── Device RNG (splitmix64 → uniform/normal) ────────────────────────────────
//
// Stateless, counter-based: given (seed, index, call_counter) produce a
// pseudo-random uint64. Each pixel advances its own counter across the
// handful of samples it needs per frame, so no cross-thread synchronisation
// is required.

inline uint64_t splitmix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

inline float uniform01(uint64_t & counter, uint64_t seed, uint32_t pixel_idx)
{
    const uint64_t mix = splitmix64(seed ^ (static_cast<uint64_t>(pixel_idx) * 0xD1B54A32D192ED03ULL) ^ counter);
    ++counter;
    // Map top 24 bits to [0, 1).
    return (mix >> 40) * (1.0f / 16777216.0f);
}

inline float normal01(uint64_t & counter, uint64_t seed, uint32_t pixel_idx)
{
    // Box-Muller on two uniforms. Draws two samples per call; we use one
    // and discard the second to keep the API simple (same pattern as
    // curand_normal).
    const float u1 = sycl::fmax(uniform01(counter, seed, pixel_idx), 1e-7f);
    const float u2 = uniform01(counter, seed, pixel_idx);
    return sycl::sqrt(-2.0f * sycl::log(u1)) *
           sycl::cos(2.0f * rpmath::kPi * u2);
}

// ── SyclBackend ─────────────────────────────────────────────────────────────

class SyclBackend final : public Backend {
public:
    SyclBackend(uint64_t seed, sycl::queue q, bool integrated)
        : seed_(seed), q_(std::move(q)), integrated_(integrated)
    {
        // Derive a non-zero effective seed: production callers pass 0 for
        // non-deterministic; we want new output every run AND distinct
        // output across sensors constructed in the same tick.
        // time(nullptr) has 1-second resolution which is too coarse — mix
        // steady_clock + pid + this-pointer instead.
        effective_seed_ = (seed_ != 0) ? seed_ : deriveNonDeterministicSeed(this);
    }

    ~SyclBackend() override
    {
        auto maybeFree = [&](void * p) { if (p) sycl::free(p, q_); };
        maybeFree(u_depth_);
        maybeFree(u_range_);
        maybeFree(u_signal_);
        maybeFree(u_refl_);
        maybeFree(u_nearir_);
        maybeFree(u_raw_frame_);
        maybeFree(u_beam_alt_);
        maybeFree(u_beam_az_);
    }

    void processRaw(
        const float * raw_host,
        const float * beam_alt_host,
        const float * beam_az_host,
        const ResampleParams & rp,
        uint32_t *    range_out,
        uint16_t *    signal_out,
        uint8_t *     reflectivity_out,
        uint16_t *    nearir_out,
        const RayProcessParams & pp) override
    {
        const int raw_n = rp.raw_n;
        const int out_n = rp.H * rp.W;

        ensureBuffers(out_n);
        ensureResampleBuffers(raw_n, rp.H);

        std::memcpy(u_raw_frame_, raw_host,      static_cast<size_t>(raw_n) * sizeof(float));
        std::memcpy(u_beam_alt_,  beam_alt_host, static_cast<size_t>(rp.H) * sizeof(float));
        std::memcpy(u_beam_az_,   beam_az_host,  static_cast<size_t>(rp.H) * sizeof(float));

        launchResampleKernel(rp, out_n);
        // No laser_retro channel from the panel rig — null retro selects
        // base_reflectivity / unit intensity in the process kernel.
        launchRayKernel(
            u_depth_, nullptr,
            u_range_, u_signal_, u_refl_, u_nearir_, pp, out_n);
        q_.wait();

        std::memcpy(range_out,        u_range_,  static_cast<size_t>(out_n) * sizeof(uint32_t));
        std::memcpy(signal_out,       u_signal_, static_cast<size_t>(out_n) * sizeof(uint16_t));
        std::memcpy(reflectivity_out, u_refl_,   static_cast<size_t>(out_n) * sizeof(uint8_t));
        std::memcpy(nearir_out,       u_nearir_, static_cast<size_t>(out_n) * sizeof(uint16_t));
    }

    const char * name() const override { return integrated_ ? "sycl-igpu" : "sycl"; }

private:
    void launchRayKernel(
        const float * depth, const float * retro,
        uint32_t * range_out, uint16_t * signal_out,
        uint8_t * refl_out, uint16_t * nearir_out,
        const RayProcessParams & p, int n)
    {
        const uint64_t seed = effective_seed_;
        const int H = p.H, W = p.W;
        const float base_signal = p.base_signal;
        const float base_refl = p.base_reflectivity;
        const float rmin = p.range_noise_min_std;
        const float rmax = p.range_noise_max_std;
        const float maxr = p.max_range;
        const float sig_scale = p.signal_noise_scale;
        const float nir_scale = p.nearir_noise_scale;
        const float drop_close = p.dropout_rate_close;
        const float drop_far = p.dropout_rate_far;
        const float edge = p.edge_discon_threshold;

        q_.parallel_for(sycl::range<1>{static_cast<size_t>(n)},
            [=](sycl::id<1> it) {
                const uint32_t idx = static_cast<uint32_t>(it[0]);
                uint64_t counter = 0;

                float d = depth[idx];
                const bool valid = sycl::isfinite(d) && d > rpmath::kValidDepthMin;
                if (!valid) {
                    range_out[idx]  = 0u;
                    signal_out[idx] = 0u;
                    refl_out[idx]   = static_cast<uint8_t>(base_refl);
                    nearir_out[idx] = 0u;
                    return;
                }

                // Depth-discontinuity suppression
                if (edge > 0.f &&
                    rpmath::edgeDiscontinuity(depth, static_cast<int>(idx), H, W, edge) &&
                    uniform01(counter, seed, idx) < rpmath::kEdgeSuppressProb) {
                    range_out[idx]  = 0u;
                    signal_out[idx] = 0u;
                    refl_out[idx]   = static_cast<uint8_t>(base_refl);
                    nearir_out[idx] = 0u;
                    return;
                }

                // Dropouts
                if (drop_close > 0.f || drop_far > 0.f) {
                    const float retro_val = rpmath::retroForNoise(retro, static_cast<int>(idx));
                    const float p_drop = rpmath::dropoutProbability(
                        d, retro_val, drop_close, drop_far, maxr);
                    if (uniform01(counter, seed, idx) < p_drop) {
                        range_out[idx]  = 0u;
                        signal_out[idx] = 0u;
                        refl_out[idx]   = static_cast<uint8_t>(base_refl);
                        nearir_out[idx] = 0u;
                        return;
                    }
                }

                // Range noise
                if (rmin > 0.f || rmax > 0.f) {
                    const float retro_val = rpmath::retroForNoise(retro, static_cast<int>(idx));
                    const float sigma = rpmath::rangeNoiseSigma(d, retro_val, rmin, rmax, maxr);
                    d = sycl::fmax(d + normal01(counter, seed, idx) * sigma, 0.f);
                }

                range_out[idx] = static_cast<uint32_t>(d * rpmath::kRangeToMm);

                // Signal 1/r² + shot noise
                float intensity = 1.0f;
                if (retro) {
                    float r = retro[idx];
                    if (sycl::isfinite(r) && r > 0.f) intensity = r;
                }
                float sig = rpmath::signalFromRange(d, intensity, base_signal);
                if (sig_scale > 0.f) {
                    float sigma_sig = sycl::sqrt(sycl::fmax(sig, 0.f)) * sig_scale;
                    sig = sycl::fmax(sig + normal01(counter, seed, idx) * sigma_sig, 0.f);
                }
                signal_out[idx] = rpmath::clampU16(sig);

                // Reflectivity — shared rpmath::reflectivityToByte. Canonical
                // derivation + upstream Ouster refs in ray_processor_cpu_impl.cpp.
                if (retro && sycl::isfinite(retro[idx]) && retro[idx] > 0.f) {
                    refl_out[idx] = rpmath::reflectivityToByte(retro[idx]);
                } else {
                    refl_out[idx] = static_cast<uint8_t>(base_refl);
                }

                // Near-IR
                float nir = (retro && sycl::isfinite(retro[idx]) && retro[idx] > 0.f)
                    ? retro[idx] * rpmath::kNearIrScale : 0.f;
                if (nir_scale > 0.f && nir > 0.f) {
                    float sigma_nir = sycl::sqrt(sycl::fmax(nir, 0.f)) * nir_scale;
                    nir = sycl::fmax(nir + normal01(counter, seed, idx) * sigma_nir, 0.f);
                }
                nearir_out[idx] = rpmath::clampU16(nir);
            });
    }

    void launchResampleKernel(const ResampleParams & rp, int n)
    {
        const float * raw = u_raw_frame_;
        const float * beam_alt = u_beam_alt_;
        const float * beam_az = u_beam_az_;
        float * depth_out = u_depth_;
        const int W = rp.W;
        const float beam_origin_m = rp.beam_origin_m;
        const float deg_per_col = 360.0f / static_cast<float>(W);
        const float kInf = std::numeric_limits<float>::infinity();
        const ResampleParams rp_copy = rp;  // by-value capture into the kernel

        q_.parallel_for(sycl::range<1>{static_cast<size_t>(n)},
            [=](sycl::id<1> it) {
                const int tid = static_cast<int>(it[0]);
                const int beam = tid / W;
                const int m    = tid % W;

                // Output column m is the Ouster measurement id: m = 0
                // forward (+x), azimuth decreasing with m.
                const float el = beam_alt[beam];
                const float az = beam_az[beam] -
                    static_cast<float>(m) * deg_per_col;

                float depth = rpmath::sampleBeamRange(
                    raw, rp_copy, el, az, kInf);
                depth = rpmath::applyBeamOrigin(depth, el, beam_origin_m);
                depth_out[tid] = depth;
            });
    }

    template <typename T>
    void allocShared(T * & ptr, size_t count)
    {
        if (ptr) sycl::free(ptr, q_);
        ptr = sycl::malloc_shared<T>(count, q_);
        if (!ptr) throw std::runtime_error("sycl::malloc_shared failed");
    }

    void ensureBuffers(int n)
    {
        if (n <= buf_n_) return;
        allocShared(u_depth_,  static_cast<size_t>(n));
        allocShared(u_range_,  static_cast<size_t>(n));
        allocShared(u_signal_, static_cast<size_t>(n));
        allocShared(u_refl_,   static_cast<size_t>(n));
        allocShared(u_nearir_, static_cast<size_t>(n));
        buf_n_ = n;
    }

    void ensureResampleBuffers(int raw_n, int H)
    {
        if (raw_n > raw_buf_n_) {
            allocShared(u_raw_frame_, static_cast<size_t>(raw_n));
            raw_buf_n_ = raw_n;
        }
        if (H > beam_buf_n_) {
            allocShared(u_beam_alt_, static_cast<size_t>(H));
            allocShared(u_beam_az_,  static_cast<size_t>(H));
            beam_buf_n_ = H;
        }
    }

    uint64_t seed_ = 0;
    uint64_t effective_seed_ = 0;
    sycl::queue q_;
    bool integrated_ = false;

    float *    u_depth_     = nullptr;
    uint32_t * u_range_     = nullptr;
    uint16_t * u_signal_    = nullptr;
    uint8_t *  u_refl_      = nullptr;
    uint16_t * u_nearir_    = nullptr;
    float *    u_raw_frame_ = nullptr;
    float *    u_beam_alt_  = nullptr;
    float *    u_beam_az_   = nullptr;
    int buf_n_ = 0;
    int raw_buf_n_ = 0;
    int beam_buf_n_ = 0;
};

}  // namespace

// ── Factory ─────────────────────────────────────────────────────────────────
// Selects a GPU via sycl::gpu_selector. If no SYCL-accessible GPU is found
// (e.g. no Intel iGPU driver, no Arc, no AdaptiveCpp target), returns
// nullptr so the dispatcher falls through to the CPU backend.
//
// host_unified_memory is deprecated in SYCL 2020 but still reported by DPC++
// and AdaptiveCpp; we treat it as a hint that shared USM is cheap. Failing
// the query is not fatal — we still allocate shared, it just may page-migrate.

std::unique_ptr<Backend> makeSyclBackend(uint64_t seed)
{
    try {
        sycl::queue q{sycl::gpu_selector_v,
            sycl::property::queue::in_order{}};

        const auto dev = q.get_device();

        bool integrated = false;
        try {
            integrated = dev.get_info<sycl::info::device::host_unified_memory>();
        } catch (...) {
            integrated = false;
        }

        if (!dev.has(sycl::aspect::usm_shared_allocations)) {
            // Every modern SYCL-capable device supports shared USM; if not,
            // we can't run this backend.
            return nullptr;
        }

        const auto name = dev.get_info<sycl::info::device::name>();
        std::fprintf(stderr,
            "[gz_gpu_ouster_lidar] SYCL backend: device='%s' integrated=%s\n",
            name.c_str(), integrated ? "yes" : "no");

        return std::make_unique<SyclBackend>(seed, std::move(q), integrated);
    } catch (const sycl::exception & e) {
        std::fprintf(stderr,
            "[gz_gpu_ouster_lidar] SYCL backend unavailable: %s\n", e.what());
        return nullptr;
    } catch (...) {
        return nullptr;
    }
}

}  // namespace gz_gpu_ouster_lidar
