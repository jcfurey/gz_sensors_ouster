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

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
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
           sycl::cos(2.0f * 3.14159265358979f * u2);
}

// ── SyclBackend ─────────────────────────────────────────────────────────────

class SyclBackend final : public Backend {
public:
    SyclBackend(uint64_t seed, sycl::queue q, bool integrated)
        : seed_(seed), q_(std::move(q)), integrated_(integrated)
    {
        // Derive a non-zero effective seed: production callers pass 0 for
        // non-deterministic; we want new output every run in that case.
        effective_seed_ = (seed_ != 0) ? seed_ :
            static_cast<uint64_t>(std::time(nullptr)) ^ 0xC0FFEEULL;
    }

    ~SyclBackend() override
    {
        auto maybeFree = [&](void * p) { if (p) sycl::free(p, q_); };
        maybeFree(u_depth_);
        maybeFree(u_retro_);
        maybeFree(u_range_);
        maybeFree(u_signal_);
        maybeFree(u_refl_);
        maybeFree(u_nearir_);
        maybeFree(u_raw_frame_);
        maybeFree(u_beam_alt_);
        maybeFree(u_beam_az_);
    }

    void process(
        const float * depth_host,
        const float * retro_host,
        uint32_t *    range_out,
        uint16_t *    signal_out,
        uint8_t *     reflectivity_out,
        uint16_t *    nearir_out,
        const RayProcessParams & p) override
    {
        const int n = p.H * p.W;
        ensureBuffers(n);

        std::memcpy(u_depth_, depth_host, static_cast<size_t>(n) * sizeof(float));
        if (retro_host) {
            std::memcpy(u_retro_, retro_host, static_cast<size_t>(n) * sizeof(float));
        }

        launchRayKernel(
            u_depth_, retro_host ? u_retro_ : nullptr,
            u_range_, u_signal_, u_refl_, u_nearir_, p, n);
        q_.wait();

        std::memcpy(range_out,        u_range_,  static_cast<size_t>(n) * sizeof(uint32_t));
        std::memcpy(signal_out,       u_signal_, static_cast<size_t>(n) * sizeof(uint16_t));
        std::memcpy(reflectivity_out, u_refl_,   static_cast<size_t>(n) * sizeof(uint8_t));
        std::memcpy(nearir_out,       u_nearir_, static_cast<size_t>(n) * sizeof(uint16_t));
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
        const int raw_n = rp.gpu_H * rp.gpu_W * rp.gpu_chan;
        const int out_n = rp.H * rp.W;

        ensureBuffers(out_n);
        ensureResampleBuffers(raw_n, rp.H);

        std::memcpy(u_raw_frame_, raw_host,      static_cast<size_t>(raw_n) * sizeof(float));
        std::memcpy(u_beam_alt_,  beam_alt_host, static_cast<size_t>(rp.H) * sizeof(float));
        std::memcpy(u_beam_az_,   beam_az_host,  static_cast<size_t>(rp.H) * sizeof(float));

        launchResampleKernel(rp, out_n);
        launchRayKernel(
            u_depth_, u_retro_,
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
                const bool valid = sycl::isfinite(d) && d > 0.001f;
                if (!valid) {
                    range_out[idx]  = 0u;
                    signal_out[idx] = 0u;
                    refl_out[idx]   = static_cast<uint8_t>(base_refl);
                    nearir_out[idx] = 0u;
                    return;
                }

                // Depth-discontinuity suppression
                if (edge > 0.f) {
                    const int beam = idx / W;
                    const int col  = idx % W;
                    bool suppress = false;
                    auto checkN = [&](int nb, int nc) {
                        if (nb < 0 || nb >= H || nc < 0 || nc >= W) return;
                        float nd = depth[nb * W + nc];
                        if (!sycl::isfinite(nd) || nd < 0.001f ||
                            sycl::fabs(nd - d) > edge)
                            suppress = true;
                    };
                    checkN(beam - 1, col);
                    checkN(beam + 1, col);
                    checkN(beam, col - 1);
                    checkN(beam, col + 1);
                    if (suppress && uniform01(counter, seed, idx) < 0.5f) {
                        range_out[idx]  = 0u;
                        signal_out[idx] = 0u;
                        refl_out[idx]   = static_cast<uint8_t>(base_refl);
                        nearir_out[idx] = 0u;
                        return;
                    }
                }

                // Dropouts
                if (drop_close > 0.f || drop_far > 0.f) {
                    float t = sycl::fmin(d / sycl::fmax(maxr, 0.1f), 1.f);
                    float p_drop = drop_close + t * (drop_far - drop_close);
                    float retro_val = (retro && sycl::isfinite(retro[idx]) && retro[idx] > 0.f)
                        ? retro[idx] : 0.5f;
                    float refl_factor = sycl::fmin(1.0f / sycl::fmax(retro_val, 0.33f), 3.0f);
                    p_drop *= refl_factor;
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
                    float t = sycl::fmin(d / sycl::fmax(maxr, 0.1f), 1.f);
                    float sigma = rmin + t * (rmax - rmin);
                    float retro_val = (retro && sycl::isfinite(retro[idx]) && retro[idx] > 0.f)
                        ? retro[idx] : 0.5f;
                    sigma *= sycl::fmin(1.0f / sycl::fmax(retro_val, 0.5f), 2.0f);
                    d = sycl::fmax(d + normal01(counter, seed, idx) * sigma, 0.f);
                }

                range_out[idx] = static_cast<uint32_t>(d * 1000.f);

                // Signal 1/r² + shot noise
                float intensity = 1.0f;
                if (retro) {
                    float r = retro[idx];
                    if (sycl::isfinite(r) && r > 0.f) intensity = r;
                }
                const float r_sq = d * d;
                float sig = base_signal * intensity / sycl::fmax(r_sq, 0.0001f);
                if (sig_scale > 0.f) {
                    float sigma_sig = sycl::sqrt(sycl::fmax(sig, 0.f)) * sig_scale;
                    sig = sycl::fmax(sig + normal01(counter, seed, idx) * sigma_sig, 0.f);
                }
                signal_out[idx] = static_cast<uint16_t>(
                    sycl::fmin(sycl::fmax(sig, 0.f), 65535.f));

                // Reflectivity
                if (retro && sycl::isfinite(retro[idx]) && retro[idx] > 0.f) {
                    float rv = retro[idx];
                    uint8_t refl;
                    if (rv <= 1.0f) {
                        refl = static_cast<uint8_t>(sycl::fmin(rv * 100.f, 100.f));
                    } else {
                        refl = static_cast<uint8_t>(
                            sycl::fmin(100.f + sycl::log2(rv) * 22.f, 255.f));
                    }
                    refl_out[idx] = refl;
                } else {
                    refl_out[idx] = static_cast<uint8_t>(base_refl);
                }

                // Near-IR
                float nir = (retro && sycl::isfinite(retro[idx]) && retro[idx] > 0.f)
                    ? retro[idx] * 256.f : 0.f;
                if (nir_scale > 0.f && nir > 0.f) {
                    float sigma_nir = sycl::sqrt(sycl::fmax(nir, 0.f)) * nir_scale;
                    nir = sycl::fmax(nir + normal01(counter, seed, idx) * sigma_nir, 0.f);
                }
                nearir_out[idx] = static_cast<uint16_t>(
                    sycl::fmin(sycl::fmax(nir, 0.f), 65535.f));
            });
    }

    void launchResampleKernel(const ResampleParams & rp, int n)
    {
        const float * raw = u_raw_frame_;
        const float * beam_alt = u_beam_alt_;
        const float * beam_az = u_beam_az_;
        float * depth_out = u_depth_;
        float * retro_out = u_retro_;
        const int H = rp.H, W = rp.W;
        const int gpu_H = rp.gpu_H, gpu_W = rp.gpu_W, gpu_chan = rp.gpu_chan;
        const float min_alt = rp.min_alt;
        const float v_range = rp.v_range;
        const float deg_per_col = rp.deg_per_col;
        const float beam_origin_m = rp.beam_origin_m;
        const int half_W = rp.half_W;
        const float kInf = std::numeric_limits<float>::infinity();

        q_.parallel_for(sycl::range<1>{static_cast<size_t>(n)},
            [=](sycl::id<1> it) {
                const int tid = static_cast<int>(it[0]);
                const int beam = tid / W;
                const int col  = tid % W;

                const float beam_angle = beam_alt[beam];
                const float v_frac = (beam_angle - min_alt) / v_range;
                const float row_f = v_frac * (gpu_H - 1);
                const int row_lo = sycl::max(sycl::min(static_cast<int>(sycl::floor(row_f)), gpu_H - 1), 0);
                const int row_hi = sycl::min(row_lo + 1, gpu_H - 1);
                const float v_alpha = row_f - row_lo;

                const float az_offset_cols = beam_az[beam] / deg_per_col;
                float col_wrapped = sycl::fmod(
                    static_cast<float>(col) + az_offset_cols + gpu_W,
                    static_cast<float>(gpu_W));
                if (col_wrapped < 0.f) col_wrapped += gpu_W;
                const int col_lo = static_cast<int>(sycl::floor(col_wrapped)) % gpu_W;
                const int col_hi = (col_lo + 1) % gpu_W;
                const float h_alpha = col_wrapped - sycl::floor(col_wrapped);

                const int idx_00 = (row_lo * gpu_W + col_lo) * gpu_chan;
                const int idx_01 = (row_lo * gpu_W + col_hi) * gpu_chan;
                const int idx_10 = (row_hi * gpu_W + col_lo) * gpu_chan;
                const int idx_11 = (row_hi * gpu_W + col_hi) * gpu_chan;

                const float d00 = raw[idx_00], d01 = raw[idx_01];
                const float d10 = raw[idx_10], d11 = raw[idx_11];
                const bool v00 = !sycl::isinf(d00), v01 = !sycl::isinf(d01);
                const bool v10 = !sycl::isinf(d10), v11 = !sycl::isinf(d11);
                const int n_valid = v00 + v01 + v10 + v11;

                float depth;
                if (n_valid == 0) {
                    depth = kInf;
                } else if (n_valid == 4) {
                    const float d_top = d00 * (1.f - h_alpha) + d01 * h_alpha;
                    const float d_bot = d10 * (1.f - h_alpha) + d11 * h_alpha;
                    depth = d_top * (1.f - v_alpha) + d_bot * v_alpha;
                } else {
                    float sum = 0.f;
                    if (v00) sum += d00;
                    if (v01) sum += d01;
                    if (v10) sum += d10;
                    if (v11) sum += d11;
                    depth = sum / static_cast<float>(n_valid);
                }

                if (sycl::isfinite(depth) && beam_origin_m > 0.f) {
                    const float elev_rad = beam_angle * 3.14159265358979f / 180.f;
                    depth = sycl::fmax(0.f, depth - beam_origin_m * sycl::cos(elev_rad));
                }

                const int m_id = (half_W - col + W) % W;
                const int ouster_idx = beam * W + m_id;
                depth_out[ouster_idx] = depth;

                if (gpu_chan >= 2) {
                    const float r00 = raw[idx_00 + 1], r01 = raw[idx_01 + 1];
                    const float r10 = raw[idx_10 + 1], r11 = raw[idx_11 + 1];
                    float retro;
                    if (n_valid == 4) {
                        const float r_top = r00 * (1.f - h_alpha) + r01 * h_alpha;
                        const float r_bot = r10 * (1.f - h_alpha) + r11 * h_alpha;
                        retro = r_top * (1.f - v_alpha) + r_bot * v_alpha;
                    } else if (n_valid > 0) {
                        float sum = 0.f;
                        if (v00) sum += r00;
                        if (v01) sum += r01;
                        if (v10) sum += r10;
                        if (v11) sum += r11;
                        retro = sum / static_cast<float>(n_valid);
                    } else {
                        retro = 0.f;
                    }
                    retro_out[ouster_idx] = retro;
                } else {
                    retro_out[ouster_idx] = 0.f;
                }
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
        allocShared(u_retro_,  static_cast<size_t>(n));
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
    float *    u_retro_     = nullptr;
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
