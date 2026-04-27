// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0
//
// AMD HIP / ROCm backend.
//
// Kernel bodies are near-1:1 mirrors of ray_processor_cuda.cu. When
// fixing a bug in the math below, update the CUDA file too. Long comments
// explaining the noise/resample model live in ray_processor_cuda.cu; this
// file keeps only HIP-specific commentary.
//
// APU handling: on integrated GPUs (Ryzen APUs) the device shares DRAM
// with the CPU. We detect this via hipDeviceProp_t::integrated and switch
// to hipMallocManaged / hipHostMalloc paths that skip explicit H2D/D2H
// copies.

#include "backend.hpp"

#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <memory>
#include <stdexcept>

namespace gz_gpu_ouster_lidar {

namespace {

// ── Helpers ──────────────────────────────────────────────────────────────────

void checkHip(hipError_t err, const char * file, int line)
{
    if (err != hipSuccess) {
        char msg[256];
        std::snprintf(msg, sizeof(msg),
            "HIP error at %s:%d — %s", file, line, hipGetErrorString(err));
        throw std::runtime_error(msg);
    }
}
#define HIP_CHECK(expr) checkHip((expr), __FILE__, __LINE__)

constexpr int kBlock = 256;

// ── Kernels (mirror CUDA) ────────────────────────────────────────────────────

__global__ void initRandStatesHip(hiprandState * states, unsigned long seed, int n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        hiprand_init(seed, static_cast<unsigned long long>(idx), 0, &states[idx]);
    }
}

__global__ void resampleKernelHip(
    const float * __restrict__ raw,
    const float * __restrict__ beam_alt,
    const float * __restrict__ beam_az,
    float * __restrict__       depth_out,
    float * __restrict__       retro_out,
    int H, int W,
    int gpu_H, int gpu_W, int gpu_chan,
    float min_alt, float v_range,
    float deg_per_col, float beam_origin_m,
    int half_W)
{
    const int n = H * W;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    const int beam = tid / W;
    const int col  = tid % W;

    const float beam_angle = beam_alt[beam];

    const float v_frac = (beam_angle - min_alt) / v_range;
    const float row_f  = v_frac * (gpu_H - 1);
    const int row_lo   = max(min(__float2int_rd(row_f), gpu_H - 1), 0);
    const int row_hi   = min(row_lo + 1, gpu_H - 1);
    const float v_alpha = row_f - row_lo;

    const float az_offset_cols = beam_az[beam] / deg_per_col;
    float col_wrapped = fmodf(static_cast<float>(col) + az_offset_cols + gpu_W,
                              static_cast<float>(gpu_W));
    if (col_wrapped < 0.f) col_wrapped += gpu_W;
    const int col_lo = __float2int_rd(col_wrapped) % gpu_W;
    const int col_hi = (col_lo + 1) % gpu_W;
    const float h_alpha = col_wrapped - floorf(col_wrapped);

    const int idx_00 = (row_lo * gpu_W + col_lo) * gpu_chan;
    const int idx_01 = (row_lo * gpu_W + col_hi) * gpu_chan;
    const int idx_10 = (row_hi * gpu_W + col_lo) * gpu_chan;
    const int idx_11 = (row_hi * gpu_W + col_hi) * gpu_chan;

    const float d00 = raw[idx_00], d01 = raw[idx_01];
    const float d10 = raw[idx_10], d11 = raw[idx_11];

    const bool v00 = !isinf(d00), v01 = !isinf(d01);
    const bool v10 = !isinf(d10), v11 = !isinf(d11);
    const int n_valid = v00 + v01 + v10 + v11;

    float depth;
    if (n_valid == 0) {
        depth = __int_as_float(0x7f800000);  // +inf
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

    if (isfinite(depth) && beam_origin_m > 0.f) {
        const float elev_rad = beam_angle * 3.14159265358979f / 180.f;
        depth = fmaxf(0.f, depth - beam_origin_m * cosf(elev_rad));
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
}

__global__ void rayProcessKernelHip(
    const float * __restrict__ depth,
    const float * __restrict__ retro,
    uint32_t * __restrict__    range_out,
    uint16_t * __restrict__    signal_out,
    uint8_t *  __restrict__    refl_out,
    uint16_t * __restrict__    nearir_out,
    int H, int W,
    float base_signal,
    float base_reflectivity,
    float range_noise_min_std,
    float range_noise_max_std,
    float max_range,
    float signal_noise_scale,
    float nearir_noise_scale,
    float dropout_rate_close,
    float dropout_rate_far,
    float edge_discon_threshold,
    hiprandState * __restrict__ rand_states)
{
    const int n = H * W;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float d = depth[idx];
    const bool valid = isfinite(d) && d > 0.001f;

    if (!valid) {
        range_out[idx]  = 0u;
        signal_out[idx] = 0u;
        refl_out[idx]   = static_cast<uint8_t>(base_reflectivity);
        nearir_out[idx] = 0u;
        return;
    }

    hiprandState * rs = (rand_states != nullptr) ? &rand_states[idx] : nullptr;

    if (edge_discon_threshold > 0.f) {
        const int beam = idx / W;
        const int col  = idx % W;
        bool suppress = false;
        auto checkNeighbor = [&](int nb, int nc) {
            if (nb < 0 || nb >= H || nc < 0 || nc >= W) return;
            float nd = depth[nb * W + nc];
            if (!isfinite(nd) || nd < 0.001f || fabsf(nd - d) > edge_discon_threshold)
                suppress = true;
        };
        checkNeighbor(beam - 1, col);
        checkNeighbor(beam + 1, col);
        checkNeighbor(beam, col - 1);
        checkNeighbor(beam, col + 1);

        if (suppress && rs != nullptr && hiprand_uniform(rs) < 0.5f) {
            range_out[idx]  = 0u;
            signal_out[idx] = 0u;
            refl_out[idx]   = static_cast<uint8_t>(base_reflectivity);
            nearir_out[idx] = 0u;
            return;
        }
    }

    if (rs != nullptr && (dropout_rate_close > 0.f || dropout_rate_far > 0.f)) {
        float t = fminf(d / fmaxf(max_range, 0.1f), 1.f);
        float p_dropout = dropout_rate_close + t * (dropout_rate_far - dropout_rate_close);
        float retro_val = (retro != nullptr && isfinite(retro[idx]) && retro[idx] > 0.f)
            ? retro[idx] : 0.5f;
        float refl_factor = fminf(1.0f / fmaxf(retro_val, 0.33f), 3.0f);
        p_dropout *= refl_factor;
        if (hiprand_uniform(rs) < p_dropout) {
            range_out[idx]  = 0u;
            signal_out[idx] = 0u;
            refl_out[idx]   = static_cast<uint8_t>(base_reflectivity);
            nearir_out[idx] = 0u;
            return;
        }
    }

    if (rs != nullptr && (range_noise_min_std > 0.f || range_noise_max_std > 0.f)) {
        float t = fminf(d / fmaxf(max_range, 0.1f), 1.f);
        float sigma = range_noise_min_std + t * (range_noise_max_std - range_noise_min_std);
        float retro_val = (retro != nullptr && isfinite(retro[idx]) && retro[idx] > 0.f)
            ? retro[idx] : 0.5f;
        sigma *= fminf(1.0f / fmaxf(retro_val, 0.5f), 2.0f);
        d = fmaxf(d + hiprand_normal(rs) * sigma, 0.f);
    }

    range_out[idx] = static_cast<uint32_t>(d * 1000.f);

    float intensity = 1.0f;
    if (retro != nullptr) {
        float r = retro[idx];
        if (isfinite(r) && r > 0.f) intensity = r;
    }
    const float r_sq = d * d;
    float sig = base_signal * intensity / fmaxf(r_sq, 0.0001f);
    if (rs != nullptr && signal_noise_scale > 0.f) {
        float sigma_sig = sqrtf(fmaxf(sig, 0.f)) * signal_noise_scale;
        sig = fmaxf(sig + hiprand_normal(rs) * sigma_sig, 0.f);
    }
    signal_out[idx] = static_cast<uint16_t>(fminf(fmaxf(sig, 0.f), 65535.f));

    if (retro != nullptr && isfinite(retro[idx]) && retro[idx] > 0.f) {
        float rv = retro[idx];
        uint8_t refl;
        if (rv <= 1.0f) {
            refl = static_cast<uint8_t>(fminf(rv * 100.f, 100.f));
        } else {
            refl = static_cast<uint8_t>(fminf(100.f + log2f(rv) * 22.f, 255.f));
        }
        refl_out[idx] = refl;
    } else {
        refl_out[idx] = static_cast<uint8_t>(base_reflectivity);
    }

    float nir = (retro != nullptr && isfinite(retro[idx]) && retro[idx] > 0.f)
        ? retro[idx] * 256.f : 0.f;
    if (rs != nullptr && nearir_noise_scale > 0.f && nir > 0.f) {
        float sigma_nir = sqrtf(fmaxf(nir, 0.f)) * nearir_noise_scale;
        nir = fmaxf(nir + hiprand_normal(rs) * sigma_nir, 0.f);
    }
    nearir_out[idx] = static_cast<uint16_t>(fminf(fmaxf(nir, 0.f), 65535.f));
}

// ── HipBackend ──────────────────────────────────────────────────────────────

class HipBackend final : public Backend {
public:
    HipBackend(uint64_t seed, hipStream_t stream, bool integrated)
        : seed_(seed), stream_(stream), integrated_(integrated) {}

    ~HipBackend() override
    {
        auto maybeFree = [](void * p) { if (p) hipFree(p); };
        maybeFree(d_depth_);
        maybeFree(d_retro_);
        maybeFree(d_range_);
        maybeFree(d_signal_);
        maybeFree(d_refl_);
        maybeFree(d_nearir_);
        maybeFree(d_raw_frame_);
        maybeFree(d_beam_alt_);
        maybeFree(d_beam_az_);
        maybeFree(d_rand_states_);
        if (stream_) hipStreamDestroy(stream_);
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

        const bool need_rand = noiseEnabled(pp);
        if (need_rand) ensureRandStates(out_n);

        h2d(d_raw_frame_, raw_host, static_cast<size_t>(raw_n) * sizeof(float));
        h2d(d_beam_alt_, beam_alt_host, static_cast<size_t>(rp.H) * sizeof(float));
        h2d(d_beam_az_,  beam_az_host,  static_cast<size_t>(rp.H) * sizeof(float));

        const int grid_r = (out_n + kBlock - 1) / kBlock;
        hipLaunchKernelGGL(resampleKernelHip, dim3(grid_r), dim3(kBlock), 0, stream_,
            static_cast<const float *>(d_raw_frame_),
            static_cast<const float *>(d_beam_alt_),
            static_cast<const float *>(d_beam_az_),
            static_cast<float *>(d_depth_),
            static_cast<float *>(d_retro_),
            rp.H, rp.W, rp.gpu_H, rp.gpu_W, rp.gpu_chan,
            rp.min_alt, rp.v_range,
            rp.deg_per_col, rp.beam_origin_m, rp.half_W);
        HIP_CHECK(hipGetLastError());

        hipLaunchKernelGGL(rayProcessKernelHip, dim3(grid_r), dim3(kBlock), 0, stream_,
            static_cast<const float *>(d_depth_),
            static_cast<const float *>(d_retro_),
            static_cast<uint32_t *>(d_range_),
            static_cast<uint16_t *>(d_signal_),
            static_cast<uint8_t *>(d_refl_),
            static_cast<uint16_t *>(d_nearir_),
            pp.H, pp.W,
            pp.base_signal, pp.base_reflectivity,
            pp.range_noise_min_std, pp.range_noise_max_std, pp.max_range,
            pp.signal_noise_scale, pp.nearir_noise_scale,
            pp.dropout_rate_close, pp.dropout_rate_far,
            pp.edge_discon_threshold,
            need_rand ? static_cast<hiprandState *>(d_rand_states_) : nullptr);
        HIP_CHECK(hipGetLastError());

        d2hResults(range_out, signal_out, reflectivity_out, nearir_out, out_n);
        HIP_CHECK(hipStreamSynchronize(stream_));
    }

    const char * name() const override { return integrated_ ? "hip-apu" : "hip"; }

private:
    // APU (integrated GPU) shares DRAM with the CPU. hipMallocManaged lets
    // the runtime map the allocation into both address spaces without an
    // explicit copy. On discrete AMD GPUs we use hipMalloc + async memcpy
    // because explicit transfers over PCIe are still faster than page-
    // migrated managed memory.
    void allocDev(void * & ptr, size_t bytes)
    {
        if (ptr) { HIP_CHECK(hipFree(ptr)); ptr = nullptr; }
        if (integrated_) {
            HIP_CHECK(hipMallocManaged(&ptr, bytes, hipMemAttachGlobal));
        } else {
            HIP_CHECK(hipMalloc(&ptr, bytes));
        }
    }

    // On APUs (managed memory), the destination pointer is the same address
    // the host can read; skip the async copy entirely and memcpy on host.
    // On discrete GPUs, use hipMemcpyAsync over the stream.
    void h2d(void * dst, const void * src, size_t bytes)
    {
        if (integrated_) {
            std::memcpy(dst, src, bytes);
        } else {
            HIP_CHECK(hipMemcpyAsync(dst, src, bytes,
                hipMemcpyHostToDevice, stream_));
        }
    }

    void d2h(void * dst, const void * src, size_t bytes)
    {
        if (integrated_) {
            // Sync to ensure kernels have finished before the host reads.
            HIP_CHECK(hipStreamSynchronize(stream_));
            std::memcpy(dst, src, bytes);
        } else {
            HIP_CHECK(hipMemcpyAsync(dst, src, bytes,
                hipMemcpyDeviceToHost, stream_));
        }
    }

    void ensureBuffers(int n)
    {
        if (n <= buf_n_) return;
        allocDev(d_depth_,  static_cast<size_t>(n) * sizeof(float));
        allocDev(d_retro_,  static_cast<size_t>(n) * sizeof(float));
        allocDev(d_range_,  static_cast<size_t>(n) * sizeof(uint32_t));
        allocDev(d_signal_, static_cast<size_t>(n) * sizeof(uint16_t));
        allocDev(d_refl_,   static_cast<size_t>(n) * sizeof(uint8_t));
        allocDev(d_nearir_, static_cast<size_t>(n) * sizeof(uint16_t));
        buf_n_ = n;
    }

    void ensureResampleBuffers(int raw_n, int H)
    {
        if (raw_n > raw_buf_n_) {
            allocDev(d_raw_frame_, static_cast<size_t>(raw_n) * sizeof(float));
            raw_buf_n_ = raw_n;
        }
        if (H > beam_buf_n_) {
            allocDev(d_beam_alt_, static_cast<size_t>(H) * sizeof(float));
            allocDev(d_beam_az_,  static_cast<size_t>(H) * sizeof(float));
            beam_buf_n_ = H;
        }
    }

    void ensureRandStates(int n)
    {
        if (n <= rand_n_) return;
        if (d_rand_states_) { HIP_CHECK(hipFree(d_rand_states_)); d_rand_states_ = nullptr; }
        HIP_CHECK(hipMalloc(&d_rand_states_,
            static_cast<size_t>(n) * sizeof(hiprandState)));
        const unsigned long rng_seed = (seed_ != 0)
            ? static_cast<unsigned long>(seed_)
            : static_cast<unsigned long>(clock());
        const int grid = (n + kBlock - 1) / kBlock;
        hipLaunchKernelGGL(initRandStatesHip, dim3(grid), dim3(kBlock), 0, stream_,
            static_cast<hiprandState *>(d_rand_states_), rng_seed, n);
        HIP_CHECK(hipGetLastError());
        rand_n_ = n;
    }

    void d2hResults(
        uint32_t * range_out, uint16_t * signal_out,
        uint8_t * reflectivity_out, uint16_t * nearir_out, int n)
    {
        d2h(range_out,        d_range_,  static_cast<size_t>(n) * sizeof(uint32_t));
        d2h(signal_out,       d_signal_, static_cast<size_t>(n) * sizeof(uint16_t));
        d2h(reflectivity_out, d_refl_,   static_cast<size_t>(n) * sizeof(uint8_t));
        d2h(nearir_out,       d_nearir_, static_cast<size_t>(n) * sizeof(uint16_t));
    }

    uint64_t seed_ = 0;
    hipStream_t stream_ = nullptr;
    bool integrated_ = false;

    void * d_depth_ = nullptr;
    void * d_retro_ = nullptr;
    void * d_range_ = nullptr;
    void * d_signal_ = nullptr;
    void * d_refl_ = nullptr;
    void * d_nearir_ = nullptr;
    void * d_rand_states_ = nullptr;
    void * d_raw_frame_ = nullptr;
    void * d_beam_alt_ = nullptr;
    void * d_beam_az_ = nullptr;
    int buf_n_ = 0;
    int rand_n_ = 0;
    int raw_buf_n_ = 0;
    int beam_buf_n_ = 0;
};

}  // namespace

// ── Factory ─────────────────────────────────────────────────────────────────
// Returns nullptr if no ROCm-usable device is present. Picks device 0 and
// reads its hipDeviceProp_t::integrated flag to choose the APU code path.

std::unique_ptr<Backend> makeHipBackend(uint64_t seed)
{
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess || device_count == 0) {
        hipGetLastError();
        return nullptr;
    }

    hipDeviceProp_t props;
    err = hipGetDeviceProperties(&props, 0);
    if (err != hipSuccess) {
        hipGetLastError();
        return nullptr;
    }
    const bool integrated = props.integrated != 0;

    hipStream_t s = nullptr;
    err = hipStreamCreate(&s);
    if (err != hipSuccess) {
        hipGetLastError();
        return nullptr;
    }

    std::fprintf(stderr,
        "[gz_gpu_ouster_lidar] HIP backend: device='%s' integrated=%s "
        "(managed-memory path %s)\n",
        props.name, integrated ? "yes" : "no",
        integrated ? "ON" : "OFF");

    return std::make_unique<HipBackend>(seed, s, integrated);
}

}  // namespace gz_gpu_ouster_lidar
