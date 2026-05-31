// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0
//
// NVIDIA CUDA backend. Kernel bodies are near-1:1 mirrored in
// ray_processor_hip.cpp (AMD HIP). When fixing a bug in the math here,
// check there too. The SYCL backend (ray_processor_sycl.cpp) re-expresses
// the same math in a SYCL kernel model.

#include "ray_processor_cuda.cuh"
#include "backend.hpp"
#include "ray_processor_math.hpp"

#include <cuda_runtime.h>
#include <math_constants.h>
#include <curand_kernel.h>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>

namespace gz_gpu_ouster_lidar {

// ── Helpers ──────────────────────────────────────────────────────────────────

static void checkCuda(cudaError_t err, const char * file, int line)
{
    if (err != cudaSuccess) {
        char msg[256];
        std::snprintf(msg, sizeof(msg),
            "CUDA error at %s:%d — %s", file, line, cudaGetErrorString(err));
        throw std::runtime_error(msg);
    }
}
#define CUDA_CHECK(expr) checkCuda((expr), __FILE__, __LINE__)

static constexpr int kBlock = 256;

// ── Kernels ──────────────────────────────────────────────────────────────────

__global__ void initRandStates(curandState * states, unsigned long seed, int n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, static_cast<unsigned long long>(idx), 0, &states[idx]);
    }
}

/// Resample GpuRays uniform grid → exact Ouster beam geometry.
/// One thread per output pixel (beam × col).  Performs 2D bilinear
/// interpolation with azimuth wrapping and beam-origin correction.
__global__ void resampleKernel(
    const float * __restrict__ raw,          // gpu_H × gpu_W × gpu_chan
    const float * __restrict__ beam_alt,     // H beam altitude angles (degrees)
    const float * __restrict__ beam_az,      // H beam azimuth offsets (degrees)
    float * __restrict__       depth_out,    // H × W
    float * __restrict__       retro_out,    // H × W
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

    // Fractional row in GpuRays grid (vertical interpolation).
    const float v_frac = (beam_angle - min_alt) / v_range;
    const float row_f  = v_frac * (gpu_H - 1);
    const int row_lo   = max(min(__float2int_rd(row_f), gpu_H - 1), 0);
    const int row_hi   = min(row_lo + 1, gpu_H - 1);  // NOLINT(build/include_what_you_use) — CUDA device-side min, not std::min
    const float v_alpha = row_f - row_lo;

    // Per-beam azimuth offset → fractional column shift.
    const float az_offset_cols = beam_az[beam] / deg_per_col;
    float col_wrapped = fmodf(static_cast<float>(col) + az_offset_cols + gpu_W,
                              static_cast<float>(gpu_W));
    if (col_wrapped < 0.f) col_wrapped += gpu_W;
    const int col_lo = __float2int_rd(col_wrapped) % gpu_W;
    const int col_hi = (col_lo + 1) % gpu_W;
    const float h_alpha = col_wrapped - floorf(col_wrapped);

    // 2D bilinear: four corner depth samples.
    const int idx_00 = (row_lo * gpu_W + col_lo) * gpu_chan;
    const int idx_01 = (row_lo * gpu_W + col_hi) * gpu_chan;
    const int idx_10 = (row_hi * gpu_W + col_lo) * gpu_chan;
    const int idx_11 = (row_hi * gpu_W + col_hi) * gpu_chan;

    const float d00 = raw[idx_00], d01 = raw[idx_01];
    const float d10 = raw[idx_10], d11 = raw[idx_11];

    const bool v00 = !isinf(d00), v01 = !isinf(d01);
    const bool v10 = !isinf(d10), v11 = !isinf(d11);
    const int n_valid = v00 + v01 + v10 + v11;

    // Bilinear blend + beam-origin correction (shared with all backends).
    float depth = rpmath::bilinearOrAverage(
        d00, d01, d10, d11, h_alpha, v_alpha, v00, v01, v10, v11,
        n_valid, CUDART_INF_F);
    depth = rpmath::applyBeamOrigin(depth, beam_angle, beam_origin_m);

    // Azimuth remapping: Gazebo col 0 = −π, Ouster m_id 0 = encoder 0.
    const int m_id = (half_W - col + W) % W;
    const int ouster_idx = beam * W + m_id;
    depth_out[ouster_idx] = depth;

    // Retro channel — same bilinear (empty = 0, not +inf).
    if (gpu_chan >= 2) {
        const float r00 = raw[idx_00 + 1], r01 = raw[idx_01 + 1];
        const float r10 = raw[idx_10 + 1], r11 = raw[idx_11 + 1];
        retro_out[ouster_idx] = rpmath::bilinearOrAverage(
            r00, r01, r10, r11, h_alpha, v_alpha, v00, v01, v10, v11,
            n_valid, 0.f);
    } else {
        retro_out[ouster_idx] = 0.f;
    }
}

// ── Resample launcher ────────────────────────────────────────────────────────

void launchResampleKernel(
    const float * d_raw_frame,
    const float * d_beam_alt,
    const float * d_beam_az,
    float *       d_depth_out,
    float *       d_retro_out,
    const ResampleParams & rp,
    void * stream)
{
    const int n = rp.H * rp.W;
    const int grid = (n + kBlock - 1) / kBlock;
    resampleKernel<<<grid, kBlock, 0, static_cast<cudaStream_t>(stream)>>>(
        d_raw_frame, d_beam_alt, d_beam_az,
        d_depth_out, d_retro_out,
        rp.H, rp.W, rp.gpu_H, rp.gpu_W, rp.gpu_chan,
        rp.min_alt, rp.v_range,
        rp.deg_per_col, rp.beam_origin_m, rp.half_W);
    CUDA_CHECK(cudaGetLastError());
}

/// Process depth + retro → range_mm, signal, reflectivity.
///
/// Each thread handles one pixel (beam × column).
/// Input layout: depth_buf[beam * W + col] = depth in metres (inf = miss).
/// Output: range in mm, signal via 1/r² model, reflectivity from retro.
///
/// Noise model:
///   Range:        Gaussian, σ linearly interpolated from min_std..max_std over [0, max_range]
///   Signal:       Poisson-approximated shot noise: σ = sqrt(signal) * scale
///   Near-IR:      Same Poisson model on retro channel
///   Dropouts:     Uniform random < linearly-interpolated dropout probability → zero return
///   Edge suppression: if any cardinal neighbor depth differs by > threshold → suppress
__global__ void rayProcessKernel(
    const float * __restrict__   depth,
    const float * __restrict__   retro,   // may be nullptr
    uint32_t * __restrict__      range_out,
    uint16_t * __restrict__      signal_out,
    uint8_t *  __restrict__      refl_out,
    uint16_t * __restrict__      nearir_out,
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
    curandState * __restrict__   rand_states)
{
    const int n = H * W;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float d = depth[idx];
    const bool valid = isfinite(d) && d > rpmath::kValidDepthMin;

    if (!valid) {
        range_out[idx]  = 0u;
        signal_out[idx] = 0u;
        refl_out[idx]   = static_cast<uint8_t>(base_reflectivity);
        nearir_out[idx] = 0u;
        return;
    }

    curandState * rs = (rand_states != nullptr) ? &rand_states[idx] : nullptr;

    // ── Depth-discontinuity suppression ──────────────────────────────────────
    // Suppress returns at depth edges where a cardinal neighbor is much farther
    // or missing.  This mimics the mixed-return rejection in real Ouster firmware.
    // (rs is non-null exactly when noise is enabled, which edge_discon_threshold>0
    // implies, so the curand draw only happens at a detected edge.)
    if (edge_discon_threshold > 0.f && rs != nullptr &&
        rpmath::edgeDiscontinuity(depth, idx, H, W, edge_discon_threshold) &&
        curand_uniform(rs) < rpmath::kEdgeSuppressProb) {
        range_out[idx]  = 0u;
        signal_out[idx] = 0u;
        refl_out[idx]   = static_cast<uint8_t>(base_reflectivity);
        nearir_out[idx] = 0u;
        return;
    }

    // ── Random dropouts ──────────────────────────────────────────────────────
    // Dropout probability increases with range AND decreases with reflectivity.
    // Real sensors lose more returns on low-reflectivity targets at distance.
    if (rs != nullptr && (dropout_rate_close > 0.f || dropout_rate_far > 0.f)) {
        const float retro_val = rpmath::retroForNoise(retro, idx);
        const float p_dropout = rpmath::dropoutProbability(
            d, retro_val, dropout_rate_close, dropout_rate_far, max_range);
        if (curand_uniform(rs) < p_dropout) {
            range_out[idx]  = 0u;
            signal_out[idx] = 0u;
            refl_out[idx]   = static_cast<uint8_t>(base_reflectivity);
            nearir_out[idx] = 0u;
            return;
        }
    }

    // ── Range noise (Gaussian, σ scales with range and reflectivity) ─────────
    // Lower reflectivity targets produce noisier range measurements.
    if (rs != nullptr && (range_noise_min_std > 0.f || range_noise_max_std > 0.f)) {
        const float retro_val = rpmath::retroForNoise(retro, idx);
        const float sigma = rpmath::rangeNoiseSigma(
            d, retro_val, range_noise_min_std, range_noise_max_std, max_range);
        d = fmaxf(d + curand_normal(rs) * sigma, 0.f);
    }

    // Range in millimetres
    range_out[idx] = static_cast<uint32_t>(d * rpmath::kRangeToMm);

    // ── Signal: 1/r² model with Poisson shot noise ──────────────────────────
    float intensity = 1.0f;
    if (retro != nullptr) {
        float r = retro[idx];
        if (isfinite(r) && r > 0.f) {
            intensity = r;
        }
    }
    float sig = rpmath::signalFromRange(d, intensity, base_signal);

    // Shot noise: σ = √(signal) × scale
    if (rs != nullptr && signal_noise_scale > 0.f) {
        float sigma_sig = sqrtf(fmaxf(sig, 0.f)) * signal_noise_scale;
        sig = fmaxf(sig + curand_normal(rs) * sigma_sig, 0.f);
    }
    signal_out[idx] = rpmath::clampU16(sig);

    // ── Reflectivity from retro (Ouster calibrated scale) ──────────────────
    // 0-100 = Lambertian (linear), 101-255 = retroreflective (log).
    // See ray_processor_cpu_impl.cpp for the full derivation; the mapping
    // itself is rpmath::reflectivityToByte (shared by all backends).
    if (retro != nullptr && isfinite(retro[idx]) && retro[idx] > 0.f) {
        refl_out[idx] = rpmath::reflectivityToByte(retro[idx]);
    } else {
        refl_out[idx] = static_cast<uint8_t>(base_reflectivity);
    }

    // ── Near-IR: retro → uint16 photon-count analogue with Poisson noise ────
    float nir = (retro != nullptr && isfinite(retro[idx]) && retro[idx] > 0.f)
        ? retro[idx] * rpmath::kNearIrScale : 0.f;
    if (rs != nullptr && nearir_noise_scale > 0.f && nir > 0.f) {
        float sigma_nir = sqrtf(fmaxf(nir, 0.f)) * nearir_noise_scale;
        nir = fmaxf(nir + curand_normal(rs) * sigma_nir, 0.f);
    }
    nearir_out[idx] = rpmath::clampU16(nir);
}

// ── Launcher wrappers (called from .cpp via .cuh) ────────────────────────────

void launchRayProcessKernel(
    const float * d_depth,
    const float * d_retro,
    uint32_t *    d_range,
    uint16_t *    d_signal,
    uint8_t *     d_refl,
    uint16_t *    d_nearir,
    const RayProcessParams & p,
    void *        d_rand_states,
    void *        stream)
{
    const int n = p.H * p.W;
    const int grid = (n + kBlock - 1) / kBlock;
    rayProcessKernel<<<grid, kBlock, 0, static_cast<cudaStream_t>(stream)>>>(
        d_depth, d_retro, d_range, d_signal, d_refl, d_nearir,
        p.H, p.W,
        p.base_signal, p.base_reflectivity,
        p.range_noise_min_std, p.range_noise_max_std, p.max_range,
        p.signal_noise_scale, p.nearir_noise_scale,
        p.dropout_rate_close, p.dropout_rate_far,
        p.edge_discon_threshold,
        static_cast<curandState *>(d_rand_states));
    CUDA_CHECK(cudaGetLastError());
}

void launchInitRandKernel(
    void * d_states, unsigned long seed, int n, void * stream)
{
    const int grid = (n + kBlock - 1) / kBlock;
    initRandStates<<<grid, kBlock, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<curandState *>(d_states), seed, n);
    CUDA_CHECK(cudaGetLastError());
}

// ── CudaBackend ─────────────────────────────────────────────────────────────

namespace {

class CudaBackend final : public Backend {
public:
    CudaBackend(uint64_t seed, cudaStream_t stream) : seed_(seed), stream_(stream) {}

    ~CudaBackend() override
    {
        if (d_depth_)       cudaFree(d_depth_);
        if (d_retro_)       cudaFree(d_retro_);
        if (d_range_)       cudaFree(d_range_);
        if (d_signal_)      cudaFree(d_signal_);
        if (d_refl_)        cudaFree(d_refl_);
        if (d_nearir_)      cudaFree(d_nearir_);
        if (d_raw_frame_)   cudaFree(d_raw_frame_);
        if (d_beam_alt_)    cudaFree(d_beam_alt_);
        if (d_beam_az_)     cudaFree(d_beam_az_);
        if (d_rand_states_) cudaFree(d_rand_states_);
        if (stream_)        cudaStreamDestroy(stream_);
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

        CUDA_CHECK(cudaMemcpyAsync(d_raw_frame_, raw_host,
            static_cast<size_t>(raw_n) * sizeof(float),
            cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(d_beam_alt_, beam_alt_host,
            static_cast<size_t>(rp.H) * sizeof(float),
            cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(d_beam_az_, beam_az_host,
            static_cast<size_t>(rp.H) * sizeof(float),
            cudaMemcpyHostToDevice, stream_));

        launchResampleKernel(
            static_cast<const float *>(d_raw_frame_),
            static_cast<const float *>(d_beam_alt_),
            static_cast<const float *>(d_beam_az_),
            static_cast<float *>(d_depth_),
            static_cast<float *>(d_retro_),
            rp, stream_);

        launchRayProcessKernel(
            static_cast<const float *>(d_depth_),
            static_cast<const float *>(d_retro_),
            static_cast<uint32_t *>(d_range_),
            static_cast<uint16_t *>(d_signal_),
            static_cast<uint8_t *>(d_refl_),
            static_cast<uint16_t *>(d_nearir_),
            pp,
            need_rand ? d_rand_states_ : nullptr,
            stream_);

        d2hResults(range_out, signal_out, reflectivity_out, nearir_out, out_n);
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    const char * name() const override { return "cuda"; }

private:
    static void realloc_(void * & ptr, size_t bytes)
    {
        if (ptr) { CUDA_CHECK(cudaFree(ptr)); ptr = nullptr; }
        CUDA_CHECK(cudaMalloc(&ptr, bytes));
    }

    void ensureBuffers(int n)
    {
        if (n <= buf_n_) return;
        realloc_(d_depth_,  static_cast<size_t>(n) * sizeof(float));
        realloc_(d_retro_,  static_cast<size_t>(n) * sizeof(float));
        realloc_(d_range_,  static_cast<size_t>(n) * sizeof(uint32_t));
        realloc_(d_signal_, static_cast<size_t>(n) * sizeof(uint16_t));
        realloc_(d_refl_,   static_cast<size_t>(n) * sizeof(uint8_t));
        realloc_(d_nearir_, static_cast<size_t>(n) * sizeof(uint16_t));
        buf_n_ = n;
    }

    void ensureResampleBuffers(int raw_n, int H)
    {
        if (raw_n > raw_buf_n_) {
            realloc_(d_raw_frame_, static_cast<size_t>(raw_n) * sizeof(float));
            raw_buf_n_ = raw_n;
        }
        if (H > beam_buf_n_) {
            realloc_(d_beam_alt_, static_cast<size_t>(H) * sizeof(float));
            realloc_(d_beam_az_,  static_cast<size_t>(H) * sizeof(float));
            beam_buf_n_ = H;
        }
    }

    void ensureRandStates(int n)
    {
        if (n <= rand_n_) return;
        if (d_rand_states_) { CUDA_CHECK(cudaFree(d_rand_states_)); d_rand_states_ = nullptr; }
        CUDA_CHECK(cudaMalloc(&d_rand_states_,
            static_cast<size_t>(n) * sizeof(curandState)));

        // seed_ == 0 means non-deterministic (production). Any non-zero seed
        // configured at construction time makes the noise reproducible.
        // For non-deterministic, mix steady_clock + pid + this-pointer so two
        // sensors constructed in the same tick get independent noise.
        const unsigned long rng_seed = (seed_ != 0)
            ? static_cast<unsigned long>(seed_)
            : static_cast<unsigned long>(deriveNonDeterministicSeed(this));
        launchInitRandKernel(d_rand_states_, rng_seed, n, stream_);
        rand_n_ = n;
    }

    void d2hResults(
        uint32_t * range_out,
        uint16_t * signal_out,
        uint8_t *  reflectivity_out,
        uint16_t * nearir_out,
        int n)
    {
        CUDA_CHECK(cudaMemcpyAsync(range_out, d_range_,
            static_cast<size_t>(n) * sizeof(uint32_t),
            cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaMemcpyAsync(signal_out, d_signal_,
            static_cast<size_t>(n) * sizeof(uint16_t),
            cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaMemcpyAsync(reflectivity_out, d_refl_,
            static_cast<size_t>(n) * sizeof(uint8_t),
            cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaMemcpyAsync(nearir_out, d_nearir_,
            static_cast<size_t>(n) * sizeof(uint16_t),
            cudaMemcpyDeviceToHost, stream_));
    }

    uint64_t seed_ = 0;
    cudaStream_t stream_ = nullptr;

    // Device buffers — noise processing
    void * d_depth_   = nullptr;
    void * d_retro_   = nullptr;
    void * d_range_   = nullptr;
    void * d_signal_  = nullptr;
    void * d_refl_    = nullptr;
    void * d_nearir_  = nullptr;
    void * d_rand_states_ = nullptr;

    // Device buffers — resampling
    void * d_raw_frame_ = nullptr;
    void * d_beam_alt_  = nullptr;
    void * d_beam_az_   = nullptr;

    int buf_n_      = 0;
    int rand_n_     = 0;
    int raw_buf_n_  = 0;
    int beam_buf_n_ = 0;
};

}  // namespace

// ── Factory ─────────────────────────────────────────────────────────────────
// Returns nullptr when no CUDA device is present or stream creation fails —
// e.g. a binary built with CUDA but running on a container without GPU
// passthrough. The dispatcher then tries the next backend (HIP / SYCL / CPU).

std::unique_ptr<Backend> makeCudaBackend(uint64_t seed)
{
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        cudaGetLastError();  // clear sticky error so later CUDA calls aren't poisoned
        return nullptr;
    }

    cudaStream_t s = nullptr;
    err = cudaStreamCreate(&s);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    return std::make_unique<CudaBackend>(seed, s);
}

}  // namespace gz_gpu_ouster_lidar
