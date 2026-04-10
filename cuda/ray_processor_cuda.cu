#include "ray_processor_cuda.cuh"
#include "gz_gpu_ouster_lidar/cuda_ray_processor.hpp"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdint>
#include <ctime>
#include <cstdio>
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

/// Process depth + retro → range_mm, signal, reflectivity.
///
/// Each thread handles one pixel (beam × column).
/// Input layout: depth_buf[beam * W + col] = depth in metres (inf = miss).
/// Output: range in mm, signal via 1/r² model, reflectivity from retro.
__global__ void rayProcessKernel(
    const float * __restrict__   depth,
    const float * __restrict__   retro,   // may be nullptr
    uint32_t * __restrict__      range_out,
    uint16_t * __restrict__      signal_out,
    uint8_t *  __restrict__      refl_out,
    int n,
    float base_signal,
    float base_reflectivity,
    float noise_std,
    curandState * __restrict__   rand_states)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float d = depth[idx];
    const bool valid = isfinite(d) && d > 0.001f;

    if (!valid) {
        range_out[idx]  = 0u;
        signal_out[idx] = 0u;
        refl_out[idx]   = static_cast<uint8_t>(base_reflectivity);
        return;
    }

    // Optional Gaussian range noise
    if (noise_std > 0.f && rand_states != nullptr) {
        d = fmaxf(d + curand_normal(&rand_states[idx]) * noise_std, 0.f);
    }

    // Range in millimetres
    range_out[idx] = static_cast<uint32_t>(d * 1000.f);

    // Signal: 1/r² model scaled by retro intensity
    float intensity = 1.0f;
    if (retro != nullptr) {
        float r = retro[idx];
        if (isfinite(r) && r > 0.f) {
            intensity = r;
        }
    }
    const float r_sq = d * d;
    const float sig = fminf(base_signal * intensity / fmaxf(r_sq, 0.0001f), 65535.f);
    signal_out[idx] = static_cast<uint16_t>(sig);

    // Reflectivity from retro
    if (retro != nullptr && isfinite(retro[idx]) && retro[idx] > 0.f) {
        refl_out[idx] = static_cast<uint8_t>(fminf(retro[idx] * 1000.f, 255.f));
    } else {
        refl_out[idx] = static_cast<uint8_t>(base_reflectivity);
    }
}

// ── Launcher wrappers (called from .cpp via .cuh) ────────────────────────────

void launchRayProcessKernel(
    const float * d_depth,
    const float * d_retro,
    uint32_t *    d_range,
    uint16_t *    d_signal,
    uint8_t *     d_refl,
    int n,
    float base_signal,
    float base_reflectivity,
    float range_noise_std,
    void *        d_rand_states,
    void *        stream)
{
    const int grid = (n + kBlock - 1) / kBlock;
    rayProcessKernel<<<grid, kBlock, 0, static_cast<cudaStream_t>(stream)>>>(
        d_depth, d_retro, d_range, d_signal, d_refl,
        n, base_signal, base_reflectivity, range_noise_std,
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

// ── CudaRayProcessor ────────────────────────────────────────────────────────

CudaRayProcessor::CudaRayProcessor()
{
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    stream_ = static_cast<void *>(s);
}

CudaRayProcessor::~CudaRayProcessor()
{
    if (d_depth_)       cudaFree(d_depth_);
    if (d_retro_)       cudaFree(d_retro_);
    if (d_range_)       cudaFree(d_range_);
    if (d_signal_)      cudaFree(d_signal_);
    if (d_refl_)        cudaFree(d_refl_);
    if (d_rand_states_) cudaFree(d_rand_states_);
    if (stream_)        cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
}

void CudaRayProcessor::ensureBuffers(int n)
{
    if (n <= buf_n_) return;

    auto realloc = [&](void * & ptr, size_t bytes) {
        if (ptr) { CUDA_CHECK(cudaFree(ptr)); ptr = nullptr; }
        CUDA_CHECK(cudaMalloc(&ptr, bytes));
    };

    realloc(d_depth_,  static_cast<size_t>(n) * sizeof(float));
    realloc(d_retro_,  static_cast<size_t>(n) * sizeof(float));
    realloc(d_range_,  static_cast<size_t>(n) * sizeof(uint32_t));
    realloc(d_signal_, static_cast<size_t>(n) * sizeof(uint16_t));
    realloc(d_refl_,   static_cast<size_t>(n) * sizeof(uint8_t));

    buf_n_ = n;
}

void CudaRayProcessor::ensureRandStates(int n)
{
    if (n <= rand_n_) return;

    if (d_rand_states_) { CUDA_CHECK(cudaFree(d_rand_states_)); d_rand_states_ = nullptr; }
    CUDA_CHECK(cudaMalloc(&d_rand_states_, static_cast<size_t>(n) * sizeof(curandState)));

    launchInitRandKernel(
        d_rand_states_,
        static_cast<unsigned long>(clock()),
        n, stream_);

    rand_n_ = n;
}

void CudaRayProcessor::process(
    const float * depth_host,
    const float * retro_host,
    uint32_t *    range_out,
    uint16_t *    signal_out,
    uint8_t *     reflectivity_out,
    const RayProcessParams & p)
{
    const int n = p.H * p.W;
    auto stream = static_cast<cudaStream_t>(stream_);

    ensureBuffers(n);
    if (p.range_noise_std > 0.f) ensureRandStates(n);

    // H2D: depth (always) and retro (if available)
    CUDA_CHECK(cudaMemcpyAsync(
        d_depth_, depth_host,
        static_cast<size_t>(n) * sizeof(float),
        cudaMemcpyHostToDevice, stream));

    if (retro_host) {
        CUDA_CHECK(cudaMemcpyAsync(
            d_retro_, retro_host,
            static_cast<size_t>(n) * sizeof(float),
            cudaMemcpyHostToDevice, stream));
    }

    // Launch kernel
    launchRayProcessKernel(
        static_cast<const float *>(d_depth_),
        retro_host ? static_cast<const float *>(d_retro_) : nullptr,
        static_cast<uint32_t *>(d_range_),
        static_cast<uint16_t *>(d_signal_),
        static_cast<uint8_t *>(d_refl_),
        n, p.base_signal, p.base_reflectivity, p.range_noise_std,
        (p.range_noise_std > 0.f) ? d_rand_states_ : nullptr,
        stream_);

    // D2H: results
    CUDA_CHECK(cudaMemcpyAsync(
        range_out, d_range_,
        static_cast<size_t>(n) * sizeof(uint32_t),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        signal_out, d_signal_,
        static_cast<size_t>(n) * sizeof(uint16_t),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        reflectivity_out, d_refl_,
        static_cast<size_t>(n) * sizeof(uint8_t),
        cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace gz_gpu_ouster_lidar
