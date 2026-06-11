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
#include "raycast_math.hpp"

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

/// Resample the perspective panel rig → exact Ouster beam geometry.
/// One thread per output pixel (beam × measurement id). Computes the exact
/// beam direction (cylindrical/hemispherical model with per-beam azimuth
/// offsets), projects into the covering panel and bilinearly samples planar
/// depth (shared rpmath::sampleBeamRange). `rp` is passed by value: ~0.8 KB,
/// well inside the 4 KB kernel-parameter limit.
__global__ void resampleKernel(
    const float * __restrict__ raw,          // packed panel depth images
    const float * __restrict__ beam_alt,     // H beam altitude angles (degrees)
    const float * __restrict__ beam_az,      // H beam azimuth offsets (degrees)
    float * __restrict__       depth_out,    // H × W
    ResampleParams rp)
{
    const int n = rp.H * rp.W;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    const int beam = tid / rp.W;
    const int m    = tid % rp.W;

    const float el = beam_alt[beam];
    const float az = rpmath::beamRayAzimuthDeg(
        beam_az[beam], m, 360.0f / static_cast<float>(rp.W));

    float depth = rpmath::sampleBeamRange(raw, rp, el, az, CUDART_INF_F);
    depth = rpmath::applyBeamOrigin(depth, el, rp.beam_origin_m);
    depth_out[tid] = depth;
}

// ── Resample launcher ────────────────────────────────────────────────────────

void launchResampleKernel(
    const float * d_raw_frame,
    const float * d_beam_alt,
    const float * d_beam_az,
    float *       d_depth_out,
    const ResampleParams & rp,
    void * stream)
{
    const int n = rp.H * rp.W;
    const int grid = (n + kBlock - 1) / kBlock;
    resampleKernel<<<grid, kBlock, 0, static_cast<cudaStream_t>(stream)>>>(
        d_raw_frame, d_beam_alt, d_beam_az, d_depth_out, rp);
    CUDA_CHECK(cudaGetLastError());
}

/// Per-thread arguments for the raycast kernel that don't vary per ray.
/// Passed by value (~70 B) like ResampleParams.
struct RcCastArgs {
    float sr[9];
    float st[3];
    rc::ScanParams sp;
    int n_instances;
};

/// Full per-beam raycast: one thread per output pixel, executing the SAME
/// rcCastOneRay the CPU fallback runs (shared raycast_math.hpp).
__global__ void castScanKernel(
    const rc::RcInstance * __restrict__ instances,
    const float * __restrict__ verts,
    const int * __restrict__ tris,
    const int * __restrict__ order,
    const rc::MeshBvhNode * __restrict__ nodes,
    const rc::InstanceXform * __restrict__ xforms,
    const float * __restrict__ beam_alt,
    const float * __restrict__ beam_az,
    // cppcheck-suppress passedByValueCallback ; GPU kernel arguments must be
    // passed by value — the device cannot dereference a host reference.
    RcCastArgs args,
    float * __restrict__ range_out,
    float * __restrict__ retro_out)
{
    const int n = args.sp.H * args.sp.W;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float range, retro;
    rc::rcCastOneRay(instances, args.n_instances, verts, tris, order, nodes,
                     xforms, beam_alt, beam_az, args.sr, args.st, args.sp,
                     idx, CUDART_INF_F, range, retro);
    range_out[idx] = range;
    retro_out[idx] = retro;
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
        if (d_rc_insts_)    cudaFree(d_rc_insts_);
        if (d_rc_verts_)    cudaFree(d_rc_verts_);
        if (d_rc_tris_)     cudaFree(d_rc_tris_);
        if (d_rc_order_)    cudaFree(d_rc_order_);
        if (d_rc_nodes_)    cudaFree(d_rc_nodes_);
        if (d_rc_xforms_)   cudaFree(d_rc_xforms_);
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
        const int raw_n = rp.raw_n;
        const int out_n = rp.H * rp.W;

        ensureBuffers(out_n);
        ensureResampleBuffers(raw_n, rp.H);

        const bool need_rand = noiseEnabled(pp);
        if (need_rand) ensureRandStates(out_n);

        CUDA_CHECK(cudaMemcpyAsync(d_raw_frame_, raw_host,
            static_cast<size_t>(raw_n) * sizeof(float),
            cudaMemcpyHostToDevice, stream_));
        uploadBeamTables(beam_alt_host, beam_az_host, rp.H);

        launchResampleKernel(
            static_cast<const float *>(d_raw_frame_),
            static_cast<const float *>(d_beam_alt_),
            static_cast<const float *>(d_beam_az_),
            static_cast<float *>(d_depth_),
            rp, stream_);

        // The panel rig carries no laser_retro channel — pass null retro so
        // the kernel uses base_reflectivity / unit intensity.
        launchRayProcessKernel(
            static_cast<const float *>(d_depth_),
            nullptr,
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

    void processDepth(
        const float * depth_host,
        const float * retro_host,
        uint32_t *    range_out,
        uint16_t *    signal_out,
        uint8_t *     reflectivity_out,
        uint16_t *    nearir_out,
        const RayProcessParams & pp) override
    {
        const int out_n = pp.H * pp.W;
        ensureBuffers(out_n);

        const bool need_rand = noiseEnabled(pp);
        if (need_rand) ensureRandStates(out_n);

        CUDA_CHECK(cudaMemcpyAsync(d_depth_, depth_host,
            static_cast<size_t>(out_n) * sizeof(float),
            cudaMemcpyHostToDevice, stream_));
        if (retro_host) {
            ensureRetroBuffer(out_n);
            CUDA_CHECK(cudaMemcpyAsync(d_retro_, retro_host,
                static_cast<size_t>(out_n) * sizeof(float),
                cudaMemcpyHostToDevice, stream_));
        }

        launchRayProcessKernel(
            static_cast<const float *>(d_depth_),
            retro_host ? static_cast<const float *>(d_retro_) : nullptr,
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

    void castScan(
        const rc::SceneView & scene,
        uint64_t scene_version,
        const rc::InstanceXform * xforms,
        const float * beam_alt_deg,
        const float * beam_az_deg,
        const float sensor_r[9],
        const float sensor_t[3],
        const rc::ScanParams & sp,
        float * range_out,
        float * retro_out) override
    {
        const int out_n = sp.H * sp.W;
        ensureBuffers(out_n);
        ensureRetroBuffer(out_n);
        ensureSceneBuffers(scene, scene_version);
        ensureResampleBuffers(0, sp.H);  // beam-angle buffers only

        if (scene.n_instances > 0) {
            CUDA_CHECK(cudaMemcpyAsync(d_rc_xforms_, xforms,
                static_cast<size_t>(scene.n_instances) *
                    sizeof(rc::InstanceXform),
                cudaMemcpyHostToDevice, stream_));
        }
        uploadBeamTables(beam_alt_deg, beam_az_deg, sp.H);

        RcCastArgs args;
        for (int i = 0; i < 9; ++i) args.sr[i] = sensor_r[i];
        for (int i = 0; i < 3; ++i) args.st[i] = sensor_t[i];
        args.sp = sp;
        args.n_instances = scene.n_instances;

        const int grid = (out_n + kBlock - 1) / kBlock;
        castScanKernel<<<grid, kBlock, 0, stream_>>>(
            static_cast<const rc::RcInstance *>(d_rc_insts_),
            static_cast<const float *>(d_rc_verts_),
            static_cast<const int *>(d_rc_tris_),
            static_cast<const int *>(d_rc_order_),
            static_cast<const rc::MeshBvhNode *>(d_rc_nodes_),
            static_cast<const rc::InstanceXform *>(d_rc_xforms_),
            static_cast<const float *>(d_beam_alt_),
            static_cast<const float *>(d_beam_az_),
            args,
            static_cast<float *>(d_depth_),
            static_cast<float *>(d_retro_));
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(range_out, d_depth_,
            static_cast<size_t>(out_n) * sizeof(float),
            cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaMemcpyAsync(retro_out, d_retro_,
            static_cast<size_t>(out_n) * sizeof(float),
            cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    const char * name() const override { return "cuda"; }

private:
    /// Upload the flat scene arrays, cached by the caller's version id —
    /// geometry is immutable between rebuilds, so steady state uploads
    /// only the per-scan transforms.
    void ensureSceneBuffers(const rc::SceneView & sv, uint64_t version)
    {
        if (version == rc_scene_version_ && rc_scene_version_valid_) return;

        auto upload = [this](void * & ptr, int & cap, const void * src,
                             size_t bytes) {
            if (bytes == 0) return;
            if (static_cast<size_t>(cap) < bytes) {
                realloc_(ptr, bytes);
                cap = static_cast<int>(bytes);
            }
            CUDA_CHECK(cudaMemcpyAsync(ptr, src, bytes,
                cudaMemcpyHostToDevice, stream_));
        };
        upload(d_rc_insts_, rc_insts_cap_, sv.instances,
               static_cast<size_t>(sv.n_instances) * sizeof(rc::RcInstance));
        upload(d_rc_verts_, rc_verts_cap_, sv.verts,
               static_cast<size_t>(sv.n_vert_floats) * sizeof(float));
        upload(d_rc_tris_, rc_tris_cap_, sv.tris,
               static_cast<size_t>(sv.n_tri_ints) * sizeof(int));
        upload(d_rc_order_, rc_order_cap_, sv.order,
               static_cast<size_t>(sv.n_order) * sizeof(int));
        upload(d_rc_nodes_, rc_nodes_cap_, sv.nodes,
               static_cast<size_t>(sv.n_nodes) * sizeof(rc::MeshBvhNode));
        if (rc_xforms_cap_ < sv.n_instances) {
            realloc_(d_rc_xforms_,
                static_cast<size_t>(sv.n_instances) *
                    sizeof(rc::InstanceXform));
            rc_xforms_cap_ = sv.n_instances;
        }
        rc_scene_version_ = version;
        rc_scene_version_valid_ = true;
    }

    static void realloc_(void * & ptr, size_t bytes)
    {
        if (ptr) { CUDA_CHECK(cudaFree(ptr)); ptr = nullptr; }
        CUDA_CHECK(cudaMalloc(&ptr, bytes));
    }

    void ensureBuffers(int n)
    {
        if (n <= buf_n_) return;
        realloc_(d_depth_,  static_cast<size_t>(n) * sizeof(float));
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
            beam_src_alt_ = nullptr;  // device buffers changed: re-upload
        }
    }

    // Beam calibration tables are constant for a sensor's lifetime (derived
    // from the metadata JSON at construction), so cache them on the device
    // keyed by source pointer + count instead of re-copying every frame —
    // transfer minimisation per docs/SYSTEMS_REFERENCES.md. Assumes the
    // caller's arrays are immutable while the backend lives, which holds for
    // the plugin's metadata-derived tables.
    void uploadBeamTables(const float * alt, const float * az, int H)
    {
        if (alt == beam_src_alt_ && az == beam_src_az_ && H == beam_src_h_) {
            return;
        }
        CUDA_CHECK(cudaMemcpyAsync(d_beam_alt_, alt,
            static_cast<size_t>(H) * sizeof(float),
            cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(d_beam_az_, az,
            static_cast<size_t>(H) * sizeof(float),
            cudaMemcpyHostToDevice, stream_));
        beam_src_alt_ = alt;
        beam_src_az_ = az;
        beam_src_h_ = H;
    }

    // Retro device buffer is only used by processDepth (the raycast mode
    // restores laser_retro); allocated lazily so the panels path pays
    // nothing for it.
    void ensureRetroBuffer(int n)
    {
        if (n <= retro_n_) return;
        realloc_(d_retro_, static_cast<size_t>(n) * sizeof(float));
        retro_n_ = n;
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
    void * d_retro_   = nullptr;   // processDepth only (lazy)
    void * d_range_   = nullptr;
    void * d_signal_  = nullptr;
    void * d_refl_    = nullptr;
    void * d_nearir_  = nullptr;
    void * d_rand_states_ = nullptr;

    // Device buffers — resampling
    void * d_raw_frame_ = nullptr;
    void * d_beam_alt_  = nullptr;
    void * d_beam_az_   = nullptr;
    // uploadBeamTables cache identity (host source pointers + count).
    const float * beam_src_alt_ = nullptr;
    const float * beam_src_az_  = nullptr;
    int beam_src_h_ = 0;

    // Device buffers — raycast scene (cached by scene_version)
    void * d_rc_insts_  = nullptr;
    void * d_rc_verts_  = nullptr;
    void * d_rc_tris_   = nullptr;
    void * d_rc_order_  = nullptr;
    void * d_rc_nodes_  = nullptr;
    void * d_rc_xforms_ = nullptr;
    int rc_insts_cap_ = 0;
    int rc_verts_cap_ = 0;
    int rc_tris_cap_ = 0;
    int rc_order_cap_ = 0;
    int rc_nodes_cap_ = 0;
    int rc_xforms_cap_ = 0;
    uint64_t rc_scene_version_ = 0;
    bool rc_scene_version_valid_ = false;

    int buf_n_      = 0;
    int retro_n_    = 0;
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
