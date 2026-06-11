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
#include "ray_processor_math.hpp"
#include "raycast_math.hpp"

#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
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
    const float * __restrict__ raw,          // packed panel depth images
    const float * __restrict__ beam_alt,
    const float * __restrict__ beam_az,
    float * __restrict__       depth_out,
    // cppcheck-suppress passedByValueCallback ; GPU kernel arguments must be
    // passed by value — the device cannot dereference a host reference. (~0.8
    // KB, well inside the 4 KB kernel-parameter limit.)
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

    float depth = rpmath::sampleBeamRange(
        raw, rp, el, az, __int_as_float(0x7f800000));  // miss → +inf
    depth = rpmath::applyBeamOrigin(depth, el, rp.beam_origin_m);
    depth_out[tid] = depth;
}

/// Per-thread arguments for the raycast kernel (mirror of CUDA).
struct RcCastArgsHip {
    float sr[9];
    float st[3];
    rc::ScanParams sp;
    int n_instances;
};

__global__ void castScanKernelHip(
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
    RcCastArgsHip args,
    float * __restrict__ range_out,
    float * __restrict__ retro_out)
{
    const int n = args.sp.H * args.sp.W;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float range, retro;
    rc::rcCastOneRay(instances, args.n_instances, verts, tris, order, nodes,
                     xforms, beam_alt, beam_az, args.sr, args.st, args.sp,
                     idx, __int_as_float(0x7f800000), range, retro);
    range_out[idx] = range;
    retro_out[idx] = retro;
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
    float false_alarm_rate,
    float edge_discon_threshold,
    hiprandState * __restrict__ rand_states)
{
    const int n = H * W;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float d = depth[idx];
    const bool valid = isfinite(d) && d > rpmath::kValidDepthMin;

    if (!valid) {
        // Solar-background false alarm (see the CPU reference in
        // ray_processor_cpu_impl.cpp for the rationale).
        if (false_alarm_rate > 0.f && rand_states != nullptr &&
            hiprand_uniform(&rand_states[idx]) < false_alarm_rate) {
            const float d_fa =
                hiprand_uniform(&rand_states[idx]) * max_range;
            range_out[idx]  =
                static_cast<uint32_t>(d_fa * rpmath::kRangeToMm);
            signal_out[idx] = 1u;
            refl_out[idx]   = static_cast<uint8_t>(base_reflectivity);
            nearir_out[idx] = 0u;
            return;
        }
        range_out[idx]  = 0u;
        signal_out[idx] = 0u;
        refl_out[idx]   = static_cast<uint8_t>(base_reflectivity);
        nearir_out[idx] = 0u;
        return;
    }

    hiprandState * rs = (rand_states != nullptr) ? &rand_states[idx] : nullptr;

    if (edge_discon_threshold > 0.f && rs != nullptr &&
        rpmath::edgeDiscontinuity(depth, idx, H, W, edge_discon_threshold) &&
        hiprand_uniform(rs) < rpmath::kEdgeSuppressProb) {
        range_out[idx]  = 0u;
        signal_out[idx] = 0u;
        refl_out[idx]   = static_cast<uint8_t>(base_reflectivity);
        nearir_out[idx] = 0u;
        return;
    }

    if (rs != nullptr && (dropout_rate_close > 0.f || dropout_rate_far > 0.f)) {
        const float retro_val = rpmath::retroForNoise(retro, idx);
        const float p_dropout = rpmath::dropoutProbability(
            d, retro_val, dropout_rate_close, dropout_rate_far, max_range);
        if (hiprand_uniform(rs) < p_dropout) {
            range_out[idx]  = 0u;
            signal_out[idx] = 0u;
            refl_out[idx]   = static_cast<uint8_t>(base_reflectivity);
            nearir_out[idx] = 0u;
            return;
        }
    }

    if (rs != nullptr && (range_noise_min_std > 0.f || range_noise_max_std > 0.f)) {
        const float retro_val = rpmath::retroForNoise(retro, idx);
        const float sigma = rpmath::rangeNoiseSigma(
            d, retro_val, range_noise_min_std, range_noise_max_std, max_range);
        d = fmaxf(d + hiprand_normal(rs) * sigma, 0.f);
    }

    range_out[idx] = static_cast<uint32_t>(d * rpmath::kRangeToMm);

    float intensity = 1.0f;
    if (retro != nullptr) {
        float r = retro[idx];
        if (isfinite(r) && r > 0.f) intensity = r;
    }
    float sig = rpmath::signalFromRange(d, intensity, base_signal);
    if (rs != nullptr && signal_noise_scale > 0.f) {
        float sigma_sig = sqrtf(fmaxf(sig, 0.f)) * signal_noise_scale;
        sig = fmaxf(sig + hiprand_normal(rs) * sigma_sig, 0.f);
    }
    signal_out[idx] = rpmath::clampU16(sig);

    // Ouster reflectivity scale (shared rpmath::reflectivityToByte); canonical
    // derivation + upstream refs live in ray_processor_cpu_impl.cpp.
    if (retro != nullptr && isfinite(retro[idx]) && retro[idx] > 0.f) {
        refl_out[idx] = rpmath::reflectivityToByte(retro[idx]);
    } else {
        refl_out[idx] = static_cast<uint8_t>(base_reflectivity);
    }

    float nir = (retro != nullptr && isfinite(retro[idx]) && retro[idx] > 0.f)
        ? retro[idx] * rpmath::kNearIrScale : 0.f;
    if (rs != nullptr && nearir_noise_scale > 0.f && nir > 0.f) {
        float sigma_nir = sqrtf(fmaxf(nir, 0.f)) * nearir_noise_scale;
        nir = fmaxf(nir + hiprand_normal(rs) * sigma_nir, 0.f);
    }
    nearir_out[idx] = rpmath::clampU16(nir);
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
        maybeFree(d_rc_insts_);
        maybeFree(d_rc_verts_);
        maybeFree(d_rc_tris_);
        maybeFree(d_rc_order_);
        maybeFree(d_rc_nodes_);
        maybeFree(d_rc_xforms_);
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
        const int raw_n = rp.raw_n;
        const int out_n = rp.H * rp.W;

        ensureBuffers(out_n);
        ensureResampleBuffers(raw_n, rp.H);

        const bool need_rand = noiseEnabled(pp);
        if (need_rand) ensureRandStates(out_n);

        h2d(d_raw_frame_, raw_host, static_cast<size_t>(raw_n) * sizeof(float));
        uploadBeamTables(beam_alt_host, beam_az_host, rp.H);

        const int grid_r = (out_n + kBlock - 1) / kBlock;
        hipLaunchKernelGGL(resampleKernelHip, dim3(grid_r), dim3(kBlock), 0, stream_,
            static_cast<const float *>(d_raw_frame_),
            static_cast<const float *>(d_beam_alt_),
            static_cast<const float *>(d_beam_az_),
            static_cast<float *>(d_depth_),
            rp);
        HIP_CHECK(hipGetLastError());

        // No laser_retro channel from the panel rig — null retro selects
        // base_reflectivity / unit intensity in the process kernel.
        hipLaunchKernelGGL(rayProcessKernelHip, dim3(grid_r), dim3(kBlock), 0, stream_,
            static_cast<const float *>(d_depth_),
            static_cast<const float *>(nullptr),
            static_cast<uint32_t *>(d_range_),
            static_cast<uint16_t *>(d_signal_),
            static_cast<uint8_t *>(d_refl_),
            static_cast<uint16_t *>(d_nearir_),
            pp.H, pp.W,
            pp.base_signal, pp.base_reflectivity,
            pp.range_noise_min_std, pp.range_noise_max_std, pp.max_range,
            pp.signal_noise_scale, pp.nearir_noise_scale,
            pp.dropout_rate_close, pp.dropout_rate_far,
            pp.false_alarm_rate,
            pp.edge_discon_threshold,
            need_rand ? static_cast<hiprandState *>(d_rand_states_) : nullptr);
        HIP_CHECK(hipGetLastError());

        d2hResults(range_out, signal_out, reflectivity_out, nearir_out, out_n);
        HIP_CHECK(hipStreamSynchronize(stream_));
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

        h2d(d_depth_, depth_host, static_cast<size_t>(out_n) * sizeof(float));
        if (retro_host) {
            ensureRetroBuffer(out_n);
            h2d(d_retro_, retro_host,
                static_cast<size_t>(out_n) * sizeof(float));
        }

        const int grid = (out_n + kBlock - 1) / kBlock;
        hipLaunchKernelGGL(rayProcessKernelHip, dim3(grid), dim3(kBlock), 0, stream_,
            static_cast<const float *>(d_depth_),
            static_cast<const float *>(retro_host ? d_retro_ : nullptr),
            static_cast<uint32_t *>(d_range_),
            static_cast<uint16_t *>(d_signal_),
            static_cast<uint8_t *>(d_refl_),
            static_cast<uint16_t *>(d_nearir_),
            pp.H, pp.W,
            pp.base_signal, pp.base_reflectivity,
            pp.range_noise_min_std, pp.range_noise_max_std, pp.max_range,
            pp.signal_noise_scale, pp.nearir_noise_scale,
            pp.dropout_rate_close, pp.dropout_rate_far,
            pp.false_alarm_rate,
            pp.edge_discon_threshold,
            need_rand ? static_cast<hiprandState *>(d_rand_states_) : nullptr);
        HIP_CHECK(hipGetLastError());

        d2hResults(range_out, signal_out, reflectivity_out, nearir_out, out_n);
        HIP_CHECK(hipStreamSynchronize(stream_));
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
            h2d(d_rc_xforms_, xforms,
                static_cast<size_t>(scene.n_instances) *
                    sizeof(rc::InstanceXform));
        }
        uploadBeamTables(beam_alt_deg, beam_az_deg, sp.H);

        RcCastArgsHip args;
        for (int i = 0; i < 9; ++i) args.sr[i] = sensor_r[i];
        for (int i = 0; i < 3; ++i) args.st[i] = sensor_t[i];
        args.sp = sp;
        args.n_instances = scene.n_instances;

        const int grid = (out_n + kBlock - 1) / kBlock;
        hipLaunchKernelGGL(castScanKernelHip, dim3(grid), dim3(kBlock), 0, stream_,
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
        HIP_CHECK(hipGetLastError());

        d2h(range_out, d_depth_, static_cast<size_t>(out_n) * sizeof(float));
        d2h(retro_out, d_retro_, static_cast<size_t>(out_n) * sizeof(float));
        HIP_CHECK(hipStreamSynchronize(stream_));
    }

    const char * name() const override { return integrated_ ? "hip-apu" : "hip"; }

private:
    /// Upload the flat scene arrays, cached by the caller's version id.
    void ensureSceneBuffers(const rc::SceneView & sv, uint64_t version)
    {
        if (version == rc_scene_version_ && rc_scene_version_valid_) return;

        auto upload = [this](void * & ptr, int & cap, const void * src,
                             size_t bytes) {
            if (bytes == 0) return;
            if (static_cast<size_t>(cap) < bytes) {
                allocDev(ptr, bytes);
                cap = static_cast<int>(bytes);
            }
            h2d(ptr, src, bytes);
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
            allocDev(d_rc_xforms_,
                static_cast<size_t>(sv.n_instances) *
                    sizeof(rc::InstanceXform));
            rc_xforms_cap_ = sv.n_instances;
        }
        rc_scene_version_ = version;
        rc_scene_version_valid_ = true;
    }

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
            beam_src_alt_ = nullptr;  // device buffers changed: re-upload
        }
    }

    // Beam calibration tables are constant for a sensor's lifetime; upload
    // once, keyed by host source pointer + count (see the CUDA twin and
    // docs/SYSTEMS_REFERENCES.md).
    void uploadBeamTables(const float * alt, const float * az, int H)
    {
        if (alt == beam_src_alt_ && az == beam_src_az_ && H == beam_src_h_) {
            return;
        }
        h2d(d_beam_alt_, alt, static_cast<size_t>(H) * sizeof(float));
        h2d(d_beam_az_,  az,  static_cast<size_t>(H) * sizeof(float));
        beam_src_alt_ = alt;
        beam_src_az_ = az;
        beam_src_h_ = H;
    }

    // Retro device buffer is only used by processDepth (raycast mode);
    // allocated lazily so the panels path pays nothing for it.
    void ensureRetroBuffer(int n)
    {
        if (n <= retro_n_) return;
        allocDev(d_retro_, static_cast<size_t>(n) * sizeof(float));
        retro_n_ = n;
    }

    void ensureRandStates(int n)
    {
        if (n <= rand_n_) return;
        if (d_rand_states_) { HIP_CHECK(hipFree(d_rand_states_)); d_rand_states_ = nullptr; }
        HIP_CHECK(hipMalloc(&d_rand_states_,
            static_cast<size_t>(n) * sizeof(hiprandState)));
        // seed_ == 0 means non-deterministic. Mix steady_clock + pid +
        // this-pointer so multiple sensors constructed in the same tick get
        // independent noise (clock() at 10 ms resolution gives identical
        // seeds and perfectly correlated noise across LiDARs).
        const unsigned long rng_seed = (seed_ != 0)
            ? static_cast<unsigned long>(seed_)
            : static_cast<unsigned long>(deriveNonDeterministicSeed(this));
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
    void * d_retro_ = nullptr;   // processDepth only (lazy)
    void * d_range_ = nullptr;
    void * d_signal_ = nullptr;
    void * d_refl_ = nullptr;
    void * d_nearir_ = nullptr;
    void * d_rand_states_ = nullptr;
    void * d_raw_frame_ = nullptr;
    void * d_beam_alt_ = nullptr;
    // uploadBeamTables cache identity (host source pointers + count).
    const float * beam_src_alt_ = nullptr;
    const float * beam_src_az_  = nullptr;
    int beam_src_h_ = 0;
    void * d_beam_az_ = nullptr;
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
    int buf_n_ = 0;
    int retro_n_ = 0;
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
