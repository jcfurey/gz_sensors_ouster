// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace gz_gpu_ouster_lidar {

/// Parameters for GPU beam resampling kernel
struct ResampleParams {
    int H;              ///< Ouster beam count (output rows)
    int W;              ///< Ouster columns per frame (output cols)
    int gpu_H;          ///< GpuRays vertical samples (input rows)
    int gpu_W;          ///< GpuRays horizontal samples (input cols)
    int gpu_chan;        ///< GpuRays channels per pixel (typically 3)
    float min_alt;      ///< Vertical FOV min (degrees, with margin)
    float v_range;      ///< Vertical FOV range (degrees)
    float deg_per_col;  ///< 360 / W
    float beam_origin_m; ///< lidar_origin_to_beam_origin in metres
    int half_W;         ///< W / 2 (azimuth remapping offset)
};

/// Parameters for CUDA ray post-processing kernel
struct RayProcessParams {
    int H;                     ///< Beam count (pixels per column)
    int W;                     ///< Columns per frame

    // ── Signal model ─────────────────────────────────────────────────────────
    float base_signal;         ///< Baseline signal for 1/r² model (photons·m²)
    float base_reflectivity;   ///< Default reflectivity value [0–255]

    // ── Range noise ──────────────────────────────────────────────────────────
    float range_noise_min_std; ///< Min range noise σ at 0 m (metres), e.g. 0.005
    float range_noise_max_std; ///< Max range noise σ at max_range (metres), e.g. 0.03
    float max_range;           ///< Sensor max range (metres) for noise scaling

    // ── Photon / signal noise ────────────────────────────────────────────────
    float signal_noise_scale;  ///< Signal Poisson noise scale (0 = off, 1 = physical)
    float nearir_noise_scale;  ///< Near-IR Poisson noise scale (0 = off, 1 = physical)

    // ── Random dropouts ──────────────────────────────────────────────────────
    float dropout_rate_close;  ///< Dropout probability at 0 m (e.g. 0.001)
    float dropout_rate_far;    ///< Dropout probability at max_range (e.g. 0.05)

    // ── Depth-discontinuity suppression ──────────────────────────────────────
    float edge_discon_threshold; ///< Depth jump threshold (metres) to suppress neighbor, 0=off
};

/// CUDA ray post-processor: resamples GpuRays → Ouster beam geometry, then
/// synthesises channel arrays (range_mm, signal, reflectivity, near_ir).
class CudaRayProcessor {
public:
    CudaRayProcessor();
    ~CudaRayProcessor();

    CudaRayProcessor(const CudaRayProcessor &) = delete;
    CudaRayProcessor & operator=(const CudaRayProcessor &) = delete;

    /// Resample raw GpuRays buffer to Ouster beam geometry on GPU, then run
    /// noise model and produce final channel outputs.  Single CUDA stream,
    /// one host sync at the end.
    ///
    /// @param raw_host         gpu_H × gpu_W × gpu_chan float array from GpuRays
    /// @param beam_alt_host    H float array of beam altitude angles (degrees)
    /// @param beam_az_host     H float array of beam azimuth offsets (degrees)
    /// @param rp               Resample parameters
    /// @param range_out        H×W uint32 output (range in mm)
    /// @param signal_out       H×W uint16 output
    /// @param reflectivity_out H×W uint8 output
    /// @param nearir_out       H×W uint16 output
    /// @param pp               Noise processing parameters
    void processRaw(
        const float * raw_host,
        const float * beam_alt_host,
        const float * beam_az_host,
        const ResampleParams & rp,
        uint32_t *    range_out,
        uint16_t *    signal_out,
        uint8_t *     reflectivity_out,
        uint16_t *    nearir_out,
        const RayProcessParams & pp);

    /// Legacy interface: process pre-resampled depth/retro buffers (CPU fallback path).
    void process(
        const float * depth_host,
        const float * retro_host,
        uint32_t *    range_out,
        uint16_t *    signal_out,
        uint8_t *     reflectivity_out,
        uint16_t *    nearir_out,
        const RayProcessParams & p);

    /// Returns true when the CUDA path is inactive — either the library was
    /// built without CUDA, or CUDA was compiled in but no usable device was
    /// detected at runtime (e.g. container started without GPU passthrough).
    /// Callers can use this for diagnostics; both process() and processRaw()
    /// transparently dispatch to the CPU implementation in this state.
    bool usesCpuFallback() const { return use_cpu_fallback_; }

private:
    void ensureBuffers(int n);
    void ensureResampleBuffers(int raw_n, int out_n, int H);
    void ensureRandStates(int n);

    bool use_cpu_fallback_ = false;
    void * stream_ = nullptr;

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

}  // namespace gz_gpu_ouster_lidar
