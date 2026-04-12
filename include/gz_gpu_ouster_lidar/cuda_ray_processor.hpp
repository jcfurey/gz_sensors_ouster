// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace gz_gpu_ouster_lidar {

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

    // ── Random dropouts ──────────────────────────────────────────────────────
    float dropout_rate_close;  ///< Dropout probability at 0 m (e.g. 0.001)
    float dropout_rate_far;    ///< Dropout probability at max_range (e.g. 0.05)

    // ── Depth-discontinuity suppression ──────────────────────────────────────
    float edge_discon_threshold; ///< Depth jump threshold (metres) to suppress neighbor, 0=off
};

/// CUDA ray post-processor: converts GpuRays float buffer → Ouster-compatible
/// channel arrays (range_mm, signal, reflectivity).
class CudaRayProcessor {
public:
    CudaRayProcessor();
    ~CudaRayProcessor();

    CudaRayProcessor(const CudaRayProcessor &) = delete;
    CudaRayProcessor & operator=(const CudaRayProcessor &) = delete;

    /// Process raw GpuRays depth buffer.
    /// @param depth_host     H×W float array (depth in metres, row-major)
    /// @param retro_host     H×W float array (retro intensity, row-major), may be nullptr
    /// @param range_out      H×W uint32 output (range in mm, Ouster row/col order)
    /// @param signal_out     H×W uint16 output
    /// @param reflectivity_out H×W uint8 output
    /// @param p              Processing parameters
    void process(
        const float * depth_host,
        const float * retro_host,
        uint32_t *    range_out,
        uint16_t *    signal_out,
        uint8_t *     reflectivity_out,
        const RayProcessParams & p);

private:
    void ensureBuffers(int n);
    void ensureRandStates(int n);

    void * stream_ = nullptr;

    // Device buffers
    void * d_depth_   = nullptr;
    void * d_retro_   = nullptr;
    void * d_range_   = nullptr;
    void * d_signal_  = nullptr;
    void * d_refl_    = nullptr;
    void * d_rand_states_ = nullptr;

    int buf_n_  = 0;
    int rand_n_ = 0;
};

}  // namespace gz_gpu_ouster_lidar
