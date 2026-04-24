// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>

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

// Forward declaration for pimpl.
class Backend;

/// Vendor-neutral ray post-processor: resamples GpuRays → Ouster beam
/// geometry, then synthesises channel arrays (range_mm, signal,
/// reflectivity, near_ir). Internally dispatches to whichever backend was
/// detected at construction time — CUDA (NVIDIA), HIP (AMD incl. APUs),
/// SYCL (Intel), or CPU.
///
/// The class name is retained from the original CUDA-only implementation
/// for source-compatibility with downstream code that declared dependencies
/// on it. The behaviour is now vendor-neutral.
class CudaRayProcessor {
public:
    /// Construct the processor.
    /// @param seed  RNG seed for the noise model. 0 (default) means
    ///              non-deterministic (clock() on GPU, std::random_device on
    ///              CPU). Any non-zero value produces deterministic output
    ///              across runs — primarily useful for tests.
    explicit CudaRayProcessor(uint64_t seed = 0);
    ~CudaRayProcessor();

    CudaRayProcessor(const CudaRayProcessor &) = delete;
    CudaRayProcessor & operator=(const CudaRayProcessor &) = delete;

    /// Resample raw GpuRays buffer to Ouster beam geometry, then run the
    /// noise model and produce final channel outputs. Fast path for the
    /// GPU pipeline; one host synchronisation per call.
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

    /// Legacy interface: process pre-resampled depth/retro buffers.
    void process(
        const float * depth_host,
        const float * retro_host,
        uint32_t *    range_out,
        uint16_t *    signal_out,
        uint8_t *     reflectivity_out,
        uint16_t *    nearir_out,
        const RayProcessParams & p);

    /// Returns true when the active backend is the CPU fallback (no GPU
    /// path is compiled in, or all GPU probes failed at construction).
    bool usesCpuFallback() const;

    /// Short name of the active backend: "cuda", "hip", "sycl", or "cpu".
    /// Primarily for logging / diagnostics.
    const char * backendName() const;

    /// Configured RNG seed (0 = non-deterministic).
    uint64_t seed() const { return seed_; }

private:
    uint64_t seed_ = 0;
    std::unique_ptr<Backend> backend_;
};

}  // namespace gz_gpu_ouster_lidar
