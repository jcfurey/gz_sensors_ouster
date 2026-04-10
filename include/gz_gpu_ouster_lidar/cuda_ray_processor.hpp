#pragma once

#include <cstdint>

namespace gz_gpu_ouster_lidar {

/// Parameters for CUDA ray post-processing kernel
struct RayProcessParams {
    int H;                     ///< Beam count (pixels per column)
    int W;                     ///< Columns per frame
    float base_signal;         ///< Baseline signal for 1/r² model
    float base_reflectivity;   ///< Default reflectivity value
    float range_noise_std;     ///< Gaussian range noise σ (metres), 0 = off
    uint64_t dt_per_col_ns;    ///< Nanoseconds between columns (rolling shutter)
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
