// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0
//
// Internal backend interface for CudaRayProcessor. Each vendor path (CUDA
// for NVIDIA, HIP for AMD, SYCL for Intel) implements this, plus an
// always-available CPU backend used as a fallback when no GPU is present.
//
// Runtime preference order is decided by the dispatcher in
// ray_processor_dispatch.cpp: CUDA → HIP → SYCL → CPU, each factory
// returning nullptr if the backend is not compiled in or no usable device
// is detected.
//
// This header is NOT part of the public API — it lives in cuda/ rather
// than include/.

#pragma once

#include "gz_gpu_ouster_lidar/cuda_ray_processor.hpp"

#include <cstdint>
#include <memory>

namespace gz_gpu_ouster_lidar {

class Backend {
public:
    virtual ~Backend() = default;

    virtual void processRaw(
        const float * raw_host,
        const float * beam_alt_host,
        const float * beam_az_host,
        const ResampleParams & rp,
        uint32_t *    range_out,
        uint16_t *    signal_out,
        uint8_t *     reflectivity_out,
        uint16_t *    nearir_out,
        const RayProcessParams & pp) = 0;

    /// Short identifier: "cuda", "hip", "sycl", or "cpu".
    virtual const char * name() const = 0;
};

// Factories. Return nullptr if the backend is not compiled in OR if it is
// compiled in but no usable device is detected at runtime. Implementations
// of the non-compiled-in backends live in backend_stubs.cpp.
std::unique_ptr<Backend> makeCudaBackend(uint64_t seed);
std::unique_ptr<Backend> makeHipBackend(uint64_t seed);
std::unique_ptr<Backend> makeSyclBackend(uint64_t seed);
std::unique_ptr<Backend> makeCpuBackend(uint64_t seed);

/// True if any noise/dropout/edge-suppression term is non-zero. Backends
/// use this to skip RNG state allocation and the curand/hiprand init
/// kernel when the user has fully disabled noise.
inline bool noiseEnabled(const RayProcessParams & p)
{
    return p.range_noise_min_std > 0.f || p.range_noise_max_std > 0.f ||
           p.signal_noise_scale > 0.f || p.nearir_noise_scale > 0.f ||
           p.dropout_rate_close > 0.f || p.dropout_rate_far > 0.f ||
           p.edge_discon_threshold > 0.f;
}

}  // namespace gz_gpu_ouster_lidar
