// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0
//
// Non-CUDA build: CudaRayProcessor is a thin adapter that delegates to the
// shared pure-CPU implementations in ray_processor_cpu_impl.cpp.

#include "gz_gpu_ouster_lidar/cuda_ray_processor.hpp"
#include "ray_processor_cpu_impl.hpp"

namespace gz_gpu_ouster_lidar {

CudaRayProcessor::CudaRayProcessor(uint64_t seed)
{
    // No CUDA in this build — permanently on CPU.
    use_cpu_fallback_ = true;
    seed_ = seed;
}

CudaRayProcessor::~CudaRayProcessor() = default;

void CudaRayProcessor::ensureBuffers(int /*n*/) {}
void CudaRayProcessor::ensureResampleBuffers(int /*raw_n*/, int /*out_n*/, int /*H*/) {}
void CudaRayProcessor::ensureRandStates(int /*n*/) {}

void CudaRayProcessor::process(
    const float * depth_host,
    const float * retro_host,
    uint32_t *    range_out,
    uint16_t *    signal_out,
    uint8_t *     reflectivity_out,
    uint16_t *    nearir_out,
    const RayProcessParams & p)
{
    processCpu(depth_host, retro_host,
               range_out, signal_out, reflectivity_out, nearir_out, p, seed_);
}

void CudaRayProcessor::processRaw(
    const float * raw_host,
    const float * beam_alt_host,
    const float * beam_az_host,
    const ResampleParams & rp,
    uint32_t *    range_out,
    uint16_t *    signal_out,
    uint8_t *     reflectivity_out,
    uint16_t *    nearir_out,
    const RayProcessParams & pp)
{
    processRawCpu(raw_host, beam_alt_host, beam_az_host, rp,
                  range_out, signal_out, reflectivity_out, nearir_out, pp,
                  seed_);
}

}  // namespace gz_gpu_ouster_lidar
