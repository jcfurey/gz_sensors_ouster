#pragma once

#include "gz_gpu_ouster_lidar/cuda_ray_processor.hpp"

namespace gz_gpu_ouster_lidar {

// Forward declarations for CUDA internal types — keeps .cuh out of public headers.
void launchRayProcessKernel(
    const float * d_depth,
    const float * d_retro,
    uint32_t *    d_range,
    uint16_t *    d_signal,
    uint8_t *     d_refl,
    const RayProcessParams & p,
    void *        d_rand_states,
    void *        stream);

void launchInitRandKernel(
    void * d_states, unsigned long seed, int n, void * stream);

}  // namespace gz_gpu_ouster_lidar
