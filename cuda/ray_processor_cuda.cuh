// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gz_gpu_ouster_lidar/ray_processor.hpp"

namespace gz_gpu_ouster_lidar {

// Forward declarations for CUDA internal types — keeps .cuh out of public headers.
void launchResampleKernel(
    const float * d_raw_frame,
    const float * d_beam_alt,
    const float * d_beam_az,
    float *       d_depth_out,
    float *       d_retro_out,
    const ResampleParams & rp,
    void * stream);

void launchRayProcessKernel(
    const float * d_depth,
    const float * d_retro,
    uint32_t *    d_range,
    uint16_t *    d_signal,
    uint8_t *     d_refl,
    uint16_t *    d_nearir,
    const RayProcessParams & p,
    void *        d_rand_states,
    void *        stream);

void launchInitRandKernel(
    void * d_states, unsigned long seed, int n, void * stream);

}  // namespace gz_gpu_ouster_lidar
