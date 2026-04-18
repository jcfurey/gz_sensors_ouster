// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0
//
// Pure-CPU implementations of the ray-processing pipeline. Compiled into the
// static library in both CUDA and non-CUDA builds so that the CUDA path can
// dispatch to these at runtime when no CUDA device is present.

#pragma once

#include "gz_gpu_ouster_lidar/cuda_ray_processor.hpp"

#include <cstdint>

namespace gz_gpu_ouster_lidar {

void processCpu(
    const float * depth_host,
    const float * retro_host,
    uint32_t *    range_out,
    uint16_t *    signal_out,
    uint8_t *     reflectivity_out,
    uint16_t *    nearir_out,
    const RayProcessParams & p);

void processRawCpu(
    const float * raw_host,
    const float * beam_alt_host,
    const float * beam_az_host,
    const ResampleParams & rp,
    uint32_t *    range_out,
    uint16_t *    signal_out,
    uint8_t *     reflectivity_out,
    uint16_t *    nearir_out,
    const RayProcessParams & pp);

}  // namespace gz_gpu_ouster_lidar
