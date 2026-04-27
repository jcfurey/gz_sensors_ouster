// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0
//
// Compatibility shim. The class is now `RayProcessor` (the dispatcher
// is vendor-neutral; "Cuda" was the original NVIDIA-only backend). Include
// `gz_gpu_ouster_lidar/ray_processor.hpp` directly in new code.

#pragma once

#include "gz_gpu_ouster_lidar/ray_processor.hpp"

namespace gz_gpu_ouster_lidar {

/// Deprecated alias for source compatibility with pre-rename callers.
using CudaRayProcessor = RayProcessor;

}  // namespace gz_gpu_ouster_lidar
