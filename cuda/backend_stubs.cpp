// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0
//
// Stub factories for GPU backends that are NOT compiled in. Each factory
// here returns nullptr so the dispatcher falls through to the next option.
// When a backend IS compiled in, its real .cu / .cpp file defines the
// factory and the corresponding HAVE_* macro is set by CMake, causing the
// stub here to be elided.

#include "backend.hpp"

namespace gz_gpu_ouster_lidar {

#ifndef GZ_GPU_OUSTER_HAVE_CUDA
std::unique_ptr<Backend> makeCudaBackend(uint64_t /*seed*/) { return nullptr; }
#endif

#ifndef GZ_GPU_OUSTER_HAVE_HIP
std::unique_ptr<Backend> makeHipBackend(uint64_t /*seed*/) { return nullptr; }
#endif

#ifndef GZ_GPU_OUSTER_HAVE_SYCL
std::unique_ptr<Backend> makeSyclBackend(uint64_t /*seed*/) { return nullptr; }
#endif

}  // namespace gz_gpu_ouster_lidar
