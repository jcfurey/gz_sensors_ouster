// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0
//
// Runtime dispatcher for CudaRayProcessor. Owns a Backend chosen at
// construction time by probing each vendor path in order.
//
// Preference order:
//   1. CUDA  (NVIDIA, via ray_processor_cuda.cu)
//   2. HIP   (AMD discrete + APUs, via ray_processor_hip.cpp)
//   3. SYCL  (Intel iGPU/Arc, via ray_processor_sycl.cpp)
//   4. CPU   (always present, OpenMP-parallelised)
//
// The env var GZ_OUSTER_BACKEND (one of cuda|hip|sycl|cpu) can force a
// specific backend for testing; if the requested backend is unavailable
// the dispatcher falls through to the default order with a warning.
//
// The class name "CudaRayProcessor" is retained for ABI/source compatibility
// — despite the name it is now vendor-neutral.

#include "gz_gpu_ouster_lidar/cuda_ray_processor.hpp"
#include "backend.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace gz_gpu_ouster_lidar {

namespace {

enum class Pref { kAuto, kCuda, kHip, kSycl, kCpu };

Pref readPrefEnv()
{
    const char * env = std::getenv("GZ_OUSTER_BACKEND");
    if (!env || !*env) return Pref::kAuto;
    if (std::strcmp(env, "cuda") == 0) return Pref::kCuda;
    if (std::strcmp(env, "hip")  == 0) return Pref::kHip;
    if (std::strcmp(env, "sycl") == 0) return Pref::kSycl;
    if (std::strcmp(env, "cpu")  == 0) return Pref::kCpu;
    std::fprintf(stderr,
        "[gz_gpu_ouster_lidar] Unknown GZ_OUSTER_BACKEND='%s'; using auto.\n",
        env);
    return Pref::kAuto;
}

std::unique_ptr<Backend> tryBackend(Pref p, uint64_t seed)
{
    switch (p) {
        case Pref::kCuda: return makeCudaBackend(seed);
        case Pref::kHip:  return makeHipBackend(seed);
        case Pref::kSycl: return makeSyclBackend(seed);
        case Pref::kCpu:  return makeCpuBackend(seed);
        default:          return nullptr;
    }
}

std::unique_ptr<Backend> pickBackend(uint64_t seed)
{
    const Pref pref = readPrefEnv();

    // If the caller forced a specific backend via env var, try it first.
    if (pref != Pref::kAuto) {
        auto forced = tryBackend(pref, seed);
        if (forced) return forced;
        std::fprintf(stderr,
            "[gz_gpu_ouster_lidar] GZ_OUSTER_BACKEND requested backend is "
            "unavailable on this system; falling back to auto selection.\n");
    }

    // Default preference order.
    for (Pref p : {Pref::kCuda, Pref::kHip, Pref::kSycl, Pref::kCpu}) {
        auto b = tryBackend(p, seed);
        if (b) return b;
    }

    // Should never happen — CPU backend is always available.
    throw std::runtime_error(
        "gz_gpu_ouster_lidar: no backend available (CPU backend missing?)");
}

}  // namespace

// ── CudaRayProcessor ────────────────────────────────────────────────────────

CudaRayProcessor::CudaRayProcessor(uint64_t seed)
    : seed_(seed), backend_(pickBackend(seed))
{
    std::fprintf(stderr,
        "[gz_gpu_ouster_lidar] Using %s backend.\n",
        backend_ ? backend_->name() : "none");
}

CudaRayProcessor::~CudaRayProcessor() = default;

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
    backend_->processRaw(raw_host, beam_alt_host, beam_az_host, rp,
                         range_out, signal_out, reflectivity_out, nearir_out,
                         pp);
}

bool CudaRayProcessor::usesCpuFallback() const
{
    return backend_ && std::strcmp(backend_->name(), "cpu") == 0;
}

const char * CudaRayProcessor::backendName() const
{
    return backend_ ? backend_->name() : "none";
}

}  // namespace gz_gpu_ouster_lidar
