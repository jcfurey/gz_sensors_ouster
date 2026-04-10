#include "gz_gpu_ouster_lidar/cuda_ray_processor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace gz_gpu_ouster_lidar {

CudaRayProcessor::CudaRayProcessor() = default;

CudaRayProcessor::~CudaRayProcessor() = default;

void CudaRayProcessor::ensureBuffers(int /*n*/) {}

void CudaRayProcessor::ensureRandStates(int /*n*/) {}

void CudaRayProcessor::process(
    const float * depth_host,
    const float * retro_host,
    uint32_t *    range_out,
    uint16_t *    signal_out,
    uint8_t *     reflectivity_out,
    const RayProcessParams & p)
{
    const int n = p.H * p.W;
    (void)p.dt_per_col_ns;

    for (int idx = 0; idx < n; ++idx) {
        float d = depth_host[idx];
        const bool valid = std::isfinite(d) && d > 0.001f;

        if (!valid) {
            range_out[idx] = 0u;
            signal_out[idx] = 0u;
            reflectivity_out[idx] = static_cast<uint8_t>(p.base_reflectivity);
            continue;
        }

        // CPU fallback intentionally skips random range noise.
        range_out[idx] = static_cast<uint32_t>(d * 1000.0f);

        float intensity = 1.0f;
        if (retro_host && std::isfinite(retro_host[idx]) && retro_host[idx] > 0.0f) {
            intensity = retro_host[idx];
        }

        const float r_sq = std::max(d * d, 0.0001f);
        const float sig = std::min(p.base_signal * intensity / r_sq, 65535.0f);
        signal_out[idx] = static_cast<uint16_t>(sig);

        if (retro_host && std::isfinite(retro_host[idx]) && retro_host[idx] > 0.0f) {
            const float refl = std::min(retro_host[idx] * 1000.0f, 255.0f);
            reflectivity_out[idx] = static_cast<uint8_t>(refl);
        } else {
            reflectivity_out[idx] = static_cast<uint8_t>(p.base_reflectivity);
        }
    }
}

}  // namespace gz_gpu_ouster_lidar
