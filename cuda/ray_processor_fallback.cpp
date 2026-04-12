// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#include "gz_gpu_ouster_lidar/cuda_ray_processor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>

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
    const int H = p.H;
    const int W = p.W;
    const int n = H * W;

    // Thread-local RNG for CPU fallback
    static thread_local std::mt19937 rng{std::random_device{}()};
    std::normal_distribution<float> norm(0.0f, 1.0f);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    const bool has_noise = p.range_noise_min_std > 0.f || p.range_noise_max_std > 0.f ||
                           p.signal_noise_scale > 0.f ||
                           p.dropout_rate_close > 0.f || p.dropout_rate_far > 0.f ||
                           p.edge_discon_threshold > 0.f;

    for (int idx = 0; idx < n; ++idx) {
        float d = depth_host[idx];
        const bool valid = std::isfinite(d) && d > 0.001f;

        if (!valid) {
            range_out[idx] = 0u;
            signal_out[idx] = 0u;
            reflectivity_out[idx] = static_cast<uint8_t>(p.base_reflectivity);
            continue;
        }

        // ── Depth-discontinuity suppression ──────────────────────────────────
        if (has_noise && p.edge_discon_threshold > 0.f) {
            const int beam = idx / W;
            const int col  = idx % W;
            bool suppress = false;

            auto checkNeighbor = [&](int nb, int nc) {
                if (nb < 0 || nb >= H || nc < 0 || nc >= W) return;
                float nd = depth_host[nb * W + nc];
                if (!std::isfinite(nd) || nd < 0.001f ||
                    std::abs(nd - d) > p.edge_discon_threshold)
                    suppress = true;
            };
            checkNeighbor(beam - 1, col);
            checkNeighbor(beam + 1, col);
            checkNeighbor(beam, col - 1);
            checkNeighbor(beam, col + 1);

            if (suppress && uni(rng) < 0.5f) {
                range_out[idx] = 0u;
                signal_out[idx] = 0u;
                reflectivity_out[idx] = static_cast<uint8_t>(p.base_reflectivity);
                continue;
            }
        }

        // ── Random dropouts ──────────────────────────────────────────────────
        if (has_noise && (p.dropout_rate_close > 0.f || p.dropout_rate_far > 0.f)) {
            float t = std::min(d / std::max(p.max_range, 0.1f), 1.0f);
            float p_drop = p.dropout_rate_close + t * (p.dropout_rate_far - p.dropout_rate_close);
            if (uni(rng) < p_drop) {
                range_out[idx] = 0u;
                signal_out[idx] = 0u;
                reflectivity_out[idx] = static_cast<uint8_t>(p.base_reflectivity);
                continue;
            }
        }

        // ── Range noise ──────────────────────────────────────────────────────
        if (has_noise && (p.range_noise_min_std > 0.f || p.range_noise_max_std > 0.f)) {
            float t = std::min(d / std::max(p.max_range, 0.1f), 1.0f);
            float sigma = p.range_noise_min_std + t * (p.range_noise_max_std - p.range_noise_min_std);
            d = std::max(d + norm(rng) * sigma, 0.0f);
        }

        range_out[idx] = static_cast<uint32_t>(d * 1000.0f);

        // ── Signal with Poisson shot noise ───────────────────────────────────
        float intensity = 1.0f;
        if (retro_host && std::isfinite(retro_host[idx]) && retro_host[idx] > 0.0f) {
            intensity = retro_host[idx];
        }

        const float r_sq = std::max(d * d, 0.0001f);
        float sig = p.base_signal * intensity / r_sq;

        if (has_noise && p.signal_noise_scale > 0.f) {
            float sigma_sig = std::sqrt(std::max(sig, 0.0f)) * p.signal_noise_scale;
            sig = std::max(sig + norm(rng) * sigma_sig, 0.0f);
        }
        signal_out[idx] = static_cast<uint16_t>(std::clamp(sig, 0.0f, 65535.0f));

        // ── Reflectivity ─────────────────────────────────────────────────────
        if (retro_host && std::isfinite(retro_host[idx]) && retro_host[idx] > 0.0f) {
            const float refl = std::min(retro_host[idx] * 1000.0f, 255.0f);
            reflectivity_out[idx] = static_cast<uint8_t>(refl);
        } else {
            reflectivity_out[idx] = static_cast<uint8_t>(p.base_reflectivity);
        }
    }
}

}  // namespace gz_gpu_ouster_lidar
