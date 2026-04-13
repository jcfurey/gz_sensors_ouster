// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#include "gz_gpu_ouster_lidar/cuda_ray_processor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace gz_gpu_ouster_lidar {

CudaRayProcessor::CudaRayProcessor() = default;

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
    const int H = p.H;
    const int W = p.W;
    const int n = H * W;

    // Thread-local RNG for CPU fallback
    static thread_local std::mt19937 rng{std::random_device{}()};
    std::normal_distribution<float> norm(0.0f, 1.0f);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    const bool has_noise = p.range_noise_min_std > 0.f || p.range_noise_max_std > 0.f ||
                           p.signal_noise_scale > 0.f || p.nearir_noise_scale > 0.f ||
                           p.dropout_rate_close > 0.f || p.dropout_rate_far > 0.f ||
                           p.edge_discon_threshold > 0.f;

    for (int idx = 0; idx < n; ++idx) {
        float d = depth_host[idx];
        const bool valid = std::isfinite(d) && d > 0.001f;

        if (!valid) {
            range_out[idx] = 0u;
            signal_out[idx] = 0u;
            reflectivity_out[idx] = static_cast<uint8_t>(p.base_reflectivity);
            nearir_out[idx] = 0u;
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
                nearir_out[idx] = 0u;
                continue;
            }
        }

        // ── Random dropouts ──────────────────────────────────────────────────
        // Dropout probability increases with range AND decreases with reflectivity.
        if (has_noise && (p.dropout_rate_close > 0.f || p.dropout_rate_far > 0.f)) {
            float t = std::min(d / std::max(p.max_range, 0.1f), 1.0f);
            float p_drop = p.dropout_rate_close + t * (p.dropout_rate_far - p.dropout_rate_close);

            float retro_val = (retro_host && std::isfinite(retro_host[idx]) && retro_host[idx] > 0.f)
                ? retro_host[idx] : 0.5f;
            float refl_factor = std::min(1.0f / std::max(retro_val, 0.33f), 3.0f);
            p_drop *= refl_factor;

            if (uni(rng) < p_drop) {
                range_out[idx] = 0u;
                signal_out[idx] = 0u;
                reflectivity_out[idx] = static_cast<uint8_t>(p.base_reflectivity);
                nearir_out[idx] = 0u;
                continue;
            }
        }

        // ── Range noise (scales with range and reflectivity) ─────────────────
        if (has_noise && (p.range_noise_min_std > 0.f || p.range_noise_max_std > 0.f)) {
            float t = std::min(d / std::max(p.max_range, 0.1f), 1.0f);
            float sigma = p.range_noise_min_std + t * (p.range_noise_max_std - p.range_noise_min_std);

            float retro_val = (retro_host && std::isfinite(retro_host[idx]) && retro_host[idx] > 0.f)
                ? retro_host[idx] : 0.5f;
            sigma *= std::min(1.0f / std::max(retro_val, 0.5f), 2.0f);

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

        // ── Reflectivity (Ouster calibrated scale) ─────────────────────────
        // 0-100 = Lambertian (linear), 101-255 = retroreflective (log).
        if (retro_host && std::isfinite(retro_host[idx]) && retro_host[idx] > 0.0f) {
            float rv = retro_host[idx];
            uint8_t refl;
            if (rv <= 1.0f) {
                refl = static_cast<uint8_t>(std::min(rv * 100.0f, 100.0f));
            } else {
                refl = static_cast<uint8_t>(std::min(100.0f + std::log2(rv) * 22.0f, 255.0f));
            }
            reflectivity_out[idx] = refl;
        } else {
            reflectivity_out[idx] = static_cast<uint8_t>(p.base_reflectivity);
        }

        // ── Near-IR with Poisson shot noise ──────────────────────────────────
        float nir = (retro_host && std::isfinite(retro_host[idx]) && retro_host[idx] > 0.0f)
            ? retro_host[idx] * 256.0f : 0.0f;
        if (has_noise && p.nearir_noise_scale > 0.f && nir > 0.f) {
            float sigma_nir = std::sqrt(std::max(nir, 0.0f)) * p.nearir_noise_scale;
            nir = std::max(nir + norm(rng) * sigma_nir, 0.0f);
        }
        nearir_out[idx] = static_cast<uint16_t>(std::clamp(nir, 0.0f, 65535.0f));
    }
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
    // CPU fallback: resample on CPU (with OpenMP if available), then process.
    const int H = rp.H, W = rp.W;
    const int gpu_H = rp.gpu_H, gpu_W = rp.gpu_W, gpu_chan = rp.gpu_chan;
    const int out_n = H * W;

    // Temporary depth/retro buffers.
    std::vector<float> depth_buf(static_cast<size_t>(out_n));
    std::vector<float> retro_buf(static_cast<size_t>(out_n), 0.0f);

    #pragma omp parallel for schedule(static) if(out_n > 65536)
    for (int idx = 0; idx < out_n; ++idx) {
        const int beam = idx / W;
        const int col  = idx % W;

        const float beam_angle = beam_alt_host[beam];
        const float v_frac = (beam_angle - rp.min_alt) / rp.v_range;
        const float row_f  = v_frac * (gpu_H - 1);
        const int row_lo   = std::clamp(static_cast<int>(std::floor(row_f)), 0, gpu_H - 1);
        const int row_hi   = std::clamp(row_lo + 1, 0, gpu_H - 1);
        const float v_alpha = static_cast<float>(row_f - row_lo);

        const float az_offset_cols = beam_az_host[beam] / rp.deg_per_col;
        float col_wrapped = std::fmod(static_cast<float>(col) + az_offset_cols + gpu_W,
                                      static_cast<float>(gpu_W));
        if (col_wrapped < 0.f) col_wrapped += gpu_W;
        const int col_lo = static_cast<int>(std::floor(col_wrapped)) % gpu_W;
        const int col_hi = (col_lo + 1) % gpu_W;
        const float h_alpha = col_wrapped - std::floor(col_wrapped);

        const int idx_00 = (row_lo * gpu_W + col_lo) * gpu_chan;
        const int idx_01 = (row_lo * gpu_W + col_hi) * gpu_chan;
        const int idx_10 = (row_hi * gpu_W + col_lo) * gpu_chan;
        const int idx_11 = (row_hi * gpu_W + col_hi) * gpu_chan;

        const float d00 = raw_host[idx_00], d01 = raw_host[idx_01];
        const float d10 = raw_host[idx_10], d11 = raw_host[idx_11];
        const bool v00 = !std::isinf(d00), v01 = !std::isinf(d01);
        const bool v10 = !std::isinf(d10), v11 = !std::isinf(d11);
        const int n_valid = v00 + v01 + v10 + v11;

        float depth;
        if (n_valid == 0) {
            depth = std::numeric_limits<float>::infinity();
        } else if (n_valid == 4) {
            const float d_top = d00 * (1.f - h_alpha) + d01 * h_alpha;
            const float d_bot = d10 * (1.f - h_alpha) + d11 * h_alpha;
            depth = d_top * (1.f - v_alpha) + d_bot * v_alpha;
        } else {
            float sum = 0.f;
            if (v00) sum += d00; if (v01) sum += d01;
            if (v10) sum += d10; if (v11) sum += d11;
            depth = sum / static_cast<float>(n_valid);
        }

        if (std::isfinite(depth) && rp.beam_origin_m > 0.f) {
            const float elev_rad = beam_angle * 3.14159265358979f / 180.f;
            depth = std::max(0.f, depth - rp.beam_origin_m * std::cos(elev_rad));
        }

        const int m_id = (rp.half_W - col + W) % W;
        const int ouster_idx = beam * W + m_id;
        depth_buf[static_cast<size_t>(ouster_idx)] = depth;

        if (gpu_chan >= 2) {
            const float r00 = raw_host[idx_00+1], r01 = raw_host[idx_01+1];
            const float r10 = raw_host[idx_10+1], r11 = raw_host[idx_11+1];
            float retro;
            if (n_valid == 4) {
                const float r_top = r00*(1.f-h_alpha) + r01*h_alpha;
                const float r_bot = r10*(1.f-h_alpha) + r11*h_alpha;
                retro = r_top*(1.f-v_alpha) + r_bot*v_alpha;
            } else if (n_valid > 0) {
                float sum = 0.f;
                if (v00) sum += r00; if (v01) sum += r01;
                if (v10) sum += r10; if (v11) sum += r11;
                retro = sum / static_cast<float>(n_valid);
            } else {
                retro = 0.f;
            }
            retro_buf[static_cast<size_t>(ouster_idx)] = retro;
        }
    }

    // Feed into existing noise pipeline.
    process(depth_buf.data(), retro_buf.data(),
            range_out, signal_out, reflectivity_out, nearir_out, pp);
}

}  // namespace gz_gpu_ouster_lidar
