// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#include "ray_processor_cpu_impl.hpp"
#include "backend.hpp"
#include "ray_processor_math.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace gz_gpu_ouster_lidar {

void processCpu(
    const float * depth_host,
    const float * retro_host,
    uint32_t *    range_out,
    uint16_t *    signal_out,
    uint8_t *     reflectivity_out,
    uint16_t *    nearir_out,
    const RayProcessParams & p,
    uint64_t      seed)
{
    const int H = p.H;
    const int W = p.W;
    const int n = H * W;

    // Seeded callers (tests) get a call-local RNG for reproducibility.
    // Default callers (production) share a thread-local non-deterministic RNG
    // to avoid reseeding every frame.
    static thread_local std::mt19937 tl_rng{std::random_device{}()};
    std::mt19937 local_rng;
    if (seed != 0) local_rng.seed(static_cast<std::mt19937::result_type>(seed));
    std::mt19937 & rng = (seed != 0) ? local_rng : tl_rng;
    std::normal_distribution<float> norm(0.0f, 1.0f);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    const bool has_noise = noiseEnabled(p);

    for (int idx = 0; idx < n; ++idx) {
        float d = depth_host[idx];
        const bool valid = std::isfinite(d) && d > rpmath::kValidDepthMin;

        if (!valid) {
            range_out[idx] = 0u;
            signal_out[idx] = 0u;
            reflectivity_out[idx] = static_cast<uint8_t>(p.base_reflectivity);
            nearir_out[idx] = 0u;
            continue;
        }

        // Depth-discontinuity suppression. (has_noise is redundant with
        // edge_discon_threshold > 0 — noiseEnabled() already covers it — but
        // kept for parity of structure with the other gates.)
        if (has_noise && p.edge_discon_threshold > 0.f) {
            if (rpmath::edgeDiscontinuity(depth_host, idx, H, W, p.edge_discon_threshold)
                && uni(rng) < rpmath::kEdgeSuppressProb) {
                range_out[idx] = 0u;
                signal_out[idx] = 0u;
                reflectivity_out[idx] = static_cast<uint8_t>(p.base_reflectivity);
                nearir_out[idx] = 0u;
                continue;
            }
        }

        // Random dropouts — probability scales with range and inverse reflectivity
        if (has_noise && (p.dropout_rate_close > 0.f || p.dropout_rate_far > 0.f)) {
            const float retro_val = rpmath::retroForNoise(retro_host, idx);
            const float p_drop = rpmath::dropoutProbability(
                d, retro_val, p.dropout_rate_close, p.dropout_rate_far, p.max_range);
            if (uni(rng) < p_drop) {
                range_out[idx] = 0u;
                signal_out[idx] = 0u;
                reflectivity_out[idx] = static_cast<uint8_t>(p.base_reflectivity);
                nearir_out[idx] = 0u;
                continue;
            }
        }

        // Range noise (scales with range and reflectivity)
        if (has_noise && (p.range_noise_min_std > 0.f || p.range_noise_max_std > 0.f)) {
            const float retro_val = rpmath::retroForNoise(retro_host, idx);
            const float sigma = rpmath::rangeNoiseSigma(
                d, retro_val, p.range_noise_min_std, p.range_noise_max_std, p.max_range);
            d = std::max(d + norm(rng) * sigma, 0.0f);
        }

        range_out[idx] = static_cast<uint32_t>(d * rpmath::kRangeToMm);

        // Signal with Poisson shot noise
        float intensity = 1.0f;
        if (retro_host && std::isfinite(retro_host[idx]) && retro_host[idx] > 0.0f) {
            intensity = retro_host[idx];
        }

        float sig = rpmath::signalFromRange(d, intensity, p.base_signal);
        if (has_noise && p.signal_noise_scale > 0.f) {
            float sigma_sig = std::sqrt(std::max(sig, 0.0f)) * p.signal_noise_scale;
            sig = std::max(sig + norm(rng) * sigma_sig, 0.0f);
        }
        signal_out[idx] = rpmath::clampU16(sig);

        // Reflectivity (Ouster calibrated scale, mirrors firmware output):
        //   0-100   Lambertian diffuse, linear in surface reflectance %
        //   101-255 retroreflective, log-scaled so traffic-sign / retro-tape
        //           returns don't saturate the linear band.
        // Refs:
        //   - ouster_client/include/ouster/chanfield.h: REFLECTIVITY field is
        //     "calibrated by range and sensor sensitivity" (the on-sensor
        //     firmware does this at production time).
        //   - Ouster Sensor Documentation, "Reflectivity" section
        //     (https://static.ouster.dev/sensor-docs/).
        // Slope choice: 22 ≈ (255 - 100) / 7, so rv ∈ (1, 128] maps into
        // [101, 254]; rv > 128 saturates at 255. That gives ~7 doublings of
        // dynamic range above the Lambertian band before clipping, which
        // matches how a real OS-1 grades retro-tape strength in practice. The
        // mapping itself lives in rpmath::reflectivityToByte (shared by all
        // backends); this comment is the canonical derivation.
        if (retro_host && std::isfinite(retro_host[idx]) && retro_host[idx] > 0.0f) {
            reflectivity_out[idx] = rpmath::reflectivityToByte(retro_host[idx]);
        } else {
            reflectivity_out[idx] = static_cast<uint8_t>(p.base_reflectivity);
        }

        // Near-IR with Poisson shot noise
        float nir = (retro_host && std::isfinite(retro_host[idx]) && retro_host[idx] > 0.0f)
            ? retro_host[idx] * rpmath::kNearIrScale : 0.0f;
        if (has_noise && p.nearir_noise_scale > 0.f && nir > 0.f) {
            float sigma_nir = std::sqrt(std::max(nir, 0.0f)) * p.nearir_noise_scale;
            nir = std::max(nir + norm(rng) * sigma_nir, 0.0f);
        }
        nearir_out[idx] = rpmath::clampU16(nir);
    }
}

void processRawCpu(
    const float * raw_host,
    const float * beam_alt_host,
    const float * beam_az_host,
    const ResampleParams & rp,
    uint32_t *    range_out,
    uint16_t *    signal_out,
    uint8_t *     reflectivity_out,
    uint16_t *    nearir_out,
    const RayProcessParams & pp,
    uint64_t      seed)
{
    const int H = rp.H, W = rp.W;
    const int gpu_H = rp.gpu_H, gpu_W = rp.gpu_W, gpu_chan = rp.gpu_chan;
    const int out_n = H * W;

    std::vector<float> depth_buf(static_cast<size_t>(out_n));
    std::vector<float> retro_buf(static_cast<size_t>(out_n), 0.0f);

    #pragma omp parallel for schedule(static) if(out_n > 65536)  // NOLINT(whitespace/parens)
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

        float depth = rpmath::bilinearOrAverage(
            d00, d01, d10, d11, h_alpha, v_alpha, v00, v01, v10, v11,
            n_valid, std::numeric_limits<float>::infinity());
        depth = rpmath::applyBeamOrigin(depth, beam_angle, rp.beam_origin_m);

        const int m_id = (rp.half_W - col + W) % W;
        const int ouster_idx = beam * W + m_id;
        depth_buf[static_cast<size_t>(ouster_idx)] = depth;

        if (gpu_chan >= 2) {
            const float r00 = raw_host[idx_00+1], r01 = raw_host[idx_01+1];
            const float r10 = raw_host[idx_10+1], r11 = raw_host[idx_11+1];
            const float retro = rpmath::bilinearOrAverage(
                r00, r01, r10, r11, h_alpha, v_alpha, v00, v01, v10, v11,
                n_valid, 0.0f);
            retro_buf[static_cast<size_t>(ouster_idx)] = retro;
        }
    }

    processCpu(depth_buf.data(), retro_buf.data(),
               range_out, signal_out, reflectivity_out, nearir_out, pp, seed);
}

}  // namespace gz_gpu_ouster_lidar
