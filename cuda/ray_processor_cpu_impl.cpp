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
    uint64_t      seed,
    const float * nir_host)
{
    const int H = p.H;
    const int W = p.W;
    const int n = H * W;

    const bool has_noise = noiseEnabled(p);

    auto processOne = [&](int idx, auto & rng, auto & norm, auto & uni) {
        float d = depth_host[idx];
        const bool valid = std::isfinite(d) && d > rpmath::kValidDepthMin;

        if (!valid) {
            // Solar-background false alarm: daytime background photons can
            // exceed the detection threshold on a no-return pixel, inventing
            // a spurious point uniformly distributed over the unambiguous
            // range window (Jin et al., IET Radar Sonar Navig. 14, 2020).
            // Minimal signal: a false alarm is by construction at the noise
            // floor.
            if (has_noise && p.false_alarm_rate > 0.f &&
                uni(rng) < p.false_alarm_rate) {
                const float d_fa = rpmath::quantizeRange(
                    p.min_range + uni(rng) *
                        std::max(p.max_range - p.min_range, 0.0f),
                    p.range_resolution);
                range_out[idx] =
                    static_cast<uint32_t>(d_fa * rpmath::kRangeToMm);
                signal_out[idx] = 1u;
                reflectivity_out[idx] =
                    static_cast<uint8_t>(p.base_reflectivity);
                nearir_out[idx] = 0u;
                return;
            }
            range_out[idx] = 0u;
            signal_out[idx] = 0u;
            reflectivity_out[idx] = static_cast<uint8_t>(p.base_reflectivity);
            nearir_out[idx] = 0u;
            return;
        }

        // Product/firmware minimum-range threshold. The raycaster also uses
        // this as its near clip, but keep the channel-stage gate for panel
        // mode and for noise that crosses the threshold.
        if (d < p.min_range || d >= p.max_range) {
            range_out[idx] = 0u;
            signal_out[idx] = 0u;
            reflectivity_out[idx] = static_cast<uint8_t>(p.base_reflectivity);
            nearir_out[idx] = 0u;
            return;
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
                return;
            }
        }

        // Random dropouts — probability scales with range and inverse reflectivity
        if (has_noise &&
            (p.dropout_rate_close > 0.f || p.dropout_rate_far > 0.f ||
             (p.detection_range_10 > 0.f && p.detection_range_80 > 0.f))) {
            const float retro_val = rpmath::retroForNoise(retro_host, idx);
            const float p_drop = rpmath::dropoutProbability(
                d, retro_val, p.dropout_rate_close, p.dropout_rate_far,
                p.max_range, p.detection_range_10, p.detection_range_80,
                p.detection_range_10_d50, p.detection_range_80_d50,
                p.detection_rolloff);
            if (uni(rng) < p_drop) {
                range_out[idx] = 0u;
                signal_out[idx] = 0u;
                reflectivity_out[idx] = static_cast<uint8_t>(p.base_reflectivity);
                nearir_out[idx] = 0u;
                return;
            }
        }

        // Range noise (scales with range and reflectivity)
        if (has_noise && (p.range_noise_min_std > 0.f || p.range_noise_max_std > 0.f)) {
            const float retro_val = rpmath::retroForNoise(retro_host, idx);
            const float sigma = rpmath::rangeNoiseSigma(
                d, retro_val, p.range_noise_min_std, p.range_noise_max_std,
                p.range_noise_reference_range);
            d = std::max(d + norm(rng) * sigma, 0.0f);
        }

        d = rpmath::quantizeRange(d, p.range_resolution);
        if (d < p.min_range || d >= p.max_range) {
            range_out[idx] = 0u;
            signal_out[idx] = 0u;
            reflectivity_out[idx] = static_cast<uint8_t>(p.base_reflectivity);
            nearir_out[idx] = 0u;
            return;
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

        if (retro_host && std::isfinite(retro_host[idx]) && retro_host[idx] > 0.0f) {
            reflectivity_out[idx] = rpmath::reflectivityToByte(retro_host[idx]);
        } else {
            reflectivity_out[idx] = static_cast<uint8_t>(p.base_reflectivity);
        }

        // Near-IR with Poisson shot noise. With a raycaster-provided
        // ambient factor (albedo × sun illumination, view-independent) use
        // it; otherwise the legacy retro-based analogue.
        float nir;
        if (nir_host != nullptr) {
            const float f = nir_host[idx];
            nir = (std::isfinite(f) && f > 0.0f)
                ? f * rpmath::kNearIrScale : 0.0f;
        } else {
            nir = (retro_host && std::isfinite(retro_host[idx]) && retro_host[idx] > 0.0f)
                ? retro_host[idx] * rpmath::kNearIrScale : 0.0f;
        }
        if (has_noise && p.nearir_noise_scale > 0.f && nir > 0.f) {
            float sigma_nir = std::sqrt(std::max(nir, 0.0f)) * p.nearir_noise_scale;
            nir = std::max(nir + norm(rng) * sigma_nir, 0.0f);
        }
        nearir_out[idx] = rpmath::clampU16(nir);
    };

    // Seeded callers (tests) stay serial with a call-local RNG for exact
    // reproducibility. Default production callers use one persistent RNG per
    // worker so an OpenMP scan neither shares RNG state nor reseeds per frame.
    if (seed != 0 || !has_noise) {
        std::mt19937 local_rng(static_cast<std::mt19937::result_type>(seed));
        std::normal_distribution<float> norm(0.0f, 1.0f);
        std::uniform_real_distribution<float> uni(0.0f, 1.0f);
        for (int idx = 0; idx < n; ++idx) {
            processOne(idx, local_rng, norm, uni);
        }
    } else {
#if defined(_OPENMP)
        #pragma omp parallel
        {
            static thread_local std::mt19937 tl_rng{std::random_device{}()};
            std::normal_distribution<float> norm(0.0f, 1.0f);
            std::uniform_real_distribution<float> uni(0.0f, 1.0f);
            #pragma omp for schedule(static)
            for (int idx = 0; idx < n; ++idx) {
                processOne(idx, tl_rng, norm, uni);
            }
        }
#else
        static thread_local std::mt19937 tl_rng{std::random_device{}()};
        std::normal_distribution<float> norm(0.0f, 1.0f);
        std::uniform_real_distribution<float> uni(0.0f, 1.0f);
        for (int idx = 0; idx < n; ++idx) {
            processOne(idx, tl_rng, norm, uni);
        }
#endif
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
    const int out_n = H * W;
    const float deg_per_col = 360.0f / static_cast<float>(W);
    const float kInf = std::numeric_limits<float>::infinity();

    std::vector<float> depth_buf(static_cast<size_t>(out_n));

    // Output column m IS the Ouster measurement id: m = 0 points forward
    // (+x) and azimuth decreases with m (clockwise encoder), matching the
    // hardware column ↔ encoder-angle convention.
    #pragma omp parallel for schedule(static) if(out_n > 65536)  // NOLINT(whitespace/parens)
    for (int idx = 0; idx < out_n; ++idx) {
        const int beam = idx / W;
        const int m    = idx % W;

        const float el = beam_alt_host[beam];
        const float az =
            rpmath::beamRayAzimuthDeg(beam_az_host[beam], m, deg_per_col);

        float depth = rpmath::sampleBeamRange(raw_host, rp, el, az, kInf);
        depth = rpmath::applyBeamOrigin(depth, el, rp.beam_origin_m);
        depth_buf[static_cast<size_t>(idx)] = depth;
    }

    // The depth-panel rig carries no laser_retro channel; passing a null
    // retro buffer makes the noise model fall back to base_reflectivity
    // and unit intensity (rpmath::retroForNoise / kDefaultRetro).
    processCpu(depth_buf.data(), nullptr,
               range_out, signal_out, reflectivity_out, nearir_out, pp, seed);
}

}  // namespace gz_gpu_ouster_lidar
