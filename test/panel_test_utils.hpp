// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0
//
// Shared helpers for tests that exercise the panel-rig resampling path.

#pragma once

#include <cmath>
#include <limits>
#include <vector>

#include "gz_gpu_ouster_lidar/ray_processor.hpp"
#include "panel_layout.hpp"

namespace gz_gpu_ouster_lidar {
namespace testutil {

/// Fill the packed rig buffer so every pixel of every panel sees a surface
/// at Euclidean range `range` (a "sphere" around the sensor). A pixel ray
/// with image-plane tangents (a, b) has planar depth range / √(1+a²+b²).
inline std::vector<float> makeUniformRangeRaw(
    const ResampleParams & rp, float range)
{
    std::vector<float> raw(static_cast<size_t>(rp.raw_n), 0.0f);
    for (int i = 0; i < rp.n_panels; ++i) {
        const ResamplePanel & p = rp.panels[i];
        for (int v = 0; v < p.height; ++v) {
            for (int u = 0; u < p.width; ++u) {
                const float a = (p.cx - static_cast<float>(u)) / p.fx;
                const float b = (p.cy - static_cast<float>(v)) / p.fy;
                raw[p.offset + v * p.width + u] =
                    range / std::sqrt(1.0f + a * a + b * b);
            }
        }
    }
    return raw;
}

/// All-miss rig buffer (every pixel +inf).
inline std::vector<float> makeAllInfRaw(const ResampleParams & rp)
{
    return std::vector<float>(static_cast<size_t>(rp.raw_n),
                              std::numeric_limits<float>::infinity());
}

/// Evenly spaced beam altitudes inside [min_alt, max_alt].
inline std::vector<float> makeBeams(int H, float min_alt, float max_alt)
{
    std::vector<float> alt(static_cast<size_t>(H));
    for (int i = 0; i < H; ++i) {
        alt[static_cast<size_t>(i)] =
            min_alt + (max_alt - min_alt) * (i + 0.5f) / static_cast<float>(H);
    }
    return alt;
}

/// No-noise RayProcessParams.
inline RayProcessParams noNoise(int H, int W, float max_range = 120.0f)
{
    RayProcessParams p{};
    p.H = H;
    p.W = W;
    p.base_signal = 800.0f;
    p.base_reflectivity = 50.0f;
    p.max_range = max_range;
    return p;
}

/// Build a verified rig for the given beam band, with far_clip set.
inline PanelLayout makeRig(float min_alt, float max_alt, int H, int W,
                           float far_clip = 120.0f, double oversample = 2.0)
{
    PanelLayout layout = buildOusterPanelLayout(
        min_alt, max_alt, H, W, oversample);
    layout.rp.far_clip = far_clip;
    return layout;
}

}  // namespace testutil
}  // namespace gz_gpu_ouster_lidar
