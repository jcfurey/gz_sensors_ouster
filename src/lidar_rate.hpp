// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0
//
// Scan-rate resolution: an Ouster sensor's frame rate is part of its
// lidar_mode (e.g. 1024x10), so the metadata is the default source of
// <lidar_hz>. An explicit SDF rate is honoured but flagged when it disagrees,
// because packet column timestamps, the IMU packet cadence and the published
// metadata would then describe different sensors.

#pragma once

#include <cmath>
#include <optional>

namespace gz_gpu_ouster_lidar {

struct LidarRateResolution {
    enum class Source { kSdf, kMetadata, kFallback };

    double hz = 10.0;
    Source source = Source::kFallback;
    /// An SDF value was given but rejected (non-finite or <= 0).
    bool sdf_invalid = false;
    /// A valid SDF value differs from the metadata frame rate.
    bool mismatch = false;
};

/// Resolve the scan rate from the optional SDF <lidar_hz> and the optional
/// metadata frame rate (data_format.fps, else the lidar_mode suffix).
/// Precedence: valid SDF value, then metadata, then `fallback_hz`.
inline LidarRateResolution resolveLidarRate(
    std::optional<double> sdf_hz,
    std::optional<double> metadata_hz,
    double fallback_hz = 10.0)
{
    const auto valid = [](const std::optional<double> & hz) {
        return hz.has_value() && std::isfinite(*hz) && *hz > 0.0;
    };
    LidarRateResolution out;
    if (sdf_hz.has_value()) {
        if (valid(sdf_hz)) {
            out.hz = *sdf_hz;
            out.source = LidarRateResolution::Source::kSdf;
            out.mismatch = valid(metadata_hz) &&
                std::abs(*sdf_hz - *metadata_hz) > 1.0e-6 * *metadata_hz;
            return out;
        }
        out.sdf_invalid = true;
    }
    if (valid(metadata_hz)) {
        out.hz = *metadata_hz;
        out.source = LidarRateResolution::Source::kMetadata;
        return out;
    }
    out.hz = fallback_hz;
    out.source = LidarRateResolution::Source::kFallback;
    return out;
}

inline const char * lidarRateSourceName(LidarRateResolution::Source source)
{
    switch (source) {
        case LidarRateResolution::Source::kSdf: return "SDF <lidar_hz>";
        case LidarRateResolution::Source::kMetadata: return "metadata";
        case LidarRateResolution::Source::kFallback: return "default";
    }
    return "unknown";
}

}  // namespace gz_gpu_ouster_lidar
