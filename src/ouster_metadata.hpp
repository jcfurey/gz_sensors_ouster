// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0
//
// All Ouster-SDK interaction for the plugin: loading + validating the
// calibration metadata JSON, deriving the sensor dimensions and beam
// intrinsics, the scan rate the metadata declares, and the WINDOW-field
// firmware advertisement. Packet layouts (lidar and IMU) are owned by the
// shared ouster_sim_core encoders built from core().

#pragma once

#include "ouster_lidar_profile.hpp"
#include <ouster_sim_core/metadata.hpp>
#include <optional>

#include <string>
#include <vector>

namespace gz_gpu_ouster_lidar {

/// Frame rate (Hz) declared by Ouster metadata JSON: data_format.fps, else
/// the lidar_mode suffix (e.g. 1024x10 -> 10). nullopt when neither is
/// present. Throws if the JSON is not valid Ouster metadata.
std::optional<double> metadataFrameRateHz(const std::string & json);

class OusterMetadata {
public:
    OusterMetadata();
    ~OusterMetadata();

    /// Load and validate the metadata file. Logs every failure mode and
    /// returns false (the plugin disables itself). `max_range` is in/out:
    /// derived from the resolved product profile unless `max_range_explicit`
    /// (SDF override). `hardware_revision` is auto or an explicit revision
    /// selector such as rev06, rev07.1, or rev08.
    /// `imu_enabled` only gates the IMU-profile log lines.
    bool load(const std::string & path, bool imu_enabled,
              const std::string & hardware_revision,
              bool max_range_explicit, double & max_range);

    const ouster_sim_core::OusterMetadata & core() const { return core_.value(); }

    // ── Products (immutable after a successful load) ─────────────────────
    std::string metadata_str;               ///< JSON as published (fw-bumped)
    int H = 0;                              ///< pixels_per_column (beam count)
    int W = 0;                              ///< columns_per_frame
    int cpp = 0;                            ///< columns_per_packet
    std::vector<double> beam_alt_angles;    ///< per-beam elevation (degrees)
    std::vector<double> beam_az_offsets;    ///< per-beam azimuth offset (deg)
    std::vector<float> beam_alt_f;          ///< float copies for GPU upload
    std::vector<float> beam_az_f;           ///< (padded to H)
    double beam_origin_mm = 0.0;            ///< lidar_origin_to_beam_origin
    OusterLidarProfile profile;              ///< resolved product physics
    // Beam altitude bounds including kBeamMarginDeg padding.
    double min_alt = 0.0;
    double max_alt = 0.0;
    double v_range = 0.0;
    /// Scan rate declared by the metadata (see metadataFrameRateHz()).
    std::optional<double> frame_rate_hz;

private:
    std::optional<ouster_sim_core::OusterMetadata> core_;
};

}  // namespace gz_gpu_ouster_lidar
