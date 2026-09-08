// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0
//
// All Ouster-SDK interaction for the plugin: loading + validating the
// calibration metadata JSON, deriving the sensor dimensions and beam
// intrinsics, the WINDOW-field firmware advertisement, and ownership of
// the PacketWriter the encoder and IMU paths write through.

#pragma once

#include "ouster_lidar_profile.hpp"
#include <ouster_sim_core/metadata.hpp>
#include <optional>

#include <memory>
#include <string>
#include <vector>

namespace ouster::sdk::core::impl {
class PacketWriter;
}

namespace gz_gpu_ouster_lidar {

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
    size_t imu_packet_size = 0;
    // SDK writer retained for the Gazebo IMU adapter only.
    std::unique_ptr<ouster::sdk::core::impl::PacketWriter> pw;

private:
    std::optional<ouster_sim_core::OusterMetadata> core_;
};

}  // namespace gz_gpu_ouster_lidar
