// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0
//
// All Ouster-SDK interaction for the plugin: loading + validating the
// calibration metadata JSON, deriving the sensor dimensions and beam
// intrinsics, the WINDOW-field firmware advertisement, and ownership of
// the PacketWriter the encoder and IMU paths write through.

#pragma once

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
    /// derived from prod_line unless `max_range_explicit` (SDF override).
    /// `imu_enabled` only gates the IMU-profile log lines.
    bool load(const std::string & path, bool imu_enabled,
              bool max_range_explicit, double & max_range);

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
    // Beam altitude bounds including kBeamMarginDeg padding.
    double min_alt = 0.0;
    double max_alt = 0.0;
    double v_range = 0.0;
    size_t imu_packet_size = 0;
    std::unique_ptr<ouster::sdk::core::impl::PacketWriter> pw;
};

}  // namespace gz_gpu_ouster_lidar
