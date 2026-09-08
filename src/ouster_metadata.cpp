// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#include "ouster_metadata.hpp"
#include "lidar_common.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <ouster/impl/packet_writer.h>
#include <ouster/types.h>

namespace gz_gpu_ouster_lidar {

static const rclcpp::Logger kLogger = lidarLogger();

OusterMetadata::OusterMetadata() = default;
OusterMetadata::~OusterMetadata() = default;

bool OusterMetadata::load(const std::string & path, bool imu_enabled,
                         const std::string & hardware_revision,
                         bool max_range_explicit, double & max_range)
{
    core_.reset();
    pw.reset();
    try {
        auto metadata = ouster_sim_core::OusterMetadata::fromFile(path);
        // Fail during configuration, before starting the scan/drain threads.
        metadata.requirePrimaryReturnProfile();
        profile = metadata.resolvedProductProfile(hardware_revision);
        if (!profile.supported) {
            throw std::invalid_argument("unsupported Ouster product/revision: " +
                metadata.productLine() + " / " + hardware_revision);
        }
        if (profile.fallback_revision) {
            RCLCPP_WARN(kLogger,
                "Cannot infer revision from prod_pn='%s'; using %s for %s. "
                "Set <hardware_revision> explicitly for calibrated physics.",
                metadata.sourceProductPartNumber().c_str(),
                toString(profile.revision), toString(profile.model));
        }
        if (max_range_explicit &&
            (!std::isfinite(max_range) || max_range <= 0.0 ||
             max_range > metadata.maximumEncodableRangeMm() / 1000.0)) {
            throw std::invalid_argument(
                "max_range must be positive and fit the active UDP RANGE field");
        }
        if (!max_range_explicit) max_range = profile.representable_range_m;
        metadata_str = metadata.publishedJson();
        H = metadata.pixelsPerColumn();
        W = static_cast<int>(metadata.columnsPerFrame());
        cpp = metadata.columnsPerPacket();
        beam_alt_angles = metadata.beamAltitudeDeg();
        beam_az_offsets = metadata.beamAzimuthDeg();
        beam_origin_mm = metadata.beamOriginM() * 1000.0;
        const auto [lo, hi] = std::minmax_element(
            beam_alt_angles.begin(), beam_alt_angles.end());
        min_alt = *lo - kBeamMarginDeg;
        max_alt = *hi + kBeamMarginDeg;
        v_range = max_alt - min_alt;
        beam_alt_f.assign(beam_alt_angles.begin(), beam_alt_angles.end());
        beam_az_f.assign(beam_az_offsets.begin(), beam_az_offsets.end());

        // IMU packet construction remains in the ROS/Gazebo adapter. All
        // lidar metadata, profile capabilities and publication JSON above
        // come from ouster_sim_core using the same SDK provider.
        ouster::sdk::core::PacketFormat format(
            ouster::sdk::core::SensorInfo(metadata.sourceJson()));
        pw = std::make_unique<ouster::sdk::core::impl::PacketWriter>(format);
        imu_packet_size = format.imu_packet_size;
        if (imu_enabled && imu_packet_size == 0) {
            RCLCPP_WARN(kLogger, "IMU packet profile unavailable; imu_packets is inactive.");
        }
        RCLCPP_INFO(kLogger,
            "Ouster profile: %s UDP=%s returns=%u range=%.3fm resolution=%.1fmm",
            profile.id.c_str(), metadata.activeLidarUdpProfile().c_str(),
            static_cast<unsigned>(metadata.activeReturnCount()), max_range,
            profile.range_resolution_m * 1000.0);
        core_ = std::move(metadata);
        return true;
    } catch (const std::exception & e) {
        RCLCPP_ERROR(kLogger, "Failed to load Ouster metadata: %s", e.what());
        return false;
    }
}

}  // namespace gz_gpu_ouster_lidar
