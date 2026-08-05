// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#include "ouster_metadata.hpp"
#include "lidar_common.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include <ouster/metadata.h>
#include <ouster/impl/packet_writer.h>
#include <ouster/lidar_scan.h>
#include <ouster/types.h>

namespace gz_gpu_ouster_lidar {

static const rclcpp::Logger kLogger = lidarLogger();

OusterMetadata::OusterMetadata() = default;
OusterMetadata::~OusterMetadata() = default;

bool OusterMetadata::load(const std::string & path, bool imu_enabled,
                          const std::string & hardware_revision,
                          bool max_range_explicit, double & max_range)
{
    // Reject pathological metadata files up-front: Ouster metadata is ~10 KB
    // of JSON. Reading /dev/zero or a multi-GB file would stall Configure.
    constexpr std::uintmax_t kMaxMetadataBytes = 10u * 1024u * 1024u;
    std::error_code ec;
    const auto fsize = std::filesystem::file_size(path, ec);
    if (ec) {
        RCLCPP_ERROR(kLogger, "Cannot stat metadata: %s (%s)",
            path.c_str(), ec.message().c_str());
        return false;
    }
    if (fsize > kMaxMetadataBytes) {
        RCLCPP_ERROR(kLogger,
            "Metadata file too large: %s is %ju bytes (limit %ju)",
            path.c_str(),
            static_cast<std::uintmax_t>(fsize),
            static_cast<std::uintmax_t>(kMaxMetadataBytes));
        return false;
    }

    // Read raw JSON
    std::ifstream fs(path);
    if (!fs.is_open()) {
        RCLCPP_ERROR(kLogger, "Cannot open metadata: %s", path.c_str());
        return false;
    }
    std::ostringstream ss;
    ss << fs.rdbuf();
    metadata_str = ss.str();

    // Parse via Ouster SDK for PacketWriter.
    // SensorInfo / PacketFormat / PacketWriter can throw on malformed or
    // incompatible metadata; catch here to avoid unwinding into Gazebo.
    try {
        ouster::sdk::core::SensorInfo info(metadata_str);
        ouster::sdk::core::PacketFormat pf(info);
        pw = std::make_unique<ouster::sdk::core::impl::PacketWriter>(pf);

        H   = pw->pixels_per_column;
        W   = static_cast<int>(info.format.columns_per_frame);
        cpp = pw->columns_per_packet;

        // Upper bounds well above any shipping Ouster (max is OS1-128 @ 2048
        // cols). Rejects corrupted/malicious metadata that would otherwise
        // drive multi-GB buffer allocations below.
        constexpr int kMaxH = 256;
        constexpr int kMaxW = 4096;
        if (H <= 0 || H > kMaxH || W <= 0 || W > kMaxW) {
            RCLCPP_ERROR(kLogger,
                "Metadata dimensions out of range: H=%d (1..%d), W=%d (1..%d)",
                H, kMaxH, W, kMaxW);
            return false;
        }

        if (cpp <= 0 || cpp > W || W % cpp != 0) {
            RCLCPP_ERROR(kLogger,
                "columns_per_frame (%d) not divisible by columns_per_packet (%d)",
                W, cpp);
            return false;
        }

        // IMU packet sizing (used only when imu_enabled).
        // Known Ouster IMU packet sizes:
        //   48 bytes  - LEGACY profile
        //   other > 0 - ACCEL32_GYRO32_NMEA (uses PacketWriter setters)
        // A zero or unrecognised size disables IMU packet emission; the
        // sensor_msgs/Imu publisher still works.
        imu_packet_size = pf.imu_packet_size;
        if (imu_enabled && imu_packet_size > 0) {
            constexpr size_t kLegacyImuSize = 48;
            if (imu_packet_size != kLegacyImuSize) {
                RCLCPP_INFO(kLogger,
                    "IMU packet size=%zu bytes (non-LEGACY profile); "
                    "using PacketWriter NMEA timestamp setter.",
                    imu_packet_size);
            }
        } else if (imu_enabled) {
            RCLCPP_WARN(kLogger,
                "IMU enabled but metadata reports imu_packet_size=0; "
                "imu_packets topic will be inactive.");
        }

        const auto fw = info.get_version();
        const auto udp_profile = info.format.udp_profile_lidar;
        const bool low_data_profile =
            udp_profile == ouster::sdk::core::UDPProfileLidar::RNG15_RFL8_NIR8 ||
            udp_profile == ouster::sdk::core::UDPProfileLidar::RNG15_RFL8_NIR8_DUAL ||
            udp_profile == ouster::sdk::core::UDPProfileLidar::FUSA_RNG15_RFL8_NIR8_DUAL;
        OusterProfileRequest profile_request;
        profile_request.prod_line = info.prod_line;
        profile_request.prod_pn = info.prod_pn;
        profile_request.hardware_revision = hardware_revision;
        profile_request.firmware_major = static_cast<int>(fw.major);
        profile_request.firmware_minor = static_cast<int>(fw.minor);
        profile_request.beam_count = H;
        profile_request.low_data_profile = low_data_profile;
        profile = resolveOusterLidarProfile(profile_request);

        if (profile.model == OusterModel::Unknown) {
            RCLCPP_ERROR(kLogger,
                "Unsupported Ouster prod_line='%s'; set metadata prod_line to "
                "OS0, OS1, OS2, OSDome, or OS1MAX.", info.prod_line.c_str());
            return false;
        }
        if (!profile.supported) {
            RCLCPP_ERROR(kLogger,
                "Unsupported Ouster product/revision combination: %s %s. "
                "Check metadata prod_line/prod_pn or set a compatible "
                "<hardware_revision>.",
                toString(profile.model), toString(profile.revision));
            return false;
        }
        if (profile.fallback_revision) {
            RCLCPP_WARN(kLogger,
                "Cannot infer hardware revision from prod_pn='%s' and "
                "firmware='%s'; using %s fallback for %s. Set "
                "<hardware_revision> explicitly for calibrated physics.",
                info.prod_pn.c_str(), info.image_rev.c_str(),
                toString(profile.revision), toString(profile.model));
        }
        if (!max_range_explicit) max_range = profile.representable_range_m;
        RCLCPP_INFO(kLogger,
            "Ouster profile: %s generation=%s revision=%s D90[10%%=%.0fm,"
            "80%%=%.0fm] min=%.2fm representable=%.0fm resolution=%.1fmm",
            profile.id.c_str(), toString(profile.generation),
            toString(profile.revision), profile.detection_range_10_d90_m,
            profile.detection_range_80_d90_m, profile.minimum_range_m,
            max_range, profile.range_resolution_m * 1000.0);

        // Beam intrinsics are available directly on SensorInfo.
        beam_alt_angles = info.beam_altitude_angles;
        beam_az_offsets = info.beam_azimuth_angles;
        beam_origin_mm  = info.lidar_origin_to_beam_origin_mm;

        // Ensure the published firmware exposes the WINDOW channel field.
        //
        // The Ouster SDK only materialises WINDOW in the LidarScan when the
        // reported firmware is >= v3.2.0 (get_field_types() strips it below
        // that — see lidar_scan.cpp), and the version is read from image_rev,
        // not build_rev (SensorInfo::get_version()). The bundled ouster-ros
        // point-cloud processor, however, lists WINDOW unconditionally in its
        // per-profile field table and does a strict scan.field(WINDOW) lookup.
        // Metadata that reports older firmware (captured v2.4.0 dumps) or an
        // unparseable string (the "sim" placeholder) therefore makes os_cloud
        // drop WINDOW and then abort with
        // "Field 'WINDOW' not found in LidarScan".
        //
        // The simulated packets use the modern profile byte layout regardless
        // of the firmware string, so advertise a firmware that keeps the field
        // present and re-serialise the metadata we publish so every consumer
        // stays consistent. pw was already built from the parsed info above
        // and is unaffected (firmware does not change the packet layout).
        //
        // Only profiles whose modern (>= v3.2.0) layout actually carries WINDOW
        // are bumped — LEGACY-profile sims are intentionally pre-3.2 and have no
        // WINDOW field, so they are left untouched.
        const ouster::sdk::core::Version kWindowFieldMinFw{3, 2, 0};
        const auto modern_fields =
            ouster::sdk::core::get_field_types(info.format, kWindowFieldMinFw);
        const bool profile_has_window = std::any_of(
            modern_fields.begin(), modern_fields.end(),
            [](const ouster::sdk::core::FieldType & ft) {
                return ft.name == ouster::sdk::core::ChanField::WINDOW;
            });
        if (profile_has_window && info.get_version() < kWindowFieldMinFw) {
            const std::string reported =
                info.image_rev.empty() ? info.fw_rev : info.image_rev;
            RCLCPP_INFO(kLogger,
                "Metadata firmware '%s' predates the WINDOW field (< v3.2.0); "
                "advertising v3.2.0 so os_cloud retains the WINDOW channel.",
                reported.c_str());
            info.image_rev = "ousteros-image-prod-aries-v3.2.0";
            info.fw_rev    = "v3.2.0";
            metadata_str   = info.to_json_string();
        }
    } catch (const std::exception & e) {
        RCLCPP_ERROR(kLogger, "Failed to parse metadata: %s", e.what());
        return false;
    }

    if (beam_alt_angles.empty() ||
        static_cast<int>(beam_alt_angles.size()) != H) {
        RCLCPP_ERROR(kLogger, "beam_altitude_angles size (%zu) != H (%d)",
            beam_alt_angles.size(), H);
        return false;
    }

    // Cache beam altitude range (with margin) for the resample pipeline.
    const auto [min_it, max_it] = std::minmax_element(
        beam_alt_angles.begin(), beam_alt_angles.end());
    min_alt = *min_it - kBeamMarginDeg;
    max_alt = *max_it + kBeamMarginDeg;
    v_range = max_alt - min_alt;
    if (v_range <= 0.0) {
        RCLCPP_ERROR(kLogger, "Invalid beam altitude range: [%.3f, %.3f]",
            min_alt, max_alt);
        return false;
    }

    // Float copies for GPU upload; beam_az_f padded to H (some metadata
    // omits azimuth offsets).
    beam_alt_f.resize(beam_alt_angles.size());
    beam_az_f.resize(beam_az_offsets.size());
    for (size_t i = 0; i < beam_alt_angles.size(); ++i)
        beam_alt_f[i] = static_cast<float>(beam_alt_angles[i]);
    for (size_t i = 0; i < beam_az_offsets.size(); ++i)
        beam_az_f[i] = static_cast<float>(beam_az_offsets[i]);
    beam_az_f.resize(static_cast<size_t>(H), 0.0f);

    return true;
}

}  // namespace gz_gpu_ouster_lidar
