// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#include "gz_gpu_ouster_lidar/gz_gpu_ouster_lidar_system.hpp"
#include "gz_gpu_ouster_lidar/ray_processor.hpp"
#include "backend.hpp"     // noiseEnabled() — pulled in via cuda/ public includes
#include "imu_noise.hpp"   // applyImuNoise()

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>

#include <gz/plugin/Register.hh>
#include <gz/sim/Entity.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/EventManager.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/Link.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/ParentEntity.hh>
#include <gz/sim/components/Sensor.hh>
#include <gz/sim/components/World.hh>
#include <gz/sim/components/AngularVelocity.hh>
#include <gz/sim/components/LinearAcceleration.hh>
#include <gz/math/Pose3.hh>
#include <gz/rendering/DepthCamera.hh>
#include <gz/rendering/RenderEngine.hh>
#include <gz/rendering/RenderingIface.hh>
#include <gz/rendering/Scene.hh>
#include <gz/sim/rendering/Events.hh>

#include <ouster/metadata.h>
#include <ouster/impl/packet_writer.h>
#include <ouster/lidar_scan.h>
#include <ouster/types.h>

#include <rclcpp/rclcpp.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <builtin_interfaces/msg/time.hpp>
#include <ouster_sensor_msgs/msg/packet_msg.hpp>
#include <std_msgs/msg/string.hpp>

namespace gz_gpu_ouster_lidar {

static const rclcpp::Logger kLogger = rclcpp::get_logger("gz_gpu_ouster_lidar");

// Small angular pad added to the beam altitude range so the panel rig's
// vertical coverage extends a touch beyond the outermost beams; keeps
// bilinear corners of the edge beams inside rendered pixels.
static constexpr double kBeamMarginDeg = 1.0;

// Near clip plane for the panel depth cameras (metres). Matches the real
// sensor's minimum range region where returns are unreliable anyway.
static constexpr double kNearClip = 0.3;

// ── Registration ─────────────────────────────────────────────────────────────

GZ_ADD_PLUGIN(
    GzGpuOusterLidarSystem,
    ::gz::sim::System,
    ::gz::sim::ISystemConfigure,
    ::gz::sim::ISystemPostUpdate)

GZ_ADD_PLUGIN_ALIAS(
    GzGpuOusterLidarSystem,
    "gz_gpu_ouster_lidar::GzGpuOusterLidarSystem")

// ── Construction / destruction ───────────────────────────────────────────────

GzGpuOusterLidarSystem::GzGpuOusterLidarSystem() = default;

GzGpuOusterLidarSystem::~GzGpuOusterLidarSystem()
{
    // Order matters here: see render_busy_mtx_ comment in the header.
    // 1. Set shutdown_ first so a render callback that begins (or is
    //    already past the lock) can observe it and bail before touching
    //    teardown-fragile state.
    // 2. Disconnect the render hooks. gz::common::Connection::reset() does
    //    not wait for in-flight callbacks, so this only prevents *new*
    //    invocations.
    // 3. Take render_busy_mtx_ to flush any in-flight callback. By the
    //    time we own the lock and release it, no render-thread code is
    //    executing in this instance.
    // 4. Tear down the rest in reverse construction order.
    shutdown_.store(true, std::memory_order_release);
    render_conn_.reset();
    { std::lock_guard<std::recursive_mutex> lk(render_busy_mtx_); }
    DestroyPanels();

    drain_cv_.notify_all();
    if (drain_thread_.joinable()) {
        drain_thread_.join();
    }
    if (ros_executor_) {
        ros_executor_->cancel();
    }
    if (ros_spin_thread_.joinable()) {
        ros_spin_thread_.join();
    }
}

// ── ISystemConfigure ─────────────────────────────────────────────────────────

void GzGpuOusterLidarSystem::Configure(
    const ::gz::sim::Entity & entity,
    const std::shared_ptr<const sdf::Element> & sdf,
    ::gz::sim::EntityComponentManager & ecm,
    ::gz::sim::EventManager & event_mgr)
{
    // Read SDF parameters
    if (sdf->HasElement("metadata_path")) {
        metadata_path_ = sdf->Get<std::string>("metadata_path");
        // Resolve relative paths against the SDF file's directory.
        // Absolute paths are used as-is.
        if (!metadata_path_.empty() &&
            !std::filesystem::path(metadata_path_).is_absolute()) {
            std::string sdf_file = sdf->FilePath();
            if (!sdf_file.empty()) {
                auto parent = std::filesystem::path(sdf_file).parent_path();
                if (!parent.empty()) {
                    metadata_path_ = (parent / metadata_path_).string();
                }
            } else {
                RCLCPP_ERROR(kLogger,
                    "metadata_path '%s' is relative but the plugin's SDF has "
                    "no file path (likely loaded from a string); use an "
                    "absolute path.", metadata_path_.c_str());
            }
        }
    }
    if (sdf->HasElement("sensor_name")) {
        sensor_name_ = sdf->Get<std::string>("sensor_name");
    }
    if (sdf->HasElement("lidar_hz")) {
        lidar_hz_ = sdf->Get<double>("lidar_hz");
    }
    if (sdf->HasElement("visibility_mask")) {
        visibility_mask_ = sdf->Get<uint32_t>("visibility_mask");
    }
    if (sdf->HasElement("panel_oversample")) {
        panel_oversample_ = sdf->Get<double>("panel_oversample");
    }
    if (sdf->HasElement("panel_sampling")) {
        panel_sampling_ = sdf->Get<std::string>("panel_sampling");
    }
    if (sdf->HasElement("image_qos")) {
        image_qos_ = sdf->Get<std::string>("image_qos");
    }
    if (sdf->HasElement("imu_qos")) {
        imu_qos_ = sdf->Get<std::string>("imu_qos");
    }

    // Noise model SDF parameters (all optional, with sensible Ouster defaults)
    if (sdf->HasElement("range_noise_min_std")) {
        range_noise_min_std_ = sdf->Get<double>("range_noise_min_std");
    }
    if (sdf->HasElement("range_noise_max_std")) {
        range_noise_max_std_ = sdf->Get<double>("range_noise_max_std");
    }
    if (sdf->HasElement("signal_noise_scale")) {
        signal_noise_scale_ = sdf->Get<double>("signal_noise_scale");
    }
    if (sdf->HasElement("nearir_noise_scale")) {
        nearir_noise_scale_ = sdf->Get<double>("nearir_noise_scale");
    }
    if (sdf->HasElement("dropout_rate_close")) {
        dropout_rate_close_ = sdf->Get<double>("dropout_rate_close");
    }
    if (sdf->HasElement("dropout_rate_far")) {
        dropout_rate_far_ = sdf->Get<double>("dropout_rate_far");
    }
    if (sdf->HasElement("edge_discon_threshold")) {
        edge_discon_threshold_ = sdf->Get<double>("edge_discon_threshold");
    }
    if (sdf->HasElement("base_signal")) {
        base_signal_ = sdf->Get<double>("base_signal");
    }
    if (sdf->HasElement("base_reflectivity")) {
        base_reflectivity_ = sdf->Get<double>("base_reflectivity");
    }
    if (sdf->HasElement("max_range")) {
        max_range_ = sdf->Get<double>("max_range");
        max_range_explicit_ = true;
    }

    // ── IMU parameters (optional — omit imu_name to disable) ─────────────────
    if (sdf->HasElement("imu_name")) {
        imu_name_ = sdf->Get<std::string>("imu_name");
        imu_enabled_ = !imu_name_.empty();
    }
    if (sdf->HasElement("imu_hz")) {
        imu_hz_ = sdf->Get<double>("imu_hz");
    }
    if (sdf->HasElement("publish_imu_msg")) {
        publish_imu_msg_ = sdf->Get<bool>("publish_imu_msg");
    }

    // ── IMU noise model (SDF-optional, defaults match Ouster Os1) ────────────
    if (sdf->HasElement("gyro_noise_std")) {
        gyro_noise_std_ = sdf->Get<double>("gyro_noise_std");
    }
    if (sdf->HasElement("accel_noise_std")) {
        accel_noise_std_ = sdf->Get<double>("accel_noise_std");
    }
    if (sdf->HasElement("gyro_bias_walk")) {
        gyro_bias_walk_ = sdf->Get<double>("gyro_bias_walk");
    }
    if (sdf->HasElement("accel_bias_walk")) {
        accel_bias_walk_ = sdf->Get<double>("accel_bias_walk");
    }

    // ── Validate parameters ────────────────────────────────────────────────────
    range_noise_min_std_   = std::max(0.0, range_noise_min_std_);
    range_noise_max_std_   = std::max(0.0, range_noise_max_std_);
    signal_noise_scale_    = std::max(0.0, signal_noise_scale_);
    nearir_noise_scale_    = std::max(0.0, nearir_noise_scale_);
    dropout_rate_close_    = std::clamp(dropout_rate_close_, 0.0, 1.0);
    dropout_rate_far_      = std::clamp(dropout_rate_far_,   0.0, 1.0);
    edge_discon_threshold_ = std::max(0.0, edge_discon_threshold_);
    base_signal_           = std::max(0.0, base_signal_);
    base_reflectivity_     = std::clamp(base_reflectivity_, 0.0, 255.0);
    max_range_             = std::max(1.0, max_range_);
    gyro_noise_std_        = std::max(0.0, gyro_noise_std_);
    accel_noise_std_       = std::max(0.0, accel_noise_std_);
    gyro_bias_walk_        = std::max(0.0, gyro_bias_walk_);
    accel_bias_walk_       = std::max(0.0, accel_bias_walk_);
    panel_oversample_      = std::clamp(panel_oversample_, 1.0, 4.0);
    if (panel_sampling_ != "bilinear" && panel_sampling_ != "nearest") {
        RCLCPP_WARN(kLogger,
            "Unknown panel_sampling='%s'; expected bilinear|nearest. "
            "Defaulting to bilinear.", panel_sampling_.c_str());
        panel_sampling_ = "bilinear";
    }

    auto validate_qos = [](std::string & val, const char * field, const char * fallback) {
        if (val != "reliable" && val != "best_effort" && val != "sensor_data") {
            RCLCPP_WARN(kLogger,
                "Unknown %s='%s'; expected reliable|best_effort|sensor_data. "
                "Defaulting to %s.", field, val.c_str(), fallback);
            val = fallback;
        }
    };
    validate_qos(image_qos_, "image_qos", "reliable");
    validate_qos(imu_qos_,   "imu_qos",   "sensor_data");
    if (lidar_hz_ <= 0.0) {
        RCLCPP_WARN(kLogger, "lidar_hz must be > 0, got %f; defaulting to 10", lidar_hz_);
        lidar_hz_ = 10.0;
    }
    if (imu_enabled_ && imu_hz_ <= 0.0) {
        RCLCPP_WARN(kLogger, "imu_hz must be > 0, got %f; defaulting to 100", imu_hz_);
        imu_hz_ = 100.0;
    }

    if (metadata_path_.empty()) {
        RCLCPP_ERROR(kLogger, "'metadata_path' SDF parameter is required");
        return;
    }

    // Discover world name
    auto worldEntity = ::gz::sim::worldEntity(ecm);
    if (worldEntity != ::gz::sim::kNullEntity) {
        auto * nameComp = ecm.Component<::gz::sim::components::Name>(worldEntity);
        if (nameComp) {
            world_name_ = nameComp->Data();
        }
    }

    sensor_entity_ = entity;

    // Load Ouster metadata and beam angles
    if (!loadMetadata()) {
        RCLCPP_ERROR(kLogger, "Metadata loading failed; plugin disabled");
        return;
    }

    // Initialise ray processor. The CUDA path self-probes for a GPU and flips
    // to a CPU implementation when none is present (no GPU passthrough, older
    // driver, etc.); the try/catch is a belt-and-braces guard so any residual
    // throw still disables just this plugin rather than taking the whole sim
    // down.
    try {
        ray_processor_ = std::make_unique<RayProcessor>();
    } catch (const std::exception & e) {
        RCLCPP_ERROR(kLogger,
            "Ray processor init failed: %s; plugin disabled", e.what());
        return;
    }
    if (ray_processor_->usesCpuFallback()) {
        RCLCPP_WARN(kLogger,
            "No CUDA-capable device detected; gz_gpu_ouster_lidar running on "
            "CPU fallback (expect lower sim rate on high-resolution sensors).");
    }

    // Allocate channel buffers
    const int n = H_ * W_;
    range_buf_.resize(static_cast<size_t>(n), 0);
    signal_buf_.resize(static_cast<size_t>(n), 0);
    reflectivity_buf_.resize(static_cast<size_t>(n),
                             static_cast<uint8_t>(base_reflectivity_));
    nearir_buf_.resize(static_cast<size_t>(n), 0);
    // Convert beam angles to float for GPU upload.
    beam_alt_f_.resize(beam_alt_angles_.size());
    beam_az_f_.resize(beam_az_offsets_.size());
    for (size_t i = 0; i < beam_alt_angles_.size(); ++i)
        beam_alt_f_[i] = static_cast<float>(beam_alt_angles_[i]);
    for (size_t i = 0; i < beam_az_offsets_.size(); ++i)
        beam_az_f_[i] = static_cast<float>(beam_az_offsets_[i]);
    // Pad beam_az_f_ to H if shorter (some metadata omits azimuth offsets).
    beam_az_f_.resize(static_cast<size_t>(H_), 0.0f);

    // ── Build the panel rig from the beam geometry ───────────────────────────
    // Cylindrical sectors for OS0/1/2; pitched sectors + zenith cap for the
    // hemispherical OSDome. min_alt_/max_alt_ already carry kBeamMarginDeg.
    layout_ = buildOusterPanelLayout(
        min_alt_, max_alt_, H_, W_, panel_oversample_);
    if (layout_.n_panels == 0) {
        RCLCPP_ERROR(kLogger,
            "Unsupported beam geometry: altitude range [%.1f, %.1f] deg has "
            "no panel-rig coverage (supported: within ±60 deg cylindrical, "
            "or -32..+90 deg hemispherical); plugin disabled.",
            min_alt_, max_alt_);
        return;
    }
    layout_.rp.far_clip = static_cast<float>(max_range_);
    layout_.rp.beam_origin_m = static_cast<float>(beam_origin_mm_ / 1000.0);
    layout_.rp.nearest = (panel_sampling_ == "nearest") ? 1 : 0;

    // Verify the exact calibrated beams (incl. per-beam azimuth offsets)
    // against the rig. The builder already self-checks a fine grid; this
    // catches pathological metadata (offsets beyond the pads).
    const int uncovered = countUncoveredRays(
        layout_.rp, beam_alt_f_.data(), beam_az_f_.data());
    if (uncovered > 0) {
        RCLCPP_WARN(kLogger,
            "%d of %d beam rays fall outside the panel rig and will read as "
            "misses (check beam_azimuth_angles in the metadata).",
            uncovered, H_ * W_);
    }
    {
        std::string dims;
        for (int i = 0; i < layout_.n_panels; ++i) {
            dims += " " + std::to_string(layout_.cams[i].width) + "x" +
                    std::to_string(layout_.cams[i].height);
        }
        RCLCPP_INFO(kLogger,
            "Panel rig: %d %s panels, %s sampling (%.1f MiB raw):%s",
            layout_.n_panels,
            layout_.hemispherical ? "hemispherical" : "cylindrical",
            panel_sampling_.c_str(),
            layout_.rp.raw_n * sizeof(float) / 1048576.0, dims.c_str());
    }

    // Derive the Gazebo sensor entity name for pose tracking.
    // sensor_name_ is e.g. "/sensor/lidar/lidar0" → extract "lidar0".
    // Gazebo merges fixed-joint child links (including lidar_frame) into the
    // parent link (base_footprint), so there is no Link entity for
    // lidar_frame.  However the <sensor> element survives merging and carries
    // the accumulated pose.  We search for the Sensor entity by name.
    const auto last_slash = sensor_name_.rfind('/');
    const std::string lidar_id =
        (last_slash != std::string::npos)
            ? sensor_name_.substr(last_slash + 1)
            : sensor_name_;
    lidar_frame_name_ = lidar_id;  // Gazebo sensor name, e.g. "lidar0"

    // Image frame_id: match the URDF lidar frame (e.g. "lidar0/lidar_frame")
    image_frame_id_ = lidar_id + "/lidar_frame";
    imu_frame_id_ = lidar_id + "/imu_frame";

    // Initialise ROS 2 node and publishers (after frame IDs are derived).
    // If this partially fails, tear down before leaving Configure so the
    // destructor doesn't touch inconsistent state.
    try {
        initRosInterface();
    } catch (const std::exception & e) {
        RCLCPP_ERROR(kLogger,
            "ROS interface init failed: %s; plugin disabled", e.what());
        if (ros_executor_) {
            ros_executor_->cancel();
        }
        if (ros_spin_thread_.joinable()) {
            ros_spin_thread_.join();
        }
        return;
    }

    // Connect to the rendering-thread Render event (fires after scene->PreRender()
    // at Sensors.cc:496) so that panel-rig initialisation and Render() happen on
    // the correct (EGL) thread with the scene already set up.
    event_mgr_ = &event_mgr;
    render_conn_ = event_mgr.Connect<::gz::sim::events::Render>(
        std::bind(&GzGpuOusterLidarSystem::OnRender, this));

    RCLCPP_INFO(kLogger,
        "Configured: H=%d W=%d cpp=%d sensor_name=%s gz_sensor=%s hz=%.1f"
        " noise: range_sigma=[%.4f,%.4f]m signal_noise=%.1f"
        " dropout=[%.4f,%.4f] edge_discon=%.3fm max_range=%.1fm",
        H_, W_, cpp_, sensor_name_.c_str(), lidar_frame_name_.c_str(), lidar_hz_,
        range_noise_min_std_, range_noise_max_std_, signal_noise_scale_,
        dropout_rate_close_, dropout_rate_far_, edge_discon_threshold_, max_range_);
    if (imu_enabled_) {
        RCLCPP_INFO(kLogger, "  IMU: sensor=%s hz=%.1f publish_imu_msg=%s",
            imu_name_.c_str(), imu_hz_, publish_imu_msg_ ? "true" : "false");
    }

    // Start drain thread last — all state is fully initialised above.
    drain_thread_ = std::thread(&GzGpuOusterLidarSystem::drainThreadFunc, this);
}

// ── Metadata loading ─────────────────────────────────────────────────────────

bool GzGpuOusterLidarSystem::loadMetadata()
{
    // Reject pathological metadata files up-front: Ouster metadata is ~10 KB
    // of JSON. Reading /dev/zero or a multi-GB file would stall Configure.
    constexpr std::uintmax_t kMaxMetadataBytes = 10u * 1024u * 1024u;
    std::error_code ec;
    const auto fsize = std::filesystem::file_size(metadata_path_, ec);
    if (ec) {
        RCLCPP_ERROR(kLogger, "Cannot stat metadata: %s (%s)",
            metadata_path_.c_str(), ec.message().c_str());
        return false;
    }
    if (fsize > kMaxMetadataBytes) {
        RCLCPP_ERROR(kLogger,
            "Metadata file too large: %s is %ju bytes (limit %ju)",
            metadata_path_.c_str(),
            static_cast<std::uintmax_t>(fsize),
            static_cast<std::uintmax_t>(kMaxMetadataBytes));
        return false;
    }

    // Read raw JSON
    std::ifstream fs(metadata_path_);
    if (!fs.is_open()) {
        RCLCPP_ERROR(kLogger, "Cannot open metadata: %s", metadata_path_.c_str());
        return false;
    }
    std::ostringstream ss;
    ss << fs.rdbuf();
    metadata_str_ = ss.str();

    // Parse via Ouster SDK for PacketWriter.
    // SensorInfo / PacketFormat / PacketWriter can throw on malformed or
    // incompatible metadata; catch here to avoid unwinding into Gazebo.
    try {
        ouster::sdk::core::SensorInfo info(metadata_str_);
        ouster::sdk::core::PacketFormat pf(info);
        pw_ = std::make_unique<ouster::sdk::core::impl::PacketWriter>(pf);

        H_   = pw_->pixels_per_column;
        W_   = static_cast<int>(info.format.columns_per_frame);
        cpp_ = pw_->columns_per_packet;

        // Upper bounds well above any shipping Ouster (max is OS1-128 @ 2048
        // cols). Rejects corrupted/malicious metadata that would otherwise
        // drive multi-GB buffer allocations below.
        constexpr int kMaxH = 256;
        constexpr int kMaxW = 4096;
        if (H_ <= 0 || H_ > kMaxH || W_ <= 0 || W_ > kMaxW) {
            RCLCPP_ERROR(kLogger,
                "Metadata dimensions out of range: H=%d (1..%d), W=%d (1..%d)",
                H_, kMaxH, W_, kMaxW);
            return false;
        }

        if (cpp_ <= 0 || cpp_ > W_ || W_ % cpp_ != 0) {
            RCLCPP_ERROR(kLogger, "columns_per_frame (%d) not divisible by columns_per_packet (%d)",
                W_, cpp_);
            return false;
        }

        pkt_buf_.resize(pw_->lidar_packet_size, 0);

        // IMU packet buffer (used only when imu_enabled_).
        // Known Ouster IMU packet sizes:
        //   48 bytes  - LEGACY profile
        //   other > 0 - ACCEL32_GYRO32_NMEA (uses PacketWriter setters)
        // A zero or unrecognised size disables IMU packet emission; the
        // sensor_msgs/Imu publisher still works.
        imu_packet_size_ = pf.imu_packet_size;
        if (imu_enabled_ && imu_packet_size_ > 0) {
            imu_pkt_buf_.resize(imu_packet_size_, 0);
            constexpr size_t kLegacyImuSize = 48;
            if (imu_packet_size_ != kLegacyImuSize) {
                RCLCPP_INFO(kLogger,
                    "IMU packet size=%zu bytes (non-LEGACY profile); "
                    "using PacketWriter NMEA timestamp setter.",
                    imu_packet_size_);
            }
        } else if (imu_enabled_) {
            RCLCPP_WARN(kLogger,
                "IMU enabled but metadata reports imu_packet_size=0; "
                "imu_packets topic will be inactive.");
        }

        // Derive max_range from product line if not explicitly set via SDF.
        if (!max_range_explicit_ && !info.prod_line.empty()) {
            const auto & pl = info.prod_line;
            if (pl.find("OS0") != std::string::npos)      max_range_ = 50.0;
            else if (pl.find("OS1") != std::string::npos)  max_range_ = 120.0;
            else if (pl.find("OS2") != std::string::npos)  max_range_ = 240.0;
            // OSDome and others keep the default 120m
            RCLCPP_INFO(kLogger, "Derived max_range=%.0fm from prod_line=%s",
                max_range_, pl.c_str());
        }

        // Beam intrinsics are available directly on SensorInfo.
        beam_alt_angles_ = info.beam_altitude_angles;
        beam_az_offsets_ = info.beam_azimuth_angles;
        beam_origin_mm_  = info.lidar_origin_to_beam_origin_mm;

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
        // stays consistent. pw_ was already built from the parsed info above
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
            metadata_str_  = info.to_json_string();
        }
    } catch (const std::exception & e) {
        RCLCPP_ERROR(kLogger, "Failed to parse metadata: %s", e.what());
        return false;
    }

    if (beam_alt_angles_.empty() || static_cast<int>(beam_alt_angles_.size()) != H_) {
        RCLCPP_ERROR(kLogger, "beam_altitude_angles size (%zu) != H (%d)",
            beam_alt_angles_.size(), H_);
        return false;
    }

    // Cache beam altitude range (with margin) for the resample pipeline.
    const auto [min_it, max_it] = std::minmax_element(
        beam_alt_angles_.begin(), beam_alt_angles_.end());
    min_alt_ = *min_it - kBeamMarginDeg;
    max_alt_ = *max_it + kBeamMarginDeg;
    v_range_ = max_alt_ - min_alt_;
    if (v_range_ <= 0.0) {
        RCLCPP_ERROR(kLogger, "Invalid beam altitude range: [%.3f, %.3f]",
            min_alt_, max_alt_);
        return false;
    }

    return true;
}

// ── ROS 2 interface init ─────────────────────────────────────────────────────

void GzGpuOusterLidarSystem::initRosInterface()
{
    if (!rclcpp::ok()) {
        rclcpp::init(0, nullptr);
    }

    // Construct the executor lazily here, AFTER rclcpp::init() above. The
    // header member is a unique_ptr so the executor's constructor —
    // which requires a live rclcpp context — doesn't run at plugin
    // instantiation time. Without this, sim setups that don't load
    // gz_ros2_control crash here because nothing else has called
    // rclcpp::init() yet.
    ros_executor_ = std::make_unique<rclcpp::executors::SingleThreadedExecutor>();

    rclcpp::NodeOptions opts;
    opts.use_intra_process_comms(false);
    ros_node_ = std::make_shared<rclcpp::Node>(
        "gz_gpu_ouster_lidar", sensor_name_, opts);

    // Use absolute topic names derived from sensor_name_ so that Gazebo's
    // process-level namespace contamination doesn't affect topic routing.
    const std::string abs_prefix = sensor_name_;

    // Latched metadata publisher
    rclcpp::QoS meta_qos{1};
    meta_qos.reliable().transient_local();
    meta_pub_ = ros_node_->create_publisher<std_msgs::msg::String>(
        abs_prefix + "/metadata", meta_qos);

    // Build a QoS from a string keyword. Used for image/camera_info and IMU
    // pubs so deployments can match whatever their consumer expects on
    // rmw_zenoh_cpp (where pub and sub QoS must match exactly — neither
    // BEST_EFFORT-pub-to-RELIABLE-sub nor the reverse work).
    auto qos_from_string = [](const std::string & kind, size_t depth) -> rclcpp::QoS {
        if (kind == "sensor_data") return rclcpp::SensorDataQoS();
        rclcpp::QoS q{depth};
        if (kind == "best_effort") q.best_effort();
        else                        q.reliable();   // "reliable" or unknown
        return q.keep_last(depth);
    };

    // Packet publisher: SensorDataQoS (BEST_EFFORT). High-rate raw stream;
    // a dropped packet is recoverable by os_cloud and never blocks the
    // realtime path. Not user-overridable.
    const auto pkt_qos = rclcpp::SensorDataQoS();
    pkt_pub_ = ros_node_->create_publisher<ouster_sensor_msgs::msg::PacketMsg>(
        abs_prefix + "/lidar_packets", pkt_qos);

    // Image + camera_info: configurable via <image_qos> SDF tag, defaults
    // RELIABLE KEEP_LAST(5) to match RViz / rqt_image_view / image_transport
    // default subscribers. Override to "best_effort" if your consumer is
    // Foxglove configured for sensor data.
    const auto image_qos = qos_from_string(image_qos_, 5);
    range_image_pub_ = ros_node_->create_publisher<sensor_msgs::msg::Image>(
        abs_prefix + "/range_image", image_qos);
    signal_image_pub_ = ros_node_->create_publisher<sensor_msgs::msg::Image>(
        abs_prefix + "/signal_image", image_qos);
    reflec_image_pub_ = ros_node_->create_publisher<sensor_msgs::msg::Image>(
        abs_prefix + "/reflec_image", image_qos);
    nearir_image_pub_ = ros_node_->create_publisher<sensor_msgs::msg::Image>(
        abs_prefix + "/nearir_image", image_qos);

    // CameraInfo publisher (equirectangular projection model for range images)
    camera_info_pub_ = ros_node_->create_publisher<sensor_msgs::msg::CameraInfo>(
        abs_prefix + "/camera_info", image_qos);
    {
        const uint32_t H = static_cast<uint32_t>(H_);
        const uint32_t W = static_cast<uint32_t>(W_);

        // Horizontal: each column = 2π/W radians (full rotation; azimuth window
        // only affects data validity, not the column ↔ angle mapping).
        const double fx = static_cast<double>(W) / (2.0 * M_PI);
        const double cx = static_cast<double>(W) / 2.0;

        // Vertical: use actual beam altitude angles for correct VFOV.
        double fy, cy;
        if (beam_alt_angles_.size() >= 2) {
            auto [min_it, max_it] = std::minmax_element(
                beam_alt_angles_.begin(), beam_alt_angles_.end());
            const double vfov_rad = (*max_it - *min_it) * M_PI / 180.0;
            fy = static_cast<double>(H) / vfov_rad;
            const double mean_alt_rad = 0.5 * (*max_it + *min_it) * M_PI / 180.0;
            cy = static_cast<double>(H) / 2.0 - mean_alt_rad * fy;
        } else {
            fy = static_cast<double>(H) / (2.0 * M_PI);
            cy = static_cast<double>(H) / 2.0;
        }

        camera_info_msg_.header.frame_id = image_frame_id_;
        camera_info_msg_.height = H;
        camera_info_msg_.width = W;
        camera_info_msg_.distortion_model = "equidistant";
        // camera_info_msg_.distortion_model = "plumb_bob";
        camera_info_msg_.k = {fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0};
        camera_info_msg_.d = {0.0, 0.0, 0.0, 0.0, 0.0};
        camera_info_msg_.r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
        camera_info_msg_.p = {fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0,
                              1.0, 0.0};
    }

    // IMU publishers (only if imu_name was provided in SDF). QoS is
    // configurable via <imu_qos>; defaults to sensor_data to match the
    // ouster_ros driver convention so a sim/hardware topic swap doesn't
    // require changing subscriber QoS.
    if (imu_enabled_) {
        const auto imu_qos = qos_from_string(imu_qos_, 10);
        imu_pkt_pub_ = ros_node_->create_publisher<ouster_sensor_msgs::msg::PacketMsg>(
            abs_prefix + "/imu_packets", imu_qos);
        if (publish_imu_msg_) {
            imu_msg_pub_ = ros_node_->create_publisher<sensor_msgs::msg::Imu>(
                abs_prefix + "/imu", imu_qos);
        }
    }

    // ── Declare ROS 2 parameters ──────────────────────────────────────────
    // Read-only structural parameters (informational, cannot change at runtime).
    ros_node_->declare_parameter("lidar_hz", lidar_hz_);
    ros_node_->declare_parameter("max_range", max_range_);
    if (imu_enabled_) {
        ros_node_->declare_parameter("imu_hz", imu_hz_);
    }

    // Dynamically reconfigurable noise model parameters.
    ros_node_->declare_parameter("range_noise_min_std", range_noise_min_std_);
    ros_node_->declare_parameter("range_noise_max_std", range_noise_max_std_);
    ros_node_->declare_parameter("signal_noise_scale", signal_noise_scale_);
    ros_node_->declare_parameter("nearir_noise_scale", nearir_noise_scale_);
    ros_node_->declare_parameter("dropout_rate_close", dropout_rate_close_);
    ros_node_->declare_parameter("dropout_rate_far", dropout_rate_far_);
    ros_node_->declare_parameter("edge_discon_threshold", edge_discon_threshold_);
    ros_node_->declare_parameter("base_signal", base_signal_);
    ros_node_->declare_parameter("base_reflectivity", base_reflectivity_);
    if (imu_enabled_) {
        ros_node_->declare_parameter("gyro_noise_std",   gyro_noise_std_);
        ros_node_->declare_parameter("accel_noise_std",  accel_noise_std_);
        ros_node_->declare_parameter("gyro_bias_walk",   gyro_bias_walk_);
        ros_node_->declare_parameter("accel_bias_walk",  accel_bias_walk_);
    }

    param_cb_handle_ = ros_node_->add_on_set_parameters_callback(
        [this](const std::vector<rclcpp::Parameter> & params)
            -> rcl_interfaces::msg::SetParametersResult {
            for (const auto & p : params) {
                const auto & name = p.get_name();
                // Reject changes to structural parameters.
                if (name == "lidar_hz" || name == "max_range" || name == "imu_hz") {
                    RCLCPP_WARN(kLogger,
                        "rejecting runtime change to '%s' (structural; "
                        "set in SDF instead)", name.c_str());
                    rcl_interfaces::msg::SetParametersResult r;
                    r.successful = false;
                    r.reason = name + " cannot be changed at runtime";
                    return r;
                }
            }
            // Apply noise model updates under lock (sim thread reads these).
            {
                std::lock_guard<std::mutex> lk(noise_mtx_);
                for (const auto & p : params) {
                    const auto & name = p.get_name();
                    if (name == "range_noise_min_std")       range_noise_min_std_     = std::max(0.0, p.as_double());
                    else if (name == "range_noise_max_std")  range_noise_max_std_     = std::max(0.0, p.as_double());
                    else if (name == "signal_noise_scale")   signal_noise_scale_      = std::max(0.0, p.as_double());
                    else if (name == "nearir_noise_scale")   nearir_noise_scale_      = std::max(0.0, p.as_double());
                    else if (name == "dropout_rate_close")   dropout_rate_close_      = std::clamp(p.as_double(), 0.0, 1.0);
                    else if (name == "dropout_rate_far")     dropout_rate_far_        = std::clamp(p.as_double(), 0.0, 1.0);
                    else if (name == "edge_discon_threshold") edge_discon_threshold_   = std::max(0.0, p.as_double());
                    else if (name == "base_signal")          base_signal_             = std::max(0.0, p.as_double());
                    else if (name == "base_reflectivity")    base_reflectivity_       = std::clamp(p.as_double(), 0.0, 255.0);
                    else if (name == "gyro_noise_std")       gyro_noise_std_          = std::max(0.0, p.as_double());
                    else if (name == "accel_noise_std")      accel_noise_std_         = std::max(0.0, p.as_double());
                    else if (name == "gyro_bias_walk")       gyro_bias_walk_          = std::max(0.0, p.as_double());
                    else if (name == "accel_bias_walk")      accel_bias_walk_         = std::max(0.0, p.as_double());
                }
            }
            rcl_interfaces::msg::SetParametersResult r;
            r.successful = true;
            return r;
        });

    // Spin the node on a background thread so Zenoh completes peer discovery.
    // Without this, rmw_zenoh_cpp never processes incoming control messages
    // and publish() calls go undelivered even when subscribers exist.
    ros_executor_->add_node(ros_node_);
    ros_spin_thread_ = std::thread([this]() {
        ros_executor_->spin();
    });

    // Publish initial metadata AFTER executor is spinning.  rmw_zenoh_cpp
    // requires the executor thread to pump outgoing control messages before
    // publish() will actually deliver data to peers.  100 ms is enough for
    // the executor to complete its first spin cycle and for Zenoh peer
    // discovery to settle on a local router (zenohd runs on the same host).
    // This is a one-time startup cost; subsequent metadata republishing is
    // handled in OnRender() with a subscriber-count check.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std_msgs::msg::String meta_msg;
    meta_msg.data = metadata_str_;
    meta_pub_->publish(meta_msg);
    RCLCPP_INFO(kLogger, "Initial metadata published on %s/metadata", abs_prefix.c_str());
}

// ── Rendering-thread callback ────────────────────────────────────────────────
// Fired by the Sensors system on its rendering thread (events::Render, line 496
// in Sensors.cc) AFTER scene->PreRender() has been called.  All gz::rendering
// calls happen here (never from PostUpdate).

void GzGpuOusterLidarSystem::OnRender()
{
    // See render_busy_mtx_ comment in the header: hold the barrier for the
    // entire render-thread callback so the dtor can't race us into freed
    // members. shutdown_ check is the cheap early-out for the case where
    // the dtor set the flag and is now waiting on this same lock.
    std::lock_guard<std::recursive_mutex> render_lk(render_busy_mtx_);
    if (shutdown_.load(std::memory_order_acquire)) return;

    // Record that events::Render fired at least once. PostUpdate() reads this
    // to emit a one-shot diagnostic when the Sensors system never starts
    // rendering (no rendering sensor in the world); see PostUpdate().
    onrender_entries_.fetch_add(1, std::memory_order_relaxed);

    if (sensor_initialized_.load(std::memory_order_acquire)) {
        // Republish metadata periodically until os_cloud confirms receipt.
        // rmw_zenoh_cpp transient_local replay is unreliable across processes,
        // and get_subscription_count() may lag behind actual Zenoh peer state.
        // Republish every ~1s (assuming render ticks at sensor hz) until we
        // see a subscriber AND have published at least a few times.
        const auto meta_subs = meta_pub_->get_subscription_count();
        if (metadata_published_ && meta_subs == 0) {
            // Subscriber dropped (Foxglove tab refresh, os_cloud restart,
            // network blip). transient_local replay won't reliably re-deliver
            // the latched metadata when the sub reconnects on rmw_zenoh, so
            // re-arm the republish loop instead of staying latched forever.
            metadata_published_ = false;
            metadata_pub_count_ = 0;
            RCLCPP_INFO(kLogger,
                "metadata subscriber count dropped to 0; re-arming republish");
        }
        if (!metadata_published_) {
            ++metadata_pub_count_;
            // Throttle: republish roughly every 1 second worth of render ticks.
            // Clamp ticks-per-second to >= 1 to avoid division by zero when lidar_hz < 1.
            const int ticks_per_sec = std::max(1, static_cast<int>(lidar_hz_));
            if (metadata_pub_count_ <= 5 || metadata_pub_count_ % ticks_per_sec == 0) {
                std_msgs::msg::String meta_msg;
                meta_msg.data = metadata_str_;
                std::lock_guard<std::mutex> pub_lk(publish_mtx_);
                meta_pub_->publish(meta_msg);
            }
            // Only stop after subscriber detected AND we've sent enough for Zenoh to settle
            if (meta_subs > 0 && metadata_pub_count_ > ticks_per_sec * 2) {
                metadata_published_ = true;
                RCLCPP_INFO(kLogger, "metadata delivered to os_cloud (after %d publishes)",
                    metadata_pub_count_.load());
            }
        }

        // ── Throttle to lidar_hz_ ───────────────────────────────────────────
        auto now = std::chrono::steady_clock::now();
        const auto period = std::chrono::duration_cast<
            std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(1.0 / lidar_hz_));
        if (now - last_render_time_ < period) return;
        last_render_time_ = now;

        // Render every panel of the rig at the lidar_frame world pose.
        // Render() does the perspective depth pass; PostRender() reads the
        // buffer back and synchronously fires onPanelFrame, which copies
        // the data into its packed slot in pending_buf_.
        if (!panel_cams_.empty()) {
            ::gz::math::Pose3d pose;
            {
                std::lock_guard<std::mutex> lk(pose_mtx_);
                pose = cached_pose_;
            }
            std::fill(panel_filled_.begin(), panel_filled_.end(), false);
            for (size_t i = 0; i < panel_cams_.size(); ++i) {
                panel_cams_[i]->SetWorldPosition(pose.Pos());
                panel_cams_[i]->SetWorldRotation(pose.Rot() * panel_quats_[i]);
                panel_cams_[i]->Render();
                panel_cams_[i]->PostRender();
            }

            const bool complete = std::all_of(
                panel_filled_.begin(), panel_filled_.end(),
                [](bool b) { return b; });
            if (complete) {
                std::lock_guard<std::mutex> lk(frame_mtx_);
                if (frame_ready_.load(std::memory_order_acquire)) {
                    // Previous frame hasn't been consumed by PostUpdate yet;
                    // we're about to overwrite it. Surface the drop so a
                    // sustained problem (sim-time stall, post-pause burst)
                    // is visible in logs instead of silent.
                    const uint64_t dropped = dropped_frames_.fetch_add(1) + 1;
                    if (ros_node_) {
                        RCLCPP_WARN_THROTTLE(kLogger,
                            *ros_node_->get_clock(), 5000,
                            "%s: dropped rig frame (PostUpdate didn't drain); "
                            "total dropped=%lu",
                            sensor_name_.c_str(),
                            static_cast<unsigned long>(dropped));
                    }
                }
                // O(1) vector swap; capacity is preserved and reused.
                pending_buf_.swap(raw_frame_buf_);
                raw_frame_n_ = layout_.rp.raw_n;
                frame_ready_ = true;
            } else if (ros_node_) {
                RCLCPP_WARN_THROTTLE(kLogger, *ros_node_->get_clock(), 5000,
                    "%s: incomplete panel rig frame (a depth camera did not "
                    "deliver); dropping this scan", sensor_name_.c_str());
            }
        }
        return;
    }

    // ── Lazy panel-rig initialisation ────────────────────────────────────────
    // Guard: metadata must have loaded and the layout must have been built.
    if (beam_alt_angles_.empty() || layout_.n_panels == 0) return;

    // Wait until the Sensors system has created the OGRE2 scene.
    auto * engine = ::gz::rendering::engine("ogre2");
    if (!engine) return;
    if (engine->SceneCount() == 0) return;

    auto scene = engine->SceneByIndex(0);
    if (!scene) return;

    auto root = scene->RootVisual();

    // One perspective depth camera per panel. Each is a plain single-pass
    // render — the Ouster beam model (cylindrical or hemispherical) is
    // applied in the resample kernel, not by the renderer, so there is no
    // cubemap and no equirect intermediate.
    panel_cams_.reserve(static_cast<size_t>(layout_.n_panels));
    panel_conns_.reserve(static_cast<size_t>(layout_.n_panels));
    panel_quats_.reserve(static_cast<size_t>(layout_.n_panels));
    for (int i = 0; i < layout_.n_panels; ++i) {
        const auto & cs = layout_.cams[i];
        auto cam = scene->CreateDepthCamera(
            sensor_name_ + "_panel" + std::to_string(i));
        if (!cam) {
            RCLCPP_ERROR(kLogger,
                "Failed to create depth camera for panel %d", i);
            DestroyPanels();
            return;
        }
        cam->SetImageWidth(cs.width);
        cam->SetImageHeight(cs.height);
        // Square pixels: aspect = w/h reproduces the layout's fx == fy
        // pinhole model exactly.
        cam->SetAspectRatio(
            static_cast<double>(cs.width) / static_cast<double>(cs.height));
        cam->SetHFOV(::gz::math::Angle(cs.hfov_rad));
        cam->SetNearClipPlane(kNearClip);
        cam->SetFarClipPlane(max_range_);
        cam->SetVisibilityMask(visibility_mask_);
        cam->CreateDepthTexture();

        if (root) {
            root->AddChild(cam);
        }

        // Panel orientation relative to the sensor frame. Gazebo's positive
        // pitch tilts the x axis down, so an upward panel axis is -pitch.
        panel_quats_.emplace_back(0.0, -cs.pitch_rad, cs.yaw_rad);
        panel_conns_.push_back(cam->ConnectNewDepthFrame(
            [this, i](const float * data, unsigned int w, unsigned int h,
                      unsigned int ch, const std::string & /*format*/) {
                onPanelFrame(static_cast<size_t>(i), data, w, h, ch);
            }));
        panel_cams_.push_back(cam);
    }
    panel_filled_.assign(static_cast<size_t>(layout_.n_panels), false);

    sensor_initialized_.store(true, std::memory_order_release);

    RCLCPP_INFO(kLogger,
        "Panel rig created: %d depth cameras, beam altitude span "
        "[%.1f, %.1f] deg, %s model",
        layout_.n_panels, min_alt_, max_alt_,
        layout_.hemispherical ? "hemispherical" : "cylindrical");
}

// ── ISystemPostUpdate ────────────────────────────────────────────────────────

void GzGpuOusterLidarSystem::PostUpdate(
    const ::gz::sim::UpdateInfo & info,
    const ::gz::sim::EntityComponentManager & ecm)
{
    // ── Silent-failure guard ─────────────────────────────────────────────────
    // The plugin attaches its depth-camera rig to the ogre2 scene owned by
    // gz-sim-sensors-system and is driven by the events::Render event. On
    // Gazebo Harmonic the Sensors system only initialises rendering (and emits
    // events::Render) when the world contains at least one *rendering* sensor
    // (camera / gpu_lidar / depth_camera ...). With only non-rendering sensors
    // (altimeter, imu) present, OnRender() never fires and no point cloud is
    // ever produced. Detect that and tell the user exactly what to fix, once.
    if (!no_render_warned_ &&
        !sensor_initialized_.load(std::memory_order_acquire) &&
        onrender_entries_.load(std::memory_order_relaxed) == 0)
    {
        constexpr int64_t kRenderWaitNs = 2'000'000'000;  // 2 s of sim time
        const int64_t sim_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                info.simTime).count();
        if (sim_ns > kRenderWaitNs) {
            no_render_warned_ = true;
            RCLCPP_ERROR(kLogger,
                "events::Render has not fired after %.1fs of sim time — gz-sim's "
                "Sensors system has not started rendering. This plugin attaches "
                "its depth-camera rig to the ogre2 scene owned by gz-sim-sensors-system, "
                "which only initialises rendering when the world contains at "
                "least one rendering sensor (camera or gpu_lidar). Add a "
                "rendering sensor (the example URDF's anchor_type defaults to a "
                "camera) or no point cloud will be produced.",
                static_cast<double>(sim_ns) / 1e9);
        }
    }

    if (info.paused) {
        was_paused_ = true;
        return;
    }

    // Reset timing state after sim resumes to prevent stale data burst.
    if (was_paused_) {
        was_paused_ = false;
        last_imu_sim_time_ = std::chrono::duration_cast<std::chrono::nanoseconds>(info.simTime);
        last_render_time_ = std::chrono::steady_clock::now();
    }

    if (!sensor_initialized_.load(std::memory_order_acquire)) return;

    // ── Locate the gpu_lidar sensor entity (once) ────────────────────────────
    // Gazebo merges fixed-joint child links into the parent, so
    // "lidar0/lidar_frame" does not exist as a Link entity.  The <sensor>
    // element placed on that frame DOES survive and carries the accumulated
    // fixed-joint pose, giving us the correct world position.
    if (!lidar_frame_found_) {
        ecm.Each<::gz::sim::components::Name, ::gz::sim::components::Sensor>(
            [this](const ::gz::sim::Entity & ent,
                   const ::gz::sim::components::Name * name,
                   const ::gz::sim::components::Sensor *) -> bool {
                if (name->Data() == lidar_frame_name_) {
                    lidar_frame_entity_ = ent;
                    lidar_frame_found_ = true;
                    RCLCPP_INFO(kLogger, "Found sensor entity: %s (id=%lu)",
                        lidar_frame_name_.c_str(), static_cast<unsigned long>(ent));
                    return false;  // stop iteration
                }
                return true;  // continue
            });
    }

    // ── Locate the IMU sensor entity (once) ────────────────────────────────
    if (imu_enabled_ && !imu_entity_found_) {
        const bool auto_detect = (imu_name_ == "auto");

        // Scope auto-detect to the LiDAR's top-level model so a sibling
        // model's IMU (or a sensor named "gimbal_imu") cannot silently bind.
        ::gz::sim::Entity lidar_model = ::gz::sim::kNullEntity;
        if (auto_detect) {
            lidar_model = ::gz::sim::topLevelModel(sensor_entity_, ecm);
        }

        std::vector<std::pair<std::string, ::gz::sim::Entity>> candidates;
        ecm.Each<::gz::sim::components::Name, ::gz::sim::components::Sensor>(
            [&](const ::gz::sim::Entity & ent,
                const ::gz::sim::components::Name * name,
                const ::gz::sim::components::Sensor *) -> bool {
                const auto & n = name->Data();
                if (auto_detect) {
                    const bool name_match =
                        (n.find("imu") != std::string::npos ||
                         n.find("IMU") != std::string::npos);
                    if (!name_match) return true;
                    if (lidar_model != ::gz::sim::kNullEntity) {
                        const auto cand_model =
                            ::gz::sim::topLevelModel(ent, ecm);
                        if (cand_model != lidar_model) return true;
                    }
                } else if (n != imu_name_) {
                    return true;
                }
                candidates.emplace_back(n, ent);
                return true;
            });

        if (!candidates.empty()) {
            if (auto_detect && candidates.size() > 1) {
                std::string names;
                for (const auto & c : candidates) {
                    names += " "; names += c.first;
                }
                RCLCPP_WARN(kLogger,
                    "IMU auto-detect found %zu candidates; picking '%s'."
                    " Candidates:%s",
                    candidates.size(), candidates.front().first.c_str(),
                    names.c_str());
            }
            imu_entity_       = candidates.front().second;
            imu_entity_found_ = true;
            imu_name_         = candidates.front().first;
            RCLCPP_INFO(kLogger, "Found IMU entity: %s (id=%lu)",
                imu_name_.c_str(),
                static_cast<unsigned long>(imu_entity_));
        }
    }

    // ── Cache world pose for the rendering thread ────────────────────────────
    if (lidar_frame_found_) {
        auto worldPose = ::gz::sim::worldPose(lidar_frame_entity_, ecm);
        std::lock_guard<std::mutex> lk(pose_mtx_);
        cached_pose_ = worldPose;
    }

    // ── Process any pending frame ────────────────────────────────────────────
    // Sim-side of the triple buffer: under the lock just swap the
    // raw_frame_buf_ slot into our sim-only process_buf_. encodeAndPublish
    // (which can take many ms on dense sensors) runs without holding
    // frame_mtx_, so the render thread is free to memcpy a new frame
    // concurrently. process_buf_ keeps its capacity across calls.
    bool have_frame = false;
    int local_n = 0;
    {
        std::lock_guard<std::mutex> lk(frame_mtx_);
        have_frame = frame_ready_;
        frame_ready_ = false;
        if (have_frame) {
            process_buf_.swap(raw_frame_buf_);
            local_n = raw_frame_n_;
        }
    }

    if (have_frame && !process_buf_.empty()) {
        const auto stamp_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                info.simTime).count();
        encodeAndPublish(stamp_ns, process_buf_.data(), local_n);
    }

    // ── Publish IMU at configured rate ──────────────────────────────────────
    if (imu_enabled_ && imu_entity_found_) {
        publishImu(info, ecm);
    }
}

// ── Panel depth frame callback ──────────────────────────────────────────────

void GzGpuOusterLidarSystem::onPanelFrame(
    size_t panel, const float * data,
    unsigned int width, unsigned int height, unsigned int channels)
{
    // See render_busy_mtx_ comment in the header. This callback is signaled
    // synchronously by the panel camera's PostRender(), so it runs nested on
    // the render thread inside OnRender() which already holds this lock —
    // hence the lock is recursive. Re-taking it here keeps the dtor's
    // shutdown barrier honest for the (defensive) case of an out-of-band
    // signal, and is a no-op cost in the normal nested path.
    std::lock_guard<std::recursive_mutex> render_lk(render_busy_mtx_);
    if (shutdown_.load(std::memory_order_acquire)) return;

    if (!data || width == 0 || height == 0 || channels == 0) return;
    if (panel >= static_cast<size_t>(layout_.n_panels)) return;

    // Defensive: the resampler's panel intrinsics assume exactly the layout
    // dimensions. If OGRE2 ever hands back a differently-sized frame the
    // projection would silently mis-sample, so drop it loudly instead.
    const auto & cs = layout_.cams[panel];
    if (width != static_cast<unsigned int>(cs.width) ||
        height != static_cast<unsigned int>(cs.height)) {
        if (ros_node_) {
            RCLCPP_ERROR_THROTTLE(kLogger, *ros_node_->get_clock(), 5000,
                "%s: panel %zu frame %ux%u (expected %dx%d); dropping",
                sensor_name_.c_str(), panel, width, height,
                cs.width, cs.height);
        }
        return;
    }

    RCLCPP_INFO_ONCE(kLogger,
        "%s: panel depth frames flowing (%ux%ux%u for panel 0 of %d)",
        sensor_name_.c_str(), width, height, channels, layout_.n_panels);

    // Copy into this panel's packed slot in the render-only pending_buf_,
    // no lock held. After warmup the resize is a no-op (capacity is kept
    // across the triple-buffer swaps) and the memcpy is the only cost.
    const size_t n = static_cast<size_t>(width) * height;
    if (pending_buf_.size() < static_cast<size_t>(layout_.rp.raw_n)) {
        pending_buf_.resize(static_cast<size_t>(layout_.rp.raw_n));
    }
    float * dst = pending_buf_.data() + layout_.rp.panels[panel].offset;
    if (channels == 1) {
        std::memcpy(dst, data, n * sizeof(float));
    } else {
        // Defensive stride copy if a depth implementation delivers packed
        // multi-channel data; channel 0 is depth.
        for (size_t i = 0; i < n; ++i) {
            dst[i] = data[i * channels];
        }
    }
    panel_filled_[panel] = true;
}

// ── Encode depth → Ouster packets ───────────────────────────────────────────

void GzGpuOusterLidarSystem::encodeAndPublish(
    int64_t stamp_ns,
    const float * raw_data, int raw_n)
{
    if (stamp_ns <= 0) return;
    if (!pw_ || pkt_buf_.empty()) return;
    if (beam_alt_angles_.empty() || H_ <= 0 || W_ <= 0 || cpp_ <= 0) return;
    if (!ray_processor_) return;
    if (raw_n != layout_.rp.raw_n) {
        RCLCPP_ERROR_THROTTLE(kLogger, *ros_node_->get_clock(), 5000,
            "%s: raw frame size %d != rig size %d; dropping",
            sensor_name_.c_str(), raw_n, layout_.rp.raw_n);
        return;
    }

    // ── Snapshot noise parameters (may be updated by ROS param callback) ───
    double snap_range_noise_min_std, snap_range_noise_max_std;
    double snap_signal_noise_scale, snap_nearir_noise_scale;
    double snap_dropout_rate_close, snap_dropout_rate_far;
    double snap_edge_discon_threshold, snap_base_signal, snap_base_reflectivity;
    double snap_max_range;
    {
        std::lock_guard<std::mutex> lk(noise_mtx_);
        snap_range_noise_min_std  = range_noise_min_std_;
        snap_range_noise_max_std  = range_noise_max_std_;
        snap_signal_noise_scale   = signal_noise_scale_;
        snap_nearir_noise_scale   = nearir_noise_scale_;
        snap_dropout_rate_close   = dropout_rate_close_;
        snap_dropout_rate_far     = dropout_rate_far_;
        snap_edge_discon_threshold = edge_discon_threshold_;
        snap_base_signal          = base_signal_;
        snap_base_reflectivity    = base_reflectivity_;
        snap_max_range            = max_range_;
    }

    // ── Resample params ──────────────────────────────────────────────────────
    // Panel geometry, intrinsics and packed offsets were computed once in
    // Configure (buildOusterPanelLayout); far_clip/beam_origin set there too.
    const ResampleParams & rp = layout_.rp;

    // ── Noise params ─────────────────────────────────────────────────────────
    RayProcessParams pp;
    pp.H = H_;
    pp.W = W_;
    pp.base_signal = static_cast<float>(snap_base_signal);
    pp.base_reflectivity = static_cast<float>(snap_base_reflectivity);
    pp.range_noise_min_std = static_cast<float>(snap_range_noise_min_std);
    pp.range_noise_max_std = static_cast<float>(snap_range_noise_max_std);
    pp.max_range = static_cast<float>(snap_max_range);
    pp.signal_noise_scale = static_cast<float>(snap_signal_noise_scale);
    pp.nearir_noise_scale = static_cast<float>(snap_nearir_noise_scale);
    pp.dropout_rate_close = static_cast<float>(snap_dropout_rate_close);
    pp.dropout_rate_far = static_cast<float>(snap_dropout_rate_far);
    pp.edge_discon_threshold = static_cast<float>(snap_edge_discon_threshold);

    // ── GPU pipeline: resample → noise → channel outputs ─────────────────────
    if (!memory_logged_) {
        // One-time per-sensor accounting after first frame, when the
        // dispatched backend has settled on actual buffer sizes. Helps
        // diagnose multi-sensor OOMs — a dense sensor (4096×512) plus
        // curand state can easily exceed 100 MB of VRAM on its own.
        const size_t raw_bytes  = static_cast<size_t>(rp.raw_n) * sizeof(float);
        const size_t channel_bytes =
            static_cast<size_t>(H_) * W_ *
            (sizeof(uint32_t) + sizeof(uint16_t) + sizeof(uint8_t) + sizeof(uint16_t));
        const size_t resample_bytes = static_cast<size_t>(H_) * W_ * sizeof(float);
        // curandState (XORWOW) is 48 B; hiprand similar; SYCL counter-based
        // RNG is stateless. Approximate with 48 B to avoid backend coupling.
        const size_t rand_bytes = noiseEnabled(pp)
            ? static_cast<size_t>(H_) * W_ * 48 : 0;
        const size_t total = raw_bytes + channel_bytes + resample_bytes + rand_bytes;
        RCLCPP_INFO(kLogger,
            "%s: GPU buffers ~%.1f MiB (%s backend) — raw=%.1f channels=%.1f "
            "resample=%.1f rand=%.1f",
            sensor_name_.c_str(), total / 1048576.0,
            ray_processor_->backendName(),
            raw_bytes / 1048576.0,
            channel_bytes / 1048576.0,
            resample_bytes / 1048576.0,
            rand_bytes / 1048576.0);
        memory_logged_ = true;
    }
    ray_processor_->processRaw(
        raw_data,
        beam_alt_f_.data(),
        beam_az_f_.data(),
        rp,
        range_buf_.data(),
        signal_buf_.data(),
        reflectivity_buf_.data(),
        nearir_buf_.data(),
        pp);

    // ── Build packets ────────────────────────────────────────────────────────
    const int n_packets = W_ / cpp_;
    const int64_t scan_period_ns = static_cast<int64_t>(1e9 / lidar_hz_);
    const int64_t scan_start_ns  = std::max(int64_t{0}, stamp_ns - scan_period_ns);

    // Per-column timestamps: round (col * period / W) on the full numerator
    // rather than accumulating col * (period / W). At 10 Hz × 1024 cols the
    // truncated form drops 256 ns per scan and drifts over long bag captures;
    // computing on the full multiply restores the last column to the scan end.

    // Map raw buffers into Eigen for PacketWriter
    using RangeMatrix = Eigen::Map<ouster::sdk::core::img_t<uint32_t>>;
    using SignalMatrix = Eigen::Map<ouster::sdk::core::img_t<uint16_t>>;
    using ReflMatrix = Eigen::Map<ouster::sdk::core::img_t<uint8_t>>;
    using NirMatrix = Eigen::Map<ouster::sdk::core::img_t<uint16_t>>;

    RangeMatrix  range_mat(range_buf_.data(), H_, W_);
    SignalMatrix  signal_mat(signal_buf_.data(), H_, W_);
    ReflMatrix    refl_mat(reflectivity_buf_.data(), H_, W_);
    NirMatrix     nearir_mat(nearir_buf_.data(), H_, W_);

    std::vector<ouster_sensor_msgs::msg::PacketMsg> new_pkts(static_cast<size_t>(n_packets));

    for (int p = 0; p < n_packets; ++p) {
        std::memset(pkt_buf_.data(), 0, pkt_buf_.size());

        const int col_start = p * cpp_;
        pw_->set_frame_id(pkt_buf_.data(), frame_id_);

        for (int c_local = 0; c_local < cpp_; ++c_local) {
            const int col_global = col_start + c_local;
            uint8_t * col = pw_->nth_col(c_local, pkt_buf_.data());
            const int64_t col_ts = scan_start_ns +
                (static_cast<int64_t>(col_global) * scan_period_ns) / W_;
            pw_->set_col_timestamp(col, static_cast<uint64_t>(col_ts));
            pw_->set_col_measurement_id(col, static_cast<uint16_t>(col_global));
            pw_->set_col_status(col, 0x01u);
        }

        pw_->set_block<uint32_t>(range_mat.data(),  W_, ouster::sdk::core::ChanField::RANGE,        pkt_buf_.data());
        pw_->set_block<uint16_t>(signal_mat.data(), W_, ouster::sdk::core::ChanField::SIGNAL,       pkt_buf_.data());
        pw_->set_block<uint8_t> (refl_mat.data(),   W_, ouster::sdk::core::ChanField::REFLECTIVITY, pkt_buf_.data());
        pw_->set_block<uint16_t>(nearir_mat.data(), W_, ouster::sdk::core::ChanField::NEAR_IR,      pkt_buf_.data());

        new_pkts[static_cast<size_t>(p)].buf.assign(pkt_buf_.begin(), pkt_buf_.end());
    }

    ++frame_id_;

    // ── Publish image topics ─────────────────────────────────────────────────
    publishImages(stamp_ns);

    // ── Wake drain thread ────────────────────────────────────────────────────
    {
        std::lock_guard<std::mutex> lk(drain_mtx_);
        drain_pkts_.swap(new_pkts);
        drain_ready_.store(true, std::memory_order_release);
    }
    drain_cv_.notify_one();
}

// ── Publish sensor images ────────────────────────────────────────────────────

void GzGpuOusterLidarSystem::publishImages(int64_t stamp_ns)
{
    const auto n = static_cast<size_t>(H_ * W_);
    const uint32_t h = static_cast<uint32_t>(H_);
    const uint32_t w = static_cast<uint32_t>(W_);
    const uint32_t step = w * static_cast<uint32_t>(sizeof(uint16_t));

    builtin_interfaces::msg::Time stamp;
    stamp.sec  = static_cast<int32_t>(stamp_ns / 1000000000LL);
    stamp.nanosec = static_cast<uint32_t>(stamp_ns % 1000000000LL);

    auto make_image = [&]() -> sensor_msgs::msg::Image {
        sensor_msgs::msg::Image msg;
        msg.header.stamp = stamp;
        msg.header.frame_id = image_frame_id_;
        msg.height = h;
        msg.width = w;
        msg.encoding = "mono16";
        msg.is_bigendian = false;
        msg.step = step;
        msg.data.resize(n * sizeof(uint16_t));
        return msg;
    };

    // Range image: mm → 4mm-resolution uint16 (matches ouster_ros os_image)
    // Values > 65535 (≈262 m) are clamped to 0 (no return).
    if (range_image_pub_->get_subscription_count() > 0) {
        auto msg = make_image();
        auto * pixels = reinterpret_cast<uint16_t *>(msg.data.data());
        for (size_t i = 0; i < n; ++i) {
            const uint32_t r = (range_buf_[i] + 2u) >> 2;
            pixels[i] = (r > 65535u) ? 0u : static_cast<uint16_t>(r);
        }
        std::lock_guard<std::mutex> pub_lk(publish_mtx_);
        range_image_pub_->publish(std::move(msg));
    }

    // Signal image: uint16 values from CUDA processing (photon counts)
    if (signal_image_pub_->get_subscription_count() > 0) {
        auto msg = make_image();
        std::memcpy(msg.data.data(), signal_buf_.data(), n * sizeof(uint16_t));
        std::lock_guard<std::mutex> pub_lk(publish_mtx_);
        signal_image_pub_->publish(std::move(msg));
    }

    // Reflectivity image: uint8 → uint16  (×257 maps [0,255] → [0,65535])
    if (reflec_image_pub_->get_subscription_count() > 0) {
        auto msg = make_image();
        auto * pixels = reinterpret_cast<uint16_t *>(msg.data.data());
        for (size_t i = 0; i < n; ++i) {
            pixels[i] = static_cast<uint16_t>(reflectivity_buf_[i]) * 257u;
        }
        std::lock_guard<std::mutex> pub_lk(publish_mtx_);
        reflec_image_pub_->publish(std::move(msg));
    }

    // Near-IR image: produced by CUDA kernel with Poisson noise via nearir_noise_scale.
    if (nearir_image_pub_->get_subscription_count() > 0) {
        auto msg = make_image();
        std::memcpy(msg.data.data(), nearir_buf_.data(), n * sizeof(uint16_t));
        std::lock_guard<std::mutex> pub_lk(publish_mtx_);
        nearir_image_pub_->publish(std::move(msg));
    }

    // CameraInfo: publish with matching timestamp for image_transport sync.
    if (camera_info_pub_->get_subscription_count() > 0) {
        camera_info_msg_.header.stamp = stamp;
        std::lock_guard<std::mutex> pub_lk(publish_mtx_);
        camera_info_pub_->publish(camera_info_msg_);
    }
}

// ── IMU publishing ──────────────────────────────────────────────────────────

void GzGpuOusterLidarSystem::publishImu(
    const ::gz::sim::UpdateInfo & info,
    const ::gz::sim::EntityComponentManager & ecm)
{
    if (!pw_) return;

    // ── Rate limiting (sim time) ─────────────────────────────────────────
    const auto sim_now = std::chrono::duration_cast<std::chrono::nanoseconds>(info.simTime);
    const auto imu_period = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(1.0 / imu_hz_));

    if (sim_now - last_imu_sim_time_ < imu_period) return;
    last_imu_sim_time_ = sim_now;

    // ── Read IMU data from ECM ───────────────────────────────────────────
    auto * angVelComp = ecm.Component<::gz::sim::components::AngularVelocity>(imu_entity_);
    auto * linAccComp = ecm.Component<::gz::sim::components::LinearAcceleration>(imu_entity_);

    if (!angVelComp || !linAccComp) return;  // IMU data not yet available

    const auto & av = angVelComp->Data();   // rad/s, body frame (gyro — unchanged)
    const auto & la_raw = linAccComp->Data();   // m/s², body frame, KINEMATIC only

    // ── Proper-acceleration compensation ─────────────────────────────────
    // `components::LinearAcceleration` is the body's coordinate (kinematic)
    // linear acceleration in its own frame — NOT what a real accelerometer
    // reads. An accelerometer measures *proper* acceleration:
    //
    //    a_proper = a_kinematic - g_body
    //
    // where g_body is the world gravity vector rotated into the body frame.
    // For a stationary rover, a_kinematic = 0 and a_proper = -g_body ≈
    // (0, 0, +9.81), which matches what a real Ouster IMU reports.
    //
    // Without this step the rover appears to be in free fall (la = 0),
    // Madgwick cannot find a gravity reference, and downstream localization
    // (Sierra, robot_localization, etc.) never publishes odom->base_footprint.
    const auto imu_world_pose = ::gz::sim::worldPose(imu_entity_, ecm);
    const auto & R = imu_world_pose.Rot();
    const ::gz::math::Vector3d gravity_world(0.0, 0.0, -9.80665);
    const auto gravity_body = R.RotateVectorReverse(gravity_world);
    const ::gz::math::Vector3d la_proper = la_raw - gravity_body;

    // ── IMU noise + bias model ───────────────────────────────────────────
    // Math lives in cuda/imu_noise.{hpp,cpp} so it's testable without
    // spinning Gazebo. Defaults match Ouster Os1 datasheet; downstream
    // filters that subscribe to /imu now see a non-pristine signal.
    if (!imu_rng_seeded_) {
        imu_rng_.seed(deriveNonDeterministicSeed(this));
        imu_rng_seeded_ = true;
    }
    double snap_gyro_noise, snap_accel_noise, snap_gyro_walk, snap_accel_walk;
    {
        std::lock_guard<std::mutex> lk(noise_mtx_);
        snap_gyro_noise  = gyro_noise_std_;
        snap_accel_noise = accel_noise_std_;
        snap_gyro_walk   = gyro_bias_walk_;
        snap_accel_walk  = accel_bias_walk_;
    }
    const Vec3 nominal_av = {av.X(), av.Y(), av.Z()};
    const Vec3 nominal_la = {la_proper.X(), la_proper.Y(), la_proper.Z()};
    const ImuNoiseSample noisy = applyImuNoise(
        nominal_av, nominal_la,
        gyro_bias_, accel_bias_,
        snap_gyro_noise, snap_accel_noise,
        snap_gyro_walk,  snap_accel_walk,
        1.0 / imu_hz_,
        imu_rng_);
    const ::gz::math::Vector3d av_meas(noisy.av.x, noisy.av.y, noisy.av.z);
    const ::gz::math::Vector3d la(noisy.la.x, noisy.la.y, noisy.la.z);
    const double gyro_white  = noisy.gyro_white_std;   // for covariance below
    const double accel_white = noisy.accel_white_std;

    const int64_t stamp_ns = sim_now.count();

    // ── Encode Ouster IMU PacketMsg ──────────────────────────────────────
    if (imu_pkt_pub_ && imu_pkt_pub_->get_subscription_count() > 0 &&
        !imu_pkt_buf_.empty()) {
        std::memset(imu_pkt_buf_.data(), 0, imu_pkt_buf_.size());
        uint8_t * buf = imu_pkt_buf_.data();
        const uint64_t ts = static_cast<uint64_t>(stamp_ns);

        // Dispatch on packet size — PacketWriter doesn't expose the profile.
        //   LEGACY (48 bytes):      write sys_ts/accel_ts/gyro_ts directly;
        //                           the SDK has no setter for these fields.
        //                           os_cloud reads gyro_ts (offset 16) for
        //                           the ROS timestamp.
        //   ACCEL32_GYRO32_NMEA:    use set_imu_nmea_ts.
        // Using exclusive branches prevents the NMEA setter from stomping on
        // the LEGACY offsets (and vice versa) when the other profile is in
        // use.
        constexpr size_t kLegacyImuSize = 48;
        if (imu_pkt_buf_.size() == kLegacyImuSize) {
            std::memcpy(buf + 0,  &ts, sizeof(uint64_t));  // sys_ts
            std::memcpy(buf + 8,  &ts, sizeof(uint64_t));  // accel_ts
            std::memcpy(buf + 16, &ts, sizeof(uint64_t));  // gyro_ts
        } else {
            pw_->set_imu_nmea_ts(buf, ts);
        }

        // Accel/gyro values — PacketWriter writes at profile-correct offsets.
        pw_->set_imu_la_x(buf, static_cast<float>(la.X()));
        pw_->set_imu_la_y(buf, static_cast<float>(la.Y()));
        pw_->set_imu_la_z(buf, static_cast<float>(la.Z()));
        pw_->set_imu_av_x(buf, static_cast<float>(av_meas.X()));
        pw_->set_imu_av_y(buf, static_cast<float>(av_meas.Y()));
        pw_->set_imu_av_z(buf, static_cast<float>(av_meas.Z()));

        ouster_sensor_msgs::msg::PacketMsg pkt;
        pkt.buf.assign(imu_pkt_buf_.begin(), imu_pkt_buf_.end());
        {
            std::lock_guard<std::mutex> pub_lk(publish_mtx_);
            imu_pkt_pub_->publish(std::move(pkt));
        }
    }

    // ── Publish sensor_msgs/Imu for convenience ─────────────────────────
    if (publish_imu_msg_ && imu_msg_pub_ &&
        imu_msg_pub_->get_subscription_count() > 0) {
        sensor_msgs::msg::Imu msg;
        msg.header.stamp.sec  = static_cast<int32_t>(stamp_ns / 1000000000LL);
        msg.header.stamp.nanosec = static_cast<uint32_t>(stamp_ns % 1000000000LL);
        msg.header.frame_id = imu_frame_id_;

        msg.angular_velocity.x = av_meas.X();
        msg.angular_velocity.y = av_meas.Y();
        msg.angular_velocity.z = av_meas.Z();

        msg.linear_acceleration.x = la.X();
        msg.linear_acceleration.y = la.Y();
        msg.linear_acceleration.z = la.Z();

        // Covariance derived from the actual noise model: diagonal = σ²
        // where σ is the per-sample white-noise standard deviation. Falls
        // back to ouster_ros defaults if the user disabled noise (so REP-145
        // consumers don't see literal zero variances).
        const double gyro_var  = (gyro_white > 0.0)  ? gyro_white  * gyro_white  : 6e-4;
        const double accel_var = (accel_white > 0.0) ? accel_white * accel_white : 0.01;
        msg.angular_velocity_covariance[0] = gyro_var;
        msg.angular_velocity_covariance[4] = gyro_var;
        msg.angular_velocity_covariance[8] = gyro_var;

        msg.linear_acceleration_covariance[0] = accel_var;
        msg.linear_acceleration_covariance[4] = accel_var;
        msg.linear_acceleration_covariance[8] = accel_var;

        // Orientation unknown (per REP-145: first element = -1)
        msg.orientation_covariance[0] = -1.0;

        {
            std::lock_guard<std::mutex> pub_lk(publish_mtx_);
            imu_msg_pub_->publish(std::move(msg));
        }
    }
}

// ── Drain thread ─────────────────────────────────────────────────────────────

void GzGpuOusterLidarSystem::drainThreadFunc()
{
    std::vector<ouster_sensor_msgs::msg::PacketMsg> local_pkts;

    while (!shutdown_.load(std::memory_order_acquire)) {
        {
            std::unique_lock<std::mutex> lk(drain_mtx_);
            drain_cv_.wait(lk, [this] {
                return drain_ready_.load(std::memory_order_acquire) ||
                       shutdown_.load(std::memory_order_acquire);
            });
            if (shutdown_.load(std::memory_order_acquire)) break;
            drain_ready_.store(false, std::memory_order_release);
            local_pkts.swap(drain_pkts_);
        }

        if (local_pkts.empty()) continue;

        // Use absolute deadlines (sleep_until) instead of accumulating
        // sleep_for(spacing) calls. At dense-sensor packet counts the per-
        // packet spacing drops to hundreds of microseconds, where CFS
        // scheduler jitter would round each sleep up and the packets would
        // bunch toward the end of the scan. Anchoring on a fixed t0 lets
        // any individual sleep finish late without pushing the next one.
        const auto period = std::chrono::nanoseconds(
            static_cast<int64_t>(1e9 / lidar_hz_));
        const auto spacing = period / static_cast<int64_t>(local_pkts.size());
        const auto t0 = std::chrono::steady_clock::now();

        try {
            for (size_t i = 0; i < local_pkts.size(); ++i) {
                if (shutdown_.load(std::memory_order_acquire)) return;
                if (i > 0) {
                    std::this_thread::sleep_until(t0 + spacing * static_cast<int64_t>(i));
                }
                {
                    std::lock_guard<std::mutex> pub_lk(publish_mtx_);
                    pkt_pub_->publish(local_pkts[i]);
                }
            }
        } catch (const std::exception & e) {
            RCLCPP_ERROR(kLogger, "drainThread publish failed: %s", e.what());
        }
    }
}

void GzGpuOusterLidarSystem::DestroyPanels()
{
    if (panel_cams_.empty()) return;

    // Disconnect frame callbacks before destroying the cameras. Safe here:
    // callbacks only fire from OnRender's PostRender calls, and the caller
    // (dtor) has already disconnected the render event and flushed the
    // render-busy barrier.
    panel_conns_.clear();

    auto * engine = ::gz::rendering::engine("ogre2");
    if (engine && engine->SceneCount() > 0) {
        auto scene = engine->SceneByIndex(0);
        if (scene) {
            for (auto & cam : panel_cams_) {
                if (cam) scene->DestroySensor(cam);
            }
        }
    }

    panel_cams_.clear();
    panel_quats_.clear();
    panel_filled_.clear();
}

}  // namespace gz_gpu_ouster_lidar
