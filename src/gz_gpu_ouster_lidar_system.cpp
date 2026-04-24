// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#include "gz_gpu_ouster_lidar/gz_gpu_ouster_lidar_system.hpp"
#include "gz_gpu_ouster_lidar/cuda_ray_processor.hpp"

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
#include <gz/rendering/GpuRays.hh>
#include <gz/rendering/RenderEngine.hh>
#include <gz/rendering/RenderingIface.hh>
#include <gz/rendering/Scene.hh>
#include <gz/sim/rendering/Events.hh>

#include <ouster/metadata.h>
#include <ouster/impl/packet_writer.h>
#include <ouster/types.h>

#include <rclcpp/rclcpp.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <builtin_interfaces/msg/time.hpp>
#include <ouster_sensor_msgs/msg/packet_msg.hpp>
#include <std_msgs/msg/string.hpp>

namespace gz_gpu_ouster_lidar {

static const rclcpp::Logger kLogger = rclcpp::get_logger("gz_gpu_ouster_lidar");

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
    render_conn_.reset();
    frame_conn_.reset();
    DestroyGpuRays();

    shutdown_.store(true, std::memory_order_release);
    drain_cv_.notify_all();
    if (drain_thread_.joinable()) {
        drain_thread_.join();
    }
    ros_executor_.cancel();
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
        cuda_processor_ = std::make_unique<CudaRayProcessor>();
    } catch (const std::exception & e) {
        RCLCPP_ERROR(kLogger,
            "Ray processor init failed: %s; plugin disabled", e.what());
        return;
    }
    if (cuda_processor_->usesCpuFallback()) {
        RCLCPP_WARN(kLogger,
            "No CUDA-capable device detected; gz_gpu_ouster_lidar running on "
            "CPU fallback (expect lower sim rate on high-resolution sensors).");
    }

    // Allocate channel buffers
    const int n = H_ * W_;
    range_buf_.resize(static_cast<size_t>(n), 0);
    signal_buf_.resize(static_cast<size_t>(n), 0);
    reflectivity_buf_.resize(static_cast<size_t>(n), 50);
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
        ros_executor_.cancel();
        if (ros_spin_thread_.joinable()) {
            ros_spin_thread_.join();
        }
        return;
    }

    // Connect to the rendering-thread Render event (fires after scene->PreRender()
    // at Sensors.cc:496) so that GpuRays initialisation and Render() happen on
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

        // IMU packet buffer (used only when imu_enabled_)
        imu_packet_size_ = pf.imu_packet_size;
        if (imu_enabled_ && imu_packet_size_ > 0) {
            imu_pkt_buf_.resize(imu_packet_size_, 0);
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
    } catch (const std::exception & e) {
        RCLCPP_ERROR(kLogger, "Failed to parse metadata: %s", e.what());
        return false;
    }

    if (beam_alt_angles_.empty() || static_cast<int>(beam_alt_angles_.size()) != H_) {
        RCLCPP_ERROR(kLogger, "beam_altitude_angles size (%zu) != H (%d)",
            beam_alt_angles_.size(), H_);
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

    // Packet publisher
    const auto qos = rclcpp::SensorDataQoS();
    pkt_pub_ = ros_node_->create_publisher<ouster_sensor_msgs::msg::PacketMsg>(
        abs_prefix + "/lidar_packets", qos);

    // Image publishers (same topics as os_image node)
    range_image_pub_ = ros_node_->create_publisher<sensor_msgs::msg::Image>(
        abs_prefix + "/range_image", qos);
    signal_image_pub_ = ros_node_->create_publisher<sensor_msgs::msg::Image>(
        abs_prefix + "/signal_image", qos);
    reflec_image_pub_ = ros_node_->create_publisher<sensor_msgs::msg::Image>(
        abs_prefix + "/reflec_image", qos);
    nearir_image_pub_ = ros_node_->create_publisher<sensor_msgs::msg::Image>(
        abs_prefix + "/nearir_image", qos);

    // CameraInfo publisher (equirectangular projection model for range images)
    camera_info_pub_ = ros_node_->create_publisher<sensor_msgs::msg::CameraInfo>(
        abs_prefix + "/camera_info", qos);
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
        camera_info_msg_.k = {fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0};
        camera_info_msg_.d = {0.0, 0.0, 0.0, 0.0, 0.0};
        camera_info_msg_.r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
        camera_info_msg_.p = {fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0,
                              1.0, 0.0};
    }

    // IMU publishers (only if imu_name was provided in SDF)
    if (imu_enabled_) {
        imu_pkt_pub_ = ros_node_->create_publisher<ouster_sensor_msgs::msg::PacketMsg>(
            abs_prefix + "/imu_packets", qos);
        if (publish_imu_msg_) {
            imu_msg_pub_ = ros_node_->create_publisher<sensor_msgs::msg::Imu>(
                abs_prefix + "/imu", qos);
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

    param_cb_handle_ = ros_node_->add_on_set_parameters_callback(
        [this](const std::vector<rclcpp::Parameter> & params)
            -> rcl_interfaces::msg::SetParametersResult {
            for (const auto & p : params) {
                const auto & name = p.get_name();
                // Reject changes to structural parameters.
                if (name == "lidar_hz" || name == "max_range" || name == "imu_hz") {
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
                }
            }
            rcl_interfaces::msg::SetParametersResult r;
            r.successful = true;
            return r;
        });

    // Spin the node on a background thread so Zenoh completes peer discovery.
    // Without this, rmw_zenoh_cpp never processes incoming control messages
    // and publish() calls go undelivered even when subscribers exist.
    ros_executor_.add_node(ros_node_);
    ros_spin_thread_ = std::thread([this]() {
        ros_executor_.spin();
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
    if (sensor_initialized_.load(std::memory_order_acquire)) {
        // Republish metadata periodically until os_cloud confirms receipt.
        // rmw_zenoh_cpp transient_local replay is unreliable across processes,
        // and get_subscription_count() may lag behind actual Zenoh peer state.
        // Republish every ~1s (assuming render ticks at sensor hz) until we
        // see a subscriber AND have published at least a few times.
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
            if (meta_pub_->get_subscription_count() > 0 && metadata_pub_count_ > ticks_per_sec * 2) {
                metadata_published_ = true;
                RCLCPP_INFO(kLogger, "metadata delivered to os_cloud (after %d publishes)",
                    metadata_pub_count_);
            }
        }

        // Skip expensive GPU raycasts when no lidar outputs are consumed.
        // IMU publishing runs independently in PostUpdate().
        if (!this->HasActiveLidarSubscribers()) {
            return;
        }

        // ── Throttle to lidar_hz_ ───────────────────────────────────────────
        auto now = std::chrono::steady_clock::now();
        const auto period = std::chrono::duration_cast<
            std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(1.0 / lidar_hz_));
        if (now - last_render_time_ < period) return;
        last_render_time_ = now;

        // Trigger the GPU render pass and fire onNewFrame callback.
        // Render() does the actual GPU raycast; PostRender() reads back
        // the data and fires the NewGpuRaysFrame event.
        if (gpu_rays_) {
            // Position the sensor at the lidar_frame world pose.
            {
                std::lock_guard<std::mutex> lk(pose_mtx_);
                gpu_rays_->SetWorldPose(cached_pose_);
            }
            gpu_rays_->Render();
            gpu_rays_->PostRender();
        }
        return;
    }

    // ── Lazy GpuRays sensor initialisation ───────────────────────────────────
    // Guard: metadata must have loaded successfully (beam angles populated).
    if (beam_alt_angles_.empty()) return;

    // Wait until the Sensors system has created the OGRE2 scene.
    auto * engine = ::gz::rendering::engine("ogre2");
    if (!engine) return;
    if (engine->SceneCount() == 0) return;

    auto scene = engine->SceneByIndex(0);
    if (!scene) return;

    // Create GpuRays sensor programmatically.
    // Each beam row gets its own ray with the exact Ouster elevation angle.
    auto gpuRays = scene->CreateGpuRays(sensor_name_ + "_gpu_rays");
    if (!gpuRays) {
        RCLCPP_ERROR(kLogger, "Failed to create GpuRays sensor");
        return;
    }

    // Horizontal: full 360° FOV, W columns
    gpuRays->SetAngleMin(-GZ_PI);
    gpuRays->SetAngleMax(GZ_PI);
    gpuRays->SetRayCount(W_);

    // Vertical: use the min/max from beam_altitude_angles
    // GpuRays fires UNIFORM vertical samples between min/max.
    // We oversample vertically and resample to exact beam angles in onNewFrame.
    double min_alt = *std::min_element(beam_alt_angles_.begin(), beam_alt_angles_.end());
    double max_alt = *std::max_element(beam_alt_angles_.begin(), beam_alt_angles_.end());

    constexpr double margin_deg = 1.0;
    min_alt -= margin_deg;
    max_alt += margin_deg;

    gpuRays->SetVerticalAngleMin(min_alt * GZ_PI / 180.0);
    gpuRays->SetVerticalAngleMax(max_alt * GZ_PI / 180.0);

    // Ogre2GpuRays cubemap face resolution = next_power_of_2(max(hs, vs)),
    // where hs = RangeCount/4 for 360° HFOV and vs ≈ VerticalRangeCount.
    // With W_=512 → hs=128. Default v_samples=64 → cubemap 128×128 (coarse,
    // causes visible squared-off corners at cubemap face boundaries).
    // Raising v_samples to 256 → cubemap 256×256, halving the artifact.
    const int v_samples = std::max(H_ * 4, 256);
    gpuRays->SetVerticalRayCount(v_samples);

    gpuRays->SetNearClipPlane(0.3);
    gpuRays->SetFarClipPlane(max_range_);
    // Preserve out-of-range samples as +/-inf instead of clamping.
    // This matches upstream GpuLidarSensor behavior and keeps miss handling explicit.
    gpuRays->SetClamp(false);
    gpuRays->SetVisibilityMask(visibility_mask_);

    // Attach to scene root; world pose is set from ECM each frame in OnRender().
    auto root = scene->RootVisual();
    if (root) {
        root->AddChild(gpuRays);
    }

    // Connect frame callback
    frame_conn_ = gpuRays->ConnectNewGpuRaysFrame(
        std::bind(&GzGpuOusterLidarSystem::onNewFrame, this,
                  std::placeholders::_1, std::placeholders::_2,
                  std::placeholders::_3, std::placeholders::_4,
                  std::placeholders::_5));
    frame_connected_ = true;

    gpu_rays_ = gpuRays;
    sensor_initialized_.store(true, std::memory_order_release);

    RCLCPP_INFO(kLogger, "GpuRays sensor created: %dx%d rays, vertical FOV [%.1f, %.1f] deg",
        W_, v_samples, min_alt, max_alt);
}

// ── ISystemPostUpdate ────────────────────────────────────────────────────────

void GzGpuOusterLidarSystem::PostUpdate(
    const ::gz::sim::UpdateInfo & info,
    const ::gz::sim::EntityComponentManager & ecm)
{
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
    bool have_frame = false;
    std::vector<float> local_raw;
    int local_raw_H = 0, local_raw_W = 0, local_raw_chan = 0;
    {
        std::lock_guard<std::mutex> lk(frame_mtx_);
        have_frame = frame_ready_;
        frame_ready_ = false;
        if (have_frame) {
            local_raw.swap(raw_frame_buf_);
            local_raw_H    = raw_frame_H_;
            local_raw_W    = raw_frame_W_;
            local_raw_chan  = raw_frame_chan_;
        }
    }

    if (have_frame && !local_raw.empty()) {
        const auto stamp_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                info.simTime).count();
        encodeAndPublish(stamp_ns, local_raw.data(),
                         local_raw_H, local_raw_W, local_raw_chan);
    }

    // ── Publish IMU at configured rate ──────────────────────────────────────
    if (imu_enabled_ && imu_entity_found_) {
        publishImu(info, ecm);
    }
}

// ── GpuRays frame callback ──────────────────────────────────────────────────

void GzGpuOusterLidarSystem::onNewFrame(
    const float * data, unsigned int width,
    unsigned int height, unsigned int channels,
    const std::string & /*format*/)
{
    // Fast path: just memcpy the raw GpuRays buffer into a staging area.
    // Bilinear resampling is done later on the GPU (CUDA) or CPU (fallback)
    // in encodeAndPublish(), avoiding heavy computation under this lock.

    if (!data || width == 0 || height == 0 || channels == 0) return;
    if (beam_alt_angles_.empty() || H_ <= 0 || W_ <= 0) return;

    const size_t n = static_cast<size_t>(width) * height * channels;

    std::lock_guard<std::mutex> lk(frame_mtx_);
    raw_frame_buf_.resize(n);
    std::memcpy(raw_frame_buf_.data(), data, n * sizeof(float));
    raw_frame_H_    = static_cast<int>(height);
    raw_frame_W_    = static_cast<int>(width);
    raw_frame_chan_ = static_cast<int>(channels);
    frame_ready_ = true;
}

// ── Encode depth → Ouster packets ───────────────────────────────────────────

void GzGpuOusterLidarSystem::encodeAndPublish(
    int64_t stamp_ns,
    const float * raw_data, int gpu_H, int gpu_W, int gpu_chan)
{
    if (stamp_ns <= 0) return;
    if (!pw_ || pkt_buf_.empty()) return;
    if (beam_alt_angles_.empty() || H_ <= 0 || W_ <= 0 || cpp_ <= 0) return;
    if (!cuda_processor_) return;

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
    double min_alt = *std::min_element(beam_alt_angles_.begin(), beam_alt_angles_.end());
    double max_alt = *std::max_element(beam_alt_angles_.begin(), beam_alt_angles_.end());
    constexpr double margin_deg = 1.0;
    min_alt -= margin_deg;
    max_alt += margin_deg;
    const double v_range = max_alt - min_alt;
    if (v_range <= 0.0) return;

    ResampleParams rp;
    rp.H = H_;
    rp.W = W_;
    rp.gpu_H = gpu_H;
    rp.gpu_W = gpu_W;
    rp.gpu_chan = gpu_chan;
    rp.min_alt = static_cast<float>(min_alt);
    rp.v_range = static_cast<float>(v_range);
    rp.deg_per_col = 360.0f / static_cast<float>(W_);
    rp.beam_origin_m = static_cast<float>(beam_origin_mm_ / 1000.0);
    rp.half_W = W_ / 2;

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
    cuda_processor_->processRaw(
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
    const int64_t dt_col_ns      = scan_period_ns / W_;

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
        std::fill(pkt_buf_.begin(), pkt_buf_.end(), 0);

        const int col_start = p * cpp_;
        pw_->set_frame_id(pkt_buf_.data(), frame_id_);

        for (int c_local = 0; c_local < cpp_; ++c_local) {
            const int col_global = col_start + c_local;
            uint8_t * col = pw_->nth_col(c_local, pkt_buf_.data());
            pw_->set_col_timestamp(col,
                static_cast<uint64_t>(scan_start_ns + col_global * dt_col_ns));
            pw_->set_col_measurement_id(col, static_cast<uint16_t>(col_global));
            pw_->set_col_status(col, 0x01u);
        }

        pw_->set_block<uint32_t>(range_mat,  ouster::sdk::core::ChanField::RANGE,        pkt_buf_.data());
        pw_->set_block<uint16_t>(signal_mat, ouster::sdk::core::ChanField::SIGNAL,       pkt_buf_.data());
        pw_->set_block<uint8_t> (refl_mat,   ouster::sdk::core::ChanField::REFLECTIVITY, pkt_buf_.data());
        pw_->set_block<uint16_t>(nearir_mat, ouster::sdk::core::ChanField::NEAR_IR,      pkt_buf_.data());

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

    const auto & av = angVelComp->Data();   // rad/s, body frame
    const auto & la = linAccComp->Data();   // m/s², body frame, includes gravity

    const int64_t stamp_ns = sim_now.count();

    // ── Encode Ouster IMU PacketMsg ──────────────────────────────────────
    if (imu_pkt_pub_ && imu_pkt_pub_->get_subscription_count() > 0 &&
        !imu_pkt_buf_.empty()) {
        std::fill(imu_pkt_buf_.begin(), imu_pkt_buf_.end(), 0);
        uint8_t * buf = imu_pkt_buf_.data();
        const uint64_t ts = static_cast<uint64_t>(stamp_ns);

        // LEGACY IMU profile (48 bytes): timestamps at fixed offsets.
        // PacketWriter only exposes set_imu_nmea_ts (for ACCEL32_GYRO32_NMEA),
        // so we write the LEGACY sys_ts/accel_ts/gyro_ts directly.
        // os_cloud reads imu_gyro_ts (offset 16) for the ROS message timestamp.
        if (imu_pkt_buf_.size() >= 48) {
            std::memcpy(buf + 0,  &ts, sizeof(uint64_t));  // sys_ts
            std::memcpy(buf + 8,  &ts, sizeof(uint64_t));  // accel_ts
            std::memcpy(buf + 16, &ts, sizeof(uint64_t));  // gyro_ts
        }

        // ACCEL32_GYRO32_NMEA profile: use PacketWriter setters.
        pw_->set_imu_nmea_ts(buf, ts);

        // Accel/gyro values — PacketWriter writes at profile-correct offsets.
        pw_->set_imu_la_x(buf, static_cast<float>(la.X()));
        pw_->set_imu_la_y(buf, static_cast<float>(la.Y()));
        pw_->set_imu_la_z(buf, static_cast<float>(la.Z()));
        pw_->set_imu_av_x(buf, static_cast<float>(av.X()));
        pw_->set_imu_av_y(buf, static_cast<float>(av.Y()));
        pw_->set_imu_av_z(buf, static_cast<float>(av.Z()));

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

        msg.angular_velocity.x = av.X();
        msg.angular_velocity.y = av.Y();
        msg.angular_velocity.z = av.Z();

        msg.linear_acceleration.x = la.X();
        msg.linear_acceleration.y = la.Y();
        msg.linear_acceleration.z = la.Z();

        // Covariance: match ouster_ros os_cloud defaults
        msg.angular_velocity_covariance[0] = 6e-4;
        msg.angular_velocity_covariance[4] = 6e-4;
        msg.angular_velocity_covariance[8] = 6e-4;

        msg.linear_acceleration_covariance[0] = 0.01;
        msg.linear_acceleration_covariance[4] = 0.01;
        msg.linear_acceleration_covariance[8] = 0.01;

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

        const auto spacing = std::chrono::nanoseconds(
            static_cast<int64_t>(1e9 / lidar_hz_) /
            static_cast<int64_t>(local_pkts.size()));

        try {
            for (size_t i = 0; i < local_pkts.size(); ++i) {
                if (shutdown_.load(std::memory_order_acquire)) return;
                {
                    std::lock_guard<std::mutex> pub_lk(publish_mtx_);
                    pkt_pub_->publish(local_pkts[i]);
                }
                if (i + 1 < local_pkts.size()) {
                    std::this_thread::sleep_for(spacing);
                }
            }
        } catch (const std::exception & e) {
            RCLCPP_ERROR(kLogger, "drainThread publish failed: %s", e.what());
        }
    }
}

bool GzGpuOusterLidarSystem::HasActiveLidarSubscribers() const
{
    if (pkt_pub_ && pkt_pub_->get_subscription_count() > 0) return true;
    if (range_image_pub_ && range_image_pub_->get_subscription_count() > 0) return true;
    if (signal_image_pub_ && signal_image_pub_->get_subscription_count() > 0) return true;
    if (reflec_image_pub_ && reflec_image_pub_->get_subscription_count() > 0) return true;
    if (nearir_image_pub_ && nearir_image_pub_->get_subscription_count() > 0) return true;
    if (camera_info_pub_ && camera_info_pub_->get_subscription_count() > 0) return true;
    return false;
}

void GzGpuOusterLidarSystem::DestroyGpuRays()
{
    if (!gpu_rays_) return;

    auto * engine = ::gz::rendering::engine("ogre2");
    if (engine && engine->SceneCount() > 0) {
        auto scene = engine->SceneByIndex(0);
        if (scene) {
            scene->DestroySensor(gpu_rays_);
        }
    }

    gpu_rays_.reset();
    frame_connected_ = false;
}

}  // namespace gz_gpu_ouster_lidar
