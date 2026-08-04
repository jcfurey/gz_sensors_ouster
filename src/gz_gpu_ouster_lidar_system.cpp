// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0
//
// Thin orchestrator: SDF parsing/validation, the gz-sim Configure /
// PostUpdate / Render callbacks, entity tracking, IMU sampling and the
// teardown ordering. The pipeline work lives in the components under
// src/ (OusterMetadata, PanelRig, RaycastMirror, PacketEncoder,
// RosInterface, FrameExchange).

#include "gz_gpu_ouster_lidar/gz_gpu_ouster_lidar_system.hpp"

#include "gz_gpu_ouster_lidar/ray_processor.hpp"
#include "backend.hpp"        // noiseEnabled() — pulled in via cuda/ includes
#include "imu_noise.hpp"      // applyImuNoise()
#include "frame_exchange.hpp"
#include "lidar_common.hpp"
#include "ouster_metadata.hpp"
#include "packet_encoder.hpp"
#include "panel_rig.hpp"
#include "raycast_mirror.hpp"
#include "ros_interface.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <gz/plugin/Register.hh>
#include <gz/sim/Entity.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/EventManager.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/World.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/Sensor.hh>
#include <gz/sim/components/AngularVelocity.hh>
#include <gz/sim/components/LinearAcceleration.hh>
#include <gz/sim/rendering/Events.hh>

#include <ouster/impl/packet_writer.h>
#include <ouster/types.h>

namespace gz_gpu_ouster_lidar {

static const rclcpp::Logger kLogger = lidarLogger();

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
    // 4. Tear down the components in reverse data-flow order: producers
    //    (rig / mirror), then the encoder's drain, then ROS.
    shutdown_.store(true, std::memory_order_release);
    render_conn_.reset();
    render_teardown_conn_.reset();
    // Taking render_busy_mtx_ flushes any in-flight render callback (see the
    // header comment); holding it across the rig fallback below also keeps
    // the teardown paths mutually exclusive with OnRenderTeardown.
    {
        std::lock_guard<std::recursive_mutex> lk(render_busy_mtx_);
        // The panel rig's OGRE2 depth cameras are normally destroyed on the
        // render thread via OnRenderTeardown() (events::RenderTeardown), so
        // their GL resources are freed on the thread that owns the GL context.
        // This fallback covers the paths where that never ran:
        //  - benign: the rig holds no cameras (panels mode where events::Render
        //    never fired, or raycast mode) — destroy() early-returns, no GL work;
        //  - KNOWN LIMITATION: the plugin's model is removed from a *running*
        //    world. gz-sim emits RenderTeardown only when the Sensors render
        //    loop exits, not on entity removal, so here we still destroy live
        //    cameras from the server thread while rendering continues. That is
        //    the pre-RenderTeardown behavior (thread-affinity risk); there is
        //    no plugin-visible hook to schedule the destroy on the render
        //    thread once our event connections are reset. Leaking instead
        //    would permanently hold rig VRAM per spawn/despawn cycle.
        if (rig_ && !rig_destroyed_) {
            rig_->destroy();
            rig_destroyed_ = true;
        }
    }
    if (mirror_) {
        mirror_->stop();
    }
    if (encoder_) {
        encoder_->stop();
    }
    if (ros_) {
        ros_->shutdown();
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
    if (sdf->HasElement("ray_mode")) {
        ray_mode_ = sdf->Get<std::string>("ray_mode");
    }
    if (sdf->HasElement("image_qos")) {
        image_qos_ = sdf->Get<std::string>("image_qos");
    }
    if (sdf->HasElement("imu_qos")) {
        imu_qos_ = sdf->Get<std::string>("imu_qos");
    }
    if (sdf->HasElement("publish_native_images")) {
        publish_native_images_ = sdf->Get<bool>("publish_native_images");
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
    if (sdf->HasElement("false_alarm_rate")) {
        false_alarm_rate_ = sdf->Get<double>("false_alarm_rate");
    }
    if (sdf->HasElement("motion_distortion")) {
        motion_distortion_ = sdf->Get<bool>("motion_distortion");
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
    // Out-of-range values are clamped AND reported — a silently rewritten
    // parameter (e.g. a typo'd negative noise std) is a debugging trap.
    auto clamp_warn = [](double & v, double lo, double hi, const char * name) {
        // NaN compares false against everything, so it needs an explicit
        // branch — and std::clamp would pass it through. Sanitise to the
        // lower bound (what the old std::max(lo, v) form effectively did).
        if (std::isnan(v)) {
            RCLCPP_WARN(kLogger, "%s is NaN; using %g", name, lo);
            v = lo;
            return;
        }
        if (v < lo || v > hi) {
            const double c = std::clamp(v, lo, hi);
            RCLCPP_WARN(kLogger, "%s=%g outside [%g, %g]; clamped to %g",
                        name, v, lo, hi, c);
            v = c;
        }
    };
    constexpr double kInfD = std::numeric_limits<double>::infinity();
    clamp_warn(range_noise_min_std_,   0.0, kInfD, "range_noise_min_std");
    clamp_warn(range_noise_max_std_,   0.0, kInfD, "range_noise_max_std");
    clamp_warn(signal_noise_scale_,    0.0, kInfD, "signal_noise_scale");
    clamp_warn(nearir_noise_scale_,    0.0, kInfD, "nearir_noise_scale");
    clamp_warn(dropout_rate_close_,    0.0, 1.0,   "dropout_rate_close");
    clamp_warn(dropout_rate_far_,      0.0, 1.0,   "dropout_rate_far");
    clamp_warn(false_alarm_rate_,      0.0, 1.0,   "false_alarm_rate");
    clamp_warn(edge_discon_threshold_, 0.0, kInfD, "edge_discon_threshold");
    clamp_warn(base_signal_,           0.0, kInfD, "base_signal");
    clamp_warn(base_reflectivity_,     0.0, 255.0, "base_reflectivity");
    clamp_warn(max_range_,             1.0, kInfD, "max_range");
    clamp_warn(gyro_noise_std_,        0.0, kInfD, "gyro_noise_std");
    clamp_warn(accel_noise_std_,       0.0, kInfD, "accel_noise_std");
    clamp_warn(gyro_bias_walk_,        0.0, kInfD, "gyro_bias_walk");
    clamp_warn(accel_bias_walk_,       0.0, kInfD, "accel_bias_walk");
    clamp_warn(panel_oversample_,      1.0, 4.0,   "panel_oversample");
    if (ray_mode_ != "panels" && ray_mode_ != "raycast") {
        RCLCPP_WARN(kLogger,
            "Unknown ray_mode='%s'; expected panels|raycast. "
            "Defaulting to raycast.", ray_mode_.c_str());
        ray_mode_ = "raycast";
    }
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
    if (!std::isfinite(lidar_hz_) || lidar_hz_ <= 0.0) {
        RCLCPP_WARN(kLogger, "lidar_hz must be > 0, got %f; defaulting to 10", lidar_hz_);
        lidar_hz_ = 10.0;
    }
    if (imu_enabled_ && (!std::isfinite(imu_hz_) || imu_hz_ <= 0.0)) {
        RCLCPP_WARN(kLogger, "imu_hz must be > 0, got %f; defaulting to 100", imu_hz_);
        imu_hz_ = 100.0;
    }

    if (metadata_path_.empty()) {
        RCLCPP_ERROR(kLogger, "'metadata_path' SDF parameter is required");
        return;
    }

    // sensor_name is used as BOTH the ROS node namespace and the absolute topic
    // prefix, so it must be a non-empty absolute name. A relative prefix makes
    // every topic relative — resolved a second time under the node namespace —
    // so it double-namespaces (e.g. "lidar0" → /lidar0/lidar0/lidar_packets).
    // An empty name is fatal (like an empty metadata_path): the pose-anchor
    // lookup would never match and the plugin would run forever publishing
    // nothing — disable it loudly instead of limping. A relative name is
    // normalised.
    if (sensor_name_.empty()) {
        RCLCPP_ERROR(kLogger,
            "'sensor_name' SDF parameter is required (e.g. "
            "/sensor/lidar/lidar0); plugin disabled");
        return;
    }
    if (sensor_name_.front() != '/') {
        RCLCPP_WARN(kLogger,
            "sensor_name '%s' is relative; prefixing '/' so the plugin's topics "
            "stay absolute (a relative prefix double-namespaces every topic "
            "under the node).", sensor_name_.c_str());
        sensor_name_ = "/" + sensor_name_;
    }

    // Discover world name + gravity. The Gravity component is created from
    // the world SDF before systems are configured, so a one-time read here
    // covers non-standard worlds (reduced/tilted gravity); the member default
    // (standard gravity) stands if the component is somehow absent.
    auto worldEntity = ::gz::sim::worldEntity(ecm);
    if (worldEntity != ::gz::sim::kNullEntity) {
        auto * nameComp = ecm.Component<::gz::sim::components::Name>(worldEntity);
        if (nameComp) {
            world_name_ = nameComp->Data();
        }
        if (const auto gravity =
                ::gz::sim::World(worldEntity).Gravity(ecm)) {
            world_gravity_ = *gravity;
        }
    }

    sensor_entity_ = entity;

    // Load Ouster metadata and beam angles
    meta_ = std::make_unique<OusterMetadata>();
    if (!meta_->load(metadata_path_, imu_enabled_, max_range_explicit_,
                     max_range_)) {
        RCLCPP_ERROR(kLogger, "Metadata loading failed; plugin disabled");
        meta_.reset();
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

    // Allocate channel buffers + the frame exchange.
    const int n = meta_->H * meta_->W;
    range_buf_.resize(static_cast<size_t>(n), 0);
    signal_buf_.resize(static_cast<size_t>(n), 0);
    reflectivity_buf_.resize(static_cast<size_t>(n),
                             static_cast<uint8_t>(base_reflectivity_));
    nearir_buf_.resize(static_cast<size_t>(n), 0);
    exchange_ = std::make_unique<FrameExchange>();
    processed_exchange_ = std::make_unique<ProcessedFrameExchange>();
    if (imu_enabled_ && meta_->imu_packet_size > 0) {
        imu_pkt_buf_.resize(meta_->imu_packet_size, 0);
    }

    // ── Build the panel rig from the beam geometry (panels mode only) ───────
    // The raycast mode needs no rig: beams are cast exactly, any altitude.
    if (ray_mode_ == "panels") {
        rig_ = std::make_unique<PanelRig>(sensor_name_);
        if (!rig_->buildLayout(*meta_, panel_oversample_, panel_sampling_,
                               max_range_)) {
            rig_.reset();
            return;
        }
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
    ros_ = std::make_unique<RosInterface>();
    try {
        RosInterfaceConfig cfg;
        cfg.sensor_name = sensor_name_;
        cfg.image_qos = image_qos_;
        cfg.imu_qos = imu_qos_;
        cfg.publish_native_images = publish_native_images_;
        cfg.image_frame_id = image_frame_id_;
        cfg.imu_frame_id = imu_frame_id_;
        cfg.metadata_str = meta_->metadata_str;
        cfg.H = meta_->H;
        cfg.W = meta_->W;
        cfg.lidar_packet_qos_depth = static_cast<size_t>(
            std::max(5, meta_->W / meta_->cpp));
        cfg.beam_alt_angles = &meta_->beam_alt_angles;
        cfg.lidar_hz = lidar_hz_;
        cfg.max_range = max_range_;
        cfg.imu_hz = imu_hz_;
        cfg.imu_enabled = imu_enabled_;
        cfg.publish_imu_msg = publish_imu_msg_;

        NoiseParams noise;
        noise.range_noise_min_std = range_noise_min_std_;
        noise.range_noise_max_std = range_noise_max_std_;
        noise.signal_noise_scale = signal_noise_scale_;
        noise.nearir_noise_scale = nearir_noise_scale_;
        noise.dropout_rate_close = dropout_rate_close_;
        noise.dropout_rate_far = dropout_rate_far_;
        noise.false_alarm_rate = false_alarm_rate_;
        noise.edge_discon_threshold = edge_discon_threshold_;
        noise.base_signal = base_signal_;
        noise.base_reflectivity = base_reflectivity_;
        noise.gyro_noise_std = gyro_noise_std_;
        noise.accel_noise_std = accel_noise_std_;
        noise.gyro_bias_walk = gyro_bias_walk_;
        noise.accel_bias_walk = accel_bias_walk_;

        ros_->init(cfg, noise);
    } catch (const std::exception & e) {
        RCLCPP_ERROR(kLogger,
            "ROS interface init failed: %s; plugin disabled", e.what());
        ros_->shutdown();
        ros_.reset();
        return;
    }

    // Connect to the rendering-thread Render event (fires after scene->PreRender()
    // at Sensors.cc:496) so that panel-rig initialisation and Render() happen on
    // the correct (EGL) thread with the scene already set up. The raycast
    // mode never touches the renderer, so it skips the event entirely (and
    // with it the world's rendering-sensor requirement).
    event_mgr_ = &event_mgr;
    if (ray_mode_ == "panels") {
        render_conn_ = event_mgr.Connect<::gz::sim::events::Render>(
            std::bind(&GzGpuOusterLidarSystem::OnRender, this));
        // Destroy the depth-camera rig on the render thread (owning GL
        // context) when gz-sim tears rendering down, rather than from the
        // dtor on the server thread.
        render_teardown_conn_ =
            event_mgr.Connect<::gz::sim::events::RenderTeardown>(
                std::bind(&GzGpuOusterLidarSystem::OnRenderTeardown, this));
    }

    RCLCPP_INFO(kLogger,
        "Configured: H=%d W=%d cpp=%d sensor_name=%s gz_sensor=%s hz=%.1f"
        " noise: range_sigma=[%.4f,%.4f]m signal_noise=%.1f"
        " dropout=[%.4f,%.4f] edge_discon=%.3fm max_range=%.1fm",
        meta_->H, meta_->W, meta_->cpp, sensor_name_.c_str(),
        lidar_frame_name_.c_str(), lidar_hz_,
        range_noise_min_std_, range_noise_max_std_, signal_noise_scale_,
        dropout_rate_close_, dropout_rate_far_, edge_discon_threshold_, max_range_);
    if (imu_enabled_) {
        RCLCPP_INFO(kLogger, "  IMU: sensor=%s hz=%.1f publish_imu_msg=%s",
            imu_name_.c_str(), imu_hz_, publish_imu_msg_ ? "true" : "false");
    }

    if (ray_mode_ == "raycast") {
        RCLCPP_INFO(kLogger,
            "Full raycast mode: exact per-beam casting against the ECM scene "
            "mirror (no rendering; laser_retro drives reflectivity).");
        mirror_ = std::make_unique<RaycastMirror>(sensor_name_);
        RaycastMirror::Params mp;
        mp.H = meta_->H;
        mp.W = meta_->W;
        mp.max_range = max_range_;
        mp.lidar_hz = lidar_hz_;
        mp.beam_origin_mm = meta_->beam_origin_mm;
        mp.motion_distortion = motion_distortion_;
        mp.beam_alt_f = &meta_->beam_alt_f;
        mp.beam_az_f = &meta_->beam_az_f;
        mirror_->start(mp, ray_processor_.get(), processed_exchange_.get());
        // Nothing render-side to initialise — unblock PostUpdate immediately.
        sensor_initialized_.store(true, std::memory_order_release);
    }

    // Start the encoder's drain thread last — all state is initialised above.
    encoder_ = std::make_unique<PacketEncoder>();
    encoder_->start(meta_.get(), ros_.get(), lidar_hz_);
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
    if (!rig_) return;

    // Record that events::Render fired at least once. PostUpdate() reads this
    // to emit a one-shot diagnostic when the Sensors system never starts
    // rendering (no rendering sensor in the world); see PostUpdate().
    onrender_entries_.fetch_add(1, std::memory_order_relaxed);

    if (sensor_initialized_.load(std::memory_order_acquire)) {
        // Skip scan renders while the sim is paused: PostUpdate won't consume
        // a frame until unpause, and a mid-pause render could capture a scene
        // the user is editing in the GUI, publishing a stale world as the
        // first post-resume cloud. (Rig creation below is NOT gated — building
        // the cameras while paused is cheap and desirable.)
        if (paused_.load(std::memory_order_acquire)) return;

        // ── Throttle to lidar_hz_ on SIM time ───────────────────────────────
        // Pace on sim time (published by PostUpdate) rather than wall-clock so
        // the panels scan rate tracks lidar_hz at any real-time factor and
        // matches the sim-time packet timestamps. Rendering when the sim time
        // has rewound (world reset: sim_ns < last) re-arms instead of stalling
        // until the clock catches back up.
        ::gz::math::Pose3d pose;
        FrameMetadata metadata;
        {
            std::lock_guard<std::mutex> lk(pose_mtx_);
            pose = cached_pose_;
            metadata.scan_end_ns = cached_pose_sim_ns_;
            metadata.epoch = cached_pose_epoch_;
        }
        const auto gate = panel_scan_gate_.advance(
            std::chrono::nanoseconds(metadata.scan_end_ns),
            periodFromHz(lidar_hz_));
        if (!gate.due) return;

        rig_->renderScan(pose, *exchange_, metadata);
        return;
    }

    // ── Lazy panel-rig initialisation ────────────────────────────────────────
    if (rig_->ensureCreated(max_range_, visibility_mask_)) {
        sensor_initialized_.store(true, std::memory_order_release);
        RCLCPP_INFO(kLogger,
            "Panel rig created: %d depth cameras, beam altitude span "
            "[%.1f, %.1f] deg, %s model",
            rig_->resampleParams().n_panels, meta_->min_alt, meta_->max_alt,
            rig_->hemispherical() ? "hemispherical" : "cylindrical");
    }
}

// ── Rendering-thread teardown callback ───────────────────────────────────────
// Fired by gz-sim on the render thread (events::RenderTeardown) while the ogre2
// engine/scene are still valid, so the depth cameras' GL resources are freed on
// the thread that owns the GL context. Doing this from the dtor (server thread)
// would free GL resources off their owning thread — corruption / segfault.

void GzGpuOusterLidarSystem::OnRenderTeardown()
{
    std::lock_guard<std::recursive_mutex> render_lk(render_busy_mtx_);
    if (rig_ && !rig_destroyed_) {
        rig_->destroy();
        rig_destroyed_ = true;
    }
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
    if (ray_mode_ == "panels" && !no_render_warned_ &&
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
                "Sensors system has not started rendering. panels mode attaches "
                "a depth-camera rig to the ogre2 scene owned by gz-sim-sensors-system, "
                "which only initialises rendering when the world contains at "
                "least one rendering sensor (camera or gpu_lidar). Add a "
                "rendering sensor (pass anchor_type:=camera in xacro, or switch "
                "to the default ray_mode:=raycast which needs no renderer) or "
                "no point cloud will be produced.",
                static_cast<double>(sim_ns) / 1e9);
        }
    }

    const auto sim_now_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(info.simTime);
    if (last_post_sim_time_.count() >= 0 && sim_now_ns < last_post_sim_time_) {
        ++sim_epoch_;
    }
    last_post_sim_time_ = sim_now_ns;
    if (encoder_) {
        encoder_->setSimulationState(info.paused, sim_epoch_);
    }

    // Publish the paused state for the render thread: OnRender skips scan
    // renders while paused. A mid-pause render could only capture a world the
    // user is editing in the GUI, and PostUpdate would not consume the frame
    // until unpause anyway — so it would publish as a stale first post-resume
    // cloud. Stored before the early-return so the flag tracks every tick.
    paused_.store(info.paused, std::memory_order_release);
    if (info.paused) {
        return;
    }

    if (!sensor_initialized_.load(std::memory_order_acquire)) return;
    if (!ros_) return;

    // Sim-thread work here can throw — rclcpp publishes (an RCLError if the
    // context is torn down mid-step) and, on the "scan" path, the GPU backend
    // (a CUDA/HIP error surfacing as an exception). gz-sim does not wrap
    // ISystemPostUpdate in a try/catch, so an escaping exception aborts the
    // whole server. Swallow + throttle-log instead: a persistent failure logs
    // every 5 s (identifying the failing stage) while the sim keeps running.
    // catch (...) is deliberate — an unknown exception type crossing into
    // gz-sim would abort just the same.
    auto guarded = [this](const char * what, auto && fn) {
        try {
            fn();
        } catch (const std::exception & e) {
            if (ros_ && ros_->clock()) {
                RCLCPP_ERROR_THROTTLE(kLogger, *ros_->clock(), 5000,
                    "%s: %s stage failed: %s",
                    sensor_name_.c_str(), what, e.what());
            }
        } catch (...) {
            if (ros_ && ros_->clock()) {
                RCLCPP_ERROR_THROTTLE(kLogger, *ros_->clock(), 5000,
                    "%s: %s stage failed (non-std exception)",
                    sensor_name_.c_str(), what);
            }
        }
    };

    // ── Metadata (re)publish — sim-thread, works in both ray modes ──────────
    guarded("metadata", [&] {
        ros_->publishMetadataIfNeeded(
            std::chrono::duration_cast<std::chrono::nanoseconds>(info.simTime));
    });

    // ── Locate the gpu_lidar sensor entity (once) ────────────────────────────
    // Gazebo merges fixed-joint child links into the parent, so
    // "lidar0/lidar_frame" does not exist as a Link entity.  The <sensor>
    // element placed on that frame DOES survive and carries the accumulated
    // fixed-joint pose, giving us the correct world position.
    //
    // Scope the search to the plugin's own top-level model so a same-named
    // sensor on a *different* model (a second robot carrying its own "lidar0")
    // cannot bind here and feed this plugin the wrong pose.
    if (!lidar_frame_found_) {
        const ::gz::sim::Entity lidar_model =
            ::gz::sim::topLevelModel(sensor_entity_, ecm);
        ecm.Each<::gz::sim::components::Name, ::gz::sim::components::Sensor>(
            [this, &ecm, lidar_model](const ::gz::sim::Entity & ent,
                   const ::gz::sim::components::Name * name,
                   const ::gz::sim::components::Sensor *) -> bool {
                if (name->Data() != lidar_frame_name_) return true;
                if (lidar_model != ::gz::sim::kNullEntity &&
                    ::gz::sim::topLevelModel(ent, ecm) != lidar_model) {
                    return true;  // same name, different model — skip
                }
                lidar_frame_entity_ = ent;
                lidar_frame_found_ = true;
                RCLCPP_INFO(kLogger, "Found sensor entity: %s (id=%lu)",
                    lidar_frame_name_.c_str(), static_cast<unsigned long>(ent));
                return false;  // stop iteration
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
        cached_pose_sim_ns_ = sim_now_ns.count();
        cached_pose_epoch_ = sim_epoch_;
    }

    // ── Full raycast mode: mirror the scene and post a scan job ─────────────
    if (mirror_ && lidar_frame_found_) {
        ::gz::math::Pose3d sensor_pose;
        {
            std::lock_guard<std::mutex> lk(pose_mtx_);
            sensor_pose = cached_pose_;
        }
        const auto process_params = makeRayProcessParams();
        if (!memory_logged_) {
            const size_t n = static_cast<size_t>(meta_->H) * meta_->W;
            const size_t intermediate_bytes = 3 * n * sizeof(float);
            const size_t channel_bytes = n *
                (sizeof(uint32_t) + sizeof(uint16_t) + sizeof(uint8_t) +
                 sizeof(uint16_t));
            const size_t rand_bytes = noiseEnabled(process_params)
                ? n * 48 : 0;
            const size_t total =
                intermediate_bytes + channel_bytes + rand_bytes;
            RCLCPP_INFO(kLogger,
                "%s: GPU buffers ~%.1f MiB (%s fused raycast) — "
                "device intermediates=%.1f channels=%.1f rand=%.1f",
                sensor_name_.c_str(), total / 1048576.0,
                ray_processor_->backendName(),
                intermediate_bytes / 1048576.0,
                channel_bytes / 1048576.0,
                rand_bytes / 1048576.0);
            memory_logged_ = true;
        }
        mirror_->postUpdate(
            info, ecm, sensor_pose, sim_epoch_, process_params);
    }

    // ── Process any pending frame ────────────────────────────────────────────
    // Sim-side of the exchange: take() is an O(1) swap into the sim-only
    // process_buf_; encodeAndPublish (which can take many ms on dense
    // sensors) runs without holding the exchange lock, so the producer is
    // free to hand over a new frame concurrently.
    int local_n = 0;
    FrameMetadata frame_metadata;
    if (ray_mode_ == "raycast" && processed_exchange_->take(
            range_buf_, signal_buf_, reflectivity_buf_, nearir_buf_,
            local_n, frame_metadata)) {
        if (frame_metadata.epoch != sim_epoch_) {
            RCLCPP_WARN_THROTTLE(kLogger, *ros_->clock(), 5000,
                "%s: dropping stale scan from epoch %lu after reset to %lu",
                sensor_name_.c_str(),
                static_cast<unsigned long>(frame_metadata.epoch),
                static_cast<unsigned long>(sim_epoch_));
        } else if (local_n != meta_->H * meta_->W) {
            RCLCPP_ERROR_THROTTLE(kLogger, *ros_->clock(), 5000,
                "%s: processed raycast frame size %d != expected %d; dropping",
                sensor_name_.c_str(), local_n, meta_->H * meta_->W);
        } else {
            guarded("scan", [&] {
                publishChannels(frame_metadata.scan_end_ns);
            });
        }
    } else if (ray_mode_ != "raycast" &&
               exchange_->take(process_buf_, local_n, frame_metadata) &&
               !process_buf_.empty()) {
        if (frame_metadata.epoch != sim_epoch_) {
            RCLCPP_WARN_THROTTLE(kLogger, *ros_->clock(), 5000,
                "%s: dropping stale scan from epoch %lu after reset to %lu",
                sensor_name_.c_str(),
                static_cast<unsigned long>(frame_metadata.epoch),
                static_cast<unsigned long>(sim_epoch_));
        } else {
            guarded("scan", [&] {
                encodeAndPublish(frame_metadata.scan_end_ns,
                                 process_buf_.data(), local_n);
            });
        }
    }

    // ── Publish IMU at configured rate ──────────────────────────────────────
    if (imu_enabled_ && imu_entity_found_) {
        guarded("imu", [&] { publishImu(info, ecm); });
    }
}

// ── Encode depth → Ouster packets ───────────────────────────────────────────

RayProcessParams GzGpuOusterLidarSystem::makeRayProcessParams() const
{
    const NoiseParams noise = ros_->noiseSnapshot();
    RayProcessParams pp{};
    pp.H = meta_->H;
    pp.W = meta_->W;
    pp.base_signal = static_cast<float>(noise.base_signal);
    pp.base_reflectivity = static_cast<float>(noise.base_reflectivity);
    pp.range_noise_min_std = static_cast<float>(noise.range_noise_min_std);
    pp.range_noise_max_std = static_cast<float>(noise.range_noise_max_std);
    pp.max_range = static_cast<float>(max_range_);
    pp.signal_noise_scale = static_cast<float>(noise.signal_noise_scale);
    pp.nearir_noise_scale = static_cast<float>(noise.nearir_noise_scale);
    pp.dropout_rate_close = static_cast<float>(noise.dropout_rate_close);
    pp.dropout_rate_far = static_cast<float>(noise.dropout_rate_far);
    pp.false_alarm_rate = static_cast<float>(noise.false_alarm_rate);
    pp.edge_discon_threshold =
        static_cast<float>(noise.edge_discon_threshold);
    return pp;
}

void GzGpuOusterLidarSystem::encodeAndPublish(
    int64_t stamp_ns,
    const float * raw_data, int raw_n)
{
    if (stamp_ns <= 0) return;
    if (!meta_ || !ray_processor_ || !encoder_ || !ros_) return;
    const int H = meta_->H;
    const int W = meta_->W;
    if (H <= 0 || W <= 0) return;

    const int expected_n = rig_ ? rig_->resampleParams().raw_n : 0;
    if (raw_n != expected_n) {
        if (ros_->clock()) {
            RCLCPP_ERROR_THROTTLE(kLogger, *ros_->clock(), 5000,
                "%s: raw frame size %d != expected %d; dropping",
                sensor_name_.c_str(), raw_n, expected_n);
        }
        return;
    }

    const RayProcessParams pp = makeRayProcessParams();

    // ── GPU pipeline: resample/cast results → noise → channel outputs ───────
    if (!memory_logged_) {
        // One-time per-sensor accounting after first frame, when the
        // dispatched backend has settled on actual buffer sizes. Helps
        // diagnose multi-sensor OOMs — a dense sensor (4096×512) plus
        // curand state can easily exceed 100 MB of VRAM on its own.
        const size_t raw_bytes  = static_cast<size_t>(raw_n) * sizeof(float);
        const size_t channel_bytes =
            static_cast<size_t>(H) * W *
            (sizeof(uint32_t) + sizeof(uint16_t) + sizeof(uint8_t) + sizeof(uint16_t));
        const size_t resample_bytes = static_cast<size_t>(H) * W * sizeof(float);
        // curandState (XORWOW) is 48 B; hiprand similar; SYCL counter-based
        // RNG is stateless. Approximate with 48 B to avoid backend coupling.
        const size_t rand_bytes = noiseEnabled(pp)
            ? static_cast<size_t>(H) * W * 48 : 0;
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
        meta_->beam_alt_f.data(),
        meta_->beam_az_f.data(),
        rig_->resampleParams(),
        range_buf_.data(),
        signal_buf_.data(),
        reflectivity_buf_.data(),
        nearir_buf_.data(),
        pp);

    publishChannels(stamp_ns);
}

void GzGpuOusterLidarSystem::publishChannels(int64_t stamp_ns)
{
    if (stamp_ns <= 0 || !encoder_ || !ros_) return;

    // ── Packets + image topics ───────────────────────────────────────────────
    encoder_->encodeScan(stamp_ns, sim_epoch_,
                         range_buf_.data(), signal_buf_.data(),
                         reflectivity_buf_.data(), nearir_buf_.data());
    ros_->publishImages(stamp_ns, range_buf_.data(), signal_buf_.data(),
                        reflectivity_buf_.data(), nearir_buf_.data());
}

// ── IMU publishing ──────────────────────────────────────────────────────────

void GzGpuOusterLidarSystem::publishImu(
    const ::gz::sim::UpdateInfo & info,
    const ::gz::sim::EntityComponentManager & ecm)
{
    if (!meta_ || !meta_->pw || !ros_) return;

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
    //
    // world_gravity_ is the WORLD's actual gravity vector (read once in
    // Configure) rather than a hardcoded 9.80665 down-Z, so non-standard
    // worlds (reduced/tilted gravity, e.g. lunar or inclined test rigs) still
    // produce a correct proper-acceleration reference.
    const auto imu_world_pose = ::gz::sim::worldPose(imu_entity_, ecm);
    const auto & R = imu_world_pose.Rot();
    const auto gravity_body = R.RotateVectorReverse(world_gravity_);
    const ::gz::math::Vector3d la_proper = la_raw - gravity_body;

    const auto sim_now =
        std::chrono::duration_cast<std::chrono::nanoseconds>(info.simTime);
    const auto imu_period = periodFromHz(imu_hz_);
    const auto batch = imu_scheduler_.advance(sim_now, imu_period);

    const Vec3 current_av{av.X(), av.Y(), av.Z()};
    const Vec3 current_la{la_proper.X(), la_proper.Y(), la_proper.Z()};

    if (batch.reset) {
        // A world reset starts a new stochastic sensor epoch. Carrying a bias
        // random walk backwards through time makes bag concatenation and
        // repeatable reset tests physically inconsistent.
        gyro_bias_ = {0.0, 0.0, 0.0};
        accel_bias_ = {0.0, 0.0, 0.0};
        imu_state_valid_ = false;
    }
    if (batch.skipped > 0) {
        RCLCPP_WARN_THROTTLE(kLogger, *ros_->clock(), 5000,
            "%s: IMU sim time jumped; skipped %lu oldest catch-up samples",
            sensor_name_.c_str(),
            static_cast<unsigned long>(batch.skipped));
    }

    const Vec3 previous_av = imu_state_valid_ ? imu_prev_av_ : current_av;
    const Vec3 previous_la = imu_state_valid_ ? imu_prev_la_ : current_la;
    const auto previous_time = imu_state_valid_ ? imu_prev_state_time_ : sim_now;
    const int64_t span_ns = (sim_now - previous_time).count();
    const double sample_dt =
        static_cast<double>(imu_period.count()) / 1.0e9;

    auto interpolate = [](const Vec3 & a, const Vec3 & b, double t) {
        return Vec3{
            a.x + (b.x - a.x) * t,
            a.y + (b.y - a.y) * t,
            a.z + (b.z - a.z) * t};
    };
    for (size_t i = 0; i < batch.size; ++i) {
        const double alpha = (span_ns > 0)
            ? std::clamp(
                static_cast<double>((batch.deadlines[i] - previous_time).count()) /
                    static_cast<double>(span_ns),
                0.0, 1.0)
            : 1.0;
        publishImuSample(
            batch.deadlines[i].count(),
            interpolate(previous_av, current_av, alpha),
            interpolate(previous_la, current_la, alpha),
            sample_dt);
    }

    imu_prev_state_time_ = sim_now;
    imu_prev_av_ = current_av;
    imu_prev_la_ = current_la;
    imu_state_valid_ = true;
}

void GzGpuOusterLidarSystem::publishImuSample(
    int64_t stamp_ns,
    const Vec3 & nominal_av,
    const Vec3 & nominal_la,
    double sample_dt)
{
    if (!meta_ || !meta_->pw || !ros_) return;

    // ── IMU noise + bias model ───────────────────────────────────────────
    // Math lives in cuda/imu_noise.{hpp,cpp} so it's testable without
    // spinning Gazebo. Defaults match Ouster Os1 datasheet; downstream
    // filters that subscribe to /imu now see a non-pristine signal.
    if (!imu_rng_seeded_) {
        imu_rng_.seed(deriveNonDeterministicSeed(this));
        imu_rng_seeded_ = true;
    }
    const NoiseParams noise = ros_->noiseSnapshot();
    const ImuNoiseSample noisy = applyImuNoise(
        nominal_av, nominal_la,
        gyro_bias_, accel_bias_,
        noise.gyro_noise_std, noise.accel_noise_std,
        noise.gyro_bias_walk, noise.accel_bias_walk,
        sample_dt,
        imu_rng_);
    const Vec3 & av_meas = noisy.av;
    const Vec3 & la = noisy.la;
    const double gyro_white  = noisy.gyro_white_std;   // for covariance below
    const double accel_white = noisy.accel_white_std;

    // ── Encode Ouster IMU PacketMsg ──────────────────────────────────────
    if (ros_->imuPacketWanted() && !imu_pkt_buf_.empty()) {
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
            meta_->pw->set_imu_nmea_ts(buf, ts);
        }

        // Accel/gyro values — PacketWriter writes at profile-correct offsets.
        meta_->pw->set_imu_la_x(buf, static_cast<float>(la.x));
        meta_->pw->set_imu_la_y(buf, static_cast<float>(la.y));
        meta_->pw->set_imu_la_z(buf, static_cast<float>(la.z));
        meta_->pw->set_imu_av_x(buf, static_cast<float>(av_meas.x));
        meta_->pw->set_imu_av_y(buf, static_cast<float>(av_meas.y));
        meta_->pw->set_imu_av_z(buf, static_cast<float>(av_meas.z));

        ros_->publishImuPacket(imu_pkt_buf_);
    }

    // ── Publish sensor_msgs/Imu for convenience ─────────────────────────
    if (publish_imu_msg_ && ros_->imuMsgWanted()) {
        sensor_msgs::msg::Imu msg;
        msg.header.stamp.sec  = static_cast<int32_t>(stamp_ns / 1000000000LL);
        msg.header.stamp.nanosec = static_cast<uint32_t>(stamp_ns % 1000000000LL);
        msg.header.frame_id = imu_frame_id_;

        msg.angular_velocity.x = av_meas.x;
        msg.angular_velocity.y = av_meas.y;
        msg.angular_velocity.z = av_meas.z;

        msg.linear_acceleration.x = la.x;
        msg.linear_acceleration.y = la.y;
        msg.linear_acceleration.z = la.z;

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

        ros_->publishImuMsg(std::move(msg));
    }
}

}  // namespace gz_gpu_ouster_lidar
