#include "gz_gpu_ouster_lidar/gz_gpu_ouster_lidar_system.hpp"
#include "gz_gpu_ouster_lidar/cuda_ray_processor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
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
#include <ouster_sensor_msgs/msg/packet_msg.hpp>
#include <std_msgs/msg/string.hpp>

namespace gz_gpu_ouster_lidar {

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
    }
    if (sdf->HasElement("sensor_name")) {
        sensor_name_ = sdf->Get<std::string>("sensor_name");
    }
    if (sdf->HasElement("lidar_hz")) {
        lidar_hz_ = sdf->Get<double>("lidar_hz");
    }

    if (metadata_path_.empty()) {
        std::cerr << "[GzGpuOusterLidar] 'metadata_path' SDF parameter is required\n";
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
    loadMetadata();

    // Initialise CUDA processor
    cuda_processor_ = std::make_unique<CudaRayProcessor>();

    // Allocate channel buffers
    const int n = H_ * W_;
    range_buf_.resize(static_cast<size_t>(n), 0);
    signal_buf_.resize(static_cast<size_t>(n), 0);
    reflectivity_buf_.resize(static_cast<size_t>(n), 50);
    depth_buf_.resize(static_cast<size_t>(n), 0.0f);
    retro_buf_.resize(static_cast<size_t>(n), 0.0f);

    // Initialise ROS 2 node and publishers
    initRosInterface();

    // Start drain thread
    drain_thread_ = std::thread(&GzGpuOusterLidarSystem::drainThreadFunc, this);

    // Connect to the rendering-thread Render event (fires after scene->PreRender()
    // at Sensors.cc:496) so that GpuRays initialisation and Render() happen on
    // the correct (EGL) thread with the scene already set up.
    event_mgr_ = &event_mgr;
    render_conn_ = event_mgr.Connect<::gz::sim::events::Render>(
        std::bind(&GzGpuOusterLidarSystem::OnRender, this));

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

    std::cout << "[GzGpuOusterLidar] Configured: H=" << H_ << " W=" << W_
          << " cpp=" << cpp_ << " sensor_name=" << sensor_name_
          << " gz_sensor=" << lidar_frame_name_
          << " hz=" << lidar_hz_ << "\n";
}

// ── Metadata loading ─────────────────────────────────────────────────────────

void GzGpuOusterLidarSystem::loadMetadata()
{
    // Read raw JSON
    std::ifstream fs(metadata_path_);
    if (!fs.is_open()) {
        std::cerr << "[GzGpuOusterLidar] Cannot open metadata: " << metadata_path_ << "\n";
        return;
    }
    std::ostringstream ss;
    ss << fs.rdbuf();
    metadata_str_ = ss.str();

    // Parse via Ouster SDK for PacketWriter
    ouster::sdk::core::SensorInfo info(metadata_str_);
    ouster::sdk::core::PacketFormat pf(info);
    pw_ = std::make_unique<ouster::sdk::core::impl::PacketWriter>(pf);

    H_   = pw_->pixels_per_column;
    W_   = static_cast<int>(info.format.columns_per_frame);
    cpp_ = pw_->columns_per_packet;

    if (cpp_ <= 0 || W_ % cpp_ != 0) {
        std::cerr << "[GzGpuOusterLidar] columns_per_frame (" << W_
              << ") not divisible by columns_per_packet (" << cpp_ << ")\n";
        return;
    }

    pkt_buf_.resize(pw_->lidar_packet_size, 0);

    // Beam intrinsics are available directly on SensorInfo.
    beam_alt_angles_ = info.beam_altitude_angles;
    beam_az_offsets_ = info.beam_azimuth_angles;
    beam_origin_mm_  = info.lidar_origin_to_beam_origin_mm;

    if (static_cast<int>(beam_alt_angles_.size()) != H_) {
        std::cerr << "[GzGpuOusterLidar] beam_altitude_angles size ("
              << beam_alt_angles_.size() << ") != H (" << H_ << ")\n";
    }
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
    meta_qos.transient_local();
    meta_pub_ = ros_node_->create_publisher<std_msgs::msg::String>(
        abs_prefix + "/metadata", meta_qos);

    std_msgs::msg::String meta_msg;
    meta_msg.data = metadata_str_;
    meta_pub_->publish(meta_msg);

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

    // Spin the node on a background thread so Zenoh completes peer discovery.
    // Without this, rmw_zenoh_cpp never processes incoming control messages
    // and publish() calls go undelivered even when subscribers exist.
    ros_executor_.add_node(ros_node_);
    ros_spin_thread_ = std::thread([this]() {
        ros_executor_.spin();
    });
}

// ── Rendering-thread callback ────────────────────────────────────────────────
// Fired by the Sensors system on its rendering thread (events::Render, line 496
// in Sensors.cc) AFTER scene->PreRender() has been called.  All gz::rendering
// calls happen here (never from PostUpdate).

void GzGpuOusterLidarSystem::OnRender()
{
    if (sensor_initialized_.load(std::memory_order_acquire)) {
        // Republish metadata on every render tick until os_cloud subscribes.
        // Zenoh's transient_local does not replay to late subscribers without
        // a running executor, so we push the message repeatedly until confirmed.
        if (!metadata_published_) {
            std_msgs::msg::String meta_msg;
            meta_msg.data = metadata_str_;
            meta_pub_->publish(meta_msg);
            if (meta_pub_->get_subscription_count() > 0) {
                metadata_published_ = true;
                std::cout << "[GzGpuOusterLidar] metadata delivered to os_cloud\n";
            }
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
        std::cerr << "[GzGpuOusterLidar] Failed to create GpuRays sensor\n";
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
    gpuRays->SetFarClipPlane(50.0);

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

    std::cout << "[GzGpuOusterLidar] GpuRays sensor created: "
              << W_ << "x" << v_samples << " rays, "
              << "vertical FOV [" << min_alt << ", " << max_alt << "] deg\n";
}

// ── ISystemPostUpdate ────────────────────────────────────────────────────────

void GzGpuOusterLidarSystem::PostUpdate(
    const ::gz::sim::UpdateInfo & info,
    const ::gz::sim::EntityComponentManager & ecm)
{
    if (info.paused) return;
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
                    std::cout << "[GzGpuOusterLidar] Found sensor entity: "
                              << lidar_frame_name_ << " (id=" << ent << ")\n";
                    return false;  // stop iteration
                }
                return true;  // continue
            });
    }

    // ── Cache world pose for the rendering thread ────────────────────────────
    if (lidar_frame_found_) {
        auto worldPose = ::gz::sim::worldPose(lidar_frame_entity_, ecm);
        std::lock_guard<std::mutex> lk(pose_mtx_);
        cached_pose_ = worldPose;
    }

    // ── Process any pending frame ────────────────────────────────────────────
    bool have_frame = false;
    {
        std::lock_guard<std::mutex> lk(frame_mtx_);
        have_frame = frame_ready_;
        frame_ready_ = false;
    }

    if (have_frame) {
        const auto stamp_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                info.simTime).count();
        encodeAndPublish(stamp_ns);
    }
}

// ── GpuRays frame callback ──────────────────────────────────────────────────

void GzGpuOusterLidarSystem::onNewFrame(
    const float * data, unsigned int width,
    unsigned int height, unsigned int channels,
    const std::string & /*format*/)
{
    // GpuRays returns width × height × channels floats.
    // Channels: [0]=depth, [1]=retro, [2]=unused
    // We need to resample vertical from the uniform grid to exact beam angles.

    const int gpu_H      = static_cast<int>(height);
    const int gpu_W      = static_cast<int>(width);
    const int gpu_chan    = static_cast<int>(channels);

    std::lock_guard<std::mutex> lk(frame_mtx_);

    // Compute the vertical angle range of the GpuRays sensor
    double min_alt = *std::min_element(beam_alt_angles_.begin(), beam_alt_angles_.end());
    double max_alt = *std::max_element(beam_alt_angles_.begin(), beam_alt_angles_.end());
    constexpr double margin_deg = 1.0;
    min_alt -= margin_deg;
    max_alt += margin_deg;

    const double v_range = max_alt - min_alt;  // degrees

    // For each Ouster beam (row), find the depth at its exact elevation angle
    // by interpolating the uniform GpuRays grid.
    const int half_W = W_ / 2;  // azimuth remapping offset

    for (int beam = 0; beam < H_ && beam < static_cast<int>(beam_alt_angles_.size()); ++beam) {
        const double beam_angle = beam_alt_angles_[static_cast<size_t>(beam)];

        // Fractional row in the GpuRays grid.
        // GpuRays row 0 = VerticalAngleMin (bottom), row gpu_H-1 = VerticalAngleMax (top).
        const double frac = (beam_angle - min_alt) / v_range;
        const double row_f = frac * (gpu_H - 1);
        const int row_lo = std::clamp(static_cast<int>(std::floor(row_f)), 0, gpu_H - 1);
        const int row_hi = std::clamp(row_lo + 1, 0, gpu_H - 1);
        const float alpha = static_cast<float>(row_f - row_lo);

        for (int col = 0; col < W_ && col < gpu_W; ++col) {
            const int idx_lo = (row_lo * gpu_W + col) * gpu_chan;
            const int idx_hi = (row_hi * gpu_W + col) * gpu_chan;

            // Bilinear interpolation of depth
            const float d_lo = data[idx_lo];
            const float d_hi = data[idx_hi];

            float depth;
            if (std::isinf(d_lo) || std::isinf(d_hi)) {
                // Either beam missed — use whichever is valid, or inf if both miss
                depth = std::isinf(d_lo) ? d_hi : d_lo;
            } else {
                depth = d_lo * (1.0f - alpha) + d_hi * alpha;
            }

            // Azimuth remapping: Gazebo col 0 = −π, Ouster m_id 0 = encoder 0 (+X_lidar).
            // Formula mirrors gz_ouster_packet_bridge: m_id = (W/2 − col + W) % W
            const int m_id = (half_W - col + W_) % W_;
            const int ouster_idx = beam * W_ + m_id;
            depth_buf_[static_cast<size_t>(ouster_idx)] = depth;

            // Retro (intensity) channel
            if (gpu_chan >= 2) {
                const float r_lo = data[idx_lo + 1];
                const float r_hi = data[idx_hi + 1];
                retro_buf_[static_cast<size_t>(ouster_idx)] =
                    r_lo * (1.0f - alpha) + r_hi * alpha;
            }
        }
    }

    frame_ready_ = true;
}

// ── Encode depth → Ouster packets ───────────────────────────────────────────

void GzGpuOusterLidarSystem::encodeAndPublish(int64_t stamp_ns)
{
    if (stamp_ns <= 0) return;

    // ── CUDA post-processing ─────────────────────────────────────────────────
    RayProcessParams params;
    params.H = H_;
    params.W = W_;
    params.base_signal = 500.0f;
    params.base_reflectivity = 50.0f;
    params.range_noise_std = 0.0f;
    params.dt_per_col_ns = static_cast<uint64_t>(
        static_cast<int64_t>(1e9 / lidar_hz_) / W_);

    cuda_processor_->process(
        depth_buf_.data(),
        retro_buf_.data(),
        range_buf_.data(),
        signal_buf_.data(),
        reflectivity_buf_.data(),
        params);

    // ── Build packets ────────────────────────────────────────────────────────
    const int n_packets = W_ / cpp_;
    const int64_t scan_period_ns = static_cast<int64_t>(1e9 / lidar_hz_);
    const int64_t scan_start_ns  = std::max(int64_t{0}, stamp_ns - scan_period_ns);
    const int64_t dt_col_ns      = scan_period_ns / W_;

    // Map raw buffers into Eigen for PacketWriter
    using RangeMatrix = Eigen::Map<ouster::sdk::core::img_t<uint32_t>>;
    using SignalMatrix = Eigen::Map<ouster::sdk::core::img_t<uint16_t>>;
    using ReflMatrix = Eigen::Map<ouster::sdk::core::img_t<uint8_t>>;

    RangeMatrix  range_mat(range_buf_.data(), H_, W_);
    SignalMatrix  signal_mat(signal_buf_.data(), H_, W_);
    ReflMatrix    refl_mat(reflectivity_buf_.data(), H_, W_);

    drain_pkts_.resize(static_cast<size_t>(n_packets));

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

        drain_pkts_[static_cast<size_t>(p)].buf.assign(pkt_buf_.begin(), pkt_buf_.end());
    }

    ++frame_id_;

    // ── Publish image topics ─────────────────────────────────────────────────
    publishImages(stamp_ns);

    // ── Wake drain thread ────────────────────────────────────────────────────
    {
        std::lock_guard<std::mutex> lk(drain_mtx_);
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
        range_image_pub_->publish(std::move(msg));
    }

    // Signal image: uint16 values from CUDA processing (photon counts)
    if (signal_image_pub_->get_subscription_count() > 0) {
        auto msg = make_image();
        std::memcpy(msg.data.data(), signal_buf_.data(), n * sizeof(uint16_t));
        signal_image_pub_->publish(std::move(msg));
    }

    // Reflectivity image: uint8 → uint16  (×257 maps [0,255] → [0,65535])
    if (reflec_image_pub_->get_subscription_count() > 0) {
        auto msg = make_image();
        auto * pixels = reinterpret_cast<uint16_t *>(msg.data.data());
        for (size_t i = 0; i < n; ++i) {
            pixels[i] = static_cast<uint16_t>(reflectivity_buf_[i]) * 257u;
        }
        reflec_image_pub_->publish(std::move(msg));
    }

    // Near-IR image: GpuRays retro channel → uint16
    // Retro represents surface reflectance; closest sim analogue to ambient IR.
    if (nearir_image_pub_->get_subscription_count() > 0) {
        auto msg = make_image();
        auto * pixels = reinterpret_cast<uint16_t *>(msg.data.data());
        for (size_t i = 0; i < n; ++i) {
            const float v = std::clamp(retro_buf_[i] * 256.0f, 0.0f, 65535.0f);
            pixels[i] = static_cast<uint16_t>(v);
        }
        nearir_image_pub_->publish(std::move(msg));
    }
}

// ── Drain thread ─────────────────────────────────────────────────────────────

void GzGpuOusterLidarSystem::drainThreadFunc()
{
    while (!shutdown_.load(std::memory_order_acquire)) {
        {
            std::unique_lock<std::mutex> lk(drain_mtx_);
            drain_cv_.wait(lk, [this] {
                return drain_ready_.load(std::memory_order_acquire) ||
                       shutdown_.load(std::memory_order_acquire);
            });
            if (shutdown_.load(std::memory_order_acquire)) break;
            drain_ready_.store(false, std::memory_order_release);
        }

        if (drain_pkts_.empty()) continue;

        const auto spacing = std::chrono::nanoseconds(
            static_cast<int64_t>(1e9 / lidar_hz_) /
            static_cast<int64_t>(drain_pkts_.size()));

        for (size_t i = 0; i < drain_pkts_.size(); ++i) {
            if (shutdown_.load(std::memory_order_acquire)) return;
            pkt_pub_->publish(drain_pkts_[i]);
            if (i + 1 < drain_pkts_.size()) {
                std::this_thread::sleep_for(spacing);
            }
        }
    }
}

}  // namespace gz_gpu_ouster_lidar
