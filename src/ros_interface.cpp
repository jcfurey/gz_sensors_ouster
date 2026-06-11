// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#include "ros_interface.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <builtin_interfaces/msg/time.hpp>

namespace gz_gpu_ouster_lidar {

static const rclcpp::Logger kLogger = lidarLogger();

RosInterface::RosInterface() = default;

RosInterface::~RosInterface()
{
    shutdown();
}

void RosInterface::shutdown()
{
    if (executor_) {
        executor_->cancel();
    }
    if (spin_thread_.joinable()) {
        spin_thread_.join();
    }
}

void RosInterface::init(const RosInterfaceConfig & cfg,
                        const NoiseParams & noise)
{
    cfg_ = cfg;
    noise_ = noise;

    if (!rclcpp::ok()) {
        rclcpp::init(0, nullptr);
    }

    // Construct the executor lazily here, AFTER rclcpp::init() above (see
    // the member comment in the header). Without this, sim setups that
    // don't load gz_ros2_control crash because nothing else has called
    // rclcpp::init() yet.
    executor_ = std::make_unique<rclcpp::executors::SingleThreadedExecutor>();

    rclcpp::NodeOptions opts;
    opts.use_intra_process_comms(false);
    node_ = std::make_shared<rclcpp::Node>(
        "gz_gpu_ouster_lidar", cfg_.sensor_name, opts);

    // Use absolute topic names derived from sensor_name so that Gazebo's
    // process-level namespace contamination doesn't affect topic routing.
    const std::string abs_prefix = cfg_.sensor_name;

    // Latched metadata publisher
    rclcpp::QoS meta_qos{1};
    meta_qos.reliable().transient_local();
    meta_pub_ = node_->create_publisher<std_msgs::msg::String>(
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
    pkt_pub_ = node_->create_publisher<ouster_sensor_msgs::msg::PacketMsg>(
        abs_prefix + "/lidar_packets", pkt_qos);

    // Image + camera_info: configurable via <image_qos> SDF tag, defaults
    // RELIABLE KEEP_LAST(5) to match RViz / rqt_image_view / image_transport
    // default subscribers. Override to "best_effort" if your consumer is
    // Foxglove configured for sensor data.
    const auto image_qos = qos_from_string(cfg_.image_qos, 5);
    range_image_pub_ = node_->create_publisher<sensor_msgs::msg::Image>(
        abs_prefix + "/range_image", image_qos);
    signal_image_pub_ = node_->create_publisher<sensor_msgs::msg::Image>(
        abs_prefix + "/signal_image", image_qos);
    reflec_image_pub_ = node_->create_publisher<sensor_msgs::msg::Image>(
        abs_prefix + "/reflec_image", image_qos);
    nearir_image_pub_ = node_->create_publisher<sensor_msgs::msg::Image>(
        abs_prefix + "/nearir_image", image_qos);

    // CameraInfo publisher (equirectangular projection model for range images)
    camera_info_pub_ = node_->create_publisher<sensor_msgs::msg::CameraInfo>(
        abs_prefix + "/camera_info", image_qos);
    {
        const uint32_t H = static_cast<uint32_t>(cfg_.H);
        const uint32_t W = static_cast<uint32_t>(cfg_.W);

        // Horizontal: each column = 2π/W radians (full rotation; azimuth window
        // only affects data validity, not the column ↔ angle mapping).
        const double fx = static_cast<double>(W) / (2.0 * M_PI);
        const double cx = static_cast<double>(W) / 2.0;

        // Vertical: use actual beam altitude angles for correct VFOV.
        // Generic defaults first; overridden when the metadata carries a
        // usable beam span. Degenerate metadata (all beams at a single
        // altitude) would otherwise divide by zero and emit inf/NaN in K.
        double fy = static_cast<double>(H) / (2.0 * M_PI);
        double cy = static_cast<double>(H) / 2.0;
        if (cfg_.beam_alt_angles && cfg_.beam_alt_angles->size() >= 2) {
            auto [min_it, max_it] = std::minmax_element(
                cfg_.beam_alt_angles->begin(), cfg_.beam_alt_angles->end());
            const double vfov_rad = (*max_it - *min_it) * M_PI / 180.0;
            if (vfov_rad > 1.0e-6) {
                fy = static_cast<double>(H) / vfov_rad;
                const double mean_alt_rad =
                    0.5 * (*max_it + *min_it) * M_PI / 180.0;
                cy = static_cast<double>(H) / 2.0 - mean_alt_rad * fy;
            } else {
                RCLCPP_WARN(kLogger,
                    "beam altitude span is ~0 deg; CameraInfo fy falls back "
                    "to the generic default");
            }
        }

        camera_info_msg_.header.frame_id = cfg_.image_frame_id;
        camera_info_msg_.height = H;
        camera_info_msg_.width = W;
        // The range image is an equirectangular panorama (u linear in
        // azimuth, v linear in elevation) — no standard ROS distortion
        // model describes it. Declare the honest non-standard name rather
        // than "equidistant" (a fisheye model) so consumers fail loudly
        // instead of silently mis-projecting; fx/fy in K are the
        // pixels-per-radian scale factors of the angular mapping.
        camera_info_msg_.distortion_model = "equirectangular";
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
    if (cfg_.imu_enabled) {
        const auto imu_qos = qos_from_string(cfg_.imu_qos, 10);
        imu_pkt_pub_ = node_->create_publisher<ouster_sensor_msgs::msg::PacketMsg>(
            abs_prefix + "/imu_packets", imu_qos);
        if (cfg_.publish_imu_msg) {
            imu_msg_pub_ = node_->create_publisher<sensor_msgs::msg::Imu>(
                abs_prefix + "/imu", imu_qos);
        }
    }

    // ── Declare ROS 2 parameters ──────────────────────────────────────────
    // Structural parameters are declared read_only so rclcpp itself rejects
    // runtime changes with a proper error (no manual rejection needed).
    rcl_interfaces::msg::ParameterDescriptor read_only;
    read_only.read_only = true;
    node_->declare_parameter("lidar_hz", cfg_.lidar_hz, read_only);
    node_->declare_parameter("max_range", cfg_.max_range, read_only);
    if (cfg_.imu_enabled) {
        node_->declare_parameter("imu_hz", cfg_.imu_hz, read_only);
    }

    // Dynamically reconfigurable noise model parameters.
    node_->declare_parameter("range_noise_min_std", noise_.range_noise_min_std);
    node_->declare_parameter("range_noise_max_std", noise_.range_noise_max_std);
    node_->declare_parameter("signal_noise_scale", noise_.signal_noise_scale);
    node_->declare_parameter("nearir_noise_scale", noise_.nearir_noise_scale);
    node_->declare_parameter("dropout_rate_close", noise_.dropout_rate_close);
    node_->declare_parameter("dropout_rate_far", noise_.dropout_rate_far);
    node_->declare_parameter("false_alarm_rate", noise_.false_alarm_rate);
    node_->declare_parameter("edge_discon_threshold", noise_.edge_discon_threshold);
    node_->declare_parameter("base_signal", noise_.base_signal);
    node_->declare_parameter("base_reflectivity", noise_.base_reflectivity);
    if (cfg_.imu_enabled) {
        node_->declare_parameter("gyro_noise_std",   noise_.gyro_noise_std);
        node_->declare_parameter("accel_noise_std",  noise_.accel_noise_std);
        node_->declare_parameter("gyro_bias_walk",   noise_.gyro_bias_walk);
        node_->declare_parameter("accel_bias_walk",  noise_.accel_bias_walk);
    }

    // Structural params (lidar_hz/max_range/imu_hz) are read_only above —
    // rclcpp rejects writes to them before this callback ever runs.
    param_cb_handle_ = node_->add_on_set_parameters_callback(
        [this](const std::vector<rclcpp::Parameter> & params)
            -> rcl_interfaces::msg::SetParametersResult {
            // Apply noise model updates under lock (sim thread reads these).
            {
                std::lock_guard<std::mutex> lk(noise_mtx_);
                for (const auto & p : params) {
                    const auto & name = p.get_name();
                    // Out-of-range values are clamped AND reported.
                    auto clamped = [&p](double lo, double hi) {
                        const double v = p.as_double();
                        // NaN: std::clamp would return NaN — sanitise to
                        // the lower bound instead (and say so).
                        if (std::isnan(v)) {
                            RCLCPP_WARN(kLogger, "%s is NaN; using %g",
                                p.get_name().c_str(), lo);
                            return lo;
                        }
                        const double c = std::clamp(v, lo, hi);
                        if (c != v) {
                            RCLCPP_WARN(kLogger,
                                "%s=%g outside [%g, %g]; clamped to %g",
                                p.get_name().c_str(), v, lo, hi, c);
                        }
                        return c;
                    };
                    constexpr double kInfD =
                        std::numeric_limits<double>::infinity();
                    NoiseParams & np = noise_;
                    if (name == "range_noise_min_std")       np.range_noise_min_std    = clamped(0.0, kInfD);
                    else if (name == "range_noise_max_std")  np.range_noise_max_std    = clamped(0.0, kInfD);
                    else if (name == "signal_noise_scale")   np.signal_noise_scale     = clamped(0.0, kInfD);
                    else if (name == "nearir_noise_scale")   np.nearir_noise_scale     = clamped(0.0, kInfD);
                    else if (name == "dropout_rate_close")   np.dropout_rate_close     = clamped(0.0, 1.0);
                    else if (name == "dropout_rate_far")     np.dropout_rate_far       = clamped(0.0, 1.0);
                    else if (name == "false_alarm_rate")     np.false_alarm_rate       = clamped(0.0, 1.0);
                    else if (name == "edge_discon_threshold") np.edge_discon_threshold  = clamped(0.0, kInfD);
                    else if (name == "base_signal")          np.base_signal            = clamped(0.0, kInfD);
                    else if (name == "base_reflectivity")    np.base_reflectivity      = clamped(0.0, 255.0);
                    else if (name == "gyro_noise_std")       np.gyro_noise_std         = clamped(0.0, kInfD);
                    else if (name == "accel_noise_std")      np.accel_noise_std        = clamped(0.0, kInfD);
                    else if (name == "gyro_bias_walk")       np.gyro_bias_walk         = clamped(0.0, kInfD);
                    else if (name == "accel_bias_walk")      np.accel_bias_walk        = clamped(0.0, kInfD);
                }
            }
            rcl_interfaces::msg::SetParametersResult r;
            r.successful = true;
            return r;
        });

    // Spin the node on a background thread so Zenoh completes peer discovery.
    // Without this, rmw_zenoh_cpp never processes incoming control messages
    // and publish() calls go undelivered even when subscribers exist.
    executor_->add_node(node_);
    spin_thread_ = std::thread([this]() {
        executor_->spin();
    });

    // Metadata publishing happens in publishMetadataIfNeeded() (driven from
    // PostUpdate, so it works in both ray modes): first publish on the first
    // sim tick — by which point the executor thread above is pumping Zenoh
    // control messages — then sim-time-throttled republish until a
    // subscriber has acked. No startup sleep needed.
    RCLCPP_INFO(kLogger, "Metadata will publish on %s/metadata",
                abs_prefix.c_str());
}

NoiseParams RosInterface::noiseSnapshot() const
{
    std::lock_guard<std::mutex> lk(noise_mtx_);
    return noise_;
}

// ── Metadata publishing ──────────────────────────────────────────────────────
// Driven from PostUpdate (sim thread) so it works in BOTH ray modes.
// rmw_zenoh_cpp transient_local replay is unreliable across processes and
// get_subscription_count() may lag actual Zenoh peer state, so republish on
// a sim-time throttle until a subscriber is visible AND enough copies have
// gone out for Zenoh discovery to settle; re-arm if the subscriber drops
// (Foxglove tab refresh, os_cloud restart, network blip).

void RosInterface::publishMetadataIfNeeded(std::chrono::nanoseconds sim_now)
{
    if (!meta_pub_) return;

    constexpr auto kRepubPeriod = std::chrono::milliseconds(500);
    constexpr int kMinPublishes = 5;  // ≥ 2 s of copies before declaring done

    const auto meta_subs = meta_pub_->get_subscription_count();
    if (metadata_published_ && meta_subs == 0) {
        metadata_published_ = false;
        metadata_pub_count_ = 0;
        last_meta_pub_time_ = std::chrono::nanoseconds(-1);
        RCLCPP_INFO(kLogger,
            "metadata subscriber count dropped to 0; re-arming republish");
    }
    if (metadata_published_) return;

    if (last_meta_pub_time_.count() < 0 ||
        sim_now - last_meta_pub_time_ >= kRepubPeriod) {
        std_msgs::msg::String meta_msg;
        meta_msg.data = cfg_.metadata_str;
        {
            std::lock_guard<std::mutex> pub_lk(publish_mtx_);
            meta_pub_->publish(meta_msg);
        }
        ++metadata_pub_count_;
        last_meta_pub_time_ = sim_now;
    }

    if (meta_subs > 0 && metadata_pub_count_ >= kMinPublishes) {
        metadata_published_ = true;
        RCLCPP_INFO(kLogger,
            "metadata delivered to os_cloud (after %d publishes)",
            metadata_pub_count_.load());
    }
}

// ── Image topics ─────────────────────────────────────────────────────────────

void RosInterface::publishImages(int64_t stamp_ns,
                                 const uint32_t * range,
                                 const uint16_t * signal,
                                 const uint8_t * refl,
                                 const uint16_t * nearir)
{
    const auto n = static_cast<size_t>(cfg_.H) * cfg_.W;
    const uint32_t h = static_cast<uint32_t>(cfg_.H);
    const uint32_t w = static_cast<uint32_t>(cfg_.W);
    const uint32_t step = w * static_cast<uint32_t>(sizeof(uint16_t));

    builtin_interfaces::msg::Time stamp;
    stamp.sec  = static_cast<int32_t>(stamp_ns / 1000000000LL);
    stamp.nanosec = static_cast<uint32_t>(stamp_ns % 1000000000LL);

    auto make_image = [&]() -> sensor_msgs::msg::Image {
        sensor_msgs::msg::Image msg;
        msg.header.stamp = stamp;
        msg.header.frame_id = cfg_.image_frame_id;
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
            const uint32_t r = (range[i] + 2u) >> 2;
            pixels[i] = (r > 65535u) ? 0u : static_cast<uint16_t>(r);
        }
        std::lock_guard<std::mutex> pub_lk(publish_mtx_);
        range_image_pub_->publish(std::move(msg));
    }

    // Signal image: uint16 values from CUDA processing (photon counts)
    if (signal_image_pub_->get_subscription_count() > 0) {
        auto msg = make_image();
        std::memcpy(msg.data.data(), signal, n * sizeof(uint16_t));
        std::lock_guard<std::mutex> pub_lk(publish_mtx_);
        signal_image_pub_->publish(std::move(msg));
    }

    // Reflectivity image: uint8 → uint16  (×257 maps [0,255] → [0,65535])
    if (reflec_image_pub_->get_subscription_count() > 0) {
        auto msg = make_image();
        auto * pixels = reinterpret_cast<uint16_t *>(msg.data.data());
        for (size_t i = 0; i < n; ++i) {
            pixels[i] = static_cast<uint16_t>(refl[i]) * 257u;
        }
        std::lock_guard<std::mutex> pub_lk(publish_mtx_);
        reflec_image_pub_->publish(std::move(msg));
    }

    // Near-IR image: produced by the noise kernel with Poisson noise via
    // nearir_noise_scale.
    if (nearir_image_pub_->get_subscription_count() > 0) {
        auto msg = make_image();
        std::memcpy(msg.data.data(), nearir, n * sizeof(uint16_t));
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

// ── Packet + IMU publish ─────────────────────────────────────────────────────

void RosInterface::publishLidarPacket(
    const ouster_sensor_msgs::msg::PacketMsg & pkt)
{
    std::lock_guard<std::mutex> pub_lk(publish_mtx_);
    pkt_pub_->publish(pkt);
}

bool RosInterface::imuPacketWanted() const
{
    return imu_pkt_pub_ && imu_pkt_pub_->get_subscription_count() > 0;
}

void RosInterface::publishImuPacket(const std::vector<uint8_t> & buf)
{
    ouster_sensor_msgs::msg::PacketMsg pkt;
    pkt.buf.assign(buf.begin(), buf.end());
    std::lock_guard<std::mutex> pub_lk(publish_mtx_);
    imu_pkt_pub_->publish(std::move(pkt));
}

bool RosInterface::imuMsgWanted() const
{
    return imu_msg_pub_ && imu_msg_pub_->get_subscription_count() > 0;
}

void RosInterface::publishImuMsg(sensor_msgs::msg::Imu && msg)
{
    std::lock_guard<std::mutex> pub_lk(publish_mtx_);
    imu_msg_pub_->publish(std::move(msg));
}

}  // namespace gz_gpu_ouster_lidar
