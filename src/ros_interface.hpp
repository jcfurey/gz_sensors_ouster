// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0
//
// All ROS 2 surface area of the plugin: node + executor lifecycle, every
// publisher (packets, images, camera_info, metadata, IMU), QoS policy,
// parameter declaration + the dynamic noise-parameter store, and the
// metadata republish state machine. Components and the plugin publish
// exclusively through this class; publish_mtx_ serialises all publish()
// calls across threads (render, sim, drain) because rmw_zenoh_cpp is not
// guaranteed thread-safe for concurrent publishes on the same node.

#pragma once

#include "lidar_common.hpp"

#include <rclcpp/rclcpp.hpp>
#include <ouster_sensor_msgs/msg/packet_msg.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/string.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace gz_gpu_ouster_lidar {

struct RosInterfaceConfig {
    std::string sensor_name;        ///< topic prefix + node namespace
    std::string image_qos = "reliable";
    std::string imu_qos = "sensor_data";
    std::string image_frame_id;
    std::string imu_frame_id;
    std::string metadata_str;
    int H = 0;
    int W = 0;
    const std::vector<double> * beam_alt_angles = nullptr;  ///< CameraInfo fy
    double lidar_hz = 10.0;
    double max_range = 120.0;
    double imu_hz = 100.0;
    bool imu_enabled = false;
    bool publish_imu_msg = true;
    /// Publish the range/signal/reflec/nearir images + camera_info directly
    /// from the plugin. Default false: in sim the ouster_ros os_image node is
    /// the single image source (driven by lidar_packets + metadata, same as on
    /// hardware), so the plugin's native renditions — which differ (no
    /// auto-exposure, independent noise draws) — would otherwise be a second,
    /// divergent source. Enable only to A/B the native images for debugging.
    bool publish_native_images = false;
};

class RosInterface {
public:
    RosInterface();
    ~RosInterface();  // calls shutdown()

    /// Create node/executor/publishers, declare parameters and start the
    /// spin thread. Throws on failure (caller logs + disables the plugin).
    void init(const RosInterfaceConfig & cfg, const NoiseParams & noise);

    /// Cancel the executor and join the spin thread. Idempotent.
    void shutdown();

    bool ready() const { return static_cast<bool>(node_); }

    /// Clock for throttled logging; null before init().
    rclcpp::Clock * clock() const
    {
        return node_ ? node_->get_clock().get() : nullptr;
    }

    /// Copy of the live noise parameters (written by the ROS parameter
    /// callback).
    NoiseParams noiseSnapshot() const;

    /// Metadata (re)publish state machine; call every sim tick.
    void publishMetadataIfNeeded(std::chrono::nanoseconds sim_now);

    /// Range/signal/reflectivity/near-IR images + CameraInfo (each only
    /// when subscribed). Buffers are H×W in the configured dimensions.
    void publishImages(int64_t stamp_ns,
                       const uint32_t * range, const uint16_t * signal,
                       const uint8_t * refl, const uint16_t * nearir);

    void publishLidarPacket(const ouster_sensor_msgs::msg::PacketMsg & pkt);

    bool imuPacketWanted() const;
    void publishImuPacket(const std::vector<uint8_t> & buf);
    bool imuMsgWanted() const;
    void publishImuMsg(sensor_msgs::msg::Imu && msg);

private:
    RosInterfaceConfig cfg_;

    // Dynamic noise parameters (ROS param callback writes, snapshots read).
    mutable std::mutex noise_mtx_;
    NoiseParams noise_;

    // publish_mtx_ serialises all publish() calls across threads.
    std::mutex publish_mtx_;
    rclcpp::Node::SharedPtr node_;
    // Lazy-construct the executor inside init() *after* rclcpp::init().
    // A value member's constructor would run at plugin instantiation time
    // (gz::plugin::Loader::Instantiate) — before any rclcpp::init() — and
    // crash with "failed to create guard condition: context is null".
    std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> executor_;
    std::thread spin_thread_;

    rclcpp::Publisher<ouster_sensor_msgs::msg::PacketMsg>::SharedPtr pkt_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr meta_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr range_image_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr signal_image_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr reflec_image_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr nearir_image_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_pub_;
    sensor_msgs::msg::CameraInfo camera_info_msg_;
    rclcpp::Publisher<ouster_sensor_msgs::msg::PacketMsg>::SharedPtr imu_pkt_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_msg_pub_;
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr
        param_cb_handle_;

    // Metadata republish state (sim thread; atomics for observability).
    std::atomic<bool> metadata_published_{false};
    std::atomic<int> metadata_pub_count_{0};
    std::chrono::nanoseconds last_meta_pub_time_{-1};
};

}  // namespace gz_gpu_ouster_lidar
