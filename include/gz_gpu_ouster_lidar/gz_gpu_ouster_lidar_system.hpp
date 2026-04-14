// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gz/sim/System.hh>
#include <gz/common/Event.hh>
#include <gz/math/Pose3.hh>
#include <gz/rendering/GpuRays.hh>

#include <rclcpp/rclcpp.hpp>
#include <ouster_sensor_msgs/msg/packet_msg.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/string.hpp>

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <atomic>
#include <condition_variable>
#include <cstdint>

namespace ouster::sdk::core {
    class SensorInfo;
    namespace impl { class PacketWriter; }
}

namespace gz_gpu_ouster_lidar {

class CudaRayProcessor;

/// Gazebo Harmonic system plugin that creates a GpuRays sensor with
/// per-beam non-uniform elevation angles matching Ouster calibration metadata,
/// then encodes the raw depth buffer into Ouster PacketMsg via CUDA.
class GzGpuOusterLidarSystem
        : public ::gz::sim::System,
            public ::gz::sim::ISystemConfigure,
            public ::gz::sim::ISystemPostUpdate
{
public:
    GzGpuOusterLidarSystem();
    ~GzGpuOusterLidarSystem() override;

    // ISystemConfigure — called once when the plugin is loaded
    void Configure(
        const ::gz::sim::Entity & entity,
        const std::shared_ptr<const sdf::Element> & sdf,
        ::gz::sim::EntityComponentManager & ecm,
        ::gz::sim::EventManager & event_mgr) override;

    // ISystemPostUpdate — called every sim step (read-only, parallel-safe)
    void PostUpdate(
        const ::gz::sim::UpdateInfo & info,
        const ::gz::sim::EntityComponentManager & ecm) override;

private:
    // ── Configuration ────────────────────────────────────────────────────────
    std::string metadata_path_;
    std::string sensor_name_;
    std::string world_name_;
    double lidar_hz_ = 10.0;
    uint32_t visibility_mask_ = 0xFFFFFFFFu;

    // ── Noise model parameters (SDF-configurable) ─────────────────────────────
    // Defaults tuned to OS1 rev6 (real hardware on this platform).
    // OS1 has tighter beams and longer range than OS0, yielding better
    // Validated against ISPRS accuracy assessment of OS1-64 and Ouster FW 1.13+
    // datasheets.  Override via SDF for OS0/OS2.
    // Guarded by noise_mtx_ (written by ROS param callback, read by sim thread).
    mutable std::mutex noise_mtx_;
    double range_noise_min_std_ = 0.003;   // 3 mm σ — OS1 FW 1.13+ best precision
    double range_noise_max_std_ = 0.015;   // 15 mm σ — OS1 empirical at max range
    double signal_noise_scale_  = 1.0;     // Poisson shot noise (physically correct)
    double nearir_noise_scale_  = 1.0;     // Near-IR follows same photon statistics
    double dropout_rate_close_  = 0.0005;  // 0.05% — OS1 near-zero at short range
    double dropout_rate_far_    = 0.03;    // 3.0% — low-reflectivity misses at max range
    double edge_discon_threshold_ = 0.15;  // 0.15 m — 1ns echo delay convention
    double base_signal_ = 800.0;           // OS1 higher photon budget than OS0
    double base_reflectivity_ = 50.0;      // Default reflectivity [0–255]
    double max_range_ = 120.0;             // Sensor max range (metres), default OS1
    bool max_range_explicit_ = false;      // true if set via SDF (don't auto-derive)

    // ── IMU configuration (SDF-optional) ─────────────────────────────────────
    std::string imu_name_;                  // Gazebo IMU sensor entity name
    double imu_hz_ = 100.0;                // IMU publish rate (Hz)
    bool imu_enabled_ = false;             // true if <imu_name> was provided
    bool publish_imu_msg_ = true;          // also publish sensor_msgs/Imu

    // ── Ouster metadata ──────────────────────────────────────────────────────
    std::string metadata_str_;
    int H_ = 0;                     // pixels_per_column (beam count)
    int W_ = 0;                     // columns_per_frame
    int cpp_ = 0;                   // columns_per_packet
    std::vector<double> beam_alt_angles_;   // per-beam elevation (degrees)
    std::vector<double> beam_az_offsets_;   // per-beam azimuth offset (degrees)
    double beam_origin_mm_ = 0.0;          // lidar_origin_to_beam_origin_mm

    // ── IMU packet format ────────────────────────────────────────────────────
    size_t imu_packet_size_ = 0;
    std::vector<uint8_t> imu_pkt_buf_;

    // ── Packet writer ────────────────────────────────────────────────────────
    std::unique_ptr<ouster::sdk::core::impl::PacketWriter> pw_;
    std::vector<uint8_t> pkt_buf_;
    uint32_t frame_id_ = 0;

    // ── GpuRays sensor ───────────────────────────────────────────────────────
    ::gz::sim::Entity sensor_entity_{::gz::sim::kNullEntity};
    std::atomic<bool> sensor_initialized_{false};

    // Rendering scene / sensor pointers (set after rendering init)
    ::gz::rendering::GpuRaysPtr gpu_rays_;
    bool frame_connected_ = false;
    gz::common::ConnectionPtr frame_conn_;

    // ── Lidar-frame pose tracking ────────────────────────────────────────────
    std::string lidar_frame_name_;          // URDF link name, e.g. "lidar0/lidar_frame"
    ::gz::sim::Entity lidar_frame_entity_{::gz::sim::kNullEntity};
    bool lidar_frame_found_ = false;
    std::mutex pose_mtx_;
    ::gz::math::Pose3d cached_pose_;

    // ── IMU entity tracking ──────────────────────────────────────────────────
    ::gz::sim::Entity imu_entity_{::gz::sim::kNullEntity};
    bool imu_entity_found_ = false;

    // ── Rendering-thread event connection ────────────────────────────────────
    ::gz::sim::EventManager *event_mgr_{nullptr};
    gz::common::ConnectionPtr render_conn_;
    std::chrono::steady_clock::time_point last_render_time_{};
    void OnRender();
    void DestroyGpuRays();

    // ── CUDA processor ───────────────────────────────────────────────────────
    std::unique_ptr<CudaRayProcessor> cuda_processor_;

    // ── Channel buffers ──────────────────────────────────────────────────────
    // Allocated as H×W, filled by CUDA, encoded into packets.
    std::vector<uint32_t> range_buf_;
    std::vector<uint16_t> signal_buf_;
    std::vector<uint8_t>  reflectivity_buf_;
    std::vector<uint16_t> nearir_buf_;          // NEAR_IR channel for packet encoding

    // Raw GpuRays frame staging buffer (fast memcpy in callback, GPU-processed later)
    std::vector<float> raw_frame_buf_;
    int raw_frame_H_ = 0;
    int raw_frame_W_ = 0;
    int raw_frame_chan_ = 0;
    std::vector<float> beam_alt_f_;     // beam_alt_angles_ as float (for GPU upload)
    std::vector<float> beam_az_f_;      // beam_az_offsets_ as float
    std::mutex frame_mtx_;
    std::atomic<bool> frame_ready_{false};

    // ── ROS 2 node & publishers ──────────────────────────────────────────────
    // publish_mtx_ serialises all publish() calls across threads (render,
    // simulation/PostUpdate, drain).  rmw_zenoh_cpp is not guaranteed
    // thread-safe for concurrent publishes on the same node.
    std::mutex publish_mtx_;
    rclcpp::Node::SharedPtr ros_node_;
    rclcpp::executors::SingleThreadedExecutor ros_executor_;
    std::thread ros_spin_thread_;
    rclcpp::Publisher<ouster_sensor_msgs::msg::PacketMsg>::SharedPtr pkt_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr meta_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr range_image_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr signal_image_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr reflec_image_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr nearir_image_pub_;
    rclcpp::Publisher<ouster_sensor_msgs::msg::PacketMsg>::SharedPtr imu_pkt_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_msg_pub_;
    std::string image_frame_id_;
    std::string imu_frame_id_;
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb_handle_;
    bool metadata_published_ = false;   // true once a subscriber has acked
    int metadata_pub_count_ = 0;       // render ticks since sensor init

    // ── Drain thread ─────────────────────────────────────────────────────────
    std::vector<ouster_sensor_msgs::msg::PacketMsg> drain_pkts_;
    std::thread drain_thread_;
    std::mutex drain_mtx_;
    std::condition_variable drain_cv_;
    std::atomic<bool> drain_ready_{false};
    std::atomic<bool> shutdown_{false};

    // ── IMU timing ───────────────────────────────────────────────────────────
    std::chrono::nanoseconds last_imu_sim_time_{0};

    // ── Pause/resume detection ───────────────────────────────────────────────
    bool was_paused_ = false;

    // ── Private methods ──────────────────────────────────────────────────────
    bool loadMetadata();
    void initRosInterface();
    void onNewFrame(const float * data, unsigned int width,
                    unsigned int height, unsigned int channels,
                    const std::string & format);
    void encodeAndPublish(int64_t stamp_ns,
                          const float * raw_data, int gpu_H, int gpu_W, int gpu_chan);
    void publishImages(int64_t stamp_ns);
    void publishImu(const ::gz::sim::UpdateInfo & info,
                    const ::gz::sim::EntityComponentManager & ecm);
    void drainThreadFunc();
    bool HasActiveLidarSubscribers() const;
};

}  // namespace gz_gpu_ouster_lidar
