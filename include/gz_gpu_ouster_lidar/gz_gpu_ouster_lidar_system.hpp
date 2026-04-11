#pragma once

#include <gz/sim/System.hh>
#include <gz/common/Event.hh>
#include <gz/math/Pose3.hh>
#include <gz/rendering/GpuRays.hh>

#include <rclcpp/rclcpp.hpp>
#include <ouster_sensor_msgs/msg/packet_msg.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <atomic>
#include <condition_variable>

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

    // ── Noise model parameters (SDF-configurable, Ouster-like defaults) ──────
    double range_noise_min_std_ = 0.005;   // 5 mm σ at close range
    double range_noise_max_std_ = 0.03;    // 30 mm σ at max range
    double signal_noise_scale_  = 1.0;     // Poisson shot noise (1 = physical)
    double nearir_noise_scale_  = 1.0;     // Near-IR shot noise
    double dropout_rate_close_  = 0.001;   // 0.1% dropout probability at 0 m
    double dropout_rate_far_    = 0.05;    // 5% dropout probability at max range
    double edge_discon_threshold_ = 0.5;   // Suppress returns at >0.5 m depth jumps
    double base_signal_ = 500.0;           // Baseline signal (photons·m²)
    double base_reflectivity_ = 50.0;      // Default reflectivity [0–255]

    // ── Ouster metadata ──────────────────────────────────────────────────────
    std::string metadata_str_;
    int H_ = 0;                     // pixels_per_column (beam count)
    int W_ = 0;                     // columns_per_frame
    int cpp_ = 0;                   // columns_per_packet
    std::vector<double> beam_alt_angles_;   // per-beam elevation (degrees)
    std::vector<double> beam_az_offsets_;   // per-beam azimuth offset (degrees)
    double beam_origin_mm_ = 0.0;          // lidar_origin_to_beam_origin_mm

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

    // ── Rendering-thread event connection ────────────────────────────────────
    ::gz::sim::EventManager *event_mgr_{nullptr};
    gz::common::ConnectionPtr render_conn_;
    std::chrono::steady_clock::time_point last_render_time_{};
    void OnRender();

    // ── CUDA processor ───────────────────────────────────────────────────────
    std::unique_ptr<CudaRayProcessor> cuda_processor_;

    // ── Channel buffers ──────────────────────────────────────────────────────
    // Allocated as H×W, filled by CUDA, encoded into packets.
    std::vector<uint32_t> range_buf_;
    std::vector<uint16_t> signal_buf_;
    std::vector<uint8_t>  reflectivity_buf_;

    // Raw GpuRays callback buffer
    std::vector<float> depth_buf_;
    std::vector<float> retro_buf_;
    std::mutex frame_mtx_;
    bool frame_ready_ = false;

    // ── ROS 2 node & publishers ──────────────────────────────────────────────
    rclcpp::Node::SharedPtr ros_node_;
    rclcpp::executors::SingleThreadedExecutor ros_executor_;
    std::thread ros_spin_thread_;
    rclcpp::Publisher<ouster_sensor_msgs::msg::PacketMsg>::SharedPtr pkt_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr meta_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr range_image_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr signal_image_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr reflec_image_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr nearir_image_pub_;
    std::string image_frame_id_;
    bool metadata_published_ = false;   // true once a subscriber has acked

    // ── Drain thread ─────────────────────────────────────────────────────────
    std::vector<ouster_sensor_msgs::msg::PacketMsg> drain_pkts_;
    std::thread drain_thread_;
    std::mutex drain_mtx_;
    std::condition_variable drain_cv_;
    std::atomic<bool> drain_ready_{false};
    std::atomic<bool> shutdown_{false};

    // ── Private methods ──────────────────────────────────────────────────────
    void loadMetadata();
    void initRosInterface();
    void onNewFrame(const float * data, unsigned int width,
                    unsigned int height, unsigned int channels,
                    const std::string & format);
    void encodeAndPublish(int64_t stamp_ns);
    void publishImages(int64_t stamp_ns);
    void drainThreadFunc();
};

}  // namespace gz_gpu_ouster_lidar
