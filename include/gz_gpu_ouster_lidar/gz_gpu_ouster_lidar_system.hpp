// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gz_gpu_ouster_lidar/ray_processor.hpp"
#include "imu_noise.hpp"     // Vec3 (used for IMU bias state members)
#include "panel_layout.hpp"  // PanelLayout (cylindrical/hemispherical rig)
#include "raycast_scene.hpp"  // rc::Scene (full per-beam raycast mode)

#include <gz/sim/System.hh>
#include <gz/common/Event.hh>
#include <gz/math/Pose3.hh>
#include <gz/math/Quaternion.hh>
#include <gz/rendering/DepthCamera.hh>

#include <rclcpp/rclcpp.hpp>
#include <ouster_sensor_msgs/msg/packet_msg.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/string.hpp>

#include <chrono>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <atomic>
#include <condition_variable>
#include <cstdint>

namespace ouster::sdk::core {
    class SensorInfo;
    namespace impl { class PacketWriter; }
}

namespace gz_gpu_ouster_lidar {

class RayProcessor;

/// Gazebo Harmonic system plugin that models the Ouster beam geometry
/// directly: a rig of perspective depth panels (cylindrical sectors for
/// OS0/1/2, plus a zenith cap for OSDome) is rendered each scan and every
/// calibrated beam is resampled from the covering panel — no GpuRays
/// cubemap. The resampled scan is encoded into Ouster PacketMsg on the GPU.
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

    // QoS overrides for image/camera_info and IMU pubs. Accepted values:
    //   "reliable"    — RELIABLE + KEEP_LAST(depth)
    //   "best_effort" — BEST_EFFORT + KEEP_LAST(depth)
    //   "sensor_data" — rclcpp::SensorDataQoS preset
    // lidar_packets always uses SensorDataQoS (high-rate, os_cloud expects it).
    std::string image_qos_ = "reliable";    // matches RViz / rqt_image_view default
    std::string imu_qos_   = "sensor_data"; // matches ouster_ros driver convention

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

    // ── IMU noise model (SDF-configurable, dynamically reconfigurable) ───────
    // Defaults match Ouster Os1 IMU datasheet (ICM-20948 class).
    // Continuous-time noise densities; per-sample std is density / sqrt(dt).
    // Bias random walks; per-sample drift std is walk * sqrt(dt).
    double gyro_noise_std_   = 1.75e-4;  // rad/s/√Hz   (≈0.01 °/s/√Hz)
    double accel_noise_std_  = 2.3e-3;   // m/s²/√Hz    (≈230 µg/√Hz)
    double gyro_bias_walk_   = 1.0e-6;   // rad/s²/√Hz  (Ouster bias instability)
    double accel_bias_walk_  = 1.0e-5;   // m/s³/√Hz    (Ouster bias instability)

    // Bias state (random walk integrand). Persists across frames.
    // Plain double-triple (defined in cuda/imu_noise.hpp) so the noise
    // model can be unit-tested without dragging gz/math along.
    Vec3 gyro_bias_{0.0, 0.0, 0.0};
    Vec3 accel_bias_{0.0, 0.0, 0.0};

    // Per-instance RNG for IMU noise. Lazy-seeded on first publish using
    // deriveNonDeterministicSeed(this) so multiple sensors get independent
    // noise streams. mt19937_64 (not mt19937) so the full 64-bit seed is
    // preserved.
    std::mt19937_64 imu_rng_;
    bool imu_rng_seeded_ = false;

    // ── Ouster metadata ──────────────────────────────────────────────────────
    std::string metadata_str_;
    int H_ = 0;                     // pixels_per_column (beam count)
    int W_ = 0;                     // columns_per_frame
    int cpp_ = 0;                   // columns_per_packet
    std::vector<double> beam_alt_angles_;   // per-beam elevation (degrees)
    std::vector<double> beam_az_offsets_;   // per-beam azimuth offset (degrees)
    double beam_origin_mm_ = 0.0;          // lidar_origin_to_beam_origin_mm

    // Cached beam-altitude bounds (populated in loadMetadata).
    // Values include kBeamMarginDeg padding so the resample window spans a
    // little beyond the outermost beams.
    double min_alt_ = 0.0;
    double max_alt_ = 0.0;
    double v_range_ = 0.0;

    // ── IMU packet format ────────────────────────────────────────────────────
    size_t imu_packet_size_ = 0;
    std::vector<uint8_t> imu_pkt_buf_;

    // ── Packet writer ────────────────────────────────────────────────────────
    std::unique_ptr<ouster::sdk::core::impl::PacketWriter> pw_;
    std::vector<uint8_t> pkt_buf_;
    uint32_t frame_id_ = 0;

    // ── Panel rig (cylindrical / hemispherical depth cameras) ───────────────
    ::gz::sim::Entity sensor_entity_{::gz::sim::kNullEntity};
    std::atomic<bool> sensor_initialized_{false};

    // Layout computed from beam metadata in Configure (panel orientations,
    // intrinsics, packed-buffer offsets). layout_.rp is the ResampleParams
    // handed to the backends each frame.
    PanelLayout layout_;
    double panel_oversample_ = 2.0;  // SDF <panel_oversample>, clamped 1..4
    // SDF <panel_sampling>: "bilinear" (smooth surfaces, default) or
    // "nearest" (raycast-like: one exact rendered ray per beam, direction
    // quantised to the pixel grid, no range blending at depth edges).
    std::string panel_sampling_ = "bilinear";

    // Rendering objects (created lazily on the render thread in OnRender).
    std::vector<::gz::rendering::DepthCameraPtr> panel_cams_;
    std::vector<gz::common::ConnectionPtr> panel_conns_;
    std::vector<::gz::math::Quaterniond> panel_quats_;  // sensor→panel rotation
    // Render-thread-only: which panels delivered a frame this tick.
    std::vector<bool> panel_filled_;

    // ── Full raycast mode (SDF <ray_mode>raycast</ray_mode>) ────────────────
    // Casts every beam exactly (calibrated direction, true beam-origin
    // parallax) against a scene mirrored from the ECM. No rendering is
    // involved: no panel cameras, no anchor-sensor requirement, and the
    // scan is throttled on sim time. laser_retro is restored as the
    // reflectivity source (visuals carry components::LaserRetro).
    std::string ray_mode_ = "panels";   // "panels" | "raycast"

    // Immutable scene geometry, shared with the worker via shared_ptr so a
    // rebuild (entity spawned/removed) never races an in-flight cast.
    // rc_scene_version_ lets the GPU backends cache the uploaded geometry.
    std::shared_ptr<const rc::Scene> rc_scene_;
    uint64_t rc_scene_version_ = 0;
    struct RaycastRef {
        ::gz::sim::Entity entity{::gz::sim::kNullEntity};
        // Extra local rotation folded into the entity pose (plane normal).
        ::gz::math::Quaterniond offset{1, 0, 0, 0};
    };
    std::vector<RaycastRef> rc_refs_;       // parallel to rc_scene_->instances
    size_t rc_visual_count_ = 0;            // rebuild trigger
    std::chrono::nanoseconds rc_last_scan_time_{-1};

    // Worker thread: PostUpdate posts a job (scene snapshot + per-instance
    // transforms + sensor pose); the worker casts the scan and hands the
    // H×W depth + H×W retro buffer through the frame triple-buffer.
    std::thread rc_thread_;
    std::mutex rc_mtx_;
    std::condition_variable rc_cv_;
    bool rc_job_ready_ = false;
    std::shared_ptr<const rc::Scene> rc_job_scene_;
    uint64_t rc_job_scene_version_ = 0;
    std::vector<rc::InstanceXform> rc_job_xforms_;
    float rc_job_sensor_r_[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    float rc_job_sensor_t_[3] = {0, 0, 0};
    std::vector<float> rc_out_;             // worker-local depth+retro

    void raycastThreadFunc();
    void rebuildRaycastScene(const ::gz::sim::EntityComponentManager & ecm,
                             size_t visual_count);
    void raycastPostUpdate(const ::gz::sim::UpdateInfo & info,
                           const ::gz::sim::EntityComponentManager & ecm);

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
    // Non-owning. The EventManager is owned by the gz-sim server and outlives
    // this plugin; we only borrow it (in Configure) to register the render
    // callback. Do not delete.
    ::gz::sim::EventManager *event_mgr_{nullptr};
    gz::common::ConnectionPtr render_conn_;
    std::chrono::steady_clock::time_point last_render_time_{};
    void OnRender();
    void DestroyPanels();

    // ── Ray processor (vendor-neutral; CUDA/HIP/SYCL/CPU at runtime) ────────
    // processor_mtx_ serialises backend calls: in raycast mode the worker
    // thread (castScan) and the sim thread (processDepth in
    // encodeAndPublish) would otherwise race on the backend's shared
    // device buffers and stream.
    std::mutex processor_mtx_;
    std::unique_ptr<RayProcessor> ray_processor_;

    // ── Channel buffers ──────────────────────────────────────────────────────
    // Allocated as H×W, filled by CUDA, encoded into packets.
    std::vector<uint32_t> range_buf_;
    std::vector<uint16_t> signal_buf_;
    std::vector<uint8_t>  reflectivity_buf_;
    std::vector<uint16_t> nearir_buf_;          // NEAR_IR channel for packet encoding

    // Raw panel-rig frame triple-buffer:
    //   pending_buf_  — render-thread scratch; each panel's depth callback
    //                   memcpys into its packed-offset slot OUTSIDE the lock.
    //   raw_frame_buf_ — handoff slot; under frame_mtx_, render swaps
    //                    pending_buf_ in, sim swaps it out.
    //   process_buf_  — sim-thread scratch; encodeAndPublish reads from here
    //                   AFTER releasing the lock.
    // After warmup, all three vectors hold capacity matching the packed rig
    // size so the per-frame memcpy doesn't reallocate. The render thread
    // never holds frame_mtx_ during the panel memcpys.
    std::vector<float> pending_buf_;
    std::vector<float> raw_frame_buf_;
    std::vector<float> process_buf_;
    int raw_frame_n_ = 0;               // floats in raw_frame_buf_ (== rp.raw_n)
    std::vector<float> beam_alt_f_;     // beam_alt_angles_ as float (for GPU upload)
    std::vector<float> beam_az_f_;      // beam_az_offsets_ as float
    std::mutex frame_mtx_;
    std::atomic<bool> frame_ready_{false};
    // Cumulative count of rig frames that arrived while a previous frame
    // was still pending in raw_frame_buf_ — i.e. PostUpdate didn't drain
    // in time and the older frame was overwritten unobserved.
    std::atomic<uint64_t> dropped_frames_{0};

    // Counts OnRender() entries, i.e. how many times events::Render has fired.
    // Stays 0 when the Sensors system never starts rendering (no rendering
    // sensor in the world); PostUpdate() uses it to emit a one-shot diagnostic.
    std::atomic<uint64_t> onrender_entries_{0};
    // Set once the above diagnostic has been logged, so it fires only once.
    bool no_render_warned_{false};

    // ── ROS 2 node & publishers ──────────────────────────────────────────────
    // publish_mtx_ serialises all publish() calls across threads (render,
    // simulation/PostUpdate, drain).  rmw_zenoh_cpp is not guaranteed
    // thread-safe for concurrent publishes on the same node.
    std::mutex publish_mtx_;
    rclcpp::Node::SharedPtr ros_node_;
    // Lazy-construct the executor inside Configure() *after* rclcpp::init().
    // If we declare it as a value member here, its constructor runs at plugin
    // instantiation time (gz::plugin::Loader::Instantiate) — before any
    // rclcpp::init() has been called by anyone — and crashes with
    // "failed to create guard condition: context argument is null".
    // This made the plugin silently dependent on gz_ros2_control loading
    // first and pulling rclcpp into existence. unique_ptr defers the
    // executor's construction until we own a context.
    std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> ros_executor_;
    std::thread ros_spin_thread_;
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
    std::string image_frame_id_;
    std::string imu_frame_id_;
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb_handle_;
    // Touched only on the render thread today (OnRender is serialized by the
    // Gazebo rendering system), but kept atomic so the metadata-republish
    // state stays well-defined if a future render path ever runs concurrently.
    std::atomic<bool> metadata_published_{false};  // true once a subscriber has acked
    std::atomic<int> metadata_pub_count_{0};       // render ticks since sensor init
    bool memory_logged_ = false;       // true after first GPU-buffer report

    // ── Drain thread ─────────────────────────────────────────────────────────
    std::vector<ouster_sensor_msgs::msg::PacketMsg> drain_pkts_;
    std::thread drain_thread_;
    std::mutex drain_mtx_;
    std::condition_variable drain_cv_;
    std::atomic<bool> drain_ready_{false};
    std::atomic<bool> shutdown_{false};

    // ── Render-thread shutdown barrier ───────────────────────────────────────
    // gz::common::Connection::reset() does NOT wait for in-flight callbacks:
    // EventT::Disconnect() flips an atomic flag that prevents future Signal
    // dispatches but immediately destroys the held std::function while a
    // concurrent Signal call may still be inside the callback (UB by the
    // letter of the standard, racy in practice). Without an explicit
    // barrier, the dtor can race the render thread, freeing drain_cv_ /
    // frame_mtx_ / publish_mtx_ out from under an in-flight OnRender or
    // onPanelFrame. Both render-thread entry points lock this for the
    // duration of their work, and the dtor takes it once after disconnect()
    // to flush any in-flight callback before tearing down the rest.
    //
    // MUST be recursive: gz-rendering signals the NewDepthFrame event
    // synchronously from inside Ogre2DepthCamera::PostRender(), so
    // onPanelFrame() runs nested on the SAME render thread, *inside*
    // OnRender() which already holds this lock across its per-panel
    // PostRender() calls. A plain std::mutex self-deadlocks the render
    // thread there (which in turn stalls the gz-sim Sensors render loop and
    // freezes /clock). The dtor still takes the lock from a different
    // thread, so it blocks until the render thread has fully unwound both
    // frames — the barrier guarantee is preserved.
    mutable std::recursive_mutex render_busy_mtx_;

    // ── IMU timing ───────────────────────────────────────────────────────────
    std::chrono::nanoseconds last_imu_sim_time_{0};

    // ── Pause/resume detection ───────────────────────────────────────────────
    bool was_paused_ = false;

    // ── Private methods ──────────────────────────────────────────────────────
    bool loadMetadata();
    void initRosInterface();
    void onPanelFrame(size_t panel, const float * data,
                      unsigned int width, unsigned int height,
                      unsigned int channels);
    void encodeAndPublish(int64_t stamp_ns,
                          const float * raw_data, int raw_n);
    void publishImages(int64_t stamp_ns);
    void publishImu(const ::gz::sim::UpdateInfo & info,
                    const ::gz::sim::EntityComponentManager & ecm);
    void drainThreadFunc();
};

}  // namespace gz_gpu_ouster_lidar
