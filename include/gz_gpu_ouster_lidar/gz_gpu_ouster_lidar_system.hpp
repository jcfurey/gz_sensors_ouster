// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "imu_noise.hpp"  // Vec3 (used for IMU bias state members)

#include <gz/sim/System.hh>
#include <gz/common/Event.hh>
#include <gz/math/Pose3.hh>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <vector>

namespace gz_gpu_ouster_lidar {

// Plugin-internal components (src/*.hpp, not installed). The plugin holds
// them through unique_ptr + the out-of-line destructor, so this public
// header needs only forward declarations.
class FrameExchange;
class OusterMetadata;
class PacketEncoder;
class PanelRig;
class RayProcessor;
class RaycastMirror;
class RosInterface;

/// Gazebo Harmonic system plugin that models the Ouster beam geometry
/// directly — a rig of perspective depth panels (cylindrical sectors for
/// OS0/1/2, plus a zenith cap for OSDome), or exact per-beam raycasting
/// against an ECM scene mirror — and encodes the scan into Ouster
/// PacketMsg on the active compute backend (CUDA/HIP/SYCL/CPU).
///
/// This class is a thin orchestrator: SDF parsing/validation, the gz-sim
/// callbacks, entity tracking and teardown ordering. The work lives in
/// focused components (src/):
///   OusterMetadata — metadata loading + PacketWriter ownership
///   PanelRig       — panel layout + depth cameras + frame assembly
///   RaycastMirror  — ECM scene mirror + cast worker thread
///   PacketEncoder  — packet building + paced drain thread
///   RosInterface   — node/publishers/parameters + noise-param store
///   FrameExchange  — producer/consumer scan-frame triple buffer
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
    // ── SDF configuration ────────────────────────────────────────────────────
    std::string metadata_path_;
    std::string sensor_name_;
    std::string world_name_;
    double lidar_hz_ = 10.0;
    uint32_t visibility_mask_ = 0xFFFFFFFFu;
    std::string ray_mode_ = "panels";        // "panels" | "raycast"
    double panel_oversample_ = 2.0;          // clamped 1..4
    std::string panel_sampling_ = "bilinear";  // "bilinear" | "nearest"
    std::string image_qos_ = "reliable";
    std::string imu_qos_ = "sensor_data";
    double max_range_ = 120.0;               // metres, default OS1
    bool max_range_explicit_ = false;        // set via SDF (don't auto-derive)

    // Noise model SDF defaults (live store sits in RosInterface; these hold
    // the parsed + clamped initial values handed over at init).
    double range_noise_min_std_ = 0.003;
    double range_noise_max_std_ = 0.015;
    double signal_noise_scale_ = 1.0;
    double nearir_noise_scale_ = 1.0;
    double dropout_rate_close_ = 0.0005;
    double dropout_rate_far_ = 0.03;
    double false_alarm_rate_ = 0.0;
    double edge_discon_threshold_ = 0.15;
    double base_signal_ = 800.0;
    double base_reflectivity_ = 50.0;

    // ── IMU configuration (SDF-optional) ─────────────────────────────────────
    std::string imu_name_;
    double imu_hz_ = 100.0;
    bool imu_enabled_ = false;
    bool publish_imu_msg_ = true;
    double gyro_noise_std_ = 1.75e-4;   // rad/s/√Hz
    double accel_noise_std_ = 2.3e-3;   // m/s²/√Hz
    double gyro_bias_walk_ = 1.0e-6;    // rad/s²/√Hz
    double accel_bias_walk_ = 1.0e-5;   // m/s³/√Hz

    // IMU bias state (random walk integrand). Persists across frames.
    Vec3 gyro_bias_{0.0, 0.0, 0.0};
    Vec3 accel_bias_{0.0, 0.0, 0.0};
    // Per-instance RNG for IMU noise; lazy-seeded so multiple sensors get
    // independent streams.
    std::mt19937_64 imu_rng_;
    bool imu_rng_seeded_ = false;
    std::vector<uint8_t> imu_pkt_buf_;
    std::chrono::nanoseconds last_imu_sim_time_{0};

    // ── Components ───────────────────────────────────────────────────────────
    std::unique_ptr<OusterMetadata> meta_;
    std::unique_ptr<RosInterface> ros_;
    std::unique_ptr<PanelRig> rig_;
    std::unique_ptr<RaycastMirror> mirror_;
    std::unique_ptr<PacketEncoder> encoder_;
    std::unique_ptr<FrameExchange> exchange_;

    // ── Ray processor (vendor-neutral; CUDA/HIP/SYCL/CPU at runtime) ────────
    // processor_mtx_ serialises backend calls: in raycast mode the mirror's
    // worker (castScan) and the sim thread (processDepth) would otherwise
    // race on the backend's shared device buffers and stream.
    std::mutex processor_mtx_;
    std::unique_ptr<RayProcessor> ray_processor_;

    // ── Channel buffers (filled by the backend, consumed by the encoder) ────
    std::vector<uint32_t> range_buf_;
    std::vector<uint16_t> signal_buf_;
    std::vector<uint8_t> reflectivity_buf_;
    std::vector<uint16_t> nearir_buf_;
    std::vector<float> process_buf_;   // sim-thread scratch from exchange_
    bool memory_logged_ = false;       // true after first GPU-buffer report

    // ── Entity tracking ──────────────────────────────────────────────────────
    ::gz::sim::Entity sensor_entity_{::gz::sim::kNullEntity};
    std::string lidar_frame_name_;
    ::gz::sim::Entity lidar_frame_entity_{::gz::sim::kNullEntity};
    bool lidar_frame_found_ = false;
    std::mutex pose_mtx_;
    ::gz::math::Pose3d cached_pose_;
    ::gz::sim::Entity imu_entity_{::gz::sim::kNullEntity};
    bool imu_entity_found_ = false;
    std::string image_frame_id_;
    std::string imu_frame_id_;

    // ── Rendering-thread event ───────────────────────────────────────────────
    // Non-owning. The EventManager is owned by the gz-sim server and outlives
    // this plugin; we only borrow it (in Configure) to register the render
    // callback. Do not delete.
    ::gz::sim::EventManager * event_mgr_{nullptr};
    gz::common::ConnectionPtr render_conn_;
    std::chrono::steady_clock::time_point last_render_time_{};
    std::atomic<bool> sensor_initialized_{false};
    // Counts OnRender() entries; stays 0 when the Sensors system never
    // starts rendering (no rendering sensor in the world); PostUpdate()
    // uses it to emit a one-shot diagnostic.
    std::atomic<uint64_t> onrender_entries_{0};
    bool no_render_warned_{false};
    bool was_paused_ = false;

    std::atomic<bool> shutdown_{false};

    // ── Render-thread shutdown barrier ───────────────────────────────────────
    // gz::common::Connection::reset() does NOT wait for in-flight callbacks:
    // EventT::Disconnect() flips an atomic flag that prevents future Signal
    // dispatches but immediately destroys the held std::function while a
    // concurrent Signal call may still be inside the callback (UB by the
    // letter of the standard, racy in practice). Without an explicit
    // barrier, the dtor can race the render thread, freeing components out
    // from under an in-flight OnRender. OnRender locks this for the
    // duration of its work, and the dtor takes it once after disconnect()
    // to flush any in-flight callback before tearing down the rest.
    //
    // MUST be recursive: gz-rendering signals the NewDepthFrame event
    // synchronously from inside Ogre2DepthCamera::PostRender(), so the
    // rig's panel callbacks run nested on the SAME render thread, inside
    // OnRender() which already holds this lock. The dtor still takes the
    // lock from a different thread, so it blocks until the render thread
    // has fully unwound — the barrier guarantee is preserved.
    mutable std::recursive_mutex render_busy_mtx_;

    // ── Private methods ──────────────────────────────────────────────────────
    void OnRender();
    void encodeAndPublish(int64_t stamp_ns, const float * raw_data, int raw_n);
    void publishImu(const ::gz::sim::UpdateInfo & info,
                    const ::gz::sim::EntityComponentManager & ecm);
};

}  // namespace gz_gpu_ouster_lidar
