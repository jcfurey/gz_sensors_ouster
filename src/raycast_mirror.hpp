// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0
//
// The raycast ray mode: mirrors the world's visual geometry out of the ECM
// into an immutable rc::Scene (rebuilt on spawn/despawn, shared with the
// worker via shared_ptr snapshots so a rebuild never races an in-flight
// cast), refreshes per-instance transforms every scan, and runs a worker
// thread that casts the scan on the active backend and hands the
// depth+retro frame through the FrameExchange.

#pragma once

#include "frame_exchange.hpp"
#include "lidar_common.hpp"
#include "raycast_scene.hpp"

#include <gz/math/Pose3.hh>
#include <gz/math/Quaternion.hh>
#include <gz/sim/Entity.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/Util.hh>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace gz_gpu_ouster_lidar {

class RayProcessor;

class RaycastMirror {
public:
    explicit RaycastMirror(std::string sensor_name);
    ~RaycastMirror();  // calls stop()

    struct Params {
        int H = 0;
        int W = 0;
        double max_range = 120.0;
        double lidar_hz = 10.0;
        double beam_origin_mm = 0.0;
        /// Rolling-shutter motion distortion: cast each column from the
        /// sensor pose at its acquisition time (interpolated from the
        /// per-tick pose history) instead of one snapshot pose.
        bool motion_distortion = false;
        const std::vector<float> * beam_alt_f = nullptr;
        const std::vector<float> * beam_az_f = nullptr;
    };

    /// Start the cast worker. `proc_mtx` serialises backend access against
    /// the sim thread's processDepth (shared device buffers/stream); the
    /// pointed-to objects must outlive stop().
    void start(const Params & p, RayProcessor * proc, std::mutex * proc_mtx,
               FrameExchange * exch);

    /// Sim thread, every PostUpdate: rebuild the mirror on visual-population
    /// changes, and at the scan cadence post a cast job (per-instance
    /// transforms + sensor pose) to the worker.
    void postUpdate(const ::gz::sim::UpdateInfo & info,
                    const ::gz::sim::EntityComponentManager & ecm,
                    const ::gz::math::Pose3d & sensor_pose);

    /// Stop and join the worker. Idempotent.
    void stop();

private:
    void rebuildScene(const ::gz::sim::EntityComponentManager & ecm,
                      size_t visual_count);
    void threadFunc();

    std::string sensor_name_;
    Params params_;
    RayProcessor * proc_ = nullptr;
    std::mutex * proc_mtx_ = nullptr;
    FrameExchange * exch_ = nullptr;

    // Immutable scene geometry, shared with the worker via shared_ptr so a
    // rebuild (entity spawned/removed) never races an in-flight cast.
    // scene_version_ lets the GPU backends cache the uploaded geometry.
    std::shared_ptr<const rc::Scene> scene_;
    uint64_t scene_version_ = 0;
    struct Ref {
        ::gz::sim::Entity entity{::gz::sim::kNullEntity};
        // Extra local rotation folded into the entity pose (plane normal).
        ::gz::math::Quaterniond offset{1, 0, 0, 0};
    };
    std::vector<Ref> refs_;          // parallel to scene_->instances
    size_t visual_count_ = 0;        // rebuild trigger
    std::chrono::nanoseconds last_scan_time_{-1};

    // Sensor pose history (sim time, sensor→world pose), recorded every
    // PostUpdate tick; spans at least one scan period so per-column poses
    // can be interpolated for motion distortion.
    std::deque<std::pair<std::chrono::nanoseconds, ::gz::math::Pose3d>>
        pose_history_;
    /// Build per-column pose tables (9W + 3W floats) for the scan ending at
    /// `scan_end`: column m's pose is SLERP/lerp-interpolated from the
    /// history at t_m = scan_end − T + (m+1)·T/W.
    void buildColumnPoses(std::chrono::nanoseconds scan_end,
                          std::vector<float> & col_r,
                          std::vector<float> & col_t) const;

    // Worker thread + job slot.
    std::thread thread_;
    std::mutex mtx_;
    std::condition_variable cv_;
    std::atomic<bool> shutdown_{false};
    bool job_ready_ = false;
    std::shared_ptr<const rc::Scene> job_scene_;
    uint64_t job_scene_version_ = 0;
    std::vector<rc::InstanceXform> job_xforms_;
    float job_sensor_r_[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    float job_sensor_t_[3] = {0, 0, 0};
    std::vector<float> job_col_r_;   // per-column poses (motion distortion;
    std::vector<float> job_col_t_;   // empty = static snapshot pose)
    // Sun illumination for the NEAR_IR ambient model: world-frame
    // propagation direction + diffuse/ambient weights (no directional light
    // in the world → ambient-only, nir = albedo).
    float job_sun_[5] = {0.0f, 0.0f, -1.0f, 0.0f, 1.0f};
    std::vector<float> out_;         // worker-local depth+retro

    // Standalone clock for throttled logs (no ROS node dependency).
    rclcpp::Clock throttle_clock_;
};

}  // namespace gz_gpu_ouster_lidar
