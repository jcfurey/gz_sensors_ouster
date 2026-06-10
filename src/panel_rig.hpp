// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0
//
// The panels ray mode: layout construction from the beam intrinsics, the
// per-panel DepthCamera lifecycle on the render thread, and per-scan frame
// assembly into the FrameExchange. Thread contract: buildLayout() runs in
// Configure (sim thread, before anything renders); ensureCreated(),
// renderScan() and destroy() run under the plugin's render-busy barrier
// (OnRender / dtor), which also serialises the nested NewDepthFrame
// callbacks fired synchronously from PostRender().

#pragma once

#include "frame_exchange.hpp"
#include "lidar_common.hpp"
#include "panel_layout.hpp"

#include <gz/common/Event.hh>
#include <gz/math/Pose3.hh>
#include <gz/math/Quaternion.hh>
#include <gz/rendering/DepthCamera.hh>

#include <string>
#include <vector>

namespace gz_gpu_ouster_lidar {

class OusterMetadata;

class PanelRig {
public:
    explicit PanelRig(std::string sensor_name);

    /// Build + verify the panel layout from the beam geometry. Logs the
    /// rig summary or the unsupported-geometry error; false disables the
    /// plugin.
    bool buildLayout(const OusterMetadata & meta, double oversample,
                     const std::string & sampling, double max_range);

    bool layoutValid() const { return layout_.n_panels > 0; }
    const ResampleParams & resampleParams() const { return layout_.rp; }
    bool hemispherical() const { return layout_.hemispherical; }

    bool created() const { return !cams_.empty(); }

    /// Render thread: create the depth cameras once the ogre2 scene exists.
    /// Returns true when the rig is live after the call. No-ops (returning
    /// false) until the Sensors system has built the scene.
    bool ensureCreated(double max_range, uint32_t visibility_mask);

    /// Render thread: render every panel at `pose` and assemble the packed
    /// frame into `exch` (drops are surfaced with throttled warnings).
    void renderScan(const ::gz::math::Pose3d & pose, FrameExchange & exch);

    /// Destroy the cameras (idempotent). Caller guarantees no concurrent
    /// render-thread activity (the plugin dtor flushes its barrier first).
    void destroy();

private:
    void onPanelFrame(size_t panel, const float * data, unsigned int width,
                      unsigned int height, unsigned int channels);

    std::string sensor_name_;
    PanelLayout layout_;

    std::vector<::gz::rendering::DepthCameraPtr> cams_;
    std::vector<gz::common::ConnectionPtr> conns_;
    std::vector<::gz::math::Quaterniond> quats_;  // sensor→panel rotation
    // Render-thread-only: which panels delivered a frame this tick.
    std::vector<bool> filled_;
    std::vector<float> pending_buf_;

    // Standalone clock for throttled logs (no ROS node dependency).
    rclcpp::Clock throttle_clock_;
};

}  // namespace gz_gpu_ouster_lidar
