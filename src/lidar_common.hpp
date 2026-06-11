// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0
//
// Shared constants + logger for the plugin's internal components. These
// headers live in src/ (not installed): they are implementation detail of
// the plugin .so, included by sibling .cpp files only.

#pragma once

#include <rclcpp/rclcpp.hpp>

namespace gz_gpu_ouster_lidar {

/// One shared logger so every component logs under the same name the
/// plugin always used.
inline const rclcpp::Logger & lidarLogger()
{
    static const rclcpp::Logger logger =
        rclcpp::get_logger("gz_gpu_ouster_lidar");
    return logger;
}

// Small angular pad added to the beam altitude range so the panel rig's
// vertical coverage extends a touch beyond the outermost beams; keeps
// bilinear corners of the edge beams inside rendered pixels.
constexpr double kBeamMarginDeg = 1.0;

// Near clip plane for the panel depth cameras and the raycast mode
// (metres). Matches the real sensor's minimum range region where returns
// are unreliable anyway.
constexpr double kNearClip = 0.3;

/// Dynamically reconfigurable noise-model parameters. SDF supplies the
/// initial values (parsed by the plugin); the live store sits in
/// RosInterface, written by the ROS parameter callback and snapshotted by
/// the encode and IMU paths.
struct NoiseParams {
    double range_noise_min_std = 0.003;
    double range_noise_max_std = 0.015;
    double signal_noise_scale = 1.0;
    double nearir_noise_scale = 1.0;
    double dropout_rate_close = 0.0005;
    double dropout_rate_far = 0.03;
    double false_alarm_rate = 0.0;
    double edge_discon_threshold = 0.15;
    double base_signal = 800.0;
    double base_reflectivity = 50.0;
    double gyro_noise_std = 1.75e-4;
    double accel_noise_std = 2.3e-3;
    double gyro_bias_walk = 1.0e-6;
    double accel_bias_walk = 1.0e-5;
};

}  // namespace gz_gpu_ouster_lidar
