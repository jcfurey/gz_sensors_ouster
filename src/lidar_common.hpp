// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0
//
// Shared constants + logger for the simulator-neutral Ouster core and the
// Gazebo plugin. This header is exported with gz_sensors_ouster_core so other
// ray producers can use the same packet/publication path.

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
