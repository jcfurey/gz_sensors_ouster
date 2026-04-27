// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0
//
// IMU noise + bias model. Decoupled from Gazebo (operates on plain Vec3
// triples) so it can be unit-tested without spinning a sim.
//
// Model:
//   white  = density / sqrt(dt)         per-sample gyro/accel noise sigma
//   drift  = walk    * sqrt(dt)         per-sample bias-walk increment sigma
//   bias_k+1 = bias_k + drift * N(0,1)  random-walk integration
//   meas   = nominal + bias + white * N(0,1)
//
// All four density/walk inputs are continuous-time (per-√Hz), matching
// what IMU datasheets quote, and scaled internally for the configured dt.

#pragma once

#include <random>

namespace gz_gpu_ouster_lidar {

/// Plain double triple. Avoids depending on gz::math from this header so
/// the noise model can be tested in isolation.
struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;

    Vec3 operator+(const Vec3 & o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 & operator+=(const Vec3 & o) { x += o.x; y += o.y; z += o.z; return *this; }
};

struct ImuNoiseSample {
    Vec3 av;                 ///< measured angular velocity (with bias + noise)
    Vec3 la;                 ///< measured linear acceleration (with bias + noise)
    double gyro_white_std;   ///< per-sample gyro sigma — for covariance reporting
    double accel_white_std;  ///< per-sample accel sigma — for covariance reporting
};

/// Apply white Gaussian noise + integrate bias random walk for one IMU sample.
///
/// @param nom_av           nominal (noiseless) angular velocity (rad/s, body frame)
/// @param nom_la           nominal proper linear acceleration (m/s², body frame)
/// @param gyro_bias        gyro bias state — mutated in place each call
/// @param accel_bias       accel bias state — mutated in place each call
/// @param gyro_noise_std   continuous-time gyro density (rad/s/√Hz)
/// @param accel_noise_std  continuous-time accel density (m/s²/√Hz)
/// @param gyro_bias_walk   gyro bias random walk (rad/s²/√Hz)
/// @param accel_bias_walk  accel bias random walk (m/s³/√Hz)
/// @param dt               sample interval (s) — typically 1/imu_hz
/// @param rng              mutated in place; pass mt19937_64 for full entropy
/// @return measured av/la and the per-sample white-noise std (for covariance)
ImuNoiseSample applyImuNoise(
    const Vec3 & nom_av,
    const Vec3 & nom_la,
    Vec3 & gyro_bias,
    Vec3 & accel_bias,
    double gyro_noise_std,
    double accel_noise_std,
    double gyro_bias_walk,
    double accel_bias_walk,
    double dt,
    std::mt19937_64 & rng);

}  // namespace gz_gpu_ouster_lidar
