// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#include "imu_noise.hpp"

#include <cmath>

namespace gz_gpu_ouster_lidar {

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
    std::mt19937_64 & rng)
{
    const double sqrt_dt = std::sqrt(dt);
    const double gyro_white  = gyro_noise_std  / sqrt_dt;
    const double accel_white = accel_noise_std / sqrt_dt;
    const double gyro_drift  = gyro_bias_walk  * sqrt_dt;
    const double accel_drift = accel_bias_walk * sqrt_dt;

    std::normal_distribution<double> norm{0.0, 1.0};

    // Bias drift: skip the 3 RNG draws when walk is disabled.
    if (gyro_drift > 0.0) {
        gyro_bias.x += gyro_drift * norm(rng);
        gyro_bias.y += gyro_drift * norm(rng);
        gyro_bias.z += gyro_drift * norm(rng);
    }
    if (accel_drift > 0.0) {
        accel_bias.x += accel_drift * norm(rng);
        accel_bias.y += accel_drift * norm(rng);
        accel_bias.z += accel_drift * norm(rng);
    }

    // Nominal + bias, then optionally white noise.
    Vec3 av = nom_av + gyro_bias;
    Vec3 la = nom_la + accel_bias;
    if (gyro_white > 0.0) {
        av.x += gyro_white * norm(rng);
        av.y += gyro_white * norm(rng);
        av.z += gyro_white * norm(rng);
    }
    if (accel_white > 0.0) {
        la.x += accel_white * norm(rng);
        la.y += accel_white * norm(rng);
        la.z += accel_white * norm(rng);
    }

    return {av, la, gyro_white, accel_white};
}

}  // namespace gz_gpu_ouster_lidar
