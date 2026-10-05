// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0
//
// Gazebo-free IMU sampling path: exact sim-time sample deadlines, linear
// interpolation between physics states, the IMU noise/bias model, and the
// native Ouster IMU packet encoding (ouster_sim_core::OusterImuPacketPipeline).
// The plugin feeds it one physics state per PostUpdate and publishes what it
// returns; tests drive it directly and decode the packets with the SDK.

#pragma once

#include "imu_noise.hpp"
#include "gz_gpu_ouster_lidar/sim_time_scheduler.hpp"

#include <ouster_sim_core/imu_packet_pipeline.hpp>
#include <ouster_sim_core/metadata.hpp>

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace gz_gpu_ouster_lidar {

/// Continuous-time IMU noise densities (see applyImuNoise()).
struct ImuNoiseDensities {
    double gyro_noise_std = 0.0;   ///< rad/s/sqrt(Hz)
    double accel_noise_std = 0.0;  ///< m/s^2/sqrt(Hz)
    double gyro_bias_walk = 0.0;   ///< rad/s^2/sqrt(Hz)
    double accel_bias_walk = 0.0;  ///< m/s^3/sqrt(Hz)
};

/// One measured IMU sample in SI units, IMU body frame.
struct ImuSample {
    int64_t stamp_ns = 0;
    Vec3 angular_velocity;     ///< rad/s (with bias + noise)
    Vec3 linear_acceleration;  ///< proper acceleration, m/s^2 (with bias + noise)
    double gyro_white_std = 0.0;   ///< per-sample sigma, for covariance
    double accel_white_std = 0.0;  ///< per-sample sigma, for covariance
};

/// Output of one OusterImuSampler::step(). Reused across steps so the
/// steady state allocates no sample storage.
struct ImuStepResult {
    bool reset = false;      ///< time rewound (or first step): new IMU epoch
    uint64_t skipped = 0;    ///< catch-up deadlines dropped after a time jump
    std::vector<ImuSample> samples;
    std::vector<ouster_sim_core::EncodedOusterImuPacket> packets;
    /// Non-empty when the packet encoder rejected a sample; the packet
    /// stream was restarted in a new epoch and the sample was not packed.
    std::string packet_error;

    void clear()
    {
        reset = false;
        skipped = 0;
        samples.clear();
        packets.clear();
        packet_error.clear();
    }
};

/// Samples simulator IMU kinematics at the Ouster IMU cadence and encodes
/// native IMU packets.
///
/// With packets enabled the sample period is the packet contract's (100 Hz
/// for LEGACY; fps x measurements-per-frame for ACCEL32_GYRO32_NMEA), and
/// every noisy sample is handed to the core pipeline exactly on one of its
/// deadlines, so the pipeline's interpolation passes it through unchanged:
/// the packets carry the same values the sensor_msgs/Imu path publishes.
class OusterImuSampler {
public:
    /// Packet-less sampler at `sample_period` (sensor_msgs/Imu only).
    explicit OusterImuSampler(std::chrono::nanoseconds sample_period);

    /// Sampler that also encodes native Ouster IMU packets. Throws (from the
    /// core pipeline) when the metadata has no usable IMU packet layout or
    /// its cadence does not divide the lidar frame period.
    OusterImuSampler(const ouster_sim_core::OusterMetadata & metadata,
                     std::chrono::nanoseconds lidar_frame_period,
                     uint64_t sensor_stream_id);

    ~OusterImuSampler();
    OusterImuSampler(const OusterImuSampler &) = delete;
    OusterImuSampler & operator=(const OusterImuSampler &) = delete;

    bool packetsEnabled() const { return pipeline_.has_value(); }
    /// Packet contract; only valid when packetsEnabled().
    const ouster_sim_core::OusterImuPacketContract & contract() const;
    std::chrono::nanoseconds samplePeriod() const { return sample_period_; }
    double sampleRateHz() const;
    /// Samples per lidar frame (packets x measurements); 0 without packets.
    uint32_t samplesPerFrame() const;
    uint64_t epoch() const { return epoch_; }

    /// Fix the noise RNG seed (tests). Otherwise the first step seeds it
    /// non-deterministically so concurrent sensors draw independent noise.
    void seed(uint64_t value);

    /// Advance to sim time `now` given the current nominal (noise-free)
    /// angular velocity (rad/s) and proper acceleration (m/s^2), both in the
    /// IMU body frame. Fills `out` with every sample deadline in (previous
    /// step, now] and the packets those samples completed.
    void step(std::chrono::nanoseconds now,
              const Vec3 & angular_velocity,
              const Vec3 & proper_acceleration,
              const ImuNoiseDensities & noise,
              ImuStepResult & out);

private:
    void resetEpoch();

    std::chrono::nanoseconds sample_period_{0};
    std::optional<ouster_sim_core::OusterImuPacketPipeline> pipeline_;
    uint64_t epoch_ = 0;
    PeriodicDeadlineScheduler scheduler_;

    // Interpolation state: the previous physics state.
    bool state_valid_ = false;
    std::chrono::nanoseconds previous_time_{0};
    Vec3 previous_av_;
    Vec3 previous_la_;

    // Noise model state (random-walk integrands persist across steps).
    Vec3 gyro_bias_;
    Vec3 accel_bias_;
    std::mt19937_64 rng_;
    bool rng_seeded_ = false;
};

}  // namespace gz_gpu_ouster_lidar
