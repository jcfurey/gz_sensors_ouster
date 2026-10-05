// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#include "imu_sampler.hpp"

#include "backend.hpp"  // deriveNonDeterministicSeed()

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace gz_gpu_ouster_lidar {

namespace {

// Every ingested state is exactly one contract period after the previous
// one, so a pipeline catch-up of one sample never synthesizes samples: a
// sim-time jump skips (rather than interpolates) the missing deadlines and
// keeps the sample index, and with it the modern measurement IDs, aligned
// with time.
constexpr std::size_t kPassThroughCatchUp = 1;

// Exact at the endpoints: a deadline on a physics tick reproduces that
// state bit for bit (and a non-finite previous state cannot leak into it).
Vec3 lerp(const Vec3 & a, const Vec3 & b, double t)
{
    if (t >= 1.0) return b;
    if (t <= 0.0) return a;
    return {a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t};
}

}  // namespace

OusterImuSampler::OusterImuSampler(std::chrono::nanoseconds sample_period)
    : sample_period_(sample_period)
{
    if (sample_period_ <= std::chrono::nanoseconds::zero()) {
        throw std::invalid_argument("IMU sample period must be positive");
    }
}

OusterImuSampler::OusterImuSampler(
    const ouster_sim_core::OusterMetadata & metadata,
    std::chrono::nanoseconds lidar_frame_period,
    uint64_t sensor_stream_id)
{
    pipeline_.emplace(metadata, lidar_frame_period, sensor_stream_id, epoch_,
                      kPassThroughCatchUp);
    sample_period_ = pipeline_->contract().sample_period;
}

OusterImuSampler::~OusterImuSampler() = default;

const ouster_sim_core::OusterImuPacketContract &
OusterImuSampler::contract() const
{
    if (!pipeline_) {
        throw std::logic_error("IMU packets are not enabled");
    }
    return pipeline_->contract();
}

double OusterImuSampler::sampleRateHz() const
{
    return 1.0e9 / static_cast<double>(sample_period_.count());
}

uint32_t OusterImuSampler::samplesPerFrame() const
{
    if (!pipeline_) return 0;
    const auto & c = pipeline_->contract();
    return static_cast<uint32_t>(c.measurements_per_packet) *
        c.packets_per_frame;
}

void OusterImuSampler::seed(uint64_t value)
{
    rng_.seed(value);
    rng_seeded_ = true;
}

void OusterImuSampler::resetEpoch()
{
    ++epoch_;
    if (pipeline_) {
        pipeline_->reset(epoch_);
    }
}

void OusterImuSampler::step(
    std::chrono::nanoseconds now,
    const Vec3 & angular_velocity,
    const Vec3 & proper_acceleration,
    const ImuNoiseDensities & noise,
    ImuStepResult & out)
{
    out.clear();
    const bool started = state_valid_;
    const auto batch = scheduler_.advance(now, sample_period_);
    if (batch.reset) {
        // A rewind starts a new stochastic sensor epoch and a new packet
        // time domain (the pipeline rejects backwards time). Carrying a bias
        // random walk backwards through time would make bag concatenation
        // and repeatable reset tests physically inconsistent.
        out.reset = true;
        gyro_bias_ = {};
        accel_bias_ = {};
        state_valid_ = false;
        if (started) {
            resetEpoch();
        }
    }
    out.skipped = batch.skipped;
    if (!rng_seeded_) {
        seed(deriveNonDeterministicSeed(this));
    }

    const Vec3 & previous_av = state_valid_ ? previous_av_ : angular_velocity;
    const Vec3 & previous_la = state_valid_ ? previous_la_ : proper_acceleration;
    const auto previous_time = state_valid_ ? previous_time_ : now;
    const int64_t span_ns = (now - previous_time).count();
    const double sample_dt = static_cast<double>(sample_period_.count()) / 1.0e9;

    for (size_t i = 0; i < batch.size; ++i) {
        const auto deadline = batch.deadlines[i];
        const double alpha = (span_ns > 0)
            ? std::clamp(
                  static_cast<double>((deadline - previous_time).count()) /
                      static_cast<double>(span_ns),
                  0.0, 1.0)
            : 1.0;
        const ImuNoiseSample noisy = applyImuNoise(
            lerp(previous_av, angular_velocity, alpha),
            lerp(previous_la, proper_acceleration, alpha),
            gyro_bias_, accel_bias_,
            noise.gyro_noise_std, noise.accel_noise_std,
            noise.gyro_bias_walk, noise.accel_bias_walk,
            sample_dt, rng_);

        ImuSample sample;
        sample.stamp_ns = deadline.count();
        sample.angular_velocity = noisy.av;
        sample.linear_acceleration = noisy.la;
        sample.gyro_white_std = noisy.gyro_white_std;
        sample.accel_white_std = noisy.accel_white_std;
        out.samples.push_back(sample);

        if (!pipeline_) continue;
        ouster_sim_core::OusterImuState state;
        state.timestamp_ns = sample.stamp_ns;
        state.linear_acceleration_mps2 = {
            noisy.la.x, noisy.la.y, noisy.la.z};
        state.angular_velocity_rad_s = {noisy.av.x, noisy.av.y, noisy.av.z};
        try {
            auto ready = pipeline_->ingest(state);
            for (auto & packet : ready) {
                out.packets.push_back(std::move(packet));
            }
        } catch (const std::exception & e) {
            // Non-finite or unrepresentable kinematics. A failed encode can
            // leave the pipeline mid-packet, so restart the packet stream in
            // a fresh epoch; the sensor_msgs/Imu sample is still reported.
            out.packet_error = e.what();
            resetEpoch();
        }
    }

    previous_time_ = now;
    previous_av_ = angular_velocity;
    previous_la_ = proper_acceleration;
    state_valid_ = true;
}

}  // namespace gz_gpu_ouster_lidar
