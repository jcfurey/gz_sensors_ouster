// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0
//
// The plugin's IMU path end to end, without Gazebo: SI kinematics at
// physics ticks -> OusterImuSampler (deadlines, interpolation, noise) ->
// native Ouster IMU packets -> Ouster SDK ImuPacket decode. The decoded
// accel()/gyro()/status()/timestamps must equal the SI inputs for both the
// LEGACY (g and deg/s on the wire) and ACCEL32_GYRO32_NMEA (SI, multiple
// measurements per packet, header + CRC) profiles.

#include "imu_sampler.hpp"

#include <ouster/packet.h>
#include <ouster/types.h>

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#ifndef GZ_TEST_METADATA_DIR
#error "GZ_TEST_METADATA_DIR must be defined at compile time"
#endif

namespace gz_gpu_ouster_lidar {
namespace {

using std::chrono_literals::operator""s;
using std::chrono_literals::operator""ms;
using std::chrono_literals::operator""ns;
using ouster_sim_core::EncodedOusterImuPacket;
using ouster_sim_core::OusterMetadata;

constexpr double kGravity = 9.80665;
constexpr double kPi = 3.14159265358979323846;
// LEGACY and the 8x8 modern cadence (1.5625 ms) both land exactly on this
// physics grid, so samples equal the kinematics at their deadline.
constexpr std::chrono::nanoseconds kPhysicsStep{312'500};
constexpr std::chrono::nanoseconds kStart = 1s;

std::string readFile(const std::string & path)
{
    std::ifstream stream(path);
    EXPECT_TRUE(stream.is_open()) << "Cannot open: " << path;
    std::ostringstream text;
    text << stream.rdbuf();
    return text.str();
}

std::string legacyJson()
{
    return readFile(std::string(GZ_TEST_METADATA_DIR) + "/os1_64_rev7.json");
}

void replaceAll(std::string & text, const std::string & from,
                const std::string & to)
{
    for (size_t at = text.find(from); at != std::string::npos;
         at = text.find(from, at + to.size())) {
        text.replace(at, from.size(), to);
    }
}

std::string modernJson(int measurements_per_packet, int packets_per_frame)
{
    auto json = legacyJson();
    replaceAll(json, "\"udp_profile_imu\": \"LEGACY\"",
               "\"udp_profile_imu\": \"ACCEL32_GYRO32_NMEA\"");
    const auto end = json.rfind('}');
    EXPECT_NE(end, std::string::npos);
    json.insert(end,
        ",\n  \"imu_data_format\": {\"imu_measurements_per_packet\": " +
        std::to_string(measurements_per_packet) +
        ", \"imu_packets_per_frame\": " + std::to_string(packets_per_frame) +
        "}\n");
    return json;
}

/// Nominal (noise-free) IMU kinematics: magnitudes well above 1 g and
/// 1 rad/s so a unit or scale slip cannot hide inside the tolerance.
struct Kinematics {
    Vec3 angular_velocity;     // rad/s
    Vec3 proper_acceleration;  // m/s^2
};

Kinematics kinematicsAt(std::chrono::nanoseconds t)
{
    const double s = static_cast<double>(t.count()) * 1.0e-9;
    return {
        {kPi * std::sin(2.0 * s), -0.5 * kPi + 0.1 * s, 1.2 * std::cos(3.0 * s)},
        {0.3 + std::sin(s), -2.0 * kGravity * std::cos(0.5 * s),
         kGravity + 0.25 * s}};
}

// Float wire precision: ~2^-23 relative, plus the legacy g / deg scaling.
double tolerance(double value)
{
    return 4.0e-6 * std::abs(value) + 1.0e-6;
}

class SdkDecoder {
public:
    explicit SdkDecoder(const OusterMetadata & metadata)
        : info_(std::make_shared<ouster::sdk::core::SensorInfo>(
              metadata.publishedJson())),
          format_(std::make_shared<ouster::sdk::core::PacketFormat>(*info_))
    {}

    ouster::sdk::core::ImuPacket decode(const EncodedOusterImuPacket & encoded) const
    {
        ouster::sdk::core::ImuPacket packet(static_cast<int>(encoded.bytes.size()));
        packet.buf = encoded.bytes;
        packet.format = format_;
        EXPECT_EQ(packet.validate(*info_),
                  ouster::sdk::core::PacketValidationFailure::NONE);
        return packet;
    }

    const ouster::sdk::core::PacketFormat & format() const { return *format_; }

private:
    std::shared_ptr<ouster::sdk::core::SensorInfo> info_;
    std::shared_ptr<ouster::sdk::core::PacketFormat> format_;
};

struct Recording {
    std::vector<ImuSample> samples;
    std::vector<EncodedOusterImuPacket> packets;
    uint64_t skipped = 0;
    bool reset_after_start = false;
};

/// Drive the sampler over [from, to] at `step` with kinematicsAt().
void drive(OusterImuSampler & sampler, std::chrono::nanoseconds from,
           std::chrono::nanoseconds to, std::chrono::nanoseconds step,
           const ImuNoiseDensities & noise, Recording & run)
{
    ImuStepResult out;
    for (auto t = from; t <= to; t += step) {
        const auto k = kinematicsAt(t);
        sampler.step(t, k.angular_velocity, k.proper_acceleration, noise, out);
        EXPECT_TRUE(out.packet_error.empty()) << out.packet_error;
        if (out.reset && t != from) run.reset_after_start = true;
        run.skipped += out.skipped;
        run.samples.insert(run.samples.end(), out.samples.begin(), out.samples.end());
        for (auto & packet : out.packets) run.packets.push_back(std::move(packet));
    }
}

void expectVec3Near(const Eigen::Array<float, Eigen::Dynamic, 3> & decoded,
                    int row, const Vec3 & expected, const char * what)
{
    EXPECT_NEAR(decoded(row, 0), expected.x, tolerance(expected.x)) << what << ".x";
    EXPECT_NEAR(decoded(row, 1), expected.y, tolerance(expected.y)) << what << ".y";
    EXPECT_NEAR(decoded(row, 2), expected.z, tolerance(expected.z)) << what << ".z";
}

// ── LEGACY ────────────────────────────────────────────────────────────────

TEST(ImuPackets, LegacySiInputsRoundTripThroughSdkDecode)
{
    const auto metadata = OusterMetadata::fromJson(legacyJson());
    OusterImuSampler sampler(metadata, 100ms, 7);
    ASSERT_TRUE(sampler.packetsEnabled());
    EXPECT_TRUE(sampler.contract().legacy);
    EXPECT_EQ(sampler.contract().packet_bytes, 48u);
    EXPECT_EQ(sampler.samplePeriod(), 10ms);
    EXPECT_DOUBLE_EQ(sampler.sampleRateHz(), 100.0);
    EXPECT_EQ(sampler.samplesPerFrame(), 10u);

    Recording run;
    drive(sampler, kStart, kStart + 1s, kPhysicsStep, {}, run);
    ASSERT_EQ(run.samples.size(), 101u);  // deadlines t0 .. t0 + 1 s inclusive
    ASSERT_EQ(run.packets.size(), run.samples.size());
    EXPECT_EQ(run.skipped, 0u);
    EXPECT_FALSE(run.reset_after_start);

    const SdkDecoder decoder(metadata);
    for (size_t i = 0; i < run.packets.size(); ++i) {
        SCOPED_TRACE("packet " + std::to_string(i));
        const auto stamp = kStart + 10ms * static_cast<int64_t>(i);
        const auto expected = kinematicsAt(stamp);
        EXPECT_EQ(run.samples[i].stamp_ns, stamp.count());
        EXPECT_EQ(run.packets[i].packet_sequence, i);
        EXPECT_EQ(run.packets[i].first_sample_timestamp_ns,
                  static_cast<uint64_t>(stamp.count()));

        const auto packet = decoder.decode(run.packets[i]);
        // ouster_ros stamps LEGACY IMU messages with gyro_ts.
        EXPECT_EQ(packet.sys_ts(), static_cast<uint64_t>(stamp.count()));
        EXPECT_EQ(packet.accel_ts(), static_cast<uint64_t>(stamp.count()));
        EXPECT_EQ(packet.gyro_ts(), static_cast<uint64_t>(stamp.count()));
        ASSERT_EQ(packet.status().size(), 1);
        EXPECT_EQ(packet.status()(0), 1u);
        expectVec3Near(packet.accel(), 0, expected.proper_acceleration, "accel");
        expectVec3Near(packet.gyro(), 0, expected.angular_velocity, "gyro");
    }
}

TEST(ImuPackets, LegacyWireUnitsAreGAndDegreesPerSecond)
{
    // Regression for SI values written straight into the LEGACY fields:
    // a resting sensor then decoded as ~96 m/s^2 "gravity".
    const auto metadata = OusterMetadata::fromJson(legacyJson());
    OusterImuSampler sampler(metadata, 100ms, 1);
    ImuStepResult out;
    sampler.step(kStart, {0.0, 0.0, kPi / 2.0}, {0.0, 0.0, kGravity}, {}, out);
    ASSERT_EQ(out.packets.size(), 1u);

    const SdkDecoder decoder(metadata);
    const auto * bytes = out.packets[0].bytes.data();
    EXPECT_NEAR(decoder.format().imu_la_z(bytes), 1.0, 1.0e-6);   // g
    EXPECT_NEAR(decoder.format().imu_av_z(bytes), 90.0, 1.0e-4);  // deg/s
    const auto packet = decoder.decode(out.packets[0]);
    EXPECT_NEAR(packet.accel()(0, 2), kGravity, 1.0e-5);
    EXPECT_NEAR(packet.gyro()(0, 2), kPi / 2.0, 1.0e-6);
}

TEST(ImuPackets, LegacyCadenceIgnoresLidarRateButFramesFollowIt)
{
    const auto metadata = OusterMetadata::fromJson(legacyJson());
    OusterImuSampler at20(metadata, 50ms, 1);
    EXPECT_EQ(at20.samplePeriod(), 10ms);
    EXPECT_EQ(at20.contract().packets_per_frame, 5u);
    EXPECT_EQ(at20.samplesPerFrame(), 5u);
}

// ── ACCEL32_GYRO32_NMEA ───────────────────────────────────────────────────

TEST(ImuPackets, ModernSiInputsRoundTripThroughSdkDecode)
{
    const auto metadata = OusterMetadata::fromJson(modernJson(8, 8));
    ASSERT_FALSE(metadata.legacyImuProfile());
    OusterImuSampler sampler(metadata, 100ms, 9);
    ASSERT_TRUE(sampler.packetsEnabled());
    const auto & contract = sampler.contract();
    EXPECT_FALSE(contract.legacy);
    EXPECT_EQ(contract.measurements_per_packet, 8u);
    EXPECT_EQ(contract.packets_per_frame, 8u);
    EXPECT_EQ(sampler.samplePeriod(), 1'562'500ns);  // 10 fps x 64 per frame
    EXPECT_DOUBLE_EQ(sampler.sampleRateHz(), 640.0);
    EXPECT_EQ(sampler.samplesPerFrame(), 64u);

    Recording run;
    drive(sampler, kStart, kStart + 200ms, kPhysicsStep, {}, run);
    ASSERT_EQ(run.samples.size(), 129u);  // 0.2 s at 640 Hz, inclusive
    ASSERT_EQ(run.packets.size(), run.samples.size() / 8);  // 16 full packets

    const SdkDecoder decoder(metadata);
    for (size_t p = 0; p < run.packets.size(); ++p) {
        SCOPED_TRACE("packet " + std::to_string(p));
        const auto & encoded = run.packets[p];
        EXPECT_EQ(encoded.packet_sequence, p);
        EXPECT_EQ(encoded.sample_count, 8u);
        const auto packet = decoder.decode(encoded);
        EXPECT_EQ(packet.packet_type(), 0x2u);
        EXPECT_EQ(packet.frame_id(), metadata.packetFrameId(p / 8));
        EXPECT_EQ(packet.init_id(), metadata.initializationId());
        EXPECT_EQ(packet.prod_sn(), metadata.sensorSerial());
        ASSERT_TRUE(packet.crc().has_value());
        EXPECT_EQ(packet.crc().value(), packet.calculate_crc());

        const auto first = kStart + 1'562'500ns * static_cast<int64_t>(p * 8);
        EXPECT_EQ(packet.nmea_ts(), static_cast<uint64_t>(first.count()));
        const auto timestamps = packet.timestamp();
        const auto status = packet.status();
        const auto ids = packet.measurement_id();
        const auto accel = packet.accel();
        const auto gyro = packet.gyro();
        ASSERT_EQ(timestamps.size(), 8);
        for (int m = 0; m < 8; ++m) {
            SCOPED_TRACE("measurement " + std::to_string(m));
            const auto stamp = first + 1'562'500ns * m;
            const auto expected = kinematicsAt(stamp);
            EXPECT_EQ(timestamps(m), static_cast<uint64_t>(stamp.count()));
            EXPECT_EQ(status(m), 1u);
            // 64 samples per frame over 1024 columns: every 16th column.
            EXPECT_EQ(ids(m), ((p * 8 + m) % 64) * 16u);
            expectVec3Near(accel, m, expected.proper_acceleration, "accel");
            expectVec3Near(gyro, m, expected.angular_velocity, "gyro");
        }
    }
}

TEST(ImuPackets, ModernInterpolatesBetweenPhysicsTicksOffTheSampleGrid)
{
    // 1 kHz physics does not land on the 1.5625 ms sample grid; the sampler
    // interpolates. A linear trajectory makes the interpolation exact.
    const auto metadata = OusterMetadata::fromJson(modernJson(8, 8));
    OusterImuSampler sampler(metadata, 100ms, 9);
    const auto linear = [](std::chrono::nanoseconds t) {
        const double s = static_cast<double>(t.count()) * 1.0e-9;
        return Kinematics{{2.0 * s, -3.0 * s, 0.5},
                          {9.0 * s - 1.0, 4.0, -7.0 * s}};
    };
    ImuStepResult out;
    std::vector<EncodedOusterImuPacket> packets;
    for (auto t = kStart; t <= kStart + 50ms; t += 1ms) {
        const auto k = linear(t);
        sampler.step(t, k.angular_velocity, k.proper_acceleration, {}, out);
        for (auto & packet : out.packets) packets.push_back(std::move(packet));
    }
    ASSERT_GE(packets.size(), 3u);
    const SdkDecoder decoder(metadata);
    for (const auto & encoded : packets) {
        const auto packet = decoder.decode(encoded);
        const auto timestamps = packet.timestamp();
        for (int m = 0; m < 8; ++m) {
            const auto expected = linear(
                std::chrono::nanoseconds(static_cast<int64_t>(timestamps(m))));
            expectVec3Near(packet.accel(), m, expected.proper_acceleration, "accel");
            expectVec3Near(packet.gyro(), m, expected.angular_velocity, "gyro");
        }
    }
}

TEST(ImuPackets, ModernCadenceFollowsLidarRateAndRejectsNonDividingPeriods)
{
    const auto metadata = OusterMetadata::fromJson(modernJson(8, 8));
    OusterImuSampler at20(metadata, 50ms, 1);
    EXPECT_EQ(at20.samplePeriod(), 781'250ns);
    EXPECT_DOUBLE_EQ(at20.sampleRateHz(), 1280.0);
    // 1/7 s is not a whole number of nanoseconds per 64 samples.
    EXPECT_THROW(OusterImuSampler(metadata, std::chrono::nanoseconds(142'857'143), 1),
                 std::invalid_argument);
}

// ── Shared behaviour ──────────────────────────────────────────────────────

TEST(ImuPackets, NoisySamplesArePackedUnchangedForBothProfiles)
{
    // The packets and sensor_msgs/Imu publish the same noisy draws: the
    // pipeline receives each sample exactly on a deadline and must pass it
    // through without re-interpolating.
    ImuNoiseDensities noise;
    noise.gyro_noise_std = 1.75e-4;
    noise.accel_noise_std = 2.3e-3;
    noise.gyro_bias_walk = 1.0e-3;
    noise.accel_bias_walk = 1.0e-2;
    for (const auto & json : {legacyJson(), modernJson(8, 8)}) {
        const auto metadata = OusterMetadata::fromJson(json);
        OusterImuSampler sampler(metadata, 100ms, 3);
        sampler.seed(1234);
        Recording run;
        drive(sampler, kStart, kStart + 300ms, 1ms, noise, run);
        const auto per_packet = sampler.contract().measurements_per_packet;
        ASSERT_GE(run.packets.size(), 3u);
        ASSERT_GE(run.samples.size(), run.packets.size() * per_packet);

        const SdkDecoder decoder(metadata);
        size_t sample = 0;
        bool any_noise = false;
        for (const auto & encoded : run.packets) {
            const auto packet = decoder.decode(encoded);
            const auto accel = packet.accel();
            const auto gyro = packet.gyro();
            for (int m = 0; m < static_cast<int>(per_packet); ++m, ++sample) {
                const auto & s = run.samples[sample];
                expectVec3Near(accel, m, s.linear_acceleration, "accel");
                expectVec3Near(gyro, m, s.angular_velocity, "gyro");
                const auto nominal = kinematicsAt(std::chrono::nanoseconds(s.stamp_ns));
                any_noise = any_noise ||
                    std::abs(s.linear_acceleration.z - nominal.proper_acceleration.z) > 1e-4;
            }
        }
        EXPECT_TRUE(any_noise) << "noise model was not applied";
    }
}

TEST(ImuPackets, RewindStartsNewPacketEpochInsteadOfThrowing)
{
    for (const auto & json : {legacyJson(), modernJson(8, 8)}) {
        const auto metadata = OusterMetadata::fromJson(json);
        OusterImuSampler sampler(metadata, 100ms, 5);
        EXPECT_EQ(sampler.epoch(), 0u);
        Recording before;
        drive(sampler, 2s, 2s + 100ms, kPhysicsStep, {}, before);
        ASSERT_FALSE(before.packets.empty());
        EXPECT_EQ(before.packets.front().epoch, 0u);

        // World reset: sim time jumps back.
        Recording after;
        ImuStepResult out;
        const auto k = kinematicsAt(500ms);
        ASSERT_NO_THROW(sampler.step(500ms, k.angular_velocity,
                                     k.proper_acceleration, {}, out));
        EXPECT_TRUE(out.reset);
        EXPECT_EQ(sampler.epoch(), 1u);
        drive(sampler, 500ms + kPhysicsStep, 600ms, kPhysicsStep, {}, after);
        after.packets.insert(after.packets.begin(), out.packets.begin(), out.packets.end());
        ASSERT_FALSE(after.packets.empty());
        EXPECT_EQ(after.packets.front().epoch, 1u);
        EXPECT_EQ(after.packets.front().packet_sequence, 0u);
        EXPECT_EQ(after.packets.front().first_sample_timestamp_ns, 500'000'000u);

        const SdkDecoder decoder(metadata);
        const auto packet = decoder.decode(after.packets.front());
        expectVec3Near(packet.accel(), 0, k.proper_acceleration, "accel");
    }
}

TEST(ImuPackets, TimeJumpSkipsDeadlinesWithoutSynthesizingSamples)
{
    const auto metadata = OusterMetadata::fromJson(legacyJson());
    OusterImuSampler sampler(metadata, 100ms, 5);
    Recording run;
    drive(sampler, kStart, kStart + 50ms, kPhysicsStep, {}, run);
    // One physics step of 1 s: 100 deadlines due, the scheduler keeps 32.
    ImuStepResult out;
    const auto jump = kStart + 50ms + 1s;
    const auto k = kinematicsAt(jump);
    sampler.step(jump, k.angular_velocity, k.proper_acceleration, {}, out);
    EXPECT_GT(out.skipped, 0u);
    EXPECT_TRUE(out.packet_error.empty()) << out.packet_error;
    EXPECT_EQ(out.packets.size(), out.samples.size());
    for (size_t i = 0; i < out.samples.size(); ++i) {
        EXPECT_EQ(out.packets[i].first_sample_timestamp_ns,
                  static_cast<uint64_t>(out.samples[i].stamp_ns));
    }
    EXPECT_EQ(out.samples.back().stamp_ns, jump.count());
}

TEST(ImuPackets, NonFiniteKinematicsRestartThePacketStream)
{
    const auto metadata = OusterMetadata::fromJson(legacyJson());
    OusterImuSampler sampler(metadata, 100ms, 5);
    ImuStepResult out;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    sampler.step(kStart, {0, 0, 0}, {0, 0, nan}, {}, out);
    EXPECT_FALSE(out.packet_error.empty());
    EXPECT_TRUE(out.packets.empty());
    ASSERT_EQ(out.samples.size(), 1u);  // still reported on /imu
    EXPECT_EQ(sampler.epoch(), 1u);

    sampler.step(kStart + 10ms, {0, 0, 0}, {0, 0, kGravity}, {}, out);
    EXPECT_TRUE(out.packet_error.empty());
    ASSERT_EQ(out.packets.size(), 1u);
    EXPECT_EQ(out.packets[0].epoch, 1u);
}

TEST(ImuPackets, PacketlessSamplerRunsAtRequestedRate)
{
    OusterImuSampler sampler(std::chrono::nanoseconds(5'000'000));
    EXPECT_FALSE(sampler.packetsEnabled());
    EXPECT_EQ(sampler.samplesPerFrame(), 0u);
    EXPECT_THROW(static_cast<void>(sampler.contract()), std::logic_error);
    ImuStepResult out;
    size_t samples = 0;
    for (auto t = kStart; t <= kStart + 100ms; t += 1ms) {
        sampler.step(t, {}, {0, 0, kGravity}, {}, out);
        EXPECT_TRUE(out.packets.empty());
        samples += out.samples.size();
    }
    EXPECT_EQ(samples, 21u);
    EXPECT_THROW(OusterImuSampler(std::chrono::nanoseconds(0)), std::invalid_argument);
}

TEST(ImuPackets, EveryShippedCalibrationBuildsAnImuPacketEncoder)
{
    for (const char * name : {"os0_128_rev7", "os1_128_rev7", "os1_64_rev7",
                              "os2_128_rev7", "osdome_128_rev7"}) {
        for (const char * suffix : {".json", "_legacy.json"}) {
            const std::string path =
                std::string(GZ_TEST_METADATA_DIR) + "/" + name + suffix;
            SCOPED_TRACE(path);
            OusterMetadata metadata = OusterMetadata::fromFile(path);
            OusterImuSampler sampler(metadata, 100ms, 1);
            EXPECT_TRUE(sampler.contract().legacy);
            EXPECT_EQ(sampler.samplePeriod(), 10ms);
        }
    }
}

}  // namespace
}  // namespace gz_gpu_ouster_lidar
