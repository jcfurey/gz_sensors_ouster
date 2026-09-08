// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#include "ouster_metadata.hpp"
#include "packet_encoder.hpp"
#include "gz_gpu_ouster_lidar/sim_time_scheduler.hpp"
#include "conformance_v1.hpp"

#include <gtest/gtest.h>
#include <ouster/types.h>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <future>
#include <mutex>
#include <unistd.h>

namespace gz_gpu_ouster_lidar {
namespace {
namespace core = ouster_sim_core;
namespace fixture = core::conformance_v1;
using namespace std::chrono_literals;

class MetadataFile {
public:
    explicit MetadataFile(const std::string & profile) {
        path = std::filesystem::temp_directory_path() /
            ("gz_ouster_conformance_" + std::to_string(getpid()) + ".json");
        write(fixture::metadataJson(OUSTER_SIM_CORE_TEST_DATA_DIR "/os1_64_rev7.json", profile));
    }
    ~MetadataFile() { std::filesystem::remove(path); }
    void write(const std::string & json) { std::ofstream(path) << json; }
    std::filesystem::path path;
};

struct Capture {
    std::mutex mutex;
    std::condition_variable cv;
    std::vector<std::vector<uint8_t>> packets;
    bool block = false;
    PacketEncoder encoder;

    ~Capture() { release(); encoder.stop(); }
    void start(const OusterMetadata & metadata, double hz = 10,
               core::PacketDeliveryMode mode = core::PacketDeliveryMode::kBurst) {
        encoder.start(&metadata, [this](const auto & packet) {
            std::unique_lock lock(mutex);
            packets.push_back(packet.buf);
            cv.notify_all();
            cv.wait(lock, [this] { return !block; });
        }, hz, mode);
    }
    void release() {
        std::lock_guard lock(mutex);
        block = false;
        cv.notify_all();
    }
    bool wait(size_t count, std::chrono::milliseconds timeout = 3s) {
        std::unique_lock lock(mutex);
        return cv.wait_for(lock, timeout, [&] { return packets.size() >= count; });
    }
    void submit(const core::OusterScanFrame & frame, uint64_t epoch = 0) {
        encoder.encodeScan(static_cast<int64_t>(frame.column_timestamp_ns.back()), epoch,
            frame.range_mm.data(), frame.signal.data(), frame.reflectivity.data(), frame.near_ir.data());
    }
};

TEST(CoreConformance, ProductionMetadataAndEncoderMatchEverySharedPrimaryProfile)
{
    for (const auto * profile : fixture::primaryProfiles) {
        for (const double hz : {10.0, 7.0}) {
            SCOPED_TRACE(std::string(profile) + " / " + std::to_string(hz));
            MetadataFile file(profile);
            OusterMetadata metadata;
            double maximum = 0;
            ASSERT_TRUE(metadata.load(file.path, false, "rev07", false, maximum));
            const auto reference = core::OusterMetadata::fromFile(file.path);
            EXPECT_EQ(metadata.metadata_str, reference.publishedJson());
            EXPECT_EQ(metadata.beam_alt_angles, reference.beamAltitudeDeg());
            EXPECT_EQ(metadata.beam_az_offsets, reference.beamAzimuthDeg());
            EXPECT_DOUBLE_EQ(maximum, reference.resolvedProductProfile("rev07").representable_range_m);
            Capture capture;
            capture.start(metadata, hz);
            const auto frame = fixture::frame(reference, 0, periodFromHz(hz).count());
            const auto expected = core::OusterPacketEncoder(reference).encode(frame);
            capture.submit(frame);
            ASSERT_TRUE(capture.wait(expected.size()));
            capture.encoder.stop();
            ASSERT_EQ(capture.packets.size(), expected.size());
            for (size_t i = 0; i < expected.size(); ++i) {
                EXPECT_EQ(capture.packets[i], expected[i].bytes) << "packet " << i;
            }
        }
    }
}

TEST(CoreConformance, ProductionLoaderAcceptsEveryShippedCalibration)
{
    size_t count = 0;
    for (const auto & file : std::filesystem::directory_iterator(GZ_TEST_METADATA_DIR)) {
        if (file.path().extension() != ".json") continue;
        SCOPED_TRACE(file.path().string());
        OusterMetadata metadata;
        double maximum = 0;
        ASSERT_TRUE(metadata.load(file.path(), true, "auto", false, maximum));
        EXPECT_GT(metadata.H, 0);
        EXPECT_GT(metadata.W, 0);
        EXPECT_GT(metadata.imu_packet_size, 0u);
        EXPECT_EQ(metadata.beam_alt_f.size(), static_cast<size_t>(metadata.H));
        EXPECT_EQ(metadata.beam_az_f.size(), static_cast<size_t>(metadata.H));
        ++count;
    }
    EXPECT_GE(count, 10u);
}

TEST(CoreConformance, RejectsUnsupportedReturnsAndUnencodableRangeBeforePublication)
{
    for (const auto * profile : fixture::secondaryProfiles) {
        MetadataFile file(profile);
        OusterMetadata metadata;
        double maximum = 120;
        EXPECT_FALSE(metadata.load(file.path, false, "rev07", false, maximum));
    }
    MetadataFile file("RNG15_RFL8_NIR8");
    auto low_json = fixture::metadataJson(OUSTER_SIM_CORE_TEST_DATA_DIR "/os1_64_rev7.json",
                                          "RNG15_RFL8_NIR8");
    const auto model = low_json.find("OS1-64");
    ASSERT_NE(model, std::string::npos);
    low_json.replace(model, 6, "OS2-64");
    file.write(low_json);
    OusterMetadata metadata;
    double maximum = 300;
    EXPECT_FALSE(metadata.load(file.path, false, "rev07", true, maximum));
    ASSERT_TRUE(metadata.load(file.path, false, "rev07", false, maximum));
    EXPECT_DOUBLE_EQ(maximum, 262.136);
    auto frame = fixture::frame(metadata.core());
    Capture capture;
    capture.start(metadata);
    for (const uint32_t invalid : {300000u, 262144u, 12001u}) {
        frame.range_mm[0] = invalid;
        EXPECT_THROW(capture.submit(frame), std::invalid_argument);
    }
    frame.range_mm[0] = 262136;
    capture.submit(frame);
    ASSERT_TRUE(capture.wait(64));
    capture.encoder.stop();
    const auto expected = core::OusterPacketEncoder(metadata.core()).encode(frame);
    EXPECT_EQ(capture.packets.front(), expected.front().bytes);
    EXPECT_EQ(capture.packets.size(), 64u);  // invalid frames consumed no identity
}

TEST(CoreConformance, MetadataFirmwareAdjustmentUsesUnmodifiedProductIdentity)
{
    MetadataFile file("RNG19_RFL8_SIG16_NIR16");
    auto json = fixture::metadataJson(OUSTER_SIM_CORE_TEST_DATA_DIR "/os1_64_rev7.json",
                                      "RNG19_RFL8_SIG16_NIR16");
    size_t pos;
    while ((pos = json.find("v3.2.0")) != std::string::npos) json.replace(pos, 6, "v2.4.0");
    file.write(json);
    OusterMetadata metadata;
    double maximum = 0;
    ASSERT_TRUE(metadata.load(file.path, false, "rev07", false, maximum));
    EXPECT_TRUE(metadata.core().firmwareAdvertisementAdjusted());
    EXPECT_EQ(metadata.core().sourceFirmwareVersion().major, 2u);
    EXPECT_EQ(metadata.metadata_str, metadata.core().publishedJson());
}

TEST(CoreConformance, CongestedSinkKeepsActiveAndNewestWholeFrameWithoutBlockingProducer)
{
    MetadataFile file("RNG19_RFL8_SIG16_NIR16");
    OusterMetadata metadata;
    double maximum = 0;
    ASSERT_TRUE(metadata.load(file.path, false, "rev07", false, maximum));
    Capture capture;
    capture.block = true;
    capture.start(metadata);
    const auto frame = fixture::frame(metadata.core());
    capture.submit(frame);
    ASSERT_TRUE(capture.wait(1));  // sink is now held inside its first publish
    auto producer = std::async(std::launch::async, [&] {
        for (int i = 0; i < 4; ++i) capture.submit(frame);
        capture.encoder.setSimulationState(true, 0);
        capture.encoder.setSimulationState(false, 0);
    });
    const auto status = producer.wait_for(1s);
    // Always release before asserting, including the lock-regression failure.
    capture.release();
    producer.get();
    EXPECT_EQ(status, std::future_status::ready);
    EXPECT_EQ(capture.encoder.droppedBatches(), 3u);
    ASSERT_TRUE(capture.wait(128));
    capture.encoder.stop();
    ASSERT_EQ(capture.packets.size(), 128u);
    ouster::sdk::core::PacketFormat format(
        ouster::sdk::core::SensorInfo(metadata.metadata_str));
    for (size_t i = 0; i < capture.packets.size(); ++i) {
        const auto * bytes = capture.packets[i].data();
        EXPECT_EQ(format.frame_id(bytes), i < 64 ? 0u : 4u);
        EXPECT_EQ(format.col_measurement_id(format.nth_col(0, bytes)), (i % 64) * 16);
    }
}

TEST(CoreConformance, PauseResetCancelsOldFrameAndRestartPublishesAgain)
{
    MetadataFile file("LEGACY");
    OusterMetadata metadata;
    double maximum = 0;
    ASSERT_TRUE(metadata.load(file.path, false, "rev07", false, maximum));
    Capture capture;
    capture.start(metadata);
    const auto frame = fixture::frame(metadata.core());
    capture.encoder.setSimulationState(true, 0);
    capture.submit(frame);
    EXPECT_FALSE(capture.wait(1, 20ms));
    capture.encoder.setSimulationState(true, 1);
    capture.submit(frame, 1);
    capture.encoder.setSimulationState(false, 1);
    ASSERT_TRUE(capture.wait(64));
    capture.encoder.stop();
    auto expected_frame = frame;
    expected_frame.revolution = 1;
    auto expected = core::OusterPacketEncoder(metadata.core()).encode(expected_frame);
    ASSERT_EQ(capture.packets.size(), 64u);
    EXPECT_EQ(capture.packets.front(), expected.front().bytes);
    capture.start(metadata);
    capture.submit(frame);
    ASSERT_TRUE(capture.wait(128));
    capture.encoder.stop();
    expected = core::OusterPacketEncoder(metadata.core()).encode(frame);
    EXPECT_EQ(capture.packets[64], expected.front().bytes);
}

TEST(CoreConformance, EpochChangeDuringBlockedPublishCancelsRemainingOldPackets)
{
    MetadataFile file("LEGACY");
    OusterMetadata metadata;
    double maximum = 0;
    ASSERT_TRUE(metadata.load(file.path, false, "rev07", false, maximum));
    Capture capture;
    capture.block = true;
    capture.start(metadata);
    const auto frame = fixture::frame(metadata.core());
    capture.submit(frame);
    ASSERT_TRUE(capture.wait(1));
    capture.encoder.setSimulationState(false, 1);
    capture.submit(frame, 1);
    capture.release();
    ASSERT_TRUE(capture.wait(65));
    capture.encoder.stop();
    ASSERT_EQ(capture.packets.size(), 65u);
    auto new_frame = frame;
    new_frame.revolution = 1;
    const auto expected = core::OusterPacketEncoder(metadata.core()).encode(new_frame);
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(capture.packets[i + 1], expected[i].bytes);
    }
}

}  // namespace
}  // namespace gz_gpu_ouster_lidar
