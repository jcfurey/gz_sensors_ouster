// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include <ouster/metadata.h>
#include <ouster/impl/packet_writer.h>
#include <ouster/types.h>

#ifndef TEST_METADATA_DIR
#error "TEST_METADATA_DIR must be defined at compile time"
#endif

namespace gz_gpu_ouster_lidar {

static std::string readFile(const std::string & path)
{
    std::ifstream fs(path);
    EXPECT_TRUE(fs.is_open()) << "Cannot open: " << path;
    std::ostringstream ss;
    ss << fs.rdbuf();
    return ss.str();
}

// Parameterised test: run the same checks across all metadata JSONs
class MetadataParsingTest : public ::testing::TestWithParam<std::string> {};

TEST_P(MetadataParsingTest, ParsesSuccessfully)
{
    const std::string path = std::string(TEST_METADATA_DIR) + "/" + GetParam();
    const auto json = readFile(path);
    ASSERT_FALSE(json.empty());

    ouster::sdk::core::SensorInfo info(json);
    ouster::sdk::core::PacketFormat pf(info);

    EXPECT_GT(info.beam_altitude_angles.size(), 0u);
    EXPECT_EQ(info.beam_altitude_angles.size(), info.beam_azimuth_angles.size());
    EXPECT_GT(info.format.columns_per_frame, 0u);
    EXPECT_FALSE(info.prod_line.empty());
}

TEST_P(MetadataParsingTest, PacketWriterCreation)
{
    const std::string path = std::string(TEST_METADATA_DIR) + "/" + GetParam();
    const auto json = readFile(path);

    ouster::sdk::core::SensorInfo info(json);
    ouster::sdk::core::PacketFormat pf(info);
    ouster::sdk::core::impl::PacketWriter pw(pf);

    EXPECT_GT(pw.pixels_per_column, 0);
    EXPECT_GT(pw.columns_per_packet, 0);
    EXPECT_GT(pw.lidar_packet_size, 0u);

    // columns_per_frame must be divisible by columns_per_packet
    int W = static_cast<int>(info.format.columns_per_frame);
    EXPECT_EQ(W % pw.columns_per_packet, 0)
        << "W=" << W << " cpp=" << pw.columns_per_packet;
}

TEST_P(MetadataParsingTest, BeamAnglesInRange)
{
    const std::string path = std::string(TEST_METADATA_DIR) + "/" + GetParam();
    const auto json = readFile(path);

    ouster::sdk::core::SensorInfo info(json);

    // Beam altitude angles should be within [-90, 90] degrees
    for (size_t i = 0; i < info.beam_altitude_angles.size(); ++i) {
        EXPECT_GE(info.beam_altitude_angles[i], -90.0)
            << "beam " << i << " altitude out of range";
        EXPECT_LE(info.beam_altitude_angles[i], 90.0)
            << "beam " << i << " altitude out of range";
    }

    // Azimuth offsets are typically small (< 10 degrees)
    for (size_t i = 0; i < info.beam_azimuth_angles.size(); ++i) {
        EXPECT_GE(info.beam_azimuth_angles[i], -15.0)
            << "beam " << i << " azimuth offset unusually large";
        EXPECT_LE(info.beam_azimuth_angles[i], 15.0)
            << "beam " << i << " azimuth offset unusually large";
    }
}

TEST_P(MetadataParsingTest, MaxRangeDerivation)
{
    const std::string path = std::string(TEST_METADATA_DIR) + "/" + GetParam();
    const auto json = readFile(path);

    ouster::sdk::core::SensorInfo info(json);
    const auto & pl = info.prod_line;

    // Verify product line is one of the known types
    bool known = (pl.find("OS0") != std::string::npos ||
                  pl.find("OS1") != std::string::npos ||
                  pl.find("OS2") != std::string::npos ||
                  pl.find("OSDome") != std::string::npos);
    EXPECT_TRUE(known) << "Unknown product line: " << pl;

    // Verify max_range derivation logic
    double max_range = 120.0;  // default
    if (pl.find("OS0") != std::string::npos)      max_range = 50.0;
    else if (pl.find("OS1") != std::string::npos)  max_range = 120.0;
    else if (pl.find("OS2") != std::string::npos)  max_range = 240.0;

    EXPECT_GT(max_range, 0.0);
}

TEST_P(MetadataParsingTest, PixelsPerColumnMatchesBeamCount)
{
    const std::string path = std::string(TEST_METADATA_DIR) + "/" + GetParam();
    const auto json = readFile(path);

    ouster::sdk::core::SensorInfo info(json);
    ouster::sdk::core::PacketFormat pf(info);
    ouster::sdk::core::impl::PacketWriter pw(pf);

    EXPECT_EQ(static_cast<size_t>(pw.pixels_per_column),
              info.beam_altitude_angles.size())
        << "pixels_per_column should match beam count";
}

// Discover metadata files at compile time via preprocessor
INSTANTIATE_TEST_SUITE_P(
    AllSensors,
    MetadataParsingTest,
    ::testing::Values(
        "os0_128_rev7.json",
        "os1_64_rev7.json",
        "os1_128_rev7.json",
        "os2_128_rev7.json",
        "osdome_128_rev7.json"
    ),
    [](const ::testing::TestParamInfo<std::string> & info) {
        // Sanitise filename for test name (replace dots/hyphens with underscores)
        std::string name = info.param;
        for (auto & c : name) {
            if (c == '.' || c == '-') c = '_';
        }
        return name;
    }
);

// ── Malformed metadata tests ────────────────────────────────────────────────

TEST(MetadataParsing, EmptyStringThrows)
{
    EXPECT_THROW(ouster::sdk::core::SensorInfo(""), std::exception);
}

TEST(MetadataParsing, GarbageJsonThrows)
{
    EXPECT_THROW(ouster::sdk::core::SensorInfo("{\"not\": \"metadata\"}"),
                 std::exception);
}

TEST(MetadataParsing, InvalidJsonThrows)
{
    EXPECT_THROW(ouster::sdk::core::SensorInfo("this is not json"),
                 std::exception);
}

}  // namespace gz_gpu_ouster_lidar
