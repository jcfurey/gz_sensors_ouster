// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0
//
// <lidar_hz> resolution against the frame rate the Ouster metadata declares
// (data_format.fps, else the lidar_mode suffix): the metadata is the default,
// an explicit SDF rate wins but is flagged on mismatch, and invalid SDF
// values fall back to the metadata instead of a hard-coded 10 Hz.

#include "lidar_rate.hpp"
#include "ouster_metadata.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <unistd.h>

#ifndef GZ_TEST_METADATA_DIR
#error "GZ_TEST_METADATA_DIR must be defined at compile time"
#endif

namespace gz_gpu_ouster_lidar {
namespace {

using Source = LidarRateResolution::Source;

std::string readFile(const std::string & path)
{
    std::ifstream stream(path);
    EXPECT_TRUE(stream.is_open()) << "Cannot open: " << path;
    std::ostringstream text;
    text << stream.rdbuf();
    return text.str();
}

void replaceAll(std::string & text, const std::string & from,
                const std::string & to)
{
    for (size_t at = text.find(from); at != std::string::npos;
         at = text.find(from, at + to.size())) {
        text.replace(at, from.size(), to);
    }
}

std::string os1Json()
{
    return readFile(std::string(GZ_TEST_METADATA_DIR) + "/os1_64_rev7.json");
}

/// The shipped OS1-64 metadata re-declared as a 1024x20 sensor.
std::string os1Json20Hz()
{
    auto json = os1Json();
    replaceAll(json, "\"lidar_mode\": \"1024x10\"", "\"lidar_mode\": \"1024x20\"");
    replaceAll(json, "\"fps\": 10", "\"fps\": 20");
    return json;
}

class TempFile {
public:
    explicit TempFile(const std::string & contents)
        : path_(std::filesystem::temp_directory_path() /
                ("gz_ouster_lidar_rate_" + std::to_string(getpid()) + ".json"))
    {
        std::ofstream(path_) << contents;
    }
    ~TempFile() { std::filesystem::remove(path_); }
    std::string path() const { return path_.string(); }

private:
    std::filesystem::path path_;
};

TEST(LidarRate, MetadataRateIsTheDefault)
{
    const auto rate = resolveLidarRate(std::nullopt, 20.0);
    EXPECT_DOUBLE_EQ(rate.hz, 20.0);
    EXPECT_EQ(rate.source, Source::kMetadata);
    EXPECT_FALSE(rate.mismatch);
    EXPECT_FALSE(rate.sdf_invalid);
}

TEST(LidarRate, MatchingSdfRateIsNotAMismatch)
{
    const auto rate = resolveLidarRate(10.0, 10.0);
    EXPECT_DOUBLE_EQ(rate.hz, 10.0);
    EXPECT_EQ(rate.source, Source::kSdf);
    EXPECT_FALSE(rate.mismatch);
}

TEST(LidarRate, DisagreeingSdfRateIsHonouredButFlagged)
{
    for (const double sdf : {5.0, 20.0, 10.5}) {
        const auto rate = resolveLidarRate(sdf, 10.0);
        EXPECT_DOUBLE_EQ(rate.hz, sdf);
        EXPECT_EQ(rate.source, Source::kSdf);
        EXPECT_TRUE(rate.mismatch) << sdf;
    }
}

TEST(LidarRate, InvalidSdfRateFallsBackToMetadata)
{
    for (const double bad : {0.0, -10.0, std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::infinity()}) {
        const auto rate = resolveLidarRate(bad, 20.0);
        EXPECT_TRUE(rate.sdf_invalid) << bad;
        EXPECT_FALSE(rate.mismatch) << bad;
        EXPECT_DOUBLE_EQ(rate.hz, 20.0) << bad;
        EXPECT_EQ(rate.source, Source::kMetadata) << bad;
    }
}

TEST(LidarRate, NoMetadataRateUsesSdfOrFallback)
{
    const auto sdf = resolveLidarRate(15.0, std::nullopt);
    EXPECT_DOUBLE_EQ(sdf.hz, 15.0);
    EXPECT_EQ(sdf.source, Source::kSdf);
    EXPECT_FALSE(sdf.mismatch);

    const auto none = resolveLidarRate(std::nullopt, std::nullopt);
    EXPECT_DOUBLE_EQ(none.hz, 10.0);
    EXPECT_EQ(none.source, Source::kFallback);

    const auto invalid = resolveLidarRate(-1.0, 0.0, 12.5);
    EXPECT_TRUE(invalid.sdf_invalid);
    EXPECT_DOUBLE_EQ(invalid.hz, 12.5);
    EXPECT_EQ(invalid.source, Source::kFallback);
}

TEST(LidarRate, EveryShippedCalibrationDeclaresTenHz)
{
    size_t count = 0;
    for (const auto & file : std::filesystem::directory_iterator(GZ_TEST_METADATA_DIR)) {
        if (file.path().extension() != ".json") continue;
        SCOPED_TRACE(file.path().string());
        const auto hz = metadataFrameRateHz(readFile(file.path().string()));
        ASSERT_TRUE(hz.has_value());
        EXPECT_DOUBLE_EQ(*hz, 10.0);
        ++count;
    }
    EXPECT_GE(count, 10u);
}

TEST(LidarRate, MetadataFrameRateFollowsLidarMode)
{
    EXPECT_EQ(metadataFrameRateHz(os1Json20Hz()), std::optional<double>(20.0));

    // Without data_format.fps the SDK derives the rate from lidar_mode.
    auto no_fps = os1Json20Hz();
    replaceAll(no_fps, "\"fps\": 20", "\"fps_removed\": 20");
    EXPECT_EQ(metadataFrameRateHz(no_fps), std::optional<double>(20.0));
}

TEST(LidarRate, LoaderExposesMetadataRateForPluginDefault)
{
    TempFile file(os1Json20Hz());
    OusterMetadata metadata;
    double max_range = 0.0;
    ASSERT_TRUE(metadata.load(file.path(), false, "rev07", false, max_range));
    ASSERT_TRUE(metadata.frame_rate_hz.has_value());
    EXPECT_DOUBLE_EQ(*metadata.frame_rate_hz, 20.0);

    // No <lidar_hz>: the plugin scans at the metadata's 20 Hz.
    EXPECT_DOUBLE_EQ(resolveLidarRate(std::nullopt, metadata.frame_rate_hz).hz, 20.0);
    // The historical 10 Hz example value is now detected as inconsistent.
    EXPECT_TRUE(resolveLidarRate(10.0, metadata.frame_rate_hz).mismatch);
}

}  // namespace
}  // namespace gz_gpu_ouster_lidar
