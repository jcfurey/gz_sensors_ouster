// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cmath>

#include "ouster_lidar_profile.hpp"

namespace gz_gpu_ouster_lidar {
namespace {

OusterLidarProfile profile(
    const char * product, const char * revision,
    int firmware_major = 3, int firmware_minor = 1,
    int beams = 128, bool low_data = false)
{
    OusterProfileRequest request;
    request.prod_line = product;
    request.hardware_revision = revision;
    request.firmware_major = firmware_major;
    request.firmware_minor = firmware_minor;
    request.beam_count = beams;
    request.low_data_profile = low_data;
    return resolveOusterLidarProfile(request);
}

TEST(OusterLidarProfile, CoversPublishedProductGenerations)
{
    const auto gen1 = profile("OS1-64", "gen1", 2, 2, 64);
    EXPECT_EQ(gen1.generation, OusterGeneration::Gen1);
    EXPECT_DOUBLE_EQ(gen1.detection_range_10_d90_m, 50.0);
    EXPECT_DOUBLE_EQ(gen1.detection_range_80_d90_m, 110.0);
    EXPECT_DOUBLE_EQ(gen1.minimum_range_m, 0.8);
    EXPECT_EQ(gen1.max_returns, 1);

    for (const char * revision : {"revC", "revD", "rev05"}) {
        const auto os1 = profile("OS1-64", revision, 2, 5, 64);
        EXPECT_EQ(os1.generation, OusterGeneration::Gen2);
        EXPECT_DOUBLE_EQ(os1.detection_range_10_d90_m, 45.0);
        EXPECT_DOUBLE_EQ(os1.detection_range_80_d90_m, 100.0);
        EXPECT_DOUBLE_EQ(os1.representable_range_m, 270.0);
        EXPECT_EQ(os1.max_returns, 1);
    }

    for (const char * revision : {"rev06", "rev06.2"}) {
        const auto os1 = profile("OS1-128", revision, 2, 5);
        EXPECT_EQ(os1.generation, OusterGeneration::Gen2);
        EXPECT_EQ(os1.max_returns, 2);
    }

    EXPECT_DOUBLE_EQ(
        profile("OS2-128", "rev06", 2, 4).representable_range_m,
        465.0);

    const auto rev7 = profile("OS1-128", "rev07");
    EXPECT_EQ(rev7.generation, OusterGeneration::Gen3);
    EXPECT_DOUBLE_EQ(rev7.detection_range_10_d90_m, 90.0);
    EXPECT_DOUBLE_EQ(rev7.detection_range_80_d90_m, 170.0);

    const auto rev8 = profile("OS1MAX-256", "rev08", 4, 0, 256);
    EXPECT_EQ(rev8.generation, OusterGeneration::Gen4);
    EXPECT_EQ(rev8.model, OusterModel::OS1Max);
    EXPECT_DOUBLE_EQ(rev8.detection_range_10_d90_m, 200.0);
    EXPECT_DOUBLE_EQ(rev8.detection_range_80_d90_m, 350.0);
    EXPECT_DOUBLE_EQ(rev8.representable_range_m, 500.0);
}

TEST(OusterLidarProfile, CoversEveryRev8Model)
{
    const auto os0 = profile("OS0-128", "rev08", 4, 0);
    const auto os1 = profile("OS1-128", "rev08", 4, 0);
    const auto dome = profile("OSDome-128", "rev08", 4, 0);
    const auto max = profile("OS1MAX-256", "rev08", 4, 0, 256);

    for (const auto * p : {&os0, &os1, &dome, &max}) {
        EXPECT_TRUE(p->supported);
        EXPECT_EQ(p->generation, OusterGeneration::Gen4);
        EXPECT_DOUBLE_EQ(p->representable_range_m, 500.0);
    }
    EXPECT_DOUBLE_EQ(os0.detection_range_10_d90_m, 35.0);
    EXPECT_DOUBLE_EQ(os1.detection_range_10_d90_m, 90.0);
    EXPECT_DOUBLE_EQ(dome.detection_range_10_d90_m, 20.0);
    EXPECT_DOUBLE_EQ(dome.precision_min_std_m, 0.005);
    EXPECT_DOUBLE_EQ(dome.precision_max_std_m, 0.050);
    EXPECT_DOUBLE_EQ(max.detection_range_10_d90_m, 200.0);
}

TEST(OusterLidarProfile, RejectsImpossibleProductRevisionPairs)
{
    EXPECT_FALSE(profile("OS2-128", "rev08", 4, 0).supported);
    EXPECT_FALSE(profile("OSDome-128", "rev06", 2, 5).supported);
    EXPECT_FALSE(profile("OS0-128", "gen1", 1, 13).supported);
    EXPECT_FALSE(profile("OS2-128", "rev07.1", 3, 1).supported);
}

TEST(OusterLidarProfile, CoversEveryRev7Model)
{
    const auto os0 = profile("OS0-128", "rev07");
    const auto os1 = profile("OS1-64", "rev07", 3, 1, 64);
    const auto os2 = profile("OS2-128", "rev07", 2, 5);
    const auto dome = profile("OSDome-128", "rev07");

    EXPECT_DOUBLE_EQ(os0.detection_range_10_d90_m, 35.0);
    EXPECT_DOUBLE_EQ(os0.detection_range_80_d90_m, 75.0);
    EXPECT_DOUBLE_EQ(os1.detection_range_10_d90_m, 90.0);
    EXPECT_DOUBLE_EQ(os1.detection_range_80_d90_m, 170.0);
    EXPECT_DOUBLE_EQ(os2.detection_range_10_d90_m, 200.0);
    EXPECT_DOUBLE_EQ(os2.detection_range_80_d90_m, 350.0);
    EXPECT_DOUBLE_EQ(os2.minimum_range_m, 0.8);
    EXPECT_DOUBLE_EQ(dome.detection_range_10_d90_m, 20.0);
    EXPECT_DOUBLE_EQ(dome.detection_range_80_d90_m, 45.0);
    EXPECT_DOUBLE_EQ(dome.representable_range_m, 233.0);
    EXPECT_TRUE(os0.supported);
    EXPECT_TRUE(os1.supported);
    EXPECT_TRUE(os2.supported);
    EXPECT_TRUE(dome.supported);
}

TEST(OusterLidarProfile, InfersRevisionFromBothPartNumberFormats)
{
    OusterProfileRequest old_pn;
    old_pn.prod_line = "OS1-128";
    old_pn.prod_pn = "860-105010-07";
    old_pn.hardware_revision = "auto";
    const auto rev7 = resolveOusterLidarProfile(old_pn);
    EXPECT_EQ(rev7.revision, OusterRevision::Rev07);
    EXPECT_TRUE(rev7.revision_inferred);
    EXPECT_FALSE(rev7.fallback_revision);

    OusterProfileRequest revc_pn = old_pn;
    revc_pn.prod_pn = "840105010C";
    const auto revc = resolveOusterLidarProfile(revc_pn);
    EXPECT_EQ(revc.revision, OusterRevision::RevC);
    EXPECT_TRUE(revc.revision_inferred);

    OusterProfileRequest new_pn = old_pn;
    new_pn.prod_line = "OS1MAX-256";
    new_pn.prod_pn = "OS1MAX-080-256-U-002-XX";
    const auto rev8 = resolveOusterLidarProfile(new_pn);
    EXPECT_EQ(rev8.revision, OusterRevision::Rev08);
    EXPECT_TRUE(rev8.revision_inferred);

    OusterProfileRequest fw4 = old_pn;
    fw4.prod_line = "OSDome-128";
    fw4.prod_pn = "synthetic";
    fw4.firmware_major = 4;
    const auto rev8_fw = resolveOusterLidarProfile(fw4);
    EXPECT_EQ(rev8_fw.revision, OusterRevision::Rev08);
    EXPECT_TRUE(rev8_fw.revision_inferred);
}

TEST(OusterLidarProfile, AmbiguousMetadataIsMarkedAsFallback)
{
    OusterProfileRequest request;
    request.prod_line = "OS2-128";
    request.prod_pn = "860-os2128";
    request.hardware_revision = "auto";
    request.firmware_major = 2;
    request.firmware_minor = 5;
    const auto p = resolveOusterLidarProfile(request);
    EXPECT_EQ(p.revision, OusterRevision::Rev06);
    EXPECT_TRUE(p.fallback_revision);
}

TEST(OusterLidarProfile, FirmwareAndPacketProfileAffectResolution)
{
    EXPECT_DOUBLE_EQ(
        profile("OS1-64", "rev05", 2, 2, 64).range_resolution_m,
        0.003);
    EXPECT_DOUBLE_EQ(
        profile("OS1-64", "rev05", 2, 5, 64).range_resolution_m,
        0.001);
    EXPECT_DOUBLE_EQ(
        profile("OS1-64", "rev07", 3, 1, 64, true).range_resolution_m,
        0.008);
}

TEST(OusterLidarProfile, ModeRangeScaleUsesDocumentedHalfRateIncrement)
{
    const auto p = profile("OS1-64", "rev07", 3, 1, 64);
    EXPECT_NEAR(ousterModeRangeScale(p, 1024, 10.0), 1.0, 1e-12);
    EXPECT_NEAR(ousterModeRangeScale(p, 512, 10.0), 1.19, 1e-12);
    EXPECT_NEAR(ousterModeRangeScale(p, 1024, 20.0), 1.0 / 1.19, 1e-12);
    EXPECT_NEAR(ousterModeRangeScale(p, 1024, 40.0),
                1.0 / (1.19 * 1.19), 1e-12);

    EXPECT_NEAR(ousterModePrecisionScale(p, 1024, 10.0), 1.0, 1e-12);
    EXPECT_NEAR(ousterModePrecisionScale(p, 512, 10.0),
                1.0 / std::sqrt(2.0), 1e-12);
    EXPECT_NEAR(ousterModePrecisionScale(p, 1024, 20.0),
                std::sqrt(2.0), 1e-12);

    const auto os2 = profile("OS2-128", "rev07", 2, 5);
    EXPECT_NEAR(ousterModeRangeScale(os2, 2048, 10.0), 1.0, 1e-12);
}

}  // namespace
}  // namespace gz_gpu_ouster_lidar
