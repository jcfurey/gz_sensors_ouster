// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0
//
// Direct unit tests for beamRayAzimuthDeg (ray_processor_math.hpp).
// Verifies the sign convention: m=0 → azimuth = −beam_az (offset subtracted),
// encoder increases clockwise (azimuth decreases with m), matching the Ouster
// SDK XYZ LUT convention (xyzlut.cpp: azimuth = −beam_azimuth_angles).

#include <gtest/gtest.h>

#include "ray_processor_math.hpp"

namespace gz_gpu_ouster_lidar {

TEST(BeamMath, BeamAzimuthColumnZeroAtZeroOffset)
{
    // No beam-azimuth offset, first column: result is zero.
    EXPECT_FLOAT_EQ(rpmath::beamRayAzimuthDeg(0.0f, 0, 1.0f), 0.0f);
}

TEST(BeamMath, BeamAzimuthDecreasesClockwise)
{
    // Ouster encoder is clockwise: 90 columns at 1°/col → azimuth = −90°.
    EXPECT_FLOAT_EQ(rpmath::beamRayAzimuthDeg(0.0f, 90, 1.0f), -90.0f);
}

TEST(BeamMath, BeamAzimuthOffsetIsSubtracted)
{
    // beam_az_deg is subtracted (not added) to match XYZ-LUT reconstruction.
    EXPECT_FLOAT_EQ(rpmath::beamRayAzimuthDeg(5.0f, 0, 1.0f), -5.0f);
}

}  // namespace gz_gpu_ouster_lidar
