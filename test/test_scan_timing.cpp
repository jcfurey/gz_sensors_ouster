// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "scan_timing.hpp"

namespace gz_gpu_ouster_lidar {

TEST(ScanTiming, ColumnsMatchRollingShutterAcquisition)
{
    constexpr int64_t end = 1'000'000'000;
    constexpr int64_t period = 100'000'000;
    EXPECT_EQ(columnTimestampNs(end, period, 0, 4), 925'000'000);
    EXPECT_EQ(columnTimestampNs(end, period, 1, 4), 950'000'000);
    EXPECT_EQ(columnTimestampNs(end, period, 3, 4), end);
}

TEST(ScanTiming, NonDivisiblePeriodStillEndsExactlyAtCapture)
{
    constexpr int64_t end = 3'000'000'000;
    constexpr int64_t period = 100'000'003;
    int64_t previous = end - period;
    for (int column = 0; column < 1024; ++column) {
        const int64_t stamp =
            columnTimestampNs(end, period, column, 1024);
        EXPECT_GT(stamp, previous);
        EXPECT_LE(stamp, end);
        previous = stamp;
    }
    EXPECT_EQ(previous, end);
}

TEST(ScanTiming, StartupNeverProducesNegativeOrFutureTimestamps)
{
    constexpr int64_t end = 40'000'000;
    constexpr int64_t period = 100'000'000;
    EXPECT_EQ(columnTimestampNs(end, period, 0, 4), 10'000'000);
    EXPECT_EQ(columnTimestampNs(end, period, 3, 4), end);
}

TEST(ScanTiming, RejectsInvalidIndicesAndDimensions)
{
    EXPECT_EQ(columnTimestampNs(100, 10, -1, 4), 0);
    EXPECT_EQ(columnTimestampNs(100, 10, 4, 4), 0);
    EXPECT_EQ(columnTimestampNs(100, 0, 0, 4), 0);
    EXPECT_EQ(columnTimestampNs(-1, 10, 0, 4), 0);
}

}  // namespace gz_gpu_ouster_lidar
