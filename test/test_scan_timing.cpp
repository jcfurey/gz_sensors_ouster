// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "scan_timing.hpp"

#include <algorithm>
#include <cstdlib>

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

TEST(ScanTiming, GroupMidTimesFallMidwayThroughEachGroupsColumns)
{
    constexpr int64_t end = 1'000'000'000;
    constexpr int64_t period = 100'000'000;
    constexpr int columns = 1024;
    constexpr int groups = 16;
    int64_t previous = end - period;
    for (int group = 0; group < groups; ++group) {
        // The columns rc::castScanGrouped casts with this group.
        int first = columns, last = -1;
        for (int m = 0; m < columns; ++m) {
            if (m * groups / columns == group) {
                first = std::min(first, m);
                last = std::max(last, m);
            }
        }
        const int64_t a = columnTimestampNs(end, period, first, columns);
        const int64_t b = columnTimestampNs(end, period, last, columns);
        const int64_t mid =
            columnGroupMidTimestampNs(end, period, group, groups, columns);
        EXPECT_LE(std::llabs(2 * mid - a - b), 1) << "group " << group;
        EXPECT_GT(mid, previous);
        previous = mid;
    }
    EXPECT_LT(previous, end);
}

TEST(ScanTiming, UnevenGroupsAndInvalidArguments)
{
    // 10 columns in 3 groups by m * 3 / 10: {0..3}, {4..6}, {7..9}.
    constexpr int64_t end = 1'000'000'000;
    constexpr int64_t period = 100'000'000;
    EXPECT_EQ(columnGroupMidTimestampNs(end, period, 0, 3, 10),
              (columnTimestampNs(end, period, 0, 10) +
               columnTimestampNs(end, period, 3, 10)) / 2);
    EXPECT_EQ(columnGroupMidTimestampNs(end, period, 1, 3, 10),
              (columnTimestampNs(end, period, 4, 10) +
               columnTimestampNs(end, period, 6, 10)) / 2);
    EXPECT_EQ(columnGroupMidTimestampNs(end, period, 2, 3, 10),
              (columnTimestampNs(end, period, 7, 10) +
               columnTimestampNs(end, period, 9, 10)) / 2);
    EXPECT_EQ(columnGroupMidTimestampNs(end, period, 3, 3, 10), 0);
    EXPECT_EQ(columnGroupMidTimestampNs(end, period, 0, 11, 10), 0);
    EXPECT_EQ(columnGroupMidTimestampNs(end, period, 0, 0, 10), 0);
}

}  // namespace gz_gpu_ouster_lidar
