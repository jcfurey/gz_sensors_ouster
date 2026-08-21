// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "gz_gpu_ouster_lidar/sim_time_scheduler.hpp"

#include <chrono>
#include <limits>
#include <vector>

namespace gz_gpu_ouster_lidar {
namespace {
using std::chrono_literals::operator""ms;
using std::chrono_literals::operator""ns;
}

TEST(SimTimeScheduler, ConvertsRatesSafely)
{
    EXPECT_EQ(periodFromHz(100.0), 10ms);
    EXPECT_EQ(periodFromHz(10.0), 100ms);
    EXPECT_EQ(periodFromHz(0.0), 0ns);
    EXPECT_EQ(periodFromHz(-1.0), 0ns);
    EXPECT_EQ(periodFromHz(std::numeric_limits<double>::infinity()), 0ns);
    EXPECT_EQ(periodFromHz(std::numeric_limits<double>::quiet_NaN()), 0ns);
}

TEST(SimTimeScheduler, LidarGatePreservesAverageRateAcrossPhysicsTicks)
{
    SimTimeGate gate;
    std::vector<std::chrono::nanoseconds> captures;
    for (auto now = 0ms; now <= 40ms; now += 4ms) {
        if (gate.advance(now, 10ms).due) captures.push_back(now);
    }
    const std::vector<std::chrono::nanoseconds> expected{
        0ms, 12ms, 20ms, 32ms, 40ms};
    EXPECT_EQ(captures, expected);
}

TEST(SimTimeScheduler, ImuDeadlinesAreExactAcrossPhysicsTicks)
{
    PeriodicDeadlineScheduler scheduler;
    std::vector<std::chrono::nanoseconds> samples;
    for (auto now = 0ms; now <= 40ms; now += 4ms) {
        const auto batch = scheduler.advance(now, 10ms);
        samples.insert(samples.end(), batch.deadlines.begin(),
                       batch.deadlines.begin() + batch.size);
    }
    const std::vector<std::chrono::nanoseconds> expected{
        0ms, 10ms, 20ms, 30ms, 40ms};
    EXPECT_EQ(samples, expected);
}

TEST(SimTimeScheduler, RewindStartsANewTimeEpochImmediately)
{
    PeriodicDeadlineScheduler scheduler;
    EXPECT_TRUE(scheduler.advance(100ms, 10ms).reset);
    EXPECT_EQ(scheduler.advance(110ms, 10ms).size, 1u);

    const auto rewound = scheduler.advance(20ms, 10ms);
    ASSERT_TRUE(rewound.reset);
    ASSERT_EQ(rewound.size, 1u);
    EXPECT_EQ(rewound.deadlines[0], 20ms);
}

TEST(SimTimeScheduler, LargeJumpBoundsCatchupAndSkipsOldest)
{
    PeriodicDeadlineScheduler scheduler;
    scheduler.advance(0ms, 1ms);
    const auto batch = scheduler.advance(100ms, 1ms);
    EXPECT_EQ(batch.size, PeriodicDeadlineScheduler::kMaxDeadlines);
    EXPECT_EQ(batch.skipped, 68u);
    EXPECT_EQ(batch.deadlines.front(), 69ms);
    EXPECT_EQ(batch.deadlines[batch.size - 1], 100ms);
}

}  // namespace gz_gpu_ouster_lidar
