// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "packet_pacing.hpp"

#include <chrono>

namespace gz_gpu_ouster_lidar {
namespace {
using namespace std::chrono_literals;
}

TEST(PacketPacing, FirstBatchUsesNominalPeriodWithIdleTail)
{
    EXPECT_EQ(packetBatchDrainSpan(100ms, 0ns, false), 80ms);
}

TEST(PacketPacing, TracksFastAndSlowRealTimeFactors)
{
    EXPECT_EQ(packetBatchDrainSpan(100ms, 50ms, true), 40ms);
    EXPECT_EQ(packetBatchDrainSpan(100ms, 200ms, true), 160ms);
}

TEST(PacketPacing, InvalidObservationFallsBackToNominal)
{
    EXPECT_EQ(packetBatchDrainSpan(100ms, 0ns, true), 80ms);
    EXPECT_EQ(packetBatchDrainSpan(100ms, -1ms, true), 80ms);
}

TEST(PacketPacing, ExtremeFastRateRetainsPositiveSpan)
{
    EXPECT_EQ(packetBatchDrainSpan(100ms, 1us, true), 80us);
}

TEST(PacketPacing, VeryLongPeriodDoesNotOverflow)
{
    const auto longest = std::chrono::nanoseconds::max();
    const auto span = packetBatchDrainSpan(longest, 0ns, false);
    EXPECT_GT(span, 0ns);
    EXPECT_LT(span, longest);
}

}  // namespace gz_gpu_ouster_lidar
