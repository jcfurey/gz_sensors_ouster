// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "packet_pacing.hpp"
#include "scan_timing.hpp"

#include <chrono>
#include <vector>

namespace gz_gpu_ouster_lidar {
namespace {
using std::chrono_literals::operator""ms;
using std::chrono_literals::operator""ns;
using std::chrono_literals::operator""us;
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

TEST(PacketPacing, PlaybackRateChangesDeliveryButNotAcquisitionTimestamps)
{
    // A bag player or fast/slow simulation changes producer arrival in wall
    // time. The drain follows that observation, while packet column stamps
    // remain a pure function of the recorded simulation-time scan end.
    constexpr auto nominal = 100ms;
    constexpr int64_t scan_end_ns = 12'000'000'000LL;
    const std::vector<std::chrono::nanoseconds> observed{
        400ms, 200ms, 100ms, 50ms, 25ms};  // 0.25x, 0.5x, 1x, 2x, 4x
    const std::vector<std::chrono::nanoseconds> expected_spans{
        320ms, 160ms, 80ms, 40ms, 20ms};

    std::vector<int64_t> reference_stamps;
    for (int column = 0; column < 4; ++column) {
        reference_stamps.push_back(columnTimestampNs(
            scan_end_ns, nominal.count(), column, 4));
    }

    for (size_t rate_case = 0; rate_case < observed.size(); ++rate_case) {
        EXPECT_EQ(packetBatchDrainSpan(nominal, observed[rate_case], true),
                  expected_spans[rate_case]);
        for (int column = 0; column < 4; ++column) {
            EXPECT_EQ(columnTimestampNs(
                scan_end_ns, nominal.count(), column, 4),
                reference_stamps[static_cast<size_t>(column)]);
        }
    }
}

}  // namespace gz_gpu_ouster_lidar
