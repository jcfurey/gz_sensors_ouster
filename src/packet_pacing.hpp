// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>

namespace gz_gpu_ouster_lidar {

// Spread a batch over 80% of the producer's observed wall-clock scan period.
// The observation follows real-time-factor changes in both directions, while
// the idle tail gives packet consumers time to finish a completed scan before
// the next batch starts. The first batch (or first after pause/reset) uses the
// nominal period. This affects delivery only; packet timestamps remain sim time.
inline std::chrono::nanoseconds packetBatchDrainSpan(
    std::chrono::nanoseconds nominal,
    std::chrono::nanoseconds observed,
    bool have_observation)
{
    auto source = (have_observation &&
                   observed > std::chrono::nanoseconds::zero())
        ? observed : nominal;
    source = std::max(source, std::chrono::nanoseconds(
        std::chrono::microseconds(100)));
    // Divide before multiplying so a valid but very long configured period
    // cannot overflow the signed nanosecond representation.
    const auto ticks = source.count();
    const auto scaled = (ticks / 5) * 4 + ((ticks % 5) * 4) / 5;
    return std::chrono::nanoseconds(std::max<int64_t>(1, scaled));
}

}  // namespace gz_gpu_ouster_lidar
