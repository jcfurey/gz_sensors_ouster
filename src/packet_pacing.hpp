// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ouster_sim_core/packet_pacing.hpp>
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
    return ouster_sim_core::packetBatchDrainSpan(
        nominal, have_observation ? std::optional{observed} : std::nullopt);
}

}  // namespace gz_gpu_ouster_lidar
