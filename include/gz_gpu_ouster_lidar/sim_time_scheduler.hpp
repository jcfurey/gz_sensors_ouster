// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0
//
// Allocation-free sim-time cadence helpers.  LiDAR frames cannot be
// synthesized between physics/render ticks, so SimTimeGate preserves their
// average rate while callers stamp the actual capture tick.  IMU samples can
// be interpolated between physics states, so PeriodicDeadlineScheduler emits
// exact sample deadlines.

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace gz_gpu_ouster_lidar {

inline std::chrono::nanoseconds periodFromHz(double hz)
{
    if (!std::isfinite(hz) || hz <= 0.0) {
        return std::chrono::nanoseconds::zero();
    }
    const long double ns = 1.0e9L / static_cast<long double>(hz);
    if (ns >= static_cast<long double>(
            std::chrono::nanoseconds::max().count())) {
        return std::chrono::nanoseconds::max();
    }
    const auto rounded = static_cast<int64_t>(std::llround(ns));
    return std::chrono::nanoseconds(std::max<int64_t>(1, rounded));
}

class SimTimeGate {
public:
    struct Result {
        bool due = false;
        bool reset = false;
        uint64_t skipped = 0;
    };

    Result advance(std::chrono::nanoseconds now,
                   std::chrono::nanoseconds period)
    {
        Result out;
        if (period <= std::chrono::nanoseconds::zero()) return out;

        if (!initialized_ || now < last_observed_) {
            initialized_ = true;
            next_deadline_ = now;
            out.reset = true;
        }
        last_observed_ = now;

        if (now < next_deadline_) return out;
        const auto intervals = (now - next_deadline_) / period;
        out.skipped = static_cast<uint64_t>(intervals);
        next_deadline_ += period * (intervals + 1);
        out.due = true;
        return out;
    }

    void reset() { initialized_ = false; }

private:
    bool initialized_ = false;
    std::chrono::nanoseconds last_observed_{0};
    std::chrono::nanoseconds next_deadline_{0};
};

class PeriodicDeadlineScheduler {
public:
    static constexpr size_t kMaxDeadlines = 32;

    struct Batch {
        std::array<std::chrono::nanoseconds, kMaxDeadlines> deadlines{};
        size_t size = 0;
        uint64_t skipped = 0;
        bool reset = false;
    };

    Batch advance(std::chrono::nanoseconds now,
                  std::chrono::nanoseconds period)
    {
        Batch out;
        if (period <= std::chrono::nanoseconds::zero()) return out;

        if (!initialized_ || now < last_observed_) {
            initialized_ = true;
            next_deadline_ = now;
            out.reset = true;
        }
        last_observed_ = now;

        if (now < next_deadline_) return out;
        const auto total_due_i64 = (now - next_deadline_) / period + 1;
        const auto total_due = static_cast<uint64_t>(total_due_i64);
        if (total_due > kMaxDeadlines) {
            out.skipped = total_due - kMaxDeadlines;
            next_deadline_ += period * static_cast<int64_t>(out.skipped);
        }

        out.size = static_cast<size_t>(
            std::min<uint64_t>(total_due, kMaxDeadlines));
        for (size_t i = 0; i < out.size; ++i) {
            out.deadlines[i] = next_deadline_;
            next_deadline_ += period;
        }
        return out;
    }

    void reset() { initialized_ = false; }

private:
    bool initialized_ = false;
    std::chrono::nanoseconds last_observed_{0};
    std::chrono::nanoseconds next_deadline_{0};
};

}  // namespace gz_gpu_ouster_lidar
