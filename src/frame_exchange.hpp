// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0
//
// Raw scan frame triple-buffer between a producer (the panel rig on the
// render thread, or the raycast worker) and the consumer (PostUpdate on the
// sim thread):
//
//   producer scratch ──publish()──► slot ──take()──► consumer scratch
//
// publish()/take() are O(1) vector swaps under a short lock, so after
// warmup all three vectors hold capacity matching the largest seen frame
// and no per-frame allocation or long lock-hold happens. The producer
// never blocks on the consumer: an unconsumed frame is overwritten and
// counted (callers surface the drop in logs).

#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>

namespace gz_gpu_ouster_lidar {

struct FrameMetadata {
    int64_t scan_end_ns = 0;
    uint64_t epoch = 0;
};

class FrameExchange {
public:
    /// Producer: swap `buf` into the slot (`buf` receives the previous
    /// slot's storage back, preserving capacity for the next frame).
    /// Returns true when an unconsumed frame was overwritten (a drop).
    bool publish(std::vector<float> & buf, int n_floats,
                 const FrameMetadata & metadata)
    {
        std::lock_guard<std::mutex> lk(mtx_);
        const bool overwrote = ready_;
        if (overwrote) {
            dropped_.fetch_add(1, std::memory_order_relaxed);
        }
        buf.swap(slot_);
        n_ = n_floats;
        metadata_ = metadata;
        ready_ = true;
        return overwrote;
    }

    /// Consumer: if a frame is pending, swap it into `out` and return true.
    bool take(std::vector<float> & out, int & n_floats,
              FrameMetadata & metadata)
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (!ready_) return false;
        ready_ = false;
        out.swap(slot_);
        n_floats = n_;
        metadata = metadata_;
        return true;
    }

    /// Cumulative count of frames overwritten before consumption.
    uint64_t dropped() const
    {
        return dropped_.load(std::memory_order_relaxed);
    }

private:
    std::mutex mtx_;
    std::vector<float> slot_;
    int n_ = 0;
    FrameMetadata metadata_;
    bool ready_ = false;
    std::atomic<uint64_t> dropped_{0};
};

/// Typed handoff for a raycast frame after channel synthesis. Keeping these
/// arrays separate avoids alignment/aliasing tricks and lets each vector's
/// storage circulate between worker, slot, and sim thread without copies.
class ProcessedFrameExchange {
public:
    bool publish(std::vector<uint32_t> & range,
                 std::vector<uint16_t> & signal,
                 std::vector<uint8_t> & reflectivity,
                 std::vector<uint16_t> & nearir,
                 int n_pixels,
                 const FrameMetadata & metadata)
    {
        std::lock_guard<std::mutex> lk(mtx_);
        const bool overwrote = ready_;
        if (overwrote) {
            dropped_.fetch_add(1, std::memory_order_relaxed);
        }
        range.swap(range_);
        signal.swap(signal_);
        reflectivity.swap(reflectivity_);
        nearir.swap(nearir_);
        n_ = n_pixels;
        metadata_ = metadata;
        ready_ = true;
        return overwrote;
    }

    bool take(std::vector<uint32_t> & range,
              std::vector<uint16_t> & signal,
              std::vector<uint8_t> & reflectivity,
              std::vector<uint16_t> & nearir,
              int & n_pixels,
              FrameMetadata & metadata)
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (!ready_) return false;
        ready_ = false;
        range.swap(range_);
        signal.swap(signal_);
        reflectivity.swap(reflectivity_);
        nearir.swap(nearir_);
        n_pixels = n_;
        metadata = metadata_;
        return true;
    }

    uint64_t dropped() const
    {
        return dropped_.load(std::memory_order_relaxed);
    }

private:
    std::mutex mtx_;
    std::vector<uint32_t> range_;
    std::vector<uint16_t> signal_;
    std::vector<uint8_t> reflectivity_;
    std::vector<uint16_t> nearir_;
    int n_ = 0;
    FrameMetadata metadata_;
    bool ready_ = false;
    std::atomic<uint64_t> dropped_{0};
};

}  // namespace gz_gpu_ouster_lidar
