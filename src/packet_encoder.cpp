// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#include "packet_encoder.hpp"
#include "ouster_metadata.hpp"
#include "packet_pacing.hpp"
#include "ros_interface.hpp"
#include "scan_timing.hpp"
#include "gz_gpu_ouster_lidar/sim_time_scheduler.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>

#include <ouster/impl/packet_writer.h>
#include <ouster/lidar_scan.h>
#include <ouster/types.h>

namespace gz_gpu_ouster_lidar {

static const rclcpp::Logger kLogger = lidarLogger();

PacketEncoder::PacketEncoder() = default;

PacketEncoder::~PacketEncoder()
{
    stop();
}

void PacketEncoder::start(const OusterMetadata * meta, RosInterface * ros,
                          double lidar_hz)
{
    meta_ = meta;
    ros_ = ros;
    lidar_hz_ = lidar_hz;
    pkt_buf_.resize(meta_->pw->lidar_packet_size, 0);
    drain_thread_ = std::thread(&PacketEncoder::drainThreadFunc, this);
}

void PacketEncoder::stop()
{
    // Set shutdown_ under drain_mtx_ so the drain thread can't miss it in the
    // window between evaluating its cv_ predicate and blocking in wait() — a
    // lost wakeup there would hang the join below at teardown.
    {
        std::lock_guard<std::mutex> lk(drain_mtx_);
        shutdown_.store(true, std::memory_order_release);
    }
    drain_cv_.notify_all();
    if (drain_thread_.joinable()) {
        drain_thread_.join();
    }
}

void PacketEncoder::setSimulationState(bool paused, uint64_t epoch)
{
    bool changed = false;
    {
        std::lock_guard<std::mutex> lk(drain_mtx_);
        changed = paused_ != paused || simulation_epoch_ != epoch;
        paused_ = paused;
        simulation_epoch_ = epoch;
        if (changed) ++state_generation_;

        // A pending frame from before a rewind must not be decoded beside
        // post-reset packets. The in-flight local batch observes the same
        // epoch change before its next publish and cancels itself.
        if (drain_ready_ && drain_epoch_ != simulation_epoch_) {
            drain_ready_ = false;
        }
    }
    if (changed) drain_cv_.notify_all();
}

void PacketEncoder::encodeScan(int64_t stamp_ns, uint64_t epoch,
                               uint32_t * range, uint16_t * signal,
                               uint8_t * refl, uint16_t * nearir)
{
    if (!meta_ || !meta_->pw || pkt_buf_.empty()) return;
    const auto produced_at = std::chrono::steady_clock::now();

    const int H = meta_->H;
    const int W = meta_->W;
    const int cpp = meta_->cpp;
    auto & pw = *meta_->pw;

    const int n_packets = W / cpp;
    const int64_t scan_period_ns = periodFromHz(lidar_hz_).count();
    // Per-column timestamps use the same (m + 1) rolling-shutter convention
    // as RaycastMirror's pose interpolation. The final measurement therefore
    // coincides exactly with the actual acquisition timestamp.

    // Map raw buffers into Eigen for PacketWriter
    using RangeMatrix = Eigen::Map<ouster::sdk::core::img_t<uint32_t>>;
    using SignalMatrix = Eigen::Map<ouster::sdk::core::img_t<uint16_t>>;
    using ReflMatrix = Eigen::Map<ouster::sdk::core::img_t<uint8_t>>;
    using NirMatrix = Eigen::Map<ouster::sdk::core::img_t<uint16_t>>;

    RangeMatrix  range_mat(range, H, W);
    SignalMatrix signal_mat(signal, H, W);
    ReflMatrix   refl_mat(refl, H, W);
    NirMatrix    nearir_mat(nearir, H, W);

    // encode_pkts_ buffers (and the vector itself) are reused across scans:
    // after the drain swap below it holds the drain thread's previously
    // published packets, whose buf capacity the assign() below reuses —
    // zero allocations in steady state.
    encode_pkts_.resize(static_cast<size_t>(n_packets));

    for (int p = 0; p < n_packets; ++p) {
        std::memset(pkt_buf_.data(), 0, pkt_buf_.size());

        const int col_start = p * cpp;
        pw.set_frame_id(pkt_buf_.data(), frame_id_);

        for (int c_local = 0; c_local < cpp; ++c_local) {
            const int col_global = col_start + c_local;
            uint8_t * col = pw.nth_col(c_local, pkt_buf_.data());
            const int64_t col_ts = columnTimestampNs(
                stamp_ns, scan_period_ns, col_global, W);
            pw.set_col_timestamp(col, static_cast<uint64_t>(col_ts));
            pw.set_col_measurement_id(col, static_cast<uint16_t>(col_global));
            pw.set_col_status(col, 0x01u);
        }

        pw.set_block<uint32_t>(range_mat.data(),  W, ouster::sdk::core::ChanField::RANGE,        pkt_buf_.data());
        pw.set_block<uint16_t>(signal_mat.data(), W, ouster::sdk::core::ChanField::SIGNAL,       pkt_buf_.data());
        pw.set_block<uint8_t> (refl_mat.data(),   W, ouster::sdk::core::ChanField::REFLECTIVITY, pkt_buf_.data());
        pw.set_block<uint16_t>(nearir_mat.data(), W, ouster::sdk::core::ChanField::NEAR_IR,      pkt_buf_.data());

        encode_pkts_[static_cast<size_t>(p)].buf.assign(
            pkt_buf_.begin(), pkt_buf_.end());
    }

    ++frame_id_;

    // ── Wake drain thread ────────────────────────────────────────────────────
    bool overwrote = false;
    uint64_t dropped = 0;
    {
        std::lock_guard<std::mutex> lk(drain_mtx_);
        // A reset may have occurred while this scan was being encoded.
        if (epoch != simulation_epoch_) return;
        overwrote = drain_ready_;
        if (overwrote) {
            dropped = dropped_batches_.fetch_add(
                1, std::memory_order_relaxed) + 1;
        }
        drain_pkts_.swap(encode_pkts_);
        drain_produced_at_ = produced_at;
        drain_epoch_ = epoch;
        drain_ready_ = true;
    }
    drain_cv_.notify_one();
    if (overwrote) {
        RCLCPP_WARN_THROTTLE(kLogger, *ros_->clock(), 5000,
            "packet drain replaced a complete pending scan; total dropped=%lu",
            static_cast<unsigned long>(dropped));
    }
}

void PacketEncoder::drainThreadFunc()
{
    std::vector<ouster_sensor_msgs::msg::PacketMsg> local_pkts;
    std::chrono::steady_clock::time_point local_produced_at{};
    std::chrono::steady_clock::time_point prev_produced_at{};
    uint64_t local_epoch = 0;
    uint64_t previous_epoch = 0;
    uint64_t local_generation = 0;
    uint64_t previous_generation = 0;
    bool have_previous = false;

    while (!shutdown_.load(std::memory_order_acquire)) {
        {
            std::unique_lock<std::mutex> lk(drain_mtx_);
            drain_cv_.wait(lk, [this] {
                return (drain_ready_ && !paused_) ||
                       shutdown_.load(std::memory_order_acquire);
            });
            if (shutdown_.load(std::memory_order_acquire)) break;
            drain_ready_ = false;
            local_pkts.swap(drain_pkts_);
            local_produced_at = drain_produced_at_;
            local_epoch = drain_epoch_;
            local_generation = state_generation_;
        }

        if (local_pkts.empty()) continue;

        // Use absolute deadlines (sleep_until) instead of accumulating
        // sleep_for(spacing) calls. At dense-sensor packet counts the per-
        // packet spacing drops to hundreds of microseconds, where CFS
        // scheduler jitter would round each sleep up and the packets would
        // bunch toward the end of the scan. Anchoring on a fixed t0 lets
        // any individual sleep finish late without pushing the next one.
        //
        // Pace from PRODUCER arrival times rather than drain start times. A
        // busy drain otherwise measures its own backlog and feeds that error
        // into the next scan. The observed interval is used above and below
        // RTF 1, so slow simulation and rate-scaled recording are respected
        // just as fast simulation is. Pause/reset invalidates the observation.
        const auto nominal = periodFromHz(lidar_hz_);
        const bool observation_valid =
            have_previous && local_epoch == previous_epoch &&
            local_generation == previous_generation;
        const auto observed = observation_valid
            ? std::chrono::duration_cast<std::chrono::nanoseconds>(
                local_produced_at - prev_produced_at)
            : std::chrono::nanoseconds::zero();
        const auto span = packetBatchDrainSpan(
            nominal, observed, observation_valid);
        const auto spacing = span /
            static_cast<int64_t>(local_pkts.size());
        prev_produced_at = local_produced_at;
        previous_epoch = local_epoch;
        previous_generation = local_generation;
        have_previous = true;

        try {
            auto t0 = std::chrono::steady_clock::now();
            bool cancelled = false;
            for (size_t i = 0; i < local_pkts.size(); ++i) {
                auto deadline = t0 + spacing * static_cast<int64_t>(i);
                std::unique_lock<std::mutex> lk(drain_mtx_);
                for (;;) {
                    if (shutdown_.load(std::memory_order_acquire)) return;
                    if (local_epoch != simulation_epoch_) {
                        cancelled = true;
                        break;
                    }
                    if (paused_) {
                        drain_cv_.wait(lk, [this, local_epoch] {
                            return shutdown_.load(std::memory_order_acquire) ||
                                   !paused_ ||
                                   local_epoch != simulation_epoch_;
                        });
                        // Resume without a catch-up burst: put the current
                        // packet one normal spacing after unpause and shift
                        // all later absolute deadlines with it.
                        const auto now = std::chrono::steady_clock::now();
                        t0 = now - spacing * static_cast<int64_t>(i) + spacing;
                        deadline = t0 + spacing * static_cast<int64_t>(i);
                        continue;
                    }
                    if (drain_cv_.wait_until(lk, deadline,
                            [this, local_epoch] {
                                return shutdown_.load(std::memory_order_acquire) ||
                                       paused_ ||
                                       local_epoch != simulation_epoch_;
                            })) {
                        continue;
                    }
                    break;
                }
                lk.unlock();
                if (cancelled) break;
                ros_->publishLidarPacket(local_pkts[i]);
            }
        } catch (const std::exception & e) {
            RCLCPP_ERROR(kLogger, "drainThread publish failed: %s", e.what());
        }
    }
}

}  // namespace gz_gpu_ouster_lidar
