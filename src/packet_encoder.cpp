// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#include "packet_encoder.hpp"
#include "ouster_metadata.hpp"
#include "ros_interface.hpp"

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
    shutdown_.store(true, std::memory_order_release);
    drain_cv_.notify_all();
    if (drain_thread_.joinable()) {
        drain_thread_.join();
    }
}

void PacketEncoder::encodeScan(int64_t stamp_ns,
                               uint32_t * range, uint16_t * signal,
                               uint8_t * refl, uint16_t * nearir)
{
    if (!meta_ || !meta_->pw || pkt_buf_.empty()) return;

    const int H = meta_->H;
    const int W = meta_->W;
    const int cpp = meta_->cpp;
    auto & pw = *meta_->pw;

    const int n_packets = W / cpp;
    const int64_t scan_period_ns = static_cast<int64_t>(1e9 / lidar_hz_);
    const int64_t scan_start_ns = std::max(int64_t{0},
                                           stamp_ns - scan_period_ns);

    // Per-column timestamps: round (col * period / W) on the full numerator
    // rather than accumulating col * (period / W). At 10 Hz × 1024 cols the
    // truncated form drops 256 ns per scan and drifts over long bag captures;
    // computing on the full multiply restores the last column to the scan end.

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
            const int64_t col_ts = scan_start_ns +
                (static_cast<int64_t>(col_global) * scan_period_ns) / W;
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
    {
        std::lock_guard<std::mutex> lk(drain_mtx_);
        drain_pkts_.swap(encode_pkts_);
        drain_ready_.store(true, std::memory_order_release);
    }
    drain_cv_.notify_one();
}

void PacketEncoder::drainThreadFunc()
{
    std::vector<ouster_sensor_msgs::msg::PacketMsg> local_pkts;
    std::chrono::steady_clock::time_point prev_batch{};
    bool have_prev_batch = false;

    while (!shutdown_.load(std::memory_order_acquire)) {
        {
            std::unique_lock<std::mutex> lk(drain_mtx_);
            drain_cv_.wait(lk, [this] {
                return drain_ready_.load(std::memory_order_acquire) ||
                       shutdown_.load(std::memory_order_acquire);
            });
            if (shutdown_.load(std::memory_order_acquire)) break;
            drain_ready_.store(false, std::memory_order_release);
            local_pkts.swap(drain_pkts_);
        }

        if (local_pkts.empty()) continue;

        // Use absolute deadlines (sleep_until) instead of accumulating
        // sleep_for(spacing) calls. At dense-sensor packet counts the per-
        // packet spacing drops to hundreds of microseconds, where CFS
        // scheduler jitter would round each sleep up and the packets would
        // bunch toward the end of the scan. Anchoring on a fixed t0 lets
        // any individual sleep finish late without pushing the next one.
        //
        // Pace across the OBSERVED wall-clock scan cadence, not the nominal
        // 1/lidar_hz: scans arrive at lidar_hz in SIM time, so at RTF > 1
        // batches land faster than nominal and spreading over the nominal
        // period would back the drain up indefinitely (dropped frames).
        // min() with nominal keeps RTF < 1 behaviour unchanged (finish
        // early, idle); the 5% margin finishes before the next batch; the
        // 1 ms floor keeps spacing sane after a scheduling hiccup. Packet
        // timestamps are sim time regardless — only wall spacing adapts.
        const auto t0 = std::chrono::steady_clock::now();
        const auto nominal = std::chrono::nanoseconds(
            static_cast<int64_t>(1e9 / lidar_hz_));
        auto period = nominal;
        if (have_prev_batch) {
            const auto observed =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    t0 - prev_batch);
            period = std::min(nominal, observed);
        }
        prev_batch = t0;
        have_prev_batch = true;
        period = std::max(std::chrono::nanoseconds(1'000'000),
                          period * 95 / 100);
        const auto spacing = period / static_cast<int64_t>(local_pkts.size());

        try {
            for (size_t i = 0; i < local_pkts.size(); ++i) {
                if (shutdown_.load(std::memory_order_acquire)) return;
                if (i > 0) {
                    std::this_thread::sleep_until(
                        t0 + spacing * static_cast<int64_t>(i));
                }
                ros_->publishLidarPacket(local_pkts[i]);
            }
        } catch (const std::exception & e) {
            RCLCPP_ERROR(kLogger, "drainThread publish failed: %s", e.what());
        }
    }
}

}  // namespace gz_gpu_ouster_lidar
