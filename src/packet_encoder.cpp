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
#include <cmath>
#include <stdexcept>

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
    if (!ros) throw std::invalid_argument("packet publisher is null");
    start(meta, [ros](const auto & packet) { ros->publishLidarPacket(packet); }, lidar_hz);
}

void PacketEncoder::start(const OusterMetadata * meta, PacketSink sink,
                          double lidar_hz, ouster_sim_core::PacketDeliveryMode mode)
{
    if (!meta || !sink || !std::isfinite(lidar_hz) || lidar_hz <= 0.0) {
        throw std::invalid_argument("packet encoder requires metadata, sink and positive rate");
    }
    auto encoder = std::make_unique<ouster_sim_core::OusterPacketEncoder>(meta->core());
    stop();
    meta_ = meta;
    sink_ = std::move(sink);
    encoder_ = std::move(encoder);
    lidar_hz_ = lidar_hz;
    delivery_mode_ = mode;
    column_timestamps_.resize(static_cast<size_t>(meta_->W));
    {
        std::lock_guard lk(drain_mtx_);
        shutdown_.store(false);
        paused_ = false;
        simulation_epoch_ = state_generation_ = 0;
        revolution_ = 0;
        dropped_batches_.store(0);
        drain_ready_ = false;
        drain_pkts_.clear();
    }
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
                               const uint32_t * range, const uint16_t * signal,
                               const uint8_t * refl, const uint16_t * nearir)
{
    if (!encoder_ || shutdown_.load()) return;
    if (!range || !signal || !refl || !nearir || stamp_ns <= 0) {
        throw std::invalid_argument("scan requires channel buffers and a positive timestamp");
    }
    const auto produced_at = std::chrono::steady_clock::now();
    const auto period = periodFromHz(lidar_hz_).count();
    for (int column = 0; column < meta_->W; ++column) {
        column_timestamps_[static_cast<size_t>(column)] = static_cast<uint64_t>(
            columnTimestampNs(stamp_ns, period, column, meta_->W));
    }
    const auto count = static_cast<size_t>(meta_->H) * meta_->W;
    const ouster_sim_core::OusterScanFrameView frame{
        revolution_, std::max(int64_t{0}, stamp_ns - period),
        static_cast<uint32_t>(meta_->W), static_cast<uint16_t>(meta_->H),
        column_timestamps_, {range, count}, {signal, count}, {refl, count}, {nearir, count}};
    encoder_->encode(frame, encode_pkts_);
    ++revolution_;

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
    if (overwrote && (dropped == 1 || (dropped & (dropped - 1)) == 0)) {
        RCLCPP_WARN(kLogger,
            "drainThread dropped %lu backlogged batches",
            static_cast<unsigned long>(dropped));
    }
}

void PacketEncoder::drainThreadFunc()
{
    using ouster_sim_core::PacketPacingPolicy;
    PacketPacingPolicy pacing(periodFromHz(lidar_hz_), delivery_mode_);
    std::vector<ouster_sim_core::EncodedLidarPacket> local_pkts;
    ouster_sensor_msgs::msg::PacketMsg message;
    uint64_t previous_epoch = 0;
    uint64_t previous_generation = 0;

    while (!shutdown_.load(std::memory_order_acquire)) {
        std::unique_lock lk(drain_mtx_);
        drain_cv_.wait(lk, [this] { return (drain_ready_ && !paused_) || shutdown_.load(); });
        if (shutdown_.load()) return;
        drain_ready_ = false;
        local_pkts.swap(drain_pkts_);
        const auto local_epoch = drain_epoch_;
        auto local_generation = state_generation_;
        if (previous_epoch != local_epoch || previous_generation != local_generation) {
            pacing.reset();
        }
        previous_epoch = local_epoch;
        previous_generation = local_generation;
        pacing.resume(std::chrono::steady_clock::now());
        if (local_pkts.empty()) continue;
        pacing.beginFrame(drain_produced_at_, std::chrono::steady_clock::now(), local_pkts.size());
        try {
            while (pacing.hasActiveFrame()) {
                if (shutdown_.load()) return;
                if (local_epoch != simulation_epoch_) {
                    pacing.reset();
                    break;
                }
                if (paused_) {
                    pacing.pause();
                    drain_cv_.wait(lk, [this, local_epoch] {
                        return shutdown_.load() || !paused_ || local_epoch != simulation_epoch_;
                    });
                    pacing.resume(std::chrono::steady_clock::now());
                    local_generation = state_generation_;
                    continue;
                }
                if (local_generation != state_generation_) {
                    // Observe even a pause/resume that finished during publish.
                    pacing.pause();
                    pacing.resume(std::chrono::steady_clock::now());
                    local_generation = state_generation_;
                }
                const auto deadline = pacing.nextDeadline().value();
                if (drain_cv_.wait_until(lk, deadline, [this, local_epoch, local_generation] {
                        return shutdown_.load() || paused_ || local_epoch != simulation_epoch_ ||
                            local_generation != state_generation_;
                    })) continue;

                auto & packet = local_pkts[pacing.nextPacketIndex()];
                message.buf.swap(packet.bytes);
                lk.unlock();
                // The potentially blocking transport owns no simulation mutex.
                sink_(message);
                lk.lock();
                message.buf.swap(packet.bytes);
                pacing.markPacketPublished();
            }
        } catch (const std::exception & e) {
            if (!lk.owns_lock()) lk.lock();
            pacing.reset();
            RCLCPP_ERROR(kLogger, "drainThread publish failed: %s", e.what());
        }
    }
}

}  // namespace gz_gpu_ouster_lidar
