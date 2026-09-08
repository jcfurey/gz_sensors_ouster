// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0
//
// Ouster packet encoding + paced publishing: builds the scan's lidar
// packets through the SDK PacketWriter (per-column timestamps, measurement
// ids, channel blocks) on the sim thread, and drains them on a dedicated
// thread with rolling-shutter inter-packet spacing that follows the
// observed wall-clock scan cadence (RTF-aware).

#pragma once

#include "lidar_common.hpp"
#include <ouster_sim_core/packet_encoder.hpp>
#include <ouster_sim_core/packet_pacing.hpp>
#include <functional>

#include <ouster_sensor_msgs/msg/packet_msg.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace gz_gpu_ouster_lidar {

class OusterMetadata;
class RosInterface;

class PacketEncoder {
public:
    PacketEncoder();
    ~PacketEncoder();  // calls stop()

    /// Size the packet buffer and start the drain thread. `meta` and `ros`
    /// must outlive stop().
    void start(const OusterMetadata * meta, RosInterface * ros,
               double lidar_hz);

    using PacketSink = std::function<void(const ouster_sensor_msgs::msg::PacketMsg &)>;
    /// Alternate sink for reusable ROS producers and transport conformance.
    /// The sink runs only on the drain thread and must outlive stop().
    void start(const OusterMetadata * meta, PacketSink sink, double lidar_hz,
               ouster_sim_core::PacketDeliveryMode mode =
                   ouster_sim_core::PacketDeliveryMode::kPaced);

    uint64_t droppedBatches() const { return dropped_batches_.load(); }

    /// Stop and join the drain thread. Idempotent.
    void stop();

    /// Sim thread: publish pause/reset state to the drain thread. No packets
    /// are emitted while paused; a new epoch cancels any pre-reset batch.
    void setSimulationState(bool paused, uint64_t epoch);

    /// Sim thread: build the scan's packets from the channel buffers and
    /// wake the drain thread. Input buffers are not modified.
    void encodeScan(int64_t stamp_ns, uint64_t epoch,
                    const uint32_t * range, const uint16_t * signal,
                    const uint8_t * refl, const uint16_t * nearir);

private:
    void drainThreadFunc();

    const OusterMetadata * meta_ = nullptr;
    PacketSink sink_;
    std::unique_ptr<ouster_sim_core::OusterPacketEncoder> encoder_;
    ouster_sim_core::PacketDeliveryMode delivery_mode_ =
        ouster_sim_core::PacketDeliveryMode::kPaced;
    double lidar_hz_ = 10.0;

    std::vector<uint64_t> column_timestamps_;
    uint64_t revolution_ = 0;

    // encode_pkts_ is the sim-thread staging vector; swapping with
    // drain_pkts_ circulates buffer capacity between the encode and drain
    // sides so steady state allocates nothing per scan.
    std::vector<ouster_sim_core::EncodedLidarPacket> encode_pkts_;
    std::vector<ouster_sim_core::EncodedLidarPacket> drain_pkts_;
    std::chrono::steady_clock::time_point drain_produced_at_{};
    uint64_t drain_epoch_ = 0;
    std::thread drain_thread_;
    std::mutex drain_mtx_;
    std::condition_variable drain_cv_;
    bool drain_ready_ = false;
    bool paused_ = false;
    uint64_t simulation_epoch_ = 0;
    uint64_t state_generation_ = 0;
    std::atomic<uint64_t> dropped_batches_{0};
    std::atomic<bool> shutdown_{false};
};

}  // namespace gz_gpu_ouster_lidar
