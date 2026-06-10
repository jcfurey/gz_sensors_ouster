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

#include <ouster_sensor_msgs/msg/packet_msg.hpp>

#include <atomic>
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

    /// Stop and join the drain thread. Idempotent.
    void stop();

    /// Sim thread: build the scan's packets from the channel buffers and
    /// wake the drain thread. (Pointers are non-const to match the SDK
    /// PacketWriter block API; the buffers are not modified.)
    void encodeScan(int64_t stamp_ns,
                    uint32_t * range, uint16_t * signal,
                    uint8_t * refl, uint16_t * nearir);

private:
    void drainThreadFunc();

    const OusterMetadata * meta_ = nullptr;
    RosInterface * ros_ = nullptr;
    double lidar_hz_ = 10.0;

    std::vector<uint8_t> pkt_buf_;
    uint32_t frame_id_ = 0;

    // encode_pkts_ is the sim-thread staging vector; swapping with
    // drain_pkts_ circulates buffer capacity between the encode and drain
    // sides so steady state allocates nothing per scan.
    std::vector<ouster_sensor_msgs::msg::PacketMsg> encode_pkts_;
    std::vector<ouster_sensor_msgs::msg::PacketMsg> drain_pkts_;
    std::thread drain_thread_;
    std::mutex drain_mtx_;
    std::condition_variable drain_cv_;
    std::atomic<bool> drain_ready_{false};
    std::atomic<bool> shutdown_{false};
};

}  // namespace gz_gpu_ouster_lidar
