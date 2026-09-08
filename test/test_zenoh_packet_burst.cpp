// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#include "ros_interface.hpp"
#include <gtest/gtest.h>
#include <atomic>
#include <csignal>
#include <cstdlib>
#include <future>
#include <unistd.h>
#include <chrono>
#include <mutex>
#include <thread>
#include <vector>

namespace gz_gpu_ouster_lidar {
namespace {
using namespace std::chrono_literals;

template <class Predicate> bool waitFor(Predicate predicate) {
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (std::chrono::steady_clock::now() < deadline) {
        if (predicate()) return true;
        std::this_thread::sleep_for(1ms);
    }
    return false;
}

TEST(ZenohPacketBurst, ProductionPublisherPreservesPacketsAndLatchedMetadata)
{
    // RosInterface uses the default context. A separate reader context forces
    // the real constrained TCP link instead of Zenoh in-session delivery.
    RosInterface publisher;
    RosInterfaceConfig config;
    config.sensor_name = "/ouster_conformance";
    config.metadata_str = "{\"fixture\":1}";
    publisher.init(config, {});
    publisher.publishMetadataIfNeeded(0ns);

    auto context = std::make_shared<rclcpp::Context>();
    context->init(0, nullptr);
    auto reader = std::make_shared<rclcpp::Node>(
        "ouster_conformance_reader", rclcpp::NodeOptions().context(context));
    std::mutex mutex;
    std::vector<unsigned> received;
    std::atomic<bool> metadata_received{false};
    auto metadata_sub = reader->create_subscription<std_msgs::msg::String>(
        config.sensor_name + "/metadata", rclcpp::QoS(1).reliable().transient_local(),
        [&](const std_msgs::msg::String & value) { metadata_received = value.data == config.metadata_str; });
    auto subscription = reader->create_subscription<ouster_sensor_msgs::msg::PacketMsg>(
        config.sensor_name + "/lidar_packets", rclcpp::QoS(rclcpp::KeepAll()).reliable(),
        [&](const ouster_sensor_msgs::msg::PacketMsg & value) {
            std::lock_guard lock(mutex);
            received.push_back(value.buf.size() == 61440 ?
                value.buf[0] * 256u + value.buf[1] : 999999u);
        });
    rclcpp::ExecutorOptions options;
    options.context = context;
    rclcpp::executors::SingleThreadedExecutor executor(options);
    executor.add_node(reader);
    std::jthread spinner([&](std::stop_token stop) {
        while (!stop.stop_requested()) {
            executor.spin_some();
            std::this_thread::sleep_for(100us);
        }
    });
    ASSERT_TRUE(waitFor([&] {
        return reader->count_publishers(config.sensor_name + "/lidar_packets") == 1;
    }));
    const auto endpoints = reader->get_publishers_info_by_topic(config.sensor_name + "/lidar_packets");
    ASSERT_EQ(endpoints.size(), 1u);
    const auto qos = endpoints.front().qos_profile().get_rmw_qos_profile();
    EXPECT_EQ(qos.reliability, RMW_QOS_POLICY_RELIABILITY_RELIABLE);
    EXPECT_EQ(qos.history, RMW_QOS_POLICY_HISTORY_KEEP_ALL);
    EXPECT_EQ(qos.durability, RMW_QOS_POLICY_DURABILITY_VOLATILE);
    ASSERT_TRUE(waitFor([&] { return metadata_received.load(); }));

    ouster_sensor_msgs::msg::PacketMsg packet;
    packet.buf.assign(61440, 0x41);
    for (unsigned i = 0; i < 1024; ++i) {
        packet.buf[0] = static_cast<uint8_t>(i / 256);
        packet.buf[1] = static_cast<uint8_t>(i % 256);
        publisher.publishLidarPacket(packet);
    }
    EXPECT_TRUE(waitFor([&] {
        std::lock_guard lock(mutex);
        return received.size() == 1024;
    }));
    spinner.request_stop();
    spinner.join();
    publisher.shutdown();
    ASSERT_EQ(received.size(), 1024u);
    for (unsigned i = 0; i < received.size(); ++i) EXPECT_EQ(received[i], i);
}

TEST(ZenohPacketBurst, BlockedTransportDoesNotHoldSimulationPublicationLock)
{
    const char * router_pid = std::getenv("OUSTER_TEST_ROUTER_PID");
    ASSERT_NE(router_pid, nullptr) << "Use the private-router test runner";
    const auto pid = static_cast<pid_t>(std::stoi(router_pid));
    ASSERT_GT(pid, 1);
    RosInterface publisher;
    RosInterfaceConfig config;
    config.sensor_name = "/ouster_blocked_transport";
    config.metadata_str = "{}";
    config.imu_enabled = true;
    publisher.init(config, {});
    auto context = std::make_shared<rclcpp::Context>();
    context->init(0, nullptr);
    auto reader = std::make_shared<rclcpp::Node>(
        "ouster_blocked_reader", rclcpp::NodeOptions().context(context));
    auto subscription = reader->create_subscription<ouster_sensor_msgs::msg::PacketMsg>(
        config.sensor_name + "/lidar_packets", rclcpp::QoS(rclcpp::KeepAll()).reliable(),
        [](const ouster_sensor_msgs::msg::PacketMsg &) {});
    ASSERT_TRUE(waitFor([&] {
        return reader->count_publishers(config.sensor_name + "/lidar_packets") == 1;
    }));

    struct RouterPause {
        pid_t pid;
        ~RouterPause() { resume(); }
        void resume() { if (pid > 0) { kill(pid, SIGCONT); pid = 0; } }
    } pause{pid};
    ASSERT_EQ(kill(pid, SIGSTOP), 0);
    std::atomic<unsigned> published{0};
    std::atomic<bool> finished{false};
    std::jthread drain([&] {
        ouster_sensor_msgs::msg::PacketMsg packet;
        packet.buf.assign(61440, 0x41);
        for (unsigned i = 0; i < 1024; ++i) {
            publisher.publishLidarPacket(packet);
            ++published;
        }
        finished = true;
    });
    // Fill the stopped router's socket until a publish demonstrably stalls.
    unsigned previous = 0;
    bool blocked = false;
    for (int i = 0; i < 10 && !finished; ++i) {
        std::this_thread::sleep_for(100ms);
        const auto current = published.load();
        if (current > 0 && current == previous && !finished) {
            blocked = true;
            break;
        }
        previous = current;
    }
    auto simulation = std::async(std::launch::async, [&] {
        publisher.publishMetadataIfNeeded(0ns);
        publisher.publishImuMsg(sensor_msgs::msg::Imu{});
    });
    const auto progress = simulation.wait_for(500ms);
    // Release on both success and failure before joining blocked publishers.
    pause.resume();
    drain.join();
    simulation.get();
    EXPECT_TRUE(blocked) << "The constrained link did not create backpressure";
    EXPECT_EQ(progress, std::future_status::ready);
}

}  // namespace
}  // namespace gz_gpu_ouster_lidar
