// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0
//
// Smoke tests for the plugin's class lifecycle. Does NOT spin a Gazebo
// world — ISystemConfigure/PostUpdate touch types that require a live
// EventManager + EntityComponentManager which we can't trivially mock.
// The intent is:
//
//   1. Construct a default-initialised plugin and immediately destruct it.
//      This catches member-init regressions where the destructor walks
//      pointers/threads that Configure() was supposed to set up.
//
//   2. Repeat the construct/destruct cycle many times to exercise any
//      state that survives instances (static initialisers, plugin
//      registration tables).
//
// If Gazebo's gz-sim8 schema changes such that the plugin's vtable can't
// link, this test will fail to *build*, which is itself useful coverage.

#include <gtest/gtest.h>

#include <rclcpp/rclcpp.hpp>

#include "gz_gpu_ouster_lidar/gz_gpu_ouster_lidar_system.hpp"

namespace gz_gpu_ouster_lidar {

// The plugin default-constructs an rclcpp::executors::SingleThreadedExecutor
// member, which provisions its wait-set guard condition off the default
// rclcpp context — that requires rclcpp::init(). Configure once for the
// suite and shut down on tear-down so each TEST gets a live context without
// repeatedly initialising rcl.
class LifecycleEnv : public ::testing::Environment
{
public:
    void SetUp() override
    {
        if (!rclcpp::ok()) {
            rclcpp::init(0, nullptr);
        }
    }
    void TearDown() override
    {
        if (rclcpp::ok()) {
            rclcpp::shutdown();
        }
    }
};
// gtest takes ownership; raw new is the documented idiom.
::testing::Environment * const kLifecycleEnv =
    ::testing::AddGlobalTestEnvironment(new LifecycleEnv);

TEST(Lifecycle, ConstructAndDestructWithoutConfigure)
{
    // All members are POD or default-constructible; the destructor must
    // tolerate the never-Configured case (null gpu_rays_, unjoinable
    // threads, never-spun executor).
    GzGpuOusterLidarSystem plugin;
    SUCCEED();  // reaching here without abort/crash is the assertion
}

TEST(Lifecycle, RepeatedConstructDestruct)
{
    // Catches accumulation bugs in any static state (e.g. plugin
    // registration tables) that survive instances.
    for (int i = 0; i < 16; ++i) {
        GzGpuOusterLidarSystem plugin;
        (void)plugin;
    }
    SUCCEED();
}

}  // namespace gz_gpu_ouster_lidar
