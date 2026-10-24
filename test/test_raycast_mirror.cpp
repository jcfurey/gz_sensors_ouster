// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0
//
// RaycastMirror on the CPU backend against a hand-stepped ECM: a wall driven
// toward the sensor while scans sweep it, to check that each column group
// sees it where it was when that group was acquired.

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <gz/math/Pose3.hh>
#include <gz/math/Vector3.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/components/Geometry.hh>
#include <gz/sim/components/Link.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/ParentEntity.hh>
#include <gz/sim/components/Pose.hh>
#include <gz/sim/components/Static.hh>
#include <gz/sim/components/Visual.hh>
#include <gz/sim/components/World.hh>
#include <sdf/Box.hh>
#include <sdf/Geometry.hh>

#include "frame_exchange.hpp"
#include "gz_gpu_ouster_lidar/ray_processor.hpp"
#include "raycast_mirror.hpp"

namespace gz_gpu_ouster_lidar {
namespace {

namespace components = ::gz::sim::components;

// gz-sim clears component change flags after every step; the test steps by
// hand, so it clears them itself.
class SteppedEcm : public ::gz::sim::EntityComponentManager {
public:
    void endStep() { SetAllComponentsUnchanged(); }
};

constexpr int kW = 64;
constexpr int kGroups = 16;
constexpr int64_t kPeriodMs = 100;     // 10 Hz
constexpr double kFront0 = 10.0;       // m, wall face at t = 0
constexpr double kSpeed = 5.0;         // m/s toward the sensor
constexpr double kHalfThickness = 0.1;

double front(int64_t t_ms) { return kFront0 - kSpeed * 1e-3 * static_cast<double>(t_ms); }

// The first tick at or after the middle of column group g's timestamps, for
// the scan ending at scan_end_ms (column m is stamped
// scan_end - T + (m + 1) T / W, and a group holds W / kGroups columns).
int64_t groupCaptureMs(int64_t scan_end_ms, int g, int groups)
{
    const int64_t period_ns = kPeriodMs * 1000000;
    const auto stamp = [&](int m) {
        return scan_end_ms * 1000000 - period_ns + (m + 1) * period_ns / kW;
    };
    const int first = g * kW / groups;
    const int last = (g + 1) * kW / groups - 1;
    const int64_t mid = (stamp(first) + stamp(last)) / 2;
    return (mid + 999999) / 1000000;
}

::gz::sim::Entity addBoxModel(SteppedEcm & ecm, ::gz::sim::Entity world,
                              const std::string & name, bool is_static,
                              const ::gz::math::Pose3d & pose,
                              const ::gz::math::Vector3d & size)
{
    const auto model = ecm.CreateEntity();
    ecm.CreateComponent(model, components::Model());
    ecm.CreateComponent(model, components::Name(name));
    ecm.CreateComponent(model, components::ParentEntity(world));
    ecm.CreateComponent(model, components::Pose(pose));
    ecm.CreateComponent(model, components::Static(is_static));
    const auto link = ecm.CreateEntity();
    ecm.CreateComponent(link, components::Link());
    ecm.CreateComponent(link, components::Name("link"));
    ecm.CreateComponent(link, components::ParentEntity(model));
    ecm.CreateComponent(link, components::Pose(::gz::math::Pose3d::Zero));
    const auto visual = ecm.CreateEntity();
    ::sdf::Box box;
    box.SetSize(size);
    ::sdf::Geometry geometry;
    geometry.SetType(::sdf::GeometryType::BOX);
    geometry.SetBoxShape(box);
    ecm.CreateComponent(visual, components::Visual());
    ecm.CreateComponent(visual, components::Name("visual"));
    ecm.CreateComponent(visual, components::ParentEntity(link));
    ecm.CreateComponent(visual, components::Pose(::gz::math::Pose3d::Zero));
    ecm.CreateComponent(visual, components::Geometry(geometry));
    return model;
}

class RaycastMirrorSweep : public ::testing::Test {
protected:
    void SetUp() override
    {
        ::setenv("GZ_OUSTER_BACKEND", "cpu", 1);
        proc_ = std::make_unique<RayProcessor>(123u);
        world_ = ecm_.CreateEntity();
        ecm_.CreateComponent(world_, components::World());
        ecm_.CreateComponent(world_, components::Name("world"));
        // A wide wall ahead, straddling the scan's seam (columns 0 and W-1),
        // so the scan meets it both first and last.
        wall_ = addBoxModel(ecm_, world_, "wall", false, wallPose(0),
                            {2 * kHalfThickness, 100.0, 4.0});
        // Static posts behind the sensor: enough instances for top-level BVHs.
        for (int i = 0; i < 8; ++i) {
            addBoxModel(ecm_, world_, "post" + std::to_string(i), true,
                        {-30.0, -14.0 + 4.0 * i, 0.0, 0.0, 0.0, 0.0},
                        {0.5, 0.5, 4.0});
        }
        ecm_.endStep();

        pp_.H = 1;
        pp_.W = kW;
        pp_.base_signal = 800.0f;
        pp_.base_reflectivity = 50.0f;
        pp_.max_range = 120.0f;
        pp_.min_range = 0.3f;
    }

    void TearDown() override { mirror_.stop(); }

    static ::gz::math::Pose3d wallPose(int64_t t_ms)
    {
        return {front(t_ms) + kHalfThickness, 0.0, 0.0, 0.0, 0.0, 0.0};
    }

    void start(int samples, bool sensor_carries_wall)
    {
        RaycastMirror::Params params;
        params.H = 1;
        params.W = kW;
        params.lidar_hz = 1000.0 / static_cast<double>(kPeriodMs);
        params.object_motion_samples = samples;
        params.sensor_model = sensor_carries_wall ? wall_ : ::gz::sim::kNullEntity;
        params.beam_alt_f = &beam_deg_;
        params.beam_az_f = &beam_deg_;
        mirror_.start(params, proc_.get(), &exchange_);
    }

    // One simulation step to t_ms, with the wall where it is at t_ms.
    void step(int64_t t_ms)
    {
        ecm_.SetComponentData<components::Pose>(wall_, wallPose(t_ms));
        ::gz::sim::UpdateInfo info;
        info.simTime = std::chrono::milliseconds(t_ms);
        info.dt = std::chrono::milliseconds(1);
        mirror_.postUpdate(info, ecm_, ::gz::math::Pose3d::Zero, 0, pp_);
        ecm_.endStep();
    }

    void steps(int64_t from_ms, int64_t to_ms)
    {
        for (int64_t t = from_ms; t <= to_ms; ++t) step(t);
    }

    // Step to t_ms, where a scan triggers, and return its ranges (mm).
    std::vector<uint32_t> scanAt(int64_t t_ms)
    {
        step(t_ms);
        std::vector<uint32_t> range;
        std::vector<uint16_t> signal, nearir;
        std::vector<uint8_t> reflectivity;
        int n = 0;
        FrameMetadata metadata;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (!exchange_.take(range, signal, reflectivity, nearir, n, metadata)) {
            if (std::chrono::steady_clock::now() > deadline) {
                ADD_FAILURE() << "no scan at " << t_ms << " ms";
                return {};
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        EXPECT_EQ(n, kW);
        EXPECT_EQ(metadata.scan_end_ns, t_ms * 1000000);
        return range;
    }

    // Columns that `scan` should see the wall with its face at
    // front(when(m)), given `still`, a scan of the wall at front(0). Returns
    // how many columns were checked.
    template <typename When>
    int expectWall(const std::vector<uint32_t> & scan,
                   const std::vector<uint32_t> & still, When when)
    {
        if (scan.size() != static_cast<size_t>(kW) ||
            still.size() != static_cast<size_t>(kW)) {
            ADD_FAILURE() << "missing scan";
            return 0;
        }
        int checked = 0;
        for (int m = 0; m < kW; ++m) {
            // Within ~56 deg of the wall's normal (range from the still scan,
            // which is kFront0 / cos(azimuth)): clear of its far edges.
            if (still[m] == 0 || still[m] > 1800.0 * kFront0) continue;
            const double expected = still[m] * front(when(m)) / kFront0;
            EXPECT_NEAR(scan[m], expected, 2.5) << "column " << m;
            ++checked;
        }
        return checked;
    }

    SteppedEcm ecm_;
    ::gz::sim::Entity world_ = ::gz::sim::kNullEntity;
    ::gz::sim::Entity wall_ = ::gz::sim::kNullEntity;
    std::vector<float> beam_deg_{0.0f};
    RayProcessParams pp_{};
    std::unique_ptr<RayProcessor> proc_;
    ProcessedFrameExchange exchange_;
    RaycastMirror mirror_{"test"};
};

TEST_F(RaycastMirrorSweep, EachColumnGroupSeesTheWallWhereItWasWhenSwept)
{
    start(kGroups, false);
    // The first trigger starts the cadence: one capture, at that tick.
    const auto still = scanAt(0);
    steps(1, kPeriodMs - 1);
    const auto swept = scanAt(kPeriodMs);
    EXPECT_GT(expectWall(swept, still, [](int m) {
        return groupCaptureMs(kPeriodMs, m * kGroups / kW, kGroups);
    }), 20);
    // The first and last columns, either side of straight ahead, are captured
    // 94 ms apart: 0.47 m of travel, where one capture would put the last
    // column 46 mm further (5.6 deg off the normal).
    ASSERT_GT(swept[0], 0u);
    ASSERT_GT(swept[kW - 1], 0u);
    EXPECT_GT(static_cast<double>(swept[0]) - swept[kW - 1], 300.0);
}

TEST_F(RaycastMirrorSweep, OneSampleCapturesMovingObjectsMidScan)
{
    start(1, false);
    const auto still = scanAt(0);
    steps(1, kPeriodMs - 1);
    EXPECT_GT(expectWall(scanAt(kPeriodMs), still, [](int) {
        return groupCaptureMs(kPeriodMs, 0, 1);
    }), 20);
}

TEST_F(RaycastMirrorSweep, TheSensorsOwnModelStaysWithTheSensorPose)
{
    // Without motion_distortion the sensor casts from its pose at the
    // trigger, so a model carrying it is captured there too.
    start(kGroups, true);
    const auto still = scanAt(0);
    steps(1, kPeriodMs - 1);
    EXPECT_GT(expectWall(scanAt(kPeriodMs), still, [](int) { return kPeriodMs; }), 20);
}

TEST_F(RaycastMirrorSweep, ARewindRestartsTheSweepFromTheNewTime)
{
    start(kGroups, false);
    const auto still = scanAt(0);
    steps(1, kPeriodMs - 1);
    scanAt(kPeriodMs);
    // Halfway through the next scan, rewind to 20 ms (a world reset): the
    // cadence restarts there with a snapshot, and captures taken before the
    // rewind are dropped.
    steps(kPeriodMs + 1, kPeriodMs + 50);
    const int64_t rewound = 20;
    EXPECT_GT(expectWall(scanAt(rewound), still, [](int) { return rewound; }), 20);
    steps(rewound + 1, rewound + kPeriodMs - 1);
    EXPECT_GT(expectWall(scanAt(rewound + kPeriodMs), still, [](int m) {
        return groupCaptureMs(rewound + kPeriodMs, m * kGroups / kW, kGroups);
    }), 20);
}

}  // namespace
}  // namespace gz_gpu_ouster_lidar
