// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0
//
// FrameExchange: the producer/consumer scan-frame triple buffer shared by
// the panel rig and the raycast worker. Covers handoff semantics, capacity
// circulation, overwrite (drop) accounting and cross-thread operation.

#include <gtest/gtest.h>
#include <atomic>
#include <thread>
#include <vector>

#include "frame_exchange.hpp"

namespace gz_gpu_ouster_lidar {

TEST(FrameExchange, EmptyTakeReturnsFalse)
{
    FrameExchange ex;
    std::vector<float> out;
    int n = -1;
    EXPECT_FALSE(ex.take(out, n));
    EXPECT_EQ(ex.dropped(), 0u);
}

TEST(FrameExchange, PublishThenTakeHandsOverTheFrame)
{
    FrameExchange ex;
    std::vector<float> in = {1.0f, 2.0f, 3.0f};
    EXPECT_FALSE(ex.publish(in, 3));  // nothing pending → no overwrite

    std::vector<float> out;
    int n = 0;
    ASSERT_TRUE(ex.take(out, n));
    EXPECT_EQ(n, 3);
    ASSERT_EQ(out.size(), 3u);
    EXPECT_EQ(out[0], 1.0f);
    EXPECT_EQ(out[2], 3.0f);

    // Slot consumed: the next take is empty.
    EXPECT_FALSE(ex.take(out, n));
}

TEST(FrameExchange, OverwriteCountsAsDrop)
{
    FrameExchange ex;
    std::vector<float> a = {1.0f};
    std::vector<float> b = {2.0f};
    EXPECT_FALSE(ex.publish(a, 1));
    EXPECT_TRUE(ex.publish(b, 1));  // unconsumed frame overwritten
    EXPECT_EQ(ex.dropped(), 1u);

    std::vector<float> out;
    int n = 0;
    ASSERT_TRUE(ex.take(out, n));
    EXPECT_EQ(out[0], 2.0f);  // consumer sees the newest frame
}

TEST(FrameExchange, CapacityCirculatesThroughSwaps)
{
    FrameExchange ex;
    std::vector<float> buf(1024, 7.0f);
    const float * original_storage = buf.data();
    ex.publish(buf, 1024);

    std::vector<float> out;
    int n = 0;
    ASSERT_TRUE(ex.take(out, n));
    // The producer's original storage ends up at the consumer untouched —
    // swaps, not copies.
    EXPECT_EQ(out.data(), original_storage);
    EXPECT_EQ(out[123], 7.0f);
}

TEST(FrameExchange, CrossThreadStress)
{
    FrameExchange ex;
    constexpr int kFrames = 2000;
    std::atomic<bool> done{false};

    std::thread producer([&] {
        std::vector<float> buf;
        for (int i = 1; i <= kFrames; ++i) {
            buf.assign(4, static_cast<float>(i));
            ex.publish(buf, 4);
        }
        done = true;
    });

    std::vector<float> out;
    int n = 0;
    float last = 0.0f;
    uint64_t taken = 0;
    for (;;) {
        if (ex.take(out, n)) {
            ASSERT_EQ(n, 4);
            // Frames arrive in order (newest-wins): monotonically increasing.
            EXPECT_GE(out[0], last);
            last = out[0];
            ++taken;
        } else if (done.load()) {
            // Producer finished and the slot is empty — fully drained.
            break;
        } else {
            std::this_thread::yield();
        }
    }
    producer.join();

    // Conservation: every produced frame was either taken or dropped.
    EXPECT_EQ(taken + ex.dropped(), static_cast<uint64_t>(kFrames));
    EXPECT_EQ(last, static_cast<float>(kFrames));
}

}  // namespace gz_gpu_ouster_lidar
