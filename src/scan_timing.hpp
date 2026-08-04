// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>

namespace gz_gpu_ouster_lidar {

// A scan stamped at scan_end_ns covers (scan_end - period, scan_end].  This
// matches the rolling-shutter pose used by RaycastMirror: measurement column
// zero is one column interval after the start and the last column is acquired
// exactly at scan_end.  At startup, clamp the incomplete pre-zero interval so
// packet timestamps can never be negative or lie in the future.
inline int64_t columnTimestampNs(int64_t scan_end_ns,
                                 int64_t scan_period_ns,
                                 int column,
                                 int columns)
{
    if (scan_end_ns < 0 || scan_period_ns <= 0 ||
        column < 0 || column >= columns || columns <= 0) {
        return 0;
    }

    const int64_t scan_start_ns =
        std::max(int64_t{0}, scan_end_ns - scan_period_ns);
    const int64_t span_ns = scan_end_ns - scan_start_ns;
    const int64_t ordinal = static_cast<int64_t>(column) + 1;

    // Split quotient/remainder to avoid overflowing ordinal * span_ns for
    // unusually large simulated epochs.
    const int64_t offset_ns =
        (span_ns / columns) * ordinal +
        ((span_ns % columns) * ordinal) / columns;
    return scan_start_ns + offset_ns;
}

}  // namespace gz_gpu_ouster_lidar
