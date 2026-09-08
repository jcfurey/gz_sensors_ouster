// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ouster_sim_core/product_profile.hpp>

namespace gz_gpu_ouster_lidar {

// Source-compatible ROS facade over the canonical simulator-neutral catalog.
using ouster_sim_core::OusterModel;
using ouster_sim_core::OusterGeneration;
using ouster_sim_core::OusterRevision;
using OusterLidarProfile = ouster_sim_core::OusterProductProfile;
using ouster_sim_core::parseOusterModel;
using ouster_sim_core::parseOusterRevision;
using ouster_sim_core::toString;
using ouster_sim_core::ousterModeRangeScale;
using ouster_sim_core::ousterModePrecisionScale;

struct OusterProfileRequest {
    std::string prod_line;
    std::string prod_pn;
    std::string hardware_revision = "auto";
    int firmware_major = 0;
    int firmware_minor = 0;
    int beam_count = 0;
    bool low_data_profile = false;
};

OusterLidarProfile resolveOusterLidarProfile(const OusterProfileRequest & request);

}  // namespace gz_gpu_ouster_lidar
