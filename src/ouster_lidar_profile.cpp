// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#include "ouster_lidar_profile.hpp"

namespace gz_gpu_ouster_lidar {

OusterLidarProfile resolveOusterLidarProfile(const OusterProfileRequest & request)
{
    ouster_sim_core::OusterProductProfileRequest core;
    core.product_line = request.prod_line;
    core.product_part_number = request.prod_pn;
    core.hardware_revision = request.hardware_revision;
    core.firmware_major = request.firmware_major;
    core.firmware_minor = request.firmware_minor;
    core.beam_count = request.beam_count;
    core.low_data_profile = request.low_data_profile;
    return ouster_sim_core::resolveOusterProductProfile(core);
}

}  // namespace gz_gpu_ouster_lidar
