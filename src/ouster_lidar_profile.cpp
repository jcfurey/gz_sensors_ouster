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

SignalResponseParams signalResponseParams(
    const ouster_sim_core::OusterSignalResponse & response,
    int columns_per_frame, double lidar_hz)
{
    static_assert(kMaxSignalKnots ==
                  ouster_sim_core::OusterSignalResponse::kMaxKnots);
    SignalResponseParams params{};
    if (!response.measured()) return params;
    params.knots = response.knot_count;
    for (int i = 0; i < response.knot_count; ++i) {
        params.range_m[i] = static_cast<float>(response.range_m[i]);
        params.signal_m2[i] = static_cast<float>(response.signal_m2[i]);
    }
    params.saturation = static_cast<float>(response.saturation_signal);
    params.shot_noise = static_cast<float>(response.shot_noise_factor);
    params.relative_noise = static_cast<float>(response.relative_noise);
    params.mode_gain = static_cast<float>(ouster_sim_core::signalModeGain(
        response, columns_per_frame, lidar_hz));
    return params;
}

}  // namespace gz_gpu_ouster_lidar
