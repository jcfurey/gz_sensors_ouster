// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0
//
// Ouster product physics catalog.  Product identity is deliberately kept
// separate from packet metadata: prod_line identifies the optical model,
// while prod_pn (or an explicit SDF override) identifies the hardware
// revision.  Firmware can change packet capabilities and, for Gen 1, the
// calibrated detection performance without changing either field.

#pragma once

#include <string>

namespace gz_gpu_ouster_lidar {

enum class OusterModel {
    Unknown,
    OS0,
    OS1,
    OS2,
    OSDome,
    OS1Max,
};

enum class OusterGeneration {
    Unknown,
    Gen1,  // original OS1 / L1
    Gen2,  // Rev C/D/05/06 family (L2/L2X)
    Gen3,  // Rev7 / L3
    Gen4,  // Rev8
};

enum class OusterRevision {
    Auto,
    Unknown,
    Gen1,
    RevC,
    RevD,
    Rev05,
    Rev06,
    Rev062,
    Rev07,
    Rev071,
    Rev08,
};

struct OusterLidarProfile {
    std::string id;
    OusterModel model = OusterModel::Unknown;
    OusterGeneration generation = OusterGeneration::Unknown;
    OusterRevision revision = OusterRevision::Unknown;
    bool supported = false;
    bool revision_inferred = false;
    bool fallback_revision = false;

    // Optical detection calibration at 100 klx.  The D90 points are vendor
    // specifications.  D50 is populated where historical datasheets publish
    // it; otherwise the common physics path uses a documented smooth-rolloff
    // approximation anchored exactly at D90.
    double detection_range_10_d90_m = 0.0;
    double detection_range_80_d90_m = 0.0;
    double detection_range_10_d50_m = 0.0;
    double detection_range_80_d50_m = 0.0;
    double representable_range_m = 120.0;
    double minimum_range_m = 0.3;

    // Datasheet precision envelope and systematic accuracy.  The precision
    // envelope supplies the default range-noise ramp; an explicit SDF noise
    // value remains an override. Accuracy is retained separately because it
    // is a systematic bound, not random per-sample noise.
    double precision_min_std_m = 0.003;
    double precision_max_std_m = 0.015;
    double lambertian_accuracy_m = 0.03;
    double retroreflector_accuracy_m = 0.10;

    double range_resolution_m = 0.001;
    double beam_diameter_m = 0.0;
    double beam_divergence_fwhm_deg = 0.0;
    double false_positive_rate = 1.0e-4;
    int max_returns = 1;

    // Vendor detection ranges are measured at this column rate.  Range is
    // adjusted by the documented 15-20% per halving of gathered points.
    double reference_columns_per_second = 10240.0;
};

struct OusterProfileRequest {
    std::string prod_line;
    std::string prod_pn;
    std::string hardware_revision = "auto";
    int firmware_major = 0;
    int firmware_minor = 0;
    int beam_count = 0;
    bool low_data_profile = false;
};

OusterModel parseOusterModel(const std::string & prod_line);
OusterRevision parseOusterRevision(const std::string & value);
const char * toString(OusterModel model);
const char * toString(OusterGeneration generation);
const char * toString(OusterRevision revision);

/// Resolve product identity and return its calibrated physics profile.
/// Unknown/ambiguous part numbers use a model-specific conservative fallback
/// and set fallback_revision=true; callers should surface that in logs.
OusterLidarProfile resolveOusterLidarProfile(
    const OusterProfileRequest & request);

/// Effective range multiplier relative to the profile's datasheet mode.
/// Ouster's operating-mode table gives 1.19x for every halving of points
/// gathered. Two octaves cover the valid 5-40 Hz / 512-4096 Rev8 modes.
double ousterModeRangeScale(const OusterLidarProfile & profile,
                            int columns_per_frame, double lidar_hz);

/// Datasheet precision multiplier for the active point-gathering rate. Ouster
/// publishes 0.71x sigma for a halving and 1.41x for a doubling.
double ousterModePrecisionScale(const OusterLidarProfile & profile,
                                int columns_per_frame, double lidar_hz);

}  // namespace gz_gpu_ouster_lidar
