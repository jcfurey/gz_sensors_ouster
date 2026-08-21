// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#include "ouster_lidar_profile.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>

namespace gz_gpu_ouster_lidar {
namespace {

std::string canonical(const std::string & value)
{
    std::string out;
    out.reserve(value.size());
    for (const unsigned char c : value) {
        if (std::isalnum(c)) out.push_back(static_cast<char>(std::toupper(c)));
    }
    return out;
}

bool endsWith(const std::string & value, const std::string & suffix)
{
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

OusterGeneration generationOf(OusterRevision revision)
{
    switch (revision) {
    case OusterRevision::Gen1:
        return OusterGeneration::Gen1;
    case OusterRevision::RevC:
    case OusterRevision::RevD:
    case OusterRevision::Rev05:
    case OusterRevision::Rev06:
    case OusterRevision::Rev062:
        return OusterGeneration::Gen2;
    case OusterRevision::Rev07:
    case OusterRevision::Rev071:
        return OusterGeneration::Gen3;
    case OusterRevision::Rev08:
        return OusterGeneration::Gen4;
    default:
        return OusterGeneration::Unknown;
    }
}

bool hasDualReturns(OusterRevision revision)
{
    return revision == OusterRevision::Rev06 ||
           revision == OusterRevision::Rev062 ||
           revision == OusterRevision::Rev07 ||
           revision == OusterRevision::Rev071 ||
           revision == OusterRevision::Rev08;
}

OusterRevision revisionFromPartNumber(const std::string & prod_pn)
{
    std::string pn = prod_pn;
    std::transform(pn.begin(), pn.end(), pn.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });

    // Current part numbers carry 070/071/080 as their second group.
    if (pn.find("-080-") != std::string::npos) return OusterRevision::Rev08;
    if (pn.find("-071-") != std::string::npos) return OusterRevision::Rev071;
    if (pn.find("-070-") != std::string::npos) return OusterRevision::Rev07;

    // Historical 840/860 part numbers carry the revision as their suffix.
    if (endsWith(pn, "-06.2") || endsWith(pn, "-062"))
        return OusterRevision::Rev062;
    // Some exported metadata omits the final separator (for example
    // 840105010C). Require a digit before a bare revision letter so ordinary
    // descriptive strings such as "synthetic" cannot be misclassified.
    const bool compact_c = pn.size() >= 2 && endsWith(pn, "C") &&
                           std::isdigit(static_cast<unsigned char>(pn[pn.size() - 2]));
    const bool compact_d = pn.size() >= 2 && endsWith(pn, "D") &&
                           std::isdigit(static_cast<unsigned char>(pn[pn.size() - 2]));
    if (endsWith(pn, "-C") || compact_c) return OusterRevision::RevC;
    if (endsWith(pn, "-D") || compact_d) return OusterRevision::RevD;
    if (endsWith(pn, "-05")) return OusterRevision::Rev05;
    if (endsWith(pn, "-06")) return OusterRevision::Rev06;
    if (endsWith(pn, "-07")) return OusterRevision::Rev07;
    if (endsWith(pn, "-08")) return OusterRevision::Rev08;
    return OusterRevision::Unknown;
}

OusterRevision fallbackRevision(OusterModel model)
{
    if (model == OusterModel::OSDome) return OusterRevision::Rev07;
    if (model == OusterModel::OS1Max) return OusterRevision::Rev08;
    return OusterRevision::Rev06;
}

void setLegacyModelPhysics(OusterLidarProfile & p)
{
    p.minimum_range_m = 0.3;
    p.range_resolution_m = 0.001;
    p.lambertian_accuracy_m = 0.03;
    p.retroreflector_accuracy_m = 0.10;

    switch (p.model) {
    case OusterModel::OS0:
        p.detection_range_10_d90_m = 15.0;
        p.detection_range_80_d90_m = 45.0;
        p.detection_range_10_d50_m = 20.0;
        p.detection_range_80_d50_m = 50.0;
        p.representable_range_m = 270.0;
        p.precision_min_std_m = 0.010;
        p.precision_max_std_m = 0.050;
        p.beam_diameter_m = 0.005;
        p.beam_divergence_fwhm_deg = 0.35;
        break;
    case OusterModel::OS1:
        p.detection_range_10_d90_m = 45.0;
        p.detection_range_80_d90_m = 100.0;
        p.detection_range_10_d50_m = 55.0;
        p.detection_range_80_d50_m = 120.0;
        p.representable_range_m = 270.0;
        p.precision_min_std_m = 0.007;
        p.precision_max_std_m = 0.050;
        p.beam_diameter_m = 0.0095;
        p.beam_divergence_fwhm_deg = 0.18;
        break;
    case OusterModel::OS2:
        p.detection_range_10_d90_m = 80.0;
        p.detection_range_80_d90_m = 210.0;
        p.detection_range_10_d50_m = 100.0;
        p.detection_range_80_d50_m = 240.0;
        p.representable_range_m = 465.0;
        p.minimum_range_m = 1.0;
        p.precision_min_std_m = 0.025;
        p.precision_max_std_m = 0.080;
        p.beam_diameter_m = 0.019;
        p.beam_divergence_fwhm_deg = 0.09;
        // Historical OS2 range specifications use 2048x10, unlike OS0/OS1.
        p.reference_columns_per_second = 20480.0;
        break;
    default:
        break;
    }
}

void setRev7Physics(OusterLidarProfile & p)
{
    p.minimum_range_m = 0.5;
    p.range_resolution_m = 0.001;
    p.lambertian_accuracy_m = 0.025;
    p.retroreflector_accuracy_m = 0.05;
    p.max_returns = 2;

    switch (p.model) {
    case OusterModel::OS0:
        p.detection_range_10_d90_m = 35.0;
        p.detection_range_80_d90_m = 75.0;
        p.representable_range_m = 233.0;
        p.precision_min_std_m = 0.008;
        p.precision_max_std_m = 0.040;
        p.beam_diameter_m = 0.005;
        p.beam_divergence_fwhm_deg = 0.35;
        break;
    case OusterModel::OS1:
        p.detection_range_10_d90_m = 90.0;
        p.detection_range_80_d90_m = 170.0;
        p.representable_range_m = 233.0;
        p.precision_min_std_m = 0.005;
        p.precision_max_std_m = 0.030;
        p.beam_diameter_m = 0.0095;
        p.beam_divergence_fwhm_deg = 0.18;
        break;
    case OusterModel::OS2:
        p.detection_range_10_d90_m = 200.0;
        p.detection_range_80_d90_m = 350.0;
        p.representable_range_m = 404.0;
        p.minimum_range_m = 0.8;
        p.precision_min_std_m = 0.020;
        p.precision_max_std_m = 0.100;
        p.beam_diameter_m = 0.019;
        p.beam_divergence_fwhm_deg = 0.09;
        p.reference_columns_per_second = 20480.0;
        break;
    case OusterModel::OSDome:
        p.detection_range_10_d90_m = 20.0;
        p.detection_range_80_d90_m = 45.0;
        p.representable_range_m = 233.0;
        p.precision_min_std_m = 0.010;
        p.precision_max_std_m = 0.100;
        p.beam_diameter_m = 0.005;
        p.beam_divergence_fwhm_deg = 0.35;
        break;
    default:
        break;
    }
}

void setRev8Physics(OusterLidarProfile & p)
{
    p.minimum_range_m = 0.5;
    p.range_resolution_m = 0.001;
    p.representable_range_m = 500.0;
    p.precision_min_std_m = 0.0025;
    p.precision_max_std_m = 0.015;
    p.lambertian_accuracy_m = 0.0125;
    p.retroreflector_accuracy_m = 0.025;
    p.max_returns = 2;

    switch (p.model) {
    case OusterModel::OS0:
        p.detection_range_10_d90_m = 35.0;
        p.detection_range_80_d90_m = 75.0;
        p.beam_diameter_m = 0.005;
        p.beam_divergence_fwhm_deg = 0.35;
        break;
    case OusterModel::OS1:
        p.detection_range_10_d90_m = 90.0;
        p.detection_range_80_d90_m = 170.0;
        p.beam_diameter_m = 0.0095;
        p.beam_divergence_fwhm_deg = 0.18;
        break;
    case OusterModel::OSDome:
        p.detection_range_10_d90_m = 20.0;
        p.detection_range_80_d90_m = 45.0;
        p.precision_min_std_m = 0.005;
        p.precision_max_std_m = 0.050;
        p.beam_diameter_m = 0.005;
        p.beam_divergence_fwhm_deg = 0.35;
        break;
    case OusterModel::OS1Max:
        p.detection_range_10_d90_m = 200.0;
        p.detection_range_80_d90_m = 350.0;
        p.beam_diameter_m = 0.019;
        p.beam_divergence_fwhm_deg = 0.09;
        break;
    default:
        break;
    }
}

bool compatible(OusterModel model, OusterRevision revision)
{
    switch (revision) {
    case OusterRevision::Gen1:
        return model == OusterModel::OS1;
    case OusterRevision::RevC:
    case OusterRevision::RevD:
    case OusterRevision::Rev05:
    case OusterRevision::Rev06:
    case OusterRevision::Rev062:
        return model == OusterModel::OS0 || model == OusterModel::OS1 ||
               model == OusterModel::OS2;
    case OusterRevision::Rev07:
        return model == OusterModel::OS0 || model == OusterModel::OS1 ||
               model == OusterModel::OS2 || model == OusterModel::OSDome;
    case OusterRevision::Rev071:
        // Rev7.1 was a rolling reliability update to the L3 family. OS2
        // remained on Rev7.0 / FW 2.5.
        return model == OusterModel::OS0 || model == OusterModel::OS1 ||
               model == OusterModel::OSDome;
    case OusterRevision::Rev08:
        return model == OusterModel::OS0 || model == OusterModel::OS1 ||
               model == OusterModel::OSDome || model == OusterModel::OS1Max;
    default:
        return false;
    }
}

}  // namespace

OusterModel parseOusterModel(const std::string & prod_line)
{
    const std::string value = canonical(prod_line);
    if (value.find("OS1MAX") != std::string::npos) return OusterModel::OS1Max;
    if (value.find("OSDOME") != std::string::npos) return OusterModel::OSDome;
    if (value.find("OS0") != std::string::npos) return OusterModel::OS0;
    if (value.find("OS1") != std::string::npos) return OusterModel::OS1;
    if (value.find("OS2") != std::string::npos) return OusterModel::OS2;
    return OusterModel::Unknown;
}

OusterRevision parseOusterRevision(const std::string & value)
{
    const std::string revision = canonical(value);
    if (revision.empty() || revision == "AUTO") return OusterRevision::Auto;
    if (revision == "GEN1" || revision == "REVGEN1") return OusterRevision::Gen1;
    if (revision == "C" || revision == "REVC") return OusterRevision::RevC;
    if (revision == "D" || revision == "REVD") return OusterRevision::RevD;
    if (revision == "05" || revision == "5" || revision == "REV05" || revision == "REV5")
        return OusterRevision::Rev05;
    if (revision == "06" || revision == "6" || revision == "REV06" || revision == "REV6")
        return OusterRevision::Rev06;
    if (revision == "062" || revision == "62" || revision == "REV062" || revision == "REV62")
        return OusterRevision::Rev062;
    if (revision == "07" || revision == "7" || revision == "REV07" || revision == "REV7")
        return OusterRevision::Rev07;
    if (revision == "071" || revision == "71" || revision == "REV071" || revision == "REV71")
        return OusterRevision::Rev071;
    if (revision == "08" || revision == "8" || revision == "REV08" || revision == "REV8")
        return OusterRevision::Rev08;
    return OusterRevision::Unknown;
}

const char * toString(OusterModel model)
{
    switch (model) {
    case OusterModel::OS0: return "OS0";
    case OusterModel::OS1: return "OS1";
    case OusterModel::OS2: return "OS2";
    case OusterModel::OSDome: return "OSDome";
    case OusterModel::OS1Max: return "OS1 MAX";
    default: return "unknown";
    }
}

const char * toString(OusterGeneration generation)
{
    switch (generation) {
    case OusterGeneration::Gen1: return "Gen1";
    case OusterGeneration::Gen2: return "Gen2";
    case OusterGeneration::Gen3: return "Gen3/L3";
    case OusterGeneration::Gen4: return "Gen4/L4";
    default: return "unknown";
    }
}

const char * toString(OusterRevision revision)
{
    switch (revision) {
    case OusterRevision::Auto: return "auto";
    case OusterRevision::Gen1: return "gen1";
    case OusterRevision::RevC: return "revC";
    case OusterRevision::RevD: return "revD";
    case OusterRevision::Rev05: return "rev05";
    case OusterRevision::Rev06: return "rev06";
    case OusterRevision::Rev062: return "rev06.2";
    case OusterRevision::Rev07: return "rev07";
    case OusterRevision::Rev071: return "rev07.1";
    case OusterRevision::Rev08: return "rev08";
    default: return "unknown";
    }
}

OusterLidarProfile resolveOusterLidarProfile(const OusterProfileRequest & request)
{
    OusterLidarProfile p;
    p.model = parseOusterModel(request.prod_line);

    const OusterRevision requested = parseOusterRevision(request.hardware_revision);
    if (requested != OusterRevision::Auto && requested != OusterRevision::Unknown) {
        p.revision = requested;
    } else {
        p.revision = revisionFromPartNumber(request.prod_pn);
        p.revision_inferred = p.revision != OusterRevision::Unknown;

        // Product/firmware deductions that are unambiguous without a real PN.
        if (p.revision == OusterRevision::Unknown &&
            request.firmware_major >= 4 &&
            (p.model == OusterModel::OS0 || p.model == OusterModel::OS1 ||
             p.model == OusterModel::OSDome || p.model == OusterModel::OS1Max)) {
            p.revision = OusterRevision::Rev08;
            p.revision_inferred = true;
        } else if (p.revision == OusterRevision::Unknown &&
                   p.model == OusterModel::OS1Max) {
            p.revision = OusterRevision::Rev08;
            p.revision_inferred = true;
        } else if (p.revision == OusterRevision::Unknown &&
                   p.model == OusterModel::OS1 &&
                   (request.beam_count == 16 || request.firmware_major == 1)) {
            p.revision = OusterRevision::Gen1;
            p.revision_inferred = true;
        } else if (p.revision == OusterRevision::Unknown &&
                   request.firmware_major >= 3 &&
                   (p.model == OusterModel::OS0 || p.model == OusterModel::OS1 ||
                    p.model == OusterModel::OS2 || p.model == OusterModel::OSDome)) {
            p.revision = OusterRevision::Rev07;
            p.revision_inferred = true;
        }
    }

    if (p.revision == OusterRevision::Unknown || requested == OusterRevision::Unknown) {
        p.revision = fallbackRevision(p.model);
        p.fallback_revision = true;
    }
    p.generation = generationOf(p.revision);
    p.max_returns = hasDualReturns(p.revision) ? 2 : 1;
    p.supported = compatible(p.model, p.revision);

    // Keep an incompatible selection recognizable to the caller instead of
    // silently coercing it into another product (for example OS2 Rev8).
    if (!p.supported) {
        p.id = std::string(toString(p.model)) + "-" + toString(p.revision);
        return p;
    }

    if (p.revision == OusterRevision::Gen1) {
        p.minimum_range_m = 0.8;
        p.range_resolution_m = 0.003;
        p.representable_range_m = 200.0;
        p.beam_diameter_m = 0.010;
        p.beam_divergence_fwhm_deg = 0.13;
        p.max_returns = 1;
        if (request.firmware_major <= 1 && request.firmware_major != 0) {
            p.detection_range_10_d90_m = 40.0;
            p.detection_range_80_d90_m = 105.0;
            p.detection_range_10_d50_m = 60.0;
            p.detection_range_80_d50_m = 120.0;
            p.precision_min_std_m = 0.015;
            p.precision_max_std_m = 0.100;
        } else {
            p.detection_range_10_d90_m = 50.0;
            p.detection_range_80_d90_m = 110.0;
            p.detection_range_10_d50_m = 65.0;
            p.detection_range_80_d50_m = 150.0;
            p.precision_min_std_m = 0.010;
            p.precision_max_std_m = 0.050;
        }
        p.lambertian_accuracy_m = 0.05;
        p.retroreflector_accuracy_m = 0.10;
    } else if (p.revision == OusterRevision::Rev07 ||
               p.revision == OusterRevision::Rev071) {
        setRev7Physics(p);
    } else if (p.revision == OusterRevision::Rev08) {
        setRev8Physics(p);
    } else {
        setLegacyModelPhysics(p);
        // Full configurable profiles gained 1 mm range resolution in FW 2.4;
        // older/legacy packet generations retain 3 mm physical resolution.
        if (request.firmware_major < 2 ||
            (request.firmware_major == 2 && request.firmware_minor < 4)) {
            p.range_resolution_m = 0.003;
        }
    }

    if (request.low_data_profile) p.range_resolution_m = 0.008;

    p.id = std::string(toString(p.model)) + "-" + toString(p.revision);
    return p;
}

double ousterModeRangeScale(const OusterLidarProfile & profile,
                            int columns_per_frame, double lidar_hz)
{
    if (profile.reference_columns_per_second <= 0.0 ||
        columns_per_frame <= 0 || lidar_hz <= 0.0) {
        return 1.0;
    }
    const double actual = static_cast<double>(columns_per_frame) * lidar_hz;
    const double octaves = std::clamp(
        std::log2(profile.reference_columns_per_second / actual), -2.0, 2.0);
    return std::pow(1.19, octaves);
}

double ousterModePrecisionScale(const OusterLidarProfile & profile,
                                int columns_per_frame, double lidar_hz)
{
    if (profile.reference_columns_per_second <= 0.0 ||
        columns_per_frame <= 0 || lidar_hz <= 0.0) {
        return 1.0;
    }
    const double actual = static_cast<double>(columns_per_frame) * lidar_hz;
    const double octaves = std::clamp(
        std::log2(actual / profile.reference_columns_per_second), -2.0, 2.0);
    return std::pow(std::sqrt(2.0), octaves);
}

}  // namespace gz_gpu_ouster_lidar
