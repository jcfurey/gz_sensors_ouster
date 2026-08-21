// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0
//
// Shared, backend-agnostic ray-processing math.
//
// The CUDA, HIP, SYCL and CPU backends all implement the *same* noise,
// reflectivity and resampling model. Historically each backend carried its
// own near-verbatim copy of that math (~60-86% identical between files), so a
// fix or tweak had to be applied four times and silently drifted otherwise.
// This header is the single source of truth for the deterministic, RNG-free
// scalar math; each backend keeps only its own buffer management, kernel
// launch boilerplate and (intentionally backend-specific) RNG.
//
// Portability:
//   * Functions are marked GZ_OUSTER_HD so they are __host__ __device__ under
//     nvcc/hipcc and plain inline under a host or SYCL compiler. SYCL's
//     single-source model device-compiles any header function reached from a
//     kernel lambda, so no attribute is needed there.
//   * Transcendentals route through the `gzm` shim: sycl:: math in a SYCL TU,
//     std:: math everywhere else (CUDA/HIP provide device overloads of the
//     std:: math functions, and the host obviously does too).
//   * min/max use plain comparisons rather than fmin/fmax. Every call site
//     here feeds non-NaN inputs (guarded upstream), so the comparison form is
//     bit-identical to std::min/std::max and to the device fminf/fmaxf the
//     backends used before, while needing no library function.

#pragma once

#include "gz_gpu_ouster_lidar/ray_processor.hpp"

#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__HIP__)
  #define GZ_OUSTER_HD __host__ __device__
#else
  #define GZ_OUSTER_HD
#endif

#if defined(SYCL_LANGUAGE_VERSION)
  #include <sycl/sycl.hpp>
#else
  #include <cmath>
#endif

namespace gz_gpu_ouster_lidar {
namespace rpmath {

// ── Named constants (replace magic numbers previously inlined per backend) ───
constexpr float kPi              = 3.14159265358979f;
constexpr float kValidDepthMin   = 0.001f;    ///< metres; below this a return is a miss
constexpr float kPanelMinCos     = 1.0e-3f;   ///< min ray·axis cosine for panel hit
constexpr float kFarClipFrac     = 0.999f;    ///< planar depth ≥ far×this → miss
constexpr float kMinDenom        = 0.0001f;   ///< guards the 1/r² signal denominator
constexpr float kRangeToMm       = 1000.0f;
constexpr float kU16Max          = 65535.0f;
constexpr float kEdgeSuppressProb = 0.5f;     ///< P(suppress) at a detected depth edge
constexpr float kNearIrScale     = 256.0f;    ///< retro → near-IR photon-count analogue

// Reflectivity model (Ouster calibrated scale).
constexpr float kLambertianMax   = 100.0f;    ///< 0..100 diffuse band
constexpr float kReflByteMax     = 255.0f;
constexpr float kRetroLogSlope   = 22.0f;     ///< (255-100)/7 → 7 doublings of retro headroom

// Reflectivity weighting of dropout / range noise on dark targets.
constexpr float kDropoutRetroFloor = 0.33f;   ///< retro floor → up to 3× dropout
constexpr float kDropoutRetroMax   = 3.0f;
constexpr float kRangeRetroFloor   = 0.25f;   ///< retro floor → up to 2× range noise (1/√0.25)
constexpr float kRangeRetroMax     = 2.0f;
constexpr float kDefaultRetro      = 0.5f;    ///< assumed retro when the channel is absent
constexpr float kMaxRangeFloor     = 0.1f;    ///< guards the d / max_range ratio

// ── Math shim ────────────────────────────────────────────────────────────────
namespace gzm {

GZ_OUSTER_HD inline float fmin_(float a, float b) { return a < b ? a : b; }
GZ_OUSTER_HD inline float fmax_(float a, float b) { return a > b ? a : b; }

#if defined(SYCL_LANGUAGE_VERSION)
GZ_OUSTER_HD inline float sqrt_(float x)     { return sycl::sqrt(x); }
GZ_OUSTER_HD inline float log2_(float x)     { return sycl::log2(x); }
GZ_OUSTER_HD inline float log_(float x)      { return sycl::log(x); }
GZ_OUSTER_HD inline float exp_(float x)      { return sycl::exp(x); }
GZ_OUSTER_HD inline float cos_(float x)      { return sycl::cos(x); }
GZ_OUSTER_HD inline float sin_(float x)      { return sycl::sin(x); }
GZ_OUSTER_HD inline float asin_(float x)     { return sycl::asin(x); }
GZ_OUSTER_HD inline float atan2_(float y, float x) { return sycl::atan2(y, x); }
GZ_OUSTER_HD inline float floor_(float x)    { return sycl::floor(x); }
GZ_OUSTER_HD inline float fabs_(float x)     { return sycl::fabs(x); }
GZ_OUSTER_HD inline bool  isfinite_(float x) { return sycl::isfinite(x); }
#else
GZ_OUSTER_HD inline float sqrt_(float x)     { return std::sqrt(x); }
GZ_OUSTER_HD inline float log2_(float x)     { return std::log2(x); }
GZ_OUSTER_HD inline float log_(float x)      { return std::log(x); }
GZ_OUSTER_HD inline float exp_(float x)      { return std::exp(x); }
GZ_OUSTER_HD inline float cos_(float x)      { return std::cos(x); }
GZ_OUSTER_HD inline float sin_(float x)      { return std::sin(x); }
GZ_OUSTER_HD inline float asin_(float x)     { return std::asin(x); }
GZ_OUSTER_HD inline float atan2_(float y, float x) { return std::atan2(y, x); }
GZ_OUSTER_HD inline float floor_(float x)    { return std::floor(x); }
GZ_OUSTER_HD inline float fabs_(float x)     { return std::fabs(x); }
GZ_OUSTER_HD inline bool  isfinite_(float x) { return std::isfinite(x); }
#endif

}  // namespace gzm

// ── Resampling helpers ───────────────────────────────────────────────────────

/// 2D bilinear blend of four corner samples with per-corner validity.
/// n_valid == 4  → full bilinear; 1..3 → mean of the valid corners;
/// 0 → `empty_value` (the depth path passes +inf, the retro path passes 0).
GZ_OUSTER_HD inline float bilinearOrAverage(
    float a00, float a01, float a10, float a11,
    float h_alpha, float v_alpha,
    bool v00, bool v01, bool v10, bool v11,
    int n_valid, float empty_value)
{
    if (n_valid == 0) return empty_value;
    if (n_valid == 4) {
        const float top = a00 * (1.0f - h_alpha) + a01 * h_alpha;
        const float bot = a10 * (1.0f - h_alpha) + a11 * h_alpha;
        return top * (1.0f - v_alpha) + bot * v_alpha;
    }
    float sum = 0.0f;
    if (v00) sum += a00;
    if (v01) sum += a01;
    if (v10) sum += a10;
    if (v11) sum += a11;
    return sum / static_cast<float>(n_valid);
}

// ── Panel-rig sampling (cylindrical / hemispherical beam model) ──────────────
//
// Beam directions use the Gazebo sensor frame: x forward, y left, z up.
// Each panel is a perspective depth render; image u grows rightward (−y),
// v grows downward (−z), pixel centres at integer coordinates with the
// principal point at ((w−1)/2, (h−1)/2).

/// Unit direction of a beam given elevation and azimuth in degrees.
GZ_OUSTER_HD inline void beamDirection(float el_deg, float az_deg,
                                       float & dx, float & dy, float & dz)
{
    const float el = el_deg * kPi / 180.0f;
    const float az = az_deg * kPi / 180.0f;
    const float ce = gzm::cos_(el);
    dx = ce * gzm::cos_(az);
    dy = ce * gzm::sin_(az);
    dz = gzm::sin_(el);
}

/// Project a unit direction into one panel. On success returns true and
/// fills (u, v) — continuous pixel coordinates inside the image — plus the
/// ray·axis cosine used to convert planar depth back to Euclidean range.
GZ_OUSTER_HD inline bool projectToPanel(const ResamplePanel & p,
    float dx, float dy, float dz, float & u, float & v, float & cosp)
{
    const float xc = p.r[0] * dx + p.r[1] * dy + p.r[2] * dz;
    if (xc < kPanelMinCos) return false;
    const float yc = p.r[3] * dx + p.r[4] * dy + p.r[5] * dz;
    const float zc = p.r[6] * dx + p.r[7] * dy + p.r[8] * dz;
    u = p.cx - p.fx * (yc / xc);
    v = p.cy - p.fy * (zc / xc);
    cosp = xc;
    return u >= 0.0f && u <= static_cast<float>(p.width - 1) &&
           v >= 0.0f && v <= static_cast<float>(p.height - 1);
}

/// First panel containing the direction, or -1 when uncovered.
/// (u, v, cosp) are valid only for a non-negative return.
GZ_OUSTER_HD inline int panelForDirection(const ResampleParams & rp,
    float dx, float dy, float dz, float & u, float & v, float & cosp)
{
    for (int i = 0; i < rp.n_panels; ++i) {
        if (projectToPanel(rp.panels[i], dx, dy, dz, u, v, cosp)) return i;
    }
    return -1;
}

/// Per-beam ray azimuth (degrees) for output column m, where m is the Ouster
/// measurement id: m = 0 points forward (+x) and azimuth decreases with m
/// (clockwise encoder), matching the hardware column ↔ encoder-angle
/// convention. beam_azimuth is SUBTRACTED to match the Ouster SDK XYZ LUT
/// (xyzlut.cpp: azimuth = -beam_azimuth_angles; direction =
/// cos(encoder + azimuth) = cos(encoder - beam_az)); the opposite sign lands
/// each return 2*beam_az away from where os_cloud reconstructs it, splitting
/// every object into one arc per beam-azimuth group (4 for OS1-64).
GZ_OUSTER_HD inline float beamRayAzimuthDeg(
    float beam_az_deg, int m, float deg_per_col)
{
    return -beam_az_deg - static_cast<float>(m) * deg_per_col;
}

/// Resample one beam ray from the panel rig: bilinear on planar depth in the
/// covering panel, then divide by the ray cosine to recover range along the
/// beam. Returns `inf_value` for misses, clipped pixels and uncovered rays.
/// `inf_value` is caller-supplied because each backend has its own portable
/// spelling of +inf.
GZ_OUSTER_HD inline float sampleBeamRange(
    const float * raw, const ResampleParams & rp,
    float el_deg, float az_deg, float inf_value)
{
    float dx, dy, dz;
    beamDirection(el_deg, az_deg, dx, dy, dz);

    float u, v, cosp;
    const int pi = panelForDirection(rp, dx, dy, dz, u, v, cosp);
    if (pi < 0) return inf_value;
    const ResamplePanel & p = rp.panels[pi];
    const float far_lim = rp.far_clip * kFarClipFrac;

    if (rp.nearest) {
        // Nearest-pixel mode: take the single closest rendered ray. The
        // rasterised pixel IS an exact ray-scene intersection, so this is a
        // true raycast with the beam direction quantised to the pixel grid
        // (≤ half a pixel, i.e. ≤ 1/(2·oversample) of the beam spacing).
        // No interpolation → no fore/background range blending at edges.
        const int ui = static_cast<int>(u + 0.5f);
        const int vi = static_cast<int>(v + 0.5f);
        const float d = raw[p.offset + vi * p.width + ui];
        if (!gzm::isfinite_(d) || d <= kValidDepthMin || d >= far_lim) {
            return inf_value;
        }
        return d / cosp;
    }

    const int u0 = static_cast<int>(gzm::floor_(u));
    const int v0 = static_cast<int>(gzm::floor_(v));
    const int u1 = (u0 + 1 < p.width)  ? u0 + 1 : p.width - 1;
    const int v1 = (v0 + 1 < p.height) ? v0 + 1 : p.height - 1;
    const float ha = u - static_cast<float>(u0);
    const float va = v - static_cast<float>(v0);

    const float * img = raw + p.offset;
    const float d00 = img[v0 * p.width + u0];
    const float d01 = img[v0 * p.width + u1];
    const float d10 = img[v1 * p.width + u0];
    const float d11 = img[v1 * p.width + u1];

    const bool b00 = gzm::isfinite_(d00) && d00 > kValidDepthMin && d00 < far_lim;
    const bool b01 = gzm::isfinite_(d01) && d01 > kValidDepthMin && d01 < far_lim;
    const bool b10 = gzm::isfinite_(d10) && d10 > kValidDepthMin && d10 < far_lim;
    const bool b11 = gzm::isfinite_(d11) && d11 > kValidDepthMin && d11 < far_lim;
    const int n_valid = static_cast<int>(b00) + static_cast<int>(b01) +
                        static_cast<int>(b10) + static_cast<int>(b11);

    const float planar = bilinearOrAverage(
        d00, d01, d10, d11, ha, va, b00, b01, b10, b11, n_valid, inf_value);
    if (!gzm::isfinite_(planar)) return inf_value;
    return planar / cosp;
}

/// Beam-origin parallax correction: convert the panel-rendered Euclidean range
/// (measured from the lidar origin) into the reported Ouster range.
///
/// The panel depth cameras render from the lidar origin, so `depth` is the range
/// from that origin along the beam. The Ouster XYZ LUT reconstructs each point as
///   xyz = (R - n)·d_hat + n·[cos(enc), sin(enc), 0]
/// where R is the reported range and n = beam_origin_m. Projecting onto d_hat
/// gives (R - n) = depth - n·cos(elev), i.e. R = depth + n·(1 - cos(elev)). The
/// beam-origin distance t = depth - n·cos(elev); the reported range is t + n,
/// matching the raycast path (range = t0 + n_off) and real-sensor reconstruction.
/// The previous form returned t (omitting the + n), reconstructing every point
/// ~n (≈1.4–2.8 cm) closer than the raycast mode and the LUT. No-op at elev = 0
/// (cos = 1) and n = 0. No-op for non-finite depth.
GZ_OUSTER_HD inline float applyBeamOrigin(float depth, float beam_angle_deg,
                                          float beam_origin_m)
{
    if (gzm::isfinite_(depth) && beam_origin_m > 0.0f) {
        // Keep the multiply order (deg * π / 180) the backends used inline, so
        // results stay bit-identical across backends, not merely close.
        const float elev_rad = beam_angle_deg * kPi / 180.0f;
        return gzm::fmax_(0.0f,
            depth - beam_origin_m * gzm::cos_(elev_rad) + beam_origin_m);
    }
    return depth;
}

// ── Noise / channel model helpers (RNG-free; callers supply the draws) ───────

/// Range fraction t ∈ [0,1] used to interpolate range-dependent quantities.
GZ_OUSTER_HD inline float rangeFraction(float d, float max_range)
{
    return gzm::fmin_(d / gzm::fmax_(max_range, kMaxRangeFloor), 1.0f);
}

/// Retro value to use for the dropout / range-noise weighting: the measured
/// retro when present and positive, else a neutral default.
GZ_OUSTER_HD inline float retroForNoise(const float * retro, int idx)
{
    if (retro != nullptr) {
        const float r = retro[idx];
        if (gzm::isfinite_(r) && r > 0.0f) return r;
    }
    return kDefaultRetro;
}

/// Power-law interpolation of the vendor's 10% and 80% Lambertian range
/// points.  Unlike the old fixed sqrt(rho) law, this preserves both published
/// calibration anchors for every Ouster product/revision.
GZ_OUSTER_HD inline float calibratedRangeAtReflectivity(
    float retro_val, float range_10, float range_80)
{
    if (range_10 <= 0.0f || range_80 <= range_10 || retro_val <= 0.0f) return 0.0f;
    const float exponent = gzm::log_(range_80 / range_10) / gzm::log_(8.0f);
    return range_10 * gzm::exp_(exponent * gzm::log_(retro_val / 0.1f));
}

/// Product-calibrated detection probability. D90 is exact at both vendor
/// reflectivity anchors. Historical profiles also supply measured D50 points;
/// newer datasheets publish only D90, so D50 is placed `rolloff` beyond D90
/// (15% by default) to avoid the old non-physical hard discontinuity.
GZ_OUSTER_HD inline float detectionProbability(
    float d, float retro_val,
    float range_10_d90, float range_80_d90,
    float range_10_d50, float range_80_d50,
    float rolloff, float max_range)
{
    if (d <= 0.0f || d >= max_range) return 0.0f;
    if (range_10_d90 <= 0.0f || range_80_d90 <= range_10_d90) return 1.0f;  // profile detection disabled
    if (retro_val <= 0.0f) return 0.0f;

    const float d90 = calibratedRangeAtReflectivity(
        retro_val, range_10_d90, range_80_d90);
    if (d90 <= 0.0f) return 0.0f;

    float d50 = calibratedRangeAtReflectivity(
        retro_val, range_10_d50, range_80_d50);
    if (d50 <= d90) d50 = d90 * (1.0f + gzm::fmax_(rolloff, 0.01f));

    constexpr float kLn9 = 2.19722457733622f;
    const float slope = kLn9 / (d50 - d90);
    return 1.0f / (1.0f + gzm::exp_(slope * (d - d50)));
}

/// Total dropout probability, clamped to [0,1].  The calibrated detection
/// probability is combined with the user-tunable stochastic miss term.
GZ_OUSTER_HD inline float dropoutProbability(float d, float retro_val,
    float drop_close, float drop_far, float max_range,
    float range_10_d90, float range_80_d90,
    float range_10_d50, float range_80_d50, float rolloff)
{
    const float t = rangeFraction(d, max_range);
    const float p = drop_close + t * (drop_far - drop_close);
    const float refl = gzm::fmin_(1.0f / gzm::fmax_(retro_val, kDropoutRetroFloor),
                                  kDropoutRetroMax);
    const float random_keep = 1.0f - gzm::fmin_(p * refl, 1.0f);
    const float calibrated_keep = detectionProbability(
        d, retro_val, range_10_d90, range_80_d90,
        range_10_d50, range_80_d50, rolloff, max_range);
    return 1.0f - random_keep * calibrated_keep;
}

/// Quantise a range to the active Ouster UDP/product resolution. A zero step
/// keeps the legacy exact-depth test/debug path unchanged.
GZ_OUSTER_HD inline float quantizeRange(float d, float resolution)
{
    if (resolution <= 0.0f) return d;
    return gzm::floor_(d / resolution + 0.5f) * resolution;
}

/// Gaussian range-noise σ. Interpolated over range, then scaled by inverse
/// √reflectivity (dark targets up to 2× noisier).
///
/// The √ exponent follows ToF ranging theory: timing precision is
/// σ ∝ 1/√N for N detected signal photons, and N ∝ ρ for a fixed range —
/// so σ ∝ 1/√ρ. (E.g. "Influence of Waveform Characteristics on LiDAR
/// Ranging Accuracy and Precision", Sensors 18(4), 2018; SPAD dToF precision
/// bounds, arXiv:2507.11404.) Range dependence stays the configurable
/// min→max ramp: datasheet precision-vs-range curves fold in firmware
/// filtering that a pure r/√ρ law does not capture.
GZ_OUSTER_HD inline float rangeNoiseSigma(float d, float retro_val,
    float min_std, float max_std, float reference_range)
{
    const float t = rangeFraction(d, reference_range);
    float sigma = min_std + t * (max_std - min_std);
    sigma *= gzm::fmin_(1.0f / gzm::sqrt_(gzm::fmax_(retro_val, kRangeRetroFloor)),
                        kRangeRetroMax);
    return sigma;
}

/// 1/r² signal model with a clamped denominator.
GZ_OUSTER_HD inline float signalFromRange(float d, float intensity, float base_signal)
{
    const float r_sq = gzm::fmax_(d * d, kMinDenom);
    return base_signal * intensity / r_sq;
}

/// Convert an Ouster calibrated reflectivity byte back to the physical
/// reflectance used by the ray caster. This is the inverse of
/// reflectivityToByte() apart from the byte quantisation.
GZ_OUSTER_HD inline float reflectivityByteToRetro(float value)
{
    const float b = gzm::fmin_(gzm::fmax_(value, 0.0f), kReflByteMax);
    if (b <= kLambertianMax) {
        return b / kLambertianMax;
    }
    constexpr float kLn2 = 0.6931471805599453f;
    return gzm::exp_((b - kLambertianMax) * kLn2 / kRetroLogSlope);
}

/// Map a Gazebo retro value to the Ouster reflectivity byte:
///   rv ∈ [0,1]  → linear  [0,100]      (Lambertian diffuse)
///   rv > 1      → log map  [101,255]    (retroreflective)
/// Missing laser_retro values are resolved to physical reflectance by the ray
/// caster before incidence, obscurant attenuation and return arbitration.
GZ_OUSTER_HD inline uint8_t reflectivityToByte(float rv)
{
    if (rv <= 1.0f) {
        return static_cast<uint8_t>(gzm::fmin_(rv * kLambertianMax, kLambertianMax));
    }
    return static_cast<uint8_t>(
        gzm::fmin_(kLambertianMax + gzm::log2_(rv) * kRetroLogSlope, kReflByteMax));
}

/// Clamp a float to the uint16 channel range.
GZ_OUSTER_HD inline uint16_t clampU16(float v)
{
    return static_cast<uint16_t>(gzm::fmin_(gzm::fmax_(v, 0.0f), kU16Max));
}

/// True if any cardinal neighbour of `idx` is a miss or differs in depth by
/// more than `threshold` — i.e. `idx` sits on a depth discontinuity. Callers
/// gate this on `threshold > 0` and apply the kEdgeSuppressProb RNG draw.
GZ_OUSTER_HD inline bool edgeDiscontinuity(const float * depth, int idx,
                                           int H, int W, float threshold)
{
    const float d = depth[idx];
    const int beam = idx / W;
    const int col  = idx % W;
    const int nb_r[4] = {beam - 1, beam + 1, beam, beam};
    const int nb_c[4] = {col, col, col - 1, col + 1};
    bool suppress = false;
    for (int k = 0; k < 4; ++k) {
        const int r = nb_r[k];
        const int c = nb_c[k];
        if (r < 0 || r >= H || c < 0 || c >= W) continue;
        const float nd = depth[r * W + c];
        if (!gzm::isfinite_(nd) || nd < kValidDepthMin ||
            gzm::fabs_(nd - d) > threshold) {
            suppress = true;
        }
    }
    return suppress;
}

}  // namespace rpmath
}  // namespace gz_gpu_ouster_lidar
