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
constexpr float kRangeRetroFloor   = 0.5f;    ///< retro floor → up to 2× range noise
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
GZ_OUSTER_HD inline float cos_(float x)      { return sycl::cos(x); }
GZ_OUSTER_HD inline float fabs_(float x)     { return sycl::fabs(x); }
GZ_OUSTER_HD inline bool  isfinite_(float x) { return sycl::isfinite(x); }
#else
GZ_OUSTER_HD inline float sqrt_(float x)     { return std::sqrt(x); }
GZ_OUSTER_HD inline float log2_(float x)     { return std::log2(x); }
GZ_OUSTER_HD inline float cos_(float x)      { return std::cos(x); }
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

/// Beam-origin parallax correction: subtract the lidar-origin-to-beam-origin
/// offset projected onto the beam elevation. No-op for non-finite depth or a
/// zero offset.
GZ_OUSTER_HD inline float applyBeamOrigin(float depth, float beam_angle_deg,
                                          float beam_origin_m)
{
    if (gzm::isfinite_(depth) && beam_origin_m > 0.0f) {
        // Keep the multiply-then-divide order (deg * π / 180) the backends
        // used inline, so results are bit-identical, not merely close.
        const float elev_rad = beam_angle_deg * kPi / 180.0f;
        return gzm::fmax_(0.0f, depth - beam_origin_m * gzm::cos_(elev_rad));
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

/// Dropout probability, clamped to [0,1]. Rises with range and falls with
/// reflectivity (dark targets drop out up to 3× more often).
GZ_OUSTER_HD inline float dropoutProbability(float d, float retro_val,
    float drop_close, float drop_far, float max_range)
{
    const float t = rangeFraction(d, max_range);
    const float p = drop_close + t * (drop_far - drop_close);
    const float refl = gzm::fmin_(1.0f / gzm::fmax_(retro_val, kDropoutRetroFloor),
                                  kDropoutRetroMax);
    return gzm::fmin_(p * refl, 1.0f);
}

/// Gaussian range-noise σ. Interpolated over range, then scaled by inverse
/// reflectivity (dark targets up to 2× noisier).
GZ_OUSTER_HD inline float rangeNoiseSigma(float d, float retro_val,
    float min_std, float max_std, float max_range)
{
    const float t = rangeFraction(d, max_range);
    float sigma = min_std + t * (max_std - min_std);
    sigma *= gzm::fmin_(1.0f / gzm::fmax_(retro_val, kRangeRetroFloor), kRangeRetroMax);
    return sigma;
}

/// 1/r² signal model with a clamped denominator.
GZ_OUSTER_HD inline float signalFromRange(float d, float intensity, float base_signal)
{
    const float r_sq = gzm::fmax_(d * d, kMinDenom);
    return base_signal * intensity / r_sq;
}

/// Map a Gazebo retro value to the Ouster reflectivity byte:
///   rv ∈ [0,1]  → linear  [0,100]      (Lambertian diffuse)
///   rv > 1      → log map  [101,255]    (retroreflective)
/// Callers handle the "no retro channel" case with base_reflectivity.
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
