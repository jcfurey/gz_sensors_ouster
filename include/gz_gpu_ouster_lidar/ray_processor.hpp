// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>

namespace gz_gpu_ouster_lidar {

/// Maximum panels in a rig: 8 azimuth sectors + zenith cap + headroom.
constexpr int kMaxResamplePanels = 12;

/// One perspective depth panel of the cylindrical / hemispherical rig.
///
/// The raw input buffer is a concatenation of per-panel planar-depth images
/// (one float per pixel: distance along the panel's forward axis; ±inf for
/// clipped pixels). A beam ray is resampled by rotating its direction into
/// the panel frame, projecting through the pinhole model below, bilinearly
/// sampling planar depth, and dividing by the ray/axis cosine to recover
/// Euclidean range.
struct ResamplePanel {
    float r[9];     ///< Row-major sensor→panel rotation (d_panel = R · d_sensor)
    float fx;       ///< Focal length, pixels (horizontal)
    float fy;       ///< Focal length, pixels (vertical; == fx, square pixels)
    float cx;       ///< Principal point x: (width - 1) / 2
    float cy;       ///< Principal point y: (height - 1) / 2
    int width;      ///< Panel image width (pixels)
    int height;     ///< Panel image height (pixels)
    int offset;     ///< Float index of this panel's block in the raw buffer
};

/// Parameters for the beam resampling kernel: exact Ouster beam directions
/// (cylindrical for OS0/1/2, hemispherical for OSDome) sampled from a rig of
/// perspective depth panels. Replaces the former GpuRays-cubemap equirect
/// grid: one bilinear pass straight from a single perspective render to each
/// calibrated beam, instead of cubemap→equirect→beam double interpolation.
struct ResampleParams {
    int H;              ///< Ouster beam count (output rows)
    int W;              ///< Ouster columns per frame (output cols)
    int n_panels;       ///< Number of valid entries in panels[]
    int raw_n;          ///< Total floats in the packed raw buffer
    /// 1 → nearest-pixel sampling: each beam takes the single closest
    /// rendered ray (a true raycast with direction quantised to the panel
    /// pixel grid; no range blending at depth edges). 0 → bilinear (default):
    /// smoother surfaces, but silhouette pixels blend fore/background range.
    int nearest;
    float far_clip;     ///< Metres; planar depth at/beyond this is a miss
    float beam_origin_m; ///< lidar_origin_to_beam_origin in metres
    ResamplePanel panels[kMaxResamplePanels];
};

/// Parameters for CUDA ray post-processing kernel
struct RayProcessParams {
    int H;                     ///< Beam count (pixels per column)
    int W;                     ///< Columns per frame

    // ── Signal model ─────────────────────────────────────────────────────────
    float base_signal;         ///< Baseline signal for 1/r² model (photons·m²)
    float base_reflectivity;   ///< Default reflectivity value [0–255]

    // ── Range noise ──────────────────────────────────────────────────────────
    float range_noise_min_std; ///< Min range noise σ at 0 m (metres), e.g. 0.005
    float range_noise_max_std; ///< Max range noise σ at max_range (metres), e.g. 0.03
    float max_range;           ///< Sensor max range (metres) for noise scaling

    // ── Photon / signal noise ────────────────────────────────────────────────
    float signal_noise_scale;  ///< Signal Poisson noise scale (0 = off, 1 = physical)
    float nearir_noise_scale;  ///< Near-IR Poisson noise scale (0 = off, 1 = physical)

    // ── Random dropouts ──────────────────────────────────────────────────────
    float dropout_rate_close;  ///< Dropout probability at 0 m (e.g. 0.001)
    float dropout_rate_far;    ///< Dropout probability at max_range (e.g. 0.05)

    // ── Solar-background false alarms ────────────────────────────────────────
    float false_alarm_rate;    ///< P(spurious return) per no-return pixel per
                               ///< frame (daytime background photons; 0 = off)

    // ── Depth-discontinuity suppression ──────────────────────────────────────
    float edge_discon_threshold; ///< Depth jump threshold (metres) to suppress neighbor, 0=off
};

// Forward declaration for pimpl.
class Backend;

// Raycast types (full definitions in raycast_scene.hpp / raycast_math.hpp;
// only references/pointers appear in this interface).
namespace rc {
struct SceneView;
struct InstanceXform;
struct ScanParams;
}  // namespace rc

/// Vendor-neutral ray post-processor: resamples the panel-rig depth buffer
/// → Ouster beam geometry, then synthesises channel arrays (range_mm,
/// signal, reflectivity, near_ir). Internally dispatches to whichever backend was
/// detected at construction time — CUDA (NVIDIA), HIP (AMD incl. APUs),
/// SYCL (Intel), or CPU.
class RayProcessor {
public:
    /// Construct the processor.
    /// @param seed  RNG seed for the noise model. 0 (default) means
    ///              non-deterministic (mixed from steady_clock + pid +
    ///              this-pointer on GPU, std::random_device on CPU). Any
    ///              non-zero value produces deterministic output across runs.
    explicit RayProcessor(uint64_t seed = 0);
    ~RayProcessor();

    RayProcessor(const RayProcessor &) = delete;
    RayProcessor & operator=(const RayProcessor &) = delete;

    /// Resample the packed panel-rig depth buffer to Ouster beam geometry,
    /// then run the noise model and produce final channel outputs. Fast path
    /// for the GPU pipeline; one host synchronisation per call.
    ///
    /// @param raw_host         rp.raw_n floats: concatenated per-panel
    ///                         planar-depth images (see ResamplePanel)
    /// @param beam_alt_host    H float array of beam altitude angles (degrees)
    /// @param beam_az_host     H float array of beam azimuth offsets (degrees)
    /// @param rp               Resample parameters
    /// @param range_out        H×W uint32 output (range in mm)
    /// @param signal_out       H×W uint16 output
    /// @param reflectivity_out H×W uint8 output
    /// @param nearir_out       H×W uint16 output
    /// @param pp               Noise processing parameters
    void processRaw(
        const float * raw_host,
        const float * beam_alt_host,
        const float * beam_az_host,
        const ResampleParams & rp,
        uint32_t *    range_out,
        uint16_t *    signal_out,
        uint8_t *     reflectivity_out,
        uint16_t *    nearir_out,
        const RayProcessParams & pp);

    /// Noise/channel stage only, for callers that already hold exact
    /// per-beam ranges (the full-raycast mode). depth_host is H×W metres
    /// (+inf for a miss); retro_host is an optional H×W laser_retro array
    /// (nullptr → base_reflectivity / unit intensity).
    /// `nir_host` (optional): per-pixel NEAR_IR ambient factor from the
    /// raycaster (albedo × sun illumination); nullptr keeps the legacy
    /// retro-based near-IR.
    void processDepth(
        const float * depth_host,
        const float * retro_host,
        uint32_t *    range_out,
        uint16_t *    signal_out,
        uint8_t *     reflectivity_out,
        uint16_t *    nearir_out,
        const RayProcessParams & pp,
        const float * nir_host = nullptr);

    /// Full per-beam raycast on the active backend (CUDA/HIP/SYCL kernel,
    /// or the OpenMP CPU fallback — identical shared math either way).
    /// GPU backends cache the uploaded scene geometry by `scene_version`
    /// and re-upload only when it changes; the per-scan transforms upload
    /// every call. Outputs: reported Ouster range in metres (+inf miss)
    /// and the nearest hit's laser_retro (0 on miss/unset), both H×W.
    void castScan(
        const rc::SceneView & scene,
        uint64_t scene_version,
        const rc::InstanceXform * xforms,
        const float * beam_alt_deg,
        const float * beam_az_deg,
        const float sensor_r[9],
        const float sensor_t[3],
        const rc::ScanParams & sp,
        float * range_out,
        float * retro_out,
        const float * col_r = nullptr,
        const float * col_t = nullptr,
        float * nir_out = nullptr);

    /// Returns true when the active backend is the CPU fallback (no GPU
    /// path is compiled in, or all GPU probes failed at construction).
    bool usesCpuFallback() const;

    /// Short name of the active backend: "cuda", "hip", "sycl", or "cpu".
    /// Primarily for logging / diagnostics.
    const char * backendName() const;

    /// Configured RNG seed (0 = non-deterministic).
    uint64_t seed() const { return seed_; }

private:
    uint64_t seed_ = 0;
    std::unique_ptr<Backend> backend_;
};

}  // namespace gz_gpu_ouster_lidar
