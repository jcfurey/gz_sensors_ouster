// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0
//
// Host-side builder for the perspective depth-panel rig that models the
// Ouster beam geometry directly:
//
//   * Cylindrical sensors (OS0 / OS1 / OS2, vertical FOV within ±60°):
//     4 azimuth sectors of ~94° HFOV around the full circle, pitch 0.
//   * Hemispherical sensors (OSDome, beams up to +90°): 8 azimuth sectors
//     pitched upward covering the lower band, plus one zenith cap panel.
//
// Each panel is an ordinary single-pass perspective depth render. Beam rays
// are resampled straight from the covering panel (see sampleBeamRange in
// ray_processor_math.hpp) — there is no intermediate cubemap or equirect
// grid. Panel resolution is derived from the sensor's angular resolution
// times an oversample factor, so interpolation error is bounded by design
// rather than by next_power_of_2 face sizing.
//
// Pure math, no Gazebo / rendering dependencies — unit-testable as-is.

#pragma once

#include "gz_gpu_ouster_lidar/ray_processor.hpp"

namespace gz_gpu_ouster_lidar {

/// Camera-facing description of one panel, used by the plugin to configure
/// the corresponding rendering::DepthCamera. Angles follow the Gazebo
/// convention (x forward, y left, z up); pitch is the elevation of the
/// panel's optical axis, positive up.
struct PanelCameraSpec {
    double yaw_rad = 0.0;    ///< Panel axis azimuth
    double pitch_rad = 0.0;  ///< Panel axis elevation (positive up)
    double hfov_rad = 0.0;   ///< Full horizontal FOV
    int width = 0;           ///< Image width (pixels)
    int height = 0;          ///< Image height (pixels)
};

struct PanelLayout {
    int n_panels = 0;            ///< 0 → unsupported beam geometry
    bool hemispherical = false;  ///< true when the zenith cap is present
    PanelCameraSpec cams[kMaxResamplePanels];
    /// Resample parameters with H, W, n_panels, raw_n and panels[] filled.
    /// The caller still sets far_clip and beam_origin_m.
    ResampleParams rp{};
};

/// Build the panel rig for a sensor whose beams span
/// [min_alt_deg, max_alt_deg] elevation with H beams and W columns.
///
/// @param oversample     panel angular resolution as a multiple of the
///                       sensor's finest angular resolution (clamped 1..4)
/// @param max_panel_dim  hard cap on panel width/height in pixels
///
/// Returns n_panels == 0 when the elevation band cannot be covered
/// (below -60° in dome mode, beyond +90.5° in any mode).
PanelLayout buildOusterPanelLayout(
    double min_alt_deg, double max_alt_deg,
    int H, int W,
    double oversample = 2.0,
    int max_panel_dim = 4096);

/// Count beam rays (H beams × W columns, including per-beam azimuth
/// offsets) that no panel covers. 0 for a healthy layout.
int countUncoveredRays(
    const ResampleParams & rp,
    const float * beam_alt_deg,
    const float * beam_az_deg);

}  // namespace gz_gpu_ouster_lidar
