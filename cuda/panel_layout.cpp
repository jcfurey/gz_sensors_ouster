// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#include "panel_layout.hpp"
#include "ray_processor_math.hpp"

#include <algorithm>
#include <cmath>

namespace gz_gpu_ouster_lidar {

namespace {

constexpr double kDegToRad = rpmath::kPi / 180.0;

// Angular pads beyond the requested coverage so bilinear corners near a
// panel boundary always have support inside the overlapping neighbour.
constexpr double kAzPadDeg = 2.0;
constexpr double kElPadDeg = 1.5;

// Mode thresholds. Cylindrical handles up to ±60° elevation (OS0 is ±~46°
// after metadata margin); anything reaching higher gets the dome rig.
constexpr double kCylMaxElDeg = 60.0;
// Dome side panels hand over to the zenith cap at this elevation. The cap's
// ±50° square FOV guarantees coverage down to 40° elevation on the cardinal
// directions, so a 47° handover leaves ≥ 7° of overlap.
constexpr double kDomeSideTopDeg = 47.0;
constexpr double kDomeCapHalfDeg = 50.0;
constexpr double kDomeMinElDeg = -32.0;  // side band lower limit (dome rig)
// Beam elevations slightly past +90° (zenith) wrap through the cap panel,
// whose square FOV covers up to ~50° past the axis; allow generous headroom.
constexpr double kDomeMaxElDeg = 120.0;

/// Geometric description of one panel before pixel quantisation.
struct PanelSpecD {
    double yaw_deg;
    double pitch_deg;
    double tan_half_h;
    double tan_half_v;
};

void fillRotation(float * r, double yaw_rad, double pitch_rad)
{
    // d_panel = Ry(pitch) · Rz(-yaw) · d_sensor  (transpose of the panel
    // orientation R = Rz(yaw) · Ry(-pitch); positive pitch aims the axis up).
    const double cy = std::cos(yaw_rad),   sy = std::sin(yaw_rad);
    const double cp = std::cos(pitch_rad), sp = std::sin(pitch_rad);
    r[0] = static_cast<float>(cp * cy);
    r[1] = static_cast<float>(cp * sy);
    r[2] = static_cast<float>(sp);
    r[3] = static_cast<float>(-sy);
    r[4] = static_cast<float>(cy);
    r[5] = 0.0f;
    r[6] = static_cast<float>(-sp * cy);
    r[7] = static_cast<float>(-sp * sy);
    r[8] = static_cast<float>(cp);
}

int evenClamp(double px, int max_dim)
{
    int v = static_cast<int>(std::ceil(px));
    v += v & 1;
    return std::clamp(v, 16, max_dim);
}

/// Quantise one panel to pixels at focal length `f` (pixels per unit
/// tangent) and append it to the layout. Square pixels (fy == fx) so the
/// rendering camera's HFOV + aspect-ratio model reproduces the intrinsics
/// exactly. If the height would exceed max_dim the focal length is reduced
/// to preserve vertical coverage (resolution degrades, geometry doesn't).
void appendPanel(PanelLayout & out, const PanelSpecD & s, double f, int max_dim)
{
    double fx = f;
    int w = evenClamp(2.0 * fx * s.tan_half_h, max_dim);
    fx = w / (2.0 * s.tan_half_h);
    int h = evenClamp(2.0 * fx * s.tan_half_v, max_dim);
    if (h >= max_dim && 2.0 * fx * s.tan_half_v > max_dim) {
        fx = max_dim / (2.0 * s.tan_half_v);
        w = evenClamp(2.0 * fx * s.tan_half_h, max_dim);
        fx = w / (2.0 * s.tan_half_h);
        h = max_dim;
    }

    const int i = out.n_panels;
    ResamplePanel & p = out.rp.panels[i];
    fillRotation(p.r, s.yaw_deg * kDegToRad, s.pitch_deg * kDegToRad);
    p.fx = static_cast<float>(fx);
    p.fy = static_cast<float>(fx);
    p.cx = static_cast<float>(w - 1) / 2.0f;
    p.cy = static_cast<float>(h - 1) / 2.0f;
    p.width = w;
    p.height = h;
    p.offset = out.rp.raw_n;
    out.rp.raw_n += w * h;

    PanelCameraSpec & c = out.cams[i];
    c.yaw_rad = s.yaw_deg * kDegToRad;
    c.pitch_rad = s.pitch_deg * kDegToRad;
    c.hfov_rad = 2.0 * std::atan(s.tan_half_h);
    c.width = w;
    c.height = h;

    out.n_panels = i + 1;
    out.rp.n_panels = out.n_panels;
}

/// True when every direction in the elevation band (sampled on a fine grid
/// over the full azimuth circle) projects into at least one panel.
bool coversBand(const ResampleParams & rp, double el_lo, double el_hi)
{
    constexpr double kStepDeg = 0.25;
    float u, v, cosp;
    for (double el = el_lo; el <= el_hi + 1e-9; el += kStepDeg) {
        for (double az = 0.0; az < 360.0; az += kStepDeg) {
            float dx, dy, dz;
            rpmath::beamDirection(
                static_cast<float>(el), static_cast<float>(az), dx, dy, dz);
            if (rpmath::panelForDirection(rp, dx, dy, dz, u, v, cosp) < 0) {
                return false;
            }
        }
    }
    return true;
}

}  // namespace

PanelLayout buildOusterPanelLayout(
    double min_alt_deg, double max_alt_deg,
    int H, int W,
    double oversample, int max_panel_dim)
{
    PanelLayout out;
    if (H <= 0 || W <= 0 || max_alt_deg <= min_alt_deg) return out;

    const double el_lo = min_alt_deg - kElPadDeg;
    const double el_hi = max_alt_deg + kElPadDeg;
    const bool cylindrical =
        el_hi <= kCylMaxElDeg && el_lo >= -kCylMaxElDeg;
    if (!cylindrical && (el_hi > kDomeMaxElDeg || el_lo < kDomeMinElDeg)) {
        return out;  // unsupported geometry → n_panels stays 0
    }

    // Focal length from the sensor's finest angular resolution. A pinhole's
    // coarsest angular pixel pitch is at the image centre (1/f radians), so
    // f ≥ oversample / Δangle guarantees ≥ oversample panel samples per beam
    // step everywhere in the panel.
    oversample = std::clamp(oversample, 1.0, 4.0);
    const double az_density = W / (2.0 * rpmath::kPi);
    const double el_density = H / ((max_alt_deg - min_alt_deg) * kDegToRad);
    const double f = oversample * std::max(az_density, el_density);

    // Pixel-aware angular pad: containment requires u ∈ [0, w−1], so the
    // outer half pixel of each border is unusable, and bilinear corners
    // need one more pixel of support. 2.5 pixels of slack expressed as an
    // angle keeps the fixed angular pads meaningful even on low-resolution
    // rigs (tiny f) where half a pixel can be larger than the pad itself.
    const double px_pad_deg = (2.5 / f) / kDegToRad;

    // Coverage-verified construction: build at scale 1, and if the fine grid
    // check finds an uncovered direction (it shouldn't — the analytic FOVs
    // below already include the azimuth-edge secant correction and the pixel
    // pad), widen the FOVs by 5% and retry a few times rather than ship a
    // hole.
    for (int attempt = 0; attempt < 5; ++attempt) {
        const double grow = 1.0 + 0.05 * attempt;
        out.n_panels = 0;
        out.rp = ResampleParams{};
        out.hemispherical = !cylindrical;

        if (cylindrical) {
            // 4 azimuth sectors, pitch 0. A fixed elevation appears at a
            // larger image inclination toward the sector's azimuth edge:
            // tan_v_required = tan(el) / cos(az_edge).
            const double half_az = 45.0 + kAzPadDeg + px_pad_deg;
            const double el_abs =
                std::max(std::abs(el_lo), std::abs(el_hi)) + px_pad_deg;
            const double tan_h = grow * std::tan(half_az * kDegToRad);
            const double tan_v = grow *
                std::tan(el_abs * kDegToRad) / std::cos(half_az * kDegToRad);
            for (int k = 0; k < 4; ++k) {
                appendPanel(out,
                    {90.0 * k, 0.0, tan_h, tan_v}, f, max_panel_dim);
            }
        } else {
            // 8 pitched side sectors covering [el_lo, kDomeSideTopDeg],
            // plus a square zenith cap whose inscribed coverage reaches
            // down to 90 − kDomeCapHalfDeg elevation on every azimuth.
            const double half_az = 22.5 + kAzPadDeg + px_pad_deg;
            const double pitch = 0.5 * (el_lo + kDomeSideTopDeg);
            const double half_band =
                0.5 * (kDomeSideTopDeg - el_lo) + kElPadDeg + px_pad_deg;
            const double tan_h = grow * std::tan(half_az * kDegToRad);
            const double tan_v = grow *
                std::tan(half_band * kDegToRad) / std::cos(half_az * kDegToRad);
            for (int k = 0; k < 8; ++k) {
                appendPanel(out,
                    {45.0 * k, pitch, tan_h, tan_v}, f, max_panel_dim);
            }
            const double tan_cap = grow *
                std::tan((kDomeCapHalfDeg + px_pad_deg) * kDegToRad);
            appendPanel(out, {0.0, 90.0, tan_cap, tan_cap}, f, max_panel_dim);
        }

        out.rp.H = H;
        out.rp.W = W;
        // Coverage check runs against an arbitrary positive far clip; the
        // caller overwrites with the real sensor range.
        out.rp.far_clip = 1.0e6f;

        if (coversBand(out.rp, min_alt_deg, max_alt_deg)) return out;
    }

    out.n_panels = 0;  // never achieved coverage — report unsupported
    out.rp.n_panels = 0;
    return out;
}

int countUncoveredRays(
    const ResampleParams & rp,
    const float * beam_alt_deg,
    const float * beam_az_deg)
{
    const float deg_per_col = 360.0f / static_cast<float>(rp.W);
    int uncovered = 0;
    for (int beam = 0; beam < rp.H; ++beam) {
        for (int m = 0; m < rp.W; ++m) {
            const float az = beam_az_deg[beam] -
                static_cast<float>(m) * deg_per_col;
            float dx, dy, dz, u, v, cosp;
            rpmath::beamDirection(beam_alt_deg[beam], az, dx, dy, dz);
            if (rpmath::panelForDirection(rp, dx, dy, dz, u, v, cosp) < 0) {
                ++uncovered;
            }
        }
    }
    return uncovered;
}

}  // namespace gz_gpu_ouster_lidar
