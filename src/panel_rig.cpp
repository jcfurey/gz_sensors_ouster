// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#include "panel_rig.hpp"
#include "ouster_metadata.hpp"

#include <algorithm>
#include <cstring>
#include <string>
#include <utility>

#include <gz/rendering/RenderEngine.hh>
#include <gz/rendering/RenderingIface.hh>
#include <gz/rendering/Scene.hh>

namespace gz_gpu_ouster_lidar {

static const rclcpp::Logger kLogger = lidarLogger();

PanelRig::PanelRig(std::string sensor_name)
    : sensor_name_(std::move(sensor_name))
{
}

bool PanelRig::buildLayout(const OusterMetadata & meta, double oversample,
                           const std::string & sampling, double max_range)
{
    // Cylindrical sectors for OS0/1/2; pitched sectors + zenith cap for the
    // hemispherical OSDome. min_alt/max_alt already carry kBeamMarginDeg.
    layout_ = buildOusterPanelLayout(
        meta.min_alt, meta.max_alt, meta.H, meta.W, oversample);
    if (layout_.n_panels == 0) {
        RCLCPP_ERROR(kLogger,
            "Unsupported beam geometry: altitude range [%.1f, %.1f] deg has "
            "no panel-rig coverage (supported: bands within ±60 deg use the "
            "cylindrical rig; higher bands use the hemispherical rig, which "
            "accepts -32 deg up to +120 deg — elevations past +90 wrap "
            "through the zenith cap); plugin disabled.",
            meta.min_alt, meta.max_alt);
        return false;
    }
    layout_.rp.far_clip = static_cast<float>(max_range);
    layout_.rp.beam_origin_m = static_cast<float>(meta.beam_origin_mm / 1000.0);
    layout_.rp.nearest = (sampling == "nearest") ? 1 : 0;

    // Verify the exact calibrated beams (incl. per-beam azimuth offsets)
    // against the rig. The builder already self-checks a fine grid; this
    // catches pathological metadata (offsets beyond the pads).
    const int uncovered = countUncoveredRays(
        layout_.rp, meta.beam_alt_f.data(), meta.beam_az_f.data());
    if (uncovered > 0) {
        RCLCPP_WARN(kLogger,
            "%d of %d beam rays fall outside the panel rig and will read as "
            "misses (check beam_azimuth_angles in the metadata).",
            uncovered, meta.H * meta.W);
    }
    {
        std::string dims;
        for (int i = 0; i < layout_.n_panels; ++i) {
            dims += " " + std::to_string(layout_.cams[i].width) + "x" +
                    std::to_string(layout_.cams[i].height);
        }
        RCLCPP_INFO(kLogger,
            "Panel rig: %d %s panels, %s sampling (%.1f MiB raw):%s",
            layout_.n_panels,
            layout_.hemispherical ? "hemispherical" : "cylindrical",
            sampling.c_str(),
            layout_.rp.raw_n * sizeof(float) / 1048576.0, dims.c_str());
    }
    return true;
}

bool PanelRig::ensureCreated(double max_range, uint32_t visibility_mask)
{
    if (created()) return true;
    if (layout_.n_panels == 0) return false;

    // Wait until the Sensors system has created the OGRE2 scene.
    auto * engine = ::gz::rendering::engine("ogre2");
    if (!engine) return false;
    if (engine->SceneCount() == 0) return false;

    auto scene = engine->SceneByIndex(0);
    if (!scene) return false;

    auto root = scene->RootVisual();

    // One perspective depth camera per panel. Each is a plain single-pass
    // render — the Ouster beam model (cylindrical or hemispherical) is
    // applied in the resample kernel, not by the renderer, so there is no
    // cubemap and no equirect intermediate.
    cams_.reserve(static_cast<size_t>(layout_.n_panels));
    conns_.reserve(static_cast<size_t>(layout_.n_panels));
    quats_.reserve(static_cast<size_t>(layout_.n_panels));
    for (int i = 0; i < layout_.n_panels; ++i) {
        const auto & cs = layout_.cams[i];
        auto cam = scene->CreateDepthCamera(
            sensor_name_ + "_panel" + std::to_string(i));
        if (!cam) {
            RCLCPP_ERROR(kLogger,
                "Failed to create depth camera for panel %d", i);
            destroy();
            return false;
        }
        cam->SetImageWidth(cs.width);
        cam->SetImageHeight(cs.height);
        // Square pixels: aspect = w/h reproduces the layout's fx == fy
        // pinhole model exactly.
        cam->SetAspectRatio(
            static_cast<double>(cs.width) / static_cast<double>(cs.height));
        cam->SetHFOV(::gz::math::Angle(cs.hfov_rad));
        cam->SetNearClipPlane(kNearClip);
        cam->SetFarClipPlane(max_range);
        cam->SetVisibilityMask(visibility_mask);
        cam->CreateDepthTexture();

        if (root) {
            root->AddChild(cam);
        }

        // Panel orientation relative to the sensor frame. Gazebo's positive
        // pitch tilts the x axis down, so an upward panel axis is -pitch.
        quats_.emplace_back(0.0, -cs.pitch_rad, cs.yaw_rad);
        conns_.push_back(cam->ConnectNewDepthFrame(
            [this, i](const float * data, unsigned int w, unsigned int h,
                      unsigned int ch, const std::string & /*format*/) {
                onPanelFrame(static_cast<size_t>(i), data, w, h, ch);
            }));
        cams_.push_back(cam);
    }
    filled_.assign(static_cast<size_t>(layout_.n_panels), false);
    return true;
}

void PanelRig::renderScan(const ::gz::math::Pose3d & pose,
                          FrameExchange & exch)
{
    if (cams_.empty()) return;

    // Render every panel of the rig at the lidar_frame world pose.
    // Render() does the perspective depth pass; PostRender() reads the
    // buffer back and synchronously fires onPanelFrame, which copies the
    // data into its packed slot in pending_buf_.
    std::fill(filled_.begin(), filled_.end(), false);
    for (size_t i = 0; i < cams_.size(); ++i) {
        cams_[i]->SetWorldPosition(pose.Pos());
        cams_[i]->SetWorldRotation(pose.Rot() * quats_[i]);
        cams_[i]->Render();
        cams_[i]->PostRender();
    }

    const bool complete = std::all_of(
        filled_.begin(), filled_.end(), [](bool b) { return b; });
    if (complete) {
        // O(1) vector swap; capacity is preserved and reused. A still-
        // pending previous frame means PostUpdate didn't drain in time —
        // surface the drop so a sustained problem (sim-time stall,
        // post-pause burst) is visible in logs instead of silent.
        if (exch.publish(pending_buf_, layout_.rp.raw_n)) {
            RCLCPP_WARN_THROTTLE(kLogger, throttle_clock_, 5000,
                "%s: dropped rig frame (PostUpdate didn't drain); "
                "total dropped=%lu", sensor_name_.c_str(),
                static_cast<unsigned long>(exch.dropped()));
        }
    } else {
        RCLCPP_WARN_THROTTLE(kLogger, throttle_clock_, 5000,
            "%s: incomplete panel rig frame (a depth camera did not "
            "deliver); dropping this scan", sensor_name_.c_str());
    }
}

void PanelRig::onPanelFrame(size_t panel, const float * data,
                            unsigned int width, unsigned int height,
                            unsigned int channels)
{
    // Fired synchronously from PostRender() inside renderScan(), which the
    // plugin already serialises under its render-busy barrier — no extra
    // locking needed here.
    if (!data || width == 0 || height == 0 || channels == 0) return;
    if (panel >= static_cast<size_t>(layout_.n_panels)) return;

    // Defensive: the resampler's panel intrinsics assume exactly the layout
    // dimensions. If OGRE2 ever hands back a differently-sized frame the
    // projection would silently mis-sample, so drop it loudly instead.
    const auto & cs = layout_.cams[panel];
    if (width != static_cast<unsigned int>(cs.width) ||
        height != static_cast<unsigned int>(cs.height)) {
        RCLCPP_ERROR_THROTTLE(kLogger, throttle_clock_, 5000,
            "%s: panel %zu frame %ux%u (expected %dx%d); dropping",
            sensor_name_.c_str(), panel, width, height,
            cs.width, cs.height);
        return;
    }

    RCLCPP_INFO_ONCE(kLogger,
        "%s: panel depth frames flowing (%ux%ux%u for panel 0 of %d)",
        sensor_name_.c_str(), width, height, channels, layout_.n_panels);

    // Copy into this panel's packed slot in the render-only pending_buf_,
    // no lock held. After warmup the resize is a no-op (capacity is kept
    // across the triple-buffer swaps) and the memcpy is the only cost.
    const size_t n = static_cast<size_t>(width) * height;
    if (pending_buf_.size() < static_cast<size_t>(layout_.rp.raw_n)) {
        pending_buf_.resize(static_cast<size_t>(layout_.rp.raw_n));
    }
    float * dst = pending_buf_.data() + layout_.rp.panels[panel].offset;
    if (channels == 1) {
        std::memcpy(dst, data, n * sizeof(float));
    } else {
        // Defensive stride copy if a depth implementation delivers packed
        // multi-channel data; channel 0 is depth.
        for (size_t i = 0; i < n; ++i) {
            dst[i] = data[i * channels];
        }
    }
    filled_[panel] = true;
}

void PanelRig::destroy()
{
    if (cams_.empty()) return;

    // Disconnect frame callbacks before destroying the cameras. Safe here:
    // callbacks only fire from renderScan's PostRender calls, and the
    // caller has already flushed the render-busy barrier.
    conns_.clear();

    auto * engine = ::gz::rendering::engine("ogre2");
    if (engine && engine->SceneCount() > 0) {
        auto scene = engine->SceneByIndex(0);
        if (scene) {
            for (auto & cam : cams_) {
                if (cam) scene->DestroySensor(cam);
            }
        }
    }

    cams_.clear();
    quats_.clear();
    filled_.clear();
}

}  // namespace gz_gpu_ouster_lidar
