// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0
//
// Participating media (smoke / dust / fog) for the raycast ray mode: the
// ECM→rc::RcObscurant side of the model whose physics lives in
// cuda/raycast_math.hpp (rcApplyObscurants) and whose rationale is written
// up in docs/MODEL_REFERENCES.md §11.
//
// Two sources feed the same volume list:
//
//   * Gazebo <particle_emitter> elements, mirrored automatically so the
//     smoke you SEE is the smoke you SCAN. gz gives no physical density for
//     an emitter — rate/lifetime/particle_size are authored for visual
//     appeal, not radiometry — so the extinction coefficient is derived from
//     <particle_scatter_ratio>, which is already Gazebo's "how much does
//     this emitter affect range sensors" knob (gz-rendering applies it to
//     GpuRays), scaled by a documented per-sensor coefficient.
//
//   * <obscurant> blocks on the plugin itself, for authored volumes with
//     direct control of σ_ext (or visibility), lidar ratio and albedo. Use
//     these when the plume shape or optical density matters.

#pragma once

#include "raycast_scene.hpp"

#include <gz/math/Pose3.hh>
#include <gz/math/Vector3.hh>
#include <gz/msgs/particle_emitter.pb.h>
#include <gz/sim/EntityComponentManager.hh>
#include <sdf/Element.hh>

#include <cstddef>
#include <vector>

namespace gz_gpu_ouster_lidar {

/// Extinction-to-backscatter ratio S = σ_ext/β_π [sr]. 50 sr sits between
/// the dust (≈40–50) and biomass-smoke (≈50–70) measurements and is a
/// reasonable default for an unspecified obscurant; fog/water cloud is
/// nearer 18–20. (Müller et al., JGR 112 D16202, 2007.)
constexpr double kObscurantLidarRatio = 50.0;
/// Single-scattering albedo ω, for the NEAR_IR airlight term. Smoke and dust
/// are weakly absorbing in the near IR (ω ≈ 0.8–0.9); pure soot is far lower.
constexpr double kObscurantAlbedo = 0.8;
/// Effective one-pulse range gate ΔR = c·τ_pulse/2 [m]; ≈0.6 m for the ~4 ns
/// pulse of a mid-range ToF lidar.
constexpr double kObscurantPulseGate = 0.6;
/// σ_ext produced by an emitter whose <particle_scatter_ratio> is 1.0 [1/m].
/// With gz's default ratio of 0.65 this gives σ ≈ 0.65 /m — a meteorological
/// optical range of ~6 m, i.e. thick smoke you cannot see through. Turn it
/// down for haze.
constexpr double kParticleExtinction = 1.0;
/// Fraction of the mean particle travel distance (v̄ · lifetime) by which an
/// emitter's own <size> volume is dilated to cover the plume the particles
/// actually occupy. 0 uses the emitter volume verbatim.
constexpr double kParticleGrowth = 1.0;
/// gz-rendering's default <particle_scatter_ratio>, assumed when an emitter
/// leaves the field unset.
constexpr double kGzDefaultScatterRatio = 0.65;
/// Koschmieder's constant: visibility V is the range at which contrast falls
/// to 2%, so σ_ext = ln(1/0.02) / V.
constexpr double kKoschmieder = 3.912;

/// One authored <obscurant> volume, in world coordinates.
struct ObscurantVolume {
    rc::ObscurantType type = rc::ObscurantType::kEllipsoid;
    ::gz::math::Pose3d pose;                  ///< world frame
    ::gz::math::Vector3d size{1.0, 1.0, 1.0};  ///< FULL extents (metres)
    double extinction = 0.0;                  ///< σ_ext [1/m]
    double lidar_ratio = kObscurantLidarRatio;
    double albedo = kObscurantAlbedo;
};

/// Everything the mirror needs to turn a world into a list of obscurants.
struct ObscurantConfig {
    bool mirror_particles = true;      ///< auto-mirror <particle_emitter>
    double particle_extinction = kParticleExtinction;
    double particle_growth = kParticleGrowth;
    double lidar_ratio = kObscurantLidarRatio;  ///< default S for emitters
    double albedo = kObscurantAlbedo;           ///< default ω for emitters
    double pulse_gate_m = kObscurantPulseGate;
    std::vector<ObscurantVolume> volumes;

    /// True when this configuration can ever produce an obscurant, i.e. when
    /// the per-scan gather is worth running at all.
    bool active() const
    {
        if (mirror_particles && particle_extinction > 0.0) return true;
        for (const auto & v : volumes) {
            if (v.extinction > 0.0) return true;
        }
        return false;
    }
};

/// Meteorological optical range [m] → extinction coefficient [1/m].
/// Returns 0 for a non-positive visibility (treated as "not specified").
double extinctionFromVisibility(double visibility_m);

/// Read the obscuration knobs and every <obscurant> child of the plugin's
/// SDF element. Unparseable or optically empty volumes are warned about and
/// skipped rather than aborting the plugin; a null element yields defaults.
ObscurantConfig parseObscurantConfig(const ::sdf::ElementConstPtr & elem);

/// Fill an rc::RcObscurant from a local→world pose and LOCAL HALF extents.
/// Stores the inverse (world→local) transform the kernel expects.
void makeObscurant(rc::ObscurantType type,
                   const ::gz::math::Pose3d & world_pose,
                   const ::gz::math::Vector3d & half,
                   double sigma, double lidar_ratio, double albedo,
                   rc::RcObscurant & out);

/// Convert one authored volume. False when it is optically empty.
bool obscurantFromVolume(const ObscurantVolume & vol, rc::RcObscurant & out);

/// Convert one Gazebo particle emitter, given its world pose. False when the
/// emitter contributes nothing: not emitting, or a zero scatter ratio /
/// extinction scale.
///
/// The volume is the emitter's own <size> region dilated isotropically by
/// `particle_growth` × the mean particle travel distance. gz does not
/// document a stable emission axis in the message, so the dilation is
/// deliberately isotropic — it always CONTAINS the plume, and an authored
/// <obscurant> is the way to get an exact shape.
bool obscurantFromEmitter(const ::gz::msgs::ParticleEmitter & em,
                          const ::gz::math::Pose3d & world_pose,
                          const ObscurantConfig & cfg,
                          rc::RcObscurant & out);

/// Truncate to `max_keep` entries, keeping the ones nearest `sensor_pos`.
/// Returns how many were dropped.
///
/// Distance is measured to the volume's bounding sphere rather than its
/// centre, so a large cloud the sensor is standing inside always outranks a
/// small distant one. No-op when the list already fits.
size_t keepNearest(std::vector<rc::RcObscurant> & obs,
                   const ::gz::math::Vector3d & sensor_pos, size_t max_keep);

/// Gather every obscurant the sensor should see this scan: authored volumes
/// first, then mirrored particle emitters, truncated to rc::kMaxObscurants
/// nearest the sensor. `out` is cleared first; its capacity is reused.
/// Returns how many volumes the cap discarded, so the caller can say so —
/// smoke that quietly stops obscuring is otherwise very hard to diagnose.
size_t gatherObscurants(const ObscurantConfig & cfg,
                        const ::gz::sim::EntityComponentManager & ecm,
                        const ::gz::math::Vector3d & sensor_pos,
                        std::vector<rc::RcObscurant> & out);

}  // namespace gz_gpu_ouster_lidar
