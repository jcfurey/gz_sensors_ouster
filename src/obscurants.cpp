// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#include "obscurants.hpp"

#include "lidar_common.hpp"

#include <gz/math/Matrix3.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/ParticleEmitter.hh>

#include <algorithm>
#include <cmath>
#include <string>

namespace gz_gpu_ouster_lidar {

namespace {

/// Bounding-sphere radius of a volume's local half extents.
float boundingRadius(const rc::RcObscurant & ob)
{
    return std::sqrt(ob.half[0] * ob.half[0] + ob.half[1] * ob.half[1] +
                     ob.half[2] * ob.half[2]);
}

/// World-space centre of a volume, recovered from the stored world→local
/// transform: p_l = r·p_w + t is zero at the centre, so p_w = −rᵀ·t.
::gz::math::Vector3d centreOf(const rc::RcObscurant & ob)
{
    return {-(ob.r[0] * ob.t[0] + ob.r[3] * ob.t[1] + ob.r[6] * ob.t[2]),
            -(ob.r[1] * ob.t[0] + ob.r[4] * ob.t[1] + ob.r[7] * ob.t[2]),
            -(ob.r[2] * ob.t[0] + ob.r[5] * ob.t[1] + ob.r[8] * ob.t[2])};
}

/// Scalar accessor for gz's wrapped Float / Boolean message fields.
double floatOr(bool has, double value, double fallback)
{
    return has ? value : fallback;
}

}  // namespace

double extinctionFromVisibility(double visibility_m)
{
    return (visibility_m > 0.0) ? (kKoschmieder / visibility_m) : 0.0;
}

ObscurantConfig parseObscurantConfig(const ::sdf::ElementConstPtr & elem)
{
    ObscurantConfig cfg;
    if (!elem) return cfg;

    if (elem->HasElement("particle_obscuration")) {
        cfg.mirror_particles = elem->Get<bool>("particle_obscuration");
    }
    if (elem->HasElement("particle_extinction")) {
        cfg.particle_extinction =
            std::max(0.0, elem->Get<double>("particle_extinction"));
    }
    if (elem->HasElement("particle_growth")) {
        cfg.particle_growth =
            std::max(0.0, elem->Get<double>("particle_growth"));
    }
    if (elem->HasElement("obscurant_lidar_ratio")) {
        cfg.lidar_ratio = elem->Get<double>("obscurant_lidar_ratio");
    }
    if (elem->HasElement("obscurant_albedo")) {
        cfg.albedo = elem->Get<double>("obscurant_albedo");
    }
    if (elem->HasElement("obscurant_multiple_scattering")) {
        cfg.multiple_scattering =
            elem->Get<double>("obscurant_multiple_scattering");
    }
    if (elem->HasElement("pulse_length")) {
        cfg.pulse_gate_m = std::max(0.0, elem->Get<double>("pulse_length"));
    }

    for (::sdf::ElementConstPtr e = elem->FindElement("obscurant"); e;
         e = e->GetNextElement("obscurant")) {
        ObscurantVolume v;
        v.lidar_ratio = cfg.lidar_ratio;
        v.albedo = cfg.albedo;
        v.multiple_scattering = cfg.multiple_scattering;

        const std::string type =
            e->HasElement("type") ? e->Get<std::string>("type") : "ellipsoid";
        if (type == "box") {
            v.type = rc::ObscurantType::kBox;
        } else if (type == "cylinder") {
            v.type = rc::ObscurantType::kCylinder;
        } else if (type == "ellipsoid" || type == "sphere") {
            v.type = rc::ObscurantType::kEllipsoid;
        } else {
            RCLCPP_WARN(lidarLogger(),
                "obscurant type '%s' unknown (expected box|ellipsoid|"
                "cylinder|sphere); using ellipsoid", type.c_str());
        }

        if (e->HasElement("pose")) {
            v.pose = e->Get<::gz::math::Pose3d>("pose");
        }
        if (e->HasElement("size")) {
            v.size = e->Get<::gz::math::Vector3d>("size");
        }
        // Two ways to say the same thing. Visibility (meteorological optical
        // range) is the friendlier knob — "you can see 8 m in this" — and
        // converts through Koschmieder's law; an explicit extinction wins
        // when both are given.
        if (e->HasElement("extinction")) {
            v.extinction = e->Get<double>("extinction");
        } else if (e->HasElement("visibility")) {
            v.extinction =
                extinctionFromVisibility(e->Get<double>("visibility"));
        }
        if (e->HasElement("lidar_ratio")) {
            v.lidar_ratio = e->Get<double>("lidar_ratio");
        }
        if (e->HasElement("albedo")) {
            v.albedo = e->Get<double>("albedo");
        }
        if (e->HasElement("multiple_scattering")) {
            v.multiple_scattering = e->Get<double>("multiple_scattering");
        }

        if (!(v.extinction > 0.0)) {
            RCLCPP_WARN(lidarLogger(),
                "obscurant at (%.2f %.2f %.2f) has no positive <extinction> "
                "or <visibility>; ignored",
                v.pose.Pos().X(), v.pose.Pos().Y(), v.pose.Pos().Z());
            continue;
        }
        cfg.volumes.push_back(v);
    }
    return cfg;
}

void makeObscurant(rc::ObscurantType type,
                   const ::gz::math::Pose3d & world_pose,
                   const ::gz::math::Vector3d & half,
                   double sigma, double lidar_ratio, double albedo,
                   double multiple_scattering, rc::RcObscurant & out)
{
    // local→world rotation, then store its transpose (= world→local) plus
    // t = −rᵀ·T, matching rc::Scene::computeXform's convention.
    const ::gz::math::Matrix3d m(world_pose.Rot());
    float r_lw[9];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            r_lw[3 * i + j] = static_cast<float>(m(i, j));
        }
    }
    out.r[0] = r_lw[0]; out.r[1] = r_lw[3]; out.r[2] = r_lw[6];
    out.r[3] = r_lw[1]; out.r[4] = r_lw[4]; out.r[5] = r_lw[7];
    out.r[6] = r_lw[2]; out.r[7] = r_lw[5]; out.r[8] = r_lw[8];

    const float t_lw[3] = {static_cast<float>(world_pose.Pos().X()),
                           static_cast<float>(world_pose.Pos().Y()),
                           static_cast<float>(world_pose.Pos().Z())};
    for (int a = 0; a < 3; ++a) {
        out.t[a] = -(out.r[3 * a] * t_lw[0] + out.r[3 * a + 1] * t_lw[1] +
                     out.r[3 * a + 2] * t_lw[2]);
    }

    out.half[0] = static_cast<float>(std::max(half.X(), 0.0));
    out.half[1] = static_cast<float>(std::max(half.Y(), 0.0));
    out.half[2] = static_cast<float>(std::max(half.Z(), 0.0));
    out.sigma = static_cast<float>(std::max(sigma, 0.0));
    // A non-positive lidar ratio would mean infinite backscatter; fall back
    // to the documented default rather than emitting a return of ∞ power.
    out.lidar_ratio = static_cast<float>(
        lidar_ratio > 0.0 ? lidar_ratio : kObscurantLidarRatio);
    out.albedo = static_cast<float>(std::clamp(albedo, 0.0, 1.0));
    // η is a fraction of the extinction, so it lives in (0, 1]; 0 would make
    // the medium perfectly transparent to the laser while still returning
    // backscatter, which is not a physical state.
    out.ms_factor = static_cast<float>(
        std::clamp(multiple_scattering, rc::kRcMinMultipleScattering, 1.0));
    out.type = type;
}

bool obscurantFromVolume(const ObscurantVolume & vol, rc::RcObscurant & out)
{
    if (vol.extinction <= 0.0) return false;
    makeObscurant(vol.type, vol.pose, vol.size / 2.0, vol.extinction,
                  vol.lidar_ratio, vol.albedo, vol.multiple_scattering, out);
    return true;
}

bool obscurantFromEmitter(const ::gz::msgs::ParticleEmitter & em,
                          const ::gz::math::Pose3d & world_pose,
                          const ObscurantConfig & cfg,
                          rc::RcObscurant & out)
{
    if (cfg.particle_extinction <= 0.0) return false;
    // An emitter that has been switched off (or was authored off) emits no
    // particles, so it obscures nothing. gz leaves the field unset for
    // emitters that were never toggled, which means "emitting" by default.
    if (em.has_emitting() && !em.emitting().data()) return false;

    const double scatter = floatOr(em.has_particle_scatter_ratio(),
                                   em.particle_scatter_ratio().data(),
                                   kGzDefaultScatterRatio);
    if (scatter <= 0.0) return false;
    const double sigma = cfg.particle_extinction * scatter;

    // Steady-state plume extent: particles live `lifetime` seconds and travel
    // at the mean of the emitter's velocity range, so they occupy roughly the
    // emitter volume dilated by v̄ · lifetime.
    const double v_min = floatOr(em.has_min_velocity(),
                                 em.min_velocity().data(), 0.0);
    const double v_max = floatOr(em.has_max_velocity(),
                                 em.max_velocity().data(), 0.0);
    const double lifetime = floatOr(em.has_lifetime(),
                                    em.lifetime().data(), 0.0);
    const double grow = std::max(cfg.particle_growth, 0.0) *
                        0.5 * (v_min + v_max) * std::max(lifetime, 0.0);

    ::gz::math::Vector3d half{0.0, 0.0, 0.0};
    if (em.has_size()) {
        half.Set(std::abs(em.size().x()) / 2.0, std::abs(em.size().y()) / 2.0,
                 std::abs(em.size().z()) / 2.0);
    }

    rc::ObscurantType type = rc::ObscurantType::kEllipsoid;
    switch (em.type()) {
        case ::gz::msgs::ParticleEmitter::BOX:
            type = rc::ObscurantType::kBox;
            break;
        case ::gz::msgs::ParticleEmitter::CYLINDER:
            type = rc::ObscurantType::kCylinder;
            break;
        case ::gz::msgs::ParticleEmitter::POINT:
            // gz ignores <size> for a point emitter: the plume is whatever
            // the particles travel, so the dilation IS the volume.
            half.Set(0.0, 0.0, 0.0);
            break;
        default:
            break;  // ELLIPSOID (and any future type) keeps the default
    }
    half += ::gz::math::Vector3d{grow, grow, grow};
    if (half.X() <= 0.0 && half.Y() <= 0.0 && half.Z() <= 0.0) return false;

    makeObscurant(type, world_pose, half, sigma, cfg.lidar_ratio, cfg.albedo,
                  cfg.multiple_scattering, out);
    return true;
}

size_t keepNearest(std::vector<rc::RcObscurant> & obs,
                   const ::gz::math::Vector3d & sensor_pos, size_t max_keep)
{
    if (obs.size() <= max_keep) return 0;
    const size_t dropped = obs.size() - max_keep;
    std::partial_sort(
        obs.begin(), obs.begin() + static_cast<std::ptrdiff_t>(max_keep),
        obs.end(),
        [&sensor_pos](const rc::RcObscurant & a, const rc::RcObscurant & b) {
            const double da =
                (centreOf(a) - sensor_pos).Length() - boundingRadius(a);
            const double db =
                (centreOf(b) - sensor_pos).Length() - boundingRadius(b);
            return da < db;
        });
    obs.resize(max_keep);
    return dropped;
}

size_t gatherObscurants(const ObscurantConfig & cfg,
                        const ::gz::sim::EntityComponentManager & ecm,
                        const ::gz::math::Vector3d & sensor_pos,
                        std::vector<rc::RcObscurant> & out)
{
    out.clear();

    rc::RcObscurant ob;
    for (const auto & vol : cfg.volumes) {
        if (obscurantFromVolume(vol, ob)) out.push_back(ob);
    }

    if (cfg.mirror_particles) {
        ecm.Each<::gz::sim::components::ParticleEmitter>(
            [&](const ::gz::sim::Entity & ent,
                const ::gz::sim::components::ParticleEmitter * em) -> bool {
                const auto pose = ::gz::sim::worldPose(ent, ecm);
                if (obscurantFromEmitter(em->Data(), pose, cfg, ob)) {
                    out.push_back(ob);
                }
                return true;
            });
    }

    return keepNearest(out, sensor_pos,
                       static_cast<size_t>(rc::kMaxObscurants));
}

}  // namespace gz_gpu_ouster_lidar
