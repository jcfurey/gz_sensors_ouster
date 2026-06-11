// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#include "raycast_mirror.hpp"

#include "gz_gpu_ouster_lidar/ray_processor.hpp"

#include <cstring>
#include <string>
#include <unordered_map>
#include <utility>

#include <gz/sim/components/Geometry.hh>
#include <gz/sim/components/LaserRetro.hh>
#include <gz/sim/components/Light.hh>
#include <gz/sim/components/Material.hh>
#include <gz/sim/components/Transparency.hh>
#include <gz/sim/components/Visual.hh>
#include <gz/common/Mesh.hh>
#include <gz/common/MeshManager.hh>
#include <gz/common/SubMesh.hh>
#include <gz/math/Matrix3.hh>

// sdf/Geometry.hh only forward-declares the shape classes; the accessors
// (BoxShape()->Size(), ...) need the full definitions on every distro.
#include <sdf/Box.hh>
#include <sdf/Cylinder.hh>
#include <sdf/Geometry.hh>
#include <sdf/Mesh.hh>
#include <sdf/Plane.hh>
#include <sdf/Sphere.hh>

namespace gz_gpu_ouster_lidar {

static const rclcpp::Logger kLogger = lidarLogger();

namespace {

/// Row-major rotation + translation from a gz pose.
void poseToRT(const ::gz::math::Pose3d & pose, float r[9], float t[3])
{
    const ::gz::math::Matrix3d m(pose.Rot());
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            r[3 * i + j] = static_cast<float>(m(i, j));
        }
    }
    t[0] = static_cast<float>(pose.Pos().X());
    t[1] = static_cast<float>(pose.Pos().Y());
    t[2] = static_cast<float>(pose.Pos().Z());
}

/// Flatten a gz mesh (all TRIANGLES submeshes, scale baked in).
bool appendGzMesh(const ::gz::common::Mesh & mesh,
                  const ::gz::math::Vector3d & scale,
                  std::vector<float> & verts, std::vector<int> & tris)
{
    for (unsigned int si = 0; si < mesh.SubMeshCount(); ++si) {
        auto sm = mesh.SubMeshByIndex(si).lock();
        if (!sm) continue;
        if (sm->SubMeshPrimitiveType() != ::gz::common::SubMesh::TRIANGLES) {
            continue;
        }
        const int base = static_cast<int>(verts.size()) / 3;
        for (unsigned int v = 0; v < sm->VertexCount(); ++v) {
            const auto & p = sm->Vertex(v);
            verts.push_back(static_cast<float>(p.X() * scale.X()));
            verts.push_back(static_cast<float>(p.Y() * scale.Y()));
            verts.push_back(static_cast<float>(p.Z() * scale.Z()));
        }
        for (unsigned int k = 0; k + 2 < sm->IndexCount(); k += 3) {
            tris.push_back(base + static_cast<int>(sm->Index(k)));
            tris.push_back(base + static_cast<int>(sm->Index(k + 1)));
            tris.push_back(base + static_cast<int>(sm->Index(k + 2)));
        }
    }
    return !tris.empty();
}

}  // namespace

RaycastMirror::RaycastMirror(std::string sensor_name)
    : sensor_name_(std::move(sensor_name))
{
}

RaycastMirror::~RaycastMirror()
{
    stop();
}

void RaycastMirror::start(const Params & p, RayProcessor * proc,
                          std::mutex * proc_mtx, FrameExchange * exch)
{
    params_ = p;
    proc_ = proc;
    proc_mtx_ = proc_mtx;
    exch_ = exch;
    thread_ = std::thread(&RaycastMirror::threadFunc, this);
}

void RaycastMirror::stop()
{
    shutdown_.store(true, std::memory_order_release);
    cv_.notify_all();
    if (thread_.joinable()) {
        thread_.join();
    }
}

void RaycastMirror::rebuildScene(
    const ::gz::sim::EntityComponentManager & ecm, size_t visual_count)
{
    auto scene = std::make_shared<rc::Scene>();
    std::vector<Ref> refs;
    std::unordered_map<std::string, int> mesh_cache;  // key → BVH root node
    int skipped = 0;

    ecm.Each<::gz::sim::components::Visual,
             ::gz::sim::components::Geometry>(
        [&](const ::gz::sim::Entity & ent,
            const ::gz::sim::components::Visual *,
            const ::gz::sim::components::Geometry * geom) -> bool {
            const sdf::Geometry & g = geom->Data();
            rc::GeomType type = rc::GeomType::kBox;
            float size[3] = {0.0f, 0.0f, 0.0f};
            int root_node = -1;
            Ref ref;
            ref.entity = ent;

            switch (g.Type()) {
                case sdf::GeometryType::BOX: {
                    const auto box = g.BoxShape()->Size();
                    type = rc::GeomType::kBox;
                    size[0] = static_cast<float>(box.X() / 2.0);
                    size[1] = static_cast<float>(box.Y() / 2.0);
                    size[2] = static_cast<float>(box.Z() / 2.0);
                    break;
                }
                case sdf::GeometryType::SPHERE:
                    type = rc::GeomType::kSphere;
                    size[0] = static_cast<float>(g.SphereShape()->Radius());
                    break;
                case sdf::GeometryType::CYLINDER:
                    type = rc::GeomType::kCylinder;
                    size[0] =
                        static_cast<float>(g.CylinderShape()->Radius());
                    size[1] =
                        static_cast<float>(g.CylinderShape()->Length() / 2.0);
                    break;
                case sdf::GeometryType::PLANE: {
                    const auto * plane = g.PlaneShape();
                    type = rc::GeomType::kPlane;
                    size[0] = static_cast<float>(plane->Size().X() / 2.0);
                    size[1] = static_cast<float>(plane->Size().Y() / 2.0);
                    // Local frame has the plane at z = 0; fold the SDF
                    // normal into the entity pose as an extra rotation.
                    ::gz::math::Quaterniond q;
                    q.SetFrom2Axes(::gz::math::Vector3d::UnitZ,
                                   plane->Normal().Normalized());
                    ref.offset = q;
                    break;
                }
                case sdf::GeometryType::MESH: {
                    const auto * shape = g.MeshShape();
                    const std::string resolved = ::gz::sim::asFullPath(
                        shape->Uri(), shape->FilePath());
                    const auto & scale = shape->Scale();
                    const std::string key = resolved + "|" +
                        std::to_string(scale.X()) + "," +
                        std::to_string(scale.Y()) + "," +
                        std::to_string(scale.Z());
                    auto it = mesh_cache.find(key);
                    if (it == mesh_cache.end()) {
                        const ::gz::common::Mesh * gz_mesh =
                            ::gz::common::MeshManager::Instance()->Load(
                                resolved);
                        if (!gz_mesh) {
                            RCLCPP_WARN(kLogger,
                                "raycast: cannot load mesh '%s'; visual "
                                "skipped", resolved.c_str());
                            ++skipped;
                            return true;
                        }
                        std::vector<float> verts;
                        std::vector<int> tris;
                        if (!appendGzMesh(*gz_mesh, scale, verts, tris)) {
                            ++skipped;
                            return true;
                        }
                        const int root = scene->addMesh(verts, tris);
                        if (root < 0) {
                            ++skipped;
                            return true;
                        }
                        it = mesh_cache.emplace(key, root).first;
                    }
                    type = rc::GeomType::kMesh;
                    root_node = it->second;
                    break;
                }
                default:
                    // capsule / ellipsoid / heightmap / polyline: not mirrored.
                    ++skipped;
                    return true;
            }

            const auto * lr =
                ecm.Component<::gz::sim::components::LaserRetro>(ent);
            const float retro = lr ? static_cast<float>(lr->Data()) : 0.0f;

            // Material model for the monostatic return (see
            // rcApparentReflectance): specular coefficient from the visual
            // material's <specular> colour (mean RGB), transmittance from
            // the visual's <transparency>. Both default to 0 (pure
            // Lambertian, opaque) when unset.
            float spec = 0.0f;
            if (const auto * mat =
                    ecm.Component<::gz::sim::components::Material>(ent)) {
                const auto & s = mat->Data().Specular();
                spec = static_cast<float>((s.R() + s.G() + s.B()) / 3.0);
            }
            float transmit = 0.0f;
            if (const auto * tr =
                    ecm.Component<::gz::sim::components::Transparency>(ent)) {
                transmit = static_cast<float>(tr->Data());
            }

            scene->addInstance(type, size, retro, root_node, spec, transmit);
            refs.push_back(ref);
            return true;
        });

    refs_ = std::move(refs);
    visual_count_ = visual_count;
    {
        // Publish the new immutable scene; an in-flight cast keeps its own
        // shared_ptr to the previous one. The version bump tells the GPU
        // backends to re-upload the geometry arrays.
        std::lock_guard<std::mutex> lk(mtx_);
        scene_ = std::move(scene);
        ++scene_version_;
    }

    RCLCPP_INFO(kLogger,
        "raycast scene mirror v%lu: %d instances (%d meshes, %d visuals "
        "skipped) from %zu visuals",
        static_cast<unsigned long>(scene_version_),
        scene_->instanceCount(), scene_->meshCount(), skipped,
        visual_count);
}

void RaycastMirror::buildColumnPoses(std::chrono::nanoseconds scan_end,
                                     std::vector<float> & col_r,
                                     std::vector<float> & col_t) const
{
    // Column m of a spinning lidar is acquired at
    //   t_m = scan_end − T + (m+1)·T/W
    // (measurement ids are in acquisition order; the last column is the
    // freshest, coinciding with the scan trigger). Interpolate the sensor
    // pose at each t_m from the per-tick history: linear position +
    // quaternion SLERP between the bracketing samples — the standard
    // pose-interpolation treatment for rolling-shutter sensors (Lovegrove
    // et al., BMVC 2013; Furgale et al., ICRA 2012 — full splines are for
    // estimation; playback of known sim poses only needs interpolation).
    const int W = params_.W;
    const auto T = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(1.0 / params_.lidar_hz));
    col_r.resize(static_cast<size_t>(9 * W));
    col_t.resize(static_cast<size_t>(3 * W));

    size_t hint = 0;  // history is time-sorted; t_m is increasing in m
    for (int m = 0; m < W; ++m) {
        const auto t_m = scan_end - T + (T * (m + 1)) / W;

        // Find the first history sample at/after t_m, starting from hint.
        while (hint < pose_history_.size() &&
               pose_history_[hint].first < t_m) {
            ++hint;
        }
        ::gz::math::Pose3d pose;
        if (hint == 0) {
            pose = pose_history_.front().second;   // before history: clamp
        } else if (hint >= pose_history_.size()) {
            pose = pose_history_.back().second;    // after history: clamp
        } else {
            const auto & [tb, pb] = pose_history_[hint];
            const auto & [ta, pa] = pose_history_[hint - 1];
            const double span =
                static_cast<double>((tb - ta).count());
            const double a = (span > 0.0)
                ? static_cast<double>((t_m - ta).count()) / span : 1.0;
            pose.Pos() = pa.Pos() + (pb.Pos() - pa.Pos()) * a;
            pose.Rot() = ::gz::math::Quaterniond::Slerp(
                a, pa.Rot(), pb.Rot(), true);
        }

        float r[9], t[3];
        poseToRT(pose, r, t);
        std::memcpy(&col_r[static_cast<size_t>(9 * m)], r, sizeof(r));
        std::memcpy(&col_t[static_cast<size_t>(3 * m)], t, sizeof(t));
    }
}

void RaycastMirror::postUpdate(
    const ::gz::sim::UpdateInfo & info,
    const ::gz::sim::EntityComponentManager & ecm,
    const ::gz::math::Pose3d & sensor_pose)
{
    // Record the sensor pose every tick (cheap) so per-column poses can be
    // interpolated over the scan period; trim history older than one period
    // plus margin.
    if (params_.motion_distortion) {
        const auto now =
            std::chrono::duration_cast<std::chrono::nanoseconds>(info.simTime);
        if (pose_history_.empty() || pose_history_.back().first < now) {
            pose_history_.emplace_back(now, sensor_pose);
        }
        const auto keep = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(1.2 / params_.lidar_hz));
        while (pose_history_.size() > 2 &&
               now - pose_history_.front().first > keep) {
            pose_history_.pop_front();
        }
    }

    // Rebuild the geometry mirror when the visual population changes
    // (spawn/despawn). Pure pose changes are handled per scan below.
    size_t visual_count = 0;
    ecm.Each<::gz::sim::components::Visual,
             ::gz::sim::components::Geometry>(
        [&visual_count](const ::gz::sim::Entity &,
                        const ::gz::sim::components::Visual *,
                        const ::gz::sim::components::Geometry *) -> bool {
            ++visual_count;
            return true;
        });
    if (!scene_ || visual_count != visual_count_) {
        rebuildScene(ecm, visual_count);
    }

    // ── Throttle to lidar_hz on sim time ────────────────────────────────────
    const auto sim_now =
        std::chrono::duration_cast<std::chrono::nanoseconds>(info.simTime);
    const auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(1.0 / params_.lidar_hz));
    if (last_scan_time_.count() >= 0 && sim_now - last_scan_time_ < period) {
        return;
    }
    last_scan_time_ = sim_now;

    // ── Per-scan transforms + sensor pose ────────────────────────────────────
    std::vector<rc::InstanceXform> xforms(refs_.size());
    for (size_t i = 0; i < refs_.size(); ++i) {
        const auto pose =
            ::gz::sim::worldPose(refs_[i].entity, ecm) *
            ::gz::math::Pose3d(::gz::math::Vector3d::Zero, refs_[i].offset);
        float r[9], t[3];
        poseToRT(pose, r, t);
        scene_->computeXform(static_cast<int>(i), r, t, xforms[i]);
    }

    // Per-column poses for motion distortion (needs ≥ 2 history samples to
    // interpolate; falls back to the snapshot pose otherwise).
    std::vector<float> col_r, col_t;
    if (params_.motion_distortion && pose_history_.size() >= 2) {
        buildColumnPoses(sim_now, col_r, col_t);
    }

    // Sun for the NEAR_IR ambient model: first directional light in the
    // world (propagation direction rotated to world frame, diffuse mean as
    // intensity). Ouster's NEAR_IR counts ambient sunlight reflected off
    // the scene; with no sun the channel falls back to ambient-only
    // (nir = albedo). Weights: 0.3 ambient + 0.7·diffuse sun keeps
    // nir ∈ [0, albedo] like the legacy analogue.
    float sun[5] = {0.0f, 0.0f, -1.0f, 0.0f, 1.0f};
    ecm.Each<::gz::sim::components::Light>(
        [&](const ::gz::sim::Entity & ent,
            const ::gz::sim::components::Light * light) -> bool {
            if (light->Data().Type() != sdf::LightType::DIRECTIONAL) {
                return true;
            }
            auto dir = ::gz::sim::worldPose(ent, ecm).Rot().RotateVector(
                light->Data().Direction());
            dir.Normalize();
            const auto & c = light->Data().Diffuse();
            const float diffuse = static_cast<float>(
                (c.R() + c.G() + c.B()) / 3.0);
            sun[0] = static_cast<float>(dir.X());
            sun[1] = static_cast<float>(dir.Y());
            sun[2] = static_cast<float>(dir.Z());
            sun[3] = 0.7f * ((diffuse < 1.0f) ? diffuse : 1.0f);
            sun[4] = 0.3f;
            return false;  // first directional light wins
        });

    {
        std::lock_guard<std::mutex> lk(mtx_);
        job_scene_ = scene_;
        job_scene_version_ = scene_version_;
        job_xforms_ = std::move(xforms);
        poseToRT(sensor_pose, job_sensor_r_, job_sensor_t_);
        job_col_r_ = std::move(col_r);
        job_col_t_ = std::move(col_t);
        std::memcpy(job_sun_, sun, sizeof(sun));
        job_ready_ = true;
    }
    cv_.notify_one();
}

void RaycastMirror::threadFunc()
{
    while (!shutdown_.load(std::memory_order_acquire)) {
        std::shared_ptr<const rc::Scene> scene;
        std::vector<rc::InstanceXform> xforms;
        std::vector<float> col_r, col_t;
        uint64_t version = 0;
        float sr[9], st[3], sun[5];
        {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_.wait(lk, [this] {
                return job_ready_ ||
                       shutdown_.load(std::memory_order_acquire);
            });
            if (shutdown_.load(std::memory_order_acquire)) break;
            job_ready_ = false;
            scene = job_scene_;
            version = job_scene_version_;
            xforms = std::move(job_xforms_);
            col_r = std::move(job_col_r_);
            col_t = std::move(job_col_t_);
            std::memcpy(sr, job_sensor_r_, sizeof(sr));
            std::memcpy(st, job_sensor_t_, sizeof(st));
            std::memcpy(sun, job_sun_, sizeof(sun));
        }
        if (!scene || !proc_) continue;

        const int n = params_.H * params_.W;
        out_.resize(static_cast<size_t>(3 * n));

        rc::ScanParams sp;
        sp.H = params_.H;
        sp.W = params_.W;
        sp.max_range = static_cast<float>(params_.max_range);
        sp.near_clip = static_cast<float>(kNearClip);
        sp.beam_origin_m = static_cast<float>(params_.beam_origin_mm / 1000.0);
        sp.sun_dir[0] = sun[0];
        sp.sun_dir[1] = sun[1];
        sp.sun_dir[2] = sun[2];
        sp.sun_diffuse = sun[3];
        sp.sun_ambient = sun[4];

        // Cast on the active backend — CUDA/HIP/SYCL kernel, or the
        // OpenMP CPU fallback (identical shared math). proc_mtx_
        // serialises against the sim thread's processDepth call.
        {
            std::lock_guard<std::mutex> proc_lk(*proc_mtx_);
            proc_->castScan(scene->view(), version, xforms.data(),
                            params_.beam_alt_f->data(),
                            params_.beam_az_f->data(),
                            sr, st, sp,
                            out_.data(), out_.data() + n,
                            col_r.empty() ? nullptr : col_r.data(),
                            col_t.empty() ? nullptr : col_t.data(),
                            out_.data() + 2 * n);
        }

        if (exch_->publish(out_, 3 * n)) {
            RCLCPP_WARN_THROTTLE(kLogger, throttle_clock_, 5000,
                "%s: dropped raycast frame (PostUpdate didn't drain); "
                "total dropped=%lu", sensor_name_.c_str(),
                static_cast<unsigned long>(exch_->dropped()));
        }
    }
}

}  // namespace gz_gpu_ouster_lidar
