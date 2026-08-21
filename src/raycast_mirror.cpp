// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0

#include "raycast_mirror.hpp"

#include "gz_gpu_ouster_lidar/ray_processor.hpp"

#include <algorithm>
#include <cstring>
#include <exception>
#include <filesystem>
#include <limits>
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
#include <gz/common/Image.hh>
#include <gz/common/SubMesh.hh>
#include <gz/math/Matrix3.hh>

// sdf/Geometry.hh only forward-declares the shape classes; the accessors
// (BoxShape()->Size(), ...) need the full definitions on every distro.
#include <sdf/Box.hh>
#include <sdf/Cylinder.hh>
#include <sdf/Geometry.hh>
#include <sdf/Mesh.hh>
#include <sdf/Plane.hh>
#include <sdf/Pbr.hh>
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
                  std::vector<float> & verts, std::vector<int> & tris,
                  std::vector<float> & texcoords, bool & has_texcoords)
{
    has_texcoords = true;
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
            if (sm->HasTexCoord(v)) {
                const auto uv = sm->TexCoord(v);
                texcoords.push_back(static_cast<float>(uv.X()));
                texcoords.push_back(static_cast<float>(uv.Y()));
            } else {
                texcoords.push_back(0.0f);
                texcoords.push_back(0.0f);
                has_texcoords = false;
            }
        }
        for (unsigned int k = 0; k + 2 < sm->IndexCount(); k += 3) {
            tris.push_back(base + static_cast<int>(sm->Index(k)));
            tris.push_back(base + static_cast<int>(sm->Index(k + 1)));
            tris.push_back(base + static_cast<int>(sm->Index(k + 2)));
        }
    }
    return !tris.empty();
}

/// Standard SDF has no LiDAR response-map field. Associate one without
/// vendor-specific tags by looking beside the visible PBR albedo map:
///   brick.png -> brick.ouster.png
/// The companion is RGBA8: diffuse-865nm, passive-NIR, specular, opacity.
std::string responseMapPath(const sdf::Material & material)
{
    const sdf::Pbr * pbr = material.PbrMaterial();
    if (!pbr) return {};
    const sdf::PbrWorkflow * workflow =
        pbr->Workflow(sdf::PbrWorkflowType::METAL);
    if (!workflow) workflow = pbr->Workflow(sdf::PbrWorkflowType::SPECULAR);
    if (!workflow || workflow->AlbedoMap().empty()) return {};

    const std::string albedo = ::gz::sim::asFullPath(
        workflow->AlbedoMap(), material.FilePath());
    if (albedo.empty()) return {};
    std::filesystem::path path(albedo);
    const std::string extension = path.extension().string();
    path.replace_filename(path.stem().string() + ".ouster" + extension);
    std::error_code ec;
    return std::filesystem::is_regular_file(path, ec) ? path.string() : std::string{};
}

struct CachedResponseTexture {
    int offset = -1;
    int width = 0;
    int height = 0;
};

struct CachedMesh {
    int root_node = -1;
    bool has_texcoords = false;
};

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
                          ProcessedFrameExchange * exch)
{
    params_ = p;
    proc_ = proc;
    exch_ = exch;
    thread_ = std::thread(&RaycastMirror::threadFunc, this);
}

void RaycastMirror::stop()
{
    // Set shutdown_ under mtx_ so it can't be missed between the worker's
    // predicate check and its cv_.wait() — otherwise a stop() racing that
    // window would notify before the worker blocks, and the worker would sleep
    // forever, hanging the join below.
    {
        std::lock_guard<std::mutex> lk(mtx_);
        shutdown_.store(true, std::memory_order_release);
    }
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
    std::unordered_map<std::string, CachedMesh> mesh_cache;
    std::unordered_map<std::string, CachedResponseTexture> response_cache;
    int skipped = 0;
    int response_instances = 0;

    ecm.Each<::gz::sim::components::Visual,
             ::gz::sim::components::Geometry>(
        [&](const ::gz::sim::Entity & ent,
            const ::gz::sim::components::Visual *,
            const ::gz::sim::components::Geometry * geom) -> bool {
            const sdf::Geometry & g = geom->Data();
            rc::GeomType type = rc::GeomType::kBox;
            float size[3] = {0.0f, 0.0f, 0.0f};
            int root_node = -1;
            bool has_texcoords = true;  // analytic primitives derive UVs
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
                        std::vector<float> texcoords;
                        bool mesh_has_texcoords = false;
                        if (!appendGzMesh(*gz_mesh, scale, verts, tris,
                                         texcoords, mesh_has_texcoords)) {
                            ++skipped;
                            return true;
                        }
                        const int root = scene->addMesh(verts, tris,
                                                        texcoords);
                        if (root < 0) {
                            ++skipped;
                            return true;
                        }
                        it = mesh_cache.emplace(
                            key, CachedMesh{root, mesh_has_texcoords}).first;
                    }
                    type = rc::GeomType::kMesh;
                    root_node = it->second.root_node;
                    has_texcoords = it->second.has_texcoords;
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
            const auto * mat =
                ecm.Component<::gz::sim::components::Material>(ent);
            if (mat) {
                const auto & s = mat->Data().Specular();
                spec = static_cast<float>((s.R() + s.G() + s.B()) / 3.0);
            }
            float transmit = 0.0f;
            if (const auto * tr =
                    ecm.Component<::gz::sim::components::Transparency>(ent)) {
                transmit = static_cast<float>(tr->Data());
            }

            CachedResponseTexture response;
            if (mat) {
                const std::string response_path =
                    responseMapPath(mat->Data());
                if (!response_path.empty() && !has_texcoords) {
                    RCLCPP_WARN(kLogger,
                        "raycast: response map '%s' ignored because mesh "
                        "visual %lu has no texture coordinates",
                        response_path.c_str(),
                        static_cast<unsigned long>(ent));
                } else if (!response_path.empty()) {
                    auto it = response_cache.find(response_path);
                    if (it == response_cache.end()) {
                        ::gz::common::Image image;
                        CachedResponseTexture loaded;
                        const bool dimensions_fit =
                            image.Load(response_path) == 0 && image.Valid() &&
                            image.Width() <= static_cast<unsigned int>(
                                std::numeric_limits<int>::max()) &&
                            image.Height() <= static_cast<unsigned int>(
                                std::numeric_limits<int>::max());
                        if (dimensions_fit) {
                            const auto rgba = image.RGBAData();
                            loaded.width = static_cast<int>(image.Width());
                            loaded.height = static_cast<int>(image.Height());
                            loaded.offset = scene->addResponseTexture(
                                loaded.width, loaded.height, rgba);
                        }
                        if (loaded.offset < 0) {
                            RCLCPP_WARN(kLogger,
                                "raycast: cannot load RGBA response map '%s'; "
                                "using scalar material response",
                                response_path.c_str());
                        }
                        it = response_cache.emplace(response_path,
                                                    loaded).first;
                    }
                    response = it->second;
                    if (response.offset >= 0) ++response_instances;
                }
            }

            scene->addInstance(type, size, retro, root_node, spec, transmit,
                               lr != nullptr, response.offset,
                               response.width, response.height);
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
        "raycast scene mirror v%lu: %d instances (%d meshes, %d response "
        "maps on %d instances, %d visuals skipped) from %zu visuals",
        static_cast<unsigned long>(scene_version_),
        scene_->instanceCount(), scene_->meshCount(),
        static_cast<int>(std::count_if(
            response_cache.begin(), response_cache.end(),
            [](const auto & item) { return item.second.offset >= 0; })),
        response_instances, skipped,
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
    const ::gz::math::Pose3d & sensor_pose,
    uint64_t epoch,
    const RayProcessParams & process_params)
{
    // Record the sensor pose every tick (cheap) so per-column poses can be
    // interpolated over the scan period; trim history older than one period
    // plus margin.
    if (params_.motion_distortion) {
        const auto now =
            std::chrono::duration_cast<std::chrono::nanoseconds>(info.simTime);
        // Sim-time rewind (world reset): the history holds poses stamped ahead
        // of `now`, so the append gate below would never admit new samples and
        // the trim delta would go negative — buildColumnPoses would then clamp
        // every column to the stale pre-reset pose for the whole rewound span.
        // Drop the history and start fresh from the post-reset pose.
        if (!pose_history_.empty() && now < pose_history_.back().first) {
            pose_history_.clear();
        }
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

    // Evaluate component changes on every tick to catch edge mutations that
    // gz-sim clears at the end of each simulation step.
    ecm.Each<::gz::sim::components::Visual,
             ::gz::sim::components::Geometry>(
        [&](const ::gz::sim::Entity & ent,
            const ::gz::sim::components::Visual *,
            const ::gz::sim::components::Geometry *) -> bool {
            auto changed = [&](::gz::sim::ComponentTypeId type) {
                return ecm.ComponentState(ent, type) !=
                    ::gz::sim::ComponentState::NoChange;
            };
            if (changed(::gz::sim::components::Geometry::typeId) ||
                changed(::gz::sim::components::LaserRetro::typeId) ||
                changed(::gz::sim::components::Material::typeId) ||
                changed(::gz::sim::components::Transparency::typeId)) {
                pending_rebuild_ = true;
                return false;  // early stop
            }
            return true;
        });

    // Gate on sim time before traversing the scene. The deadline advances
    // from its previous target rather than re-anchoring at the current tick,
    // so a physics step that does not divide the scan period produces bounded
    // 8/12 ms-style jitter instead of a permanently low average rate. The
    // frame is stamped with sim_now because that is when geometry is captured.
    const auto sim_now =
        std::chrono::duration_cast<std::chrono::nanoseconds>(info.simTime);
    const auto gate = scan_gate_.advance(sim_now, periodFromHz(params_.lidar_hz));
    if (!gate.due) return;

    // Rebuild when visual identity, geometry/material state, or pending changes occur.
    size_t visual_count = 0;
    uint64_t visual_signature = 0;
    ecm.Each<::gz::sim::components::Visual,
             ::gz::sim::components::Geometry>(
        [&](const ::gz::sim::Entity & ent,
            const ::gz::sim::components::Visual *,
            const ::gz::sim::components::Geometry * geom) -> bool {
            ++visual_count;
            uint64_t x = static_cast<uint64_t>(ent) ^
                (static_cast<uint64_t>(geom->Data().Type()) << 56);
            x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
            x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
            visual_signature ^= x ^ (x >> 31);
            return true;
        });
    if (!scene_ || visual_count != visual_count_ ||
        visual_signature != visual_signature_ || pending_rebuild_) {
        rebuildScene(ecm, visual_count);
        visual_signature_ = visual_signature;
        pending_rebuild_ = false;
    }

    // ── Per-scan transforms + sensor pose ────────────────────────────────────
    // Reclaim the worker's previous storage when the job slot is idle.
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (!job_ready_) {
            if (post_xforms_.empty()) post_xforms_.swap(job_xforms_);
            if (post_col_r_.empty()) post_col_r_.swap(job_col_r_);
            if (post_col_t_.empty()) post_col_t_.swap(job_col_t_);
        }
    }
    post_xforms_.resize(refs_.size());
    for (size_t i = 0; i < refs_.size(); ++i) {
        const auto pose =
            ::gz::sim::worldPose(refs_[i].entity, ecm) *
            ::gz::math::Pose3d(::gz::math::Vector3d::Zero, refs_[i].offset);
        float r[9], t[3];
        poseToRT(pose, r, t);
        scene_->computeXform(static_cast<int>(i), r, t, post_xforms_[i]);
    }

    // Per-column poses for motion distortion (needs ≥ 2 history samples to
    // interpolate; falls back to the snapshot pose otherwise).
    post_col_r_.clear();
    post_col_t_.clear();
    if (params_.motion_distortion && pose_history_.size() >= 2) {
        buildColumnPoses(sim_now, post_col_r_, post_col_t_);
    }

    // Sun for the NEAR_IR ambient model: first directional light in the
    // world (propagation direction rotated to world frame, diffuse mean as
    // intensity). Ouster's NEAR_IR counts ambient sunlight reflected off
    // the scene; with no sun the channel falls back to ambient-only
    // (nir = albedo). Weights: 0.3 ambient + 0.7·diffuse sun keeps
    // hit values in [0.3·albedo, albedo] (full Lambert shadow → facing the
    // sun), capped at the legacy analogue's albedo ceiling; misses stay 0.
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

    // Smoke / dust / fog volumes, re-gathered every scan: particle emitters
    // ride on links that move, and both their `emitting` flag and an
    // authored volume's density can change mid-run.
    post_obscurants_.clear();
    if (params_.obscurants != nullptr && params_.obscurants->active()) {
        const size_t dropped = gatherObscurants(
            *params_.obscurants, ecm, sensor_pose.Pos(), post_obscurants_);
        if (dropped > 0) {
            RCLCPP_WARN_THROTTLE(kLogger, throttle_clock_, 10000,
                "%s: %zu obscurant volume(s) beyond the %d-volume cap were "
                "dropped this scan; the nearest %d are kept, so more distant "
                "smoke will not obscure", sensor_name_.c_str(), dropped,
                rc::kMaxObscurants, rc::kMaxObscurants);
        }
    }

    bool overwrote_job = false;
    uint64_t dropped_jobs = 0;
    {
        std::lock_guard<std::mutex> lk(mtx_);
        overwrote_job = job_ready_;
        if (overwrote_job) {
            dropped_jobs = dropped_jobs_.fetch_add(
                1, std::memory_order_relaxed) + 1;
        }
        job_scene_ = scene_;
        job_scene_version_ = scene_version_;
        job_xforms_.swap(post_xforms_);
        poseToRT(sensor_pose, job_sensor_r_, job_sensor_t_);
        job_col_r_.swap(post_col_r_);
        job_col_t_.swap(post_col_t_);
        job_obscurants_.swap(post_obscurants_);
        job_rng_salt_ = ++scan_counter_;
        job_metadata_ = FrameMetadata{sim_now.count(), epoch};
        job_process_params_ = process_params;
        std::memcpy(job_sun_, sun, sizeof(sun));
        job_ready_ = true;
    }
    cv_.notify_one();
    if (overwrote_job) {
        RCLCPP_WARN_THROTTLE(kLogger, throttle_clock_, 5000,
            "%s: raycast worker replaced a complete pending job; "
            "total dropped=%lu", sensor_name_.c_str(),
            static_cast<unsigned long>(dropped_jobs));
    }
}

void RaycastMirror::threadFunc()
{
    // Persistent locals exchange storage with the single job slot. Together
    // with the producer staging vectors this becomes an allocation-free
    // triple-buffer after the first two scans.
    std::vector<rc::InstanceXform> xforms;
    std::vector<float> col_r, col_t;
    while (!shutdown_.load(std::memory_order_acquire)) {
        std::shared_ptr<const rc::Scene> scene;
        FrameMetadata metadata;
        RayProcessParams process_params;
        uint64_t version = 0;
        float sr[9], st[3], sun[5];
        // Obscurants are copied rather than swapped: at most
        // rc::kMaxObscurants entries (~576 B), and the kernel wants them in
        // ScanParams by value anyway.
        rc::RcObscurant obscurants[rc::kMaxObscurants];
        int n_obscurants = 0;
        uint32_t rng_salt = 0;
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
            xforms.swap(job_xforms_);
            col_r.swap(job_col_r_);
            col_t.swap(job_col_t_);
            metadata = job_metadata_;
            process_params = job_process_params_;
            std::memcpy(sr, job_sensor_r_, sizeof(sr));
            std::memcpy(st, job_sensor_t_, sizeof(st));
            std::memcpy(sun, job_sun_, sizeof(sun));
            n_obscurants = static_cast<int>(std::min<size_t>(
                job_obscurants_.size(), rc::kMaxObscurants));
            std::copy_n(job_obscurants_.begin(), n_obscurants, obscurants);
            rng_salt = job_rng_salt_;
        }
        if (!scene || !proc_) continue;

        try {
            const int n = params_.H * params_.W;
            range_out_.resize(static_cast<size_t>(n));
            signal_out_.resize(static_cast<size_t>(n));
            reflectivity_out_.resize(static_cast<size_t>(n));
            nearir_out_.resize(static_cast<size_t>(n));
            scratch_.resize(static_cast<size_t>(3 * n));

            rc::ScanParams sp;
            sp.H = params_.H;
            sp.W = params_.W;
            sp.max_range = static_cast<float>(params_.max_range);
            sp.near_clip = static_cast<float>(params_.min_range);
            sp.beam_origin_m =
                static_cast<float>(params_.beam_origin_mm / 1000.0);
            sp.sun_dir[0] = sun[0];
            sp.sun_dir[1] = sun[1];
            sp.sun_dir[2] = sun[2];
            sp.sun_diffuse = sun[3];
            sp.sun_ambient = sun[4];
            sp.fallback_retro =
                rpmath::reflectivityByteToRetro(process_params.base_reflectivity);
            sp.base_signal = process_params.base_signal;
            sp.n_obscurants = n_obscurants;
            std::copy_n(obscurants, n_obscurants, sp.obscurants);
            sp.rng_salt = rng_salt;
            if (params_.obscurants != nullptr) {
                sp.pulse_gate_m =
                    static_cast<float>(params_.obscurants->pulse_gate_m);
            }

            // CUDA launches raycast and channel/noise kernels back-to-back on
            // one stream, with no host round trip for depth/retro/NIR. Other
            // backends use the exact fallback composition through scratch_
            // until they grow a native fused override.
            proc_->castScanProcessed(
                scene->view(), version, xforms.data(),
                params_.beam_alt_f->data(), params_.beam_az_f->data(),
                sr, st, sp,
                range_out_.data(), signal_out_.data(),
                reflectivity_out_.data(), nearir_out_.data(),
                process_params,
                scratch_.data(), scratch_.data() + n,
                scratch_.data() + 2 * n,
                col_r.empty() ? nullptr : col_r.data(),
                col_t.empty() ? nullptr : col_t.data());

            if (exch_->publish(range_out_, signal_out_, reflectivity_out_,
                               nearir_out_, n, metadata)) {
                RCLCPP_WARN_THROTTLE(kLogger, throttle_clock_, 5000,
                    "%s: dropped raycast frame (PostUpdate didn't drain); "
                    "total dropped=%lu", sensor_name_.c_str(),
                    static_cast<unsigned long>(exch_->dropped()));
            }
        } catch (const std::exception & e) {
            // Moving fused GPU work off PostUpdate also moves it outside that
            // callback's exception guard. Never let a device/allocation error
            // escape a std::thread entry point: that would call std::terminate
            // and kill the whole Gazebo server.
            RCLCPP_ERROR_THROTTLE(kLogger, throttle_clock_, 5000,
                "%s: raycast worker failed: %s",
                sensor_name_.c_str(), e.what());
        } catch (...) {
            RCLCPP_ERROR_THROTTLE(kLogger, throttle_clock_, 5000,
                "%s: raycast worker failed (non-std exception)",
                sensor_name_.c_str());
        }
    }
}

}  // namespace gz_gpu_ouster_lidar
