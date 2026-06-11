// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0
//
// Host-side scene builder for the full per-beam raycast mode
// (<ray_mode>raycast</ray_mode>): every Ouster beam is cast as an exact ray
// — calibrated elevation, per-beam azimuth offset, encoder column, and the
// true beam-origin parallax (ray origins offset on the beam-origin circle,
// matching the Ouster XYZ-LUT reconstruction xyz = (r−n)·d̂ + n·[cosθ,sinθ,0])
// — against a mirror of the scene geometry. No rendering, no pixel grid, no
// interpolation of any kind.
//
// The scene is built once into flat arrays (instances + concatenated,
// globally rebased mesh/BVH data — see raycast_math.hpp) so the SAME data
// runs on every backend: the CUDA/HIP/SYCL kernels upload the arrays and
// device-execute the shared rcCastOneRay; the CPU fallback (castScan below)
// OpenMP-parallelises the identical function. Per-scan rigid transforms
// (InstanceXform) are recomputed from entity world poses every scan.
//
// Pure math, no Gazebo / rendering dependencies — unit-testable as-is. The
// plugin owns the ECM→Scene mirroring.

#pragma once

#include "raycast_math.hpp"

#include <cstdint>
#include <vector>

namespace gz_gpu_ouster_lidar {
namespace rc {

/// Non-owning view of a flat scene: exactly what a backend uploads.
struct SceneView {
    const RcInstance * instances = nullptr;
    int n_instances = 0;
    const float * verts = nullptr;
    int n_vert_floats = 0;
    const int * tris = nullptr;
    int n_tri_ints = 0;
    const int * order = nullptr;
    int n_order = 0;
    const MeshBvhNode * nodes = nullptr;
    int n_nodes = 0;
};

/// Flat scene storage. Build with addMesh()/addInstance(); immutable
/// afterwards (the plugin shares it with the cast worker via
/// shared_ptr<const Scene> and rebuilds a fresh one on entity changes).
class Scene {
public:
    /// Append a triangle mesh (instance-local coordinates, scale already
    /// baked in) and build its BVH into the global arrays. Returns the
    /// global root-node index, or -1 for an empty/degenerate mesh.
    int addMesh(const std::vector<float> & verts,
                const std::vector<int> & tris);

    /// Append an instance. For kMesh pass the root node from addMesh().
    /// `retro` is the diffuse reflectance (laser_retro), `spec` the specular
    /// coefficient (visual material specular), `transmit` the transmittance
    /// (visual transparency) — see rcApparentReflectance / rcCastOneRay.
    /// Returns the instance index.
    int addInstance(GeomType type, const float size[3], float retro,
                    int root_node = -1, float spec = 0.0f,
                    float transmit = 0.0f);

    SceneView view() const;

    int instanceCount() const { return static_cast<int>(instances_.size()); }
    int meshCount() const { return mesh_count_; }
    const RcInstance & instance(int i) const
    {
        return instances_[static_cast<size_t>(i)];
    }

    /// Fill an InstanceXform from the instance's local→world pose
    /// (row-major rotation r_lw, translation t_lw); recomputes the world
    /// AABB from the instance's cached local bounds.
    void computeXform(int instance_idx,
                      const float r_lw[9], const float t_lw[3],
                      InstanceXform & out) const;

private:
    struct LocalBounds {
        float bmin[3];
        float bmax[3];
    };

    std::vector<RcInstance> instances_;
    std::vector<LocalBounds> bounds_;     // parallel to instances_
    std::vector<float> verts_;
    std::vector<int> tris_;
    std::vector<int> order_;
    std::vector<MeshBvhNode> nodes_;
    // Per-root local AABB of each added mesh, keyed by root node index.
    int mesh_count_ = 0;
};

/// CPU reference cast (the always-available fallback; OpenMP-parallel).
/// Output layout matches the rest of the pipeline: row = beam, column =
/// Ouster measurement id (m = 0 forward, azimuth decreasing with m).
/// `sensor_r`/`sensor_t` is the sensor→world pose (row-major rotation +
/// translation).
///
/// range_out[H×W]: reported Ouster range in metres (+inf for a miss) — the
/// value r such that the standard XYZ LUT xyz = (r−n)·d̂ + n·[cosθ,sinθ,0]
/// reconstructs the true hit point. retro_out[H×W] (optional, may be null):
/// APPARENT reflectance of the reported return — kd·cos(α) + ks·cos(2α)⁸,
/// transmission-weighted for glass (see rcCastOneRay) — 0 when the material
/// is unset or on a miss.
void castScan(const SceneView & scene,
              const InstanceXform * xforms,
              const float * beam_alt_deg,
              const float * beam_az_deg,
              const float sensor_r[9], const float sensor_t[3],
              const ScanParams & sp,
              float * range_out, float * retro_out,
              const float * col_r = nullptr, const float * col_t = nullptr,
              float * nir_out = nullptr);

}  // namespace rc
}  // namespace gz_gpu_ouster_lidar
