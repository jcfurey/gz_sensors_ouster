// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0
//
// Self-contained CPU raycaster for the full per-beam raycast mode
// (<ray_mode>raycast</ray_mode>): every Ouster beam is cast as an exact ray
// — calibrated elevation, per-beam azimuth offset, encoder column, and the
// true beam-origin parallax (ray origins offset on the beam-origin circle,
// matching the Ouster XYZ-LUT reconstruction xyz = (r−n)·d̂ + n·[cosθ,sinθ,0])
// — against a mirror of the scene geometry. No rendering, no pixel grid, no
// interpolation of any kind.
//
// The scene is split into immutable geometry (Scene: primitive parameters
// and triangle meshes with a per-mesh BVH, built once) and per-scan rigid
// transforms (InstanceXform, recomputed from entity world poses every scan).
// castScan() is OpenMP-parallel over the H×W output rays.
//
// Pure math, no Gazebo / rendering dependencies — unit-testable as-is. The
// plugin owns the ECM→Scene mirroring.

#pragma once

#include <cstdint>
#include <vector>

namespace gz_gpu_ouster_lidar {
namespace rc {

enum class GeomType : int {
    kPlane = 0,    ///< z = 0 plane, finite half-extents size[0..1] (0 → infinite)
    kBox,          ///< axis-aligned box, half-extents size[0..2]
    kSphere,       ///< radius size[0]
    kCylinder,     ///< z axis, radius size[0], half-length size[1]
    kMesh,         ///< triangle mesh, index `mesh` into Scene::meshes
};

struct MeshBvhNode {
    float bmin[3];
    float bmax[3];
    int left = -1;    ///< child node index, -1 for leaf
    int right = -1;
    int first = 0;    ///< leaf: first index into MeshData::order
    int count = 0;    ///< leaf: triangle count
};

/// Immutable triangle mesh in instance-local coordinates (scale baked in).
struct MeshData {
    std::vector<float> verts;       ///< xyz triples
    std::vector<int> tris;          ///< vertex-index triples
    std::vector<int> order;         ///< BVH leaf → triangle index permutation
    std::vector<MeshBvhNode> nodes; ///< nodes[0] is the root
    float bmin[3] = {0, 0, 0};      ///< local AABB (set by buildMeshBvh)
    float bmax[3] = {0, 0, 0};
};

/// Immutable per-instance geometry. Transforms live in InstanceXform.
struct Instance {
    GeomType type = GeomType::kBox;
    float size[3] = {0, 0, 0};
    int mesh = -1;
    float retro = 0.0f;            ///< laser_retro of the visual (0 = unset)
    float local_bmin[3] = {0, 0, 0};  ///< local AABB (set by finalizeInstance)
    float local_bmax[3] = {0, 0, 0};
};

struct Scene {
    std::vector<Instance> instances;
    std::vector<MeshData> meshes;
};

/// Per-scan rigid transform of one instance.
struct InstanceXform {
    float r[9];     ///< world→local rotation (row-major)
    float t[3];     ///< world→local translation: p_l = r·p_w + t
    float bmin[3];  ///< world-space AABB for the broad-phase reject
    float bmax[3];
};

struct ScanParams {
    int H = 0;                 ///< beam count (output rows)
    int W = 0;                 ///< columns per frame (output cols)
    float max_range = 120.0f;  ///< metres; hits beyond this are misses
    float near_clip = 0.3f;    ///< metres; hits closer than this are ignored
    float beam_origin_m = 0.0f;  ///< lidar-origin→beam-origin offset (metres)
};

/// Build the BVH (and local AABB) for a mesh whose verts/tris are filled in.
/// Returns false for an empty/degenerate mesh.
bool buildMeshBvh(MeshData & mesh);

/// Compute the instance's local AABB from its geometry (meshes resolved
/// through `scene`). Call once after the instance geometry is set.
void finalizeInstance(const Scene & scene, Instance & inst);

/// Fill an InstanceXform from the instance's local→world pose
/// (row-major rotation r_lw, translation t_lw).
void computeXform(const Instance & inst,
                  const float r_lw[9], const float t_lw[3],
                  InstanceXform & out);

/// Cast the full scan. Output layout matches the rest of the pipeline:
/// row = beam, column = Ouster measurement id (m = 0 forward, azimuth
/// decreasing with m). `sensor_r`/`sensor_t` is the sensor→world pose
/// (row-major rotation + translation).
///
/// range_out[H×W]: reported Ouster range in metres (+inf for a miss) — the
/// value r such that the standard XYZ LUT xyz = (r−n)·d̂ + n·[cosθ,sinθ,0]
/// reconstructs the true hit point. retro_out[H×W] (optional, may be null):
/// laser_retro of the nearest hit instance, 0 when unset or on a miss.
void castScan(const Scene & scene,
              const InstanceXform * xforms,
              const float * beam_alt_deg,
              const float * beam_az_deg,
              const float sensor_r[9], const float sensor_t[3],
              const ScanParams & sp,
              float * range_out, float * retro_out);

}  // namespace rc
}  // namespace gz_gpu_ouster_lidar
