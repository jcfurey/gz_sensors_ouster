// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0
//
// CPU backend: always available. Wraps the free functions in
// ray_processor_cpu_impl.cpp. Used as the last-resort fallback when every
// GPU backend either isn't compiled in or fails its runtime device probe.

#include "backend.hpp"
#include "ray_processor_cpu_impl.hpp"

#include <memory>

namespace gz_gpu_ouster_lidar {

namespace {

class CpuBackend final : public Backend {
public:
    explicit CpuBackend(uint64_t seed) : seed_(seed) {}

    void processRaw(
        const float * raw_host,
        const float * beam_alt_host,
        const float * beam_az_host,
        const ResampleParams & rp,
        uint32_t *    range_out,
        uint16_t *    signal_out,
        uint8_t *     reflectivity_out,
        uint16_t *    nearir_out,
        const RayProcessParams & pp) override
    {
        processRawCpu(raw_host, beam_alt_host, beam_az_host, rp,
                      range_out, signal_out, reflectivity_out, nearir_out,
                      pp, frameSeed());
    }

    void processDepth(
        const float * depth_host,
        const float * retro_host,
        uint32_t *    range_out,
        uint16_t *    signal_out,
        uint8_t *     reflectivity_out,
        uint16_t *    nearir_out,
        const RayProcessParams & pp,
        const float * nir_host) override
    {
        processCpu(depth_host, retro_host,
                   range_out, signal_out, reflectivity_out, nearir_out,
                   pp, frameSeed(), nir_host);
    }

    void castScan(
        const rc::SceneView & scene,
        uint64_t /*scene_version*/,
        const rc::InstanceXform * xforms,
        const float * beam_alt_deg,
        const float * beam_az_deg,
        const float sensor_r[9],
        const float sensor_t[3],
        const rc::ScanParams & sp,
        float * range_out,
        float * retro_out,
        const float * col_r,
        const float * col_t,
        float * nir_out) override
    {
        // Top-level BVH over the instances' world AABBs, rebuilt from this
        // scan's transforms (cheap: O(n log n) over instances, versus the
        // O(H*W*n_instances) AABB tests it saves in the cast below). Reuses
        // tlas_'s capacity across scans. Below kTlasMinInstances it stays
        // empty and rc::castScan falls back to the linear scan.
        rc::buildTlas(xforms, scene.n_instances, tlas_);

        // OpenMP-parallel reference implementation; no upload, no cache.
        rc::castScan(scene, xforms, beam_alt_deg, beam_az_deg,
                     sensor_r, sensor_t, sp, range_out, retro_out,
                     col_r, col_t, nir_out,
                     tlas_.nodes.data(), tlas_.order.data(),
                     static_cast<int>(tlas_.nodes.size()));
    }

    const char * name() const override { return "cpu"; }

private:
    uint64_t frameSeed()
    {
        if (seed_ == 0) return 0;
        // Explicit seeds define a reproducible sequence, not a frozen frame.
        // SplitMix's Weyl increment makes adjacent frame seeds independent.
        uint64_t x = seed_ + frame_counter_++ * 0x9E3779B97F4A7C15ULL;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        return x ^ (x >> 31);
    }

    uint64_t seed_;
    uint64_t frame_counter_ = 0;
    rc::Tlas tlas_;   // per-scan top-level BVH; capacity reused
};

}  // namespace

std::unique_ptr<Backend> makeCpuBackend(uint64_t seed)
{
    return std::make_unique<CpuBackend>(seed);
}

}  // namespace gz_gpu_ouster_lidar
