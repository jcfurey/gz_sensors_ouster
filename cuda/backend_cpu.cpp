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
                      pp, seed_);
    }

    void processDepth(
        const float * depth_host,
        const float * retro_host,
        uint32_t *    range_out,
        uint16_t *    signal_out,
        uint8_t *     reflectivity_out,
        uint16_t *    nearir_out,
        const RayProcessParams & pp) override
    {
        processCpu(depth_host, retro_host,
                   range_out, signal_out, reflectivity_out, nearir_out,
                   pp, seed_);
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
        float * retro_out) override
    {
        // OpenMP-parallel reference implementation; no upload, no cache.
        rc::castScan(scene, xforms, beam_alt_deg, beam_az_deg,
                     sensor_r, sensor_t, sp, range_out, retro_out);
    }

    const char * name() const override { return "cpu"; }

private:
    uint64_t seed_;
};

}  // namespace

std::unique_ptr<Backend> makeCpuBackend(uint64_t seed)
{
    return std::make_unique<CpuBackend>(seed);
}

}  // namespace gz_gpu_ouster_lidar
