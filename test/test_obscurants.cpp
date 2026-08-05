// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0
//
// Participating media (smoke / dust / fog) in the raycast ray mode: volume
// geometry, Beer–Lambert extinction against the closed-form value, exact
// inversion of the optical-depth profile, the medium's own backscatter
// return, NEAR_IR airlight, and the downstream consequences the model is
// supposed to produce for free (dimmer signal, lower reflectivity byte,
// targets falling past the detection limit).
//
// All of it runs through the shared rc:: math the CUDA/HIP/SYCL kernels
// device-compile, so these assertions cover every backend.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "raycast_scene.hpp"

namespace gz_gpu_ouster_lidar {

namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();
const float kIdentityR[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
const float kZeroT[3] = {0, 0, 0};

/// Lidar ratio large enough that the medium's own return is negligible,
/// isolating the extinction half of the model.
constexpr float kNoBackscatter = 1.0e6f;

rc::ScanParams baseParams()
{
    rc::ScanParams sp;
    sp.H = 1;
    sp.W = 1;
    sp.max_range = 120.0f;
    sp.near_clip = 0.3f;
    return sp;
}

/// Axis-aligned obscurant centred on the +x axis, identity rotation.
rc::RcObscurant obscurantAt(rc::ObscurantType type, float cx,
                            float hx, float hy, float hz, float sigma,
                            float lidar_ratio = 50.0f, float albedo = 0.8f,
                            float ms_factor = 1.0f)
{
    rc::RcObscurant ob;
    for (int i = 0; i < 9; ++i) ob.r[i] = kIdentityR[i];
    ob.t[0] = -cx;
    ob.t[1] = 0.0f;
    ob.t[2] = 0.0f;
    ob.half[0] = hx;
    ob.half[1] = hy;
    ob.half[2] = hz;
    ob.sigma = sigma;
    ob.lidar_ratio = lidar_ratio;
    ob.albedo = albedo;
    ob.ms_factor = ms_factor;
    ob.type = type;
    return ob;
}

void addObscurant(rc::ScanParams & sp, const rc::RcObscurant & ob)
{
    ASSERT_LT(sp.n_obscurants, rc::kMaxObscurants);
    sp.obscurants[sp.n_obscurants++] = ob;
}

struct RayResult {
    float range = 0.0f;
    float retro = 0.0f;
    float nir = 0.0f;
};

/// Cast the single beam (elevation 0, azimuth 0 → straight down +x) of a
/// 1×1 scan. W = 1 makes deg_per_col 360 and column 0 the forward encoder
/// position, so this is the plain forward ray from the origin.
RayResult castOne(const rc::Scene & scene,
                  const std::vector<rc::InstanceXform> & xf,
                  const rc::ScanParams & sp_in)
{
    rc::ScanParams sp = sp_in;
    sp.H = 1;
    sp.W = 1;
    const float alt[1] = {0.0f};
    const float az[1] = {0.0f};
    RayResult r;
    rc::castScan(scene.view(), xf.empty() ? nullptr : xf.data(), alt, az,
                 kIdentityR, kZeroT, sp, &r.range, &r.retro,
                 nullptr, nullptr, &r.nir);
    return r;
}

/// A wall: a box instance whose front face sits at x = `face_x`.
void makeWall(rc::Scene & scene, std::vector<rc::InstanceXform> & xf,
              float face_x, float retro)
{
    const float half[3] = {0.5f, 8.0f, 8.0f};
    const int idx = scene.addInstance(rc::GeomType::kBox, half, retro);
    const float t[3] = {face_x + 0.5f, 0.0f, 0.0f};
    rc::InstanceXform x;
    scene.computeXform(idx, kIdentityR, t, x);
    xf.push_back(x);
}

const rc::RcV3 kOrigin{0.0f, 0.0f, 0.0f};
const rc::RcV3 kForward{1.0f, 0.0f, 0.0f};

}  // namespace

// ── Volume geometry ──────────────────────────────────────────────────────────

TEST(ObscurantGeometry, BoxSpanIsEntryAndExit)
{
    const auto ob = obscurantAt(rc::ObscurantType::kBox, 10.0f,
                                2.0f, 5.0f, 5.0f, 0.1f);
    float a = 0.0f, b = 0.0f;
    ASSERT_TRUE(rc::rcObscurantSpan(ob, kOrigin, kForward, 0.0f, 100.0f,
                                    a, b));
    EXPECT_NEAR(a, 8.0f, 1e-4f);
    EXPECT_NEAR(b, 12.0f, 1e-4f);
}

TEST(ObscurantGeometry, EllipsoidSpanUsesSemiAxes)
{
    // Semi-axis 3 along x, centre at 10 → chord [7, 13] through the centre.
    const auto ob = obscurantAt(rc::ObscurantType::kEllipsoid, 10.0f,
                                3.0f, 1.0f, 1.0f, 0.1f);
    float a = 0.0f, b = 0.0f;
    ASSERT_TRUE(rc::rcObscurantSpan(ob, kOrigin, kForward, 0.0f, 100.0f,
                                    a, b));
    EXPECT_NEAR(a, 7.0f, 1e-3f);
    EXPECT_NEAR(b, 13.0f, 1e-3f);

    // A ray offset beyond the y semi-axis misses entirely.
    const rc::RcV3 off{0.0f, 2.0f, 0.0f};
    EXPECT_FALSE(rc::rcObscurantSpan(ob, off, kForward, 0.0f, 100.0f, a, b));
}

TEST(ObscurantGeometry, CylinderSpanClipsToBothCapsAndTube)
{
    // z-axis cylinder, radius 2 in x/y, half-length 4.
    const auto ob = obscurantAt(rc::ObscurantType::kCylinder, 10.0f,
                                2.0f, 2.0f, 4.0f, 0.1f);
    float a = 0.0f, b = 0.0f;
    ASSERT_TRUE(rc::rcObscurantSpan(ob, kOrigin, kForward, 0.0f, 100.0f,
                                    a, b));
    EXPECT_NEAR(a, 8.0f, 1e-3f);
    EXPECT_NEAR(b, 12.0f, 1e-3f);

    // Straight up the axis: inside the tube for all t, clipped by the caps.
    const rc::RcV3 up{10.0f, 0.0f, -20.0f};
    const rc::RcV3 dz{0.0f, 0.0f, 1.0f};
    ASSERT_TRUE(rc::rcObscurantSpan(ob, up, dz, 0.0f, 100.0f, a, b));
    EXPECT_NEAR(a, 16.0f, 1e-3f);
    EXPECT_NEAR(b, 24.0f, 1e-3f);
}

TEST(ObscurantGeometry, FlatEmitterVolumeStillHasThickness)
{
    // gz particle emitters are routinely authored flat (<size>10 10 0</size>).
    // A zero half-extent must not divide by zero or swallow the ray.
    const auto ob = obscurantAt(rc::ObscurantType::kBox, 10.0f,
                                2.0f, 5.0f, 0.0f, 0.1f);
    float a = 0.0f, b = 0.0f;
    ASSERT_TRUE(rc::rcObscurantSpan(ob, kOrigin, kForward, 0.0f, 100.0f,
                                    a, b));
    EXPECT_TRUE(std::isfinite(a));
    EXPECT_TRUE(std::isfinite(b));
    EXPECT_NEAR(a, 8.0f, 1e-4f);
}

TEST(ObscurantGeometry, ContainsMatchesSpan)
{
    const auto ob = obscurantAt(rc::ObscurantType::kEllipsoid, 10.0f,
                                3.0f, 2.0f, 2.0f, 0.1f);
    float a = 0.0f, b = 0.0f;
    ASSERT_TRUE(rc::rcObscurantSpan(ob, kOrigin, kForward, 0.0f, 100.0f,
                                    a, b));
    const float mid = 0.5f * (a + b);
    EXPECT_TRUE(rc::rcObscurantContains(ob, rc::RcV3{mid, 0.0f, 0.0f}));
    EXPECT_FALSE(rc::rcObscurantContains(ob, rc::RcV3{a - 0.1f, 0.0f, 0.0f}));
    EXPECT_FALSE(rc::rcObscurantContains(ob, rc::RcV3{b + 0.1f, 0.0f, 0.0f}));
}

// ── Optical depth ────────────────────────────────────────────────────────────

TEST(ObscurantOpticalDepth, EqualsSigmaTimesPathLength)
{
    rc::ScanParams sp = baseParams();
    addObscurant(sp, obscurantAt(rc::ObscurantType::kBox, 10.0f,
                                 2.0f, 5.0f, 5.0f, 0.25f));
    // 4 m of path at σ = 0.25 → τ = 1.
    EXPECT_NEAR(rc::rcOpticalDepth(sp, kOrigin, kForward, 0.0f, 100.0f),
                1.0f, 1e-5f);
    // Clipping the integration range clips the optical depth with it.
    EXPECT_NEAR(rc::rcOpticalDepth(sp, kOrigin, kForward, 0.0f, 10.0f),
                0.5f, 1e-5f);
}

TEST(ObscurantOpticalDepth, OverlapsAddExactly)
{
    // Two boxes overlapping on [9, 11]: τ = 0.1·4 + 0.2·4 = 1.2, and the
    // overlap must be double-counted (two clouds in the same place really
    // are twice as thick), which the Σ∫ form gives for free.
    rc::ScanParams sp = baseParams();
    addObscurant(sp, obscurantAt(rc::ObscurantType::kBox, 9.0f,
                                 2.0f, 5.0f, 5.0f, 0.1f));   // [7, 11]
    addObscurant(sp, obscurantAt(rc::ObscurantType::kBox, 11.0f,
                                 2.0f, 5.0f, 5.0f, 0.2f));   // [9, 13]
    EXPECT_NEAR(rc::rcOpticalDepth(sp, kOrigin, kForward, 0.0f, 100.0f),
                0.1f * 4.0f + 0.2f * 4.0f, 1e-5f);
}

TEST(ObscurantOpticalDepth, WeightedAlbedoIsTauWeighted)
{
    rc::ScanParams sp = baseParams();
    addObscurant(sp, obscurantAt(rc::ObscurantType::kBox, 5.0f,
                                 1.0f, 5.0f, 5.0f, 0.5f, 50.0f, 0.2f));
    addObscurant(sp, obscurantAt(rc::ObscurantType::kBox, 20.0f,
                                 1.0f, 5.0f, 5.0f, 0.5f, 50.0f, 1.0f));
    float albedo = -1.0f;
    const float tau =
        rc::rcOpticalDepth(sp, kOrigin, kForward, 0.0f, 100.0f, &albedo);
    EXPECT_NEAR(tau, 2.0f, 1e-5f);        // 2 × (0.5 × 2 m)
    EXPECT_NEAR(albedo, 0.6f, 1e-5f);     // equal τ → plain mean
}

TEST(ObscurantOpticalDepth, DepthInversionIsExactAcrossSegments)
{
    // Two disjoint slabs of different density: τ(s) is piecewise linear with
    // a gap, so a naive uniform-medium inversion would land in the wrong
    // segment. Round-tripping every target τ pins the walk.
    rc::ScanParams sp = baseParams();
    addObscurant(sp, obscurantAt(rc::ObscurantType::kBox, 5.0f,
                                 1.0f, 5.0f, 5.0f, 0.3f));   // [4, 6]
    addObscurant(sp, obscurantAt(rc::ObscurantType::kBox, 20.0f,
                                 2.0f, 5.0f, 5.0f, 0.7f));   // [18, 22]
    const float tau_total =
        rc::rcOpticalDepth(sp, kOrigin, kForward, 0.0f, 40.0f);
    ASSERT_NEAR(tau_total, 0.3f * 2.0f + 0.7f * 4.0f, 1e-5f);

    for (int k = 1; k < 20; ++k) {
        const float target = tau_total * static_cast<float>(k) / 20.0f;
        const float s =
            rc::rcDepthAtOpticalDepth(sp, kOrigin, kForward, 0.0f, 40.0f,
                                      target);
        EXPECT_NEAR(rc::rcOpticalDepth(sp, kOrigin, kForward, 0.0f, s),
                    target, 1e-4f) << "at k=" << k;
    }
}

TEST(ObscurantOpticalDepth, DepthInversionClampsToFarEndWhenUnreachable)
{
    rc::ScanParams sp = baseParams();
    addObscurant(sp, obscurantAt(rc::ObscurantType::kBox, 5.0f,
                                 1.0f, 5.0f, 5.0f, 0.3f));
    EXPECT_NEAR(rc::rcDepthAtOpticalDepth(sp, kOrigin, kForward, 0.0f, 40.0f,
                                          100.0f),
                40.0f, 1e-4f);
}

// ── Extinction of a hard target ──────────────────────────────────────────────

TEST(ObscurantExtinction, MatchesClosedFormBeerLambert)
{
    rc::Scene scene;
    std::vector<rc::InstanceXform> xf;
    makeWall(scene, xf, 20.0f, 0.8f);

    const rc::ScanParams clear = baseParams();
    const RayResult a = castOne(scene, xf, clear);
    ASSERT_NEAR(a.range, 20.0f, 1e-3f);
    ASSERT_NEAR(a.retro, 0.8f, 1e-4f);

    rc::ScanParams smoky = baseParams();
    addObscurant(smoky, obscurantAt(rc::ObscurantType::kBox, 10.0f,
                                    2.0f, 5.0f, 5.0f, 0.05f,
                                    kNoBackscatter));
    const RayResult b = castOne(scene, xf, smoky);

    // 4 m at σ = 0.05 → τ = 0.2, two-way transmittance exp(-0.4).
    EXPECT_NEAR(b.range, 20.0f, 1e-3f) << "extinction must not move the range";
    EXPECT_NEAR(b.retro, 0.8f * std::exp(-0.4f), 1e-5f);
}

TEST(ObscurantExtinction, IsZeroWhenTheBeamMissesTheVolume)
{
    rc::Scene scene;
    std::vector<rc::InstanceXform> xf;
    makeWall(scene, xf, 20.0f, 0.8f);

    rc::ScanParams sp = baseParams();
    // Cloud parked well off the +x axis.
    rc::RcObscurant ob = obscurantAt(rc::ObscurantType::kBox, 10.0f,
                                     2.0f, 2.0f, 2.0f, 2.0f);
    ob.t[1] = -20.0f;  // world→local translation shifts the centre to y=+20
    addObscurant(sp, ob);

    const RayResult r = castOne(scene, xf, sp);
    EXPECT_NEAR(r.range, 20.0f, 1e-3f);
    EXPECT_NEAR(r.retro, 0.8f, 1e-6f);
}

TEST(ObscurantExtinction, EmptyConfigIsBitIdenticalToNoMediaAtAll)
{
    rc::Scene scene;
    std::vector<rc::InstanceXform> xf;
    makeWall(scene, xf, 20.0f, 0.8f);

    const RayResult base = castOne(scene, xf, baseParams());

    rc::ScanParams sp = baseParams();
    sp.rng_salt = 12345;          // salt set, but no volumes
    sp.pulse_gate_m = 3.0f;
    const RayResult same = castOne(scene, xf, sp);

    EXPECT_EQ(base.range, same.range);
    EXPECT_EQ(base.retro, same.retro);
    EXPECT_EQ(base.nir, same.nir);
}

TEST(ObscurantExtinction, ThickerSmokeAlwaysDimsMore)
{
    rc::Scene scene;
    std::vector<rc::InstanceXform> xf;
    makeWall(scene, xf, 30.0f, 0.9f);

    float previous = kInf;
    for (float sigma : {0.0f, 0.01f, 0.05f, 0.1f, 0.3f}) {
        rc::ScanParams sp = baseParams();
        if (sigma > 0.0f) {
            addObscurant(sp, obscurantAt(rc::ObscurantType::kBox, 10.0f,
                                         3.0f, 5.0f, 5.0f, sigma,
                                         kNoBackscatter));
        }
        const RayResult r = castOne(scene, xf, sp);
        ASSERT_NEAR(r.range, 30.0f, 1e-3f);
        EXPECT_LT(r.retro, previous) << "sigma=" << sigma;
        previous = r.retro;
    }
}

// ── Forward scattering: Platt's multiple-scattering factor ───────────────────
//
// σ_ext removes light in every direction, but forward-peaked media deflect
// most of it by only milliradians, so a real receiver still collects it.
// η credits that back: the round trip attenuates by exp(−2·η·τ).

TEST(ObscurantForwardScatter, EtaScalesTheAttenuatingOpticalDepth)
{
    rc::ScanParams sp = baseParams();
    // 4 m at σ = 0.25 → τ = 1; η = 0.6 → τ_eff = 0.6.
    addObscurant(sp, obscurantAt(rc::ObscurantType::kBox, 10.0f,
                                 2.0f, 5.0f, 5.0f, 0.25f, 50.0f, 0.8f, 0.6f));
    float tau_eff = -1.0f;
    const float tau = rc::rcOpticalDepth(sp, kOrigin, kForward, 0.0f, 100.0f,
                                         nullptr, &tau_eff);
    EXPECT_NEAR(tau, 1.0f, 1e-5f) << "physical depth must be untouched";
    EXPECT_NEAR(tau_eff, 0.6f, 1e-5f);
}

TEST(ObscurantForwardScatter, EtaMixesPerVolumeAlongOnePath)
{
    // Two media with different η on one ray: the attenuating depth is the
    // η-weighted sum, not η applied to the total.
    rc::ScanParams sp = baseParams();
    addObscurant(sp, obscurantAt(rc::ObscurantType::kBox, 5.0f,
                                 1.0f, 5.0f, 5.0f, 0.5f, 50.0f, 0.8f, 1.0f));
    addObscurant(sp, obscurantAt(rc::ObscurantType::kBox, 20.0f,
                                 1.0f, 5.0f, 5.0f, 0.5f, 50.0f, 0.8f, 0.5f));
    float tau_eff = -1.0f;
    const float tau = rc::rcOpticalDepth(sp, kOrigin, kForward, 0.0f, 100.0f,
                                         nullptr, &tau_eff);
    EXPECT_NEAR(tau, 2.0f, 1e-5f);                 // 2 × (0.5 × 2 m)
    EXPECT_NEAR(tau_eff, 1.0f * 1.0f + 0.5f * 1.0f, 1e-5f);
}

TEST(ObscurantForwardScatter, DefaultEtaIsTheSingleScatteringLimit)
{
    // η = 1 must reproduce the pre-existing behaviour bit for bit, so the
    // correction is opt-in and nobody's world changes under them.
    rc::Scene scene;
    std::vector<rc::InstanceXform> xf;
    makeWall(scene, xf, 20.0f, 0.8f);

    rc::ScanParams unset = baseParams();
    addObscurant(unset, obscurantAt(rc::ObscurantType::kBox, 10.0f,
                                    2.0f, 5.0f, 5.0f, 0.2f, kNoBackscatter));
    ASSERT_FLOAT_EQ(unset.obscurants[0].ms_factor, 1.0f) << "default is 1";

    rc::ScanParams explicit_one = baseParams();
    addObscurant(explicit_one,
                 obscurantAt(rc::ObscurantType::kBox, 10.0f, 2.0f, 5.0f, 5.0f,
                             0.2f, kNoBackscatter, 0.8f, 1.0f));

    const RayResult a = castOne(scene, xf, unset);
    const RayResult b = castOne(scene, xf, explicit_one);
    EXPECT_EQ(a.range, b.range);
    EXPECT_EQ(a.retro, b.retro);
    // ...and it is exactly the closed-form single-scattering value.
    EXPECT_NEAR(a.retro, 0.8f * std::exp(-2.0f * 0.2f * 4.0f), 1e-5f);
}

TEST(ObscurantForwardScatter, LowerEtaLetsMoreLightThrough)
{
    rc::Scene scene;
    std::vector<rc::InstanceXform> xf;
    makeWall(scene, xf, 25.0f, 0.9f);

    // τ = 0.3 × 6 m = 1.8. Attenuation must follow exp(-2·η·τ) exactly.
    float previous = 0.0f;
    for (float eta : {1.0f, 0.8f, 0.6f, 0.4f}) {
        rc::ScanParams sp = baseParams();
        addObscurant(sp, obscurantAt(rc::ObscurantType::kBox, 12.0f,
                                     3.0f, 6.0f, 6.0f, 0.3f, kNoBackscatter,
                                     0.8f, eta));
        const RayResult r = castOne(scene, xf, sp);
        EXPECT_NEAR(r.range, 25.0f, 1e-3f);
        EXPECT_NEAR(r.retro, 0.9f * std::exp(-2.0f * eta * 1.8f), 1e-5f)
            << "eta=" << eta;
        EXPECT_GT(r.retro, previous) << "less loss must mean a brighter return";
        previous = r.retro;
    }
}

TEST(ObscurantForwardScatter, RecoversTargetsTheSingleScatteringLimitDrops)
{
    // The point of the correction: at high optical depth η = 1 is pessimistic
    // enough to push a real target past the detection limit that η < 1 keeps
    // it inside.
    rc::Scene scene;
    std::vector<rc::InstanceXform> xf;
    makeWall(scene, xf, 45.0f, 0.8f);

    rc::ScanParams pessimistic = baseParams();
    addObscurant(pessimistic, obscurantAt(rc::ObscurantType::kBox, 15.0f,
                                          4.0f, 8.0f, 8.0f, 0.2f,
                                          kNoBackscatter));
    rc::ScanParams corrected = baseParams();
    addObscurant(corrected, obscurantAt(rc::ObscurantType::kBox, 15.0f,
                                        4.0f, 8.0f, 8.0f, 0.2f,
                                        kNoBackscatter, 0.8f, 0.5f));

    const RayResult a = castOne(scene, xf, pessimistic);
    const RayResult b = castOne(scene, xf, corrected);
    EXPECT_FLOAT_EQ(rpmath::dropoutProbability(a.range, a.retro, 0.0005f,
                                               0.03f, 120.0f), 1.0f);
    EXPECT_LT(rpmath::dropoutProbability(b.range, b.retro, 0.0005f, 0.03f,
                                         120.0f), 1.0f);
}

TEST(ObscurantForwardScatter, AmbientChannelIgnoresEta)
{
    // Koschmieder's airlight term IS the multiply-scattered light for a wide
    // passive field of view, so applying η there as well would correct for
    // the same physics twice. The NEAR_IR value must not move with η.
    rc::Scene scene;
    std::vector<rc::InstanceXform> xf;
    makeWall(scene, xf, 20.0f, 0.2f);

    float reference = -1.0f;
    for (float eta : {1.0f, 0.5f, 0.2f}) {
        rc::ScanParams sp = baseParams();
        addObscurant(sp, obscurantAt(rc::ObscurantType::kBox, 10.0f,
                                     2.0f, 5.0f, 5.0f, 0.5f, kNoBackscatter,
                                     0.9f, eta));
        const RayResult r = castOne(scene, xf, sp);
        if (reference < 0.0f) {
            reference = r.nir;
            const float trans = std::exp(-2.0f);   // physical τ = 0.5 × 4
            EXPECT_NEAR(reference, 0.2f * trans + 0.9f * (1.0f - trans),
                        1e-5f);
        } else {
            EXPECT_FLOAT_EQ(r.nir, reference) << "eta=" << eta;
        }
    }
}

TEST(ObscurantForwardScatter, ScatterDepthSamplingPenetratesDeeper)
{
    // Sampling lives in η·τ space, so recovering forward-scattered light
    // must also push the medium's own returns further into the cloud.
    rc::Scene scene;
    std::vector<rc::InstanceXform> xf;

    auto medianDepth = [&](float eta) {
        rc::ScanParams sp = baseParams();
        addObscurant(sp, obscurantAt(rc::ObscurantType::kBox, 10.0f,
                                     5.0f, 5.0f, 5.0f, 1.0f, 50.0f, 0.8f,
                                     eta));
        std::vector<float> depths;
        for (uint32_t i = 1; i <= 600; ++i) {
            sp.rng_salt = i;
            const RayResult r = castOne(scene, xf, sp);
            if (std::isfinite(r.range)) depths.push_back(r.range);
        }
        std::sort(depths.begin(), depths.end());
        return depths.empty() ? 0.0f : depths[depths.size() / 2];
    };

    const float deep = medianDepth(0.25f);
    const float shallow = medianDepth(1.0f);
    EXPECT_GT(deep, shallow);
    // Mean penetration is 1/(2ησ): 0.5 m at η = 0.25 against 0.125 m at 1.
    EXPECT_NEAR(deep - 5.0f, 4.0f * (shallow - 5.0f), 0.15f);
}

// ── Backscatter from the medium itself ───────────────────────────────────────

namespace {

/// Cast the same ray under many RNG salts. The scatter-depth draw is a pure
/// hash of (pixel index, salt), so varying the salt on a one-pixel scan
/// samples exactly the distribution a full frame would.
std::vector<RayResult> castOverSalts(const rc::Scene & scene,
                                     const std::vector<rc::InstanceXform> & xf,
                                     rc::ScanParams sp, int n)
{
    std::vector<RayResult> out;
    out.reserve(static_cast<size_t>(n));
    for (int i = 1; i <= n; ++i) {
        sp.rng_salt = static_cast<uint32_t>(i);
        out.push_back(castOne(scene, xf, sp));
    }
    return out;
}

double fractionWithRangeBelow(const std::vector<RayResult> & rs, float limit)
{
    const auto n = std::count_if(rs.begin(), rs.end(),
        [limit](const RayResult & r) { return r.range < limit; });
    return static_cast<double>(n) / static_cast<double>(rs.size());
}

}  // namespace

TEST(ObscurantBackscatter, DenseSmokeReturnsFromItsNearFace)
{
    rc::Scene scene;
    std::vector<rc::InstanceXform> xf;
    makeWall(scene, xf, 30.0f, 0.9f);

    rc::ScanParams sp = baseParams();
    // σ = 2 /m over [5, 15]: two-way optical depth 40, so the wall is gone
    // and the mean penetration is 1/(2σ) = 0.25 m past the near face.
    addObscurant(sp, obscurantAt(rc::ObscurantType::kBox, 10.0f,
                                 5.0f, 5.0f, 5.0f, 2.0f));

    const auto rs = castOverSalts(scene, xf, sp, 2000);
    for (const auto & r : rs) {
        ASSERT_TRUE(std::isfinite(r.range));
        EXPECT_GE(r.range, 5.0f) << "no return may come from before the cloud";
        EXPECT_LE(r.range, 15.0f);
    }
    // Analytic: P(depth < 5 + d) = 1 - exp(-2σd) → 86% by 0.5 m, 99.7% by 1.5.
    EXPECT_GT(fractionWithRangeBelow(rs, 5.5f), 0.80);
    EXPECT_GT(fractionWithRangeBelow(rs, 6.5f), 0.99);
}

TEST(ObscurantBackscatter, ThinHazeLetsTheTargetThrough)
{
    rc::Scene scene;
    std::vector<rc::InstanceXform> xf;
    makeWall(scene, xf, 30.0f, 0.9f);

    rc::ScanParams sp = baseParams();
    addObscurant(sp, obscurantAt(rc::ObscurantType::kBox, 10.0f,
                                 5.0f, 5.0f, 5.0f, 0.002f));

    const auto rs = castOverSalts(scene, xf, sp, 500);
    const auto wall = std::count_if(rs.begin(), rs.end(),
        [](const RayResult & r) { return std::abs(r.range - 30.0f) < 0.1f; });
    EXPECT_GT(static_cast<double>(wall) / rs.size(), 0.95);
}

TEST(ObscurantBackscatter, SmokeAgainstOpenSkyStillReturns)
{
    // No geometry at all: a beam through smoke is a RETURN, not a miss.
    rc::Scene scene;
    std::vector<rc::InstanceXform> xf;

    rc::ScanParams sp = baseParams();
    addObscurant(sp, obscurantAt(rc::ObscurantType::kBox, 10.0f,
                                 5.0f, 5.0f, 5.0f, 1.0f));

    const auto rs = castOverSalts(scene, xf, sp, 500);
    int finite = 0;
    for (const auto & r : rs) {
        if (!std::isfinite(r.range)) continue;
        ++finite;
        EXPECT_GE(r.range, 5.0f);
        EXPECT_LE(r.range, 15.0f);
        EXPECT_GT(r.retro, 0.0f);
    }
    EXPECT_GT(finite, 450);
}

TEST(ObscurantBackscatter, ReturnAmplitudeMatchesTheLidarEquation)
{
    // Salt 0 is not used by castOverSalts; pick a salt whose draw lands very
    // close to the near face so the analytic value is unambiguous.
    rc::Scene scene;
    std::vector<rc::InstanceXform> xf;

    constexpr float kSigma = 1.0f;
    constexpr float kS = 40.0f;
    rc::ScanParams sp = baseParams();
    sp.pulse_gate_m = 0.6f;
    addObscurant(sp, obscurantAt(rc::ObscurantType::kBox, 10.0f,
                                 5.0f, 5.0f, 5.0f, kSigma, kS));

    const auto rs = castOverSalts(scene, xf, sp, 400);
    for (const auto & r : rs) {
        ASSERT_TRUE(std::isfinite(r.range));
        // ρ = π·β·ΔR·exp(-2τ) with β = σ/S and τ = σ·(depth - 5).
        const float tau = kSigma * (r.range - 5.0f);
        const float expected = static_cast<float>(M_PI) * (kSigma / kS) *
                               sp.pulse_gate_m * std::exp(-2.0f * tau);
        EXPECT_NEAR(r.retro, expected, 1e-5f) << "range=" << r.range;
    }
}

TEST(ObscurantBackscatter, ReturnsInsideTheBlindZoneAreDiscarded)
{
    // Sensor standing inside dense smoke: everything nearer than near_clip
    // is unreportable, so those draws must yield a miss rather than a
    // bogus 5 cm return.
    rc::Scene scene;
    std::vector<rc::InstanceXform> xf;

    rc::ScanParams sp = baseParams();
    sp.near_clip = 1.0f;
    addObscurant(sp, obscurantAt(rc::ObscurantType::kBox, 0.0f,
                                 20.0f, 20.0f, 20.0f, 3.0f));

    const auto rs = castOverSalts(scene, xf, sp, 1000);
    int blanked = 0;
    for (const auto & r : rs) {
        if (!std::isfinite(r.range)) {
            ++blanked;
            EXPECT_EQ(r.retro, 0.0f);
        } else {
            EXPECT_GE(r.range, sp.near_clip);
        }
    }
    // P(scatter within 1 m) = 1 - exp(-2·3·1) ≈ 99.8%.
    EXPECT_GT(blanked, 900);
}

TEST(ObscurantBackscatter, IsReproducibleForTheSameSalt)
{
    rc::Scene scene;
    std::vector<rc::InstanceXform> xf;
    makeWall(scene, xf, 30.0f, 0.9f);

    rc::ScanParams sp = baseParams();
    addObscurant(sp, obscurantAt(rc::ObscurantType::kBox, 10.0f,
                                 5.0f, 5.0f, 5.0f, 1.0f));
    sp.rng_salt = 7;
    const RayResult a = castOne(scene, xf, sp);
    const RayResult b = castOne(scene, xf, sp);
    EXPECT_EQ(a.range, b.range);
    EXPECT_EQ(a.retro, b.retro);

    sp.rng_salt = 8;
    const RayResult c = castOne(scene, xf, sp);
    EXPECT_NE(a.range, c.range) << "successive scans must decorrelate";
}

TEST(ObscurantBackscatter, HashDrawIsUniformAndOpenIntervalled)
{
    // The draw feeds log(1 - ξ); an exact 0 or 1 would produce ±inf.
    int bins[10] = {0};
    for (uint32_t i = 0; i < 20000; ++i) {
        const float u = rc::rcHashUnit(i, 991u);
        ASSERT_GT(u, 0.0f);
        ASSERT_LT(u, 1.0f);
        ++bins[static_cast<int>(u * 10.0f)];
    }
    for (int b : bins) {
        EXPECT_GT(b, 1700) << "hash draw is badly non-uniform";
        EXPECT_LT(b, 2300) << "hash draw is badly non-uniform";
    }
}

// ── NEAR_IR airlight ─────────────────────────────────────────────────────────

TEST(ObscurantNearIr, SmokeGlowsWhileTheLaserChannelsDim)
{
    rc::Scene scene;
    std::vector<rc::InstanceXform> xf;
    makeWall(scene, xf, 20.0f, 0.2f);   // dark wall

    const RayResult clear = castOne(scene, xf, baseParams());
    // Default illumination is ambient-only with weight 1.
    ASSERT_NEAR(clear.nir, 0.2f, 1e-5f);

    rc::ScanParams sp = baseParams();
    addObscurant(sp, obscurantAt(rc::ObscurantType::kBox, 10.0f,
                                 2.0f, 5.0f, 5.0f, 0.5f, kNoBackscatter,
                                 0.9f));
    const RayResult smoky = castOne(scene, xf, sp);

    // Koschmieder: L = L_target·exp(-τ) + ω·illum·(1 - exp(-τ)), τ = 0.5·4.
    const float trans = std::exp(-2.0f);
    EXPECT_NEAR(smoky.nir, 0.2f * trans + 0.9f * (1.0f - trans), 1e-5f);
    EXPECT_GT(smoky.nir, clear.nir) << "smoke must brighten NEAR_IR";
    EXPECT_LT(smoky.retro, clear.retro) << "...while dimming the laser return";
}

TEST(ObscurantNearIr, BrightTargetBehindSmokeIsWashedTowardTheAirlight)
{
    rc::Scene scene;
    std::vector<rc::InstanceXform> xf;
    makeWall(scene, xf, 20.0f, 1.0f);   // bright wall, brighter than ω

    rc::ScanParams sp = baseParams();
    addObscurant(sp, obscurantAt(rc::ObscurantType::kBox, 10.0f,
                                 3.0f, 5.0f, 5.0f, 1.0f, kNoBackscatter,
                                 0.6f));
    const RayResult r = castOne(scene, xf, sp);
    EXPECT_LT(r.nir, 1.0f);
    EXPECT_GT(r.nir, 0.6f) << "airlight is the floor, not the value";
}

// ── Downstream consequences (the point of folding it into retro) ─────────────

TEST(ObscurantDownstream, TargetsPastTheDetectionLimitAreDroppedForFree)
{
    // The extinction is applied to the apparent reflectance, so the existing
    // √ρ detection limit in the noise model does the dropping with no
    // knowledge of smoke. Verify at the exact boundary the model defines.
    rc::Scene scene;
    std::vector<rc::InstanceXform> xf;
    makeWall(scene, xf, 60.0f, 0.8f);

    const rc::ScanParams clear = baseParams();
    const RayResult a = castOne(scene, xf, clear);
    EXPECT_LT(rpmath::dropoutProbability(a.range, a.retro, 0.0005f, 0.03f,
                                         120.0f),
              1.0f) << "a clear 60 m wall is well inside the detection limit";

    rc::ScanParams smoky = baseParams();
    addObscurant(smoky, obscurantAt(rc::ObscurantType::kBox, 20.0f,
                                    5.0f, 8.0f, 8.0f, 0.2f, kNoBackscatter));
    const RayResult b = castOne(scene, xf, smoky);
    ASSERT_NEAR(b.range, 60.0f, 1e-3f);
    EXPECT_FLOAT_EQ(rpmath::dropoutProbability(b.range, b.retro, 0.0005f,
                                               0.03f, 120.0f),
                    1.0f)
        << "through τ=2 smoke the same wall must fall past the limit";
}

TEST(ObscurantDownstream, SignalAndReflectivityByteBothFall)
{
    rc::Scene scene;
    std::vector<rc::InstanceXform> xf;
    makeWall(scene, xf, 15.0f, 0.9f);

    const RayResult clear = castOne(scene, xf, baseParams());

    rc::ScanParams sp = baseParams();
    addObscurant(sp, obscurantAt(rc::ObscurantType::kBox, 7.0f,
                                 2.0f, 5.0f, 5.0f, 0.15f, kNoBackscatter));
    const RayResult smoky = castOne(scene, xf, sp);

    EXPECT_LT(rpmath::signalFromRange(smoky.range, smoky.retro, 800.0f),
              rpmath::signalFromRange(clear.range, clear.retro, 800.0f));
    EXPECT_LT(rpmath::reflectivityToByte(smoky.retro),
              rpmath::reflectivityToByte(clear.retro));
}

// ── Interaction with the rest of the raycast model ───────────────────────────

TEST(ObscurantInteraction, EllipsoidAndCylinderVolumesAttenuateToo)
{
    rc::Scene scene;
    std::vector<rc::InstanceXform> xf;
    makeWall(scene, xf, 25.0f, 0.7f);
    const RayResult clear = castOne(scene, xf, baseParams());

    for (auto type : {rc::ObscurantType::kEllipsoid,
                      rc::ObscurantType::kCylinder}) {
        rc::ScanParams sp = baseParams();
        addObscurant(sp, obscurantAt(type, 12.0f, 3.0f, 3.0f, 3.0f, 0.4f,
                                     kNoBackscatter));
        const RayResult r = castOne(scene, xf, sp);
        EXPECT_NEAR(r.range, clear.range, 1e-3f);
        EXPECT_LT(r.retro, clear.retro * 0.1f)
            << "type=" << static_cast<int>(type);
    }
}

TEST(ObscurantInteraction, MaxVolumesAreAllIntegrated)
{
    rc::Scene scene;
    std::vector<rc::InstanceXform> xf;
    makeWall(scene, xf, 40.0f, 0.8f);

    rc::ScanParams sp = baseParams();
    for (int i = 0; i < rc::kMaxObscurants; ++i) {
        addObscurant(sp, obscurantAt(rc::ObscurantType::kBox,
                                     3.0f + 2.0f * static_cast<float>(i),
                                     0.5f, 4.0f, 4.0f, 0.1f, kNoBackscatter));
    }
    ASSERT_EQ(sp.n_obscurants, rc::kMaxObscurants);
    const RayResult r = castOne(scene, xf, sp);
    // Each volume contributes σ × 1 m of path = 0.1 of optical depth. Derive
    // the expectation from the cap so raising it does not silently turn this
    // into a weaker assertion.
    const float tau = 0.1f * static_cast<float>(rc::kMaxObscurants);
    EXPECT_NEAR(r.retro, 0.8f * std::exp(-2.0f * tau), 1e-5f);
}

TEST(ObscurantInteraction, MediumScalesWhicheverCandidateWinsUnchanged)
{
    // A transparent pane in front of a wall exercises the glass continuation
    // path. Whichever candidate that arbitration picks, haze in FRONT of the
    // whole stack must leave the choice alone and scale its reflectance by
    // exactly exp(-2τ) — the medium is an attenuator, not a re-ranker.
    rc::Scene scene;
    std::vector<rc::InstanceXform> xf;
    const float pane[3] = {0.02f, 4.0f, 4.0f};
    const int pane_idx = scene.addInstance(rc::GeomType::kBox, pane, 0.05f,
                                           -1, 0.0f, 0.9f);
    const float pane_t[3] = {5.0f, 0.0f, 0.0f};
    rc::InstanceXform px;
    scene.computeXform(pane_idx, kIdentityR, pane_t, px);
    xf.push_back(px);
    makeWall(scene, xf, 12.0f, 0.9f);

    const RayResult clear = castOne(scene, xf, baseParams());
    ASSERT_TRUE(std::isfinite(clear.range));
    ASSERT_GT(clear.retro, 0.0f);

    rc::ScanParams sp = baseParams();
    // Haze on [2, 4], entirely in front of the pane, so it is on the path to
    // every candidate: σ·L = 0.02 × 2 → two-way exp(-0.08).
    addObscurant(sp, obscurantAt(rc::ObscurantType::kBox, 3.0f,
                                 1.0f, 4.0f, 4.0f, 0.02f, kNoBackscatter));
    const RayResult hazy = castOne(scene, xf, sp);
    EXPECT_FLOAT_EQ(hazy.range, clear.range) << "arbitration must not shift";
    EXPECT_NEAR(hazy.retro, clear.retro * std::exp(-0.08f), 1e-6f);
}

}  // namespace gz_gpu_ouster_lidar
