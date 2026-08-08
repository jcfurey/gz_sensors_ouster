// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0
//
// Participating media (smoke / dust / fog) in the raycast ray mode: volume
// geometry, Beer–Lambert extinction against the closed-form value, the
// medium's stochastic range-resolved backscatter return, NEAR_IR airlight,
// and the downstream consequences the model is
// supposed to produce for free (dimmer signal, lower reflectivity byte,
// targets moving down the calibrated detection rolloff).
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
              float face_x, float retro, bool has_retro = true)
{
    const float half[3] = {0.5f, 8.0f, 8.0f};
    const int idx = scene.addInstance(rc::GeomType::kBox, half, retro,
                                      -1, 0.0f, 0.0f, has_retro);
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

// ── Extinction of a hard target ──────────────────────────────────────────────

TEST(ObscurantExtinction, MissingRetroUsesPhysicalFallbackBeforeMedia)
{
    rc::Scene scene;
    std::vector<rc::InstanceXform> xf;
    // The zero is only storage: has_retro=false models an omitted SDF tag.
    makeWall(scene, xf, 20.0f, 0.0f, false);

    rc::ScanParams clear = baseParams();
    clear.fallback_retro = 0.5f;
    const RayResult a = castOne(scene, xf, clear);
    ASSERT_NEAR(a.range, 20.0f, 1e-3f);
    EXPECT_NEAR(a.retro, 0.5f, 1e-6f);
    EXPECT_NEAR(a.nir, 0.5f, 1e-6f);

    rc::ScanParams smoky = clear;
    addObscurant(smoky, obscurantAt(rc::ObscurantType::kBox, 10.0f,
                                    2.0f, 5.0f, 5.0f, 0.05f,
                                    kNoBackscatter));
    const RayResult b = castOne(scene, xf, smoky);
    EXPECT_NEAR(b.range, 20.0f, 1e-3f);
    EXPECT_NEAR(b.retro, 0.5f * std::exp(-0.4f), 1e-5f);
}

TEST(ObscurantExtinction, FallbackSupportsRetroreflectiveByteRange)
{
    rc::Scene scene;
    std::vector<rc::InstanceXform> xf;
    makeWall(scene, xf, 20.0f, 0.0f, false);

    rc::ScanParams sp = baseParams();
    sp.fallback_retro = rpmath::reflectivityByteToRetro(144.0f);
    const RayResult r = castOne(scene, xf, sp);
    EXPECT_GT(r.retro, 1.0f);
    EXPECT_EQ(rpmath::reflectivityToByte(r.retro), 144u);
}

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
    // enough to push a real target down the detection rolloff that η < 1 keeps
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
    const float p_a = rpmath::dropoutProbability(
        a.range, a.retro, 0.0005f, 0.03f, 150.0f,
        45.0f, 100.0f, 55.0f, 120.0f, 0.15f);
    const float p_b = rpmath::dropoutProbability(
        b.range, b.retro, 0.0005f, 0.03f, 150.0f,
        45.0f, 100.0f, 55.0f, 120.0f, 0.15f);
    EXPECT_GT(p_a, p_b);
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

namespace {

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

double fractionWithRange(const std::vector<RayResult> & rs,
                         float lo, float hi)
{
    const auto n = std::count_if(rs.begin(), rs.end(),
        [lo, hi](const RayResult & r) { return r.range >= lo && r.range < hi; });
    return static_cast<double>(n) / static_cast<double>(rs.size());
}

float medianFiniteRange(std::vector<RayResult> rs)
{
    std::vector<float> ranges;
    for (const auto & r : rs) {
        if (std::isfinite(r.range)) ranges.push_back(r.range);
    }
    std::sort(ranges.begin(), ranges.end());
    return ranges.empty() ? 0.0f : ranges[ranges.size() / 2];
}

}  // namespace

TEST(ObscurantForwardScatter, LowerEtaSamplesDeeperIntoTheMedium)
{
    // The draw follows beta*exp(-2*eta*tau)/R^2. Recovering more forward-
    // scattered light therefore broadens the range distribution into the
    // cloud, while range spreading still keeps it biased toward the sensor.
    rc::Scene scene;
    std::vector<rc::InstanceXform> xf;

    auto medianDepth = [&](float eta) {
        rc::ScanParams sp = baseParams();
        addObscurant(sp, obscurantAt(rc::ObscurantType::kBox, 10.0f,
                                     5.0f, 5.0f, 5.0f, 1.0f, 50.0f, 0.8f,
                                     eta));
        return medianFiniteRange(castOverSalts(scene, xf, sp, 2000));
    };

    EXPECT_GT(medianDepth(0.25f), medianDepth(1.0f) + 0.2f);
}

// ── Backscatter from the medium itself ───────────────────────────────────────

TEST(ObscurantBackscatter, DenseSmokeIsSpeckledButNearFaceWeighted)
{
    rc::Scene scene;
    std::vector<rc::InstanceXform> xf;
    makeWall(scene, xf, 30.0f, 0.0f, false);

    rc::ScanParams sp = baseParams();
    // sigma=2/m over [5,15]: the wall is gone and most sampled power lies
    // within the first few attenuation lengths, without collapsing to x=5.
    addObscurant(sp, obscurantAt(rc::ObscurantType::kBox, 10.0f,
                                 5.0f, 5.0f, 5.0f, 2.0f));

    const auto rs = castOverSalts(scene, xf, sp, 2000);
    float lo = 100.0f, hi = 0.0f;
    int medium_returns = 0;
    int near_returns = 0;
    for (const auto & r : rs) {
        ASSERT_TRUE(std::isfinite(r.range));
        if (r.range >= 29.9f) continue;
        ++medium_returns;
        EXPECT_GE(r.range, 5.0f);
        EXPECT_LE(r.range, 15.0f);
        if (r.range < 5.5f) ++near_returns;
        lo = std::min(lo, r.range);
        hi = std::max(hi, r.range);
    }
    EXPECT_GT(medium_returns, 1000);
    EXPECT_LT(medium_returns, 1500)
        << "even dense smoke must not return on every intersecting beam";
    EXPECT_GT(hi - lo, 0.5f) << "the plume must not collapse to a hard shell";
    EXPECT_GT(static_cast<double>(near_returns) / medium_returns, 0.80);
}

TEST(ObscurantBackscatter, ThinHazeLetsTheTargetThrough)
{
    rc::Scene scene;
    std::vector<rc::InstanceXform> xf;
    // Regression: an omitted laser_retro used to enter arbitration as zero,
    // allowing any positive smoke return to beat the hard target.
    makeWall(scene, xf, 30.0f, 0.0f, false);

    rc::ScanParams sp = baseParams();
    addObscurant(sp, obscurantAt(rc::ObscurantType::kBox, 10.0f,
                                 5.0f, 5.0f, 5.0f, 0.002f));

    const auto rs = castOverSalts(scene, xf, sp, 500);
    EXPECT_GT(fractionWithRange(rs, 29.9f, 30.1f), 0.95);
}

TEST(ObscurantBackscatter, SmokeAgainstOpenSkyStillReturns)
{
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
    EXPECT_GT(finite, 225);
    EXPECT_LT(finite, 375)
        << "the medium must remain a sparse aerosol, not a solid silhouette";
}

TEST(ObscurantBackscatter, ReturnAmplitudeMatchesTheLidarEquationAtItsDraw)
{
    rc::Scene scene;
    std::vector<rc::InstanceXform> xf;
    constexpr float kSigma = 1.0f;
    constexpr float kS = 40.0f;
    rc::ScanParams sp = baseParams();
    sp.pulse_gate_m = 0.6f;
    addObscurant(sp, obscurantAt(rc::ObscurantType::kBox, 10.0f,
                                 5.0f, 5.0f, 5.0f, kSigma, kS));

    int finite = 0;
    for (const auto & r : castOverSalts(scene, xf, sp, 400)) {
        if (!std::isfinite(r.range)) continue;
        ++finite;
        const float tau = kSigma * (r.range - 5.0f);
        const float expected = static_cast<float>(M_PI) * (kSigma / kS) *
                               sp.pulse_gate_m * std::exp(-2.0f * tau);
        EXPECT_NEAR(r.retro, expected, 1e-5f) << "range=" << r.range;
    }
    EXPECT_GT(finite, 175);
    EXPECT_LT(finite, 300);
}

TEST(ObscurantBackscatter, SensorInsideSmokeStartsSamplingAtNearClip)
{
    rc::Scene scene;
    std::vector<rc::InstanceXform> xf;
    rc::ScanParams sp = baseParams();
    sp.near_clip = 1.0f;
    addObscurant(sp, obscurantAt(rc::ObscurantType::kBox, 0.0f,
                                 20.0f, 20.0f, 20.0f, 3.0f));

    int finite = 0;
    for (const auto & r : castOverSalts(scene, xf, sp, 1000)) {
        if (!std::isfinite(r.range)) continue;
        ++finite;
        EXPECT_GE(r.range, sp.near_clip);
        const float expected = static_cast<float>(M_PI) * (3.0f / 50.0f) *
                               sp.pulse_gate_m * std::exp(-2.0f * 3.0f * r.range);
        EXPECT_NEAR(r.retro, expected, 1e-6f);
    }
    EXPECT_GT(finite, 25);
    EXPECT_LT(finite, 100);
}

TEST(ObscurantBackscatter, SameSaltIsReproducibleAndNewSaltEvolves)
{
    rc::Scene scene;
    std::vector<rc::InstanceXform> xf;
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
    EXPECT_NE(a.range, c.range);
}

TEST(ObscurantBackscatter, OverlapIsWeightedByBackscatterNotOnlyExtinction)
{
    // The weak background spans [5,15]. A high-beta overlap on [10,12]
    // should draw many returns into that later interval despite its range and
    // attenuation penalties; the old extinction-only CDF could not express
    // this when lidar ratios differed.
    rc::Scene scene;
    std::vector<rc::InstanceXform> xf;
    rc::ScanParams weak = baseParams();
    addObscurant(weak, obscurantAt(rc::ObscurantType::kBox, 10.0f,
                                   5.0f, 5.0f, 5.0f, 0.1f, 100.0f));
    rc::ScanParams overlap = weak;
    addObscurant(overlap, obscurantAt(rc::ObscurantType::kBox, 11.0f,
                                      1.0f, 5.0f, 5.0f, 1.0f, 10.0f));

    const double baseline = fractionWithRange(
        castOverSalts(scene, xf, weak, 2000), 10.0f, 12.0f);
    const double boosted = fractionWithRange(
        castOverSalts(scene, xf, overlap, 2000), 10.0f, 12.0f);
    EXPECT_GT(boosted, baseline + 0.2);
    EXPECT_GT(boosted, 0.25);
}

TEST(ObscurantBackscatter, PhotonGateTracksIntegratedReceivedPower)
{
    rc::Scene scene;
    std::vector<rc::InstanceXform> xf;
    constexpr float kSigma = 0.35f;
    constexpr float kS = 50.0f;
    constexpr float kLo = 5.0f;
    constexpr float kHi = 15.0f;

    rc::ScanParams sp = baseParams();
    addObscurant(sp, obscurantAt(rc::ObscurantType::kBox, 10.0f,
                                 5.0f, 5.0f, 5.0f, kSigma, kS));

    // High-resolution midpoint integration is an independent reference for
    // lambda = base_signal*pi*integral(beta*exp(-2*tau)/R^2 dR).
    constexpr int kSteps = 20000;
    const double dr = (kHi - kLo) / kSteps;
    double profile_mass = 0.0;
    for (int i = 0; i < kSteps; ++i) {
        const double r = kLo + (i + 0.5) * dr;
        const double tau = kSigma * (r - kLo);
        profile_mass += (kSigma / kS) * std::exp(-2.0 * tau) /
                        (r * r) * dr;
    }
    const double expected =
        1.0 - std::exp(-sp.base_signal * M_PI * profile_mass);

    const auto rs = castOverSalts(scene, xf, sp, 20000);
    const double observed = static_cast<double>(std::count_if(
        rs.begin(), rs.end(), [](const RayResult & r) {
            return std::isfinite(r.range);
        })) / rs.size();
    EXPECT_NEAR(observed, expected, 0.02);
    EXPECT_GT(observed, 0.0);
    EXPECT_LT(observed, 1.0);
}

TEST(ObscurantBackscatter, ZeroSensorGainCannotDetectTheMedium)
{
    rc::Scene scene;
    std::vector<rc::InstanceXform> xf;
    rc::ScanParams sp = baseParams();
    sp.base_signal = 0.0f;
    addObscurant(sp, obscurantAt(rc::ObscurantType::kBox, 10.0f,
                                 5.0f, 5.0f, 5.0f, 1.0f));

    for (const auto & r : castOverSalts(scene, xf, sp, 100)) {
        EXPECT_FALSE(std::isfinite(r.range));
    }
}

TEST(ObscurantSampling, DecayInversionIsAccurateAcrossTheSeriesSwitch)
{
    // rcDecayInvert solves int_0^x exp(-2ku) du = u * int_0^L exp(-2ku) du.
    // The closed form loses meaning as k -> 0, so a thin segment takes a
    // series; the tolerance here is tight enough to reject the leading term
    // alone, which is 4.8e-4 short right at the q = 1e-3 switch — 5.7 cm on
    // a 120 m segment, a visible bias in thin-haze return ranges.
    auto exact = [](double u, double k, double L) {
        const double q = 2.0 * k * L;
        return -std::log(1.0 - u * (1.0 - std::exp(-q))) / (2.0 * k);
    };
    constexpr double kL = 120.0;
    for (double q : {1e-5, 5e-4, 1e-3, 1e-2, 0.5, 5.0}) {
        const double k = q / (2.0 * kL);
        for (double u : {0.05, 0.25, 0.5, 0.75, 0.95}) {
            const double want = exact(u, k, kL);
            const float got = rc::rcDecayInvert(static_cast<float>(u),
                                                static_cast<float>(k),
                                                static_cast<float>(kL));
            EXPECT_NEAR(got, want, 1e-5 * want) << "q=" << q << " u=" << u;
        }
    }
}

TEST(ObscurantSampling, ZeroEtaVolumeContributesNoBackscatter)
{
    // The segment weight tests only beta, which is safe because the span
    // collection screens out eta <= 0 first. Pin that pairing: a volume with
    // eta = 0 — built directly, bypassing the clamp in makeObscurant — must
    // produce no medium return, since backscattering while staying perfectly
    // transparent to the same beam would be energy from nowhere.
    rc::Scene scene;
    std::vector<rc::InstanceXform> xf;

    rc::ScanParams live = baseParams();
    addObscurant(live, obscurantAt(rc::ObscurantType::kBox, 10.0f,
                                   5.0f, 5.0f, 5.0f, 1.0f, 50.0f, 0.8f, 1.0f));
    rc::ScanParams inert = live;
    inert.obscurants[0].ms_factor = 0.0f;

    int live_hits = 0, inert_hits = 0;
    for (uint32_t i = 1; i <= 400; ++i) {
        live.rng_salt = i;
        inert.rng_salt = i;
        if (std::isfinite(castOne(scene, xf, live).range)) ++live_hits;
        if (std::isfinite(castOne(scene, xf, inert).range)) ++inert_hits;
    }
    // The control only has to prove the configuration is live, so that zero
    // means something; the exact fraction is set by the Poisson detection
    // probability and is not what this test is about (it measures ~235/400).
    EXPECT_GT(live_hits, 100) << "the control must return from the medium";
    EXPECT_EQ(inert_hits, 0);
}

TEST(ObscurantSampling, NonPositiveLidarRatioAttenuatesButNeverReturns)
{
    // The complementary case the beta-only guard has to keep handling: a
    // volume with no backscatter still has to attenuate what is behind it.
    rc::Scene scene;
    std::vector<rc::InstanceXform> xf;
    makeWall(scene, xf, 20.0f, 0.8f);

    rc::ScanParams sp = baseParams();
    addObscurant(sp, obscurantAt(rc::ObscurantType::kBox, 10.0f,
                                 2.0f, 5.0f, 5.0f, 0.2f, 50.0f));
    sp.obscurants[0].lidar_ratio = 0.0f;      // no backscatter at all

    for (uint32_t i = 1; i <= 50; ++i) {
        sp.rng_salt = i;
        const RayResult r = castOne(scene, xf, sp);
        ASSERT_NEAR(r.range, 20.0f, 1e-3f) << "must still report the wall";
        // 4 m at sigma 0.2 -> tau 0.8, two-way exp(-1.6).
        EXPECT_NEAR(r.retro, 0.8f * std::exp(-1.6f), 1e-5f);
    }
}

TEST(ObscurantSampling, DecayInversionDegradesToUniformAsDensityVanishes)
{
    // With no extinction the decayed mass is uniform, so the draw must be
    // exactly u*L — the limit the series exists to reach without dividing
    // by k.
    for (float u : {0.1f, 0.5f, 0.9f}) {
        EXPECT_NEAR(rc::rcDecayInvert(u, 0.0f, 40.0f), u * 40.0f, 1e-5f);
    }
}

TEST(ObscurantSampling, DecayInversionSpansTheSegmentMonotonically)
{
    // Basic sanity the sampler relies on: the draw stays inside [0, L] and
    // increases with u, for a thin segment and a thick one alike.
    for (float k : {1.0e-6f, 0.05f, 2.0f}) {
        float previous = -1.0f;
        for (float u : {0.01f, 0.2f, 0.4f, 0.6f, 0.8f, 0.99f}) {
            const float x = rc::rcDecayInvert(u, k, 12.0f);
            EXPECT_GE(x, 0.0f) << "k=" << k;
            EXPECT_LE(x, 12.0f) << "k=" << k;
            EXPECT_GT(x, previous) << "k=" << k << " u=" << u;
            previous = x;
        }
    }
}

TEST(ObscurantBackscatter, HashDrawIsUniformAndOpenIntervalled)
{
    int bins[10] = {0};
    for (uint32_t i = 0; i < 20000; ++i) {
        const float u = rc::rcHashUnit(i, 991u, 2u);
        ASSERT_GT(u, 0.0f);
        ASSERT_LT(u, 1.0f);
        ++bins[static_cast<int>(u * 10.0f)];
    }
    for (int count : bins) {
        EXPECT_GT(count, 1700);
        EXPECT_LT(count, 2300);
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
    // Extinction is applied to apparent reflectance, so the calibrated product
    // detection curve performs the dropping without special smoke knowledge.
    rc::Scene scene;
    std::vector<rc::InstanceXform> xf;
    makeWall(scene, xf, 60.0f, 0.8f);

    const rc::ScanParams clear = baseParams();
    const RayResult a = castOne(scene, xf, clear);
    const float clear_drop = rpmath::dropoutProbability(
        a.range, a.retro, 0.0005f, 0.03f, 150.0f,
        45.0f, 100.0f, 55.0f, 120.0f, 0.15f);
    EXPECT_LT(clear_drop, 0.1f)
        << "a clear 60 m wall is well inside the OS1 envelope";

    rc::ScanParams smoky = baseParams();
    addObscurant(smoky, obscurantAt(rc::ObscurantType::kBox, 20.0f,
                                    5.0f, 8.0f, 8.0f, 0.2f, kNoBackscatter));
    const RayResult b = castOne(scene, xf, smoky);
    ASSERT_NEAR(b.range, 60.0f, 1e-3f);
    const float smoky_drop = rpmath::dropoutProbability(
        b.range, b.retro, 0.0005f, 0.03f, 150.0f,
        45.0f, 100.0f, 55.0f, 120.0f, 0.15f);
    EXPECT_GT(smoky_drop, clear_drop);
    EXPECT_GT(smoky_drop, 0.9f)
        << "through tau=2 smoke the wall should be overwhelmingly lost";
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

// ── Per-candidate attenuation ────────────────────────────────────────────────
//
// Candidates on one beam sit at DIFFERENT depths, so the medium dims them by
// different amounts. The arbitration therefore has to compare what the
// detector would actually receive, not the clear-air powers.

namespace {

/// Sub-millimetre pane so the glass continuation clears its own back face
/// (kRcSegEps is 1 mm), with a bright object behind it.
void makePaneAndObject(rc::Scene & scene, std::vector<rc::InstanceXform> & xf,
                       float pane_x, float pane_retro, float transmit,
                       float obj_face_x, float obj_retro)
{
    const float pane[3] = {2.0e-4f, 4.0f, 4.0f};
    const int pi = scene.addInstance(rc::GeomType::kBox, pane, pane_retro,
                                     -1, 0.0f, transmit);
    const float pt[3] = {pane_x, 0.0f, 0.0f};
    rc::InstanceXform px;
    scene.computeXform(pi, kIdentityR, pt, px);
    xf.push_back(px);
    makeWall(scene, xf, obj_face_x, obj_retro);
}

}  // namespace

TEST(ObscurantArbitration, CandidatesCompeteAfterTheirOwnAttenuation)
{
    // Pane (transmit 0.9, retro 0.05) at 5 m, bright object at 12 m, uniform
    // smoke over the whole path. Clear air: the object wins by 25x. But the
    // object is 7 m deeper, so it loses exp(-2*sigma*14) more than the pane —
    // the winner crosses over at sigma = ln(25.3)/(2*7) = 0.231 /m, a
    // visibility of about 17 m. Comparing unattenuated powers keeps reporting
    // the object well past that.
    rc::Scene scene;
    std::vector<rc::InstanceXform> xf;
    makePaneAndObject(scene, xf, 5.0f, 0.05f, 0.9f, 12.0f, 0.9f);

    auto reported = [&](float sigma) {
        rc::ScanParams sp = baseParams();
        if (sigma > 0.0f) {
            // Slab spanning [0, 15] m; backscatter suppressed so this
            // isolates the arbitration from the medium's own return.
            addObscurant(sp, obscurantAt(rc::ObscurantType::kBox, 7.5f,
                                         7.5f, 6.0f, 6.0f, sigma,
                                         kNoBackscatter));
        }
        return castOne(scene, xf, sp);
    };

    // Below the crossover the object still wins, as in clear air.
    for (float sigma : {0.0f, 0.10f, 0.20f}) {
        const RayResult r = reported(sigma);
        EXPECT_NEAR(r.range, 12.0f, 1e-2f) << "sigma=" << sigma;
        EXPECT_NEAR(r.retro, 0.9f * 0.81f * std::exp(-2.0f * sigma * 12.0f),
                    1e-5f) << "sigma=" << sigma;
    }

    // Past it the pane wins — and is reported with the pane's own optical
    // depth (5 m), not the object's.
    const RayResult thick = reported(0.30f);
    EXPECT_NEAR(thick.range, 5.0f, 1e-2f)
        << "past the crossover the near surface must win";
    EXPECT_NEAR(thick.retro, 0.05f * (1.0f - 0.9f) * std::exp(-2.0f * 0.3f * 5.0f),
                1e-6f);
}

TEST(ObscurantArbitration, EqualDepthCandidatesAreUnaffected)
{
    // The correction must only bite when candidate depths differ. Haze
    // entirely in FRONT of the whole stack attenuates every candidate
    // identically, so the winner and the ratio between candidates are
    // unchanged — only the overall scale moves.
    rc::Scene scene;
    std::vector<rc::InstanceXform> xf;
    makePaneAndObject(scene, xf, 5.0f, 0.05f, 0.9f, 12.0f, 0.9f);

    const RayResult clear = castOne(scene, xf, baseParams());

    rc::ScanParams sp = baseParams();
    addObscurant(sp, obscurantAt(rc::ObscurantType::kBox, 2.0f,
                                 1.0f, 4.0f, 4.0f, 0.4f, kNoBackscatter));
    const RayResult hazy = castOne(scene, xf, sp);
    EXPECT_FLOAT_EQ(hazy.range, clear.range);
    EXPECT_NEAR(hazy.retro, clear.retro * std::exp(-2.0f * 0.4f * 2.0f), 1e-7f);
}

TEST(ObscurantArbitration, MirrorGhostIntegratesItsBentPathNotAStraightLine)
{
    // A mirror at 4 m reflects the beam sideways onto a target. The ghost is
    // REPORTED at the folded path length (~14 m), but the pulse never travels
    // the straight line out to 14 m. Smoke placed only along that straight
    // continuation — behind the mirror, off the real path — must therefore
    // not attenuate the ghost at all.
    rc::Scene scene;
    std::vector<rc::InstanceXform> xf;

    // Mirror at x=4, tilted 45 deg about z. Its face normal is local +x,
    // which maps to world (0.707, 0.707, 0), so a +x beam reflects to -y.
    const float mirror[3] = {0.05f, 3.0f, 3.0f};
    const int mi = scene.addInstance(rc::GeomType::kBox, mirror, 0.0f, -1,
                                     1.0f, 0.0f);
    const float c = std::cos(static_cast<float>(M_PI) / 4.0f);
    const float rm[9] = {c, -c, 0, c, c, 0, 0, 0, 1};
    const float mt[3] = {4.0f, 0.0f, 0.0f};
    rc::InstanceXform mx;
    scene.computeXform(mi, rm, mt, mx);
    xf.push_back(mx);

    // Target off to -y, where the reflected leg lands.
    const float tgt[3] = {3.0f, 0.3f, 3.0f};
    const int ti = scene.addInstance(rc::GeomType::kBox, tgt, 0.9f);
    const float tt[3] = {4.0f, -10.0f, 0.0f};
    rc::InstanceXform tx;
    scene.computeXform(ti, kIdentityR, tt, tx);
    xf.push_back(tx);

    const RayResult clear = castOne(scene, xf, baseParams());
    ASSERT_GT(clear.range, 8.0f) << "expected the ghost, not the mirror face";

    // Smoke ONLY behind the mirror, on the straight line the ghost is
    // reported along but never travels.
    rc::ScanParams sp = baseParams();
    addObscurant(sp, obscurantAt(rc::ObscurantType::kBox, 9.0f,
                                 3.0f, 2.0f, 2.0f, 0.5f, kNoBackscatter));
    const RayResult behind = castOne(scene, xf, sp);
    EXPECT_FLOAT_EQ(behind.range, clear.range);
    EXPECT_NEAR(behind.retro, clear.retro, 1e-7f)
        << "medium behind the mirror is not on the ghost's path";
}

}  // namespace gz_gpu_ouster_lidar
