// Copyright 2026 John C. Furey
// SPDX-License-Identifier: Apache-2.0
//
// The world-side half of the obscuration model: turning Gazebo particle
// emitters and authored <obscurant> volumes into the rc::RcObscurant PODs
// the raycast kernel consumes, and choosing which ones survive the fixed
// per-scan cap.

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/ParentEntity.hh>
#include <gz/sim/components/ParticleEmitter.hh>
#include <gz/sim/components/Pose.hh>
#include <sdf/Root.hh>
#include <sdf/parser.hh>

#include "obscurants.hpp"

namespace gz_gpu_ouster_lidar {

namespace {

/// Wrap plugin-element children in a minimal world and hand back the
/// <plugin> element, exactly as gz-sim would pass it to Configure().
::sdf::ElementConstPtr pluginElement(const std::string & children)
{
    static std::vector<::sdf::SDFPtr> keep_alive;  // elements are non-owning
    auto doc = std::make_shared<::sdf::SDF>();
    ::sdf::init(doc);
    const std::string xml =
        "<?xml version='1.0'?><sdf version='1.9'><world name='w'>"
        "<plugin filename='f' name='n'>" + children +
        "</plugin></world></sdf>";
    if (!::sdf::readString(xml, doc)) return nullptr;
    keep_alive.push_back(doc);
    return doc->Root()->GetElement("world")->GetElement("plugin");
}

::gz::msgs::ParticleEmitter boxEmitter()
{
    ::gz::msgs::ParticleEmitter em;
    em.set_name("smoke");
    em.set_type(::gz::msgs::ParticleEmitter::BOX);
    em.mutable_size()->set_x(4.0);
    em.mutable_size()->set_y(6.0);
    em.mutable_size()->set_z(0.0);      // flat patch, as gz demos author it
    em.mutable_lifetime()->set_data(10.0f);
    em.mutable_min_velocity()->set_data(0.2f);
    em.mutable_max_velocity()->set_data(0.4f);
    em.mutable_particle_scatter_ratio()->set_data(0.5f);
    return em;
}

/// Map a world point through the stored world→local transform.
::gz::math::Vector3d toLocal(const rc::RcObscurant & ob,
                             const ::gz::math::Vector3d & p)
{
    return {ob.r[0] * p.X() + ob.r[1] * p.Y() + ob.r[2] * p.Z() + ob.t[0],
            ob.r[3] * p.X() + ob.r[4] * p.Y() + ob.r[5] * p.Z() + ob.t[1],
            ob.r[6] * p.X() + ob.r[7] * p.Y() + ob.r[8] * p.Z() + ob.t[2]};
}

rc::RcObscurant at(double x, double radius)
{
    rc::RcObscurant ob;
    makeObscurant(rc::ObscurantType::kEllipsoid,
                  ::gz::math::Pose3d(x, 0, 0, 0, 0, 0),
                  {radius, radius, radius}, 0.5, 50.0, 0.8, 1.0, ob);
    return ob;
}

}  // namespace

// ── Visibility ↔ extinction ──────────────────────────────────────────────────

TEST(ObscurantConfig, VisibilityConvertsThroughKoschmieder)
{
    // σ = 3.912 / V, so V = 3.912 m is exactly σ = 1 /m.
    EXPECT_NEAR(extinctionFromVisibility(3.912), 1.0, 1e-9);
    EXPECT_NEAR(extinctionFromVisibility(100.0), 0.03912, 1e-9);
    // Non-positive means "not specified", not "infinitely dense".
    EXPECT_EQ(extinctionFromVisibility(0.0), 0.0);
    EXPECT_EQ(extinctionFromVisibility(-5.0), 0.0);
}

// ── Transform convention ─────────────────────────────────────────────────────

TEST(ObscurantConfig, StoresTheWorldToLocalTransform)
{
    rc::RcObscurant ob;
    // 90° about z, translated: local +x maps to world +y.
    makeObscurant(rc::ObscurantType::kBox,
                  ::gz::math::Pose3d(3, 4, 5, 0, 0, GZ_PI / 2),
                  {1, 2, 3}, 0.25, 40.0, 0.7, 0.65, ob);

    const auto centre = toLocal(ob, {3, 4, 5});
    EXPECT_NEAR(centre.X(), 0.0, 1e-5);
    EXPECT_NEAR(centre.Y(), 0.0, 1e-5);
    EXPECT_NEAR(centre.Z(), 0.0, 1e-5);

    // World +y at unit distance from the centre is local +x.
    const auto along = toLocal(ob, {3, 5, 5});
    EXPECT_NEAR(along.X(), 1.0, 1e-5);
    EXPECT_NEAR(along.Y(), 0.0, 1e-5);

    EXPECT_FLOAT_EQ(ob.half[0], 1.0f);
    EXPECT_FLOAT_EQ(ob.half[2], 3.0f);
    EXPECT_FLOAT_EQ(ob.sigma, 0.25f);
    EXPECT_FLOAT_EQ(ob.lidar_ratio, 40.0f);
    EXPECT_FLOAT_EQ(ob.albedo, 0.7f);
    EXPECT_FLOAT_EQ(ob.ms_factor, 0.65f);
    EXPECT_EQ(ob.type, rc::ObscurantType::kBox);
}

TEST(ObscurantConfig, RejectsDegenerateOpticalParameters)
{
    rc::RcObscurant ob;
    // A zero or negative lidar ratio would divide by zero in the kernel and
    // emit a return of infinite power.
    makeObscurant(rc::ObscurantType::kBox, {}, {1, 1, 1}, 0.5, 0.0, 2.0, 5.0,
                  ob);
    EXPECT_FLOAT_EQ(ob.lidar_ratio,
                    static_cast<float>(kObscurantLidarRatio));
    EXPECT_FLOAT_EQ(ob.albedo, 1.0f) << "albedo must clamp into [0, 1]";
    EXPECT_FLOAT_EQ(ob.ms_factor, 1.0f)
        << "eta above 1 would mean losing more than all the light";

    makeObscurant(rc::ObscurantType::kBox, {}, {-1, 1, 1}, -3.0, 50.0, -1.0,
                  -0.5, ob);
    EXPECT_FLOAT_EQ(ob.half[0], 0.0f);
    EXPECT_FLOAT_EQ(ob.sigma, 0.0f);
    EXPECT_FLOAT_EQ(ob.albedo, 0.0f);
    // eta floors above zero: a medium that returns backscatter while being
    // perfectly transparent to the same beam is not a physical state.
    EXPECT_GT(ob.ms_factor, 0.0f);
    EXPECT_LT(ob.ms_factor, 0.01f);
}

// ── Authored volumes ─────────────────────────────────────────────────────────

TEST(ObscurantConfig, AuthoredVolumeUsesHalfOfTheFullSize)
{
    ObscurantVolume v;
    v.type = rc::ObscurantType::kCylinder;
    v.pose = ::gz::math::Pose3d(10, 0, 1, 0, 0, 0);
    v.size = {6, 6, 4};
    v.extinction = 0.8;
    rc::RcObscurant ob;
    ASSERT_TRUE(obscurantFromVolume(v, ob));
    EXPECT_FLOAT_EQ(ob.half[0], 3.0f);
    EXPECT_FLOAT_EQ(ob.half[2], 2.0f);
    EXPECT_FLOAT_EQ(ob.sigma, 0.8f);
    EXPECT_EQ(ob.type, rc::ObscurantType::kCylinder);
}

TEST(ObscurantConfig, VolumeWithoutExtinctionIsRejected)
{
    ObscurantVolume v;
    v.size = {5, 5, 5};
    rc::RcObscurant ob;
    EXPECT_FALSE(obscurantFromVolume(v, ob));
}

TEST(ObscurantConfig, ActiveTracksWhetherAnythingCanObscure)
{
    ObscurantConfig cfg;
    EXPECT_TRUE(cfg.active()) << "particle mirroring is on by default";

    cfg.mirror_particles = false;
    EXPECT_FALSE(cfg.active());

    ObscurantVolume dead;
    cfg.volumes.push_back(dead);
    EXPECT_FALSE(cfg.active()) << "a zero-extinction volume is not active";

    cfg.volumes.back().extinction = 0.1;
    EXPECT_TRUE(cfg.active());

    cfg.volumes.clear();
    cfg.mirror_particles = true;
    cfg.particle_extinction = 0.0;
    EXPECT_FALSE(cfg.active()) << "mirroring with zero scale obscures nothing";
}

// ── Particle emitters ────────────────────────────────────────────────────────

TEST(ObscurantConfig, EmitterVolumeIsDilatedByParticleTravel)
{
    ObscurantConfig cfg;
    rc::RcObscurant ob;
    ASSERT_TRUE(obscurantFromEmitter(boxEmitter(), {}, cfg, ob));

    // v̄ = 0.3 m/s over a 10 s lifetime → 3 m of travel, added to each half
    // extent (size/2 = 2, 3, 0).
    EXPECT_FLOAT_EQ(ob.half[0], 5.0f);
    EXPECT_FLOAT_EQ(ob.half[1], 6.0f);
    EXPECT_FLOAT_EQ(ob.half[2], 3.0f) << "a flat patch gains real thickness";
    EXPECT_EQ(ob.type, rc::ObscurantType::kBox);

    cfg.particle_growth = 0.0;
    ASSERT_TRUE(obscurantFromEmitter(boxEmitter(), {}, cfg, ob));
    EXPECT_FLOAT_EQ(ob.half[0], 2.0f);
    EXPECT_FLOAT_EQ(ob.half[2], 0.0f);
}

TEST(ObscurantConfig, EmitterExtinctionScalesWithScatterRatio)
{
    ObscurantConfig cfg;
    cfg.particle_extinction = 2.0;
    rc::RcObscurant ob;
    ASSERT_TRUE(obscurantFromEmitter(boxEmitter(), {}, cfg, ob));
    EXPECT_FLOAT_EQ(ob.sigma, 1.0f);   // 2.0 × 0.5

    // An emitter that never set the field takes gz-rendering's default.
    auto em = boxEmitter();
    em.clear_particle_scatter_ratio();
    ASSERT_TRUE(obscurantFromEmitter(em, {}, cfg, ob));
    EXPECT_FLOAT_EQ(ob.sigma,
                    static_cast<float>(2.0 * kGzDefaultScatterRatio));
}

TEST(ObscurantConfig, EmitterThatIsNotEmittingObscuresNothing)
{
    ObscurantConfig cfg;
    rc::RcObscurant ob;

    auto em = boxEmitter();
    em.mutable_emitting()->set_data(false);
    EXPECT_FALSE(obscurantFromEmitter(em, {}, cfg, ob));

    em.mutable_emitting()->set_data(true);
    EXPECT_TRUE(obscurantFromEmitter(em, {}, cfg, ob));

    // An unset flag means "was never toggled", i.e. still emitting.
    em.clear_emitting();
    EXPECT_TRUE(obscurantFromEmitter(em, {}, cfg, ob));

    em.mutable_particle_scatter_ratio()->set_data(0.0f);
    EXPECT_FALSE(obscurantFromEmitter(em, {}, cfg, ob));
}

TEST(ObscurantConfig, PointEmitterBecomesTheTravelBall)
{
    // gz ignores <size> for a point emitter, so the plume IS the dilation.
    ObscurantConfig cfg;
    auto em = boxEmitter();
    em.set_type(::gz::msgs::ParticleEmitter::POINT);
    rc::RcObscurant ob;
    ASSERT_TRUE(obscurantFromEmitter(em, {}, cfg, ob));
    EXPECT_EQ(ob.type, rc::ObscurantType::kEllipsoid);
    EXPECT_FLOAT_EQ(ob.half[0], 3.0f);
    EXPECT_FLOAT_EQ(ob.half[1], 3.0f);
    EXPECT_FLOAT_EQ(ob.half[2], 3.0f);

    // ...and with no travel at all it collapses, so it is dropped rather
    // than emitted as a zero-volume cloud.
    em.mutable_lifetime()->set_data(0.0f);
    EXPECT_FALSE(obscurantFromEmitter(em, {}, cfg, ob));
}

TEST(ObscurantConfig, EmitterTypesMapToVolumeTypes)
{
    ObscurantConfig cfg;
    rc::RcObscurant ob;
    const std::pair<::gz::msgs::ParticleEmitter::EmitterType,
                    rc::ObscurantType> cases[] = {
        {::gz::msgs::ParticleEmitter::BOX, rc::ObscurantType::kBox},
        {::gz::msgs::ParticleEmitter::CYLINDER, rc::ObscurantType::kCylinder},
        {::gz::msgs::ParticleEmitter::ELLIPSOID,
         rc::ObscurantType::kEllipsoid},
    };
    for (const auto & [in, want] : cases) {
        auto em = boxEmitter();
        em.set_type(in);
        ASSERT_TRUE(obscurantFromEmitter(em, {}, cfg, ob));
        EXPECT_EQ(ob.type, want) << "emitter type " << static_cast<int>(in);
    }
}

TEST(ObscurantConfig, EmitterInheritsTheConfiguredOpticalDefaults)
{
    ObscurantConfig cfg;
    cfg.lidar_ratio = 18.0;   // fog
    cfg.albedo = 0.95;
    cfg.multiple_scattering = 0.6;
    rc::RcObscurant ob;
    ASSERT_TRUE(obscurantFromEmitter(boxEmitter(), {}, cfg, ob));
    EXPECT_FLOAT_EQ(ob.lidar_ratio, 18.0f);
    EXPECT_FLOAT_EQ(ob.albedo, 0.95f);
    EXPECT_FLOAT_EQ(ob.ms_factor, 0.6f);
}

TEST(ObscurantConfig, EmitterIsPlacedAtItsWorldPose)
{
    ObscurantConfig cfg;
    cfg.particle_growth = 0.0;
    rc::RcObscurant ob;
    const ::gz::math::Pose3d world(7, -2, 1.5, 0, 0, 0);
    ASSERT_TRUE(obscurantFromEmitter(boxEmitter(), world, cfg, ob));
    const auto centre = toLocal(ob, world.Pos());
    EXPECT_NEAR(centre.X(), 0.0, 1e-5);
    EXPECT_NEAR(centre.Y(), 0.0, 1e-5);
    EXPECT_NEAR(centre.Z(), 0.0, 1e-5);
}

// ── SDF parsing ──────────────────────────────────────────────────────────────

TEST(ObscurantSdf, EmptyPluginKeepsTheDocumentedDefaults)
{
    const auto cfg = parseObscurantConfig(pluginElement(""));
    EXPECT_TRUE(cfg.mirror_particles);
    EXPECT_DOUBLE_EQ(cfg.particle_extinction, kParticleExtinction);
    EXPECT_DOUBLE_EQ(cfg.particle_growth, kParticleGrowth);
    EXPECT_DOUBLE_EQ(cfg.lidar_ratio, kObscurantLidarRatio);
    EXPECT_DOUBLE_EQ(cfg.albedo, kObscurantAlbedo);
    EXPECT_DOUBLE_EQ(cfg.multiple_scattering, 1.0)
        << "the default must be the pure single-scattering limit";
    EXPECT_DOUBLE_EQ(cfg.pulse_gate_m, kObscurantPulseGate);
    EXPECT_TRUE(cfg.volumes.empty());
}

TEST(ObscurantSdf, NullElementIsSafe)
{
    const auto cfg = parseObscurantConfig(nullptr);
    EXPECT_TRUE(cfg.volumes.empty());
    EXPECT_DOUBLE_EQ(cfg.pulse_gate_m, kObscurantPulseGate);
}

TEST(ObscurantSdf, ScalarKnobsAreRead)
{
    const auto cfg = parseObscurantConfig(pluginElement(
        "<particle_obscuration>false</particle_obscuration>"
        "<particle_extinction>0.4</particle_extinction>"
        "<particle_growth>0.25</particle_growth>"
        "<obscurant_lidar_ratio>18</obscurant_lidar_ratio>"
        "<obscurant_albedo>0.95</obscurant_albedo>"
        "<obscurant_multiple_scattering>0.6</obscurant_multiple_scattering>"
        "<pulse_length>0.9</pulse_length>"));
    EXPECT_FALSE(cfg.mirror_particles);
    EXPECT_DOUBLE_EQ(cfg.particle_extinction, 0.4);
    EXPECT_DOUBLE_EQ(cfg.particle_growth, 0.25);
    EXPECT_DOUBLE_EQ(cfg.lidar_ratio, 18.0);
    EXPECT_DOUBLE_EQ(cfg.albedo, 0.95);
    EXPECT_DOUBLE_EQ(cfg.multiple_scattering, 0.6);
    EXPECT_DOUBLE_EQ(cfg.pulse_gate_m, 0.9);
}

TEST(ObscurantSdf, NegativeScalarsAreFloored)
{
    const auto cfg = parseObscurantConfig(pluginElement(
        "<particle_extinction>-1</particle_extinction>"
        "<particle_growth>-2</particle_growth>"
        "<pulse_length>-3</pulse_length>"));
    EXPECT_DOUBLE_EQ(cfg.particle_extinction, 0.0);
    EXPECT_DOUBLE_EQ(cfg.particle_growth, 0.0);
    EXPECT_DOUBLE_EQ(cfg.pulse_gate_m, 0.0);
}

TEST(ObscurantSdf, ReadsEveryObscurantBlock)
{
    const auto cfg = parseObscurantConfig(pluginElement(
        "<obscurant>"
        "  <type>box</type>"
        "  <pose>10 2 1.5 0 0 0.5</pose>"
        "  <size>6 4 3</size>"
        "  <extinction>0.6</extinction>"
        "  <lidar_ratio>30</lidar_ratio>"
        "  <albedo>0.5</albedo>"
        "  <multiple_scattering>0.7</multiple_scattering>"
        "</obscurant>"
        "<obscurant>"
        "  <type>cylinder</type>"
        "  <visibility>10</visibility>"
        "</obscurant>"));
    ASSERT_EQ(cfg.volumes.size(), 2u);

    EXPECT_EQ(cfg.volumes[0].type, rc::ObscurantType::kBox);
    EXPECT_DOUBLE_EQ(cfg.volumes[0].pose.Pos().X(), 10.0);
    EXPECT_DOUBLE_EQ(cfg.volumes[0].pose.Rot().Yaw(), 0.5);
    EXPECT_DOUBLE_EQ(cfg.volumes[0].size.Y(), 4.0);
    EXPECT_DOUBLE_EQ(cfg.volumes[0].extinction, 0.6);
    EXPECT_DOUBLE_EQ(cfg.volumes[0].lidar_ratio, 30.0);
    EXPECT_DOUBLE_EQ(cfg.volumes[0].albedo, 0.5);
    EXPECT_DOUBLE_EQ(cfg.volumes[0].multiple_scattering, 0.7);

    EXPECT_EQ(cfg.volumes[1].type, rc::ObscurantType::kCylinder);
    EXPECT_NEAR(cfg.volumes[1].extinction, kKoschmieder / 10.0, 1e-12);
}

TEST(ObscurantSdf, VolumeInheritsTheGlobalOpticalDefaults)
{
    const auto cfg = parseObscurantConfig(pluginElement(
        "<obscurant_lidar_ratio>18</obscurant_lidar_ratio>"
        "<obscurant_albedo>0.95</obscurant_albedo>"
        "<obscurant_multiple_scattering>0.55</obscurant_multiple_scattering>"
        "<obscurant><extinction>0.2</extinction></obscurant>"));
    ASSERT_EQ(cfg.volumes.size(), 1u);
    EXPECT_DOUBLE_EQ(cfg.volumes[0].lidar_ratio, 18.0);
    EXPECT_DOUBLE_EQ(cfg.volumes[0].albedo, 0.95);
    EXPECT_DOUBLE_EQ(cfg.volumes[0].multiple_scattering, 0.55);
}

TEST(ObscurantSdf, ExplicitExtinctionBeatsVisibility)
{
    const auto cfg = parseObscurantConfig(pluginElement(
        "<obscurant>"
        "  <extinction>0.25</extinction><visibility>1</visibility>"
        "</obscurant>"));
    ASSERT_EQ(cfg.volumes.size(), 1u);
    EXPECT_DOUBLE_EQ(cfg.volumes[0].extinction, 0.25);
}

TEST(ObscurantSdf, OpticallyEmptyOrUnknownBlocksDoNotAbortTheRest)
{
    const auto cfg = parseObscurantConfig(pluginElement(
        "<obscurant><size>1 1 1</size></obscurant>"          // no density
        "<obscurant><type>banana</type>"                     // unknown type
        "  <extinction>0.3</extinction></obscurant>"));
    ASSERT_EQ(cfg.volumes.size(), 1u) << "only the dense one survives";
    EXPECT_EQ(cfg.volumes[0].type, rc::ObscurantType::kEllipsoid)
        << "an unknown type must fall back, not drop the volume";
    EXPECT_DOUBLE_EQ(cfg.volumes[0].extinction, 0.3);
}

// ── Nearest-N selection ──────────────────────────────────────────────────────

TEST(ObscurantConfig, KeepNearestIsANoOpWhenItAlreadyFits)
{
    std::vector<rc::RcObscurant> obs{at(50, 1), at(10, 1), at(30, 1)};
    EXPECT_EQ(keepNearest(obs, {0, 0, 0}, 8), 0u);
    ASSERT_EQ(obs.size(), 3u);
    EXPECT_FLOAT_EQ(obs[0].t[0], -50.0f) << "order must be preserved";
}

TEST(ObscurantConfig, KeepNearestDropsTheFarthest)
{
    std::vector<rc::RcObscurant> obs{at(50, 1), at(10, 1), at(30, 1),
                                     at(5, 1)};
    // The drop count is what the mirror surfaces to the user; silent
    // truncation is what made this cap hard to diagnose in the first place.
    EXPECT_EQ(keepNearest(obs, {0, 0, 0}, 2), 2u);
    ASSERT_EQ(obs.size(), 2u);
    std::vector<float> centres{-obs[0].t[0], -obs[1].t[0]};
    std::sort(centres.begin(), centres.end());
    EXPECT_FLOAT_EQ(centres[0], 5.0f);
    EXPECT_FLOAT_EQ(centres[1], 10.0f);
}

TEST(ObscurantConfig, KeepNearestMeasuresToTheVolumeNotItsCentre)
{
    // A huge cloud the sensor is standing inside has a distant centre but
    // zero distance to its surface; it must outrank a small nearby puff.
    std::vector<rc::RcObscurant> obs{at(20, 1.0), at(60, 100.0)};
    keepNearest(obs, {0, 0, 0}, 1);
    ASSERT_EQ(obs.size(), 1u);
    EXPECT_FLOAT_EQ(obs[0].half[0], 100.0f);
}

// ── Gathering out of a live ECM ──────────────────────────────────────────────
//
// This is the half that only shows up in a running world: emitters hang off a
// link inside a model, so their world pose comes from the parent chain, not
// from the emitter's own Pose component.

namespace {

/// model(at `pose`) -> link -> particle emitter, exactly the nesting
/// SdfEntityCreator builds for <model><link><particle_emitter>.
::gz::sim::Entity spawnEmitter(::gz::sim::EntityComponentManager & ecm,
                               const std::string & name,
                               const ::gz::math::Pose3d & model_pose,
                               const ::gz::msgs::ParticleEmitter & em,
                               const ::gz::math::Pose3d & link_pose = {},
                               const ::gz::math::Pose3d & em_pose = {})
{
    const auto model = ecm.CreateEntity();
    ecm.CreateComponent(model, ::gz::sim::components::Model());
    ecm.CreateComponent(model, ::gz::sim::components::Name(name));
    ecm.CreateComponent(model, ::gz::sim::components::Pose(model_pose));

    const auto link = ecm.CreateEntity();
    ecm.CreateComponent(link, ::gz::sim::components::ParentEntity(model));
    ecm.CreateComponent(link, ::gz::sim::components::Pose(link_pose));

    const auto ent = ecm.CreateEntity();
    ecm.CreateComponent(ent, ::gz::sim::components::ParentEntity(link));
    ecm.CreateComponent(ent, ::gz::sim::components::Pose(em_pose));
    ecm.CreateComponent(ent, ::gz::sim::components::ParticleEmitter(em));
    return ent;
}

}  // namespace

TEST(ObscurantEcm, MirrorsEveryEmitterAtItsWorldPose)
{
    ::gz::sim::EntityComponentManager ecm;
    for (int i = 0; i < 5; ++i) {
        spawnEmitter(ecm, "cloud" + std::to_string(i),
                     ::gz::math::Pose3d(3.0 * i, 0, 1, 0, 0, 0), boxEmitter());
    }

    ObscurantConfig cfg;
    cfg.particle_growth = 0.0;
    std::vector<rc::RcObscurant> out;
    EXPECT_EQ(gatherObscurants(cfg, ecm, {0, 0, 0}, out), 0u);
    ASSERT_EQ(out.size(), 5u);

    // Each volume must sit at its own model's pose, not stacked at the origin.
    std::vector<double> xs;
    for (const auto & ob : out) {
        xs.push_back(-(ob.r[0] * ob.t[0] + ob.r[3] * ob.t[1] +
                       ob.r[6] * ob.t[2]));
    }
    std::sort(xs.begin(), xs.end());
    for (int i = 0; i < 5; ++i) {
        EXPECT_NEAR(xs[i], 3.0 * i, 1e-6) << "volume " << i << " misplaced";
    }
}

TEST(ObscurantEcm, ComposesTheWholeParentChain)
{
    // A pose on the model, the link AND the emitter itself must all compose:
    // reading only the emitter's own Pose leaves every cloud at the origin.
    ::gz::sim::EntityComponentManager ecm;
    spawnEmitter(ecm, "nested", ::gz::math::Pose3d(10, 0, 0, 0, 0, 0),
                 boxEmitter(), ::gz::math::Pose3d(0, 4, 0, 0, 0, 0),
                 ::gz::math::Pose3d(0, 0, 2, 0, 0, 0));

    ObscurantConfig cfg;
    cfg.particle_growth = 0.0;
    std::vector<rc::RcObscurant> out;
    gatherObscurants(cfg, ecm, {0, 0, 0}, out);
    ASSERT_EQ(out.size(), 1u);
    const auto centre = toLocal(out[0], {10, 4, 2});
    EXPECT_NEAR(centre.X(), 0.0, 1e-6);
    EXPECT_NEAR(centre.Y(), 0.0, 1e-6);
    EXPECT_NEAR(centre.Z(), 0.0, 1e-6);
}

TEST(ObscurantEcm, AuthoredVolumesAndEmittersBothAppear)
{
    ::gz::sim::EntityComponentManager ecm;
    spawnEmitter(ecm, "cloud", ::gz::math::Pose3d(5, 0, 1, 0, 0, 0),
                 boxEmitter());

    ObscurantConfig cfg;
    ObscurantVolume v;
    v.pose = ::gz::math::Pose3d(-5, 0, 1, 0, 0, 0);
    v.extinction = 0.3;
    cfg.volumes.push_back(v);

    std::vector<rc::RcObscurant> out;
    gatherObscurants(cfg, ecm, {0, 0, 0}, out);
    EXPECT_EQ(out.size(), 2u);

    cfg.mirror_particles = false;
    gatherObscurants(cfg, ecm, {0, 0, 0}, out);
    ASSERT_EQ(out.size(), 1u) << "particle mirroring must be switchable off";
    EXPECT_FLOAT_EQ(out[0].sigma, 0.3f);
}

TEST(ObscurantEcm, ReportsWhatTheCapDiscarded)
{
    ::gz::sim::EntityComponentManager ecm;
    const int n = rc::kMaxObscurants + 3;
    for (int i = 0; i < n; ++i) {
        spawnEmitter(ecm, "cloud" + std::to_string(i),
                     ::gz::math::Pose3d(2.0 * (i + 1), 0, 1, 0, 0, 0),
                     boxEmitter());
    }
    ObscurantConfig cfg;
    std::vector<rc::RcObscurant> out;
    EXPECT_EQ(gatherObscurants(cfg, ecm, {0, 0, 0}, out), 3u);
    EXPECT_EQ(out.size(), static_cast<size_t>(rc::kMaxObscurants));
}

TEST(ObscurantEcm, ClearsPriorContentsOnEveryGather)
{
    ::gz::sim::EntityComponentManager ecm;
    spawnEmitter(ecm, "cloud", ::gz::math::Pose3d(5, 0, 1, 0, 0, 0),
                 boxEmitter());
    ObscurantConfig cfg;
    std::vector<rc::RcObscurant> out{at(1, 1), at(2, 1)};
    gatherObscurants(cfg, ecm, {0, 0, 0}, out);
    EXPECT_EQ(out.size(), 1u) << "stale volumes would accumulate every scan";
}

}  // namespace gz_gpu_ouster_lidar
