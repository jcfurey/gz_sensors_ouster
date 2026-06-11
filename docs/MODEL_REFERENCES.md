# Sensor Model — Literature References

This document maps every physical effect the plugin models to the equation it
implements, where it lives in the code, and the literature that grounds it.
It also lists the deliberate simplifications, so the boundary between "modeled"
and "not modeled" is explicit.

Conventions: ρ is the target's diffuse reflectance (`laser_retro`), R the
range, α the angle between the beam and the surface normal at the hit.

## Modeled effects

### 1. Received signal — extended-Lambertian lidar equation

**Model:** `signal = base_signal · ρ_app / R²` with
`ρ_app = ρ · cos(α)` (raycast mode), implemented in
`rpmath::signalFromRange` (`cuda/ray_processor_math.hpp`) and
`rc::rcCosIncidence` / `rcCastOneRay` (`cuda/raycast_math.hpp`).

For a target larger than the beam footprint with Lambertian reflectance, the
lidar range equation reduces to `P_r ∝ ρ · cos(α) / R²`. This is the standard
form used both for radiometric calibration of real scanners and by physically
based simulators:

- Kashani, Olsen, Parrish, Wilson — *A Review of LIDAR Radiometric Processing:
  From Ad Hoc Intensity Correction to Rigorous Radiometric Calibration*,
  Sensors 15(11), 2015. <https://pmc.ncbi.nlm.nih.gov/articles/PMC4701271/>
- Kaasalainen et al. — *Analysis of Incidence Angle and Distance Effects on
  Terrestrial Laser Scanner Intensity: Search for Correction Methods*,
  Remote Sensing 3(10), 2011. <https://www.mdpi.com/2072-4292/3/10/2207>
- Winiwarter et al. — *Virtual laser scanning with HELIOS++*, Remote Sensing
  of Environment 269, 2022 (the reference open-source simulator; same
  radiometric form). <https://arxiv.org/abs/2101.09154>

Folding cos(α) into the *apparent reflectance* at cast time (rather than into
the signal alone) is deliberate: the Ouster firmware derives its calibrated
reflectivity from the received signal and range, so a real sensor's
reflectivity output *also* drops on oblique surfaces, as do its
detection/precision statistics. One factor at the source propagates to all
of them consistently. Caveats: the pure cosine law is experimentally reliable
only to ~20° incidence (Kaasalainen et al.), hence the conservative
`kRcMinCosInc` clamp; **panels mode** has no per-hit normal (depth images
only) and does not apply the factor.

### 2. Range precision — photon-budget scaling

**Model:** `σ(R, ρ) = lerp(min_std, max_std; R/max_range) · min(1/√ρ_app, 2)`,
implemented in `rpmath::rangeNoiseSigma`.

ToF timing precision scales as `σ ∝ 1/√N` for N detected signal photons, and
N ∝ ρ at fixed range — so the reflectance dependence is `1/√ρ`:

- Hu et al. — *Influence of Waveform Characteristics on LiDAR Ranging Accuracy
  and Precision*, Sensors 18(4), 2018 ("random error = characteristic time of
  the waveform / √(detected photons)").
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC5948876/>
- *Performance Bounds of Ranging Precision in SPAD-Based dToF LiDAR*,
  arXiv:2507.11404 (optimal precision ∝ 1/√N bounds for SPAD detection).

The *range* dependence stays a configurable linear ramp rather than the pure
`R/√ρ` law: published datasheet precision-vs-range curves (e.g. Ouster OS1,
±0.7–5 cm band) fold in firmware filtering that an analytic law does not
capture, so the ramp endpoints are left to the user (see the Sensor Tuning
Guide in the README).

### 3. Dropout — detection failure on weak returns

**Model:** `P_drop = lerp(close, far; R/max_range) · min(1/max(ρ_app, ⅓), 3)`,
implemented in `rpmath::dropoutProbability`.

Detection probability falls when the return signal approaches the detector
threshold; received signal ∝ ρ/R², so miss probability rises with range and
inverse reflectance. The `1/ρ` weighting follows the first-order `1/SNR`
heuristic; intensity-thresholded ray dropping is the standard treatment in
simulation. Additionally, beyond the **reflectance-dependent detection
limit** `d_max(ρ) = max_range·√(ρ/0.8)` the return is always dropped: the
1/R² lidar equation makes the threshold range scale with √ρ, and vendor
range specs are quoted at 80% Lambertian — the OS1's published 120 m @ 80% /
~45 m @ 10% pair matches the √ law's 42 m prediction. (Same construct as
the reflectance limit function `RL(d)` of arXiv:2208.10295 §II-D.)

- Hahner et al. — *LiDAR Snowfall Simulation for Robust 3D Object Detection*,
  CVPR 2022 (returns culled when attenuated intensity falls below the
  detection threshold). <https://arxiv.org/abs/2203.15118>

This is a heuristic, not a calibrated detection model — a rigorous treatment
would integrate the full detection statistics (see §2 references).

### 3b. Solar-background false alarms

**Model:** each no-return pixel becomes a spurious point with probability
`false_alarm_rate` (default 0 = off), at a range uniform over
`(0, max_range]` and a noise-floor signal of 1, implemented in each
backend's noise stage (CPU reference: `ray_processor_cpu_impl.cpp`).

Daytime background photons exceed the detection threshold at a constant
rate, producing false alarms uniformly distributed in time — hence uniform
in range over the unambiguous window — strongest for photon-sensitive
(SPAD/APD) receivers; detection and false-alarm probabilities follow
Neyman–Pearson threshold statistics:

- Jin et al. — *Receiver performance and detection statistics of single
  photon lidar*, IET Radar Sonar Navig. 14, 2020.
- Haider et al., Sensors 22(19), 2022 — sunlight-induced noise listed among
  the receiver effects required for accurate virtual lidar.

Unmodeled refinement: the false-alarm rate should rise with scene/sky
radiance (sun position, bright surfaces) rather than being uniform.

### 4. Signal / near-IR shot noise

**Model:** `σ_channel = √(channel) · noise_scale` (Poisson shot-noise
analogue, Gaussian approximation), applied to `signal` and `near_ir` in each
backend's noise stage.

Photon-counting statistics are Poisson; for the photon counts these channels
represent, `Var = N` (σ = √N) and the Gaussian approximation is standard.
Same photon-budget references as §2.

### 5. Calibrated reflectivity byte

**Model:** ρ_app ∈ [0,1] → linear [0,100]; ρ_app > 1 → log₂ map into
[101,255], implemented in `rpmath::reflectivityToByte` (derivation comment in
`cuda/ray_processor_cpu_impl.cpp`).

Matches Ouster's documented two-band calibrated-reflectivity encoding (linear
percent for diffuse targets, compressed band for retroreflectors):

- Ouster Sensor Docs, *Sensor Data* (REFLECTIVITY field).
  <https://static.ouster.dev/sensor-docs/image_route1/image_route2/sensor_data/sensor-data.html>
- `ouster_client/include/ouster/chanfield.h` (REFLECTIVITY is "calibrated by
  range and sensor sensitivity").

### 6. Near-IR channel semantics

**Model:** `near_ir = ρ · kNearIrScale`, range-independent, in the backend
noise stage.

Ouster's NEAR_IR channel counts **ambient** near-infrared photons (sunlight
reflected off the scene — "the camera in the lidar"), not laser return. Two
properties follow and are honoured by the model: (a) the value tracks the
surface's NIR albedo, for which `laser_retro` is the available proxy, and
(b) like any camera image of an extended scene it is **radiance-invariant
with range** — hence deliberately no 1/R² here:

- Ouster Sensor Docs, *Sensor Data* (NEAR_IR / ambient).
- Ouster blog — *Lidar as a camera*.
  <https://ouster.com/insights/blog/the-camera-is-in-the-lidar>

Unmodeled: actual sun illumination/shadowing (a sim sun model would be
needed); the sensor-incidence cosine folded into ρ_app in raycast mode is a
slight mis-model for this one channel (ambient Lambertian radiance does not
depend on the *viewing* angle), accepted to keep one reflectance value per
return.

### 7. IMU noise — white noise density + bias random walk

**Model:** discrete per-sample sigmas `σ_w = density/√Δt`,
`σ_b,step = walk·√Δt`, bias integrated as a random walk, implemented in
`cuda/imu_noise.{hpp,cpp}`.

This is exactly the two-parameter IMU model used across the
calibration/estimation literature (Allan-variance white-noise + rate-random-
walk segments), with the same continuous→discrete conversions:

- Kalibr wiki — *IMU Noise Model*.
  <https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model>
- IEEE Std 952 (Allan-variance characterisation of gyros) — the underlying
  standard for the density/walk parametrisation datasheets quote.

### 8. Specular and transparent surfaces (raycast mode)

**Model:** per-visual material `(kd, ks, τ)` mirrored from SDF
(`<laser_retro>`, material `<specular>` mean RGB, `<transparency>`); the
monostatic apparent reflectance is

```
ρ_app = kd·cos(α) + ks·max(0, cos 2α)⁸
```

and a transparent first hit (τ ≥ 0.05) casts one continuation segment: the
pane returns `(1−τ)·ρ_app` and the object behind returns `τ²·ρ_app` (the
pulse crosses the pane twice); the **strongest received power `ρ/R²` wins**
(single-return mode). Implemented in `rc::rcApparentReflectance` /
`rcCastOneRay` (`cuda/raycast_math.hpp`) and the ECM material mirroring in
`src/raycast_mirror.cpp`.

This reproduces the empirically documented lidar-on-glass behaviour — a
strong pane return only near surface-normal incidence, the object behind the
glass at weakened intensity otherwise — and the missing-points signature of
glossy/black paint (high ks, tiny kd → returns only from sensor-facing
patches, elevated dropout elsewhere):

- Velas et al. — *Detection and Utilization of Reflection in 3D Lidar
  Scans*, arXiv:1909.12483 (§III: the three glass return cases).
  <https://arxiv.org/abs/1909.12483>
- *Investigation of Automotive LiDAR Vision in Rain from Material and
  Optical Perspectives*, Sensors 24(10), 2024 — material-dependent missing
  points / reduced reflectivity on specular and dark surfaces.
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC11124791/>

The cos(2α)ⁿ lobe is the monostatic Phong form (receiver at the emitter:
`(r̂·(−d̂))ⁿ = cosⁿ 2α`); n = 8 is a fixed qualitative width since SDF
exposes no per-material shininess. Demo objects exercising the model live in
`examples/worlds/turtlebot3_ouster_headless.sdf` (`glass_pane`,
`box_behind_glass`, `glossy_black_box`).

The *mirror ghost* path is also modeled: a hit with `ks ≥ 0.5` casts the
specular bounce; an object found there competes as a ghost candidate along
the **original** beam at the total path length, weighted `((1−τ)·ks)²·ρ`
(the pulse interacts with the mirror twice — glass ghosts weak, true
mirrors strong), exactly the artifact Velas et al. detect and exploit. SDF
has no roughness channel, so `ks = 0.5` is the gloss/mirror discriminator:
keep paint below it, mirrors and glass near 1.

### 9. Motion distortion — rolling-shutter sweep (raycast mode)

**Model:** with `<motion_distortion>true</motion_distortion>` (default off),
column m is cast from the sensor pose at its acquisition time
`t_m = t_scan − T + (m+1)·T/W`, interpolated (linear position + quaternion
SLERP) from a per-sim-tick pose history. Implemented in
`RaycastMirror::buildColumnPoses` (`src/raycast_mirror.cpp`) with per-column
pose tables threaded through `castScan` on all four backends
(`rcCastOneRay` selects `col_r[9m]/col_t[3m]`).

A spinning lidar acquires its W columns over a full period; platform motion
during the sweep skews the cloud by roughly the distance travelled per scan
(decimetres at walking speed, metres in vehicles — deskewing improves
mapping accuracy by up to ~3 m at speed):

- Zhao et al. — *Registration-based point cloud deskewing and dynamic lidar
  simulation*, The Photogrammetric Record 39, 2024.
- Manivasagam et al. — *LiDARsim*, CVPR 2020 (simulates per-ray sensor
  poses); UTIAS Motion-Distorted Lidar Simulation Dataset.
- Lovegrove et al. — *Spline Fusion*, BMVC 2013 / Furgale et al., ICRA 2012
  — the continuous-time pose treatment for rolling-shutter sensors; sim
  playback of known poses needs only the interpolation, not the spline
  estimation machinery.

Conventions and caveats: relative intra-scan timing matches the packet
encoder exactly (its per-column timestamps are spaced `T/W` apart), so
IMU-based de-skew pipelines see consistent data. The encoder stamps the
scan window starting at the trigger, while the simulated acquisition times
end at it — absolute timestamps lead the geometry by one period, which only
matters for TF-lookup-based de-skew against sim ground truth. Ego motion
only: other agents' poses stay at the scan-trigger snapshot (the dominant
term; per-agent sweep interpolation is LiDARsim-style future work). Panels
mode renders one snapshot and cannot apply this.

### 10. Beam geometry — XYZ-LUT conventions

**Model:** azimuth `enc − beam_azimuth` (`rpmath::beamRayAzimuthDeg`),
beam-origin parallax (ray origins on the beam-origin circle; range reported
so `xyz = (r−n)·d̂ + n·[cosθ,sinθ,0]` reconstructs exactly), and
`pixel_shift_by_row = round(−beam_azimuth·W/360)`.

Grounded directly in the vendor implementation rather than papers: the Ouster
SDK XYZ LUT (`ouster_client` `xyzlut.cpp`) and destagger
(`lidar_scan_impl.h`), and the Ouster sensor documentation coordinate-frame
sections. Verified in-tree by `test_raycast.BeamOriginParallaxMatchesXyzLut`.

## Known gaps (deliberately not modeled)

Ordered roughly by expected impact on downstream perception realism.

| Effect | What a full model adds | Reference |
|---|---|---|
| Agent motion during sweep | §9 distorts for EGO motion; other agents' poses stay at the scan-trigger snapshot. Fast crossing traffic also smears in reality (LiDARsim interpolates per-agent poses too). | *Lidar with Velocity*, arXiv:2111.09497; HiMo, arXiv:2503.00803 |
| Beam divergence / footprint | Finite-footprint returns: edge mixing, multi-return, footprint-averaged ranges on oblique/rough surfaces; energy is ~2-D Gaussian over the footprint. HELIOS++ subsamples the beam cone. | Winiwarter et al. 2022 |
| Retroreflector blooming / crosstalk | Very strong returns (signs, plates) saturate detectors and scatter into neighbouring channels — halo points, range bias. This plugin encodes ρ > 1 in the reflectivity byte but produces no artifacts. | *LiDAR Blooming Artifacts Estimation … with Synthetic Data Modeling*, IEEE (10.1109/10774004), 2024 |
| Atmospheric attenuation | `P_r ∝ e^(−2ζR)`; ζ from rain rate / fog visibility via Mie scattering. Hooks cleanly into `signalFromRange` if weather sim is ever needed. | Rasshofer et al., Adv. Radio Sci. 9, 2011; MDPI Sensors 23(15):6891, 2023 |
| Weather scatterers (rain/fog/snow) | Backscatter returns *off the weather itself* (early false hits), not just attenuation. Physics-based augmentation is a mature line of work and a good template. | Hahner et al., *Fog Simulation on Real LiDAR Point Clouds*, ICCV 2021 (arXiv:2108.05249); Kilic et al., *LISA*, arXiv:2107.07004; Hahner et al., *LiDAR Snowfall Simulation*, CVPR 2022 |
| Multi-return / full waveform | Second returns through vegetation, edge splits. | Winiwarter et al. 2022 |
| Incidence angle in panels mode | Depth-image normals (from gradients) could approximate cos(α); currently panels mode applies no incidence factor. | §1 references |
| Retroreflective BRDF | Retroreflectors (ρ > 1) are *angle-insensitive* (corner cubes return along the incident path); §8's diffuse+specular split still attenuates them by cos(α). | Kashani et al. 2015 |
| Sun model for NEAR_IR | Scene illumination, shadows, sky background. | §6 references |
| Range walk | Amplitude-dependent timing bias (strong returns trigger earlier). | Hu et al. 2018 |

## Further reading — the lidar-simulation landscape

Context for where this plugin's approach (analytic geometry + parametric
noise) sits among published alternatives.

**Physics-/geometry-based simulators** (this plugin's family):

- Winiwarter et al. — *HELIOS++*, RSE 269, 2022. Ray tracing + full waveform,
  beam-cone subsampling; the academic reference simulator.
  <https://arxiv.org/abs/2101.09154>
- *Physical LiDAR Simulation in Real-Time Engine*, arXiv:2208.10295 — game-
  engine lidar with physically based intensity, closest in spirit to the
  panels/raycast split here.
- Haider et al. — *Development of High-Fidelity Automotive LiDAR Sensor Model
  with Standardized Interfaces*, Sensors 22(19), 2022 (PMC9572647). Models the
  full receive chain (optics, APD, amplifier, sun noise) and reports
  validation metrics (signal MAPE 1.7%, point count 8.5%, mean intensity
  9.3%) — a useful benchmark for "how good is good" if this model is ever
  validated against a real OS1.
- CARLA (<https://carla.org>) — raycast lidar with linear-in-distance
  intensity attenuation and stochastic raydrop; the de-facto AV research
  baseline, less physical than this plugin's model.

**Data-driven / learned sensor models** (complementary approach: learn the
residual realism a parametric model misses):

- Manivasagam et al. — *LiDARsim*, CVPR 2020. Real-world assets + raycasting +
  a learned raydrop network; demonstrated sim-trained perception transferring
  to real data.
- Guillard et al. — *Learning to Simulate Realistic LiDARs*, arXiv:2209.10986.
  Learns raydrop + intensity from paired camera/lidar data.
- Huang et al. — *Neural LiDAR Fields for Novel View Synthesis*, ICCV 2023.
  Neural-field lidar rendering with beam divergence and two-return modeling.
- Hamdi et al. — *Data-driven Camera and Lidar Simulation Models for
  Autonomous Driving: A Review*, arXiv:2402.10079. Survey covering the
  generative end (R2DM, RangeLDM, LiDM) of the spectrum.

**Why it matters**: studies across this literature consistently find raydrop
(which returns go missing) and intensity fidelity to be the two largest
contributors to the lidar sim-to-real gap — which is why this plugin's
dropout/reflectance modeling (§§1–3) carries most of the realism weight, and
why motion distortion and specular dropout top the gaps table above.
