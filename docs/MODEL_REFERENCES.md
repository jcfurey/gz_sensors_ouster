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
simulation:

- Hahner et al. — *LiDAR Snowfall Simulation for Robust 3D Object Detection*,
  CVPR 2022 (returns culled when attenuated intensity falls below the
  detection threshold). <https://arxiv.org/abs/2203.15118>

This is a heuristic, not a calibrated detection model — a rigorous treatment
would integrate the full detection statistics (see §2 references).

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

### 8. Beam geometry — XYZ-LUT conventions

**Model:** azimuth `enc − beam_azimuth` (`rpmath::beamRayAzimuthDeg`),
beam-origin parallax (ray origins on the beam-origin circle; range reported
so `xyz = (r−n)·d̂ + n·[cosθ,sinθ,0]` reconstructs exactly), and
`pixel_shift_by_row = round(−beam_azimuth·W/360)`.

Grounded directly in the vendor implementation rather than papers: the Ouster
SDK XYZ LUT (`ouster_client` `xyzlut.cpp`) and destagger
(`lidar_scan_impl.h`), and the Ouster sensor documentation coordinate-frame
sections. Verified in-tree by `test_raycast.BeamOriginParallaxMatchesXyzLut`.

## Known gaps (deliberately not modeled)

| Effect | What a full model adds | Reference |
|---|---|---|
| Beam divergence / footprint | Finite-footprint returns: edge mixing, multi-return, footprint-averaged ranges on oblique/rough surfaces. HELIOS++ subsamples the beam cone. | Winiwarter et al. 2022 |
| Atmospheric attenuation | `P_r ∝ e^(−2ζR)`; ζ from rain rate / fog visibility via Mie scattering. Hooks cleanly into `signalFromRange` if weather sim is ever needed. | Rasshofer et al., *Influences of weather phenomena on automotive laser radar systems*, Adv. Radio Sci. 9, 2011; MDPI Sensors 23(15):6891, 2023 |
| Multi-return / full waveform | Second returns through vegetation, edge splits. | Winiwarter et al. 2022 |
| Incidence angle in panels mode | Depth-image normals (from gradients) could approximate cos(α); currently panels mode applies no incidence factor. | §1 references |
| Specular / retroreflective BRDF | cos(α) assumes Lambertian; retroreflectors (ρ > 1) physically return *more* at normal incidence and are angle-insensitive. | Kashani et al. 2015 |
| Sun model for NEAR_IR | Scene illumination, shadows, sky background. | §6 references |
| Range walk | Amplitude-dependent timing bias (strong returns trigger earlier). | Hu et al. 2018 |
