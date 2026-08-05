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

**Model (raycast mode):** `near_ir = albedo · (a + d·max(0, n̂·(−ŝ)))
· kNearIrScale` — the surface's diffuse albedo (`laser_retro`) under a
Lambert sun term, where ŝ is the world's first directional light
(propagation direction, diffuse mean as intensity; weights a = 0.3,
d = 0.7·intensity). No directional light → ambient-only `albedo·1`.
Computed per hit at cast time (the normal is available there) and carried
as a third frame plane into the noise stage. Panels mode keeps the legacy
`ρ·kNearIrScale` analogue. Range-independent in both cases.

Ouster's NEAR_IR channel counts **ambient** near-infrared photons (sunlight
reflected off the scene — "the camera in the lidar"), not laser return. Two
properties follow and are honoured by the model: (a) the value tracks the
surface's NIR albedo, for which `laser_retro` is the available proxy, and
(b) like any camera image of an extended scene it is **radiance-invariant
with range** — hence deliberately no 1/R² here:

- Ouster Sensor Docs, *Sensor Data* (NEAR_IR / ambient).
- Ouster blog — *Lidar as a camera*.
  <https://ouster.com/insights/blog/the-camera-is-in-the-lidar>

The dedicated albedo factor also removes the earlier caveat of the
sensor-incidence cosine leaking into this channel via ρ_app: ambient
Lambertian radiance is view-independent, and the NIR plane now uses the raw
albedo with only the sun-incidence Lambert term. Unmodeled: shadows (no
occlusion ray toward the sun) and sky background on misses.

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

### 11. Participating media — smoke, dust and fog (raycast mode)

**Model:** every beam is integrated through a list of obscurant volumes
(`rc::RcObscurant`, `cuda/raycast_math.hpp`), each homogeneous with
extinction coefficient σ_ext [1/m]. Three effects, all in
`rc::rcApplyObscurants`:

```
τ(R)   = ∫₀ᴿ σ_ext ds                        one-way optical depth
τ_e(R) = ∫₀ᴿ η · σ_ext ds                    what the ACTIVE round trip sees
ρ_app ← ρ_app · exp(−2τ_e)                   two-way extinction of the target
ρ_med  = π · β_π · ΔR · exp(−2τ_e,s)         backscatter from the medium itself
NIR    = NIR_target·e^(−τ) + ω·I·(1 − e^(−τ))  airlight on the ambient channel
```

with β_π = σ_ext/S from the **lidar ratio** S = σ_ext/β_π [sr], ΔR = c·τ_pulse/2
the one-pulse range gate, ω the single-scattering albedo, and η Platt's
multiple-scattering factor (below). The single-return detector reports
whichever of the medium and the hard target carries more received power
ρ/R² — the same strongest-return rule §8 uses for glass and mirrors.

**Forward vs back scattering.** Both are modeled, but they enter in different
places, and it is worth being explicit about which:

- *Backscatter* is the return, to single-scattering order, through the lidar
  ratio. S is the measured extinction-to-backscatter ratio, so it already
  folds in both the albedo and the phase function at 180°:
  β_π = σ_ext·ω·P(π)/4π, i.e. S = 4π/(ω·P(π)).
- *Forward scattering* enters as loss, since σ_ext = σ_abs + σ_sca removes
  light in every direction. Taken literally that is too pessimistic:
  smoke, dust and fog are strongly forward-peaked (asymmetry g ≈ 0.7–0.9), so
  much of what σ_ext removes is deflected by only milliradians and a real
  receiver still collects it. **Platt's multiple-scattering factor η** is the
  standard first-order correction — keep the single-scattering form and
  attenuate by exp(−2·η·τ) instead. η ≈ 1 for optically thin media or a
  narrow field of view (the default, reproducing the pure single-scattering
  limit exactly), η ≈ 0.5–0.8 for dense fog and smoke at typical lidar fields
  of view. Real η varies along the path as multiple scattering builds up; one
  path-averaged value is Platt's own working approximation and is as far as
  this model can go without a beam-cone model.

  η applies to the active laser round trip only. The NEAR_IR composite below
  is a passive wide-field transfer whose airlight term already *is* the
  multiply-scattered light, so applying η there too would correct for the
  same physics twice.

Three properties fall out rather than being coded:

- A visual without `<laser_retro>` gets the configured `base_reflectivity`
  converted from the calibrated Ouster byte scale back to physical reflectance
  *before* incidence, extinction and strongest-return arbitration. Untagged
  hard targets therefore attenuate and compete with smoke exactly like authored
  materials; an explicitly authored zero is not mistaken for omission during
  ray casting.
- Because extinction multiplies the *apparent reflectance*, the whole
  downstream pipeline responds with no knowledge that smoke exists: SIGNAL
  dims by exp(−2τ), the calibrated REFLECTIVITY byte drops, range noise
  widens on the √ρ weighting, and once ρ·exp(−2τ) falls past the √(ρ/0.8)
  detection limit of §3 the return disappears entirely.
- A beam that hits nothing can still return, because the medium is a target.
  That is how a phantom obstacle appears in real smoke.
- NEAR_IR moves the *opposite* way from the laser channels — lit smoke
  scatters ambient light into the receiver, so the ambient image brightens
  while the range image darkens, the signature of fog on a real Ouster.

The scatter depth is sampled by inverse CDF over the truncated two-way
transmittance profile, so dense smoke returns from a speckled shell just
inside its near face and thin smoke returns sparsely from throughout — the
1/R² term is kept in the amplitude but omitted from the sampling density,
which varies slowly across the ≲1/σ penetration depth. Sampling uses an
integer hash of (pixel, per-scan salt) rather than a backend RNG, so the CPU
and every GPU kernel draw identically.

**Sourcing.** Gazebo `<particle_emitter>` elements are mirrored
automatically (`src/obscurants.cpp`), so the smoke a world already shows is
the smoke the LiDAR scans. gz exposes no physical density for an emitter —
rate/lifetime/particle_size are authored for visual appeal — so σ_ext comes
from `<particle_scatter_ratio>`, which is already Gazebo's "how much does
this emitter affect range sensors" knob (gz-rendering applies it to
GpuRays), scaled by `<particle_extinction>`. The volume is the emitter's own
`<size>` region dilated by the mean particle travel distance; the dilation
is isotropic because the message documents no stable emission axis, so it
always *contains* the plume. `<obscurant>` blocks give exact volumes with an
explicit σ_ext (or `<visibility>`, via Koschmieder's σ = 3.912/V).

Parameter values: S ≈ 18–20 sr for fog and water cloud, 40–50 sr for dust,
50–70 sr for biomass-burning smoke; ω ≈ 0.8–0.9 in the near IR for weakly
absorbing smoke and dust; η ≈ 1 thin, 0.5–0.8 dense.

Two coupling caveats worth knowing when tuning:

- **ω and S are not independent.** S = 4π/(ω·P(π)) ties them to the phase
  function, but nothing enforces it — the laser path reads S alone and only
  the ambient channel reads ω, so an inconsistent pair describes no real
  aerosol even though it simulates fine.
- **Scatter-depth sampling is extinction-weighted**, while the true return
  profile is backscatter-weighted. These are the same thing for one medium;
  where volumes with *different* lidar ratios overlap on one ray, which
  volume a return is drawn from is mildly biased. The amplitude reported for
  wherever it lands is exact either way.

- Rasshofer et al. — *Influences of weather phenomena on automotive laser
  radar systems*, Adv. Radio Sci. 9, 2011. The extinction + backscatter
  decomposition this implements.
- Platt — *Lidar and radiometric observations of cirrus clouds*, J. Atmos.
  Sci. 30, 1973, and *Remote sounding of high clouds III*, J. Appl.
  Meteorol. 20, 1981. Source of the multiple-scattering factor η and of the
  practice of treating it as a single path-averaged number.
- Hahner et al. — *Fog Simulation on Real LiDAR Point Clouds for 3D Object
  Detection*, ICCV 2021, arXiv:2108.05249; Kilic et al. — *LISA: Lidar Light
  Scattering Augmentation*, arXiv:2107.07004. The same physics applied as
  point-cloud augmentation; useful cross-checks on the artifact shapes.
- Müller et al. — *Aerosol-type-dependent lidar ratios observed with Raman
  lidar*, JGR 112 D16202, 2007; Ackermann — *The extinction-to-backscatter
  ratio of tropospheric aerosol*, J. Atmos. Ocean. Technol. 15, 1998. Source
  of the lidar-ratio values above.
- Koschmieder 1924, for the airlight composite and the visibility ↔
  extinction conversion.

Panels mode has no equivalent path (it only ever sees a rendered depth
image), and the plugin warns if obscurants are configured there. Verified
in-tree by `test_obscurants` (closed-form Beer–Lambert, exact inversion of
the optical-depth profile, fallback-material extinction/arbitration, and the
sampled amplitude against the lidar equation) and `test_obscurant_config`;
`examples/worlds/ouster_smoke.sdf` demonstrates it with an untagged-wall
density ladder whose first three rungs preserve the wall and last two report
the medium.

## Known gaps (deliberately not modeled)

Ordered roughly by expected impact on downstream perception realism.

| Effect | What a full model adds | Reference |
|---|---|---|
| Agent motion during sweep | §9 distorts for EGO motion; other agents' poses stay at the scan-trigger snapshot. Fast crossing traffic also smears in reality (LiDARsim interpolates per-agent poses too). | *Lidar with Velocity*, arXiv:2111.09497; HiMo, arXiv:2503.00803 |
| Beam divergence / footprint | Finite-footprint returns: edge mixing, multi-return, footprint-averaged ranges on oblique/rough surfaces; energy is ~2-D Gaussian over the footprint. HELIOS++ subsamples the beam cone. | Winiwarter et al. 2022 |
| Retroreflector blooming / crosstalk | Very strong returns (signs, plates) saturate detectors and scatter into neighbouring channels — halo points, range bias. This plugin encodes ρ > 1 in the reflectivity byte but produces no artifacts. | *LiDAR Blooming Artifacts Estimation … with Synthetic Data Modeling*, IEEE (10.1109/10774004), 2024 |
| Discrete precipitation (rain / snow) | §11 models a *continuous* medium, which fits smoke, dust and fog. Rain and snow are sparse discrete scatterers: individual drops or flakes crossing single beams give isolated near returns and per-beam flicker rather than a smooth transmittance profile, and the drop-size distribution ties σ_ext to rain rate. | Hahner et al., *LiDAR Snowfall Simulation*, CVPR 2022; Kilic et al., *LISA*, arXiv:2107.07004 |
| Multiple scattering — beyond the transmittance | §11's η corrects the *magnitude* of the loss, but multiply-scattered photons also arrive late and off-axis: real dense-medium returns are pulse-stretched and range-biased long, and the beam broadens with depth. Both need a beam-cone model, which is its own gap above. | Platt 1973/1981; Bissonnette, Appl. Opt. 35, 1996 |
| Density structure inside a plume | Obscurant volumes are homogeneous with a hard boundary; real plumes have soft, turbulent, time-varying density, so simulated cloud edges are crisper than real ones. | §11 references |
| Multi-return / full waveform | Second returns through vegetation, edge splits. | Winiwarter et al. 2022 |
| Incidence angle in panels mode | Depth-image normals (from gradients) could approximate cos(α); currently panels mode applies no incidence factor. | §1 references |
| Retroreflective BRDF | Retroreflectors (ρ > 1) are *angle-insensitive* (corner cubes return along the incident path); §8's diffuse+specular split still attenuates them by cos(α). | Kashani et al. 2015 |
| NEAR_IR shadows / sky | §6 models the sun's Lambert term but casts no shadow ray (surfaces in shadow still read lit) and misses report 0 instead of sky background. | §6 references |
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
