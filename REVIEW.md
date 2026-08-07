# gz_sensors_ouster — code review of `cam_wip` @ 5a372a0

Ten parallel reviewers produced 71 candidate findings. Each was then handed to an
adversarial verifier instructed to refute it and to correct inflated severity.
**12 were refuted, 59 survived.** Severities below are the verifier-corrected ones.

Baseline on this commit: the build is clean (no compiler warnings from repo code)
and `colcon test` reports 398 passed, 0 failed, 2 skipped.

## HIGH (8)

### `.github/workflows/ci.yaml:164` — cppcheck CI gate is red on HEAD: 6 diagnostics, --error-exitcode=1

*Build & CI* · build

The `cppcheck` job runs with `--error-exitcode=1` over `src/ include/ cuda/`. I ran that exact command line on Ubuntu 24.04 with cppcheck 2.13.0-2ubuntu3 — the identical package `ubuntu-latest` installs at ci.yaml:154 — and it exits 1 with six diagnostics, none of which carry an `--inline-suppr` comment:

  src/obscurants.hpp:102 style useStlAlgorithm  (raw loop over `volumes`, suggests std::any_of)
  src/ouster_lidar_profile.cpp:14 performance passedByValue  (`std::string canonical(std::string value)`)
  src/packet_encoder.cpp:167 style variableScope  (`local_epoch`)
  src/packet_encoder.cpp:169 style variableScope  (`local_generation`)
  src/packet_encoder.cpp:248 style knownConditionTrueFalse  (`paused_` always false)
  src/packet_encoder.cpp:249 style knownConditionTrueFalse  (`local_epoch != simulation_epoch_` always false)

The two `packet_encoder.cpp` hits are cppcheck false positives — it cannot see that `paused_` and `simulation_epoch_` are mutated by another thread under `drain_mtx_` (the wait predicates at lines 232-236 and 245-250 are re-evaluated after the mutex is released and reacquired). But cppcheck does not know that, and `--error-exitcode=1` does not care. Two of the three offending files (`src/obscurants.hpp`, `src/ouster_lidar_profile.cpp`) are among the new files added since AUDIT.md, which is how this got in: nothing pushed to `cam_wip` ever ran the gate (see the branch-filter finding).

**Failure scenario.** Open any pull request from the current branch. The `cppcheck (static analysis)` required status check fails immediately at ci.yaml:164 with exit 1 and six style/performance messages, before a single line of the PR's own diff is considered. Every PR is red until these six are fixed or suppressed.

**Fix.** Fix the two real ones (make `canonical` take `const std::string &` at src/ouster_lidar_profile.cpp:14; hoist `local_epoch`/`local_generation` into the `while` body at src/packet_encoder.cpp:167,169; use std::any_of at src/obscurants.hpp:102) and add `// cppcheck-suppress knownConditionTrueFalse` above src/packet_encoder.cpp:248 and :249 with a comment naming the cross-thread mutation — `--inline-suppr` is already enabled at ci.yaml:166 precisely for this. Then pin the cppcheck version in CI (e.g. a container or an explicit apt version) so a runner-image bump cannot silently turn the gate red again.

**Verifier.** Reproduced verbatim. I ran the exact ci.yaml:164-171 command line with cppcheck 2.13.0 (the version the runner installs) and got EXIT=1 with exactly the six listed diagnostics at exactly the cited lines: src/obscurants.hpp:102 useStlAlgorithm (confirmed raw 'for (const auto & v : volumes)' inside active()), src/ouster_lidar_profile.cpp:14 passedByValue (confirmed 'std::string canonical(std::string value)'), src/packet_encoder.cpp:167/:169 variableScope (local_epoch/local_generation declared in drainThreadFunc's preamble, used only inside the inner loop), and :248/:249 knownConditionTrueFalse. --inline-suppr is already passed at ci.yaml:166 and none of the six are suppressed. The reviewer's read of the packet_encoder pair as false positives is also right: paused_ and simulation_epoch_ are read under drain_mtx_ in the wait predicates at packet_encoder.cpp:231-236 and 245-250 and mutated by the producer thread, which cppcheck cannot model. Severity stays high: this is a currently-broken build-ci artifact that exits 1 on every PR regardless of the diff.

---

### `.github/workflows/ci.yaml:191` — cpplint CI gate is red on HEAD: 3 errors in test/

*Build & CI* · build

`cpplint --recursive src/ include/ cuda/ test/` with the repo's CPPLINT.cfg exits 1 on the current tree. I ran it with cpplint 1.6.1 — the exact version pinned at ci.yaml:184 — and got `Total errors found: 3`:

  test/test_packet_pacing.cpp:14      build/namespaces  `using namespace std::chrono_literals;`
  test/test_sim_time_scheduler.cpp:14 build/namespaces  `using namespace std::chrono_literals;`
  test/test_metadata_parsing.cpp:104  whitespace/blank_line  redundant blank line before the closing `}`

Note that `build/namespaces` is NOT in the filter list in CPPLINT.cfg:31, so this is a category the project deliberately keeps enabled. Both offending files were added in the commits after AUDIT.md. cpplint does process `.cu`/`.cuh` (I verified all 18 files under cuda/ including ray_processor_cuda.cu are linted), so coverage is fine — the gate simply never ran on the branch these files landed on.

**Failure scenario.** Any PR from this branch fails the `cpplint (Google C++ style)` check at ci.yaml:191 with exit 1 and three errors, independent of the PR's contents.

**Fix.** Replace the two `using namespace std::chrono_literals;` with using-declarations (`using std::chrono_literals::operator""ms;` etc.) or add `// NOLINT(build/namespaces)`, and delete the blank line at test/test_metadata_parsing.cpp:104. Longer term, wire cpplint+cppcheck into a pre-commit hook or the `.claude/hooks/session-start.sh` provisioning (it already installs both, at lines 77 and 94) so they run before the code is pushed rather than after.

**Verifier.** Reproduced verbatim. 'cpplint --recursive src/ include/ cuda/ test/' with cpplint 1.6.1 (the version pinned at ci.yaml:184) prints 'Total errors found: 3' and exits 1: test/test_packet_pacing.cpp:14 and test/test_sim_time_scheduler.cpp:14 build/namespaces, test/test_metadata_parsing.cpp:104 whitespace/blank_line. I confirmed build/namespaces is not silenced — the filter list is at CPPLINT.cfg:25 (the finding says :31, a harmless line-number slip) and contains only whitespace/braces, whitespace/indent, whitespace/comments, whitespace/line_length, build/include_order, build/include_subdir, runtime/references, runtime/int, runtime/indentation_namespace, build/c++11. Same standing as [0]: a confirmed red gate on HEAD.

---

### `src/gz_gpu_ouster_lidar_system.cpp:906` — mirror_->postUpdate() is the one PostUpdate stage outside the guarded() exception firewall

*Lifecycle & threading* · correctness

PostUpdate builds an explicit exception firewall at lines 761-777 with a comment that spells out the stakes: "gz-sim does not wrap ISystemPostUpdate in a try/catch, so an escaping exception aborts the whole server." Every stage is then routed through it — `guarded("metadata", ...)` (780), `guarded("scan", ...)` (931, 945), `guarded("imu", ...)` (954). Every stage except the biggest one. Line 906 calls `mirror_->postUpdate(info, ecm, sensor_pose, sim_epoch_, process_params)` bare. That call is the entry point to `RaycastMirror::rebuildScene()`, which is by far the most throw-prone code in the whole per-tick path: it loads meshes off disk (`MeshManager::Instance()->Load`, raycast_mirror.cpp:235), loads RGBA textures (`gz::common::Image::Load`, :310), does unbounded `std::vector` growth for verts/tris/texcoords (:75-92) and `scene->addMesh` (:253). And it contains a guaranteed-throwing API call: `responseMapPath()` at raycast_mirror.cpp:116 uses `std::filesystem::is_regular_file(path)` — the overload that throws `std::filesystem_error` rather than the `std::error_code&` overload that does not. The irony is that the raycast *worker* thread does have this firewall (raycast_mirror.cpp:697-709) with a comment explaining that an escape there "would call std::terminate and kill the whole Gazebo server" — so the author understood the hazard exactly and then left the sim-thread half of the same class unprotected.

**Failure scenario.** A world contains a visual whose PBR albedo map resolves to a path the sim process cannot stat — the file lives on an NFS/autofs mount that has gone away, under a directory with the execute bit cleared, or behind a symlink loop (ELOOP). `responseMapPath()` reaches raycast_mirror.cpp:116, `std::filesystem::is_regular_file` throws `filesystem_error`, the exception unwinds out of `rebuildScene` -> `RaycastMirror::postUpdate` -> `GzGpuOusterLidarSystem::PostUpdate` -> gz-sim's `SimulationRunner` PostUpdate worker thread, which has no handler, so std::terminate aborts the entire Gazebo server. Every other sensor, the physics engine and the GUI die because one texture directory was unreadable. Same outcome for a std::bad_alloc from a large mesh on line 253.

**Fix.** Wrap the call in the existing firewall: `guarded("raycast mirror", [&] { mirror_->postUpdate(info, ecm, sensor_pose, sim_epoch_, process_params); });`. Independently, switch raycast_mirror.cpp:116 to the non-throwing overload: `std::error_code ec; return std::filesystem::is_regular_file(path, ec) ? path.string() : std::string{};`.

**Verifier.** Confirmed by reading the code. gz_gpu_ouster_lidar_system.cpp:761-777 defines `guarded`; it wraps metadata (780), scan (931, 945) and imu (954). Line 906 `mirror_->postUpdate(info, ecm, sensor_pose, sim_epoch_, process_params);` is bare — the lambda is already in scope, so this is an omission, not a scoping constraint. RaycastMirror::postUpdate (raycast_mirror.cpp:419) has no try/catch of its own and calls rebuildScene (492), which does throwing work: unbounded vector growth in appendGzMesh (raycast_mirror.cpp:75-91), scene->addMesh (253), gz::common::Image::Load (311), and critically responseMapPath at raycast_mirror.cpp:116, which uses `std::filesystem::is_regular_file(path)` — the overload specified to throw std::filesystem_error when the underlying status() reports an error (EACCES on a parent dir, ELOOP, ESTALE; ENOENT is not an error so a merely-missing file is safe). Reached whenever a visual has a PBR albedo map (108-111), i.e. any textured world. The worker thread half of the same class does carry a firewall (raycast_mirror.cpp:697-709) with a comment stating an escape 'would call std::terminate and kill the whole Gazebo server', so the asymmetry is real. Not covered by AUDIT.md (which predates raycast_mirror.cpp entirely). Kept at high rather than critical: the trigger needs an unusual filesystem or OOM condition, but the blast radius is the whole server and the fix is one line.

---

### `cuda/ray_processor_math.hpp:303` — detectionProbability saturates at rho=0.01, so extinction stops affecting dropout past tau_eff~2

*Obscurants & noise* · correctness

`calibratedRangeAtReflectivity` floors the apparent reflectance before the power law:

```cpp
const float rho = gzm::fmax_(retro_val, 0.01f);   // <-- line 303
const float exponent = gzm::log_(range_80 / range_10) / gzm::log_(8.0f);
return range_10 * gzm::exp_(exponent * gzm::log_(rho / 0.1f));
```

The floor exists to keep `log_` finite, but it collapses six-plus decades of reflectance onto a single value. Every target dimmer than 1% apparent reflectance is treated as a 1% target, so `detectionProbability` -- and therefore `dropoutProbability` -- stops responding to extinction the moment `rho * exp(-2*eta*tau)` drops below 0.01.

This is exactly the mechanism docs/MODEL_REFERENCES.md §11 promises does the work: "calibrated detection probability from §3 falls smoothly as the attenuated return approaches the product's D90/D50 envelope" and "§11 ... makes targets disappear entirely once the smoke is thick enough" (`cuda/raycast_math.hpp:1258-1259`). It does not. The obscurant model's headline downstream consequence is capped.

**Failure scenario.** Measured with the OS1 profile (range_10_d90=45, range_80_d90=100, range_10_d50=55, range_80_d50=120, max_range=120), target at 20 m:

  rho=5.0e-02  P(keep)=0.9818
  rho=1.0e-02  P(keep)=0.8075
  rho=1.0e-03  P(keep)=0.8075
  rho=1.0e-06  P(keep)=0.8075
  rho=1.0e-20  P(keep)=0.8075

A 0.8-reflectance wall at 20 m behind sigma=0.5 /m smoke over 10 m (tau=5, two-way transmittance 4.5e-5 -- optically invisible in reality) is still reported on 81% of beams, and no amount of extra smoke changes that number. The dropout ceiling from the calibrated curve at 20 m is 19% regardless of how dark the return is.

**Fix.** Do not floor rho for the detection curve. Either extrapolate the power law down (rho can be arbitrarily small; `log_(rho/0.1f)` is finite for any rho > 0, so only rho <= 0 needs a branch, and that branch should return d90 = 0 i.e. "never detected"), or add an explicit SNR floor: below some apparent reflectance (e.g. the reflectance whose returned signal equals the detector noise floor) return detection probability 0 rather than reusing the 1% anchor. Add a test that P(keep) is strictly decreasing across rho = 1e-2, 1e-3, 1e-4 -- the current suite (test_obscurants.cpp:812 TargetsPastTheDetectionLimitAreDroppedForFree) only exercises a 60 m target where range, not reflectance, does the dropping.

**Verifier.** Confirmed. ray_processor_math.hpp:303 is exactly `const float rho = gzm::fmax_(retro_val, 0.01f);`, so d90/d50 are constant for every apparent reflectance at or below 1%, and detectionProbability (312-332) / dropoutProbability (336-350) inherit that. I reproduced the reviewer's numbers by hand with the real OS1 profile from src/ouster_lidar_profile.cpp:118-121 (45/100/55/120) and max_range 120 (gz_gpu_ouster_lidar_system.hpp:91): exponent = ln(100/45)/ln8 = 0.384, d90(0.01) = 18.6 m, d50(0.01) = 23.2 m, slope 0.480, P(det|20 m) = 0.821, times random_keep 0.984 = 0.808 — matching their 0.8075 and invariant for all rho below 0.01. Worse at short range: a wall at 5 m behind arbitrarily opaque smoke gives P(det) ~= 0.9998. rho = 0.8*exp(-2*tau_eff) crosses 0.01 at tau_eff = 2.19, which examples/worlds/ouster_smoke.sdf's densest rung already sits on (sigma 1.0/m, tau 1.8, wall rho 0.0137), so this is in the shipped operating range, not an extreme. It also directly falsifies docs/MODEL_REFERENCES.md:362 ("falls smoothly as the attenuated return approaches the product's D90/D50 envelope") and raycast_math.hpp:1258-1259. Their read of the existing test is right too: test_obscurants.cpp:812 only passes because its 60 m wall is far past d50, where range does the dropping. Not in AUDIT.md.

---

### `cuda/ray_processor_math.hpp:326` — d50 clamped to max_range but d90 is not, giving every retroreflector a flat 10% dropout at all ranges

*Obscurants & noise* · correctness

`detectionProbability` repairs a d50 <= d90 crossover and then immediately re-creates it:

```cpp
float d50 = calibratedRangeAtReflectivity(retro_val, range_10_d50, range_80_d50);
if (d50 <= d90) d50 = d90 * (1.0f + gzm::fmax_(rolloff, 0.01f));   // 325: fixes it
d50 = gzm::fmin_(d50, max_range);                                  // 326: breaks it again
if (d50 <= d90) return d <= d90 ? 0.9f : 0.0f;                     // 327: degenerate branch
```

d90 is never clamped. As soon as the apparent reflectance is high enough that d90 >= max_range, line 326 forces d50 <= d90 and line 327 replaces the whole logistic rolloff with a constant 0.9 -- a hard-coded 10% dropout at *every* range, including point blank. It is also discontinuous: crossing that reflectance threshold drops detection from ~1.0 to exactly 0.9 everywhere.

This hits precisely the retroreflective band the reflectivity model was built for (`reflectivityToByte`, rv > 1 -> bytes 101-255), and the shipped example worlds are inside it: `examples/worlds/*.sdf` author `<laser_retro>8.0</laser_retro>` (x3) and `<laser_retro>12.0</laser_retro>`.

**Failure scenario.** Measured directly (rpmath::detectionProbability):

  OS1, max_range=120:
    rho=1.20  d90=116.85 d50=139.72 -> P(det) @5m=1.0000 @20m=1.0000 @60m=1.0000
    rho=1.30  d90=120.49 d50=143.98 -> P(det) @5m=0.9000 @20m=0.9000 @60m=0.9000
    rho=8.00 / 12.00 (the showcase signs) -> 0.9000 at every range
  OS2, max_range=240: the cliff starts at rho=1.2.

A retroreflective sign at 5 m loses 10% of its returns for no physical reason, while an otherwise identical rho=1.2 surface at the same range loses none. Any perception consumer measuring return density on retro targets sees a spurious 10% hole rate that is independent of range.

**Fix.** Clamp consistently, or not at all. Simplest correct form: drop line 326 entirely (the logistic already returns ~0 for d >= max_range, and the caller rejects d >= max_range before this is reached at ray_processor_math.hpp:318). If a clamp is wanted for numerical reasons, clamp d90 first and re-apply the rolloff afterwards: `d90 = fmin_(d90, max_range); d50 = fmin_(d50, max_range); if (d50 <= d90) d50 = d90 * (1 + rolloff);`. Add a regression test that P(det) is monotone non-decreasing in retro_val at a fixed range across rho = 1.0, 1.3, 5.0, 20.0.

**Verifier.** Confirmed line-for-line: 325 repairs the crossover (d50 = d90*(1+rolloff)), 326 `d50 = gzm::fmin_(d50, max_range);` re-creates it whenever d90 >= max_range, and 327 then returns a constant 0.9 for every d — and d < max_range is already guaranteed by the guard at 318, so the `: 0.0f` arm is dead and 0.9 is the only outcome. d90 is never clamped anywhere in the function. Threshold checks out: d90 = 45*(rho/0.1)^0.384 reaches 120 at rho = 1.286, so rho=1.2 gives 116.9 (full logistic, P~1.0) and rho=1.3 gives 120.5 (flat 0.9) — a genuine discontinuity and non-monotonicity in retro. Unlike finding [0], this fires in shipped content today: examples/worlds/ouster_demo.sdf:143 and turtlebot3_ouster_headless.sdf:153 author laser_retro 8.0, ouster_showcase.sdf:772 authors 12.0, and showcase lines 316-528 go up to 300.0 (d90 ~= 973 m). Every one of those surfaces loses 10% of its returns at point-blank range for no physical reason. Not in AUDIT.md. Keeping HIGH: it is reachable in the default example worlds, not latent.

---

### `config/metadata/os0_128_rev7.json:263` — All 10 shipped metadata JSONs put lidar_origin_to_beam_origin_mm in the wrong matrix cell and the wrong units

*Ouster protocol* · correctness

Every `config/metadata/*.json` declares the beam origin twice and the two disagree by a factor of 1000 and by a matrix slot:
```
"lidar_origin_to_beam_origin_mm": 27.67,        // line 262 — correct, millimetres
"beam_to_lidar_transform": [                    // line 263
  1,0,0, 0,          <- (0,3) should be 27.67
  0,1,0, 0,
  0,0,1, 0.02767,    <- (2,3) holds the value, in METRES
  0,0,0, 1 ]
```
The array is row-major (`mat4d_to_array` doc, `typedefs.h:156`), confirmed by the sibling `lidar_to_sensor_transform` which correctly carries 36.18 **mm** at index 11. Real Ouster metadata puts the beam origin at index 3 in mm. The SDK only synthesises `beam_to_lidar_transform(0,3) = lidar_origin_to_beam_origin_mm` when the key is *absent* (`metadata.cpp:849-851`); here it is present, so the bogus value wins. `convert_legacy_to_nonlegacy` (`metadata.cpp:924-927`) faithfully forwards it from the flat layout these files use.

Consequence in `xyzlut.cpp:25-30, 72-81`: `beam_to_lidar_euclidean_distance_mm` becomes 0.02767 instead of 27.67, so `lut.offset` is ~2.8e-5 m instead of 2.767e-2 m — effectively zero. Meanwhile the plugin deliberately reports the range that inverts the *correct* LUT: raycast mode originates each ray at `n·[cos enc, sin enc, 0]` and reports `r = t0 + n_off` (`cuda/raycast_math.hpp:1387, 1410-1411`), and panels mode applies `rpmath::applyBeamOrigin` = `depth − n·cos(el) + n` (`cuda/ray_processor_math.hpp:264-275`), with `n` taken from the *correct* `lidar_origin_to_beam_origin_mm` (`src/panel_rig.cpp:43`). `docs/MODEL_REFERENCES.md:293-295` states this reconstructs the hit point "exactly".

It does not. `os_cloud` reconstructs `r·d̂` rather than `(r−n)d̂ + n·ê`, an error of `n·|ê − d̂| = 2n·sin(el/2)`: 21.2 mm for OS0-128 at ±45°, 22.3 mm for OSDome at ±90°, 0 at the horizon. That is a systematic, elevation-banded radial bulge larger than the OS0's own 10 mm accuracy spec, and it is invisible at el = 0 — which is the only elevation the parallax test exercises (see the separate test finding).

**Failure scenario.** Run any shipped world with `os_cloud` and place a flat wall at 10 m. Points from the horizon beams land at 10.000 m; points from the ±45° edge beams of an OS0-128 land ~21 mm off along the beam. Diffing the simulated cloud against Gazebo ground-truth geometry shows a smooth elevation-dependent bias that no noise parameter can remove, and any sim-vs-hardware calibration comparison is silently poisoned.

**Fix.** In all ten `config/metadata/*.json`, move the beam origin to the correct cell in millimetres, e.g. for OS0: `"beam_to_lidar_transform": [1,0,0,27.67, 0,1,0,0, 0,0,1,0, 0,0,0,1]` (15.806 for OS1/OSDome, 13.762 for OS2). Add an assertion to `test_metadata_parsing.cpp` that `info.beam_to_lidar_transform(0,3) == info.lidar_origin_to_beam_origin_mm` and `(2,3) == 0` for every file.

**Verifier.** Confirmed. os0_128_rev7.json:262-278 has lidar_origin_to_beam_origin_mm=27.67 but beam_to_lidar_transform with 0 at index 3 and 0.02767 at index 11; the same shape is in all ten files (grep of lidar_origin_to_beam_origin_mm: 27.67 / 15.806 / 13.762). Row-major is confirmed by typedefs.h:148 (`mat4d_from_array` assumed row-major) via JsonTools::decode_transform_array (json_tools.cpp:99-109), and by the sibling lidar_to_sensor_transform in the same file (os0_128_rev7.json:299-316) which correctly carries 36.18 mm at index 11 as a Z translation. metadata.cpp:840-853 only synthesises (0,3) when the key is ABSENT, so the present-but-wrong value wins. Consequence in xyzlut.cpp:25-29: (2,3)!=0 so beam_to_lidar_euclidean_distance_mm = sqrt(0 + 0.02767^2) = 0.02767 mm, and lut.offset (lines 74-82) collapses to ~2.8e-5 m. Meanwhile the plugin reports the range that inverts the CORRECT lut: raycast originates at n*[cos enc, sin enc, 0] and reports t0+n_off (raycast_math.hpp:1409-1411) and panels apply depth - n*cos(el) + n (ray_processor_math.hpp:271-272) with n from the correct lidar_origin_to_beam_origin_mm (panel_rig.cpp:43). metadata_str is published verbatim - ouster_metadata.cpp:201 only re-serialises when firmware < 3.2.0 and the profile carries WINDOW, which is false for both the modern (v3.2.0) and LEGACY files. Elevation-banded radial error 2n*sin(el/2) = 21.2 mm at OS0's +/-45 deg is real.

---

### `src/packet_encoder.cpp:127` — Low-bandwidth UDP profiles are accepted by the loader but throw in every encode, so zero packets ever ship

*Ouster protocol* · api-misuse · reported as critical, downgraded by verifier

`encodeScan` unconditionally writes four channel blocks:
```
pw.set_block<uint32_t>(range_mat.data(),  W, ChanField::RANGE,        pkt_buf_.data());
pw.set_block<uint16_t>(signal_mat.data(), W, ChanField::SIGNAL,       pkt_buf_.data());   // line 127
pw.set_block<uint8_t> (refl_mat.data(),   W, ChanField::REFLECTIVITY, pkt_buf_.data());
pw.set_block<uint16_t>(nearir_mat.data(), W, ChanField::NEAR_IR,      pkt_buf_.data());
```
`PacketWriter::set_block` does `impl::FieldInfo f_info = impl_->fields.at(field_name);` (parsing.cpp:1108) on a `std::map<std::string, FieldInfo>` (parsing.cpp:419). The low-bandwidth field tables `LB_FIELD_INFO` (parsing.cpp:212-218), `DUAL_LB_FIELD_INFO` (parsing.cpp:333-343), `LB_WINDOW_FIELD_INFO` (parsing.cpp:221-227) and `ZM_LB_FIELD_INFO` (parsing.cpp:311-319) contain NO `SIGNAL` entry — so `.at("SIGNAL")` throws `std::out_of_range`. `LB_WINDOW_FIELD_INFO` additionally has no `NEAR_IR`, so line 129 would throw too.

This is not a hypothetical profile the plugin refuses. `src/ouster_metadata.cpp:112-115` explicitly enumerates `RNG15_RFL8_NIR8`, `RNG15_RFL8_NIR8_DUAL` and `FUSA_RNG15_RFL8_NIR8_DUAL` as `low_data_profile`, passes that flag into the profile resolver, and `src/ouster_lidar_profile.cpp:436` sets `range_resolution_m = 0.008` for them. `load()` returns true, the profile line is logged as `resolution=8.0mm`, metadata is published — and then the sensor is dead.

The throw is swallowed by the `guarded("scan", ...)` wrapper at `src/gz_gpu_ouster_lidar_system.cpp:931`/`945`, so there is no crash and no clue: just `"<sensor>: scan stage failed: map::at"` throttled to once per 5 s, forever. `test_metadata_parsing.cpp` only instantiates the ten shipped RNG19/LEGACY files, so nothing in the suite touches this path.

**Failure scenario.** Point `<metadata_file>` at any metadata whose `udp_profile_lidar` is `RNG15_RFL8_NIR8` (or `_DUAL` / `FUSA_...` / `RNG15_RFL8_WIN8`). The plugin configures cleanly and logs the resolved Ouster profile at 8 mm resolution. `encodeScan` then throws `std::out_of_range` on the first and every subsequent scan at line 127, before the drain swap at line 149, so `drain_pkts_` is never populated. `/<sensor>/lidar_packets` publishes nothing for the life of the sim; `os_cloud` sits on a valid metadata string waiting for packets that never arrive, and the only diagnostic is a 5-s-throttled "scan stage failed: map::at".

**Fix.** Resolve the field set once in `PacketEncoder::start()` from `meta_->pw` (e.g. probe with a `has_field` helper or cache which of RANGE/SIGNAL/REFLECTIVITY/NEAR_IR the active profile carries) and skip the missing blocks in the loop. Alternatively, reject unsupported profiles up front in `OusterMetadata::load()` — it already inspects `udp_profile_lidar` at line 111 — with an explicit error naming the profile, instead of admitting it and failing silently per-scan.

**Verifier.** Confirmed by reading all three layers. packet_encoder.cpp:126-129 unconditionally calls set_block for RANGE/SIGNAL/REFLECTIVITY/NEAR_IR; PacketWriter::set_block does `impl_->fields.at(field_name)` (parsing.cpp:1109) on a std::map (parsing.cpp:419), and LB_FIELD_INFO (parsing.cpp:212-218), LB_WINDOW_FIELD_INFO (221-227), ZM_LB_FIELD_INFO (311-319) and DUAL_LB_FIELD_INFO (333-343) contain no SIGNAL key -> std::out_of_range. The plugin genuinely admits these profiles: ouster_metadata.cpp:112-115 enumerates RNG15_RFL8_NIR8 / _DUAL / FUSA_ as low_data_profile, feeds it to the resolver, ouster_lidar_profile.cpp:436 sets range_resolution_m=0.008, and load() returns true with no profile rejection anywhere. The throw is swallowed by `guarded("scan", ...)` (gz_gpu_ouster_lidar_system.cpp:761-778, call sites 931/945), and it fires before the drain swap at packet_encoder.cpp:149, so drain_pkts_ is never populated and publishImages (publishChannels line 1065) is skipped too. Not critical only because none of the 10 shipped config/metadata/*.json use a low-bandwidth profile (all are RNG19_RFL8_SIG16_NIR16 or LEGACY) - it needs user-supplied metadata.

---

### `cuda/raycast_math.hpp:1475` — Glass continuation ray self-hits the pane's own back face; nothing behind solid transparent geometry is ever visible

*Raycast math* · correctness · reported as critical, downgraded by verifier

`rcCastOneRay` starts the behind-glass continuation at `seg_start = t0 + kRcSegEps` (1 mm) and re-casts against the *whole* scene with no exclusion of the instance it just hit (raycast_math.hpp:1474-1497). For any transparent geometry thicker than 1 mm the continuation origin is *inside* that instance, and `rcHitBox` is explicitly written to return the exit point in that case — line 350: `const float t = (lo > tmin) ? lo : hi;  // Entry point if outside the box, exit point if inside`. So `inst1 == inst0` and the one permitted continuation segment is consumed by the pane's own rear face.

Every shipped transparent visual is a solid box: `examples/worlds/turtlebot3_ouster_headless.sdf:179` (`glass_pane`, box 2 x 0.02 x 1.5, transparency 0.9) and `examples/worlds/ouster_showcase.sdf:599,618` (`F_pane_60`/`F_pane_92`, box 0.08 x 2.4 x 2.4).

Reproduced with the headless-world numbers (sensor at origin, pane at y=3.5, `box_behind_glass` at y=4.5): reported range 3.5100 m, retro 1.2150 — the pane's back face at +2 cm, never the box at 4.30 m. The 1.2150 is exactly `(fallback 0.5·cos + ks 1.0·lobe 1.0) · tau^2 = 1.5 · 0.81`. Feeding 1.2150 into `rpmath::reflectivityToByte` (ray_processor_math.hpp:405) yields byte 106, i.e. an ordinary window is emitted in the band the code reserves for retro-tape.

`TEST(Raycast, GlassTransmissionReportsStrongestReturn)` (test/test_raycast.cpp:410) cannot catch this: it builds the pane as a zero-thickness `kPlane` (test_raycast.cpp:426), the one geometry for which the continuation origin is not inside the pane. Swapping that plane for the 8 cm box the showcase world actually authors flips the result from 4.85 m / 0.7618 to 2.0400 m / 0.9310.

docs/MODEL_REFERENCES.md:220 ("the object behind the glass at weakened intensity otherwise") and the world comments at ouster_showcase.sdf:592 ("tau 0.92 - nearly clear; the wall behind dominates") describe behaviour the code cannot produce for any beam at any incidence angle.

**Failure scenario.** Load examples/worlds/turtlebot3_ouster_headless.sdf in raycast mode. Every beam that hits glass_pane (box 2 x 0.02 x 1.5, transparency 0.9, specular 1.0, at y=3.5) reports range 3.510 m with apparent reflectance 1.215 -> REFLECTIVITY byte 106. box_behind_glass (laser_retro 0.9, at y=4.5, true range 4.30 m) never appears in the cloud at any beam or column. The published point is a phantom surface 2 cm behind the pane, classified as retroreflective.

**Fix.** Pass the front instance index into the continuation cast and skip it in `rcTestInstance` (an `ignore_inst` parameter threaded through `rcNearestHit`), or advance the continuation origin past the front instance's exit: re-intersect `instances[inst0]` from `t0` in its own local frame, take the exit parameter `t_exit`, and start the segment at `t0 + t_exit + kRcSegEps`. The latter also gives physically correct in-pane path length. Then change `TEST(Raycast, GlassTransmissionReportsStrongestReturn)` to use a solid `kBox` pane (matching the shipped worlds) instead of the zero-thickness `kPlane` at test_raycast.cpp:426, otherwise the regression is invisible to CI.

**Verifier.** Confirmed by reading the code, not just the cited line. raycast_math.hpp:1475 sets seg_start = t0 + kRcSegEps with kRcSegEps = 1.0e-3f (raycast_math.hpp:220), and the continuation calls the same whole-scene rcNearestHit at raycast_math.hpp:1479. rcNearestHit (raycast_math.hpp:791-831) and rcTestInstance (raycast_math.hpp:758-778) take no exclusion parameter and have no inst != inst0 guard, and the only inst1 check at raycast_math.hpp:1483 is `inst1 >= 0`. rcHitBox is explicitly written to return the exit face for an origin inside the box (raycast_math.hpp:344-351: 'Unbounded interval here (not [tmin,tmax]) so an origin inside the box still yields the exit point'), so for any pane thicker than 1 mm the continuation immediately re-hits the front instance. The shipped worlds are all solid boxes: turtlebot3_ouster_headless.sdf:177-181 (glass_pane box 2 x 0.02 x 1.5, transparency 0.9, specular 1.0) and ouster_showcase.sdf:596-600 / 614-618 (F_pane_60 / F_pane_92, box 0.08 x 2.4 x 2.4). raycast_mirror.cpp:190-197 keeps SDF boxes as rc::GeomType::kBox (not meshes), so rcHitBox is the path taken. The back-face candidate is also NOT multiplied by (1-tau) — raycast_math.hpp:1463 applies (1-tau) only to the front surface, while raycast_math.hpp:1486-1490 multiplies the continuation by tau*tau only — so the self-hit's rho is (kd*cos + ks*lobe)*tau^2, which beats the front surface on rho/R^2 and is what gets published. The regression test cannot see it: test_raycast.cpp:422-427 builds the pane as rc::GeomType::kPlane with psize = {1,1,0} (zero thickness), the one geometry where the 1 mm advance lands past the surface and rcHitPlane (raycast_math.hpp:396-399, requires t > tmin) cannot re-fire. Severity corrected from critical to high: it is wrong published data for every transparent visual, but it is confined to the transparency feature (panel mode and opaque geometry are unaffected) and causes no crash or memory corruption.

---

## MEDIUM (25)

### `cuda/ray_processor_sycl.cpp:671` — SYCL backend never checks kernel errors: no async_handler, wait() not wait_and_throw()

*Backend divergence* · api-misuse · reported as high, downgraded by verifier

The queue is built as `sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}}` (line 671-672) with **no `async_handler`**, and every synchronisation point in the backend is `q_.wait()` — line 144 (`processRaw`), line 182 (`processDepth`), line 302 (`castScan`). There is no error check of any kind after the three `q_.parallel_for` submissions at lines 288, 387 and 521.

Per SYCL 2020, `queue::wait()` does NOT report asynchronous errors to the caller; only `wait_and_throw()`, `throw_asynchronous()` or queue destruction invoke the handler, and with no handler installed the implementation's default is to `std::terminate()` at that point. So a kernel that faults (bad USM pointer, out-of-resources, device lost) either produces nothing and is silently ignored, or blows up at shutdown far from the cause.

This is a straight divergence from the sibling backends, which check after *every* launch and throw a catchable `std::runtime_error`: `ray_processor_cuda.cu:93, 364, 373, 606` (`CUDA_CHECK(cudaGetLastError())`) and `ray_processor_hip.cpp:350, 374, 429, 523, 697` (`HIP_CHECK(hipGetLastError())`). `raycast_mirror.cpp:695-707` wraps `castScanProcessed` in try/catch precisely so a device error is logged instead of killing the Gazebo server — that guard is dead code on SYCL because nothing ever throws.

Note the allocation path IS handled (`allocShared` throws on a null `malloc_shared`, line 543); it is only kernel execution/launch that is unguarded.

**Failure scenario.** An Intel Arc GPU hits a device-lost / out-of-resources condition mid-scan on `processRaw`. `q_.parallel_for` (line 387) fails asynchronously. `q_.wait()` (line 144) returns normally, and lines 146-149 `std::memcpy` the *previous frame's* (or, on frame 1, uninitialised `malloc_shared`) contents of `u_range_`/`u_signal_`/`u_refl_`/`u_nearir_` into the caller's buffers. `publishChannels` then encodes those into genuine-looking Ouster UDP packets and ROS images. On CUDA or HIP the identical failure throws `std::runtime_error` and is logged. Downstream consumers get a plausible but stale/garbage point cloud with no diagnostic anywhere.

**Fix.** Construct the queue with an `async_handler` that rethrows as `std::runtime_error` (matching CUDA_CHECK/HIP_CHECK), e.g. `sycl::queue q{sycl::gpu_selector_v, [](sycl::exception_list el){ for (auto & e : el) std::rethrow_exception(e); }, sycl::property::queue::in_order{}}`, and replace all three `q_.wait()` calls (lines 144, 182, 302) with `q_.wait_and_throw()`. Wrap the resulting `sycl::exception` into `std::runtime_error` at the backend boundary so callers see the same exception type as the other backends.

**Verifier.** Confirmed by reading the file. ray_processor_sycl.cpp:671-672 constructs `sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}}` with no async_handler argument, and the only synchronisation points are bare `q_.wait()` at :144 (processRaw), :182 (processDepth) and :302 (castScan). The three submissions at :288, :387, :521 have no post-launch check of any kind — the only throw in the whole backend is `allocShared` at :543. The sibling backends do check: CUDA_CHECK(cudaGetLastError()) at ray_processor_cuda.cu:93,364,373,606 plus CUDA_CHECK(cudaStreamSynchronize) at :460,:508,:640,:683, and HIP_CHECK(hipGetLastError()) at ray_processor_hip.cpp:350,374,429,523,697 plus HIP_CHECK(hipStreamSynchronize) at :377,:432,:531 — every one of those line numbers is accurate. raycast_mirror.cpp:695-707 does wrap the castScanProcessed call in catch(const std::exception&)/catch(...) (its comment at :697-700 says it exists so a device/allocation error is logged instead of terminating the Gazebo server), and that guard is effectively dead on SYCL for asynchronous faults. After wait() returns, processRaw :146-149 unconditionally memcpys u_range_/u_signal_/u_refl_/u_nearir_ into the caller's buffers, which publishChannels then encodes — stale or uninitialised USM shipped as real returns. Two corrections to the reviewer: (a) a *synchronous* launch failure in DPC++ does throw a sycl::exception out of parallel_for, and sycl::exception derives from std::exception, so raycast_mirror.cpp:695 would catch that case — only genuinely asynchronous faults are silently swallowed; (b) it therefore is not true that 'nothing ever throws'. The defect is still real and is a genuine backend divergence, but it needs a GPU fault (device-lost / out-of-resources) to bite, which is not a routine path — HIGH is inflated.

---

### `.github/workflows/ci.yaml:6` — CI push trigger watches a branch that no longer exists; the real integration branch runs nothing

*Build & CI* · build · reported as high, downgraded by verifier

`on.push.branches: [main, develop]`. There is no `develop` ref in this repository any more — `git for-each-ref` shows exactly three remote branches: `origin/main`, `origin/cam_wip`, `origin/claude/ros2-gazebo-compatibility-xc64y8`. AUDIT.md:3-5 confirms `develop` *was* the integration line at audit time; it has since been replaced by `cam_wip` (see commit c3c6045 "Merge remote-tracking branch 'github/claude/...' into cam_wip"), and nobody updated the trigger. So half the push filter is dead, and the branch that actually accumulates work is not covered at all. The only remaining coverage is the unfiltered `pull_request:` at line 7 — which does nothing for a direct push or a local merge into `cam_wip`, exactly the pattern commit c3c6045 shows is in use. This is the mechanism by which the two red quality gates above got committed and stayed committed.

**Failure scenario.** Developer merges a feature branch into `cam_wip` locally and pushes (as in c3c6045). Zero CI jobs run: no build, no tests, no cppcheck, no cpplint, no doxygen, no smoke builds. The tree stays broken until somebody happens to open a PR, at which point three unrelated gates fail at once and the PR author gets blamed for pre-existing breakage.

**Fix.** Change line 6 to `branches: [main, cam_wip]` (or drop the filter entirely and let every push run CI), and add a `concurrency: {group: ${{ github.workflow }}-${{ github.ref }}, cancel-in-progress: true}` block so the extra coverage does not double the runner bill on PR branches.

**Verifier.** Factually confirmed. ci.yaml:6 reads 'branches: [main, develop]'. git for-each-ref returns exactly refs/heads/main, refs/heads/claude/ros2-gazebo-compatibility-xc64y8, refs/remotes/origin/{main,cam_wip,claude/ros2-gazebo-compatibility-xc64y8} — no develop anywhere. AUDIT.md:3 does say the prior audit ran on 'develop', and commit c3c6045 'Merge remote-tracking branch github/claude/... into cam_wip' exists, so the local-merge-then-push pattern into cam_wip is real and runs zero jobs. Downgraded from high: the unfiltered 'pull_request:' at ci.yaml:7 still gates every PR, which is the path that actually protects main, so the gap is missing early feedback on an integration branch rather than an unprotected merge path. The finding's causal claim ('this is the mechanism by which the two red gates got committed') is plausible but unproven — I did not verify that the offending commits never went through a PR.

---

### `CMakeLists.txt:101` — Tri-state backend option silently discards an explicitly typed -DGZ_GPU_OUSTER_USE_*:BOOL=... override

*Build & CI* · build

`gz_ouster_backend_option` treats *any* BOOL-typed cache entry as a legacy artifact and FORCE-overwrites it with AUTO. The comment at lines 90-92 justifies this with: "A BOOL-typed entry can only come from the old scheme (plain -DVAR=... command-line entries are UNINITIALIZED), so it is safe to migrate to AUTO." That claim is false. `-DVAR:BOOL=value` on the command line, and `set(VAR OFF CACHE BOOL "")` in a `-C` initial-cache file or toolchain file, both create a genuine BOOL entry.

I reproduced the macro verbatim in a standalone CMake project (cmake 3.28, same major as CI):
  cmake -DGZ_GPU_OUSTER_USE_CUDA:BOOL=ON   -> "MIGRATING legacy ... (BOOL 'ON') to AUTO"  -> MODE=AUTO
  cmake -DGZ_GPU_OUSTER_USE_CUDA:BOOL=OFF  -> "MIGRATING legacy ... (BOOL 'OFF') to AUTO" -> MODE=AUTO
  cmake -DGZ_GPU_OUSTER_USE_CUDA=ON        -> MODE=ON   (untyped works)
The explicit value is destroyed and replaced with AUTO, and the only diagnostic is a STATUS line claiming to be migrating a legacy cache — actively misleading, since there is no legacy cache involved.

The OFF case is the dangerous direction: lines 190-192 and 216-218 document OFF as the escape hatch that skips the toolchain probe entirely "so OFF also dodges a broken or hanging toolchain", and line 227-234 makes AUTO force SYCL on whenever CMAKE_CXX_COMPILER_ID is IntelLLVM. So OFF is silently inverted into ON.

**Failure scenario.** A packager builds with icpx and passes `colcon build --cmake-args -DGZ_GPU_OUSTER_USE_SYCL:BOOL=OFF` to work around a broken oneAPI install. The macro rewrites it to AUTO, the icpx check at CMakeLists.txt:228 fires, GZ_GPU_OUSTER_ENABLE_SYCL is set ON, cuda/CMakeLists.txt:104 adds ray_processor_sycl.cpp, and the build dies in the toolchain the user was explicitly avoiding. Mirror case: `-DGZ_GPU_OUSTER_USE_CUDA:BOOL=ON` on a box with no nvcc silently produces a CPU-only plugin instead of the documented FATAL_ERROR at lines 181-184, and ships as 'CUDA-accelerated'.

**Fix.** Only migrate when the entry is BOOL *and* its value is one CMake wrote itself, or better: detect the legacy scheme by a separate sentinel (e.g. `if(DEFINED CACHE{GZ_OUSTER_BACKEND_OPT_SCHEME})`) written by the new code, and migrate only when that sentinel is absent AND the type is BOOL. Simplest correct fix: drop the migration entirely (the pre-tri-state build dirs are long gone) and just normalize whatever value is present, so `:BOOL=ON`/`:BOOL=OFF` fall through the existing `MATCHES "^(ON|TRUE|YES|Y|1)$"` / `^(OFF|FALSE|NO|N|0|)$` branches at lines 113-116, which already handle them correctly.

**Verifier.** Confirmed by reading and by reproduction. CMakeLists.txt:99-107 does 'get_property(_gz_ouster_opt_type CACHE ${VAR} PROPERTY TYPE)' and, if it STREQUAL BOOL, FORCE-overwrites the entry with AUTO. The justifying comment at CMakeLists.txt:90-92 ('A BOOL-typed entry can only come from the old scheme') is false. I reproduced the macro standalone under cmake 3.28.3: '-DGZ_GPU_OUSTER_USE_SYCL:BOOL=OFF' prints the migration message and yields MODE=AUTO; ':BOOL=ON' likewise yields AUTO; the untyped '-D...=OFF' correctly yields MODE=OFF. The downstream consequences are as described: CMakeLists.txt:228 makes AUTO force SYCL on under CMAKE_CXX_COMPILER_ID IntelLLVM, so ':BOOL=OFF' under icpx is inverted to ON; and the FATAL_ERROR for a forced-but-unavailable CUDA lives only in the MODE STREQUAL ON branch (CMakeLists.txt:179-184), so ':BOOL=ON' with no nvcc falls into AUTO and silently produces a CPU-only build. A -C initial-cache or toolchain file doing 'set(VAR OFF CACHE BOOL "")' hits the same path since both are processed before line 101. Medium is right — requires a slightly unusual but entirely legitimate invocation.

---

### `cuda/ray_processor_cuda.cu:161` — AUDIT M2 is only half-fixed: the noise/channel kernel body is still copy-pasted across all four backends

*Cleanup* · cleanup

AUDIT.md M2 is marked resolved ("Shared cuda/ray_processor_math.hpp now drives all four backends"), but only the LEAF math was extracted. The ~140-line control flow of the per-pixel channel kernel is still four near-verbatim copies: `rayProcessKernel` (cuda/ray_processor_cuda.cu:161-331), `rayProcessKernelHip` (cuda/ray_processor_hip.cpp:133-282), the lambda in `launchRayKernel` (cuda/ray_processor_sycl.cpp:387-506) and `processCpu` (cuda/ray_processor_cpu_impl.cpp:44-191). Only the ~15 RNG call sites actually differ. The identical four-line "emit a miss" block (`range_out=0; signal_out=0; refl_out=base_reflectivity; nearir_out=0`) appears exactly 5 times per backend — 20 copies repo-wide (grep `signal_out\[idx\] = 0u`: cuda.cu:216,226,241,261,281 / hip.cpp:187,197,207,224,242 / sycl.cpp:411,419,430,446,464 / cpu_impl.cpp:70,81,94,113,132). AUDIT.md's M1 remediation note ("the dedup makes them visibly identical") is also false: the edge-suppression gate is STILL spelled three different ways — cpu_impl.cpp:90 `has_noise && p.edge_discon_threshold > 0.f`, cuda.cu:237 `edge_discon_threshold > 0.f && rs != nullptr`, sycl.cpp:426 bare `edge > 0.f`. Still open from AUDIT.md.

**Failure scenario.** A maintainer adds a sixth early-exit (e.g. a firmware blind-zone gate) or changes what a suppressed return writes to `nearir_out`. They edit cuda.cu's five exit blocks and cpu_impl.cpp's five, miss one of hip.cpp's or sycl.cpp's, and AMD/Intel users silently ship a different point cloud than NVIDIA/CPU users. Nothing in CI catches it: hip-smoke/sycl-smoke are compile-only (.github/workflows/ci.yaml:244,269) and no test compares backend outputs.

**Fix.** Hoist the kernel body into `rpmath` as a `GZ_OUSTER_HD` function template parameterised on an RNG functor (`template <class Rng> processOnePixel(idx, params, in, out, Rng& rng)`), with a single `emitNoReturn(out, idx, base_reflectivity)` helper. Each backend then supplies only its RNG adaptor (curand / hiprand / splitmix64 / std::mt19937) and the launch boilerplate — which is what AUDIT.md M2 claimed was already done.

**Verifier.** Verified by reading all four kernels. rayProcessKernel (cuda/ray_processor_cuda.cu:161-331), rayProcessKernelHip (cuda/ray_processor_hip.cpp:133-282+), the SYCL lambda (cuda/ray_processor_sycl.cpp:387-506) and processCpu (cuda/ray_processor_cpu_impl.cpp:44-191) are near-verbatim: same gate order (invalid->false-alarm, min/max window, edge suppression, dropout, range noise, requantize+recheck, signal, reflectivity, NIR), and the 4-line miss block really does appear 5x per backend (grep confirms exactly the 20 cited lines). The three spellings of the edge gate are real as cited: cpu_impl.cpp:90 `has_noise && p.edge_discon_threshold > 0.f`, cuda.cu:237 `edge_discon_threshold > 0.f && rs != nullptr`, sycl.cpp:426 bare `edge > 0.f` (HIP:203 matches CUDA). CI corroboration also checks out: hip-smoke (.github/workflows/ci.yaml:244) and sycl-smoke (:269) are `hipcc -c`/compile-only, so nothing catches a divergence. TWO CORRECTIONS to the framing, which is why the title overstates: (a) AUDIT.md M2 named the LEAF math (reflectivity slope-22, dropout, range noise, signal model, resample), and that math genuinely is shared now — every backend calls rpmath::dropoutProbability/rangeNoiseSigma/signalFromRange/reflectivityToByte/clampU16 — so 'still open from AUDIT.md' is wrong; the residual control-flow duplication is a NEW observation. (b) AUDIT M1 already recorded the gate spellings as logically equivalent (noiseEnabled() at cuda/backend.hpp:120+ includes edge_discon_threshold>0), and they still are, so there is no current output divergence. Held at medium as a maintenance hazard rather than a bug: the divergent spellings that survive today are the evidence the hazard is not hypothetical.

---

### `cuda/ray_processor_hip.cpp:435` — HIP and SYCL never override castScanProcessed, so the fused raycast path round-trips 3 float planes through host memory

*Cleanup* · efficiency

cuda/backend.hpp:80-82 documents the contract: "The default is the exact two-stage host path ... discrete GPU backends override this to keep intermediate float planes resident on-device." Grep for `castScanProcessed` finds exactly one override in the whole repo: cuda/ray_processor_cuda.cu:643. `HipBackend` (ray_processor_hip.cpp:286-764) and `SyclBackend` (ray_processor_sycl.cpp:78-655) implement only `castScan`, so they inherit backend.hpp:102-109, which calls `castScan(...)` (D2H of depth/retro/NIR into `depth_scratch`/`retro_scratch`/`nir_scratch`) and then `processDepth(...)` (H2D of the same three planes back onto the device). HIP on a discrete AMD card takes that hit on every scan; the data never needed to leave the GPU. All the pieces to fix it already exist in the HIP file — `castScan` is factored so the kernel launch, `d_depth_`, `d_retro_` and `d_nir_f_` are right there, exactly like the CUDA `launchCastScan`/`castScanProcessed` split.

**Failure scenario.** OS1-128 at 4096x512 = 2,097,152 pixels. Fused raycast on a discrete Radeon does 3 x 2,097,152 x 4 B = 25 MB D2H followed by the same 25 MB H2D every scan — ~50 MB of avoidable PCIe traffic per frame, ~500 MB/s at 10 Hz — on the raycast worker thread that gates the publish rate (src/raycast_mirror.cpp:678). The identical NVIDIA config transfers only the final channel arrays.

**Fix.** Split HIP's `castScan` into a `launchCastScan(...)` helper (mirroring cuda/ray_processor_cuda.cu:511-607) and add a `castScanProcessed` override that calls it, then launches `rayProcessKernelHip` on `d_depth_`/`d_retro_`/`d_nir_f_` and only `d2hResults(...)`. Do the same for SYCL, or — if USM makes it a wash there — narrow backend.hpp:80-82 to say only CUDA overrides.

**Verifier.** Confirmed. cuda/backend.hpp:80-82 states 'discrete GPU backends override this to keep intermediate float planes resident on-device'; the default body at backend.hpp:102-109 calls castScan then processDepth. Repo-wide grep finds exactly one override: cuda/ray_processor_cuda.cu:643. HipBackend implements only castScan (cuda/ray_processor_hip.cpp:435), whose tail does `d2h(range_out, d_depth_)`, `d2h(retro_out, d_retro_)` and `d2h(nir_out, d_nir_f_)` before hipStreamSynchronize; processDepth (hip.cpp:380) then h2d's d_depth_/d_retro_/d_nir_f_ straight back (hip.cpp:396-406). On a discrete Radeon h2d/d2h are hipMemcpyAsync (hip.cpp:598, :610), so the PCIe round-trip is real and is on the path src/raycast_mirror.cpp:678 calls every scan. Two caveats that do not refute it: HIP's integrated path allocates managed memory and degrades the copies to std::memcpy (hip.cpp:596, :608, name() returns 'hip-apu'), and SYCL is malloc_shared USM throughout (sycl.cpp:542), so there the waste is a redundant host memcpy of 3 planes (sycl.cpp:304-310 out, :165-175 back in), not bus traffic — which the finding itself concedes. Severity medium is right: perf cliff on one backend, no wrong data.

---

### `src/gz_gpu_ouster_lidar_system.cpp:750` — Panels mode gates metadata publishing and the entire IMU path behind the render thread creating the depth-camera rig

*Lifecycle & threading* · correctness · reported as high, downgraded by verifier

`if (!sensor_initialized_.load(std::memory_order_acquire)) return;` sits above *everything*: the metadata republish state machine (781), the sensor/IMU entity discovery (794, 815), and the IMU publish (954). In raycast mode that flag is set synchronously in Configure (608), so it is a no-op. In panels mode it is set only by the render thread at line 669, after `rig_->ensureCreated()` succeeds — which requires gz-sim's Sensors system to have built an ogre2 scene (panel_rig.cpp:80-85). The plugin already knows this can never happen in some worlds: lines 707-728 detect exactly that case and emit a detailed ERROR telling the user to add a rendering sensor. But the same condition also silently takes down two data paths that have nothing whatsoever to do with rendering. The IMU reads `components::AngularVelocity`/`LinearAcceleration` straight off the ECM (1078-1079) and publishes through RosInterface; the metadata publisher just pushes a fixed JSON string (ros_interface.cpp:336-340). Neither touches ogre2, gz::rendering, the rig, or the exchange. `publishMetadataIfNeeded` has exactly one call site (grep confirms ros_interface.cpp:309 defined, :781 called), so when that early return fires, /metadata is never published at all.

**Failure scenario.** A user runs `ray_mode=panels` in a headless world whose only other sensors are an IMU and an altimeter (no camera or gpu_lidar). events::Render never fires, `sensor_initialized_` stays false forever, PostUpdate returns at line 750 on every tick. The plugin logs the "add a rendering sensor" error at 2 s of sim time and then goes completely silent: no /metadata (so ouster_ros os_cloud never initialises and never produces a cloud even if packets later appeared), no /imu, no /imu_packets. The user's downstream Madgwick/robot_localization stack loses its IMU too, so odom->base_footprint never publishes — a failure two subsystems removed from the actual cause. Nothing in the log connects the missing IMU to the missing camera.

**Fix.** Split the guard. Move the metadata republish and the IMU block above line 750, gating each on what it actually needs (`ros_` for metadata; `imu_enabled_ && imu_entity_found_` for IMU, with the IMU entity search hoisted with it). Keep `sensor_initialized_` gating only the scan/exchange path that genuinely depends on the rig.

**Verifier.** Structurally confirmed. `if (!sensor_initialized_.load(...)) return;` at line 750 precedes the metadata republish (780-783), the lidar/IMU entity discovery (794, 815) and the IMU publish (953-955). sensor_initialized_ is written in exactly two places (grep): Configure line 608 for ray_mode=='raycast' (so a no-op there) and OnRender line 669, only after rig_->ensureCreated succeeds — which returns false until gz::rendering::engine("ogre2") exists with a scene (panel_rig.cpp:80-85). Both gated paths are genuinely renderer-independent: publishMetadataIfNeeded (ros_interface.cpp:309-345) only touches meta_pub_ and a string, and publishImu (gz_gpu_ouster_lidar_system.cpp:1071-1090) reads components::AngularVelocity/LinearAcceleration off the ECM. The plugin's own diagnostic at 707-728 proves the authors consider the never-renders world reachable. Downgraded from high to medium: in that configuration no point cloud is produced anyway and the user does get a loud one-shot ERROR at 2 s naming the exact fix, so the incremental damage is the IMU going dark plus /metadata never appearing — real, but a second-order effect of an already-diagnosed misconfiguration, not silent wrong data.

---

### `src/raycast_mirror.cpp:459` — Raycast scene-rebuild change detection samples per-tick ComponentState only on scan-due ticks, missing ~99% of edits

*Lifecycle & threading* · correctness

`postUpdate` returns at line 459 (`if (!gate.due) return;`) *before* the `ecm.Each` at 467-489 that computes `visual_data_changed` from `ecm.ComponentState(ent, type) != ComponentState::NoChange` (479-488). But `ComponentState` is a one-iteration edge signal, not a level: gz-sim's SimulationRunner clears the one-time-changed set at the end of every update cycle, so a component marked changed on tick N reads as `OneTimeChange` on tick N and `NoChange` on tick N+1. The gate only lets this code run once per scan period. At the default lidar_hz=10 with a 1 ms physics step, that is 1 tick in 100 — so 99% of one-time component changes are structurally invisible to the rebuild trigger. The two backstop signals do not cover it: `visual_count` (472) only catches spawn/despawn, and `visual_signature` (473-477) hashes only the entity id and `geom->Data().Type()`, so it is blind to changes in LaserRetro value, Material specular, Transparency, or a geometry edit that keeps the same shape type (e.g. resizing a box).

**Failure scenario.** During a running sim the user sets `laser_retro` on a wall visual to 0.9 (GUI component editor, or an ECM-writing system plugin). The write lands on a physics tick that is not scan-due — a 99% chance at 10 Hz / 1 kHz. `visual_data_changed` is never observed as true, `visual_count` and `visual_signature` are unchanged, `rebuildScene` is not called, and the mirrored `rc::Scene` keeps the old retro forever. Reflectivity in the published cloud silently stays at the stale value for the rest of the run with no log line. Same for resizing a box via <geometry> edits, or toggling a material's transparency.

**Fix.** Evaluate the change scan on every tick and latch it, rather than sampling it at scan cadence. Move the `ecm.Each` change-detection block above the `scan_gate_.advance` early return and OR its result into a `pending_rebuild_` member that is consumed (and cleared) when the gate next fires. The Each is cheap relative to rebuildScene; only the rebuild itself needs to stay on the scan cadence.

**Verifier.** Confirmed. raycast_mirror.cpp:458-459 computes `gate = scan_gate_.advance(sim_now, periodFromHz(params_.lidar_hz))` and returns when !gate.due; SimTimeGate::advance (include/gz_gpu_ouster_lidar/sim_time_scheduler.hpp:44-63) sets due at most once per period, so at lidar_hz=10 with a 1 ms step the block below runs on 1 tick in 100. The change scan sits after that return, at 467-489, and reads `ecm.ComponentState(ent, type) != ComponentState::NoChange` (479-482) — a per-iteration edge signal that gz-sim clears via SetAllComponentsUnchanged at the end of each update cycle, so a change on a non-due tick is never observed. The two backstops are as described and do not cover it: visual_count (472) catches only spawn/despawn, and visual_signature (473-477) hashes only `ent` and `geom->Data().Type()`, so a laser_retro edit, a Material specular change, a Transparency change, or a box resize that keeps GeometryType::BOX all leave it unchanged. Result: rebuildScene (492) is not called and the mirrored scene keeps stale retro/material data indefinitely. Medium is correct — latent, and it only bites on runtime component edits.

---

### `test/test_lifecycle.cpp:58` — test_lifecycle only exercises the never-Configured destructor, so the entire teardown ordering it exists to protect is untested

*Lifecycle & threading* · test-coverage

`GzGpuOusterLidarSystem`'s destructor carries a 35-line comment (src/gz_gpu_ouster_lidar_system.cpp:71-104) describing a four-step ordering contract: set shutdown_, disconnect render hooks, flush the barrier, then tear down producers -> encoder drain -> ROS. That ordering is load-bearing — `encoder_->stop()` (115) must precede `ros_->shutdown()` (117) or the drain thread publishes on a cancelled executor, and `mirror_->stop()` (111) must join the cast worker before `ray_processor_` and `processed_exchange_` are destroyed as members. The only test of any of this constructs a default plugin and lets it die (58, 66-69). In that state `rig_`, `mirror_`, `encoder_` and `ros_` are all null unique_ptrs, so the destructor body degenerates to one atomic store, two no-op `ConnectionPtr::reset()`s, an uncontended mutex acquire, and four failed null checks. Not one line of the ordering contract executes. The test's own comment is stale on top of that: it claims the plugin "default-constructs an rclcpp::executors::SingleThreadedExecutor member" (line 28-31) and mentions `gpu_rays_` (line 57) — the executor moved into RosInterface as a lazily-constructed unique_ptr (ros_interface.hpp:117-121) and gpu_rays_ no longer exists. Separately, CMakeLists.txt:305-397 shows no test target compiles src/packet_encoder.cpp or src/raycast_mirror.cpp, so the two thread-owning classes — the drain thread's pause/epoch cancellation state machine (packet_encoder.cpp:222-258) and the cast worker's job handoff (raycast_mirror.cpp:599-711) — have zero automated coverage of any kind.

**Failure scenario.** A maintainer refactors the destructor and moves `ros_->shutdown()` above `encoder_->stop()` — a natural-looking cleanup, since ROS is 'the outermost layer'. The drain thread is mid-batch inside `ros_->publishLidarPacket` (packet_encoder.cpp:257) when the executor is cancelled and the node torn down, and publishes on a dead rclcpp context. Both Lifecycle tests pass unchanged, because with every pointer null neither `stop()` nor `shutdown()` is ever reached. Same for swapping steps 1 and 3 of the render barrier, or dropping `mirror_->stop()` entirely.

**Fix.** Add a PacketEncoder-focused gtest that compiles src/packet_encoder.cpp with a stub RosInterface: start(), encodeScan() a batch, assert packets arrive at the stub, then setSimulationState(paused=true) and assert delivery halts, bump the epoch and assert the pending batch is cancelled, and finally stop() under load to assert the join completes and no publish lands after shutdown. Do the same for RaycastMirror::start/postUpdate/stop with a stub RayProcessor. Also fix the stale comments at test_lifecycle.cpp:28-31 and :57.

**Verifier.** Confirmed on every claim I could check. test_lifecycle.cpp has exactly two TESTs (53, 62), both of which default-construct GzGpuOusterLidarSystem and destruct it. All five teardown-relevant members are unique_ptrs default-initialised to null (header 155-162: meta_, ros_, rig_, mirror_, encoder_), so the destructor at src/gz_gpu_ouster_lidar_system.cpp:83-118 executes one atomic store, two no-op ConnectionPtr::reset()s, an uncontended recursive_mutex acquire, and four failed null checks — the ordering contract documented at 71-82 (mirror_->stop 111 -> encoder_->stop 115 -> ros_->shutdown 117) is entirely unexecuted, so reordering 113-118 cannot fail these tests. Stale comments verified: line 56 references `gpu_rays_`, which grep shows exists nowhere in src/ or include/; lines 28-31 claim the plugin default-constructs a SingleThreadedExecutor member, but it now lives in RosInterface as a lazily-constructed unique_ptr (src/ros_interface.hpp:117-121) inside a unique_ptr that Configure creates, so the stated reason for the rclcpp::init() environment is no longer true. CMake claim verified: CMakeLists.txt:275-277 lists packet_encoder.cpp/panel_rig.cpp/raycast_mirror.cpp only in the plugin library; no ament_add_gtest target in 305-397 compiles them, and test_lifecycle links ${PROJECT_NAME} but never calls into those classes. Kept at medium as a maintenance hazard, not raised — no shipped behaviour is wrong today.

---

### `cuda/ray_processor_cpu_impl.cpp:90` — Aerosol speckle feeds the depth-edge suppressor, silently deleting ~48% of all returns in fog

*Obscurants & noise* · correctness · reported as high, downgraded by verifier

`rcSampleMediumReturn` (`cuda/raycast_math.hpp:1158`) draws each medium range independently per pixel from the backscatter profile, so adjacent beams inside one homogeneous cloud differ by metres. That stochastic range then lands in the same depth buffer the channel stage runs its mixed-return heuristic over -- `castScan` writes into `depth_scratch`, `processDepth` reads it (`cuda/backend.hpp:103-108`):

```cpp
if (has_noise && p.edge_discon_threshold > 0.f) {
    if (rpmath::edgeDiscontinuity(depth_host, idx, H, W, p.edge_discon_threshold)
        && uni(rng) < rpmath::kEdgeSuppressProb) { ... suppress ... }
}
```

`edge_discon_threshold` defaults to 0.15 m (`src/lidar_common.hpp:40`) and `kEdgeSuppressProb` is 0.5, so any pixel with a neighbour more than 15 cm away is deleted with probability 1/2. Aerosol speckle trips it on essentially every pixel -- and not only on the medium returns themselves: a beam that cleanly hits a wall at 30 m is suppressed because its neighbour drew a smoke return at 7 m.

The result is that the carefully calibrated Poisson photon gate (`cuda/raycast_math.hpp:1190-1196`, verified accurate to ~0.003 against numerical integration) is halved by a downstream heuristic that physically should not fire for a genuine distributed return. docs/MODEL_REFERENCES.md §11 states only that "Electronic shot, range and dropout noise remain downstream sensor effects" and never mentions edge suppression; the whole point of the commit 1b5dd1c redesign was to control medium point density, and this silently overrides it by 2x. Identical code in all four backends: cuda/ray_processor_cuda.cu:237, cuda/ray_processor_hip.cpp:203, cuda/ray_processor_sycl.cpp:427.

**Failure scenario.** Measured in-tree on a 32x512 scan, sensor inside a cylindrical room (r=30 m), fog ellipsoid of 12 m semi-axes centred on the sensor, sigma=0.15 /m, edge_discon_threshold=0.15 m:

  no smoke: 16384 valid returns,     0 edge-flagged  ( 0.0%)
  smoke:    16384 valid returns, 15766 edge-flagged  (96.2%)

At the default kEdgeSuppressProb=0.5 that is ~48% of ALL returns in the scan deleted on top of the Poisson gate -- both the aerosol points and the hard-target points adjacent to them. The realised medium detection rate is therefore about half the documented 1 - exp(-lambda), and hard geometry near smoke acquires holes the model never predicted.

**Fix.** Exclude medium candidates from the mixed-return heuristic. The cleanest route is to carry a per-pixel flag out of rcCastOneRay (the ray caster already knows the medium won arbitration at cuda/raycast_math.hpp:1349-1352) and skip the edge gate for both a flagged pixel and any neighbour comparison against one -- a distributed-medium return is not a mixed edge return. Failing that, document the interaction and fold the expected suppression into the Poisson gate. Also add a test asserting the observed medium detection rate matches 1 - exp(-lambda) through the FULL castScanProcessed path, not just through castScan (test_obscurants.cpp:704 PhotonGateTracksIntegratedReceivedPower bypasses the channel stage entirely, which is why this is invisible today).

**Verifier.** Mechanism confirmed. rcSampleMediumReturn (raycast_math.hpp:1158-1242) draws each pixel's medium range from independent hashes (rcHashUnit(pixel, salt, stream), lines 1194/1201/1209/1231) over a segment that can span metres, and writes it back through rcApplyObscurants:1349-1352 into range_out. The raycast path is fused via raycast_mirror.cpp:678 -> Backend::castScanProcessed (backend.hpp:102-108), which feeds depth_scratch straight into processDepth, and gz_gpu_ouster_lidar_system.cpp:989-990 sets pp.edge_discon_threshold unconditionally from the 0.15 m default (src/lidar_common.hpp:40) — no raycast-mode opt-out exists. edgeDiscontinuity (ray_processor_math.hpp:423-443) flags a pixel if ANY cardinal neighbour is non-finite/below kValidDepthMin OR differs by more than the threshold, so the strongest case is even cleaner than the one the reviewer argued: the sparse phantom returns the model deliberately produces against open sky (ouster_smoke.sdf Zone C, raycast_math.hpp:1424-1440) are by construction surrounded by misses, so every one of them is flagged and half are deleted — the realized rate is 0.5*(1-exp(-lambda)), not the 1-exp(-lambda) documented at docs/MODEL_REFERENCES.md §11. Their test-gap claim also checks out: test_obscurants.cpp:704 works on castOne, never through the channel stage. Two reasons to knock the severity down. First, their headline "48% of ALL returns" rests on a 0.0%-flagged no-smoke baseline for a cylindrical room, which cannot be right — for a 32-beam OS1 at r=30 m the outermost elevation pairs differ by ~0.33 m > 0.15 m and must flag, so their baseline instrumentation is suspect and the marginal effect is smaller than stated. Second, the output error is point density on a stochastic channel, not wrong geometry, and edge_discon_threshold is a documented user knob (config/plugin_example.sdf:130).

---

### `cuda/ray_processor_math.hpp:291` — retro==0 sentinel: fully-extinguished targets snap back to base_reflectivity and mid-gray noise weighting

*Obscurants & noise* · correctness · reported as critical, downgraded by verifier

`retroForNoise` treats a non-positive retro as "the channel is absent" and substitutes `kDefaultRetro = 0.5f`:

```cpp
GZ_OUSTER_HD inline float retroForNoise(const float * retro, int idx)
{
    if (retro != nullptr) {
        const float r = retro[idx];
        if (gzm::isfinite_(r) && r > 0.0f) return r;   // <-- line 291
    }
    return kDefaultRetro;
}
```

The same `retro > 0` sentinel gates the REFLECTIVITY byte in every backend (`cuda/ray_processor_cpu_impl.cpp:168-172`, `cuda/ray_processor_cuda.cu:310-314`, `cuda/ray_processor_hip.cpp:263-264`, `cuda/ray_processor_sycl.cpp:486-487`), falling back to `base_reflectivity` (default 50).

That convention was safe when 0 only ever meant "laser_retro omitted". It is no longer: `rcApplyObscurants` (`cuda/raycast_math.hpp:1327`) does `rho *= trans_2way;` with `trans_2way = exp(-tau_eff)^2`, which **underflows to exactly 0.0f in float once tau_eff > ~52** (>~44 with FTZ on GPU). `src/raycast_mirror.cpp:337-339` also sets `has_retro = (lr != nullptr)`, so an explicitly authored `<laser_retro>0</laser_retro>` produces rho == 0 too. Both land on the same sentinel.

So the model is **non-monotonic in smoke density**. Measured in-tree (20 m wall, retro 0.8, medium return disabled, OS1 detection curve):

```
sigma= 5.0 tau_eff=50  retro=3.08e-44  retroForNoise=3.08e-44  byte=0   p_drop=0.1925
sigma= 6.0 tau_eff=60  retro=0.000e+00 retroForNoise=0.5       byte=50  p_drop=0.0109
```

Thickening the smoke makes the wall **17x more likely to be reported**, and reports it at REFLECTIVITY 50 (mid-gray). This flatly contradicts the header's own claim at `cuda/raycast_math.hpp:1256-1259` that the single multiply "lowers the calibrated REFLECTIVITY byte ... and makes targets disappear entirely once the smoke is thick enough", and docs/MODEL_REFERENCES.md §11 ("the calibrated REFLECTIVITY byte drops"). Commit ad40ab9 ("apply fallback reflectivity before obscuration") fixed the ambiguity inside the ray caster but left the downstream sentinel untouched, so the ordering change is only half correct.

**Failure scenario.** Verified by direct execution against the in-tree rc:: math (compiled cuda/raycast_scene.cpp + a driver). Scene: 20 m-thick plume [60,70] m with sigma=6 /m (tau_eff=60, true two-way transmittance 7.7e-53), wall at 100 m with laser_retro 0.8, OS1 detection profile, base_reflectivity 50, 4000 scan salts. Result: 3980/4000 beams report the wall at exactly 100.0 m with retro == 0 -> retroForNoise 0.5 -> REFLECTIVITY byte 50 and P(keep) = 0.498. A target behind smoke that attenuates by 1e-52 is reported as a mid-gray surface on half the beams. Same output for any surface authored <laser_retro>0</laser_retro>: byte 50, P(keep) 0.989.

**Fix.** Stop overloading 0.0f as "channel absent". Options, cheapest first: (a) have the ray caster emit a negative sentinel (e.g. -1.0f) for "no retro channel" and keep 0 meaning "physically zero return", then change `retroForNoise` to `r >= 0.0f` and the four reflectivity-byte sites to `retro[idx] >= 0.0f`; (b) since ad40ab9 already resolves the fallback inside the caster, the raycast path never needs a sentinel at all -- pass a separate `has_retro` flag/buffer, or simply treat a null `retro` pointer as the only "absent" case (which is what the panels path already relies on). Additionally, clamp `trans_2way` away from float underflow (e.g. compute `exp(-2*tau_eff)` directly rather than squaring, and floor the product at a small positive epsilon) so extinction degrades smoothly instead of snapping to exactly 0.

**Verifier.** Mechanism confirmed by reading. ray_processor_math.hpp:287-294 does return kDefaultRetro (=0.5f, line 69) whenever retro[idx] is not >0, and the identical `retro[idx] > 0.f` gate guards the REFLECTIVITY byte in all four backends (ray_processor_cpu_impl.cpp:168-172, ray_processor_cuda.cu:309-313, ray_processor_hip.cpp:263-267, ray_processor_sycl.cpp:486-490) — all falling back to base_reflectivity. Both producers of a hard rho==0 are real: (a) raycast_math.hpp:1327 `rho *= trans_2way` with trans_2way = exp(-tau_eff)^2 (lines 1323-1324) underflows to exactly 0.0f once 2*tau_eff exceeds ~103, and nothing in rcApplyObscurants floors it (the only `rho <= 0` guard, line 1236, applies to the MEDIUM candidate only); (b) src/raycast_mirror.cpp:337 passes `lr != nullptr` as has_retro, and raycast_math.hpp:681/749 then use inst.retro verbatim, so an authored <laser_retro>0</laser_retro> reaches retro_out as 0. That directly contradicts raycast_math.hpp:1455 ("an explicitly authored zero remains zero") and docs/MODEL_REFERENCES.md §11 ("an explicitly authored zero is not mistaken for omission"). Not in AUDIT.md. Two corrections to the reviewer, both severity-reducing: the underflow branch needs tau_eff > ~52, i.e. ~30x the densest shipped configuration (examples/worlds/ouster_smoke.sdf tops out at sigma 1.0/m over 1.8 m => tau_eff 1.8), and no shipped world authors laser_retro 0 — so the path is latent, not something users hit today. Also note the reviewer's proposed fix (r >= 0.0f) would route rho==0 into calibratedRangeAtReflectivity, where log_(0) makes d90==0 and line 321 returns 1.0 ("profile detection disabled") — i.e. always detected. The sentinel overload is real at three sites, not one.

---

### `src/gz_gpu_ouster_lidar_system.cpp:318` — max_range is unbounded above, so an SDF override past 524.287 m silently aliases in the 19-bit RANGE field

*Ouster protocol* · correctness

`clamp_warn(max_range_, 1.0, kInfD, "max_range")` bounds `max_range` below but not above. Ranges are written as raw millimetres with no field-width check: `range_out[idx] = static_cast<uint32_t>(d * rpmath::kRangeToMm);` (`cuda/ray_processor_cpu_impl.cpp:137`, and the identical line in the CUDA/HIP/SYCL backends).

`FieldInfo::set` (`parsing.cpp:83-96`) does `word &= mask; *ptr &= ~mask; *ptr |= word;` — it masks, it does not saturate or throw. `RANGE` is `field_info(0, 19)` for every RNG19 profile (`parsing.cpp:229, 245, 283, 295, 322`), i.e. mask `0x7FFFF`, max 524287 mm = 524.287 m. LEGACY is 20 bits = 1048.575 m; the low-bandwidth profiles are 15 bits with a 3-bit upshift = 262.136 m.

The defaults are safe only by accident: the largest `representable_range_m` in the profile tables is 500.0 (`ouster_lidar_profile.cpp:203`). Nothing stops a user from raising it.

**Failure scenario.** `<max_range>600</max_range>` on an OS2 with the shipped `RNG19_RFL8_SIG16_NIR16` metadata. A target at 560 m passes the `d < p.max_range` gate at `ray_processor_cpu_impl.cpp:130`, is written as 560000 mm, and `set` masks it to `560000 & 0x7FFFF = 35712` — os_cloud renders the point at **35.7 m**, in front of the sensor, with a plausible signal and reflectivity. No warning anywhere in the pipeline.

**Fix.** After the profile resolves, clamp `max_range_` against the active profile's RANGE field width (derivable from `pw->fields`/`udp_profile_lidar`, or hard-coded per profile family: 524.287 m for RNG19, 1048.575 m for LEGACY, 262.136 m for RNG15) and log the clamp. Belt-and-braces: saturate rather than truncate at `ray_processor_cpu_impl.cpp:137` and the three GPU equivalents.

**Verifier.** Every link verified. gz_gpu_ouster_lidar_system.cpp:318 is `clamp_warn(max_range_, 1.0, kInfD, "max_range")` - no upper bound (the lambda is at :291-296). ray_processor_cpu_impl.cpp:130 gates on `d >= p.max_range` and :137 writes `static_cast<uint32_t>(d * rpmath::kRangeToMm)` with kRangeToMm=1000 (ray_processor_math.hpp:54); the same line exists in ray_processor_cuda.cu:287, ray_processor_hip.cpp:247, ray_processor_sycl.cpp:469. FieldInfo::set (parsing.cpp:82-95) ends in `word &= mask; *ptr &= ~mask; *ptr |= word;` - it truncates, never saturates or throws. RANGE is field_info(0, 19) in SINGLE_FIELD_INFO (parsing.cpp:283) and the other RNG19 tables, mask 0x7FFFF. 560000 & 0x7FFFF = 35712, so the arithmetic in the failure scenario is exact. Kept at medium rather than higher: it needs an SDF max_range above any real Ouster spec (largest table value is 500.0 at ouster_lidar_profile.cpp:203, under the 524.287 m limit), so defaults are safe - but the override is a plausible user action and produces plausible-looking phantom points with zero diagnostics.

---

### `src/ouster_lidar_profile.cpp:373` — os2_128_rev7.json can never resolve to Rev07 — the resolver hands it Gen2 physics with a 2.5x wrong detection range

*Ouster protocol* · correctness · reported as high, downgraded by verifier

The firmware-based revision inference deliberately excludes OS2:
```
} else if (p.revision == OusterRevision::Unknown &&
           request.firmware_major >= 3 &&
           (p.model == OusterModel::OS0 || p.model == OusterModel::OS1 ||
            p.model == OusterModel::OSDome)) {     // line 375-376 — no OS2
    p.revision = OusterRevision::Rev07;
```
The shipped `config/metadata/os2_128_rev7.json` has `prod_line: "OS2-128"`, `prod_pn: "860-os2128"` and `image_rev: ousteros-image-prod-aries-v3.2.0`. Trace it: `revisionFromPartNumber("860-OS2128")` matches none of the patterns at lines 68-87 → `Unknown`. Branch fw>=4 → no. OS1Max → no. OS1/beam16 → no. Branch fw>=3 → OS2 excluded → no. So `p.revision` stays `Unknown`, line 383 applies `fallbackRevision(OS2)` = **Rev06**, and `setLegacyModelPhysics` runs instead of `setRev7Physics`.

Delta shipped vs. intended (compare lines 128-141 against 174-184):
- `detection_range_10_d90_m`: 80.0 instead of 200.0
- `detection_range_80_d90_m`: 210.0 instead of 350.0
- `representable_range_m` (and therefore the default `max_range`): 465.0 instead of 404.0
- `minimum_range_m`: 1.0 instead of 0.8
- `precision_min/max_std_m`: 0.025/0.080 instead of 0.020/0.100
- reported generation: Gen2 instead of Gen3/L3

`compatible()` explicitly allows OS2+Rev07 (line 254-256) and `setRev7Physics` has a full OS2 branch, so the values exist — they are just unreachable from a plain OS2 metadata. `test_ouster_lidar_profile.cpp:153-164` pins the *fallback* behaviour at fw 2.5 and `:101-116` tests OS2 Rev07 only via an explicit `hardware_revision` string, so nothing in the suite ever resolves the shipped file.

**Failure scenario.** Load `config/metadata/os2_128_rev7.json` with no `<hardware_revision>` (the documented default is `auto`). The plugin logs a WARN telling the user it cannot infer the revision, then runs the OS2 dropout model with D90@10% = 80 m instead of 200 m. Every return past ~80 m on a 10%-reflectance target is thinned by the logistic detection curve that should not kick in until 200 m, so a long-range OS2 simulation loses more than half its useful range while the ROS param `lidar_profile` reports `OS2-rev06`.

**Fix.** Add `OusterModel::OS2` to the fw>=3 inference branch (with a fw<=2.5 guard if the Rev7.0/FW2.5 constraint in the comment at line 258-259 is meant to be enforced), or set `prod_pn` in the shipped OS2 files to a real Rev7 part number that `revisionFromPartNumber` recognises (e.g. one containing `-070-`). Add a test that resolves every file in `config/metadata/` and asserts `fallback_revision == false` and that the resolved revision matches the filename.

**Verifier.** The mechanism is exactly as described. os2_128_rev7.json has prod_pn="860-os2128", image_rev=v3.2.0; revisionFromPartNumber (ouster_lidar_profile.cpp:60-88) matches nothing (uppercased "860-OS2128" ends in '8', so compact_c/compact_d at :78-81 fail); the fw>=4, OS1Max, OS1/beam16 and fw>=3 branches (:358-379) all exclude OS2, so line 382-384 applies fallbackRevision(OS2)=Rev06 (:95) and line 427 runs setLegacyModelPhysics. All six deltas check out against :128-141 vs :174-184 (80 vs 200, 210 vs 350, 465 vs 404, 1.0 vs 0.8, 0.025/0.080 vs 0.020/0.100). Test claim also verified: test_ouster_lidar_profile.cpp:153-164 pins exactly this fallback at fw 2.5, and :101-116 only reaches OS2 Rev7 via an explicit revision string. Downgraded from high for two reasons the finding glosses over: (1) it is not silent - ouster_metadata.cpp:140-147 emits a WARN naming the fallback revision actually used; (2) every shipped example defaults hardware_revision to rev07 (examples/urdf/ouster_standalone.urdf.xacro:17, ouster_macro.xacro:41, sensor_stack.urdf.xacro:19-20, launch files), and README.md:639-641/703 explicitly tells users to set it explicitly for synthetic metadata, so the auto path is the off-road case. Also note the exclusion looks deliberate: the comment at :257-259 says OS2 stayed on Rev7.0/FW2.5, so the reviewer's proposed 'add OS2 to the fw>=3 branch' fights the model - the shipped JSON's v3.2.0 image_rev is the more likely culprit.

---

### `src/ouster_lidar_profile.cpp:382` — An unrecognised <hardware_revision> silently discards a correct part-number inference and is never reported

*Ouster protocol* · correctness

```
const OusterRevision requested = parseOusterRevision(request.hardware_revision);
if (requested != Auto && requested != Unknown) { p.revision = requested; }
else { /* infer from prod_pn, may succeed */ }

if (p.revision == Unknown || requested == Unknown) {     // line 382
    p.revision = fallbackRevision(p.model);
    p.fallback_revision = true;
}
```
The `requested == Unknown` disjunct clobbers a *successful* part-number inference. `parseOusterRevision` (lines 283-303) accepts only an exact canonicalised set: `rev7.0`, `7.0`, `gen2`, `revB`, `Rev 8` (→ `REV8` ok) etc. — `rev7.0` canonicalises to `REV70`, which matches nothing, and `gen2` matches nothing. There is no error path for an unrecognised value: nothing in `ouster_metadata.cpp` compares `parseOusterRevision(hardware_revision)` against `Unknown`, so the user gets no "unknown revision" message. Instead they get the WARN at `ouster_metadata.cpp:140-147` — "Cannot infer hardware revision from prod_pn='...'; ... Set `<hardware_revision>` explicitly" — which is actively misleading in this case: the code *did* infer, and it advised the user to do the thing they already did.

**Failure scenario.** SDF contains `<hardware_revision>rev7.0</hardware_revision>` with metadata whose `prod_pn` is `840-102145-D`. `revisionFromPartNumber` correctly returns `RevD`, but line 382 sees `requested == Unknown` and overwrites it with `fallbackRevision(OS1)` = `Rev06`. The user is told to set a revision they already set, and the sensor silently runs Rev06 physics instead of either RevD or the Rev07 they asked for.

**Fix.** Distinguish the three cases. Return/flag `Unknown` from the resolver as an error (`p.supported = false` or a dedicated `revision_unparsed` flag) so `OusterMetadata::load()` can emit `"unrecognised <hardware_revision>='%s'; valid: gen1, revC, revD, rev05, rev06, rev06.2, rev07, rev07.1, rev08"` and fail, and drop `requested == Unknown` from the line-382 condition so it no longer overrides a good inference.

**Verifier.** Confirmed. canonical() (ouster_lidar_profile.cpp:14-22) strips non-alnum and uppercases, so "rev7.0" -> "REV70" which matches none of the arms in parseOusterRevision (:283-303) -> Unknown. Line 351 then falls into the else branch and the prod_pn inference at :354 can succeed, but line 382's `|| requested == OusterRevision::Unknown` unconditionally overwrites it with fallbackRevision(p.model). Verified there is no error path: I read ouster_metadata.cpp end to end and it never calls parseOusterRevision nor inspects a parse failure - the user only gets the WARN at :140-147 which claims the revision could not be inferred from prod_pn, which is exactly backwards. One correction to the reviewer's example: RevD and Rev06 both land in setLegacyModelPhysics (:426-433), so their optical physics are identical; the observable divergence there is max_returns (hasDualReturns at :51-58 gives Rev06 -> 2, RevD -> 1) and the reported profile.id. The larger real harm is the typo case (user writes rev7.0 wanting Rev07 and gets Rev06 physics: OS0 15/45 m instead of 35/75 m). Needs a malformed SDF value and does emit a warning, so medium, not high.

---

### `cuda/backend.hpp:102` — HIP and SYCL never override castScanProcessed, so raycast mode does a full host round trip per scan

*Performance* · performance

The default `Backend::castScanProcessed` (lines 103-108) is the naive two-stage host path: `castScan(...)` writes depth/retro/nir into HOST scratch, then `processDepth(...)` uploads those same three planes straight back to the device. CUDA overrides it (ray_processor_cuda.cu:643) and keeps the intermediates resident in `d_depth_`/`d_retro_`/`d_nir_f_`. HIP does not — grep of cuda/ray_processor_hip.cpp shows `override` only on the dtor, processRaw, processDepth, castScan (line 448) and name (line 534). SYCL likewise (cuda/ray_processor_sycl.cpp, `override` only at 91/127/160/203/314). So on AMD and Intel, `RaycastMirror::threadFunc` (src/raycast_mirror.cpp:678) hands in `scratch_` and every scan pays: D2H of 3 float planes + device sync (hip castScan ends with a stream sync; sycl with `q_.wait()`), then H2D of the same 3 planes + a second sync, then D2H of the 4 channel planes. The comment at raycast_mirror.cpp:674-677 acknowledges this, but it is still shipped cost.

**Failure scenario.** OS1-128 @ 2048 on an AMD APU or Intel Arc: 3 x 262,144 x 4 B = 3.15 MiB down plus 3.15 MiB back up = 6.3 MiB of avoidable PCIe/USM traffic per scan, plus one extra full device synchronization that serializes the cast kernel against the channel kernel. At 20 Hz that is ~126 MB/s of pointless transfer and an extra ~0.5-2 ms of pipeline stall per scan on the raycast worker — for a sensor whose whole point is real-time cadence.

**Fix.** Both backends already have every piece: `launchCastScan`-equivalent bodies that write `d_depth_`/`d_retro_`/`d_nir_f_`, and `launchRayProcessKernel`/the SYCL channel kernel that read them. Split HIP's `castScan` (line 435) the way CUDA splits it (a private `launchCastScan` + two public entry points) and add the `castScanProcessed` override that chains the two kernels on the same stream/queue with a single sync at the end. Same for SYCL. That deletes both extra copies and one sync.

**Verifier.** Verified by grepping `override` in each backend: ray_processor_hip.cpp has it only at 291/329/388/448/534 and ray_processor_sycl.cpp only at 91/127/160/203/314 — neither overrides castScanProcessed, while ray_processor_cuda.cu:643 does (with a private launchCastScan at :511). HIP's castScan ends with d2h of depth/retro/nir into the caller's host scratch plus hipStreamSynchronize (ray_processor_hip.cpp:524-531), so the default composition at backend.hpp:103-108 really does push those three planes back H2D inside processDepth. RaycastMirror::threadFunc passes scratch_ (raycast_mirror.cpp:678-687) and the comment at :673-676 explicitly acknowledges the gap. Kept at medium rather than higher because the code is correct and the limitation is documented/deliberate, and both backends report integrated devices (`hip-apu` / `sycl-igpu`, :534 / :314) where the transfer is over shared memory and much cheaper than the PCIe figure quoted.

---

### `cuda/ray_processor_cpu_impl.cpp:44` — processCpu is single-threaded while the raycast it feeds is OpenMP-parallel

*Performance* · performance · reported as high, downgraded by verifier

`for (int idx = 0; idx < n; ++idx)` at line 44 has no `#pragma omp parallel for`, unlike `processRawCpu`'s resample loop at line 216 and `rc::castScan` at raycast_scene.cpp:379, which are both parallel. The reason is structural, not intentional: lines 35-40 create ONE `std::mt19937` (`tl_rng` or `local_rng`) plus shared `normal_distribution`/`uniform_real_distribution` objects that the loop body mutates, so the loop cannot be parallelised as written. On the CPU backend the raycast path is `castScanProcessed` -> (backend.hpp:103) `castScan` (all cores) -> (backend.hpp:106) `processDepth` -> `processCpu` (one core). That is a textbook Amdahl tail on the raycast worker thread. The per-pixel body is not cheap: `dropoutProbability` (ray_processor_math.hpp:336) -> `detectionProbability` -> two `calibratedRangeAtReflectivity` calls = 4 log + 2 exp, plus the sigmoid exp at line 331, plus a sqrt in `rangeNoiseSigma`, plus a sqrt each for signal and near-IR shot noise, plus 3-4 mt19937 draws and one or two normal draws.

**Failure scenario.** OS1-128 @ 2048x10 Hz on the CPU fallback (the path Configure logs at gz_gpu_ouster_lidar_system.cpp:451 for any container without GPU passthrough): 262,144 pixels x ~7 transcendentals + RNG ~= 200-400 cycles/pixel = 52-105M cycles = 17-35 ms per scan on ONE core, while the 16-core machine's other 15 cores sit idle waiting. At 10 Hz that is 17-35% of a core spent serially; raise lidar_hz to 20 and the worker cannot keep up and starts logging 'dropped raycast frame' from raycast_mirror.cpp:692.

**Fix.** Parallelise the loop with a per-chunk RNG instead of a shared one — exactly what the GPU backends already do (cuda/ray_processor_cuda.cu:222 gives every pixel its own `curandState`). Derive a counter-based stream per pixel from (seed, idx), or seed one `std::mt19937` per OpenMP thread from `splitmix64(seed ^ omp_get_thread_num())` and use `#pragma omp parallel for schedule(static)`. Determinism is preserved for the seeded test path if the stream is keyed on `idx` rather than on draw order.

**Verifier.** Confirmed: ray_processor_cpu_impl.cpp:44 `for (int idx = 0; idx < n; ++idx)` has no omp pragma, while processRawCpu:216 and rc::castScan (raycast_scene.cpp:379) both carry `#pragma omp parallel for schedule(static) if(...)`. The shared-RNG blocker is real (lines 35-40: one mt19937 plus shared distribution objects mutated in the body). The default Backend::castScanProcessed (backend.hpp:103-108) does call castScan then processDepth, and CpuBackend::processDepth (backend_cpu.cpp:47) routes straight into processCpu, so the CPU raycast path really is parallel-cast-then-serial-channel. OpenMP is optional (CMakeLists.txt:256 find_package QUIET), so on a build without it the asymmetry vanishes entirely. Downgraded from high: pure throughput, no wrong output, and the fix has a real cost the reviewer glosses — parallelising changes the RNG draw order and would break the seeded bit-identical CPU golden path referenced in AUDIT.md unless the stream is re-keyed per pixel.

---

### `cuda/raycast_math.hpp:1118` — Medium sampler burns 3-6 exp/log per segment on a quadrature it then throws away

*Performance* · performance · reported as critical, downgraded by verifier

`rcSelectMediumSegment` computes `profile_mass += weight * rcProposalAcceptanceMean(use_exp, k, next - t, r0, r1);` unconditionally (line 1118), but only writes it out when `profile_mass_out != nullptr` (lines 1122-1124, 1142). `rcSampleMediumReturn` calls the walker SEVEN times per ray: once at line 1183 with `&profile_mass` (needed), then up to six more times at line 1203 inside the rejection loop with `nullptr` (line 1205 — the result is discarded). `rcProposalAcceptanceMean` (line 1032) is a 3-point Gauss-Legendre loop; each iteration evaluates `exp_(-q)` + `log_(1 - u*(1-exp_(-q)))` (use_exp branch) or `exp_(-2*k*x)` — 3 to 6 transcendentals per call, once per segment walked. So every rejection attempt pays the full quadrature for every segment it steps past before the pick lands, and 100% of that work is dead. The whole walk is re-derived from scratch too: line 1088's `rcOpticalDepth` re-intersects every obscurant volume, and the segment structure (a, b, beta, k, tau, use_exp, weight) is byte-identical across all seven walks — only the scalar `pick` differs.

**Failure scenario.** ouster_smoke.sdf with one fog volume, OS1-128 at 2048x10Hz (H*W = 262,144 rays). Every ray that intersects the plume and passes the Poisson gate at line 1194 enters the 6-attempt loop. With S=1 segment that is 6 wasted `rcProposalAcceptanceMean` calls x ~6 transcendentals = ~36 dead exp/log per ray, ~9.4M per scan, ~94M/s at 10 Hz. Two overlapping volumes (S~3 segments) triples it to ~28M dead transcendentals per scan. On the CPU backend this alone exceeds the entire per-scan budget; on CUDA it is ~36 extra SFU-serialized ops per thread with no `--use_fast_math` (cuda/CMakeLists.txt sets no fast-math flag, so these are the full-precision libdevice routines).

**Fix.** Guard it: `if (profile_mass_out != nullptr) profile_mass += weight * rcProposalAcceptanceMean(...);`. That is a one-line, zero-behaviour-change fix that removes ~85% of the sampler's transcendental cost. Then go structural: hoist the segment walk out of the attempt loop entirely — build the (a, b, beta, k, tau, use_exp, weight) list and its prefix sums ONCE into stack arrays of `2*kMaxObscurants+1` entries (the loop at line 1092 already bounds it), and have each of the six attempts do a binary search over the prefix sums instead of re-walking and re-integrating.

**Verifier.** Code confirms the claim: raycast_math.hpp:1118 accumulates `profile_mass += weight * rcProposalAcceptanceMean(...)` unconditionally, and the value is only consumed at :1122-1124 / :1142 behind `profile_mass_out != nullptr`. rcSampleMediumReturn passes `&profile_mass` once (:1185) and `nullptr` in the rejection loop (:1205), so every attempt-loop walk pays a 3-point quadrature (2 transcendentals per point at :1047/:1055) that is provably discarded. The one-line guard is zero-behaviour-change. Severity is NOT critical: no wrong data, no crash — it is dead arithmetic. The magnitude is also inflated ~6x: the loop at :1199 returns on the first accepted sample (:1239), so a typical obscurant ray pays 1-2 dead calls, not 6, and rays that intersect no obscurant return at :1177 before any of this. Reachable only when sp.n_obscurants>0 and pulse_gate_m>0 (ouster_smoke.sdf exists, examples/worlds/).

---

### `cuda/raycast_math.hpp:444` — BVH and TLAS traversal push children in fixed order, defeating the `best`/`limit` cull

*Performance* · performance

`rcHitMesh` pushes `stack[sp++] = n.left; stack[sp++] = n.right;` (lines 444-446) with no regard for which child the ray enters first, and `rcNearestHit`'s TLAS loop does the same (lines 829-832). Both traversals maintain a shrinking cutoff (`limit` at line 427, `best` at line 822) that is only effective if the NEAR child is visited first — a far-child-first descent tests geometry that a subsequent near hit would have culled entirely. This is the single best-known constant-factor win in a BVH ray tracer and the code is already structured to benefit: the cull test `rcHitAabb(o, d, n.bmin, n.bmax, tmin, limit)` is right there at the top of the loop.

**Failure scenario.** turtlebot3_ouster_warehouse.sdf: a beam entering a shelving mesh descends into the far half of the shelf first, intersects triangles at t=8 m, then descends the near half and finds t=2 m — the 8 m triangle tests were pure waste. Typical published speedups for adding ordered traversal to an unordered median-split BVH are 1.3-2x on the traversal phase, which for the raycast mode IS the frame. Same effect one level up in the TLAS: with kTlasLeaf=2 and a median split, a ray commonly tests a far cluster of instances before the wall 2 m in front of it.

**Fix.** Compute the entry distance for both children (rcSlabAxis already yields `lo`) and push the farther child first so the nearer pops first — or, cheaper and nearly as good, use the ray's sign bit on the node's split axis. Store the split axis in `MeshBvhNode` (there is padding room next to `left`/`right`) at build time in `buildBvhNode` (raycast_scene.cpp:63-66 already picks it) and `buildTlasNode` (raycast_scene.cpp:133-139), then swap the push order when `d[axis] < 0`. Zero traversal-loop cost, no extra memory traffic.

**Verifier.** Confirmed at both sites: rcHitMesh pushes `stack[sp++] = n.left; stack[sp++] = n.right;` (:443-445) and rcNearestHit's TLAS loop does the same (:829-831), with no entry-distance comparison, while both maintain the shrinking cutoff the ordering would exploit (`limit` at :421/:426, `best` at :801/:823). Both builders already pick a split axis that is thrown away (raycast_scene.cpp:58-64 for the BLAS, :137-143 for the TLAS). One factual error in the proposed fix: MeshBvhNode (raycast_math.hpp:35-42) is 6 floats + 4 ints = 40 bytes tightly packed — there is no padding room; the axis would have to reuse `first`/`count`, which are unused on internal nodes. Severity stays medium: this is a missing optimisation in the hot loop, not a defect producing wrong output.

---

### `examples/launch/ouster_standalone.launch.py:180` — os_cloud/os_image never declare a `metadata` parameter — all three launches pass one and it is silently dropped

*Launch / URDF / worlds* · api-misuse · reported as high, downgraded by verifier

Every example launch passes `'metadata': metadata` to both ouster_ros nodes, with 6-line comments claiming it "Seed[s] the processing pipeline before packets arrive... waiting for the bag's one-shot metadata message before creating the packet subscription can otherwise lose the first scans" (ouster_standalone.launch.py:176-180, :239; turtlebot3_ouster.launch.py:177, :219; sensor_stack.launch.py:85, :117).

At the pinned ouster-ros commit (gz_sensors_ouster.repos: 6ab9402c1a8275f600945c3d8dfd5a73b40585c8), neither node declares that parameter. os_cloud declares tf_bcast params + timestamp_mode, ptp_utc_tai_offset, proc_mask, use_system_default_qos, scan_ring, point_type, organized, destagger, min_range, max_range, v_reduction, min_scan_valid_columns_ratio, mask_path (src/os_cloud_node.cpp:57-71). os_image declares timestamp_mode, ptp_utc_tai_offset, use_system_default_qos, min_scan_valid_columns_ratio, mask_path, distortion_model, sensor_frame, publish_camera_info (src/os_image_node.cpp:45-53). Both get metadata ONLY from the latched `metadata` TOPIC via create_metadata_subscriber() (os_processing_node_base.cpp:13-21), and both create their `lidar_packets` subscription inside metadata_handler() (os_image_node.cpp:142, reached from :65).

Verified empirically in this workspace: launching each node with `-p metadata:=...` and then `ros2 param list` shows no `metadata` entry — rclcpp discards overrides for undeclared parameters without error. So the mitigation the comments describe does not exist; the race they claim to close is still wide open, and the launch files carry ~20 lines of confidently-wrong rationale.

**Failure scenario.** `ros2 launch gz_sensors_ouster ouster_standalone.launch.py`, or replay of a bag recorded from it at `--rate 10`. The launch author believes os_cloud/os_image are pre-seeded from the JSON on disk. They are not: both nodes sit with no `lidar_packets` subscription until the latched `/sensor/lidar/lidar0/metadata` String arrives. Every packet published in that window is dropped, and os_cloud rejects the partial scan. A maintainer debugging the missing first scans reads the comment, concludes pre-seeding is already handled, and looks elsewhere.

**Fix.** Delete the `'metadata': metadata` entries and the pre-seeding comments from all three launch files (they are inert), and rely on the transient_local metadata publisher the plugin already uses (ros_interface.cpp:73-76), which matches the subscriber's latching QoS. If genuine pre-seeding is wanted, it has to be added upstream in ouster-ros (declare_parameter("metadata", "") + load-from-file in on_init) and the pin bumped — not asserted from the launch file.

**Verifier.** Confirmed against the pinned checkout. /root/ros2_ws/src/ouster-ros is at 6ab9402c1a8275f600945c3d8dfd5a73b40585c8; os_cloud_node.cpp:57-71 declares tf_bcast params + timestamp_mode/ptp_utc_tai_offset/proc_mask/use_system_default_qos/scan_ring/point_type/organized/destagger/min_range/max_range/v_reduction/min_scan_valid_columns_ratio/mask_path — no `metadata`. os_image_node.cpp:45-53 likewise. OusterProcessingNodeBase (include/ouster_ros/os_processing_node_base.h:20-22) forwards NodeOptions unchanged, so no automatically_declare_parameters_from_overrides / allow_undeclared — the override is stored and never read. Both nodes get metadata only from the latched topic (os_processing_node_base.cpp:13-21) and build lidar_packets subscriptions inside metadata_handler. The parameter at ouster_standalone.launch.py:180 and :239, turtlebot3_ouster.launch.py:178/:219 and sensor_stack.launch.py:85/:117 is inert, and the 6-line rationale above each is wrong. Severity corrected down: the claimed race is already covered by the plugin's own transient_local metadata publisher (src/ros_interface.cpp:72-76) which additionally re-arms and republishes when the subscriber count drops to 0 (src/ros_interface.cpp:317-350), so nothing actually breaks — the defect is ~20 lines of confidently-wrong rationale plus dead config in three files, a maintenance hazard rather than wrong behaviour.

---

### `package.xml:54` — turtlebot3_description is a hard runtime dependency of a shipped launch file but is undeclared in package.xml

*Launch / URDF / worlds* · build

package.xml lists exec_depends for the whole example bringup — ros_gz_sim, ros_gz_bridge, xacro, robot_state_publisher, launch_ros, ament_index_python, rviz2 (lines 48-54) — but omits turtlebot3_description, which turtlebot3_ouster.launch.py:49-50 calls `get_package_share_directory('turtlebot3_description')` on at module scope, and which turtlebot3_ouster.urdf.xacro:39 pulls in via `$(find turtlebot3_description)/urdf/turtlebot3_waffle.urdf`. There is no try/except and no conditional.

Reproduced here: executing `generate_launch_description()` for the other two launch files succeeds (16 and 19 entities), while turtlebot3_ouster.launch.py raises `PackageNotFoundError: "package 'turtlebot3_description' not found"` before producing anything. The Dockerfile papers over this by git-cloning ROBOTIS/turtlebot3 and copying turtlebot3_description into the workspace (Dockerfile:138-141), so the dependency is real, known, and satisfied out-of-band by exactly one consumer — while `rosdep install --from-paths src` on this repo alone installs nothing for it.

This is the launch file the default Docker smoke path uses (docker/entrypoint.sh:27).

**Failure scenario.** A user follows the README on a plain ROS 2 Jazzy machine, runs `rosdep install --from-paths src --ignore-src -y` (which succeeds, because nothing declares the dep), builds, then runs `ros2 launch gz_sensors_ouster turtlebot3_ouster.launch.py`. It dies immediately with a PackageNotFoundError traceback from ament_index — not an actionable message about a missing TurtleBot3 description.

**Fix.** Add `<exec_depend>turtlebot3_description</exec_depend>` to package.xml, and additionally wrap the lookup in turtlebot3_ouster.launch.py:49-50 with a try/except that raises a message naming the package and pointing at the Dockerfile's source-clone recipe, so the failure is self-explaining when the dep genuinely isn't packaged for the user's distro.

**Verifier.** Confirmed. package.xml:48-54 lists exec_depends ros_gz_sim, ros_gz_bridge, xacro, robot_state_publisher, launch_ros, ament_index_python, rviz2 and nothing else; grep for 'turtlebot3' in package.xml returns nothing. turtlebot3_ouster.launch.py:49-50 calls get_package_share_directory('turtlebot3_description') unguarded at the top of generate_launch_description(), and turtlebot3_ouster.urdf.xacro:39 includes $(find turtlebot3_description)/urdf/turtlebot3_waffle.urdf. gz_sensors_ouster.repos contains only gz_sensors_ouster and ouster-ros, so the README's own workspace-setup recipe (README.md:167-176, vcs import + rosdep install) leaves the package absent, while README.md:330 tells users to run exactly this launch. Only the Dockerfile satisfies it out-of-band by cloning ROBOTIS/turtlebot3 and copying turtlebot3_description into the workspace (Dockerfile:135-141). Severity kept at medium: a documented, reproducible hard failure of a shipped launch file for any non-Docker user — though note the fix is not purely mechanical, since if turtlebot3_description is not rosdep-resolvable on the target distro, adding the exec_depend would make `rosdep install` fail for users who never touch the TurtleBot example.

---

### `cuda/ray_processor_cpu_impl.cpp:141` — Apparent reflectance of exactly 0 is treated as "no retro data": a perfectly black target returns full signal and reflectivity 50

*Raycast math* · correctness · reported as high, downgraded by verifier

The channel stage tests `retro_host[idx] > 0.0f` to decide whether the raycast supplied a reflectance at all:
- line 141-143: `intensity = 1.0f;` unless `retro_host[idx] > 0.0f` -> a zero reflectance produces `signalFromRange(d, 1.0, base_signal)`, i.e. the signal of a *perfect white Lambertian*.
- line 168-172: same test -> `reflectivity_out[idx] = base_reflectivity` (50) instead of `reflectivityToByte(0)` = 0.
- `rpmath::retroForNoise` (cuda/ray_processor_math.hpp:291) has the identical `r > 0.0f` test, so the dropout multiplier and range-noise multiplier also snap back to `kDefaultRetro = 0.5` instead of the maximum dark-target penalty.

But zero is a *meaningful, explicitly supported* value here. `Scene::addInstance` documents it (cuda/raycast_scene.hpp:104): "Set `has_retro` false only when the SDF omitted laser_retro; this keeps omission distinct from an explicitly authored zero during ray casting", and rcCastOneRay repeats it (raycast_math.hpp:1455): "an explicitly authored zero remains zero". The obscurant model relies on the same channel: raycast_math.hpp:1256 claims the `rho *= trans_2way` multiply "correctly dims SIGNAL, lowers the calibrated REFLECTIVITY byte" — but `exp(-2*tau_eff)` underflows to exactly 0.0f in float32 once `tau_eff > ~52` (reachable: sigma = 3.912/visibility, so 1 m visibility over a 15 m path gives tau ~ 58), at which point the dimming inverts to maximum brightness.

Measured, box at 5 m, base_signal 800, base_reflectivity 50, noise off:
  laser_retro 0.00 -> signal 39, reflectivity byte 50
  laser_retro 0.02 -> signal  0, reflectivity byte  2
  laser_retro 1.00 -> signal 39, reflectivity byte 100
A rho of 0 is non-monotonically brighter than a rho of 0.02 and indistinguishable from a rho of 1.0.

All four backends carry the same test: ray_processor_cuda.cu:310,323; ray_processor_hip.cpp:263,274; ray_processor_sycl.cpp:486,498.

**Failure scenario.** Author a visual with `<laser_retro>0</laser_retro>` and no `<specular>` (a perfect absorber) at 5 m in raycast mode. castScan correctly emits retro = 0.0. processCpu then reports SIGNAL 39 and REFLECTIVITY 50 — identical to a laser_retro 1.0 white target and brighter than the laser_retro 0.02 target next to it. The same inversion fires for any hard target behind fog dense enough for exp(-2*tau_eff) to underflow.

**Fix.** Distinguish "no retro channel" from "measured zero". The nullptr check already exists; drop the `> 0.0f` clause from the value test at ray_processor_cpu_impl.cpp:141 and :168 (keep `isfinite`) so rho = 0 flows through as intensity 0 -> signal 0 and reflectivityToByte(0) = 0. Give `rpmath::retroForNoise` (ray_processor_math.hpp:287-294) the same treatment: return kDefaultRetro only when `retro == nullptr` or the value is non-finite, not when it is 0. Apply the identical edit to the three GPU kernels listed above, and add a test asserting signal(rho=0) < signal(rho=0.02) < signal(rho=1).

**Verifier.** Cited lines say what the finding claims: ray_processor_cpu_impl.cpp:140-143 gates intensity on `retro_host && std::isfinite(...) && retro_host[idx] > 0.0f` (else intensity stays 1.0f from line 139), ray_processor_cpu_impl.cpp:168-172 does the same for reflectivityToByte vs base_reflectivity, and rpmath::retroForNoise (ray_processor_math.hpp:287-294) repeats `r > 0.0f` before falling back to kDefaultRetro. The null-pointer case is already covered separately, so the `> 0.0f` clause is doing extra work: ray_processor_cpu_impl.cpp:229-231 documents that panel mode passes a NULL retro buffer ('The depth-panel rig carries no laser_retro channel; passing a null retro buffer...'), so the only caller with a non-null retro buffer is the raycast path via Backend::castScanProcessed (backend.hpp:103-108, which feeds retro_scratch straight into processDepth). In that path 0 is unambiguous: raycast_mirror.cpp:273-275 sets has_retro from `lr != nullptr`, rcMaterialAtHit resolves an omitted laser_retro to fallback_retro (raycast_math.hpp:681), and raycast_scene.hpp:100-105 states the whole point is 'keeps omission distinct from an explicitly authored zero'. So an authored <laser_retro>0</laser_retro> with no <specular> yields rho exactly 0 at a finite range and is then read out as intensity 1.0 and reflectivity byte 50 — brighter than the same object at laser_retro 0.02. Note the reviewer's proposed fix is over-broad (dropping `> 0.0f` unconditionally is fine only because panel mode passes nullptr; that dependency must be kept). Severity corrected from high to medium: it needs a deliberately authored zero (no shipped world does it — grep of examples/ finds no laser_retro 0), and the fog-underflow variant needs exp(-2*tau_eff) to underflow float32, i.e. tau_eff > ~52, a regime where the point is already at the noise floor.

---

### `src/raycast_mirror.cpp:194` — raycast_mirror.cpp's SDF→rc::Scene geometry extraction has zero test coverage

*Test quality* · test-coverage

`RaycastMirror::rebuildScene` converts every `<visual>` in the world into an `rc::Scene` instance, and the unit conventions are all hand-written half-extent divisions: box `Size().X()/2.0` (lines 194-196), cylinder radius + `Length()/2.0` (lines 206-208), plane `Size().X()/2.0` plus a `SetFrom2Axes` quaternion folded into `ref.offset` to rotate the SDF normal onto local +z (lines 213-220). Nothing in test/ exercises any of it — grep for `raycast_mirror` across test/ hits only test_worlds.py, and that only reads XML tag names. test_raycast.cpp builds `rc::Scene` by calling `scene.addInstance(...)` directly with hand-written sizes (e.g. `cyl_size = {0.5f, 1.0f}` at line 179), so it validates the kernel's interpretation of the convention but never that the mirror produces it. test_lifecycle.cpp links the plugin library but only default-constructs and destructs it (lines 53-71, both bodies end in `SUCCEED()`).

**Failure scenario.** Drop the `/ 2.0` on line 194-196 (or on the cylinder's `Length() / 2.0` at line 208). Every box/cylinder visual in every world becomes twice its true size to the lidar while rendering correctly in the GUI — walls thicken by a metre, the turtlebot's own chassis occludes beams, ranges are systematically short. The entire suite stays green: test_raycast never calls rebuildScene, test_worlds.py only checks that the geometry tag is in {box, sphere, cylinder, plane, mesh} (test_worlds.py:88, 177). The plane-normal quaternion at 217-220 has the same exposure — inverting the two axes in `SetFrom2Axes` flips every ground plane's normal, which changes `rcCosIncidence` and therefore every ground return's reflectivity, undetected.

**Fix.** Add a gtest that builds an `EntityComponentManager` with Visual+Geometry+Pose components (test_obscurant_config.cpp:473-494 already demonstrates this pattern for ParticleEmitter), runs the scene rebuild, and casts a known ray through the result — e.g. a 2×2×2 `<box>` at x=5 must return range 4.0, a `<cylinder>` with `<length>2</length>` must have its cap at ±1, a `<plane>` with `<normal>1 0 0</normal>` must be hit face-on by a +x beam. That pins every half-extent conversion and the normal quaternion in one test.

**Verifier.** Confirmed, including the exact line numbers. Box half-extents at src/raycast_mirror.cpp:194-196 (`box.X()/2.0`, `.Y()`, `.Z()`), cylinder radius + `Length()/2.0` at :205-208, plane `Size().X()/2.0` at :213-214 and the SetFrom2Axes normal folded into ref.offset at :217-220. `grep -rn 'raycast_mirror|RaycastMirror|rebuildScene' test/ CMakeLists.txt` returns exactly two hits: a comment in test_worlds.py:84 and the source listing at CMakeLists.txt:277 — no test constructs a RaycastMirror or calls rebuildScene. test_lifecycle.cpp only default-constructs and destructs the plugin (:53-71, both bodies end in SUCCEED()), and test_worlds.py:168-184 only checks the geometry tag name is in MIRRORED_GEOMETRY, so halving/doubling a half-extent or inverting the plane-normal quaternion is invisible to the whole suite while test_raycast.cpp validates only the kernel's side of the same convention with hand-written sizes.

---

### `test/test_metadata_parsing.cpp:10` — test_metadata_parsing.cpp tests the vendored Ouster SDK, not src/ouster_metadata.cpp

*Test quality* · test-coverage · reported as high, downgraded by verifier

The includes at lines 10-12 are `<ouster/metadata.h>`, `<ouster/impl/packet_writer.h>`, `<ouster/types.h>` — nothing from this repo. Every TEST_P constructs `ouster::sdk::core::SensorInfo(json)` and `PacketWriter(pf)` directly. So the 50 parameterised assertions verify that Ouster's own SDK can parse Ouster's own JSON, and that `beam_altitude_angles.size() > 0` (line 41), `columns_per_frame > 0` (line 43), `pixels_per_column > 0` (line 56) etc.

The repo's actual metadata code, `OusterMetadata::load()` (src/ouster_metadata.cpp:25, declared src/ouster_metadata.hpp:34), has zero coverage anywhere in test/. It is what derives `H`, `W`, `cpp`, `beam_alt_f`/`beam_az_f` (the float copies padded to H that are uploaded to the GPU), `beam_origin_mm`, `min_alt`/`max_alt`/`v_range` (which drive the panel rig band), resolves `profile` via resolveOusterLidarProfile, performs the WINDOW-field firmware bump on `metadata_str`, and mutates the caller's `max_range` when `max_range_explicit` is false.

**Failure scenario.** Swap the two lines that fill `beam_alt_f` and `beam_az_f` in OusterMetadata::load, or drop the `beam_origin_mm` unit conversion (mm vs m). Every beam in every scan is then mis-aimed / every point is reported ~1.4-2.8 cm off, and the whole test suite stays green: test_metadata_parsing never calls load(), test_ouster_lidar_profile builds `OusterProfileRequest` by hand (test_ouster_lidar_profile.cpp:17-25) rather than from a JSON file, and test_raycast/test_resample supply beam tables by hand.

**Fix.** Add a test that constructs `OusterMetadata` and calls `load()` on each of the ten JSONs already under config/metadata/ (TEST_METADATA_DIR is already plumbed at CMakeLists.txt:332), asserting H/W/cpp against the JSON, `beam_alt_f.size() == H`, `beam_alt_f[i] == beam_alt_angles[i]`, `beam_origin_mm` against `lidar_origin_to_beam_origin_mm`, `profile.supported == true`, and that `max_range` is left alone when `max_range_explicit` is true and overwritten from the profile when it is false. Also add a malformed-JSON case that asserts load() returns false rather than throwing.

**Verifier.** Confirmed. test_metadata_parsing.cpp includes only <ouster/metadata.h>, <ouster/impl/packet_writer.h>, <ouster/types.h> (lines 10-12) and every TEST_P constructs ouster::sdk::core::SensorInfo/PacketFormat/PacketWriter directly (38-39, 52-54, 71, 95, 112-114). `grep -rn 'OusterMetadata|ouster_metadata' test/ CMakeLists.txt` returns exactly one hit — CMakeLists.txt:274, the source listing — so OusterMetadata::load (src/ouster_metadata.cpp:25-238) has zero test callers. That is a lot of untested logic: the H/W/cpp bounds rejection (:75-87), the profile resolution and Unknown/unsupported rejection (:124-139), the `if (!max_range_explicit) max_range = profile.representable_range_m` mutation of the caller's parameter (:148), the min_alt/max_alt/v_range computation (:216-225), the beam_alt_f/beam_az_f float copies and pad-to-H (:229-235), and the try/catch at :203-206 that converts an SDK throw into `return false` — the three malformed-JSON tests at lines 150-165 assert the SDK throws, never that load() returns false. One detail in the finding is wrong: load() stores `beam_origin_mm = info.lidar_origin_to_beam_origin_mm` verbatim (:160), there is no mm→m conversion in load() to delete; that half of the failure scenario does not apply. The rest stands. Severity corrected: coverage gap, no defect in shipped code.

---

### `test/test_obscurants.cpp:165` — FlatEmitterVolumeStillHasThickness only exercises the one span branch that contains no division

*Test quality* · test-coverage

The test's comment (lines 166-168) states the hazard: "gz particle emitters are routinely authored flat (<size>10 10 0</size>). A zero half-extent must not divide by zero or swallow the ray." But line 169 builds a `rc::ObscurantType::kBox`, and the box branch of `rcObscurantSpan` (cuda/raycast_math.hpp:890-895) is three `rcSlabAxis` calls with no division by hx/hy/hz at all. With hz=0 and a ray at z=0 travelling along +x, `rcSlabAxis(0, 0, -0, 0, ...)` (raycast_math.hpp:278-280) returns `0 >= 0 && 0 <= 0` → true, entirely independently of the clamp.

The divisions the clamp exists for are in the ellipsoid branch (`o_l.x / hx`, raycast_math.hpp:897-898) and the cylinder branch (`o_l.x / hx`, `o_l.y / hy`, :911-912) — plus `rcObscurantContains` (:948, :952). Neither is ever tested with a zero half-extent. Note the flat case reaches those branches in production: `obscurantFromEmitter` (src/obscurants.cpp:225, 238-240) maps gz's default ELLIPSOID emitter type to `kEllipsoid`, and the `half` guard at :242 only rejects a volume when *all three* extents are non-positive, so `<size>10 10 0</size>` with `particle_growth=0` yields half = (5, 5, 0).

**Failure scenario.** Delete the three `rpmath::gzm::fmax_(ob.half[i], kRcObscurantMinHalf)` clamps at raycast_math.hpp:884-886. `ObscurantGeometry.FlatEmitterVolumeStillHasThickness` still passes (box branch, no division). A world with a flat ELLIPSOID particle emitter then computes `os.z = o_l.z / 0.0f` and `ds.z = d_l.z / 0.0f` at :897-898 → ±inf or NaN → `qa` is NaN, `qa < 1.0e-20f` is false, `disc` is NaN, `disc < 0.0f` is false, and `lo`/`hi` become NaN. Every beam through that cloud gets a NaN optical depth, and `tau`/`tau_eff` (:983-985) poison the retro for the whole ray.

**Fix.** Parameterise the test over all three ObscurantTypes with the zero half-extent (the file already does this in `EllipsoidAndCylinderVolumesAttenuateToo`, line 869), and assert the span thickness explicitly — `EXPECT_NEAR(b, 12.0f, 1e-4f)` alongside the existing `EXPECT_NEAR(a, 8.0f)` — rather than only `isfinite(b)`. Add the same three-type sweep for `rcObscurantContains`.

**Verifier.** Confirmed on every point. The test builds rc::ObscurantType::kBox with hz=0 (test_obscurants.cpp:169-170) and casts from kOrigin{0,0,0} along kForward{1,0,0} (:112-113). The box branch (cuda/raycast_math.hpp:888-893) is three rcSlabAxis calls with no division by hx/hy/hz; with d_l.z==0 rcSlabAxis takes the parallel early-out `return o >= bmin && o <= bmax` (:277-279) → 0 >= -0 && 0 <= 0 → true, so deleting the three kRcObscurantMinHalf clamps at :884-886 leaves the test green. The divisions the clamp protects are ellipsoid `o_l.x/hx` etc. (:897-898) and cylinder (:911-912), plus rcObscurantContains (:939-941, :948, :952) — and every ellipsoid/cylinder test in the file uses non-zero half-extents (:133-134, :149-150, :181-182, :869-872). Production reachability also checks out: obscurantFromEmitter defaults to rc::ObscurantType::kEllipsoid (src/obscurants.cpp:224) and only the default branch is taken for gz's ELLIPSOID emitters (:238-239), while the reject guard at :241 requires ALL THREE extents non-positive, so <size>10 10 0</size> with growth 0 yields half=(5,5,0) and reaches the divide. Kept at medium: the guard is present and correct today, so this is a mis-aimed test, not a live bug.

---

### `test/test_parameter_validation.cpp:16` — test_parameter_validation.cpp validates a private copy of Configure()'s clamps, not Configure()

*Test quality* · test-coverage · reported as high, downgraded by verifier

Lines 16-45 define `struct ValidatedParams { ... void validate() }` whose comment says "Mirrors the validation logic from GzGpuOusterLidarSystem::Configure()". Every one of the nine TESTs in the file then calls `p.validate()` on that local struct. The file never includes the plugin header, never constructs the plugin, and never touches `src/gz_gpu_ouster_lidar_system.cpp`. It is a test of `std::max` and `std::clamp`.

The mirror has ALREADY drifted from the thing it claims to mirror:
- Production `clamp_warn` (gz_gpu_ouster_lidar_system.cpp:291-306) has an explicit `std::isnan(v)` branch whose comment says "NaN compares false against everything ... std::clamp would pass it through". The test's mirror at line 36 is exactly the buggy form: `dropout_rate_close = std::clamp(dropout_rate_close, 0.0, 1.0)`. `std::clamp(NaN, 0.0, 1.0)` returns NaN. The test file therefore encodes the pre-fix behaviour, and `NonFiniteSensorRatesResetToDefaults` (line 151) only feeds NaN to `lidar_hz`/`imu_hz`, never to a clamped rate.
- Production clamps six parameters the mirror does not even have a field for: `false_alarm_rate_` [0,1] (:314), `min_range_` (:319), `detection_rolloff_` [0.01,1.0] (:320), `gyro_bias_walk_`/`accel_bias_walk_` (:323-324), `panel_oversample_` [1,4] (:325).

**Failure scenario.** Delete line 317 (`clamp_warn(base_reflectivity_, 0.0, 255.0, "base_reflectivity")`) from src/gz_gpu_ouster_lidar_system.cpp. A world authoring `<base_reflectivity>300</base_reflectivity>` now reaches line 460, `reflectivity_buf_.resize(n, static_cast<uint8_t>(base_reflectivity_))` — an out-of-range double→uint8_t conversion (UB; in practice 44) — and line 967 `pp.base_reflectivity = static_cast<float>(...)`, so every miss pixel ships REFLECTIVITY byte 44 instead of 255. All nine tests in test_parameter_validation.cpp still pass, because they only exercise the copy. Same for deleting the `panel_oversample_` clamp: `panel_oversample=0` reaches buildLayout at :473 with a degenerate rig.

**Fix.** Extract the clamping block (gz_gpu_ouster_lidar_system.cpp:288-356) into a free function taking a plain params struct — e.g. `validateNoiseParams(LidarParams&)` in a header — call it from Configure(), and point the test at that symbol. Then add cases for the parameters currently missing from the mirror (false_alarm_rate, min_range, detection_rolloff, panel_oversample) and a NaN case for dropout_rate_close/far that would fail against the `std::clamp` form.

**Verifier.** Confirmed by reading both files. test_parameter_validation.cpp:16-45 defines a local `ValidatedParams::validate()` explicitly commented "Mirrors the validation logic from GzGpuOusterLidarSystem::Configure()"; all nine TESTs (lines 47-176) call only `p.validate()`, and the file includes only <gtest>, <algorithm>, <cmath>, <limits> — nothing from src/ or include/. The drift claim is also accurate: production `clamp_warn` at src/gz_gpu_ouster_lidar_system.cpp:291-306 has the explicit `std::isnan(v)` branch (295-299) that sanitises NaN to `lo`, whereas the mirror at test line 36-37 uses bare `std::clamp`, which passes NaN through — the mirror encodes the pre-fix behaviour, and the only NaN test (line 151-159) feeds lidar_hz/imu_hz, never a clamped rate. Production clamps six params with no mirror field: false_alarm_rate_ (:314), min_range_ (:319), detection_rolloff_ (:320), gyro_bias_walk_/accel_bias_walk_ (:323-324), panel_oversample_ (:325). The mutation is real: deleting :317 leaves base_reflectivity_=300 flowing into `reflectivity_buf_.resize(n, static_cast<uint8_t>(base_reflectivity_))` at :459-460 and `pp.base_reflectivity` at :967, with all nine tests still green. Severity corrected down: the shipping code is correct today; this is a coverage/maintenance defect, not wrong behaviour in the field.

---

### `test/test_resample.cpp:406` — BilinearAveragesOnlyValidPixels passes two identical values, so "average" is unobservable

*Test quality* · test-coverage

The call at lines 406-410 is `bilinearOrAverage(40.0f, 40.0f, inf, inf, 0.5f, 0.5f, true, true, false, false, 2, inf)` and the assertion at line 413 is `EXPECT_NEAR(result, 40.0f, 0.01f)`. Both valid samples are 40, so mean, max, min, first-valid and last-valid all produce 40. The test name claims the function averages the valid pixels; the inputs make averaging unobservable.

The partial-validity branch (cuda/ray_processor_math.hpp:123-128, `sum / n_valid`) is the silhouette-edge path. The only other test of it, `BilinearOneValidPixelReturnsThatPixel` (line 416), uses n_valid=1 where every candidate rule also collapses to the same answer. n_valid==3 is never exercised at all, and the n_valid==4 bilinear branch is only reached via `NearestSamplingDoesNotBlendAcrossEdges` (line 192) with a scene where d00==d10 and d01==d11, so even swapping `h_alpha` and `v_alpha` at ray_processor_math.hpp:119-121 is invisible there.

**Failure scenario.** Replace `return sum / static_cast<float>(n_valid);` (ray_processor_math.hpp:128) with a max-of-valid reduction. Both bilinear tests pass. In a real scene, a beam straddling a silhouette with three valid neighbours at 5 m, 5 m, 50 m now reports 50 m instead of 20 m — a point floating 30 m behind the object, which is precisely the artifact `NearestSamplingDoesNotBlendAcrossEdges` exists to guard against.

**Fix.** Use distinct values: `bilinearOrAverage(10, 50, inf, inf, 0.5, 0.5, true, true, false, false, 2, inf)` must be exactly 30.0 (a max/first/last mutation gives 50/10/50 and fails). Add an n_valid==3 case (e.g. 10, 20, 60, inf → 30.0), and add an n_valid==4 case with four distinct corners and asymmetric alphas (h_alpha=0.25, v_alpha=0.75) so an h/v swap in the bilinear branch is caught.

**Verifier.** Confirmed. test_resample.cpp:406-413 passes a00=a01=40.0f with n_valid=2 and asserts EXPECT_NEAR(result, 40.0f, 0.01f) — mean, max, min, first-valid and last-valid all return 40, so replacing `sum / static_cast<float>(n_valid)` (cuda/ray_processor_math.hpp:128) with any reduction over the valid set passes. The only other partial-validity test, BilinearOneValidPixelReturnsThatPixel (:416-428), uses n_valid=1, which collapses identically. I checked the rest of the file: the only other bilinearOrAverage callers are through sampleBeamRange, and AllInfProducesZeroRange (:238) / FarClipReadsAsMiss (:263) hit the n_valid==0 early return at ray_processor_math.hpp:117, while NearestSamplingDoesNotBlendAcrossEdges (:165-203) builds a scene that is constant per column (raw[v*w+u] depends only on u, lines 183-187), so a00==a10 and a01==a11 and an h_alpha/v_alpha swap at :118-121 is invisible there. n_valid==3 is never exercised anywhere. The proposed distinct-value inputs are the correct fix.

---

## LOW (26)

### `test/test_raycast.cpp:770` — FusedCastAndProcessMatchesTwoStage test forces the CPU backend and zero noise, making it f(x)==f(x)

*Backend divergence* · test-coverage · reported as medium, downgraded by verifier

`TEST(Raycast, FusedCastAndProcessMatchesTwoStageCpuPath)` is the only test guarding the fused-vs-two-stage equivalence, and it cannot fail for the reason it exists.

1. Line 772 pins `::setenv("GZ_OUSTER_BACKEND", "cpu", 1)`. `CpuBackend` (backend_cpu.cpp:17-99) does **not** override `castScanProcessed`, so `RayProcessor::castScanProcessed` resolves to the three-line default at backend.hpp:102-109, which literally calls `castScan(...)` then `processDepth(...)`. The `two_stage` arm (lines 798-806) calls those same two methods by hand. The assertions at lines 817-820 compare the default forwarder against a manual inline of the default forwarder.
2. CUDA is the *only* backend that overrides `castScanProcessed` (ray_processor_cuda.cu:643-684, keeping depth/retro/NIR device-resident and skipping the host round-trip). That override — the actual divergence risk — is excluded by construction by the setenv on line 772, and is never compiled in CI regardless.
3. `RayProcessParams pp{}` (line 785) sets only `H`, `W`, `base_signal`, `base_reflectivity`, `max_range`. Every noise term is therefore 0, so `noiseEnabled(pp)` (backend.hpp:126-133) is false and neither arm draws a single random number. The `{123u}` seeds on lines 798 and 809 are decorative — the test passes with any seeds, and would pass if the two paths consumed RNG draws in completely different orders.

Secondary: line 772 and line 698 `setenv` without ever restoring, unlike the `BackendEnvGuard` RAII pattern in test_dispatch.cpp:28-56, so `Raycast.ProcessDepthThroughRayProcessor` (line 731, which does not setenv) silently inherits whatever the previously-run test left behind.

**Failure scenario.** Someone adds a HIP or SYCL `castScanProcessed` override (the TODO is written into raycast_mirror.cpp:672-675: "Other backends use the exact fallback composition through scratch_ until they grow a native fused override") and gets the NIR plane or the retro plumbing wrong — e.g. passes `nullptr` for `nir_in` so the fused path falls back to the legacy retro-based near-IR at ray_processor_hip.cpp:274-275 while the two-stage path uses the raycast ambient factor. Every NEAR_IR pixel differs between the two entry points, and this test stays green because it never runs that backend. Equally, a fused CPU override that consumed a different number of RNG draws would pass, because no draws happen at all.

**Fix.** Enable at least one noise term (e.g. `pp.range_noise_min_std = 0.02f; pp.dropout_rate_far = 0.05f;`) so the seeded RNG streams are actually compared, and drive the test over every compiled-in backend rather than pinning "cpu" — parameterise on the `GZ_OUSTER_BACKEND` values whose factory returns non-null, using the `BackendEnvGuard` from test_dispatch.cpp:28 so the env var does not leak into `Raycast.ProcessDepthThroughRayProcessor` at line 731.

**Verifier.** Confirmed, with two corrections. Verified: test_raycast.cpp:770 is TEST(Raycast, FusedCastAndProcessMatchesTwoStageCpuPath) and :772 is ::setenv("GZ_OUSTER_BACKEND", "cpu", 1). CpuBackend (backend_cpu.cpp:17-99) overrides only processRaw/processDepth/castScan/name — no castScanProcessed — so the fused arm at :810 resolves to the default at backend.hpp:102-109, which is literally castScan(...) then processDepth(...), i.e. the same two calls the two_stage arm makes by hand at :798-806. castScanProcessed is overridden in exactly one place in the repo, ray_processor_cuda.cu:643 (grep confirms HIP and SYCL do not override it, matching the TODO comment at raycast_mirror.cpp:672-675), and the setenv excludes it. RayProcessParams pp{} at :785 sets only H, W, base_signal, base_reflectivity, max_range, so noiseEnabled(pp) (backend.hpp:126-133) is false and no RNG is drawn in either arm — the {123u} seeds at :798 and :809 are indeed decorative. Corrections to the reviewer: (a) the test is not literally worthless — it does pin the scratch-pointer plumbing through ray_processor_dispatch.cpp:170-174 and backend.hpp:103-108, so swapping retro_scratch/nir_scratch there would be caught; (b) the proposed fix does not work. Enabling a noise term would still not make the arms distinguishable on CPU, because CpuBackend::frameSeed() (backend_cpu.cpp:85-94) derives the seed from frame_counter_, which is 0 for the first processDepth call on each of the two freshly-constructed RayProcessor objects, and processCpu re-seeds a call-local mt19937 from it (ray_processor_cpu_impl.cpp:36-38). Both arms would still be bit-identical by construction. The only real fix is parameterising over backends whose factory returns non-null. Test-quality gap with no shipped-data impact and no override yet existing to diverge from.

---

### `.github/workflows/ci.yaml:22` — SPDX license gate reports success when `find` fails outright

*Build & CI* · build

`for f in $(find src include cuda -type f \( -name '*.cpp' ... \)); do`. The exit status of a command substitution in a `for` word list is discarded — `set -e` (GitHub's default `bash -e {0}` for this unshelled step) does not apply. I verified this: with all three directories absent, `find` prints three "No such file or directory" errors to stderr, the loop body never executes, `$missing` stays empty, and the script prints "All source files have SPDX headers." and exits 0. The step also only inspects `head -2` of each file and never covers `test/` (all 21 test files do currently carry SPDX headers, so that part is latent rather than broken).

**Failure scenario.** Someone reorganises `cuda/` into `backends/` (a plausible follow-up given cuda/CMakeLists.txt:1-3 already notes the directory name is a misnomer) and forgets to update line 22. The SPDX gate goes green forever afterwards while silently checking nothing, and license headers can be dropped from every new file without CI noticing.

**Fix.** Replace the `for` loop with `find src include cuda test -type f \( ... \) -print0 | xargs -0r grep -L 'SPDX-License-Identifier'` and check the output, or simply prepend `set -euo pipefail` and assign the file list with `files=$(find ...)` on its own line so a find failure aborts the step. Add `test/` to the search roots while you are in there.

**Verifier.** Confirmed by reading and by execution. ci.yaml:22 is 'for f in $(find src include cuda -type f ...)'; the exit status of the command substitution in the for-word-list is discarded. I ran the identical loop under 'bash -e' with three nonexistent directories: find printed three 'No such file or directory' errors to stderr, the body never executed, missing stayed empty, and the script printed 'All source files have SPDX headers.' with EXIT=0. The secondary claims also hold: the search roots at line 22 omit test/, and I verified every file under test/ currently carries an SPDX header in its first two lines, so that part is latent rather than presently broken. Low is correct — it needs a future directory rename to bite.

---

### `Dockerfile:178` — Dockerfile test gate misreports a real test failure as 'no tests ran'

*Build & CI* · build

The gate is one `&&`/`||` chain:
  RUN source ... && source install/setup.bash && colcon test ... && colcon test-result --verbose --all && n="$(...)" && { [ -n "$n" ] && [ "$n" -gt 0 ]; } || { echo "ERROR: no tests ran (BUILD_TESTING wiring broken)"; exit 1; } && echo "Executed $n test(s)."
Shell precedence groups this as `(((((A && B) && C) && D) && E) && F) || G`, so `G` fires on failure of *any* link in the chain, not just the zero-test check `F`. `colcon test-result --verbose --all` at line 176 exits non-zero whenever a test actually fails, which drops straight into `G`. The build does correctly fail (good), but the only message the operator sees is a flat lie about the cause. The CI equivalent at ci.yaml:121-140 does not have this problem — it uses separate statements, so `colcon test-result` failing produces its own diagnostic.

**Failure scenario.** `test_obscurants` starts failing. `docker build` aborts printing "ERROR: no tests ran (BUILD_TESTING wiring broken)". The operator spends the next half hour auditing `if(BUILD_TESTING)` in CMakeLists.txt:305 and the `-DBUILD_TESTING=ON` flag at Dockerfile:169, none of which are wrong, because the message points at the one thing that is definitely fine.

**Fix.** Split the gate into separate statements the way ci.yaml:121-140 does, e.g. `RUN set -e; source ...; colcon test ...; colcon test-result --verbose --all; n=$(colcon test-result --all | awk '/^Summary:/ {print $2+0; exit}'); if [ -z "$n" ] || [ "$n" -eq 0 ]; then echo "ERROR: no tests ran"; exit 1; fi; echo "Executed $n test(s)."` so the zero-test message can only be emitted by the zero-test condition.

**Verifier.** Confirmed. Dockerfile:173-179 is a single && chain: source, source, 'colcon test' (:175), 'colcon test-result --verbose --all' (:176), the n= assignment (:177), then '{ [ -n "$n" ] && [ "$n" -gt 0 ]; } || { echo "ERROR: no tests ran (BUILD_TESTING wiring broken)"; exit 1; }' at :178. && and || are equal-precedence left-associative, so the || group is the fallback for the entire preceding chain, not just the zero-test test. A genuine test failure makes 'colcon test-result --verbose --all' at :176 exit non-zero and the operator is told the BUILD_TESTING wiring is broken — pointing at Dockerfile:169 (-DBUILD_TESTING=ON) and CMakeLists.txt:305 (if(BUILD_TESTING)), both of which I confirmed are correct. The contrast with ci.yaml:121-140 is accurate: that version uses separate statements so test-result failure surfaces on its own. Build still fails, only the diagnostic lies — low is right.

---

### `package.xml:27` — gz-msgs is used in a header but declared nowhere: not in package.xml, not find_package'd, not linked

*Build & CI* · build · reported as medium, downgraded by verifier

`src/obscurants.hpp:26` does `#include <gz/msgs/particle_emitter.pb.h>`, and `src/obscurants.cpp` is a source of the main plugin target (CMakeLists.txt:272) and is compiled a second time into `test_obscurant_config` (CMakeLists.txt:364-365). Yet:
  - package.xml declares gz_sim_vendor, gz_common_vendor, gz_rendering_vendor, gz_sensors_vendor, gz_plugin_vendor (lines 23-27) and no gz_msgs_vendor;
  - CMakeLists.txt:44-48 find_package()es gz-sim, gz-common, gz-rendering, gz-sensors, gz-plugin and never gz-msgs;
  - target_link_libraries at CMakeLists.txt:293-302 lists no gz-msgs target.
The generated protobuf header resolves purely through `gz-sim::gz-sim`'s INTERFACE_INCLUDE_DIRECTORIES. The smoking gun is CMakeLists.txt:366-367, where `test_obscurant_config` has to link `gz-sim::gz-sim` for no reason other than to drag in headers it never calls gz-sim for — the dependency is real, it is just laundered through the wrong target. The same applies to `<gz/math/...>` (9 include sites across src/include) with no gz_math_vendor depend and no gz-math find_package/link. CI at ci.yaml:94-99 likewise apt-installs only the five vendor packages and relies on transitive apt pulls.

**Failure scenario.** Gazebo Jetty (gz-sim 10, the `lyrical` matrix leg at ci.yaml:50) demotes gz-msgs from a PUBLIC to a PRIVATE dependency of gz-sim — a normal thing for an upstream to do at a major bump. The transitive include path disappears and the build fails with `fatal error: gz/msgs/particle_emitter.pb.h: No such file or directory` in both the plugin and test_obscurant_config, with nothing in package.xml or CMakeLists to point at the cause. Because that leg is `continue-on-error` (ci.yaml:51), the failure is also non-blocking and easy to ignore until Jetty becomes the default.

**Fix.** Add `<depend>gz_msgs_vendor</depend>` and `<depend>gz_math_vendor</depend>` to package.xml alongside the other vendor shims, add `find_package(gz-msgs REQUIRED)` and `find_package(gz-math REQUIRED)` next to CMakeLists.txt:44-48, and add `gz-msgs::gz-msgs` / `gz-math::gz-math` to the link list at CMakeLists.txt:293-302 (and to test_obscurant_config, replacing the incidental gz-sim link at line 367). Also add `ros-${ROS_DISTRO}-gz-msgs-vendor` to the apt list at ci.yaml:94-99 so the CI install matches the declaration.

**Verifier.** The facts check out but the failure scenario is speculative, so this is low, not medium. Confirmed: src/obscurants.hpp:26 includes <gz/msgs/particle_emitter.pb.h> (the only gz/msgs include in the tree); package.xml:23-27 lists gz_sim/gz_common/gz_rendering/gz_sensors/gz_plugin vendors and no gz_msgs_vendor or gz_math_vendor; grep for 'gz-msgs|gz-math|gz_msgs|gz_math' across CMakeLists.txt and cuda/CMakeLists.txt returns nothing, so neither is find_package'd nor in the link list at CMakeLists.txt:293-302; gz/math is included from 6 files including the public header include/gz_gpu_ouster_lidar/gz_gpu_ouster_lidar_system.hpp. So the undeclared-direct-dependency violation is real. But the stated trigger — gz-sim 10 demoting gz-msgs to PRIVATE — cannot happen: gz-sim's own public API (EntityComponentManager, the components) exposes gz::msgs and gz::math types, so those include dirs are structurally PUBLIC forever. The reviewer's 'smoking gun' is also wrong: test_obscurant_config links gz-sim::gz-sim because src/obscurants.cpp genuinely uses it — ::gz::sim::EntityComponentManager at obscurants.cpp:269, ecm.Each<::gz::sim::components::ParticleEmitter> at :281-284, ::gz::sim::worldPose at :284. Also a small factual slip: ci.yaml:94-99 installs four vendor packages plus libeigen3-dev, not five (gz-common-vendor is absent too).

---

### `README.md:436` — README, four launch files and three xacros still describe panels mode as a "GpuRays rig"; GpuRays does not appear anywhere in the code

*Cleanup* · cleanup · reported as medium, downgraded by verifier

grep for `GpuRays|gpu_rays|cubemap` over src/, cuda/ and include/ returns zero hits in any implementation file. Panels mode is a rig of `gz::rendering::DepthCamera` (src/panel_rig.hpp:21 `#include <gz/rendering/DepthCamera.hh>`, :68 `std::vector<DepthCameraPtr> cams_`, src/panel_rig.cpp:98 `scene->CreateDepthCamera(...)`), and include/gz_gpu_ouster_lidar/ray_processor.hpp:35-37 says so explicitly: "Replaces the former GpuRays-cubemap equirect grid." Yet nine user-facing places still say the opposite: README.md:436 "the plugin drives a GpuRays rig off events::Render"; examples/urdf/ouster_macro.xacro:19 and :83 (same sentence, twice, in the macro users copy); examples/urdf/turtlebot3_ouster.urdf.xacro:19 and :30; examples/launch/sensor_stack.launch.py:10 and :191; examples/launch/ouster_standalone.launch.py:10 and :108; examples/launch/turtlebot3_ouster.launch.py:102. test/test_lifecycle.cpp:56 even documents a member `gpu_rays_` that no longer exists.

**Failure scenario.** A user hits a panels-mode rendering artifact and searches gz-rendering's GpuRays/cubemap issue tracker and source, because every document the package ships told them that is the code path. They are debugging a subsystem the plugin stopped using; the actual code is a nine-panel perspective DepthCamera rig with its own layout logic in cuda/panel_layout.cpp.

**Fix.** Replace "GpuRays rig" with "perspective DepthCamera panel rig" at README.md:436, ouster_macro.xacro:19,83, turtlebot3_ouster.urdf.xacro:19,30, sensor_stack.launch.py:10,191, ouster_standalone.launch.py:10,108, turtlebot3_ouster.launch.py:102, and drop the stale `gpu_rays_` mention at test_lifecycle.cpp:56.

**Verifier.** Confirmed, and undercounted. grep for GpuRays|gpu_rays|cubemap finds no implementation use: panels mode is gz::rendering::DepthCamera (src/panel_rig.hpp:21 include, :68 std::vector<DepthCameraPtr> cams_, src/panel_rig.cpp:98 scene->CreateDepthCamera), and include/gz_gpu_ouster_lidar/ray_processor.hpp:35-37 states it 'Replaces the former GpuRays-cubemap equirect grid'. All nine cited user-facing lines are present verbatim (README.md:436, ouster_macro.xacro:19,83, turtlebot3_ouster.urdf.xacro:19,30, sensor_stack.launch.py:10,191, ouster_standalone.launch.py:10,108, turtlebot3_ouster.launch.py:102) plus the stale gpu_rays_ comment at test/test_lifecycle.cpp:56. The reviewer MISSED one that matters more than any of those: package.xml:9 still advertises 'Fires rays via gz::rendering::GpuRays' — that string ships in the package index. Severity corrected to low: it is documentation drift only; no code path, output or build is affected.

---

### `README.md:433` — README contradicts itself on where raycast mode runs: "cast on the CPU" at :433 vs "accelerates both ray modes" at :613

*Cleanup* · cleanup · reported as medium, downgraded by verifier

README.md:433-434 tells users raycast mode means "beams are cast on the CPU against an ECM scene mirror, with no render engine." README.md:613-615, 180 lines later, says "The selected backend accelerates **both** ray modes — the per-beam casting in `raycast` and the resample/noise pipeline in `panels` (verified: with `--gpus all` the plugin logs `Using cuda backend.` in raycast)." The code agrees with :613: src/raycast_mirror.cpp:678 calls `proc_->castScanProcessed(...)`, which goes through cuda/ray_processor_dispatch.cpp:170 to whichever backend `pickBackend` chose in CUDA -> HIP -> SYCL -> CPU order (dispatch.cpp:75). Raycast is CPU-only exactly when no GPU backend is compiled in or probes successfully. Note the surrounding docs get it right — examples/urdf/ouster_macro.xacro:16-18 correctly says raycast "needs no GPU" (not "uses no GPU").

**Failure scenario.** A user sizing a GPU-less simulation host reads README:433, concludes raycast is a CPU path, and provisions accordingly — then finds their 4096x512 sensor is unusable because the CPU fallback's per-beam cast is 5-10 ms/frame (README's own estimate at :887). Conversely a user with a GPU reads :433 and never investigates why raycast is slow on a machine where the CUDA backend failed to probe.

**Fix.** Rewrite README.md:433-434 to "beams are cast against an ECM scene mirror on the selected compute backend (CUDA/HIP/SYCL/CPU) with no render engine involved; a non-rendering `altimeter` anchor is sufficient and the world needs no display," which is both accurate and consistent with :613-615.

**Verifier.** Confirmed by reading both passages. README.md:433-434 reads 'beams are cast on the CPU against an ECM scene mirror, with no render engine'; README.md:613-615 reads 'The selected backend accelerates both ray modes — the per-beam casting in raycast and the resample/noise pipeline in panels (verified: with --gpus all the plugin logs Using cuda backend. in raycast)'. The code sides with :613: src/raycast_mirror.cpp:678 calls proc_->castScanProcessed, which reaches cuda/ray_processor_dispatch.cpp:170 and thence whichever backend pickBackend selected in CUDA->HIP->SYCL->CPU order (dispatch.cpp:75). Raycast is CPU-only only when every GPU probe fails. Severity corrected to low: a self-contradictory README sentence, no functional impact; the provisioning failure scenario is plausible but speculative and the correct statement is already 180 lines away in the same file.

---

### `README.md:944` — README's test table is 9 suites out of date and omits the two largest, covering the newest headline feature

*Cleanup* · test-coverage

CMakeLists.txt:305-406 registers 20 test targets: 15 gtest (`ament_add_gtest`/`ament_auto_add_gtest` at :310,317,331,337,343,348,353,358,364,372,377,380,383,388,392,397) and 5 pytest (`ament_add_pytest_test` at :401-405). README.md's table at :944-953 lists 11 rows. Missing entirely: `test_obscurants` (933 lines), `test_obscurant_config` (597 lines), `test_ouster_lidar_profile` (199 lines), `test_beam_math`, and all five Python structural suites (`test_worlds` 392 lines, `test_launch_files`, `test_xacro`, `test_repos_file`, `test_display_hardening`). The two largest omissions are exactly the suites covering the smoke/dust/fog obscurant model, which the README devotes a full section to and which is the most recently added feature (commits f98c8c4, 06eadd7).

**Failure scenario.** A reviewer or downstream evaluator reads the README's Tests section to decide whether the obscurant model is verified, sees no entry for it in the coverage table, and either (a) concludes the participating-media physics has no automated coverage and duplicates the 1,530 lines of tests that already exist, or (b) writes off the feature as unverified. The prose immediately below the table ("These tests run on the CPU backend...") also implies the table is exhaustive.

**Fix.** Add rows for test_obscurants, test_obscurant_config, test_ouster_lidar_profile and test_beam_math, plus a single row covering the five pytest structural suites, to README.md:944-953. Since the list has already drifted twice, consider generating the target list from CMakeLists.txt in CI (`grep -oE 'ament(_auto)?_add_(gtest|pytest_test)\\(\\w+'`) and failing if a registered target is absent from the README.

**Verifier.** Verified. CMakeLists.txt registers 20 targets — 15 gtest (:310,317,331,337,343,348,353,358,364,372,377,380,383,388,392,397 — note :313/:329 are comments, the 16 hits reduce to 15 targets) and 5 pytest (:401-405). The README table at :944-953 has 11 rows covering 12 gtest targets (one row folds scan_timing/sim_time_scheduler/packet_pacing). Absent: test_obscurants, test_obscurant_config, test_ouster_lidar_profile, test_beam_math and all five pytest suites = 9 registered targets undocumented, and the obscurant suites are indeed the two largest. The prose at :955 ('These tests run on the CPU backend...') does read as if the table were exhaustive. Low is the correct severity and the reviewer already graded it low.

---

### `cuda/ray_processor_cuda.cuh:11` — cuda/ray_processor_cuda.cuh is a vestigial header whose only includer is the file that defines everything in it

*Cleanup* · api-misuse

grep for `ray_processor_cuda.cuh` across the repo returns exactly one hit: cuda/ray_processor_cuda.cu:9, the translation unit that also DEFINES all three declared functions (`launchResampleKernel` at .cu:81, `launchRayProcessKernel` at .cu:335, `launchInitRandKernel` at .cu:367) and is the only caller of each (.cu:437, .cu:447/495/670, .cu:825). Its header comment claims it exists to keep "CUDA internal types" out of public headers, but it declares functions, not types, and nothing outside the .cu ever wanted them. The cost is real: because the declarations sit at namespace scope, these three symbols get external linkage in the static library, unlike HIP and SYCL which wrap their entire backend in an anonymous namespace (cuda/ray_processor_hip.cpp:32, cuda/ray_processor_sycl.cpp:40) and export only their `make*Backend` factory.

**Failure scenario.** A future contributor adds an unrelated `launchResampleKernel` helper to another TU in gz_gpu_ouster_lidar_cuda and gets an ODR violation / duplicate-symbol link error that would be impossible if the CUDA backend followed the same anonymous-namespace discipline as the other two GPU backends. The header also invites a reader to believe the launchers are an intentional cross-TU seam, which they are not.

**Fix.** Delete cuda/ray_processor_cuda.cuh, drop the include at ray_processor_cuda.cu:9, and move the three launchers plus `checkCuda`/`kBlock` into the existing anonymous namespace that already wraps `CudaBackend` (ray_processor_cuda.cu:378), matching ray_processor_hip.cpp:32 and ray_processor_sycl.cpp:40. Nothing outside the TU references them.

**Verifier.** Confirmed exactly as described. Repo-wide grep for ray_processor_cuda.cuh returns one hit: cuda/ray_processor_cuda.cu:9. That same TU defines all three declared functions (launchResampleKernel, launchRayProcessKernel at :335, launchInitRandKernel at :367) and is their only caller. The header's own comment claims it keeps 'CUDA internal types' out of public headers, yet it declares only functions. The linkage asymmetry is real: these three sit at gz_gpu_ouster_lidar namespace scope (external linkage) while the CudaBackend class below them is inside an anonymous namespace opened at ray_processor_cuda.cu:378, and both sibling backends wrap everything from the top — cuda/ray_processor_hip.cpp:32 and cuda/ray_processor_sycl.cpp:40 both open `namespace {` immediately inside the package namespace. Low is right: no current symbol collision exists (HIP/SYCL's like-named launchers are class members inside anon namespaces), so this is cleanup, not a bug.

---

### `cuda/raycast_math.hpp:745` — Dead overload rcApparentReflectance(const RcInstance&, float, float) has zero call sites

*Cleanup* · cleanup

cuda/raycast_math.hpp declares two overloads of `rcApparentReflectance`: the (float diffuse, float spec, float cos_inc) form at :729 and an (const RcInstance&, float cos_inc, float fallback_retro) convenience wrapper at :745-751. A repo-wide grep for `rcApparentReflectance` finds: the two definitions; :750 (inside the wrapper's own body, delegating to the float form); :857 in `rcHitReflectance`, which calls the FLOAT form with `material.diffuse, material.spec, cos_inc`; and five comment-only mentions (src/raycast_mirror.cpp:278, cuda/raycast_scene.hpp:105, raycast_math.hpp:1450, test/test_raycast.cpp:388,826,844). No call site takes the RcInstance overload. It was superseded when the material path moved to `rcMaterialAtHit`/`RcMaterialSample` (raycast_math.hpp:675-692), which resolves the response-map texture the RcInstance overload cannot see.

**Failure scenario.** A contributor adding a new cast path (a second-return arbitration, say) reaches for the ergonomic-looking `rcApparentReflectance(inst, cos_inc, sp.fallback_retro)` overload. It reads `inst.retro`/`inst.spec` directly and silently ignores the RGBA response map entirely, so any surface with a `.ouster.png` companion texture (src/raycast_mirror.cpp:101-117) returns the scalar SDF material instead of the per-texel diffuse/specular — a wrong reflectivity byte on exactly the surfaces the response-map feature exists for.

**Fix.** Delete the overload at cuda/raycast_math.hpp:745-751. If a convenience form is wanted, route it through `rcMaterialAtHit` so it cannot bypass the response map.

**Verifier.** Confirmed. cuda/raycast_math.hpp:745-751 defines the RcInstance overload; repo-wide grep for rcApparentReflectance returns only the two definitions (:729, :745), the delegation inside the wrapper's own body (:750), the real call site at :857 which passes material.diffuse/material.spec from RcMaterialSample, and comment-only mentions (src/raycast_mirror.cpp:278, cuda/raycast_scene.hpp:105, raycast_math.hpp:1450, test/test_raycast.cpp:388,826,844, docs/MODEL_REFERENCES.md:224). No caller uses the RcInstance form, and it does read inst.retro/inst.spec directly, bypassing the response-map path that rcMaterialAtHit resolves. It is an inline header function so there is no code-size or linkage cost, and the 'contributor reaches for it' scenario is hypothetical — low is correct.

---

### `src/gz_gpu_ouster_lidar_system.cpp:451` — CPU-fallback warning says "No CUDA-capable device detected" in a four-backend plugin

*Cleanup* · cleanup

src/gz_gpu_ouster_lidar_system.cpp:450-454 logs `"No CUDA-capable device detected; gz_gpu_ouster_lidar running on CPU fallback"` whenever `usesCpuFallback()` is true. But `usesCpuFallback()` (cuda/ray_processor_dispatch.cpp:177-180) just tests `strcmp(backend_->name(), "cpu") == 0`, and the dispatcher tries CUDA -> HIP -> SYCL -> CPU (dispatch.cpp:75). The preceding comment at :438-442 has the same problem ("The CUDA path self-probes for a GPU"). The dispatcher itself already prints the accurate line at dispatch.cpp:92-94 (`Using %s backend.`).

**Failure scenario.** An AMD Radeon user runs a build where GZ_GPU_OUSTER_ENABLE_HIP was off (cuda/CMakeLists.txt:88 — it defaults to AUTO and silently disables when hipcc is absent). The plugin tells them "No CUDA-capable device detected", which is true and useless: they have no CUDA device and never will. They go hunting for NVIDIA drivers instead of rebuilding with ROCm, which is the actual fix.

**Fix.** Change the message to name the backends that were probed and lost, e.g. "No GPU backend available (CUDA/HIP/SYCL all unavailable or not compiled in); running on the CPU fallback — expect lower sim rate on high-resolution sensors." Update the comment at :438-442 to say "the dispatcher probes CUDA/HIP/SYCL" rather than "the CUDA path".

**Verifier.** Confirmed verbatim. src/gz_gpu_ouster_lidar_system.cpp:450-454 logs 'No CUDA-capable device detected; gz_gpu_ouster_lidar running on CPU fallback', gated on usesCpuFallback(), which is just strcmp(backend_->name(), "cpu")==0 (cuda/ray_processor_dispatch.cpp:177-179) — true after CUDA, HIP and SYCL all failed, since pickBackend tries {kCuda,kHip,kSycl,kCpu} (dispatch.cpp:75). The preceding comment at :438-442 has the same CUDA-only framing, and dispatch.cpp:92-94 already prints the accurate 'Using %s backend.' line. Misleading log text only; low.

---

### `src/ouster_lidar_profile.hpp:73` — Four OusterLidarProfile fields are write-only or entirely unused: 20 dead assignments of datasheet values

*Cleanup* · cleanup · reported as medium, downgraded by verifier

A repo-wide grep (excluding .git) for `false_positive_rate|beam_diameter|lambertian_accuracy|retroreflector_accuracy` returns only the declarations in src/ouster_lidar_profile.hpp:73,74,77,79 and assignments in src/ouster_lidar_profile.cpp — zero reads anywhere in src/, cuda/, include/, test/ or README. Specifically: `false_positive_rate` (hpp:79) is declared with a default of 1.0e-4 and is never assigned OR read; `beam_diameter_m` (hpp:77) is assigned 12 times (cpp:114,125,137,162,171,181,191,214,220,228,234,401) and never read; `lambertian_accuracy_m` (hpp:73) and `retroreflector_accuracy_m` (hpp:74) are assigned 4 times each (cpp:102-103,151-152,206-207,419-420) and never read. Contrast the fields right beside them — `representable_range_m`, `max_returns`, `beam_divergence_fwhm_deg` — which ARE consumed (src/ouster_metadata.cpp:148, src/gz_gpu_ouster_lidar_system.cpp:525-527).

**Failure scenario.** A contributor reads `p.beam_divergence_fwhm_deg = 0.18; p.beam_diameter_m = 0.0095;` in the OS1 Rev7 block (cpp:171-172), sees both carefully curated to real datasheet numbers, and assumes the beam-spot model is wired up. It is not — only divergence reaches ros_interface.cpp:202 as a read-only ROS param. A wrong beam_diameter_m value can be committed, reviewed and merged with zero observable effect, so the field is unverifiable by construction.

**Fix.** Delete `false_positive_rate` outright. For the other three, either wire them up (beam_diameter_m belongs in the footprint model; the two accuracy_m fields are a systematic range bias the channel model does not currently apply) or delete the fields and their 20 assignments. If they are being staged for future work, at minimum surface them as read-only ROS params next to `beam_divergence_fwhm_deg` (src/ros_interface.cpp:202) so a wrong value is at least observable.

**Verifier.** Grep across the whole tree (excluding .git) reproduces exactly the cited hit list: declarations at src/ouster_lidar_profile.hpp:73,74,77,79 plus 20 assignments in src/ouster_lidar_profile.cpp and nothing else — no reads in src/, cuda/, include/, test/ or docs. false_positive_rate (hpp:79) is neither assigned nor read. The contrast is also accurate: representable_range_m is read at src/ouster_metadata.cpp:148 and beam_divergence_fwhm_deg at src/gz_gpu_ouster_lidar_system.cpp:525-526 and src/ros_interface.cpp:202. Severity corrected down: there is zero runtime consequence — no wrong data, no crash, no perf effect. This is dead state and an unverifiable-datasheet-value hazard, i.e. cleanup. Note the header comment at hpp:66-69 documents deliberate retention of the two accuracy fields ('retained separately because it is a systematic bound'), so 'delete them' is not obviously the right fix; the field-is-unverifiable point stands either way.

---

### `src/gz_gpu_ouster_lidar_system.cpp:794` — Entity discovery re-scans the whole ECM every physics tick forever when the sensor or IMU is never found, and never warns

*Lifecycle & threading* · efficiency · reported as medium, downgraded by verifier

`if (!lidar_frame_found_)` (794) runs `ecm.Each<Name, Sensor>` over every sensor entity in the world, calling `topLevelModel(ent, ecm)` per name match (803). `if (imu_enabled_ && !imu_entity_found_)` (815) does the same, and additionally heap-allocates a fresh `std::vector<std::pair<std::string, Entity>> candidates` (825) plus a `std::string` copy per candidate (844) on every single invocation. Both are latched-once flags with no failure timeout: if the name never matches, they re-run at the full physics rate (1000 Hz at the default 1 ms step) for the entire life of the process. Worse, there is no diagnostic on the failure path — line 808 logs only on success. Contrast this with the panels-mode no-render case, which gets a carefully worded one-shot ERROR after 2 s of sim time (711-727). Raycast mode gets nothing: `mirror_->postUpdate` is guarded by `lidar_frame_found_` at line 879, so a name mismatch means the raycast worker is never fed a single job and the plugin publishes only metadata, forever, in complete silence.

**Failure scenario.** A user writes `<sensor_name>/sensor/lidar/os_1</sensor_name>` in the plugin SDF but the Gazebo <sensor> element is named `lidar0`. The plugin derives `lidar_frame_name_ = "os_1"` (486-491), never matches, and: (a) burns a full-ECM `Each` traversal plus a per-tick heap allocation at 1 kHz for the whole session, and (b) publishes zero point clouds with no error message. The user sees /metadata appear, concludes the plugin loaded correctly, and has no signal pointing at the name mismatch.

**Fix.** Throttle the retries (only search when `info.simTime` has advanced past the last attempt by, say, 250 ms) and add a one-shot ERROR mirroring the panels-mode diagnostic: after ~2 s of sim time with `!lidar_frame_found_`, log the searched name, the top-level model, and the sensor names actually present in that model. Hoist `candidates` to a member so the IMU search stops allocating per tick.

**Verifier.** Half right, and the half that is quantified is wrong. Verified true: lines 794-812 and 815-867 are unlatched retries with no throttle, line 808/863 log only on success, and grep for lidar_frame_found_ shows no diagnostic anywhere — so a <sensor_name> mismatch means line 879's `mirror_ && lidar_frame_found_` never fires, no scan job is ever posted, and the plugin publishes only /metadata in total silence. That part I confirmed. Wrong as written: (a) it is not a 'full-ECM traversal' — ecm.Each<components::Name, components::Sensor> (797, 826) visits only entities carrying both components (the world's sensors, typically single digits) through gz-sim's cached view; (b) 'heap-allocates a fresh std::vector ... plus a std::string copy per candidate on every single invocation' is false in the very scenario the finding describes: `std::vector candidates` (825) default-constructs without allocating, and the emplace_back at 844 is only reached on a name match — in the never-matches case there is zero allocation per tick. Mitigating further: Configure already logs the name being searched (`gz_sensor=%s`, line 575-580), so the user does have the searched string in the log. Downgraded to low: the surviving substance is a missing one-shot diagnostic, not a perf cliff.

---

### `src/gz_gpu_ouster_lidar_system.cpp:131` — Obscurant "raycast mode only" warning is emitted before ray_mode is validated, so a typo warns then silently applies

*Obscurants & noise* · correctness

`parseObscurants` is called at line 260 and checks `ray_mode_ != "raycast"` at line 131 to decide whether to warn that obscuration will be ignored. But `ray_mode_` is only validated and normalised much later, at lines 326-330:

```cpp
if (ray_mode_ != "panels" && ray_mode_ != "raycast") {
    RCLCPP_WARN(kLogger, "Unknown ray_mode='%s'; ... Defaulting to raycast.", ...);
    ray_mode_ = "raycast";
}
```

So a misspelled mode produces the wrong diagnostic pair.

**Failure scenario.** World authored with `<ray_mode>racast</ray_mode>` plus `<obscurant>` volumes. At startup the log says "obscurant/particle obscuration is modelled in the raycast ray mode only; ray_mode='racast' will ignore it", and then "Unknown ray_mode='racast' ... Defaulting to raycast." The obscurants are in fact applied. A user chasing why their smoke has no effect is sent looking in exactly the wrong place, and the informative INFO line (volume count, sigma, S, albedo, eta, pulse gate) is never printed.

**Fix.** Move the `parseObscurants(sdf)` call to after the ray_mode validation block (i.e. below line 330), or hoist the ray_mode normalisation to immediately after it is read at line 196.

**Verifier.** Confirmed, exactly as described. ray_mode_ is read raw at gz_gpu_ouster_lidar_system.cpp:195-197, parseObscurants(sdf) is called at line 260, its `if (ray_mode_ != "raycast")` warning is at line 131-136 with the informative INFO in the else arm at 137-146, and the normalisation that rewrites an unknown value to "raycast" is at 326-331 — 66 lines later. The config is stored unconditionally (line 148) and consumed at line 605 (mp.obscurants = obscurants_.get()) with no ray_mode gate, so with <ray_mode>racast</ray_mode> the user is told obscuration will be ignored, it is then silently applied, and the volume/sigma/S/albedo/eta/pulse-gate INFO line is never printed. Default is "raycast" (gz_gpu_ouster_lidar_system.hpp:81), so this only triggers on a typo or on the literal "panels" (where the warning is correct) — genuinely low. Not in AUDIT.md.

---

### `src/packet_encoder.cpp:126` — Dual-return UDP profiles emit an all-zero second return while max_returns=2 is advertised

*Ouster protocol* · correctness · reported as medium, downgraded by verifier

`encodeScan` writes exactly four channel blocks — `RANGE`, `SIGNAL`, `REFLECTIVITY`, `NEAR_IR` (lines 126-129). `RANGE2`, `SIGNAL2` and `REFLECTIVITY2` are never written, and `pkt_buf_` is zeroed at line 111, so a dual-return packet ships a fully zero second return.

Meanwhile the plugin advertises two returns: `OusterLidarProfile::max_returns` is 2 for Rev06/06.2/07/07.1/08 (`ouster_lidar_profile.cpp:51-58, 153, 208`), it is copied to `cfg.max_returns` at `src/gz_gpu_ouster_lidar_system.cpp:527`, and declared as the read-only ROS parameter `max_returns` at `src/ros_interface.cpp:203`. Consumers sizing themselves off that parameter will expect a second cloud that either does not exist (single-return metadata) or is pure zeros (dual metadata).

On the consumer side `os_cloud` reads `info.num_returns()` from the UDP profile and creates one publisher per return (`os_cloud_node.cpp:131, 136-140`), so a `RNG19_RFL8_SIG16_NIR16_DUAL` metadata produces a second `points2` topic whose every point has range 0.

**Failure scenario.** Supply metadata with `udp_profile_lidar: RNG19_RFL8_SIG16_NIR16_DUAL` (`DUAL_FIELD_INFO`, parsing.cpp:265-280, does contain SIGNAL so nothing throws). `os_cloud` advertises `/points` and `/points2`. `/points` is correct; `/points2` is 131072 points all at range 0 published at 10 Hz. Any perception stack that fuses both returns — the normal reason to run a dual profile — ingests a full cloud of origin points every frame.

**Fix.** Either (a) reject dual/multi-return `udp_profile_lidar` values in `OusterMetadata::load()` with an explicit "single-return simulation only" error, and clamp the advertised `max_returns` ROS parameter to 1 regardless of what the profile table says; or (b) populate the second return from the raycaster's second-strongest hit. Option (a) at minimum — advertising `max_returns=2` while shipping one return is the part that misleads consumers.

**Verifier.** Mechanically confirmed: packet_encoder.cpp:111 memsets pkt_buf_ and :126-129 writes only RANGE/SIGNAL/REFLECTIVITY/NEAR_IR - RANGE2/SIGNAL2/REFLECTIVITY2 are never touched, and DUAL_FIELD_INFO (parsing.cpp:265-280) does carry SIGNAL so nothing throws, it just ships zeros. max_returns is likewise 2 for Rev06/06.2/07/07.1/08 (ouster_lidar_profile.cpp:51-58, 153, 208, 387), copied at gz_gpu_ouster_lidar_system.cpp:527 and declared read-only at ros_interface.cpp:203. Downgraded to low: (a) no shipped metadata triggers it - all ten config/metadata/*.json are RNG19_RFL8_SIG16_NIR16 or LEGACY, both single-return, so os_cloud creates one publisher; (b) multi-return is an explicitly documented modelling gap (docs/MODEL_REFERENCES.md:461, 'Multi-return / full waveform - Second returns through vegetation, edge splits'); and (c) max_returns is presented as a hardware-revision spec alongside D90/precision/beam divergence, matching the README hardware table's 'Returns' column (README.md:650-658), not as a promise about emitted data. The actionable residue is the missing guard: OusterMetadata::load() accepts a dual udp_profile_lidar without a single word of warning.

---

### `cuda/ray_processor_math.hpp:304` — calibratedRangeAtReflectivity recomputes a sensor-lifetime constant log() twice per pixel

*Performance* · performance · reported as high, downgraded by verifier

`const float exponent = gzm::log_(range_80 / range_10) / gzm::log_(8.0f);` — `log_(8.0f)` is a constant the compiler folds, but `log_(range_80 / range_10)` is not: `range_80`/`range_10` arrive as runtime `RayProcessParams` fields. Those fields are set once in `Configure` (src/gz_gpu_ouster_lidar_system.cpp:974-981, from `detection_range_10_d90_ * mode_range_scale_` etc., all fixed at load time) and merely copied by `makeRayProcessParams` each frame — they are invariant for the sensor's entire lifetime, not just per frame. `detectionProbability` (line 312) calls `calibratedRangeAtReflectivity` twice per pixel (d90 at line 319, d50 at line 323), and `dropoutProbability` calls `detectionProbability` for every valid pixel. So the same two logarithms are recomputed H*W times per scan for a value that could be computed once at Configure. Identical waste in all four backends, since they all route through this shared header.

**Failure scenario.** OS1-128 @ 2048: 262,144 pixels x 2 redundant `std::log` = 524,288 dead log calls per scan. At ~20-40 cycles each on x86 that is 10-21M cycles = 3.5-7 ms per scan, on the single-threaded `processCpu` path (see the previous finding), i.e. up to 20% of that stage's cost. On CUDA/HIP it is 2 extra non-fast-math logf per thread in `rayProcessKernel`. The gate at ray_processor_cpu_impl.cpp:104 / ray_processor_cuda.cu:252 is `detection_range_10 > 0 && detection_range_80 > 0`, which every real Ouster profile satisfies, so this fires on every valid pixel of every scan by default.

**Fix.** Add two precomputed fields to `RayProcessParams` (e.g. `detection_exponent_d90`, `detection_exponent_d50`) filled once in `makeRayProcessParams` (or better, cached in Configure), and change `calibratedRangeAtReflectivity` to take the exponent instead of deriving it. While there, note `retroForNoise(retro, idx)` is called twice per pixel (cpu_impl.cpp:105 and :122; cuda.cu:253 and :271) for the same `idx` — hoist that load too.

**Verifier.** Line 304 says exactly what is claimed (`gzm::log_(range_80 / range_10) / gzm::log_(8.0f)`), detectionProbability calls it twice (:319 d90, :323 d50), and the params are load-time constants (gz_gpu_ouster_lidar_system.cpp:974-981 copies detection_range_*_d90_/_d50_ set once at :412-415). The d50 call is not short-circuited by the :302 guard for real profiles — ouster_lidar_profile.cpp:109-132/407-415 set nonzero d50 ranges — so both logs do execute per pixel. But this is a micro-optimisation, not a defect: one extra logf out of the ~7 transcendentals + 3-4 mt19937 draws already in the per-pixel body, no behavioural difference, and it only fires when noise is enabled (gate at cpu_impl.cpp:102-104). 'HIGH' is unjustifiable for a constant-factor hoist.

---

### `cuda/raycast_math.hpp:1408` — rcCastOneRay evaluates 6 sin/cos per ray for a direction that never changes

*Performance* · performance · reported as medium, downgraded by verifier

Lines 1408-1412 compute `ce = cos_(el)`, `cos_(az)`, `sin_(az)`, `sin_(el)`, `cos_(az_enc)`, `sin_(az_enc)` — six transcendentals — to build `d_s` and `o_s`. Both are SENSOR-FRAME quantities that depend only on `beam = idx / sp.W`, `m = idx % sp.W`, `beam_alt_deg[beam]`, `beam_az_deg[beam]` and `deg_per_col = 360/W`. The beam tables are metadata-derived and immutable for the sensor's lifetime (cuda/ray_processor_cuda.cu:761-767 already relies on exactly that to skip re-uploading them). Nothing here varies scan to scan; the world pose enters only afterwards, at lines 1414-1415, as a pure rotate+translate. The angles are also separable: `az_rad = A_beam + az_enc` where `A_beam = -beam_az_deg[beam]*pi/180` and `az_enc = -m*deg_per_col*pi/180`, so `cos(az)`/`sin(az)` follow from angle-addition on two tables of size H and W. Worse, `o_s` (lines 1411-1412) pays two of the six even when `n_off == 0` — the multiply by `n_off` happens after the trig.

**Failure scenario.** OS1-128 @ 2048: 262,144 rays x 6 = 1.57M sin/cos per scan, recomputed identically every scan forever. On the OpenMP CPU cast that is ~1.5-2 ms/scan across 8 threads; on CUDA/HIP, with no `--use_fast_math` in cuda/CMakeLists.txt, these are the full-precision libdevice `sinf`/`cosf` (tens of instructions each, SFU-bound) rather than 1-instruction `__sinf`, and in a sparse world (a handful of instances, shallow TLAS) they are a measurable fraction of the whole kernel.

**Fix.** Precompute two small tables once at `RaycastMirror::start` and upload them beside `beam_alt`/`beam_az` (they cache by source pointer already): a per-beam table of {sin el, cos el, sin A_beam, cos A_beam} (H x 4 floats) and a per-column table of {sin az_enc, cos az_enc} (W x 2 floats). Then `d_s` and `o_s` are ~6 multiply-adds with zero transcendentals. For H=128, W=2048 the tables are 2 KB + 16 KB — they live in cache. If you prefer not to change the interface, at minimum guard the `o_s` trig behind `n_off > 0`.

**Verifier.** Lines 1408-1412 do compute exactly six transcendentals (cos el, cos az, sin az, sin el, cos az_enc, sin az_enc), and both d_s and o_s depend only on beam/column indices and the immutable beam tables; the world pose is applied afterwards at :1414-1415 as rcXformPoint/rcRotate, so the values are genuinely per-(beam,column) constants. The `o_s` trig at :1411-1412 is also unconditional on n_off, as claimed. No fast-math flag exists in cuda/CMakeLists.txt (grep: no -use_fast_math). But this is textbook ray-setup cost, not a defect: it is dwarfed by rcNearestHit traversal in the same function, and the proposed H*4 + W*2 tables add memory traffic and an interface change on all four backends for a small constant. Medium is inflated; this is a low-priority optimisation.

---

### `cuda/raycast_math.hpp:1583` — NEAR_IR block re-samples the response map and re-derives the normal for a hit already resolved

*Performance* · performance · reported as medium, downgraded by verifier

Line 1457 calls `rcHitReflectance(..., &material0)`, which internally computes `o_l`, `d_l`, `p_l`, calls `rcMaterialAtHit` (the full response-map path: `rcHitUv` + `rcSampleResponse`) and then `rcCosIncidence` (which computes `rcSurfaceNormalLocal`). Then lines 1562-1585 redo all of it for the winning hit: `rcXformPoint`/`rcRotate` into local frame (1563-1564), `rcSurfaceNormalLocal` again (1567), and `rcMaterialAtHit` again (1583). In the overwhelmingly common case — no transparent surface (line 1474 gate), no mirror (line 1506 gate) — `w_inst == inst0`, `w_tri == tri0`, `w_o == o`, `w_d == d`, `w_t == t0`, so `winning_material` is bit-identical to `material0` that is already sitting in a local variable. `rcSampleResponse` (line 645) is not cheap: 4 channels x 4 taps = 16 texel loads, each through `rcResponseChannel` with two `%` operations in `rcWrapIndex`, i.e. 64 integer modulos, plus `floor_` and — for sphere/cylinder instances — an `atan2_` in `rcHitUv` (lines 573, 586), or a full barycentric solve with a divide for mesh hits (lines 605-620).

**Failure scenario.** ouster_showcase.sdf / ouster_demo_panels.sdf, which exist precisely to exercise `.ouster.png` response maps. Every hit ray with a response map pays the 16-tap bilinear twice and the surface-normal derivation twice. At OS1-128 @ 2048 with ~80% hit rate that is ~210k redundant response samples (~3.4M texel loads, ~13M modulos) plus ~210k redundant normal computations per scan, every scan.

**Fix.** Carry the front-surface result forward. Track a `bool winner_is_front = true` (cleared in the glass branch at line 1494 and the mirror branch at line 1549), and in the NIR block reuse `material0.nir` and the normal from the first hit when it is still set; only recompute in the two override branches. Cleanest version: have `rcHitReflectance` also return the local-frame unit normal via an out-param (it already computes it inside `rcCosIncidence`) and stash `(material, normal)` for each candidate as it wins.

**Verifier.** Confirmed duplication: rcHitReflectance (:839-857) computes o_l/d_l/p_l, calls rcMaterialAtHit (:851) and rcCosIncidence (:856, which itself calls rcSurfaceNormalLocal at :710); the NIR block then redoes rcXformPoint/rcRotate (:1563-1564), rcSurfaceNormalLocal (:1567) and rcMaterialAtHit (:1583) for w_inst/w_tri, which equal inst0/tri0 unless the glass (:1474) or mirror (:1506) branch overrode them. So material0 is bit-identical to winning_material in the common case and is discarded. Two corrections: the cost claim is doubled — rcSampleResponse is 4 channels x 4 taps = 16 rcResponseChannel calls x 2 modulos = 32 integer modulos, not 64 — and rcMaterialAtHit early-returns at :684-689 for any instance without a response map, so on ordinary worlds the redundant call is a handful of scalar copies, leaving only the duplicate normal. This is a refactor, not a bug: low.

---

### `src/packet_encoder.cpp:131` — encodeScan copies the whole scan a second time through a staging buffer on the sim thread

*Performance* · performance · reported as high, downgraded by verifier

Every packet is built into the member scratch `pkt_buf_` (memset at line 111, filled by `pw.set_*` at lines 114-129) and then bulk-copied into its destination message: `encode_pkts_[p].buf.assign(pkt_buf_.begin(), pkt_buf_.end());`. `pkt_buf_` exists only for this purpose — grep confirms it is written nowhere else (packet_encoder.hpp:60, sized once in `start` at line 36). The PacketWriter API already writes through a `uint8_t*`, so the packet could be built directly into `encode_pkts_[p].buf`, deleting the copy outright. This runs on the SIM thread (PostUpdate -> publishChannels -> encodeScan), so it is a direct hit to the physics step, not to a worker.

**Failure scenario.** OS1-128 legacy profile: lidar_packet_size = 24,832 B, columns_per_packet = 16, W = 2048 -> n_packets = 128. The loop therefore memsets 3.18 MB and then copies another 3.18 MB per scan, for 6.36 MB of avoidable memory traffic. At 20 Hz that is ~64 MB/s of pure `memcpy` executed inside PostUpdate on every sim tick that produces a scan; the copy alone is ~0.3-0.5 ms of sim-thread stall per scan on typical DDR4.

**Fix.** Delete `pkt_buf_` and encode in place: after `encode_pkts_.resize(n_packets)` (line 108), for each p do `auto & buf = encode_pkts_[p].buf; buf.assign(meta_->pw->lidar_packet_size, 0);` (reuses the capacity circulated back by the drain swap at line 149, same cost as the current memset) and pass `buf.data()` to every `pw.set_*` call. Zero behaviour change, removes one full-scan copy per frame.

**Verifier.** Confirmed: pkt_buf_ is declared at packet_encoder.hpp:60, sized only in start() (:36), and is written nowhere except the encode loop; line 131-132 does `encode_pkts_[p].buf.assign(pkt_buf_.begin(), pkt_buf_.end())`, a full per-packet copy that in-place encoding would delete. It does run on the sim thread (PostUpdate -> publishChannels at gz_gpu_ouster_lidar_system.cpp:932/1054 -> encodeScan at :1062). Severity downgraded hard from HIGH: the copy is a minority of encodeScan's cost — the four pw.set_block calls (:126-129) bit-pack H*cpp*4 fields per packet, far more work than a 24 KB memcpy — and the memset at :111 is still needed either way, so the actual saving is ~0.15-0.3 ms/scan, not a correctness or cadence issue.

---

### `CMakeLists.txt:433` — ament_auto_package installs whatever junk is in examples/, including pytest __pycache__ directories

*Launch / URDF / worlds* · build

`ament_auto_package(INSTALL_TO_SHARE config examples)` copies the source `examples/` tree verbatim. `.gitignore:19-20` correctly ignores `__pycache__/` and `*.pyc` for git, but CMake's install(DIRECTORY) has no such filter, so anything a developer's tooling leaves in the tree ships. Confirmed in this workspace: `share/gz_sensors_ouster/examples/launch/__pycache__/` exists in the built install tree, containing depthcam_derisk.launch.cpython-312.pyc, ouster_standalone.launch.cpython-312.pyc, sensor_stack.launch.cpython-312.pyc and turtlebot3_ouster.launch.cpython-312.pyc — byte-code from a prior in-source pytest run that was never meant to be a package artifact. The same applies to test/__pycache__ if the test dir were ever installed.

**Failure scenario.** A developer runs `pytest test/` in the source tree, then `colcon build`; the resulting install tree (and the Docker image layered on it) carries stale .pyc files for launch files that may since have been edited. `ros2 launch` loads the .py so behaviour is unaffected, but the package is no longer reproducible from its sources and image size/provenance audits pick up unexplained binaries.

**Fix.** Replace the bare `examples` argument with an explicit install(DIRECTORY ... PATTERN "__pycache__" EXCLUDE PATTERN "*.pyc" EXCLUDE) for examples/, or clean the tree in CI before packaging. Keeping the `config` half of the ament_auto_package call is fine — it holds only JSON and an SDF.

**Verifier.** Confirmed, including the mechanism. CMakeLists.txt:433 is `ament_auto_package(INSTALL_TO_SHARE config examples)`, and /opt/ros/jazzy/share/ament_cmake_auto/cmake/ament_auto_package.cmake:123-128 expands that to a bare install(DIRECTORY "${_dir}" DESTINATION share/${PROJECT_NAME}) with no PATTERN EXCLUDE. The install tree at /root/ros2_ws/install/gz_sensors_ouster/share/gz_sensors_ouster/examples/launch/__pycache__/ contains all four .pyc files. I ruled out the obvious alternative explanation (runtime-generated bytecode from `ros2 launch` importing the installed .py): the installed .pyc mtimes are 2026-07-15 and 2026-07-28 while the installed .py files are 2026-08-07, and three of the four have different sizes/hashes from the current source-tree .pyc — i.e. they are stale copies carried in by install(DIRECTORY), not regenerated imports. .gitignore:19-20 ignores them for git but has no effect on CMake. Packaging hygiene only; behaviour is unaffected since ros2 launch loads the .py.

---

### `docker/entrypoint.sh:27` — Docker headless smoke runs the os_image decoder that the `images` launch arg exists to switch off

*Launch / URDF / worlds* · efficiency

`images` is declared in turtlebot3_ouster.launch.py:119-122 with the description "Run the ouster_ros os_image node (the sim image source). Set false on headless/CI runs that do not consume the image topics", and the identical rationale is repeated in ouster_standalone.launch.py:131-135 ("pass images:=false on headless/CI runs that don't consume the image topics ... to save the per-scan decode CPU"). The one place in the repo that is a headless CI run — docker/entrypoint.sh:27-28's `smoke` target — passes `headless:=true ray_mode:=raycast rviz:=false` and leaves `images` at its default `true`. docker/smoke_check.py subscribes only to `/sensor/lidar/lidar0/points` (line 33), so nothing consumes range/signal/reflec/nearir at all, yet os_image decodes every scan into four images plus camera_info regardless of subscribers.

**Failure scenario.** `docker run --rm gzouster smoke` on a CPU-only CI runner: the raycast worker and the unused os_image decoder contend for the same cores, real-time factor drops, and the 120 s SMOKE_TIMEOUT (entrypoint.sh:34) gets closer to expiring than it needs to — an intermittent smoke-test failure whose cause is a launch flag the repo already added for exactly this case.

**Fix.** Add `images:=false` to the `smoke` invocation at docker/entrypoint.sh:27-28. (The `drive|gui` case at :54-55 should keep images:=true — RViz's ouster.rviz displays those four topics.)

**Verifier.** Confirmed. docker/entrypoint.sh:27-28 launches with `headless:=true ray_mode:=raycast rviz:=false` and no `images` override; turtlebot3_ouster.launch.py:119-122 declares images with default_value='true' and the description 'Set false on headless/CI runs that do not consume the image topics'. docker/smoke_check.py:21-24 subscribes to a single PointCloud2 topic and nothing else. I also checked the pinned os_image node for subscriber gating — grep for get_subscription_count/count_subscribers in ouster-ros/src/os_image_node.cpp returns nothing, so it does decode every scan regardless of subscribers, as the launch comment claims. The 'intermittent smoke failure' consequence is speculative (smoke_check.py exits on the FIRST non-empty cloud, well inside the 120 s budget), but the one-line inconsistency between the flag's documented purpose and the repo's only headless CI invocation is real and actionable.

---

### `examples/urdf/sensor_stack.urdf.xacro:28` — sensor_stack.urdf.xacro shadows xacro's built-in `pi`, emitting a warning on every expansion

*Launch / URDF / worlds* · cleanup

`<xacro:property name="pi" value="3.14159265359"/>` redefines a symbol xacro already provides in its global table. Running `xacro examples/urdf/sensor_stack.urdf.xacro metadata_front:=... metadata_rear:=...` prints `warning: redefining global symbol: pi` on stderr. In sensor_stack.launch.py the expansion runs inside a `Command` substitution feeding `ParameterValue(..., value_type=str)` (lines 167-177), so this warning lands in the robot_state_publisher bring-up output on every launch. The redefinition is also a truncated literal (3.14159265359 vs math.pi) used at :72 for the rear sensor's yaw, and it is a string property rather than a numeric one, so it does not compose in arithmetic expressions the way the built-in does.

**Failure scenario.** `ros2 launch gz_sensors_ouster sensor_stack.launch.py` prints a xacro warning on every start, training users to ignore xacro stderr — the same channel that would carry a real macro error. A later edit that writes `rpy="0 0 ${pi/2}"` gets a TypeError on a string instead of 1.5708.

**Fix.** Delete line 28 and use xacro's built-in `pi` at line 72 (`rpy="0 0 ${pi}"` works unchanged), matching turtlebot3_ouster.urdf.xacro and ouster_standalone.urdf.xacro, neither of which redefines it.

**Verifier.** Primary claim verified empirically: sensor_stack.urdf.xacro:28 is `<xacro:property name="pi" value="3.14159265359"/>` and running `xacro examples/urdf/sensor_stack.urdf.xacro metadata_front:=... metadata_rear:=...` prints `warning: redefining global symbol: pi` to stderr (rc=0). It is the only file under examples/urdf/ that redefines pi, and the only use is the rear-sensor yaw at :72. The stated secondary failure is WRONG, though: I tested `${pi/2}` against a string-valued pi property and xacro evaluates it fine (emits 1.570796326795), so there is no TypeError waiting for a future edit — the only real cost is the stderr warning and the truncated literal. Severity low as filed.

---

### `examples/worlds/turtlebot3_ouster_sewer.sdf:22` — Sewer world has no directional light, so its NEAR_IR channel saturates to full albedo everywhere

*Launch / URDF / worlds* · correctness · reported as medium, downgraded by verifier

turtlebot3_ouster_sewer.sdf declares three `type="point"` lights (lines 22, 26, 30) and a deliberately dim `<scene><ambient>0.08 0.1 0.12 1</ambient>` (line 18) to sell a dark underground gallery. The raycast NEAR_IR model reads NONE of that. src/raycast_mirror.cpp:532-550 iterates lights and `return true`s (skips) on anything whose `Type() != sdf::LightType::DIRECTIONAL`, so the sun array keeps its initialiser at :531 — `{0,0,-1, sun_diffuse=0.0f, sun_ambient=1.0f}`. The kernel then computes `illum = sp.sun_ambient` and skips the Lambert term because `sp.sun_diffuse > 0.0f` is false (cuda/raycast_math.hpp:1569-1586), giving `nir = material.nir * 1.0` — the maximum the model can produce.

Net effect: the pitch-black sewer produces a NEAR_IR image that is both perfectly flat and BRIGHTER than every lit world. Compare turtlebot3_ouster_warehouse.sdf:21-27, whose directional skylight yields sun_diffuse = 0.7*mean(0.85,0.82,0.76) = 0.567 and sun_ambient = 0.3, capping illum at 0.867. The scene ambient value and all three lamps are pure decoration as far as the lidar is concerned.

**Failure scenario.** `ros2 launch gz_sensors_ouster turtlebot3_ouster.launch.py world:=sewer rviz:=true`. The RViz NearIrImage display shows a uniformly bright image with no lamp falloff, brighter than the same display under `world:=warehouse` in daylight. Anyone using the sewer world to develop or validate NEAR_IR-based perception in low light gets the exact opposite of the condition they are trying to simulate.

**Fix.** Add a low-intensity `<light type="directional">` to the sewer world (e.g. `<diffuse>0.12 0.12 0.14 1</diffuse>` pointing down the gallery) so sun_diffuse/sun_ambient land at 0.084/0.3 instead of 0.0/1.0, or change the no-sun fallback in raycast_mirror.cpp:531 to a dim ambient (and read `<scene><ambient>`) rather than 1.0. Either way, document in the world header that point lights do not drive NEAR_IR.

**Verifier.** Mechanism confirmed end to end. turtlebot3_ouster_sewer.sdf has only three `type="point"` lights (:22, :26, :30) — grep for 'light type' across examples/worlds shows it is the ONLY world of nine without a directional light. src/raycast_mirror.cpp:531 initialises sun[5] = {0,0,-1, 0.0f, 1.0f} and the Each<Light> lambda at :532-535 returns true (skip) for any non-DIRECTIONAL light, so sun_diffuse stays 0 and sun_ambient stays 1.0. cuda/raycast_math.hpp:1569-1571 then sets illum = sun_ambient and skips the Lambert branch because `sp.sun_diffuse > 0.0f` is false, giving nir_val = winning_material.nir * 1.0 at :1586 — the maximum. Warehouse's skylight (turtlebot3_ouster_warehouse.sdf:21-26, diffuse 0.85/0.82/0.76) gives 0.7*0.81 = 0.567 + 0.3 ambient, so the dark sewer is indeed the brightest-illuminated world. Two corrections to the write-up: the image is NOT 'perfectly flat' — nir still varies per surface because winning_material.nir comes from the response texture/diffuse (cuda/raycast_math.hpp:669, :681-682) — only the illumination term is flat; and the no-sun fallback is explicitly documented as intended at raycast_mirror.cpp:524-530 ('with no sun the channel falls back to ambient-only (nir = albedo)'). So this is a world-asset authoring gap in a demo scene, not a code defect.

---

### `test/test_launch_files.py:108` — test_launch_files.py green-lights the dead metadata parameter by counting a string

*Launch / URDF / worlds* · test-coverage · reported as medium, downgraded by verifier

`test_ouster_consumers_preseed_metadata_before_fast_bag_playback` is documented as "Avoid losing early packets while a one-shot metadata callback starts" but its entire body is `assert src.count("'metadata': metadata") >= 2`. It is a substring count over the launch source. It cannot distinguish a parameter the consumer node actually declares from one rclcpp throws away (which is the real situation — see the os_cloud/os_image finding), and it will keep passing forever regardless of what ouster-ros does with the parameter. Neighbouring tests have the same shape (`test_ouster_consumers_preserve_packet_acquisition_time` at :85 only greps for the string 'TIME_FROM_INTERNAL_OSC'), but this one is the case where the grep is currently asserting a behaviour that does not happen.

**Failure scenario.** Someone removes `metadata` from os_cloud's parameter set upstream, or (as today) it was never there: the test stays green and the suite reports that fast-bag pre-seeding is verified. Conversely, renaming the local variable from `metadata` to `meta_path` — a pure no-op refactor — turns the test red for no behavioural reason.

**Fix.** Either drop the test along with the dead parameter, or make it assert something real: run `ros2 param describe` against a spawned os_cloud/os_image in an integration test, or at minimum assert against the declared-parameter list extracted from the pinned ouster-ros source, so the test fails when the parameter does not exist.

**Verifier.** Confirmed: test/test_launch_files.py:104-108 is exactly `assert src.count("'metadata': metadata") >= 2` over the launch source text, parametrized over EXAMPLE_LAUNCHES (:17-21), and its docstring claims it verifies pre-seeding before fast bag playback — a behaviour that does not exist per finding [0]. The rename-breaks-it claim also holds: sensor_stack.launch.py only satisfies the count because the helper parameter is literally named `metadata` (_os_cloud at :52/:85, _os_image at :99/:117). Severity corrected down: every test in this file is a source-text grep by design (module docstring :1-5; the neighbouring TIME_FROM_INTERNAL_OSC check at :85 is the same shape), so this is an inherited convention, and the test's only real cost is locking in the dead parameter from [0]. It ships no wrong data and hides no live bug.

---

### `cuda/raycast_math.hpp:540` — rcHitUv treats a plane's size == 0 as 1e-12 while rcHitPlane and addInstance treat it as 1e6, collapsing an infinite plane's response map to one texel

*Raycast math* · numerics · reported as medium, downgraded by verifier

`size[0] == 0` on a `kPlane` is the codebase's own encoding for "infinite", and two of the three places that read it agree on that:
- `rcHitPlane` (raycast_math.hpp:401-402): `const float hx = (size[0] > 0.0f) ? size[0] : kRcHugeExtent;` (1e6)
- `Scene::addInstance` (cuda/raycast_scene.cpp:269-270): the identical ternary for the local AABB
- `test/test_raycast.cpp:487` builds one deliberately: `const float psize[3] = {0.0f, 0.0f, 0.0f};  // infinite plane`

`rcHitUv`'s kPlane branch (raycast_math.hpp:540-543) instead does `fmax_(inst.size[0], 1.0e-12f)`, producing `u = 0.5 + p.x / 2e-12`.

Measured on an infinite plane with a 4x4 response map attached:
  hit x =  0.00 -> u = 0.5,     wrapped 0.5
  hit x =  0.50 -> u = 2.5e+11, wrapped 0.0
  hit x =  3.00 -> u = 1.5e+12, wrapped 0.0
  hit x = 12.50 -> u = 6.25e+12, wrapped 0.0
(for contrast, the same plane with half-extents 5 gives 0.5 / 0.55 / 0.8)

Once |u| exceeds 2^24 every float is an exact integer, so `u - floor_(u)` in `rcSampleResponse` (raycast_math.hpp:650) is identically 0 and the whole plane samples texel column 0. v is 3.5e11 for every hit and collapses the same way. The material becomes one constant RGBA sample instead of a spatially varying response, with no warning anywhere — `rcHitUv` returns `true`, so `rcMaterialAtHit` (raycast_math.hpp:690) accepts the garbage rather than falling back to the scalar SDF material.

**Failure scenario.** A ground plane authored as `<plane><size>0 0</size></plane>` (the documented infinite form, exercised by test_raycast.cpp:487) whose PBR albedo has an aligned `.ouster.png` companion. Every hit on the ground, at every range and azimuth, returns the single texel at UV (0,0) — a flat, uniform diffuse/NIR/specular response instead of the per-pixel map. On a finite plane the same map varies correctly, so the failure is silent and geometry-dependent.

**Fix.** Use the same rule as rcHitPlane in the kPlane branch of rcHitUv (raycast_math.hpp:540-541): `const float hx = (inst.size[0] > 0.0f) ? inst.size[0] : kRcHugeExtent;`. Better still, `return false` when either half-extent is non-positive so an unbounded plane falls back to the scalar material via rcMaterialAtHit's existing early-out, and factor the `(size > 0) ? size : kRcHugeExtent` expression into one shared helper so rcHitPlane, addInstance and rcHitUv cannot drift again.

**Verifier.** The three-way inconsistency is exactly as described. rcHitPlane uses `(size[0] > 0.0f) ? size[0] : kRcHugeExtent` (raycast_math.hpp:401-402, kRcHugeExtent = 1.0e6f at line 205), Scene::addInstance uses the identical ternary for the local AABB (raycast_scene.cpp:268-272), and the header documents 'finite half-extents size[0..1] (0 -> infinite)' at raycast_math.hpp:28 — but rcHitUv's kPlane branch uses fmax_(inst.size[0], 1.0e-12f) (raycast_math.hpp:540-543), giving u = 0.5 + p.x/2e-12. Above 2^24 every float is an integer, so `u - floor_(u)` in rcSampleResponse (raycast_math.hpp:650-651) is identically 0 and the whole surface samples one texel column, with rcHitUv still returning true so rcMaterialAtHit (raycast_math.hpp:690) accepts it instead of falling back to the scalar material. Primitives are eligible for response maps — raycast_mirror.cpp:185 sets `has_texcoords = true` for analytic primitives, so the map-load guard at raycast_mirror.cpp:296 does not exclude planes. Severity corrected from medium to low: reachability requires a deliberately degenerate `<size>0 0</size>` plane (sdf::PlaneShape defaults to 1x1, and every shipped world uses a finite size — depthcam_derisk.sdf:58 and ouster_demo.sdf:67 use 100x100, ouster_showcase.sdf:123 uses 300x300, ouster_smoke.sdf:131 uses 80x80) combined with an aligned .ouster.png companion; the zero-size form appears only in test_raycast.cpp:487.

---

### `test/test_dispatch.cpp:78` — test_dispatch's two auto-selection tests assert a condition that cannot fail

*Test quality* · test-coverage · reported as medium, downgraded by verifier

`Dispatch.UnknownEnvFallsBackToAuto` (line 78) and `Dispatch.AutoSelectionAlwaysResolves` (line 88) both assert only `EXPECT_STRNE(proc.backendName(), "none")`. `RayProcessor::backendName()` (cuda/ray_processor_dispatch.cpp:182-185) returns "none" only when `backend_` is null, and `pickBackend()` (:61-83) either returns a non-null backend or throws `std::runtime_error`. `backend_` is therefore never null after construction, so the string can never be "none" and the assertion can never fire. Both tests reduce to `RayProcessor proc;` with no assertion at all.

These two tests are the claimed remediation for AUDIT.md H1 ("GPU backends are never exercised; backend selection is untested") — the table at AUDIT.md:111 says "Added test_dispatch (backend selection coverage)". Only `EnvForcesCpuBackend` (line 62-68) carries a real assertion, and it covers the one branch that was never in doubt.

**Failure scenario.** Change `readPrefEnv()` (ray_processor_dispatch.cpp:44-47) so an unrecognised GZ_OUSTER_BACKEND returns `Pref::kCuda` instead of `Pref::kAuto` — i.e. a typo'd env var silently selects the wrong backend instead of auto. On a CPU-only build `tryBackend(kCuda)` returns nullptr, the loop at :75 lands on CPU, `backendName()` is "cpu", and both tests pass. On a CUDA machine the user who typed `GZ_OUSTER_BACKEND=cpuu` silently gets the GPU kernel; still green. Likewise, reordering the preference loop at :75 to put CPU first (silently disabling every GPU backend) passes both tests.

**Fix.** Assert on the actual selection rather than on the impossible sentinel: capture `backendName()` with the env unset, then assert `UnknownEnvFallsBackToAuto` produces the *same* string (that is what "falls back to auto" means and it is falsifiable in every build). Add a test that `GZ_OUSTER_BACKEND=cuda` on a CPU-only build yields `usesCpuFallback()==true` and that the name is one of the four known values.

**Verifier.** Confirmed. RayProcessor::backendName() (cuda/ray_processor_dispatch.cpp:182-185) returns "none" only when backend_ is null; backend_ is initialised from pickBackend() in the ctor init list (:89-90), and pickBackend (:61-83) either returns a non-null unique_ptr or throws std::runtime_error (:81). So after construction backend_ is never null and EXPECT_STRNE(proc.backendName(), "none") at test_dispatch.cpp:78 and :88 cannot fire. The mutation is real too: readPrefEnv (:36-48) returning kCuda instead of kAuto for an unrecognised value leaves both tests green on a CPU-only build (tryBackend(kCuda) → nullptr → loop at :75 lands on CPU). Severity corrected down: these two tests are redundant, not wrong, and the suite is not as hollow as implied — EnvForcesCpuBackend (:62-68) asserts backendName()=="cpu" and usesCpuFallback(), and ProcessRawThroughWrapper (:103) plus SeededCpuNoise (:136) drive real end-to-end dispatch. This is a cleanup, worth one line of code.

---

### `test/test_packet_pacing.cpp:68` — test_packet_pacing's rate-invariance loop compares columnTimestampNs to itself

*Test quality* · test-coverage · reported as medium, downgraded by verifier

In `PacketPacing.PlaybackRateChangesDeliveryButNotAcquisitionTimestamps`, `reference_stamps` is filled at lines 60-63 by `columnTimestampNs(scan_end_ns, nominal.count(), column, 4)`. The inner loop at lines 68-72 then asserts `columnTimestampNs(scan_end_ns, nominal.count(), column, 4) == reference_stamps[column]` — byte-identical arguments, no state touched between the two calls, and `rate_case` appears nowhere in the call. The assertion is `f(a) == f(a)` and can only fail if `columnTimestampNs` is nondeterministic. `columnTimestampNs` (src/scan_timing.hpp:16-19) takes no playback-rate parameter, so the "not acquisition timestamps" half of the test name is structurally unfalsifiable.

**Failure scenario.** Rewrite `columnTimestampNs` to stamp columns from the *start* of the scan instead of the end (`ordinal = column` rather than `column + 1`, scan_timing.hpp:29) — a one-column rolling-shutter skew across the whole point cloud. Lines 68-72 still pass identically for all five rate cases; only the separate test_scan_timing.cpp:14-16 catches it. In other words this loop contributes nothing that test_scan_timing does not already cover, while its name advertises a rate-invariance property nobody is checking.

**Fix.** Either delete the inner loop (the drain-span assertion at line 66 is the real content, and scan_timing is covered by test_scan_timing.cpp), or make it falsifiable: derive the packet stamps through the code path that actually sees the observed wall-clock arrival — feed the encoder/drain the observed span and assert the emitted column stamps equal the reference computed from sim time alone.

**Verifier.** Confirmed literally. reference_stamps is filled at test_packet_pacing.cpp:60-63 with columnTimestampNs(scan_end_ns, nominal.count(), column, 4); the inner loop at :68-72 calls it with byte-identical arguments and compares to that same vector. rate_case (:65) appears nowhere in the call, and columnTimestampNs (src/scan_timing.hpp:16-37) is a pure function of (scan_end_ns, scan_period_ns, column, columns) with no rate parameter and no state, so the assertion is f(a)==f(a). The `ordinal = column + 1` mutation at scan_timing.hpp:29 would indeed pass here (both sides move together). Severity corrected down hard: the enclosing test's real content — EXPECT_EQ(packetBatchDrainSpan(...), expected_spans[rate_case]) at :66 — is a genuine, falsifiable assertion across five rate cases, and scan_timing itself is covered by test_scan_timing.cpp. This is a redundant loop with an over-promising test name, not a latent bug.

---

## Refuted (12)

Raised by a finder, killed by the verifier. Recorded so they are not re-reported.

**`CMakeLists.txt:420`** — CMake installs 4 internal backend headers (~2200 lines of GPU kernel math) into the public include dir where nothing can reach them  
The central claim is wrong and the proposed fix would break downstream builds. include/gz_gpu_ouster_lidar/ray_processor.hpp is a PUBLIC installed header that forward-declares rc::SceneView, rc::InstanceXform and rc::ScanParams (ray_processor.hpp:96-99) and then uses them by reference in the public castScan (:170-181) and castScanProcessed (:187-205) signatures. Its own comment at :94-95 says 'full definitions in raycast_scene.hpp / raycast_math.hpp'. A downstream caller cannot construct a rc::ScanParams or a rc::SceneView to pass without those definitions, and cuda/raycast_scene.hpp:24 includes raycast_math.hpp, which includes ray_processor_math.hpp (raycast_math.hpp:18) — so three of the four 'pointless' headers are exactly the transitive closure the public raycast API needs, and reducing install(FILES) to imu_noise.hpp (the proposed fix) would make castScan uncallable from an installed tree. The 'nothing can reach them' claim is also literally false: they install into include/gz_gpu_ouster_lidar/, which is on the consumer's include path. What survives is far smaller than reported: panel_layout.hpp is the one entry no public header needs (ResamplePanel/ResampleParams are fully defined in ray_processor.hpp:11-51), and the CMakeLists.txt:408-409 comment mis-describes the mechanism as 'quote-include' when it is really 'forward-declared type needs its definition'. That is a stale-comment nit, not a medium API-surface defect.

**`examples/worlds/depthcam_derisk.sdf:1`** — Resolved phase-0 de-risk world + launch file are still shipped to users and parametrized into every structural test  
Style opinion with a failure scenario that cannot occur today. The file header (examples/worlds/depthcam_derisk.sdf:15-18) explicitly states the experiment is RESOLVED and that the world 'is kept as a historical harness' — retention is a deliberate, documented choice, not an oversight. The claim that it is 'parametrized into every structural assertion' in test_worlds.py is false: only 4 of the suite's parametrizations use WORLD_NAMES (test/test_worlds.py:24 parses, :29 has-physics, :65 physics pacing, :168 mirrorable geometry) — all genuinely universal invariants that any shipped SDF should satisfy — while every world-specific assertion already uses an explicit allowlist (:34-41 imu, :55 and :60 RAYCAST_WORLDS altimeter), i.e. the suite is already structured so a tightening lands on the enumerated worlds, not the harness. The stated failure requires a hypothetical future test change that has not been made. The only concrete residue is the four stale GpuRays/cubemap comments (sdf:6,9,10,18; launch.py:6,14,17), which is already finding [4]'s territory.

**`test/test_raycast.cpp:142`** — BeamOriginParallaxMatchesXyzLut tests only elevation 0, where the beam-origin term is algebraically zero  
The stated failure scenario is false and the central claim is wrong for the path this test exercises. `cast()` (test_raycast.cpp:55-66) calls rc::castScan, i.e. the raycast path in raycast_math.hpp, which never touches applyBeamOrigin - so the proposed mutation at ray_processor_math.hpp:272 is not a regression this test could ever have been expected to catch, and it does NOT survive 'the whole suite': test_resample.cpp:292-326 (BeamOriginReportsXyzLutRange) drives processRaw with beam_origin_m=0.05 and asserts range == 10000 mm +/- 30, so dropping the `+ beam_origin_m` yields 9950 and fails that test - the comment at :316-321 says so explicitly. The 'vacuous loop' claim is also wrong: in raycast mode the origin is n*[cos enc, sin enc, 0] (raycast_math.hpp:1410-1411) so t0 = 4 - n and the reported r = t0 + n_off; deleting or sign-flipping the `+ n_off` gives 4 - 2n = 3.945 for n=0.0277 and blows the 1e-4 tolerance at test_raycast.cpp:146. The loop over three n values does pin that compensation. What genuinely remains is much narrower and lower severity: at el=0, az=0, m=0 the encoder unit vector and the beam direction are colinear, so the test cannot distinguish an origin placed on the encoder circle from one placed along the beam, and no test covers a non-zero elevation or builds a real XYZLut.

**`cuda/panel_layout.cpp:228`** — countUncoveredRays flips the beam-azimuth sign relative to the sampler it is supposed to validate  
The sign discrepancy is real - panel_layout.cpp:228-229 computes `beam_az_deg[beam] - m*deg_per_col` while every consumer uses rpmath::beamRayAzimuthDeg = `-beam_az_deg - m*deg_per_col` (ray_processor_math.hpp:186-190) - but the failure scenario cannot occur, so this is a style/maintenance inconsistency, not a bug. buildOusterPanelLayout tiles the FULL 360 deg azimuth circle (4 panels at 90 deg spacing with half_az = 45 + 2 + px_pad, panel_layout.cpp:175-184; or 8 at 45 deg spacing plus a zenith cap, :189-202) and does not return a layout until coversBand (:109-124, called at :211) has verified that EVERY azimuth from 0 to 360 in 0.25 deg steps projects into some panel, at every elevation in [min_alt - 1.5, max_alt + 1.5]. There is therefore no azimuth pad to fall past: for a beam elevation inside the band - guaranteed, since meta.min_alt/max_alt already carry kBeamMarginDeg (ouster_metadata.cpp:216-220) - coverage is azimuth-complete, so countUncoveredRays returns 0 for +12 deg offsets exactly as it does for -12 deg. `grow` (:166) scales tan_h symmetrically and cannot make the pad one-sided. Fixing it to use the shared helper is still worth doing; it just changes no output today.

**`cuda/panel_layout.cpp:228`** — countUncoveredRays uses the wrong beam-azimuth sign — the exact bug beamRayAzimuthDeg's doc comment warns about  
The sign divergence at panel_layout.cpp:227-228 (`beam_az_deg[beam] - m*deg_per_col`) versus rpmath::beamRayAzimuthDeg (ray_processor_math.hpp:186-190, `-beam_az_deg - m*deg_per_col`) is real, but the claimed failure cannot occur. The panel rig is azimuthally complete by construction AND verified: buildOusterPanelLayout only returns a layout when coversBand passes (panel_layout.cpp:211), and coversBand (panel_layout.cpp:106-121) sweeps az from 0 to 360 in 0.25 deg steps for every elevation in [min_alt_deg, max_alt_deg] — no azimuth is ever left uncovered inside the band. The cylindrical rig is 4 panels at 90 deg spacing with half_az = 45 + 2 + px_pad (panel_layout.cpp:170-181) and the dome is 8 panels at 45 deg with half_az = 22.5 + 2 + px_pad plus a zenith cap (panel_layout.cpp:186-202), so coverage is a function of elevation only. panelForDirection (ray_processor_math.hpp:167-174) uses no range/far_clip term, so nothing changes between build time and the panel_rig.cpp:49 call. And every beam elevation is strictly inside the verified band: ouster_metadata.cpp:218-219 sets min_alt = min(beam_alt) - kBeamMarginDeg and max_alt = max(beam_alt) + kBeamMarginDeg. Consequently countUncoveredRays returns 0 for both azimuth conventions, for any metadata (symmetric or asymmetric offsets); neither the silent-miss case nor the spurious-warning case is reachable. It is a one-line convention inconsistency worth cleaning up, not a behavioural bug — severity low.

**`cuda/raycast_math.hpp:739`** — rcApparentReflectance is unbounded: kd + ks pushes glossy paint into the reflectivity byte band reserved for retroreflectors  
The arithmetic is right (rcApparentReflectance at raycast_math.hpp:729-743 has no clamp; reflectivityToByte at ray_processor_math.hpp:405-412 maps 1.4382 to 100 + log2(1.4382)*22 = 111) but this is a modelling opinion, not a defect. Nothing is violated: no invariant, comment or doc anywhere states rho_app <= 1 — docs/MODEL_REFERENCES.md:216 writes the formula as-is, and reflectivityToByte explicitly handles rv > 1 and saturates at kReflByteMax, so there is no overflow, UB or corruption. Physically a monostatic specular lobe at near-normal incidence DOES return more power than a white Lambertian target, which is precisely why a real OS-1 reads high reflectivity off mirrors, wet asphalt and sign faces; capping it at 1.0 as proposed would remove modelled behaviour, not fix a bug. The proposed clamp `fmin_(rho, fmax_(diffuse, 1.0f))` is a design change requiring a physics argument, not a correctness fix. Also note the glass_pane 1.2150 -> byte 106 example is not evidence for this finding at all: it is the back-face self-hit of finding [0] (the continuation return at raycast_math.hpp:1486-1490 is multiplied by tau^2 but never by (1-tau)); with [0] fixed the pane's front face yields 1.5*0.1 = 0.15, well inside the Lambertian band. Downgraded to low as a documentation/tuning note.

**`cuda/CMakeLists.txt:108`** — -fsycl is applied target-wide, silently swapping the CPU backend's math from std:: to sycl::  
The cited fact is right (cuda/CMakeLists.txt:108-109 is target-wide `target_compile_options(... PRIVATE $<$<CXX_COMPILER_ID:IntelLLVM>:-fsycl>)`, versus the file-scoped `set_source_files_properties(ray_processor_hip.cpp ... LANGUAGE HIP)` at :87-88, and ray_processor_math.hpp:39/78 does key off SYCL_LANGUAGE_VERSION), but the conclusion is backwards and the failure scenario cannot occur. (1) The proposed fix would introduce an ODR violation. rpmath::gzm::cos_ etc. are `inline` free functions with external linkage (ray_processor_math.hpp:75-102) consumed by every TU in the same static library `gz_gpu_ouster_lidar_cuda`. If -fsycl were scoped to ray_processor_sycl.cpp only, that TU would emit sycl::-bodied definitions of the exact same inline symbols that ray_processor_cpu_impl.cpp / panel_layout.cpp / raycast_scene.cpp emit with std:: bodies — one definition would win at link time arbitrarily, which is strictly worse and non-deterministic. Applying the flag target-wide is the configuration that keeps the shim self-consistent across the library. (2) The concrete failure named — 'a golden/regression digest recorded on a gcc build fails on the icpx build' — references a test that does not exist: `grep -rn 'digest|golden' test/ CMakeLists.txt .github` returns nothing. AUDIT.md's digest was a one-off local check by the prior auditor, not a checked-in gate. (3) The ≤4 ULP divergence is asserted, not demonstrated; in DPC++ and AdaptiveCpp the host overloads of sycl::cos/sqrt/log forward to the C library, so no divergence was shown. What is left is a loosely-worded comment at ray_processor_math.hpp:19-21 — a documentation nit, not a bug.

**`cuda/panel_layout.cpp:228`** — countUncoveredRays computes beam azimuth with the opposite sign from every backend kernel  
The sign discrepancy is real — panel_layout.cpp:228-229 writes `beam_az_deg[beam] - m*deg_per_col` while every sampler calls rpmath::beamRayAzimuthDeg (ray_processor_math.hpp:186-189) returning `-beam_az_deg - m*deg_per_col` (ray_processor_cpu_impl.cpp:222-223, ray_processor_sycl.cpp:528-529, raycast_math.hpp:1404-1406) — but the claimed failure is impossible, and I proved it empirically rather than by argument. The rig covers the FULL 360 deg of azimuth by construction: cylindrical is 4 panels of half_az = 45+2+px_pad at 90 deg spacing (panel_layout.cpp:175-184), dome is 8 panels of half_az = 22.5+2+px_pad at 45 deg spacing plus a zenith cap (:189-202). More decisively, buildOusterPanelLayout only returns a non-empty layout after coversBand (:109-124, called at :211) has verified every direction on a 0.25 deg grid over el in [min_alt,max_alt] x az in [0,360). So no azimuth offset of any magnitude can move a ray out of coverage; only an out-of-band ELEVATION can, and elevation is untouched by the azimuth sign. I compiled panel_layout.cpp standalone against a correct-sign reimplementation: for the cylindrical OS0 rig (-46..46, H=32, W=1024) and the dome rig (-1..91, H=64, W=1024), with per-beam azimuth offsets of 0/3/8/20/45/90 deg, BOTH formulas return 0 uncovered in every case; with deliberately out-of-band beams (el -55..55 on a -46..46 rig) both return the same non-zero count (944 vs 944 at offset 0, 936 vs 936 at offset 8) — i.e. the warning at panel_rig.cpp:51-56 fires identically either way. The reviewer's 'whole beam row of zeros with no log line' cannot happen. Genuine but cosmetic: the validator should call the shared helper so it cannot drift.

**`cuda/ray_processor_hip.cpp:343`** — HIP sizes the ray-process kernel grid from rp.H*rp.W while the kernel bounds itself with pp.H*pp.W  
The lines say what is claimed (ray_processor_hip.cpp:332 out_n = rp.H*rp.W, :343 grid_r, reused at :344 and :354 while rayProcessKernelHip is passed pp.H/pp.W at :361; ray_processor_cuda.cu:347-348 does compute its own grid from p.H*p.W). But this is unreachable and the 'backend divergence' framing is wrong. Unreachable: the only production caller is gz_gpu_ouster_lidar_system.cpp:1043, which passes rig_->resampleParams() (built from meta.H/meta.W at panel_rig.cpp:30-31) together with pp from makeRayProcessParams(), which sets pp.H = meta_->H / pp.W = meta_->W at :964-965; every test caller (test_resample.cpp:126,156,229,255,284,313,369,371 and test_dispatch.cpp:127) passes layout.rp built from the same H,W it passes to noNoise(H,W). Nothing in-tree can produce a mismatch. Wrong framing: the CPU backend has exactly the same implicit coupling — ray_processor_cpu_impl.cpp:206-211 sizes depth_buf from rp.H*rp.W and then hands it to processCpu, which iterates p.H*p.W (:28-30, :44) — so it would over-read identically. CUDA's separate grid computation is not a validation either; as the finding itself concedes, CUDA would just OOB-write instead. So three of four backends share the assumption and none validates it; there is no HIP-specific defect, only a missing precondition on an internal interface. Adding the rp/pp consistency check at ray_processor_dispatch.cpp:109 is reasonable hardening, but MEDIUM for an input no caller can generate is inflated.

**`examples/worlds/ouster_demo_panels.sdf:102`** — ouster_demo_panels.sdf claims geometry "byte-identical" to ouster_demo.sdf but adds a laser_retro the raycast world deliberately omits  
The diff is real (I diffed both files from <model name="ground_plane"> to EOF: the only model-level delta is the extra <laser_retro>0.5</laser_retro> at ouster_demo_panels.sdf:102), but the finding misreads the comment and overstates the consequence. (1) The comment at :79-83 claims *geometry* is byte-identical — geometry is byte-identical; laser_retro is a material/response tag, and the very same comment goes on to say the laser_retro tags are inert in panels mode and 'kept solely to mirror the raycast world', i.e. carrying them is the stated intent. (2) The failure scenario is a behavioural no-op: cuda/ray_processor_math.hpp:69 sets kDefaultRetro = 0.5f and src/raycast_mirror.cpp:663-664 sets sp.fallback_retro = reflectivityByteToRetro(base_reflectivity) with base_reflectivity defaulting to 50 → rho 0.5, exactly the value the tag spells out (ouster_demo.sdf's own comment says so). So even the hypothetical copy-back the finding fears produces identical output at the default configuration. This reduces to a comment-wording nit with a speculative, output-neutral failure mode.

**`test/test_worlds.py:16`** — test_worlds.py's glob-driven parametrize turns a missing worlds directory into five silent skips  
The mechanism is real (WORLD_NAMES at test_worlds.py:16 is glob-derived and an empty parametrize set is reported as SKIPPED) but the stated failure scenario cannot occur: moving/renaming examples/worlds does NOT leave the suite green. test_worlds.py is full of hard-coded world paths that would raise FileNotFoundError — the explicit-list parametrize at :34-41 (eight named .sdf files), RAYCAST_WORLDS at :48-52, and direct reads at :101, :147, :150, :207-210, :239, :252, :309, :374, :380, :384, :388. The same holds for test_launch_files.py: EXAMPLE_LAUNCHES is hard-coded at :17-21 and read at :26-38, so a moved examples/launch fails loudly regardless of LAUNCH_NAMES. The CI guard at .github/workflows/ci.yaml:133-138 never has to catch it. The secondary claim is likewise already guarded: if the brick_graffiti.png reference were removed from all three worlds, test_response_textured_worlds_reference_the_visible_companion_base (:206-213) asserts `'../media/materials/textures/brick_graffiti.png' in text` for each of the three named worlds and fails before the vacuous loop at :217-232 matters. What remains is a defensive one-liner, not a defect.

**`include/gz_gpu_ouster_lidar/gz_gpu_ouster_lidar_system.hpp:238`** — render_busy_mtx_ is recursive on a justification that the code does not implement  
The factual observations hold — grep shows render_busy_mtx_ is acquired at exactly three sites (dtor 90, OnRender 627, OnRenderTeardown 686), the nested callback path panel_rig.cpp:150-151 -> onPanelFrame (174-219) takes no lock and says so at 178-180, and the registered lambda (panel_rig.cpp:126-129) calls only PanelRig::onPanelFrame, never back into the plugin — so nothing re-enters the mutex today. But that makes this a comment-accuracy nit, not a defect: a recursive_mutex behaves identically to std::mutex on every path that actually exists, and the 'failure' is explicitly conditioned on a gz-sim feature that does not exist (the dtor comment at 99-103 says there is 'no plugin-visible hook to schedule the destroy on the render thread'). A defect whose trigger is a hypothetical future upstream API is a style preference dressed as a bug; if anything the recursive choice is the conservative one given gz-rendering does fire NewDepthFrame synchronously and a future callback that did touch plugin state would need it. Refuted as a bug; at most a low-value doc fix to the paragraph at hpp:237-242.
