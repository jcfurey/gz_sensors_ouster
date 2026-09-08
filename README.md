# gz_sensors_ouster

Gazebo system plugin (Harmonic / Ionic / Jetty) that simulates Ouster
LiDAR sensors (OS0, OS1, OS2, OSDome and OS1 MAX) across Gen1 through Rev8,
with GPU-accelerated ray casting, product-calibrated detection and noise, and
native Ouster packet output. Downstream nodes like
`ouster_ros` `os_cloud` consume the packets identically to real
hardware -- no driver changes needed.

## Features

- Per-beam elevation and azimuth geometry from Ouster calibration JSON
- Multi-vendor GPU post-processing: **CUDA** (NVIDIA), **HIP** (AMD incl.
  APUs, with unified-memory fast path), **SYCL** (Intel iGPU + Arc, via
  oneAPI DPC++ or AdaptiveCpp). Automatic CPU fallback when no GPU
  toolchain is compiled in or no device is found at runtime.
- Native `PacketMsg` encoding via Ouster SDK `PacketWriter` (RANGE, SIGNAL,
  REFLECTIVITY, NEAR_IR channels)
- Exported `gz_sensors_ouster_core` ROS facade over the pinned, ROS-free
  `ouster_sim_core` submodule shared with AGX. Metadata, product profiles,
  packet bytes and pacing use the same implementation. Both providers run
  the shared conformance fixtures.
- Simulated IMU packets from Gazebo's IMU sensor (optional, auto-detect)
- Noise parameters reconfigurable at runtime via `ros2 param set`
- Latched metadata republishing for rmw_zenoh_cpp compatibility
- Rolling-shutter packet timing via drain thread
- Works on **any GPU** for ray casting (OGRE2 + OpenGL is vendor-agnostic);
  the GPU noise path is additionally accelerated on NVIDIA/AMD/Intel.

## Shared core contract

Clone with `--recurse-submodules`, or run `git submodule update --init --recursive`
after checkout, including before building a Docker image from this directory.
`third_party/ouster_sim_core` is embedded plain CMake; Gazebo supplies the SDK
through `ouster_ros`, while AGX supplies its native SDK in a separate process.
Only the parent facade library is installed. Public core headers are installed
alongside the Gazebo facade headers.

The production metadata loader rejects dual-return UDP profiles until the
common frame contract supports their secondary channels. Supported primary
profiles are LEGACY, RNG19_RFL8_SIG16_NIR16 and RNG15_RFL8_NIR8. Optional wire
fields are written only when present; modern packets include identity and CRC.
RNG15 ranges must be multiples of 8 mm and at most 262.136 m. Invalid ranges
fail before a batch enters the delivery queue.

CPU/CUDA/HIP/SYCL optical stages use the common dense-buffer presence contract:
finite zero means an authored black surface, invalid or absent input uses a
fallback, and true misses have four zero channels. Versioned noiseless fixtures
compare production backend outputs with the scalar core. Stochastic draws
remain backend-specific. Gazebo's depth API still supplies a single range;
transferring richer effects must preserve the core's separate physical path
length and reported range before claiming full noisy optical equivalence.

`test_core_conformance` exercises the production loader/encoder and bounded
frame delivery, including pause, rewind, restart and blocked sinks.
`test_optical_conformance` runs every available backend (unavailable devices
are reported as skipped). The shared contract suites run against Gazebo's SDK
provider. `test_zenoh_packet_burst` uses an owned private router to verify 1,024
large packets, late metadata delivery, and simulation-thread progress while
packet transport is blocked. Rendering, hardware calibration and advanced
material/IMU transfers remain engine-specific work.

## Published Topics

All topics are published under the configured `<sensor_name>` prefix.

For example, with `<sensor_name>/sensor/lidar/lidar0</sensor_name>`:

| Topic | Type | Rate | Description |
|-------|------|------|-------------|
| `.../lidar_packets` | `ouster_sensor_msgs/PacketMsg` | lidar_hz | Native Ouster lidar packets |
| `.../metadata` | `std_msgs/String` | Latched | Ouster calibration JSON |
| `.../range_image` | `sensor_msgs/Image` | lidar_hz | Range in mm (mono16). Native; off unless `publish_native_images` (see below) |
| `.../signal_image` | `sensor_msgs/Image` | lidar_hz | Signal photon counts (mono16). Native; off unless `publish_native_images` |
| `.../reflec_image` | `sensor_msgs/Image` | lidar_hz | Reflectivity (mono16). Native; off unless `publish_native_images` |
| `.../nearir_image` | `sensor_msgs/Image` | lidar_hz | Near-IR (mono16). Native; off unless `publish_native_images` |
| `.../camera_info` | `sensor_msgs/CameraInfo` | lidar_hz | Range-image camera metadata (H×W, frame_id). Native; off unless `publish_native_images`. `distortion_model` is the non-standard string `equirectangular`: u is linear in azimuth, v linear in elevation; fx/fy in K are pixels-per-radian. Standard pinhole/fisheye consumers must not reproject with it. |
| `.../imu_packets` | `ouster_sensor_msgs/PacketMsg` | imu_hz | Native Ouster IMU packets (if IMU enabled) |
| `.../imu` | `sensor_msgs/Imu` | imu_hz | Standard ROS IMU message (if IMU enabled) |

The native image + CameraInfo topics are **off by default** (`publish_native_images=false`):
in sim the ouster_ros `os_image` node is the single image source, driven by the same
`lidar_packets` + `metadata` it consumes on hardware, so what you see in sim matches the
real driver (auto-exposure, beam-uniformity, destaggering). The plugin's native renditions
are raw (no auto-exposure) and draw noise independently, so running both would publish two
divergent versions of each topic. Set `publish_native_images=true` only to A/B the native
images. When enabled, image/CameraInfo and IMU topics are still published lazily — only when
a subscriber is present.

## Supported ROS 2 / Gazebo versions

The build is **version-agnostic**: `CMakeLists.txt` discovers Gazebo through the
`gz_*_vendor` CMake shims (`find_package(gz-sim)` with no version suffix), so
the same source builds against whichever Gazebo your ROS 2 distro vendors. CI
builds all three:

| ROS 2 distro | Gazebo release | gz-sim | Status |
|--------------|----------------|--------|--------|
| Jazzy        | Harmonic       | 8      | Supported (CI, required) |
| Kilted       | Ionic          | 9      | Supported (CI, required) |
| Lyrical      | Jetty          | 10     | Supported (CI, advisory while the distro is new) |

To build against a Gazebo version other than the one your distro vendors (for
example Jetty on Jazzy), set `GZ_RELAX_VERSION_MATCH=1` so the vendor package
builds the requested Gazebo from source, matching just the major version.

## Prerequisites

- **Gazebo** Harmonic, Ionic, or Jetty — provided by the `gz_*_vendor`
  packages of your ROS 2 distro (see the table above)
- **ROS 2** Jazzy, Kilted, or Lyrical with `rclcpp`, `sensor_msgs`, `std_msgs`
- **Ouster SDK** (via the `ouster-ros` submodule -- run
  `git submodule update --init --recursive`)
- **Eigen3**
- **GPU toolchain** (optional, any of):
  - **CUDA Toolkit** (NVIDIA) -- detected automatically via `nvcc`
  - **ROCm / HIP** (AMD, incl. APUs) -- pass `-DGZ_GPU_OUSTER_USE_HIP=ON`
    with `hipcc` available; APU unified memory is used automatically
  - **Intel oneAPI DPC++** (Intel iGPU, Arc) -- pass
    `-DCMAKE_CXX_COMPILER=icpx` and the SYCL backend enables itself, or
    use AdaptiveCpp with `-DGZ_GPU_OUSTER_USE_SYCL=ON`
  - With none of the above the plugin compiles on CPU only (OpenMP-
    parallelised); produces identical results at lower throughput.

For IMU simulation, your world SDF must also load the Gazebo IMU system:
```xml
<plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
```

### World requirements

The default `raycast` mode casts beams against the ECM scene mirror with no
render engine. The only world requirement is `gz-sim-physics-system` (pose
updates). The bundled examples also load `gz-sim-altimeter-system` so the
non-rendering `altimeter` pose anchor is handled without Gazebo warnings.

**`panels` mode** adds two requirements:

1. Load the Sensors system with the ogre2 engine:
   ```xml
   <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
     <render_engine>ogre2</render_engine>
   </plugin>
   ```

2. Contain **at least one rendering sensor** (`camera`, `gpu_lidar`,
   `depth_camera`, …). On Gazebo Harmonic the Sensors system only initialises
   rendering — building the scene and emitting `events::Render` — once such a
   sensor exists in the ECM. With only non-rendering sensors (e.g. an
   `altimeter` pose anchor plus an `imu`), the Sensors system never starts
   rendering, `OnRender()` never fires, the panel rig is never created, and **no
   point cloud is produced**. The plugin logs a one-shot error after ~2 s of
   sim time: *"events::Render has not fired … add a rendering sensor"*.

The bundled `ouster_demo_panels.sdf` world is the panels-mode (rendering)
counterpart of `ouster_demo.sdf`. The example launches select it automatically:

```bash
ros2 launch gz_sensors_ouster ouster_standalone.launch.py ray_mode:=panels
```

Passing `ray_mode:=panels` loads `ouster_demo_panels.sdf` and derives
`anchor_type:=camera` automatically — no separate flag needed. Panels needs a
render-capable host (ogre2/GPU); on broken-render hosts use the default
`ray_mode:=raycast` instead.

### Spatial Ouster response textures (raycast mode)

SDF has no standard LiDAR-response texture field. In `raycast` mode this
plugin therefore looks beside a visual's PBR albedo map for an optional
aligned companion named `<stem>.ouster<extension>`:

```text
brick_graffiti.png          visible PBR albedo
brick_graffiti.ouster.png   LiDAR response, same UV layout and dimensions
```

The companion must be RGBA8. Its channels are physical response inputs, not a
false-color display:

| Channel | Meaning | Range |
|---------|---------|-------|
| R | absolute diffuse reflectance at the Ouster's 865 nm laser wavelength (`kd`) | 0–1 |
| G | passive near-IR albedo used by `NEAR_IR` | 0–1 |
| B | monostatic specular coefficient (`ks`) | 0–1 |
| A | opacity; transmittance is `1 − A` | 0–1 |

Sampling is bilinear and repeating. Analytic primitives get deterministic UVs;
triangle meshes use their authored vertex UVs. With no companion, the existing
scalar mapping remains unchanged: `<laser_retro>`, material `<specular>`, and
visual `<transparency>`. Response textures affect return strength and channel
contrast; range relief still has to be modeled as geometry. Panels mode has no
material-ID/UV return buffer and continues to use its scalar/rendered fallback.

The bundled `brick_graffiti.png` / `brick_graffiti.ouster.png` pair is used by
the TurtleBot warehouse, hills, and sewer worlds.

## Workspace setup

`ouster-ros` is not available at the required API version via apt, so both
packages must be source-built. The provided `gz_sensors_ouster.repos` file
pins the exact commits used by CI:

```bash
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws
vcs import --recursive src < /path/to/gz_sensors_ouster.repos   # or after cloning:
# vcs import --recursive src < src/gz_sensors_ouster/gz_sensors_ouster.repos

# Install system dependencies (Gazebo vendor packages, Eigen, etc.)
rosdep update && rosdep install --from-paths src --rosdistro=jazzy -y --ignore-src

source /opt/ros/jazzy/setup.bash
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
```

> **Note:** `ouster-ros` is pinned to an exact commit SHA in
> `gz_sensors_ouster.repos`.  To advance the dependency, update the SHA there
> and in the three other places that mirror it: `ci.yaml`,
> `Dockerfile`, and `.claude/hooks/session-start.sh`.

## Build

```bash
colcon build --packages-select gz_sensors_ouster
```

By default CMake probes your toolchain and enables the appropriate GPU
backend automatically. Force a specific combination with:

```bash
colcon build --packages-select gz_sensors_ouster --cmake-args \
  -DGZ_GPU_OUSTER_USE_CUDA=ON \
  -DGZ_GPU_OUSTER_USE_HIP=OFF \
  -DGZ_GPU_OUSTER_USE_SYCL=OFF
```

## GPU Backends

At runtime the plugin dispatches to the first backend that successfully
initialises a device, in preference order: **CUDA → HIP → SYCL → CPU**.
Each backend reports itself in the Gazebo log on startup:

```
[gz_gpu_ouster_lidar] HIP backend: device='Radeon 780M (gfx1103)' integrated=yes (managed-memory path ON)
[gz_gpu_ouster_lidar] Using hip-apu backend.
```

| Backend | Vendor | Notes |
|---------|--------|-------|
| `cuda`     | NVIDIA | Requires CUDA Toolkit + driver; uses `curand` for noise. |
| `hip`      | AMD discrete | Requires ROCm; uses `hiprand`. |
| `hip-apu`  | AMD APU | Same as `hip` but allocates via `hipMallocManaged`, skipping H2D/D2H over the shared-memory die. |
| `sycl`     | Intel Arc | Requires oneAPI DPC++ (`icpx`) or AdaptiveCpp. Uses `sycl::malloc_shared`. |
| `sycl-igpu`| Intel iGPU | Same backend, integrated-memory fast path. |
| `cpu`      | Any | OpenMP-parallelised; noise parity with GPU paths. Automatic fallback when no GPU device is found. |

### Environment override

Set `GZ_OUSTER_BACKEND` to force a specific backend (useful for
debugging a cross-vendor build):

```bash
GZ_OUSTER_BACKEND=cpu  ros2 launch my_sim bringup.launch.py
GZ_OUSTER_BACKEND=sycl ros2 launch my_sim bringup.launch.py
```

If the forced backend is unavailable on the host, the dispatcher logs a
warning and falls through to the auto-selection order.

## SDF Usage

See [`config/plugin_example.sdf`](config/plugin_example.sdf) for the
full annotated example.

Minimal (path is relative to the SDF file's directory):

```xml
<plugin filename="libgz_sensors_ouster.so"
        name="gz_gpu_ouster_lidar::GzGpuOusterLidarSystem">
  <metadata_path>metadata/os1_64_rev7.json</metadata_path>
  <sensor_name>/sensor/lidar/lidar0</sensor_name>
  <lidar_hz>10.0</lidar_hz>
  <!-- Optional: Gazebo render visibility mask (default: all bits set) -->
  <!-- <visibility_mask>4294967295</visibility_mask> -->
</plugin>
```

With IMU auto-detection:

```xml
<plugin filename="libgz_sensors_ouster.so"
        name="gz_gpu_ouster_lidar::GzGpuOusterLidarSystem">
  <metadata_path>metadata/os1_64_rev7.json</metadata_path>
  <sensor_name>/sensor/lidar/lidar0</sensor_name>
  <lidar_hz>10.0</lidar_hz>
  <imu_name>auto</imu_name>
</plugin>
```

> **Path resolution**: `metadata_path` is resolved relative to the
> directory of the SDF file containing the `<plugin>` element. Absolute
> paths are used as-is. The included metadata files install to
> `share/gz_sensors_ouster/config/metadata/`.

### Multiple Sensors

Add one `<plugin>` block per sensor. Each gets its own topics, panel
rig, GPU stream (or CPU OpenMP pool), and drain thread. Use
different `sensor_name` prefixes and metadata files:

```xml
<!-- Front OS1-64 (primary, with IMU) -->
<plugin filename="libgz_sensors_ouster.so"
        name="gz_gpu_ouster_lidar::GzGpuOusterLidarSystem">
  <metadata_path>metadata/os1_64_rev7.json</metadata_path>
  <sensor_name>/sensor/lidar/front</sensor_name>
  <lidar_hz>10.0</lidar_hz>
  <imu_name>auto</imu_name>
</plugin>

<!-- Rear OS0-128 (short-range, no IMU) -->
<plugin filename="libgz_sensors_ouster.so"
        name="gz_gpu_ouster_lidar::GzGpuOusterLidarSystem">
  <metadata_path>metadata/os0_128_rev7.json</metadata_path>
  <sensor_name>/sensor/lidar/rear</sensor_name>
  <lidar_hz>10.0</lidar_hz>
</plugin>
```

Each sensor adds the panel-rig render targets (~10-50 MB GPU VRAM
depending on density; the exact size is logged at startup) and two
threads (ROS executor + drain). Ogre2 renders sensors sequentially on
the render thread, so 2-3 sensors at 10 Hz is comfortable. For 4+
sensors, consider staggering scan rates or reducing beam density.

## Examples (URDF + world + launch)

Ready-to-run examples live in [`examples/`](examples/) and install to
`share/gz_sensors_ouster/examples/`:

| File | Purpose |
|------|---------|
| `urdf/ouster_macro.xacro` | Reusable `ouster_sensor` xacro macro (the building block) |
| `urdf/ouster_standalone.urdf.xacro` | Single OS1-64 + IMU on a pedestal |
| `urdf/sensor_stack.urdf.xacro` | Platform with front OS1-64+IMU and rear OS0-128 |
| `urdf/turtlebot3_ouster.urdf.xacro` | A drivable TurtleBot3 waffle carrying the Ouster (optional; used by the Docker test) |
| `worlds/ouster_demo.sdf` | Demo world (physics + altimeter + IMU systems, ground + obstacles). GPU-free: raycast mode, no rendering Sensors system |
| `worlds/ouster_demo_panels.sdf` | Panels-mode counterpart of `ouster_demo.sdf` (loads gz-sim-sensors-system/ogre2). Selected automatically by example launches when `ray_mode:=panels` |
| `worlds/turtlebot3_ouster_headless.sdf` | GPU-free arena (no rendering Sensors system) for raycast mode |
| `worlds/turtlebot3_ouster_warehouse.sdf` | Driveable warehouse with shelving, crates, retro signage, and response-textured masonry |
| `worlds/turtlebot3_ouster_hills.sdf` | Rolling mesh terrain, sun-driven NEAR_IR, boulders/trees, and response-textured buildings |
| `worlds/turtlebot3_ouster_sewer.sdf` | Segmented sewer conduit with masonry response maps, wet channels, glossy water, and retro markers |
| `worlds/ouster_showcase.sdf` | **Guided tour of the sensor model** — labelled zones for range, reflectance, retroreflectors, calibrated detection rolloff, specular/mirror ghosts, glass, curvature, sun/NEAR_IR and a moving beacon. Raycast (GPU-free), built only from primitives so it needs no downloads. See [Showcase world](#showcase-world) |
| `launch/ouster_standalone.launch.py` | Bring up the standalone example end-to-end |
| `launch/sensor_stack.launch.py` | Bring up the multi-sensor example |
| `launch/turtlebot3_ouster.launch.py` | Bring up the optional TurtleBot3 waffle + Ouster demo (drivable, raycast by default) |

Run (after `colcon build` + `source install/setup.bash`):

```bash
ros2 launch gz_sensors_ouster ouster_standalone.launch.py
# multi-sensor:  ros2 launch gz_sensors_ouster sensor_stack.launch.py
# with RViz:     ros2 launch gz_sensors_ouster ouster_standalone.launch.py rviz:=true
# panels mode:   ros2 launch gz_sensors_ouster ouster_standalone.launch.py ray_mode:=panels
# Rev8 physics:  ros2 launch gz_sensors_ouster ouster_standalone.launch.py hardware_revision:=rev08
# WSL/headless:  ros2 launch gz_sensors_ouster ouster_standalone.launch.py headless:=true
# TurtleBot:     ros2 launch gz_sensors_ouster turtlebot3_ouster.launch.py world:=warehouse
# Other scenes:  ... world:=hills   or   ... world:=sewer
```

The core package and the standalone/sensor-stack examples do not require
`turtlebot3_description`. To run only the optional TurtleBot demo, install its
description package separately when your ROS distribution provides it:

```bash
sudo apt install ros-${ROS_DISTRO}-turtlebot3-description
```

If no binary package exists for the distribution, add
`turtlebot3_description` to the source workspace instead. The supplied Docker
build already does this with a pinned TurtleBot3 checkout. Launching the demo
without the optional package reports these installation choices directly.

Each launch starts Gazebo with the demo world, runs
`robot_state_publisher` on the xacro, spawns the model with
`ros_gz_sim create` (which loads the system plugin), and bridges only
`/clock` (the LiDAR/IMU/image topics are published directly by the
plugin via `rclcpp`, so they need no bridge).

`ouster_standalone.launch.py` additionally runs the `ouster_ros`
`os_cloud` node (in the `/sensor/lidar/lidar0` namespace) so the plugin's
`lidar_packets` are assembled into a `PointCloud2` on
`/sensor/lidar/lidar0/points`, exactly as for a real Ouster — verify with
`ros2 topic hz /sensor/lidar/lidar0/points`. `os_cloud` is configured with
`point_cloud_frame:=lidar0/lidar_frame` (the cloud is stamped in the URDF lidar
frame) and a **distinct** `sensor_frame:=lidar0/os_sensor` so the driver uses the
identity XYZ LUT — see [Frames vs `os_cloud`](#frames-vs-ouster_ros-os_cloud) for
why that pairing matters — and the cloud lands in the `robot_state_publisher` TF
tree (RViz fixed frame `base_footprint`). A small `static_transform_publisher`
redundantly mirrors the URDF mount joint (`base_link → lidar0/lidar_frame`) so
the cloud still reaches `base_link` even when `robot_state_publisher` is not
running. The RViz config includes a `PointCloud` display for that topic.

### Showcase world

`worlds/ouster_showcase.sdf` is the "see everything the model does" world. Where
`ouster_demo.sdf` answers *does it produce points*, the showcase answers *what
does each part of the sensor model actually look like* — each effect gets its
own labelled zone, laid out radially around the origin so one 360° scan sweeps
through all of them:

```bash
ros2 launch gz_sensors_ouster ouster_standalone.launch.py \
    world:=ouster_showcase.sdf
# headless (WSL/SSH/CI):  ... world:=ouster_showcase.sdf headless:=true
# with RViz:              ... world:=ouster_showcase.sdf rviz:=true
```

| Azimuth | Zone | What to look for |
|---------|------|------------------|
| 0° | Range ladder | Six identical 80%-reflectance targets at 6–105 m: range-noise σ ramps with distance, signal falls as 1/r², the far pair thins out from `dropout_rate_far` |
| 45° | Reflectance ladder | Seven panels at one range, `laser_retro` 0.02→1.0: an even staircase across the 0–100 reflectivity band; dark panels are noisier and drop out more |
| 58–72° | Retroreflector ladder | Nine posts, `laser_retro` 1.3→300 → the calibrated log band walked end to end (bytes 108, 130, 151, 173, 195, 220, 242, 255, 255), like signs and retro tape |
| 82–129° | **Signal / intensity ladder** | Nine posts at *fixed* reflectivity walking in from 19.2 m to 1.2 m. SIGNAL is `base_signal·ρ/r²`, so the 1/r² term — not reflectivity — is what sweeps it; the closest post saturates the channel outright |
| 135° | Detection rolloff | Rev7 OS1 dark targets at 90 m and 115 m sit at the published 90%-detection point and in its smooth tail; a bright target at 115 m remains highly detectable. Returns thin statistically instead of disappearing at an artificial hard boundary. |
| 180° | Specular / gloss | Matte (kₛ 0.10), glossy (0.45) and mirror (0.95) faces: the specular lobe, and above the mirror threshold a **ghost return** behind the mirror (the orange post's reflection) |
| 215° | Glass | Panes at τ=0.60 and τ=0.92 with walls behind: the pane wins on the first, the wall behind wins on the second — "lidar sees through the window" |
| 270° | Curvature | Spheres and cylinders of several radii: apparent reflectance ρ·cos(α) falls off toward each silhouette |
| 305° | Sun / NEAR_IR | Four identical panels at different tilts. NEAR_IR is reflected *ambient*, so orientation alone separates them while range and reflectivity do not |
| 330° | Moving beacon | A rotating arm with a retroreflective paddle — exercises dynamic transforms, and honestly shows that `motion_distortion` corrects **ego** motion only, not moving targets |
| — | Perimeter + ground | Walls at 60 m for long-range structure; a straight wall spanning many columns makes any azimuth/encoder-convention error obvious (it would break into arcs) |

#### Channel coverage

The zones are sized so that **one scan drives every Ouster channel across its
whole domain**, not a narrow slice — a consumer that only ever sees a thin band
of values can look correct in sim and fall over on real data. Measured from the
`PointCloud2` that `os_cloud` assembles (12 scans, OS1-64 defaults):

| Channel | min | max | Coverage |
|---------|-----|-----|----------|
| `intensity` (SIGNAL) | 0 | **65535** | 4.8 decades, ~7800 distinct values |
| `reflectivity` | 0 | **255** | ~206 distinct values across 0–255 |
| `ambient` (NEAR_IR) | 0 | **65535** | saturates on the sun-facing retro panel |
| `range` | 0 | **~110000 mm** | out to the 110 m perimeter (inside the Rev7 OS1's 233 m representable window) |

Two design points make the hard channels reachable:

- **SIGNAL is the one a naive scene starves.** It goes as `ρ/r²`, so with
  nothing closer than 6 m it never leaves single digits — before the intensity
  ladder existed this world peaked at **195** of a possible 65535 (0.3% of the
  channel), with a third of objects returning literally 0. The ladder walks in
  to 1.2 m at *constant* `laser_retro`, so range alone sweeps it.
- **Occlusion and calibrated detection are load-bearing.** The perimeter sits
  at 110 m rather than 60 m, because at 60 m it silently blocked the 70 m and
  105 m range targets (max observed range was 69 m). Its `laser_retro` is 0.75,
  which keeps a Rev7 OS1 return at that range highly probable while the dark
  Zone D target visibly thins through its product-calibrated rolloff.

Two further properties worth knowing:

- **Self-contained.** Built entirely from `box`/`sphere`/`cylinder`/`plane`
  primitives — no Fuel downloads, no external meshes. That is also a
  correctness requirement: the raycast mirror only handles those plus `mesh`,
  and silently skips `capsule`/`ellipsoid`/`heightmap`/`polyline`, which would
  be visible in the GUI but invisible to the lidar. A structural test
  (`test_worlds.py::test_visual_geometry_is_mirrorable`) enforces this for every
  shipped world. The mesh/BVH path is covered by the TurtleBot3 example instead.
- **Raycast mode.** It loads no render engine, so it runs headless anywhere. The
  startup log should read `raycast scene mirror v1: 59 instances (0 meshes,
  0 visuals skipped) from 59 visuals` — a non-zero "skipped" count means
  something in the world is invisible to the sensor.

### How the URDF wires to the plugin

The plugin is a Gazebo **system** plugin on the model; it casts its own rays
rather than using a gz `<sensor type="gpu_lidar">`. The macro takes a
**`ray_mode`** (default **`raycast`**) and emits, per sensor:

- A **pose-anchor `<sensor>`** named exactly like the last segment of
  `<sensor_name>` (e.g. `lidar0`). The plugin looks this entity up to read its
  world pose (the ray-cast origin). What it must be depends on `ray_mode`:
  - **`raycast`** (default) — beams are cast on the CPU against an ECM scene
    mirror, with no render engine. A non-rendering **`altimeter`** anchor is
    sufficient; the world needs no GPU. This is the anchor default.
  - **`panels`** — the plugin drives a GpuRays rig off `events::Render`, so the
    anchor must be a *rendering* sensor and the world must load
    `gz-sim-sensors-system` (see [World requirements](#world-requirements)).
    The example URDFs derive `anchor_type:=camera` automatically when
    `ray_mode:=panels`; the example launches switch to `ouster_demo_panels.sdf`
    automatically. No separate `anchor_type` flag needed for the bundled examples.
- An optional real **`<sensor type="imu">`** (name contains `imu`) when
  `enable_imu` is set. This requires `gz-sim-imu-system` in the world
  (the demo world loads it) — the plugin reads the IMU components that
  system populates.
- URDF links named **`<name>/lidar_frame`** and **`<name>/imu_frame`** so
  `robot_state_publisher` publishes TF frames that match the frame_ids
  the plugin stamps on its image/IMU messages.

> **Metadata path:** the launch files pass an **absolute**
> `metadata_path` into xacro. Relative paths resolve against the SDF
> file's directory, which does not exist for a model spawned from the
> `robot_description` topic.

### Frames vs. `ouster_ros` `os_cloud`

This plugin generates its points aligned with `<name>/lidar_frame` (the URDF
sensor frame, published by `robot_state_publisher`). The example launches
therefore configure `os_cloud` so the cloud's full TF chain to `base_link`
is explicit:

- `point_cloud_frame:=<name>/lidar_frame` — the `PointCloud2` is stamped in the
  robot's lidar frame. The plugin's ranges are calibrated to the **standard
  (identity) Ouster XYZ LUT**, so the cloud must be reconstructed **without** the
  metadata's `lidar_to_sensor_transform` (the real Ouster 180° + 36 mm housing
  offset) — applying it would rotate/shift the whole cloud.
- The catch: `ouster_ros` applies `lidar_to_sensor_transform` **iff
  `point_cloud_frame == sensor_frame`**
  (`os_transforms_broadcaster.h::apply_lidar_to_sensor_transform()`). So the
  launches keep the two **different** — `sensor_frame:=<name>/os_sensor` — to get
  the identity LUT, and set `lidar_frame:=<name>/lidar_frame` equal to
  `point_cloud_frame` so the driver's frame validation keeps the requested frame
  (an unrecognised `point_cloud_frame` is otherwise reset to `lidar_frame` with a
  warning). Setting `point_cloud_frame == sensor_frame` instead would apply the
  housing rotation — the opposite of what a sim cloud aligned to the URDF frame
  wants.
- `pub_static_tf:=false` — with `sensor_frame != lidar_frame` the driver would
  broadcast `sensor_frame → lidar_frame`, giving `<name>/lidar_frame` a second
  parent on top of the `base_link → <name>/lidar_frame` mount, a TF-tree
  conflict. The unused `os_lidar`/`os_imu` leaf frames are dropped; RSP and the
  mount publisher own the tree.
- A `static_transform_publisher` publishes `base_link → <name>/lidar_frame`
  (matching the URDF mount joint), so the cloud reaches `base_link` even if
  `robot_state_publisher` is not running. When RSP is up it publishes the same
  edge, which is harmless (a one-time `TF_REPEATED_DATA` warning).

Net TF chain: `points (<name>/lidar_frame) → base_link → base_footprint`, via
the launch stack (RSP / the mount `static_transform_publisher`).

## Docker (standalone test)

A self-contained [`Dockerfile`](Dockerfile) builds the plugin in isolation —
ROS 2 + the new Gazebo + the pinned `ouster-ros` fork + this package — and
exercises it on a **drivable TurtleBot3 waffle** ("a vehicle for the lidar to
ride on"). By **default it builds the CUDA backend** so it can use the host's
GPU, and **falls back to the CPU (OpenMP) backend** automatically — both at build
time (if the CUDA toolkit can't be installed) and at run time (if no GPU is
visible). So the same image runs anywhere: `docker run --rm gzouster` works with
no GPU, and adding `--gpus all` switches the plugin to CUDA. The default smoke
uses **raycast** mode, which needs no render engine. Pass
`--build-arg ENABLE_CUDA=false` for a smaller CPU-only image.

`ROS_DISTRO` selects the distro (matching this package's CI matrix); **Humble is
not supported** (no `gz_*_vendor`; it ships Gazebo Fortress, not Harmonic+):

```bash
cd <this package>

# Build (default jazzy → Harmonic; kilted → Ionic; lyrical → Jetty, advisory)
docker build -t gzouster .
docker build -t gzouster --build-arg ROS_DISTRO=kilted .

# 1) Headless point-cloud smoke: waits for a PointCloud2 on
#    /sensor/lidar/lidar0/points and exits PASS/FAIL. Add --gpus all to exercise
#    the CUDA backend; without it the plugin falls back to CPU.
docker run --rm gzouster
docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all gzouster   # CUDA

# 2) Re-run the gtest suite.
docker run --rm gzouster test

# 3) Interactive: gz GUI + RViz + teleop_twist_keyboard on /cmd_vel.
#    Needs a display. On Linux with a normal X11/Wayland session, -e DISPLAY
#    and the X11 socket volume are usually sufficient:
docker run --rm -it --gpus all \
  -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix gzouster drive

# 3b) Just the windows (gz GUI + RViz), no teleop:
docker run --rm -it --gpus all \
  -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix gzouster gui
```

`drive` runs teleop_twist_keyboard in the foreground (drive with the keys);
`gui` just shows the windows. **Both bring up RViz by default.** Toggle it off
with `-e RVIZ=false`, and switch ray modes with `-e RAY_MODE=panels` (needs a
GPU). Docker `-e` flags must come **before** the image name, e.g.
`docker run ... -e RVIZ=false ... gzouster drive` — flags placed after the image
name are passed to the entrypoint as arguments, not env vars.

### WSL & headless Linux

The default **raycast** mode and the **smoke** docker target run without any
display — they work identically on bare metal, in CI, and inside WSL.

**WSL2 + Windows 11 (WSLg)** — WSLg automatically sets `DISPLAY` and creates
`/tmp/.X11-unix`. The interactive docker commands above work as-is; no extra
setup needed. For panels mode, add `--gpus all` (requires
[nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
and WSL2 GPU passthrough).

**WSL2 + Windows 10 / WSL1** — No built-in display. Two options:

1. *Headless (no display needed)* — use `headless:=true` to suppress the GUI
   client; the simulation and point cloud still run:

   ```bash
   ros2 launch gz_sensors_ouster ouster_standalone.launch.py headless:=true
   ros2 launch gz_sensors_ouster sensor_stack.launch.py      headless:=true
   # or the docker equivalent — smoke always runs headless:
   docker run --rm gzouster
   ```

2. *With a Windows X server* (VcXsrv, GWSL, MobaXterm) — start the X server on
   Windows (allow connections from WSL), then:

   ```bash
   export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0.0
   docker run --rm -it -e DISPLAY gzouster drive
   ```

**Wayland-native desktop** — If `DISPLAY` is unset but `WAYLAND_DISPLAY` is set,
Qt6 (used by gz sim and RViz) can run natively with `QT_QPA_PLATFORM=wayland`, or
fall back to XWayland with `QT_QPA_PLATFORM=xcb`. Pass the variable through docker:

```bash
docker run --rm -it \
  -e WAYLAND_DISPLAY -e XDG_RUNTIME_DIR \
  -v "$XDG_RUNTIME_DIR/$WAYLAND_DISPLAY":/tmp/wayland-0 \
  -e QT_QPA_PLATFORM=wayland \
  gzouster gui
```

**SSH / headless GPU server** — `headless:=true` also works with `ray_mode:=panels`
if the host has a GPU: gz runs server-only and ogre2 initialises via EGL without a
display, so the point cloud is produced even without a GUI client.

### Using the host GPU (CUDA backend)

The image **builds the CUDA backend by default** and runs it on the **host's**
GPU. Only the CUDA *toolkit*
(nvcc/cudart/curand) is baked in; the driver/`libcuda` comes from the host at run
time (needs `nvidia-container-toolkit` on the host — no host CUDA install
required). Tune the target GPU with `CUDA_ARCH`, or opt out entirely with
`ENABLE_CUDA=false`:

```bash
# Default build already includes CUDA. CUDA_ARCH: 86 = Ampere (RTX 30xx),
# 89 = Ada (40xx); default "80;86;89", or "75;80;86;89" for wider portability.
docker build -t gzouster --build-arg CUDA_ARCH=86 .

docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all gzouster
# the plugin logs "Using cuda backend."; without --gpus all it logs the
# "No CUDA-capable device detected ... CPU fallback" warning and still works.

docker build -t gzouster-cpu --build-arg ENABLE_CUDA=false .   # smaller CPU-only image
```

The toolkit install is **best-effort**: on a base where it can't be set up (e.g.
lyrical/26.04, a non-amd64 host, or a repo/network error) the build prints a
warning and falls back to a CPU-only build instead of failing.
`CUDA_DISTRO`/`CUDA_PKG_VERSION`/`CUDA_HOME_VERSION` override the NVIDIA apt repo
and toolkit version (defaults: Ubuntu 24.04 `noble`, CUDA 12.6 — valid for the
jazzy/kilted bases). The selected backend accelerates **both** ray modes — the
per-beam casting in `raycast` and the resample/noise pipeline in `panels`
(verified: with `--gpus all` the plugin logs `Using cuda backend.` in raycast).

The vehicle (`examples/urdf/turtlebot3_ouster.urdf.xacro`) reuses the genuine
ROBOTIS `turtlebot3_description` waffle geometry, adds a new-Gazebo
`gz-sim-diff-drive-system` (so it drives off `/cmd_vel`), and mounts the Ouster
via the `ouster_sensor` macro in `ray_mode:=raycast`. The build clones
`turtlebot3` only for its `turtlebot3_description` subpackage; both it and
`ouster-ros` are pinned to exact commits for reproducibility.

## Included Metadata Files

Example Ouster calibration JSONs are provided in `config/metadata/` for
simulation without real hardware:

| File | Sensor | Beams | VFOV | Beam Angles | Notes |
|------|--------|-------|------|-------------|-------|
| `os1_64_rev7.json` | OS1-64 | 64 | 33.2° | Real (from SDK source) | Default, recommended for testing |
| `os0_128_rev7.json` | OS0-128 | 128 | 90° | Nominal (uniform spacing) | Ultra-wide short-range |
| `os1_128_rev7.json` | OS1-128 | 128 | 45° | Nominal | High-density mid-range |
| `os2_128_rev7.json` | OS2-128 | 128 | 22.5° | Nominal | Long-range narrow |
| `osdome_128_rev7.json` | OSDome-128 | 128 | 180° | Nominal | Hemispheric (8 pitched panels + zenith cap) |

The default files use the `RNG19_RFL8_SIG16_NIR16` lidar profile, `LEGACY`
IMU profile, and 1024 columns/frame at 10 Hz. Product physics are selected by
the metadata `prod_line` plus `<hardware_revision>`; explicit example values
default to `rev07`. With `hardware_revision=auto`, a real revision-bearing
`prod_pn` is preferred and firmware/product clues are used when unambiguous.

### Hardware physics profiles

The selected model and revision control minimum and representable range,
published 100-klx D90 detection points, precision, range resolution, beam
diameter/divergence, accuracy bounds, and return capability. Related revisions
with unchanged optical specifications intentionally share a row:

| Generation / revision | Models | D90 range at 10% / 80% reflectivity (m) | Representable / minimum range (m) | Returns |
|-----------------------|--------|-------------------------------------------|-----------------------------------|---------|
| Gen1 OS1, FW 1.x | OS1 | 40 / 105 | 200 / 0.8 | 1 |
| Gen1 OS1, FW 2.x | OS1 | 50 / 110 | 200 / 0.8 | 1 |
| Rev C, D, 05 | OS0; OS1; OS2 | 15 / 45; 45 / 100; 80 / 210 | 270 / 0.3; 270 / 0.3; 465 / 1.0 | 1 |
| Rev 06, 06.2 | OS0; OS1; OS2 | 15 / 45; 45 / 100; 80 / 210 | 270 / 0.3; 270 / 0.3; 465 / 1.0 | 2 |
| Rev 07 | OS0; OS1; OS2; OSDome | 35 / 75; 90 / 170; 200 / 350; 20 / 45 | 233 / 0.5; 233 / 0.5; 404 / 0.8; 233 / 0.5 | 2 |
| Rev 07.1 | OS0; OS1; OSDome | 35 / 75; 90 / 170; 20 / 45 | 233 / 0.5; 233 / 0.5; 233 / 0.5 | 2 |
| Rev 08 | OS0; OS1; OSDome; OS1 MAX | 35 / 75; 90 / 170; 20 / 45; 200 / 350 | 500 / 0.5 (all models) | 2 |

D90 is the distance where detection probability remains at least 90%, not a
hard maximum. The plugin interpolates the published 10% and 80% reflectivity
anchors, then applies a smooth D90-to-D50 rolloff. The active scan mode also
uses Ouster's documented point-gathering factors: halving the column rate
multiplies detection range by 1.19 and precision sigma by 0.71. An explicit
`max_range`, `min_range`, or range-noise setting remains an SDF override.

### Profile variants (modern vs LEGACY)

Each sensor ships in **two variants**, selected by the `metadata_path` you
pass (or, in the example launch, the `lidar_profile:=modern|legacy` arg):

| Variant | Files | Profile | Firmware | When to use |
|---------|-------|---------|----------|-------------|
| **modern** (default) | `<name>.json` | `RNG19_RFL8_SIG16_NIR16` | v3.2.0 | Current OS sensors; **required for `os_cloud`** |
| **legacy** | `<name>_legacy.json` | `LEGACY` | v3.1.0 | Simulating pre-3.2 firmware |

Why two? `ouster-sdk` (≥ 0.16, bundled by `ouster-ros`) added a `WINDOW`
field to the `RNG19_RFL8_SIG16_NIR16` profile that exists only for
**firmware ≥ 3.2.0**. The metadata previously declared that modern profile
with firmware **v3.1.0** — a combination real hardware never produces — so
`os_cloud` allocated a `LidarScan` *without* `WINDOW` while its `ScanBatcher`
still tried to parse one, crashing with **"Field 'WINDOW' not found in
LidarScan."** The modern files now declare firmware **v3.2.0** so the field
set is consistent; the LEGACY profile has no `WINDOW` field at all, so it
also works (on any firmware). Both produce a correct point cloud — the
plugin marks column validity via the packet `STATUS` byte, not `WINDOW`.

> **Note**: For production use, replace these with real calibration data
> from your hardware (`ouster-cli sensor-info` or the sensor HTTP API).
> Real metadata includes per-unit beam angle calibration that the nominal
> files approximate.

## Parameters

All noise model parameters can be changed at runtime via
`ros2 param set <node_name> <param> <value>`.

### Required

| Parameter | Type | Description |
|-----------|------|-------------|
| `metadata_path` | string | Path to Ouster calibration JSON. Absolute paths are used as-is; relative paths are resolved against the SDF file's directory. |
| `hardware_revision` | string | Product physics revision: `auto`, `gen1`, `revC`, `revD`, `rev05`, `rev06`, `rev06.2`, `rev07`, `rev07.1`, or `rev08`. Prefer an explicit value for synthetic metadata without a real revision-bearing part number. |
| `sensor_name` | string | ROS topic prefix and node namespace (e.g. `/sensor/lidar/lidar0`). |

### Lidar

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `lidar_hz` | 10.0 | > 0 | Scan rate in Hz. |
| `max_range` | *profile* | >= 1 | Representable range in metres from the model/revision profile; explicit values override it. Also sets the GPU far clip plane. |
| `min_range` | *profile* | >= 0 | Minimum reported range from the model/revision profile; explicit values override it. |
| `detection_rolloff` | 0.15 | > 0 | Fraction beyond D90 used to place the smooth D50 point when a datasheet does not publish D50. Historical profiles use their published D50 anchors. |
| `visibility_mask` | 4294967295 | 0 to 4294967295 | **Panels mode only.** Gazebo render visibility mask applied to the panel depth cameras (a visual is seen when `visual.visibility_flags & visibility_mask != 0`), so it can include/exclude visuals from the rendered scan. Has **no effect in `raycast` mode**, which casts against every mirrored visual regardless. |
| `ray_mode` | `raycast` | `panels` \| `raycast` | `panels` renders a perspective depth-panel rig on the GPU and resamples each beam from it. `raycast` casts every beam exactly (calibrated direction, true beam-origin parallax) against an ECM scene mirror — zero interpolation error, with a per-visual material model: `laser_retro × cos(incidence)` diffuse + a specular lobe from the material `<specular>` (glossy/black paint signature) + see-through `<transparency>` (glass: pane vs object-behind, strongest return wins). Extended-Lambertian lidar equation per [docs/MODEL_REFERENCES.md](docs/MODEL_REFERENCES.md); no rendering involved (no anchor-sensor requirement). Runs on CUDA/HIP/SYCL when available with an OpenMP CPU fallback; mirrors box/sphere/cylinder/plane/mesh visuals. |
| `panel_oversample` | 2.0 | 1 to 4 | Panels mode only. Panel angular resolution as a multiple of the sensor's finest angular resolution. Higher = sharper edges, more VRAM and render time. |
| `panel_sampling` | `bilinear` | `bilinear` \| `nearest` | `bilinear` interpolates the 4 neighbouring rendered rays (smooth surfaces, but silhouettes blend fore/background range). `nearest` takes the single closest rendered ray — a true raycast with direction quantised to the pixel grid (≤ 1/(2·oversample) of the beam spacing) and no range blending at depth edges. |

### QoS overrides

`lidar_packets` uses RELIABLE + KEEP_ALL + VOLATILE. On Zenoh this applies
transport backpressure to the dedicated packet drain thread. The encoder
retains one active frame and one newest pending frame; overload replaces a
whole pending frame. Packet publication has its own mutex so backpressure
does not hold simulation-thread metadata, image or IMU publication locks.
Set `lidar_packet_reliable: true` on `os_cloud`, `os_image` and `os_pinhole`
consumers; the supplied examples do this using the pinned Ouster ROS fork.
The legacy `RosInterfaceConfig::lidar_packet_qos_depth` member remains source
compatible but no longer sets packet history. Image, camera_info and IMU pubs are
configurable for deployments that need a specific QoS to match their
consumer — necessary on rmw_zenoh_cpp where pub and sub QoS must match
exactly (neither BEST_EFFORT-pub-to-RELIABLE-sub nor the reverse work).

| Parameter | Default | Accepted values | Affects |
|-----------|---------|-----------------|---------|
| `publish_native_images` | `false` | `true` \| `false` | enables the native range/signal/reflec/nearir image + camera_info pubs (off: os_image is the image source in sim) |
| `image_qos` | `reliable` | `reliable` \| `best_effort` \| `sensor_data` | range/signal/reflec/nearir image + camera_info (when `publish_native_images`) |
| `imu_qos`   | `sensor_data` | `reliable` \| `best_effort` \| `sensor_data` | imu + imu_packets |

Defaults match the most common consumers: `reliable` for images
(matches RViz Image display, rqt_image_view, image_transport), and
`sensor_data` (BEST_EFFORT) for IMU (matches the real `ouster_ros`
driver so a sim-to-hardware topic swap doesn't require flipping
subscriber QoS).

### Noise Model

Range and detection defaults come from the selected Ouster hardware profile.
User-adjustable noise and dropout terms remain dynamically reconfigurable.

Dropout and range noise also scale with surface reflectivity: dark
surfaces (low retro) get up to 3x higher dropout and 2x more range
noise. This matches the real sensor behavior where low-reflectivity
targets produce weaker returns.

| Parameter | Default | Range | Units | Description |
|-----------|---------|-------|-------|-------------|
| `range_noise_min_std` | *profile* | >= 0 | m | Near-range precision sigma. Defaults to the selected hardware/mode precision envelope and scales with reflectivity. |
| `range_noise_max_std` | *profile* | >= 0 | m | Far precision sigma, reached at the profile's 10%-reflectivity D90 reference range. |
| `signal_noise_scale` | 1.0 | >= 0 | -- | Poisson shot noise on signal channel. 0 = off, 1 = physical. |
| `nearir_noise_scale` | 1.0 | >= 0 | -- | Poisson noise on near-IR channel (both packet and image). |
| `base_signal` | 800.0 | >= 0 | photon m^2 | Baseline for the 1/r² signal model and photon-limited aerosol detection. OS0: ~400, OS1: ~800. |
| `base_reflectivity` | 50.0 | 0-255 | -- | Default calibrated reflectivity byte when no retro channel is available. In raycast mode, an omitted `laser_retro` is converted to physical reflectance before incidence, smoke attenuation and return arbitration. |
| `dropout_rate_close` | 0.0005 | 0-1 | probability | Random miss rate at 0 m. Scales with reflectivity (low retro = more drops). |
| `dropout_rate_far` | 0.03 | 0-1 | probability | Random miss rate at the representable range, combined with the product-calibrated smooth detection probability. |
| `false_alarm_rate` | 0.0 | 0-1 | probability | Solar-background false alarms: each no-return pixel becomes a spurious point (uniform range, noise-floor signal) with this probability per frame. 0 = off. Try 0.0005-0.002 for bright daylight. |
| `motion_distortion` | false | bool | -- | Raycast mode only. Rolling-shutter sweep: each column casts from the sensor pose at its acquisition time (interpolated per sim tick), skewing the cloud by the platform's motion over one scan — as a real spinning lidar does (see [docs/MODEL_REFERENCES.md](docs/MODEL_REFERENCES.md) §9). Ego motion only; default off preserves the instantaneous snapshot. |
| `edge_discon_threshold` | 0.15 | >= 0 | m | Depth-discontinuity suppression threshold (1ns echo delay convention). 0 = off. |

### Smoke, dust and fog obscuration (raycast mode)

Beams are integrated through participating media: targets behind smoke dim by
`exp(−2·η·τ)` and eventually drop out (including untagged surfaces using
`base_reflectivity`), the smoke itself produces competing
returns, and NEAR_IR *brightens* while the laser channels darken. See
[docs/MODEL_REFERENCES.md](docs/MODEL_REFERENCES.md) §11 for the physics, and
`examples/worlds/ouster_smoke.sdf` for a demo of all of it.

The medium first passes a photon-count gate with
`P(detect) = 1 − exp(−base_signal·π·∫β(r)exp(−2τ(r))/R² dr)`. Weak or distant
smoke intersections therefore remain empty instead of outlining the authored
volume as a solid object. Conditional on detection, the range is sampled from
that complete received-power profile and compared with the attenuated hard
target by received power. Successive scans produce sparse, spatially
distributed, changing aerosol returns. Ordinary signal, range and dropout
noise are applied afterward by the shared sensor model.

Use explicit `<obscurant>` volumes when the physics matters. Gazebo
`<particle_emitter>` elements are visual effects without a physical density;
mirroring their approximate envelopes remains available as an opt-in
compatibility mode:

| Parameter | Default | Range | Units | Description |
|---|---|---|---|---|
| `particle_obscuration` | false | bool | -- | Opt in to mirroring the world's visual `<particle_emitter>` envelopes as obscurants. Prefer explicit volumes. |
| `particle_extinction` | 1.0 | >= 0 | 1/m | Extinction coefficient produced by an emitter whose `<particle_scatter_ratio>` is 1.0. gz's default ratio of 0.65 then gives σ ≈ 0.65/m — visibility ≈ 6 m, thick smoke. Lower it for haze. |
| `particle_growth` | 1.0 | >= 0 | -- | Fraction of the mean particle travel distance (`½(v_min+v_max)·lifetime`) by which an emitter's `<size>` volume is dilated to cover the plume. 0 uses `<size>` verbatim. The dilation is isotropic, so it always contains the plume; use `<obscurant>` when the shape matters. |
| `obscurant_lidar_ratio` | 50.0 | > 0 | sr | Extinction-to-backscatter ratio `S = σ_ext/β_π`, which sets how strongly the medium returns. ≈18-20 fog/cloud, 40-50 dust, 50-70 smoke. |
| `obscurant_albedo` | 0.8 | 0-1 | -- | Single-scattering albedo ω — how brightly lit media glow in NEAR_IR. Not independent of `obscurant_lidar_ratio`: `S = 4π/(ω·P(π))`. |
| `obscurant_multiple_scattering` | 1.0 | 0-1 | -- | Platt's η. Forward-peaked media deflect much of the "extinguished" light by only milliradians, so a real receiver still collects it; η credits that back, attenuating by `exp(−2·η·τ)`. 1.0 is the pure single-scattering limit (nothing changes unless you ask); 0.5-0.8 is realistic for dense fog and smoke. Applies to the laser round trip only — the NEAR_IR airlight term already accounts for it. |
| `pulse_length` | 0.6 | >= 0 | m | One-pulse range gate `ΔR = c·τ_pulse/2`; scales the medium's return amplitude. |

Authored volumes are `<obscurant>` blocks on the plugin (repeat for more,
up to 16 — beyond that the nearest to the sensor are kept and the rest are
logged as dropped):

```xml
<obscurant>
  <type>ellipsoid</type>        <!-- box | ellipsoid | cylinder -->
  <pose>10 0 1.5 0 0 0</pose>   <!-- world frame -->
  <size>6 6 3</size>            <!-- FULL extents, metres -->
  <extinction>0.6</extinction>  <!-- σ_ext [1/m] ... -->
  <!-- <visibility>6.5</visibility>   ...or say it as metres of visibility -->
  <lidar_ratio>50</lidar_ratio> <!-- optional, defaults as above -->
  <albedo>0.8</albedo>
  <multiple_scattering>0.6</multiple_scattering>
</obscurant>
```

### IMU (optional)

Requires a Gazebo IMU sensor in the SDF/URDF and the `gz-sim-imu-system`
world plugin.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `imu_name` | *(disabled)* | -- | Gazebo IMU sensor entity name. Set to `"auto"` to use the first IMU found. Omit to disable. |
| `imu_hz` | 100.0 | > 0 | IMU publish rate in Hz. |
| `publish_imu_msg` | true | bool | Also publish `sensor_msgs/Imu` alongside Ouster IMU packets. |

#### IMU noise model

White Gaussian noise plus random-walk bias on each axis. Defaults match
the Ouster Os1 IMU datasheet (ICM-20948 class). All values are
**continuous-time densities** (per-√Hz) — at runtime they're scaled by
1/√dt for white noise and √dt for bias drift, where dt = 1/imu_hz. Set
any to 0 to disable that term. All four are dynamically reconfigurable
via `ros2 param set`.

| Parameter | Default | Units | Description |
|-----------|---------|-------|-------------|
| `gyro_noise_std`  | 1.75e-4 | rad/s/√Hz | Gyro white-noise density (≈0.01 °/s/√Hz). |
| `accel_noise_std` | 2.3e-3  | m/s²/√Hz  | Accelerometer white-noise density (≈230 µg/√Hz). |
| `gyro_bias_walk`  | 1.0e-6  | rad/s²/√Hz | Gyro bias random walk (in-run instability). |
| `accel_bias_walk` | 1.0e-5  | m/s³/√Hz   | Accelerometer bias random walk. |

The published `angular_velocity_covariance` and
`linear_acceleration_covariance` diagonals are derived from the active
noise std at publish time, so downstream EKFs see covariances consistent
with the injected noise. With all noise params set to 0, the covariances
fall back to the ouster_ros driver defaults (6e-4 / 0.01) so REP-145
consumers don't get literal-zero variances.

## Sensor Tuning Guide

Start by selecting the real `<hardware_revision>` and matching model metadata;
the profile supplies the optical limits and precision. Tune environmental
terms (`base_signal`, shot noise, random dropout, solar false alarms and
obscurants) only for the simulated conditions. `max_range`, `min_range` and
range-noise parameters are escape hatches for measured unit-specific behavior,
not required per-model boilerplate.

## Performance

The rendering bottleneck is the panel rig's depth passes (4 for
cylindrical sensors, 9 for the OSDome). The GPU post-processing
pipeline (resample + noise) adds only
~2-3 ms per frame regardless of sensor density on any of the supported
backends. All numbers assume a single sensor.

> The middleware (QoS / executor / RMW choice / zero-copy) and GPU-pipeline
> (streams / pinned memory / launch overhead) design choices are mapped to
> the systems literature in
> [docs/SYSTEMS_REFERENCES.md](docs/SYSTEMS_REFERENCES.md), including which
> optimisations are deliberately not applied and when to revisit them.

### Estimated max real-time scan rate

The table below was measured **only on the CUDA backend** (NVIDIA RTX
GPUs). The HIP, SYCL and CPU figures discussed in this section are
*engineering estimates, not benchmarks* — they have not been measured,
and CI does not run any performance test (the HIP/SYCL jobs are
compile-only, with no device). Treat them as ballpark guidance until
benchmarked on real hardware:

- HIP on AMD discrete GPUs is expected to be comparable to CUDA.
- The hip-apu and sycl-igpu paths skip explicit memory copies (managed /
  shared USM), trading slightly higher kernel time for lower transfer
  overhead — expected within ~20% of the CUDA numbers.
- The CPU fallback is expected to add ~5-10 ms of per-frame CPU time at
  high-density configs (should remain real-time-capable for OS1-64 and
  OS1-128).

| Sensor config | Pixels/frame | RTX 3060 | RTX 3090 | RTX 4090 |
|---------------|-------------|----------|----------|----------|
| OS1-64 (512×64) | 33K | 40+ Hz | 40+ Hz | 40+ Hz |
| OS1-128 (1024×128) | 131K | 30 Hz | 40+ Hz | 40+ Hz |
| OS0-128 (1024×128) | 131K | 30 Hz | 40+ Hz | 40+ Hz |
| 2048×128 | 262K | 20 Hz | 30 Hz | 40+ Hz |
| 4096×128 | 524K | 10 Hz | 15 Hz | 25 Hz |
| 4096×512 | 2.1M | 5 Hz | 8-10 Hz | 12-15 Hz |

### Per-frame time budget breakdown (4096×512)

Numbers below are for the CUDA backend. HIP / SYCL costs are similar.
The CPU fallback collapses transfer rows to zero and inflates the
kernel rows to 5-10 ms each.

| Stage | Time | Notes |
|-------|------|-------|
| Panel depth passes (4 or 9) | 20-50 ms | **Dominant cost** -- GPU render |
| onPanelFrame memcpys | <1 ms | packed rig buffer copy |
| H2D transfer | ~1 ms | Raw frame to device (skipped on hip-apu / sycl-igpu) |
| resampleKernel | ~1 ms | 2M threads, panel projection + bilinear |
| rayProcessKernel | ~1 ms | 2M threads, noise + channels |
| D2H transfer | ~1 ms | Final channel results (skipped on hip-apu / sycl-igpu) |
| Packet encoding | 2-3 ms | 256 packets × 4 set_block calls |

### Tips for high-density configs

- **Reduce `<panel_oversample>`**: Panel resolution defaults to 2x the
  sensor's angular resolution. Dropping it toward 1.0 quarters the
  rendered pixels at the cost of more interpolation smoothing.
- **Stagger scan rates**: With multiple sensors, use different
  `lidar_hz` values (e.g. 10 Hz primary, 5 Hz secondary) to avoid
  simultaneous rig renders.
- **CPU fallback is viable** for low-density sensors (OS1-16, OS1-32).
  OpenMP parallelisation keeps resampling under 5 ms for <100K pixels.
- **GPU VRAM**: Each sensor uses the panel-rig render targets (logged
  as the raw-buffer size at startup) plus ~10-20 MB for backend device
  buffers (raw frame, channels, RNG state). The plugin logs the
  per-sensor breakdown on first frame; check the log if multi-sensor
  configs run into VRAM pressure.

## Tests

Tests are off by default (the workspace's `colcon_defaults.yaml` sets
`-DBUILD_TESTING=OFF`). To build + run them:

```bash
colcon build --packages-select gz_sensors_ouster --cmake-args -DBUILD_TESTING=ON
colcon test --packages-select gz_sensors_ouster
colcon test-result --verbose --test-result-base build/gz_sensors_ouster
```

| Binary | Coverage |
|--------|---------|
| `test_noise_model` | Range/signal/refl/nearir math + statistical bounds on dropouts and range noise |
| `test_resample` | Panel-rig layout (coverage, packing) + beam resample math (uniform range through cylindrical and hemispherical rigs, all-inf, far clip, beam-origin subtraction, azimuth offset, nearest-mode edge non-blending and quantisation bound) |
| `test_metadata_parsing` | Loads each shipped `config/metadata/*.json` via the Ouster SDK |
| `test_parameter_validation` | Clamping/validation rules for SDF + ROS-param inputs |
| `test_imu_noise` | IMU white-noise variance vs. density²/dt, bias drift growth, RNG-draw gating, determinism under fixed seed |
| `test_dispatch` | Backend selection: `GZ_OUSTER_BACKEND` override, auto fallback to CPU, `backendName()`/`usesCpuFallback()`, and `processRaw()` end-to-end through the `RayProcessor` wrapper |
| `test_raycast` | Full raycast mode: sphere/box/cylinder/plane/mesh intersectors, BVH vs brute-force equivalence, beam-origin parallax, response-map UV/RGBA semantics, retro of nearest hit, near-clip behaviour, zero-error uniform shell, and fused-vs-two-stage backend equivalence |
| `test_frame_exchange` | Lock-bounded newest-frame handoff, typed channel integrity, acquisition metadata, drop accounting, and cross-thread stress |
| `test_scan_timing` / `test_sim_time_scheduler` / `test_packet_pacing` | Rolling-shutter column timestamps, exact IMU deadlines, non-divisible physics cadence, rewind handling, bounded catch-up, and RTF-aware packet pacing |
| `test_lifecycle` | Plugin construct + destruct without ever calling `Configure` (catches member-init regressions; build-time vtable check against the vendored gz-sim) |

These tests run on the **CPU backend** — they exercise the shared math
(`ray_processor_math.hpp`), the class lifecycle, and the dispatcher's
backend selection, but not the GPU kernels themselves. The GPU kernels
are **compile-checked** on every PR by the `cuda-smoke`, `hip-smoke` and
`sycl-smoke` CI jobs, but are **not run on a device** in CI (no GPU
runners) — on-device verification must be done on real CUDA/ROCm/oneAPI
hardware. The shared-math refactor means the CPU tests now cover the same
arithmetic the GPU kernels execute, even though the kernels run elsewhere.

## Observability

The plugin emits a few categories of structured log lines worth
recognising when you're triaging behaviour in a running sim.

### One-time at startup

```
[gz_gpu_ouster_lidar] Using cuda backend.
[gz_gpu_ouster_lidar] HIP backend: device='Radeon 780M (gfx1103)' integrated=yes (managed-memory path ON)
```
Backend dispatch — `cuda` / `hip` / `hip-apu` / `sycl` / `sycl-igpu` / `cpu`.
Set `GZ_OUSTER_BACKEND=cpu` (or any backend name) to force a specific
choice for debugging.

```
Panel rig: 4 cylindrical panels (12.4 MiB raw): 1374x658 1374x658 1374x658 1374x658
Panel rig created: 4 depth cameras, beam altitude span [-23.6, 23.6] deg, cylindrical model
Configured: H=128 W=1024 cpp=16 sensor_name=/sensor/lidar/lidar0 ...
```
Dimensions and tuning derived from the loaded metadata.

### One-time on first frame

```
/sensor/lidar/lidar0: GPU buffers ~12.4 MiB (cuda backend) — raw=3.1 channels=0.6 resample=1.0 rand=7.7
```
Per-sensor backend memory footprint. Use this to budget multi-sensor
configs against available VRAM. The four numbers are the packed panel
rig buffer, the channel outputs (range/signal/reflec/nearir), the depth
intermediate, and the curand/hiprand state (zero on SYCL — its RNG is
counter-based and stateless).

### Repeated until acknowledged

```
metadata delivered to os_cloud (after 24 publishes)
```
Metadata republishing settles when both (a) `get_subscription_count() > 0`
and (b) ≥2 seconds of pubs have happened. If a subscriber drops to 0,
republishing automatically re-arms:

```
metadata subscriber count dropped to 0; re-arming republish
```

### Throttled WARN (5 s)

```
/sensor/lidar/lidar0: dropped rig frame (PostUpdate didn't drain); total dropped=37
```
The render thread fired a new frame before the sim thread consumed the
previous one. Rare under the lidar_hz throttle; sustained drops mean
PostUpdate is starved (sim-time stall, post-pause burst, or the GPU
pipeline can't keep up with the configured rate). The cumulative
counter is per-sensor and lifetime-of-process.

Warnings from downstream `ouster_ros` such as

```text
[os_cloud] [WARN] [lidar_packet_hander]: lidar_scans full, DROPPING PACKET
```

mean scan packets are arriving faster than that consumer completes scans. A
common simulation-specific cause is a world whose physics step is authored but
whose update rate is not, allowing sim time—and packet production—to run much
faster than wall time. Pace a custom world explicitly:

```xml
<max_step_size>0.004</max_step_size>
<real_time_factor>1.0</real_time_factor>
<real_time_update_rate>250</real_time_update_rate>
```

In general, `max_step_size × real_time_update_rate` should equal the requested
`real_time_factor`. All bundled worlds enforce that invariant. If the world is
already at 1× real time, the consumer is compute-bound: disable the unused
image decoder with `images:=false` or lower `lidar_hz`; enlarging its queue only
delays the same overload.

### Simulation time and bag playback rate

Acquisition time and delivery time are intentionally separate:

- Gazebo `info.simTime` schedules LiDAR/IMU captures and stamps native images,
  IMU messages, scan ends, and every packet column.
- A monotonic wall clock only spreads an already-stamped packet batch. Its
  drain span follows the observed producer interval, so live simulation at a
  non-1× real-time factor does not burst each scan at once.
- Pause freezes delivery; resume shifts pending deadlines instead of catching
  up in a burst. A world reset increments a time epoch and cancels pre-reset
  packets so timestamps from opposite sides of a rewind cannot form one scan.

The example `os_cloud` and `os_image` nodes use `use_sim_time:=true` and
`timestamp_mode:=TIME_FROM_INTERNAL_OSC`. They also pre-seed each decoder with
the same metadata file used by the simulated sensor. This makes the packet
subscription ready before playback begins instead of racing the bag's one-shot
metadata message at high rates; the live metadata topic remains subscribed for
updates. Consequently, `ros2 bag play --clock --rate 0.5` and `--rate 2.0`
change wall-clock delivery speed but preserve the recorded acquisition stamps
in clouds and images. Record `/clock`, `metadata`, and `lidar_packets` together.
At replay rates faster than the machine can decode, `ouster_ros` can still
report `lidar_scans full`; replay at a lower rate or omit `os_image` when only
clouds are needed.

### One-shot ERROR: Sensors system not rendering

```
events::Render has not fired after 2.0s of sim time — gz-sim's Sensors system
has not started rendering. ... Add a rendering sensor (pass anchor_type:=camera
in xacro, or switch to the default ray_mode:=raycast which needs no renderer) ...
```
You are using `panels` mode but the world has no rendering sensor, so
`gz-sim-sensors-system` never built the ogre2 scene. Either add a rendering
anchor (`anchor_type:=camera`) or switch to `ray_mode:=raycast` (the default),
which needs no render engine at all. See [World requirements](#world-requirements).

### `/imu` covariance

If you've enabled IMU noise (defaults are non-zero — see [IMU noise
model](#imu-noise-model)), the published `sensor_msgs/Imu` covariance
diagonals reflect the active per-sample noise std. Setting all four
noise params to 0 falls back to the ouster_ros default literals
(6e-4 / 0.01) so REP-145 consumers don't see literal-zero variances.

## Architecture

1. **Configure()** loads metadata, builds the panel-rig layout from the
   beam intrinsics (cylindrical sectors for OS0/1/2, pitched sectors +
   zenith cap for the OSDome), verifies that every calibrated beam ray is
   covered, creates ROS publishers, declares parameters
2. **OnRender()** (render thread) lazily creates one perspective depth
   camera per panel, then renders the whole rig each scan tick
3. **onPanelFrame()** fast-copies each panel's planar-depth buffer into its
   packed slot in a staging area (memcpy only, <1ms total)
4. **PostUpdate()** (sim thread) caches pose, swaps the raw frame out, dispatches to `encodeAndPublish()`, publishes IMU
5. **encodeAndPublish()** hands the packed rig buffer to the active
   `RayProcessor` backend (CUDA / HIP / SYCL / CPU — chosen at
   construction by probing for a usable device). The backend runs the
   resample kernel — for each beam (exact calibrated elevation, per-beam
   azimuth offset, encoder column) it projects the ray into the covering
   panel, bilinearly samples planar depth, and divides by the ray/axis
   cosine to recover Euclidean range — then the noise kernel
   (range/signal/reflectivity/near-IR with reflectivity-dependent
   effects), then PacketWriter encodes the result. The GPU backends use
   one device stream per sensor; the CPU backend uses OpenMP for
   resample and runs noise sequentially.
6. **drainThreadFunc()** publishes packets with rolling-shutter inter-packet
   timing. Pacing follows producer arrival time at the current real-time
   factor in both directions and uses 80% of the interval, leaving an idle
   tail for consumers to finish the completed scan. It stops while simulation
   is paused and cancels pre-reset batches. Packet/column **timestamps are sim
   time** and describe an idealised rotation across the scan period;
   the underlying data is a single instantaneous snapshot per scan by
   default; in raycast mode, `<motion_distortion>true</motion_distortion>`
   casts each column from its acquisition-time sensor pose instead
   (rolling-shutter sweep — [docs/MODEL_REFERENCES.md](docs/MODEL_REFERENCES.md)
   §9). Wall-clock spacing exists only to avoid bursting consumers.

With `<ray_mode>raycast</ray_mode>` steps 2-3 are replaced by an ECM scene
mirror (visual geometries extracted on geometry/material changes, world poses
refreshed per scan) and a worker thread that casts every beam exactly against
it (`cuda/raycast_scene.{hpp,cpp}`: analytic primitives + per-mesh triangle
BVH). CUDA keeps the depth, reflectance, and near-IR intermediates on one
device stream, runs channel/noise synthesis immediately after the cast, and
transfers only the compact final channels. HIP/SYCL currently use the exact
two-stage host composition; CPU uses OpenMP casting. The cast produces exact
per-beam ranges plus the **apparent reflectance** per hit —
`laser_retro × cos(incidence)`, the extended-Lambertian lidar-equation
factor (see [docs/MODEL_REFERENCES.md](docs/MODEL_REFERENCES.md)) — which
skips the panel resample kernel. The ray origins sit on the beam-origin circle
and the reported range follows the Ouster XYZ-LUT convention, so consumers
reconstruct the true hit points exactly.

## License

Apache-2.0
