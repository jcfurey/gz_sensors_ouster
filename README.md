# gz_sensors_ouster

Gazebo system plugin (Harmonic / Ionic / Jetty) that simulates Ouster
LiDAR sensors (OS0, OS1, OS2) with GPU-accelerated ray casting, realistic noise
models, and native Ouster packet output. Downstream nodes like
`ouster_ros` `os_cloud` consume the packets identically to real
hardware -- no driver changes needed.

## Features

- Per-beam elevation and azimuth geometry from Ouster calibration JSON
- Multi-vendor GPU post-processing: **CUDA** (NVIDIA), **HIP** (AMD incl.
  APUs, with unified-memory fast path), **SYCL** (Intel iGPU + Arc, via
  oneAPI DPC++ or AdaptiveCpp). Automatic CPU fallback when no GPU
  toolchain is compiled in or no device is found at runtime.
- Native `PacketMsg` encoding via Ouster SDK `PacketWriter`
  (RANGE, SIGNAL, REFLECTIVITY, NEAR_IR channels)
- Simulated IMU packets from Gazebo's IMU sensor (optional, auto-detect)
- Noise parameters reconfigurable at runtime via `ros2 param set`
- Latched metadata republishing for rmw_zenoh_cpp compatibility
- Rolling-shutter packet timing via drain thread
- Works on **any GPU** for ray casting (OGRE2 + OpenGL is vendor-agnostic);
  the GPU noise path is additionally accelerated on NVIDIA/AMD/Intel.

## Published Topics

All topics are published under the configured `<sensor_name>` prefix.

For example, with `<sensor_name>/sensor/lidar/lidar0</sensor_name>`:

| Topic | Type | Rate | Description |
|-------|------|------|-------------|
| `.../lidar_packets` | `ouster_sensor_msgs/PacketMsg` | lidar_hz | Native Ouster lidar packets |
| `.../metadata` | `std_msgs/String` | Latched | Ouster calibration JSON |
| `.../range_image` | `sensor_msgs/Image` | lidar_hz | Range in mm (mono16) |
| `.../signal_image` | `sensor_msgs/Image` | lidar_hz | Signal photon counts (mono16) |
| `.../reflec_image` | `sensor_msgs/Image` | lidar_hz | Reflectivity (mono16) |
| `.../nearir_image` | `sensor_msgs/Image` | lidar_hz | Near-IR (mono16) |
| `.../camera_info` | `sensor_msgs/CameraInfo` | lidar_hz | Range-image camera metadata (H×W, frame_id). `distortion_model` is the non-standard string `equirectangular`: u is linear in azimuth, v linear in elevation; fx/fy in K are pixels-per-radian. Standard pinhole/fisheye consumers must not reproject with it. |
| `.../imu_packets` | `ouster_sensor_msgs/PacketMsg` | imu_hz | Native Ouster IMU packets (if IMU enabled) |
| `.../imu` | `sensor_msgs/Imu` | imu_hz | Standard ROS IMU message (if IMU enabled) |

Image, CameraInfo, and IMU topics are only published when subscribers are present.

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

### World requirements (read this if no point cloud is published)

This plugin does **not** register a `<sensor type="gpu_lidar">`. It creates its
own rig of perspective depth cameras inside the ogre2 scene that
`gz-sim-sensors-system` owns, and it is
driven by that system's `events::Render` event. Therefore the world must:

1. Load the Sensors system with the ogre2 engine:
   ```xml
   <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
     <render_engine>ogre2</render_engine>
   </plugin>
   ```
> **Exception:** with `<ray_mode>raycast</ray_mode>` neither requirement
> applies — the raycast mode never touches the renderer. The two points
> below concern the default `panels` mode only.

2. Contain **at least one rendering sensor** (`camera`, `gpu_lidar`,
   `depth_camera`, …). On Gazebo Harmonic the Sensors system only initialises
   rendering — building the scene and emitting `events::Render` — once such a
   sensor exists in the ECM. With **only** non-rendering sensors (e.g. an
   `altimeter` pose anchor plus an `imu`), the Sensors system never starts
   rendering, `OnRender()` never fires, the panel rig is never created, and **no
   point cloud is produced**.

The bundled examples satisfy (2) by defaulting the pose-anchor sensor to a tiny
**`camera`** (see `examples/urdf/ouster_macro.xacro`, `anchor_type:=camera`) —
the cheapest renderer, which bootstraps the scene **without** adding a second
lidar. Use `anchor_type:=gpu_lidar` if you also want a native gz scan on
`<sensor_name>/gz_native_scan` (note: that is a second lidar raycast source and
will show as an extra cloud if visualised). If this requirement is unmet the
plugin logs a one-shot error after ~2 s of sim time: *"events::Render has not
fired … add a rendering sensor"*.

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
| `urdf/turtlebot3_ouster.urdf.xacro` | A drivable TurtleBot3 waffle carrying the Ouster (used by the Docker test) |
| `worlds/ouster_demo.sdf` | Demo world (physics + sensors + IMU systems, ground + obstacles) |
| `worlds/turtlebot3_ouster_headless.sdf` | GPU-free arena (no rendering Sensors system) for raycast mode |
| `launch/ouster_standalone.launch.py` | Bring up the standalone example end-to-end |
| `launch/sensor_stack.launch.py` | Bring up the multi-sensor example |
| `launch/turtlebot3_ouster.launch.py` | Bring up the TurtleBot3 waffle + Ouster (drivable, raycast by default) |

Run (after `colcon build` + `source install/setup.bash`):

```bash
ros2 launch gz_sensors_ouster ouster_standalone.launch.py
# multi-sensor: ros2 launch gz_sensors_ouster sensor_stack.launch.py
# with RViz:    ros2 launch gz_sensors_ouster ouster_standalone.launch.py rviz:=true
```

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
`point_cloud_frame:=lidar0/lidar_frame` and `pub_static_tf:=false` so the
cloud lands in the `robot_state_publisher` TF tree (RViz fixed frame
`base_footprint`) without a duplicate static-transform broadcaster. The
RViz config includes a `PointCloud` display for that topic.

### How the URDF wires to the plugin

The plugin is a Gazebo **system** plugin on the model; it casts its own rays
rather than using a gz `<sensor type="gpu_lidar">`. The macro takes a
**`ray_mode`** (default **`raycast`**) and emits, per sensor:

- A **pose-anchor `<sensor>`** named exactly like the last segment of
  `<sensor_name>` (e.g. `lidar0`). The plugin looks this entity up to read its
  world pose (the ray-cast origin). What it must be depends on `ray_mode`:
  - **`raycast`** (default) — beams are cast on the CPU against an ECM scene
    mirror, with no render engine, so a non-rendering **`altimeter`** anchor is
    enough and the world needs no GPU. Pass `anchor_type:=altimeter`.
  - **`panels`** — the plugin drives a GpuRays rig off `events::Render`, so the
    anchor must be a *rendering* sensor and the world must load
    `gz-sim-sensors-system` (see [World
    requirements](#world-requirements-read-this-if-no-point-cloud-is-published)).
    It defaults to a minimal **`camera`** (cheapest renderer, not a lidar, so no
    second scan); `anchor_type:=gpu_lidar` also emits a native gz scan on
    `<sensor_name>/gz_native_scan`.
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
  robot's lidar frame. It is **not** put in `os_lidar`: the metadata's
  `lidar_to_sensor_transform` is the real Ouster 180° + 36 mm offset, which
  would rotate the simulated cloud. (Set `point_cloud_frame:=<name>/os_lidar`
  if you want that physical offset applied.)
- `pub_static_tf:=true` with `sensor_frame:=<name>/lidar_frame` — the ouster
  driver broadcasts its own `<name>/lidar_frame → <name>/os_lidar` and
  `→ <name>/os_imu` static transforms from the metadata.
- A `static_transform_publisher` publishes `base_link → <name>/lidar_frame`
  (matching the URDF mount joint), so the cloud reaches `base_link` even if
  `robot_state_publisher` is not running. When RSP is up it publishes the same
  edge, which is harmless (a one-time `TF_REPEATED_DATA` warning).

Net TF chain: `points (<name>/lidar_frame) → base_link → base_footprint`, via
both the launch stack (RSP / the mount `static_transform_publisher`) and the
ouster driver.

## Docker (standalone test)

A self-contained [`Dockerfile`](Dockerfile) builds the plugin in isolation —
ROS 2 + the new Gazebo + the pinned `ouster-ros` fork + this package — and
exercises it on a **drivable TurtleBot3 waffle** ("a vehicle for the lidar to
ride on"). No CUDA/GPU toolchain is installed, so the plugin builds its CPU
(OpenMP) backend and the default smoke runs in **raycast** mode, which needs no
render engine and so runs anywhere — even with no GPU.

`ROS_DISTRO` selects the distro (matching this package's CI matrix); **Humble is
not supported** (no `gz_*_vendor`; it ships Gazebo Fortress, not Harmonic+):

```bash
cd <this package>

# Build (default jazzy → Harmonic; kilted → Ionic; lyrical → Jetty, advisory)
docker build -t gzouster .
docker build -t gzouster --build-arg ROS_DISTRO=kilted .

# 1) Headless point-cloud smoke (no GPU): waits for a PointCloud2 on
#    /sensor/lidar/lidar0/points and exits PASS/FAIL.
docker run --rm gzouster

# 2) Re-run the gtest suite.
docker run --rm gzouster test

# 3) Interactive: gz GUI + RViz + teleop_twist_keyboard on /cmd_vel (needs a
#    display; --gpus all only if you want the GUI to render on the NVIDIA card).
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

### Using the host GPU (CUDA backend)

By default the image installs no CUDA toolchain, so the plugin runs its CPU
(OpenMP) backend. To build the **CUDA backend** and run it on the **host's** GPU
— without pulling in the full `rovermax_ws` image — build with `ENABLE_CUDA=true`
and run with `--gpus all`. Only the CUDA *toolkit* (nvcc/cudart/curand) is baked
into the image; the driver/`libcuda` comes from the host at run time (needs
`nvidia-container-toolkit` on the host — no host CUDA install required):

```bash
# CUDA_ARCH: 86 = Ampere (RTX 30xx), 89 = Ada (40xx), or "75;80;86;89" for portability.
docker build -t gzouster-cuda --build-arg ENABLE_CUDA=true --build-arg CUDA_ARCH=86 .

docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all gzouster-cuda
# the plugin logs "Using cuda backend." instead of the CPU-fallback warning.
```

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
IMU profile, and 1024 columns/frame at 10 Hz. `max_range` is auto-derived
from `prod_line` when not set in SDF.

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
| `sensor_name` | string | ROS topic prefix and node namespace (e.g. `/sensor/lidar/lidar0`). |

### Lidar

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `lidar_hz` | 10.0 | > 0 | Scan rate in Hz. |
| `max_range` | *auto* | >= 1 | Max sensing range in metres. Auto-derived from metadata `prod_line` if not set (OS0: 50, OS1: 120, OS2: 240). Also sets the GPU far clip plane. |
| `visibility_mask` | 4294967295 | 0 to 4294967295 | Gazebo render visibility mask applied to the panel cameras. Use to include or exclude visuals from raycasting. |
| `ray_mode` | `panels` | `panels` \| `raycast` | `panels` renders a perspective depth-panel rig on the GPU and resamples each beam from it. `raycast` casts every beam exactly (calibrated direction, true beam-origin parallax) against an ECM scene mirror — zero interpolation error, `laser_retro` drives reflectivity, no rendering involved (no anchor-sensor requirement). CPU/OpenMP; mirrors box/sphere/cylinder/plane/mesh visuals. |
| `panel_oversample` | 2.0 | 1 to 4 | Panels mode only. Panel angular resolution as a multiple of the sensor's finest angular resolution. Higher = sharper edges, more VRAM and render time. |
| `panel_sampling` | `bilinear` | `bilinear` \| `nearest` | `bilinear` interpolates the 4 neighbouring rendered rays (smooth surfaces, but silhouettes blend fore/background range). `nearest` takes the single closest rendered ray — a true raycast with direction quantised to the pixel grid (≤ 1/(2·oversample) of the beam spacing) and no range blending at depth edges. |

### QoS overrides

`lidar_packets` always uses `SensorDataQoS` (BEST_EFFORT) because that's
what `os_cloud` expects. Image, camera_info, and IMU pubs are
configurable for deployments that need a specific QoS to match their
consumer — necessary on rmw_zenoh_cpp where pub and sub QoS must match
exactly (neither BEST_EFFORT-pub-to-RELIABLE-sub nor the reverse work).

| Parameter | Default | Accepted values | Affects |
|-----------|---------|-----------------|---------|
| `image_qos` | `reliable` | `reliable` \| `best_effort` \| `sensor_data` | range/signal/reflec/nearir image + camera_info |
| `imu_qos`   | `sensor_data` | `reliable` \| `best_effort` \| `sensor_data` | imu + imu_packets |

Defaults match the most common consumers: `reliable` for images
(matches RViz Image display, rqt_image_view, image_transport), and
`sensor_data` (BEST_EFFORT) for IMU (matches the real `ouster_ros`
driver so a sim-to-hardware topic swap doesn't require flipping
subscriber QoS).

### Noise Model

Defaults are validated against ISPRS OS1-64 accuracy assessment and
Ouster FW 1.13+ datasheets. All are dynamically reconfigurable.

Dropout and range noise also scale with surface reflectivity: dark
surfaces (low retro) get up to 3x higher dropout and 2x more range
noise. This matches the real sensor behavior where low-reflectivity
targets produce weaker returns.

| Parameter | Default | Range | Units | Description |
|-----------|---------|-------|-------|-------------|
| `range_noise_min_std` | 0.003 | >= 0 | m | Range noise sigma at 0 m. Linearly interpolated to max_std at max_range. Scales with reflectivity. |
| `range_noise_max_std` | 0.015 | >= 0 | m | Range noise sigma at max_range. |
| `signal_noise_scale` | 1.0 | >= 0 | -- | Poisson shot noise on signal channel. 0 = off, 1 = physical. |
| `nearir_noise_scale` | 1.0 | >= 0 | -- | Poisson noise on near-IR channel (both packet and image). |
| `base_signal` | 800.0 | >= 0 | photon m^2 | Baseline for 1/r^2 signal model. OS0: ~400, OS1: ~800. |
| `base_reflectivity` | 50.0 | 0-255 | -- | Default reflectivity when no retro data available. |
| `dropout_rate_close` | 0.0005 | 0-1 | probability | Random miss rate at 0 m. Scales with reflectivity (low retro = more drops). |
| `dropout_rate_far` | 0.03 | 0-1 | probability | Random miss rate at max_range. |
| `edge_discon_threshold` | 0.15 | >= 0 | m | Depth-discontinuity suppression threshold (1ns echo delay convention). 0 = off. |

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

| Sensor | `max_range` | `base_signal` | `range_noise_min_std` | `range_noise_max_std` | `dropout_rate_far` | `edge_discon_threshold` |
|--------|-------------|---------------|-----------------------|-----------------------|--------------------|--------------------------|
| OS0-128 | 50 | 400 | 0.005 | 0.020 | 0.05 | 0.20 |
| OS1-64 (default) | 120 | 800 | 0.003 | 0.015 | 0.03 | 0.15 |
| OS2-128 | 240 | 1200 | 0.003 | 0.020 | 0.04 | 0.10 |

> `max_range` is auto-derived from the metadata JSON's `prod_line` field
> when not explicitly set. The other values must be overridden in SDF if
> you are not using the OS1 defaults.

## Performance

The rendering bottleneck is the panel rig's depth passes (4 for
cylindrical sensors, 9 for the OSDome). The GPU post-processing
pipeline (resample + noise) adds only
~2-3 ms per frame regardless of sensor density on any of the supported
backends. All numbers assume a single sensor.

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
| `test_raycast` | Full raycast mode: sphere/box/cylinder/plane/mesh intersectors, BVH vs brute-force equivalence on a random triangle soup, beam-origin parallax (XYZ-LUT invariant), retro of nearest hit, near-clip behaviour, zero-error uniform shell, `processDepth` through the dispatcher |
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

### One-shot ERROR: Sensors system not rendering

```
events::Render has not fired after 2.0s of sim time — gz-sim's Sensors system
has not started rendering. ... Add a rendering sensor (the example URDF's
anchor_type defaults to a camera) ...
```
The world has no rendering sensor, so `gz-sim-sensors-system` never built the
ogre2 scene this plugin attaches to. See
[World requirements](#world-requirements-read-this-if-no-point-cloud-is-published).

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
   timing. Pacing follows the *observed* wall-clock scan cadence (capped at
   the nominal 1/lidar_hz) so running the sim faster than real time doesn't
   back the drain up. Note the timing semantics: packet/column **timestamps
   are sim time** and describe an idealised rotation across the scan period;
   the underlying data is a single instantaneous snapshot per scan (no
   rolling-shutter geometry), and wall-clock spacing exists only to avoid
   bursting consumers.

With `<ray_mode>raycast</ray_mode>` steps 2-3 are replaced by an ECM scene
mirror (visual geometries extracted once, world poses refreshed per scan,
spawn/despawn triggers a rebuild) and a worker thread that casts every
beam exactly against it (`cuda/raycast_scene.{hpp,cpp}`: analytic
primitives + per-mesh triangle BVH, OpenMP). The worker output is exact
per-beam ranges plus `laser_retro` per hit, which skips the resample
kernel and enters the pipeline at the noise stage
(`RayProcessor::processDepth`). The ray origins sit on the beam-origin
circle and the reported range follows the Ouster XYZ-LUT convention, so
consumers reconstruct the true hit points exactly.

## License

Apache-2.0
