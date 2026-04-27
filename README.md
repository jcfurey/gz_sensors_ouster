# gz_sensors_ouster

Gazebo Harmonic system plugin that simulates Ouster LiDAR sensors
(OS0, OS1, OS2) with GPU-accelerated ray casting, realistic noise
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
| `.../camera_info` | `sensor_msgs/CameraInfo` | lidar_hz | Range-image camera metadata (H×W, frame_id) |
| `.../imu_packets` | `ouster_sensor_msgs/PacketMsg` | imu_hz | Native Ouster IMU packets (if IMU enabled) |
| `.../imu` | `sensor_msgs/Imu` | imu_hz | Standard ROS IMU message (if IMU enabled) |

Image, CameraInfo, and IMU topics are only published when subscribers are present.

## Prerequisites

- **Gazebo Harmonic** (gz-sim8, gz-rendering8, gz-sensors8)
- **ROS 2** (Jazzy or later) with `rclcpp`, `sensor_msgs`, `std_msgs`
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

Add one `<plugin>` block per sensor. Each gets its own topics, GpuRays
instance, GPU stream (or CPU OpenMP pool), and drain thread. Use
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

Each sensor adds ~100 MB GPU VRAM (Ogre2 cubemap) and two threads
(ROS executor + drain). Ogre2 renders sensors sequentially on the
render thread, so 2-3 sensors at 10 Hz is comfortable. For 4+
sensors, consider staggering scan rates or reducing beam density.

## Included Metadata Files

Example Ouster calibration JSONs are provided in `config/metadata/` for
simulation without real hardware:

| File | Sensor | Beams | VFOV | Beam Angles | Notes |
|------|--------|-------|------|-------------|-------|
| `os1_64_rev7.json` | OS1-64 | 64 | 33.2° | Real (from SDK source) | Default, recommended for testing |
| `os0_128_rev7.json` | OS0-128 | 128 | 90° | Nominal (uniform spacing) | Ultra-wide short-range |
| `os1_128_rev7.json` | OS1-128 | 128 | 45° | Nominal | High-density mid-range |
| `os2_128_rev7.json` | OS2-128 | 128 | 22.5° | Nominal | Long-range narrow |
| `osdome_128_rev7.json` | OSDome-128 | 128 | 180° | Nominal | Hemispheric (experimental) |

All files use the `RNG19_RFL8_SIG16_NIR16` lidar profile, `LEGACY` IMU
profile, and 1024 columns/frame at 10 Hz. `max_range` is auto-derived
from `prod_line` when not set in SDF.

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
| `visibility_mask` | 4294967295 | 0 to 4294967295 | Gazebo render visibility mask applied to GpuRays. Use to include or exclude visuals from raycasting. |

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

The rendering bottleneck is Ogre2's GpuRays cubemap (6 faces per
frame). The GPU post-processing pipeline (resample + noise) adds only
~2-3 ms per frame regardless of sensor density on any of the supported
backends. All numbers assume a single sensor.

### Estimated max real-time scan rate

Numbers below were measured on NVIDIA RTX GPUs with the CUDA backend
active. The HIP backend on AMD discrete GPUs is comparable; the
hip-apu and sycl-igpu paths skip explicit memory copies (managed /
shared USM) and trade slightly higher kernel time for lower transfer
overhead — expect within 20% of the CUDA numbers below. The CPU
fallback path adds ~5-10 ms of per-frame CPU time at high-density
configs (still real-time-capable for OS1-64 and OS1-128).

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
| Ogre2 cubemap (6 faces) | 20-50 ms | **Dominant cost** -- GPU raycast |
| onNewFrame memcpy | <1 ms | 24 MB raw buffer copy |
| H2D transfer | ~1 ms | Raw frame to device (skipped on hip-apu / sycl-igpu) |
| resampleKernel | ~1 ms | 2M threads, bilinear interp |
| rayProcessKernel | ~1 ms | 2M threads, noise + channels |
| D2H transfer | ~1 ms | Final channel results (skipped on hip-apu / sycl-igpu) |
| Packet encoding | 2-3 ms | 256 packets × 4 set_block calls |

### Tips for high-density configs

- **Reduce `v_samples`**: The cubemap resolution scales with vertical
  ray count. For sensors where some vertical aliasing is acceptable,
  the `v_samples` calculation in `OnRender()` can be tuned down.
- **Stagger scan rates**: With multiple sensors, use different
  `lidar_hz` values (e.g. 10 Hz primary, 5 Hz secondary) to avoid
  simultaneous cubemap renders.
- **CPU fallback is viable** for low-density sensors (OS1-16, OS1-32).
  OpenMP parallelisation keeps resampling under 5 ms for <100K pixels.
- **GPU VRAM**: Each sensor uses ~100 MB for the Ogre2 cubemap plus
  ~10-20 MB for backend device buffers (raw frame, channels, RNG state).
  The plugin logs the per-sensor breakdown on first frame; check the
  log if multi-sensor configs run into VRAM pressure.

## Architecture

1. **Configure()** loads metadata, creates ROS publishers, declares parameters
2. **OnRender()** (render thread) lazily creates GpuRays, triggers GPU raycast
3. **onNewFrame()** fast-copies the raw GpuRays buffer into a staging area (memcpy only, <1ms)
4. **PostUpdate()** (sim thread) caches pose, swaps the raw frame out, dispatches to `encodeAndPublish()`, publishes IMU
5. **encodeAndPublish()** hands the raw frame to the active `RayProcessor`
   backend (CUDA / HIP / SYCL / CPU — chosen at construction by probing
   for a usable device). The backend runs the resample kernel
   (bilinear interpolation to exact Ouster beam geometry) → noise kernel
   (range/signal/reflectivity/near-IR with reflectivity-dependent
   effects), then PacketWriter encodes the result. The GPU backends use
   one device stream per sensor; the CPU backend uses OpenMP for
   resample and runs noise sequentially.
6. **drainThreadFunc()** publishes packets with rolling-shutter inter-packet timing

## License

Apache-2.0
