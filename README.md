# gz_gpu_ouster_lidar

Gazebo Harmonic system plugin that simulates Ouster LiDAR sensors
(OS0, OS1, OS2) with GPU-accelerated ray casting, realistic noise
models, and native Ouster packet output. Downstream nodes like
`ouster_ros` `os_cloud` consume the packets identically to real
hardware -- no driver changes needed.

## Features

- Per-beam elevation and azimuth geometry from Ouster calibration JSON
- CUDA post-processing with CPU fallback (range noise, signal model,
  dropouts, edge suppression)
- Native `PacketMsg` encoding via Ouster SDK `PacketWriter`
  (RANGE, SIGNAL, REFLECTIVITY, NEAR_IR channels)
- Simulated IMU packets from Gazebo's IMU sensor (optional)
- Latched metadata republishing for rmw_zenoh_cpp compatibility
- Rolling-shutter packet timing via drain thread

## Published Topics

All topics are published under `<sensor_name>` (e.g. `/sensor/lidar/lidar0`).

| Topic suffix | Type | Rate | Description |
|--------------|------|------|-------------|
| `<sensor_name>/lidar_packets` | `ouster_sensor_msgs/PacketMsg` | lidar_hz | Native Ouster lidar packets |
| `<sensor_name>/metadata` | `std_msgs/String` | Latched | Ouster calibration JSON |
| `<sensor_name>/range_image` | `sensor_msgs/Image` | lidar_hz | Range in mm (mono16) |
| `<sensor_name>/signal_image` | `sensor_msgs/Image` | lidar_hz | Signal photon counts (mono16) |
| `<sensor_name>/reflec_image` | `sensor_msgs/Image` | lidar_hz | Reflectivity (mono16) |
| `<sensor_name>/nearir_image` | `sensor_msgs/Image` | lidar_hz | Near-IR (mono16) |
| `<sensor_name>/imu_packets` | `ouster_sensor_msgs/PacketMsg` | imu_hz | Native Ouster IMU packets |
| `<sensor_name>/imu` | `sensor_msgs/Imu` | imu_hz | Standard IMU message |

Image and IMU topics are only published when subscribers are present.

## Build

```bash
colcon build --packages-select gz_gpu_ouster_lidar
```

Requires: Gazebo Harmonic (gz-sim8), ROS 2, Ouster SDK (submodule),
Eigen3. CUDA is optional (falls back to CPU).

## SDF Usage

See [`config/plugin_example.sdf`](config/plugin_example.sdf) for the
full example with all parameters.

Minimal:

```xml
<plugin filename="libgz_gpu_ouster_lidar.so"
        name="gz_gpu_ouster_lidar::GzGpuOusterLidarSystem">
  <metadata_path>os1_64_sim.json</metadata_path>
  <sensor_name>/sensor/lidar/lidar0</sensor_name>
  <lidar_hz>10.0</lidar_hz>
</plugin>
```

## Parameters

### Required

| Parameter | Type | Description |
|-----------|------|-------------|
| `metadata_path` | string | Path to Ouster calibration JSON. Supports absolute paths and relative (resolved against SDF directory). URI schemes (`model://`, `package://`) are passed through unmodified. |
| `sensor_name` | string | ROS topic prefix and node namespace (e.g. `/sensor/lidar/lidar0`) |

### Lidar

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `lidar_hz` | 10.0 | > 0 | Scan rate in Hz |
| `max_range` | 120.0 | >= 1 | Max sensing range in metres. Also sets the GPU far clip plane. OS0: 50, OS1: 120, OS2: 240. |

### Noise Model

Defaults are tuned for OS1-64 rev6. Override for OS0/OS2.

| Parameter | Default | Range | Units | Description |
|-----------|---------|-------|-------|-------------|
| `range_noise_min_std` | 0.002 | >= 0 | m | Range noise sigma at 0 m. Linearly interpolated to max_std at max_range. |
| `range_noise_max_std` | 0.008 | >= 0 | m | Range noise sigma at max_range. |
| `signal_noise_scale` | 1.0 | >= 0 | -- | Poisson shot noise on signal channel. 0 = off, 1 = physical. |
| `nearir_noise_scale` | 1.0 | >= 0 | -- | Poisson noise on near-IR image topic. |
| `base_signal` | 800.0 | >= 0 | photon m^2 | Baseline for 1/r^2 signal model. OS0: ~400, OS1: ~800. |
| `base_reflectivity` | 50.0 | 0-255 | -- | Default reflectivity when no retro data available. |
| `dropout_rate_close` | 0.0002 | 0-1 | probability | Random miss rate at 0 m. |
| `dropout_rate_far` | 0.015 | 0-1 | probability | Random miss rate at max_range. |
| `edge_discon_threshold` | 0.15 | >= 0 | m | Depth-discontinuity suppression threshold. 0 = off. |

### IMU (optional)

Requires an IMU sensor in the SDF/URDF and the `gz-sim-imu-system`
plugin loaded in the world.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `imu_name` | *(disabled)* | -- | Gazebo IMU sensor entity name. Omit to disable IMU. |
| `imu_hz` | 100.0 | > 0 | IMU publish rate in Hz. |
| `publish_imu_msg` | true | bool | Also publish `sensor_msgs/Imu` alongside Ouster IMU packets. |

## Sensor Tuning Guide

| Sensor | `max_range` | `base_signal` | `range_noise_min_std` | `range_noise_max_std` | `edge_discon_threshold` |
|--------|-------------|---------------|-----------------------|-----------------------|--------------------------|
| OS0-128 | 50 | 400 | 0.003 | 0.015 | 0.20 |
| OS1-64 (default) | 120 | 800 | 0.002 | 0.008 | 0.15 |
| OS2-128 | 240 | 1200 | 0.002 | 0.010 | 0.10 |

## Architecture

1. **Configure()** loads metadata, creates ROS publishers, connects render event
2. **OnRender()** (render thread) lazily creates GpuRays, triggers GPU raycast
3. **onNewFrame()** resamples uniform GpuRays grid to exact Ouster beam geometry via 2D bilinear interpolation
4. **PostUpdate()** (sim thread) caches pose, dispatches lidar frames to CUDA, publishes IMU
5. **encodeAndPublish()** runs CUDA noise model, encodes packets via PacketWriter
6. **drainThreadFunc()** publishes packets with rolling-shutter inter-packet timing
