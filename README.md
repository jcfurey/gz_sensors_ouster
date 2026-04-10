# gz_gpu_ouster_lidar

Gazebo Harmonic system plugin that:

1. Creates a `gz::rendering::GpuRays` sensor.
2. Resamples vertical rays to exact Ouster per-beam elevations from metadata.
3. Runs CUDA post-processing to synthesize Ouster channels:
   - `RANGE` (mm)
   - `SIGNAL`
   - `REFLECTIVITY`
4. Encodes native Ouster packets with `PacketWriter`.
5. Publishes `ouster_sensor_msgs/msg/PacketMsg` and latched `metadata`.

## Build

Inside the robot container:

```bash
colcon build --packages-select gz_gpu_ouster_lidar
```

## SDF Plugin Snippet

See `config/plugin_example.sdf`.

Required plugin parameters:

- `metadata_path`: Path to Ouster metadata JSON (required)
- `sensor_name`: Logical sensor name
- `lidar_hz`: Scan rate

## Notes

- This plugin follows the same Ouster SDK pattern used in `gz_ouster_packet_bridge`.
- Gazebo `GpuRays` vertical sampling is uniform; this plugin oversamples vertical rays
  and interpolates at exact per-beam Ouster angles.
- CUDA architectures are set to `75;80;86;89`.
