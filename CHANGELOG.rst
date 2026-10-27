^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package gz_sensors_ouster
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Forthcoming
-----------
Changes since 1.0.0, the state of ``main`` on 2026-06-18.

**Sensor model**

* [BREAKING]: Rev7 OS0 and OS1 report SIGNAL from the response measured on Ouster's public
  sample recordings instead of ``base_signal * rho / r^2``: the near-field overlap of beam and
  receiver, the detector's saturation near 5,700 counts at 1024x10, shot noise above Poisson at
  low counts and below it near saturation, and half the counts at 2048x10. An OS1 at 2.2 m on a
  29% surface reads about 3,200 counts with 1-2% scan-to-scan noise, where the old model gave 44
  with 15%. Every backend evaluates it. The new ``signal_model`` option (``auto``, ``measured``
  or ``inverse_square``) selects the model, and ``base_signal`` still scales sensitivity (800
  reproduces the recordings). Products without recordings keep the inverse-square model.
* [BREAKING]: Retroreflectors encode on Ouster's calibrated reflectivity scale, where byte 255 is
  864x a 100% diffuse target: 15.89 bytes per doubling above 100 instead of 22, which saturated
  at 132x. Diffuse targets are unchanged; a target 10x a white one reads 152 instead of 173.
* [BUGFIX]: ``base_reflectivity`` 53 and 59 read back one byte low on surfaces without
  ``laser_retro``, because ``53 / 100`` rounds to 0.52999997 in float; 214 and 240 did the same
  at the new reflectivity slope. Every byte now reads back as itself, and the default 50 stays
  exactly 0.5.
* Model Ouster hardware generations and revisions (``hardware_revision``: Gen1 through Rev8, or
  ``auto`` from the metadata part number): minimum and representable range, the published D90
  detection points with a smooth D90-to-D50 rolloff, precision, range resolution, beam
  divergence and return capability, scaled for the active scan mode.
* Spatial response textures in raycast mode: a ``<stem>.ouster.png`` beside a visual's albedo
  map supplies its 865 nm reflectance, near-IR albedo, specular coefficient and opacity per
  texel. The TurtleBot warehouse, hills and sewer worlds use one.
* [BUGFIX]: panels mode reported every range 1.4-2.8 cm short of the Ouster XYZ-LUT convention
  (the beam-origin offset was dropped); both ray modes now agree with the LUT.
* [BUGFIX]: metadata transforms, the detection rolloff, glass traversal and scene rebuilds are
  corrected.

**Smoke, dust and fog (raycast mode)**

* Beams integrate through participating media: two-way Beer-Lambert extinction of the apparent
  reflectance, which dims SIGNAL and REFLECTIVITY until targets drop out; backscatter returns
  from the medium, gated by its photon count so clouds stay sparse; and NEAR_IR airlight.
  Volumes are ``<obscurant>`` blocks (box, ellipsoid or cylinder, by extinction or visibility);
  ``examples/worlds/ouster_smoke.sdf`` demonstrates them.
* Platt's multiple-scattering factor (``<multiple_scattering>``,
  ``obscurant_multiple_scattering``, default 1) credits back forward-scattered light in dense
  media.
* Gazebo particle emitters stay visual-only unless ``particle_obscuration`` mirrors their
  envelopes as volumes.
* [BUGFIX]: a surface without ``laser_retro`` takes ``base_reflectivity`` before the medium
  attenuates it, like an authored material.
* [BUGFIX]: surface, behind-glass and mirror-ghost candidates are attenuated before the strongest
  return is chosen, so a pane in smoke no longer wins over a brighter object behind it, and a
  mirror ghost composes the transmittance of its two legs. Inconsistent albedo and lidar-ratio
  pairs are reported.
* [BUGFIX]: the medium's range sampler keeps 2e-6 relative accuracy in thin media, where it lost
  up to 5.3e-4 (about 6 cm at 120 m).

**Ray casting and performance**

* A top-level BVH over instances makes casts 2-7x faster in instance-rich scenes, on the CPU,
  CUDA, HIP and SYCL backends, with identical results.
* Raycast mode captures moving models at each of ``object_motion_samples`` (default 16) column
  groups as the sweep reaches them, so a moving object appears where it was when each column
  was acquired: within 0.4-0.6 ms of each point's time on average, against 46-55 ms before.
* CPU raycast threads take one beam row at a time, which cut a 1024x64 workcell scan from
  3.2/4.4 ms to 2.5/2.9 ms with identical output.
* The CUDA backend casts and processes a scan on one stream without a host round trip, and scans
  are scheduled on simulation time.
* Unseeded CPU noise processing runs in parallel.

**Packets, IMU and timing**

* Metadata, product profiles, packet bytes and pacing come from the ROS-free ``ouster_sim_core``
  submodule shared with AGX (now 0.7.0), exported through a ``gz_sensors_ouster_core`` facade
  and checked by the shared conformance fixtures. Complete frames survive transport congestion.
* Packets follow the metadata ``column_window``, legacy packets carry per-column frame IDs,
  encoder counts and status, and narrow channel fields saturate instead of wrapping.
* [BUGFIX]: IMU packets decode correctly: LEGACY packets carry g and deg/s (they carried m/s^2 and
  rad/s, so decoders saw about 96 m/s^2 of gravity at rest), and ACCEL32_GYRO32_NMEA packets
  carry their header, per-measurement timestamps, measurement IDs, status and CRC. The sample
  rate follows the packet profile, and ``/imu`` carries the same samples. An unencodable packet
  no longer stalls the stream.
* ``lidar_hz`` defaults to the metadata frame rate, and a disagreeing value is reported.

**Plugin lifecycle and ROS 2**

* [BUGFIX]: the panels-mode render rig is destroyed on the render thread, which crashed world
  resets; rclcpp no longer installs signal handlers in the Gazebo process; failed publishes are
  logged instead of aborting gz-sim; the pose anchor is looked up on the plugin's own model; and
  shutdown cannot miss its wake-up.
* [BUGFIX]: scans, metadata and IMU recover from a simulation-time rewind instead of stalling, the
  panels cadence follows simulation time, nothing renders while paused, and the IMU uses the
  world's gravity.
* ``sensor_name`` gains a leading ``/`` when it lacks one; an empty name disables the plugin.
* An environment hook adds the plugin to ``GZ_SIM_SYSTEM_PLUGIN_PATH``, and ``package.xml``
  declares every directly used dependency.

**Examples**

* [BUGFIX]: ``os_cloud`` rotated every example cloud 180 degrees and raised it 36 mm, because
  ``point_cloud_frame`` equalled ``sensor_frame``; the examples keep them distinct.
* Add ``ouster_showcase.sdf``, a world that walks every modeled effect and every channel's
  domain in one scan, with a ``world:=`` launch argument, and the TurtleBot warehouse, hills and
  sewer worlds.
* The standalone and sensor_stack launches run ``os_image`` for the RViz image panels
  (``images:=false`` skips it), in each sensor's frame and with matching QoS.
* [BUGFIX]: each Ouster in the xacro macro binds to its own IMU, and the TurtleBot DiffDrive
  publishes odometry at the intended rate.
* ``config/plugin_example.sdf`` documents ``signal_model`` and the measured response.

**Build and CI**

* ``GZ_GPU_OUSTER_USE_CUDA``, ``_HIP`` and ``_SYCL`` are ``AUTO``, ``ON`` or ``OFF`` overrides;
  ``OFF`` skips the toolchain probe, and CUDA builds a portable architecture list unless one is
  set.
* [BUGFIX]: backend headers install next to the public header, so its quoted includes resolve.
* CI runs on ``cam_wip`` and ``claude/**`` pushes, builds the core's real-capture conformance,
  and passes its cpplint, cppcheck, doxygen and GPU smoke-build gates.
* ``ouster-ros`` is pinned to the jcfurey cam-wip commit 57be1ca.
* [BUGFIX]: the Zenoh packet tests prove the data path before they measure it. The backpressure
  test failed every run, and the burst test about 1% of runs.

1.0.0 (2026-06-18)
------------------
* The state of ``main`` at c0f23cd, where this changelog starts; earlier changes are in the git
  history.
