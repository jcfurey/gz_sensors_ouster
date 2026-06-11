# Systems Architecture — Literature References

Companion to [MODEL_REFERENCES.md](MODEL_REFERENCES.md) (which covers the
sensor physics): this maps the plugin's **ROS 2 middleware/threading** and
**GPU pipeline** choices to the systems literature — what was checked, why
each choice stands, and which levers are deliberately not pulled.

## ROS 2 publishing architecture

The plugin runs an rclcpp node inside the Gazebo server process
(`src/ros_interface.cpp`): publishers for the packet stream, four image
topics, CameraInfo, IMU and latched metadata; a `SingleThreadedExecutor` on a
background thread services parameter callbacks and middleware housekeeping.

### QoS — audited and confirmed

| Topic | QoS | Grounding |
|---|---|---|
| `lidar_packets` | `SensorDataQoS` (BEST_EFFORT), fixed | Maruyama et al. (EMSOFT 2016) measured BEST_EFFORT outperforming RELIABLE at high rates and larger payloads; a dropped raw packet is recoverable downstream and must never block the sim thread. |
| images / CameraInfo | RELIABLE `KEEP_LAST(5)`, configurable | Matches the default subscriber QoS of RViz/image_transport; mismatched pub/sub QoS is the canonical silent-failure mode, and `rmw_zenoh_cpp` requires an exact match in both directions (documented at the `qos_from_string` helper). |
| `metadata` | RELIABLE `TRANSIENT_LOCAL(1)` | Latched config topic — the standard late-joiner pattern. The sim-time-throttled republish loop exists because transient-local replay across processes is unreliable on `rmw_zenoh_cpp` (observed; see code comment). |
| IMU | `sensor_data`, configurable | Mirrors the ouster_ros driver convention so sim↔hardware topic swaps don't require subscriber changes. |

References: Maruyama, Kato, Azumi — *Exploring the performance of ROS2*,
EMSOFT 2016. Kronauer et al. — *Latency Analysis of ROS2 Multi-Node
Systems*, arXiv:2101.02074 (end-to-end latency is dominated by the
rclcpp/rmw layers for small messages and by the DDS transport for large
payloads — which is why the high-rate packet topic stays small and
BEST_EFFORT while the bulky image topics are subscription-gated).

### RMW choice (Zenoh / CycloneDDS / FastDDS) — left to the deployment

The plugin is rmw-agnostic but carries Zenoh-specific accommodations (QoS
matching, discovery-aware metadata republish, executor pumping before first
publish). The comparative literature says the right RMW depends on the
network, so no default is imposed:

- *Performance Comparison of ROS2 Middlewares for Multi-robot Mesh
  Networks*, J. Intell. Robot. Syst. 2024 (arXiv:2407.03091) — Zenoh:
  lower delay/CPU and better reachability on dynamic mesh/Wi-Fi topologies.
- *Comparison of Middlewares in Edge-to-Edge and Edge-to-Cloud
  Communication*, arXiv:2309.07496 — CycloneDDS strongest on wired
  Ethernet (UDP multicast); Zenoh superior over Wi-Fi/4G (discovery
  traffic dominates DDS there).
- *Automotive Middleware Performance: FastDDS, Zenoh, vSomeIP*,
  arXiv:2505.02734.

Rule of thumb for this plugin's users: single machine or wired LAN → any
DDS works (CycloneDDS measured best); robot fleets / Wi-Fi / lossy links →
`rmw_zenoh_cpp`, which the code already accommodates.

### Executor & threading — audited and confirmed

- The executor services only parameter callbacks and middleware
  housekeeping — no data flows through executor callbacks (publishes happen
  directly from the sim thread / cast worker). The callback-scheduling
  anomalies analysed for single-threaded executors (non-preemptive,
  polling-point semantics, chain starvation — Casini et al., ECRTS 2019;
  multi-threaded extensions arXiv:2408.08440) therefore have no processing
  chain to act on here. A `MultiThreadedExecutor` would add context-switch
  cost for nothing (see also the real-time support survey,
  arXiv:2601.10722).
- `use_intra_process_comms(false)` is correct, not an omission: intra-process
  delivery only benefits subscribers in the same process (*Impact of ROS 2
  Node Composition in Robotic Systems*, arXiv:2305.09933); every consumer
  (os_cloud, RViz, Foxglove) is out-of-process.
- The single `publish_mtx_` serialising all publishers is a deliberate
  conservative choice for rmw thread-safety across vendors; the held-lock
  work is bounded (a move into the rmw layer). Splitting it per-publisher is
  the first thing to try if profiling ever shows packet publishes waiting on
  image copies.

### Zero-copy — deliberately not used (and why)

Loaned-message zero-copy (`design.ros2.org/articles/zero_copy.html`, via
iceoryx/CycloneDDS or FastDDS data-sharing) requires **fixed-size (bounded)
message types**. Every bulk message this plugin publishes —
`sensor_msgs/Image`, `ouster_sensor_msgs/PacketMsg` (`uint8[] buf`) — is
unsized, so the loan degenerates to a serialize+copy through shared memory
(iceoryx) or is rejected (CycloneDDS SHM). The research frontier for exactly
this gap is **Agnocast** (*Supporting Unsized Message Types for True
Zero-Copy Publish/Subscribe IPC*, arXiv:2506.16882) — worth revisiting if it
lands in a ROS distro. Until then the cheap mitigations are already in
place: every image topic is subscription-gated (zero cost with no
subscriber), and messages are `std::move`d into the rmw layer.

## GPU pipeline (CUDA / HIP / SYCL backends)

### Audited and confirmed

- **Single dedicated stream per backend, `cudaMemcpyAsync` + one
  `cudaStreamSynchronize` per stage** — the canonical small-pipeline
  pattern (NVIDIA, *How to Optimize Data Transfers in CUDA C/C++*).
- **Persistent device buffers and curand/hiprand states** (allocated once,
  re-seeded never; states stride per ray) — avoids the well-known
  `curand_init` cost per frame.
- **Version-cached scene upload** (`castScan` re-uploads geometry only when
  the scene version changes; per-scan transforms are the only steady-state
  upload) — transfer minimisation as recommended by the same literature.
- **Cached beam-table upload**: the per-beam calibration arrays are
  constant for a sensor's lifetime, so all three GPU backends upload them
  once (keyed by host pointer + count) instead of every frame.
- **Kernel launch count is 1–2 per frame**, so launch-overhead remedies
  (CUDA Graphs, persistent kernels) have nothing to amortise; AstroAccelerate
  (arXiv:2101.00941) shows stream/graph restructuring pays off at tens of
  launches per cycle, an order of magnitude above this pipeline.

### Pinned host memory — the documented next lever, not pulled

The staging buffers (`std::vector` in the cast worker and panel path) are
pageable, so the async copies are internally staged and effectively
synchronous; page-locked memory roughly doubles PCIe throughput (~3–4 →
6–12 GB/s, NVIDIA data-transfer guide; ~30% end-to-end in AstroAccelerate's
streaming workload). **At this plugin's scale it does not matter yet**: an
OS1-64 frame moves ~0.5 MB (raycast) to ~4 MB (panels) at 10 Hz — tens of
MB/s against ~16 GB/s of PCIe Gen3 — so pinning would recover well under a
millisecond per frame while adding registration-lifetime complexity across
three GPU backends (a reallocated `std::vector` silently invalidates a
`cudaHostRegister` registration). Revisit when profiles show transfer time,
or when OS2-128-at-20-Hz-class configs (~10× the volume) become the norm:
the clean shape is a `Backend::registerHostBuffer()` hook called once per
stable staging buffer.

## Summary

The systems audit found no literature-contradicted choice. The
prioritised follow-ups, in order of expected value: per-publisher mutexes
(if profiling shows contention), pinned staging buffers (if transfer time
shows up in profiles), Agnocast-style zero-copy IPC (when available in a
distro).
