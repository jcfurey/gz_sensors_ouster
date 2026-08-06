# Launch the multi-sensor "sensor stack" example (front OS1-64 + IMU, rear OS0-128)
# in the demo world.
#
#   ros2 launch gz_sensors_ouster sensor_stack.launch.py
#   ros2 launch gz_sensors_ouster sensor_stack.launch.py rviz:=true
#   ros2 launch gz_sensors_ouster sensor_stack.launch.py lidar_profile:=legacy
#   ros2 launch gz_sensors_ouster sensor_stack.launch.py ray_mode:=panels
#   ros2 launch gz_sensors_ouster sensor_stack.launch.py headless:=true
#
# Default ray_mode is *raycast* (GPU-free). Pass ray_mode:=panels for the GpuRays
# path; the launch switches to ouster_demo_panels.sdf automatically and derives the
# anchor type from ray_mode. Panels needs a render-capable host (ogre2/GPU).
# Pass headless:=true to run gz server-only (no GUI client) — required on WSL,
# SSH sessions, and any headless host without a display.
#
# Two GzGpuOusterLidarSystem instances run on one spawned model, publishing under
# /sensor/lidar/front/... and /sensor/lidar/rear/... . Only /clock is bridged;
# an ouster_ros os_cloud per sensor turns lidar_packets into PointCloud2s on
# /sensor/lidar/{front,rear}/points.
import os

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable,
    DeclareLaunchArgument,
    IncludeLaunchDescription,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _metadata_path(pkg_share, lidar_profile, modern_name):
    """Absolute metadata path, swapping in the *_legacy.json variant when
    lidar_profile == 'legacy'. modern_name is e.g. 'os1_64_rev7.json'."""
    legacy_name = modern_name[:-len('.json')] + '_legacy.json'
    name = PythonExpression(
        ["'", legacy_name, "' if '", lidar_profile, "' == 'legacy' else '",
         modern_name, "'"])
    return PathJoinSubstitution([pkg_share, 'config', 'metadata', name])


def _os_cloud(name, metadata):
    """ouster_ros os_cloud for one sensor: assemble lidar_packets into a
    PointCloud2 on /sensor/lidar/<name>/points, stamped in <name>/lidar_frame.

    The plugin already generates points in <name>/lidar_frame using the standard
    (identity) Ouster XYZ LUT, so os_cloud must NOT apply the metadata's
    lidar_to_sensor_transform (the real 180deg + 36mm housing offset) — doing so
    would rotate/shift the whole cloud. In ouster_ros that transform is applied
    iff point_cloud_frame == sensor_frame
    (os_transforms_broadcaster.h: apply_lidar_to_sensor_transform()), so we keep
    them DIFFERENT: point_cloud_frame == lidar_frame == '<name>/lidar_frame'
    (identity LUT, cloud in the URDF lidar frame) while sensor_frame is a distinct
    '<name>/os_sensor'. point_cloud_frame is set equal to lidar_frame so the
    driver's frame validation keeps it as-is (it would otherwise reset an
    unrecognised point_cloud_frame back to lidar_frame with a warning).

    pub_static_tf=False: with sensor_frame != lidar_frame the driver would
    broadcast sensor_frame -> lidar_frame, giving <name>/lidar_frame a second
    parent on top of the base_link -> <name>/lidar_frame mount (URDF / the
    _mount_stp below) — a TF-tree conflict. The os_lidar/os_imu leaf frames it
    used to publish are unused by these examples, so RSP + the mount publisher
    own the tree instead."""
    return Node(
        package='ouster_ros',
        executable='os_cloud',
        name='os_cloud',
        namespace='/sensor/lidar/' + name,
        output='screen',
        parameters=[{
            'use_sim_time': True,
            # Initialize the decoder and packet subscription before a fast bag
            # can publish its first packet batch. The metadata topic remains
            # subscribed for live updates.
            'metadata': metadata,
            'proc_mask': 'PCL',  # plugin publishes /imu itself; cloud only
            'point_cloud_frame': name + '/lidar_frame',
            'sensor_frame': name + '/os_sensor',   # != point_cloud_frame → identity LUT
            'lidar_frame': name + '/lidar_frame',  # == point_cloud_frame → no frame reset
            'imu_frame': name + '/os_imu',
            'pub_static_tf': False,
            # The plugin writes sim acquisition time into every column.
            # Preserve it across executor jitter and bag rate scaling.
            'timestamp_mode': 'TIME_FROM_INTERNAL_OSC',
        }],
    )


def _os_image(name, metadata):
    """ouster_ros os_image for one sensor: decodes lidar_packets + metadata into
    the range/signal/reflec/nearir images + camera_info under
    /sensor/lidar/<name>/. The single image source in sim (the plugin's native
    image pubs are off by default), matching the RViz Image displays. Gated
    behind images:=true (default): os_image decodes every scan whether or not
    anything subscribes, so headless/CI runs can opt out with images:=false.
    sensor_frame stamps the images/camera_info in the URDF lidar frame — the
    os_image default 'os_lidar' is broadcast by nothing here (pub_static_tf is
    false) and would also make front/rear indistinguishable by frame_id."""
    return Node(
        package='ouster_ros',
        executable='os_image',
        name='os_image',
        namespace='/sensor/lidar/' + name,
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'metadata': metadata,
            'timestamp_mode': 'TIME_FROM_INTERNAL_OSC',
            'sensor_frame': name + '/lidar_frame',
        }],
        condition=IfCondition(LaunchConfiguration('images')),
    )


def _mount_stp(name, x, y, z, yaw=0.0):
    """Explicit base_link -> <name>/lidar_frame mount transform, matching the
    URDF mount joint. robot_state_publisher also publishes this; broadcasting it
    here too guarantees the cloud reaches base_link even without RSP (harmless
    TF_REPEATED_DATA warning when both run)."""
    return Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name=name + '_mount_stp',
        arguments=['--x', str(x), '--y', str(y), '--z', str(z),
                   '--yaw', str(yaw), '--pitch', '0', '--roll', '0',
                   '--frame-id', 'base_link',
                   '--child-frame-id', name + '/lidar_frame'],
        parameters=[{'use_sim_time': True}],
    )


def generate_launch_description():
    pkg_share = get_package_share_directory('gz_sensors_ouster')
    pkg_lib = os.path.join(get_package_prefix('gz_sensors_ouster'), 'lib')

    urdf = os.path.join(pkg_share, 'examples', 'urdf', 'sensor_stack.urdf.xacro')
    bridge_cfg = os.path.join(pkg_share, 'examples', 'config', 'ouster_bridge.yaml')
    rviz_cfg = os.path.join(pkg_share, 'examples', 'rviz', 'sensor_stack.rviz')

    lidar_profile = LaunchConfiguration('lidar_profile')
    hardware_revision_front = LaunchConfiguration('hardware_revision_front')
    hardware_revision_rear = LaunchConfiguration('hardware_revision_rear')
    ray_mode = LaunchConfiguration('ray_mode')
    headless = LaunchConfiguration('headless')

    world_name = PythonExpression(
        ["'ouster_demo_panels.sdf' if '", ray_mode,
         "' == 'panels' else 'ouster_demo.sdf'"])
    world = PathJoinSubstitution([pkg_share, 'examples', 'worlds', world_name])

    # ABSOLUTE metadata paths (see ouster_standalone.launch.py for rationale).
    # lidar_profile selects the modern (RNG19, FW v3.2.0) or legacy (LEGACY
    # profile) variant of each sensor's metadata.
    metadata_front = _metadata_path(pkg_share, lidar_profile, 'os1_64_rev7.json')
    metadata_rear = _metadata_path(pkg_share, lidar_profile, 'os0_128_rev7.json')

    robot_description = ParameterValue(
        Command([
            'xacro ', urdf,
            ' metadata_front:=', metadata_front,
            ' metadata_rear:=', metadata_rear,
            ' hardware_revision_front:=', hardware_revision_front,
            ' hardware_revision_rear:=', hardware_revision_rear,
            ' ray_mode:=', ray_mode,
        ]),
        value_type=str,
    )

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('ros_gz_sim'),
                                  'launch', 'gz_sim.launch.py'])
        ),
        launch_arguments={'gz_args': [world, PythonExpression(
            ["' -s -r -v 3' if '", headless, "' == 'true' else ' -r -v 3'"])]}.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument('ray_mode', default_value='raycast',
                              description='Ray generation mode: raycast (default, GPU-free) '
                                          'or panels (GpuRays, needs ogre2/GPU). The launch '
                                          'auto-selects the world and anchor type from this.'),
        DeclareLaunchArgument('lidar_profile', default_value='modern',
                              description='Ouster generation the metadata simulates: '
                                          'modern (RNG19_RFL8_SIG16_NIR16, FW v3.2.0) | '
                                          'legacy (LEGACY profile). Applies to both sensors.'),
        DeclareLaunchArgument('hardware_revision_front', default_value='rev07',
                              description='Front Ouster hardware physics revision.'),
        DeclareLaunchArgument('hardware_revision_rear', default_value='rev07',
                              description='Rear Ouster hardware physics revision.'),
        DeclareLaunchArgument('headless', default_value='false',
                              description='Run gz server-only (no GUI client). '
                                          'Use on WSL, SSH, and any headless host.'),
        DeclareLaunchArgument('rviz', default_value='false',
                              description='Launch RViz with the example config'),
        DeclareLaunchArgument('images', default_value='true',
                              description='Run the per-sensor ouster_ros os_image nodes '
                                          '(the sim image source). Set false on headless/'
                                          'CI runs that do not consume the image topics.'),

        AppendEnvironmentVariable('GZ_SIM_SYSTEM_PLUGIN_PATH', pkg_lib),

        gz_sim,

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
        ),

        Node(
            package='ros_gz_sim',
            executable='create',
            output='screen',
            arguments=['-topic', 'robot_description',
                       '-name', 'sensor_stack', '-z', '0.05'],
        ),

        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            output='screen',
            parameters=[{'config_file': bridge_cfg, 'use_sim_time': True}],
        ),

        # One os_cloud per sensor → /sensor/lidar/{front,rear}/points, each
        # stamped in <name>/lidar_frame with the identity LUT (no housing
        # rotation); see _os_cloud for the frame rationale.
        _os_cloud('front', metadata_front),
        _os_cloud('rear', metadata_rear),

        # os_image per sensor → the range/signal/reflec/nearir images the RViz
        # config displays (single image source in sim; plugin native pubs off).
        _os_image('front', metadata_front),
        _os_image('rear', metadata_rear),

        # Explicit mount transforms base_link -> {front,rear}/lidar_frame,
        # matching the URDF (front xyz 0.45 0 0.35; rear xyz -0.45 0 0.35 yaw π).
        _mount_stp('front', 0.45, 0.0, 0.35),
        _mount_stp('rear', -0.45, 0.0, 0.35, yaw=3.14159265359),

        Node(
            package='rviz2',
            executable='rviz2',
            output='screen',
            arguments=['-d', rviz_cfg],
            parameters=[{'use_sim_time': True}],
            condition=IfCondition(LaunchConfiguration('rviz')),
        ),
    ])
