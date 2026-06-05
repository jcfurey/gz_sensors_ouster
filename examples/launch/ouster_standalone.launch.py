# Launch the standalone Ouster OS1-64 example in the demo world.
#
#   ros2 launch gz_sensors_ouster ouster_standalone.launch.py
#   ros2 launch gz_sensors_ouster ouster_standalone.launch.py rviz:=true
#   ros2 launch gz_sensors_ouster ouster_standalone.launch.py anchor_type:=gpu_lidar
#   ros2 launch gz_sensors_ouster ouster_standalone.launch.py lidar_profile:=legacy
#
# Brings up: gz sim (demo world) + robot_state_publisher (URDF) +
# `ros_gz_sim create` (spawns the model, which loads the system plugin) +
# ros_gz_bridge (/clock only) + ouster_ros os_cloud (turns the plugin's
# lidar_packets into a PointCloud2 on /sensor/lidar/lidar0/points, exactly
# as it would for a real Ouster). The LiDAR/IMU/image topics are published
# directly by the plugin, so they are NOT bridged.
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


def generate_launch_description():
    pkg_share = get_package_share_directory('gz_sensors_ouster')
    pkg_lib = os.path.join(get_package_prefix('gz_sensors_ouster'), 'lib')

    world = os.path.join(pkg_share, 'examples', 'worlds', 'ouster_demo.sdf')
    urdf = os.path.join(pkg_share, 'examples', 'urdf', 'ouster_standalone.urdf.xacro')
    bridge_cfg = os.path.join(pkg_share, 'examples', 'config', 'ouster_bridge.yaml')
    rviz_cfg = os.path.join(pkg_share, 'examples', 'rviz', 'ouster.rviz')

    anchor_type = LaunchConfiguration('anchor_type')
    lidar_profile = LaunchConfiguration('lidar_profile')

    # ABSOLUTE metadata path: relative paths resolve against an on-disk SDF dir,
    # which does not exist for a model spawned from the robot_description topic.
    # The OS1-64 metadata ships in two variants; lidar_profile selects which:
    #   modern -> os1_64_rev7.json        (RNG19_RFL8_SIG16_NIR16, FW v3.2.0)
    #   legacy -> os1_64_rev7_legacy.json (LEGACY profile, no WINDOW field)
    # os_cloud needs no matching config — it reads the profile from the
    # plugin's metadata topic.
    meta_name = PythonExpression(
        ["'os1_64_rev7_legacy.json' if '", lidar_profile,
         "' == 'legacy' else 'os1_64_rev7.json'"])
    metadata = PathJoinSubstitution([pkg_share, 'config', 'metadata', meta_name])

    robot_description = ParameterValue(
        Command([
            'xacro ', urdf,
            ' metadata_lidar0:=', metadata,
            ' anchor_type:=', anchor_type,
        ]),
        value_type=str,
    )

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('ros_gz_sim'),
                                  'launch', 'gz_sim.launch.py'])
        ),
        launch_arguments={'gz_args': [world, ' -r -v 3']}.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument('anchor_type', default_value='gpu_lidar',
                              description='Pose-anchor sensor type: gpu_lidar | altimeter. '
                                          'Must be a rendering type (gpu_lidar) unless the '
                                          'world already contains another camera/gpu_lidar, '
                                          'or the plugin never starts rendering.'),
        DeclareLaunchArgument('lidar_profile', default_value='modern',
                              description='Ouster generation the metadata simulates: '
                                          'modern (RNG19_RFL8_SIG16_NIR16, FW v3.2.0) | '
                                          'legacy (LEGACY profile). Use legacy to simulate '
                                          'pre-3.2 firmware; modern is recommended.'),
        DeclareLaunchArgument('rviz', default_value='false',
                              description='Launch RViz with the example config'),

        # Make libgz_sensors_ouster.so discoverable as a gz system plugin.
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
                       '-name', 'ouster_standalone', '-z', '0.05'],
        ),

        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            output='screen',
            parameters=[{'config_file': bridge_cfg, 'use_sim_time': True}],
        ),

        # ouster_ros os_cloud: assembles the plugin's lidar_packets + metadata
        # into a PointCloud2, identically to a real Ouster. Run in the plugin's
        # topic namespace so its relative subscriptions (lidar_packets,
        # metadata) and publication (points) line up with /sensor/lidar/lidar0.
        Node(
            package='ouster_ros',
            executable='os_cloud',
            name='os_cloud',
            namespace='/sensor/lidar/lidar0',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                # Build only the point cloud. The plugin already publishes
                # /sensor/lidar/lidar0/imu directly, so don't let os_cloud
                # republish a second 'imu' from imu_packets.
                'proc_mask': 'PCL',
                # Stamp the cloud in the URDF lidar frame that
                # robot_state_publisher already places in the TF tree, so RViz
                # (fixed frame base_footprint) can display it.
                # Stamp the cloud in the URDF lidar frame, NOT os_lidar: the
                # plugin generates points aligned with lidar0/lidar_frame, while
                # the metadata's lidar_to_sensor_transform is the real Ouster
                # 180deg + 36mm offset — publishing in os_lidar would rotate the
                # sim cloud. (Set this to lidar0/os_lidar if you want the full
                # physical sensor<->lidar offset instead.)
                'point_cloud_frame': 'lidar0/lidar_frame',
                # Let the ouster driver broadcast its own static TF subtree
                # (lidar0/lidar_frame -> lidar0/os_lidar, lidar0/os_imu) from the
                # metadata, anchored at the robot's lidar frame.
                'pub_static_tf': True,
                'sensor_frame': 'lidar0/lidar_frame',
                'lidar_frame': 'lidar0/os_lidar',
                'imu_frame': 'lidar0/os_imu',
                # Stamp on receipt with ROS (sim) time, sidestepping any epoch
                # mismatch between the packet column timestamps and /clock.
                'timestamp_mode': 'TIME_FROM_ROS_TIME',
            }],
        ),

        # Explicit mount transform base_link -> lidar0/lidar_frame, matching the
        # URDF mount joint (xyz 0 0 0.44). robot_state_publisher also publishes
        # this from the URDF; broadcasting it here too guarantees the cloud
        # reaches base_link even if RSP is not running (harmless TF_REPEATED_DATA
        # warning when both are up).
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='lidar0_mount_stp',
            arguments=['--x', '0', '--y', '0', '--z', '0.44',
                       '--yaw', '0', '--pitch', '0', '--roll', '0',
                       '--frame-id', 'base_link',
                       '--child-frame-id', 'lidar0/lidar_frame'],
            parameters=[{'use_sim_time': True}],
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            output='screen',
            arguments=['-d', rviz_cfg],
            parameters=[{'use_sim_time': True}],
            condition=IfCondition(LaunchConfiguration('rviz')),
        ),
    ])
