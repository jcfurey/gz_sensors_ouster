# Launch the multi-sensor "sensor stack" example (front OS1-64 + IMU, rear OS0-128)
# in the demo world.
#
#   ros2 launch gz_sensors_ouster sensor_stack.launch.py
#   ros2 launch gz_sensors_ouster sensor_stack.launch.py rviz:=true
#   ros2 launch gz_sensors_ouster sensor_stack.launch.py lidar_profile:=legacy
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


def _os_cloud(name):
    """ouster_ros os_cloud for one sensor: assemble lidar_packets into a
    PointCloud2 on /sensor/lidar/<name>/points, stamped in <name>/lidar_frame.
    pub_static_tf=true so the driver also broadcasts its own static TF subtree
    (<name>/lidar_frame -> <name>/os_lidar, <name>/os_imu) anchored at the robot
    lidar frame. The cloud stays in <name>/lidar_frame (NOT os_lidar) because the
    plugin generates points there and the metadata lidar_to_sensor_transform is
    the real 180deg+36mm offset, which would rotate the sim cloud."""
    return Node(
        package='ouster_ros',
        executable='os_cloud',
        name='os_cloud',
        namespace='/sensor/lidar/' + name,
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'proc_mask': 'PCL',  # plugin publishes /imu itself; cloud only
            'point_cloud_frame': name + '/lidar_frame',
            'pub_static_tf': True,
            'sensor_frame': name + '/lidar_frame',
            'lidar_frame': name + '/os_lidar',
            'imu_frame': name + '/os_imu',
            'timestamp_mode': 'TIME_FROM_ROS_TIME',
        }],
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

    world = os.path.join(pkg_share, 'examples', 'worlds', 'ouster_demo.sdf')
    urdf = os.path.join(pkg_share, 'examples', 'urdf', 'sensor_stack.urdf.xacro')
    bridge_cfg = os.path.join(pkg_share, 'examples', 'config', 'ouster_bridge.yaml')
    rviz_cfg = os.path.join(pkg_share, 'examples', 'rviz', 'sensor_stack.rviz')

    anchor_type = LaunchConfiguration('anchor_type')
    lidar_profile = LaunchConfiguration('lidar_profile')

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
        DeclareLaunchArgument('anchor_type', default_value='camera',
                              description='Render-bootstrap / pose-anchor sensor type: '
                                          'camera | gpu_lidar | altimeter. Default camera is '
                                          'the cheapest renderer and adds no second lidar. '
                                          'gpu_lidar also emits a native gz scan '
                                          '(<ns>/gz_native_scan, a 2nd lidar source). '
                                          'altimeter is non-rendering — only if the world '
                                          'already has another camera/gpu_lidar.'),
        DeclareLaunchArgument('lidar_profile', default_value='modern',
                              description='Ouster generation the metadata simulates: '
                                          'modern (RNG19_RFL8_SIG16_NIR16, FW v3.2.0) | '
                                          'legacy (LEGACY profile). Applies to both sensors.'),
        DeclareLaunchArgument('rviz', default_value='false',
                              description='Launch RViz with the example config'),

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
        # broadcasting its own os_lidar/os_imu TF subtree.
        _os_cloud('front'),
        _os_cloud('rear'),

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
