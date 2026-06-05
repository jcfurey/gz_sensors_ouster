# Launch the multi-sensor "sensor stack" example (front OS1-64 + IMU, rear OS0-128)
# in the demo world.
#
#   ros2 launch gz_sensors_ouster sensor_stack.launch.py
#   ros2 launch gz_sensors_ouster sensor_stack.launch.py rviz:=true
#
# Two GzGpuOusterLidarSystem instances run on one spawned model, publishing under
# /sensor/lidar/front/... and /sensor/lidar/rear/... . Only /clock is bridged.
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
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = get_package_share_directory('gz_sensors_ouster')
    pkg_lib = os.path.join(get_package_prefix('gz_sensors_ouster'), 'lib')

    world = os.path.join(pkg_share, 'examples', 'worlds', 'ouster_demo.sdf')
    urdf = os.path.join(pkg_share, 'examples', 'urdf', 'sensor_stack.urdf.xacro')
    bridge_cfg = os.path.join(pkg_share, 'examples', 'config', 'ouster_bridge.yaml')
    rviz_cfg = os.path.join(pkg_share, 'examples', 'rviz', 'sensor_stack.rviz')

    # ABSOLUTE metadata paths (see ouster_standalone.launch.py for rationale).
    metadata_dir = os.path.join(pkg_share, 'config', 'metadata')
    metadata_front = os.path.join(metadata_dir, 'os1_64_rev7.json')
    metadata_rear = os.path.join(metadata_dir, 'os0_128_rev7.json')

    anchor_type = LaunchConfiguration('anchor_type')

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
        DeclareLaunchArgument('anchor_type', default_value='gpu_lidar',
                              description='Pose-anchor sensor type: gpu_lidar | altimeter. '
                                          'Must be a rendering type (gpu_lidar) unless the '
                                          'world already contains another camera/gpu_lidar, '
                                          'or the plugin never starts rendering.'),
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

        Node(
            package='rviz2',
            executable='rviz2',
            output='screen',
            arguments=['-d', rviz_cfg],
            parameters=[{'use_sim_time': True}],
            condition=IfCondition(LaunchConfiguration('rviz')),
        ),
    ])
