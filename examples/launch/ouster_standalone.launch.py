# Launch the standalone Ouster OS1-64 example in the demo world.
#
#   ros2 launch gz_sensors_ouster ouster_standalone.launch.py
#   ros2 launch gz_sensors_ouster ouster_standalone.launch.py rviz:=true
#   ros2 launch gz_sensors_ouster ouster_standalone.launch.py anchor_type:=gpu_lidar
#
# Brings up: gz sim (demo world) + robot_state_publisher (URDF) +
# `ros_gz_sim create` (spawns the model, which loads the system plugin) +
# ros_gz_bridge (/clock only). The LiDAR/IMU/image topics are published
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
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
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

    # ABSOLUTE metadata path: relative paths resolve against an on-disk SDF dir,
    # which does not exist for a model spawned from the robot_description topic.
    metadata = os.path.join(pkg_share, 'config', 'metadata', 'os1_64_rev7.json')

    anchor_type = LaunchConfiguration('anchor_type')

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

        Node(
            package='rviz2',
            executable='rviz2',
            output='screen',
            arguments=['-d', rviz_cfg],
            parameters=[{'use_sim_time': True}],
            condition=IfCondition(LaunchConfiguration('rviz')),
        ),
    ])
