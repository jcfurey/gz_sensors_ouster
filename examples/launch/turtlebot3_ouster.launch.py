# Launch a TurtleBot3 waffle carrying the simulated Ouster.
#
#   ros2 launch gz_sensors_ouster turtlebot3_ouster.launch.py
#   ros2 launch gz_sensors_ouster turtlebot3_ouster.launch.py headless:=true
#   ros2 launch gz_sensors_ouster turtlebot3_ouster.launch.py rviz:=true
#   ros2 launch gz_sensors_ouster turtlebot3_ouster.launch.py ray_mode:=panels
#
# Brings up: gz sim + robot_state_publisher (waffle+Ouster URDF) +
# `ros_gz_sim create` (spawns the model, loading the system plugin) +
# ros_gz_bridge (/clock, /cmd_vel, /odom, /tf, /joint_states) + ouster_ros
# os_cloud (turns the plugin's lidar_packets into a PointCloud2 on
# /sensor/lidar/lidar0/points, exactly as for a real Ouster).
#
# Default ray_mode is *raycast* (CPU, no render engine) so this runs headless
# with no GPU. Drive it with teleop_twist_keyboard on /cmd_vel. The
# LiDAR/IMU/image topics are published directly by the plugin, so they are NOT
# bridged.
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
    # Parent of turtlebot3_description's share dir, so gz can resolve the waffle's
    # `package://turtlebot3_description/meshes/...` visual meshes.
    tb3_share_parent = os.path.dirname(
        get_package_share_directory('turtlebot3_description'))

    urdf = os.path.join(pkg_share, 'examples', 'urdf', 'turtlebot3_ouster.urdf.xacro')
    rviz_cfg = os.path.join(pkg_share, 'examples', 'rviz', 'ouster.rviz')

    ray_mode = LaunchConfiguration('ray_mode')
    lidar_profile = LaunchConfiguration('lidar_profile')
    headless = LaunchConfiguration('headless')

    # raycast runs against the GPU-free world (no rendering Sensors system);
    # panels needs the rendering ouster_demo.sdf world.
    world_name = PythonExpression(
        ["'ouster_demo.sdf' if '", ray_mode,
         "' == 'panels' else 'turtlebot3_ouster_headless.sdf'"])
    world = PathJoinSubstitution([pkg_share, 'examples', 'worlds', world_name])

    # ABSOLUTE metadata path — a model spawned from the robot_description topic
    # has no on-disk SDF dir to resolve a relative path against.
    meta_name = PythonExpression(
        ["'os1_64_rev7_legacy.json' if '", lidar_profile,
         "' == 'legacy' else 'os1_64_rev7.json'"])
    metadata = PathJoinSubstitution([pkg_share, 'config', 'metadata', meta_name])

    # gz server-only (-s) when headless, otherwise also launch the GUI client.
    gz_args = [world, PythonExpression(
        ["' -s -r -v 3' if '", headless, "' == 'true' else ' -r -v 3'"])]

    robot_description = ParameterValue(
        Command([
            'xacro ', urdf,
            ' metadata_lidar0:=', metadata,
            ' ray_mode:=', ray_mode,
        ]),
        value_type=str,
    )

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('ros_gz_sim'),
                                  'launch', 'gz_sim.launch.py'])
        ),
        launch_arguments={'gz_args': gz_args}.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument('ray_mode', default_value='raycast',
                              description='raycast (CPU, no GPU; default) | panels '
                                          '(GpuRays, needs the rendering world).'),
        DeclareLaunchArgument('lidar_profile', default_value='modern',
                              description='Ouster generation the metadata simulates: '
                                          'modern (RNG19_RFL8_SIG16_NIR16) | legacy.'),
        DeclareLaunchArgument('headless', default_value='false',
                              description='Run gz server-only (no GUI client).'),
        DeclareLaunchArgument('rviz', default_value='false',
                              description='Launch RViz with the example config.'),

        # libgz_sensors_ouster.so discoverable as a gz system plugin.
        AppendEnvironmentVariable('GZ_SIM_SYSTEM_PLUGIN_PATH', pkg_lib),
        # turtlebot3_description meshes resolvable via package:// in gz.
        AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', tb3_share_parent),

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
                       '-name', 'turtlebot3_ouster', '-z', '0.05'],
        ),

        # /clock + the diff-drive control/feedback topics (/cmd_vel, /odom, /tf,
        # /joint_states). The Ouster topics are published directly by the plugin
        # and are NOT bridged. (/clock is also covered by ouster_bridge.yaml; we
        # bridge it explicitly here so this launch is self-contained.)
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            output='screen',
            arguments=[
                '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
                '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
                '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
                '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
                '/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
            ],
            parameters=[{'use_sim_time': True}],
        ),

        # ouster_ros os_cloud: assembles the plugin's lidar_packets + metadata
        # into a PointCloud2, identically to a real Ouster. Runs in the plugin's
        # topic namespace so its relative subscriptions (lidar_packets, metadata)
        # and publication (points) line up with /sensor/lidar/lidar0.
        Node(
            package='ouster_ros',
            executable='os_cloud',
            name='os_cloud',
            namespace='/sensor/lidar/lidar0',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                # Build only the point cloud — the plugin publishes /imu itself.
                'proc_mask': 'PCL',
                # Stamp the cloud in the URDF lidar frame that
                # robot_state_publisher places in the TF tree.
                'point_cloud_frame': 'lidar0/lidar_frame',
                'pub_static_tf': True,
                'sensor_frame': 'lidar0/lidar_frame',
                'lidar_frame': 'lidar0/os_lidar',
                'imu_frame': 'lidar0/os_imu',
                'timestamp_mode': 'TIME_FROM_ROS_TIME',
            }],
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
