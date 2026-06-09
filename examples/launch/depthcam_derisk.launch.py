# PHASE-0 DE-RISK launch (not a normal example).
#
#   ros2 launch gz_sensors_ouster depthcam_derisk.launch.py
#
# Brings up a stock gz `depth_camera` (single-pass perspective render, NOT the
# GpuRays cubemap) looking at a cylinder + two boxes, and bridges its depth
# image and point cloud to ROS so you can see whether a plain depth render
# duplicates objects on THIS machine.
#
# What to look for:
#   * depth image  /depthcam        (rqt_image_view, opened automatically)
#   * point cloud  /depthcam/points (RViz, fixed frame = the cloud's frame_id)
#
#   - Each object appears ONCE  -> gz's cubemap is the bug; the DepthCamera
#                                  stitch route-around will work. Tell me and I
#                                  build it.
#   - Objects appear ~4x copies -> gz rendering is broken below the cubemap
#                                  (driver / software-GL / Ogre2); we use the
#                                  per-ray raycast path instead.
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = get_package_share_directory('gz_sensors_ouster')
    world = os.path.join(pkg_share, 'examples', 'worlds', 'depthcam_derisk.sdf')

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('ros_gz_sim'),
                                  'launch', 'gz_sim.launch.py'])
        ),
        launch_arguments={'gz_args': [world, ' -r -v 3']}.items(),
    )

    # Bridge the stock depth camera's outputs (GZ -> ROS).
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        output='screen',
        arguments=[
            '/depthcam@sensor_msgs/msg/Image[gz.msgs.Image',
            '/depthcam/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
        ],
    )

    # Easiest viewer: the raw depth image (no TF needed). Look for 1 vs ~4
    # copies of the cylinder/boxes.
    rqt = Node(
        package='rqt_image_view',
        executable='rqt_image_view',
        output='screen',
        arguments=['/depthcam'],
    )

    return LaunchDescription([gz_sim, bridge, rqt])
