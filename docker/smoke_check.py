#!/usr/bin/env python3
"""Headless smoke check for the gz_sensors_ouster plugin.

Subscribes to the Ouster point cloud the plugin + os_cloud produce and exits 0
as soon as one non-empty PointCloud2 arrives, or 1 if none arrives before the
timeout. Uses a BEST_EFFORT (sensor data) subscription so it receives from
either a best-effort or a reliable publisher.
"""
import argparse
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2


class SmokeCheck(Node):
    def __init__(self, topic):
        super().__init__('gz_sensors_ouster_smoke_check')
        self.got = False
        self.points = 0
        self.create_subscription(
            PointCloud2, topic, self._on_cloud, qos_profile_sensor_data)

    def _on_cloud(self, msg):
        self.points = msg.width * msg.height
        self.got = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--topic', default='/sensor/lidar/lidar0/points')
    ap.add_argument('--timeout', type=float, default=120.0)
    args = ap.parse_args()

    rclpy.init()
    node = SmokeCheck(args.topic)
    node.get_logger().info(
        f'waiting up to {args.timeout:.0f}s for a cloud on {args.topic} ...')

    deadline = node.get_clock().now().nanoseconds + int(args.timeout * 1e9)
    while rclpy.ok() and not node.got:
        rclpy.spin_once(node, timeout_sec=0.5)
        if node.get_clock().now().nanoseconds > deadline:
            break

    ok = node.got and node.points > 0
    if ok:
        node.get_logger().info(
            f'SMOKE PASS: received PointCloud2 with {node.points} points '
            f'on {args.topic}')
    else:
        node.get_logger().error(
            f'SMOKE FAIL: no non-empty PointCloud2 on {args.topic} within '
            f'{args.timeout:.0f}s')

    node.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
