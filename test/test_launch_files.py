"""Structural invariants for the example launch files.

Tests run without any simulation — pure Python AST parsing and text checks.
Catches regressions like removing the headless arg, reverting the ray_mode
default, or re-introducing a standalone anchor_type arg (now derived in the URDF).
"""
import ast
import pathlib
import re

import pytest

LAUNCHES = pathlib.Path(__file__).resolve().parent.parent / 'examples' / 'launch'
LAUNCH_NAMES = sorted(p.name for p in LAUNCHES.glob('*.launch.py'))

# The three example launches that must carry the headless/world-switch machinery.
EXAMPLE_LAUNCHES = [
    'ouster_standalone.launch.py',
    'sensor_stack.launch.py',
    'turtlebot3_ouster.launch.py',
]


@pytest.mark.parametrize('name', LAUNCH_NAMES)
def test_launch_parses(name):
    ast.parse((LAUNCHES / name).read_text())


@pytest.mark.parametrize('name', EXAMPLE_LAUNCHES)
def test_has_headless_arg(name):
    assert "DeclareLaunchArgument('headless'" in (LAUNCHES / name).read_text()


@pytest.mark.parametrize('name', EXAMPLE_LAUNCHES)
def test_headless_default_is_false(name):
    src = (LAUNCHES / name).read_text()
    # headless arg block must contain default_value='false'
    idx = src.index("DeclareLaunchArgument('headless'")
    assert "default_value='false'" in src[idx:idx + 300]


@pytest.mark.parametrize('name', EXAMPLE_LAUNCHES)
def test_headless_enables_server_only_flag(name):
    src = (LAUNCHES / name).read_text()
    # Match the gz server-only flag tolerant of spacing: '...' -s ...'
    m = re.search(r"'\s*-s\b", src)
    assert m, f'{name}: -s (server-only) gz flag not found'
    context = src[max(0, m.start() - 200):m.start() + 200]
    assert 'headless' in context, \
        f'{name}: -s flag exists but is not wired to the headless condition nearby'


@pytest.mark.parametrize('name', EXAMPLE_LAUNCHES)
def test_ray_mode_default_is_raycast(name):
    src = (LAUNCHES / name).read_text()
    idx = src.index("DeclareLaunchArgument('ray_mode'")
    assert "default_value='raycast'" in src[idx:idx + 300]


@pytest.mark.parametrize('name', EXAMPLE_LAUNCHES)
def test_panels_world_referenced(name):
    assert 'ouster_demo_panels.sdf' in (LAUNCHES / name).read_text()


@pytest.mark.parametrize('name', EXAMPLE_LAUNCHES)
def test_no_anchor_type_launch_arg(name):
    """anchor_type is now derived from ray_mode inside the URDF; no launch arg."""
    assert "DeclareLaunchArgument('anchor_type'" not in (LAUNCHES / name).read_text()


def test_turtlebot_launch_exposes_all_raycast_world_profiles():
    src = (LAUNCHES / 'turtlebot3_ouster.launch.py').read_text()
    assert "DeclareLaunchArgument('world'" in src
    for profile, filename in {
            'arena': 'turtlebot3_ouster_headless.sdf',
            'warehouse': 'turtlebot3_ouster_warehouse.sdf',
            'hills': 'turtlebot3_ouster_hills.sdf',
            'sewer': 'turtlebot3_ouster_sewer.sdf',
    }.items():
        assert profile in src
        assert filename in src


@pytest.mark.parametrize('name', EXAMPLE_LAUNCHES)
def test_ouster_consumers_preserve_packet_acquisition_time(name):
    """Cloud/image stamps must be invariant to executor and playback rate."""
    src = (LAUNCHES / name).read_text()
    assert 'TIME_FROM_ROS_TIME' not in src
    assert 'TIME_FROM_INTERNAL_OSC' in src


@pytest.mark.parametrize('name', EXAMPLE_LAUNCHES)
def test_ouster_consumers_follow_sim_clock_for_live_and_bag_playback(name):
    """Playback rate changes delivery, never the recorded acquisition stamps."""
    src = (LAUNCHES / name).read_text()
    assert "'use_sim_time': True" in src
    if '/clock@rosgraph_msgs/msg/Clock' not in src:
        assert 'ouster_bridge.yaml' in src
        bridge = (LAUNCHES.parent / 'config' / 'ouster_bridge.yaml').read_text()
        assert 'ros_topic_name: "/clock"' in bridge
        assert 'gz_topic_name: "/clock"' in bridge


@pytest.mark.parametrize('name', EXAMPLE_LAUNCHES)
def test_ouster_consumers_preseed_metadata_before_fast_bag_playback(name):
    """Avoid losing early packets while a one-shot metadata callback starts."""
    src = (LAUNCHES / name).read_text()
    assert src.count("'metadata': metadata") >= 2
