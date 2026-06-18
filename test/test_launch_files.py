"""Structural invariants for the example launch files.

Tests run without any simulation — pure Python AST parsing and text checks.
Catches regressions like removing the headless arg, reverting the ray_mode
default, or re-introducing a standalone anchor_type arg (now derived in the URDF).
"""
import ast
import pathlib

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
    assert "' -s -r -v 3'" in (LAUNCHES / name).read_text()


@pytest.mark.parametrize('name', EXAMPLE_LAUNCHES)
def test_ray_mode_default_is_raycast(name):
    src = (LAUNCHES / name).read_text()
    idx = src.index("DeclareLaunchArgument('ray_mode'")
    assert "default_value='raycast'" in src[idx:idx + 300]


@pytest.mark.parametrize('name', EXAMPLE_LAUNCHES)
def test_panels_world_referenced(name):
    assert 'ouster_demo_panels.sdf' in (LAUNCHES / name).read_text()


@pytest.mark.parametrize('name', ['ouster_standalone.launch.py', 'sensor_stack.launch.py'])
def test_no_anchor_type_launch_arg(name):
    """anchor_type is now derived from ray_mode inside the URDF; no launch arg."""
    assert "DeclareLaunchArgument('anchor_type'" not in (LAUNCHES / name).read_text()
