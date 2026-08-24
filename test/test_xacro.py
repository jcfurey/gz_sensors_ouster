"""URDF xacro anchor-type derivation tests.

Verifies that ray_mode:=raycast → altimeter anchor and ray_mode:=panels → camera
anchor for all three example URDFs that carry the anchor_type property.

Skipped automatically when xacro is not on PATH (e.g. running pytest outside
a sourced ROS 2 workspace).  In colcon test the package is installed and xacro
is always available.
"""
import pathlib
import shutil
import subprocess
import xml.dom.minidom

import pytest

pytestmark = pytest.mark.skipif(
    shutil.which('xacro') is None,
    reason='xacro not on PATH — source a ROS 2 workspace to enable these tests',
)

REPO = pathlib.Path(__file__).resolve().parent.parent
URDF = REPO / 'examples' / 'urdf'
META = REPO / 'config' / 'metadata'

META0 = str(META / 'os1_64_rev7.json')    # lidar0 / front
META_R = str(META / 'os0_128_rev7.json')  # rear (OS0-128)


def _expand(name: str, **kwargs) -> xml.dom.minidom.Document:
    cmd = ['xacro', str(URDF / name)]
    cmd.extend(f'{k}:={v}' for k, v in kwargs.items())
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        pytest.fail(f'xacro {name} failed:\n{r.stderr}')
    return xml.dom.minidom.parseString(r.stdout)


def _anchor_types(doc: xml.dom.minidom.Document) -> dict:
    """Return {sensor_name: sensor_type} for every <sensor> element."""
    return {s.getAttribute('name'): s.getAttribute('type')
            for s in doc.getElementsByTagName('sensor')}


def _obscurants(doc: xml.dom.minidom.Document) -> list:
    return list(doc.getElementsByTagName('obscurant'))


def _text_values(doc: xml.dom.minidom.Document, tag: str) -> list[str]:
    return [node.firstChild.nodeValue.strip()
            for node in doc.getElementsByTagName(tag)
            if node.firstChild is not None]


# ── ouster_standalone ─────────────────────────────────────────────────────────

def test_standalone_raycast_anchor_is_altimeter():
    t = _anchor_types(_expand('ouster_standalone.urdf.xacro',
                               ray_mode='raycast', metadata_lidar0=META0))
    assert t.get('lidar0') == 'altimeter', f'anchor types: {t}'


def test_standalone_panels_anchor_is_camera():
    t = _anchor_types(_expand('ouster_standalone.urdf.xacro',
                               ray_mode='panels', metadata_lidar0=META0))
    assert t.get('lidar0') == 'camera', f'anchor types: {t}'


def test_standalone_default_has_no_implicit_obscurants():
    doc = _expand('ouster_standalone.urdf.xacro',
                  ray_mode='raycast', metadata_lidar0=META0)
    assert not _obscurants(doc)


def test_standalone_smoke_profile_injects_explicit_volumes():
    doc = _expand('ouster_standalone.urdf.xacro',
                  ray_mode='raycast', metadata_lidar0=META0,
                  obscurant_profile='smoke_demo')
    volumes = _obscurants(doc)
    assert len(volumes) == 14
    # The profile is authoritative; it must not re-enable emitter mirroring.
    assert not doc.getElementsByTagName('particle_obscuration')


def test_standalone_imu_is_bound_to_its_sensor():
    doc = _expand('ouster_standalone.urdf.xacro',
                  ray_mode='raycast', metadata_lidar0=META0)
    assert _text_values(doc, 'imu_name') == ['lidar0_imu']


# ── sensor_stack (two sensors) ────────────────────────────────────────────────

def test_sensor_stack_raycast_both_altimeter():
    t = _anchor_types(_expand('sensor_stack.urdf.xacro',
                               ray_mode='raycast',
                               metadata_front=META0, metadata_rear=META_R))
    assert t.get('front') == 'altimeter', f'anchor types: {t}'
    assert t.get('rear') == 'altimeter', f'anchor types: {t}'


def test_sensor_stack_panels_both_camera():
    t = _anchor_types(_expand('sensor_stack.urdf.xacro',
                               ray_mode='panels',
                               metadata_front=META0, metadata_rear=META_R))
    assert t.get('front') == 'camera', f'anchor types: {t}'
    assert t.get('rear') == 'camera', f'anchor types: {t}'


def test_sensor_stack_imu_is_not_auto_detected():
    doc = _expand('sensor_stack.urdf.xacro',
                  ray_mode='raycast',
                  metadata_front=META0, metadata_rear=META_R)
    assert _text_values(doc, 'imu_name') == ['front_imu']


# ── turtlebot3_ouster (requires turtlebot3_description) ──────────────────────

def _has_turtlebot3() -> bool:
    r = subprocess.run(
        ['ros2', 'pkg', 'prefix', 'turtlebot3_description'],
        capture_output=True,
    )
    return r.returncode == 0


@pytest.mark.skipif(not _has_turtlebot3(),
                    reason='turtlebot3_description not installed')
def test_turtlebot3_raycast_anchor_is_altimeter():
    t = _anchor_types(_expand('turtlebot3_ouster.urdf.xacro',
                               ray_mode='raycast', metadata_lidar0=META0))
    assert t.get('lidar0') == 'altimeter', f'anchor types: {t}'


@pytest.mark.skipif(not _has_turtlebot3(),
                    reason='turtlebot3_description not installed')
def test_turtlebot3_panels_anchor_is_camera():
    t = _anchor_types(_expand('turtlebot3_ouster.urdf.xacro',
                               ray_mode='panels', metadata_lidar0=META0))
    assert t.get('lidar0') == 'camera', f'anchor types: {t}'
