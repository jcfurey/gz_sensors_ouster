"""Structural invariants for the example world SDF files.

Tests run without any simulation, display, or GPU — pure XML parsing.
Catches regressions like accidentally adding gz-sim-sensors-system to a
raycast world or removing gz-sim-altimeter-system.
"""
import pathlib
import xml.dom.minidom

import pytest

WORLDS = pathlib.Path(__file__).resolve().parent.parent / 'examples' / 'worlds'
WORLD_NAMES = sorted(p.name for p in WORLDS.glob('*.sdf'))


def _plugins(name: str) -> set:
    doc = xml.dom.minidom.parse(str(WORLDS / name))
    return {p.getAttribute('filename') for p in doc.getElementsByTagName('plugin')}


@pytest.mark.parametrize('name', WORLD_NAMES)
def test_sdf_parses(name):
    xml.dom.minidom.parse(str(WORLDS / name))


@pytest.mark.parametrize('name', WORLD_NAMES)
def test_all_worlds_have_physics(name):
    assert 'gz-sim-physics-system' in _plugins(name)


@pytest.mark.parametrize('name', ['ouster_demo.sdf',
                                   'ouster_demo_panels.sdf',
                                   'turtlebot3_ouster_headless.sdf'])
def test_example_worlds_have_imu(name):
    assert 'gz-sim-imu-system' in _plugins(name)


# ── Raycast worlds: no rendering system, yes altimeter ───────────────────────

@pytest.mark.parametrize('name', ['ouster_demo.sdf', 'turtlebot3_ouster_headless.sdf'])
def test_raycast_worlds_have_no_sensors_system(name):
    assert 'gz-sim-sensors-system' not in _plugins(name)


@pytest.mark.parametrize('name', ['ouster_demo.sdf', 'turtlebot3_ouster_headless.sdf'])
def test_raycast_worlds_have_altimeter_system(name):
    assert 'gz-sim-altimeter-system' in _plugins(name)


# ── Panels world: rendering system present, no altimeter ─────────────────────

def test_panels_world_has_sensors_system():
    assert 'gz-sim-sensors-system' in _plugins('ouster_demo_panels.sdf')


def test_panels_world_has_no_altimeter_system():
    assert 'gz-sim-altimeter-system' not in _plugins('ouster_demo_panels.sdf')


def test_panels_world_uses_ogre2():
    doc = xml.dom.minidom.parse(str(WORLDS / 'ouster_demo_panels.sdf'))
    engines = [e.firstChild.nodeValue.strip()
               for e in doc.getElementsByTagName('render_engine')
               if e.firstChild]
    assert 'ogre2' in engines
