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
                                   'ouster_showcase.sdf',
                                   'ouster_smoke.sdf',
                                   'turtlebot3_ouster_headless.sdf'])
def test_example_worlds_have_imu(name):
    assert 'gz-sim-imu-system' in _plugins(name)


# ── Raycast worlds: no rendering system, yes altimeter ───────────────────────

RAYCAST_WORLDS = ['ouster_demo.sdf', 'ouster_showcase.sdf', 'ouster_smoke.sdf',
                  'turtlebot3_ouster_headless.sdf']


@pytest.mark.parametrize('name', RAYCAST_WORLDS)
def test_raycast_worlds_have_no_sensors_system(name):
    assert 'gz-sim-sensors-system' not in _plugins(name)


@pytest.mark.parametrize('name', RAYCAST_WORLDS)
def test_raycast_worlds_have_altimeter_system(name):
    assert 'gz-sim-altimeter-system' in _plugins(name)


# The raycast scene mirror only handles box / sphere / cylinder / plane /
# mesh visuals (src/raycast_mirror.cpp). Anything else — capsule, ellipsoid,
# heightmap, polyline — is silently skipped, so it would be INVISIBLE to the
# lidar while still showing up in the GUI: a demo world that looks right and
# scans wrong. Guard the example worlds against acquiring one.
MIRRORED_GEOMETRY = {'box', 'sphere', 'cylinder', 'plane', 'mesh'}


# The showcase world's reflectance/retro ladders encode their intended
# laser_retro in the model name (B_retro_045 -> 0.45, C_retro_2p6 -> 2.6), and
# the zone comments quote the resulting reflectivity bytes. That makes a
# name/value mismatch machine-checkable — and it is worth checking, because a
# careless global search-and-replace over the SDF (e.g. retuning one zone's
# values, or bumping the perimeter walls' retro) silently rewrites any other
# model that happened to share the old literal, breaking the ladder's
# monotonicity while the world still loads and still looks fine.

def _showcase_retro_by_model() -> dict:
    doc = xml.dom.minidom.parse(str(WORLDS / 'ouster_showcase.sdf'))
    out = {}
    for model in doc.getElementsByTagName('model'):
        retros = [r.firstChild.nodeValue.strip()
                  for r in model.getElementsByTagName('laser_retro')
                  if r.firstChild]
        if retros:
            out[model.getAttribute('name')] = [float(v) for v in retros]
    return out


def test_showcase_ladder_retro_matches_model_name():
    """B_retro_045 must actually carry laser_retro 0.45, etc."""
    mismatches = []
    for model, retros in _showcase_retro_by_model().items():
        if model.startswith('B_retro_'):
            expected = int(model.rsplit('_', 1)[1]) / 100.0
        elif model.startswith('C_retro_'):
            expected = float(model.rsplit('_', 1)[1].replace('p', '.'))
        else:
            continue
        if len(retros) != 1 or abs(retros[0] - expected) > 1e-9:
            mismatches.append(f'{model}: expected {expected}, found {retros}')
    assert not mismatches, (
        'ladder model name does not match its laser_retro '
        f'(a stray global replace?): {mismatches}')


def test_showcase_ladders_are_monotonic():
    """Both reflectance ladders must increase, or the demo they exist for
    (a clean staircase in the REFLECTIVITY channel) is broken."""
    by_model = _showcase_retro_by_model()
    for prefix in ('B_retro_', 'C_retro_'):
        rungs = sorted(
            ((m, v[0]) for m, v in by_model.items() if m.startswith(prefix)),
            key=lambda kv: kv[1])
        values = [v for _, v in rungs]
        assert len(values) >= 3, f'{prefix} ladder is suspiciously short'
        assert values == sorted(set(values)), (
            f'{prefix} ladder has duplicate or non-monotonic rungs: {rungs}')


def test_showcase_comments_match_perimeter_distance():
    """The zone/coverage prose quotes the perimeter distance; keep it honest.
    Doc drift here misdescribes the scene rather than breaking it, but this
    world's whole purpose is being a readable reference."""
    text = (WORLDS / 'ouster_showcase.sdf').read_text()
    walls = [m for m in _showcase_retro_by_model() if m.startswith('J_wall_')]
    assert walls, 'expected J_wall_* perimeter models'
    doc = xml.dom.minidom.parse(str(WORLDS / 'ouster_showcase.sdf'))
    dists = set()
    for model in doc.getElementsByTagName('model'):
        if not model.getAttribute('name').startswith('J_wall_'):
            continue
        pose = model.getElementsByTagName('pose')[0].firstChild.nodeValue.split()
        dists.add(round(max(abs(float(pose[0])), abs(float(pose[1])))))
    assert len(dists) == 1, f'perimeter walls are at mixed distances: {dists}'
    actual = dists.pop()
    # Any comment quoting a different "<N> m wall/perimeter" distance is stale.
    import re as _re
    quoted = {int(v) for v in _re.findall(r'(\d+)\s*m (?:walls|perimeter)', text)}
    stale = {q for q in quoted if q != actual}
    assert not stale, (
        f'comments mention {sorted(stale)} m walls but the perimeter is at '
        f'{actual} m')


@pytest.mark.parametrize('name', WORLD_NAMES)
def test_visual_geometry_is_mirrorable(name):
    doc = xml.dom.minidom.parse(str(WORLDS / name))
    offenders = []
    for visual in doc.getElementsByTagName('visual'):
        for geom in visual.getElementsByTagName('geometry'):
            for child in geom.childNodes:
                if child.nodeType != child.ELEMENT_NODE:
                    continue
                if child.tagName not in MIRRORED_GEOMETRY:
                    offenders.append(
                        f'{visual.getAttribute("name") or "<unnamed>"}'
                        f' -> {child.tagName}')
    assert not offenders, (
        f'{name} has visual geometry the raycast mirror cannot see '
        f'(silently invisible to the lidar): {offenders}')


# ── Smoke world: the obscuration demo ────────────────────────────────────────
#
# Its whole point is a monotone density ladder, and every cloud in it is an
# ordinary <particle_emitter> — no plugin configuration. Both properties are
# easy to break silently (a retuned ratio that lands out of order still loads
# and still looks like smoke), so pin them.

def _smoke_emitters() -> dict:
    doc = xml.dom.minidom.parse(str(WORLDS / 'ouster_smoke.sdf'))
    out = {}
    for model in doc.getElementsByTagName('model'):
        for em in model.getElementsByTagName('particle_emitter'):
            ratios = [r.firstChild.nodeValue.strip()
                      for r in em.getElementsByTagName(
                          'particle_scatter_ratio') if r.firstChild]
            out[model.getAttribute('name')] = {
                'type': em.getAttribute('type'),
                'ratio': float(ratios[0]) if ratios else None,
            }
    return out


def test_smoke_world_ladder_ratio_matches_model_name():
    """A_smoke_035 must actually carry scatter ratio 0.35 — the header table
    quotes derived optical depths keyed to these names."""
    mismatches = []
    for model, em in _smoke_emitters().items():
        if not model.startswith(('A_smoke_', 'C_sky_')):
            continue
        expected = int(model.rsplit('_', 1)[1]) / 100.0
        if em['ratio'] is None or abs(em['ratio'] - expected) > 1e-9:
            mismatches.append(f'{model}: expected {expected}, found {em}')
    assert not mismatches, (
        f'smoke ladder name/ratio mismatch (a stray global replace?): '
        f'{mismatches}')


def test_smoke_world_ladder_is_monotonic():
    rungs = sorted((m, e['ratio'])
                   for m, e in _smoke_emitters().items()
                   if m.startswith('A_smoke_'))
    values = [v for _, v in rungs]
    assert len(values) >= 4, 'density ladder is suspiciously short'
    assert values == sorted(set(values)), (
        f'density ladder has duplicate or non-monotonic rungs: {rungs}')


def test_smoke_world_ladder_exercises_default_reflectivity():
    """Zone A deliberately omits laser_retro so smoke competes with the
    physical fallback, not a zero placeholder or an explicitly tagged wall."""
    doc = xml.dom.minidom.parse(str(WORLDS / 'ouster_smoke.sdf'))
    walls = [model for model in doc.getElementsByTagName('model')
             if model.getAttribute('name').startswith('A_wall_')]
    assert len(walls) >= 4, 'density ladder is suspiciously short'
    tagged = [model.getAttribute('name') for model in walls
              if model.getElementsByTagName('laser_retro')]
    assert not tagged, f'fallback ladder walls unexpectedly tagged: {tagged}'


def test_smoke_world_covers_every_emitter_shape():
    """Each emitter type takes a different branch in the volume conversion
    (src/obscurants.cpp); the gallery zone exists to exercise all of them."""
    kinds = {e['type'] for e in _smoke_emitters().values()}
    assert {'box', 'cylinder', 'ellipsoid', 'point'} <= kinds, kinds


def test_smoke_world_needs_no_plugin_configuration():
    """The demo's claim is that mirroring gz particle emitters is automatic.
    An <obscurant> block or a tweaked knob in the world would undercut it —
    and the plugin lives in the xacro, not here, so there is nowhere for one
    to legitimately appear."""
    # Match on parsed elements, not raw text: the header comment names these
    # knobs on purpose, pointing the reader at the docs.
    doc = xml.dom.minidom.parse(str(WORLDS / 'ouster_smoke.sdf'))
    for knob in ('obscurant', 'particle_extinction', 'particle_obscuration',
                 'particle_growth', 'obscurant_lidar_ratio',
                 'obscurant_albedo', 'pulse_length'):
        found = doc.getElementsByTagName(knob)
        assert not found, f'<{knob}> defeats the point of this world'


def test_smoke_world_serves_emitter_command_topics():
    """The header tells the reader to toggle clouds over gz topic; that only
    works with the particle-emitter system loaded."""
    assert 'gz-sim-particle-emitter-system' in _plugins('ouster_smoke.sdf')


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
