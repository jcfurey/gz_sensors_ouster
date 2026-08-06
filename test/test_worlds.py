"""Structural invariants for the example world SDF files.

Tests run without any simulation, display, or GPU — pure XML parsing.
Catches regressions like accidentally adding gz-sim-sensors-system to a
raycast world or removing gz-sim-altimeter-system.
"""
import pathlib
import struct
import xml.dom.minidom

import pytest

WORLDS = pathlib.Path(__file__).resolve().parent.parent / 'examples' / 'worlds'
EXAMPLES = WORLDS.parent
SMOKE_PROFILE = EXAMPLES / 'urdf' / 'ouster_smoke_obscurants.xacro'
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
                                   'turtlebot3_ouster_headless.sdf',
                                   'turtlebot3_ouster_warehouse.sdf',
                                   'turtlebot3_ouster_hills.sdf',
                                   'turtlebot3_ouster_sewer.sdf'])
def test_example_worlds_have_imu(name):
    assert 'gz-sim-imu-system' in _plugins(name)


# ── Raycast worlds: no rendering system, yes altimeter ───────────────────────

RAYCAST_WORLDS = ['ouster_demo.sdf', 'ouster_showcase.sdf', 'ouster_smoke.sdf',
                  'turtlebot3_ouster_headless.sdf',
                  'turtlebot3_ouster_warehouse.sdf',
                  'turtlebot3_ouster_hills.sdf',
                  'turtlebot3_ouster_sewer.sdf']


@pytest.mark.parametrize('name', RAYCAST_WORLDS)
def test_raycast_worlds_have_no_sensors_system(name):
    assert 'gz-sim-sensors-system' not in _plugins(name)


@pytest.mark.parametrize('name', RAYCAST_WORLDS)
def test_raycast_worlds_have_altimeter_system(name):
    assert 'gz-sim-altimeter-system' in _plugins(name)


@pytest.mark.parametrize('name', WORLD_NAMES)
def test_world_physics_is_paced_at_authored_real_time_factor(name):
    """A max step without its matching update rate can let sim time sprint,
    flooding ouster_ros lidar_scans queues with packets."""
    doc = xml.dom.minidom.parse(str(WORLDS / name))
    physics = doc.getElementsByTagName('physics')[0]

    def value(tag: str) -> float:
        node = physics.getElementsByTagName(tag)[0]
        return float(node.firstChild.nodeValue.strip())

    step = value('max_step_size')
    factor = value('real_time_factor')
    update_rate = value('real_time_update_rate')
    assert step > 0 and update_rate > 0
    assert step * update_rate == pytest.approx(factor)


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


def _png_header(path: pathlib.Path) -> tuple:
    data = path.read_bytes()[:29]
    assert data[:8] == b'\x89PNG\r\n\x1a\n'
    assert data[12:16] == b'IHDR'
    width, height = struct.unpack('>II', data[16:24])
    return width, height, data[24], data[25]


def test_brick_graffiti_response_map_is_aligned_rgba8():
    textures = EXAMPLES / 'media' / 'materials' / 'textures'
    visible = textures / 'brick_graffiti.png'
    response = textures / 'brick_graffiti.ouster.png'
    assert visible.is_file() and response.is_file()
    vw, vh, vdepth, _ = _png_header(visible)
    rw, rh, rdepth, rtype = _png_header(response)
    assert (rw, rh) == (vw, vh)
    assert vdepth == rdepth == 8
    assert rtype == 6, 'response companion must be RGBA8'


def test_response_textured_worlds_reference_the_visible_companion_base():
    for name in ('turtlebot3_ouster_warehouse.sdf',
                 'turtlebot3_ouster_hills.sdf',
                 'turtlebot3_ouster_sewer.sdf'):
        text = (WORLDS / name).read_text()
        assert '../media/materials/textures/brick_graffiti.png' in text
        # The nonstandard physics map stays out of SDF; the plugin discovers
        # the adjacent `<stem>.ouster.png` companion from this albedo path.
        assert '.ouster.png</albedo_map>' not in text


def test_response_textured_materials_use_neutral_visible_color_factors():
    """Ogre multiplies PBR albedo maps by the material color factors.

    Keep them explicit and neutral so the visible texture cannot render black
    because of renderer/version-specific implicit material defaults.
    """
    for name in WORLD_NAMES:
        doc = xml.dom.minidom.parse(str(WORLDS / name))
        for material in doc.getElementsByTagName('material'):
            maps = material.getElementsByTagName('albedo_map')
            if not maps or 'brick_graffiti.png' not in maps[0].firstChild.nodeValue:
                continue
            for tag in ('ambient', 'diffuse'):
                values = material.getElementsByTagName(tag)
                assert len(values) == 1, f'{name}: textured material lacks {tag}'
                color = [float(v) for v in
                         values[0].firstChild.nodeValue.split()]
                assert color == [1.0, 1.0, 1.0, 1.0], (
                    f'{name}: {tag} must not tint its albedo map: {color}')


def test_hilly_world_uses_an_installed_mesh_not_heightmap():
    text = (WORLDS / 'turtlebot3_ouster_hills.sdf').read_text()
    assert '../media/meshes/hilly_terrain.obj' in text
    assert '<heightmap>' not in text
    assert (EXAMPLES / 'media' / 'meshes' / 'hilly_terrain.obj').is_file()


# ── Smoke world: the obscuration demo ────────────────────────────────────────
#
# Its visible particle emitters are deliberately separate from the companion
# xacro's explicit optical volumes. Pin both layers so a visual edit cannot
# silently rewrite LiDAR physics or leave the two profiles misaligned.

def _smoke_emitters() -> dict:
    doc = xml.dom.minidom.parse(str(WORLDS / 'ouster_smoke.sdf'))
    out = {}
    for model in doc.getElementsByTagName('model'):
        for em in model.getElementsByTagName('particle_emitter'):
            def value(tag):
                nodes = [n.firstChild.nodeValue.strip()
                         for n in em.getElementsByTagName(tag) if n.firstChild]
                return nodes[0] if nodes else None

            ratio = value('particle_scatter_ratio')
            out[model.getAttribute('name')] = {
                'type': em.getAttribute('type'),
                'ratio': float(ratio) if ratio else None,
                'rate': float(value('rate')),
                'pose': value('pose'),
                'size': value('size'),
                'albedo_map': value('albedo_map'),
                'color_range_image': value('color_range_image'),
            }
    return out


def _smoke_volumes() -> list:
    doc = xml.dom.minidom.parse(str(SMOKE_PROFILE))
    return [{key: node.getAttribute(key)
             for key in ('type', 'pose', 'size', 'extinction')}
            for node in doc.getElementsByTagName('xacro:smoke_volume')]


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


def test_smoke_world_uses_consistent_ground_source_emitters():
    """Visual smoke rises from a thin source instead of filling arbitrary
    volumes or shooting radially along each model's local +X axis."""
    emitters = _smoke_emitters()
    kinds = {e['type'] for e in emitters.values()}
    assert kinds == {'box'}
    assert {e['pose'] for e in emitters.values()} == {
        '0 0 -1.15 0 -1.5708 0'}
    assert {e['size'] for e in emitters.values()} == {'0.08 1.4 1.4'}


def test_smoke_world_uses_soft_lifetime_textures():
    emitters = _smoke_emitters()
    assert {e['albedo_map'] for e in emitters.values()} == {
        '../media/particles/fog.png'}
    assert {e['color_range_image'] for e in emitters.values()} == {
        '../media/particles/fogcolors.png'}
    for relative in ('fog.png', 'fogcolors.png'):
        assert (EXAMPLES / 'media' / 'particles' / relative).is_file()


def test_smoke_world_visual_rate_tracks_density_ladder():
    rungs = sorted(
        (e['ratio'], e['rate']) for model, e in _smoke_emitters().items()
        if model.startswith('A_smoke_'))
    rates = [rate for _, rate in rungs]
    assert rates == sorted(set(rates)), rungs


def test_smoke_world_has_one_explicit_volume_per_visual_plume():
    volumes = _smoke_volumes()
    emitters = _smoke_emitters()
    assert len(volumes) == len(emitters) == 14
    assert all(v['pose'] and v['size'] and v['extinction'] for v in volumes)


def test_smoke_world_explicit_density_ladder_is_monotonic():
    """The first five companion volumes are Zone A's physical ladder."""
    extinctions = [float(v['extinction']) for v in _smoke_volumes()[:5]]
    assert extinctions == [0.05, 0.15, 0.35, 0.65, 1.0]


def test_smoke_world_uses_only_supported_explicit_volume_shapes():
    kinds = {v['type'] for v in _smoke_volumes()}
    assert kinds <= {'box', 'cylinder', 'ellipsoid'}
    assert {'box', 'cylinder', 'ellipsoid'} <= kinds


def test_standalone_launch_selects_smoke_physics_profile():
    text = (EXAMPLES / 'launch' / 'ouster_standalone.launch.py').read_text()
    assert "'smoke_demo'" in text
    assert 'obscurant_profile:=' in text


def test_smoke_world_serves_emitter_command_topics():
    """The visual overlay still needs Gazebo's particle-emitter system."""
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
