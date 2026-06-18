"""Display-hardening regression tests.

Verifies that the changes protecting headless/WSL/Wayland users from cryptic
display errors are present and complete:
  - docker/entrypoint.sh: warns early when DISPLAY and WAYLAND_DISPLAY are
    both unset, with actionable X11/WSL/Wayland guidance.
  - All three example launches: headless arg present with default=false,
    gz server-only (-s) flag wired to the headless condition.

These are text/structural checks — no simulation or display required.
"""
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
ENTRYPOINT = REPO / 'docker' / 'entrypoint.sh'
LAUNCHES = REPO / 'examples' / 'launch'

EXAMPLE_LAUNCHES = [
    'ouster_standalone.launch.py',
    'sensor_stack.launch.py',
    'turtlebot3_ouster.launch.py',
]


# ── entrypoint.sh display guard ───────────────────────────────────────────────

def test_entrypoint_checks_display_env_var():
    src = ENTRYPOINT.read_text()
    assert 'DISPLAY' in src


def test_entrypoint_checks_wayland_display_env_var():
    src = ENTRYPOINT.read_text()
    assert 'WAYLAND_DISPLAY' in src


def test_entrypoint_has_no_display_warning():
    src = ENTRYPOINT.read_text()
    assert 'no display' in src.lower()


def test_entrypoint_has_x11_guidance():
    src = ENTRYPOINT.read_text()
    assert '.X11-unix' in src or 'X11' in src


def test_entrypoint_has_wsl_guidance():
    src = ENTRYPOINT.read_text()
    assert 'WSL' in src


def test_entrypoint_guard_is_in_drive_gui_block():
    src = ENTRYPOINT.read_text()
    drive_start = src.index('drive|gui)')
    guard_pos = src.find('WAYLAND_DISPLAY', drive_start)
    assert guard_pos != -1, 'WAYLAND_DISPLAY check not found inside drive|gui block'


# ── launch file headless wiring ───────────────────────────────────────────────

def test_launches_have_headless_arg():
    for name in EXAMPLE_LAUNCHES:
        src = (LAUNCHES / name).read_text()
        assert "DeclareLaunchArgument('headless'" in src, \
            f'{name}: missing headless DeclareLaunchArgument'


def test_launches_headless_arg_defaults_to_false():
    for name in EXAMPLE_LAUNCHES:
        src = (LAUNCHES / name).read_text()
        idx = src.index("DeclareLaunchArgument('headless'")
        block = src[idx:idx + 300]
        assert "default_value='false'" in block, \
            f'{name}: headless arg default is not false'


def test_launches_wire_headless_to_server_only_flag():
    for name in EXAMPLE_LAUNCHES:
        src = (LAUNCHES / name).read_text()
        assert "' -s -r -v 3'" in src, \
            f'{name}: -s (server-only) gz flag not found'
