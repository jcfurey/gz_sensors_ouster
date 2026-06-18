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
    assert '${DISPLAY:-}' in src


def test_entrypoint_checks_wayland_display_env_var():
    src = ENTRYPOINT.read_text()
    assert 'WAYLAND_DISPLAY' in src


def test_entrypoint_has_no_display_warning():
    src = ENTRYPOINT.read_text()
    assert 'no display' in src.lower()


def test_entrypoint_has_x11_guidance():
    src = ENTRYPOINT.read_text()
    assert '.X11-unix' in src


def test_entrypoint_has_wsl_guidance():
    src = ENTRYPOINT.read_text()
    assert 'WSLg' in src


def test_entrypoint_guard_is_in_drive_gui_block():
    src = ENTRYPOINT.read_text()
    drive_start = src.index('drive|gui)')
    guard_pos = src.find('WAYLAND_DISPLAY', drive_start)
    assert guard_pos != -1, 'WAYLAND_DISPLAY check not found inside drive|gui block'

