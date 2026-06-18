"""Display-hardening regression tests.

Verifies that docker/entrypoint.sh protects headless/WSL/Wayland users from
cryptic display errors: it warns early when DISPLAY and WAYLAND_DISPLAY are
both unset, with actionable X11/WSL/Wayland guidance.

(Launch-file headless wiring is covered by test_launch_files.py.)

These are text/structural checks — no simulation or display required.
"""
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent
ENTRYPOINT = REPO / 'docker' / 'entrypoint.sh'

# The display-guard condition, tolerant of bash spacing:
#   if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
_GUARD_RE = re.compile(
    r'\[\s*-z\s*"\$\{DISPLAY:-\}"\s*\]\s*&&\s*'
    r'\[\s*-z\s*"\$\{WAYLAND_DISPLAY:-\}"\s*\]')


def _guard_block():
    """The display-warning block inside the drive|gui case: from the guard
    condition `if [ -z "${DISPLAY..." to its matching `fi`. Raises ValueError
    (failing the test) if the guard or its terminator is missing."""
    src = ENTRYPOINT.read_text()
    start = src.index('if [ -z "${DISPLAY:-}"')
    end = src.index('\n    fi', start)
    return src[start:end]


# ── entrypoint.sh display guard ───────────────────────────────────────────────

def test_entrypoint_guards_on_both_display_vars():
    # The guard must fire only when BOTH DISPLAY and WAYLAND_DISPLAY are unset;
    # tripping on either alone would warn users who actually have a display.
    assert _GUARD_RE.search(ENTRYPOINT.read_text()), \
        'display guard must test both ${DISPLAY:-} and ${WAYLAND_DISPLAY:-}'


def test_entrypoint_warns_no_display():
    assert 'no display' in _guard_block().lower(), \
        'guard block must print a "no display" warning'


def test_entrypoint_has_x11_guidance():
    block = _guard_block()
    assert '.X11-unix' in block and '-e DISPLAY' in block, \
        'X11 guidance (DISPLAY + /tmp/.X11-unix mount) missing from guard block'


def test_entrypoint_has_wsl_guidance():
    assert 'WSLg' in _guard_block(), \
        'WSL guidance (WSLg / VcXsrv) missing from guard block'


def test_entrypoint_has_wayland_guidance():
    block = _guard_block()
    assert 'WAYLAND_DISPLAY' in block and 'XDG_RUNTIME_DIR' in block, \
        'Wayland guidance (WAYLAND_DISPLAY + XDG_RUNTIME_DIR) missing from guard'


def test_entrypoint_guard_is_in_drive_gui_block():
    src = ENTRYPOINT.read_text()
    drive_start = src.index('drive|gui)')
    guard_pos = src.find('WAYLAND_DISPLAY', drive_start)
    assert guard_pos != -1, 'WAYLAND_DISPLAY check not found inside drive|gui block'
