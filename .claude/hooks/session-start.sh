#!/bin/bash
# SessionStart hook: provision a ROS 2 Jazzy build environment for
# gz_sensors_ouster in Claude Code on the web sandboxes (Ubuntu 24.04).
#
# Mirrors .github/workflows/ci.yaml (jazzy job) and the Dockerfile:
#   - ROS 2 Jazzy + colcon + rosdep + gz_*_vendor packages (apt)
#   - ouster-ros from the jcfurey fork at a pinned commit (source dep; the
#     apt ros-jazzy-ouster-ros lags and exposes an older PacketWriter API)
#   - colcon workspace at ~/ros2_ws with this repo symlinked in
#   - cpplint/cppcheck so the CI lint gates can be run locally
#
# Idempotent: every step checks for prior completion, so a cached container
# (or a re-run after a partial failure) only does the missing work.
#
# NON-FATAL: provisioning runs in a strict-mode subshell whose failure is
# captured and downgraded to a warning. The hook ALWAYS exits 0, so a flaky
# network, a missing egress-allowlist entry (packages.ros.org /
# raw.githubusercontent.com), a non-root runner, an OOM build, or a timeout
# can never block session initialization. On failure you start in a
# partially-provisioned shell and finish the build by hand (the WARN message
# prints the resume command).

set -uo pipefail   # NB: deliberately NO -e at top level (see provision())

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

ROS_DISTRO=jazzy
WS="$HOME/ros2_ws"
REPO_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"

# Same pin as ci.yaml / Dockerfile — bump all three together.
OUSTER_ROS_REPO=https://github.com/jcfurey/ouster-ros.git
OUSTER_ROS_REF=338fa84a9d988eaae762c79bdd7bacbae497280a

export DEBIAN_FRONTEND=noninteractive

# All provisioning lives here. It carries no `set -e` of its own; strictness
# comes from the subshell it is invoked in below, which keeps the -e contained
# (a failure exits the subshell, never the hook).
provision() {
  local i rosdep_ok=""

  # ── python3 must be noble's system 3.12 ─────────────────────────────────
  # The sandbox base image points python3 at a 3.11 build (update-alternatives
  # + /usr/local/bin shim). ROS 2 Jazzy debs target noble's Python 3.12 —
  # their C-extensions (numpy, rosidl) fail to import under 3.11, which breaks
  # rosidl_generator_py at configure time.
  if [ "$(readlink -f "$(command -v python3)")" != /usr/bin/python3.12 ]; then
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 100
    update-alternatives --set python3 /usr/bin/python3.12
    [ -L /usr/local/bin/python3 ] && ln -sf /usr/bin/python3.12 /usr/local/bin/python3
  fi

  # ── ROS 2 apt repository ────────────────────────────────────────────────
  if [ ! -f /etc/apt/sources.list.d/ros2.list ]; then
    apt-get update -qq
    apt-get install -y -qq --no-install-recommends curl ca-certificates gnupg
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") main" \
      > /etc/apt/sources.list.d/ros2.list
  fi

  # ── apt deps (plugin build deps per CI + tools; rosdep fills in the rest) ─
  if [ ! -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]; then
    apt-get update -qq
    apt-get install -y -qq --no-install-recommends \
      git \
      build-essential \
      cmake \
      python3-colcon-common-extensions \
      python3-rosdep \
      python3-pip \
      libeigen3-dev \
      cppcheck \
      ros-${ROS_DISTRO}-ros-base \
      ros-${ROS_DISTRO}-gz-sim-vendor \
      ros-${ROS_DISTRO}-gz-rendering-vendor \
      ros-${ROS_DISTRO}-gz-sensors-vendor \
      ros-${ROS_DISTRO}-gz-plugin-vendor
  fi

  # cpplint pinned to the version CI uses; PEP 668 requires --break-system-packages
  # on noble (ephemeral sandbox, no host Python to corrupt). Checked as a module of
  # the CURRENT python3 — a bare `command -v cpplint` can pass while the module is
  # orphaned on another interpreter (e.g. after the 3.12 repoint above).
  if ! python3 -c 'import cpplint' >/dev/null 2>&1; then
    python3 -m pip --version >/dev/null 2>&1 || {
      apt-get update -qq
      apt-get install -y -qq --no-install-recommends python3-pip
    }
    python3 -m pip install --break-system-packages --no-cache-dir cpplint==1.6.1
  fi

  # ── workspace: pinned ouster-ros + this repo (symlink) ──────────────────
  mkdir -p "$WS/src"
  if [ ! -d "$WS/src/ouster-ros/.git" ]; then
    # Clone the cam-wip branch specifically (master has a different layout that
    # leaves a duplicate sophus package in the workspace — see ci.yaml).
    git clone --branch cam-wip --recurse-submodules "$OUSTER_ROS_REPO" "$WS/src/ouster-ros"
    git -C "$WS/src/ouster-ros" checkout "$OUSTER_ROS_REF"
    git -C "$WS/src/ouster-ros" submodule sync --recursive
    git -C "$WS/src/ouster-ros" submodule update --init --recursive
  fi
  git -C "$REPO_DIR" submodule update --init --recursive
  [ -e "$WS/src/gz_sensors_ouster" ] || ln -s "$REPO_DIR" "$WS/src/gz_sensors_ouster"

  # ── system deps via rosdep ──────────────────────────────────────────────
  [ -f /etc/ros/rosdep/sources.list.d/20-default.list ] || rosdep init
  # rosdep update pulls from raw.githubusercontent.com, which intermittently
  # resets connections; retry like the Dockerfile does. On exhaustion we
  # `return 1` (not exit) so the subshell unwinds and the hook reports a
  # warning instead of dying.
  for i in 1 2 3 4 5; do
    if rosdep update --rosdistro=${ROS_DISTRO}; then rosdep_ok=1; break; fi
    echo "rosdep update failed (attempt $i); retrying in 10s..."
    sleep 10
  done
  [ -n "$rosdep_ok" ] || { echo "ERROR: rosdep update failed after 5 attempts." >&2; return 1; }
  apt-get update -qq
  rosdep install --from-paths "$WS/src" --rosdistro=${ROS_DISTRO} -y --ignore-src

  # ── build (ouster_sensor_msgs → ouster_ros → gz_sensors_ouster) ─────────
  # BUILD_TESTING=ON is required: the gtest targets are guarded by
  # if(BUILD_TESTING). Skip if the plugin is already installed (cached
  # container); incremental rebuilds after edits are the agent's job.
  if [ ! -f "$WS/install/gz_sensors_ouster/share/gz_sensors_ouster/package.xml" ]; then
    # ROS setup.bash reads unset vars (AMENT_TRACE_SETUP_FILES); relax nounset.
    set +u
    source "/opt/ros/${ROS_DISTRO}/setup.bash"
    set -u
    (cd "$WS" && colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON)
  fi
}

# ── persist whatever ROS environment exists, success or not ────────────────
# Best-effort and guarded by file existence, so even a failed/partial run this
# session still exports a working env if a prior session already built the WS.
persist_env() {
  [ -n "${CLAUDE_ENV_FILE:-}" ] || return 0
  [ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ] || return 0
  set +u
  source "/opt/ros/${ROS_DISTRO}/setup.bash"
  [ -f "$WS/install/setup.bash" ] && source "$WS/install/setup.bash"
  set -u
  for var in ROS_VERSION ROS_PYTHON_VERSION ROS_DISTRO AMENT_PREFIX_PATH \
             CMAKE_PREFIX_PATH COLCON_PREFIX_PATH LD_LIBRARY_PATH PATH PYTHONPATH; do
    [ -n "${!var:-}" ] && echo "export ${var}=\"${!var}\"" >> "$CLAUDE_ENV_FILE"
  done
}

# Run provisioning in a standalone strict subshell. Because it is NOT part of
# an if/&&/|| list, the subshell's `set -e` is honored, and because it is a
# subshell, that -e (and any exit it triggers) stays contained — $? carries the
# result back without the hook ever inheriting -e.
(
  set -euo pipefail
  provision
)
rc=$?

persist_env || true

if [ "$rc" -ne 0 ]; then
  cat >&2 <<EOF
WARN: gz_sensors_ouster provisioning did not complete (exit $rc).
      The session is starting anyway in a partially-provisioned shell.
      Likely causes on a web sandbox:
        * outbound egress to packages.ros.org / raw.githubusercontent.com
          is not on the allowlist (add them in the sandbox network settings)
        * an apt / update-alternatives / rosdep step needs root
        * the colcon build hit the hook timeout or ran out of memory
      Resume provisioning manually with:
        bash "$REPO_DIR/.claude/hooks/session-start.sh"
      or run just the failing phase (apt / rosdep / 'colcon build' in $WS).
EOF
else
  echo "ROS 2 ${ROS_DISTRO} workspace ready at $WS (gz_sensors_ouster symlinked from $REPO_DIR)."
fi

exit 0
