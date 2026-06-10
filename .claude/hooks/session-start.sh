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
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

ROS_DISTRO=jazzy
WS="$HOME/ros2_ws"
REPO_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"

# Same pin as ci.yaml / Dockerfile — bump all three together.
OUSTER_ROS_REPO=https://github.com/jcfurey/ouster-ros.git
OUSTER_ROS_REF=6ab9402c1a8275f600945c3d8dfd5a73b40585c8

export DEBIAN_FRONTEND=noninteractive

# ── python3 must be noble's system 3.12 ───────────────────────────────────────
# The sandbox base image points python3 at a 3.11 build (update-alternatives +
# /usr/local/bin shim). ROS 2 Jazzy debs target noble's Python 3.12 — their
# C-extensions (numpy, rosidl) fail to import under 3.11, which breaks
# rosidl_generator_py at configure time.
if [ "$(readlink -f "$(command -v python3)")" != /usr/bin/python3.12 ]; then
  update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 100
  update-alternatives --set python3 /usr/bin/python3.12
  [ -L /usr/local/bin/python3 ] && ln -sf /usr/bin/python3.12 /usr/local/bin/python3
fi

# ── ROS 2 apt repository ──────────────────────────────────────────────────────
if [ ! -f /etc/apt/sources.list.d/ros2.list ]; then
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends curl ca-certificates gnupg
  curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") main" \
    > /etc/apt/sources.list.d/ros2.list
fi

# ── apt deps (plugin build deps per CI + tools; rosdep fills in the rest) ─────
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
# on noble (ephemeral sandbox, no host Python to corrupt).
command -v cpplint >/dev/null 2>&1 || \
  pip install --break-system-packages --no-cache-dir cpplint==1.6.1

# ── workspace: pinned ouster-ros + this repo (symlink) ────────────────────────
mkdir -p "$WS/src"
if [ ! -d "$WS/src/ouster-ros/.git" ]; then
  # Clone the ros2 branch specifically (master has a different layout that
  # leaves a duplicate sophus package in the workspace — see ci.yaml).
  git clone --branch ros2 --recurse-submodules "$OUSTER_ROS_REPO" "$WS/src/ouster-ros"
  git -C "$WS/src/ouster-ros" checkout "$OUSTER_ROS_REF"
  git -C "$WS/src/ouster-ros" submodule sync --recursive
  git -C "$WS/src/ouster-ros" submodule update --init --recursive
fi
[ -e "$WS/src/gz_sensors_ouster" ] || ln -s "$REPO_DIR" "$WS/src/gz_sensors_ouster"

# ── system deps via rosdep ────────────────────────────────────────────────────
[ -f /etc/ros/rosdep/sources.list.d/20-default.list ] || rosdep init
# rosdep update pulls from raw.githubusercontent.com, which intermittently
# resets connections; retry like the Dockerfile does.
for i in 1 2 3 4 5; do
  rosdep update --rosdistro=${ROS_DISTRO} && break
  echo "rosdep update failed (attempt $i); retrying in 10s..."
  sleep 10
done
apt-get update -qq
rosdep install --from-paths "$WS/src" --rosdistro=${ROS_DISTRO} -y --ignore-src

# ── build (ouster_sensor_msgs → ouster_ros → gz_sensors_ouster) ───────────────
# BUILD_TESTING=ON is required: the gtest targets are guarded by
# if(BUILD_TESTING). Skip if the plugin is already installed (cached container);
# incremental rebuilds after edits are the agent's job, not the hook's.
if [ ! -f "$WS/install/gz_sensors_ouster/share/gz_sensors_ouster/package.xml" ]; then
  # ROS setup.bash reads unset vars (AMENT_TRACE_SETUP_FILES); relax nounset.
  set +u
  source "/opt/ros/${ROS_DISTRO}/setup.bash"
  set -u
  (cd "$WS" && colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON)
fi

# ── persist ROS environment for the session ───────────────────────────────────
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  set +u
  source "/opt/ros/${ROS_DISTRO}/setup.bash"
  source "$WS/install/setup.bash"
  set -u
  for var in ROS_VERSION ROS_PYTHON_VERSION ROS_DISTRO AMENT_PREFIX_PATH \
             CMAKE_PREFIX_PATH COLCON_PREFIX_PATH LD_LIBRARY_PATH PATH PYTHONPATH; do
    [ -n "${!var:-}" ] && echo "export ${var}=\"${!var}\"" >> "$CLAUDE_ENV_FILE"
  done
fi

echo "ROS 2 ${ROS_DISTRO} workspace ready at $WS (gz_sensors_ouster symlinked from $REPO_DIR)."
