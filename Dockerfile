# Standalone test image for gz_sensors_ouster.
#
# Builds the plugin in isolation against a selectable ROS 2 distro and exercises
# it on a drivable TurtleBot3 waffle in the new Gazebo. By default no CUDA
# toolchain is installed, so the plugin compiles its CPU (OpenMP) backend and the
# default headless smoke runs in *raycast* mode (no render engine / GPU needed).
#
#   docker build -t gzouster .                          # ROS 2 Jazzy (default)
#   docker build -t gzouster --build-arg ROS_DISTRO=kilted .
#   docker build -t gzouster --build-arg ROS_DISTRO=lyrical .   # advisory (brand new)
#
#   docker run --rm gzouster                            # headless point-cloud smoke
#   docker run --rm gzouster test                       # gtest suite
#   docker run --rm -it -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
#       --gpus all gzouster drive                       # interactive GUI + teleop
#
# CUDA (NVIDIA) — leverage the HOST's GPU without the rovermax_ws image. This
# installs only the CUDA *toolkit* (nvcc + cudart + curand) into the image and
# builds the plugin's CUDA backend; the GPU driver itself comes from the host at
# run time via --gpus all (needs nvidia-container-toolkit on the host). No CUDA
# install on the host is required beyond the driver.
#
#   docker build -t gzouster-cuda --build-arg ENABLE_CUDA=true \
#       --build-arg CUDA_ARCH=86 .                      # 86 = Ampere (RTX 30xx)
#   docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all gzouster-cuda
#       # plugin logs "Using cuda backend." instead of the CPU fallback.
#
# Distros: jazzy (Harmonic), kilted (Ionic), lyrical (Jetty). Humble is NOT
# supported (no gz_*_vendor; it ships Gazebo Fortress, not Harmonic+). CUDA is
# wired for the noble-based distros (jazzy/kilted); see CUDA_DISTRO to override.

ARG ROS_DISTRO=jazzy
FROM osrf/ros:${ROS_DISTRO}-desktop
ARG ROS_DISTRO

# CUDA backend (optional). ENABLE_CUDA=true installs the toolkit + builds it.
# CUDA_ARCH is a ';'-separated SM list (e.g. "86" for RTX 30xx, "89" for 40xx,
# "75;80;86;89" for a portable image). CUDA_DISTRO/CUDA_PKG_VERSION pick the
# NVIDIA apt repo + toolkit version (default: Ubuntu 24.04 'noble', CUDA 12.6).
ARG ENABLE_CUDA=false
ARG CUDA_ARCH=80;86;89
ARG CUDA_DISTRO=ubuntu2404
ARG CUDA_PKG_VERSION=12-6
ARG CUDA_HOME_VERSION=12.6

# ouster-ros is pinned to the jcfurey fork (carries an in-tree ouster-sdk
# submodule at v0.16.2+); the apt ros-${ROS_DISTRO}-ouster-ros lags and exposes
# an older PacketWriter API the plugin no longer targets. Pinned to an exact
# commit for reproducibility (same SHA as the package's CI). turtlebot3 is pinned
# for the genuine waffle *description* (geometry only).
ARG OUSTER_ROS_REPO=https://github.com/jcfurey/ouster-ros.git
ARG OUSTER_ROS_REF=6ab9402c1a8275f600945c3d8dfd5a73b40585c8
ARG TURTLEBOT3_REPO=https://github.com/ROBOTIS-GIT/turtlebot3.git
ARG TURTLEBOT3_BRANCH=jazzy
ARG TURTLEBOT3_REF=1f67e8d477df3e91729a03e43a9cd71cc67addfa

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]
WORKDIR /ws

# ── apt deps ──────────────────────────────────────────────────────────────────
# gz_*_vendor + eigen are the plugin's build deps (same set as CI). ros_gz_sim /
# ros_gz_bridge / xacro / robot_state_publisher / rviz2 / teleop_twist_keyboard
# drive the example. git for the source clones below.
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
      git \
      build-essential \
      cmake \
      python3-colcon-common-extensions \
      libeigen3-dev \
      ros-${ROS_DISTRO}-gz-sim-vendor \
      ros-${ROS_DISTRO}-gz-rendering-vendor \
      ros-${ROS_DISTRO}-gz-sensors-vendor \
      ros-${ROS_DISTRO}-gz-plugin-vendor \
      ros-${ROS_DISTRO}-ros-gz-sim \
      ros-${ROS_DISTRO}-ros-gz-bridge \
      ros-${ROS_DISTRO}-xacro \
      ros-${ROS_DISTRO}-robot-state-publisher \
      ros-${ROS_DISTRO}-rviz2 \
      ros-${ROS_DISTRO}-teleop-twist-keyboard \
 && rm -rf /var/lib/apt/lists/*

# ── CUDA toolkit (optional; ENABLE_CUDA=true) ─────────────────────────────────
# Only the toolkit (nvcc + cudart + curand) goes in the image; the driver/libcuda
# comes from the host at run time via --gpus all. We install the versioned
# packages and symlink /usr/local/cuda so CMake's CUDAToolkit lookup + the
# package's check_language(CUDA) both resolve. ENV is set unconditionally (a
# missing /usr/local/cuda/bin on a CPU image is simply absent from PATH).
RUN if [ "${ENABLE_CUDA}" = "true" ]; then \
      set -e; \
      apt-get update -qq; \
      apt-get install -y -qq --no-install-recommends wget ca-certificates gnupg; \
      wget -qO /tmp/cuda-keyring.deb \
        "https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_DISTRO}/x86_64/cuda-keyring_1.1-1_all.deb"; \
      dpkg -i /tmp/cuda-keyring.deb && rm /tmp/cuda-keyring.deb; \
      apt-get update -qq; \
      apt-get install -y -qq --no-install-recommends \
        cuda-nvcc-${CUDA_PKG_VERSION} \
        cuda-cudart-dev-${CUDA_PKG_VERSION} \
        libcurand-dev-${CUDA_PKG_VERSION}; \
      ln -sfn /usr/local/cuda-${CUDA_HOME_VERSION} /usr/local/cuda; \
      rm -rf /var/lib/apt/lists/*; \
      /usr/local/cuda/bin/nvcc --version; \
    else \
      echo "ENABLE_CUDA=${ENABLE_CUDA}: skipping CUDA toolkit (CPU backend only)"; \
    fi
# LD_LIBRARY_PATH is unset on the base image; assign it outright (no ${...:-}
# append) to avoid a trailing-colon CWD entry. The entrypoint's setup.bash later
# prepends the ROS lib paths, leaving the CUDA libs resolvable at the end.
ENV PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64

# ── ouster-ros (jcfurey fork, pinned commit, with ouster-sdk submodule) ───────
# Clone the ros2 branch specifically (its layout nests ouster-ros/ouster-sdk);
# a full clone keeps the pinned commit reachable, then resync submodules to it.
RUN git clone --branch ros2 --recurse-submodules "${OUSTER_ROS_REPO}" src/ouster-ros \
 && git -C src/ouster-ros checkout "${OUSTER_ROS_REF}" \
 && git -C src/ouster-ros submodule sync --recursive \
 && git -C src/ouster-ros submodule update --init --recursive

# ── TurtleBot3 waffle description (geometry only) ─────────────────────────────
# Clone the whole repo to a temp dir but copy ONLY turtlebot3_description into
# the build workspace, so rosdep/colcon don't drag in the dynamixel/hardware
# packages the rest of the repo needs.
RUN git clone --branch "${TURTLEBOT3_BRANCH}" "${TURTLEBOT3_REPO}" /tmp/turtlebot3 \
 && git -C /tmp/turtlebot3 checkout "${TURTLEBOT3_REF}" \
 && cp -r /tmp/turtlebot3/turtlebot3_description src/turtlebot3_description \
 && rm -rf /tmp/turtlebot3

# ── gz_sensors_ouster (this package) ──────────────────────────────────────────
COPY . src/gz_sensors_ouster

# ── system deps via rosdep ────────────────────────────────────────────────────
# `rosdep update` pulls the rosdistro index from raw.githubusercontent.com, which
# intermittently resets the connection; retry a few times so a flaky network
# doesn't fail the whole build.
RUN for i in 1 2 3 4 5; do \
      rosdep update --rosdistro=${ROS_DISTRO} && break; \
      echo "rosdep update failed (attempt $i); retrying in 10s..."; sleep 10; \
    done \
 && apt-get update -qq \
 && rosdep install --from-paths src --rosdistro=${ROS_DISTRO} -y --ignore-src \
 && rm -rf /var/lib/apt/lists/*

# ── build (ouster_sensor_msgs -> ouster_ros -> turtlebot3_description ->
#          gz_sensors_ouster). -DBUILD_TESTING=ON is required: the gtest targets
#          are guarded by if(BUILD_TESTING). ──────────────────────────────────
# When ENABLE_CUDA=true, nvcc is on PATH so the package's check_language(CUDA)
# enables the CUDA backend automatically; CMAKE_CUDA_ARCHITECTURES targets the
# GPU(s) you pass via CUDA_ARCH. With CUDA off the flag is inert (no CUDA lang).
RUN source /opt/ros/${ROS_DISTRO}/setup.bash \
 && colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"

# ── build-time test gate (mirror CI: fail if zero tests ran) ──────────────────
RUN source /opt/ros/${ROS_DISTRO}/setup.bash \
 && source install/setup.bash \
 && colcon test --packages-select gz_sensors_ouster --event-handlers console_direct+ \
 && colcon test-result --verbose --all \
 && summary="$(colcon test-result --all | grep -E '^Summary:' || true)" \
 && n="$(printf '%s' "$summary" | grep -oE '[0-9]+ test' | grep -oE '[0-9]+' | head -1)" \
 && { [ -n "$n" ] && [ "$n" -gt 0 ]; } || { echo "ERROR: no tests ran (BUILD_TESTING wiring broken)"; exit 1; } \
 && echo "Executed $n test(s)."

ENTRYPOINT ["/ws/src/gz_sensors_ouster/docker/entrypoint.sh"]
CMD ["smoke"]
