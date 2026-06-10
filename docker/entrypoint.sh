#!/usr/bin/env bash
# Entrypoint for the gz_sensors_ouster standalone test image.
#
#   smoke   (default) headless, GPU-free: launch the TurtleBot3+Ouster sim in
#           raycast mode, assert a PointCloud2 appears on
#           /sensor/lidar/lidar0/points, then exit PASS/FAIL.
#   drive   interactive: launch the sim with the gz GUI + RViz-able topics and
#           run teleop_twist_keyboard on /cmd_vel (needs -it and a display).
#   test    re-run the gtest suite.
#   bash    drop into a shell.
#
# Note: no `set -u` — the ROS/ament setup scripts reference unset variables
# (e.g. AMENT_TRACE_SETUP_FILES) and would abort under nounset.
set -eo pipefail

source "/opt/ros/${ROS_DISTRO}/setup.bash"
source /ws/install/setup.bash

cmd="${1:-smoke}"
shift || true

case "$cmd" in
  smoke)
    echo "[entrypoint] headless raycast smoke: launching sim..."
    # setsid makes the launch a process-group leader so the group kill below
    # reliably reaps the whole gz/ros2 tree, however we leave.
    setsid ros2 launch gz_sensors_ouster turtlebot3_ouster.launch.py \
      headless:=true ray_mode:=raycast rviz:=false &
    launch_pid=$!
    trap 'kill -- "-$launch_pid" 2>/dev/null || kill "$launch_pid" 2>/dev/null || true' EXIT

    set +e
    python3 "/ws/src/gz_sensors_ouster/docker/smoke_check.py" \
      --topic /sensor/lidar/lidar0/points --timeout "${SMOKE_TIMEOUT:-120}"
    rc=$?
    set -e
    exit "$rc"
    ;;

  drive|gui)
    echo "[entrypoint] interactive sim (raycast)."
    # Both bring up RViz by default; disable with `-e RVIZ=false` (note: docker
    # -e flags go BEFORE the image name). drive also runs teleop in the
    # foreground; gui just shows the windows.
    setsid ros2 launch gz_sensors_ouster turtlebot3_ouster.launch.py \
      headless:=false ray_mode:="${RAY_MODE:-raycast}" rviz:="${RVIZ:-true}" &
    launch_pid=$!
    trap 'kill -- "-$launch_pid" 2>/dev/null || kill "$launch_pid" 2>/dev/null || true' EXIT
    sleep 5
    if [ "$cmd" = "drive" ]; then
      echo "[entrypoint] drive with the teleop keys."
      ros2 run teleop_twist_keyboard teleop_twist_keyboard
    else
      wait "$launch_pid"
    fi
    ;;

  test)
    cd /ws
    colcon test --packages-select gz_sensors_ouster --event-handlers console_direct+
    colcon test-result --verbose --all
    ;;

  bash|sh)
    exec bash
    ;;

  *)
    # Anything else: treat as a command to run inside the sourced environment.
    exec "$cmd" "$@"
    ;;
esac
