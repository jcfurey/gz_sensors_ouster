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
    ros2 launch gz_sensors_ouster turtlebot3_ouster.launch.py \
      headless:=true ray_mode:=raycast rviz:=false &
    launch_pid=$!
    # Kill the whole launch process group on exit, however we leave.
    trap 'kill -- -'"$launch_pid"' 2>/dev/null || kill "$launch_pid" 2>/dev/null || true' EXIT

    set +e
    python3 "/ws/src/gz_sensors_ouster/docker/smoke_check.py" \
      --topic /sensor/lidar/lidar0/points --timeout "${SMOKE_TIMEOUT:-120}"
    rc=$?
    set -e
    exit "$rc"
    ;;

  drive|gui)
    echo "[entrypoint] interactive sim (raycast). Drive with the teleop keys."
    ros2 launch gz_sensors_ouster turtlebot3_ouster.launch.py \
      headless:=false ray_mode:="${RAY_MODE:-raycast}" rviz:="${RVIZ:-false}" &
    launch_pid=$!
    trap 'kill -- -'"$launch_pid"' 2>/dev/null || kill "$launch_pid" 2>/dev/null || true' EXIT
    sleep 5
    if [ "$cmd" = "drive" ]; then
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
