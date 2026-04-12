cmake_minimum_required(VERSION 3.18)
project(gz_gpu_ouster_lidar)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-Wall -Wextra -Wpedantic)

# ── Dependencies ──────────────────────────────────────────────────────────────

find_package(ament_cmake REQUIRED)

# Gazebo vendor packages (must be found BEFORE the actual gz-* packages;
# the vendor extras append to CMAKE_PREFIX_PATH).
find_package(gz_sim_vendor REQUIRED)
find_package(gz_rendering_vendor REQUIRED)
find_package(gz_sensors_vendor REQUIRED)
find_package(gz_plugin_vendor REQUIRED)

# Actual Gazebo libraries
find_package(gz-sim8 REQUIRED)
find_package(gz-rendering8 REQUIRED)
find_package(gz-sensors8 REQUIRED COMPONENTS gpu_lidar)
find_package(gz-plugin2 REQUIRED COMPONENTS register)

# ROS 2
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(ouster_sensor_msgs REQUIRED)
find_package(ouster_ros REQUIRED)
find_package(Eigen3 REQUIRED)

# CUDA (optional): build with CUDA when nvcc/toolkit are available,
# otherwise fall back to a CPU implementation with the same interface.
include(CheckLanguage)
check_language(CUDA)
set(GZ_GPU_OUSTER_USE_CUDA OFF)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    set(GZ_GPU_OUSTER_USE_CUDA ON)
    message(STATUS "${PROJECT_NAME}: CUDA enabled")
  else()
    message(WARNING "${PROJECT_NAME}: nvcc found but CUDAToolkit was not found; using CPU fallback")
  endif()
else()
  message(WARNING "${PROJECT_NAME}: CUDA compiler not found; using CPU fallback")
endif()

# ── Ouster SDK headers (same pattern as gz_ouster_packet_bridge) ──────────────
set(_ouster_sdk_src "${CMAKE_CURRENT_SOURCE_DIR}/../../drivers/ouster-ros/ouster-ros/ouster-sdk")
if(NOT EXISTS "${_ouster_sdk_src}/ouster_client/include")
  message(FATAL_ERROR
    "Ouster SDK not found at ${_ouster_sdk_src}. "
    "Ensure ouster-ros submodule is initialized: git submodule update --init --recursive")
endif()

# OpenMP (optional): parallelise CPU-side bilinear resampling for high-density
# sensors.  Falls back to single-threaded if not found.
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  message(STATUS "${PROJECT_NAME}: OpenMP enabled (beam resampling parallelised)")
else()
  message(STATUS "${PROJECT_NAME}: OpenMP not found; beam resampling single-threaded")
endif()

# ── CUDA static library ──────────────────────────────────────────────────────
add_subdirectory(cuda)

# ── Gazebo system plugin ─────────────────────────────────────────────────────
add_library(${PROJECT_NAME} SHARED
  src/gz_gpu_ouster_lidar_system.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE
  "${_ouster_sdk_src}/ouster_client/include"
)

target_link_libraries(${PROJECT_NAME}
  gz-sim8::gz-sim8
  gz-rendering8::gz-rendering8
  gz-sensors8::gz-sensors8
  gz-plugin2::register
  Eigen3::Eigen
  gz_gpu_ouster_lidar_cuda
  $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
)

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  sensor_msgs
  std_msgs
  ouster_sensor_msgs
  ouster_ros
)

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)

ament_package()
