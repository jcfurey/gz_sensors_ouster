cmake_minimum_required(VERSION 3.18)
project(gz_sensors_ouster VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# -Wall and -Wextra apply to both CXX and CUDA host compilation.
add_compile_options(-Wall -Wextra)

# -Wpedantic is scoped to CXX only. Reason: NVCC's cudafe1 pass generates
# tmpxft_*.stub.c intermediate files whose `# N "file"` line directives use
# GCC's extension form rather than the C-standard `#line N "file"`. GCC's
# -Wpedantic flags this, and unlike Clang (-Wno-gnu-line-marker) there is no
# sub-flag to silence just this one pedantic warning. Applying -Wpedantic to
# the CUDA host-compile pass therefore only ever flags NVCC's own output —
# never our code. Scoping it to CXX keeps every line we actually write under
# full -Wpedantic, while avoiding ~100 spurious warnings from NVCC stubs.
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>)

# ── Dependencies ──────────────────────────────────────────────────────────────

# ament_cmake_auto discovers every build/buildtool/run dependency declared in
# package.xml (rclcpp, the sensor/std msgs, ouster_sensor_msgs, ouster_ros and
# the gz_*_vendor packages) and wires their include dirs + link libs into any
# target created with ament_auto_add_library below. This finds the gz_*_vendor
# packages, which is what installs the version-agnostic gz CMake shims used
# just after.
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# Version-agnostic Gazebo discovery. Finding the gz_*_vendor packages (done
# above by ament_auto_find_build_dependencies) installs CMake shims that let us
# find_package the underlying gz libraries WITHOUT a version suffix and link
# their targets without a version in the name. The SAME CMakeLists therefore
# builds against whichever Gazebo the active ROS distro vendors:
#   ROS Jazzy   → Gazebo Harmonic (gz-sim 8)
#   ROS Kilted  → Gazebo Ionic    (gz-sim 9)
#   ROS Lyrical → Gazebo Jetty    (gz-sim 10)
# (Set GZ_RELAX_VERSION_MATCH=1 to let a vendor build a different Gazebo from
# source — e.g. Jetty on Jazzy — matching just the major version.)
find_package(gz-sim REQUIRED)
find_package(gz-rendering REQUIRED)
find_package(gz-sensors REQUIRED COMPONENTS gpu_lidar)
find_package(gz-plugin REQUIRED COMPONENTS register)

# Eigen's CMake package is named Eigen3 (the package.xml rosdep key is `eigen`),
# so it is found explicitly rather than via ament_auto.
find_package(Eigen3 REQUIRED)

# GPU backends (all optional). Each probes for its toolchain; missing
# toolchains just disable that backend and fall through to the next at
# runtime. At least one is always compiled in — the CPU backend.
#
# User-overridable: pass -DGZ_GPU_OUSTER_USE_{CUDA,HIP,SYCL}=ON/OFF on the
# cmake command line to force a backend on or off regardless of the probe.
include(CheckLanguage)

# ── CUDA (NVIDIA) ────────────────────────────────────────────────────────────
# check_language(CUDA) skips re-probing when CMAKE_CUDA_COMPILER is already
# cached — so a stale path (e.g. /usr/bin/nvcc from an older container image
# where nvcc was symlinked there) survives reconfigures and then trips
# enable_language(CUDA) with "not a full path to an existing compiler tool".
# Even when the compiler path is corrected, cached CUDA_*_LIBRARY entries
# from the previous toolkit install (e.g. /usr/lib/x86_64-linux-gnu/libcurand.so
# on an earlier Debian-layout image) survive and break the link step with
# "No rule to make target …libcurand.so".
# Guard: when the cached compiler path no longer exists, the whole CUDA
# cache is from a different toolkit and untrustworthy — wipe every CUDA_*
# and CUDAToolkit_* entry, then re-probe.
check_language(CUDA)
if(CMAKE_CUDA_COMPILER AND NOT EXISTS "${CMAKE_CUDA_COMPILER}")
  message(STATUS
    "${PROJECT_NAME}: cached CMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER} "
    "does not exist; clearing CUDA cache and re-probing")
  unset(CMAKE_CUDA_COMPILER CACHE)
  get_cmake_property(_cuda_cache_vars CACHE_VARIABLES)
  foreach(_v IN LISTS _cuda_cache_vars)
    if(_v MATCHES "^CUDA_|^CUDAToolkit")
      unset(${_v} CACHE)
    endif()
  endforeach()
  unset(_cuda_cache_vars)
  check_language(CUDA)
endif()
set(GZ_GPU_OUSTER_USE_CUDA OFF CACHE BOOL "Build the CUDA backend for NVIDIA GPUs")
if(CMAKE_CUDA_COMPILER AND EXISTS "${CMAKE_CUDA_COMPILER}")
  enable_language(CUDA)
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    set(GZ_GPU_OUSTER_USE_CUDA ON)
    message(STATUS "${PROJECT_NAME}: CUDA backend enabled (nvcc + curand)")
  else()
    message(STATUS "${PROJECT_NAME}: nvcc found but CUDAToolkit missing; CUDA backend disabled")
  endif()
else()
  message(STATUS "${PROJECT_NAME}: no CUDA compiler; CUDA backend disabled")
endif()

# ── HIP (AMD, including APUs via ROCm) ──────────────────────────────────────
check_language(HIP)
set(GZ_GPU_OUSTER_USE_HIP OFF CACHE BOOL "Build the HIP backend for AMD GPUs (incl. APUs)")
if(CMAKE_HIP_COMPILER)
  set(GZ_GPU_OUSTER_USE_HIP ON)
  message(STATUS "${PROJECT_NAME}: HIP backend enabled (${CMAKE_HIP_COMPILER})")
else()
  message(STATUS "${PROJECT_NAME}: no HIP compiler; HIP backend disabled")
endif()

# ── SYCL (Intel iGPU + Arc via oneAPI DPC++ / AdaptiveCpp) ──────────────────
# Enable automatically when the compiler identifies as IntelLLVM (icpx) or
# when SYCL is explicitly requested via -DGZ_GPU_OUSTER_USE_SYCL=ON (e.g.
# for AdaptiveCpp builds where the compiler ID is Clang).
set(GZ_GPU_OUSTER_USE_SYCL OFF CACHE BOOL "Build the SYCL backend for Intel GPUs")
if(NOT GZ_GPU_OUSTER_USE_SYCL AND CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
  set(GZ_GPU_OUSTER_USE_SYCL ON)
  message(STATUS "${PROJECT_NAME}: SYCL backend enabled (icpx detected)")
elseif(GZ_GPU_OUSTER_USE_SYCL)
  message(STATUS "${PROJECT_NAME}: SYCL backend enabled (user-requested)")
else()
  message(STATUS "${PROJECT_NAME}: SYCL backend disabled")
endif()

if(NOT GZ_GPU_OUSTER_USE_CUDA AND NOT GZ_GPU_OUSTER_USE_HIP AND NOT GZ_GPU_OUSTER_USE_SYCL)
  message(STATUS "${PROJECT_NAME}: no GPU backend enabled; CPU fallback only")
endif()

# ── Ouster SDK headers ────────────────────────────────────────────────────────
# Headers (<ouster/types.h>, <ouster/metadata.h>, <ouster/impl/packet_writer.h>)
# are propagated by ament_target_dependencies(ouster_ros) below — upstream
# ouster_ros installs ouster-sdk/ouster_client/include/ouster to its install
# prefix and re-exports it via ament_export_include_directories(include). No
# explicit path manipulation needed; works against either an apt-installed
# ros-${ROS_DISTRO}-ouster-ros or a workspace colcon build of the submodule.

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
# ament_auto_add_library auto-links the ament dependencies found above
# (rclcpp, the msg packages, ouster_ros, the gz_*_vendor packages) and adds
# include/ to the target's public include path.
ament_auto_add_library(${PROJECT_NAME} SHARED
  src/gz_gpu_ouster_lidar_system.cpp
)

# Link the non-ament targets. ament_auto_add_library already linked the ROS
# deps using target_link_libraries' *plain* signature (via
# ament_target_dependencies), so these must use the plain signature too —
# CMake forbids mixing plain and keyword (PUBLIC/PRIVATE) forms on one target.
#
# The gz/Eigen/OpenMP imported targets are safe in the plain (interface) list.
# The in-tree static lib gz_gpu_ouster_lidar_cuda is wrapped in
# $<BUILD_INTERFACE:...> so it is linked into the plugin .so at build time but
# kept out of the install/export interface — otherwise ament_auto_package's
# install(EXPORT) fails ("requires target gz_gpu_ouster_lidar_cuda that is not
# in any export set"). The plugin is dlopen'd and never linked against, so it
# needs no link interface anyway.
target_link_libraries(${PROJECT_NAME}
  gz-sim::gz-sim
  gz-rendering::gz-rendering
  gz-sensors::gz-sensors
  gz-plugin::register
  Eigen3::Eigen
  $<BUILD_INTERFACE:gz_gpu_ouster_lidar_cuda>
  $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
)

# ── Tests ────────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  # Test: noise model (CPU fallback path)
  ament_add_gtest(test_noise_model test/test_noise_model.cpp)
  target_link_libraries(test_noise_model gz_gpu_ouster_lidar_cuda)

  # Test: parameter validation. ament_auto_add_gtest links the package's ROS
  # deps (rclcpp, ouster_sensor_msgs, ouster_ros) and adds include/ automatically,
  # replacing ament_target_dependencies — which ROS 2 Lyrical (the L-turtle)
  # removed. Works identically on Jazzy/Kilted/Lyrical.
  ament_auto_add_gtest(test_parameter_validation test/test_parameter_validation.cpp)
  target_link_libraries(test_parameter_validation gz_gpu_ouster_lidar_cuda)

  # Test: metadata parsing
  set(_test_metadata_dir "${CMAKE_CURRENT_SOURCE_DIR}/config/metadata")
  if(NOT IS_DIRECTORY "${_test_metadata_dir}")
    message(FATAL_ERROR
      "Test metadata directory missing: ${_test_metadata_dir}\n"
      "The test_metadata_parsing target requires the example Ouster metadata "
      "JSONs shipped in config/metadata/. Either restore them or disable "
      "testing with -DBUILD_TESTING=OFF.")
  endif()
  # ament_auto_add_gtest auto-links ouster_ros (no ament_target_dependencies,
  # which Lyrical removed).
  ament_auto_add_gtest(test_metadata_parsing test/test_metadata_parsing.cpp)
  target_compile_definitions(test_metadata_parsing PRIVATE
    TEST_METADATA_DIR="${_test_metadata_dir}"
  )

  # Test: bilinear resampling (CPU fallback)
  ament_add_gtest(test_resample test/test_resample.cpp)
  target_link_libraries(test_resample gz_gpu_ouster_lidar_cuda)

  # Test: backend dispatch / selection (GZ_OUSTER_BACKEND override, auto
  # fallback, backendName/usesCpuFallback, processRaw through the wrapper)
  ament_add_gtest(test_dispatch test/test_dispatch.cpp)
  target_link_libraries(test_dispatch gz_gpu_ouster_lidar_cuda)

  # Test: full per-beam raycast mode (intersectors, BVH vs brute force,
  # beam-origin parallax, retro, near clip, processDepth)
  ament_add_gtest(test_raycast test/test_raycast.cpp)
  target_link_libraries(test_raycast gz_gpu_ouster_lidar_cuda)

  # Test: IMU noise model (CPU-only, no Gazebo dep)
  ament_add_gtest(test_imu_noise test/test_imu_noise.cpp)
  target_link_libraries(test_imu_noise gz_gpu_ouster_lidar_cuda)

  # Test: plugin lifecycle smoke (links against the full plugin .so so a
  # Gazebo schema bump that breaks the vtable surfaces at test build time)
  ament_add_gtest(test_lifecycle test/test_lifecycle.cpp)
  target_link_libraries(test_lifecycle ${PROJECT_NAME})
endif()

# Installs the plugin library + include/ headers, exports the ament package,
# and installs the runtime config (metadata JSONs + example SDF) to
# share/${PROJECT_NAME}/config plus the example URDFs / world / launch files to
# share/${PROJECT_NAME}/examples.
ament_auto_package(INSTALL_TO_SHARE config examples)
