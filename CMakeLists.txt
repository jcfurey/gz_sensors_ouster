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
find_package(gz-common REQUIRED COMPONENTS graphics)
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
# User-overridable: each GZ_GPU_OUSTER_USE_{CUDA,HIP,SYCL} option is tri-state
# AUTO (default, probe the toolchain) | ON (force on; configure fails if the
# toolchain is missing) | OFF (force off; the toolchain probe is skipped
# entirely, so OFF also dodges a broken or hanging toolchain). Values are
# case-insensitive; ON/OFF accept the usual CMake boolean spellings.
include(CheckLanguage)

# Record whether the USER chose CUDA architectures BEFORE any CUDA language
# enablement: enable_language(CUDA) auto-populates CMAKE_CUDA_ARCHITECTURES
# (CMP0104) with the toolchain default when unset, which would make any later
# if(NOT CMAKE_CUDA_ARCHITECTURES) check dead code. Cached (INTERNAL) so the
# first-configure answer survives reconfigures of the same build dir, where
# the auto-populated cache entry would otherwise masquerade as a user choice.
# (To change arches later in an EXISTING build dir, pass
# -DCMAKE_CUDA_ARCHITECTURES=... together with
# -DGZ_OUSTER_CUDA_ARCHS_USER_SET=ON, or start from a fresh build dir.)
if(NOT DEFINED GZ_OUSTER_CUDA_ARCHS_USER_SET)
  if(DEFINED CMAKE_CUDA_ARCHITECTURES OR DEFINED ENV{CUDAARCHS})
    set(GZ_OUSTER_CUDA_ARCHS_USER_SET ON CACHE INTERNAL
        "CMAKE_CUDA_ARCHITECTURES/CUDAARCHS was user-provided on first configure")
  else()
    set(GZ_OUSTER_CUDA_ARCHS_USER_SET OFF CACHE INTERNAL
        "CMAKE_CUDA_ARCHITECTURES/CUDAARCHS was user-provided on first configure")
  endif()
endif()

# Tri-state option declaration, legacy-cache migration and normalization.
#
# Migration: build dirs configured before the tri-state rework hold a legacy
# BOOL cache entry — the old code wrote `set(... OFF CACHE BOOL)`
# unconditionally and shadowed it with a normal variable on a successful
# probe — and `set(... CACHE ...)` does not overwrite an existing entry, so
# that stale OFF would masquerade as an explicit user override and silently
# disable a previously auto-enabled backend. A BOOL-typed entry can only come
# from the old scheme (plain -DVAR=... command-line entries are UNINITIALIZED),
# so it is safe to migrate to AUTO.
#
# Normalization: sets ${VAR}_MODE to exactly AUTO, ON or OFF. Case variants
# and the usual boolean spellings are accepted; anything else is a configure
# error (previously any unrecognized string — e.g. lowercase 'auto' — fell
# through if()'s truthiness and force-enabled the backend).
macro(gz_ouster_backend_option VAR DOC)
  set(${VAR} AUTO CACHE STRING "${DOC}")
  set_property(CACHE ${VAR} PROPERTY STRINGS AUTO ON OFF)
  string(TOUPPER "${${VAR}}" _gz_ouster_opt_norm)
  if(_gz_ouster_opt_norm STREQUAL "AUTO")
    set(${VAR}_MODE AUTO)
  elseif(_gz_ouster_opt_norm MATCHES "^(ON|TRUE|YES|Y|1)$")
    set(${VAR}_MODE ON)
  elseif(_gz_ouster_opt_norm MATCHES "^(OFF|FALSE|NO|N|0|)$")
    set(${VAR}_MODE OFF)
  else()
    message(FATAL_ERROR
      "${PROJECT_NAME}: ${VAR}='${${VAR}}' is not one of AUTO | ON | OFF")
  endif()
  unset(_gz_ouster_opt_norm)
endmacro()

gz_ouster_backend_option(GZ_GPU_OUSTER_USE_CUDA
    "CUDA (NVIDIA) backend: AUTO (probe) | ON (force) | OFF (disable)")
gz_ouster_backend_option(GZ_GPU_OUSTER_USE_HIP
    "HIP (AMD, incl. APUs) backend: AUTO (probe) | ON (force) | OFF (disable)")
gz_ouster_backend_option(GZ_GPU_OUSTER_USE_SYCL
    "SYCL (Intel) backend: AUTO (icpx auto-detect) | ON (force) | OFF (disable)")

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
# The whole probe (check_language + stale-cache cleanup) is skipped when the
# user forced OFF — a broken half-installed toolchain then costs nothing at
# configure time. The resolved decision is exported to cuda/CMakeLists.txt as
# GZ_GPU_OUSTER_ENABLE_<BACKEND>.
set(GZ_GPU_OUSTER_ENABLE_CUDA OFF)
if(NOT GZ_GPU_OUSTER_USE_CUDA_MODE STREQUAL "OFF")
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
endif()
if(GZ_GPU_OUSTER_USE_CUDA_MODE STREQUAL "AUTO")
  if(CMAKE_CUDA_COMPILER AND EXISTS "${CMAKE_CUDA_COMPILER}")
    enable_language(CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
      set(GZ_GPU_OUSTER_ENABLE_CUDA ON)
      message(STATUS "${PROJECT_NAME}: CUDA backend auto-enabled (nvcc + curand)")
    else()
      message(STATUS "${PROJECT_NAME}: nvcc found but CUDAToolkit missing; CUDA backend disabled")
    endif()
  else()
    message(STATUS "${PROJECT_NAME}: no CUDA compiler; CUDA backend disabled (auto)")
  endif()
elseif(GZ_GPU_OUSTER_USE_CUDA_MODE STREQUAL "ON")
  if(NOT (CMAKE_CUDA_COMPILER AND EXISTS "${CMAKE_CUDA_COMPILER}"))
    message(FATAL_ERROR
      "${PROJECT_NAME}: GZ_GPU_OUSTER_USE_CUDA="
      "'${GZ_GPU_OUSTER_USE_CUDA}' forces the CUDA backend on, but no CUDA "
      "compiler was found")
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(GZ_GPU_OUSTER_ENABLE_CUDA ON)
  message(STATUS "${PROJECT_NAME}: CUDA backend force-enabled (user override)")
else()  # OFF
  message(STATUS
    "${PROJECT_NAME}: CUDA backend disabled (user override; probe skipped)")
endif()

# ── HIP (AMD, including APUs via ROCm) ──────────────────────────────────────
set(GZ_GPU_OUSTER_ENABLE_HIP OFF)
if(NOT GZ_GPU_OUSTER_USE_HIP_MODE STREQUAL "OFF")
  check_language(HIP)
endif()
if(GZ_GPU_OUSTER_USE_HIP_MODE STREQUAL "AUTO")
  if(CMAKE_HIP_COMPILER)
    set(GZ_GPU_OUSTER_ENABLE_HIP ON)
    message(STATUS "${PROJECT_NAME}: HIP backend auto-enabled (${CMAKE_HIP_COMPILER})")
  else()
    message(STATUS "${PROJECT_NAME}: no HIP compiler; HIP backend disabled (auto)")
  endif()
elseif(GZ_GPU_OUSTER_USE_HIP_MODE STREQUAL "ON")
  if(NOT CMAKE_HIP_COMPILER)
    message(FATAL_ERROR
      "${PROJECT_NAME}: GZ_GPU_OUSTER_USE_HIP="
      "'${GZ_GPU_OUSTER_USE_HIP}' forces the HIP backend on, but no HIP "
      "compiler was found")
  endif()
  set(GZ_GPU_OUSTER_ENABLE_HIP ON)
  message(STATUS "${PROJECT_NAME}: HIP backend force-enabled (user override)")
else()  # OFF
  message(STATUS
    "${PROJECT_NAME}: HIP backend disabled (user override; probe skipped)")
endif()

# ── SYCL (Intel iGPU + Arc via oneAPI DPC++ / AdaptiveCpp) ──────────────────
# AUTO enables the backend when the compiler identifies as IntelLLVM (icpx);
# ON forces it on (e.g. for AdaptiveCpp builds, whose driver may identify as
# Clang or GNU depending on backend — so ON is deliberately not validated
# against a compiler-ID allowlist), OFF disables it even under icpx.
set(GZ_GPU_OUSTER_ENABLE_SYCL OFF)
if(GZ_GPU_OUSTER_USE_SYCL_MODE STREQUAL "AUTO")
  if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    set(GZ_GPU_OUSTER_ENABLE_SYCL ON)
    message(STATUS "${PROJECT_NAME}: SYCL backend auto-enabled (icpx detected)")
  else()
    message(STATUS "${PROJECT_NAME}: SYCL backend disabled "
                   "(auto; needs icpx or -DGZ_GPU_OUSTER_USE_SYCL=ON)")
  endif()
elseif(GZ_GPU_OUSTER_USE_SYCL_MODE STREQUAL "ON")
  set(GZ_GPU_OUSTER_ENABLE_SYCL ON)
  message(STATUS "${PROJECT_NAME}: SYCL backend force-enabled (user override)")
else()  # OFF
  message(STATUS "${PROJECT_NAME}: SYCL backend disabled (user override)")
endif()

if(NOT GZ_GPU_OUSTER_ENABLE_CUDA AND NOT GZ_GPU_OUSTER_ENABLE_HIP AND NOT GZ_GPU_OUSTER_ENABLE_SYCL)
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

# Canonical ROS-free contracts; ouster_ros supplies the SDK in this process.
set(OUSTER_SIM_CORE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ouster_sim_core")
if(NOT EXISTS "${OUSTER_SIM_CORE_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "Initialize ouster_sim_core: git submodule update --init --recursive")
endif()
add_library(${PROJECT_NAME}_ouster_sdk INTERFACE)
target_include_directories(${PROJECT_NAME}_ouster_sdk INTERFACE ${ouster_ros_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}_ouster_sdk INTERFACE ${ouster_ros_LIBRARIES} Eigen3::Eigen)
set(OUSTER_SIM_CORE_OUSTER_SDK_TARGET ${PROJECT_NAME}_ouster_sdk)
add_subdirectory("${OUSTER_SIM_CORE_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/ouster_sim_core")

# ── CUDA static library ──────────────────────────────────────────────────────
add_subdirectory(cuda)

# Public ROS facade: engine-independent contracts live in ouster_sim_core;
# this target owns ROS conversion, thread lifetime and complete-frame delivery.
ament_auto_add_library(${PROJECT_NAME}_core SHARED
  src/ouster_lidar_profile.cpp
  src/ouster_metadata.cpp
  src/packet_encoder.cpp
  src/ros_interface.cpp
)
target_include_directories(${PROJECT_NAME}_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}/core>
)
# Embed the complete API in the parent library, as the AGX facade does.
# Build-only linkage keeps the static/provider targets out of ament exports.
target_include_directories(${PROJECT_NAME}_core PUBLIC
  $<BUILD_INTERFACE:${OUSTER_SIM_CORE_SOURCE_DIR}/include>)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_link_libraries(${PROJECT_NAME}_core Eigen3::Eigen
    $<BUILD_INTERFACE:-Wl,--whole-archive>
    $<BUILD_INTERFACE:ouster_sim_core>
    $<BUILD_INTERFACE:-Wl,--no-whole-archive>)
else()
  target_link_libraries(${PROJECT_NAME}_core Eigen3::Eigen
    $<BUILD_INTERFACE:ouster_sim_core>)
endif()

# ── Gazebo system plugin ─────────────────────────────────────────────────────
# ament_auto_add_library auto-links the ament dependencies found above
# (rclcpp, the msg packages, ouster_ros, the gz_*_vendor packages) and adds
# include/ to the target's public include path.
ament_auto_add_library(${PROJECT_NAME} SHARED
  src/gz_gpu_ouster_lidar_system.cpp
  src/obscurants.cpp
  src/panel_rig.cpp
  src/raycast_mirror.cpp
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
  gz-common::graphics
  gz-rendering::gz-rendering
  gz-sensors::gz-sensors
  gz-plugin::register
  Eigen3::Eigen
  ${PROJECT_NAME}_core
  $<BUILD_INTERFACE:gz_gpu_ouster_lidar_cuda>
  $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
)

# ── Tests ────────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)

  # The same fixture sources run against both simulator SDK providers.
  foreach(contract revolution_assembler optical_channel_model packet_pacing
                   product_profile metadata_product_profile packet_roundtrip)
    ament_auto_add_gtest(test_shared_${contract}
      "${OUSTER_SIM_CORE_SOURCE_DIR}/test/test_${contract}.cpp")
    target_compile_definitions(test_shared_${contract} PRIVATE
      OUSTER_SIM_CORE_TEST_DATA_DIR="${OUSTER_SIM_CORE_SOURCE_DIR}/test/data")
    target_link_libraries(test_shared_${contract} ${PROJECT_NAME}_core)
  endforeach()
  ament_auto_add_gtest(test_core_conformance test/test_core_conformance.cpp TIMEOUT 30)
  target_include_directories(test_core_conformance PRIVATE "${OUSTER_SIM_CORE_SOURCE_DIR}/test/support")
  target_compile_definitions(test_core_conformance PRIVATE
    GZ_TEST_METADATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/config/metadata"
    OUSTER_SIM_CORE_TEST_DATA_DIR="${OUSTER_SIM_CORE_SOURCE_DIR}/test/data")
  target_link_libraries(test_core_conformance ${PROJECT_NAME}_core)
  ament_add_gtest(test_optical_conformance test/test_optical_conformance.cpp)
  target_include_directories(test_optical_conformance PRIVATE "${OUSTER_SIM_CORE_SOURCE_DIR}/test/support")
  target_link_libraries(test_optical_conformance gz_gpu_ouster_lidar_cuda ${PROJECT_NAME}_core)

  find_package(rmw_zenoh_cpp QUIET)
  if(rmw_zenoh_cpp_FOUND)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    ament_add_gtest_executable(test_zenoh_packet_burst test/test_zenoh_packet_burst.cpp)
    target_link_libraries(test_zenoh_packet_burst ${PROJECT_NAME}_core)
    ament_add_test(test_zenoh_packet_burst
      COMMAND "${Python3_EXECUTABLE}"
        "${OUSTER_SIM_CORE_SOURCE_DIR}/test/support/run_zenoh_packet_burst.py"
        "$<TARGET_FILE:test_zenoh_packet_burst>"
        "--gtest_output=xml:${CMAKE_CURRENT_BINARY_DIR}/test_results/${PROJECT_NAME}/test_zenoh_packet_burst.gtest.xml"
      RESULT_FILE "${CMAKE_CURRENT_BINARY_DIR}/test_results/${PROJECT_NAME}/test_zenoh_packet_burst.gtest.xml"
      TIMEOUT 40)
    set_tests_properties(test_zenoh_packet_burst PROPERTIES LABELS "integration;zenoh")
  endif()

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

  # Test: generation/revision/model physics profile catalog and inference.
  ament_add_gtest(test_ouster_lidar_profile
    test/test_ouster_lidar_profile.cpp)
  target_link_libraries(test_ouster_lidar_profile ${PROJECT_NAME}_core)
  target_include_directories(test_ouster_lidar_profile PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)

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

  # Test: participating media (smoke / dust / fog) — volume geometry,
  # Beer-Lambert extinction, medium backscatter, NEAR_IR airlight
  ament_add_gtest(test_obscurants test/test_obscurants.cpp)
  target_link_libraries(test_obscurants gz_gpu_ouster_lidar_cuda)

  # Test: ECM/SDF -> obscurant conversion (particle emitters, authored
  # volumes, nearest-N selection). Compiles src/obscurants.cpp directly so
  # the helpers stay reachable without exporting them from the plugin .so.
  ament_auto_add_gtest(test_obscurant_config
    test/test_obscurant_config.cpp src/obscurants.cpp)
  target_link_libraries(test_obscurant_config
    gz_gpu_ouster_lidar_cuda gz-sim::gz-sim)
  target_include_directories(test_obscurant_config PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)

  # Test: scan-frame triple buffer (header-only, no deps beyond the STL)
  ament_add_gtest(test_frame_exchange test/test_frame_exchange.cpp)
  target_include_directories(test_frame_exchange PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)

  # Test: exact scan-column and sim-time cadence math (header-only)
  ament_add_gtest(test_scan_timing test/test_scan_timing.cpp)
  target_include_directories(test_scan_timing PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
  ament_add_gtest(test_sim_time_scheduler test/test_sim_time_scheduler.cpp)
  target_include_directories(test_sim_time_scheduler PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
  ament_add_gtest(test_packet_pacing test/test_packet_pacing.cpp)
  target_link_libraries(test_packet_pacing ${PROJECT_NAME}_core)
  target_include_directories(test_packet_pacing PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)

  # Test: IMU noise model (CPU-only, no Gazebo dep)
  ament_add_gtest(test_imu_noise test/test_imu_noise.cpp)
  target_link_libraries(test_imu_noise gz_gpu_ouster_lidar_cuda)

  # Test: beam azimuth formula and sign convention (header-only math)
  ament_add_gtest(test_beam_math test/test_beam_math.cpp)
  target_link_libraries(test_beam_math gz_gpu_ouster_lidar_cuda)

  # Test: plugin lifecycle smoke (links against the full plugin .so so a
  # Gazebo schema bump that breaks the vtable surfaces at test build time)
  ament_add_gtest(test_lifecycle test/test_lifecycle.cpp)
  target_link_libraries(test_lifecycle ${PROJECT_NAME})

  # Python structural tests (CI-safe: no simulation, no display, no GPU needed)
  ament_add_pytest_test(test_worlds test/test_worlds.py)
  ament_add_pytest_test(test_launch_files test/test_launch_files.py)
  ament_add_pytest_test(test_xacro test/test_xacro.py)
  ament_add_pytest_test(test_repos_file test/test_repos_file.py)
  ament_add_pytest_test(test_display_hardening test/test_display_hardening.py)
endif()

# The public headers quote-include a few backend headers that live in cuda/
# (imu_noise.hpp, panel_layout.hpp, raycast_scene.hpp + their deps). At build
# time those resolve via the backend library's include dirs; in the install
# tree they must sit NEXT to the public headers so the quoted includes
# (same-directory lookup) keep resolving for downstream consumers of the
# installed gz_gpu_ouster_lidar_system.hpp.
#
# ament_auto_package installs include/gz_gpu_ouster_lidar/ to
# <prefix>/include/gz_gpu_ouster_lidar/, so these backend headers must go to the
# SAME directory — not include/${PROJECT_NAME}/... , which is a *different*
# folder (<prefix>/include/gz_sensors_ouster/gz_gpu_ouster_lidar/) where the
# quoted same-directory include would not find them.
install(FILES
  cuda/imu_noise.hpp
  cuda/panel_layout.hpp
  cuda/raycast_scene.hpp
  cuda/raycast_math.hpp
  cuda/ray_processor_math.hpp
  DESTINATION include/gz_gpu_ouster_lidar
)

install(DIRECTORY "${OUSTER_SIM_CORE_SOURCE_DIR}/include/" DESTINATION include)

install(FILES
  src/lidar_common.hpp
  src/ouster_lidar_profile.hpp
  src/ouster_metadata.hpp
  src/packet_encoder.hpp
  src/packet_pacing.hpp
  src/ros_interface.hpp
  src/scan_timing.hpp
  DESTINATION include/${PROJECT_NAME}/core
)

install(DIRECTORY examples
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)

# Installs the plugin library + include/ headers, exports the ament package,
# and installs the runtime config (metadata JSONs + example SDF) to
# share/${PROJECT_NAME}/config.
ament_auto_package(INSTALL_TO_SHARE config)
