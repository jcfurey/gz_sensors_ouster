cmake_minimum_required(VERSION 3.18)
project(gz_sensors_ouster)

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

# GPU backends (all optional). Each probes for its toolchain; missing
# toolchains just disable that backend and fall through to the next at
# runtime. At least one is always compiled in — the CPU backend.
#
# User-overridable: pass -DGZ_GPU_OUSTER_USE_{CUDA,HIP,SYCL}=ON/OFF on the
# cmake command line to force a backend on or off regardless of the probe.
include(CheckLanguage)

# ── CUDA (NVIDIA) ────────────────────────────────────────────────────────────
check_language(CUDA)
set(GZ_GPU_OUSTER_USE_CUDA OFF CACHE BOOL "Build the CUDA backend for NVIDIA GPUs")
if(CMAKE_CUDA_COMPILER)
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

# ── Ouster SDK headers (same pattern as gz_ouster_packet_bridge) ──────────────
# Try drivers/ouster-ros/ first, then drivers/lidar/ouster/ (alternate workspace layout).
set(_ouster_sdk_src "${CMAKE_CURRENT_SOURCE_DIR}/../../drivers/ouster-ros/ouster-ros/ouster-sdk")
if(NOT EXISTS "${_ouster_sdk_src}/ouster_client/include")
  set(_ouster_sdk_src "${CMAKE_CURRENT_SOURCE_DIR}/../../drivers/lidar/ouster/ouster-ros/ouster-sdk")
endif()
if(NOT EXISTS "${_ouster_sdk_src}/ouster_client/include")
  message(FATAL_ERROR
    "Ouster SDK not found. Searched:\n"
    "  ${CMAKE_CURRENT_SOURCE_DIR}/../../drivers/ouster-ros/ouster-ros/ouster-sdk\n"
    "  ${CMAKE_CURRENT_SOURCE_DIR}/../../drivers/lidar/ouster/ouster-ros/ouster-sdk\n"
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

# ── Tests ────────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  # Test: noise model (CPU fallback path)
  ament_add_gtest(test_noise_model test/test_noise_model.cpp)
  target_link_libraries(test_noise_model gz_gpu_ouster_lidar_cuda)

  # Test: parameter validation
  ament_add_gtest(test_parameter_validation test/test_parameter_validation.cpp)
  target_include_directories(test_parameter_validation PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    "${_ouster_sdk_src}/ouster_client/include"
  )
  target_link_libraries(test_parameter_validation
    gz_gpu_ouster_lidar_cuda
  )
  ament_target_dependencies(test_parameter_validation
    rclcpp
    ouster_sensor_msgs
    ouster_ros
  )

  # Test: metadata parsing
  set(_test_metadata_dir "${CMAKE_CURRENT_SOURCE_DIR}/config/metadata")
  if(NOT IS_DIRECTORY "${_test_metadata_dir}")
    message(FATAL_ERROR
      "Test metadata directory missing: ${_test_metadata_dir}\n"
      "The test_metadata_parsing target requires the example Ouster metadata "
      "JSONs shipped in config/metadata/. Either restore them or disable "
      "testing with -DBUILD_TESTING=OFF.")
  endif()
  ament_add_gtest(test_metadata_parsing test/test_metadata_parsing.cpp)
  target_include_directories(test_metadata_parsing PRIVATE
    "${_ouster_sdk_src}/ouster_client/include"
  )
  ament_target_dependencies(test_metadata_parsing
    ouster_ros
  )
  target_compile_definitions(test_metadata_parsing PRIVATE
    TEST_METADATA_DIR="${_test_metadata_dir}"
  )

  # Test: bilinear resampling (CPU fallback)
  ament_add_gtest(test_resample test/test_resample.cpp)
  target_link_libraries(test_resample gz_gpu_ouster_lidar_cuda)
endif()

ament_package()
