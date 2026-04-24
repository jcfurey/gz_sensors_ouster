# Backend library. Despite the directory name "cuda/", this is now a
# multi-backend GPU abstraction. The dispatcher + CPU backend are always
# compiled in; vendor backends (CUDA / HIP / SYCL) are added behind build
# flags.
#
# Build options (set by the root CMakeLists based on toolchain availability):
#   GZ_GPU_OUSTER_USE_CUDA — NVIDIA GPUs via nvcc + curand
#   GZ_GPU_OUSTER_USE_HIP  — AMD GPUs (discrete + APUs) via hipcc + hiprand
#   GZ_GPU_OUSTER_USE_SYCL — Intel GPUs (iGPU + Arc) via icpx / DPC++ / AdaptiveCpp
#
# All three can be enabled simultaneously. At runtime the dispatcher picks
# whichever backend successfully initialises a device, in preference order
# CUDA → HIP → SYCL → CPU. The env var GZ_OUSTER_BACKEND overrides.

set(_backend_sources
  ray_processor_dispatch.cpp
  backend_cpu.cpp
  backend_stubs.cpp
  ray_processor_cpu_impl.cpp
)

add_library(gz_gpu_ouster_lidar_cuda STATIC ${_backend_sources})

set_target_properties(gz_gpu_ouster_lidar_cuda PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

target_include_directories(gz_gpu_ouster_lidar_cuda PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# ── CUDA backend (NVIDIA) ───────────────────────────────────────────────────
if(GZ_GPU_OUSTER_USE_CUDA)
  target_sources(gz_gpu_ouster_lidar_cuda PRIVATE ray_processor_cuda.cu)
  target_compile_definitions(gz_gpu_ouster_lidar_cuda PRIVATE
    GZ_GPU_OUSTER_HAVE_CUDA=1)
  set_target_properties(gz_gpu_ouster_lidar_cuda PROPERTIES
    CUDA_ARCHITECTURES all-major
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF
  )
  target_link_libraries(gz_gpu_ouster_lidar_cuda CUDA::curand)
endif()

# ── HIP backend (AMD, incl. APUs) ───────────────────────────────────────────
# Requires ROCm toolchain. Enable with -DGZ_GPU_OUSTER_USE_HIP=ON plus
# either CMAKE_HIP_COMPILER=hipcc or the rocm/llvm toolchain on PATH.
# The integrated-GPU (APU) code path is selected at runtime by inspecting
# hipDeviceProp_t::integrated.
if(GZ_GPU_OUSTER_USE_HIP)
  enable_language(HIP)
  target_sources(gz_gpu_ouster_lidar_cuda PRIVATE ray_processor_hip.cpp)
  set_source_files_properties(ray_processor_hip.cpp PROPERTIES
    LANGUAGE HIP)
  target_compile_definitions(gz_gpu_ouster_lidar_cuda PRIVATE
    GZ_GPU_OUSTER_HAVE_HIP=1
    __HIP_PLATFORM_AMD__=1)
  # hiprand is header-only on recent ROCm; no explicit link needed. If your
  # ROCm version ships it as a shared library, uncomment:
  # find_package(hiprand CONFIG REQUIRED)
  # target_link_libraries(gz_gpu_ouster_lidar_cuda hip::hiprand)
endif()

# ── SYCL backend (Intel) ────────────────────────────────────────────────────
# Requires Intel oneAPI DPC++ (icpx) OR AdaptiveCpp (acpp). Enable with
# -DGZ_GPU_OUSTER_USE_SYCL=ON and CMAKE_CXX_COMPILER=icpx (or acpp).
# Uses sycl::malloc_shared so Intel iGPUs avoid explicit host/device copies;
# discrete Arc GPUs page-migrate transparently.
if(GZ_GPU_OUSTER_USE_SYCL)
  target_sources(gz_gpu_ouster_lidar_cuda PRIVATE ray_processor_sycl.cpp)
  target_compile_definitions(gz_gpu_ouster_lidar_cuda PRIVATE
    GZ_GPU_OUSTER_HAVE_SYCL=1)
  # icpx recognises -fsycl; AdaptiveCpp handles it via its compiler driver.
  target_compile_options(gz_gpu_ouster_lidar_cuda PRIVATE
    $<$<CXX_COMPILER_ID:IntelLLVM>:-fsycl>)
  target_link_options(gz_gpu_ouster_lidar_cuda PRIVATE
    $<$<CXX_COMPILER_ID:IntelLLVM>:-fsycl>)
endif()

# OpenMP for CPU resampling parallelism.
if(OpenMP_CXX_FOUND)
  target_link_libraries(gz_gpu_ouster_lidar_cuda OpenMP::OpenMP_CXX)
endif()
